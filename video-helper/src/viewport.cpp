#include "viewport.h"
#include "media.h"
#include "lut_loader.h"
#include "block_c_packer.h" // A2 Block C voice allocator (M5 live score packing)
#include "video_param_grammar.h" // Stage 6: shared render-graph param resolver (single source of truth)
#include "block_b_analyzer.h" // A3 Block B analyzer (M4 live audio features)
#include "mix_analyze.h"      // baked-mix offline analysis (audio parity when stopped)
#include "lua_hook.h"         // P2 Scripts-tab live preview (M8 per-frame Lua hook)
#include "js_hook.h"          // P2 Scripts-tab live preview (JS hook; shares FrameCtx)
#include "VideoFrameSharedMemory.h"
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
#include "interp.h" // retime tier 2 (async RIFE "Speed Warp") worker
#endif

#include "gpu_export_common.h" // ARBIT_HAVE_GPU_SHARED + gpuexp::Exporter alias
#if ARBIT_HAVE_GPU_SHARED
#include "fd_socket.h"
#include "SharedGpuSurfaceProtocol.h"
#endif

#include <nlohmann/json.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <set>

#if ARBIT_HAVE_GPU_SHARED
// Fence fds only exist on the dmabuf transport; give the shared code one
// close call that compiles everywhere.
#if defined (_WIN32)
static void closeFdIfValid (int) {}
#else
#include <unistd.h>
static void closeFdIfValid (int fd) { if (fd >= 0) ::close (fd); }
#endif
#endif

using json = nlohmann::json;

namespace
{
int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds> (
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Pending timestamped parameter change (graph_set_param with atBeat >= 0).
struct PendingParam
{
    std::string paramId;
    double value = 0.0;
    double atBeat = 0.0;
};

// Initialize an effect slot's params to the canonical table defaults.
void resetSlotParams (videorender::EffectSlotState& e)
{
    std::memset (e.params, 0, sizeof (e.params));
    if (const auto* def = videofx::effectDefFor (e.type))
        for (int i = 0; i < def->paramCount; ++i)
            e.params[i] = def->params[i].defaultValue;
}

// ---- Video scopes (PROTOCOL.md §Scopes) -----------------------------------

// Wire sizes. The sample frame is the renderer's fixed downscale target;
// 480x135 = 64,800 samples keeps every histogram bin within uint16_t even
// for a solid-colour frame.
constexpr int kScopeWaveW = 480, kScopeWaveH = 64;   // waveform: column x luma bin
constexpr int kScopeVecN = 128;                      // vectorscope: Cb x Cr density
static_assert (kScopeWaveW == videorender::FrameRenderer::kScopeW,
               "waveform width must match the scope readback width");
static_assert (videorender::FrameRenderer::kScopeW
                   * videorender::FrameRenderer::kScopeH <= 65535,
               "histogram bins are uint16_t — sample count must fit");

// Standard base64 ('+'/'/', '=' padding) — counterpart of the decoder in
// main.cpp; juce::Base64 on the Arbit side reads this directly.
std::string base64Encode (const uint8_t* data, size_t n)
{
    static const char* tab =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve ((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3)
    {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < n ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < n ? data[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out.push_back (tab[(v >> 18) & 63]);
        out.push_back (tab[(v >> 12) & 63]);
        out.push_back (i + 1 < n ? tab[(v >> 6) & 63] : '=');
        out.push_back (i + 2 < n ? tab[v & 63] : '=');
    }
    return out;
}
} // namespace

struct Viewport::Impl
{
    mutable std::mutex mutex;

    // Control-plane state (written by RPC thread, read by render thread).
    std::vector<ViewportSegment> segments;
    std::map<int, ClipGraphParams> clipParams; // keyed by clipId
    std::vector<PendingParam> pendingParams;
    videoshm::TransportRegion transport;
    std::string transportName;
    // M4 Block B live: the master-mix audio ring Arbit's audio thread writes.
    // The render loop drains it under `mutex` (guards against attachAudio's
    // open()/remap) and runs the BlockBAnalyzer; see the render loop.
    videoshm::AudioRingRegion audioRing;
    std::string audioName;
    uint64_t timelineGeneration = 0; // bumped by setTimeline (stream pruning)

    // Frame-perfect audio parity (sweep): the offline-analyzed master mix
    // (setAudioMix). The live ring only flows WHILE PLAYING, so while STOPPED the
    // render loop reads THESE features (same analysis the exporter runs) at the
    // current frame's time instead of zero-feeding audio-reactive shaders/mods.
    // previewMixGeneration bumps on every push so the render thread re-copies its
    // local snapshot only when the mix actually changes (the vector is large —
    // one FeatureFrame per hop for the whole song). Empty ⇒ stopped = zero-feed.
    std::vector<arbitblockb::FeatureFrame> previewMixFeats;
    double previewMixSr = 0.0;
    uint64_t previewMixGeneration = 0;

    // M5 Block C live score (setScore). Read by the render loop each frame and
    // packed via the A2 BlockCPacker. scoreGeneration is bumped on every push so
    // the render thread re-copies + resets its packer only when the note set
    // actually changes (not every frame). Empty score ⇒ shaders see the
    // zero-feed (uNoteCount 0), identical to never calling setScore.
    arbitmod::Score score;
    double scoreLookaheadBeats = 0.0;
    uint64_t scoreGeneration = 0;

    // M6 cross-domain mod matrix (setModMatrix). Routings are evaluated each
    // frame against the live clock + score + audio (the live mix from the Block
    // B audio ring) and laid over the per-frame clip-param copy. modGeneration is
    // bumped on every push so the render thread re-copies + resets its routing
    // states only when the routing set changes. Empty ⇒ no overlay (identical
    // to never calling setModMatrix).
    std::vector<arbitmod::Routing> routings;
    uint64_t modGeneration = 0;

    // P2 Scripts-tab live preview (setScript). The project-global per-frame hook
    // source + engine ("js" | "lua"). scriptGeneration is bumped on every push so
    // the render thread re-copies + recompiles only when the script actually
    // changes (not every frame). Empty source ⇒ no hook (identical to never
    // calling setScript).
    std::string scriptSource;
    std::string scriptLang;
    uint64_t scriptGeneration = 0;

    // Text overlay store (text_set_image / text_remove) — independent of the
    // timeline so texts survive segment resends. Pixel uploads and texture
    // deletes run on the render thread (GL context); the RPC thread only
    // stages pixels / flags under the mutex.
    struct TextState
    {
        // control-plane (mutex-guarded)
        double startSec = 0.0, durationSec = 0.0;
        int ownerClipId = -1;                  // -1 = project-level text
        float posX = 0.0f, posY = 0.0f;       // base NDC from text_set_image
        float opacity = 1.0f;
        float translateX = 0.0f, translateY = 0.0f; // graph_set_param offsets
        float visible = 1.0f;
        float zOrder = 0.0f;
        std::vector<uint8_t> pendingPixels;   // straight RGBA, stride w*4
        int pendingW = 0, pendingH = 0;
        bool pixelsDirty = false;
        bool removed = false;                 // render thread frees + erases
        // render-thread owned
        unsigned tex = 0;
        int texW = 0, texH = 0;
    };
    std::map<int, TextState> texts;           // keyed by textId

    // Per-clip 3D LUT staging (graph_set_lut). The RPC thread parses the
    // .cube file and stages the float grid here; the render thread (GL
    // context) uploads/deletes textures and clears the stage flags.
    struct LutState
    {
        // control-plane (mutex-guarded)
        std::string path;                 // "" = clear
        std::vector<float> pendingData;   // size^3 RGB triples
        int pendingSize = 0;
        bool dirty = false;               // upload (or clear, if path empty)
        // render-thread owned
        unsigned tex = 0;
        int size = 0;
    };
    std::map<int, LutState> luts;             // keyed by clipId

    // Video scope results (PROTOCOL.md §Scopes) — published by the render
    // thread under the mutex, read by scopeDataJson on the RPC thread.
    // seq == 0 means "no data yet". Disabled scopes keep their last arrays
    // (cheap, and scope_data only reports enabled types anyway).
    uint64_t scopeSeq = 0;
    std::vector<uint8_t> scopeWaveform;       // kScopeWaveW * kScopeWaveH u8
    std::vector<uint8_t> scopeVector;         // kScopeVecN * kScopeVecN u8
    uint16_t scopeHistogram[3][256] = {};     // R/G/B value counts

    // CPU scope accumulation over one downscaled composite frame (~65k
    // pixels at ~10 Hz — negligible). rgba is top-row-flipped by the blit,
    // which no scope cares about: pixels are samples, columns preserved.
    void computeScopes (uint32_t mask, const uint8_t* rgba, int w, int h)
    {
        std::vector<uint16_t> wave;
        std::vector<uint16_t> vec;
        uint32_t hist[3][256] = {};
        if (mask & Viewport::kScopeWaveform)
            wave.assign ((size_t) kScopeWaveW * kScopeWaveH, 0);
        if (mask & Viewport::kScopeVectorscope)
            vec.assign ((size_t) kScopeVecN * kScopeVecN, 0);

        for (int y = 0; y < h; ++y)
        {
            const uint8_t* row = rgba + (size_t) y * (size_t) w * 4;
            for (int x = 0; x < w; ++x)
            {
                const int r = row[x * 4 + 0];
                const int g = row[x * 4 + 1];
                const int b = row[x * 4 + 2];

                if (mask & Viewport::kScopeHistogram)
                {
                    ++hist[0][r];
                    ++hist[1][g];
                    ++hist[2][b];
                }
                if (mask & Viewport::kScopeWaveform)
                {
                    // Rec.709 luma, integer weights summing to 256.
                    const int luma = (54 * r + 183 * g + 19 * b) >> 8;
                    const int bin = kScopeWaveH - 1
                                  - std::clamp (luma * kScopeWaveH / 256, 0, kScopeWaveH - 1);
                    ++wave[(size_t) bin * kScopeWaveW + (size_t) x];
                }
                if (mask & Viewport::kScopeVectorscope)
                {
                    // BT.709 Cb/Cr, integer weights x256. Cb right, Cr up
                    // (row 0 = top), neutral at the centre cell (64, 64).
                    const int cb = (-29 * r - 99 * g + 128 * b) >> 8;  // -128..127
                    const int cr = (128 * r - 116 * g - 12 * b) >> 8;
                    const int vx = std::clamp (kScopeVecN / 2 + cb / 2, 0, kScopeVecN - 1);
                    const int vy = std::clamp (kScopeVecN / 2 - cr / 2, 0, kScopeVecN - 1);
                    ++vec[(size_t) vy * kScopeVecN + (size_t) vx];
                }
            }
        }

        // Density -> saturating 8-bit intensity (display gamma is the
        // panel's job; any monotone mapping works for plausibility checks).
        std::vector<uint8_t> wave8, vec8;
        if (mask & Viewport::kScopeWaveform)
        {
            wave8.resize (wave.size());
            for (size_t i = 0; i < wave.size(); ++i)
                wave8[i] = (uint8_t) std::min<uint32_t> (255u, (uint32_t) wave[i] * 40u);
        }
        if (mask & Viewport::kScopeVectorscope)
        {
            vec8.resize (vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
                vec8[i] = (uint8_t) std::min<uint32_t> (255u, (uint32_t) vec[i] * 32u);
        }

        std::lock_guard<std::mutex> lock (mutex);
        ++scopeSeq;
        if (mask & Viewport::kScopeWaveform)
            scopeWaveform = std::move (wave8);
        if (mask & Viewport::kScopeVectorscope)
            scopeVector = std::move (vec8);
        if (mask & Viewport::kScopeHistogram)
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < 256; ++i)
                    scopeHistogram[c][i] = (uint16_t) std::min<uint32_t> (hist[c][i], 65535u);
    }

    // Window control requests (GLFW calls must run on the render thread).
    std::atomic<bool> wantClose { false };
    std::atomic<bool> boundsDirty { false };
    int reqX = 0, reqY = 0, reqW = 0, reqH = 0;
    std::atomic<int> fullscreenRequest { -1 }; // -1 none, 0 windowed, 1 fullscreen

    // Stats (written by render thread).
    std::atomic<uint64_t> framesPresented { 0 };
    std::atomic<uint64_t> lastFrameHash { 0 };
    // Retime interpolation tier actually used on the most recent frame:
    // 0=nearest, 1=frame-blend, 2=rife-cuda, 3=rife-coreml, 4=rife-vulkan (reported by info()).
    std::atomic<int> interpBackend { 0 };
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    // Async RIFE worker for retime tier 2 — lazily started on first tier-2 use
    // (render-thread-owned; the worker has no GL context).
    std::unique_ptr<arbitinterp::InterpEngine> interp;
#endif
    std::atomic<uint64_t> sharedFramesSent { 0 };
    std::atomic<uint64_t> sharedFramesDroppedNoBuffer { 0 };
    std::atomic<int> sharedFreeBuffers { 0 };
    std::atomic<int> sharedBusyBuffers { 0 };
    double measuredFps = 0.0;
    // Project (canonical) frame rate that drives the VALUE grid — the rate the
    // exporter steps at (job.fps). Set from the viewport_open* RPC (clkFps) and
    // updated live by viewport_set_clkfps so the live preview snaps modulation /
    // shader-clock / score / lua / source selection to the SAME frame grid the
    // export uses (frame-perfect parity). 0 ⇒ fall back to the display targetFps.
    // Distinct from targetFps, which only paces the DISPLAY refresh — the window
    // can still repaint at 60Hz, holding the current project-fps frame. Atomic
    // because the RPC thread writes it while the render thread reads it per frame.
    std::atomic<double> valueFps { 0.0 };
    double avOffsetSec = 0.0;
    double displaySec = 0.0;
    double sourcePtsSec = 0.0;
    double sourceIdealSec = 0.0;
    bool transportOpen = false;
    bool transportPlaying = false;
    uint32_t transportGeneration = 0;
    double transportAgeSec = 0.0;
    double transportPlayheadBeats = 0.0;
    int curW = 0, curH = 0;
    arbitgl::GpuCaps gpuCaps;          // snapshot once after context creation (P1)
    std::atomic<bool> windowOpen { false };
    std::string openError;
    std::atomic<bool> openDone { false };

    // Shared (zero-copy docked) mode state — render-thread owned except
    // where noted. sharedMode is set before the thread starts and never
    // changes. The exporter is the platform backend behind gpuexp::Exporter
    // (dmabuf / IOSurface / D3D11 interop — gpu_export_common.h).
    bool sharedMode = false;
    std::atomic<bool> dmabufCapable { false }; // "shared-GPU capable" (name is wire-compat)

    // shm-docked mode state. shmMode is set and the ring (Arbit-owned,
    // opened by name) attached before the thread starts; cursor/seq are
    // render-thread only.
    bool shmMode = false;
    videoshm::Region shmRing;
    std::atomic<bool> shmResizeDirty { false };
    std::atomic<int> shmReqW { 0 }, shmReqH { 0 };
    uint32_t shmSlotCursor = 0;
    uint64_t shmFrameSeq = 0;
#if ARBIT_HAVE_GPU_SHARED
    int sharedBufferCount = 3;
    std::atomic<bool> sharedResizeDirty { false };
    std::atomic<int> sharedReqW { 0 }, sharedReqH { 0 };
    gpuexp::Exporter exporter;
    FdSocketServer fdSocket;
    std::vector<gpuexp::ExportedBuffer> buffers; // render thread only
    uint64_t frameSeq = 0;                       // render thread only
    Viewport::SharedOpenResult sharedResult;     // filled before openDone
#endif

    ClipGraphParams* paramsFor (int clipId)
    {
        return &clipParams[clipId]; // default-constructs on first use
    }

    // Parses "clip<id>/<node>/<param>". Returns false on malformed IDs.
    static bool parseParamId (const std::string& id, int& clipId,
                              std::string& node, std::string& param)
    {
        if (id.rfind ("clip", 0) != 0) return false;
        const auto slash1 = id.find ('/');
        if (slash1 == std::string::npos) return false;
        const auto slash2 = id.find ('/', slash1 + 1);
        if (slash2 == std::string::npos) return false;
        try { clipId = std::stoi (id.substr (4, slash1 - 4)); }
        catch (...) { return false; }
        node = id.substr (slash1 + 1, slash2 - slash1 - 1);
        param = id.substr (slash2 + 1);
        return ! node.empty() && ! param.empty();
    }

    // Parses "text<id>/<param>". Returns false on malformed IDs.
    static bool parseTextParamId (const std::string& id, int& textId,
                                  std::string& param)
    {
        if (id.rfind ("text", 0) != 0) return false;
        const auto slash = id.find ('/');
        if (slash == std::string::npos) return false;
        try { textId = std::stoi (id.substr (4, slash - 4)); }
        catch (...) { return false; }
        param = id.substr (slash + 1);
        return ! param.empty();
    }

    // Applies one text overlay param. Caller holds the mutex. Creates the
    // entry on first use (mirrors paramsFor) so automation values streamed
    // before text_set_image are not lost.
    bool applyText (int textId, const std::string& param, double value)
    {
        auto& t = texts[textId];
        if (param == "opacity")    { t.opacity = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "translateX") { t.translateX = (float) value; return true; }
        if (param == "translateY") { t.translateY = (float) value; return true; }
        if (param == "posX")       { t.posX = (float) value; return true; }
        if (param == "posY")       { t.posY = (float) value; return true; }
        if (param == "visible")    { t.visible = value >= 0.5 ? 1.0f : 0.0f; return true; }
        if (param == "zOrder")     { t.zOrder = (float) value; return true; }
        return false;
    }

    // Applies one parsed param. Caller holds the mutex.
    bool apply (int clipId, const std::string& node, const std::string& param,
                double value)
    {
        return applyTo (*paramsFor (clipId), node, param, value);
    }

    // Writes one parsed param onto a ClipGraphParams. Used by apply() (on the stored
    // params) and by the M6 mod-matrix overlay (on a per-frame copy, so the stored base
    // is never mutated). Delegates to the shared videoparam resolver — the SAME grammar
    // exporter.cpp uses, so the live viewport and the export can no longer diverge.
    static bool applyTo (ClipGraphParams& p, const std::string& node,
                         const std::string& param, double value)
    {
        if (videoparam::applyGraphParam (p, node, param, value))
            return true;
        // ISF/generator INPUT ("clip<id>/gen/<name>") — exporter-only in the shared
        // grammar, so handled here as the viewport's caller-side glue (mirrors
        // exporter.cpp::applyGraphParam). Stored unclamped; the shader clamps.
        if (node == "gen") { p.genParams[param] = value; return true; }
        return false;
    }

    // Reads one parsed param from a ClipGraphParams (the `base` a mod routing modulates:
    // combine(mode, base, value)). The inverse of applyTo, via the shared resolver.
    static bool readFrom (const ClipGraphParams& p, const std::string& node,
                          const std::string& param, double& out)
    {
        if (videoparam::getGraphParam (p, node, param, out))
            return true;
        // ISF/generator INPUT base for a mod-matrix routing (mirrors
        // exporter.cpp::getGraphParam). An unseeded name returns false so the
        // routing is skipped rather than applied to a junk base.
        if (node == "gen")
        {
            const auto it = p.genParams.find (param);
            if (it == p.genParams.end()) return false;
            out = it->second;
            return true;
        }
        return false;
    }
};

Viewport::Viewport() = default;
Viewport::~Viewport() { close(); }

const char* Viewport::sharedGpuPathTag()
{
#if ARBIT_HAVE_GPU_SHARED
    return gpuexp::kSharedGpuTag;
#else
    return "";
#endif
}

std::string Viewport::open (int width, int height, int x, int y,
                            bool alwaysOnTop, double targetFps, double clkFps)
{
    if (running_.load())
    {
        if (impl_ != nullptr && impl_->sharedMode)
            return "viewport is in shared (dmabuf) mode — close it first";
        if (impl_ != nullptr && impl_->shmMode)
            return "viewport is in shm (docked) mode — close it first";
        return {}; // already open
    }

    impl_ = std::make_unique<Impl>();
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    running_ = true;
    thread_ = std::thread ([this, width, height, x, y, alwaysOnTop, targetFps]
                           { renderLoop (width, height, x, y, alwaysOnTop, targetFps); });

    // Wait for the window to come up (or fail) so the RPC reply is accurate.
    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "viewport window failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    return {};
}

std::string Viewport::openShared (int width, int height, double targetFps,
                                  int bufferCount, SharedOpenResult& out,
                                  double clkFps)
{
#if ARBIT_HAVE_GPU_SHARED
    if (running_.load())
        return impl_ != nullptr && impl_->sharedMode
                   ? "shared viewport already open"
                   : "viewport window already open — close it first";

    impl_ = std::make_unique<Impl>();
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    impl_->sharedMode = true;
    impl_->sharedBufferCount = std::clamp (bufferCount, 2, 4);
    running_ = true;
    thread_ = std::thread ([this, width, height, targetFps]
                           { renderLoop (width, height, -1, -1, false, targetFps); });

    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "shared viewport failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    out = impl_->sharedResult;
    return {};
#else
    (void) width; (void) height; (void) targetFps; (void) bufferCount; (void) out; (void) clkFps;
    return "helper built without shared-GPU export support";
#endif
}

std::string Viewport::resizeShared (int width, int height)
{
#if ARBIT_HAVE_GPU_SHARED
    if (! running_.load() || impl_ == nullptr || ! impl_->sharedMode)
        return "shared viewport not open";
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return "bad shared viewport size";
    impl_->sharedReqW = width;
    impl_->sharedReqH = height;
    impl_->sharedResizeDirty = true;
    return {};
#else
    (void) width; (void) height;
    return "helper built without shared-GPU export support";
#endif
}

std::string Viewport::openShm (int width, int height, double targetFps,
                               const std::string& shmName, double clkFps)
{
    if (running_.load())
        return impl_ != nullptr && impl_->shmMode
                   ? "shm viewport already open"
                   : "another viewport mode is open — close it first";
    if (shmName.empty())
        return "missing shmName";
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return "bad shm viewport size";

    impl_ = std::make_unique<Impl>();
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    impl_->shmMode = true;
    if (! impl_->shmRing.open (shmName))
    {
        impl_.reset();
        return "failed to open frame shm region: " + shmName;
    }
    const auto* hdr = impl_->shmRing.header();
    if (hdr == nullptr || hdr->slotCount == 0 || hdr->slotBytes == 0)
    {
        impl_.reset();
        return "bad frame shm region: " + shmName;
    }

    running_ = true;
    thread_ = std::thread ([this, width, height, targetFps]
                           { renderLoop (width, height, -1, -1, false, targetFps); });

    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "shm viewport failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    return {};
}

std::string Viewport::resizeShm (int width, int height)
{
    if (! running_.load() || impl_ == nullptr || ! impl_->shmMode)
        return "shm viewport not open";
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return "bad shm viewport size";
    impl_->shmReqW = width;
    impl_->shmReqH = height;
    impl_->shmResizeDirty = true;
    return {};
}

void Viewport::close()
{
    if (impl_ != nullptr)
        impl_->wantClose = true;
    running_ = false;
    if (thread_.joinable())
        thread_.join();
    impl_.reset();
}

void Viewport::setBounds (int x, int y, int width, int height)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->reqX = x; impl_->reqY = y; impl_->reqW = width; impl_->reqH = height;
    impl_->boundsDirty = true;
}

void Viewport::setFullscreen (bool fullscreen)
{
    if (impl_ == nullptr) return;
    impl_->fullscreenRequest = fullscreen ? 1 : 0;
}

void Viewport::setTimeline (std::vector<ViewportSegment> segments)
{
    if (impl_ == nullptr) return;
    std::sort (segments.begin(), segments.end(),
               [] (const ViewportSegment& a, const ViewportSegment& b)
               { return a.displayStartSec < b.displayStartSec; });
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->segments = std::move (segments);
    ++impl_->timelineGeneration;
}

void Viewport::setScore (arbitmod::Score score, double lookaheadBeats)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->score = std::move (score);
    impl_->scoreLookaheadBeats = lookaheadBeats;
    ++impl_->scoreGeneration;
}

void Viewport::setModMatrix (std::vector<arbitmod::Routing> routings)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->routings = std::move (routings);
    ++impl_->modGeneration;
}

void Viewport::setScript (std::string source, std::string lang)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->scriptSource = std::move (source);
    impl_->scriptLang   = (lang == "lua") ? "lua" : "js";
    ++impl_->scriptGeneration;
}

void Viewport::attachTransport (const std::string& shmName)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->transportName = shmName;
    impl_->transport.open (shmName);
}

void Viewport::attachAudio (const std::string& shmName)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->audioName = shmName;
    impl_->audioRing.open (shmName);
}

void Viewport::setAudioMix (const std::string& wavPath)
{
    if (impl_ == nullptr) return;
    // Decode + offline-analyze OUTSIDE the lock (runs on the RPC thread, can take
    // a moment for a long mix); publish the result under the lock with a bumped
    // generation so the render thread picks it up. Empty path / failed decode ⇒
    // clear (stopped preview falls back to zero-feed). The exporter and this path
    // share videohelper::analyzeMixWavOffline ⇒ identical frames ⇒ stopped-preview
    // audio == export audio at the same timeline time.
    std::vector<arbitblockb::FeatureFrame> feats;
    double sr = 0.0;
    if (! wavPath.empty())
        feats = videohelper::analyzeMixWavOffline (wavPath, sr);

    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->previewMixFeats = std::move (feats);
    impl_->previewMixSr    = sr;
    ++impl_->previewMixGeneration;
}

std::string Viewport::setCanvas (int width, int height)
{
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192)
        return "bad canvas size";
    canvasW_ = width;
    canvasH_ = height;
    return {};
}

std::string Viewport::setCanvasBackground (double r, double g, double b, double a)
{
    bgR_ = r; bgG_ = g; bgB_ = b; bgA_ = a;
    return {};
}

std::string Viewport::setPostFx (double bloomIntensity, double bloomThreshold,
                                 double bloomRadius, int tonemap, double exposure)
{
    bloomIntensity_ = bloomIntensity > 0.0 ? bloomIntensity : 0.0;
    bloomThreshold_ = bloomThreshold;
    bloomRadius_    = bloomRadius > 0.0 ? bloomRadius : 0.0;
    tonemap_        = (tonemap >= 0 && tonemap <= 2) ? tonemap : 0;
    exposure_       = exposure > 0.0 ? exposure : 1.0;
    return {};
}

std::string Viewport::setParam (const std::string& paramId, double value, double atBeat)
{
    // Viewport-only view transform (PROTOCOL.md §Project canvas & view
    // transform). Stored on the Viewport so values survive close/reopen and
    // can be set before the viewport opens. Never timestamped.
    if (paramId.rfind ("view/", 0) == 0)
    {
        const std::string p = paramId.substr (5);
        if (p == "zoom") { viewZoom_ = std::clamp (value, 0.02, 32.0); return {}; }
        if (p == "panX") { viewPanX_ = value; return {}; }
        if (p == "panY") { viewPanY_ = value; return {}; }
        return "unknown view param: " + paramId;
    }

    if (impl_ == nullptr) return "viewport not open";
    int clipId = 0, textId = 0;
    std::string node, param, textParam;
    const bool isClip = Impl::parseParamId (paramId, clipId, node, param);
    const bool isText = ! isClip && Impl::parseTextParamId (paramId, textId, textParam);
    if (! isClip && ! isText)
        return "malformed paramId (expected clip<id>/<node>/<param>, text<id>/<param> or view/<param>): " + paramId;

    std::lock_guard<std::mutex> lock (impl_->mutex);
    if (atBeat >= 0.0)
    {
        impl_->pendingParams.push_back ({ paramId, value, atBeat });
        return {};
    }
    const bool ok = isClip ? impl_->apply (clipId, node, param, value)
                           : impl_->applyText (textId, textParam, value);
    if (! ok)
        return "unknown param: " + paramId;
    return {};
}

std::string Viewport::setTextImage (int textId, std::vector<uint8_t> rgba,
                                    int width, int height,
                                    double startSec, double durationSec,
                                    double posX, double posY,
                                    double opacity, double zOrder,
                                    int ownerClipId)
{
    if (impl_ == nullptr) return "viewport not open";
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192)
        return "bad text image size";
    if (rgba.size() < (size_t) width * (size_t) height * 4)
        return "text image pixel buffer too small";

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto& t = impl_->texts[textId];
    t.removed = false;
    t.pendingPixels = std::move (rgba);
    t.pendingW = width;
    t.pendingH = height;
    t.pixelsDirty = true;
    t.startSec = startSec;
    t.durationSec = durationSec;
    t.ownerClipId = ownerClipId;
    t.posX = (float) posX;
    t.posY = (float) posY;
    t.opacity = (float) std::clamp (opacity, 0.0, 1.0);
    t.zOrder = (float) zOrder;
    // translateX/translateY/visible are param-only and deliberately kept.
    return {};
}

std::string Viewport::removeText (int textId)
{
    if (impl_ == nullptr) return "viewport not open";
    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto it = impl_->texts.find (textId);
    if (it == impl_->texts.end())
        return {}; // idempotent
    it->second.removed = true;     // render thread frees the texture + erases
    it->second.pixelsDirty = false;
    it->second.pendingPixels.clear();
    return {};
}

std::string Viewport::setEffects (int clipId, const std::vector<EffectSlotSpec>& slots)
{
    if (impl_ == nullptr) return "viewport not open";

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto* p = impl_->paramsFor (clipId);

    // Bulk replace: the call describes the whole rack.
    for (auto& e : p->effects)
        e = videorender::EffectSlotState {};

    for (const auto& s : slots)
    {
        if (s.slot < 0 || s.slot >= videofx::kMaxEffectSlots)
            return "bad slot index: " + std::to_string (s.slot);
        if (s.type < -1 || s.type >= videofx::kEffectTypeCount)
            return "bad effect type: " + std::to_string (s.type);
        auto& e = p->effects[s.slot];
        e.type = s.type;
        e.enabled = s.enabled && s.type >= 0;
        if (s.type < 0)
            continue;
        resetSlotParams (e);
        for (const auto& [name, value] : s.params)
        {
            const int idx = videofx::effectParamIndex (s.type, name.c_str());
            if (idx < 0)
                return "unknown param '" + name + "' for effect type "
                     + std::to_string (s.type);
            const auto& pd = videofx::kEffectDefs[s.type].params[idx];
            e.params[idx] = std::clamp ((float) value, pd.minValue, pd.maxValue);
        }
    }
    return {};
}

std::string Viewport::setLut (int clipId, const std::string& path)
{
    if (impl_ == nullptr) return "viewport not open";

    std::vector<float> data;
    int size = 0;
    if (! path.empty())
    {
        const std::string err = loadCubeLut (path, data, size);
        if (! err.empty())
            return err;
    }

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto& l = impl_->luts[clipId];
    l.path = path;
    l.pendingData = std::move (data);
    l.pendingSize = size;
    l.dirty = true;
    return {};
}

std::string Viewport::describeGraph() const
{
    json nodes = json::array();
    json texts = json::array();
    arbitgl::GpuCaps caps;
    if (impl_ != nullptr)
    {
        std::lock_guard<std::mutex> lock (impl_->mutex);
        caps = impl_->gpuCaps;
        for (const auto& [textId, t] : impl_->texts)
        {
            if (t.removed)
                continue;
            const std::string prefix = "text" + std::to_string (textId);
            texts.push_back ({
                { "textId", textId },
                { "width", t.pixelsDirty ? t.pendingW : t.texW },
                { "height", t.pixelsDirty ? t.pendingH : t.texH },
                { "startSec", t.startSec },
                { "durationSec", t.durationSec },
                { "ownerClipId", t.ownerClipId },
                { "posX", t.posX },
                { "posY", t.posY },
                { "params", {
                    { prefix + "/opacity", t.opacity },
                    { prefix + "/translateX", t.translateX },
                    { prefix + "/translateY", t.translateY },
                    { prefix + "/visible", t.visible },
                    { prefix + "/zOrder", t.zOrder },
                } },
            });
        }
        for (const auto& [clipId, p] : impl_->clipParams)
        {
            const std::string prefix = "clip" + std::to_string (clipId);
            json effects = json::array();
            for (int slot = 0; slot < videofx::kMaxEffectSlots; ++slot)
            {
                const auto& e = p.effects[slot];
                if (e.type < 0)
                    continue;
                json params = json::object();
                if (const auto* def = videofx::effectDefFor (e.type))
                    for (int i = 0; i < def->paramCount; ++i)
                        params[def->params[i].name] = e.params[i];
                effects.push_back ({
                    { "slot", slot },
                    { "type", e.type },
                    { "enabled", e.enabled },
                    { "params", params },
                });
            }
            // retimeQuality is a per-segment field (uniform across a clip's
            // slices); surface the first slice's tier under the retime stage.
            int retimeQuality = 0;
            for (const auto& s : impl_->segments)
                if (s.clipId == clipId) { retimeQuality = s.retimeQuality; break; }
            json node = {
                { "clipId", clipId },
                { "topology", { "source", "retime", "effects", "mask", "transform2d", "display" } },
                { "params", {
                    { prefix + "/source/opacity", p.opacity },
                    { prefix + "/source/visible", p.visible },
                    { prefix + "/source/zOrder", p.zOrder },
                    { prefix + "/source/blendMode", p.blendMode },
                    { prefix + "/retime/quality", retimeQuality },
                    { prefix + "/transform2d/scale", p.scale },
                    { prefix + "/transform2d/translateX", p.translateX },
                    { prefix + "/transform2d/translateY", p.translateY },
                    { prefix + "/transform2d/rotation", p.rotationDeg },
                    { prefix + "/transform2d/cropLeft", p.cropLeft },
                    { prefix + "/transform2d/cropRight", p.cropRight },
                    { prefix + "/transform2d/cropTop", p.cropTop },
                    { prefix + "/transform2d/cropBottom", p.cropBottom },
                    { prefix + "/mask/type", p.maskType },
                    { prefix + "/mask/cx", p.maskCx },
                    { prefix + "/mask/cy", p.maskCy },
                    { prefix + "/mask/w", p.maskW },
                    { prefix + "/mask/h", p.maskH },
                    { prefix + "/mask/feather", p.maskFeather },
                    { prefix + "/mask/invert", p.maskInvert },
                } },
                { "effects", effects },
            };
            if (const auto it = impl_->luts.find (clipId);
                it != impl_->luts.end() && ! it->second.path.empty())
            {
                node["lutPath"] = it->second.path;
                node["lutSize"] = it->second.dirty ? it->second.pendingSize
                                                   : it->second.size;
            }
            nodes.push_back (std::move (node));
        }
    }
    const json canvas = {
        { "width", canvasW_.load() },
        { "height", canvasH_.load() },
        { "viewZoom", viewZoom_.load() },
        { "viewPanX", viewPanX_.load() },
        { "viewPanY", viewPanY_.load() },
    };
    // P1: report the GPU capability snapshot so agents/UI can gate
    // compute-class features (and the export path can flag them unavailable).
    const json gpu = {
        { "glMajor", caps.glMajor },
        { "glMinor", caps.glMinor },
        { "computeShaders", caps.computeShaders },
        { "particles", caps.particles },
        { "ssbo", caps.ssbo },
        { "imageLoadStore", caps.imageLoadStore },
        { "maxComputeWorkGroupCount", { caps.maxComputeWorkGroupCount[0],
                                        caps.maxComputeWorkGroupCount[1],
                                        caps.maxComputeWorkGroupCount[2] } },
        { "maxComputeWorkGroupSize", { caps.maxComputeWorkGroupSize[0],
                                       caps.maxComputeWorkGroupSize[1],
                                       caps.maxComputeWorkGroupSize[2] } },
        { "maxComputeWorkGroupInvocations", caps.maxComputeWorkGroupInvocations },
    };
    return json { { "nodes", nodes }, { "texts", texts },
                  { "canvas", canvas }, { "gpu", gpu } }.dump();
}

ViewportInfo Viewport::info() const
{
    ViewportInfo vi;
    vi.canvasWidth = canvasW_.load();
    vi.canvasHeight = canvasH_.load();
    vi.viewZoom = viewZoom_.load();
    vi.viewPanX = viewPanX_.load();
    vi.viewPanY = viewPanY_.load();
    if (impl_ == nullptr) return vi;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    vi.open = impl_->windowOpen.load();
    vi.width = impl_->curW;
    vi.height = impl_->curH;
    vi.measuredFps = impl_->measuredFps;
    vi.avOffsetSec = impl_->avOffsetSec;
    const int ib = impl_->interpBackend.load();
    vi.interpolationActive = ib != 0;
    vi.interpolationBackend = ib == 1 ? "frame-blend"
                            : ib == 2 ? "rife-cuda"
                            : ib == 3 ? "rife-coreml"
                            : ib == 4 ? "rife-vulkan"
                            : "nearest";
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    if (impl_->interp != nullptr)
    {
        vi.interpReady = impl_->interp->available();
        vi.interpWorkerBackend = impl_->interp->backend();
        vi.interpInferences = impl_->interp->inferenceCount();
        vi.interpCacheHits = impl_->interp->cacheHits();
        vi.interpCacheMisses = impl_->interp->cacheMisses();
        vi.interpError = impl_->interp->lastError();
    }
#endif
    vi.interpTargetFps = interpTargetFps_.load (std::memory_order_relaxed);
    vi.gpuPath = "opengl";
    vi.framesPresented = impl_->framesPresented.load();
    vi.lastFrameHash = impl_->lastFrameHash.load();
    vi.displaySec = impl_->displaySec;
    vi.sourcePtsSec = impl_->sourcePtsSec;
    vi.sourceIdealSec = impl_->sourceIdealSec;
    vi.transportOpen = impl_->transportOpen;
    vi.transportPlaying = impl_->transportPlaying;
    vi.transportGeneration = impl_->transportGeneration;
    vi.transportAgeSec = impl_->transportAgeSec;
    vi.transportPlayheadBeats = impl_->transportPlayheadBeats;
#if ARBIT_HAVE_GPU_SHARED
    const char* sharedPath = gpuexp::kSharedPathName;
#else
    const char* sharedPath = "dmabuf-docked"; // unreachable: sharedMode never set
#endif
    vi.presentPath = ! vi.open ? "none"
                   : impl_->sharedMode ? sharedPath
                   : impl_->shmMode ? "shm-docked" : "glfw-window";
    vi.dmabufCapable = impl_->dmabufCapable.load();
    vi.sharedFramesSent = impl_->sharedFramesSent.load();
    vi.sharedFramesDroppedNoBuffer = impl_->sharedFramesDroppedNoBuffer.load();
    vi.sharedFreeBuffers = impl_->sharedFreeBuffers.load();
    vi.sharedBusyBuffers = impl_->sharedBusyBuffers.load();
    vi.gpuCaps = impl_->gpuCaps;
    return vi;
}

uint64_t Viewport::frameHash() const
{
    return impl_ != nullptr ? impl_->lastFrameHash.load() : 0;
}

void Viewport::setInterpolationSuspended (bool suspended)
{
    // Viewport-level atomic (not impl_): safe to call from the export worker
    // thread even as close() resets impl_.
    interpSuspended_.store (suspended, std::memory_order_relaxed);
}

void Viewport::setInterpTargetFps (double fps)
{
    // Viewport-level atomic (not impl_): survives close/reopen. Clamp junk to 0
    // (= legacy slowed-only gate). The render loop reads it per frame.
    interpTargetFps_.store (fps > 0.0 ? fps : 0.0, std::memory_order_relaxed);
}

void Viewport::setInterpMaxLongSide (int px)
{
    interpMaxLongSide_.store (px > 0 ? px : 0, std::memory_order_relaxed);
}

void Viewport::setValueFps (double fps)
{
    // Live project value-grid fps update (frame-perfect parity). impl_->valueFps
    // is atomic; the render loop re-reads it each frame. No-op when closed.
    if (impl_ != nullptr)
        impl_->valueFps.store (fps, std::memory_order_relaxed);
}

void Viewport::setScopeMask (uint32_t mask)
{
    scopeMask_ = mask & (kScopeWaveform | kScopeVectorscope | kScopeHistogram);
}

uint32_t Viewport::scopeMask() const
{
    return scopeMask_.load();
}

std::string Viewport::scopeDataJson() const
{
    const uint32_t mask = scopeMask_.load();
    json enabled = json::array();
    if (mask & kScopeWaveform)    enabled.push_back ("waveform");
    if (mask & kScopeVectorscope) enabled.push_back ("vectorscope");
    if (mask & kScopeHistogram)   enabled.push_back ("histogram");

    json out = {
        { "open", impl_ != nullptr && impl_->windowOpen.load() },
        { "enabled", enabled },
        { "seq", 0 },
        { "sampleWidth", videorender::FrameRenderer::kScopeW },
        { "sampleHeight", videorender::FrameRenderer::kScopeH },
        { "samples", videorender::FrameRenderer::kScopeW
                         * videorender::FrameRenderer::kScopeH },
    };
    if (impl_ == nullptr)
        return out.dump();

    std::lock_guard<std::mutex> lock (impl_->mutex);
    out["seq"] = impl_->scopeSeq;
    if (impl_->scopeSeq == 0)
        return out.dump();

    if ((mask & kScopeWaveform)
        && impl_->scopeWaveform.size() == (size_t) kScopeWaveW * kScopeWaveH)
    {
        out["waveform"] = base64Encode (impl_->scopeWaveform.data(),
                                        impl_->scopeWaveform.size());
        out["waveformWidth"] = kScopeWaveW;
        out["waveformHeight"] = kScopeWaveH;
    }
    if ((mask & kScopeVectorscope)
        && impl_->scopeVector.size() == (size_t) kScopeVecN * kScopeVecN)
    {
        out["vectorscope"] = base64Encode (impl_->scopeVector.data(),
                                           impl_->scopeVector.size());
        out["vectorscopeSize"] = kScopeVecN;
    }
    if (mask & kScopeHistogram)
    {
        json r = json::array(), g = json::array(), b = json::array();
        for (int i = 0; i < 256; ++i)
        {
            r.push_back (impl_->scopeHistogram[0][i]);
            g.push_back (impl_->scopeHistogram[1][i]);
            b.push_back (impl_->scopeHistogram[2][i]);
        }
        out["histogram"] = { { "r", r }, { "g", g }, { "b", b } };
    }
    return out.dump();
}

void Viewport::renderLoop (int width, int height, int x, int y,
                           bool alwaysOnTop, double targetFps)
{
    auto& im = *impl_;
    const bool shared = im.sharedMode;
    const bool shm = im.shmMode;
    const bool offscreen = shared || shm; // hidden window, no swapchain pacing

    if (glfwInit() != GLFW_TRUE)
    {
        im.openError = "glfwInit failed (no display?)";
        im.openDone = true;
        return;
    }

    // OpenGL 4.3 core (4.1 on macOS, which caps desktop GL there) — the
    // existing #version 330 shaders compile unchanged in a 4.3 core context;
    // the bump unlocks compute/SSBO/image-load-store where available (P1).
#if defined (__APPLE__)
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
#else
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE); // macOS requirement, harmless elsewhere
    glfwWindowHint (GLFW_FLOATING, alwaysOnTop ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint (GLFW_RESIZABLE, GLFW_TRUE);
    if (offscreen)
    {
        // Hidden window providing an offscreen context. We render into FBOs
        // and never swap, so the window being unmapped is fine. shm mode
        // deliberately takes the platform default context API (GLX/WGL/CGL
        // all fine — readback is plain GL).
        glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
#if ARBIT_HAVE_DMABUF && ! ARBIT_HAVE_VKFD
        // dmabuf export needs an EGL-backed context (eglCreateImage +
        // eglExportDMABUFImageMESA): on Wayland GLFW is EGL anyway, on X11
        // the hint requests EGL-on-X11. The IOSurface/D3D11 exporters use
        // the platform default (CGL/WGL). The VkFd backend imports via
        // GL_EXT_memory_object_fd (core GL, GLX or EGL) so it does NOT force
        // EGL — forcing EGL-on-X11 can fail on the NVIDIA driver, the very
        // case VkFd exists to serve, so it is skipped when VkFd is compiled.
        if (shared)
            glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#endif
    }
    GLFWwindow* win = glfwCreateWindow (width, height, "Arbit Video", nullptr, nullptr);
    if (win == nullptr)
    {
        im.openError = shared
            ? "glfwCreateWindow failed (EGL GL 4.3 context unavailable?)"
            : "glfwCreateWindow failed (GL 4.3 core unavailable?)";
        glfwTerminate();
        im.openDone = true;
        return;
    }
    if (! offscreen && x >= 0 && y >= 0)
        glfwSetWindowPos (win, x, y);
    glfwMakeContextCurrent (win);
    glfwSwapInterval (offscreen ? 0 : 1); // window mode: vsync paces the loop

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        im.openError = "GL core loader failed, missing: " + missing;
        glfwDestroyWindow (win);
        glfwTerminate();
        im.openDone = true;
        return;
    }
    // P1: soft-load the optional 4.3 entry points and snapshot the caps once.
    arbitgl::loadGl43Functions (gl);
    {
        arbitgl::GpuCaps caps;
        arbitgl::queryGpuCaps (gl, caps);
        std::lock_guard<std::mutex> lock (im.mutex);
        im.gpuCaps = caps;
    }

    // Shrink an shm render size to what one ring slot can hold (aspect kept).
    auto clampToSlot = [&im] (int& w, int& h)
    {
        const auto* hdr = im.shmRing.header();
        if (hdr == nullptr)
            return;
        const double need = (double) w * (double) h * 4.0;
        if (need <= (double) hdr->slotBytes)
            return;
        const double s = std::sqrt ((double) hdr->slotBytes / need);
        w = std::max (16, (int) ((double) w * s));
        h = std::max (16, (int) ((double) h * s));
    };

    videorender::FrameRenderer renderer;
    int fbW = 0, fbH = 0;
    if (offscreen)
    {
        fbW = width;
        fbH = height;
        if (shm)
            clampToSlot (fbW, fbH);
    }
    else
        glfwGetFramebufferSize (win, &fbW, &fbH);

    std::string rendererError;
    if (! renderer.initialize (&gl, std::max (fbW, 1), std::max (fbH, 1), rendererError))
    {
        im.openError = "renderer init failed: " + rendererError;
        glfwDestroyWindow (win);
        glfwTerminate();
        im.openDone = true;
        return;
    }

#if ARBIT_HAVE_GPU_SHARED
    // Shared-mode bring-up: capability probe, exported buffer ring, fd socket.
    auto destroySharedBuffers = [&im]
    {
        for (auto& b : im.buffers)
            im.exporter.destroy (b);
        im.buffers.clear();
    };
    auto allocateSharedBuffers = [&im] (int w, int h, std::string& err) -> bool
    {
        im.buffers.resize ((size_t) im.sharedBufferCount);
        for (auto& b : im.buffers)
            if (! im.exporter.allocate (b, w, h, err))
                return false;
        return true;
    };
    auto sendBuffers = [&im]
    {
        // BUFFERS = BuffersPayload + N BufferInfo records; fds attached in
        // buffer order, planes within a buffer in plane order.
        std::vector<uint8_t> payload (sizeof (gpusurf::BuffersPayload)
                                      + im.buffers.size() * sizeof (gpusurf::BufferInfo));
        auto* bp = (gpusurf::BuffersPayload*) payload.data();
        bp->bufferCount = (uint32_t) im.buffers.size();
        auto* infos = (gpusurf::BufferInfo*) (payload.data() + sizeof (gpusurf::BuffersPayload));
        int fds[gpusurf::kMaxBuffers * gpusurf::kMaxPlanes];
        int fdCount = 0;
        for (size_t i = 0; i < im.buffers.size(); ++i)
        {
            const auto& b = im.buffers[i];
            auto& bi = infos[i];
            bi = gpusurf::BufferInfo {};
            bi.index = (uint32_t) i;
            bi.fourcc = b.fourcc;
            bi.width = (uint32_t) b.width;
            bi.height = (uint32_t) b.height;
            bi.modifier = b.modifier;
            bi.planeCount = (uint32_t) b.planeCount;
            for (int p = 0; p < b.planeCount; ++p)
            {
                bi.planes[p].strideBytes = b.strides[p];
                bi.planes[p].offsetBytes = b.offsets[p];
                fds[fdCount++] = b.fds[p];
            }
        }
        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::Buffers,
                             payload.data(), payload.size(), fds, fdCount);
    };
    auto sendHello = [&im]
    {
        gpusurf::HelloPayload hello;
        if (im.exporter.fenceSyncAvailable())
            hello.flags |= gpusurf::kFlagFenceSync;
        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::Hello, &hello, sizeof (hello));
    };

    if (shared)
    {
        std::string err;
        if (! im.exporter.initialize (&gl, err))
        {
            im.openError = "no shared-GPU export support: " + err;
            renderer.shutdown();
            glfwDestroyWindow (win);
            glfwTerminate();
            im.openDone = true;
            return;
        }
        im.dmabufCapable = true;

        if (! im.fdSocket.listen (im.sharedResult.socketPath, err)
            || ! allocateSharedBuffers (fbW, fbH, err))
        {
            im.openError = err;
            destroySharedBuffers();
            im.fdSocket.close();
            im.exporter.shutdown();
            renderer.shutdown();
            glfwDestroyWindow (win);
            glfwTerminate();
            im.openDone = true;
            return;
        }
        im.sharedResult.fourcc = im.buffers[0].fourcc;
        im.sharedResult.modifier = im.buffers[0].modifier;
        im.sharedResult.bufferCount = (int) im.buffers.size();
        im.sharedResult.device = im.exporter.devicePath();
    }
#endif // ARBIT_HAVE_GPU_SHARED

    im.windowOpen = true;
    im.openDone = true;

    const double frameBudgetSec = targetFps > 0.0 ? 1.0 / targetFps : 1.0 / 60.0;
    std::deque<int64_t> frameTimes;
    DecodedFrame df; // decode scratch, reused across frames
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    std::vector<uint8_t> interpScratch; // tier-2 fetch scratch (worker result -> GL upload)
#endif
    // Retime tier actually drawn this frame (reported via interpBackend):
    // 0=nearest, 1=frame-blend, 2=rife-cuda, 3=rife-coreml, 4=rife-vulkan.
    int interpTierThisFrame = 0;

    // Per-segment decode cursor: persistent decoder + GL texture per clipId,
    // so several segments (even of the same file) decode independently when
    // composited together.
    struct ClipStream
    {
        std::string path;
        std::unique_ptr<MediaContext> media;
        unsigned tex = 0;
        int texW = 0, texH = 0;
        double lastSrcSec = -1.0e9; // pts of the uploaded frame
        bool openFailed = false;
        double seqFps = 0.0;        // image-sequence hint the stream was opened with
        int seqStart = -1;
        // Frame Blend (retime tier 1): two bracket-frame textures straddling
        // srcSec (texA=earlier pts, texB=later) + the blend output size. tex
        // holds the alpha-mixed result the compositor samples.
        unsigned texA = 0, texB = 0;
        double blendF0Pts = -1.0e9, blendF1Pts = -1.0e9;
        bool blendPairValid = false;
        int blendW = 0, blendH = 0;
        // tex currently holds a SYNTHESIZED frame (frame-blend or RIFE) that
        // represents srcSec exactly rather than a single decoded pts. Lets the
        // nearest path know it must re-decode if it takes over, while keeping
        // lastSrcSec = srcSec so the A/V-offset diagnostic stays honest (~0).
        bool texSynthesized = false;
    };
    std::map<int, ClipStream> streams; // render-thread only, keyed by clipId
    uint64_t streamsGeneration = ~0ull;

    // One gathered layer = an active segment + a params snapshot (+ optional
    // transition A side).
    struct ActiveLayer
    {
        ViewportSegment seg;
        ClipGraphParams params;
        bool transitionActive = false;
        double transitionProgress = 0.0;
        bool haveFrom = false;
        ViewportSegment fromSeg;
        ClipGraphParams fromParams;
    };

    // Decode the frame for a segment at srcSec into its clip stream;
    // returns nullptr when the source cannot be opened / has no frame yet.
    auto decodeLayer = [&] (const ViewportSegment& s, double srcSec) -> ClipStream*
    {
        // Generator clips (gen:// sentinel) carry no decodable media — their
        // pixels come from the shader/effect path, not a MediaContext. Skip
        // decode (matches the exporter). Live shader rendering in the viewport
        // is a later slice; for now the layer is simply empty here.
        if (s.sourcePath.rfind ("gen://", 0) == 0)
            return nullptr;

        auto& cs = streams[s.clipId];
        if (cs.path != s.sourcePath || cs.seqFps != s.sourceFps || cs.seqStart != s.seqStart)
        {
            cs.media.reset();
            cs.path = s.sourcePath;
            cs.seqFps = s.sourceFps;
            cs.seqStart = s.seqStart;
            cs.openFailed = false;
            cs.texW = 0;
            cs.lastSrcSec = -1.0e9;
            cs.blendPairValid = false;
            cs.blendW = cs.blendH = 0;
        }
        if (cs.media == nullptr && ! cs.openFailed)
        {
            cs.media = std::make_unique<MediaContext>();
            if (! cs.media->open (s.sourcePath, true, s.sourceFps, s.seqStart).empty())
            {
                cs.media.reset();
                cs.openFailed = true;
            }
        }
        if (cs.media == nullptr)
            return nullptr;

        const double fps = std::max (cs.media->info().fps, 1.0);

        // Should this segment be interpolated this frame? Two regimes:
        //   * Project target FPS set (interpTargetFps_ > 0): frame-rate
        //     up-conversion — interpolate whenever the segment's effective
        //     cadence (sourceFps * rate) is below the target, so a 24 fps clip in
        //     a 60 fps project is smoothed at ANY speed (mirrors the exporter
        //     trigger sourceFps * rate < job.fps).
        //   * No target (== 0): legacy slow-motion-only gate (rate < 0.999) — no
        //     behavior change until the user opts into a target.
        // The chosen engine is still per-clip retimeQuality (tier 2 RIFE / tier 1
        // Frame Blend / 0 = leave alone), gated below.
        const double interpTargetFps = interpTargetFps_.load (std::memory_order_relaxed);
        const bool belowTarget = (interpTargetFps > 0.0)
                                     ? (fps * s.rate < interpTargetFps * 0.999)
                                     : (s.rate < 0.999);

#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
        // Speed Warp (retime tier 2): async RIFE. The render thread only reads
        // finished frames + posts requests; all decode/inference run on the
        // worker (its own decoder + RifeEngine, no GL context). On a cache miss
        // it falls through to Frame Blend (tier 1) below, so the picture is never
        // worse than tier 1 while the worker warms up. Suspended only while an
        // export holds the GPU (a resource gate, not a quality decision).
        //
        // NO fps-based auto-demote: the user's chosen tier is authoritative. We
        // never silently drop RIFE because the frame rate dipped — that's a
        // terrible surprise. (A cache miss still falls back to Frame Blend for the
        // single frame, then catches up; the *selected* tier stays tier 2.)
        if (s.retimeQuality >= 2 && belowTarget
            && ! interpSuspended_.load (std::memory_order_relaxed))
        {
            if (im.interp == nullptr)
            {
                // Publish the pointer under im.mutex (info() reads it under that
                // lock on the RPC thread); construct/start outside so the lock
                // never covers thread spawn or model build.
                auto engine = std::make_unique<arbitinterp::InterpEngine>();
                {
                    std::lock_guard<std::mutex> lk (im.mutex);
                    im.interp = std::move (engine);
                }
                im.interp->start();   // worker thread does the model build, off this thread
            }
            if (im.interp->available())
            {
                // Interpolate at the render-surface size, capped to
                // interpMaxLongSide_ (0 ⇒ engine default 360p). The source frame
                // is aspect-fit inside an ipw×iph box; we never oversample beyond
                // the panel. Higher cap = crisper preview, slower inference —
                // realtime 4K can't hold 60 (the EXPORT path is uncapped/full-res).
                int ipw = 0, iph = 0;
                if (const int cap = interpMaxLongSide_.load (std::memory_order_relaxed); cap > 0)
                {
                    const int eff = std::min (cap, std::max (width, height));
                    ipw = iph = std::max (16, eff);
                }
                const arbitinterp::InterpEngine::Request req {
                    s.clipId, s.sourcePath, s.sourceFps, s.seqStart, srcSec, ipw, iph };
                int rw = 0, rh = 0;
                if (im.interp->fetch (s.clipId, srcSec, interpScratch, rw, rh) && rw > 0)
                {
                    cs.tex = renderer.uploadRgba (interpScratch.data(), rw, rh,
                                                  rw * 4, cs.tex);
                    cs.texW = rw;
                    cs.texH = rh;
                    cs.lastSrcSec = srcSec;        // synthesized frame == srcSec exactly
                    cs.texSynthesized = true;      // re-decode if nearest takes over
                    cs.blendPairValid = false;     // hand cs.tex back cleanly to tier 1
                    cs.blendW = cs.blendH = 0;     // force a resize-correct realloc there
                    interpTierThisFrame = im.interp->tierCode();   // 2 cuda / 3 coreml / 4 vulkan
                    im.interp->request (req);      // keep the bracket(s) around srcSec warm
                    return &cs;
                }
                im.interp->request (req);          // miss: warm it, fall to Frame Blend
            }
        }
#endif

        // Frame Blend (retime tier 1): when the segment is slowed below the
        // display rate, GL alpha-mix the two source frames bracketing srcSec
        // (ports the exporter rifeDecodeLayer bracket machine — two getFrames +
        // forward-advance + rebuild-on-jump) instead of holding the nearest
        // frame. Any decode/seek failure leaves blendPairValid=false and falls
        // through to the nearest-frame path below.
        if (s.retimeQuality >= 1 && belowTarget)
        {
            const double frameDur = 1.0 / fps;
            // Next distinct frame after afterPts (steps widen for pts jitter);
            // rejects a size change so the bracket pair stays equally sized.
            auto decodeNextAfter = [&] (double afterPts, DecodedFrame& out,
                                        int w, int h) -> bool
            {
                for (const double k : { 1.05, 1.6, 2.3 })
                    if (cs.media->getFrame (afterPts + k * frameDur, 1920, 1080, out).empty()
                        && out.width > 0 && out.ptsSec > afterPts + 1e-6
                        && (w <= 0 || (out.width == w && out.height == h)))
                        return true;
                return false;
            };
            if (! cs.blendPairValid || srcSec < cs.blendF0Pts - 1e-6
                || srcSec > cs.blendF1Pts + 8.0 * frameDur)
            {
                // (Re)build the bracket pair around srcSec.
                cs.blendPairValid = false;
                if (cs.media->getFrame (srcSec, 1920, 1080, df).empty() && df.width > 0)
                {
                    const int w = df.width, h = df.height;
                    cs.texA = renderer.uploadRgba (df.rgba.data(), w, h,
                                                   df.strideBytes, cs.texA);
                    cs.blendF0Pts = df.ptsSec;
                    DecodedFrame df1;
                    if (decodeNextAfter (df.ptsSec, df1, w, h))
                    {
                        cs.texB = renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                       df1.height, df1.strideBytes, cs.texB);
                        cs.blendF1Pts = df1.ptsSec;
                        if (cs.blendW != w || cs.blendH != h)
                        {
                            if (cs.tex != 0) renderer.deleteTexture (cs.tex);
                            cs.tex = 0;
                            cs.blendW = w;
                            cs.blendH = h;
                        }
                        cs.texW = w;
                        cs.texH = h;
                        cs.blendPairValid = true;
                    }
                }
            }
            else
            {
                // Forward-advance the pair so it keeps straddling srcSec as slow
                // playback walks through source frames.
                int guard = 0;
                while (cs.blendPairValid && srcSec >= cs.blendF1Pts && guard++ < 16)
                {
                    cs.blendF0Pts = cs.blendF1Pts;
                    std::swap (cs.texA, cs.texB);   // old f1 becomes the new f0
                    DecodedFrame df1;
                    if (! decodeNextAfter (cs.blendF0Pts, df1, cs.blendW, cs.blendH))
                    {
                        cs.blendPairValid = false;  // clip tail: nearest covers it
                        break;
                    }
                    cs.texB = renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                   df1.height, df1.strideBytes, cs.texB);
                    cs.blendF1Pts = df1.ptsSec;
                }
            }

            if (cs.blendPairValid && srcSec < cs.blendF1Pts && cs.blendW > 0)
            {
                const double span = std::max (1e-9, cs.blendF1Pts - cs.blendF0Pts);
                const float mix = (float) std::clamp (
                    (srcSec - cs.blendF0Pts) / span, 0.0, 1.0);
                cs.tex = renderer.frameBlendInto (cs.texA, cs.texB, cs.tex,
                                                  cs.blendW, cs.blendH, mix);
                cs.lastSrcSec = srcSec;  // blended frame == srcSec exactly
                cs.texSynthesized = true; // re-decode if nearest takes over
                interpTierThisFrame = 1;
                return &cs;
            }
        }

        // Nearest-frame retime (v1): skip the decode when the wanted time is
        // within half a frame of the uploaded frame's pts. A synthesized frame
        // (blend/RIFE) currently in cs.tex always forces a re-decode here.
        if (cs.texW == 0 || cs.texSynthesized
            || std::abs (srcSec - cs.lastSrcSec) >= 0.5 / fps)
        {
            if (cs.media->getFrame (srcSec, 1920, 1080, df).empty() && df.width > 0)
            {
                cs.tex = renderer.uploadRgba (df.rgba.data(), df.width, df.height,
                                              df.strideBytes, cs.tex);
                cs.texW = df.width;
                cs.texH = df.height;
                cs.lastSrcSec = df.ptsSec;
                cs.texSynthesized = false;
            }
        }
        return cs.texW > 0 ? &cs : nullptr;
    };

    // Common per-clip params (everything except the source texture) — shared by
    // decoded-media layers and shader-generator layers (which carry no texture;
    // their pixels come from the compiled ShaderGenerator in renderComposite).
    auto fillDescCommon = [&im] (videorender::LayerDesc& d, const ClipGraphParams& p,
                                 double displaySec, int clipId)
    {
        d.clipId = clipId;             // keys the feedback-trail history texture
        d.scale = p.scale;
        d.translateX = p.translateX;
        d.translateY = p.translateY;
        d.rotationDeg = p.rotationDeg;
        d.cropLeft = p.cropLeft;
        d.cropRight = p.cropRight;
        d.cropTop = p.cropTop;
        d.cropBottom = p.cropBottom;
        d.opacity = p.opacity;
        d.blendMode = p.blendMode;
        d.effects = p.effects;
        d.effectCount = videofx::kMaxEffectSlots;
        d.timeSec = displaySec;
        d.maskType = p.maskType;
        d.maskCx = p.maskCx;
        d.maskCy = p.maskCy;
        d.maskW = p.maskW;
        d.maskH = p.maskH;
        d.maskFeather = p.maskFeather;
        d.maskInvert = p.maskInvert != 0;
        d.genParams = p.genParams;   // ISF/generator INPUTS (M7) — values for this frame
        // tex/size are render-thread-owned (serviced below under the mutex),
        // safe to read here on the render thread.
        if (const auto it = im.luts.find (clipId); it != im.luts.end())
        {
            d.lutTexture = it->second.tex;
            d.lutSize = it->second.size;
        }
    };

    auto fillDesc = [&fillDescCommon] (videorender::LayerDesc& d, const ClipGraphParams& p,
                                       const ClipStream& cs, double displaySec, int clipId)
    {
        d.texture = cs.tex;
        d.texWidth = cs.texW;
        d.texHeight = cs.texH;
        fillDescCommon (d, p, displaySec, clipId);
    };

    // Shader-generator programs compiled on this (the render) thread, keyed by
    // clipId -> the source last compiled. Lazy-compiled when a gen://shader
    // segment's source first appears or changes; cleared when the clip leaves.
    std::map<int, std::string> shaderCompiled;

    // ISF INPUT defaults discovered at compile (clipId -> name -> default),
    // populated from setClipShader's GenParam list. Seeded under any explicit
    // gen value before mod routings apply (mirrors the exporter's clipGenDefaults
    // in stateAt) so a routing targeting a never-set ISF input still finds a base.
    // Persists across frames; refreshed only when a shader (re)compiles.
    std::map<int, std::map<std::string, double>> clipGenDefaults;

    // M5 Block C live score (render-thread owned). The packer is a stateful
    // voice allocator advanced once per timeline frame; unlike the exporter it
    // is NOT warmed from frame 0 (a live viewport has no canonical t=0), so row
    // assignment after a backward scrub may differ from a forward playthrough —
    // the export path stays the deterministic ground truth. localScore/localGen
    // mirror the control-plane copy so we only re-copy + reset on a real change.
    arbitblockc::BlockCPacker scorePacker;
    arbitmod::Score localScore;
    double localScoreLookahead = 0.0;
    uint64_t scoreGenSeen = ~0ull;
    long scorePackedFrame = -1;
    videorender::NoteFeatures cachedNoteFeatures;

    // M6 mod matrix (render-thread owned). localRoutings mirrors the control-
    // plane copy; routingStates carry the one-pole smoothing memory advanced
    // once per timeline frame. Not warmed from frame 0 (live has no canonical
    // t=0) — the export path stays the deterministic ground truth.
    std::vector<arbitmod::Routing> localRoutings;
    std::vector<arbitmod::RoutingState> routingStates;
    uint64_t modGenSeen = ~0ull;
    long modAdvancedFrame = -1;

    // P2 Scripts-tab per-frame hook (render-thread owned). The selected engine is
    // (re)compiled only when the script set changes; scriptOverrides holds the
    // CURRENT timeline frame's {paramId->value}, applied as the TOP layer in
    // overlayMods (after the mod matrix — mirrors exporter stateAt). The hook runs
    // once per unique timeline frame (scriptRanFrame dedups paused/repeated
    // frames); it is NOT warmed from frame 0 like the exporter (live has no
    // canonical t=0) so cross-frame globals are best-effort, export is ground
    // truth. Both engine objects exist; only the selected one is compiled/run.
    arbitlua::LuaHook scriptLua;
    arbitjs::JsHook   scriptJs;
    std::string localScriptSource;
    std::string localScriptLang;
    uint64_t scriptGenSeen = ~0ull;
    bool scriptActive = false;          // a hook compiled OK and has frame()
    bool scriptWantJs = true;           // which engine the current source selects
    bool scriptErrLogged = false;       // throttle frame() error spam to once
    long scriptRanFrame = -1;           // last timeline frame frame() was run for
    std::map<std::string, double> scriptOverrides;     // this frame's overrides
    std::vector<arbitlua::HookNote> scriptNotes;       // scratch (rebuilt per run)
    std::vector<arbitlua::HookLink> scriptLinks;       // scratch (rebuilt per run)

    // M4 Block B live audio (render-thread owned). The analyzer is stepped each
    // frame with whatever new mix samples arrived in the audio ring; its latest
    // FeatureFrame is cached as audio features for both audio-reactive shaders
    // (uRMS/uPeak/uOnset/uAudioBands) and audio-sourced mod routings. Like the
    // packers above it is NOT frame-0 warmed — it tracks live playback, so the
    // export path (whole-WAV analysis) stays the deterministic ground truth. The
    // read cursor snaps forward while paused so resuming never replays silence.
    arbitblockb::BlockBAnalyzer audioAnalyzer;
    std::vector<arbitblockb::FeatureFrame> audioFrameScratch;
    std::vector<float> audioDrainBuf (videoshm::kAudioRingCapacity, 0.0f);
    uint64_t audioReadCursor = 0;
    uint32_t audioAnalyzerSr = 0;
    videorender::AudioFeatures cachedAudioFeatures;

    // Frame-perfect audio parity (sweep): render-thread snapshot of the baked
    // master mix (Impl::previewMixFeats). Re-copied only when previewMixGeneration
    // changes (rare — only on a re-bake), since the vector is large. While the
    // transport is STOPPED and this is non-empty, the loop reads features from
    // HERE at the frame's time (previewMixFeatureAt) instead of zero-feeding —
    // making a stopped/scrubbed preview frame match export at the same time.
    std::vector<arbitblockb::FeatureFrame> localPreviewFeats;
    double localPreviewSr = 0.0;
    uint64_t previewMixGenSeen = ~0ull;

    int frameCounter = 0;
    int glErrorsLogged = 0;
    int64_t nextTickNs = nowNs(); // shared-mode pacing (no vsync without a swap)
    int64_t nextScopeNs = 0;      // ~10 Hz scope readback pacing
    std::vector<uint8_t> scopeRgba; // scope readback scratch, reused

    while (running_.load() && ! im.wantClose.load() && ! glfwWindowShouldClose (win))
    {
        glfwPollEvents();

        // Window control requests from the RPC thread (window mode only).
        if (! offscreen)
        {
            if (im.boundsDirty.exchange (false))
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                glfwSetWindowPos (win, im.reqX, im.reqY);
                glfwSetWindowSize (win, im.reqW, im.reqH);
            }
            const int fsReq = im.fullscreenRequest.exchange (-1);
            if (fsReq == 1)
            {
                auto* mon = glfwGetPrimaryMonitor();
                const auto* mode = glfwGetVideoMode (mon);
                glfwSetWindowMonitor (win, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
            }
            else if (fsReq == 0)
            {
                glfwSetWindowMonitor (win, nullptr, 100, 100, width, height, 0);
            }
        }

        // shm-mode resize: just a render-size change — frames self-describe
        // their dims per slot, so no ring renegotiation is needed as long as
        // a frame fits one slot (clamped; Arbit recreates the ring to grow).
        if (shm && im.shmResizeDirty.exchange (false))
        {
            int newW = im.shmReqW.load();
            int newH = im.shmReqH.load();
            if (newW >= 16 && newH >= 16)
            {
                clampToSlot (newW, newH);
                fbW = newW;
                fbH = newH;
            }
        }

#if ARBIT_HAVE_GPU_SHARED
        if (shared)
        {
            // Socket pump: greet new clients, process FRAME_RELEASE, detect
            // disconnect (consumer gone => every announced buffer is free
            // again; the dmabuf memory itself stays valid via our own fds).
            if (im.fdSocket.poll())
            {
                sendHello();
                sendBuffers();
                for (auto& b : im.buffers)
                    b.busy = false;
            }
            for (;;)
            {
                FdSocketServer::Received msg;
                const auto r = im.fdSocket.recvMsg (msg);
                if (r == FdSocketServer::RecvResult::NoData)
                    break;
                if (r == FdSocketServer::RecvResult::Disconnected)
                {
                    for (auto& b : im.buffers)
                        b.busy = false;
                    break;
                }
                for (int fd : msg.fds) // v1 expects no fds from the consumer
                    closeFdIfValid (fd);
                if (msg.type == (uint32_t) gpusurf::MsgType::FrameRelease
                    && msg.payload.size() >= sizeof (gpusurf::FrameReleasePayload))
                {
                    gpusurf::FrameReleasePayload rel;
                    std::memcpy (&rel, msg.payload.data(), sizeof (rel));
                    if (rel.bufferIndex < im.buffers.size())
                        im.buffers[rel.bufferIndex].busy = false;
                }
            }

            // Resize renegotiation: tear everything down and re-announce.
            if (im.sharedResizeDirty.exchange (false))
            {
                const int newW = im.sharedReqW.load();
                const int newH = im.sharedReqH.load();
                if (newW > 0 && newH > 0 && (newW != fbW || newH != fbH))
                {
                    if (im.fdSocket.hasClient())
                        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::BuffersGone, nullptr, 0);
                    destroySharedBuffers();
                    std::string err;
                    if (! allocateSharedBuffers (newW, newH, err))
                    {
                        std::fprintf (stderr, "[viewport] shared resize failed: %s\n", err.c_str());
                        im.wantClose = true;
                        continue;
                    }
                    fbW = newW;
                    fbH = newH;
                    renderer.setOutputSize (fbW, fbH);
                    if (im.fdSocket.hasClient())
                        sendBuffers();
                }
            }

            // Nobody connected yet (or consumer died): don't burn the GPU.
            if (! im.fdSocket.hasClient())
            {
                std::this_thread::sleep_for (std::chrono::milliseconds (20));
                continue;
            }
        }
#endif // ARBIT_HAVE_GPU_SHARED

        const int64_t frameStartNs = nowNs();

        // --- Clock: read transport and extrapolate against the wall clock.
        videoshm::TransportBlock tb {};
        bool haveClock = false;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            for (int attempt = 0; attempt < 3 && ! haveClock; ++attempt)
                haveClock = im.transport.isOpen() && im.transport.read (tb);
        }

        // Canonical VALUE-grid frame rate (the export job.fps), read live each
        // frame so a project-fps change re-snaps the preview immediately. Drives
        // every per-frame value feed (modulation / shader clock / score / lua /
        // source selection) so the live preview lands on the SAME frame grid the
        // exporter steps — preview == export by construction. Falls back to the
        // display targetFps (legacy 60) when no project fps was supplied; the
        // DISPLAY refresh still uses targetFps independently (frameBudgetSec).
        const double vfRaw = im.valueFps.load (std::memory_order_relaxed);
        const double valueFps = vfRaw > 0.0 ? vfRaw
                                            : (targetFps > 0.0 ? targetFps : 30.0);

        double displaySec = 0.0, bpm = 120.0, beatsPerBar = 4.0;
        bool playing = false;
        if (haveClock)
        {
            bpm = tb.bpm > 0.0 ? tb.bpm : 120.0;
            beatsPerBar = tb.beatsPerBar > 0.0 ? tb.beatsPerBar : 4.0;
            playing = tb.playing != 0;
            double beats = tb.playheadBeats;
            if (playing)
            {
                const double elapsed = (double) (frameStartNs - tb.hostTimeNs) * 1e-9;
                beats += std::max (0.0, std::min (elapsed, 0.5)) * bpm / 60.0;
            }
            displaySec = beats * 60.0 / bpm;
        }
        // Frame-perfect (faithful preview): snap the value-path clock to the
        // project frame grid so the live frame is sampled at exactly the time
        // the exporter steps (t = n/valueFps). This removes the continuous
        // wall-clock vs n/fps divergence (audit findings #1/#14/#15) and makes
        // the whole frame internally coherent: srcSec, makeShaderClock,
        // d.timeSec, the routing/lua/score beats and transition progress all
        // derive from this one snapped displaySec. The DISPLAY still repaints at
        // targetFps — it simply holds the current project-fps frame between grid
        // steps. Snapping a zero/paused displaySec is harmless (rounds to a grid
        // point), and matches what the exporter would render for that region.
        displaySec = (double) std::llround (displaySec * valueFps) / valueFps;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.displaySec = displaySec;
            im.transportOpen = haveClock;
            im.transportPlaying = playing;
            im.transportGeneration = haveClock ? tb.generation.load (std::memory_order_relaxed) : 0;
            im.transportAgeSec = haveClock ? std::max (0.0, (double) (frameStartNs - tb.hostTimeNs) * 1e-9) : 0.0;
            im.transportPlayheadBeats = haveClock ? tb.playheadBeats : 0.0;
        }

        // --- M4 Block B live: drain the master-mix ring into the analyzer and
        // cache this frame's audio features. While playing we step the analyzer
        // with the newly-arrived samples so its onset/flux state stays warm; the
        // latest FeatureFrame feeds audio-reactive shaders and audio-sourced mod
        // routings below. While paused no audio flows, so we snap the read cursor
        // forward (resume never replays the silence backlog), reset the analyzer,
        // and zero-feed — a paused preview frame stays stable. Samples are copied
        // out under `im.mutex` (guards against attachAudio remapping the region);
        // the FFT runs outside the lock.
        bool haveAudio = false;
        {
            int drained = 0;
            uint32_t sr = 0;
            bool ringOpen = false;
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                ringOpen = im.audioRing.isOpen();
                if (ringOpen && playing)
                {
                    int n;
                    while (drained < (int) audioDrainBuf.size()
                           && (n = im.audioRing.read (audioDrainBuf.data() + drained,
                                                      (int) audioDrainBuf.size() - drained,
                                                      &audioReadCursor, &sr)) > 0)
                        drained += n;
                }
                else if (ringOpen)
                {
                    audioReadCursor = im.audioRing.writeCursor();
                }
                // Frame-perfect audio parity (sweep): refresh the baked-mix
                // snapshot under the same lock. Only on a re-bake (rare), so the
                // large copy never happens per-frame.
                if (im.previewMixGeneration != previewMixGenSeen)
                {
                    localPreviewFeats = im.previewMixFeats;
                    localPreviewSr    = im.previewMixSr;
                    previewMixGenSeen = im.previewMixGeneration;
                }
            }
            if (ringOpen && playing)
            {
                if (sr > 0 && sr != audioAnalyzerSr)
                {
                    audioAnalyzer.setSampleRate ((float) sr);
                    audioAnalyzerSr = sr;
                }
                if (drained > 0)
                {
                    audioFrameScratch.clear();
                    audioAnalyzer.pushSamples (audioDrainBuf.data(), drained, audioFrameScratch);
                    if (! audioFrameScratch.empty())
                    {
                        const arbitblockb::FeatureFrame& f = audioFrameScratch.back();
                        cachedAudioFeatures.rms      = f.rms;
                        cachedAudioFeatures.peak     = f.peak;
                        cachedAudioFeatures.onset    = f.onset;
                        cachedAudioFeatures.onsetAge = f.onsetAge;
                        cachedAudioFeatures.bands.assign (f.bands.begin(), f.bands.end());
                        // Determinism (audit #3): when several analysis hops land
                        // in one drain, keeping only .back() drops an onset pulse
                        // that fired on an earlier hop — so an onset-triggered mod
                        // silently misses under load. OR the pulse across every
                        // frame produced this tick (onsetAge already tracks the
                        // freshest age, so the continuous features stay .back()).
                        for (const auto& af : audioFrameScratch)
                            if (af.onset) { cachedAudioFeatures.onset = true; break; }
                    }
                }
                haveAudio = ! cachedAudioFeatures.bands.empty();
            }
            else if (ringOpen)
            {
                if (audioAnalyzerSr != 0) { audioAnalyzer.reset(); audioAnalyzerSr = 0; }
                cachedAudioFeatures = videorender::AudioFeatures {};
            }
        }

        // Frame-perfect audio parity (sweep): the offline-analyzed baked mix,
        // sampled at a timeline time exactly as the exporter's audioFeatureAt does
        // (idx = floor(t·sr/kHop) − 1; sub-frame-0 clamps to frame 0 — the same
        // parity contract). Returns the zero-feed when no mix is set.
        auto previewMixFeatureAt = [&] (double sec) -> videorender::AudioFeatures
        {
            videorender::AudioFeatures af;
            if (localPreviewFeats.empty() || localPreviewSr <= 0.0)
                return af;
            const double targetSample = std::max (0.0, sec) * localPreviewSr;
            long idx = (long) std::floor (targetSample / (double) arbitblockb::kHop) - 1;
            if (idx < 0) idx = 0;
            if (idx >= (long) localPreviewFeats.size()) idx = (long) localPreviewFeats.size() - 1;
            const arbitblockb::FeatureFrame& f = localPreviewFeats[(size_t) idx];
            af.rms = f.rms; af.peak = f.peak; af.onset = f.onset; af.onsetAge = f.onsetAge;
            af.bands.assign (f.bands.begin(), f.bands.end());
            return af;
        };
        // While STOPPED/scrubbing no live audio flows, so the drain above zero-fed
        // (or held) cachedAudioFeatures — diverging from export. If a baked mix is
        // present, feed THIS frame's offline feature instead so the stopped frame
        // matches export at the same time. While playing the live ring stays
        // authoritative (zero added monitoring latency — the chosen design).
        const bool useBakedAudio = (! playing && ! localPreviewFeats.empty());
        if (useBakedAudio)
        {
            cachedAudioFeatures = previewMixFeatureAt (displaySec);
            haveAudio = ! cachedAudioFeatures.bands.empty();
        }

        // Block B mod source for M6 routings (built once; fed at the current
        // frame in the advance loop — see below).
        arbitmod::Audio liveAud {};
        if (haveAudio)
        {
            liveAud.rms   = cachedAudioFeatures.rms;
            liveAud.peak  = cachedAudioFeatures.peak;
            liveAud.onset = cachedAudioFeatures.onset;
            for (size_t i = 0; i < cachedAudioFeatures.bands.size() && i < 64; ++i)
                liveAud.bands[i] = cachedAudioFeatures.bands[i];
        }

        // --- Gather every segment active at displaySec + transition sources.
        std::vector<ActiveLayer> act;
        std::vector<videorender::ImageLayerDesc> overlays; // active text layers
        uint64_t timelineGen = 0;
        std::set<int> liveClipIds;
        {
            std::lock_guard<std::mutex> lock (im.mutex);

            // Apply pending timestamped params whose beat has passed.
            const double playheadBeats = haveClock ? displaySec * bpm / 60.0 : 0.0;
            auto& pend = im.pendingParams;
            pend.erase (std::remove_if (pend.begin(), pend.end(),
                [&] (const PendingParam& pp)
                {
                    if (pp.atBeat > playheadBeats) return false;
                    int cid; std::string node, par;
                    if (Impl::parseParamId (pp.paramId, cid, node, par))
                        im.apply (cid, node, par, pp.value);
                    else if (Impl::parseTextParamId (pp.paramId, cid, par))
                        im.applyText (cid, par, pp.value);
                    return true;
                }), pend.end());

            timelineGen = im.timelineGeneration;
            if (timelineGen != streamsGeneration)
                for (const auto& s : im.segments)
                    liveClipIds.insert (s.clipId);

            // M5: re-copy the live score + reset the packer only when the owned
            // note set actually changed (cheap per-frame check vs a vector copy).
            if (im.scoreGeneration != scoreGenSeen)
            {
                localScore = im.score;
                localScoreLookahead = im.scoreLookaheadBeats;
                scoreGenSeen = im.scoreGeneration;
                scorePacker.reset();
                scorePackedFrame = -1;
                cachedNoteFeatures = videorender::NoteFeatures {};
            }

            // M6: re-copy the routings + reset their smoothing state only when
            // the routing set changed. The per-clip overlay happens in the
            // segment gather loop below.
            if (im.modGeneration != modGenSeen)
            {
                localRoutings = im.routings;
                modGenSeen = im.modGeneration;
                routingStates.assign (localRoutings.size(), arbitmod::RoutingState {});
                modAdvancedFrame = -1;
            }

            // P2: re-copy + recompile the per-frame script hook only when the
            // script set changed (compiling is the heavy step — never per frame).
            // The engine is Lua OR JS (scriptLang selects; empty source ⇒ no
            // hook). A compile error logs once and leaves the hook inert
            // (scriptActive false), so the preview just shows the un-scripted
            // graph — never a crash. Mirrors exporter.cpp's compile-once block.
            if (im.scriptGeneration != scriptGenSeen)
            {
                localScriptSource = im.scriptSource;
                localScriptLang   = im.scriptLang;
                scriptGenSeen     = im.scriptGeneration;
                scriptActive      = false;
                scriptWantJs      = (localScriptLang != "lua");
                scriptErrLogged   = false;
                scriptRanFrame    = -1;
                scriptOverrides.clear();
                scriptLua.close();
                scriptJs.close();
                if (! localScriptSource.empty())
                {
                    std::string serr;
                    scriptActive = scriptWantJs
                        ? scriptJs.compile (localScriptSource, serr)
                        : scriptLua.compile (localScriptSource, serr);
                    if (! scriptActive)
                        std::fprintf (stderr, "[%s] live hook disabled: %s\n",
                                      scriptWantJs ? "js" : "lua", serr.c_str());
                }
            }

            // Advance each routing's smoothing/LFO/envelope state to the current
            // timeline frame (once per frame, monotonic — same as the exporter).
            // The discarded combined value leaves RoutingState::smoothed == the
            // modulation value the overlay reads. Audio-sourced routings see the
            // live mix at the current frame (liveAud, built above) — warm-up
            // frames on a scrub still zero-feed since there is no audio history;
            // score-sourced routings see the live notes. The warm-up window is
            // sized to the slowest smoother's convergence (see below) so the
            // smoothed value matches the frame-0-warmed export to epsilon.
            if (! localRoutings.empty())
            {
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const long g = (long) std::llround (displaySec * clkFps);
                const double dtBeats = (1.0 / clkFps) * bpm / 60.0;
                // Playhead seeked backward (scrub-back, looping, or a viewport
                // reopen whose first frame renders at the previous playhead before
                // the user seeks earlier): modAdvancedFrame is monotonic, so
                // without this reset startF would exceed g and the advance loop
                // below would never run — freezing every routing's smoothed value
                // at its last position (the live mod matrix appears stuck). Reset
                // so the ~1 s warm-up window re-runs ending at the new frame.
                if (g < modAdvancedFrame)
                    modAdvancedFrame = -1;
                // Warm-up window (audit #13): a flat ~1s was too short for slow
                // one-pole smoothers, so the first preview frames after a play-
                // start / seek differed from the export (which warms from frame
                // 0). A one-pole's initial-state influence decays as exp(-T/tau),
                // so ~9.2·tau of warm-up leaves <1e-4 residual — i.e. the smoothed
                // value matches the frame-0-warmed export to epsilon. Size the
                // window to the slowest enabled smoother (0 ⇒ nothing to warm),
                // capped at 30s so a pathologically slow tau can't spin the loop.
                // (Audio/score-sourced routings still zero-feed during warm-up —
                // no live history — exactly as before; the export stays ground
                // truth for those, matching the Pass-3 audio limitation.)
                float maxTauBeats = 0.0f;
                for (const auto& r : localRoutings)
                    if (r.enabled) maxTauBeats = std::max (maxTauBeats, r.smoothingBeats);
                long warmFrames = 0;
                if (maxTauBeats > 0.0f)
                {
                    const double warmBeats = 9.2 * (double) maxTauBeats;
                    warmFrames = (long) std::ceil (warmBeats * 60.0 * clkFps
                                                   / std::max (bpm, 1.0));
                    warmFrames = std::min (warmFrames,
                                           (long) std::llround (clkFps * 30.0));
                }
                const long startF = std::max (modAdvancedFrame + 1, g - warmFrames);
                const arbitmod::Audio zeroAud {};
                for (long f = startF; f <= g; ++f)
                {
                    arbitmod::Clock clk;
                    clk.beat        = (float) ((double) f / clkFps * bpm / 60.0);
                    clk.bpm         = (float) bpm;
                    clk.beatsPerBar = (float) beatsPerBar;
                    // Audio per frame. While PLAYING only the current frame has real
                    // (live-ring) audio; warm-up frames have no history → zero-fed.
                    // While STOPPED with a baked mix, feed EVERY frame the offline
                    // feature at its time — exactly as the exporter's
                    // advanceRoutingsToFrame does — so audio-sourced routings warm to
                    // the same smoothed state export produces (frame-perfect parity).
                    arbitmod::Audio bakedAud {};
                    const arbitmod::Audio* audPtr;
                    if (useBakedAudio)
                    {
                        const videorender::AudioFeatures vaf =
                            previewMixFeatureAt ((double) f / clkFps);
                        bakedAud.rms = vaf.rms; bakedAud.peak = vaf.peak; bakedAud.onset = vaf.onset;
                        const int nb = std::min (64, (int) vaf.bands.size());
                        for (int b = 0; b < nb; ++b) bakedAud.bands[(size_t) b] = vaf.bands[(size_t) b];
                        audPtr = &bakedAud;
                    }
                    else
                        audPtr = (f == g) ? &liveAud : &zeroAud;
                    for (size_t ri = 0; ri < localRoutings.size(); ++ri)
                    {
                        if (! localRoutings[ri].enabled) continue;
                        arbitmod::evaluateRouting (localRoutings[ri], routingStates[ri],
                                                   0.0f, localScore, clk, *audPtr, (float) dtBeats);
                    }
                }
                if (g > modAdvancedFrame) modAdvancedFrame = g;
            }

            // P2: run the per-frame script hook ONCE for this timeline frame and
            // capture its {paramId->value} overrides (applied per clip as the TOP
            // layer in overlayMods below — mirrors exporter.cpp::stateAt). Deduped
            // on the rounded timeline frame so a paused/repeated frame is stable
            // and the script's ctx.frame does not race the render rate. The ctx is
            // built field-for-field like the exporter's: live clock + the Block B
            // live mix (liveAud, zero-fed when no audio) + the notes SOUNDING at
            // this beat from the live Block C score, with their JI cents/ratio.
            if (scriptActive)
            {
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const long g = (long) std::llround (displaySec * clkFps);
                if (g != scriptRanFrame)
                {
                    scriptRanFrame = g;
                    arbitlua::FrameCtx ctx;
                    ctx.t           = displaySec;
                    ctx.beat        = displaySec * bpm / 60.0;
                    ctx.bpm         = bpm;
                    ctx.beatsPerBar = beatsPerBar;
                    ctx.frame       = g;
                    ctx.rootFreq    = localScore.rootFreq;
                    // Block B live audio (zero-fed when no ring / before playback).
                    ctx.rms = liveAud.rms; ctx.peak = liveAud.peak; ctx.onset = liveAud.onset;
                    ctx.onsetAge = haveAudio ? cachedAudioFeatures.onsetAge : 0.0f;
                    const int liveBands = haveAudio
                        ? (int) std::min<size_t> (64, cachedAudioFeatures.bands.size()) : 0;
                    ctx.bands = liveBands > 0 ? liveAud.bands : nullptr;
                    ctx.bandCount = liveBands;
                    // Block C score: links (score-global) + notes sounding at this beat.
                    scriptLinks.clear();
                    for (const auto& l : localScore.links)
                    {
                        arbitlua::HookLink hl;
                        hl.slaveNoteId = l.slaveNoteId; hl.masterNoteId = l.masterNoteId;
                        hl.num = l.slaveHarmonic; hl.den = l.masterHarmonic;
                        hl.ratio = l.masterHarmonic != 0
                                 ? (double) l.slaveHarmonic / l.masterHarmonic : 0.0;
                        scriptLinks.push_back (hl);
                    }
                    scriptNotes.clear();
                    const float beatF = (float) ctx.beat;
                    for (const auto& n : localScore.notes)
                        if (n.activeAt (beatF))
                        {
                            arbitlua::HookNote hn;
                            hn.midi = n.midiNote; hn.freq = n.freqHz; hn.velocity = n.velocity;
                            hn.cents = arbitmod::centsFromRoot (n.freqHz, localScore.rootFreq);
                            hn.age = ctx.beat - n.startBeat;
                            hn.trackId = n.trackId; hn.ratioNum = n.ratioNum; hn.ratioDen = n.ratioDen;
                            hn.isRoot = n.isRoot;
                            for (int p = 0; p < 6; ++p) hn.primes[p] = n.primes[p];
                            scriptNotes.push_back (hn);
                        }
                    ctx.notes = scriptNotes.empty() ? nullptr : scriptNotes.data();
                    ctx.noteCount = (int) scriptNotes.size();
                    ctx.links = scriptLinks.empty() ? nullptr : scriptLinks.data();
                    ctx.linkCount = (int) scriptLinks.size();
                    scriptOverrides.clear();
                    std::string serr;
                    const bool sok = scriptWantJs
                        ? scriptJs.runFrame (ctx, scriptOverrides, serr)
                        : scriptLua.runFrame (ctx, scriptOverrides, serr);
                    if (! sok && ! scriptErrLogged)
                    {
                        std::fprintf (stderr, "[%s] live frame() error: %s\n",
                                      scriptWantJs ? "js" : "lua", serr.c_str());
                        scriptErrLogged = true;
                    }
                }
            }

            // M6: lay each routing targeting this clip over the per-frame param
            // copy (combine(mode, base, smoothed) — the same overlay the exporter
            // applies in stateAt). routingStates is already advanced to this
            // frame. A routing to an unknown/unreadable param is skipped (readFrom
            // returns false) rather than written onto a junk base. The stored
            // base (im.clipParams) is never mutated, so disabling/removing a
            // routing restores the underlying value next frame.
            auto overlayMods = [&] (ClipGraphParams& p, int clipId)
            {
                // Seed ISF INPUT defaults under any explicit gen value (emplace never
                // overwrites) so a routing to a never-set input modulates around the
                // shader's default — exactly as exporter.cpp::stateAt does. Defaults
                // come from the prior compile of this clip's shader (clipGenDefaults).
                if (const auto dit = clipGenDefaults.find (clipId); dit != clipGenDefaults.end())
                    for (const auto& [k, v] : dit->second)
                        p.genParams.emplace (k, v);
                for (size_t ri = 0; ri < localRoutings.size(); ++ri)
                {
                    const arbitmod::Routing& r = localRoutings[ri];
                    if (! r.enabled) continue;
                    int cid; std::string node, par;
                    if (! Impl::parseParamId (r.destination, cid, node, par)) continue;
                    if (cid != clipId) continue;
                    double base = 0.0;
                    if (! Impl::readFrom (p, node, par, base)) continue;
                    Impl::applyTo (p, node, par,
                                   arbitmod::combine (r.mode, (float) base,
                                                      routingStates[ri].smoothed));
                }
                // P2: the per-frame script hook is the TOP layer — apply this
                // frame's overrides for THIS clip after static/baked/mod-matrix
                // (mirrors exporter.cpp::stateAt). scriptOverrides was computed
                // once for this timeline frame above; a script always wins where
                // it sets a value.
                for (const auto& [pid, value] : scriptOverrides)
                {
                    int cid; std::string node, par;
                    if (! Impl::parseParamId (pid, cid, node, par)) continue;
                    if (cid != clipId) continue;
                    Impl::applyTo (p, node, par, value);
                }
            };

            constexpr double kAbutEps = 1e-3;
            for (const auto& s : im.segments)
            {
                const double segDur = (s.outSec - s.inSec) / std::max (s.rate, 1e-9);
                if (! (displaySec >= s.displayStartSec
                       && displaySec < s.displayStartSec + segDur))
                    continue;

                ActiveLayer al;
                al.seg = s;
                al.params = *im.paramsFor (s.clipId);
                overlayMods (al.params, s.clipId);

                // Leading-edge transition window of this segment.
                if (s.transitionType != 0 && s.transitionDurationSec > 0.0
                    && displaySec < s.displayStartSec + s.transitionDurationSec)
                {
                    al.transitionActive = true;
                    al.transitionProgress = std::clamp (
                        (displaySec - s.displayStartSec) / s.transitionDurationSec,
                        0.0, 1.0);

                    // A = the previous segment on the same trackLayer when it
                    // abuts/overlaps; otherwise A stays empty (= black).
                    const ViewportSegment* best = nullptr;
                    for (const auto& c : im.segments)
                        if (&c != &s && c.trackLayer == s.trackLayer
                            && c.displayStartSec < s.displayStartSec
                            && (best == nullptr
                                || c.displayStartSec > best->displayStartSec))
                            best = &c;
                    if (best != nullptr)
                    {
                        const double bestDur =
                            (best->outSec - best->inSec) / std::max (best->rate, 1e-9);
                        if (best->displayStartSec + bestDur
                            >= s.displayStartSec - kAbutEps)
                        {
                            al.haveFrom = true;
                            al.fromSeg = *best;
                            al.fromParams = *im.paramsFor (best->clipId);
                            overlayMods (al.fromParams, best->clipId);
                        }
                    }
                }
                act.push_back (std::move (al));
            }

            // --- Per-clip LUTs: service staged uploads/clears (GL context is
            // current on this thread).
            for (auto it = im.luts.begin(); it != im.luts.end();)
            {
                auto& l = it->second;
                if (l.dirty)
                {
                    if (l.path.empty() || l.pendingSize < 2)
                    {
                        if (l.tex != 0)
                            renderer.deleteTexture (l.tex);
                        l.tex = 0;
                        l.size = 0;
                    }
                    else
                    {
                        l.tex = renderer.uploadLut3D (l.pendingData.data(),
                                                      l.pendingSize, l.tex);
                        l.size = l.pendingSize;
                    }
                    l.pendingData.clear();
                    l.pendingData.shrink_to_fit();
                    l.dirty = false;
                }
                if (l.path.empty() && l.tex == 0)
                    it = im.luts.erase (it);
                else
                    ++it;
            }

            // --- Text overlays: service uploads/removals (GL context is
            // current on this thread) and gather the time-active ones.
            for (auto it = im.texts.begin(); it != im.texts.end();)
            {
                auto& t = it->second;
                if (t.removed)
                {
                    if (t.tex != 0)
                        renderer.deleteTexture (t.tex);
                    it = im.texts.erase (it);
                    continue;
                }
                if (t.pixelsDirty)
                {
                    t.tex = renderer.uploadRgba (t.pendingPixels.data(),
                                                 t.pendingW, t.pendingH,
                                                 t.pendingW * 4, t.tex);
                    t.texW = t.pendingW;
                    t.texH = t.pendingH;
                    t.pendingPixels.clear();
                    t.pendingPixels.shrink_to_fit();
                    t.pixelsDirty = false;
                }
                if (t.tex != 0 && t.visible >= 0.5f && t.opacity > 0.001f
                    && displaySec >= t.startSec
                    && displaySec < t.startSec + t.durationSec)
                {
                    videorender::ImageLayerDesc ov;
                    ov.texture = t.tex;
                    ov.width = t.texW;
                    ov.height = t.texH;
                    ov.posX = t.posX + t.translateX;
                    ov.posY = t.posY + t.translateY;
                    ov.opacity = t.opacity;
                    ov.zOrder = (int) t.zOrder;
                    ov.ownerClipId = t.ownerClipId;
                    overlays.push_back (ov);
                }
                ++it;
            }
        }

        // Texts composite above all video layers, ordered by zOrder (map
        // iteration keeps equal-zOrder texts stable by textId).
        std::stable_sort (overlays.begin(), overlays.end(),
                          [] (const videorender::ImageLayerDesc& a,
                              const videorender::ImageLayerDesc& b)
                          { return a.zOrder < b.zOrder; });

        // A segment serving as the A side of an active cross transition is
        // represented inside that transition — suppress its standalone layer
        // (matches the reference: during a transition the two frames render
        // blended, not stacked).
        {
            std::set<int> fromIds;
            for (const auto& al : act)
                if (al.haveFrom)
                    fromIds.insert (al.fromSeg.clipId);
            if (! fromIds.empty())
                act.erase (std::remove_if (act.begin(), act.end(),
                    [&] (const ActiveLayer& al)
                    {
                        return ! al.transitionActive
                            && fromIds.count (al.seg.clipId) > 0;
                    }), act.end());
        }

        // Composite order: (trackLayer, zOrder), displayStartSec for stability.
        std::sort (act.begin(), act.end(),
                   [] (const ActiveLayer& a, const ActiveLayer& b)
                   {
                       if (a.seg.trackLayer != b.seg.trackLayer)
                           return a.seg.trackLayer < b.seg.trackLayer;
                       if (a.params.zOrder != b.params.zOrder)
                           return a.params.zOrder < b.params.zOrder;
                       return a.seg.displayStartSec < b.seg.displayStartSec;
                   });

        // Drop decode streams for clips that left the timeline.
        if (timelineGen != streamsGeneration)
        {
            for (auto it = streams.begin(); it != streams.end();)
            {
                if (liveClipIds.count (it->first) == 0)
                {
                    renderer.deleteTexture (it->second.tex);
                    renderer.deleteTexture (it->second.texA);   // frame-blend brackets
                    renderer.deleteTexture (it->second.texB);
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
                    if (im.interp != nullptr) im.interp->forget (it->first); // tier-2 decoder + cache
#endif
                    it = streams.erase (it);
                }
                else
                    ++it;
            }
            // Free compiled shader programs for clips that left the timeline.
            for (auto it = shaderCompiled.begin(); it != shaderCompiled.end();)
            {
                if (liveClipIds.count (it->first) == 0)
                {
                    renderer.clearClipShader (it->first);
                    it = shaderCompiled.erase (it);
                }
                else
                    ++it;
            }
            streamsGeneration = timelineGen;
        }

        if (! offscreen)
            glfwGetFramebufferSize (win, &fbW, &fbH);
        if (fbW <= 0 || fbH <= 0)
        {
            std::this_thread::sleep_for (std::chrono::milliseconds (50));
            continue;
        }
        // Resolution parity (audit #16): composite at the CANVAS resolution —
        // exactly what the export does — so pixel-scale effects (blur/bloom
        // radius, feedback trails, particle size) match it, then frame the result
        // into the panel/window at fbW×fbH via the present target. With no canvas
        // set, fall back to compositing at the window size (legacy, no present
        // target). The composite is byte-identical to the export's because both
        // run renderComposite with output == canvas and a canvas-aspect ref.
        const int cw = canvasW_.load();
        const int ch = canvasH_.load();
        if (cw > 0 && ch > 0)
        {
            renderer.setOutputSize (cw, ch);
            renderer.setPresentSize (fbW, fbH);
        }
        else
        {
            renderer.setOutputSize (fbW, fbH);
            renderer.setPresentSize (0, 0);
        }
        // Project canvas + view transform (PROTOCOL.md §Project canvas &
        // view transform) — viewport-only; the exporter's own FrameRenderer
        // never sets these.
        renderer.setCanvas (cw, ch);
        renderer.setView ((float) viewZoom_.load(),
                          (float) viewPanX_.load(),
                          (float) viewPanY_.load());
        // Frame-global render settings mirrored from the export job so preview ==
        // export (master-plan principle #1): the canvas background colour and the
        // bloom/tonemap post stack. Neutral defaults skip the post pass and clear
        // to the editor charcoal — byte-identical to before these were wired.
        renderer.setBackgroundColor ((float) bgR_.load(), (float) bgG_.load(),
                                     (float) bgB_.load(), (float) bgA_.load());
        // Lay any mod routing targeting the global "post/<node>/<param>" namespace
        // over the static post values (combine(mode, base, smoothed) — the same
        // overlay the exporter applies per frame), so e.g. a kick->bloom routing
        // pulses the bloom in the live preview exactly as it will on export.
        // routingStates was advanced to this frame while `act` was built above.
        float pBloomI = (float) bloomIntensity_.load();
        float pBloomT = (float) bloomThreshold_.load();
        float pBloomR = (float) bloomRadius_.load();
        float pExp    = (float) exposure_.load();
        for (size_t ri = 0; ri < localRoutings.size(); ++ri)
        {
            const arbitmod::Routing& r = localRoutings[ri];
            if (! r.enabled) continue;
            if (r.destination.rfind ("post/", 0) != 0) continue;
            float* dst = r.destination == "post/bloom/intensity" ? &pBloomI
                       : r.destination == "post/bloom/threshold" ? &pBloomT
                       : r.destination == "post/bloom/radius"    ? &pBloomR
                       : r.destination == "post/grade/exposure"  ? &pExp
                                                                  : nullptr;
            if (dst == nullptr) continue;   // tonemap mode stays static (an int select)
            *dst = arbitmod::combine (r.mode, *dst, routingStates[ri].smoothed);
        }
        renderer.setPostFx (pBloomI, pBloomT, pBloomR, tonemap_.load(), pExp);

        // --- Decode all visible layers and build the renderer's layer list.
        interpTierThisFrame = 0; // decodeLayer raises this to the tier it drew
        std::vector<videorender::LayerDesc> descs;
        std::vector<videorender::LayerDesc> fromDescs;
        descs.reserve (act.size());
        fromDescs.reserve (act.size()); // stable: fromLayer pointers held by descs
        bool haveMain = false;
        double mainPts = 0.0, mainIdeal = 0.0;

        // Shader-generator clips (gen://shader): compile/refresh each active
        // program on this thread (GL current) and note which clipIds render via
        // the shader path. A clip stays a shader clip across frames until its
        // source text changes; a failed compile keeps the clip's last good
        // program (setClipShader) and is simply not re-added this frame.
        std::set<int> shaderClips;
        for (const auto& al : act)
        {
            if (al.seg.shaderSource.empty())
                continue;
            const auto it = shaderCompiled.find (al.seg.clipId);
            if (it == shaderCompiled.end() || it->second != al.seg.shaderSource)
            {
                std::string shaderLog;
                std::vector<videorender::GenParam> shaderParams;
                renderer.setClipShader (al.seg.clipId, al.seg.shaderSource,
                                        shaderLog, shaderParams);
                shaderCompiled[al.seg.clipId] = al.seg.shaderSource;

                // Capture this shader's ISF INPUT defaults (per component key, as
                // exporter.cpp does) so mod routings to an un-set input find a base.
                auto& defs = clipGenDefaults[al.seg.clipId];
                defs.clear();
                for (const auto& gp : shaderParams)
                {
                    const int nc = videorender::genParamComponentCount (gp.type);
                    for (int k = 0; k < nc; ++k)
                        defs[gp.name + videorender::genParamComponentSuffix (gp.type, k)] =
                            (nc == 1) ? gp.defaultV : gp.defaultVec[k];
                }
            }
            if (renderer.hasClipShader (al.seg.clipId))
                shaderClips.insert (al.seg.clipId);
        }

        // P4: particle clips also read the Block C score (to seed from a track's
        // notes), so the score must be packed when one is on-screen even with no
        // shader clip present.
        bool haveParticleClip = false;
        for (const auto& al : act)
            if (al.seg.sourcePath.rfind ("gen://particles", 0) == 0)
            { haveParticleClip = true; break; }

        // M5: pack the live Block C score once per timeline frame (only when a
        // shader or particle clip is on-screen to read it). The packed beat
        // matches makeShaderClock (displaySec·bpm/60); re-packing a held frame is
        // skipped via scorePackedFrame so a paused/repeated frame stays stable.
        const bool haveScore = (! shaderClips.empty() || haveParticleClip)
                             && ! localScore.notes.empty();
        if (haveScore)
        {
            const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
            const long g = (long) std::llround (displaySec * clkFps);
            if (g != scorePackedFrame)
            {
                const double beat = displaySec * bpm / 60.0;
                const arbitblockc::PackResult pr = scorePacker.pack (
                    localScore, (float) beat, (float) localScoreLookahead);
                cachedNoteFeatures.notesTex  = pr.notesTex;
                cachedNoteFeatures.linksTex  = pr.linksTex;
                cachedNoteFeatures.noteCount = pr.noteCount;
                cachedNoteFeatures.linkCount = pr.linkCount;
                cachedNoteFeatures.rootFreq  = localScore.rootFreq;
                scorePackedFrame = g;
            }
        }

        for (const auto& al : act)
        {
            if (al.params.visible < 0.5f || al.params.opacity <= 0.001f)
                continue;

            videorender::LayerDesc d;
            const bool isAdjustment = al.seg.sourcePath.rfind ("gen://adjustment", 0) == 0;
            const bool isParticles  = al.seg.sourcePath.rfind ("gen://particles", 0) == 0;
            const bool isShader = ! isAdjustment && ! isParticles
                               && shaderClips.count (al.seg.clipId) > 0;
            ClipStream* cs = nullptr;
            double srcSec = 0.0;
            if (isAdjustment)
            {
                // M8 "effect the world" in the live preview: no source of its
                // own — the shared FrameRenderer runs this clip's effect rack
                // over the composite beneath it (renderComposite's isAdjustment
                // branch), mixed back at the clip's opacity. Matches the export
                // path; the (trackLayer, zOrder) composite sort above placed it
                // after the layers it grades — same ordering as exporter.cpp.
                d.isAdjustment = true;
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
            }
            else if (isShader)
            {
                // Shader layer: no media decode; the Block A clock is a pure
                // function of display time + tempo — the SAME formula the
                // exporter uses, so preview == export. The Block C score is now
                // live too (cachedNoteFeatures, set below), and so is Block B
                // audio (cachedAudioFeatures, drained from the live mix ring).
                const double segDur = (al.seg.outSec - al.seg.inSec)
                                    / std::max (al.seg.rate, 1e-9);
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const int frameIdx = (int) std::llround (
                    (displaySec - al.seg.displayStartSec) * clkFps);
                d.shaderSource = true;
                d.shaderClock = videorender::makeShaderClock (
                    displaySec, al.seg.displayStartSec, segDur, bpm, beatsPerBar,
                    clkFps, playing, frameIdx);
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
                if (haveScore)
                {
                    d.noteFeatures = cachedNoteFeatures;
                    d.notesPresent = true;   // gate renderComposite reads (renderer.cpp:1726)
                }
                if (haveAudio)
                {
                    d.audioFeatures = cachedAudioFeatures;
                    d.audioPresent = true;   // uploads uRMS/uPeak/uOnset/uAudioBands
                }
            }
            else if (isParticles)
            {
                // P4 particle clip: a GPU compute pool, no media decode. Same
                // Block A clock formula as the exporter ⇒ preview == export; the
                // live Block C score (cachedNoteFeatures) seeds from the chosen
                // spawn track. v1 has no audio reactivity (Block B master-mix only).
                const double segDur = (al.seg.outSec - al.seg.inSec)
                                    / std::max (al.seg.rate, 1e-9);
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const int frameIdx = (int) std::llround (
                    (displaySec - al.seg.displayStartSec) * clkFps);
                d.particleSource = true;
                d.shaderClock = videorender::makeShaderClock (
                    displaySec, al.seg.displayStartSec, segDur, bpm, beatsPerBar,
                    clkFps, playing, frameIdx);
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
                if (haveScore)
                {
                    d.noteFeatures = cachedNoteFeatures;
                    d.notesPresent = true;
                }
            }
            else
            {
                srcSec = al.seg.inSec
                    + (displaySec - al.seg.displayStartSec) * al.seg.rate;
                cs = decodeLayer (al.seg, srcSec);
                if (cs == nullptr)
                    continue;
                fillDesc (d, al.params, *cs, displaySec, al.seg.clipId);
            }

            if (al.transitionActive)
            {
                d.transitionType = al.seg.transitionType;
                d.transitionProgress = (float) al.transitionProgress;
                if (al.haveFrom && al.fromParams.visible >= 0.5f
                    && al.fromParams.opacity > 0.001f)
                {
                    // The outgoing clip ended where this one starts; clamp
                    // its source time into its own span (its last frame
                    // holds for abutting cuts).
                    double fromSrc = al.fromSeg.inSec
                        + (displaySec - al.fromSeg.displayStartSec) * al.fromSeg.rate;
                    fromSrc = std::clamp (fromSrc, al.fromSeg.inSec,
                                          std::max (al.fromSeg.inSec, al.fromSeg.outSec));
                    if (ClipStream* fs = decodeLayer (al.fromSeg, fromSrc))
                    {
                        videorender::LayerDesc fd;
                        fillDesc (fd, al.fromParams, *fs, displaySec, al.fromSeg.clipId);
                        fromDescs.push_back (fd);
                        d.fromLayer = &fromDescs.back();
                    }
                }
            }

            if (! haveMain && cs != nullptr)
            {
                haveMain = true;
                mainPts = cs->lastSrcSec;
                mainIdeal = srcSec;
            }
            descs.push_back (d);
        }

        // --- Composite + present. An empty layer list still renders the
        // background so the composited hash always reflects the frame shown.
        const unsigned compositeTex =
            renderer.renderComposite (descs.data(), (int) descs.size(),
                                      overlays.data(), (int) overlays.size());
        // Report which retime tier actually drew this frame (1=frame-blend).
        im.interpBackend.store (interpTierThisFrame, std::memory_order_relaxed);
        // Export-frame border + overscan dim (viewport-only; no-op when no
        // canvas is set). presentTex is what the user sees; scopes keep
        // measuring the undecorated composite.
        const unsigned presentTex = renderer.applyCanvasFrame (compositeTex);
        if (! offscreen)
            renderer.presentToWindow (presentTex, fbW, fbH);

        // frameHash = async PBO hash of the undecorated composite (canvas-res,
        // == the export's composite for A/B parity; the hash readback region is
        // sized to outW_, which is now the canvas, so this must be the composite,
        // not the window-res presentTex — audit #16).
        if ((frameCounter++ & 3) == 0)
        {
            uint64_t h = 0;
            if (renderer.hashComposite (compositeTex, h))
                im.lastFrameHash = h;
        }

        // Video scopes (PROTOCOL.md §Scopes): downscaled async readback of
        // the same composited frame, ~10 Hz, ONLY while a scope is enabled.
        if (const uint32_t scopes = scopeMask_.load(); scopes != 0)
        {
            const int64_t tScope = nowNs();
            if (tScope >= nextScopeNs)
            {
                nextScopeNs = tScope + 100'000'000ll;
                if (renderer.readbackScope (compositeTex, scopeRgba))
                    im.computeScopes (scopes, scopeRgba.data(),
                                      videorender::FrameRenderer::kScopeW,
                                      videorender::FrameRenderer::kScopeH);
            }
        }

        for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError())
            if (glErrorsLogged < 16)
            {
                std::fprintf (stderr, "[viewport] GL error 0x%04x\n", err);
                ++glErrorsLogged;
            }

        // A/V offset: presented frame pts vs the ideal source time.
        if (haveMain)
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.sourcePtsSec = mainPts;
            im.sourceIdealSec = mainIdeal;
            if (playing)
                im.avOffsetSec = mainPts - mainIdeal;
        }

        if (! offscreen)
            glfwSwapBuffers (win);
        else if (shm)
        {
            // Pipelined PBO readback (one frame of latency): map the
            // previous frame's pixels and seqlock-write them into the next
            // ring slot. Round-robin slots; the reader scans for the highest
            // per-slot frameSeq (SlotHeader.reserved[0/1]) with an even
            // generation, so a torn slot is simply skipped.
            int pw = 0, ph = 0;
            if (const uint8_t* px = renderer.beginMappedReadback (presentTex, pw, ph))
            {
                auto* hdr = im.shmRing.header();
                const size_t bytes = (size_t) pw * (size_t) ph * 4;
                if (hdr != nullptr && hdr->slotCount > 0 && bytes > 0
                    && bytes <= hdr->slotBytes)
                {
                    const uint32_t idx = im.shmSlotCursor++ % hdr->slotCount;
                    auto* slot = im.shmRing.slot (idx);
                    slot->generation.fetch_add (1, std::memory_order_acq_rel); // odd
                    slot->width = (uint32_t) pw;
                    slot->height = (uint32_t) ph;
                    slot->strideBytes = (uint32_t) pw * 4;
                    slot->ptsSec = displaySec;
                    slot->mediaId = 0;
                    const uint64_t seq = ++im.shmFrameSeq;
                    slot->reserved[0] = (uint32_t) (seq & 0xffffffffu);
                    slot->reserved[1] = (uint32_t) (seq >> 32);
                    slot->reserved[2] = 1; // payload format: BGRA8, top row first
                    std::memcpy (im.shmRing.slotPayload (idx), px, bytes);
                    slot->generation.fetch_add (1, std::memory_order_acq_rel); // even
                    im.sharedFramesSent.fetch_add (1);
                }
                renderer.endMappedReadback();
            }
        }
#if ARBIT_HAVE_GPU_SHARED
        else
        {
            // FREE -> RENDERING -> QUEUED handoff: copy the composite into
            // the oldest free exported buffer, order the GPU work (v2: attach
            // an acquire fence fd to FRAME_READY so the consumer's GPU waits
            // without stalling our pipeline; fallback: glFinish as in v1),
            // announce. A buffer that has been announced and not yet released
            // is never written — when none are free the frame is dropped.
            gpuexp::ExportedBuffer* target = nullptr;
            int freeBuffers = 0;
            int busyBuffers = 0;
            for (auto& b : im.buffers)
            {
                if (b.busy)
                    ++busyBuffers;
                else
                {
                    ++freeBuffers;
                    if (target == nullptr)
                        target = &b;
                }
            }
            im.sharedFreeBuffers = freeBuffers;
            im.sharedBusyBuffers = busyBuffers;
            if (target != nullptr)
            {
                im.exporter.blit (presentTex, *target);

                int fenceFd = im.exporter.createNativeFenceFd();
                if (fenceFd < 0)
                    glFinish();

                gpusurf::FrameReadyPayload ready;
                ready.bufferIndex = (uint32_t) (target - im.buffers.data());
                ready.flags = fenceFd >= 0 ? gpusurf::kFlagFenceSync : 0;
                ready.frameSeq = ++im.frameSeq;
                ready.displaySec = displaySec;
                ready.frameHash = im.lastFrameHash.load();
                const bool sent = im.fdSocket.sendMsg (
                    (uint32_t) gpusurf::MsgType::FrameReady, &ready, sizeof (ready),
                    fenceFd >= 0 ? &fenceFd : nullptr, fenceFd >= 0 ? 1 : 0);
                if (fenceFd >= 0)
                    closeFdIfValid (fenceFd); // sendmsg dup'ed it (or send failed)
                if (sent)
                {
                    target->busy = true;
                    im.sharedFramesSent.fetch_add (1);
                }
                else
                    im.fdSocket.dropClient(); // send failed: treat as gone
            }
            else
                im.sharedFramesDroppedNoBuffer.fetch_add (1);
        }
#endif // ARBIT_HAVE_GPU_SHARED
        im.framesPresented.fetch_add (1);

        // Measured fps over a 1 s sliding window.
        const int64_t t = nowNs();
        frameTimes.push_back (t);
        while (! frameTimes.empty() && t - frameTimes.front() > 1'000'000'000ll)
            frameTimes.pop_front();
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.measuredFps = (double) frameTimes.size();
            im.curW = fbW; im.curH = fbH;
        }

        // Pacing: vsync paces window mode; offscreen modes (no swap) sleep
        // out the remainder of the frame budget against an absolute tick.
        if (offscreen)
        {
            const auto frameDurNs = (int64_t) (frameBudgetSec * 1.0e9);
            nextTickNs = std::max (nextTickNs + frameDurNs, t - frameDurNs);
            const int64_t sleepNs = nextTickNs - nowNs();
            if (sleepNs > 0)
                std::this_thread::sleep_for (std::chrono::nanoseconds (sleepNs));
        }
        else if (frameTimes.size() >= 2
                 && (double) (t - frameTimes[frameTimes.size() - 2]) * 1e-9 < frameBudgetSec * 0.25)
            std::this_thread::sleep_for (
                std::chrono::duration<double> (frameBudgetSec * 0.5));
    }

    im.windowOpen = false;
    for (auto& [clipId, cs] : streams)
    {
        renderer.deleteTexture (cs.tex);
        renderer.deleteTexture (cs.texA);   // frame-blend brackets
        renderer.deleteTexture (cs.texB);
    }
    streams.clear();
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    {
        // Clear the pointer under im.mutex (so a concurrent info() RPC sees
        // null), then let it destruct OUTSIDE the lock — ~InterpEngine joins the
        // worker (can block ~one inference), which must not be held under im.mutex.
        std::unique_ptr<arbitinterp::InterpEngine> dying;
        {
            std::lock_guard<std::mutex> lk (im.mutex);
            dying = std::move (im.interp);
        }
    }   // dying joins the tier-2 worker + frees its RIFE/CUDA session here
#endif
    {
        // Texts and LUTs are cleared on viewport close (Arbit re-pushes on
        // open via pushClipGraphState / text re-registration).
        std::lock_guard<std::mutex> lock (im.mutex);
        for (auto& [textId, t] : im.texts)
            if (t.tex != 0)
                renderer.deleteTexture (t.tex);
        im.texts.clear();
        for (auto& [clipId, l] : im.luts)
            if (l.tex != 0)
                renderer.deleteTexture (l.tex);
        im.luts.clear();
    }
#if ARBIT_HAVE_GPU_SHARED
    if (shared)
    {
        if (im.fdSocket.hasClient())
            im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::BuffersGone, nullptr, 0);
        destroySharedBuffers();
        im.fdSocket.close();
        im.exporter.shutdown();
    }
#endif
    renderer.shutdown();
    glfwDestroyWindow (win);
    glfwTerminate();
}
