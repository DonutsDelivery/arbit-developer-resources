#pragma once

// Viewport — helper-owned preview window with a per-frame render graph.
//
// The helper presents frames in its own OS window (GLFW + OpenGL 3.3 core)
// so pixels never cross the process boundary for preview. Arbit is
// control-plane only: it streams the timeline clock through the transport
// shm block (VideoFrameSharedMemory.h) and parameter values via
// graph_set_param / graph_set_effects.
//
// Render graph (v2, fixed topology per clip):
//   Source(clip) -> Retime/Interpolate -> Effects rack -> Transform2D -> Display
// Every parameter has a stable string ID "clip<id>/<node>/<param>" — see
// video-helper/PROTOCOL.md. All segments active at the display time are
// composited (ordered by trackLayer, then zOrder) through the GL 3.3
// FrameRenderer (renderer.h): per-clip effects uber-shader, crop/letterbox/
// transform, leading-edge transitions, and blend-mode compositing.
//
// frameHash (viewport_info) hashes the COMPOSITED output, not the decoded
// source frame: every few frames the render loop issues a small asynchronous
// PBO readback (first ~1 MB of the composited frame) and hashes the
// previously issued one (FNV-1a). Effect/param/transform changes therefore
// change the hash, but the value lags the visible frame by a few frames —
// tests should poll viewport_info after a short settle.

#include "renderer.h" // videorender::EffectSlotState (GL-gated, like this file)
#include "gpu_caps.h" // arbitgl::GpuCaps (P1 capability layer; dependency-free)
#include "mod_defs.h" // arbitmod::Score (M5 Block C live score push); plain C++17

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// One constant-rate retime span (same shape as ExportSegment — VideoBeatMath
// on the Arbit side is the single source of truth for the mapping).
struct ViewportSegment
{
    std::string sourcePath;
    int clipId = 0;
    double inSec = 0.0, outSec = 0.0, rate = 1.0, displayStartSec = 0.0;
    int retimeQuality = 0;              // retime interpolation tier (0=Nearest,
                                        // 1=Frame Blend, 2=Speed Warp/RIFE);
                                        // mirrors VideoSidecar ExportSegment
    int trackLayer = 0;                 // compositing layer (v2)
    int transitionType = 0;             // videofx::TransitionType wire value;
                                        // leading-edge transition (v2)
    double transitionDurationSec = 0.0;
    double sourceFps = 0.0;             // image sequences: pattern frame rate
                                        // (0 = not a sequence / demuxer default)
    int seqStart = -1;                  // image sequences: first frame number

    // Shader-generator source (M1 producer / M3 engine): non-empty ⇒ this is a
    // gen://shader clip whose pixels come from this GLSL program rendered with
    // the Block A clock, not a decoded MediaContext. Binds to the clip by
    // clipId and survives the whole-list resend (mirrors the export jobSpec's
    // clips[].shaderSource). Compiled lazily on the render thread.
    std::string shaderSource;
};

// Per-clip graph parameters (Transform2D + Source compositing + effects).
// Vocabulary mirrors the original editor's CompositorLayer/keyframe model.
struct ClipGraphParams
{
    float scale = 1.0f;
    float translateX = 0.0f;   // normalized, -1..1 of viewport width
    float translateY = 0.0f;
    float rotationDeg = 0.0f;
    float cropLeft = 0.0f, cropRight = 0.0f, cropTop = 0.0f, cropBottom = 0.0f;
    float opacity = 1.0f;
    float visible = 1.0f;
    float zOrder = 0.0f;
    int blendMode = 0;         // videofx::BlendMode wire value (honored in v2)
    videorender::EffectSlotState effects[videofx::kMaxEffectSlots];

    // Shape mask node ("clip<id>/mask/<param>") — PROTOCOL.md §Render graph.
    int maskType = 0;          // 0 none, 1 rect, 2 ellipse
    float maskCx = 0.5f, maskCy = 0.5f;
    float maskW = 0.8f, maskH = 0.8f;
    float maskFeather = 0.05f;
    int maskInvert = 0;

    // ISF/generator INPUT values for a gen://shader clip, keyed by the param name
    // (the "clip<id>/gen/<name>" sink) — mirrors the exporter's ClipRenderState
    // .genParams so the live viewport drives the shader's uniforms identically.
    // Keys absent here fall back to the shader's own ISF default at upload time.
    std::map<std::string, double> genParams;
};

// One slot description for graph_set_effects (bulk rack configuration).
struct EffectSlotSpec
{
    int slot = -1;
    int type = -1;             // videofx::EffectType, -1 = empty slot
    bool enabled = true;
    std::vector<std::pair<std::string, double>> params; // wire name -> value
};

struct ViewportInfo
{
    bool open = false;
    int width = 0, height = 0;
    double measuredFps = 0.0;
    double avOffsetSec = 0.0;        // presented pts vs ideal source time
    bool interpolationActive = false;
    std::string interpolationBackend; // "nearest" | "rife-cuda"
    // Retime tier 2 (async RIFE) worker diagnostics — 0/false unless the worker
    // has been started (lazily, on first tier-2 use). interpReady flips true once
    // RIFE initializes on a realtime EP; the counters confirm inference + cache.
    bool interpReady = false;
    std::string interpWorkerBackend;  // "" | "rife-cuda" | "rife-coreml" | "rife-cpu"
    uint64_t interpInferences = 0;
    uint64_t interpCacheHits = 0;
    uint64_t interpCacheMisses = 0;
    // Frame-rate up-conversion target (0 = legacy slowed-only gate, no target).
    double interpTargetFps = 0.0;
    // Why RIFE is unavailable ("" when ready/warming): lets the UI show the
    // fallback reason instead of a silent slide to Frame Blend.
    std::string interpError;
    std::string gpuPath;              // "opengl" (presentation API in use)
    uint64_t framesPresented = 0;
    uint64_t lastFrameHash = 0;
    double displaySec = 0.0;          // timeline time currently being rendered
    double sourcePtsSec = 0.0;        // decoded source pts for the primary layer
    double sourceIdealSec = 0.0;      // requested source time for the primary layer

    // Transport-shm diagnostics. measuredFps can stay smooth while these
    // values reveal that the media clock itself stopped advancing.
    bool transportOpen = false;
    bool transportPlaying = false;
    uint32_t transportGeneration = 0;
    double transportAgeSec = 0.0;
    double transportPlayheadBeats = 0.0;

    // Active rung of the present-path ladder as far as the helper can see:
    // "dmabuf-docked" (zero-copy shared offscreen mode), "shm-docked"
    // (offscreen render + CPU frame ring), "glfw-window" (own OS window),
    // "none" (closed). The Arbit side reports "cpu-shm" itself when it falls
    // back to the request_frame ring.
    std::string presentPath = "none";
    bool dmabufCapable = false;      // true once a dmabuf probe succeeded
    uint64_t sharedFramesSent = 0;
    uint64_t sharedFramesDroppedNoBuffer = 0;
    int sharedFreeBuffers = 0;
    int sharedBusyBuffers = 0;

    // Project canvas + viewport-only view transform (PROTOCOL.md §Project
    // canvas & view transform). canvas 0/0 = unset (composite at surface
    // size, legacy behavior).
    int canvasWidth = 0, canvasHeight = 0;
    double viewZoom = 1.0, viewPanX = 0.0, viewPanY = 0.0;

    // GPU capability snapshot taken once after context creation (P1). Lets the
    // plugin/agents gate compute-class UI and the export path flag
    // "compute unavailable" (e.g. macOS 4.1) rather than failing.
    arbitgl::GpuCaps gpuCaps;
};

class Viewport
{
public:
    Viewport();
    ~Viewport(); // ctor/dtor out-of-line: Impl is incomplete here

    // All methods are called from the RPC (stdin) thread; the render loop
    // runs on its own thread. Returns empty string on success.
    // clkFps is the project (canonical) value-grid frame rate — the rate the
    // exporter steps at. It drives the live preview's modulation/shader-clock/
    // score/lua/source-frame snapping so preview == export. 0 ⇒ fall back to the
    // display targetFps. Independent of targetFps, which only paces the window.
    std::string open (int width, int height, int x, int y, bool alwaysOnTop,
                      double targetFps, double clkFps = 0.0);
    void close();

    // Zero-copy docked mode (viewport_open_shared): renders the same graph
    // offscreen into shared GPU buffers and signals frames over a
    // helper-owned unix socket (SharedGpuSurfaceProtocol.h). The buffer
    // mechanism is per-platform (gpu_export_common.h): dmabuf-exported FBO
    // textures on Linux, global IOSurfaces on macOS, D3D11 keyed-mutex
    // shared textures via WGL_NV_DX_interop2 on Windows. Mutually exclusive
    // with open() — close() ends either mode. Returns empty string on
    // success; a non-empty error (no capable context/driver, or helper
    // built without a shared-GPU backend) tells Arbit to demote down the
    // present-path ladder (shm-docked, then glfw-window).
    struct SharedOpenResult
    {
        std::string socketPath;
        uint32_t fourcc = 0;
        uint64_t modifier = 0;
        int bufferCount = 0;
        std::string device;          // exporting DRM render node (may be empty)
    };
    std::string openShared (int width, int height, double targetFps,
                            int bufferCount, SharedOpenResult& out,
                            double clkFps = 0.0);  // project value-grid fps (see open())
    // Wire tag of the compiled shared-GPU backend ("dmabuf" / "iosurface" /
    // "d3d11"), or "" when none is compiled in — reported as gpuPath by
    // viewport_open_shared so Arbit can log/diagnose the mechanism.
    static const char* sharedGpuPathTag();
    // Reallocates the shared buffers (BUFFERS_GONE + fresh BUFFERS on the
    // socket). No-op error when shared mode is not active.
    std::string resizeShared (int width, int height);

    // CPU shared-memory docked mode (viewport_open_shm): renders the same
    // graph offscreen and writes BGRA frames into an Arbit-owned
    // VideoFrameSharedMemory ring via pipelined PBO readback (one frame of
    // latency, one CPU copy per frame). The portable docked rung — works on
    // GLX/X11, Windows and macOS, and inside DAW-hosted plugins, where
    // dmabuf export/import is unavailable. Mutually exclusive with open()/
    // openShared() — close() ends any mode. Render size is clamped so a
    // frame fits one ring slot. Returns empty string on success.
    std::string openShm (int width, int height, double targetFps,
                         const std::string& shmName,
                         double clkFps = 0.0);  // project value-grid fps (see open())
    // Changes the offscreen render size (no ring renegotiation — Arbit
    // recreates the ring and reopens when the new size outgrows a slot).
    std::string resizeShm (int width, int height);
    void setBounds (int x, int y, int width, int height);
    void setFullscreen (bool fullscreen);

    void setTimeline (std::vector<ViewportSegment> segments);
    void attachTransport (const std::string& shmName);

    // M4 Block B live: attach the audio-sample ring Arbit's audio thread writes
    // the master mix into. The render loop drains it into a BlockBAnalyzer each
    // frame and feeds the resulting features (uRMS/uPeak/uOnset/uAudioBands) to
    // audio-reactive shader clips — the live counterpart of the export path's
    // decodeWavToMonoFloat → analyzer. No-op until attached.
    void attachAudio (const std::string& shmName);

    // Frame-perfect audio parity (sweep): hand the viewport a baked master-mix
    // WAV. The live ring only carries audio WHILE PLAYING, so a STOPPED/scrubbed
    // preview historically zero-fed audio-reactive shaders/mods — diverging from
    // export, which reads the offline-analyzed mix at the frame's time. When a
    // mix is set, the viewport offline-analyzes it ONCE (videohelper::
    // analyzeMixWavOffline — the exporter's exact path) and, while stopped, feeds
    // those features instead of zero. Empty path clears it (back to zero-feed
    // when stopped). Decode/analyze runs on the calling RPC thread, not the
    // render thread. No effect while playing (live ring stays authoritative).
    void setAudioMix (const std::string& wavPath);

    // M5 Block C live preview: the owned note/link score that score-reactive
    // shader clips read (uNotes/uLinks/uNoteCount). Mirrors the export jobSpec's
    // `score`; Arbit pushes it whenever the owned notes/links change while the
    // viewport is open. The render loop packs it once per frame via the A2
    // BlockCPacker (audio-tier note features are zero until the Block B live shm
    // lands). Empty score = the zero-feed (uNoteCount 0), byte-identical to
    // never calling this.
    void setScore (arbitmod::Score score, double lookaheadBeats);

    // M6 mod-matrix live preview: the owned cross-domain routings (clock/score/
    // audio → clip param). Mirrors the export jobSpec's `modMatrix`; Arbit pushes
    // them when the owned routings change while the viewport is open. The render
    // loop evaluates each routing per frame (arbitmod::evaluateRouting) against
    // the live clock + score + audio (the live mix from the Block B audio ring)
    // and lays combine(mode, base, value) over the per-frame clip-param copy, so
    // routings layer on the live graph exactly as they do on export. Empty =
    // no overlay (identical to never calling this).
    void setModMatrix (std::vector<arbitmod::Routing> routings);

    // P2 Scripts-tab live preview: the project-global per-frame hook (Lua OR JS).
    // Mirrors the export jobSpec's luaScript/jsScript/scriptLang; Arbit pushes it
    // whenever the Scripts tab compiles a script (or on viewport open / helper
    // restart replay) while the viewport is open. The render loop compiles the
    // selected engine once, then runs frame(ctx) once per timeline frame against
    // the live clock + score + audio and applies the returned {paramId=value}
    // overrides as the TOP layer (after the mod matrix) — exactly as the exporter
    // does in stateAt, so preview == export. `lang` ∈ { "js", "lua" } (anything
    // else ⇒ "js"). Empty source = no hook (identical to never calling this).
    // Unlike the exporter the hook is NOT warmed from timeline frame 0 (a live
    // viewport has no canonical t=0), so a script keeping cross-frame globals is
    // best-effort live; the export path stays the deterministic ground truth.
    void setScript (std::string source, std::string lang);

    // Project canvas (viewport_set_canvas — PROTOCOL.md §Project canvas &
    // view transform): the export-frame coordinate space layers letterbox
    // against. Stored on the Viewport (not Impl) so it survives close/
    // reopen; applies on the next rendered frame. Returns "" on success.
    std::string setCanvas (int width, int height);

    // Project canvas background colour (viewport_set_canvas_background) + the
    // bloom/tonemap post stack (viewport_set_post). Both are FRAME-GLOBAL render
    // settings the exporter applies via FrameRenderer::setBackgroundColor /
    // setPostFx; the viewport must apply the SAME values each frame or preview !=
    // export (master-plan principle #1). Stored on the Viewport (not Impl), read
    // by the render loop every frame. Neutral defaults (post off, charcoal bg)
    // reproduce the prior behaviour exactly. Return "" on success.
    std::string setCanvasBackground (double r, double g, double b, double a);
    std::string setPostFx (double bloomIntensity, double bloomThreshold,
                           double bloomRadius, int tonemap, double exposure);

    // paramId = "clip<id>/<node>/<param>" or "text<id>/<param>". atBeat < 0
    // means immediate. Timestamped changes are accepted by the protocol now
    // (applied when the playhead passes atBeat); interpolation-mode hints are
    // parsed but v1 only implements immediate + linear-at-frame-rate
    // application. v2 nodes: "transform2d" (incl. crop), "source" (incl.
    // blendMode), "effect<slot>" (type/enabled/<wire param name>, slots 0-7).
    // Text params: opacity, posX, posY, translateX, translateY, visible, zOrder.
    std::string setParam (const std::string& paramId, double value, double atBeat);

    // Text overlays (PROTOCOL.md §Text overlays). Arbit rasterizes the text
    // with JUCE and ships straight (non-premultiplied) RGBA, top row first,
    // stride width*4; the helper composites it as an image layer at its
    // natural pixel size, centred at (posX, posY) NDC (+posY = down on
    // screen, matching LayerDesc::translateY). Re-sending the same textId
    // replaces the image/texture. Texts live independently of the timeline
    // (they survive viewport_set_timeline) and are cleared on close().
    std::string setTextImage (int textId, std::vector<uint8_t> rgba,
                              int width, int height,
                              double startSec, double durationSec,
                              double posX, double posY,
                              double opacity, double zOrder,
                              int ownerClipId = -1);
    std::string removeText (int textId);

    // Bulk (re)configure a clip's whole effects rack (graph_set_effects).
    std::string setEffects (int clipId, const std::vector<EffectSlotSpec>& slots);

    // Per-clip .cube 3D LUT (graph_set_lut). The file is parsed on the
    // calling (RPC) thread; the render thread uploads the 3D texture.
    // Empty path clears the clip's LUT. Returns empty string on success.
    std::string setLut (int clipId, const std::string& path);

    std::string describeGraph() const; // JSON string of nodes + params + effects

    ViewportInfo info() const;
    uint64_t frameHash() const;

    // Retime tier 2 (RIFE "Speed Warp") load-shed: while an export/proxy/render
    // job holds the GPU, suspend so the viewport drops to Frame Blend rather than
    // running a second CUDA session that contends for VRAM. Thread-safe.
    void setInterpolationSuspended (bool suspended);

    // Project-wide target frame rate for playback frame-rate up-conversion.
    // When > 0, ANY segment whose effective cadence (sourceFps * rate) is below
    // this target is interpolated UP to it — RIFE at retime tier 2, Frame Blend
    // at tier 1 — not just slowed clips. This mirrors the exporter's trigger
    // (sourceFps * rate < job.fps), so a 24 fps clip in a 60 fps project plays
    // smooth at ANY speed. 0 ⇒ legacy slowed-only gate (interpolate only when
    // rate < 0.999), i.e. no behavior change until a target is set. Thread-safe;
    // lives on the Viewport so it survives open/close. Per-clip retimeQuality
    // still selects the engine (0 = off for that clip).
    void setInterpTargetFps (double fps);

    // Max long-side (px) for REALTIME interpolation decode+inference. The preview
    // interpolates at the render-surface size, capped to this — higher = crisper
    // but slower (true 4K realtime is ~5 fps, can't hold 60; the EXPORT path is
    // uncapped and renders full-res). 0 ⇒ legacy 360p clamp. Thread-safe; on the
    // Viewport so it survives open/close.
    void setInterpMaxLongSide (int px);

    // Live update of the project (canonical) value-grid frame rate — the rate
    // the exporter steps at. Re-snaps the preview's modulation/shader-clock/
    // score/lua/source grid on the next rendered frame (frame-perfect parity).
    // Persistent like setCanvas: applies to the currently-open viewport. 0 ⇒
    // fall back to the display targetFps. No-op when no viewport is open.
    void setValueFps (double fps);

    // Video scopes (scope_enable / scope_data — PROTOCOL.md §Scopes).
    // Viewport-only diagnostics computed from a downscaled copy of the
    // composited frame at ~10 Hz, ONLY while mask != 0 and the viewport is
    // open. The mask lives on the Viewport (not Impl) so it survives
    // close/reopen. scopeDataJson() returns the latest arrays as a JSON
    // object string (the render thread publishes results under the mutex).
    enum ScopeBits : uint32_t
    {
        kScopeWaveform    = 1u,
        kScopeVectorscope = 2u,
        kScopeHistogram   = 4u,
    };
    void setScopeMask (uint32_t mask);
    uint32_t scopeMask() const;
    std::string scopeDataJson() const;

private:
    void renderLoop (int width, int height, int x, int y, bool alwaysOnTop,
                     double targetFps);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread thread_;
    std::atomic<bool> running_ { false };
    std::atomic<uint32_t> scopeMask_ { 0 };

    // Project canvas + viewport-only view transform ("view/zoom|panX|panY"
    // params). On the Viewport (not Impl) so they survive close/reopen; the
    // render loop reads them every frame.
    std::atomic<int> canvasW_ { 0 }, canvasH_ { 0 };
    std::atomic<double> viewZoom_ { 1.0 }, viewPanX_ { 0.0 }, viewPanY_ { 0.0 };

    // Frame-global render settings mirrored from the export job for preview ==
    // export (see setCanvasBackground / setPostFx). Defaults = the editor
    // charcoal background + a neutral (skipped) post pass.
    std::atomic<double> bgR_ { 0.04 }, bgG_ { 0.04 }, bgB_ { 0.05 }, bgA_ { 1.0 };
    std::atomic<double> bloomIntensity_ { 0.0 }, bloomThreshold_ { 1.0 },
                        bloomRadius_ { 0.0 }, exposure_ { 1.0 };
    std::atomic<int> tonemap_ { 0 };

    // Retime tier 2 (RIFE) load-shed flag. Lives on the Viewport (not Impl) so it
    // survives open/close and is safe to set from the export WORKER thread via
    // setInterpolationSuspended() without touching impl_ (which close() resets).
    std::atomic<bool> interpSuspended_ { false };

    // Project-wide playback frame-rate up-conversion target (see
    // setInterpTargetFps). 0 ⇒ legacy slowed-only interpolation. On the Viewport
    // (not Impl) so it survives open/close; the render loop reads it every frame.
    std::atomic<double> interpTargetFps_ { 0.0 };

    // Realtime interpolation resolution cap (long side, px); see
    // setInterpMaxLongSide. 0 ⇒ engine default 360p clamp.
    std::atomic<int> interpMaxLongSide_ { 0 };
};
