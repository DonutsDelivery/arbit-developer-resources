#pragma once

// Exporter — renders an export jobSpec to a video file.
//
// The display timeline is sampled at the output fps; each output frame maps
// through its segment (constant rate sourceSpan/displaySpan) to a source
// time, decoded via MediaContext. Gaps between segments render black.
// The master audio mix (WAV pre-rendered by Arbit) is encoded to AAC and
// muxed in; audio is authoritative for overall duration when longer.
//
// WP6 export parity: when the helper is built with viewport support and a GL
// context can be created, frames are composited through the SAME FrameRenderer
// pipeline the live viewport uses (multi-layer, effects rack, transforms,
// blend modes, leading-edge transitions), driven by the clips' static graph
// state plus a baked automation timeline (paramTimeline: linear interpolation
// between baked points, constant extrapolation at the ends). Headless / GL
// failure / minterpolate jobs fall back to the legacy single-segment CPU path
// (effects skipped) and report glCompositing = false.

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mod_defs.h"       // arbitmod::Score — Block C (M5); dependency-free
#include "beat_timeline.h"  // authoritative tempo/meter mapping; dependency-free
#include "video_control_plan.h"
#include "render_snapshot.h"
#include "visual_plan_telemetry.h"

using ExportSegment = videowire::RenderSegment;

// One effects-rack slot (mirrors graph_set_effects).
struct ExportEffectSlot
{
    int slot = -1;
    int type = -1;               // videofx::EffectType, -1 = empty
    bool enabled = true;
    std::vector<std::pair<std::string, double>> params; // wire name -> value
};

// Static (un-automated) render-graph state for one clip — the same values
// pushClipGraphState streams to the live viewport.
struct ExportClipState
{
    int clipId = 0;
    double scale = 1.0;
    double translateX = 0.0, translateY = 0.0;
    double rotationDeg = 0.0;
    double cropLeft = 0.0, cropRight = 0.0, cropTop = 0.0, cropBottom = 0.0;
    double opacity = 1.0;
    bool visible = true;
    int zOrder = 0;
    int blendMode = 0;           // videofx::BlendMode wire value
    std::vector<ExportEffectSlot> effects;

    // Shape mask (clip<id>/mask/* — same semantics as the viewport).
    int maskType = 0;            // 0 none, 1 rect, 2 ellipse
    double maskCx = 0.5, maskCy = 0.5;
    double maskW = 0.8, maskH = 0.8;
    double maskFeather = 0.05;
    bool maskInvert = false;

    // Per-clip .cube 3D LUT, shipped as a PATH in the jobSpec (the helper
    // parses the file at export start — PROTOCOL.md §LUT). "" = none.
    std::string lutPath;

    // Static ISF INPUT overrides (M7): "clip<id>/gen/<name>" values set once for
    // the whole clip (name -> value), the static layer beneath baked automation
    // and mod-matrix routings. Keys not present fall back to the shader's own ISF
    // default. Only meaningful for a shader-generator clip (shaderSource set).
    std::map<std::string, double> genParams;

    // Imported image INPUTs (M7 image follow-up): "<inputName>" -> file path.
    // Each names an ISF `"TYPE": "image"` INPUT whose `sampler2D` the helper
    // decodes once (FFmpeg image2 — software, like every offline decode) and
    // binds while rendering this clip's shader. STATIC: an image file does not
    // change over the clip, so (unlike genParams) it is not baked / mod-matrix /
    // Lua driven. A missing or undecodable path leaves the sampler reading 1x1
    // black (zero-feed). Only meaningful for a shader-generator clip
    // (shaderSource set); ignored otherwise.
    std::map<std::string, std::string> genImages;

    // Shader-generator source (M3). When non-empty this clip is a procedural
    // generator: the exporter compiles the source through the dialect front
    // door and renders it (against the Block A clock) instead of decoding a
    // media frame — the clip's segments carry timing only, their sourcePath is
    // the `gen://shader` sentinel and no MediaContext is opened. "" = a normal
    // decoded-media clip.
    std::string shaderSource;

    // Adjustment clip (M8 "effect the world"). When true this clip has NO source
    // of its own (no media, no shaderSource): its effects rack runs over the
    // composite of every layer beneath it and the result is mixed back at the
    // clip's opacity. Its segments carry timing/zOrder/trackLayer only — their
    // sourcePath is the `gen://adjustment` sentinel and no MediaContext is opened.
    // Mutually exclusive with shaderSource. false = a normal source clip.
    bool isAdjustment = false;
};

// One baked automation sample. paramId = "clip<id>/<node>/<param>" or
// "text<id>/<param>" (same namespace as graph_set_param). The exporter
// interpolates linearly between samples of the same paramId and holds the
// first/last value outside them.
struct ExportParamSample
{
    std::string paramId;
    double atSec = 0.0;          // display-timeline seconds
    double value = 0.0;          // denormalized (native range) value
};

// Text overlay: timing/placement plus the rasterized pixels. Arbit renders
// the text (same TextOverlayRasterizer pixels text_set_image ships to the
// live viewport) and base64-encodes the block into the jobSpec — inline
// rather than shm because exports must work headless with no viewport open.
// Overlays with empty rgba (no/bad pixels) keep their timing but draw
// nothing. GL path only: the CPU fallback skips compositing entirely.
struct ExportTextOverlay
{
    int textId = 0;
    int ownerClipId = -1;
    double startSec = 0.0, durationSec = 0.0;
    double posX = 0.0, posY = 0.0; // NDC, same convention as ImageLayerDesc
    double opacity = 1.0;
    int zOrder = 0;
    int width = 0, height = 0;        // pixel size (stride = width*4)
    std::vector<uint8_t> rgba;        // straight RGBA, top row first
};

struct ExportJob
{
    // Helper-startup capability; never parsed from RPC jobSpec.
    std::string depthCacheRoot;
    std::string outPath;
    uint64_t authoringRevision = 0;
    uint64_t exportableRevision = 0;
    int width = 1920;
    int height = 1080;
    double fps = 30.0;
    std::string codec = "h264";          // h264 | h265 | vp9 | prores | ffv1
    std::string proresProfile = "hq";    // when codec==prores: hq (422 HQ) | 4444 | 4444xq
    std::string encoder = "auto";        // auto | software | nvenc | videotoolbox
    std::string interpolation = "none";  // none | minterpolate | rife | auto
    std::string audioPath;               // master mix WAV ("" = no audio track)
    double durationSec = 0.0;            // display timeline length

    // Loudness normalization target in LUFS (ITU-R BS.1770). 0 = off. A negative
    // value (e.g. -14 for streaming) measures the master's integrated loudness
    // and applies a constant gain to hit the target — but DOWNWARD ONLY: a master
    // already at or below the target is never boosted (it has been through the
    // mix/limiter chain; re-amplifying it would undo that and risk clipping).
    double lufsTarget = 0.0;

    // Export range on the display timeline (PROTOCOL.md §Export). startSec
    // shifts where output frame 0 samples the timeline; endSec > 0 sets the
    // exclusive end (0 = content end). When endSec is explicit the audio
    // track is trimmed to exactly endSec - startSec (the range-rendered mix
    // WAV carries a decay tail past the range edge that full-timeline
    // exports keep but range exports must cut).
    double startSec = 0.0;
    double endSec = 0.0;

    // Intra-only output (every frame a keyframe, no B-frames) — used by the
    // render-cache build (PROTOCOL.md §Render cache) so the baked file seeks
    // frame-independently like an edit proxy.
    bool intraOnly = false;

    // Feedback pre-roll (generative-visuals-research.md risk 13). For a
    // mid-timeline export range (startSec > 0), cross-frame feedback trails
    // depend on frames before the range that the export never rendered. When
    // > 0 the GL frame loop renders this many seconds of warm-up frames before
    // startSec (clamped to the timeline start) — composited but NOT encoded —
    // to populate the feedback history so the range matches a full-clip export.
    // The plugin supplies a beats-derived value (default 8 beats); 0 = off.
    double feedbackPreRollSec = 0.0;

    // Legacy scalar fallbacks remain on the wire for older callers. beatTimeline
    // is authoritative whenever tempoMap/timeSignatureMap are present and is used
    // by every score, modulation, script, shader, seek, and warm-up clock path.
    double bpm = 120.0;
    double beatsPerBar = 4.0;
    videotime::BeatTimeline beatTimeline;

    // Canvas background colour (M1 "canvas background param"): RGBA 0..1 the
    // composite is cleared to beneath all layers. Default = editor charcoal, so
    // a job without "canvasBackground" looks exactly as before.
    float bgColor[4] = { 0.04f, 0.04f, 0.05f, 1.0f };

    // HDR post stack (visuals-quality increment 2 — bloom + tonemap over the
    // final composite). Default-neutral (bloomIntensity 0, tonemap 0, exposure
    // 1) ⇒ FrameRenderer::applyPostFx is skipped and the export is byte-
    // identical to a job without it. Parsed from the job's "post" object.
    struct PostFx
    {
        float bloomIntensity = 0.0f;   // 0 = no bloom
        float bloomThreshold = 1.0f;   // extract HDR luminance above this
        float bloomRadius    = 8.0f;   // gaussian blur radius (px; blur caps at 20)
        int   tonemap        = 0;      // 0 clamp, 1 Reinhard, 2 ACES
        float exposure       = 1.0f;   // linear exposure multiply pre-tonemap
    } post;

    std::vector<ExportSegment> segments; // sorted by displayStartSec
    std::vector<videowire::CompiledVisualLayerPlan> visualLayerPlans;
    std::vector<videowire::VisualEventScheduleBinding> visualEventSchedules;

    std::vector<ExportClipState> clips;        // static graph state per clip
    std::vector<ExportParamSample> paramTimeline; // baked automation
    std::vector<ExportTextOverlay> texts;      // rasterized text overlays

    // Block C symbolic score (M5 — "the visuals read the score"). The full
    // note/link timeline; the GL frame loop runs the A2 BlockCPacker over it
    // once per frame (at the frame's beat) to feed shader generators' uNotes/
    // uLinks/uNoteCount/uLinkCount/uRootFreq. Empty notes ⇒ Block C stays
    // zero-fed. The packer is a stateful voice allocator, so row assignment
    // depends on note entry/exit history — to keep it consistent the loop WARMS
    // the packer from timeline frame 0 (a CPU-only catch-up on the first frame
    // requested). A frame-aligned mid-range export (startSec on the fps grid)
    // therefore gets the SAME rows as a from-the-top export; a non-aligned
    // startSec snaps the warm-up grid to the nearest frame. Per-note CONTENT
    // (midi/freq/cents/age) is always beat-correct regardless. Parsed even in
    // non-viewport builds (mod_defs.h is plain C++17 with no GL/GPL); only
    // consumed by the GL frame loop.
    arbitmod::Score score;
    double scoreLookaheadBeats = 4.0;   // residency lookahead in beats (~1 bar in 4/4)

    // Cross-domain modulation matrix (M6 — "the music moves the picture"). Each
    // routing maps a musical SOURCE (Block A clock / Block B audio / Block C
    // score, evaluated by arbitmod::evaluateSource) onto a render-graph clip
    // param in the SAME "clip<id>/<node>/<param>" namespace as paramTimeline,
    // with depth/curve/smoothing/mode. The GL frame loop evaluates every routing
    // once per frame in monotonic order (the one-pole smoothing carries BEAT
    // memory, so — exactly like the score packer — the loop WARMS routing state
    // from timeline frame 0; a frame-aligned mid-range export then matches a
    // from-the-top export). The modulated value is combine(mode, base, value)
    // where base is the param's static+baked value, so routings LAYER on top of
    // automation. Symbolic/clock/score sources warm exactly from frame 0;
    // audio-derived sources only have features inside the export range (the mix
    // WAV is range-scoped), so before startSec their Audio is zero-fed. The
    // per-routing mutable RoutingState is NOT here — it is owned by the frame
    // loop (runGlFrameLoop), one per export. Empty ⇒ no modulation, output
    // byte-identical to a no-modMatrix job. Parsed even in non-viewport builds
    // (mod_defs.h is plain C++17, no GL/GPL); only consumed by the GL frame loop.
    std::vector<arbitmod::Routing> routings;
    videocontrol::Plan controlPlan;

    // Per-frame Lua hook (M8 — "agent-programmable media machine"). A single
    // script defining a global `frame(ctx)` that returns a table of
    // "clip<id>/<node>/<param>" -> value overrides — the Turing-complete general
    // case of the mod matrix. Applied as the TOP layer in stateAt (static -> ISF
    // default -> baked -> mod-matrix -> Lua), so the script wins where it sets a
    // value. ctx carries the frame's clock (t/beat/bpm/bar/frame), Block B audio
    // (rms/peak/onset/onsetAge/bands) and a coarse Block C summary
    // (noteCount/rootFreq). The lua_State persists for the whole export and is
    // warmed from timeline frame 0 (like the score packer / routing driver) so a
    // script that keeps cross-frame state in globals stays deterministic across a
    // mid-range export. Empty ⇒ no hook. Inert (no-op) in builds without Lua.
    std::string luaScript;

    // Per-frame JavaScript hook (visual-engine P2) — the JS analogue of
    // luaScript, sharing the exact same FrameCtx contract and top-of-stateAt
    // application. scriptLang selects which engine runs ("lua" | "js"); when
    // empty it is inferred (a non-empty jsScript ⇒ js, else lua). Only ONE hook
    // is active per job. Inert (no-op) in builds without QuickJS.
    std::string jsScript;
    std::string scriptLang;
    uint32_t scriptCpuMs = 0;
    uint32_t scriptMemoryMiB = 0;
};

// Live progress/cancel state shared between the export worker thread and
// the RPC thread (export_progress / export_cancel — PROTOCOL.md §Export).
// All fields are atomics: the worker writes, the RPC thread reads, and the
// RPC thread sets `abort` which every frame loop (and the audio transcode
// loop) polls. phase: 0 idle, 1 setup, 2 audio, 3 video, 4 finalize.
struct ExportProgress
{
    std::atomic<int64_t> frame { 0 };       // output frames encoded so far
    std::atomic<int64_t> totalFrames { 0 };
    std::atomic<double> encodeFps { 0.0 };  // wall-clock encode rate
    std::atomic<int> phase { 0 };
    std::atomic<bool> abort { false };      // set by export_cancel
    videowire::VisualPlanTelemetry visualTelemetry;
};

// Returns empty string on success, error message on failure. A cancelled
// export returns exactly "cancelled" and removes the partial output file.
// usedEncoderOut reports the actual encoder name chosen (e.g. "libx264").
// glCompositingOut reports whether frames went through the GL FrameRenderer
// (true) or the legacy CPU letterbox path (false: headless, GL init failure,
// helper built without viewport, or interpolation == "minterpolate").
// interpolationBackendOut reports the slow-motion backend actually used:
// "nearest" (frame repeat), "minterpolate", "rife-cuda" or "rife-cpu".
// progress (optional) receives frame counts / phase and carries the abort
// flag; it must outlive the call.
std::string runExport (const ExportJob& job, std::string& usedEncoderOut,
                       bool& glCompositingOut,
                       std::string& interpolationBackendOut,
                       ExportProgress* progress = nullptr);

struct CompositeFrameResult
{
    std::vector<uint8_t> rgba;
    int width = 0;
    int height = 0;
    std::string compositorBackend;
    std::string presentationBackend;
};

// Render one exact display-timeline frame through the production offscreen
// compositor used by export. No output path is accepted or created.
std::string renderCompositeFrame (const ExportJob& job, double timelineSec,
                                  CompositeFrameResult& result);

// RecorderSession — push-frame video-only encode session, for the live
// piano-roll recorder (record_open / record_push_frame / record_close RPCs).
// Unlike runExport (which composites a whole timeline from an ExportJob),
// this takes ALREADY-RENDERED frames one at a time from the caller (Arbit's
// own capture loop) and muxes them straight to a file — no audio, no DPX/
// ProRes, no GL compositing. It shares the exact same encoder-selection
// (pickVideoEncoder), colour tagging (tagColorBt709), and packet-writing
// (encodeAndWrite) codepaths runExport uses, so a recording gets the same
// hardware-encoder preference and BT.709 tagging as a normal export — no
// separate/duplicated muxer logic to maintain. Implementation detail (AVFormat/
// AVCodec/SwsContext) stays behind a pimpl so this header stays FFmpeg-free,
// matching the rest of exporter.h.
class RecorderSession
{
public:
    RecorderSession();
    ~RecorderSession();
    RecorderSession (const RecorderSession&) = delete;
    RecorderSession& operator= (const RecorderSession&) = delete;

    // codec: "h264" (default) or "h265" — the only codecs a live recording
    // needs; ProRes/DPX/VP9 stay export-only. encoder: "auto" (prefer hw)
    // | "software". Returns "" on success, error message otherwise.
    std::string open (const std::string& outPath, int width, int height,
                      double fps, const std::string& codec = "h264",
                      const std::string& encoder = "auto");

    // Pushes one frame. pixels must be strideBytes * height bytes, 8-bit
    // BGRA (bgra=true, matching the shm ring's payload format — PROTOCOL.md
    // §Live viewport) or RGBA (bgra=false). Converts to the encoder's pixel
    // format internally (BT.709 limited-range YUV420P, same as export).
    std::string pushFrame (const uint8_t* pixels, int strideBytes, bool bgra = true);

    // Flushes the encoder and finalizes the container. Safe to call once;
    // a session destroyed without close() first is finalized best-effort by
    // the destructor (error swallowed, matching Region's RAII elsewhere).
    std::string close();

    bool isOpen() const;
    int64_t framesEncoded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
