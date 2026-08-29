// shader_generator.h — ShaderGenerator: a procedural GLSL layer source.
//
// Media Machine M3 (Layer 2, generative-visuals-research.md §4 / master-plan
// M3). A shader generator turns a user/agent-authored GLSL fragment shader into
// a layer texture rendered helper-side every frame, against the **Arbit uniform
// contract** (Block A clock, Block B audio features, Block C symbolic notes).
// Unlike the geometric effect passes (which transform an input layer), a
// generator synthesises pixels from the contract uniforms — it IS a clip's
// source, not an effect on one.
//
// Design notes:
//   * Self-contained GL unit. Owns its own program, fullscreen-quad VAO/VBO,
//     FBO and output texture, plus tiny 1x1 black "default" textures bound to
//     the contract samplers so a shader that reads audio/notes compiles and
//     runs even before Block B/C are fed (M3 ships Block A live; M4/M5 fill the
//     rest with NO shader or prelude change — the contract is the stable seam).
//   * Runtime recompile. The FrameRenderer compiles its fixed programs once at
//     initialize() and never again; a generator is the opposite — its source
//     changes at the user's/agent's whim. So it lives outside FrameRenderer's
//     program set, with its own compile that keeps the **last good program** on
//     failure (a broken edit never blanks the composite — master-plan risk
//     posture; the GL compile log is surfaced to the caller).
//   * Dialect front door. Raw source is wrapped by arbitshader::wrapToContract
//     (shader_dialect.h) — bare GLSL / Shadertoy / ISF all become one
//     #version 330 core shader against the contract prelude. That dependency
//     (nlohmann/json) is confined to the .cpp so this header stays light enough
//     to include from renderer.h.
//   * No GL in the destructor. Like FrameRenderer, GL teardown is explicit
//     (shutdown(gl) while the context is current); the dtor only asserts the
//     handles were released. This keeps the type cheaply storable and avoids
//     freeing GL objects without a current context.
//
// Export == preview by construction: the clock is a pure function of display
// time + tempo (makeShaderClock), evaluated identically in the viewport and the
// export frame loop. A shader cannot tell preview from export.

#pragma once

#if ARBIT_HAVE_VIEWPORT

#include <map>
#include <string>
#include <vector>
#include "beat_timeline.h"

namespace arbitgl { struct GlFuncs; }

namespace videorender
{

// One generator parameter discovered from the source (ISF INPUTS today; future
// dialects may add more). Mirrors arbitshader::GenInput's scalar shape without
// pulling the dialect/json headers into this interface. Destined to become the
// automatable "clip<id>/gen/<name>" param set (M3+).
struct GenParam
{
    std::string name;
    int    type = 0;          // arbitshader::InputType wire value (0 = bool, …)
    double defaultV = 0.0;    // scalar default (bool/long/float/event/unknown)
    double minV = 0.0;
    double maxV = 1.0;
    double defaultVec[4] = { 0.0, 0.0, 0.0, 0.0 };  // vector default (point2D .xy / color .rgba)
    std::string importedPath;  // M7: ISF IMPORTED image's declared PATH (empty = a user INPUT)
};

// Generator-param shape helpers (M7). This header stays dialect-free, so the
// arbitshader::InputType wire values are hardcoded here (kept in sync with the
// enum): 0 bool, 1 long, 2 float, 3 point2D, 4 color, 5 image, 6 audio,
// 7 audioFFT, 8 event, 9 unknown. A param is uploaded as N scalar components:
// 1 for the scalar types, 2 for point2D (vec2), 4 for color (vec4); image/audio
// samplers are 0 (not value params — skipped). The "clip<id>/gen/<name><suffix>"
// param key uses the per-component suffix ("" for scalars, ".x"/".y", ".r".."‍.a").
inline int genParamComponentCount (int type)
{
    switch (type)
    {
        case 3:                 return 2;   // point2D → vec2
        case 4:                 return 4;   // color   → vec4
        case 5: case 6: case 7: return 0;   // image / audio / audioFFT samplers
        default:                return 1;   // bool / long / float / event / unknown
    }
}

inline const char* genParamComponentSuffix (int type, int i)
{
    if (type == 4) { static const char* c[4] = { ".r", ".g", ".b", ".a" }; return c[i & 3]; }
    if (type == 3) { static const char* c[2] = { ".x", ".y" };             return c[i & 1]; }
    return "";   // scalar: the key is the bare name
}

inline bool genParamIsIntScalar (int type)   // bool / long / event → glUniform1i
{
    return type == 0 || type == 1 || type == 8;
}

// Block A clock + (zeroed for now) Block B/C scalars uploaded every frame.
// Built by makeShaderClock so the viewport and exporter share ONE formula.
struct ShaderClock
{
    double timeSec     = 0.0;       // uTime: clip-relative wall seconds
    double timeDelta   = 1.0 / 30;  // uTimeDelta: one frame in seconds
    int    frame       = 0;         // uFrame: frame index since clip start
    double beat        = 0.0;       // uBeat: timeline-absolute beat
    double bpm         = 120.0;     // uBPM
    double beatPhase   = 0.0;       // uBeatPhase: fract(beat)
    double barPhase    = 0.0;       // uBarPhase: fract(beat / beatsPerBar)
    double beatsPerBar = 4.0;       // uBeatsPerBar
    double clipBeat    = 0.0;       // uClipBeat: beats since clip start
    double clipLength  = 0.0;       // uClipLength: clip duration in beats
    bool   playing     = true;      // uPlaying
};

// Derive the clock from the authoritative project beat timeline. Both render
// paths call this exact function, including tempo ramps and meter changes.
ShaderClock makeShaderClock (double displaySec, double clipStartSec,
                             double clipDurationSec,
                             const videotime::BeatTimeline& timeline, double fps,
                             bool playing, int frame);

// Block B audio features (master-plan §4.2 / M4) — the master-mix analysis
// (RMS / peak / onset / 64 log bands) the generator uploads to uRMS / uPeak /
// uOnset / uOnsetAge and the uAudioBands sampler. Unlike ShaderClock these are
// NOT a pure function of the clock; they come from the A3 analyzer over the mix
// WAV (export) or the live audio shm (viewport, M4 Slice B). Pass nullptr to
// render() to keep the zero-feed contract (scalars 0, samplers black) — so an
// audio-less viewport / transition-source layer behaves exactly as before.
// Plain copyable data so it rides LayerDesc through the composite (like
// ShaderClock); both render paths fill it from the SAME constants header
// (block_b_defs.h) ⇒ export parity.
struct AudioFeatures
{
    float rms      = 0.0f;        // uRMS    (0..1, smoothed)
    float peak     = 0.0f;        // uPeak   (0..1, instant attack)
    float onset    = 0.0f;        // uOnset  (0..1, decays after a transient)
    float onsetAge = 0.0f;        // uOnsetAge (seconds since the last onset)
    std::vector<float> bands;     // uAudioBands magnitudes (64 expected; empty = bind black)
};

// Block C symbolic note/link textures (master-plan §4.3 / M5 — "the visuals
// read the score") — the packed note/link timeline the generator uploads to
// the uNotes (4-texel x 128-row RGBA32F) and uLinks (256x1) samplers, plus
// uNoteCount/uLinkCount/uRootFreq. Produced by the A2 BlockCPacker over the
// score at the frame's beat: a STATEFUL voice allocator, so the producer packs
// once per frame in monotonic order (rows stay stable across a note's
// residency). Pass nullptr to render() to keep the zero-feed contract (samplers
// black, counts 0) — an audio-only / decoded-media layer behaves as before.
// Plain copyable data so it rides LayerDesc through the composite (like
// ShaderClock / AudioFeatures); both render paths fill it from the SAME packer
// (block_c_packer.h) ⇒ export parity.
struct NoteFeatures
{
    std::vector<float> notesTex;   // 128*4*4 floats, row-major (row,texel,channel); empty = bind black
    std::vector<float> linksTex;   // 256*4 floats per edge (slaveRow, masterRow, num, den)
    int   noteCount = 0;           // uNoteCount: highest occupied row + 1 (loop bound, covers holes)
    int   linkCount = 0;           // uLinkCount: link edges with both endpoints resident
    float rootFreq  = 261.625565f; // uRootFreq (Hz)
    float historyBeats = 0.0f;     // uScoreHistoryBeats
    float lookaheadBeats = 0.0f;   // uScoreLookaheadBeats
};

class ShaderGenerator
{
public:
    ShaderGenerator() = default;
    ~ShaderGenerator();
    ShaderGenerator (const ShaderGenerator&) = delete;
    ShaderGenerator& operator= (const ShaderGenerator&) = delete;

    // Compile a new source. Wraps via the dialect front door, then GL-compiles.
    // On success: replaces the active program, refreshes params(), ok()==true.
    // On wrap OR compile failure: keeps the last good program (if any), fills
    // log() with the diagnostics, ok()==false — and render() keeps drawing the
    // last good frame. The GL context must be current. Returns ok().
    bool setSource (const arbitgl::GlFuncs* gl, const std::string& rawSource);

    // Render the active program into the owned w*h RGBA8 texture with `clock`,
    // and return that texture (owned; overwritten on the next call). Returns 0
    // when there is no program to run yet (never compiled / first edit failed)
    // — the caller skips the layer. Binds its own FBO/viewport/program; the
    // caller must restore its own FBO + viewport afterwards. GL must be current.
    // `audio` (Block B): when non-null its features feed uRMS/uPeak/uOnset/
    // uOnsetAge + the uAudioBands sampler; null keeps the zero-feed contract.
    // `notes` (Block C): when non-null the packed score feeds uNotes/uLinks +
    // uNoteCount/uLinkCount/uRootFreq; null keeps those samplers black / 0.
    // `genValues` (M7): per-frame values for the discovered generator params
    // (ISF INPUTS), keyed by param name (the "clip<id>/gen/<name>" sink). Each
    // SCALAR param (bool/long/float/event) is uploaded to its uniform — from
    // this map when present, else from the param's own ISF default. nullptr ⇒
    // every param at its default (the pre-M7 behavior). Vector (point2D/color)
    // params keep the GLSL default. Ignored for params the program optimized
    // out (location < 0).
    // `genImages` (M7 image follow-up): GL textures for the shader's `image`-type
    // INPUTS (`sampler2D <name>`), keyed by param name. Each image INPUT is bound
    // to its own texture unit starting at TEXTURE4 (units 0–3 are the contract
    // audio/notes/links samplers); a param with no entry here (or a null/0 handle)
    // binds 1x1 black so the sampler read returns 0 — the zero-feed contract. The
    // textures are owned by the caller (decoded + uploaded once, static for the
    // clip), NOT by this generator: render() only binds, never deletes them.
    unsigned render (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                     int width, int height, const AudioFeatures* audio = nullptr,
                     const NoteFeatures* notes = nullptr,
                     const std::map<std::string, double>* genValues = nullptr,
                     const std::map<std::string, unsigned>* genImages = nullptr);

    // Free every GL object. Call while the context is current (the dtor does
    // NOT touch GL). Safe to call repeatedly.
    void shutdown (const arbitgl::GlFuncs* gl);

    bool ok() const { return ok_; }
    const std::string& log() const { return log_; }
    const std::vector<GenParam>& params() const { return params_; }
    bool hasProgram() const { return program_ != 0; }

private:
    // Contract uniform locations, cached after each successful compile.
    struct Locs
    {
        int uResolution = -1, uTime = -1, uTimeDelta = -1, uFrame = -1;
        int uBeat = -1, uBPM = -1, uBeatPhase = -1, uBarPhase = -1;
        int uBeatsPerBar = -1, uClipBeat = -1, uClipLength = -1, uPlaying = -1;
        int uRMS = -1, uPeak = -1, uOnset = -1, uOnsetAge = -1;
        int uAudioBands = -1, uAudioBands2D = -1;
        int uNotes = -1, uLinks = -1, uNoteCount = -1, uLinkCount = -1;
        int uScoreHistoryBeats = -1, uScoreLookaheadBeats = -1;
        int uLastOnsetBeat = -1, uRootFreq = -1, uChord = -1;
        // Block D — camera (P5 raymarch); -1 when the shader doesn't read them.
        int uCamPos = -1, uCamTarget = -1, uCamUp = -1;
        int uCamFov = -1, uCamNear = -1, uCamFar = -1;
    };

    unsigned compile (const arbitgl::GlFuncs* gl, const std::string& wrappedGlsl,
                      std::string& errorOut) const;
    void cacheLocs (const arbitgl::GlFuncs* gl, unsigned prog);
    void ensureQuad (const arbitgl::GlFuncs* gl);
    void ensureDefaults (const arbitgl::GlFuncs* gl);
    void ensureTarget (const arbitgl::GlFuncs* gl, int width, int height);
    // ISF multipass (M7): (re)allocate the named TARGET buffers at w*h (RGBA16F,
    // cleared to black so a PERSISTENT target's first-frame read is defined).
    void ensurePassTargets (const arbitgl::GlFuncs* gl, int width, int height);
    void releasePassTargets (const arbitgl::GlFuncs* gl);
    // Lazily (re)allocate the float band textures to hold `n` magnitudes, as a
    // GL_R32F 1D texture (uAudioBands, for arbitBand) and a GL_R32F nx1 2D
    // texture (uAudioBands2D, the ISF audioFFT / Shadertoy iChannel0 alias).
    void ensureBandsTex (const arbitgl::GlFuncs* gl, int n);
    // Lazily allocate the Block C textures once: uNotes as a 4 x 128 GL_RGBA32F
    // 2D texture and uLinks as a 256 x 1 GL_RGBA32F 2D texture (GL_NEAREST —
    // the packed texels are structured data, never interpolated).
    void ensureNotesTex (const arbitgl::GlFuncs* gl);

    unsigned program_  = 0;   // active (last good) program, 0 = none
    bool     ok_       = false;
    std::string log_;
    std::vector<GenParam> params_;
    std::vector<int> paramLocs_;   // M7: uniform location per params_ entry (-1 = absent/optimized out)
    Locs locs_;

    unsigned vao_ = 0, vbo_ = 0, fbo_ = 0;
    unsigned outTex_ = 0;
    int outW_ = 0, outH_ = 0;
    unsigned blackTex2D_ = 0;   // 1x1 black, for uNotes/uLinks/uAudioBands2D
    unsigned blackTex1D_ = 0;   // 1x1 black, for sampler1D uAudioBands
    unsigned bandsTex1D_ = 0;   // GL_R32F n-texel, uAudioBands when audio present
    unsigned bandsTex2D_ = 0;   // GL_R32F nx1, uAudioBands2D (iChannel0 alias)
    int      bandsTexN_  = 0;   // current allocated band count (0 = unallocated)
    unsigned notesTex2D_ = 0;   // GL_RGBA32F 4x128, uNotes when notes present (M5)
    unsigned linksTex2D_ = 0;   // GL_RGBA32F 256x1, uLinks when notes present
    bool     notesTexAlloc_ = false;  // true once the Block C textures are allocated

    // ISF multipass (M7). The PASS list captured from the dialect wrap, plus the
    // per-TARGET render buffers. A pass renders the body (branching on PASSINDEX)
    // into its TARGET texture; later passes / the output pass sample earlier
    // targets by name. PERSISTENT targets are double-buffered — a pass reads the
    // previous frame while writing the current one (the ISF feedback path). For a
    // single implicit output pass (no PASSES / no PERSISTENT) multipass_ stays
    // false and render() keeps the pre-M7 fast path: one draw into outTex_.
    struct PassInfo   { std::string target; bool persistent = false; };
    struct PassTarget { unsigned tex[2] = { 0, 0 }; bool persistent = false; };
    std::vector<PassInfo> passes_;
    std::map<std::string, PassTarget> passTargets_;     // render buffers by TARGET name
    std::map<std::string, int> targetSamplerLocs_;      // sampler2D uniform loc per target
    int      passIndexLoc_ = -1;        // PASSINDEX uniform (-1 = single pass / absent)
    unsigned passParity_   = 0;         // flips per frame: swaps persistent read/write
    int      passTargW_ = 0, passTargH_ = 0;   // size the pass targets are allocated at
    bool     multipass_ = false;        // >1 pass OR any persistent target
};

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
