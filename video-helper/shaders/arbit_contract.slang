// arbit_contract.slang — the shared Arbit per-frame uniform contract, as a
// Slang module (visual-engine plan P3).
//
// `import arbit_contract;` in a Slang shader to read the same clock / audio /
// JI-score uniforms the GLSL renderer feeds (the names mirror the auto-prepended
// GLSL contract in shader_dialect.h:551 verbatim, so a Slang shader lowered to
// GLSL binds to the exact same uniform locations the renderer sets).
//
// Authoring routes (until in-process libslang lands, ARBIT_HAVE_SLANG):
//   slangc -target spirv shader.slang -o shader.spv   → import with lang=spirv
//   slangc -target glsl  shader.slang -o shader.frag  → import with lang=glsl
// The SPIR-V route is the universal adapter (spirv_ingest.h) and works today.

module arbit_contract;

// --- Block A: clock -------------------------------------------------------
public uniform float2 uResolution;
public uniform float  uTime;
public uniform float  uTimeDelta;
public uniform int    uFrame;
public uniform float  uBeat;
public uniform float  uBPM;
public uniform float  uBeatPhase;
public uniform float  uBarPhase;
public uniform float  uBeatsPerBar;
public uniform float  uClipBeat;
public uniform float  uClipLength;
public uniform int    uPlaying;

// --- Block B: mix audio ---------------------------------------------------
public uniform float  uRMS;
public uniform float  uPeak;
public uniform float  uOnset;
public uniform float  uOnsetAge;
public uniform Sampler1D uAudioBands;        // 64-band spectrum
public uniform Sampler2D uAudioBands2D;      // 64x1 alias (ISF audioFFT / iChannel0)

// --- Block C: JI score ----------------------------------------------------
public uniform Sampler2D uNotes;             // 128 rows x 4 RGBA32F texels
public uniform Sampler2D uLinks;             // 256 x 1 harmonic-link graph
public uniform int    uNoteCount;
public uniform int    uLinkCount;
public uniform float  uRootFreq;             // song root frequency (Hz)
public uniform float  uScoreHistoryBeats;    // visible beats behind playhead
public uniform float  uScoreLookaheadBeats;  // visible beats ahead of playhead

// --- Block D: camera (3D / SDF raymarch, P5) ------------------------------
// Names + order mirror the GLSL prelude (shader_dialect.h arbitPrelude) verbatim
// so a Slang shader lowered to GLSL binds the exact same uniform locations the
// renderer sets. A Raymarch clip ships uCamPos/uCamTarget/uCamFov as genParams.
public uniform float3 uCamPos;               // eye position (default 0,0,5)
public uniform float3 uCamTarget;            // look-at point (default 0,0,0)
public uniform float3 uCamUp;                // up vector (default 0,1,0)
public uniform float  uCamFov;               // vertical FOV in degrees (default 45)
public uniform float  uCamNear;              // near plane (default 0.05)
public uniform float  uCamFar;               // far plane / max march distance (default 60)

// --- Convenience accessors ------------------------------------------------
// One audio band in [0,1] (i in [0,63]).
public float arbitBand(int i)
{
    return uAudioBands.Sample((float(i) + 0.5) / 64.0).r;
}

// The i-th sounding note's packed row (RGBA32F): typically
// (midi/freq, velocity, centsFromRoot, flags) — see block_c_packer.h.
public float4 arbitNote(int i)
{
    return uNotes.Load(int3(0, i, 0));
}

// Normalized beat phase within the bar, [0,1).
public float arbitBarPhase() { return frac(uBarPhase); }

// Block D camera ray for a fragCoord (aspect-correct, look-at basis, perspective
// from uCamFov). Mirrors the GLSL arbitCameraRay helper exactly.
public void arbitCameraRay(float2 fragCoord, out float3 ro, out float3 rd)
{
    // uv.y negated: Arbit presents the render-target Y-flipped vs Shadertoy's
    // y-up fragCoord, so world-up maps to the displayed top.
    float2 uv = (fragCoord - 0.5 * uResolution) / uResolution.y;
    float3 f = normalize(uCamTarget - uCamPos);
    float3 r = normalize(cross(f, uCamUp));
    float3 u = cross(r, f);
    float  z = 1.0 / tan(radians(uCamFov) * 0.5);
    ro = uCamPos;
    rd = normalize(uv.x * r - uv.y * u + z * f);
}
