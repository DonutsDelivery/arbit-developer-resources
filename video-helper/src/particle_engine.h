// particle_engine.h — ParticleEngine: a GPU compute-driven particle layer source.
//
// Media Machine P4 (visual-engine-phased-plan §P4). A particle clip is a
// generator whose pixels are a simulated particle pool, NOT a fragment program:
// a GL 4.3 compute shader advances a persistent std430 SSBO of particles every
// frame, then a point-sprite pass rasterises them into an owned RGBA8 layer
// texture that the FrameRenderer composites by zOrder like any other layer
// (no compositor change). It is the compute peer of ShaderGenerator — same
// clipId-keyed ownership, same "render into a texture the composite chain then
// treats as a decoded frame" posture, same Block A clock / Block C score feed.
//
// Seeding by the score: dead particles re-seed from the notes of ONE chosen
// track (the "spawn-source"). The compute kernel reads the same packed uNotes
// texture the shader generators read (block_c_packer.h: texel1.z = trackId),
// filters rows by spawnTrack, and seeds position/colour/velocity from the
// matched note's pitch + velocity. So "a chosen track's notes seed/colour/force
// the particles" — the P4 thesis — with no new score plumbing.
//
// Capability gate: the engine self-gates on the GL 4.3 compute entry points
// being resolved in the GlFuncs table (loadGl43Functions soft-loads them; they
// stay null on a < 4.3 / macOS context). When unavailable, render() returns 0 —
// the layer renders nothing (a parked wire, never a crash). The plugin's
// New-Generator menu mirrors this gate via viewport_info.gpu.computeShaders.
//
// Determinism (export parity): a fresh FrameRenderer per export job means a
// fresh SSBO (zero-initialised ⇒ all particles dead ⇒ re-seed on frame 0). Each
// invocation owns exactly one particle index (no shared-write race) and re-seeds
// via a deterministic hash(index, frame) — never rand() — so back-to-back
// exports of the same timeline are byte-identical. The simulation only advances
// when the clock is playing, so a paused/repeated viewport frame holds still.

#pragma once

#if ARBIT_HAVE_VIEWPORT

#include <memory>
#include <string>

namespace arbitgl { struct GlFuncs; }

namespace videorender
{

struct ShaderClock;    // shader_generator.h (Block A clock; shared formula)
struct NoteFeatures;   // shader_generator.h (Block C packed score)
#if defined (__APPLE__) && ARBIT_HAVE_METAL_BACKEND
class MetalParticleEngine;
#endif

// v1 particle params. These ride the clip's genParams map ("clip<id>/gen/<name>")
// exactly like a shader's ISF INPUTs, so they serialize + mod-route for free; the
// renderer unpacks them from LayerDesc::genParams into this struct per frame.
struct ParticleParams
{
    int   count      = 512;    // pool size (legacy clips clamp to kMaxParticles)
    int   spawnTrack = 0;      // track whose notes seed/colour/force particles
    float size       = 2.0f;   // point sprite size in px
    float gravity    = 0.0f;   // downward acceleration (screen units / s^2)
    float force      = 1.0f;   // note-velocity -> spawn-velocity scale
    int   seed       = 0;      // deterministic curated-source seed
    float lifetime   = 0.0f;   // seconds; <= 0 keeps note-reactive legacy range
    float red = -1.0f, green = -1.0f, blue = -1.0f, alpha = 1.0f; // red < 0 keeps hue
};

// Populated only when ARBIT_PARTICLE_DIAGNOSTICS=1. This keeps production free
// of synchronous readbacks while letting the native smoke localise blank output.
struct ParticleDiagnostics
{
    int noteRows = 0;
    int liveParticles = -1;       // simulation state before rasterisation
    int visibleCandidates = -1;   // live particles with finite in-range positions
    int drawAlphaPixels = -1;     // backend target before bridge/readback
    int readbackAlphaPixels = -1; // texture returned to the GL caller
};

class ParticleEngine
{
public:
    ParticleEngine();
    ~ParticleEngine();
    ParticleEngine (const ParticleEngine&) = delete;
    ParticleEngine& operator= (const ParticleEngine&) = delete;

    // Legacy generator clips retain their historical cap. The typed paid node
    // has a separate, deliberately smaller wire/resource contract.
    static constexpr int kMaxParticles = 100000;
    static constexpr int kMaxVisualParticles = 4096;

    void resetSimulation (const arbitgl::GlFuncs* gl);

    // True iff this GL context exposes the GL 4.3 compute / SSBO entry points
    // (loadGl43Functions resolved them). The plugin gate mirrors this via
    // viewport_info.gpu.computeShaders.
    static bool computeAvailable (const arbitgl::GlFuncs* gl);

    // Advance the pool for this frame and rasterise it into the owned w*h RGBA8
    // texture; returns that texture (owned; overwritten next call), or 0 when
    // compute is unavailable or a program failed to compile (caller skips the
    // layer). Lazily compiles the fixed compute + draw programs and allocates the
    // SSBO/target. `clock` drives the sim (only advances when clock.playing).
    // `notes` (Block C): when non-null, dead particles re-seed from notes on
    // params.spawnTrack; null/empty ⇒ no seeding (pool empties). Binds its own
    // FBO/viewport; the caller must restore its own afterwards. GL must be current.
    unsigned render (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                     int width, int height, const ParticleParams& params,
                     const NoteFeatures* notes);

    // Free every GL object. Call while the context is current (the dtor does NOT
    // touch GL). Safe to call repeatedly.
    void shutdown (const arbitgl::GlFuncs* gl);

    bool ok() const { return ok_; }
    const std::string& log() const { return log_; }
    const ParticleDiagnostics& diagnostics() const { return diagnostics_; }

private:
    bool ensurePrograms (const arbitgl::GlFuncs* gl);
    void ensurePool (const arbitgl::GlFuncs* gl, int count);
    void ensureTarget (const arbitgl::GlFuncs* gl, int width, int height);
    void uploadNotes (const arbitgl::GlFuncs* gl, const NoteFeatures* notes);
    unsigned compileCompute (const arbitgl::GlFuncs* gl, const char* src, std::string& err) const;
    unsigned compileDraw (const arbitgl::GlFuncs* gl, const char* vs, const char* fs, std::string& err) const;

#if defined (__APPLE__)
    // macOS GL-4.1 ping-pong-FBO fallback (no compute / SSBO). Particle state lives
    // in two RGBA32F textures per ping-pong set: posVel (pos.xy, vel.xy) and life
    // (life, maxLife, hue, pad), one texel per particle on a stateW x stateH grid.
    // ensureFallback (re)compiles the sim/draw programs and (re)allocates the
    // state textures + their MRT FBOs for `count` particles; renderFallback runs
    // the per-frame sim (fullscreen-quad fragment pass → MRT) then the point-sprite
    // draw (vertex shader reads state via texelFetch). See particle_engine.cpp.
    bool     ensureFallback (const arbitgl::GlFuncs* gl, int count);
    unsigned renderFallback (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                             int width, int height, const ParticleParams& params,
                             const NoteFeatures* notes, int count);

    unsigned simProg_    = 0;          // fragment particle-update program (MRT)
    unsigned drawFbProg_ = 0;          // texelFetch point-sprite program
    unsigned posVelTex_[2] = { 0, 0 }; // ping-pong: pos.xy / vel.xy (RGBA32F)
    unsigned lifeTex_[2]   = { 0, 0 }; // ping-pong: life / maxLife / hue / pad
    unsigned simFbo_[2]    = { 0, 0 }; // FBO i = {posVelTex_[i]@0, lifeTex_[i]@1}
    int      fbRead_       = 0;        // which set holds the current (readable) state
    int      stateTexW_    = 0;        // state-grid width  (index = y*stateW + x)
    int      stateTexH_    = 0;        // state-grid height (rows to hold `count`)
    int      fbPoolCount_  = -1;       // particles the fallback textures are sized for

    // sim uniforms
    int sUCount_ = -1, sUSpawnTrack_ = -1, sUGravity_ = -1, sUForce_ = -1, sULifetime_ = -1, sUDt_ = -1,
        sUFrame_ = -1, sUNoteCount_ = -1, sUAspect_ = -1, sUNotes_ = -1,
        sUPosVel_ = -1, sULife_ = -1, sUStateW_ = -1;
    // draw uniforms
    int dUPosVel_ = -1, dULife_ = -1, dUStateW_ = -1, dUPointSize_ = -1, dUColor_ = -1;

  #if ARBIT_HAVE_METAL_BACKEND
    // P6 migration slice: opt-in native Metal compute+raster output bridged
    // through IOSurface into the existing GL compositor. The GL fallback below
    // remains active whenever Metal is disabled or unavailable.
    std::unique_ptr<MetalParticleEngine> metal_;
  #endif
#endif

    // Frame-gated catch-up state (export parity, audit #7) — applies to BOTH the
    // compute and the macOS fallback path, so it lives outside the __APPLE__
    // guard. The sim must advance exactly once per integer timeline frame, NOT
    // once per render() call — a 60Hz viewport calls render() ~2x per 30fps
    // project frame, and a hitching tick skips frames. Both paths advance
    // (clock.frame - lastSimFrame_) unit steps (capped) so the preview
    // reproduces the export's per-frame sequence.
    bool     simSeeded_    = false;    // has the sim run at least one playing step?
    int      lastSimFrame_ = 0;        // last integer timeline frame advanced to
    static constexpr int kMaxReplaySteps = 10000; // explicit fail-closed random-access bound

    // How many unit-frame advance passes to run this call; writes the uFrame the
    // FIRST pass uses (the last pass lands exactly on clock.frame, matching what
    // the export's single per-frame pass uses). 0 = hold (same frame re-shown).
    // Paused keeps the prior behavior: one dt=0 pass, and the seed cursor is left
    // untouched so resuming advances from where playback paused. Defined in the
    // .cpp (ShaderClock is only forward-declared here — its members aren't
    // visible in this header).
    int planSimSteps (const ShaderClock& clock, int& firstFrame);

    bool triedPrograms_ = false;   // compiled (or failed) once
    bool ok_ = false;
    std::string log_;
    ParticleDiagnostics diagnostics_;

    unsigned computeProg_ = 0;     // particle update kernel
    unsigned drawProg_    = 0;     // point-sprite rasteriser
    unsigned ssbo_        = 0;     // std430 particle pool
    int      poolCount_   = 0;     // particles currently allocated in ssbo_
    unsigned vao_         = 0;     // empty VAO (core profile needs one for draws)
    unsigned fbo_         = 0;
    unsigned outTex_      = 0;
    int      outW_ = 0, outH_ = 0;
    unsigned notesTex_    = 0;     // 4 x 128 RGBA32F (uNotes); 0 until first upload

    // Cached uniform locations.
    int uCount_ = -1, uSpawnTrack_ = -1, uGravity_ = -1, uForce_ = -1, uLifetime_ = -1;
    int uDt_ = -1, uFrame_ = -1, uNoteCount_ = -1, uAspect_ = -1, uNotesC_ = -1;
    int uPointSize_ = -1, uColor_ = -1;
};

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
