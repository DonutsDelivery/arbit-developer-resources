// particle_engine.cpp — see particle_engine.h.

#if ARBIT_HAVE_VIEWPORT

#include "particle_engine.h"
#include "gl_loader.h"
#include "shader_generator.h"   // ShaderClock, NoteFeatures
#if defined (__APPLE__) && ARBIT_HAVE_METAL_BACKEND
 #include "gpu_backend/particle_engine_metal.h"
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// Defensive: these GL 4.3 / point-sprite tokens come from <GL/glext.h> (pulled in
// by gl_loader.h), but guard them so an older glext still compiles.
#ifndef GL_COMPUTE_SHADER
 #define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
 #define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
 #define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#endif
#ifndef GL_PROGRAM_POINT_SIZE
 #define GL_PROGRAM_POINT_SIZE 0x8642
#endif

namespace videorender
{

ParticleEngine::ParticleEngine() = default;

// uNotes texture geometry (block_c_packer.h: 128 rows x 4 texels, RGBA32F).
static constexpr int kNoteTexW = 4;
static constexpr int kNoteTexH = 128;

// One particle = 8 floats (std430): pos.xy, vel.xy, life, maxLife, hue, pad.
static constexpr int kFloatsPerParticle = 8;

// --- GLSL -------------------------------------------------------------------

// Compute update kernel. Each invocation owns particle index i; dead particles
// re-seed from a note on uSpawnTrack (read from the packed uNotes texture,
// texel1.z == trackId, texel0.y == velocity/127). Re-seed selection is a
// deterministic hash(i, frame) so back-to-back exports are byte-identical.
static const char* kComputeSrc = R"GLSL(
#version 430
layout(local_size_x = 256) in;

struct Particle { vec2 pos; vec2 vel; float life; float maxLife; float hue; float pad; };
layout(std430, binding = 0) buffer Pool { Particle p[]; };

uniform int   uCount;
uniform int   uSpawnTrack;
uniform float uGravity;
uniform float uForce;
uniform float uLifetime;
uniform float uDt;
uniform int   uFrame;
uniform int   uNoteCount;
uniform float uAspect;        // width/height, keeps motion isotropic
uniform sampler2D uNotes;     // 4 x 128 RGBA32F

float hash11 (uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7fffffffU) / float(0x7fffffff);
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount)) return;

    Particle pt = p[i];
    pt.life -= uDt / max(pt.maxLife, 1e-3);

    if (pt.life <= 0.0)
    {
        // Count live notes on the spawn track (velocity > 0 ⇒ not a packer hole).
        int matchCount = 0;
        int rows = min(uNoteCount, 128);
        for (int r = 0; r < rows; ++r)
        {
            vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);   // freq, cents, trackId, isRoot
            if (int(t1.z + 0.5) == uSpawnTrack)
            {
                vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0); // midi, vel, age, remain
                if (t0.y > 0.001) ++matchCount;
            }
        }
        if (matchCount > 0)
        {
            int pick = int(hash11(i * 747U + uint(uFrame) * 13U) * float(matchCount));
            pick = min(pick, matchCount - 1);
            int seen = 0; int chosen = -1;
            for (int r = 0; r < rows; ++r)
            {
                vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);
                vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0);
                if (int(t1.z + 0.5) == uSpawnTrack && t0.y > 0.001)
                {
                    if (seen == pick) { chosen = r; break; }
                    ++seen;
                }
            }
            if (chosen >= 0)
            {
                vec4 t0 = texelFetch(uNotes, ivec2(0, chosen), 0);
                float midi = t0.x;
                float vel  = t0.y;                       // 0..1
                float px   = clamp((midi - 36.0) / 60.0, 0.0, 1.0) * 0.8 + 0.1;
                pt.pos  = vec2(px, 0.12);                // seed low on screen
                float ang = (hash11(i * 31U + uint(uFrame)) - 0.5) * 2.2;  // upward spread
                float spd = (0.25 + vel * 0.75) * uForce;
                pt.vel  = vec2(sin(ang) * spd / max(uAspect, 1e-3), cos(ang) * spd);
                pt.maxLife = uLifetime > 0.0 ? uLifetime : 0.6 + hash11(i * 97U) * 1.2;
                pt.life = 1.0;
                pt.hue  = fract(midi / 12.0);            // hue by pitch class
            }
            else { pt.life = 0.0; pt.pos = vec2(-10.0); }
        }
        else { pt.life = 0.0; pt.pos = vec2(-10.0); }   // no notes ⇒ park offscreen
    }
    else
    {
        pt.vel.y -= uGravity * uDt;
        pt.pos   += pt.vel * uDt;
    }

    p[i] = pt;
}
)GLSL";

// Point-sprite rasteriser. The vertex shader reads the pool by gl_VertexID (no
// vertex attributes), so an empty VAO suffices.
static const char* kDrawVertSrc = R"GLSL(
#version 430
struct Particle { vec2 pos; vec2 vel; float life; float maxLife; float hue; float pad; };
layout(std430, binding = 0) buffer Pool { Particle p[]; };
uniform float uPointSize;
out float vLife;
out float vHue;
void main()
{
    Particle pt = p[gl_VertexID];
    vLife = clamp(pt.life, 0.0, 1.0);
    vHue  = pt.hue;
    if (pt.life <= 0.0)
    {
        gl_Position  = vec4(-2.0, -2.0, 0.0, 1.0);  // clipped
        gl_PointSize = 0.0;
        return;
    }
    gl_Position  = vec4(pt.pos.x * 2.0 - 1.0, pt.pos.y * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = uPointSize * (0.5 + vLife * 0.8);
}
)GLSL";

static const char* kDrawFragSrc = R"GLSL(
#version 430
in float vLife;
in float vHue;
uniform vec4 uColor;
out vec4 frag;
vec3 hsv2rgb (float h)
{
    vec3 c = abs(fract(h + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0;
    return clamp(c, 0.0, 1.0);
}
void main()
{
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    // smoothstep requires edge0 < edge1. Reversed edges are undefined GLSL and
    // produced zero alpha on Apple's compilers despite working on Mesa.
    float a = (1.0 - smoothstep(0.0, 0.5, r)) * vLife;
    vec3 rgb = uColor.r >= 0.0 ? uColor.rgb : hsv2rgb(vHue);
    frag = vec4(rgb, a * uColor.a);                   // straight alpha (composites via blendNormal)
}
)GLSL";

#if defined (__APPLE__)
// --- GL-4.1 ping-pong-FBO fallback shaders ---------------------------------
//
// macOS desktop GL caps at 4.1: no compute shaders / SSBOs. The same simulation
// runs as a fullscreen-quad FRAGMENT pass that reads the previous particle state
// from two RGBA32F textures and writes the next state to two via MRT (one texel
// per particle, gl_FragCoord.xy ↔ gl_GlobalInvocationID.xy). The draw pass then
// reads the state in the VERTEX shader via texelFetch by gl_VertexID. The update
// logic is a line-for-line port of kComputeSrc so behaviour matches the compute
// path (each macOS export stays internally deterministic — hash(i,frame), no rand).

// Fullscreen triangle from gl_VertexID — no VBO (empty VAO suffices).
static const char* kFbSimVertSrc = R"GLSL(
#version 410 core
void main()
{
    vec2 v = vec2(float((gl_VertexID & 1) << 2) - 1.0,
                  float((gl_VertexID & 2) << 1) - 1.0);
    gl_Position = vec4(v, 0.0, 1.0);
}
)GLSL";

// Particle-update kernel as a fragment shader. Output 0 = pos.xy/vel.xy,
// output 1 = life/maxLife/hue/pad. Mirrors kComputeSrc exactly.
static const char* kFbSimFragSrc = R"GLSL(
#version 410 core
uniform int   uCount;
uniform int   uSpawnTrack;
uniform float uGravity;
uniform float uForce;
uniform float uLifetime;
uniform float uDt;
uniform int   uFrame;
uniform int   uNoteCount;
uniform float uAspect;
uniform sampler2D uNotes;     // 4 x 128 RGBA32F
uniform sampler2D uPosVel;    // prev pos.xy / vel.xy
uniform sampler2D uLife;      // prev life / maxLife / hue / pad
uniform int   uStateW;        // state-grid width (index = y*uStateW + x)
layout(location = 0) out vec4 outPosVel;
layout(location = 1) out vec4 outLife;

float hash11 (uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7fffffffU) / float(0x7fffffff);
}

void main()
{
    ivec2 tc = ivec2(gl_FragCoord.xy);
    uint i = uint(tc.y * uStateW + tc.x);
    vec4 pv = texelFetch(uPosVel, tc, 0);
    vec4 lf = texelFetch(uLife, tc, 0);
    vec2 pos = pv.xy;  vec2 vel = pv.zw;
    float life = lf.x; float maxLife = lf.y; float hue = lf.z;

    if (i >= uint(uCount)) { outPosVel = vec4(-10.0, -10.0, 0.0, 0.0); outLife = vec4(0.0); return; }

    life -= uDt / max(maxLife, 1e-3);

    if (life <= 0.0)
    {
        int matchCount = 0;
        int rows = min(uNoteCount, 128);
        for (int r = 0; r < rows; ++r)
        {
            vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);   // freq, cents, trackId, isRoot
            if (int(t1.z + 0.5) == uSpawnTrack)
            {
                vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0); // midi, vel, age, remain
                if (t0.y > 0.001) ++matchCount;
            }
        }
        if (matchCount > 0)
        {
            int pick = int(hash11(i * 747U + uint(uFrame) * 13U) * float(matchCount));
            pick = min(pick, matchCount - 1);
            int seen = 0; int chosen = -1;
            for (int r = 0; r < rows; ++r)
            {
                vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);
                vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0);
                if (int(t1.z + 0.5) == uSpawnTrack && t0.y > 0.001)
                {
                    if (seen == pick) { chosen = r; break; }
                    ++seen;
                }
            }
            if (chosen >= 0)
            {
                vec4 t0 = texelFetch(uNotes, ivec2(0, chosen), 0);
                float midi = t0.x;
                float v    = t0.y;                       // 0..1
                float px   = clamp((midi - 36.0) / 60.0, 0.0, 1.0) * 0.8 + 0.1;
                pos  = vec2(px, 0.12);                    // seed low on screen
                float ang = (hash11(i * 31U + uint(uFrame)) - 0.5) * 2.2;  // upward spread
                float spd = (0.25 + v * 0.75) * uForce;
                vel  = vec2(sin(ang) * spd / max(uAspect, 1e-3), cos(ang) * spd);
                maxLife = uLifetime > 0.0 ? uLifetime : 0.6 + hash11(i * 97U) * 1.2;
                life = 1.0;
                hue  = fract(midi / 12.0);                // hue by pitch class
            }
            else { life = 0.0; pos = vec2(-10.0); }
        }
        else { life = 0.0; pos = vec2(-10.0); }          // no notes ⇒ park offscreen
    }
    else
    {
        vel.y -= uGravity * uDt;
        pos   += vel * uDt;
    }

    outPosVel = vec4(pos, vel);
    outLife   = vec4(life, maxLife, hue, 0.0);
}
)GLSL";

// Point-sprite draw: the vertex shader reads particle i (gl_VertexID) from the
// state textures via texelFetch (the SSBO read in kDrawVertSrc).
static const char* kFbDrawVertSrc = R"GLSL(
#version 410 core
uniform sampler2D uPosVel;
uniform sampler2D uLife;
uniform int   uStateW;
uniform float uPointSize;
out float vLife;
out float vHue;
void main()
{
    ivec2 tc = ivec2(gl_VertexID % uStateW, gl_VertexID / uStateW);
    vec4 pv = texelFetch(uPosVel, tc, 0);
    vec4 lf = texelFetch(uLife, tc, 0);
    vLife = clamp(lf.x, 0.0, 1.0);
    vHue  = lf.z;
    if (lf.x <= 0.0)
    {
        gl_Position  = vec4(-2.0, -2.0, 0.0, 1.0);  // clipped
        gl_PointSize = 0.0;
        return;
    }
    gl_Position  = vec4(pv.x * 2.0 - 1.0, pv.y * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = uPointSize * (0.5 + vLife * 0.8);
}
)GLSL";

// Identical to kDrawFragSrc but #version 410 (430 won't compile on macOS).
static const char* kFbDrawFragSrc = R"GLSL(
#version 410 core
in float vLife;
in float vHue;
uniform vec4 uColor;
out vec4 frag;
vec3 hsv2rgb (float h)
{
    vec3 c = abs(fract(h + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0;
    return clamp(c, 0.0, 1.0);
}
void main()
{
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    float a = (1.0 - smoothstep(0.0, 0.5, r)) * vLife;
    vec3 rgb = uColor.r >= 0.0 ? uColor.rgb : hsv2rgb(vHue);
    frag = vec4(rgb, a * uColor.a);
}
)GLSL";

#ifndef GL_COLOR_ATTACHMENT1
 #define GL_COLOR_ATTACHMENT1 0x8CE1
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
 #define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
#endif // __APPLE__

// --- helpers ----------------------------------------------------------------

unsigned ParticleEngine::compileCompute (const arbitgl::GlFuncs* gl, const char* src,
                                         std::string& err) const
{
    const unsigned sh = gl->CreateShader (GL_COMPUTE_SHADER);
    gl->ShaderSource (sh, 1, &src, nullptr);
    gl->CompileShader (sh);
    GLint ok = 0; gl->GetShaderiv (sh, GL_COMPILE_STATUS, &ok);
    if (! ok)
    {
        char buf[2048] = {0}; gl->GetShaderInfoLog (sh, sizeof buf, nullptr, buf);
        err = std::string ("compute: ") + buf;
        gl->DeleteShader (sh);
        return 0;
    }
    const unsigned prog = gl->CreateProgram();
    gl->AttachShader (prog, sh);
    gl->LinkProgram (prog);
    gl->DeleteShader (sh);
    GLint linked = 0; gl->GetProgramiv (prog, GL_LINK_STATUS, &linked);
    if (! linked)
    {
        char buf[2048] = {0}; gl->GetProgramInfoLog (prog, sizeof buf, nullptr, buf);
        err = std::string ("compute link: ") + buf;
        gl->DeleteProgram (prog);
        return 0;
    }
    return prog;
}

unsigned ParticleEngine::compileDraw (const arbitgl::GlFuncs* gl, const char* vs,
                                      const char* fs, std::string& err) const
{
    auto one = [&] (GLenum type, const char* s, const char* tag) -> unsigned
    {
        const unsigned sh = gl->CreateShader (type);
        gl->ShaderSource (sh, 1, &s, nullptr);
        gl->CompileShader (sh);
        GLint ok = 0; gl->GetShaderiv (sh, GL_COMPILE_STATUS, &ok);
        if (! ok)
        {
            char buf[2048] = {0}; gl->GetShaderInfoLog (sh, sizeof buf, nullptr, buf);
            err = std::string (tag) + ": " + buf;
            gl->DeleteShader (sh);
            return 0;
        }
        return sh;
    };
    const unsigned v = one (GL_VERTEX_SHADER, vs, "draw vert");
    if (! v) return 0;
    const unsigned f = one (GL_FRAGMENT_SHADER, fs, "draw frag");
    if (! f) { gl->DeleteShader (v); return 0; }
    const unsigned prog = gl->CreateProgram();
    gl->AttachShader (prog, v);
    gl->AttachShader (prog, f);
    gl->LinkProgram (prog);
    gl->DeleteShader (v);
    gl->DeleteShader (f);
    GLint linked = 0; gl->GetProgramiv (prog, GL_LINK_STATUS, &linked);
    if (! linked)
    {
        char buf[2048] = {0}; gl->GetProgramInfoLog (prog, sizeof buf, nullptr, buf);
        err = std::string ("draw link: ") + buf;
        gl->DeleteProgram (prog);
        return 0;
    }
    return prog;
}

bool ParticleEngine::computeAvailable (const arbitgl::GlFuncs* gl)
{
#if defined (__APPLE__)
    // macOS caps GL at 4.1 — no compute/SSBO — but the ping-pong-FBO fragment
    // fallback (renderFallback) runs on any 4.1 core context. Gate on its core
    // entry points: FBO attach + MRT (glDrawBuffers). gpu.particles mirrors this.
    return gl != nullptr
        && gl->GenFramebuffers      != nullptr
        && gl->FramebufferTexture2D != nullptr
        && gl->DrawBuffers          != nullptr;
#else
    return gl != nullptr
        && gl->DispatchCompute != nullptr
        && gl->MemoryBarrier   != nullptr
        && gl->BindBufferBase  != nullptr
        && gl->GenBuffers      != nullptr
        && gl->BufferData      != nullptr;
#endif
}

bool ParticleEngine::ensurePrograms (const arbitgl::GlFuncs* gl)
{
    if (triedPrograms_) return ok_;
    triedPrograms_ = true;

    std::string err;
    computeProg_ = compileCompute (gl, kComputeSrc, err);
    if (computeProg_ == 0) { log_ = err; ok_ = false; return false; }
    drawProg_ = compileDraw (gl, kDrawVertSrc, kDrawFragSrc, err);
    if (drawProg_ == 0) { log_ = err; ok_ = false; return false; }

    uCount_      = gl->GetUniformLocation (computeProg_, "uCount");
    uSpawnTrack_ = gl->GetUniformLocation (computeProg_, "uSpawnTrack");
    uGravity_    = gl->GetUniformLocation (computeProg_, "uGravity");
    uForce_      = gl->GetUniformLocation (computeProg_, "uForce");
    uLifetime_   = gl->GetUniformLocation (computeProg_, "uLifetime");
    uDt_         = gl->GetUniformLocation (computeProg_, "uDt");
    uFrame_      = gl->GetUniformLocation (computeProg_, "uFrame");
    uNoteCount_  = gl->GetUniformLocation (computeProg_, "uNoteCount");
    uAspect_     = gl->GetUniformLocation (computeProg_, "uAspect");
    uNotesC_     = gl->GetUniformLocation (computeProg_, "uNotes");
    uPointSize_  = gl->GetUniformLocation (drawProg_, "uPointSize");
    uColor_      = gl->GetUniformLocation (drawProg_, "uColor");

    if (vao_ == 0) gl->GenVertexArrays (1, &vao_);
    ok_ = true;
    return true;
}

void ParticleEngine::ensurePool (const arbitgl::GlFuncs* gl, int count)
{
    if (ssbo_ != 0 && poolCount_ == count) return;
    poolCount_ = count;
    if (ssbo_ == 0) gl->GenBuffers (1, &ssbo_);
    // Zero-initialise so every particle starts dead (life 0) ⇒ re-seeds on the
    // first frame — the determinism contract (fresh renderer ⇒ identical run).
    std::vector<float> zeros ((size_t) count * kFloatsPerParticle, 0.0f);
    gl->BindBuffer (GL_SHADER_STORAGE_BUFFER, ssbo_);
    gl->BufferData (GL_SHADER_STORAGE_BUFFER,
                    (GLsizeiptr) (zeros.size() * sizeof (float)),
                    zeros.data(), GL_DYNAMIC_DRAW);
    gl->BindBuffer (GL_SHADER_STORAGE_BUFFER, 0);
}

void ParticleEngine::ensureTarget (const arbitgl::GlFuncs* gl, int width, int height)
{
    if (outTex_ != 0 && outW_ == width && outH_ == height) return;
    outW_ = width; outH_ = height;
    if (outTex_ == 0) glGenTextures (1, &outTex_);
    glBindTexture (GL_TEXTURE_2D, outTex_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    if (fbo_ == 0) gl->GenFramebuffers (1, &fbo_);
    gl->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, outTex_, 0);
    gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
}

void ParticleEngine::uploadNotes (const arbitgl::GlFuncs* gl, const NoteFeatures* notes)
{
    (void) gl;
    if (notesTex_ == 0)
    {
        glGenTextures (1, &notesTex_);
        glBindTexture (GL_TEXTURE_2D, notesTex_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // structured texels
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, kNoteTexW, kNoteTexH, 0,
                      GL_RGBA, GL_FLOAT, nullptr);
    }
    // Only re-upload when there are notes; with uNoteCount 0 the kernel never
    // reads the texture, so a stale allocation is harmless (zero-feed contract).
    if (notes != nullptr
        && (int) notes->notesTex.size() >= kNoteTexW * kNoteTexH * 4)
    {
        glBindTexture (GL_TEXTURE_2D, notesTex_);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, kNoteTexW, kNoteTexH, 0,
                      GL_RGBA, GL_FLOAT, notes->notesTex.data());
    }
}

void ParticleEngine::resetSimulation (const arbitgl::GlFuncs* gl)
{
    // Reuse the established complete teardown path. This is uncommon (seek or
    // structural revision) and guarantees all GL/Metal/fallback pools restart
    // from their documented zero/dead state rather than partially rewinding.
    shutdown(gl);
}

int ParticleEngine::planSimSteps (const ShaderClock& clock, int& firstFrame)
{
    if (! clock.playing) { firstFrame = clock.frame; return 0; }
    int steps;
    if (! simSeeded_)
    {
        if (clock.frame < 0 || clock.frame >= kMaxReplaySteps) return -1;
        steps = clock.frame + 1;
        firstFrame = 0;
    }
    else if (clock.frame == lastSimFrame_) { steps = 0; firstFrame = clock.frame; }
    else if (clock.frame <  lastSimFrame_) { return -1; }
    else
    {
        const int d = clock.frame - lastSimFrame_;
        if (d > kMaxReplaySteps) return -1;
        steps = d;
        firstFrame = lastSimFrame_ + 1;
    }
    simSeeded_ = true;
    lastSimFrame_ = clock.frame;
    return steps;
}

unsigned ParticleEngine::render (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                                 int width, int height, const ParticleParams& params,
                                 const NoteFeatures* notes)
{
    // Renderer boundary: unlike older generator code, particles restore every GL
    // state they mutate. This lets callers compose or fail without hidden state
    // coupling; owned object contents are the only persistent side effect.
    struct StateBoundary
    {
        const arbitgl::GlFuncs* gl;
        GLint program = 0, activeTexture = 0, drawFbo = 0, readFbo = 0, vao = 0, ssbo = 0;
        GLint viewport[4] {}, texture2d[3] {}, blendSrcRgb = 0, blendDstRgb = 0;
        GLint blendSrcAlpha = 0, blendDstAlpha = 0, drawBuffer = 0, readBuffer = 0;
        GLfloat clear[4] {};
        GLboolean blend = GL_FALSE, pointSize = GL_FALSE;
        explicit StateBoundary (const arbitgl::GlFuncs* funcs) : gl(funcs)
        {
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
            glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFbo);
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFbo);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
            glGetIntegerv(GL_VIEWPORT, viewport);
            glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcAlpha);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstAlpha);
            glGetIntegerv(GL_DRAW_BUFFER0, &drawBuffer);
            glGetIntegerv(GL_READ_BUFFER, &readBuffer);
            glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
            blend = glIsEnabled(GL_BLEND); pointSize = glIsEnabled(GL_PROGRAM_POINT_SIZE);
            for (int unit = 0; unit < 3; ++unit)
            {
                gl->ActiveTexture(GL_TEXTURE0 + unit);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture2d[unit]);
            }
#if !defined (__APPLE__)
            gl->GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0, &ssbo);
#endif
        }
        ~StateBoundary()
        {
#if !defined (__APPLE__)
            gl->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, (GLuint) ssbo);
#endif
            for (int unit = 0; unit < 3; ++unit)
            {
                gl->ActiveTexture(GL_TEXTURE0 + unit);
                glBindTexture(GL_TEXTURE_2D, (GLuint) texture2d[unit]);
            }
            gl->ActiveTexture((GLenum) activeTexture);
            gl->BindVertexArray((GLuint) vao);
            gl->UseProgram((GLuint) program);
            gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint) drawFbo);
            gl->BindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint) readFbo);
            glDrawBuffer((GLenum) drawBuffer);
            glReadBuffer((GLenum) readBuffer);
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            gl->BlendFuncSeparate((GLenum) blendSrcRgb, (GLenum) blendDstRgb,
                                  (GLenum) blendSrcAlpha, (GLenum) blendDstAlpha);
            glClearColor(clear[0], clear[1], clear[2], clear[3]);
            blend ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
            pointSize ? glEnable(GL_PROGRAM_POINT_SIZE) : glDisable(GL_PROGRAM_POINT_SIZE);
        }
    } boundary(gl);
    if (! computeAvailable (gl)) return 0;
    if (width <= 0 || height <= 0) return 0;
    if (simSeeded_ && clock.frame < lastSimFrame_)
        resetSimulation(gl);

    int count = params.count;
    if (count < 1) count = 1;
    if (count > kMaxParticles) count = kMaxParticles;

#if defined (__APPLE__)
  #if ARBIT_HAVE_METAL_BACKEND
    if (MetalParticleEngine::enabled())
    {
        if (metal_ == nullptr)
            metal_ = std::make_unique<MetalParticleEngine>();
        if (const unsigned texture = metal_->render (gl, clock, width, height,
                                                      params, notes))
        {
            diagnostics_ = metal_->diagnostics();
            ok_ = true;
            log_ = "metal";
            return texture;
        }
        log_ = metal_->log();
    }
  #endif
    // macOS GL 4.1 fallback: ping-pong fragment/MRT simulation.
    return renderFallback (gl, clock, width, height, params, notes, count);
#else
    if (! ensurePrograms (gl)) return 0;
    ensurePool (gl, count);
    ensureTarget (gl, width, height);
    uploadNotes (gl, notes);

    const int noteCount =
        (notes != nullptr && ! notes->notesTex.empty()) ? notes->noteCount : 0;

    // 1) Advance the pool (compute) — frame-gated catch-up (audit #7): one pass
    //    per integer timeline frame crossed since the last call, NOT one per
    //    render(). A single render() at 60Hz over a 30fps project would otherwise
    //    double-step the sim vs the export (which runs one pass per frame).
    int firstFrame = clock.frame;
    const int simSteps = planSimSteps (clock, firstFrame);
    if (simSteps < 0) { log_ = "particle replay exceeds deterministic bound"; return 0; }
    gl->UseProgram (computeProg_);
    gl->Uniform1i (uCount_, count);
    gl->Uniform1i (uSpawnTrack_, params.spawnTrack);
    gl->Uniform1f (uGravity_, params.gravity);
    gl->Uniform1f (uForce_, params.force > 0.0f ? params.force : 0.0f);
    gl->Uniform1f (uLifetime_, params.lifetime);
    gl->Uniform1f (uDt_, clock.playing ? (float) clock.timeDelta : 0.0f);
    gl->Uniform1i (uNoteCount_, noteCount);
    gl->Uniform1f (uAspect_, height > 0 ? (float) width / (float) height : 1.0f);
    gl->ActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, notesTex_);
    gl->Uniform1i (uNotesC_, 0);
#if !defined (__APPLE__)
    gl->BindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, ssbo_);
    for (int s = 0; s < simSteps; ++s)   // simSteps==0 ⇒ hold the pool, rasterise as-is
    {
        gl->Uniform1i (uFrame_, firstFrame + s + params.seed);
        gl->DispatchCompute ((unsigned) ((count + 255) / 256), 1, 1);
        gl->MemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT);
    }
#endif

    // 2) Rasterise as point sprites into the layer texture.
    gl->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    glViewport (0, 0, width, height);
    glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_PROGRAM_POINT_SIZE);
    gl->UseProgram (drawProg_);
    gl->Uniform1f (uPointSize_, params.size > 0.0f ? params.size : 1.0f);
    gl->Uniform4f (uColor_, params.red, params.green, params.blue, params.alpha);
#if !defined (__APPLE__)
    gl->BindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, ssbo_);
#endif
    gl->BindVertexArray (vao_);
    glDrawArrays (GL_POINTS, 0, count);
    gl->BindVertexArray (0);
    glDisable (GL_PROGRAM_POINT_SIZE);
    glDisable (GL_BLEND);

    // Caller restores its own FBO + viewport (matches ShaderGenerator::render).
    return outTex_;
#endif // __APPLE__
}

#if defined (__APPLE__)
bool ParticleEngine::ensureFallback (const arbitgl::GlFuncs* gl, int count)
{
    // Compile the two fallback programs once (reusing the shared failure latch).
    if (! triedPrograms_)
    {
        triedPrograms_ = true;
        std::string err;
        simProg_ = compileDraw (gl, kFbSimVertSrc, kFbSimFragSrc, err);
        if (simProg_ == 0) { log_ = err; ok_ = false; return false; }
        drawFbProg_ = compileDraw (gl, kFbDrawVertSrc, kFbDrawFragSrc, err);
        if (drawFbProg_ == 0) { log_ = err; ok_ = false; return false; }

        sUCount_      = gl->GetUniformLocation (simProg_, "uCount");
        sUSpawnTrack_ = gl->GetUniformLocation (simProg_, "uSpawnTrack");
        sUGravity_    = gl->GetUniformLocation (simProg_, "uGravity");
        sUForce_      = gl->GetUniformLocation (simProg_, "uForce");
        sULifetime_   = gl->GetUniformLocation (simProg_, "uLifetime");
        sUDt_         = gl->GetUniformLocation (simProg_, "uDt");
        sUFrame_      = gl->GetUniformLocation (simProg_, "uFrame");
        sUNoteCount_  = gl->GetUniformLocation (simProg_, "uNoteCount");
        sUAspect_     = gl->GetUniformLocation (simProg_, "uAspect");
        sUNotes_      = gl->GetUniformLocation (simProg_, "uNotes");
        sUPosVel_     = gl->GetUniformLocation (simProg_, "uPosVel");
        sULife_       = gl->GetUniformLocation (simProg_, "uLife");
        sUStateW_     = gl->GetUniformLocation (simProg_, "uStateW");
        dUPosVel_     = gl->GetUniformLocation (drawFbProg_, "uPosVel");
        dULife_       = gl->GetUniformLocation (drawFbProg_, "uLife");
        dUStateW_     = gl->GetUniformLocation (drawFbProg_, "uStateW");
        dUPointSize_ = gl->GetUniformLocation (drawFbProg_, "uPointSize");
        dUColor_     = gl->GetUniformLocation (drawFbProg_, "uColor");

        if (vao_ == 0) gl->GenVertexArrays (1, &vao_);
        ok_ = true;
    }
    if (! ok_) return false;

    // (Re)allocate the two ping-pong state sets when the pool size changes. One
    // texel per particle on a 256-wide grid; both sets zero-initialised so every
    // particle starts dead (life 0) ⇒ re-seeds on frame 0 — the determinism
    // contract (fresh engine ⇒ identical run), matching ensurePool's zero-fill.
    if (fbPoolCount_ != count || posVelTex_[0] == 0)
    {
        fbPoolCount_ = count;
        stateTexW_ = 256;
        stateTexH_ = (count + stateTexW_ - 1) / stateTexW_;
        if (stateTexH_ < 1) stateTexH_ = 1;
        const std::vector<float> zeros ((size_t) stateTexW_ * (size_t) stateTexH_ * 4, 0.0f);
        for (int s = 0; s < 2; ++s)
        {
            if (posVelTex_[s] == 0) glGenTextures (1, &posVelTex_[s]);
            if (lifeTex_[s]   == 0) glGenTextures (1, &lifeTex_[s]);
            const unsigned texes[2] = { posVelTex_[s], lifeTex_[s] };
            for (unsigned tex : texes)
            {
                glBindTexture (GL_TEXTURE_2D, tex);
                glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, stateTexW_, stateTexH_, 0,
                              GL_RGBA, GL_FLOAT, zeros.data());
            }
            if (simFbo_[s] == 0) gl->GenFramebuffers (1, &simFbo_[s]);
            gl->BindFramebuffer (GL_FRAMEBUFFER, simFbo_[s]);
            gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, posVelTex_[s], 0);
            gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                                      GL_TEXTURE_2D, lifeTex_[s], 0);
            // RGBA32F is required colour-renderable in core GL 3.0+, so this MRT
            // FBO should always be complete on a 4.1 core context — but this path
            // has never run on real hardware, so verify rather than assume. A
            // driver that rejects 32F-MRT latches ok_ = false (render() then
            // returns 0 ⇒ the layer renders nothing, never garbage/crash) with a
            // diagnostic, matching how ensurePrograms latches a compile failure.
            const unsigned st = gl->CheckFramebufferStatus (GL_FRAMEBUFFER);
            if (st != GL_FRAMEBUFFER_COMPLETE)
            {
                gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
                log_ = "particle fallback FBO incomplete (RGBA32F MRT unsupported?), status "
                     + std::to_string (st);
                ok_ = false;
                return false;
            }
        }
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        fbRead_ = 0;
    }
    return true;
}

unsigned ParticleEngine::renderFallback (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                                         int width, int height, const ParticleParams& params,
                                         const NoteFeatures* notes, int count)
{
    if (! ensureFallback (gl, count)) return 0;
    ensureTarget (gl, width, height);
    uploadNotes (gl, notes);

    const int noteCount =
        (notes != nullptr && ! notes->notesTex.empty()) ? notes->noteCount : 0;
    diagnostics_ = {};
    diagnostics_.noteRows = noteCount;

    // 1) Advance the pool — frame-gated catch-up (audit #7), mirroring the
    //    compute path: one fullscreen sim pass per integer timeline frame crossed
    //    (ping-ponging the state textures), NOT one per render() call, so the
    //    60Hz preview reproduces the export's per-frame sequence. simSteps==0 ⇒
    //    hold the pool and rasterise the current state.
    int firstFrame = clock.frame;
    const int simSteps = planSimSteps (clock, firstFrame);
    gl->UseProgram (simProg_);
    gl->Uniform1i (sUCount_, count);
    gl->Uniform1i (sUSpawnTrack_, params.spawnTrack);
    gl->Uniform1f (sUGravity_, params.gravity);
    gl->Uniform1f (sUForce_, params.force > 0.0f ? params.force : 0.0f);
    gl->Uniform1f (sULifetime_, params.lifetime);
    gl->Uniform1f (sUDt_, clock.playing ? (float) clock.timeDelta : 0.0f);
    gl->Uniform1i (sUNoteCount_, noteCount);
    gl->Uniform1f (sUAspect_, height > 0 ? (float) width / (float) height : 1.0f);
    gl->Uniform1i (sUStateW_, stateTexW_);
    gl->ActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, notesTex_);
    gl->Uniform1i (sUNotes_, 0);
    gl->Uniform1i (sUPosVel_, 1);
    gl->Uniform1i (sULife_, 2);
    const GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDisable (GL_BLEND);
    gl->BindVertexArray (vao_);
    for (int s = 0; s < simSteps; ++s)
    {
        const int writeIdx = 1 - fbRead_;
        gl->Uniform1i (sUFrame_, firstFrame + s + params.seed);
        gl->ActiveTexture (GL_TEXTURE1);
        glBindTexture (GL_TEXTURE_2D, posVelTex_[fbRead_]);
        gl->ActiveTexture (GL_TEXTURE2);
        glBindTexture (GL_TEXTURE_2D, lifeTex_[fbRead_]);
        gl->BindFramebuffer (GL_FRAMEBUFFER, simFbo_[writeIdx]);
        gl->DrawBuffers (2, bufs);
        glViewport (0, 0, stateTexW_, stateTexH_);
        glDrawArrays (GL_TRIANGLES, 0, 3);
        fbRead_ = writeIdx;          // just-written set becomes current
    }

    const char* diagnostic = std::getenv ("ARBIT_PARTICLE_DIAGNOSTICS");
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        std::vector<float> life ((size_t) stateTexW_ * (size_t) stateTexH_ * 4);
        std::vector<float> posVel ((size_t) stateTexW_ * (size_t) stateTexH_ * 4);
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, simFbo_[fbRead_]);
        glReadBuffer (GL_COLOR_ATTACHMENT1);
        glReadPixels (0, 0, stateTexW_, stateTexH_, GL_RGBA, GL_FLOAT, life.data());
        glReadBuffer (GL_COLOR_ATTACHMENT0);
        glReadPixels (0, 0, stateTexW_, stateTexH_, GL_RGBA, GL_FLOAT, posVel.data());
        diagnostics_.liveParticles = 0;
        diagnostics_.visibleCandidates = 0;
        for (int i = 0; i < count; ++i)
        {
            const auto base = (size_t) i * 4;
            if (life[base] > 0.0f)
            {
                ++diagnostics_.liveParticles;
                if (std::isfinite (posVel[base]) && std::isfinite (posVel[base + 1])
                    && posVel[base] >= 0.0f && posVel[base] <= 1.0f
                    && posVel[base + 1] >= 0.0f && posVel[base + 1] <= 1.0f)
                    ++diagnostics_.visibleCandidates;
            }
        }
    }

    // 2) Rasterise as point sprites into the layer texture (outTex_/fbo_).
    gl->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    const GLenum one = GL_COLOR_ATTACHMENT0;
    gl->DrawBuffers (1, &one);   // restore single draw buffer on the layer FBO
    glViewport (0, 0, width, height);
    glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_PROGRAM_POINT_SIZE);
    gl->UseProgram (drawFbProg_);
    gl->Uniform1f (dUPointSize_, params.size > 0.0f ? params.size : 1.0f);
    gl->Uniform4f (dUColor_, params.red, params.green, params.blue, params.alpha);
    gl->Uniform1i (dUStateW_, stateTexW_);
    gl->ActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, posVelTex_[fbRead_]);   // fbRead_ = latest after catch-up
    gl->Uniform1i (dUPosVel_, 0);
    gl->ActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, lifeTex_[fbRead_]);
    gl->Uniform1i (dULife_, 1);
    gl->BindVertexArray (vao_);
    glDrawArrays (GL_POINTS, 0, count);
    gl->BindVertexArray (0);
    glDisable (GL_PROGRAM_POINT_SIZE);
    glDisable (GL_BLEND);
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        std::vector<unsigned char> pixels ((size_t) width * (size_t) height * 4);
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, fbo_);
        glReadBuffer (GL_COLOR_ATTACHMENT0);
        glReadPixels (0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        diagnostics_.drawAlphaPixels = 0;
        for (size_t i = 3; i < pixels.size(); i += 4)
            if (pixels[i] != 0) ++diagnostics_.drawAlphaPixels;
        diagnostics_.readbackAlphaPixels = diagnostics_.drawAlphaPixels;
    }
    // fbRead_ already points at the just-written set (advanced inside the loop).
    // Caller restores its own FBO + viewport (matches ShaderGenerator::render).
    ok_ = true;
    log_ = "opengl-fragment";
    return outTex_;
}
#endif // __APPLE__

void ParticleEngine::shutdown (const arbitgl::GlFuncs* gl)
{
    if (gl == nullptr) return;
    if (computeProg_) { gl->DeleteProgram (computeProg_); computeProg_ = 0; }
    if (drawProg_)    { gl->DeleteProgram (drawProg_);    drawProg_ = 0; }
    if (ssbo_)        { gl->DeleteBuffers (1, &ssbo_);    ssbo_ = 0; }
    if (vao_)         { gl->DeleteVertexArrays (1, &vao_); vao_ = 0; }
    if (fbo_)         { gl->DeleteFramebuffers (1, &fbo_); fbo_ = 0; }
    if (outTex_)      { glDeleteTextures (1, &outTex_);   outTex_ = 0; }
    if (notesTex_)    { glDeleteTextures (1, &notesTex_); notesTex_ = 0; }
#if defined (__APPLE__)
  #if ARBIT_HAVE_METAL_BACKEND
    if (metal_ != nullptr)
    {
        metal_->shutdown (gl);
        metal_.reset();
    }
  #endif
    if (simProg_)    { gl->DeleteProgram (simProg_);    simProg_ = 0; }
    if (drawFbProg_) { gl->DeleteProgram (drawFbProg_); drawFbProg_ = 0; }
    for (int s = 0; s < 2; ++s)
    {
        if (posVelTex_[s]) { glDeleteTextures (1, &posVelTex_[s]); posVelTex_[s] = 0; }
        if (lifeTex_[s])   { glDeleteTextures (1, &lifeTex_[s]);   lifeTex_[s] = 0; }
        if (simFbo_[s])    { gl->DeleteFramebuffers (1, &simFbo_[s]); simFbo_[s] = 0; }
    }
    fbPoolCount_ = -1; stateTexW_ = stateTexH_ = 0; fbRead_ = 0;
#endif
    poolCount_ = 0; outW_ = outH_ = 0;
    triedPrograms_ = false; ok_ = false;
    simSeeded_ = false; lastSimFrame_ = 0;   // restart the frame-gated catch-up clean
    diagnostics_ = {};
}

ParticleEngine::~ParticleEngine()
{
    // GL teardown is explicit via shutdown(); the dtor must not touch GL.
}

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
