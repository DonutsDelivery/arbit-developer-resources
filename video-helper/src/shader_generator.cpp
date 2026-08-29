// shader_generator.cpp — see shader_generator.h.

#include "shader_generator.h"

#if ARBIT_HAVE_VIEWPORT

#include "gl_loader.h"
#include "shader_dialect.h"

#include <cmath>
#include <set>

namespace videorender
{

// Minimal vertex stage: a fullscreen NDC quad. The contract fragment shader
// reads gl_FragCoord/uResolution, so no varyings are needed (unlike the
// renderer's kVertexShader, which threads TexCoord + uTransform). Keeping our
// own vertex shader means a generator never depends on uTransform being set.
static const char* kGenVertex = R"GLSL(#version 330 core
layout (location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

// Fullscreen triangle strip in NDC.
static const float kGenQuad[8] = { -1.f, -1.f,  1.f, -1.f,  -1.f, 1.f,  1.f, 1.f };

ShaderClock makeShaderClock (double displaySec, double clipStartSec,
                             double clipDurationSec,
                             const videotime::BeatTimeline& timeline, double fps,
                             bool playing, int frame)
{
    ShaderClock c;
    const double clipRel = displaySec - clipStartSec;
    const auto timelineClock = timeline.clockAtSeconds (displaySec);
    const double clipStartBeat = timeline.secondsToBeat (clipStartSec);
    const double clipEndBeat = timeline.secondsToBeat (clipStartSec + clipDurationSec);
    c.timeSec     = clipRel > 0.0 ? clipRel : 0.0;
    c.timeDelta   = fps > 1e-6 ? 1.0 / fps : 1.0 / 30.0;
    c.frame       = frame;
    c.bpm         = timelineClock.bpm;
    c.beatsPerBar = timelineClock.beatsPerBar;
    c.beat        = timelineClock.beat;
    c.beatPhase   = c.beat - std::floor (c.beat);
    c.barPhase    = timelineClock.barPhase;
    c.clipBeat    = std::max (0.0, c.beat - clipStartBeat);
    c.clipLength  = std::max (0.0, clipEndBeat - clipStartBeat);
    c.playing     = playing;
    return c;
}

ShaderGenerator::~ShaderGenerator()
{
    // GL teardown is explicit (shutdown). Nothing to do here — touching GL
    // without a current context would be a bug. Handles, if any, are leaked to
    // the driver on process exit exactly like FrameRenderer's own programs.
}

unsigned ShaderGenerator::compile (const arbitgl::GlFuncs* gl,
                                   const std::string& wrappedGlsl,
                                   std::string& errorOut) const
{
    auto stage = [&] (GLenum type, const char* src) -> unsigned
    {
        const unsigned sh = gl->CreateShader (type);
        gl->ShaderSource (sh, 1, &src, nullptr);
        gl->CompileShader (sh);
        GLint okv = GL_FALSE;
        gl->GetShaderiv (sh, GL_COMPILE_STATUS, &okv);
        if (okv != GL_TRUE)
        {
            char info[4096] = {};
            GLsizei len = 0;
            gl->GetShaderInfoLog (sh, (GLsizei) sizeof (info) - 1, &len, info);
            errorOut = std::string (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
                     + " compile: " + info;
            gl->DeleteShader (sh);
            return 0;
        }
        return sh;
    };

    const unsigned vs = stage (GL_VERTEX_SHADER, kGenVertex);
    if (vs == 0) return 0;
    const char* fsrc = wrappedGlsl.c_str();
    const unsigned fs = stage (GL_FRAGMENT_SHADER, fsrc);
    if (fs == 0) { gl->DeleteShader (vs); return 0; }

    const unsigned prog = gl->CreateProgram();
    gl->AttachShader (prog, vs);
    gl->AttachShader (prog, fs);
    gl->LinkProgram (prog);
    gl->DeleteShader (vs);
    gl->DeleteShader (fs);

    GLint okv = GL_FALSE;
    gl->GetProgramiv (prog, GL_LINK_STATUS, &okv);
    if (okv != GL_TRUE)
    {
        char info[4096] = {};
        GLsizei len = 0;
        gl->GetProgramInfoLog (prog, (GLsizei) sizeof (info) - 1, &len, info);
        errorOut = std::string ("link: ") + info;
        gl->DeleteProgram (prog);
        return 0;
    }
    return prog;
}

void ShaderGenerator::cacheLocs (const arbitgl::GlFuncs* gl, unsigned prog)
{
    auto L = [&] (const char* n) { return gl->GetUniformLocation (prog, n); };
    locs_.uResolution   = L ("uResolution");
    locs_.uTime         = L ("uTime");
    locs_.uTimeDelta    = L ("uTimeDelta");
    locs_.uFrame        = L ("uFrame");
    locs_.uBeat         = L ("uBeat");
    locs_.uBPM          = L ("uBPM");
    locs_.uBeatPhase    = L ("uBeatPhase");
    locs_.uBarPhase     = L ("uBarPhase");
    locs_.uBeatsPerBar  = L ("uBeatsPerBar");
    locs_.uClipBeat     = L ("uClipBeat");
    locs_.uClipLength   = L ("uClipLength");
    locs_.uPlaying      = L ("uPlaying");
    locs_.uRMS          = L ("uRMS");
    locs_.uPeak         = L ("uPeak");
    locs_.uOnset        = L ("uOnset");
    locs_.uOnsetAge     = L ("uOnsetAge");
    locs_.uAudioBands   = L ("uAudioBands");
    locs_.uAudioBands2D = L ("uAudioBands2D");
    locs_.uNotes        = L ("uNotes");
    locs_.uLinks        = L ("uLinks");
    locs_.uNoteCount    = L ("uNoteCount");
    locs_.uLinkCount    = L ("uLinkCount");
    locs_.uScoreHistoryBeats = L ("uScoreHistoryBeats");
    locs_.uScoreLookaheadBeats = L ("uScoreLookaheadBeats");
    locs_.uLastOnsetBeat= L ("uLastOnsetBeat");
    locs_.uRootFreq     = L ("uRootFreq");
    locs_.uChord        = L ("uChord");
    // Block D — camera (P5).
    locs_.uCamPos       = L ("uCamPos");
    locs_.uCamTarget    = L ("uCamTarget");
    locs_.uCamUp        = L ("uCamUp");
    locs_.uCamFov       = L ("uCamFov");
    locs_.uCamNear      = L ("uCamNear");
    locs_.uCamFar       = L ("uCamFar");
}

bool ShaderGenerator::setSource (const arbitgl::GlFuncs* gl, const std::string& rawSource)
{
    arbitshader::WrapResult wrap = arbitshader::wrapToContract (rawSource);

    // Compose a human/agent-readable log from diagnostics (errors first).
    std::string diag;
    for (const auto& d : wrap.diagnostics)
        diag += (d.level == arbitshader::WrapDiagnostic::Error ? "error: " : "warning: ")
              + d.message + "\n";

    if (wrap.hasError())
    {
        log_ = diag;
        ok_ = false;
        return false;     // keep last good program (if any)
    }

    std::string glErr;
    const unsigned prog = compile (gl, wrap.glsl, glErr);
    if (prog == 0)
    {
        log_ = diag + "error: " + glErr + "\n";
        ok_ = false;
        return false;     // keep last good program (if any)
    }

    if (program_ != 0)
        gl->DeleteProgram (program_);
    program_ = prog;
    cacheLocs (gl, program_);

    params_.clear();
    params_.reserve (wrap.params.size());
    for (const auto& gi : wrap.params)
    {
        GenParam gp { gi.name, (int) gi.type, gi.defaultScalar, gi.minScalar, gi.maxScalar,
                      { 0.0, 0.0, 0.0, 0.0 }, gi.importedPath };
        for (int k = 0; k < 4; ++k) gp.defaultVec[k] = gi.defaultVec[(size_t) k];
        params_.push_back (std::move (gp));
    }

    // M7: cache each generator param's uniform location now that params_ and the
    // program are both current (the prelude declared `uniform <type> <name>;` for
    // every ISF INPUT). A param the compiler dropped as unused reports -1 and is
    // skipped at upload time.
    paramLocs_.clear();
    paramLocs_.reserve (params_.size());
    for (const auto& p : params_)
        paramLocs_.push_back (gl->GetUniformLocation (program_, p.name.c_str()));

    // ISF multipass (M7): capture the PASS list from the wrap and cache the
    // PASSINDEX + per-TARGET sampler locations. The target buffers themselves are
    // program-independent, but their NAMES belong to this source, so release any
    // left from a previous shader (GL is current here — setSource compiled).
    releasePassTargets (gl);
    passes_.clear();
    targetSamplerLocs_.clear();
    multipass_ = false;
    passParity_ = 0;
    for (const auto& ps : wrap.passes)
    {
        passes_.push_back ({ ps.target, ps.persistent });
        if (ps.persistent) multipass_ = true;
    }
    if (passes_.size() > 1) multipass_ = true;
    passIndexLoc_ = gl->GetUniformLocation (program_, "PASSINDEX");
    if (multipass_)
        for (const auto& ps : passes_)
            if (! ps.target.empty()
                && targetSamplerLocs_.find (ps.target) == targetSamplerLocs_.end())
                targetSamplerLocs_[ps.target] = gl->GetUniformLocation (program_, ps.target.c_str());

    ok_ = true;
    log_ = diag;          // warnings only (or empty)
    return true;
}

void ShaderGenerator::ensureQuad (const arbitgl::GlFuncs* gl)
{
    if (vao_ != 0) return;
    gl->GenVertexArrays (1, &vao_);
    gl->BindVertexArray (vao_);
    gl->GenBuffers (1, &vbo_);
    gl->BindBuffer (GL_ARRAY_BUFFER, vbo_);
    gl->BufferData (GL_ARRAY_BUFFER, sizeof (kGenQuad), kGenQuad, GL_STATIC_DRAW);
    gl->VertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (float), (const void*) 0);
    gl->EnableVertexAttribArray (0);
    gl->BindVertexArray (0);
    gl->GenFramebuffers (1, &fbo_);
}

void ShaderGenerator::ensureDefaults (const arbitgl::GlFuncs* gl)
{
    (void) gl;
    if (blackTex2D_ == 0)
    {
        const unsigned char zero[4] = { 0, 0, 0, 0 };
        glGenTextures (1, &blackTex2D_);
        glBindTexture (GL_TEXTURE_2D, blackTex2D_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, zero);
    }
    if (blackTex1D_ == 0)
    {
        const unsigned char zero = 0;
        glGenTextures (1, &blackTex1D_);
        glBindTexture (GL_TEXTURE_1D, blackTex1D_);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexImage1D (GL_TEXTURE_1D, 0, GL_R8, 1, 0, GL_RED, GL_UNSIGNED_BYTE, &zero);
    }
}

// Block C texture dims (M5) — a fixed wire contract mirroring block_c_packer.h
// (kTexelsPerNote, kMaxNotes, kMaxLinks) and the shader prelude's sampler
// comments. uNotes = 4 texels (x) x 128 rows (y) RGBA32F; uLinks = 256 x 1.
static constexpr int kNoteTexW = 4;     // texels per note (x)
static constexpr int kNoteTexH = 128;   // note rows (y)
static constexpr int kLinkTexW = 256;   // link edges (x)

void ShaderGenerator::ensureBandsTex (const arbitgl::GlFuncs* gl, int n)
{
    (void) gl;
    if (n < 1) n = 1;
    if (bandsTex1D_ != 0 && bandsTex2D_ != 0 && n == bandsTexN_)
        return;   // already allocated at this size; render() will glTexSubImage.

    if (bandsTex1D_ == 0)
    {
        glGenTextures (1, &bandsTex1D_);
        glBindTexture (GL_TEXTURE_1D, bandsTex1D_);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }
    glBindTexture (GL_TEXTURE_1D, bandsTex1D_);
    glTexImage1D (GL_TEXTURE_1D, 0, GL_R32F, n, 0, GL_RED, GL_FLOAT, nullptr);

    if (bandsTex2D_ == 0)
    {
        glGenTextures (1, &bandsTex2D_);
        glBindTexture (GL_TEXTURE_2D, bandsTex2D_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture (GL_TEXTURE_2D, bandsTex2D_);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_R32F, n, 1, 0, GL_RED, GL_FLOAT, nullptr);

    bandsTexN_ = n;
}

void ShaderGenerator::ensureNotesTex (const arbitgl::GlFuncs* gl)
{
    (void) gl;
    if (notesTexAlloc_) return;   // fixed-size wire contract — allocate once.

    glGenTextures (1, &notesTex2D_);
    glBindTexture (GL_TEXTURE_2D, notesTex2D_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // structured texels
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, kNoteTexW, kNoteTexH, 0,
                  GL_RGBA, GL_FLOAT, nullptr);

    glGenTextures (1, &linksTex2D_);
    glBindTexture (GL_TEXTURE_2D, linksTex2D_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, kLinkTexW, 1, 0,
                  GL_RGBA, GL_FLOAT, nullptr);

    notesTexAlloc_ = true;
}

void ShaderGenerator::ensureTarget (const arbitgl::GlFuncs* gl, int width, int height)
{
    (void) gl;
    width = width > 0 ? width : 1;
    height = height > 0 ? height : 1;
    if (outTex_ != 0 && width == outW_ && height == outH_) return;
    if (outTex_ != 0) { glDeleteTextures (1, &outTex_); outTex_ = 0; }
    outW_ = width;
    outH_ = height;
    glGenTextures (1, &outTex_);
    glBindTexture (GL_TEXTURE_2D, outTex_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // HDR float (visuals-quality increment 1): a generator shader may output
    // values >1.0; keeping this target RGBA16F preserves that headroom into the
    // composite so a later bloom/tonemap pass can use it. Were this RGBA8 the
    // shader output would clamp here, before HDR could mean anything downstream.
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA16F, outW_, outH_, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
}

void ShaderGenerator::releasePassTargets (const arbitgl::GlFuncs* gl)
{
    (void) gl;
    for (auto& kv : passTargets_)
        for (int b = 0; b < 2; ++b)
            if (kv.second.tex[b] != 0) { glDeleteTextures (1, &kv.second.tex[b]); kv.second.tex[b] = 0; }
    passTargets_.clear();
    passTargW_ = passTargH_ = 0;
}

void ShaderGenerator::ensurePassTargets (const arbitgl::GlFuncs* gl, int width, int height)
{
    if (! multipass_) return;
    width  = width  > 0 ? width  : 1;
    height = height > 0 ? height : 1;

    // The render size drives every TARGET (per-pass WIDTH/HEIGHT is not evaluated
    // in v1 — documented in the wrap warning). On a resize, drop and reallocate.
    if (width != passTargW_ || height != passTargH_)
        releasePassTargets (gl);
    passTargW_ = width;
    passTargH_ = height;

    // The output pass uses outTex_; every NAMED target gets its own buffer. A
    // target is double-buffered when ANY pass writing it is PERSISTENT (so a
    // pass can read the previous frame while writing this one). New buffers are
    // cleared to black so a PERSISTENT first-frame read is defined (not garbage).
    for (const auto& ps : passes_)
    {
        if (ps.target.empty()) continue;
        PassTarget& pt = passTargets_[ps.target];
        if (ps.persistent) pt.persistent = true;
        const int n = pt.persistent ? 2 : 1;
        for (int b = 0; b < n; ++b)
        {
            if (pt.tex[b] != 0) continue;
            glGenTextures (1, &pt.tex[b]);
            glBindTexture (GL_TEXTURE_2D, pt.tex[b]);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
                          GL_RGBA, GL_HALF_FLOAT, nullptr);
            gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, pt.tex[b], 0);
            glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
            glClear (GL_COLOR_BUFFER_BIT);
        }
    }
}

unsigned ShaderGenerator::render (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                                  int width, int height, const AudioFeatures* audio,
                                  const NoteFeatures* notes,
                                  const std::map<std::string, double>* genValues,
                                  const std::map<std::string, unsigned>* genImages)
{
    if (program_ == 0)
        return 0;

    ensureQuad (gl);
    ensureDefaults (gl);
    ensureTarget (gl, width, height);

    gl->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, outTex_, 0);
    glViewport (0, 0, outW_, outH_);
    glDisable (GL_BLEND);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);

    gl->UseProgram (program_);

    // Block A — clock (live).
    if (locs_.uResolution  >= 0) gl->Uniform2f (locs_.uResolution, (float) outW_, (float) outH_);
    if (locs_.uTime        >= 0) gl->Uniform1f (locs_.uTime, (float) clock.timeSec);
    if (locs_.uTimeDelta   >= 0) gl->Uniform1f (locs_.uTimeDelta, (float) clock.timeDelta);
    if (locs_.uFrame       >= 0) gl->Uniform1i (locs_.uFrame, clock.frame);
    if (locs_.uBeat        >= 0) gl->Uniform1f (locs_.uBeat, (float) clock.beat);
    if (locs_.uBPM         >= 0) gl->Uniform1f (locs_.uBPM, (float) clock.bpm);
    if (locs_.uBeatPhase   >= 0) gl->Uniform1f (locs_.uBeatPhase, (float) clock.beatPhase);
    if (locs_.uBarPhase    >= 0) gl->Uniform1f (locs_.uBarPhase, (float) clock.barPhase);
    if (locs_.uBeatsPerBar >= 0) gl->Uniform1f (locs_.uBeatsPerBar, (float) clock.beatsPerBar);
    if (locs_.uClipBeat    >= 0) gl->Uniform1f (locs_.uClipBeat, (float) clock.clipBeat);
    if (locs_.uClipLength  >= 0) gl->Uniform1f (locs_.uClipLength, (float) clock.clipLength);
    if (locs_.uPlaying     >= 0) gl->Uniform1i (locs_.uPlaying, clock.playing ? 1 : 0);

    // Block B — audio features (live when `audio` present, else zero-fed). M4.
    const float rms      = audio ? audio->rms      : 0.0f;
    const float peak     = audio ? audio->peak     : 0.0f;
    const float onset    = audio ? audio->onset    : 0.0f;
    const float onsetAge = audio ? audio->onsetAge : 0.0f;
    if (locs_.uRMS          >= 0) gl->Uniform1f (locs_.uRMS, rms);
    if (locs_.uPeak         >= 0) gl->Uniform1f (locs_.uPeak, peak);
    if (locs_.uOnset        >= 0) gl->Uniform1f (locs_.uOnset, onset);
    if (locs_.uOnsetAge     >= 0) gl->Uniform1f (locs_.uOnsetAge, onsetAge);
    // Block C — symbolic score (live when `notes` present, else zero-fed). M5.
    const bool haveNotes = notes != nullptr && ! notes->notesTex.empty();
    if (locs_.uNoteCount    >= 0) gl->Uniform1i (locs_.uNoteCount, haveNotes ? notes->noteCount : 0);
    if (locs_.uLinkCount    >= 0) gl->Uniform1i (locs_.uLinkCount, haveNotes ? notes->linkCount : 0);
    if (locs_.uRootFreq     >= 0) gl->Uniform1f (locs_.uRootFreq, haveNotes ? notes->rootFreq : 0.0f);
    if (locs_.uScoreHistoryBeats >= 0)
        gl->Uniform1f (locs_.uScoreHistoryBeats, haveNotes ? notes->historyBeats : 0.0f);
    if (locs_.uScoreLookaheadBeats >= 0)
        gl->Uniform1f (locs_.uScoreLookaheadBeats, haveNotes ? notes->lookaheadBeats : 0.0f);
    // uLastOnsetBeat / uChord stay reserved (in Locs, not in the v1 prelude) —
    // a later slice can fill them without a contract change.
    if (locs_.uLastOnsetBeat>= 0) gl->Uniform1f (locs_.uLastOnsetBeat, 0.0f);
    if (locs_.uChord        >= 0) gl->Uniform4f (locs_.uChord, 0.0f, 0.0f, 0.0f, 0.0f);

    // Block D — camera (P5 raymarch). Contract uniforms any shader may read; a
    // Raymarch clip ships uCamPos/uCamTarget/uCamFov as genParams so the camera
    // automates + mod-routes like any other value. Keys are the uniform name +
    // .x/.y/.z component suffix, matching GeneratorSpec::seedRaymarchDefaults
    // plugin-side; uCamUp/uCamNear/uCamFar use static defaults in v1. The camera
    // is program-global, set once here (NOT per multipass pass).
    auto camVal = [&] (const char* key, float dflt) -> float
    {
        if (genValues != nullptr)
            if (const auto it = genValues->find (key); it != genValues->end())
                return (float) it->second;
        return dflt;
    };
    if (locs_.uCamPos    >= 0) gl->Uniform3f (locs_.uCamPos,
                                   camVal ("uCamPos.x", 0.0f), camVal ("uCamPos.y", 0.0f), camVal ("uCamPos.z", 5.0f));
    if (locs_.uCamTarget >= 0) gl->Uniform3f (locs_.uCamTarget,
                                   camVal ("uCamTarget.x", 0.0f), camVal ("uCamTarget.y", 0.0f), camVal ("uCamTarget.z", 0.0f));
    if (locs_.uCamUp     >= 0) gl->Uniform3f (locs_.uCamUp,
                                   camVal ("uCamUp.x", 0.0f), camVal ("uCamUp.y", 1.0f), camVal ("uCamUp.z", 0.0f));
    if (locs_.uCamFov    >= 0) gl->Uniform1f (locs_.uCamFov,  camVal ("uCamFov", 45.0f));
    if (locs_.uCamNear   >= 0) gl->Uniform1f (locs_.uCamNear, camVal ("uCamNear", 0.05f));
    if (locs_.uCamFar    >= 0) gl->Uniform1f (locs_.uCamFar,  camVal ("uCamFar", 60.0f));

    // ISF INPUTS (M7) — upload each generator param to its uniform: the caller's
    // value for "clip<id>/gen/<name><component>" (genValues) when set, else the
    // param's own ISF default. This is what makes an imported shader's INPUTS
    // controllable (static / baked automation / mod matrix). Each param is N
    // scalar components (genParamComponentCount): 1 = bool/long/float/event/
    // unknown, 2 = point2D (vec2, keys .x/.y), 4 = color (vec4, keys .r/.g/.b/.a);
    // image/audio sampler INPUTS are 0 (skipped — not value params yet).
    for (size_t i = 0; i < params_.size(); ++i)
    {
        const int loc = paramLocs_[i];
        if (loc < 0) continue;
        const GenParam& gp = params_[i];
        const int n = genParamComponentCount (gp.type);
        if (n == 0) continue;   // sampler INPUT — no scalar uniform to set
        float comp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < n; ++k)
        {
            double v = (n == 1) ? gp.defaultV : gp.defaultVec[k];
            if (genValues != nullptr)
            {
                const std::string key = gp.name + genParamComponentSuffix (gp.type, k);
                if (const auto it = genValues->find (key); it != genValues->end())
                    v = it->second;
            }
            comp[k] = (float) v;
        }
        if (n == 1)
        {
            if (genParamIsIntScalar (gp.type)) gl->Uniform1i (loc, (int) std::llround (comp[0]));
            else                               gl->Uniform1f (loc, comp[0]);
        }
        else if (n == 2) gl->Uniform2f (loc, comp[0], comp[1]);
        else if (n == 4) gl->Uniform4f (loc, comp[0], comp[1], comp[2], comp[3]);
    }

    // Bind the contract samplers on distinct units (two samplers of different
    // types must never share a unit). Block B audio bands feed the float band
    // textures when present, else safe black (reads return 0). The same band
    // data fills BOTH uAudioBands (1D, for arbitBand) and uAudioBands2D (nx1 2D,
    // the ISF audioFFT / Shadertoy iChannel0 alias) so no dialect reads black.
    const bool haveBands = audio != nullptr && ! audio->bands.empty();
    if (haveBands)
        ensureBandsTex (gl, (int) audio->bands.size());

    // A shader may also carry NAMED `audio`/`audioFFT` INPUTS (sampler2D), distinct
    // from the contract uAudioBands2D — the spectrum must be uploaded into the nx1
    // 2D band texture if EITHER consumes it (the named INPUTS are bound on TEXTURE4+
    // in the sampler loop below).
    bool haveNamedAudioInput = false;
    for (size_t i = 0; i < params_.size(); ++i)
        if ((params_[i].type == 6 || params_[i].type == 7) && paramLocs_[i] >= 0)
        {
            haveNamedAudioInput = true;
            break;
        }

    if (locs_.uAudioBands >= 0)
    {
        gl->ActiveTexture (GL_TEXTURE0);
        if (haveBands)
        {
            glBindTexture (GL_TEXTURE_1D, bandsTex1D_);
            glTexSubImage1D (GL_TEXTURE_1D, 0, 0, (int) audio->bands.size(),
                             GL_RED, GL_FLOAT, audio->bands.data());
        }
        else
        {
            glBindTexture (GL_TEXTURE_1D, blackTex1D_);
        }
        gl->Uniform1i (locs_.uAudioBands, 0);
    }
    // Upload the spectrum into the nx1 2D band texture ONCE if anything samples it
    // — the contract uAudioBands2D (iChannel0 alias) OR a named audio/audioFFT
    // INPUT — then bind the contract sampler on its fixed unit. (The named INPUTS
    // bind to the same bandsTex2D_ on TEXTURE4+ below.)
    if (haveBands && (locs_.uAudioBands2D >= 0 || haveNamedAudioInput))
    {
        glBindTexture (GL_TEXTURE_2D, bandsTex2D_);
        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, (int) audio->bands.size(), 1,
                         GL_RED, GL_FLOAT, audio->bands.data());
    }
    if (locs_.uAudioBands2D >= 0)
    {
        gl->ActiveTexture (GL_TEXTURE1);
        glBindTexture (GL_TEXTURE_2D, haveBands ? bandsTex2D_ : blackTex2D_);
        gl->Uniform1i (locs_.uAudioBands2D, 1);
    }
    // Block C note/link textures (M5): when `notes` carries a full packed score
    // (sizes match the fixed 4x128 / 256x1 layout) upload it into the float
    // textures; otherwise bind black so reads return 0 (the M3/M4 zero-feed).
    if (haveNotes && (locs_.uNotes >= 0 || locs_.uLinks >= 0))
        ensureNotesTex (gl);
    if (locs_.uNotes >= 0)
    {
        gl->ActiveTexture (GL_TEXTURE2);
        if (haveNotes && notes->notesTex.size() >= (size_t) (kNoteTexW * kNoteTexH * 4))
        {
            glBindTexture (GL_TEXTURE_2D, notesTex2D_);
            glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, kNoteTexW, kNoteTexH,
                             GL_RGBA, GL_FLOAT, notes->notesTex.data());
        }
        else
        {
            glBindTexture (GL_TEXTURE_2D, blackTex2D_);
        }
        gl->Uniform1i (locs_.uNotes, 2);
    }
    if (locs_.uLinks >= 0)
    {
        gl->ActiveTexture (GL_TEXTURE3);
        if (haveNotes && notes->linksTex.size() >= (size_t) (kLinkTexW * 4))
        {
            glBindTexture (GL_TEXTURE_2D, linksTex2D_);
            glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, kLinkTexW, 1,
                             GL_RGBA, GL_FLOAT, notes->linksTex.data());
        }
        else
        {
            glBindTexture (GL_TEXTURE_2D, blackTex2D_);
        }
        gl->Uniform1i (locs_.uLinks, 3);
    }

    // Named sampler2D INPUTS — bind each on its own unit starting at TEXTURE4
    // (units 0–3 are the contract audio(1D/2D)/notes/links samplers, so these
    // never collide with them):
    //   * `image` (type 5, M7 image follow-up): the clip's externally-decoded
    //     texture (uploaded once by the caller, e.g. the exporter's clip-image
    //     cache) keyed by INPUT name; 1x1 black when none supplied.
    //   * `audio`/`audioFFT` (types 6/7): the nx1 spectrum band texture (same
    //     data as the contract uAudioBands2D / iChannel0), so an imported ISF
    //     shader's OWN named audio INPUT reacts to the mix; 1x1 black in silence.
    // A black binding makes `texture(name, …)` return 0 — the zero-feed contract.
    // image textures are owned by the caller; render() only binds, never frees.
    // `nextUnit` continues past the last INPUT sampler so the multipass TARGET
    // samplers (below) get their own units, never colliding with these.
    int nextUnit = 4;
    for (size_t i = 0; i < params_.size(); ++i)
    {
        const int t = params_[i].type;
        const bool isImage = (t == 5);            // InputType::Image
        const bool isAudio = (t == 6 || t == 7);  // InputType::Audio / AudioFFT
        if (! isImage && ! isAudio) continue;
        const int loc = paramLocs_[i];
        if (loc < 0) continue;                    // optimized out by the compiler
        unsigned tex = blackTex2D_;
        if (isImage)
        {
            if (genImages != nullptr)
                if (const auto it = genImages->find (params_[i].name);
                    it != genImages->end() && it->second != 0)
                    tex = it->second;
        }
        else if (haveBands)                       // audio / audioFFT spectrum
            tex = bandsTex2D_;
        gl->ActiveTexture (GL_TEXTURE0 + nextUnit);
        glBindTexture (GL_TEXTURE_2D, tex);
        gl->Uniform1i (loc, nextUnit);
        ++nextUnit;
    }

    // Single implicit output pass (no PASSES / no PERSISTENT): the pre-M7 fast
    // path — one draw into outTex_, already attached + cleared above.
    if (! multipass_)
    {
        gl->BindVertexArray (vao_);
        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
        gl->BindVertexArray (0);
        return outTex_;
    }

    // --- ISF multipass (M7) ---
    // Each PASS renders the body (branching on PASSINDEX) into its TARGET buffer;
    // the output pass renders into outTex_. The contract + INPUT uniforms/samplers
    // set above are program state, identical for every pass; only PASSINDEX, the
    // bound FBO target, and the per-pass TARGET samplers change below.
    ensurePassTargets (gl, outW_, outH_);

    // Output pass = the last pass with an empty TARGET (ISF convention). If a
    // shader names every target (unusual), the final pass is shown instead — its
    // named buffer is bypassed for display, so any cross-frame persistence of THAT
    // pass is skipped (documented v1 limitation; the common feedback pattern uses
    // a separate empty output pass and is unaffected).
    int outputPass = -1;
    for (size_t pi = 0; pi < passes_.size(); ++pi)
        if (passes_[pi].target.empty()) outputPass = (int) pi;
    if (outputPass < 0) outputPass = (int) passes_.size() - 1;

    std::set<std::string> written;   // named targets produced so far THIS frame
    gl->BindVertexArray (vao_);
    for (size_t pi = 0; pi < passes_.size(); ++pi)
    {
        const PassInfo& ps = passes_[pi];
        const bool toOutput = ps.target.empty() || (int) pi == outputPass;

        // Choose the texture this pass writes. A non-output PERSISTENT target
        // writes its BACK buffer (parity^1) so its read sampler (parity) still
        // sees the previous frame in the same pass — no read/write on one texture.
        unsigned writeTex = outTex_;
        if (! toOutput)
        {
            PassTarget& pt = passTargets_[ps.target];
            writeTex = pt.persistent ? pt.tex[(passParity_ + 1) & 1] : pt.tex[0];
            if (writeTex == 0) writeTex = outTex_;   // allocation guard
        }
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, writeTex, 0);
        glViewport (0, 0, outW_, outH_);
        if (passIndexLoc_ >= 0) gl->Uniform1i (passIndexLoc_, (int) pi);

        // Bind every named TARGET as its sampler for THIS pass.
        //   * own target: PERSISTENT → previous frame (feedback); transient
        //     self-read is undefined → black.
        //   * other target, PERSISTENT → this frame's write if an earlier pass
        //     already produced it, else the previous frame.
        //   * other target, transient → its single buffer (this frame's write).
        int unit = nextUnit;
        for (const auto& kv : targetSamplerLocs_)
        {
            const int loc = kv.second;
            if (loc < 0) continue;
            PassTarget& pt = passTargets_[kv.first];
            unsigned rtex = blackTex2D_;
            if (kv.first == ps.target)
            {
                if (ps.persistent && pt.tex[passParity_ & 1] != 0)
                    rtex = pt.tex[passParity_ & 1];
            }
            else if (pt.persistent)
            {
                const int idx = written.count (kv.first) ? ((passParity_ + 1) & 1)
                                                         : (passParity_ & 1);
                if (pt.tex[idx] != 0) rtex = pt.tex[idx];
            }
            else if (pt.tex[0] != 0)
            {
                rtex = pt.tex[0];
            }
            gl->ActiveTexture (GL_TEXTURE0 + unit);
            glBindTexture (GL_TEXTURE_2D, rtex);
            gl->Uniform1i (loc, unit);
            ++unit;
        }

        glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
        if (! ps.target.empty()) written.insert (ps.target);
    }
    gl->BindVertexArray (0);
    passParity_ ^= 1;   // next frame swaps every persistent target's read/write
    return outTex_;
}

void ShaderGenerator::shutdown (const arbitgl::GlFuncs* gl)
{
    if (program_ != 0) { gl->DeleteProgram (program_); program_ = 0; }
    if (vao_ != 0) { gl->DeleteVertexArrays (1, &vao_); vao_ = 0; }
    if (vbo_ != 0) { gl->DeleteBuffers (1, &vbo_); vbo_ = 0; }
    if (fbo_ != 0) { gl->DeleteFramebuffers (1, &fbo_); fbo_ = 0; }
    if (outTex_ != 0) { glDeleteTextures (1, &outTex_); outTex_ = 0; }
    if (blackTex2D_ != 0) { glDeleteTextures (1, &blackTex2D_); blackTex2D_ = 0; }
    if (blackTex1D_ != 0) { glDeleteTextures (1, &blackTex1D_); blackTex1D_ = 0; }
    if (bandsTex1D_ != 0) { glDeleteTextures (1, &bandsTex1D_); bandsTex1D_ = 0; }
    if (bandsTex2D_ != 0) { glDeleteTextures (1, &bandsTex2D_); bandsTex2D_ = 0; }
    bandsTexN_ = 0;
    if (notesTex2D_ != 0) { glDeleteTextures (1, &notesTex2D_); notesTex2D_ = 0; }
    if (linksTex2D_ != 0) { glDeleteTextures (1, &linksTex2D_); linksTex2D_ = 0; }
    notesTexAlloc_ = false;
    releasePassTargets (gl);   // ISF multipass buffers
    passes_.clear();
    targetSamplerLocs_.clear();
    passIndexLoc_ = -1;
    passParity_ = 0;
    multipass_ = false;
    outW_ = outH_ = 0;
    ok_ = false;
}

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
