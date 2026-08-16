#include "gl_loader.h"
#include "particle_engine.h"
#include "shader_generator.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
uint32_t hashTexture(arbitgl::GlFuncs& gl, unsigned texture, int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers(1, &fbo);
    gl.BindFramebuffer(GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    std::vector<uint8_t> pixels((size_t) width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.DeleteFramebuffers(1, &fbo);
    uint32_t hash = 2166136261u;
    for (const auto value : pixels) { hash ^= value; hash *= 16777619u; }
    return hash;
}

videorender::NoteFeatures builtInNote()
{
    videorender::NoteFeatures notes;
    notes.notesTex.resize(128u * 4u * 4u, 0.0f);
    notes.notesTex[0] = 60.0f; notes.notesTex[1] = 1.0f;
    notes.notesTex[4] = 261.625565f; notes.notesTex[6] = 0.0f;
    notes.noteCount = 1;
    return notes;
}

struct GlState
{
    GLint program {}, activeTexture {}, readFbo {}, drawFbo {}, drawBuffer {}, vao {}, ssbo {};
    GLint textures[3] {}, blendSrcRgb {}, blendDstRgb {}, blendSrcAlpha {}, blendDstAlpha {};
    GLint viewport[4] {};
    GLfloat clear[4] {};
    GLboolean blend {}, pointSize {};

    static GlState capture(arbitgl::GlFuncs& gl)
    {
        GlState s;
        glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &s.activeTexture);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFbo);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFbo);
        glGetIntegerv(GL_DRAW_BUFFER0, &s.drawBuffer);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0, &s.ssbo);
        glGetIntegerv(GL_BLEND_SRC_RGB, &s.blendSrcRgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &s.blendDstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &s.blendSrcAlpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &s.blendDstAlpha);
        glGetIntegerv(GL_VIEWPORT, s.viewport);
        glGetFloatv(GL_COLOR_CLEAR_VALUE, s.clear);
        s.blend = glIsEnabled(GL_BLEND);
        s.pointSize = glIsEnabled(GL_PROGRAM_POINT_SIZE);
        for (int unit = 0; unit < 3; ++unit)
        {
            gl.ActiveTexture(GL_TEXTURE0 + unit);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.textures[unit]);
        }
        gl.ActiveTexture((GLenum) s.activeTexture);
        return s;
    }

    bool operator==(const GlState& other) const
    {
        return program == other.program && activeTexture == other.activeTexture
            && readFbo == other.readFbo && drawFbo == other.drawFbo
            && drawBuffer == other.drawBuffer && vao == other.vao && ssbo == other.ssbo
            && std::equal(std::begin(textures), std::end(textures), std::begin(other.textures))
            && blendSrcRgb == other.blendSrcRgb && blendDstRgb == other.blendDstRgb
            && blendSrcAlpha == other.blendSrcAlpha && blendDstAlpha == other.blendDstAlpha
            && std::equal(std::begin(viewport), std::end(viewport), std::begin(other.viewport))
            && std::equal(std::begin(clear), std::end(clear), std::begin(other.clear))
            && blend == other.blend && pointSize == other.pointSize;
    }
};
}

int main()
{
    constexpr int width = 160, height = 90;
    if (! glfwInit()) return 77;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    auto* window = glfwCreateWindow(width, height, "visual.particles GL", nullptr, nullptr);
    if (window == nullptr) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions(gl, missing)) return 1;
    if (! arbitgl::loadGl43Functions(gl)) return 1;

    videorender::ShaderClock clock;
    clock.frame = 60; clock.timeSec = 1.0; clock.timeDelta = 1.0 / 60.0; clock.playing = true;
    videorender::ParticleParams params;
    params.count = 256; params.size = 5.0f; params.force = 1.25f;
    params.seed = 77; params.lifetime = 2.5f;
    params.red = 0.1f; params.green = 0.8f; params.blue = 0.4f; params.alpha = 0.75f;
    const auto notes = builtInNote();

    videorender::ParticleEngine first, second;
    GLuint sentinelTextures[3] {}, sentinelFbos[2] {}, sentinelVao {}, sentinelSsbo {};
    glGenTextures(3, sentinelTextures);
    gl.GenFramebuffers(2, sentinelFbos);
    gl.GenVertexArrays(1, &sentinelVao);
    gl.GenBuffers(1, &sentinelSsbo);
    for (int unit = 0; unit < 3; ++unit)
    {
        gl.ActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, sentinelTextures[unit]);
    }
    gl.ActiveTexture(GL_TEXTURE2);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, sentinelFbos[0]);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, sentinelFbos[1]);
    glDrawBuffer(GL_NONE);
    gl.BindVertexArray(sentinelVao);
    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sentinelSsbo);
    glViewport(7, 9, 101, 47);
    glClearColor(0.125f, 0.25f, 0.5f, 0.75f);
    glEnable(GL_BLEND);
    glEnable(GL_PROGRAM_POINT_SIZE);
    gl.BlendFuncSeparate(GL_ONE, GL_ZERO, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    const auto stateBefore = GlState::capture(gl);
    const auto a = first.render(&gl, clock, width, height, params, &notes);
    const bool successStateRestored = GlState::capture(gl) == stateBefore;

    auto failingGl = gl;
    failingGl.CreateShader = [](GLenum) -> GLuint { return 0; };
    videorender::ParticleEngine failing;
    const auto failedTexture = failing.render(&failingGl, clock, width, height, params, &notes);
    const bool failureStateRestored = GlState::capture(gl) == stateBefore;

    const auto b = second.render(&gl, clock, width, height, params, &notes);
    const auto hashA = a != 0 ? hashTexture(gl, a, width, height) : 0;
    const auto hashB = b != 0 ? hashTexture(gl, b, width, height) : 0;

    videorender::ParticleParams legacy;
    legacy.count = videorender::ParticleEngine::kMaxParticles;
    videorender::ParticleEngine legacyA, legacyB, empty;
    const auto legacyTexA = legacyA.render(&gl, clock, width, height, legacy, &notes);
    const auto legacyTexB = legacyB.render(&gl, clock, width, height, legacy, &notes);
    const auto emptyTex = empty.render(&gl, clock, width, height, legacy, nullptr);
    const auto legacyHashA = legacyTexA ? hashTexture(gl, legacyTexA, width, height) : 0;
    const auto legacyHashB = legacyTexB ? hashTexture(gl, legacyTexB, width, height) : 0;
    const auto emptyHash = emptyTex ? hashTexture(gl, emptyTex, width, height) : 0;
    legacyA.shutdown(&gl); legacyB.shutdown(&gl); empty.shutdown(&gl);
    const auto firstLog = first.log();
    const auto secondLog = second.log();
    first.shutdown(&gl); second.shutdown(&gl); failing.shutdown(&gl);
    gl.DeleteBuffers(1, &sentinelSsbo);
    gl.DeleteVertexArrays(1, &sentinelVao);
    gl.DeleteFramebuffers(2, sentinelFbos);
    glDeleteTextures(3, sentinelTextures);
    glfwDestroyWindow(window); glfwTerminate();
    if (! successStateRestored || ! failureStateRestored || failedTexture != 0
        || a == 0 || b == 0 || hashA == 0 || hashA != hashB
        || legacyTexA == 0 || legacyHashA == 0 || legacyHashA != legacyHashB
        || emptyTex == 0 || emptyHash != 3996078533u)
    {
        std::cerr << "visual.particles OpenGL determinism/state policy failed: "
                  << hashA << " / " << hashB << " legacy=" << legacyHashA << " / "
                  << legacyHashB << " empty=" << emptyHash
                  << " successState=" << successStateRestored
                  << " failureState=" << failureStateRestored
                  << " first=" << firstLog << " second=" << secondLog << '\n';
        return 1;
    }
    std::cout << "visual.particles OpenGL PASS hash=" << hashA << '\n';
    return 0;
}
