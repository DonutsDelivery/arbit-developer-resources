#include "gl_loader.h"
#include "particle_engine.h"
#include "shader_generator.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct PixelStats
{
    uint32_t hash = 2166136261u;
    int litPixels = 0;
    uint64_t alphaSum = 0;
};

PixelStats readTexture (arbitgl::GlFuncs& gl, unsigned texture, int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
        gl.DeleteFramebuffers (1, &fbo);
        return {};
    }

    std::vector<uint8_t> pixels (static_cast<size_t> (width) * height * 4);
    glReadPixels (0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);

    PixelStats stats;
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        stats.hash ^= pixels[i];
        stats.hash *= 16777619u;
        if ((i & 3u) == 3u)
        {
            stats.alphaSum += pixels[i];
            if (pixels[i] != 0)
                ++stats.litPixels;
        }
    }
    return stats;
}

videorender::NoteFeatures oneNote()
{
    videorender::NoteFeatures notes;
    notes.notesTex.resize (128u * 4u * 4u, 0.0f);
    notes.notesTex[0] = 60.0f;
    notes.notesTex[1] = 1.0f;
    notes.notesTex[4] = 261.625565f;
    notes.notesTex[6] = 0.0f;
    notes.noteCount = 1;
    return notes;
}

} // namespace

int main()
{
    constexpr int width = 320;
    constexpr int height = 180;

    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "GLFW initialization failed; run this test from macOS Terminal\n";
        return 1;
    }
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow (width, height, "Arbit Metal particle smoke",
                                           nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "OpenGL 4.1 context creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::cerr << "OpenGL loader failed: " << missing << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    videorender::ShaderClock clock;
    clock.frame = 0;
    clock.timeSec = 0.0;
    clock.timeDelta = 1.0 / 30.0;
    clock.playing = true;
    videorender::ParticleParams params;
    params.count = 512;
    params.spawnTrack = 0;
    params.size = 4.0f;
    params.force = 1.0f;
    // Exercise the widened compute/draw uniform ABI, not legacy defaults.
    params.seed = 73;
    params.lifetime = 2.75f;
    params.red = 0.15f;
    params.green = 0.65f;
    params.blue = 0.35f;
    params.alpha = 0.70f;
    const auto notes = oneNote();

    setenv ("ARBIT_PARTICLE_DIAGNOSTICS", "1", 1);
    setenv ("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine fallback;
    unsigned fallbackTexture = 0;
    // Sample while the deterministic upward burst remains inside the viewport.
    // By frame 60 every particle has legitimately travelled above y=1.
    for (int frame = 0; frame <= 15; ++frame)
    {
        clock.frame = frame;
        clock.timeSec = frame * clock.timeDelta;
        fallbackTexture = fallback.render (&gl, clock, width, height, params, &notes);
    }
    PixelStats fallbackStats;
    if (fallbackTexture != 0)
        fallbackStats = readTexture (gl, fallbackTexture, width, height);
    const auto fallbackDiagnostics = fallback.diagnostics();
    fallback.shutdown (&gl);

    setenv ("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine metal;
    unsigned metalTexture = 0;
    for (int frame = 0; frame <= 15; ++frame)
    {
        clock.frame = frame;
        clock.timeSec = frame * clock.timeDelta;
        metalTexture = metal.render (&gl, clock, width, height, params, &notes);
    }
    const std::string metalLog = metal.log();
    PixelStats metalStats;
    if (metalTexture != 0)
        metalStats = readTexture (gl, metalTexture, width, height);
    const auto metalDiagnostics = metal.diagnostics();
    metal.shutdown (&gl);

    // A fresh engine rendered directly at frame 15 must replay frames 0..15 and
    // land on the same deterministic image as sequential preview stepping.
    videorender::ParticleEngine metalReplay;
    const unsigned replayTexture = metalReplay.render (&gl, clock, width, height, params, &notes);
    const PixelStats replayStats = replayTexture != 0
        ? readTexture (gl, replayTexture, width, height) : PixelStats {};
    const auto replayDiagnostics = metalReplay.diagnostics();
    metalReplay.shutdown (&gl);

    glfwDestroyWindow (window);
    glfwTerminate();

    std::cout << "GL fallback: hash=" << fallbackStats.hash
              << " litPixels=" << fallbackStats.litPixels
              << " alphaSum=" << fallbackStats.alphaSum << '\n';
    std::cout << "GL stages: notes=" << fallbackDiagnostics.noteRows
              << " live=" << fallbackDiagnostics.liveParticles
              << " visible=" << fallbackDiagnostics.visibleCandidates
              << " drawAlpha=" << fallbackDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << fallbackDiagnostics.readbackAlphaPixels << '\n';
    std::cout << "Metal bridge: hash=" << metalStats.hash
              << " litPixels=" << metalStats.litPixels
              << " alphaSum=" << metalStats.alphaSum
              << " backend=" << metalLog << '\n';
    std::cout << "Metal stages: notes=" << metalDiagnostics.noteRows
              << " live=" << metalDiagnostics.liveParticles
              << " visible=" << metalDiagnostics.visibleCandidates
              << " drawAlpha=" << metalDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << metalDiagnostics.readbackAlphaPixels << '\n';
    std::cout << "Metal direct replay: hash=" << replayStats.hash
              << " live=" << replayDiagnostics.liveParticles
              << " drawAlpha=" << replayDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << replayDiagnostics.readbackAlphaPixels << '\n';

    if (fallbackTexture == 0 || fallbackStats.litPixels == 0
        || fallbackDiagnostics.noteRows != 1 || fallbackDiagnostics.liveParticles <= 0
        || fallbackDiagnostics.drawAlphaPixels <= 0)
    {
        std::cerr << "OpenGL reference path produced no particles\n";
        return 1;
    }
    if (metalTexture == 0 || metalLog != "metal" || metalStats.litPixels == 0
        || metalDiagnostics.noteRows != 1 || metalDiagnostics.liveParticles <= 0
        || metalDiagnostics.drawAlphaPixels <= 0
        || metalDiagnostics.readbackAlphaPixels <= 0
        || replayTexture == 0 || replayStats.hash != metalStats.hash
        || replayDiagnostics.liveParticles != metalDiagnostics.liveParticles
        || replayDiagnostics.drawAlphaPixels <= 0
        || replayDiagnostics.readbackAlphaPixels <= 0)
    {
        std::cerr << "Metal particle/IOSurface bridge failed: " << metalLog << '\n';
        return 1;
    }

    const double litRatio = static_cast<double> (metalStats.litPixels)
                          / static_cast<double> (fallbackStats.litPixels);
    if (litRatio < 0.5 || litRatio > 2.0)
    {
        std::cerr << "Metal/GL particle coverage differs unexpectedly (ratio "
                  << litRatio << ")\n";
        return 1;
    }
    std::cout << "Metal particle bridge smoke passed; coverage ratio="
              << litRatio << '\n';
    return 0;
}
