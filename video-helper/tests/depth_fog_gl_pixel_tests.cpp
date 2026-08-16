#include "gl_loader.h"
#include "renderer.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = 3;
constexpr int kHeight = 1;
constexpr int kTolerance = 3;
constexpr std::array<uint8_t, 4> kSource { 32, 96, 192, 255 };
constexpr std::array<uint8_t, 4> kFog { 224, 48, 16, 128 };
constexpr std::array<uint16_t, 3> kDepth { 0, 32768, 65535 };
constexpr double kNear = 0.0;
constexpr double kFar = 1.0;
constexpr double kDensity = 2.0;

std::array<int, 4> oracle(uint16_t rawDepth)
{
    // Independent specification oracle: normalized UNORM16 depth, Beer-Lambert
    // transmittance, then straight RGB interpolation. This intentionally does
    // not call any renderer/fog helper or reproduce shader source text.
    const double normalizedDepth = static_cast<double>(rawDepth) / 65535.0;
    const double progress = std::max(0.0, std::min(1.0,
        (normalizedDepth - kNear) / (kFar - kNear)));
    const double fogOpacity = (static_cast<double>(kFog[3]) / 255.0)
        * (1.0 - std::exp(-kDensity * progress));
    std::array<int, 4> expected {};
    for (int channel = 0; channel < 3; ++channel)
        expected[channel] = static_cast<int>(std::lround(
            kSource[channel] * (1.0 - fogOpacity) + kFog[channel] * fogOpacity));
    expected[3] = kSource[3];
    return expected;
}

bool closeEnough(const uint8_t* actual, const std::array<int, 4>& expected)
{
    for (int channel = 0; channel < 4; ++channel)
        if (std::abs(static_cast<int>(actual[channel]) - expected[channel]) > kTolerance)
            return false;
    return true;
}

void printPixel(const char* name, const uint8_t* actual, const std::array<int, 4>& expected)
{
    std::cout << name << " actual=(" << static_cast<int>(actual[0]) << ','
              << static_cast<int>(actual[1]) << ',' << static_cast<int>(actual[2]) << ','
              << static_cast<int>(actual[3]) << ") oracle=(" << expected[0] << ','
              << expected[1] << ',' << expected[2] << ',' << expected[3] << ")\n";
}
}

int main()
{
#if !defined(__linux__)
    std::cerr << "This acceptance target requires native Linux OpenGL\n";
    return 77;
#else
    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": " << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "BLOCKED: glfwInit failed; no native EGL/GLX display/context available\n";
        return 77;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(16, 16, "Depth Fog pixel acceptance", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "BLOCKED: GLFW could not create a native Linux GL 4.3 offscreen context\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (!arbitgl::loadGlFunctions(gl, error))
    {
        std::cerr << "BLOCKED: production GL loader missing: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 77;
    }

    // Force production FrameRenderer shader admission to fail, and prove its
    // owner unwinds to a non-ready state without leaving a GL error behind.
    auto rejectedGl = gl;
    rejectedGl.CreateShader = [](GLenum) -> GLuint { return 0; };
    videorender::FrameRenderer rejected;
    std::string rejection;
    const bool rejectedAdmission = !rejected.initialize(&rejectedGl, kWidth, kHeight, rejection)
        && !rejected.ready() && !rejection.empty();
    rejected.shutdown();
    // The deliberately invalid CreateShader handle can set GL_INVALID_VALUE;
    // consume that injected diagnostic before proving the same context remains
    // usable by a fresh production owner.
    while (glGetError() != GL_NO_ERROR) {}

    videorender::FrameRenderer renderer;
    if (!renderer.initialize(&gl, kWidth, kHeight, error))
    {
        std::cerr << "FrameRenderer initialization failed: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::array<uint8_t, kWidth * kHeight * 4> source {};
    for (int x = 0; x < kWidth; ++x)
        std::copy(kSource.begin(), kSource.end(), source.begin() + x * 4);
    const unsigned sourceTexture = renderer.uploadRgba(source.data(), kWidth, kHeight,
                                                        kWidth * 4, 0);
    const unsigned depthTexture = renderer.uploadR16(kDepth.data(), kWidth, kHeight, 0);

    videorender::LayerDesc layer;
    layer.texture = sourceTexture;
    layer.texWidth = kWidth;
    layer.texHeight = kHeight;
    layer.depthTexture = depthTexture;
    layer.depthWidth = kWidth;
    layer.depthHeight = kHeight;
    layer.depthFog = true;
    layer.fogNear = static_cast<float>(kNear);
    layer.fogFar = static_cast<float>(kFar);
    layer.fogDensity = static_cast<float>(kDensity);
    layer.fogRed = kFog[0] / 255.0f;
    layer.fogGreen = kFog[1] / 255.0f;
    layer.fogBlue = kFog[2] / 255.0f;
    layer.fogAlpha = kFog[3] / 255.0f;

    std::vector<uint8_t> pixels;
    const bool rendered = sourceTexture != 0 && depthTexture != 0
        && renderer.renderToPixels(&layer, 1, pixels, error)
        && pixels.size() == static_cast<size_t>(kWidth * kHeight * 4);
    bool pixelsPass = rendered;
    for (int x = 0; x < kWidth && rendered; ++x)
    {
        const auto expected = oracle(kDepth[x]);
        printPixel(x == 0 ? "near" : (x == 1 ? "mid" : "far"), pixels.data() + x * 4, expected);
        pixelsPass = pixelsPass && closeEnough(pixels.data() + x * 4, expected);
    }
    const bool depthChangesOutput = rendered
        && !std::equal(pixels.begin(), pixels.begin() + 4, pixels.begin() + 4)
        && !std::equal(pixels.begin() + 4, pixels.begin() + 8, pixels.begin() + 8);

    renderer.deleteTexture(sourceTexture);
    renderer.deleteTexture(depthTexture);
    const bool externalTexturesDeleted = glIsTexture(sourceTexture) == GL_FALSE
        && glIsTexture(depthTexture) == GL_FALSE;
    renderer.shutdown();
    const bool ownerClean = !renderer.ready() && glGetError() == GL_NO_ERROR;

    glfwDestroyWindow(window);
    glfwTerminate();
    if (!rejectedAdmission || !pixelsPass || !depthChangesOutput
        || !externalTexturesDeleted || !ownerClean)
    {
        std::cerr << "Depth Fog native pixel acceptance FAIL: rendered=" << rendered
                  << " rejectedAdmission=" << rejectedAdmission
                  << " depthChangesOutput=" << depthChangesOutput
                  << " texturesDeleted=" << externalTexturesDeleted
                  << " ownerClean=" << ownerClean << " error=" << error
                  << " rejection=" << rejection << '\n';
        return 1;
    }
    std::cout << "Depth Fog native OpenGL PASS tolerance=+/-" << kTolerance
              << " channels; production FrameRenderer readback, depth-dependent pixels, "
                 "shader-failure unwind, and texture cleanup verified\n";
    return 0;
#endif
}
