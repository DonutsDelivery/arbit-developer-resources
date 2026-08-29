#include "gl_loader.h"
#include "gpu_backend/frame_renderer_metal.h"
#include "renderer.h"
#include "visual_plan_executor.h"

#include <GLFW/glfw3.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace
{

std::array<uint8_t, 4> readCenter (arbitgl::GlFuncs& gl, unsigned texture,
                                   int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    std::array<uint8_t, 4> pixel = {};
    glReadPixels (width / 2, height / 2, 1, 1,
                  GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);
    return pixel;
}

std::array<uint8_t, 4> readPixel (arbitgl::GlFuncs& gl, unsigned texture,
                                  int x, int y)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    std::array<uint8_t, 4> pixel = {};
    glReadPixels (x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);
    return pixel;
}

bool isRed (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] < 20 && pixel[2] < 20 && pixel[3] > 220;
}

bool isBlue (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] < 20 && pixel[2] > 220 && pixel[3] > 220;
}

bool isCyan (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] > 220 && pixel[2] > 220 && pixel[3] > 220;
}

bool isGreen (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] < 20 && pixel[1] > 220 && pixel[2] < 20 && pixel[3] > 220;
}

bool isMagenta (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] < 20 && pixel[2] > 220 && pixel[3] > 220;
}

bool isYellow (const std::array<uint8_t, 4>& pixel)
{
    return pixel[0] > 220 && pixel[1] > 220 && pixel[2] < 20 && pixel[3] > 220;
}

void setSurfaceInt (CFMutableDictionaryRef properties, CFStringRef key, int value)
{
    CFNumberRef number = CFNumberCreate (
        kCFAllocatorDefault, kCFNumberIntType, &value);
    CFDictionarySetValue (properties, key, number);
    CFRelease (number);
}

videowire::CompiledVisualLayerPlan typedMattePlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 501;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.refine",
                       "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 42, 43, 12 };
    plan.edges = { { 11, 0, 43, 0 }, { 41, 0, 42, 0 },
                   { 42, 1, 43, 1 }, { 43, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"metal-smoke-matte\" state=\"available\" "
          "cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" "
          "framePrefix=\"subject-\" frameExtension=\".rgba\" firstFrame=\"42\" "
          "frameDigits=\"3\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 42, "visual.matte.refine", "native-gpu",
          "<NodeParams invert=\"1\" black=\"0.2\" white=\"0.8\" "
          "erodeDilate=\"1\" feather=\"1\" choke=\"0.1\"/>" },
        { 43, "visual.matte.apply", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}


} // namespace

int main (int argc, char** argv)
{
    constexpr int width = 64;
    constexpr int height = 32;
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "GLFW initialization failed\n";
        return 1;
    }
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow (width, height,
                                           "Arbit Metal frame compositor smoke",
                                           nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "OpenGL 4.1 context creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (! arbitgl::loadGlFunctions (gl, error))
    {
        std::cerr << "OpenGL loader failed: " << error << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    videorender::MetalFrameRenderer renderer;
    if (! renderer.initialize (&gl, width, height, error))
    {
        std::cerr << "Metal compositor initialization failed: " << error << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    renderer.setCanvas (width, height);
    renderer.setPresentSize (width, height);
    renderer.setBackgroundColor (0, 0, 0, 1);

    std::vector<uint8_t> red (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < red.size(); i += 4)
    {
        red[i + 1] = 0;
        red[i + 2] = 0;
    }
    constexpr unsigned sourceHandle = 42;
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    videorender::LayerDesc layer;
    layer.texture = sourceHandle;
    layer.texWidth = width;
    layer.texHeight = height;

    std::array<uint8_t, 4> pixel = {};
    for (int frame = 0; frame < 12; ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        if (output == 0)
        {
            std::cerr << "Metal compositor rejected a supported frame: "
                      << renderer.lastError() << '\n';
            renderer.shutdown (&gl);
            glfwDestroyWindow (window);
            glfwTerminate();
            return 1;
        }
        pixel = readCenter (gl, output, width, height);
    }

    if (! isRed (pixel))
    {
        std::cerr << "Metal compositor did not preserve the uploaded red frame\n";
        renderer.deleteTexture (sourceHandle);
        renderer.shutdown (&gl);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    // Physical Apple Silicon oracle: this uses the production R16 owner and
    // typed layer boundary, then exercises deletion/stale-handle rejection.
    constexpr unsigned depthHandle = 43;
    std::vector<uint16_t> depth (static_cast<size_t> (width) * height, 65535);
    if (! renderer.uploadR16 (depthHandle, depth.data(), width, height))
    {
        std::cerr << "Metal R16 depth upload failed: " << renderer.lastError() << '\n';
        return 1;
    }
    layer.depthFog = true;
    layer.depthTexture = depthHandle;
    layer.depthWidth = width;
    layer.depthHeight = height;
    layer.fogNear = 0.0f;
    layer.fogFar = 1.0f;
    layer.fogDensity = 1.0f;
    layer.fogRed = 0.0f;
    layer.fogGreen = 0.0f;
    layer.fogBlue = 1.0f;
    layer.fogAlpha = 1.0f;
    unsigned fogOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, fogOutput, width, height);
    const float fogAmount = 1.0f - std::exp (-1.0f);
    const int expectedRed = static_cast<int> (std::lround (255.0f * (1.0f - fogAmount)));
    const int expectedBlue = static_cast<int> (std::lround (255.0f * fogAmount));
    if (fogOutput == 0 || std::abs (static_cast<int> (pixel[0]) - expectedRed) > 3
        || pixel[1] > 3 || std::abs (static_cast<int> (pixel[2]) - expectedBlue) > 3)
    {
        std::cerr << "Metal Depth Fog production pixel oracle failed\n";
        return 1;
    }
    for (int mode = 2; mode <= 4; ++mode)
    {
        layer.depthFog = false;
        layer.depthEffect = mode;
        layer.depthParam0 = mode == 2 ? 4.0f : (mode == 3 ? 0.1f : 1.0f);
        layer.depthParam1 = mode == 2 ? 0.0f : (mode == 3 ? 0.0f : 0.5f);
        layer.depthParam2 = mode == 2 ? 1.0f : 0.5f;
        layer.depthColorRed = layer.depthColorGreen = layer.depthColorBlue = 1.0f;
        if (renderer.renderComposite (&gl, &layer, 1, nullptr, 0) == 0)
        {
            std::cerr << "Metal depth primitive mode " << mode << " was rejected: "
                      << renderer.lastError() << '\n';
            return 1;
        }
    }
    renderer.deleteTexture (depthHandle);
    if (renderer.renderComposite (&gl, &layer, 1, nullptr, 0) != 0)
    {
        std::cerr << "Metal Depth Fog accepted a deleted/stale depth handle\n";
        return 1;
    }
    layer.depthFog = false;
    layer.depthEffect = 0;
    layer.depthTexture = 0;
    layer.depthWidth = layer.depthHeight = 0;

    std::vector<uint8_t> dimGray (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < dimGray.size(); i += 4)
        dimGray[i] = dimGray[i + 1] = dimGray[i + 2] = 64;
    renderer.uploadRgba (sourceHandle, dimGray.data(), width, height, width * 4);
    renderer.setPostFx (0.0f, 1.0f, 0.0f, 0, 2.0f);
    unsigned postOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, postOutput, width, height);
    if (pixel[0] < 120 || pixel[0] > 136
        || pixel[1] < 120 || pixel[1] > 136 || pixel[2] < 120 || pixel[2] > 136)
    {
        std::cerr << "Metal HDR exposure did not double dim gray\n";
        return 1;
    }

    std::vector<uint8_t> bloomSource (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < bloomSource.size(); i += 4)
        bloomSource[i] = bloomSource[i + 1] = bloomSource[i + 2] = 0;
    for (int y = 0; y < height; ++y)
        bloomSource[(static_cast<size_t> (y) * width + width / 2) * 4] = 255;
    renderer.uploadRgba (sourceHandle, bloomSource.data(), width, height, width * 4);
    renderer.setPostFx (1.0f, 0.1f, 5.0f, 0, 1.0f);
    postOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, postOutput, width / 2 + 3, height / 2);
    if (pixel[0] <= 5)
    {
        std::cerr << "Metal HDR bloom did not spread the bright line\n";
        return 1;
    }
    renderer.setPostFx (0.0f, 1.0f, 0.0f, 0, 1.0f);
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    unsigned resetOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, resetOutput, width, height);
    if (! isRed (pixel))
    {
        std::cerr << "Metal source/post reset did not restore red; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }

    videorender::EffectSlotState invert;
    invert.type = static_cast<int> (videofx::EffectType::Invert);
    invert.enabled = true;
    invert.params[0] = 1.0f;
    layer.effects = &invert;
    layer.effectCount = 1;
    for (int frame = 0; frame < 12 && ! isCyan (pixel); ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        pixel = readCenter (gl, output, width, height);
    }
    if (! isCyan (pixel))
    {
        std::cerr << "Metal color effect suite did not invert red to cyan; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        renderer.deleteTexture (sourceHandle);
        renderer.shutdown (&gl);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    layer.effects = nullptr;
    layer.effectCount = 0;

    videorender::LayerDesc adjustment;
    adjustment.isAdjustment = true;
    adjustment.clipId = 99;
    adjustment.opacity = 1.0f;
    adjustment.effects = &invert;
    adjustment.effectCount = 1;
    const std::array<videorender::LayerDesc, 2> adjustedLayers = {
        layer, adjustment };
    unsigned adjustedOutput = renderer.renderComposite (
        &gl, adjustedLayers.data(), static_cast<int> (adjustedLayers.size()), nullptr, 0);
    pixel = readCenter (gl, adjustedOutput, width, height);
    if (! isCyan (pixel))
    {
        std::cerr << "Metal adjustment layer did not invert the composite beneath it\n";
        return 1;
    }

    constexpr unsigned ownedOverlayHandle = 46;
    std::vector<uint8_t> ownedBlue (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < ownedBlue.size(); i += 4)
        ownedBlue[i] = ownedBlue[i + 1] = 0;
    renderer.uploadRgba (ownedOverlayHandle, ownedBlue.data(), width, height, width * 4);
    videorender::ImageLayerDesc ownedOverlay;
    ownedOverlay.texture = ownedOverlayHandle;
    ownedOverlay.width = width;
    ownedOverlay.height = height;
    ownedOverlay.ownerClipId = adjustment.clipId;
    unsigned overlayOnlyOutput = renderer.renderComposite (
        &gl, &layer, 1, &ownedOverlay, 1);
    pixel = readCenter (gl, overlayOnlyOutput, width, height);
    if (! isBlue (pixel))
    {
        std::cerr << "Metal owned overlay did not composite before adjustment; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    for (int frame = 0; frame < 4 && ! isYellow (pixel); ++frame)
    {
        adjustedOutput = renderer.renderComposite (
            &gl, adjustedLayers.data(), static_cast<int> (adjustedLayers.size()),
            &ownedOverlay, 1);
        pixel = readCenter (gl, adjustedOutput, width, height);
    }
    if (! isYellow (pixel))
    {
        std::cerr << "Metal adjustment did not process its owned blue overlay; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    renderer.deleteTexture (ownedOverlayHandle);

    constexpr unsigned lutHandle = 45;
    std::array<float, 24> magentaLut = {};
    for (size_t i = 0; i < magentaLut.size(); i += 3)
    {
        magentaLut[i] = 1.0f;
        magentaLut[i + 2] = 1.0f;
    }
    renderer.uploadLut3D (lutHandle, magentaLut.data(), 2);
    layer.lutTexture = lutHandle;
    layer.lutSize = 2;
    const unsigned lutOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, lutOutput, width, height);
    if (! isMagenta (pixel))
    {
        std::cerr << "Metal 3D LUT did not map the source to magenta; got "
                  << static_cast<int> (pixel[0]) << ','
                  << static_cast<int> (pixel[1]) << ','
                  << static_cast<int> (pixel[2]) << ','
                  << static_cast<int> (pixel[3]) << '\n';
        return 1;
    }
    layer.lutTexture = 0;
    layer.lutSize = 0;
    renderer.deleteTexture (lutHandle);

    std::vector<uint8_t> edge (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            const uint8_t value = x < width / 2 ? 255 : 0;
            edge[offset] = value;
            edge[offset + 1] = 0;
            edge[offset + 2] = 0;
        }
    renderer.uploadRgba (sourceHandle, edge.data(), width, height, width * 4);
    videorender::EffectSlotState blur;
    blur.type = static_cast<int> (videofx::EffectType::Blur);
    blur.enabled = true;
    blur.params[0] = 4.0f;
    layer.effects = &blur;
    layer.effectCount = 1;
    unsigned filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width / 2, height / 2);
    if (pixel[0] <= 20 || pixel[0] >= 220)
    {
        std::cerr << "Metal Gaussian blur did not soften a hard edge; red="
                  << static_cast<int> (pixel[0]) << '\n';
        return 1;
    }

    std::vector<uint8_t> ridge (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            const uint8_t value = x == width / 2 ? 192 : 128;
            ridge[offset] = ridge[offset + 1] = ridge[offset + 2] = value;
        }
    renderer.uploadRgba (sourceHandle, ridge.data(), width, height, width * 4);
    videorender::EffectSlotState sharpen;
    sharpen.type = static_cast<int> (videofx::EffectType::Sharpen);
    sharpen.enabled = true;
    sharpen.params[0] = 1.0f;
    layer.effects = &sharpen;
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width / 2, height / 2);
    if (pixel[0] <= 200)
    {
        std::cerr << "Metal unsharp mask did not increase ridge contrast; red="
                  << static_cast<int> (pixel[0]) << '\n';
        return 1;
    }

    std::vector<uint8_t> asymmetric (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            asymmetric[offset] = x < width / 2 ? 255 : 0;
            asymmetric[offset + 1] = x < width / 2 ? 0 : 255;
            asymmetric[offset + 2] = 0;
        }
    renderer.uploadRgba (sourceHandle, asymmetric.data(), width, height, width * 4);
    videorender::EffectSlotState geometryEffect;
    geometryEffect.type = static_cast<int> (videofx::EffectType::Mirror);
    geometryEffect.enabled = true;
    geometryEffect.params[0] = 0.0f;
    layer.effects = &geometryEffect;
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readPixel (gl, filteredOutput, width * 3 / 4, height / 2);
    if (! isRed (pixel))
    {
        std::cerr << "Metal mirror effect did not reflect the left half\n";
        return 1;
    }

    const std::array<videofx::EffectType, 8> statelessGeometry = {
        videofx::EffectType::Kaleidoscope, videofx::EffectType::Mirror,
        videofx::EffectType::Tile, videofx::EffectType::Warp,
        videofx::EffectType::Displace, videofx::EffectType::PolarSwirl,
        videofx::EffectType::DisplaceRgb, videofx::EffectType::Pixelate };
    for (const auto type : statelessGeometry)
    {
        geometryEffect = {};
        geometryEffect.type = static_cast<int> (type);
        geometryEffect.enabled = true;
        const auto* definition = videofx::effectDefFor (geometryEffect.type);
        for (int param = 0; definition != nullptr && param < definition->paramCount; ++param)
            geometryEffect.params[param] = definition->params[param].defaultValue;
        filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
        if (filteredOutput == 0)
        {
            std::cerr << "Metal compositor rejected stateless geometry effect "
                      << geometryEffect.type << '\n';
            return 1;
        }
    }

    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    videorender::EffectSlotState feedback;
    feedback.type = static_cast<int> (videofx::EffectType::FeedbackTrail);
    feedback.enabled = true;
    feedback.params[0] = 0.9f;
    feedback.params[1] = 1.0f;
    feedback.params[2] = 0.0f;
    layer.clipId = 7;
    layer.effects = &feedback;
    layer.effectCount = 1;
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    std::vector<uint8_t> black (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < black.size(); i += 4)
        black[i] = black[i + 1] = black[i + 2] = 0;
    renderer.uploadRgba (sourceHandle, black.data(), width, height, width * 4);
    filteredOutput = renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    pixel = readCenter (gl, filteredOutput, width, height);
    if (pixel[0] < 200 || pixel[1] > 20 || pixel[2] > 20)
    {
        std::cerr << "Metal feedback trail did not retain the previous red frame\n";
        return 1;
    }
    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    layer.clipId = 0;
    layer.effects = nullptr;
    layer.effectCount = 0;
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);

    constexpr unsigned incomingHandle = 44;
    std::vector<uint8_t> green (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < green.size(); i += 4)
    {
        green[i] = 0;
        green[i + 2] = 0;
    }
    renderer.uploadRgba (incomingHandle, green.data(), width, height, width * 4);
    const videorender::LayerDesc outgoing = layer;
    videorender::LayerDesc incoming = layer;
    incoming.texture = incomingHandle;
    incoming.transitionType = 2;
    incoming.fromLayer = &outgoing;
    incoming.transitionProgress = 0.0f;
    unsigned transitionOutput = renderer.renderComposite (&gl, &incoming, 1, nullptr, 0);
    pixel = readCenter (gl, transitionOutput, width, height);
    if (! isRed (pixel))
    {
        std::cerr << "Metal dissolve did not begin on the outgoing frame\n";
        return 1;
    }
    incoming.transitionProgress = 1.0f;
    transitionOutput = renderer.renderComposite (&gl, &incoming, 1, nullptr, 0);
    pixel = readCenter (gl, transitionOutput, width, height);
    if (! isGreen (pixel))
    {
        std::cerr << "Metal dissolve did not end on the incoming frame\n";
        return 1;
    }
    renderer.deleteTexture (incomingHandle);

    constexpr int overlayWidth = 16;
    constexpr int overlayHeight = 16;
    constexpr unsigned overlayHandle = 43;
    std::vector<uint8_t> blue (
        static_cast<size_t> (overlayWidth) * overlayHeight * 4, 255);
    for (size_t i = 0; i < blue.size(); i += 4)
    {
        blue[i] = 0;
        blue[i + 1] = 0;
    }
    renderer.uploadRgba (overlayHandle, blue.data(), overlayWidth, overlayHeight,
                         overlayWidth * 4);
    videorender::ImageLayerDesc overlay;
    overlay.texture = overlayHandle;
    overlay.width = overlayWidth;
    overlay.height = overlayHeight;
    for (int frame = 0; frame < 8; ++frame)
    {
        const unsigned output = renderer.renderComposite (&gl, &layer, 1, &overlay, 1);
        if (output == 0)
        {
            std::cerr << "Metal compositor rejected a supported overlay\n";
            renderer.deleteTexture (overlayHandle);
            renderer.deleteTexture (sourceHandle);
            renderer.shutdown (&gl);
            glfwDestroyWindow (window);
            glfwTerminate();
            return 1;
        }
        pixel = readCenter (gl, output, width, height);
    }

    renderer.uploadRgba (sourceHandle, red.data(), width, height, width * 4);
    renderer.renderComposite (&gl, &layer, 1, nullptr, 0);
    CFMutableDictionaryRef properties = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    setSurfaceInt (properties, kIOSurfaceWidth, width);
    setSurfaceInt (properties, kIOSurfaceHeight, height);
    setSurfaceInt (properties, kIOSurfaceBytesPerElement, 4);
    setSurfaceInt (properties, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CFDictionarySetValue (properties, kIOSurfaceIsGlobal, kCFBooleanTrue);
#pragma clang diagnostic pop
    IOSurfaceRef directSurface = IOSurfaceCreate (properties);
    CFRelease (properties);
    if (directSurface == nullptr
        || ! renderer.renderCompositeToIOSurface (
            &gl, directSurface, width, height, &layer, 1, nullptr, 0))
    {
        std::cerr << "Metal compositor did not render directly to IOSurface\n";
        return 1;
    }
    IOSurfaceLock (directSurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (directSurface));
    const size_t rowBytes = IOSurfaceGetBytesPerRow (directSurface);
    const size_t center = static_cast<size_t> (height / 2) * rowBytes
                        + static_cast<size_t> (width / 2) * 4;
    const std::array<uint8_t, 4> directPixel = {
        bytes[center + 2], bytes[center + 1], bytes[center], bytes[center + 3] };
    IOSurfaceUnlock (directSurface, kIOSurfaceLockReadOnly, nullptr);
    if (! isRed (directPixel))
    {
        std::cerr << "Direct Metal IOSurface output did not contain the red frame\n";
        return 1;
    }

    const auto consumer = std::filesystem::path (argc > 0 ? argv[0] : "")
                            .parent_path() / "arbit-metal-iosurface-consumer";
    std::ostringstream consumerCommand;
    consumerCommand << std::quoted (consumer.string()) << ' '
                    << IOSurfaceGetID (directSurface) << ' ' << width << ' ' << height;
    const int consumerStatus = std::system (consumerCommand.str().c_str());
    if (consumerStatus == -1 || ! WIFEXITED (consumerStatus)
        || WEXITSTATUS (consumerStatus) != 0)
    {
        std::cerr << "Cross-process Metal consumer could not see the rendered IOSurface\n";
        return 1;
    }
    renderer.clearDirectOutputs();
    CFRelease (directSurface);

    renderer.deleteTexture (overlayHandle);
    renderer.deleteTexture (sourceHandle);
    renderer.shutdown (&gl);

    // The production shared-viewport mode must work without a GL context or
    // GL-backed source IDs. Exercise opaque Metal uploads, GPU frame blending,
    // and direct IOSurface output as one strict path.
    videorender::MetalFrameRenderer directRenderer;
    if (! directRenderer.initialize (nullptr, width, height, error, true))
    {
        std::cerr << "Metal-only compositor initialization failed: " << error << '\n';
        return 1;
    }
    directRenderer.setBackgroundColor (0, 0, 0, 1);
    constexpr unsigned redHandle = 0x80000001u;
    constexpr unsigned blueHandle = 0x80000002u;
    constexpr unsigned mixHandle = 0x80000003u;
    std::vector<uint8_t> fullBlue (static_cast<size_t> (width) * height * 4, 255);
    for (size_t i = 0; i < fullBlue.size(); i += 4)
        fullBlue[i] = fullBlue[i + 1] = 0;
    directRenderer.uploadRgba (redHandle, red.data(), width, height, width * 4);
    directRenderer.uploadRgba (blueHandle, fullBlue.data(), width, height, width * 4);
    directRenderer.setFrameBlend (mixHandle, redHandle, blueHandle,
                                  width, height, 0.5f);
    videorender::LayerDesc mixedLayer;
    mixedLayer.texture = mixHandle;
    mixedLayer.texWidth = width;
    mixedLayer.texHeight = height;

    properties = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    setSurfaceInt (properties, kIOSurfaceWidth, width);
    setSurfaceInt (properties, kIOSurfaceHeight, height);
    setSurfaceInt (properties, kIOSurfaceBytesPerElement, 4);
    setSurfaceInt (properties, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
    IOSurfaceRef metalOnlySurface = IOSurfaceCreate (properties);
    CFRelease (properties);
    if (metalOnlySurface == nullptr
        || ! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor did not render its frame blend\n";
        return 1;
    }

    // Execute the real typed-plan lowering seam, then bind both uploaded RGBA
    // resources through the strict production Metal compositor. The matte has
    // two uniform regions so samples away from its edge remain deterministic
    // while still exercising bounded morphology, feather, levels, choke, and
    // invert in the production shader.
    constexpr unsigned matteHandle = 0x80000005u;
    std::vector<uint8_t> matteRgba (static_cast<size_t> (width) * height * 4, 255);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t offset = (static_cast<size_t> (y) * width + x) * 4;
            matteRgba[offset] = x < width / 2 ? 64 : 192;
            matteRgba[offset + 1] = 17;
            matteRgba[offset + 2] = 239;
        }
    directRenderer.uploadRgba (matteHandle, matteRgba.data(), width, height, width * 4);

    auto mattePlan = typedMattePlan();
    videorender::LayerDesc matteLayer;
    matteLayer.clipId = mattePlan.clipId;
    matteLayer.texture = redHandle;
    matteLayer.texWidth = width;
    matteLayer.texHeight = height;
    matteLayer.matteTexture = matteHandle;
    matteLayer.matteWidth = width;
    matteLayer.matteHeight = height;
    std::string matteError;
    if (! videowire::executeVisualLayerPlan (
            { mattePlan }, mattePlan.clipId, matteLayer, matteError)
        || ! matteLayer.matteApply || ! matteLayer.matteInvert
        || matteLayer.matteBlack != 0.2f || matteLayer.matteWhite != 0.8f
        || matteLayer.matteErodeDilate != 1.0f || matteLayer.matteFeather != 1.0f
        || matteLayer.matteChoke != 0.1f)
    {
        std::cerr << "Typed matte plan did not lower bounded refinement parameters: "
                  << matteError << '\n';
        return 1;
    }
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &matteLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected the typed matte frame: "
                  << directRenderer.lastError() << '\n';
        return 1;
    }

    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    const auto* matteBytes = static_cast<const uint8_t*> (
        IOSurfaceGetBaseAddress (metalOnlySurface));
    const size_t matteRowBytes = IOSurfaceGetBytesPerRow (metalOnlySurface);
    const auto mattePixelAt = [&](int x)
    {
        const size_t offset = static_cast<size_t> (height / 2) * matteRowBytes
                            + static_cast<size_t> (x) * 4;
        return std::array<uint8_t, 4> { matteBytes[offset + 2], matteBytes[offset + 1],
                                        matteBytes[offset], matteBytes[offset + 3] };
    };
    const auto matteLeft = mattePixelAt (width / 8);
    const auto matteRight = mattePixelAt (width * 7 / 8);
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    if (matteLeft[0] < 195 || matteLeft[0] > 220 || matteLeft[1] > 4
        || matteLeft[2] > 4 || matteLeft[3] < 248
        || matteRight[0] > 4 || matteRight[1] > 4 || matteRight[2] > 4
        || matteRight[3] < 248)
    {
        std::cerr << "Typed matte pixels did not reflect the uploaded red channel; left="
                  << static_cast<int> (matteLeft[0]) << ','
                  << static_cast<int> (matteLeft[1]) << ','
                  << static_cast<int> (matteLeft[2]) << ','
                  << static_cast<int> (matteLeft[3]) << " right="
                  << static_cast<int> (matteRight[0]) << ','
                  << static_cast<int> (matteRight[1]) << ','
                  << static_cast<int> (matteRight[2]) << ','
                  << static_cast<int> (matteRight[3]) << '\n';
        return 1;
    }

    auto malformedMattePlan = mattePlan;
    malformedMattePlan.edges[2].toPort = 0;
    if (videowire::executeVisualLayerPlan (
            { malformedMattePlan }, malformedMattePlan.clipId, matteLayer, matteError)
        || matteError != "typed matte graph has unsupported production topology")
    {
        std::cerr << "Typed matte executor accepted malformed topology\n";
        return 1;
    }
    videorender::LayerDesc texturelessMatte = matteLayer;
    texturelessMatte.matteTexture = 0;
    if (videowire::executeVisualLayerPlan (
            { mattePlan }, mattePlan.clipId, texturelessMatte, matteError)
        || matteError != "typed matte GPU texture is unavailable")
    {
        std::cerr << "Typed matte executor accepted a missing texture\n";
        return 1;
    }
    videorender::LayerDesc missingResourceLayer = mixedLayer;
    missingResourceLayer.texture = 0x8fffffffu;
    if (directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height,
            &missingResourceLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal accepted a missing GPU resource instead of failing closed\n";
        return 1;
    }
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal did not recover after a rejected frame\n";
        return 1;
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    const size_t metalOnlyRowBytes = IOSurfaceGetBytesPerRow (metalOnlySurface);
    const size_t metalOnlyCenter = static_cast<size_t> (height / 2) * metalOnlyRowBytes
                                 + static_cast<size_t> (width / 2) * 4;
    const std::array<uint8_t, 4> mixedPixel = {
        bytes[metalOnlyCenter + 2], bytes[metalOnlyCenter + 1],
        bytes[metalOnlyCenter], bytes[metalOnlyCenter + 3] };
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);

    // Strict production-path coverage: every public effect, blend mode,
    // transition, mask shape, and a combined transform/crop frame must render
    // without a GL context. Individual visual assertions above catch shader
    // correctness; this matrix prevents newly added combinations from silently
    // reintroducing an API fallback.
    for (int effectType = 0; effectType < videofx::kEffectTypeCount; ++effectType)
    {
        videorender::EffectSlotState effectState;
        effectState.type = effectType;
        effectState.enabled = true;
        const auto* definition = videofx::effectDefFor (effectType);
        for (int param = 0; definition != nullptr && param < definition->paramCount; ++param)
            effectState.params[param] = definition->params[param].defaultValue;
        mixedLayer.clipId = 1000 + effectType;
        mixedLayer.effects = &effectState;
        mixedLayer.effectCount = 1;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height, &mixedLayer, 1, nullptr, 0))
        {
            std::cerr << "Strict Metal rejected effect " << effectType << ": "
                      << directRenderer.lastError() << '\n';
            return 1;
        }
    }
    mixedLayer.effects = nullptr;
    mixedLayer.effectCount = 0;

    videorender::LayerDesc redLayer = mixedLayer;
    redLayer.texture = redHandle;
    videorender::LayerDesc blueLayer = mixedLayer;
    blueLayer.texture = blueHandle;
    blueLayer.opacity = 0.5f;
    std::array<videorender::LayerDesc, 2> blendLayers = { redLayer, blueLayer };
    for (int blendMode = 0; blendMode < static_cast<int> (videofx::BlendMode::Count);
         ++blendMode)
    {
        blendLayers[1].blendMode = blendMode;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height,
                blendLayers.data(), static_cast<int> (blendLayers.size()), nullptr, 0))
        {
            std::cerr << "Strict Metal rejected blend mode " << blendMode << '\n';
            return 1;
        }
    }

    videorender::LayerDesc transitionLayer = blueLayer;
    transitionLayer.opacity = 1.0f;
    transitionLayer.fromLayer = &redLayer;
    transitionLayer.transitionProgress = 0.5f;
    for (int transition = 1;
         transition < static_cast<int> (videofx::TransitionType::Count); ++transition)
    {
        transitionLayer.transitionType = transition;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height,
                &transitionLayer, 1, nullptr, 0))
        {
            std::cerr << "Strict Metal rejected transition " << transition << '\n';
            return 1;
        }
    }

    for (int maskType = 1; maskType <= 2; ++maskType)
    {
        videorender::LayerDesc masked = redLayer;
        masked.maskType = maskType;
        masked.maskInvert = maskType == 2;
        masked.maskFeather = 0.15f;
        masked.scale = 0.75f;
        masked.rotationDeg = 17.0f;
        masked.translateX = 0.1f;
        masked.translateY = -0.1f;
        masked.cropLeft = 0.05f;
        masked.cropBottom = 0.1f;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height, &masked, 1, nullptr, 0))
        {
            std::cerr << "Strict Metal rejected mask/transform combination "
                      << maskType << '\n';
            return 1;
        }
    }

    constexpr unsigned strictLutHandle = 0x80000004u;
    directRenderer.uploadLut3D (strictLutHandle, magentaLut.data(), 2);
    videorender::LayerDesc lutLayer = redLayer;
    lutLayer.lutTexture = strictLutHandle;
    lutLayer.lutSize = 2;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &lutLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected a 3D LUT layer\n";
        return 1;
    }
    directRenderer.deleteTexture (strictLutHandle);

    videorender::LayerDesc particleLayer;
    particleLayer.clipId = 77;
    particleLayer.particleSource = true;
    particleLayer.texWidth = width;
    particleLayer.texHeight = height;
    particleLayer.genParams["count"] = 512;
    particleLayer.genParams["size"] = 4.0;
    particleLayer.genParams["force"] = 1.0;
    particleLayer.notesPresent = true;
    particleLayer.noteFeatures.notesTex.resize (128u * 4u * 4u, 0.0f);
    particleLayer.noteFeatures.notesTex[0] = 60.0f;
    particleLayer.noteFeatures.notesTex[1] = 1.0f;
    particleLayer.noteFeatures.notesTex[4] = 261.625565f;
    particleLayer.noteFeatures.noteCount = 1;
    particleLayer.shaderClock.playing = true;
    particleLayer.shaderClock.timeDelta = 1.0 / 30.0;
    videorender::EffectSlotState particleBlur;
    particleBlur.type = static_cast<int> (videofx::EffectType::Blur);
    particleBlur.enabled = true;
    particleBlur.params[0] = 2.0f;
    particleLayer.effects = &particleBlur;
    particleLayer.effectCount = 1;
    for (int frame = 0; frame < 4; ++frame)
    {
        particleLayer.shaderClock.frame = frame;
        if (! directRenderer.renderCompositeToIOSurface (
                nullptr, metalOnlySurface, width, height,
                &particleLayer, 1, nullptr, 0))
        {
            std::cerr << "Metal-only compositor rejected a particle layer\n";
            return 1;
        }
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    int litParticlePixels = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const size_t pixelOffset = static_cast<size_t> (y) * metalOnlyRowBytes
                                     + static_cast<size_t> (x) * 4;
            if (bytes[pixelOffset] > 4 || bytes[pixelOffset + 1] > 4
                || bytes[pixelOffset + 2] > 4)
                ++litParticlePixels;
        }
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    const bool particlePassed = litParticlePixels > 0
        && directRenderer.particleBackend() == "metal-compute";

    std::string shaderLog;
    std::vector<videorender::GenParam> shaderParams;
    if (! directRenderer.setClipShader (
            88, "void main(){ fragColor=(uTime>0.5 && uResolution.x>1.0)"
                "?vec4(0.0,1.0,0.0,1.0):vec4(1.0,0.0,0.0,1.0); }",
            shaderLog, shaderParams))
    {
        std::cerr << "Native Metal generator compile failed: " << shaderLog << '\n';
        return 1;
    }
    videorender::LayerDesc shaderLayer;
    shaderLayer.clipId = 88;
    shaderLayer.shaderSource = true;
    shaderLayer.shaderClock.timeSec = 1.0;
    shaderLayer.effects = &invert;
    shaderLayer.effectCount = 1;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &shaderLayer, 1, nullptr, 0))
    {
        std::cerr << "Strict Metal rejected a generator/effect combination\n";
        return 1;
    }
    shaderLayer.effects = nullptr;
    shaderLayer.effectCount = 0;
    if (! directRenderer.renderCompositeToIOSurface (
            nullptr, metalOnlySurface, width, height, &shaderLayer, 1, nullptr, 0))
    {
        std::cerr << "Metal-only compositor rejected its native generator: "
                  << directRenderer.lastError() << '\n';
        return 1;
    }
    IOSurfaceLock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    bytes = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (metalOnlySurface));
    const std::array<uint8_t, 4> shaderPixel = {
        bytes[metalOnlyCenter + 2], bytes[metalOnlyCenter + 1],
        bytes[metalOnlyCenter], bytes[metalOnlyCenter + 3] };
    const size_t shaderInterior = static_cast<size_t> (height / 2) * metalOnlyRowBytes
                                + static_cast<size_t> (width / 8) * 4;
    const std::array<uint8_t, 4> shaderInteriorPixel = {
        bytes[shaderInterior + 2], bytes[shaderInterior + 1],
        bytes[shaderInterior], bytes[shaderInterior + 3] };
    IOSurfaceUnlock (metalOnlySurface, kIOSurfaceLockReadOnly, nullptr);
    directRenderer.clearClipShader (88);
    directRenderer.clearDirectOutputs();
    CFRelease (metalOnlySurface);
    directRenderer.deleteTexture (matteHandle);
    directRenderer.deleteTexture (mixHandle);
    directRenderer.deleteTexture (blueHandle);
    directRenderer.deleteTexture (redHandle);
    directRenderer.shutdown (nullptr);
    if (mixedPixel[0] < 120 || mixedPixel[0] > 136
        || mixedPixel[1] > 8 || mixedPixel[2] < 120 || mixedPixel[2] > 136
        || mixedPixel[3] < 248)
    {
        std::cerr << "Metal-only frame blend was not half red/half blue\n";
        return 1;
    }
    if (! particlePassed)
    {
        std::cerr << "Metal-only particle layer produced no visible pixels\n";
        return 1;
    }
    if (! isGreen (shaderPixel))
    {
        std::cerr << "Native Metal generator was not rendered into the compositor; got "
                  << static_cast<int> (shaderPixel[0]) << ','
                  << static_cast<int> (shaderPixel[1]) << ','
                  << static_cast<int> (shaderPixel[2]) << ','
                  << static_cast<int> (shaderPixel[3]) << '\n';
        return 1;
    }
    if (! isGreen (shaderInteriorPixel))
    {
        std::cerr << "Native Metal generator did not fill the canvas; interior RGBA="
                  << static_cast<int> (shaderInteriorPixel[0]) << ','
                  << static_cast<int> (shaderInteriorPixel[1]) << ','
                  << static_cast<int> (shaderInteriorPixel[2]) << ','
                  << static_cast<int> (shaderInteriorPixel[3]) << '\n';
        return 1;
    }

    glfwDestroyWindow (window);
    glfwTerminate();

    std::cout << "Metal compositor center RGBA="
              << static_cast<int> (pixel[0]) << ','
              << static_cast<int> (pixel[1]) << ','
              << static_cast<int> (pixel[2]) << ','
              << static_cast<int> (pixel[3]) << '\n';
    if (! isBlue (pixel))
    {
        std::cerr << "Metal compositor did not composite the blue overlay\n";
        return 1;
    }
    std::cout << "Metal frame compositor smoke passed\n";
    return 0;
}
