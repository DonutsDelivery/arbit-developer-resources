// gpu_backend/frame_renderer_metal.h -- native Metal production compositor.
#pragma once

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND && ARBIT_HAVE_VIEWPORT

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace arbitgl { struct GlFuncs; }
namespace videopreview { struct State; }
namespace videowire { class VisualPlanTelemetry; }

namespace videorender
{

struct LayerDesc;
struct ImageLayerDesc;
struct GenParam;

class MetalFrameRenderer
{
public:
    MetalFrameRenderer();
    ~MetalFrameRenderer();
    MetalFrameRenderer (const MetalFrameRenderer&) = delete;
    MetalFrameRenderer& operator= (const MetalFrameRenderer&) = delete;

    bool initialize (const arbitgl::GlFuncs* gl, int width, int height,
                     std::string& errorOut, bool directOnly = false);
    void shutdown (const arbitgl::GlFuncs* gl);
    void setOutputSize (const arbitgl::GlFuncs* gl, int width, int height);
    void setCanvas (int width, int height);
    void setPresentSize (int width, int height);
    void setView (float zoom, float panX, float panY);
    void setBackgroundColor (float r, float g, float b, float a);
    void setPostFx (float bloomIntensity, float bloomThreshold,
                    float bloomRadius, int tonemap, float exposure);
    void setInspection (int transformClipId, unsigned requestedHandle,
                        const videopreview::State& presentation);
    unsigned inspectionHandle() const;
    bool inspectionResourceAdmitted() const;

    void uploadRgba (unsigned textureHandle, const uint8_t* rgba,
                     int width, int height, int strideBytes);
    bool uploadR16 (unsigned textureHandle, const uint16_t* pixels,
                    int width, int height);
    void setFrameBlend (unsigned outputHandle, unsigned textureA,
                        unsigned textureB, int width, int height, float mix);
    void uploadLut3D (unsigned textureHandle, const float* rgbTriples, int size);
    void deleteTexture (unsigned textureHandle);
    bool setClipShader (int clipId, const std::string& source,
                        std::string& logOut, std::vector<GenParam>& paramsOut);
    void clearClipShader (int clipId);
    bool hasClipShader (int clipId) const;
    void setClipImage (int clipId, const std::string& name,
                       const uint8_t* rgba, int width, int height, int strideBytes);

    // Returns a GL_TEXTURE_2D view only for the explicit legacy interop test
    // configuration. Production Metal sessions render directly to IOSurface;
    // zero is a hard frame rejection and must never trigger an API fallback.
    unsigned renderComposite (const arbitgl::GlFuncs* gl,
                              const LayerDesc* layers, int numLayers,
                              const ImageLayerDesc* overlays, int numOverlays);
    bool renderCompositeToIOSurface (const arbitgl::GlFuncs* gl, void* ioSurface,
                                     int width, int height,
                                     const LayerDesc* layers, int numLayers,
                                     const ImageLayerDesc* overlays, int numOverlays);
    void clearDirectOutputs();

    bool ready() const;
    const std::string& lastError() const;
    const std::string& particleBackend() const;
    void setVisualTelemetryOwner (videowire::VisualPlanTelemetry* owner);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace videorender

#endif
