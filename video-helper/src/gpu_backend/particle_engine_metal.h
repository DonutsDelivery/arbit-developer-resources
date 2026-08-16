// gpu_backend/particle_engine_metal.h -- experimental P6 Metal particle path.
#pragma once

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND

#include <memory>
#include <cstdint>
#include <string>

namespace arbitgl { struct GlFuncs; }

namespace videorender
{

struct NoteFeatures;
struct ParticleParams;
struct ParticleDiagnostics;
struct ShaderClock;

class MetalParticleEngine
{
public:
    MetalParticleEngine();
    ~MetalParticleEngine();
    MetalParticleEngine (const MetalParticleEngine&) = delete;
    MetalParticleEngine& operator= (const MetalParticleEngine&) = delete;

    // Deliberately opt-in until GL-vs-Metal frame parity has been measured on
    // real Apple hardware. Set ARBIT_VIDEO_METAL=1 in the helper environment.
    static bool enabled();

    unsigned render (const arbitgl::GlFuncs* gl, const ShaderClock& clock,
                     int width, int height, const ParticleParams& params,
                     const NoteFeatures* notes);
    void shutdown (const arbitgl::GlFuncs* gl);

    const std::string& log() const;
    const ParticleDiagnostics& diagnostics() const;

private:
    friend class MetalFrameRenderer;
    uint32_t renderMetalViewUnlocked (const ShaderClock& clock,
                                      int width, int height,
                                      const ParticleParams& params,
                                      const NoteFeatures* notes);
    uint32_t renderViewUnlocked (const arbitgl::GlFuncs* gl,
                                 const ShaderClock& clock,
                                 int width, int height,
                                 const ParticleParams& params,
                                 const NoteFeatures* notes,
                                 bool nativeOnly);
    void shutdownUnlocked (const arbitgl::GlFuncs* gl);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace videorender

#endif
