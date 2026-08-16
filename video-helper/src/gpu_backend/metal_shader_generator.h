// Native runtime shader-generator path for the strict macOS Metal viewport.
#pragma once

#if defined(__APPLE__) && ARBIT_HAVE_METAL_GENERATORS && ARBIT_HAVE_VIEWPORT

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace videorender
{

struct AudioFeatures;
struct GenParam;
struct NoteFeatures;
struct ShaderClock;

class MetalShaderGenerator
{
public:
    MetalShaderGenerator();
    ~MetalShaderGenerator();
    MetalShaderGenerator (const MetalShaderGenerator&) = delete;
    MetalShaderGenerator& operator= (const MetalShaderGenerator&) = delete;

    bool setSource (const std::string& rawSource);
    void setImage (const std::string& name, const uint8_t* rgba,
                   int width, int height, int strideBytes);
    uint32_t renderViewUnlocked (const ShaderClock& clock, int width, int height,
                                 const AudioFeatures* audio,
                                 const NoteFeatures* notes,
                                 const std::map<std::string, double>* genValues);
    void shutdownUnlocked();

    bool hasProgram() const;
    const std::string& log() const;
    const std::vector<GenParam>& params() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace videorender

#endif
