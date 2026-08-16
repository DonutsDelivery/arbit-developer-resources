#include "metal_shader_generator.h"

#if defined(__APPLE__) && ARBIT_HAVE_METAL_GENERATORS && ARBIT_HAVE_VIEWPORT

#include "../shader_dialect.h"
#include "../shader_generator.h"

#import <Metal/Metal.h>

#include <MoltenVKShaderConverter/SPIRVConversion.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <unordered_map>

#define SOKOL_METAL
#include "sokol_gfx.h"

namespace videorender
{
namespace
{

struct MetalPassInfo { std::string target; bool persistent = false; };

struct MetalTarget
{
    int width = 0, height = 0;
    id<MTLTexture> texture = nil;
    sg_image image = {};
    sg_view view = {};
};

void destroyTarget (MetalTarget& target)
{
    if (target.view.id != 0) sg_destroy_view (target.view);
    if (target.image.id != 0) sg_destroy_image (target.image);
#if ! __has_feature(objc_arc)
    [target.texture release];
#endif
    target = {};
}

bool makeTarget (id<MTLDevice> device, MetalTarget& target, int width, int height,
                 const char* label)
{
    (void) device;
    if (target.image.id != 0 && target.width == width && target.height == height)
        return true;
    destroyTarget (target);
    sg_image_desc imageDesc = {};
    imageDesc.usage.color_attachment = true;
    imageDesc.width = width;
    imageDesc.height = height;
    imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
    imageDesc.sample_count = 1;
    imageDesc.label = label;
    target.image = sg_make_image (&imageDesc);
    const sg_mtl_image_info native = sg_mtl_query_image_info (target.image);
    target.texture = (__bridge id<MTLTexture>) native.tex[native.active_slot];
#if ! __has_feature(objc_arc)
    [target.texture retain];
#endif
    sg_view_desc viewDesc = {};
    viewDesc.texture.image = target.image;
    target.view = sg_make_view (&viewDesc);
    target.width = width;
    target.height = height;
    return sg_query_image_state (target.image) == SG_RESOURCESTATE_VALID
        && sg_query_view_state (target.view) == SG_RESOURCESTATE_VALID;
}

std::string trim (std::string value)
{
    const auto first = value.find_first_not_of (" \t\r");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of (" \t\r");
    return value.substr (first, last - first + 1);
}

// Vulkan GLSL requires ordinary uniforms to live in a block and every opaque
// resource to have an explicit binding. Keep the public dialect unchanged and
// mechanically adapt only its generated prelude before glslang sees it.
std::string metalGlsl (const std::string& wrapped)
{
    struct Uniform { std::string type, name; };
    std::vector<Uniform> values;
    std::vector<std::string> body;
    int samplerBinding = 1;
    bool inPrelude = true;

    std::istringstream input (wrapped);
    std::string line;
    while (std::getline (input, line))
    {
        const std::string clean = trim (line);
        if (clean.rfind ("#version", 0) == 0)
        {
            body.push_back ("#version 450");
            continue;
        }
        if (clean == "// --- user shader ---") inPrelude = false;
        if (inPrelude && clean.rfind ("uniform ", 0) == 0)
        {
            std::istringstream declaration (clean.substr (8));
            Uniform uniform;
            declaration >> uniform.type >> uniform.name;
            if (! uniform.name.empty() && uniform.name.back() == ';') uniform.name.pop_back();
            if (uniform.type.rfind ("sampler", 0) == 0)
                body.push_back ("layout(set=0,binding=" + std::to_string (samplerBinding++)
                                + ") uniform " + uniform.type + " " + uniform.name + ";");
            else
                values.push_back (uniform);
            continue;
        }
        if (inPrelude && clean == "out vec4 fragColor;")
        {
            body.push_back ("layout(location=0) out vec4 fragColor;");
            continue;
        }
        body.push_back (line);
    }

    std::ostringstream output;
    bool emitted = false;
    for (const auto& current : body)
    {
        if (! emitted && trim (current) == "// --- Arbit uniform contract (auto-prepended) ---")
        {
            output << current << '\n';
            output << "layout(std140,set=0,binding=0) uniform ArbitUniformBlock {\n";
            for (const auto& uniform : values)
                output << "    " << uniform.type << ' ' << uniform.name << ";\n";
            output << "} arbitUniforms;\n";
            for (const auto& uniform : values)
                output << "#define " << uniform.name << " arbitUniforms." << uniform.name << '\n';
            emitted = true;
        }
        else
            output << current << '\n';
    }
    return output.str();
}

bool compileSpirv (const std::string& source, std::vector<uint32_t>& words,
                   std::string& error)
{
    static const bool initialized = glslang_initialize_process() != 0;
    if (! initialized) { error = "glslang process initialization failed"; return false; }
    const glslang_input_t input = {
        GLSLANG_SOURCE_GLSL, GLSLANG_STAGE_FRAGMENT, GLSLANG_CLIENT_VULKAN,
        GLSLANG_TARGET_VULKAN_1_2, GLSLANG_TARGET_SPV, GLSLANG_TARGET_SPV_1_5,
        source.c_str(), 450, GLSLANG_CORE_PROFILE, false, false,
        GLSLANG_MSG_DEFAULT_BIT, glslang_default_resource()
    };
    glslang_shader_t* shader = glslang_shader_create (&input);
    if (shader == nullptr) { error = "glslang shader allocation failed"; return false; }
    glslang_shader_set_options (shader, GLSLANG_SHADER_AUTO_MAP_LOCATIONS);
    auto fail = [&] (const char* phase)
    {
        error = std::string (phase) + ": " + glslang_shader_get_info_log (shader)
              + "\n" + glslang_shader_get_info_debug_log (shader);
    };
    if (! glslang_shader_preprocess (shader, &input))
    {
        fail ("GLSL preprocess"); glslang_shader_delete (shader); return false;
    }
    glslang_shader_set_preprocessed_code (shader, glslang_shader_get_preprocessed_code (shader));
    if (! glslang_shader_parse (shader, &input))
    {
        fail ("GLSL parse"); glslang_shader_delete (shader); return false;
    }
    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader (program, shader);
    if (! glslang_program_link (program,
            GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
    {
        error = std::string ("GLSL link: ") + glslang_program_get_info_log (program)
              + "\n" + glslang_program_get_info_debug_log (program);
        glslang_program_delete (program); glslang_shader_delete (shader); return false;
    }
    glslang_program_SPIRV_generate (program, GLSLANG_STAGE_FRAGMENT);
    words.resize (glslang_program_SPIRV_get_size (program));
    glslang_program_SPIRV_get (program, words.data());
    if (const char* messages = glslang_program_SPIRV_get_messages (program)) error = messages;
    glslang_program_delete (program);
    glslang_shader_delete (shader);
    return ! words.empty();
}

id<MTLTexture> makeTexture (id<MTLDevice> device, MTLPixelFormat format,
                            MTLTextureType type, int width, int height)
{
    MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = type;
    descriptor.pixelFormat = format;
    descriptor.width = std::max (width, 1);
    descriptor.height = type == MTLTextureType1D ? 1 : std::max (height, 1);
    descriptor.depth = 1;
    descriptor.mipmapLevelCount = 1;
    descriptor.arrayLength = 1;
    descriptor.sampleCount = 1;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> result = [device newTextureWithDescriptor:descriptor];
#if ! __has_feature(objc_arc)
    [descriptor release];
#endif
    return result;
}

void replaceRgba8 (id<MTLTexture> texture, const uint8_t* rgba,
                   int width, int height, int stride)
{
    [texture replaceRegion:MTLRegionMake2D (0, 0, width, height)
               mipmapLevel:0 withBytes:rgba bytesPerRow:stride];
}

} // namespace

struct MetalShaderGenerator::Impl
{
    id<MTLDevice> device = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    MTLRenderPipelineReflection* reflection = nil;
    id<MTLSamplerState> sampler = nil;
    MetalTarget output;
    std::map<std::string, MetalTarget> passTargets[2];
    unsigned parity = 0;
    std::vector<MetalPassInfo> passes;
    bool multipass = false;
    std::vector<GenParam> params;
    std::string log;
    std::unordered_map<std::string, id<MTLTexture>> images;
    id<MTLTexture> black1D = nil, black2D = nil, bands1D = nil, bands2D = nil;
    id<MTLTexture> notesTexture = nil, linksTexture = nil;
    int bandsCount = 0;

    void releasePipeline()
    {
#if ! __has_feature(objc_arc)
        [pipeline release];
        [reflection release];
#endif
        pipeline = nil;
        reflection = nil;
    }

    void releaseTexture (id<MTLTexture>& texture)
    {
#if ! __has_feature(objc_arc)
        [texture release];
#endif
        texture = nil;
    }

    bool ensureDefaults()
    {
        if (black1D != nil) return true;
        black1D = makeTexture (device, MTLPixelFormatR32Float, MTLTextureType1D, 1, 1);
        black2D = makeTexture (device, MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 1, 1);
        const float zero = 0.0f;
        const uint8_t clear[4] = {};
        [black1D replaceRegion:MTLRegionMake1D (0, 1) mipmapLevel:0
                     withBytes:&zero bytesPerRow:sizeof (zero)];
        replaceRgba8 (black2D, clear, 1, 1, 4);
        MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
        descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler = [device newSamplerStateWithDescriptor:descriptor];
#if ! __has_feature(objc_arc)
        [descriptor release];
#endif
        return black1D != nil && black2D != nil && sampler != nil;
    }

    void clearPassTargets()
    {
        for (auto& bank : passTargets)
        {
            for (auto& item : bank) destroyTarget (item.second);
            bank.clear();
        }
        parity = 0;
    }

    bool ensureRenderTargets (int width, int height)
    {
        if (! makeTarget (device, output, width, height, "arbit-metal-generator-output"))
            return false;
        if (! multipass) return true;
        for (const auto& pass : passes)
        {
            if (pass.target.empty()) continue;
            if (! makeTarget (device, passTargets[0][pass.target], width, height,
                              "arbit-metal-generator-pass-a")) return false;
            if (pass.persistent
                && ! makeTarget (device, passTargets[1][pass.target], width, height,
                                 "arbit-metal-generator-pass-b")) return false;
        }
        return true;
    }

    MTLStructMember* uniformMember (NSString* name) const
    {
        for (MTLArgument* argument in reflection.fragmentArguments)
            if (argument.type == MTLArgumentTypeBuffer && argument.bufferStructType != nil)
                for (MTLStructMember* member in argument.bufferStructType.members)
                    if ([member.name isEqualToString:name]) return member;
        return nil;
    }

    MTLArgument* uniformArgument() const
    {
        for (MTLArgument* argument in reflection.fragmentArguments)
            if (argument.type == MTLArgumentTypeBuffer && argument.bufferStructType != nil)
                return argument;
        return nil;
    }

    void bindTexture (id<MTLRenderCommandEncoder> encoder,
                      const std::string& name, id<MTLTexture> texture)
    {
        NSString* wanted = [NSString stringWithUTF8String:name.c_str()];
        for (MTLArgument* argument in reflection.fragmentArguments)
        {
            if (argument.type == MTLArgumentTypeTexture
                && [argument.name isEqualToString:wanted])
                [encoder setFragmentTexture:texture atIndex:argument.index];
            else if (argument.type == MTLArgumentTypeSampler
                     && ([argument.name isEqualToString:wanted]
                         || [argument.name hasPrefix:wanted]))
                [encoder setFragmentSamplerState:sampler atIndex:argument.index];
        }
    }

    template <typename T>
    void put (std::vector<uint8_t>& bytes, const char* name, const T& value) const
    {
        MTLStructMember* member = uniformMember ([NSString stringWithUTF8String:name]);
        if (member == nil || member.offset + sizeof (T) > bytes.size()) return;
        std::memcpy (bytes.data() + member.offset, &value, sizeof (T));
    }

    void encode (id<MTLCommandBuffer> command, MetalTarget& target, int passIndex,
                 const ShaderClock& clock, const AudioFeatures* audio,
                 const NoteFeatures* notes,
                 const std::map<std::string, double>* values,
                 const std::map<std::string, id<MTLTexture>>& targetReads)
    {
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake (0, 0, 0, 0);
        id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:pipeline];
        [encoder setViewport:(MTLViewport) { 0, 0, (double) target.width,
                                             (double) target.height, 0, 1 }];

        if (MTLArgument* argument = uniformArgument())
        {
            std::vector<uint8_t> bytes (argument.bufferDataSize, 0);
            const float resolution[2] = { (float) target.width, (float) target.height };
            put (bytes, "uResolution", resolution);
            put (bytes, "uTime", (float) clock.timeSec);
            put (bytes, "uTimeDelta", (float) clock.timeDelta);
            put (bytes, "uFrame", clock.frame);
            put (bytes, "uBeat", (float) clock.beat);
            put (bytes, "uBPM", (float) clock.bpm);
            put (bytes, "uBeatPhase", (float) clock.beatPhase);
            put (bytes, "uBarPhase", (float) clock.barPhase);
            put (bytes, "uBeatsPerBar", (float) clock.beatsPerBar);
            put (bytes, "uClipBeat", (float) clock.clipBeat);
            put (bytes, "uClipLength", (float) clock.clipLength);
            const int playing = clock.playing ? 1 : 0;
            put (bytes, "uPlaying", playing);
            put (bytes, "uRMS", audio ? audio->rms : 0.0f);
            put (bytes, "uPeak", audio ? audio->peak : 0.0f);
            put (bytes, "uOnset", audio ? audio->onset : 0.0f);
            put (bytes, "uOnsetAge", audio ? audio->onsetAge : 0.0f);
            const bool haveNotes = notes != nullptr && ! notes->notesTex.empty();
            put (bytes, "uNoteCount", haveNotes ? notes->noteCount : 0);
            put (bytes, "uLinkCount", haveNotes ? notes->linkCount : 0);
            put (bytes, "uRootFreq", haveNotes ? notes->rootFreq : 0.0f);
            auto value = [values] (const char* name, float fallback)
            {
                if (values != nullptr)
                    if (const auto found = values->find (name); found != values->end())
                        return (float) found->second;
                return fallback;
            };
            const float camPos[3] = { value ("uCamPos.x", 0), value ("uCamPos.y", 0), value ("uCamPos.z", 5) };
            const float camTarget[3] = { value ("uCamTarget.x", 0), value ("uCamTarget.y", 0), value ("uCamTarget.z", 0) };
            const float camUp[3] = { value ("uCamUp.x", 0), value ("uCamUp.y", 1), value ("uCamUp.z", 0) };
            put (bytes, "uCamPos", camPos); put (bytes, "uCamTarget", camTarget); put (bytes, "uCamUp", camUp);
            put (bytes, "uCamFov", value ("uCamFov", 45));
            put (bytes, "uCamNear", value ("uCamNear", 0.05f));
            put (bytes, "uCamFar", value ("uCamFar", 60));
            put (bytes, "PASSINDEX", passIndex);
            for (const auto& param : params)
            {
                const int count = genParamComponentCount (param.type);
                if (count == 0) continue;
                if (genParamIsIntScalar (param.type))
                {
                    const int v = (int) std::llround (value (param.name.c_str(), (float) param.defaultV));
                    put (bytes, param.name.c_str(), v);
                    continue;
                }
                float component[4] = {};
                for (int i = 0; i < count; ++i)
                {
                    const std::string key = param.name + genParamComponentSuffix (param.type, i);
                    component[i] = value (key.c_str(), count == 1 ? (float) param.defaultV
                                                                  : (float) param.defaultVec[i]);
                }
                if (count == 1) put (bytes, param.name.c_str(), component[0]);
                else if (count == 2)
                {
                    const float pair[2] = { component[0], component[1] };
                    put (bytes, param.name.c_str(), pair);
                }
                else put (bytes, param.name.c_str(), component);
            }
            [encoder setFragmentBytes:bytes.data() length:bytes.size() atIndex:argument.index];
        }

        const bool haveBands = audio != nullptr && ! audio->bands.empty();
        bindTexture (encoder, "uAudioBands", haveBands ? bands1D : black1D);
        bindTexture (encoder, "uAudioBands2D", haveBands ? bands2D : black2D);
        const bool haveNotes = notes != nullptr && ! notes->notesTex.empty();
        bindTexture (encoder, "uNotes", haveNotes ? notesTexture : black2D);
        bindTexture (encoder, "uLinks", haveNotes ? linksTexture : black2D);
        for (const auto& param : params)
        {
            if (param.type == 5)
            {
                const auto image = images.find (param.name);
                bindTexture (encoder, param.name,
                             image != images.end() ? image->second : black2D);
            }
            else if (param.type == 6 || param.type == 7)
                bindTexture (encoder, param.name, haveBands ? bands2D : black2D);
        }
        for (const auto& targetRead : targetReads)
            bindTexture (encoder, targetRead.first, targetRead.second);
        for (MTLArgument* argument in reflection.fragmentArguments)
            if (argument.type == MTLArgumentTypeSampler)
                [encoder setFragmentSamplerState:sampler atIndex:argument.index];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
    }
};

MetalShaderGenerator::MetalShaderGenerator() : impl_ (std::make_unique<Impl>()) {}
MetalShaderGenerator::~MetalShaderGenerator() = default;

bool MetalShaderGenerator::setSource (const std::string& rawSource)
{
    const auto wrap = arbitshader::wrapToContract (rawSource);
    std::string diagnostics;
    for (const auto& item : wrap.diagnostics)
        diagnostics += (item.level == arbitshader::WrapDiagnostic::Error ? "error: " : "warning: ")
                     + item.message + "\n";
    if (wrap.hasError()) { impl_->log = diagnostics; return false; }

    std::vector<uint32_t> spirv;
    std::string compilerLog;
    if (! compileSpirv (metalGlsl (wrap.glsl), spirv, compilerLog))
    {
        impl_->log = diagnostics + "error: " + compilerLog;
        return false;
    }
    char* msl = nullptr;
    char* conversionLog = nullptr;
    if (! mvkConvertSPIRVToMSL (spirv.data(), spirv.size(),
                                &msl, &conversionLog, false, false) || msl == nullptr)
    {
        impl_->log = diagnostics + "error: SPIR-V to MSL conversion failed: "
                   + (conversionLog != nullptr ? conversionLog : "no diagnostic");
        std::free (msl); std::free (conversionLog);
        return false;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    impl_->device = queue.device;
    NSError* error = nil;
    NSString* mslSource = [NSString stringWithUTF8String:msl];
    id<MTLLibrary> fragmentLibrary = [impl_->device newLibraryWithSource:mslSource
                                                                 options:nil error:&error];
    std::free (msl); std::free (conversionLog);
    if (fragmentLibrary == nil)
    {
        impl_->log = diagnostics + "error: Metal library compile: "
                   + (error != nil ? error.localizedDescription.UTF8String : "unknown error");
        return false;
    }
    static NSString* vertexSource = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct O{float4 position [[position]];};\n"
        "vertex O arbitVertex(uint id [[vertex_id]]){constexpr float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};O o;o.position=float4(p[id],0,1);return o;}";
    id<MTLLibrary> vertexLibrary = [impl_->device newLibraryWithSource:vertexSource options:nil error:&error];
    id<MTLFunction> vertex = [vertexLibrary newFunctionWithName:@"arbitVertex"];
    id<MTLFunction> fragment = [fragmentLibrary newFunctionWithName:@"main0"];
    if (fragment == nil && fragmentLibrary.functionNames.count > 0)
        fragment = [fragmentLibrary newFunctionWithName:fragmentLibrary.functionNames[0]];
    MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = vertex;
    descriptor.fragmentFunction = fragment;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    MTLRenderPipelineReflection* reflection = nil;
    id<MTLRenderPipelineState> pipeline = [impl_->device
        newRenderPipelineStateWithDescriptor:descriptor
        options:MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
        reflection:&reflection error:&error];
#if ! __has_feature(objc_arc)
    [descriptor release]; [vertex release]; [fragment release];
    [vertexLibrary release]; [fragmentLibrary release];
#endif
    if (pipeline == nil)
    {
        impl_->log = diagnostics + "error: Metal pipeline compile: "
                   + (error != nil ? error.localizedDescription.UTF8String : "unknown error");
        return false;
    }
    impl_->releasePipeline();
    impl_->pipeline = pipeline;
    impl_->reflection = reflection;
    impl_->params.clear();
    for (const auto& input : wrap.params)
    {
        GenParam param { input.name, (int) input.type, input.defaultScalar,
                         input.minScalar, input.maxScalar,
                         { 0, 0, 0, 0 }, input.importedPath };
        for (int i = 0; i < 4; ++i) param.defaultVec[i] = input.defaultVec[(size_t) i];
        impl_->params.push_back (std::move (param));
    }
    impl_->passes.clear();
    impl_->multipass = false;
    for (const auto& pass : wrap.passes)
    {
        impl_->passes.push_back ({ pass.target, pass.persistent });
        impl_->multipass |= pass.persistent;
    }
    impl_->multipass |= impl_->passes.size() > 1;
    impl_->clearPassTargets();
    impl_->log = diagnostics + compilerLog;
    return impl_->ensureDefaults();
}

void MetalShaderGenerator::setImage (const std::string& name, const uint8_t* rgba,
                                     int width, int height, int strideBytes)
{
    if (rgba == nullptr || width <= 0 || height <= 0 || impl_->device == nil) return;
    auto& texture = impl_->images[name];
    if (texture == nil || texture.width != (NSUInteger) width || texture.height != (NSUInteger) height)
    {
        impl_->releaseTexture (texture);
        texture = makeTexture (impl_->device, MTLPixelFormatRGBA8Unorm,
                               MTLTextureType2D, width, height);
    }
    replaceRgba8 (texture, rgba, width, height, strideBytes);
}

uint32_t MetalShaderGenerator::renderViewUnlocked (
    const ShaderClock& clock, int width, int height, const AudioFeatures* audio,
    const NoteFeatures* notes, const std::map<std::string, double>* genValues)
{
    if (impl_->pipeline == nil || ! impl_->ensureDefaults()
        || ! impl_->ensureRenderTargets (width, height)) return 0;
    const bool haveBands = audio != nullptr && ! audio->bands.empty();
    if (haveBands)
    {
        const int count = (int) audio->bands.size();
        if (impl_->bandsCount != count)
        {
            impl_->releaseTexture (impl_->bands1D); impl_->releaseTexture (impl_->bands2D);
            impl_->bands1D = makeTexture (impl_->device, MTLPixelFormatR32Float,
                                          MTLTextureType1D, count, 1);
            impl_->bands2D = makeTexture (impl_->device, MTLPixelFormatR32Float,
                                          MTLTextureType2D, count, 1);
            impl_->bandsCount = count;
        }
        [impl_->bands1D replaceRegion:MTLRegionMake1D (0, count) mipmapLevel:0
                            withBytes:audio->bands.data() bytesPerRow:count * sizeof (float)];
        [impl_->bands2D replaceRegion:MTLRegionMake2D (0, 0, count, 1) mipmapLevel:0
                            withBytes:audio->bands.data() bytesPerRow:count * sizeof (float)];
    }
    const bool haveNotes = notes != nullptr && ! notes->notesTex.empty();
    if (haveNotes)
    {
        if (impl_->notesTexture == nil)
            impl_->notesTexture = makeTexture (impl_->device, MTLPixelFormatRGBA32Float,
                                                MTLTextureType2D, 4, 128);
        if (impl_->linksTexture == nil)
            impl_->linksTexture = makeTexture (impl_->device, MTLPixelFormatRGBA32Float,
                                                MTLTextureType2D, 256, 1);
        if (notes->notesTex.size() >= 4u * 128u * 4u)
            [impl_->notesTexture replaceRegion:MTLRegionMake2D (0, 0, 4, 128) mipmapLevel:0
                                       withBytes:notes->notesTex.data() bytesPerRow:4 * 4 * sizeof (float)];
        if (notes->linksTex.size() >= 256u * 4u)
            [impl_->linksTexture replaceRegion:MTLRegionMake2D (0, 0, 256, 1) mipmapLevel:0
                                       withBytes:notes->linksTex.data() bytesPerRow:256 * 4 * sizeof (float)];
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (! impl_->multipass)
    {
        impl_->encode (command, impl_->output, 0, clock, audio, notes, genValues, {});
    }
    else
    {
        int outputPass = -1;
        for (size_t i = 0; i < impl_->passes.size(); ++i)
            if (impl_->passes[i].target.empty()) outputPass = (int) i;
        if (outputPass < 0) outputPass = (int) impl_->passes.size() - 1;
        std::set<std::string> written;
        for (size_t i = 0; i < impl_->passes.size(); ++i)
        {
            const auto& pass = impl_->passes[i];
            const bool toOutput = pass.target.empty() || (int) i == outputPass;
            MetalTarget& write = toOutput ? impl_->output
                : impl_->passTargets[pass.persistent ? ((impl_->parity + 1) & 1) : 0][pass.target];
            std::map<std::string, id<MTLTexture>> reads;
            for (const auto& candidate : impl_->passes)
            {
                if (candidate.target.empty()) continue;
                id<MTLTexture> texture = impl_->black2D;
                if (candidate.target == pass.target)
                {
                    if (pass.persistent)
                        texture = impl_->passTargets[impl_->parity & 1][candidate.target].texture;
                }
                else if (candidate.persistent)
                    texture = impl_->passTargets[written.count (candidate.target)
                        ? ((impl_->parity + 1) & 1) : (impl_->parity & 1)][candidate.target].texture;
                else
                    texture = impl_->passTargets[0][candidate.target].texture;
                reads[candidate.target] = texture != nil ? texture : impl_->black2D;
            }
            impl_->encode (command, write, (int) i, clock, audio, notes, genValues, reads);
            if (! pass.target.empty()) written.insert (pass.target);
        }
        ++impl_->parity;
    }
    [command commit];
    return impl_->output.view.id;
}

void MetalShaderGenerator::shutdownUnlocked()
{
    impl_->releasePipeline();
    destroyTarget (impl_->output);
    impl_->clearPassTargets();
    for (auto& item : impl_->images) impl_->releaseTexture (item.second);
    impl_->images.clear();
    impl_->releaseTexture (impl_->black1D); impl_->releaseTexture (impl_->black2D);
    impl_->releaseTexture (impl_->bands1D); impl_->releaseTexture (impl_->bands2D);
    impl_->releaseTexture (impl_->notesTexture); impl_->releaseTexture (impl_->linksTexture);
#if ! __has_feature(objc_arc)
    [impl_->sampler release];
#endif
    impl_->sampler = nil;
}

bool MetalShaderGenerator::hasProgram() const { return impl_->pipeline != nil; }
const std::string& MetalShaderGenerator::log() const { return impl_->log; }
const std::vector<GenParam>& MetalShaderGenerator::params() const { return impl_->params; }

} // namespace videorender

#endif
