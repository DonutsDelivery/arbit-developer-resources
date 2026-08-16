#include "backend.h"
#if ARBIT_HAVE_VIEWPORT
#include "particle_engine_metal.h"
#if ARBIT_HAVE_METAL_GENERATORS
#include "metal_shader_generator.h"
#endif
#include "../gl_loader.h"
#include "../particle_engine.h"
#include "../renderer.h"
#include "../shader_generator.h"
#endif

#import <CoreFoundation/CoreFoundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <OpenGL/CGLIOSurface.h>
#import <OpenGL/OpenGL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#define SOKOL_IMPL
#define SOKOL_METAL
#include "sokol_gfx.h"

namespace
{

// The viewport is owned by another translation unit and may be torn down during
// process-static destruction. Keep the backend lock alive until the process is
// gone so a late renderer shutdown can never lock a destroyed std::mutex.
std::mutex& sokolMutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}
id<MTLDevice> gMetalDevice = nil;
std::string gSokolError;

bool ensureSokolMetal()
{
    if (sg_isvalid())
        return true;
    if (! gSokolError.empty())
        return false;

    gMetalDevice = MTLCreateSystemDefaultDevice();
    if (gMetalDevice == nil)
    {
        gSokolError = "Metal returned no default device";
        return false;
    }

    sg_desc desc = {};
    desc.environment.metal.device = (__bridge const void*) gMetalDevice;
    desc.environment.defaults.color_format = SG_PIXELFORMAT_BGRA8;
    desc.environment.defaults.depth_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    desc.environment.defaults.sample_count = 1;
    sg_setup (&desc);
    if (! sg_isvalid())
    {
        gSokolError = "sokol_gfx Metal initialization failed";
        return false;
    }
    return true;
}

uint32_t fnv1a (const void* data, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*> (data);
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

std::string deviceName (id<MTLDevice> device)
{
    if (device == nil || device.name == nil)
        return {};
    return std::string ([device.name UTF8String]);
}

void readBackSubmittedWork (id<MTLDevice> device, id<MTLBuffer> computeBuffer,
                            id<MTLTexture> renderTexture,
                            std::array<uint32_t, 4>& computed,
                            std::array<uint8_t, 4 * 4 * 4>& pixels)
{
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLBuffer> textureReadback = [device newBufferWithLength:pixels.size()
                                                        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (computeBuffer.storageMode == MTLStorageModeManaged)
        [blit synchronizeResource:computeBuffer];
    if (renderTexture != nil && textureReadback != nil)
    {
        [blit copyFromTexture:renderTexture
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake (0, 0, 0)
                   sourceSize:MTLSizeMake (4, 4, 1)
                     toBuffer:textureReadback
            destinationOffset:0
       destinationBytesPerRow:16
     destinationBytesPerImage:pixels.size()];
    }
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if (computeBuffer != nil && computeBuffer.contents != nullptr)
        std::memcpy (computed.data(), computeBuffer.contents, sizeof (computed));
    if (textureReadback != nil && textureReadback.contents != nullptr)
        std::memcpy (pixels.data(), textureReadback.contents, pixels.size());

#if ! __has_feature(objc_arc)
    [textureReadback release];
#endif
}

bool resourceValid (sg_resource_state state)
{
    return state == SG_RESOURCESTATE_VALID;
}

} // namespace

namespace arbitgpu
{

BackendInfo queryNativeBackend()
{
    BackendInfo result;
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            result.error = "Metal returned no default device";
            return result;
        }

        result.available = true;
        result.compute = true;
        result.backend = "metal";
        result.device = deviceName (device);
    }
    return result;
}

BackendSelfTest runNativeBackendSelfTest()
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    BackendSelfTest result;

    @autoreleasepool
    {
        if (! ensureSokolMetal())
        {
            result.error = gSokolError;
            return result;
        }

        result.available = true;
        result.backend = "metal";
        result.device = deviceName (gMetalDevice);

        result.compute = sg_query_features().compute;
        if (! result.compute || sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            result.error = "sokol_gfx did not select a compute-capable macOS Metal backend";
            return result;
        }

        const std::array<uint32_t, 4> initial = { 1u, 2u, 3u, 4u };
        sg_buffer_desc bufferDesc = {};
        bufferDesc.usage.storage_buffer = true;
        bufferDesc.data.ptr = initial.data();
        bufferDesc.data.size = sizeof (initial);
        bufferDesc.label = "arbit-metal-selftest-buffer";
        const sg_buffer buffer = sg_make_buffer (&bufferDesc);

        sg_view_desc storageViewDesc = {};
        storageViewDesc.storage_buffer.buffer = buffer;
        const sg_view storageView = sg_make_view (&storageViewDesc);

        static const char* computeSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Values { uint value[4]; };
kernel void _main(device Values& values [[buffer(8)]],
                  uint index [[thread_position_in_grid]])
{
    if (index < 4) values.value[index] = values.value[index] * 3u + 7u;
}
)metal";

        sg_shader_desc computeShaderDesc = {};
        computeShaderDesc.compute_func.source = computeSource;
        computeShaderDesc.mtl_threads_per_threadgroup.x = 4;
        computeShaderDesc.mtl_threads_per_threadgroup.y = 1;
        computeShaderDesc.mtl_threads_per_threadgroup.z = 1;
        computeShaderDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeShaderDesc.views[0].storage_buffer.readonly = false;
        computeShaderDesc.views[0].storage_buffer.msl_buffer_n = 8;
        computeShaderDesc.label = "arbit-metal-selftest-compute-shader";
        const sg_shader computeShader = sg_make_shader (&computeShaderDesc);

        sg_pipeline_desc computePipelineDesc = {};
        computePipelineDesc.compute = true;
        computePipelineDesc.shader = computeShader;
        computePipelineDesc.label = "arbit-metal-selftest-compute-pipeline";
        const sg_pipeline computePipeline = sg_make_pipeline (&computePipelineDesc);

        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = 4;
        imageDesc.height = 4;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA8;
        imageDesc.sample_count = 1;
        imageDesc.label = "arbit-metal-selftest-image";
        const sg_image image = sg_make_image (&imageDesc);

        sg_view_desc colorViewDesc = {};
        colorViewDesc.color_attachment.image = image;
        const sg_view colorView = sg_make_view (&colorViewDesc);

        static const char* vertexSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut _main(uint vertexId [[vertex_id]])
{
    const float2 positions[3] = { float2(-1.0, -1.0),
                                  float2( 3.0, -1.0),
                                  float2(-1.0,  3.0) };
    VertexOut out;
    out.position = float4(positions[vertexId], 0.0, 1.0);
    return out;
}
)metal";
        static const char* fragmentSource = R"metal(
#include <metal_stdlib>
using namespace metal;
fragment float4 _main() { return float4(0.25, 0.5, 0.75, 1.0); }
)metal";

        sg_shader_desc renderShaderDesc = {};
        renderShaderDesc.vertex_func.source = vertexSource;
        renderShaderDesc.fragment_func.source = fragmentSource;
        renderShaderDesc.label = "arbit-metal-selftest-render-shader";
        const sg_shader renderShader = sg_make_shader (&renderShaderDesc);

        sg_pipeline_desc renderPipelineDesc = {};
        renderPipelineDesc.shader = renderShader;
        renderPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
        renderPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        renderPipelineDesc.label = "arbit-metal-selftest-render-pipeline";
        const sg_pipeline renderPipeline = sg_make_pipeline (&renderPipelineDesc);

        const bool resourcesOk = resourceValid (sg_query_buffer_state (buffer))
            && resourceValid (sg_query_view_state (storageView))
            && resourceValid (sg_query_shader_state (computeShader))
            && resourceValid (sg_query_pipeline_state (computePipeline))
            && resourceValid (sg_query_image_state (image))
            && resourceValid (sg_query_view_state (colorView))
            && resourceValid (sg_query_shader_state (renderShader))
            && resourceValid (sg_query_pipeline_state (renderPipeline));
        if (! resourcesOk)
        {
            result.error = "Metal self-test resource creation failed";
            sg_destroy_pipeline (renderPipeline);
            sg_destroy_shader (renderShader);
            sg_destroy_view (colorView);
            sg_destroy_image (image);
            sg_destroy_pipeline (computePipeline);
            sg_destroy_shader (computeShader);
            sg_destroy_view (storageView);
            sg_destroy_buffer (buffer);
            return result;
        }

        sg_pass computePass = {};
        computePass.compute = true;
        computePass.label = "arbit-metal-selftest-compute-pass";
        sg_begin_pass (&computePass);
        sg_apply_pipeline (computePipeline);
        sg_bindings computeBindings = {};
        computeBindings.views[0] = storageView;
        sg_apply_bindings (&computeBindings);
        sg_dispatch (1, 1, 1);
        sg_end_pass();

        sg_pass renderPass = {};
        renderPass.attachments.colors[0] = colorView;
        renderPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
        renderPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        renderPass.action.colors[0].clear_value = { 0.0f, 0.0f, 0.0f, 1.0f };
        renderPass.label = "arbit-metal-selftest-render-pass";
        sg_begin_pass (&renderPass);
        sg_apply_pipeline (renderPipeline);
        sg_draw (0, 3, 1);
        sg_end_pass();
        sg_commit();

        const sg_mtl_buffer_info nativeBuffer = sg_mtl_query_buffer_info (buffer);
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>) nativeBuffer.buf[nativeBuffer.active_slot];
        std::array<uint32_t, 4> computed = {};
        const sg_mtl_image_info nativeImage = sg_mtl_query_image_info (image);
        id<MTLTexture> mtlImage = (__bridge id<MTLTexture>) nativeImage.tex[nativeImage.active_slot];
        std::array<uint8_t, 4 * 4 * 4> pixels = {};
        readBackSubmittedWork (gMetalDevice, mtlBuffer, mtlImage, computed, pixels);

        const std::array<uint32_t, 4> expected = { 10u, 13u, 16u, 19u };
        result.computeChecksum = fnv1a (computed.data(), sizeof (computed));
        result.computePassed = computed == expected;
        result.renderChecksum = fnv1a (pixels.data(), pixels.size());
        result.renderPassed = pixels[0] >= 63 && pixels[0] <= 64
            && pixels[1] >= 127 && pixels[1] <= 128
            && pixels[2] >= 191 && pixels[2] <= 192
            && pixels[3] == 255;

        sg_destroy_pipeline (renderPipeline);
        sg_destroy_shader (renderShader);
        sg_destroy_view (colorView);
        sg_destroy_image (image);
        sg_destroy_pipeline (computePipeline);
        sg_destroy_shader (computeShader);
        sg_destroy_view (storageView);
        sg_destroy_buffer (buffer);

        if (! result.computePassed)
            result.error = "Metal compute result mismatch";
        else if (! result.renderPassed)
            result.error = "Metal offscreen render result mismatch";
    }

    return result;
}

} // namespace arbitgpu

#if ARBIT_HAVE_VIEWPORT
namespace
{

void iosurfaceSetInt (CFMutableDictionaryRef dict, CFStringRef key, int32_t value)
{
    CFNumberRef number = CFNumberCreate (kCFAllocatorDefault, kCFNumberSInt32Type, &value);
    CFDictionarySetValue (dict, key, number);
    CFRelease (number);
}

struct MetalComputeParams
{
    int32_t count;
    int32_t spawnTrack;
    float gravity;
    float force;
    float dt;
    int32_t frame;
    int32_t noteCount;
    float aspect;
    float lifetime;
    float padding[3];
};
static_assert (sizeof (MetalComputeParams) == 48, "MSL compute uniform layout changed");

struct MetalDrawParams
{
    float pointSize;
    float padding[3];
    float color[4];
};
static_assert (sizeof (MetalDrawParams) == 32, "MSL draw uniform layout changed");

struct MetalParticle
{
    float pos[2];
    float vel[2];
    float life;
    float maxLife;
    float hue;
    float padding;
};
static_assert (sizeof (MetalParticle) == 32, "MSL particle layout changed");

const char* kMetalParticleCompute = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Params {
    int count; int spawnTrack; float gravity; float force;
    float dt; int frame; int noteCount; float aspect;
    // Scalar padding keeps this MSL block byte-identical to the 48-byte C++ wire
    // block. float3 would align to 16 and silently make Params 64 bytes.
    float lifetime; float padding0; float padding1; float padding2;
};
struct Particle {
    float2 pos; float2 vel; float life; float maxLife; float hue; float pad;
};
float hash11(uint n) {
    n = (n << 13u) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    return float(n & 0x7fffffffu) / float(0x7fffffffu);
}
kernel void _main(constant Params& u [[buffer(0)]],
                  device Particle* particles [[buffer(8)]],
                  const device float4* notes [[buffer(9)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= uint(u.count)) return;
    Particle p = particles[i];
    p.life -= u.dt / max(p.maxLife, 1.0e-3f);
    if (p.life <= 0.0f) {
        int matches = 0;
        const int rows = min(u.noteCount, 128);
        for (int row = 0; row < rows; ++row) {
            const float4 t0 = notes[row * 4];
            const float4 t1 = notes[row * 4 + 1];
            if (int(t1.z + 0.5f) == u.spawnTrack && t0.y > 0.001f) ++matches;
        }
        if (matches > 0) {
            int pick = min(int(hash11(i * 747u + uint(u.frame) * 13u) * float(matches)),
                           matches - 1);
            int seen = 0;
            int chosen = -1;
            for (int row = 0; row < rows; ++row) {
                const float4 t0 = notes[row * 4];
                const float4 t1 = notes[row * 4 + 1];
                if (int(t1.z + 0.5f) == u.spawnTrack && t0.y > 0.001f) {
                    if (seen == pick) { chosen = row; break; }
                    ++seen;
                }
            }
            if (chosen >= 0) {
                const float4 t0 = notes[chosen * 4];
                const float midi = t0.x;
                const float velocity = t0.y;
                const float px = clamp((midi - 36.0f) / 60.0f, 0.0f, 1.0f) * 0.8f + 0.1f;
                p.pos = float2(px, 0.12f);
                const float angle = (hash11(i * 31u + uint(u.frame)) - 0.5f) * 2.2f;
                const float speed = (0.25f + velocity * 0.75f) * u.force;
                p.vel = float2(sin(angle) * speed / max(u.aspect, 1.0e-3f), cos(angle) * speed);
                p.maxLife = u.lifetime > 0.0f ? u.lifetime : 0.6f + hash11(i * 97u) * 1.2f;
                p.life = 1.0f;
                p.hue = fract(midi / 12.0f);
            } else {
                p.life = 0.0f; p.pos = float2(-10.0f);
            }
        } else {
            p.life = 0.0f; p.pos = float2(-10.0f);
        }
    } else {
        p.vel.y -= u.gravity * u.dt;
        p.pos += p.vel * u.dt;
    }
    particles[i] = p;
}
)metal";

const char* kMetalParticleVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Particle {
    float2 pos; float2 vel; float life; float maxLife; float hue; float pad;
};
struct DrawParams {
    float pointSize; float padding0; float padding1; float padding2; float4 color;
};
struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float life [[user(locn0)]];
    float hue [[user(locn1)]];
    float4 color [[user(locn2)]];
};
vertex VertexOut _main(const device Particle* particles [[buffer(8)]],
                       constant DrawParams& u [[buffer(0)]],
                       uint index [[vertex_id]])
{
    const Particle p = particles[index];
    VertexOut out;
    out.life = clamp(p.life, 0.0f, 1.0f);
    out.hue = p.hue;
    out.color = u.color;
    if (p.life <= 0.0f) {
        out.position = float4(-2.0f, -2.0f, 0.0f, 1.0f);
        out.pointSize = 1.0f;
    } else {
        out.position = float4(p.pos * 2.0f - 1.0f, 0.0f, 1.0f);
        out.pointSize = max(1.0f, u.pointSize * (0.5f + out.life * 0.8f));
    }
    return out;
}
)metal";

const char* kMetalParticleFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
    float life [[user(locn0)]];
    float hue [[user(locn1)]];
    float4 color [[user(locn2)]];
};
float3 hsv2rgb(float h) {
    const float3 c = abs(fract(h + float3(0.0f, 0.6667f, 0.3333f)) * 6.0f - 3.0f) - 1.0f;
    return clamp(c, 0.0f, 1.0f);
}
fragment float4 _main(VertexOut in [[stage_in]], float2 pointCoord [[point_coord]])
{
    const float radius = length(pointCoord - float2(0.5f));
    if (radius > 0.5f) discard_fragment();
    const float alpha = (1.0f - smoothstep(0.0f, 0.5f, radius)) * in.life;
    const float3 rgb = in.color.r >= 0.0f ? in.color.rgb : hsv2rgb(in.hue);
    return float4(rgb, alpha * in.color.a);
}
)metal";

} // namespace

namespace videorender
{

struct MetalParticleEngine::Impl
{
    std::string error;
    bool programsReady = false;
    int poolCount = 0;
    int outWidth = 0;
    int outHeight = 0;
    bool simSeeded = false;
    int lastSimFrame = 0;

    sg_buffer particleBuffer = {};
    sg_view particleView = {};
    sg_buffer notesBuffer = {};
    sg_view notesView = {};
    sg_shader computeShader = {};
    sg_pipeline computePipeline = {};
    sg_shader drawShader = {};
    sg_pipeline drawPipeline = {};
    sg_image outputImage = {};
    sg_view outputView = {};
    sg_view outputTextureView = {};
    bool nativeTarget = false;
    ParticleDiagnostics diagnostics;

    IOSurfaceRef surface = nullptr;
    id<MTLTexture> metalTexture = nil;
    unsigned rectangleTexture = 0;
    unsigned rectangleFbo = 0;
    unsigned outputTexture = 0;
    unsigned outputFbo = 0;

    bool ensurePrograms()
    {
        if (programsReady)
            return true;

        sg_shader_desc computeDesc = {};
        computeDesc.compute_func.source = kMetalParticleCompute;
        computeDesc.mtl_threads_per_threadgroup = { 256, 1, 1 };
        computeDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.uniform_blocks[0].size = sizeof (MetalComputeParams);
        computeDesc.uniform_blocks[0].msl_buffer_n = 0;
        computeDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.views[0].storage_buffer.readonly = false;
        computeDesc.views[0].storage_buffer.msl_buffer_n = 8;
        computeDesc.views[1].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.views[1].storage_buffer.readonly = true;
        computeDesc.views[1].storage_buffer.msl_buffer_n = 9;
        computeDesc.label = "arbit-metal-particle-compute";
        computeShader = sg_make_shader (&computeDesc);

        sg_pipeline_desc computePipelineDesc = {};
        computePipelineDesc.compute = true;
        computePipelineDesc.shader = computeShader;
        computePipelineDesc.label = "arbit-metal-particle-compute-pipeline";
        computePipeline = sg_make_pipeline (&computePipelineDesc);

        sg_shader_desc drawDesc = {};
        drawDesc.vertex_func.source = kMetalParticleVertex;
        drawDesc.fragment_func.source = kMetalParticleFragment;
        drawDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        drawDesc.uniform_blocks[0].size = sizeof (MetalDrawParams);
        drawDesc.uniform_blocks[0].msl_buffer_n = 0;
        drawDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_VERTEX;
        drawDesc.views[0].storage_buffer.readonly = true;
        drawDesc.views[0].storage_buffer.msl_buffer_n = 8;
        drawDesc.label = "arbit-metal-particle-draw";
        drawShader = sg_make_shader (&drawDesc);

        sg_pipeline_desc drawPipelineDesc = {};
        drawPipelineDesc.shader = drawShader;
        drawPipelineDesc.primitive_type = SG_PRIMITIVETYPE_POINTS;
        drawPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        drawPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        drawPipelineDesc.colors[0].blend.enabled = true;
        drawPipelineDesc.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        drawPipelineDesc.label = "arbit-metal-particle-draw-pipeline";
        drawPipeline = sg_make_pipeline (&drawPipelineDesc);

        sg_buffer_desc notesDesc = {};
        notesDesc.size = 4u * 128u * 4u * sizeof (float);
        notesDesc.usage.storage_buffer = true;
        notesDesc.usage.dynamic_update = true;
        notesDesc.label = "arbit-metal-particle-notes";
        notesBuffer = sg_make_buffer (&notesDesc);
        sg_view_desc notesViewDesc = {};
        notesViewDesc.storage_buffer.buffer = notesBuffer;
        notesView = sg_make_view (&notesViewDesc);

        programsReady = resourceValid (sg_query_shader_state (computeShader))
            && resourceValid (sg_query_pipeline_state (computePipeline))
            && resourceValid (sg_query_shader_state (drawShader))
            && resourceValid (sg_query_pipeline_state (drawPipeline))
            && resourceValid (sg_query_buffer_state (notesBuffer))
            && resourceValid (sg_query_view_state (notesView));
        if (! programsReady)
            error = "Metal particle shader/pipeline creation failed";
        return programsReady;
    }

    bool ensurePool (int count)
    {
        if (particleBuffer.id != 0 && poolCount == count)
            return true;
        if (particleView.id != 0) sg_destroy_view (particleView);
        if (particleBuffer.id != 0) sg_destroy_buffer (particleBuffer);
        particleView = {};
        particleBuffer = {};
        poolCount = count;
        simSeeded = false;
        lastSimFrame = 0;

        const std::vector<MetalParticle> initial (static_cast<size_t> (count));
        sg_buffer_desc desc = {};
        desc.usage.storage_buffer = true;
        desc.data.ptr = initial.data();
        desc.data.size = initial.size() * sizeof (MetalParticle);
        desc.label = "arbit-metal-particle-pool";
        particleBuffer = sg_make_buffer (&desc);
        sg_view_desc viewDesc = {};
        viewDesc.storage_buffer.buffer = particleBuffer;
        particleView = sg_make_view (&viewDesc);
        if (! resourceValid (sg_query_buffer_state (particleBuffer))
            || ! resourceValid (sg_query_view_state (particleView)))
        {
            error = "Metal particle storage-buffer creation failed";
            return false;
        }
        return true;
    }

    void destroyTarget (const arbitgl::GlFuncs* gl)
    {
        if (outputTextureView.id != 0) sg_destroy_view (outputTextureView);
        if (outputView.id != 0) sg_destroy_view (outputView);
        if (outputImage.id != 0) sg_destroy_image (outputImage);
        outputView = {};
        outputTextureView = {};
        outputImage = {};
        if (rectangleFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &rectangleFbo);
        if (outputFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &outputFbo);
        if (rectangleTexture != 0) glDeleteTextures (1, &rectangleTexture);
        if (outputTexture != 0) glDeleteTextures (1, &outputTexture);
        rectangleFbo = outputFbo = rectangleTexture = outputTexture = 0;
#if ! __has_feature(objc_arc)
        [metalTexture release];
#endif
        metalTexture = nil;
        if (surface != nullptr) CFRelease (surface);
        surface = nullptr;
        nativeTarget = false;
        outWidth = outHeight = 0;
    }

    bool ensureTarget (const arbitgl::GlFuncs* gl, int width, int height,
                       bool nativeOnly)
    {
        if (outputImage.id != 0 && outWidth == width && outHeight == height
            && nativeTarget == nativeOnly)
            return true;
        destroyTarget (gl);

        if (nativeOnly)
        {
            sg_image_desc imageDesc = {};
            imageDesc.usage.color_attachment = true;
            imageDesc.width = width;
            imageDesc.height = height;
            imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
            imageDesc.sample_count = 1;
            imageDesc.label = "arbit-metal-particle-native-output";
            outputImage = sg_make_image (&imageDesc);
            sg_view_desc attachmentDesc = {};
            attachmentDesc.color_attachment.image = outputImage;
            outputView = sg_make_view (&attachmentDesc);
            sg_view_desc textureDesc = {};
            textureDesc.texture.image = outputImage;
            outputTextureView = sg_make_view (&textureDesc);
            if (! resourceValid (sg_query_image_state (outputImage))
                || ! resourceValid (sg_query_view_state (outputView))
                || ! resourceValid (sg_query_view_state (outputTextureView)))
            {
                error = "Metal particle native target creation failed";
                destroyTarget (gl);
                return false;
            }
            nativeTarget = true;
            outWidth = width;
            outHeight = height;
            return true;
        }

        CFMutableDictionaryRef props = CFDictionaryCreateMutable (
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        iosurfaceSetInt (props, kIOSurfaceWidth, width);
        iosurfaceSetInt (props, kIOSurfaceHeight, height);
        iosurfaceSetInt (props, kIOSurfaceBytesPerElement, 4);
        iosurfaceSetInt (props, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
        surface = IOSurfaceCreate (props);
        CFRelease (props);
        if (surface == nullptr)
        {
            error = "Metal particle IOSurface creation failed";
            return false;
        }

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor iosurface:surface plane:0];
        if (metalTexture == nil)
        {
            error = "Metal particle IOSurface texture creation failed";
            destroyTarget (gl);
            return false;
        }

        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = width;
        imageDesc.height = height;
        imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        imageDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        // sokol's Metal resource-pool insertion balances one retain after it
        // stores an injected texture. Preserve this owner's +1 from
        // newTextureWithDescriptor so destroyTarget can release it safely.
        [metalTexture retain];
#endif
        imageDesc.mtl_textures[0] = (__bridge const void*) metalTexture;
        imageDesc.label = "arbit-metal-particle-output";
        outputImage = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.color_attachment.image = outputImage;
        outputView = sg_make_view (&viewDesc);
        sg_view_desc textureViewDesc = {};
        textureViewDesc.texture.image = outputImage;
        outputTextureView = sg_make_view (&textureViewDesc);
        if (! resourceValid (sg_query_image_state (outputImage))
            || ! resourceValid (sg_query_view_state (outputView))
            || ! resourceValid (sg_query_view_state (outputTextureView)))
        {
            error = "Metal particle output attachment creation failed";
            destroyTarget (gl);
            return false;
        }

        CGLContextObj cgl = CGLGetCurrentContext();
        if (cgl == nullptr)
        {
            error = "Metal particle bridge has no current CGL context";
            destroyTarget (gl);
            return false;
        }
        glGenTextures (1, &rectangleTexture);
        glBindTexture (GL_TEXTURE_RECTANGLE, rectangleTexture);
        const CGLError cglError = CGLTexImageIOSurface2D (
            cgl, GL_TEXTURE_RECTANGLE, GL_RGBA8, width, height,
            GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
        if (cglError != kCGLNoError)
        {
            error = std::string ("Metal particle CGL IOSurface import failed: ")
                  + CGLErrorString (cglError);
            glBindTexture (GL_TEXTURE_RECTANGLE, 0);
            destroyTarget (gl);
            return false;
        }
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture (GL_TEXTURE_RECTANGLE, 0);

        gl->GenFramebuffers (1, &rectangleFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, rectangleFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_RECTANGLE, rectangleTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            error = "Metal particle IOSurface GL framebuffer incomplete";
            gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
            destroyTarget (gl);
            return false;
        }

        glGenTextures (1, &outputTexture);
        glBindTexture (GL_TEXTURE_2D, outputTexture);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl->GenFramebuffers (1, &outputFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, outputFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, outputTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            error = "Metal particle compositor framebuffer incomplete";
            gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
            destroyTarget (gl);
            return false;
        }
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        outWidth = width;
        outHeight = height;
        nativeTarget = false;
        return true;
    }

    int planSteps (const ShaderClock& clock, int& firstFrame)
    {
        if (! clock.playing) { firstFrame = clock.frame; return 0; }
        int steps = 0;
        if (! simSeeded)
        {
            if (clock.frame < 0 || clock.frame >= 10000) return -1;
            steps = clock.frame + 1;
            firstFrame = 0;
        }
        else if (clock.frame == lastSimFrame) { steps = 0; firstFrame = clock.frame; }
        else if (clock.frame < lastSimFrame) return -1;
        else
        {
            steps = clock.frame - lastSimFrame;
            if (steps > 10000) return -1;
            firstFrame = lastSimFrame + 1;
        }
        simSeeded = true;
        lastSimFrame = clock.frame;
        return steps;
    }
};

MetalParticleEngine::MetalParticleEngine() : impl_ (std::make_unique<Impl>()) {}
MetalParticleEngine::~MetalParticleEngine() = default;

bool MetalParticleEngine::enabled()
{
    const char* value = std::getenv ("ARBIT_VIDEO_METAL");
    // Physical Apple-silicon validation covers native compute, offscreen
    // rendering, and the IOSurface bridge. Keep an explicit recovery switch,
    // but use the native path by default on Apple builds.
    return value == nullptr || std::strcmp (value, "0") != 0;
}

uint32_t MetalParticleEngine::renderViewUnlocked (
    const arbitgl::GlFuncs* gl, const ShaderClock& clock,
    int width, int height, const ParticleParams& params,
    const NoteFeatures* notes, bool nativeOnly)
{
    if (width <= 0 || height <= 0)
        return 0;
    if (! ensureSokolMetal())
    {
        impl_->error = gSokolError;
        return 0;
    }
    if (! sg_query_features().compute || ! impl_->ensurePrograms())
        return 0;

    int count = params.count;
    if (count < 1) count = 1;
    if (count > ParticleEngine::kMaxParticles) count = ParticleEngine::kMaxParticles;
    if (impl_->simSeeded && clock.frame < impl_->lastSimFrame)
    {
        if (impl_->particleView.id != 0) sg_destroy_view (impl_->particleView);
        if (impl_->particleBuffer.id != 0) sg_destroy_buffer (impl_->particleBuffer);
        impl_->particleView = {};
        impl_->particleBuffer = {};
        impl_->poolCount = 0;
        impl_->simSeeded = false;
        impl_->lastSimFrame = 0;
    }
    if (! impl_->ensurePool (count)
        || ! impl_->ensureTarget (gl, width, height, nativeOnly))
        return 0;

    std::array<float, 4 * 128 * 4> noteData = {};
    int noteCount = 0;
    if (notes != nullptr && notes->notesTex.size() >= noteData.size())
    {
        std::copy_n (notes->notesTex.data(), noteData.size(), noteData.data());
        noteCount = notes->noteCount;
    }
    impl_->diagnostics = {};
    impl_->diagnostics.noteRows = noteCount;
    const sg_range noteRange = { noteData.data(), sizeof (noteData) };
    sg_update_buffer (impl_->notesBuffer, &noteRange);

    int firstFrame = clock.frame;
    const int steps = impl_->planSteps (clock, firstFrame);
    if (steps < 0)
    {
        impl_->error = "Metal particle replay exceeds deterministic bound";
        return 0;
    }
    sg_pass computePass = {};
    computePass.compute = true;
    computePass.label = "arbit-metal-particle-compute-pass";
    sg_begin_pass (&computePass);
    sg_apply_pipeline (impl_->computePipeline);
    sg_bindings computeBindings = {};
    computeBindings.views[0] = impl_->particleView;
    computeBindings.views[1] = impl_->notesView;
    sg_apply_bindings (&computeBindings);
    for (int step = 0; step < steps; ++step)
    {
        const MetalComputeParams uniforms = {
            count,
            params.spawnTrack,
            params.gravity,
            params.force > 0.0f ? params.force : 0.0f,
            clock.playing ? static_cast<float> (clock.timeDelta) : 0.0f,
            firstFrame + step + params.seed,
            noteCount,
            height > 0 ? static_cast<float> (width) / static_cast<float> (height) : 1.0f,
            params.lifetime,
            { 0.0f, 0.0f, 0.0f },
        };
        const sg_range range = { &uniforms, sizeof (uniforms) };
        sg_apply_uniforms (0, &range);
        sg_dispatch ((count + 255) / 256, 1, 1);
    }
    sg_end_pass();

    sg_pass drawPass = {};
    drawPass.attachments.colors[0] = impl_->outputView;
    drawPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
    drawPass.action.colors[0].store_action = SG_STOREACTION_STORE;
    drawPass.action.colors[0].clear_value = { 0.0f, 0.0f, 0.0f, 0.0f };
    drawPass.label = "arbit-metal-particle-draw-pass";
    sg_begin_pass (&drawPass);
    sg_apply_pipeline (impl_->drawPipeline);
    sg_bindings drawBindings = {};
    drawBindings.views[0] = impl_->particleView;
    sg_apply_bindings (&drawBindings);
    const MetalDrawParams drawParams = {
        params.size > 0.0f ? params.size : 1.0f,
        { 0.0f, 0.0f, 0.0f },
        { params.red, params.green, params.blue, params.alpha },
    };
    const sg_range drawRange = { &drawParams, sizeof (drawParams) };
    sg_apply_uniforms (0, &drawRange);
    sg_draw (0, count, 1);
    sg_end_pass();
    impl_->error.clear();
    return impl_->outputTextureView.id;
}

uint32_t MetalParticleEngine::renderMetalViewUnlocked (
    const ShaderClock& clock, int width, int height,
    const ParticleParams& params, const NoteFeatures* notes)
{
    return renderViewUnlocked (nullptr, clock, width, height, params, notes, true);
}

unsigned MetalParticleEngine::render (const arbitgl::GlFuncs* gl,
                                      const ShaderClock& clock,
                                      int width, int height,
                                      const ParticleParams& params,
                                      const NoteFeatures* notes)
{
    if (gl == nullptr)
        return 0;
    std::lock_guard<std::mutex> lock (sokolMutex());
    if (renderViewUnlocked (gl, clock, width, height, params, notes, false) == 0)
        return 0;
    sg_commit();

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    const char* diagnostic = std::getenv ("ARBIT_PARTICLE_DIAGNOSTICS");
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        const sg_mtl_buffer_info info = sg_mtl_query_buffer_info (impl_->particleBuffer);
        id<MTLBuffer> source = (__bridge id<MTLBuffer>) info.buf[info.active_slot];
        const NSUInteger byteCount = static_cast<NSUInteger> (impl_->poolCount)
                                   * sizeof (MetalParticle);
        id<MTLBuffer> copy = [gMetalDevice newBufferWithLength:byteCount
                                                       options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder copyFromBuffer:source sourceOffset:0 toBuffer:copy destinationOffset:0
                           size:byteCount];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto* particles = static_cast<const MetalParticle*> ([copy contents]);
        impl_->diagnostics.liveParticles = 0;
        impl_->diagnostics.visibleCandidates = 0;
        for (int i = 0; i < impl_->poolCount; ++i)
        {
            if (particles[i].life > 0.0f)
            {
                ++impl_->diagnostics.liveParticles;
                if (std::isfinite (particles[i].pos[0]) && std::isfinite (particles[i].pos[1])
                    && particles[i].pos[0] >= 0.0f && particles[i].pos[0] <= 1.0f
                    && particles[i].pos[1] >= 0.0f && particles[i].pos[1] <= 1.0f)
                    ++impl_->diagnostics.visibleCandidates;
            }
        }
#if ! __has_feature(objc_arc)
        [copy release];
#endif
        std::vector<unsigned char> pixels ((size_t) width * (size_t) height * 4);
        [impl_->metalTexture getBytes:pixels.data() bytesPerRow:(NSUInteger) width * 4
                           fromRegion:MTLRegionMake2D (0, 0, width, height) mipmapLevel:0];
        impl_->diagnostics.drawAlphaPixels = 0;
        for (size_t i = 3; i < pixels.size(); i += 4)
            if (pixels[i] != 0) ++impl_->diagnostics.drawAlphaPixels;
    }

    gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->rectangleFbo);
    gl->BindFramebuffer (GL_DRAW_FRAMEBUFFER, impl_->outputFbo);
    gl->BlitFramebuffer (0, 0, width, height, 0, 0, width, height,
                         GL_COLOR_BUFFER_BIT, GL_NEAREST);
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        std::vector<unsigned char> pixels ((size_t) width * (size_t) height * 4);
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->outputFbo);
        glReadBuffer (GL_COLOR_ATTACHMENT0);
        glReadPixels (0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        impl_->diagnostics.readbackAlphaPixels = 0;
        for (size_t i = 3; i < pixels.size(); i += 4)
            if (pixels[i] != 0) ++impl_->diagnostics.readbackAlphaPixels;
    }
    gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return impl_->outputTexture;
}

void MetalParticleEngine::shutdown (const arbitgl::GlFuncs* gl)
{
    if (impl_ == nullptr)
        return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    shutdownUnlocked (gl);
}

void MetalParticleEngine::shutdownUnlocked (const arbitgl::GlFuncs* gl)
{
    impl_->destroyTarget (gl);
    if (impl_->particleView.id != 0) sg_destroy_view (impl_->particleView);
    if (impl_->particleBuffer.id != 0) sg_destroy_buffer (impl_->particleBuffer);
    if (impl_->notesView.id != 0) sg_destroy_view (impl_->notesView);
    if (impl_->notesBuffer.id != 0) sg_destroy_buffer (impl_->notesBuffer);
    if (impl_->drawPipeline.id != 0) sg_destroy_pipeline (impl_->drawPipeline);
    if (impl_->drawShader.id != 0) sg_destroy_shader (impl_->drawShader);
    if (impl_->computePipeline.id != 0) sg_destroy_pipeline (impl_->computePipeline);
    if (impl_->computeShader.id != 0) sg_destroy_shader (impl_->computeShader);
    impl_->particleView = {};
    impl_->particleBuffer = {};
    impl_->notesView = {};
    impl_->notesBuffer = {};
    impl_->drawPipeline = {};
    impl_->drawShader = {};
    impl_->computePipeline = {};
    impl_->computeShader = {};
    impl_->programsReady = false;
}

const std::string& MetalParticleEngine::log() const
{
    return impl_->error;
}

const ParticleDiagnostics& MetalParticleEngine::diagnostics() const
{
    return impl_->diagnostics;
}

} // namespace videorender
#endif

#if ARBIT_HAVE_VIEWPORT
namespace videorender
{

namespace
{

struct MetalGeometryParams
{
    float transform[16];
    float crop[4];
};

struct MetalMaskParams
{
    float maskRect[4];
    float opacity;
    int32_t maskType;
    float maskFeather;
    int32_t maskInvert;
    float matteRefine[4];
    float matteTexel[2];
    float matteChoke;
    int32_t matteFlags;
};

struct MetalBlendParams
{
    float opacity;
    int32_t blendMode;
    float padding[2];
};

struct MetalFrameMixParams
{
    float mix;
    float padding[3];
};

struct MetalDepthFogParams
{
    float rangeDensity[4];
    float color[4];
};

struct MetalTransitionParams
{
    float progress;
    int32_t transitionType;
    int32_t blendMode;
    float padding;
};

struct MetalFilterParams
{
    float texelX;
    float texelY;
    float amount;
    float padding;
};

struct MetalUvEffectParams
{
    float resolutionX;
    float resolutionY;
    float time;
    int32_t mode;
    float values[4];
};

struct MetalFeedbackParams
{
    float decay;
    float zoom;
    float swirl;
    float padding;
};

struct MetalPostParams
{
    float threshold;
    float intensity;
    float exposure;
    int32_t tonemap;
};

struct MetalCanvasParams
{
    float rect[4];
    float texel[2];
    float padding[2];
};

struct MetalPreviewParams
{
    float zoom;
    float alignmentPadding;
    float pan[2];
    float split;
    int32_t layout;
    int32_t background;
    float trailingPadding;
    float padding[2];
};

struct MetalDrawShapeParams
{
    float rect[4];
    float color[4];
};

enum MetalFxValue
{
    MfxBrightness = 0, MfxContrast, MfxSaturation, MfxHueShift, MfxExposure,
    MfxGamma, MfxVignetteAmount, MfxVignetteSoftness, MfxWarmth, MfxCoolness,
    MfxVintage, MfxSepia, MfxBw, MfxInvert, MfxPosterize, MfxNoise,
    MfxKeyR, MfxKeyG, MfxKeyB, MfxKeyTolerance, MfxKeySoftness, MfxKeySpill,
    MfxLumaLow, MfxLumaHigh, MfxLumaSoftness, MfxLumaInvert,
    MfxLiftR, MfxLiftG, MfxLiftB, MfxGammaR, MfxGammaG, MfxGammaB,
    MfxGainR, MfxGainG, MfxGainB,
    MfxCount
};

struct MetalEffectParams
{
    int32_t mask = 0;
    float time = 0.0f;
    float lutEnabled = 0.0f;
    float lutSize = 0.0f;
    float values[MfxCount] = {};
    float padding = 0.0f;
};

static_assert (sizeof (MetalGeometryParams) == 80, "Metal geometry uniforms changed");
static_assert (sizeof (MetalMaskParams) == 64, "Metal mask uniforms changed");
static_assert (sizeof (MetalBlendParams) == 16, "Metal blend uniforms changed");
static_assert (sizeof (MetalFrameMixParams) == 16, "Metal frame-mix uniforms changed");
static_assert (sizeof (MetalDepthFogParams) == 32, "Metal depth-fog uniforms changed");
static_assert (sizeof (MetalTransitionParams) == 16, "Metal transition uniforms changed");
static_assert (sizeof (MetalFilterParams) == 16, "Metal filter uniforms changed");
static_assert (sizeof (MetalUvEffectParams) == 32, "Metal UV effect uniforms changed");
static_assert (sizeof (MetalFeedbackParams) == 16, "Metal feedback uniforms changed");
static_assert (sizeof (MetalPostParams) == 16, "Metal post uniforms changed");
static_assert (sizeof (MetalCanvasParams) == 32, "Metal canvas uniforms changed");
static_assert (sizeof (MetalDrawShapeParams) == 32, "Metal draw-shape uniforms changed");
static_assert (sizeof (MetalEffectParams) == 160, "Metal effect uniforms changed");

const char* kMetalLayerVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Geometry { float4x4 transform; float4 crop; };
struct Out {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
    float2 rawUV [[user(locn1)]];
};
vertex Out _main(uint vertexId [[vertex_id]], constant Geometry& g [[buffer(0)]])
{
    const float2 pos[6] = { float2(-1,-1), float2(1,-1), float2(1,1),
                            float2(-1,-1), float2(1,1), float2(-1,1) };
    const float2 uv[6] = { float2(0,0), float2(1,0), float2(1,1),
                           float2(0,0), float2(1,1), float2(0,1) };
    Out out;
    out.position = g.transform * float4(pos[vertexId], 0, 1);
    out.rawUV = uv[vertexId];
    out.uv = mix(float2(g.crop.x, g.crop.z),
                 float2(1.0f - g.crop.y, 1.0f - g.crop.w), uv[vertexId]);
    return out;
}
)metal";

const char* kMetalLayerFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
    float2 rawUV [[user(locn1)]];
};
struct Mask { float4 rect; float opacity; int type; float feather; int invert;
              float4 matteRefine; float2 matteTexel; float matteChoke; int matteFlags; };
struct Effects { int mask; float time; float lutEnabled; float lutSize; float values[35]; float padding; };
struct DepthFog { float4 rangeDensity; float4 color; };
float luminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
float randomValue(float2 st) {
    return fract(sin(dot(st, float2(12.9898f, 78.233f))) * 43758.5453f);
}
float hue2rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}
float3 rgb2hsl(float3 c) {
    const float maxC = max(max(c.r, c.g), c.b);
    const float minC = min(min(c.r, c.g), c.b);
    const float l = (maxC + minC) * 0.5f;
    float h = 0.0f, s = 0.0f;
    if (maxC != minC) {
        const float d = maxC - minC;
        s = l > 0.5f ? d / (2.0f - maxC - minC) : d / (maxC + minC);
        if (maxC == c.r) h = (c.g - c.b) / d + (c.g < c.b ? 6.0f : 0.0f);
        else if (maxC == c.g) h = (c.b - c.r) / d + 2.0f;
        else h = (c.r - c.g) / d + 4.0f;
        h /= 6.0f;
    }
    return float3(h, s, l);
}
float3 hsl2rgb(float3 hsl) {
    if (hsl.y == 0.0f) return float3(hsl.z);
    const float q = hsl.z < 0.5f ? hsl.z * (1.0f + hsl.y)
                                  : hsl.z + hsl.y - hsl.z * hsl.y;
    const float p = 2.0f * hsl.z - q;
    return float3(hue2rgb(p, q, hsl.x + 1.0f / 3.0f),
                  hue2rgb(p, q, hsl.x),
                  hue2rgb(p, q, hsl.x - 1.0f / 3.0f));
}
float4 applyEffects(float4 color, float2 uv, constant Effects& e) {
    float3 rgb = color.rgb;
    float alpha = color.a;
    if ((e.mask & 131072) != 0) {
        const float3 key = float3(e.values[16], e.values[17], e.values[18]);
        const float2 pixCbCr = float2(rgb.b - luminance(rgb), rgb.r - luminance(rgb));
        const float2 keyCbCr = float2(key.b - luminance(key), key.r - luminance(key));
        const float dist = length(pixCbCr - keyCbCr);
        alpha *= smoothstep(e.values[19],
                            e.values[19] + max(e.values[20], 1.0e-5f), dist);
        const float spill = e.values[21];
        if (spill > 0.0f) {
            if (key.g >= key.r && key.g >= key.b)
                rgb.g = mix(rgb.g, min(rgb.g, max(rgb.r, rgb.b)), spill);
            else if (key.b >= key.r)
                rgb.b = mix(rgb.b, min(rgb.b, max(rgb.r, rgb.g)), spill);
            else
                rgb.r = mix(rgb.r, min(rgb.r, max(rgb.g, rgb.b)), spill);
        }
    }
    if ((e.mask & 262144) != 0) {
        const float luma = luminance(rgb);
        const float softness = max(e.values[24], 1.0e-5f);
        float keep = smoothstep(e.values[22] - softness, e.values[22], luma)
                   * (1.0f - smoothstep(e.values[23], e.values[23] + softness, luma));
        if (e.values[25] >= 0.5f) keep = 1.0f - keep;
        alpha *= keep;
    }
    if ((e.mask & 16) != 0 && e.values[4] != 0.0f) rgb *= pow(2.0f, e.values[4]);
    if ((e.mask & 1) != 0 && e.values[0] != 0.0f) rgb += float3(e.values[0]);
    if ((e.mask & 2) != 0 && e.values[1] != 1.0f) rgb = (rgb - 0.5f) * e.values[1] + 0.5f;
    if ((e.mask & 32) != 0 && e.values[5] != 1.0f)
        rgb = pow(max(rgb, float3(0.0f)), float3(1.0f / e.values[5]));
    if ((e.mask & 4) != 0 && e.values[2] != 1.0f)
        rgb = mix(float3(luminance(rgb)), rgb, e.values[2]);
    if ((e.mask & 8) != 0 && e.values[3] != 0.0f) {
        float3 hsl = rgb2hsl(rgb);
        const float shifted = hsl.x + e.values[3] / 360.0f;
        hsl.x = shifted - floor(shifted);
        rgb = hsl2rgb(hsl);
    }
    if ((e.mask & 524288) != 0) {
        const float3 lift = float3(e.values[26], e.values[27], e.values[28]);
        const float3 gamma = max(float3(e.values[29], e.values[30], e.values[31]),
                                 float3(1.0e-3f));
        const float3 gain = float3(e.values[32], e.values[33], e.values[34]);
        rgb = clamp(rgb * gain + lift * (1.0f - rgb), 0.0f, 1.0f);
        rgb = pow(rgb, 1.0f / gamma);
    }
    if ((e.mask & 512) != 0 && e.values[8] > 0.0f)
        rgb = mix(rgb, rgb * float3(1.1f, 0.9f, 0.7f), e.values[8] * 0.5f);
    if ((e.mask & 1024) != 0 && e.values[9] > 0.0f)
        rgb = mix(rgb, rgb * float3(0.8f, 0.9f, 1.2f), e.values[9] * 0.5f);
    if ((e.mask & 2048) != 0 && e.values[10] > 0.0f) {
        const float luma = luminance(rgb);
        float3 vintage = mix(float3(luma), rgb, 0.7f) * float3(1.1f, 1.0f, 0.9f);
        vintage = max(vintage, float3(0.03f));
        rgb = mix(rgb, vintage, e.values[10]);
    }
    if ((e.mask & 4096) != 0 && e.values[11] > 0.0f) {
        const float3 sepia = float3(dot(rgb, float3(0.393f, 0.769f, 0.189f)),
                                    dot(rgb, float3(0.349f, 0.686f, 0.168f)),
                                    dot(rgb, float3(0.272f, 0.534f, 0.131f)));
        rgb = mix(rgb, sepia, e.values[11]);
    }
    if ((e.mask & 8192) != 0 && e.values[12] > 0.0f)
        rgb = mix(rgb, float3(luminance(rgb)), e.values[12]);
    if ((e.mask & 16384) != 0 && e.values[13] > 0.0f)
        rgb = mix(rgb, 1.0f - rgb, e.values[13]);
    if ((e.mask & 32768) != 0 && e.values[14] > 0.0f)
        rgb = floor(rgb * e.values[14]) / e.values[14];
    if ((e.mask & 65536) != 0 && e.values[15] > 0.0f) {
        const float noise = randomValue(uv + float2(e.time)) * 2.0f - 1.0f;
        rgb += float3(noise * e.values[15] * 0.2f);
    }
    if ((e.mask & 256) != 0 && e.values[6] > 0.0f) {
        const float dist = length(uv - 0.5f) * 1.4142f;
        const float vignette = 1.0f - smoothstep(1.0f - e.values[6] - e.values[7],
                                                 1.0f - e.values[6] + 0.01f, dist);
        rgb *= vignette;
    }
    return float4(clamp(rgb, 0.0f, 1.0f), alpha);
}
float coverage(float2 uv, constant Mask& m)
{
    if (m.type == 0) return 1.0f;
    const float f = max(m.feather, 1.0e-5f);
    float cov = 1.0f;
    if (m.type == 1) {
        const float2 d = abs(uv - m.rect.xy) - 0.5f * m.rect.zw;
        cov = 1.0f - smoothstep(-f, 0.0f, max(d.x, d.y));
    } else {
        const float2 r = (uv - m.rect.xy) / max(0.5f * m.rect.zw, float2(1.0e-5f));
        const float fr = f / max(0.25f * (m.rect.z + m.rect.w), 1.0e-5f);
        cov = 1.0f - smoothstep(-fr, 0.0f, length(r) - 1.0f);
    }
    return m.invert != 0 ? 1.0f - cov : cov;
}
float rawMatteCoverage(float2 uv, constant Mask& m, texture2d<float> matte,
                       texture2d<float> matteB, sampler imageSampler) {
    float value = matte.sample(imageSampler, uv).r;
    const int combineMode = (m.matteFlags >> 2) - 1;
    if (combineMode >= 0) {
        const float b = matteB.sample(imageSampler, uv).r;
        if (combineMode == 0) value = max(value, b);
        else if (combineMode == 1) value = min(value, b);
        else if (combineMode == 2) value = max(value - b, 0.0f);
        else value = abs(value - b);
    }
    return value;
}
float matteCoverage(float2 uv, constant Mask& m, texture2d<float> matte,
                    texture2d<float> matteB,
                    sampler imageSampler) {
    if ((m.matteFlags & 1) == 0) return 1.0f;
    const int radius = int(clamp(abs(m.matteRefine.z), 0.0f, 4.0f));
    float value = rawMatteCoverage(uv, m, matte, matteB, imageSampler);
    if (radius > 0) {
        float aggregate = m.matteRefine.z >= 0.0f ? 0.0f : 1.0f;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= radius && abs(y) <= radius) {
                const float sampleValue = rawMatteCoverage(
                    uv + float2(x, y) * m.matteTexel, m, matte, matteB, imageSampler);
                aggregate = m.matteRefine.z >= 0.0f ? max(aggregate, sampleValue)
                                                    : min(aggregate, sampleValue);
            }
        value = aggregate;
    }
    const int feather = int(clamp(m.matteRefine.w, 0.0f, 4.0f));
    if (feather > 0) {
        float sum = 0.0f, count = 0.0f;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= feather && abs(y) <= feather) {
                sum += rawMatteCoverage(uv + float2(x, y) * m.matteTexel,
                                        m, matte, matteB, imageSampler);
                count += 1.0f;
            }
        value = sum / max(count, 1.0f);
    }
    value = clamp((value - m.matteRefine.x)
                  / max(m.matteRefine.y - m.matteRefine.x, 1.0e-5f), 0.0f, 1.0f);
    value = clamp(value + m.matteChoke, 0.0f, 1.0f);
    return (m.matteFlags & 2) != 0 ? 1.0f - value : value;
}
fragment float4 _main(In in [[stage_in]], constant Mask& m [[buffer(0)]],
                      constant Effects& e [[buffer(1)]],
                      constant DepthFog& fog [[buffer(2)]],
                      texture2d<float> image [[texture(0)]],
                      texture3d<float> lut [[texture(1)]],
                      texture2d<float> matte [[texture(2)]],
                      texture2d<float> matteB [[texture(3)]],
                      texture2d<float> depth [[texture(4)]],
                      sampler imageSampler [[sampler(0)]])
{
    const int depthMode = int(fog.rangeDensity.w + 0.5f);
    const float depthValue = depth.sample(imageSampler, in.rawUV).r;
    float2 sampleUV = in.uv;
    if (depthMode == 3)
        sampleUV = clamp(in.uv + fog.rangeDensity.xy * (depthValue - fog.rangeDensity.z), 0.0f, 1.0f);
    float4 sampled = image.sample(imageSampler, sampleUV);
    if (depthMode == 2) {
        const float radius = clamp(abs(depthValue - fog.rangeDensity.y)
            / max(fog.rangeDensity.z, 1.0e-5f), 0.0f, 1.0f) * fog.rangeDensity.x;
        const float2 stepUV = radius / float2(image.get_width(), image.get_height());
        sampled = (sampled * 4.0f + image.sample(imageSampler, sampleUV + float2(stepUV.x, 0))
            + image.sample(imageSampler, sampleUV - float2(stepUV.x, 0))
            + image.sample(imageSampler, sampleUV + float2(0, stepUV.y))
            + image.sample(imageSampler, sampleUV - float2(0, stepUV.y))) / 8.0f;
    }
    float4 c = applyEffects(sampled, sampleUV, e);
    if (e.lutEnabled > 0.5f) {
        const float size = max(e.lutSize, 2.0f);
        const float3 uvw = clamp(c.rgb, 0.0f, 1.0f) * ((size - 1.0f) / size)
                         + 0.5f / size;
        c = float4(lut.sample(imageSampler, uvw).rgb, c.a);
    }
    if (depthMode == 1) {
        const float range = clamp((depthValue - fog.rangeDensity.x)
            / max(fog.rangeDensity.y - fog.rangeDensity.x, 1.0e-5f), 0.0f, 1.0f);
        const float amount = fog.color.a * (1.0f - exp(-fog.rangeDensity.z * range));
        c.rgb = mix(c.rgb, fog.color.rgb, amount);
    } else if (depthMode == 4) {
        c.rgb *= fog.color.rgb * max(0.0f, fog.rangeDensity.y
            + fog.rangeDensity.x * (1.0f - depthValue));
    }
    return float4(c.rgb, c.a * m.opacity * coverage(in.rawUV, m)
                             * matteCoverage(in.rawUV, m, matte, matteB, imageSampler));
}
)metal";

const char* kMetalFullscreenVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; float2 uv [[user(locn0)]]; };
vertex Out _main(uint vertexId [[vertex_id]])
{
    const float2 pos[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
    const float2 uv[3] = { float2(0,0), float2(2,0), float2(0,2) };
    Out out; out.position = float4(pos[vertexId], 0, 1); out.uv = uv[vertexId];
    return out;
}
)metal";

const char* kMetalBlendFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float opacity; int mode; float2 padding; };
float3 overlay(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  base >= 0.5f);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> frontTex [[texture(0)]],
                      texture2d<float> backTex [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 front = frontTex.sample(imageSampler, in.uv);
    const float4 back = backTex.sample(imageSampler, in.uv);
    const float alpha = front.a * p.opacity;
    float3 blended = front.rgb;
    if (p.mode == 1) blended = min(back.rgb + front.rgb, float3(1.0f));
    else if (p.mode == 2) blended = back.rgb * front.rgb;
    else if (p.mode == 3) blended = 1.0f - (1.0f - back.rgb) * (1.0f - front.rgb);
    else if (p.mode == 4) blended = overlay(back.rgb, front.rgb);
    return float4(mix(back.rgb, blended, alpha), max(back.a, alpha));
}
)metal";

const char* kMetalFrameMixFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float mixValue; float3 padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> earlier [[texture(0)]],
                      texture2d<float> later [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    return mix(earlier.sample(imageSampler, in.uv),
               later.sample(imageSampler, in.uv),
               clamp(p.mixValue, 0.0f, 1.0f));
}
)metal";

const char* kMetalTransitionFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float progress; int transitionType; int blendMode; float padding; };
float3 overlay(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  base >= 0.5f);
}
float easeInOutCubic(float t) {
    if (t < 0.5f) return 4.0f * t * t * t;
    const float p = 2.0f * t - 2.0f;
    return 0.5f * p * p * p + 1.0f;
}
float4 transitionColor(float4 from, float4 to, float2 uv, constant Params& p) {
    const float progress = clamp(p.progress, 0.0f, 1.0f);
    if (p.transitionType == 0) return mix(from, to, easeInOutCubic(progress));
    if (p.transitionType == 1) {
        if (progress < 0.5f) return float4(from.rgb * (1.0f - progress * 2.0f), from.a);
        return float4(to.rgb * ((progress - 0.5f) * 2.0f), to.a);
    }
    if (p.transitionType == 2) return uv.x > 1.0f - progress ? to : from;
    if (p.transitionType == 3) return uv.x < progress ? to : from;
    if (p.transitionType == 4) return uv.y > 1.0f - progress ? to : from;
    if (p.transitionType == 5) return uv.y < progress ? to : from;
    return progress >= 0.5f ? to : from;
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> fromTex [[texture(0)]],
                      texture2d<float> toTex [[texture(1)]],
                      texture2d<float> backTex [[texture(2)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 front = transitionColor(fromTex.sample(imageSampler, in.uv),
                                         toTex.sample(imageSampler, in.uv), in.uv, p);
    const float4 back = backTex.sample(imageSampler, in.uv);
    float3 blended = front.rgb;
    if (p.blendMode == 1) blended = min(back.rgb + front.rgb, float3(1.0f));
    else if (p.blendMode == 2) blended = back.rgb * front.rgb;
    else if (p.blendMode == 3) blended = 1.0f - (1.0f - back.rgb) * (1.0f - front.rgb);
    else if (p.blendMode == 4) blended = overlay(back.rgb, front.rgb);
    return float4(mix(back.rgb, blended, front.a), max(back.a, front.a));
}
)metal";

const char* kMetalBlitFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
fragment float4 _main(In in [[stage_in]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{ return image.sample(imageSampler, in.uv); }
)metal";

const char* kMetalPreviewFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float zoom; float2 pan; float split; int layout; int background; float2 padding; };
float3 previewBackground(float2 uv, int mode) {
    if (mode == 1) return float3(0.0f);
    if (mode == 2) return float3(1.0f);
    float c = fmod(floor(uv.x * 32.0f) + floor(uv.y * 32.0f), 2.0f);
    return mix(float3(0.18f), float3(0.32f), c);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> finalImage [[texture(0)]],
                      texture2d<float> previewImage [[texture(1)]],
                      sampler imageSampler [[sampler(0)]]) {
    if (p.layout == 1 && in.uv.x < p.split)
        return finalImage.sample(imageSampler, float2(in.uv.x / p.split, in.uv.y));
    const float4 overlayRect = float4(0.58f, 0.04f, 0.98f, 0.44f);
    if (p.layout == 2 && (in.uv.x < overlayRect.x || in.uv.x > overlayRect.z
                      || in.uv.y < overlayRect.y || in.uv.y > overlayRect.w))
        return finalImage.sample(imageSampler, in.uv);
    float2 region = p.layout == 1
        ? float2((in.uv.x - p.split) / (1.0f - p.split), in.uv.y)
        : (in.uv - overlayRect.xy) / (overlayRect.zw - overlayRect.xy);
    float2 uv = (region - 0.5f) / p.zoom + 0.5f - p.pan;
    float4 sample = all(uv >= 0.0f) && all(uv <= 1.0f)
        ? previewImage.sample(imageSampler, uv) : float4(0.0f);
    return float4(mix(previewBackground(region, p.background), sample.rgb, sample.a), 1.0f);
}
)metal";

const char* kMetalBlurFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 texelStep; float radius; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const int radius = int(min(p.radius, 20.0f) + 0.5f);
    if (radius <= 0) return image.sample(imageSampler, in.uv);
    const float sigma = max(p.radius * 0.5f, 0.5f);
    float4 sum = float4(0.0f);
    float weightSum = 0.0f;
    for (int i = -20; i <= 20; ++i) {
        if (i < -radius || i > radius) continue;
        const float weight = exp(-float(i * i) / (2.0f * sigma * sigma));
        sum += image.sample(imageSampler, in.uv + p.texelStep * float(i)) * weight;
        weightSum += weight;
    }
    return sum / weightSum;
}
)metal";

const char* kMetalSharpenFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 texelSize; float amount; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 center = image.sample(imageSampler, in.uv);
    float3 blurred = float3(0.0f);
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            blurred += image.sample(imageSampler,
                in.uv + float2(float(x), float(y)) * p.texelSize).rgb;
    blurred /= 9.0f;
    return float4(clamp(center.rgb + p.amount * (center.rgb - blurred),
                        0.0f, 1.0f), center.a);
}
)metal";

const char* kMetalLutFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 unused; float size; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture3d<float> lut [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 color = image.sample(imageSampler, in.uv);
    const float size = max(p.size, 2.0f);
    const float3 uvw = clamp(color.rgb, 0.0f, 1.0f) * ((size - 1.0f) / size)
                     + 0.5f / size;
    return float4(lut.sample(imageSampler, uvw).rgb, color.a);
}
)metal";

const char* kMetalUvEffectFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 resolution; float time; int mode; float4 values; };
float4 sampleClamped(texture2d<float> image, sampler imageSampler, float2 uv) {
    return image.sample(imageSampler, clamp(uv, 0.0f, 1.0f));
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    float2 uv = in.uv;
    if (p.mode == 0) {
        const float aspect = p.resolution.x / max(p.resolution.y, 1.0f);
        const float2 point = (uv - 0.5f) * float2(aspect, 1.0f);
        const float radius = length(point);
        float angle = atan2(point.y, point.x);
        const float wedge = 6.28318530718f / max(p.values.x, 2.0f);
        angle -= (p.values.y + p.values.z * p.time) * 6.28318530718f;
        angle = fmod(angle, wedge);
        if (angle < 0.0f) angle += wedge;
        angle = abs(angle - 0.5f * wedge);
        const float2 folded = float2(cos(angle), sin(angle))
                            * (radius / max(p.values.w, 0.01f));
        uv = folded / float2(aspect, 1.0f) + 0.5f;
    } else if (p.mode == 1) {
        const int mode = int(p.values.x + 0.5f);
        if (mode == 0 && uv.x > 0.5f) uv.x = 1.0f - uv.x;
        else if (mode == 1 && uv.x < 0.5f) uv.x = 1.0f - uv.x;
        else if (mode == 2 && uv.y > 0.5f) uv.y = 1.0f - uv.y;
        else if (mode >= 3) {
            if (uv.x > 0.5f) uv.x = 1.0f - uv.x;
            if (uv.y > 0.5f) uv.y = 1.0f - uv.y;
        }
    } else if (p.mode == 2) {
        const float count = max(p.values.x, 1.0f);
        float2 tiled = uv * count;
        const float2 cell = floor(tiled);
        uv = fract(tiled);
        if (p.values.y > 0.5f) {
            if (fmod(cell.x, 2.0f) >= 1.0f) uv.x = 1.0f - uv.x;
            if (fmod(cell.y, 2.0f) >= 1.0f) uv.y = 1.0f - uv.y;
        }
    } else if (p.mode == 3) {
        const float phase = p.time * p.values.z;
        uv.x += p.values.x * sin(uv.y * p.values.y + phase);
        uv.y += p.values.x * cos(uv.x * p.values.y + phase * 1.3f);
    } else if (p.mode == 4) {
        const float2 texel = 1.0f / max(p.resolution, float2(1.0f));
        const float3 weights = float3(0.299f, 0.587f, 0.114f);
        const float lx = dot(sampleClamped(image, imageSampler,
            uv + float2(texel.x, 0.0f)).rgb, weights)
            - dot(sampleClamped(image, imageSampler,
            uv - float2(texel.x, 0.0f)).rgb, weights);
        const float ly = dot(sampleClamped(image, imageSampler,
            uv + float2(0.0f, texel.y)).rgb, weights)
            - dot(sampleClamped(image, imageSampler,
            uv - float2(0.0f, texel.y)).rgb, weights);
        uv += float2(lx, ly) * p.values.x;
    } else if (p.mode == 5) {
        const float aspect = p.resolution.x / max(p.resolution.y, 1.0f);
        const float2 point = (uv - 0.5f) * float2(aspect, 1.0f);
        const float radius = length(point);
        const float falloff = 1.0f - smoothstep(0.0f, max(p.values.y, 1.0e-3f), radius);
        const float angle = p.values.x * 6.28318530718f * falloff;
        const float c = cos(angle), s = sin(angle);
        const float2 rotated = float2(c * point.x - s * point.y,
                                      s * point.x + c * point.y);
        uv = rotated / float2(aspect, 1.0f) + 0.5f;
    } else if (p.mode == 6) {
        const float angle = p.values.y * 6.28318530718f;
        const float2 direction = float2(cos(angle), sin(angle)) * p.values.x;
        const float red = sampleClamped(image, imageSampler, uv + direction).r;
        const float4 center = image.sample(imageSampler, uv);
        const float blue = sampleClamped(image, imageSampler, uv - direction).b;
        return float4(red, center.g, blue, center.a);
    } else if (p.mode == 7) {
        const float block = max(p.values.x, 1.0f);
        const float2 resolution = max(p.resolution, float2(1.0f));
        uv = (floor(uv * resolution / block) + 0.5f) * block / resolution;
    }
    return sampleClamped(image, imageSampler, uv);
}
)metal";

const char* kMetalFeedbackFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float decay; float zoom; float swirl; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture2d<float> history [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float2 point = in.uv - 0.5f;
    const float c = cos(p.swirl), s = sin(p.swirl);
    const float2 trailUv = float2(c * point.x - s * point.y,
                                  s * point.x + c * point.y) * p.zoom + 0.5f;
    const float4 previous = history.sample(imageSampler,
        clamp(trailUv, 0.0f, 1.0f)) * clamp(p.decay, 0.0f, 0.999f);
    const float4 current = image.sample(imageSampler, in.uv);
    float4 result = 1.0f - (1.0f - current) * (1.0f - previous);
    result.a = max(current.a, previous.a);
    return result;
}
)metal";

const char* kMetalBloomThresholdFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float threshold; float intensity; float exposure; int tonemap; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float3 color = image.sample(imageSampler, in.uv).rgb;
    const float luminance = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    const float knee = max(p.threshold * 0.5f, 1.0e-4f);
    const float value = clamp((luminance - p.threshold + knee) / (2.0f * knee),
                              0.0f, 1.0f);
    return float4(color * (value * value), 1.0f);
}
)metal";

const char* kMetalPostCombineFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float threshold; float intensity; float exposure; int tonemap; };
float3 aces(float3 value) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return clamp((value * (a * value + b)) /
                 (value * (c * value + d) + e), 0.0f, 1.0f);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture2d<float> bloom [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float3 base = image.sample(imageSampler, in.uv).rgb;
    const float3 glow = bloom.sample(imageSampler, in.uv).rgb;
    const float3 hdr = (base + glow * p.intensity) * p.exposure;
    float3 ldr = clamp(hdr, 0.0f, 1.0f);
    if (p.tonemap == 1) ldr = hdr / (hdr + 1.0f);
    else if (p.tonemap == 2) ldr = aces(hdr);
    return float4(ldr, 1.0f);
}
)metal";

const char* kMetalDrawShapeFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float4 rect; float4 color; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]])
{
    float2 halfSize = max(abs(p.rect.zw) * 0.5f, float2(0.0f));
    float2 local = abs(in.uv - p.rect.xy);
    bool inside;
    if (p.rect.z < 0.0f)
    {
        if (any(halfSize <= float2(0.0f)))
            inside = false;
        else
        {
            float2 normalized = local / halfSize;
            inside = dot(normalized, normalized) <= 1.0f;
        }
    }
    else
    {
        float2 d = local - halfSize;
        inside = max(d.x, d.y) <= 0.0f;
    }
    return inside ? p.color : float4(0.0f);
}
)metal";

const char* kMetalCanvasFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float4 rect; float2 texel; float2 padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    float4 color = image.sample(imageSampler, in.uv);
    const bool inside = in.uv.x >= p.rect.x && in.uv.x <= p.rect.z
                     && in.uv.y >= p.rect.y && in.uv.y <= p.rect.w;
    if (!inside) color.rgb *= 0.45f;
    const float border = 1.5f;
    const float dx = min(abs(in.uv.x - p.rect.x), abs(in.uv.x - p.rect.z)) / p.texel.x;
    const float dy = min(abs(in.uv.y - p.rect.y), abs(in.uv.y - p.rect.w)) / p.texel.y;
    const bool spanX = in.uv.x >= p.rect.x - border * p.texel.x
                    && in.uv.x <= p.rect.z + border * p.texel.x;
    const bool spanY = in.uv.y >= p.rect.y - border * p.texel.y
                    && in.uv.y <= p.rect.w + border * p.texel.y;
    if ((dx <= border && spanY) || (dy <= border && spanX))
        color = float4(0.61f, 0.42f, 0.87f, 1.0f);
    return color;
}
)metal";

void makeTransform (float tx, float ty, float rotationDeg, float sx, float sy,
                    float out[16])
{
    const float radians = rotationDeg * 0.01745329251994329577f;
    const float c = std::cos (radians), s = std::sin (radians);
    const float values[16] = {
        c * sx, s * sx, 0.0f, 0.0f,
       -s * sy, c * sy, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        tx, ty, 0.0f, 1.0f,
    };
    std::copy (std::begin (values), std::end (values), out);
}

int metalTransitionType (int wireType)
{
    switch (wireType)
    {
        case 1: return 1;
        case 2: return 0;
        case 3: return 2;
        case 4: return 3;
        case 5: return 4;
        case 6: return 5;
        default: return 0;
    }
}

int metalUvEffectMode (int type)
{
    switch (static_cast<videofx::EffectType> (type))
    {
        case videofx::EffectType::Kaleidoscope: return 0;
        case videofx::EffectType::Mirror: return 1;
        case videofx::EffectType::Tile: return 2;
        case videofx::EffectType::Warp: return 3;
        case videofx::EffectType::Displace: return 4;
        case videofx::EffectType::PolarSwirl: return 5;
        case videofx::EffectType::DisplaceRgb: return 6;
        case videofx::EffectType::Pixelate: return 7;
        default: return -1;
    }
}

MetalEffectParams makeEffectParams (const LayerDesc& layer)
{
    static constexpr float neutral[MfxCount] = {
        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.18f, 0.10f, 0.0f,
        0.0f, 1.0f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    MetalEffectParams result;
    result.time = static_cast<float> (layer.timeSec);
    result.lutEnabled = layer.lutTexture != 0 && layer.lutSize >= 2 ? 1.0f : 0.0f;
    result.lutSize = static_cast<float> (layer.lutSize);
    std::copy (std::begin (neutral), std::end (neutral), result.values);
    for (int i = 0; layer.effects != nullptr && i < layer.effectCount; ++i)
    {
        const auto& effect = layer.effects[i];
        if (! effect.enabled || effect.type < 0
            || effect.type >= videofx::kEffectTypeCount)
            continue;
        result.mask |= videofx::kEffectBits[effect.type];
        switch (static_cast<videofx::EffectType> (effect.type))
        {
            case videofx::EffectType::Brightness: result.values[MfxBrightness] = effect.params[0]; break;
            case videofx::EffectType::Contrast: result.values[MfxContrast] = effect.params[0]; break;
            case videofx::EffectType::Saturation: result.values[MfxSaturation] = effect.params[0]; break;
            case videofx::EffectType::Hue: result.values[MfxHueShift] = effect.params[0]; break;
            case videofx::EffectType::Exposure: result.values[MfxExposure] = effect.params[0]; break;
            case videofx::EffectType::Gamma: result.values[MfxGamma] = effect.params[0]; break;
            case videofx::EffectType::Vignette:
                result.values[MfxVignetteAmount] = effect.params[0];
                result.values[MfxVignetteSoftness] = effect.params[1]; break;
            case videofx::EffectType::Warm: result.values[MfxWarmth] = effect.params[0]; break;
            case videofx::EffectType::Cool: result.values[MfxCoolness] = effect.params[0]; break;
            case videofx::EffectType::Vintage: result.values[MfxVintage] = effect.params[0]; break;
            case videofx::EffectType::Sepia: result.values[MfxSepia] = effect.params[0]; break;
            case videofx::EffectType::BlackAndWhite: result.values[MfxBw] = effect.params[0]; break;
            case videofx::EffectType::Invert: result.values[MfxInvert] = effect.params[0]; break;
            case videofx::EffectType::Posterize: result.values[MfxPosterize] = effect.params[0]; break;
            case videofx::EffectType::Noise: result.values[MfxNoise] = effect.params[0]; break;
            case videofx::EffectType::ChromaKey:
                for (int p = 0; p < 6; ++p) result.values[MfxKeyR + p] = effect.params[p];
                break;
            case videofx::EffectType::LumaKey:
                for (int p = 0; p < 4; ++p) result.values[MfxLumaLow + p] = effect.params[p];
                break;
            case videofx::EffectType::ColorWheels:
                for (int p = 0; p < 9; ++p) result.values[MfxLiftR + p] = effect.params[p];
                break;
            default: break;
        }
    }
    return result;
}

MetalEffectParams neutralEffectParams()
{
    const LayerDesc neutral;
    return makeEffectParams (neutral);
}

} // namespace

struct MetalFrameRenderer::Impl
{
    struct Target
    {
        sg_image image = {};
        sg_view attachment = {};
        sg_view texture = {};
    };

    struct Source
    {
        int width = 0, height = 0;
        sg_image image = {};
        sg_view view = {};
        std::vector<uint8_t> pixels;
        bool dirty = false;
        bool frameBlend = false;
        unsigned textureA = 0, textureB = 0;
        float blendMix = 0.0f;
        Target blendTarget;
    };

    struct Lut
    {
        int size = 0;
        sg_image image = {};
        sg_view view = {};
    };

    struct Depth
    {
        int width = 0, height = 0;
        sg_image image = {};
        sg_view view = {};
    };

    struct FeedbackHistory
    {
        Target target[2];
        int current = 0;
        bool ready = false;
    };

    struct DirectOutput
    {
        int width = 0, height = 0;
        id<MTLTexture> metalTexture = nil;
        sg_image image = {};
        sg_view attachment = {};
    };

    std::string error;
    videowire::VisualPlanTelemetry* visualTelemetry = nullptr;
    int width = 0, height = 0;
    int canvasWidth = 0, canvasHeight = 0;
    int presentWidth = 0, presentHeight = 0;
    float zoom = 1.0f, panX = 0.0f, panY = 0.0f;
    float bg[4] = { 0.04f, 0.04f, 0.05f, 1.0f };
    float bloomIntensity = 0.0f, bloomThreshold = 1.0f, bloomRadius = 0.0f;
    float exposure = 1.0f;
    int tonemap = 0;
    bool programsReady = false;
    bool directOnly = false;
    std::unordered_map<unsigned, Source> sources;
    std::unordered_map<unsigned, Lut> luts;
    std::unordered_map<unsigned, Depth> depths;
    std::unordered_map<int, std::unique_ptr<MetalParticleEngine>> particles;
#if ARBIT_HAVE_METAL_GENERATORS
    std::unordered_map<int, std::unique_ptr<MetalShaderGenerator>> generators;
#endif
    std::string particleBackend = "none";
    std::unordered_map<int, FeedbackHistory> feedback;
    std::unordered_map<void*, DirectOutput> directOutputs;
    void* requestedDirectSurface = nullptr;
    int requestedDirectWidth = 0, requestedDirectHeight = 0;
    uint64_t frameParity = 0;
    Lut identityLut;

    sg_sampler sampler = {};
    sg_shader layerShader = {}, blendShader = {}, frameMixShader = {}, transitionShader = {}, blitShader = {};
    sg_shader blurShader = {}, sharpenShader = {}, lutShader = {}, uvEffectShader = {};
    sg_shader feedbackShader = {};
    sg_shader bloomThresholdShader = {}, postCombineShader = {};
    sg_shader canvasShader = {}, previewShader = {}, drawShapeShader = {};
    sg_pipeline layerPipeline = {}, blendPipeline = {}, frameMixPipeline = {}, transitionPipeline = {}, blitPipeline = {};
    sg_pipeline blurPipeline = {}, sharpenPipeline = {}, lutPipeline = {}, uvEffectPipeline = {};
    sg_pipeline feedbackPipeline = {};
    sg_pipeline bloomThresholdPipeline = {}, postCombinePipeline = {};
    sg_pipeline canvasPipeline = {}, previewPipeline = {}, drawShapePipeline = {};
    Target layerTarget, transitionFrom, effect[2], accum[2], inspectionTarget;
    int inspectionClipId = -1;
    unsigned inspectionRequestedHandle = 0;
    unsigned inspectionRetainedHandle = 0;
    bool inspectionAdmitted = false;
    videopreview::State inspectionPresentation;

    IOSurfaceRef surface = nullptr;
    id<MTLTexture> metalTexture = nil;
    sg_image outputImage = {};
    sg_view outputAttachment = {};
    unsigned rectangleTexture = 0, rectangleFbo = 0;
    unsigned outputTexture = 0, outputFbo = 0;

    void destroySource (Source& source)
    {
        destroyTarget (source.blendTarget);
        if (source.view.id != 0) sg_destroy_view (source.view);
        if (source.image.id != 0) sg_destroy_image (source.image);
        source = {};
    }

    void destroyTarget (Target& target)
    {
        if (target.texture.id != 0) sg_destroy_view (target.texture);
        if (target.attachment.id != 0) sg_destroy_view (target.attachment);
        if (target.image.id != 0) sg_destroy_image (target.image);
        target = {};
    }

    void destroyLut (Lut& lut)
    {
        if (lut.view.id != 0) sg_destroy_view (lut.view);
        if (lut.image.id != 0) sg_destroy_image (lut.image);
        lut = {};
    }

    void destroyDepth (Depth& depth)
    {
        if (depth.view.id != 0) sg_destroy_view (depth.view);
        if (depth.image.id != 0) sg_destroy_image (depth.image);
        depth = {};
    }

    void destroyDirectOutput (DirectOutput& output)
    {
        if (output.attachment.id != 0) sg_destroy_view (output.attachment);
        if (output.image.id != 0) sg_destroy_image (output.image);
#if ! __has_feature(objc_arc)
        [output.metalTexture release];
#endif
        output = {};
    }

    void clearDirectOutputs()
    {
        for (auto& item : directOutputs) destroyDirectOutput (item.second);
        directOutputs.clear();
    }

    DirectOutput* directOutput (void* surfacePtr, int w, int h)
    {
        if (surfacePtr == nullptr || w <= 0 || h <= 0) return nullptr;
        auto& output = directOutputs[surfacePtr];
        if (output.image.id != 0 && output.width == w && output.height == h)
            return &output;
        destroyDirectOutput (output);
        IOSurfaceRef surface = static_cast<IOSurfaceRef> (surfacePtr);
        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:w height:h mipmapped:NO];
        // The IOSurface is sampled by a Metal device in another process.
        // Managed storage would give each process a private GPU backing store,
        // so the consumer could legally observe undefined pixels even after
        // this queue completes. Shared storage keeps the IOSurface itself as
        // the single producer/consumer allocation.
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        output.metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor
                                                            iosurface:surface plane:0];
        if (output.metalTexture == nil) return nullptr;
        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = w;
        imageDesc.height = h;
        imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        imageDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        [output.metalTexture retain];
#endif
        imageDesc.mtl_textures[0] = (__bridge const void*) output.metalTexture;
        imageDesc.label = "arbit-metal-direct-iosurface";
        output.image = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.color_attachment.image = output.image;
        output.attachment = sg_make_view (&viewDesc);
        output.width = w;
        output.height = h;
        if (! resourceValid (sg_query_image_state (output.image))
            || ! resourceValid (sg_query_view_state (output.attachment)))
        {
            destroyDirectOutput (output);
            return nullptr;
        }
        return &output;
    }

    bool makeLut (Lut& lut, const float* rgbTriples, int size, const char* label)
    {
        destroyLut (lut);
        if (rgbTriples == nullptr || size < 2) return false;
        const size_t voxels = static_cast<size_t> (size) * size * size;
        std::vector<float> rgba (voxels * 4);
        for (size_t i = 0; i < voxels; ++i)
        {
            rgba[i * 4] = rgbTriples[i * 3];
            rgba[i * 4 + 1] = rgbTriples[i * 3 + 1];
            rgba[i * 4 + 2] = rgbTriples[i * 3 + 2];
            rgba[i * 4 + 3] = 1.0f;
        }
        sg_image_desc desc = {};
        desc.type = SG_IMAGETYPE_3D;
        desc.width = size;
        desc.height = size;
        desc.num_slices = size;
        desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
        desc.data.mip_levels[0] = { rgba.data(), rgba.size() * sizeof (float) };
        desc.label = label;
        lut.image = sg_make_image (&desc);
        sg_view_desc viewDesc = {};
        viewDesc.texture.image = lut.image;
        lut.view = sg_make_view (&viewDesc);
        lut.size = size;
        return resourceValid (sg_query_image_state (lut.image))
            && resourceValid (sg_query_view_state (lut.view));
    }

    void destroyOutputs (const arbitgl::GlFuncs* gl)
    {
        for (auto& item : feedback)
        {
            destroyTarget (item.second.target[0]);
            destroyTarget (item.second.target[1]);
        }
        feedback.clear();
        destroyTarget (layerTarget);
        destroyTarget (transitionFrom);
        destroyTarget (effect[0]);
        destroyTarget (effect[1]);
        destroyTarget (accum[0]);
        destroyTarget (accum[1]);
        destroyTarget (inspectionTarget);
        inspectionRetainedHandle = 0;
        inspectionAdmitted = false;
        if (outputAttachment.id != 0) sg_destroy_view (outputAttachment);
        if (outputImage.id != 0) sg_destroy_image (outputImage);
        outputAttachment = {};
        outputImage = {};
        if (rectangleFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &rectangleFbo);
        if (outputFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &outputFbo);
        if (rectangleTexture != 0) glDeleteTextures (1, &rectangleTexture);
        if (outputTexture != 0) glDeleteTextures (1, &outputTexture);
        rectangleTexture = rectangleFbo = outputTexture = outputFbo = 0;
#if ! __has_feature(objc_arc)
        [metalTexture release];
#endif
        metalTexture = nil;
        if (surface != nullptr) CFRelease (surface);
        surface = nullptr;
        width = height = 0;
    }

    bool makeTarget (Target& target, int w, int h, const char* label)
    {
        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = w;
        imageDesc.height = h;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA16F;
        imageDesc.sample_count = 1;
        imageDesc.label = label;
        target.image = sg_make_image (&imageDesc);
        sg_view_desc attachmentDesc = {};
        attachmentDesc.color_attachment.image = target.image;
        target.attachment = sg_make_view (&attachmentDesc);
        sg_view_desc textureDesc = {};
        textureDesc.texture.image = target.image;
        target.texture = sg_make_view (&textureDesc);
        return resourceValid (sg_query_image_state (target.image))
            && resourceValid (sg_query_view_state (target.attachment))
            && resourceValid (sg_query_view_state (target.texture));
    }

    bool ensurePrograms()
    {
        if (programsReady) return true;

        sg_sampler_desc samplerDesc = {};
        samplerDesc.min_filter = SG_FILTER_LINEAR;
        samplerDesc.mag_filter = SG_FILTER_LINEAR;
        samplerDesc.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_w = SG_WRAP_CLAMP_TO_EDGE;
        sampler = sg_make_sampler (&samplerDesc);

        static constexpr float identity[24] = {
            0,0,0, 1,0,0, 0,1,0, 1,1,0,
            0,0,1, 1,0,1, 0,1,1, 1,1,1,
        };
        if (! makeLut (identityLut, identity, 2, "arbit-metal-identity-lut"))
        {
            error = "Metal identity LUT creation failed";
            return false;
        }

        sg_shader_desc layerDesc = {};
        layerDesc.vertex_func.source = kMetalLayerVertex;
        layerDesc.fragment_func.source = kMetalLayerFragment;
        layerDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        layerDesc.uniform_blocks[0].size = sizeof (MetalGeometryParams);
        layerDesc.uniform_blocks[0].msl_buffer_n = 0;
        layerDesc.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[1].size = sizeof (MetalMaskParams);
        layerDesc.uniform_blocks[1].msl_buffer_n = 0;
        layerDesc.uniform_blocks[2].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[2].size = sizeof (MetalEffectParams);
        layerDesc.uniform_blocks[2].msl_buffer_n = 1;
        layerDesc.uniform_blocks[3].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[3].size = sizeof (MetalDepthFogParams);
        layerDesc.uniform_blocks[3].msl_buffer_n = 2;
        layerDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[0].texture.msl_texture_n = 0;
        layerDesc.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[1].texture.image_type = SG_IMAGETYPE_3D;
        layerDesc.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[1].texture.msl_texture_n = 1;
        layerDesc.views[2].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[2].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[2].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[2].texture.msl_texture_n = 2;
        layerDesc.views[3].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[3].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[3].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[3].texture.msl_texture_n = 3;
        layerDesc.views[4].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[4].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[4].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[4].texture.msl_texture_n = 4;
        layerDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        layerDesc.samplers[0].msl_sampler_n = 0;
        layerDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        layerDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "lut" };
        layerDesc.texture_sampler_pairs[2] = { SG_SHADERSTAGE_FRAGMENT, 2, 0, "matte" };
        layerDesc.texture_sampler_pairs[3] = { SG_SHADERSTAGE_FRAGMENT, 3, 0, "depth" };
        layerDesc.label = "arbit-metal-frame-layer";
        layerShader = sg_make_shader (&layerDesc);

        sg_pipeline_desc layerPipelineDesc = {};
        layerPipelineDesc.shader = layerShader;
        layerPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        layerPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        layerPipelineDesc.label = "arbit-metal-frame-layer-pipeline";
        layerPipeline = sg_make_pipeline (&layerPipelineDesc);

        sg_shader_desc blendDesc = {};
        blendDesc.vertex_func.source = kMetalFullscreenVertex;
        blendDesc.fragment_func.source = kMetalBlendFragment;
        blendDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blendDesc.uniform_blocks[0].size = sizeof (MetalBlendParams);
        blendDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            blendDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            blendDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            blendDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            blendDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        blendDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blendDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        blendDesc.samplers[0].msl_sampler_n = 0;
        blendDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "frontTex" };
        blendDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "backTex" };
        blendDesc.label = "arbit-metal-frame-blend";
        blendShader = sg_make_shader (&blendDesc);

        sg_pipeline_desc blendPipelineDesc = {};
        blendPipelineDesc.shader = blendShader;
        blendPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        blendPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        blendPipelineDesc.label = "arbit-metal-frame-blend-pipeline";
        blendPipeline = sg_make_pipeline (&blendPipelineDesc);

        sg_shader_desc frameMixDesc = {};
        frameMixDesc.vertex_func.source = kMetalFullscreenVertex;
        frameMixDesc.fragment_func.source = kMetalFrameMixFragment;
        frameMixDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        frameMixDesc.uniform_blocks[0].size = sizeof (MetalFrameMixParams);
        frameMixDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            frameMixDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            frameMixDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            frameMixDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            frameMixDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        frameMixDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        frameMixDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        frameMixDesc.samplers[0].msl_sampler_n = 0;
        frameMixDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "earlier" };
        frameMixDesc.texture_sampler_pairs[1] = {
            SG_SHADERSTAGE_FRAGMENT, 1, 0, "later" };
        frameMixDesc.label = "arbit-metal-frame-mix";
        frameMixShader = sg_make_shader (&frameMixDesc);
        sg_pipeline_desc frameMixPipelineDesc = {};
        frameMixPipelineDesc.shader = frameMixShader;
        frameMixPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        frameMixPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        frameMixPipelineDesc.label = "arbit-metal-frame-mix-pipeline";
        frameMixPipeline = sg_make_pipeline (&frameMixPipelineDesc);

        sg_shader_desc transitionDesc = {};
        transitionDesc.vertex_func.source = kMetalFullscreenVertex;
        transitionDesc.fragment_func.source = kMetalTransitionFragment;
        transitionDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        transitionDesc.uniform_blocks[0].size = sizeof (MetalTransitionParams);
        transitionDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 3; ++i)
        {
            transitionDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            transitionDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            transitionDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            transitionDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        transitionDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        transitionDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        transitionDesc.samplers[0].msl_sampler_n = 0;
        transitionDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "fromTex" };
        transitionDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "toTex" };
        transitionDesc.texture_sampler_pairs[2] = { SG_SHADERSTAGE_FRAGMENT, 2, 0, "backTex" };
        transitionDesc.label = "arbit-metal-frame-transition";
        transitionShader = sg_make_shader (&transitionDesc);
        sg_pipeline_desc transitionPipelineDesc = {};
        transitionPipelineDesc.shader = transitionShader;
        transitionPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        transitionPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        transitionPipelineDesc.label = "arbit-metal-frame-transition-pipeline";
        transitionPipeline = sg_make_pipeline (&transitionPipelineDesc);

        sg_shader_desc blitDesc = {};
        blitDesc.vertex_func.source = kMetalFullscreenVertex;
        blitDesc.fragment_func.source = kMetalBlitFragment;
        blitDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        blitDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        blitDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        blitDesc.views[0].texture.msl_texture_n = 0;
        blitDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blitDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        blitDesc.samplers[0].msl_sampler_n = 0;
        blitDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        blitDesc.label = "arbit-metal-frame-blit";
        blitShader = sg_make_shader (&blitDesc);
        sg_pipeline_desc blitPipelineDesc = {};
        blitPipelineDesc.shader = blitShader;
        blitPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        blitPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        blitPipelineDesc.label = "arbit-metal-frame-blit-pipeline";
        blitPipeline = sg_make_pipeline (&blitPipelineDesc);

        sg_shader_desc previewDesc = {};
        previewDesc.vertex_func.source = kMetalFullscreenVertex;
        previewDesc.fragment_func.source = kMetalPreviewFragment;
        previewDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        previewDesc.uniform_blocks[0].size = sizeof (MetalPreviewParams);
        previewDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            previewDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            previewDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            previewDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            previewDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        previewDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        previewDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        previewDesc.samplers[0].msl_sampler_n = 0;
        previewDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "finalImage" };
        previewDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "previewImage" };
        previewDesc.label = "arbit-metal-node-preview";
        previewShader = sg_make_shader (&previewDesc);
        sg_pipeline_desc previewPipelineDesc = {};
        previewPipelineDesc.shader = previewShader;
        previewPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        previewPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        previewPipelineDesc.label = "arbit-metal-node-preview-pipeline";
        previewPipeline = sg_make_pipeline (&previewPipelineDesc);

        auto makeFilter = [&] (const char* fragment, const char* label,
                               sg_shader& shader, sg_pipeline& pipeline,
                               bool hasLut)
        {
            sg_shader_desc desc = {};
            desc.vertex_func.source = kMetalFullscreenVertex;
            desc.fragment_func.source = fragment;
            desc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.uniform_blocks[0].size = sizeof (MetalFilterParams);
            desc.uniform_blocks[0].msl_buffer_n = 0;
            desc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            desc.views[0].texture.image_type = SG_IMAGETYPE_2D;
            desc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            desc.views[0].texture.msl_texture_n = 0;
            desc.texture_sampler_pairs[0] = {
                SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
            if (hasLut)
            {
                desc.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
                desc.views[1].texture.image_type = SG_IMAGETYPE_3D;
                desc.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
                desc.views[1].texture.msl_texture_n = 1;
                desc.texture_sampler_pairs[1] = {
                    SG_SHADERSTAGE_FRAGMENT, 1, 0, "lut" };
            }
            desc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
            desc.samplers[0].msl_sampler_n = 0;
            desc.label = label;
            shader = sg_make_shader (&desc);
            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.shader = shader;
            pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
            pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
            pipelineDesc.label = label;
            pipeline = sg_make_pipeline (&pipelineDesc);
        };
        makeFilter (kMetalBlurFragment, "arbit-metal-frame-blur",
                    blurShader, blurPipeline, false);
        makeFilter (kMetalSharpenFragment, "arbit-metal-frame-sharpen",
                    sharpenShader, sharpenPipeline, false);
        makeFilter (kMetalLutFragment, "arbit-metal-frame-lut",
                    lutShader, lutPipeline, true);

        sg_shader_desc uvDesc = {};
        uvDesc.vertex_func.source = kMetalFullscreenVertex;
        uvDesc.fragment_func.source = kMetalUvEffectFragment;
        uvDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.uniform_blocks[0].size = sizeof (MetalUvEffectParams);
        uvDesc.uniform_blocks[0].msl_buffer_n = 0;
        uvDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        uvDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        uvDesc.views[0].texture.msl_texture_n = 0;
        uvDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        uvDesc.samplers[0].msl_sampler_n = 0;
        uvDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        uvDesc.label = "arbit-metal-frame-uv-effect";
        uvEffectShader = sg_make_shader (&uvDesc);
        sg_pipeline_desc uvPipelineDesc = {};
        uvPipelineDesc.shader = uvEffectShader;
        uvPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        uvPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        uvPipelineDesc.label = "arbit-metal-frame-uv-effect-pipeline";
        uvEffectPipeline = sg_make_pipeline (&uvPipelineDesc);

        sg_shader_desc feedbackDesc = {};
        feedbackDesc.vertex_func.source = kMetalFullscreenVertex;
        feedbackDesc.fragment_func.source = kMetalFeedbackFragment;
        feedbackDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        feedbackDesc.uniform_blocks[0].size = sizeof (MetalFeedbackParams);
        feedbackDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            feedbackDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            feedbackDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            feedbackDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            feedbackDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        feedbackDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        feedbackDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        feedbackDesc.samplers[0].msl_sampler_n = 0;
        feedbackDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        feedbackDesc.texture_sampler_pairs[1] = {
            SG_SHADERSTAGE_FRAGMENT, 1, 0, "history" };
        feedbackDesc.label = "arbit-metal-frame-feedback";
        feedbackShader = sg_make_shader (&feedbackDesc);
        sg_pipeline_desc feedbackPipelineDesc = {};
        feedbackPipelineDesc.shader = feedbackShader;
        feedbackPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        feedbackPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        feedbackPipelineDesc.label = "arbit-metal-frame-feedback-pipeline";
        feedbackPipeline = sg_make_pipeline (&feedbackPipelineDesc);

        auto makePost = [&] (const char* fragment, const char* label,
                             int textureCount, sg_shader& shader,
                             sg_pipeline& pipeline)
        {
            sg_shader_desc desc = {};
            desc.vertex_func.source = kMetalFullscreenVertex;
            desc.fragment_func.source = fragment;
            desc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.uniform_blocks[0].size = sizeof (MetalPostParams);
            desc.uniform_blocks[0].msl_buffer_n = 0;
            for (int i = 0; i < textureCount; ++i)
            {
                desc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
                desc.views[i].texture.image_type = SG_IMAGETYPE_2D;
                desc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
                desc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
            }
            desc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
            desc.samplers[0].msl_sampler_n = 0;
            desc.texture_sampler_pairs[0] = {
                SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
            if (textureCount > 1)
                desc.texture_sampler_pairs[1] = {
                    SG_SHADERSTAGE_FRAGMENT, 1, 0, "bloom" };
            desc.label = label;
            shader = sg_make_shader (&desc);
            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.shader = shader;
            pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
            pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
            pipelineDesc.label = label;
            pipeline = sg_make_pipeline (&pipelineDesc);
        };
        makePost (kMetalBloomThresholdFragment, "arbit-metal-frame-bloom-threshold",
                  1, bloomThresholdShader, bloomThresholdPipeline);
        makePost (kMetalPostCombineFragment, "arbit-metal-frame-post-combine",
                  2, postCombineShader, postCombinePipeline);

        sg_shader_desc canvasDesc = {};
        canvasDesc.vertex_func.source = kMetalFullscreenVertex;
        canvasDesc.fragment_func.source = kMetalCanvasFragment;
        canvasDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.uniform_blocks[0].size = sizeof (MetalCanvasParams);
        canvasDesc.uniform_blocks[0].msl_buffer_n = 0;
        canvasDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        canvasDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        canvasDesc.views[0].texture.msl_texture_n = 0;
        canvasDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        canvasDesc.samplers[0].msl_sampler_n = 0;
        canvasDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        canvasDesc.label = "arbit-metal-frame-canvas";
        canvasShader = sg_make_shader (&canvasDesc);
        sg_pipeline_desc canvasPipelineDesc = {};
        canvasPipelineDesc.shader = canvasShader;
        canvasPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        canvasPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        canvasPipelineDesc.label = "arbit-metal-frame-canvas-pipeline";
        canvasPipeline = sg_make_pipeline (&canvasPipelineDesc);

        sg_shader_desc drawShapeDesc = {};
        drawShapeDesc.vertex_func.source = kMetalFullscreenVertex;
        drawShapeDesc.fragment_func.source = kMetalDrawShapeFragment;
        drawShapeDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        drawShapeDesc.uniform_blocks[0].size = sizeof (MetalDrawShapeParams);
        drawShapeDesc.uniform_blocks[0].msl_buffer_n = 0;
        drawShapeDesc.label = "arbit-metal-draw-shape";
        drawShapeShader = sg_make_shader (&drawShapeDesc);
        sg_pipeline_desc drawShapePipelineDesc = {};
        drawShapePipelineDesc.shader = drawShapeShader;
        drawShapePipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        drawShapePipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        drawShapePipelineDesc.label = "arbit-metal-draw-shape-pipeline";
        drawShapePipeline = sg_make_pipeline (&drawShapePipelineDesc);

        programsReady = resourceValid (sg_query_sampler_state (sampler))
            && resourceValid (sg_query_shader_state (layerShader))
            && resourceValid (sg_query_pipeline_state (layerPipeline))
            && resourceValid (sg_query_shader_state (blendShader))
            && resourceValid (sg_query_pipeline_state (blendPipeline))
            && resourceValid (sg_query_shader_state (frameMixShader))
            && resourceValid (sg_query_pipeline_state (frameMixPipeline))
            && resourceValid (sg_query_shader_state (transitionShader))
            && resourceValid (sg_query_pipeline_state (transitionPipeline))
            && resourceValid (sg_query_shader_state (blitShader))
            && resourceValid (sg_query_pipeline_state (blitPipeline))
            && resourceValid (sg_query_shader_state (previewShader))
            && resourceValid (sg_query_pipeline_state (previewPipeline))
            && resourceValid (sg_query_shader_state (blurShader))
            && resourceValid (sg_query_pipeline_state (blurPipeline))
            && resourceValid (sg_query_shader_state (sharpenShader))
            && resourceValid (sg_query_pipeline_state (sharpenPipeline))
            && resourceValid (sg_query_shader_state (lutShader))
            && resourceValid (sg_query_pipeline_state (lutPipeline))
            && resourceValid (sg_query_shader_state (uvEffectShader))
            && resourceValid (sg_query_pipeline_state (uvEffectPipeline))
            && resourceValid (sg_query_shader_state (feedbackShader))
            && resourceValid (sg_query_pipeline_state (feedbackPipeline))
            && resourceValid (sg_query_shader_state (bloomThresholdShader))
            && resourceValid (sg_query_pipeline_state (bloomThresholdPipeline))
            && resourceValid (sg_query_shader_state (postCombineShader))
            && resourceValid (sg_query_pipeline_state (postCombinePipeline))
            && resourceValid (sg_query_shader_state (canvasShader))
            && resourceValid (sg_query_pipeline_state (canvasPipeline))
            && resourceValid (sg_query_shader_state (drawShapeShader))
            && resourceValid (sg_query_pipeline_state (drawShapePipeline));
        if (! programsReady) error = "Metal production compositor pipeline creation failed";
        return programsReady;
    }

    bool ensureOutputs (const arbitgl::GlFuncs* gl, int w, int h)
    {
        if (accum[0].image.id != 0 && width == w && height == h
            && (directOnly || outputTexture != 0))
            return true;
        destroyOutputs (gl);
        if (! makeTarget (layerTarget, w, h, "arbit-metal-frame-layer-target")
            || ! makeTarget (transitionFrom, w, h, "arbit-metal-frame-transition-from")
            || ! makeTarget (effect[0], w, h, "arbit-metal-frame-effect-a")
            || ! makeTarget (effect[1], w, h, "arbit-metal-frame-effect-b")
            || ! makeTarget (accum[0], w, h, "arbit-metal-frame-accum-a")
            || ! makeTarget (accum[1], w, h, "arbit-metal-frame-accum-b")
            || ! makeTarget (inspectionTarget, w, h, "arbit-metal-inspection-retained"))
        {
            error = "Metal production compositor target creation failed";
            destroyOutputs (gl);
            return false;
        }

        width = w;
        height = h;
        if (directOnly)
            return true;

        CFMutableDictionaryRef props = CFDictionaryCreateMutable (
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        iosurfaceSetInt (props, kIOSurfaceWidth, w);
        iosurfaceSetInt (props, kIOSurfaceHeight, h);
        iosurfaceSetInt (props, kIOSurfaceBytesPerElement, 4);
        iosurfaceSetInt (props, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
        surface = IOSurfaceCreate (props);
        CFRelease (props);
        if (surface == nullptr) { error = "Metal compositor IOSurface creation failed"; return false; }

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:w height:h mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor iosurface:surface plane:0];
        if (metalTexture == nil) { error = "Metal compositor IOSurface texture creation failed"; return false; }

        sg_image_desc outputDesc = {};
        outputDesc.usage.color_attachment = true;
        outputDesc.width = w;
        outputDesc.height = h;
        outputDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        outputDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        // sg_make_image transfers the injected pointer through a temporary
        // which sokol releases after its resource pool has retained it. Keep
        // the compositor's original ownership independent of that transfer.
        [metalTexture retain];
#endif
        outputDesc.mtl_textures[0] = (__bridge const void*) metalTexture;
        outputDesc.label = "arbit-metal-frame-iosurface";
        outputImage = sg_make_image (&outputDesc);
        sg_view_desc outputViewDesc = {};
        outputViewDesc.color_attachment.image = outputImage;
        outputAttachment = sg_make_view (&outputViewDesc);
        if (! resourceValid (sg_query_image_state (outputImage))
            || ! resourceValid (sg_query_view_state (outputAttachment)))
        { error = "Metal compositor IOSurface attachment creation failed"; return false; }

        CGLContextObj cgl = CGLGetCurrentContext();
        if (cgl == nullptr) { error = "Metal compositor has no current CGL bridge context"; return false; }
        glGenTextures (1, &rectangleTexture);
        glBindTexture (GL_TEXTURE_RECTANGLE, rectangleTexture);
        const CGLError cglError = CGLTexImageIOSurface2D (
            cgl, GL_TEXTURE_RECTANGLE, GL_RGBA8, w, h,
            GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
        if (cglError != kCGLNoError)
        { error = std::string ("Metal compositor CGL IOSurface import failed: ") + CGLErrorString (cglError); return false; }
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture (GL_TEXTURE_RECTANGLE, 0);
        gl->GenFramebuffers (1, &rectangleFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, rectangleFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_RECTANGLE, rectangleTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { error = "Metal compositor IOSurface GL framebuffer incomplete"; return false; }

        glGenTextures (1, &outputTexture);
        glBindTexture (GL_TEXTURE_2D, outputTexture);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl->GenFramebuffers (1, &outputFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, outputFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, outputTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { error = "Metal compositor bridge framebuffer incomplete"; return false; }
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        return true;
    }

    bool supports (const LayerDesc* layers, int numLayers,
                   const ImageLayerDesc* overlays, int numOverlays) const
    {
        auto supportsLayer = [this] (const LayerDesc& layer)
        {
            if (layer.drawShape && (layer.isAdjustment || layer.transitionType != 0))
                return false;
            if (layer.shaderSource)
            {
#if ARBIT_HAVE_METAL_GENERATORS
                const auto generator = generators.find (layer.clipId);
                if (generator == generators.end() || generator->second == nullptr
                    || ! generator->second->hasProgram())
                    return false;
#else
                return false;
#endif
            }
            if (! layer.isAdjustment && layer.texture != 0
                && sources.find (layer.texture) == sources.end())
                return false;
            if (layer.lutTexture != 0 && luts.find (layer.lutTexture) == luts.end())
                return false;
            if ((layer.depthFog || layer.depthEffect != 0)
                && (layer.depthTexture == 0 || layer.depthWidth <= 0 || layer.depthHeight <= 0
                    || depths.find (layer.depthTexture) == depths.end()))
                return false;
            return true;
        };
        for (int i = 0; i < numLayers; ++i)
        {
            const auto& layer = layers[i];
            if (! supportsLayer (layer)) return false;
            if (layer.transitionType != 0 && layer.fromLayer != nullptr
                && ! supportsLayer (*layer.fromLayer))
                return false;
        }
        for (int i = 0; i < numOverlays; ++i)
            if (overlays[i].texture != 0
                && sources.find (overlays[i].texture) == sources.end())
                return false;
        return true;
    }
};

MetalFrameRenderer::MetalFrameRenderer() : impl_ (std::make_unique<Impl>()) {}
MetalFrameRenderer::~MetalFrameRenderer() = default;

bool MetalFrameRenderer::initialize (const arbitgl::GlFuncs* gl, int width, int height,
                                     std::string& errorOut, bool directOnly)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool
    {
        impl_->directOnly = directOnly;
        if (! ensureSokolMetal() || ! impl_->ensurePrograms()
            || ! impl_->ensureOutputs (gl, width, height))
        {
            errorOut = ! impl_->error.empty() ? impl_->error : gSokolError;
            return false;
        }
    }
    errorOut.clear();
    return true;
}

void MetalFrameRenderer::shutdown (const arbitgl::GlFuncs* gl)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
#if ARBIT_HAVE_METAL_GENERATORS
    for (auto& item : impl_->generators)
        if (item.second != nullptr) item.second->shutdownUnlocked();
    impl_->generators.clear();
#endif
    for (auto& item : impl_->particles)
        if (item.second != nullptr) item.second->shutdownUnlocked (gl);
    impl_->particles.clear();
    for (auto& item : impl_->sources) impl_->destroySource (item.second);
    impl_->sources.clear();
    for (auto& item : impl_->luts) impl_->destroyLut (item.second);
    impl_->luts.clear();
    for (auto& item : impl_->depths) impl_->destroyDepth (item.second);
    impl_->depths.clear();
    impl_->destroyLut (impl_->identityLut);
    impl_->clearDirectOutputs();
    impl_->destroyOutputs (gl);
    if (impl_->layerPipeline.id != 0) sg_destroy_pipeline (impl_->layerPipeline);
    if (impl_->blendPipeline.id != 0) sg_destroy_pipeline (impl_->blendPipeline);
    if (impl_->frameMixPipeline.id != 0) sg_destroy_pipeline (impl_->frameMixPipeline);
    if (impl_->transitionPipeline.id != 0) sg_destroy_pipeline (impl_->transitionPipeline);
    if (impl_->blitPipeline.id != 0) sg_destroy_pipeline (impl_->blitPipeline);
    if (impl_->blurPipeline.id != 0) sg_destroy_pipeline (impl_->blurPipeline);
    if (impl_->sharpenPipeline.id != 0) sg_destroy_pipeline (impl_->sharpenPipeline);
    if (impl_->lutPipeline.id != 0) sg_destroy_pipeline (impl_->lutPipeline);
    if (impl_->uvEffectPipeline.id != 0) sg_destroy_pipeline (impl_->uvEffectPipeline);
    if (impl_->feedbackPipeline.id != 0) sg_destroy_pipeline (impl_->feedbackPipeline);
    if (impl_->bloomThresholdPipeline.id != 0) sg_destroy_pipeline (impl_->bloomThresholdPipeline);
    if (impl_->postCombinePipeline.id != 0) sg_destroy_pipeline (impl_->postCombinePipeline);
    if (impl_->canvasPipeline.id != 0) sg_destroy_pipeline (impl_->canvasPipeline);
    if (impl_->previewPipeline.id != 0) sg_destroy_pipeline (impl_->previewPipeline);
    if (impl_->drawShapePipeline.id != 0) sg_destroy_pipeline (impl_->drawShapePipeline);
    if (impl_->layerShader.id != 0) sg_destroy_shader (impl_->layerShader);
    if (impl_->blendShader.id != 0) sg_destroy_shader (impl_->blendShader);
    if (impl_->frameMixShader.id != 0) sg_destroy_shader (impl_->frameMixShader);
    if (impl_->transitionShader.id != 0) sg_destroy_shader (impl_->transitionShader);
    if (impl_->blitShader.id != 0) sg_destroy_shader (impl_->blitShader);
    if (impl_->blurShader.id != 0) sg_destroy_shader (impl_->blurShader);
    if (impl_->sharpenShader.id != 0) sg_destroy_shader (impl_->sharpenShader);
    if (impl_->lutShader.id != 0) sg_destroy_shader (impl_->lutShader);
    if (impl_->uvEffectShader.id != 0) sg_destroy_shader (impl_->uvEffectShader);
    if (impl_->feedbackShader.id != 0) sg_destroy_shader (impl_->feedbackShader);
    if (impl_->bloomThresholdShader.id != 0) sg_destroy_shader (impl_->bloomThresholdShader);
    if (impl_->postCombineShader.id != 0) sg_destroy_shader (impl_->postCombineShader);
    if (impl_->canvasShader.id != 0) sg_destroy_shader (impl_->canvasShader);
    if (impl_->previewShader.id != 0) sg_destroy_shader (impl_->previewShader);
    if (impl_->drawShapeShader.id != 0) sg_destroy_shader (impl_->drawShapeShader);
    if (impl_->sampler.id != 0) sg_destroy_sampler (impl_->sampler);
    impl_->layerPipeline = impl_->blendPipeline = impl_->frameMixPipeline = {};
    impl_->transitionPipeline = impl_->blitPipeline = {};
    impl_->layerShader = impl_->blendShader = impl_->frameMixShader = {};
    impl_->transitionShader = impl_->blitShader = {};
    impl_->blurPipeline = impl_->sharpenPipeline = impl_->lutPipeline = {};
    impl_->blurShader = impl_->sharpenShader = impl_->lutShader = {};
    impl_->uvEffectPipeline = {};
    impl_->uvEffectShader = {};
    impl_->feedbackPipeline = {};
    impl_->feedbackShader = {};
    impl_->bloomThresholdPipeline = impl_->postCombinePipeline = {};
    impl_->bloomThresholdShader = impl_->postCombineShader = {};
    impl_->canvasPipeline = {};
    impl_->previewPipeline = {};
    impl_->drawShapePipeline = {};
    impl_->canvasShader = {};
    impl_->previewShader = {};
    impl_->drawShapeShader = {};
    impl_->sampler = {};
    impl_->programsReady = false;
}

void MetalFrameRenderer::setOutputSize (const arbitgl::GlFuncs* gl, int width, int height)
{
    if (width == impl_->width && height == impl_->height) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool { impl_->ensureOutputs (gl, width, height); }
}

void MetalFrameRenderer::setCanvas (int width, int height)
{ impl_->canvasWidth = width; impl_->canvasHeight = height; }

void MetalFrameRenderer::setPresentSize (int width, int height)
{ impl_->presentWidth = width; impl_->presentHeight = height; }

void MetalFrameRenderer::setView (float zoom, float panX, float panY)
{ impl_->zoom = zoom; impl_->panX = panX; impl_->panY = panY; }

void MetalFrameRenderer::setBackgroundColor (float r, float g, float b, float a)
{ impl_->bg[0] = r; impl_->bg[1] = g; impl_->bg[2] = b; impl_->bg[3] = a; }

void MetalFrameRenderer::setPostFx (float bloomIntensity, float bloomThreshold,
                                    float bloomRadius, int tonemap, float exposure)
{
    impl_->bloomIntensity = std::max (bloomIntensity, 0.0f);
    impl_->bloomThreshold = bloomThreshold;
    impl_->bloomRadius = std::max (bloomRadius, 0.0f);
    impl_->tonemap = tonemap >= 0 && tonemap <= 2 ? tonemap : 0;
    impl_->exposure = exposure > 0.0f ? exposure : 1.0f;
}

void MetalFrameRenderer::setInspection (
    int transformClipId, unsigned requestedHandle,
    const videopreview::State& presentation)
{
    impl_->inspectionClipId = transformClipId;
    impl_->inspectionRequestedHandle = requestedHandle;
    impl_->inspectionPresentation = presentation;
    impl_->inspectionAdmitted = false;
    impl_->inspectionRetainedHandle = 0;
}

unsigned MetalFrameRenderer::inspectionHandle() const
{ return impl_->inspectionAdmitted ? impl_->inspectionRetainedHandle : 0; }

bool MetalFrameRenderer::inspectionResourceAdmitted() const
{ return impl_->inspectionAdmitted; }

void MetalFrameRenderer::uploadRgba (unsigned handle, const uint8_t* rgba,
                                     int width, int height, int strideBytes)
{
    if (handle == 0 || rgba == nullptr || width <= 0 || height <= 0) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& source = impl_->sources[handle];
    if (source.frameBlend || source.image.id == 0
        || source.width != width || source.height != height)
    {
        impl_->destroySource (source);
        sg_image_desc imageDesc = {};
        imageDesc.usage.stream_update = true;
        imageDesc.width = width;
        imageDesc.height = height;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA8;
        imageDesc.label = "arbit-metal-frame-source";
        source.image = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.texture.image = source.image;
        source.view = sg_make_view (&viewDesc);
        source.width = width;
        source.height = height;
    }
    if (! resourceValid (sg_query_image_state (source.image))) return;
    const size_t packedBytes = static_cast<size_t> (width) * static_cast<size_t> (height) * 4;
    source.pixels.resize (packedBytes);
    for (int y = 0; y < height; ++y)
        std::memcpy (source.pixels.data() + static_cast<size_t> (y) * width * 4,
                     rgba + static_cast<size_t> (y) * strideBytes,
                     static_cast<size_t> (width) * 4);
    source.dirty = true;
}

bool MetalFrameRenderer::uploadR16 (unsigned handle, const uint16_t* pixels,
                                    int width, int height)
{
    if (handle == 0 || pixels == nullptr || width <= 0 || height <= 0
        || width > 16384 || height > 16384) return false;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& depth = impl_->depths[handle];
    impl_->destroyDepth (depth);
    sg_image_desc desc = {};
    desc.width = width; desc.height = height; desc.pixel_format = SG_PIXELFORMAT_R16;
    desc.data.mip_levels[0] = { pixels, static_cast<size_t> (width) * height * sizeof (uint16_t) };
    desc.label = "arbit-metal-depth-r16";
    depth.image = sg_make_image (&desc);
    sg_view_desc viewDesc = {}; viewDesc.texture.image = depth.image;
    depth.view = sg_make_view (&viewDesc); depth.width = width; depth.height = height;
    if (! resourceValid (sg_query_image_state (depth.image))
        || ! resourceValid (sg_query_view_state (depth.view)))
    {
        impl_->destroyDepth (depth); impl_->depths.erase (handle);
        impl_->error = "Metal R16 depth texture creation failed"; return false;
    }
    return true;
}

void MetalFrameRenderer::setFrameBlend (unsigned handle, unsigned textureA,
                                        unsigned textureB, int width, int height,
                                        float mix)
{
    if (handle == 0 || textureA == 0 || textureB == 0
        || width <= 0 || height <= 0)
        return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& source = impl_->sources[handle];
    if (! source.frameBlend || source.width != width || source.height != height)
    {
        impl_->destroySource (source);
        source.width = width;
        source.height = height;
        source.frameBlend = true;
    }
    source.textureA = textureA;
    source.textureB = textureB;
    source.blendMix = std::clamp (mix, 0.0f, 1.0f);
}

void MetalFrameRenderer::uploadLut3D (unsigned handle, const float* rgbTriples, int size)
{
    if (handle == 0 || rgbTriples == nullptr || size < 2) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& lut = impl_->luts[handle];
    if (! impl_->makeLut (lut, rgbTriples, size, "arbit-metal-frame-lut"))
        impl_->error = "Metal 3D LUT upload failed";
}

void MetalFrameRenderer::deleteTexture (unsigned handle)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto it = impl_->sources.find (handle);
    if (it != impl_->sources.end())
    {
        impl_->destroySource (it->second);
        impl_->sources.erase (it);
    }
    auto lut = impl_->luts.find (handle);
    if (lut != impl_->luts.end())
    {
        impl_->destroyLut (lut->second);
        impl_->luts.erase (lut);
    }
    auto depth = impl_->depths.find (handle);
    if (depth != impl_->depths.end())
    {
        impl_->destroyDepth (depth->second);
        impl_->depths.erase (depth);
    }
}

bool MetalFrameRenderer::setClipShader (int clipId, const std::string& source,
                                        std::string& logOut,
                                        std::vector<GenParam>& paramsOut)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& generator = impl_->generators[clipId];
    if (generator == nullptr) generator = std::make_unique<MetalShaderGenerator>();
    const bool ok = generator->setSource (source);
    logOut = generator->log();
    paramsOut = generator->params();
    return ok;
#else
    (void) clipId; (void) source; (void) paramsOut;
    logOut = "native Metal shader compiler was not linked";
    return false;
#endif
}

void MetalFrameRenderer::clearClipShader (int clipId)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    const auto generator = impl_->generators.find (clipId);
    if (generator == impl_->generators.end()) return;
    if (generator->second != nullptr) generator->second->shutdownUnlocked();
    impl_->generators.erase (generator);
#else
    (void) clipId;
#endif
}

bool MetalFrameRenderer::hasClipShader (int clipId) const
{
#if ARBIT_HAVE_METAL_GENERATORS
    const auto generator = impl_->generators.find (clipId);
    return generator != impl_->generators.end() && generator->second != nullptr
        && generator->second->hasProgram();
#else
    (void) clipId;
    return false;
#endif
}

void MetalFrameRenderer::setClipImage (int clipId, const std::string& name,
                                       const uint8_t* rgba, int width, int height,
                                       int strideBytes)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    const auto generator = impl_->generators.find (clipId);
    if (generator != impl_->generators.end() && generator->second != nullptr)
        generator->second->setImage (name, rgba, width, height, strideBytes);
#else
    (void) clipId; (void) name; (void) rgba; (void) width; (void) height;
    (void) strideBytes;
#endif
}

unsigned MetalFrameRenderer::renderComposite (const arbitgl::GlFuncs* gl,
                                              const LayerDesc* layers, int numLayers,
                                              const ImageLayerDesc* overlays, int numOverlays)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool
    {
        const bool direct = impl_->requestedDirectSurface != nullptr;
        if (! impl_->programsReady || (! direct && impl_->outputTexture == 0)
            || ! impl_->supports (layers, numLayers, overlays, numOverlays))
            return 0;
        ++impl_->frameParity;
        impl_->particleBackend = "none";
        impl_->inspectionAdmitted = false;
        impl_->inspectionRetainedHandle = 0;

#if ARBIT_HAVE_METAL_GENERATORS
        // Submit dynamic generators before the first Sokol draw/update opens
        // this frame's compositor command buffer. Both use Sokol's Metal queue,
        // so commit order provides GPU-side synchronization without a CPU wait.
        std::unordered_map<int, sg_view> generatedViews;
        auto renderGenerator = [&] (const LayerDesc& layer) -> bool
        {
            if (! layer.shaderSource || generatedViews.count (layer.clipId) != 0)
                return true;
            const auto generator = impl_->generators.find (layer.clipId);
            if (generator == impl_->generators.end() || generator->second == nullptr)
                return false;
            sg_view view = {};
            const auto started = std::chrono::steady_clock::now();
            view.id = generator->second->renderViewUnlocked (
                layer.shaderClock, impl_->width, impl_->height,
                layer.audioPresent ? &layer.audioFeatures : nullptr,
                layer.notesPresent ? &layer.noteFeatures : nullptr,
                &layer.genParams);
            if (impl_->visualTelemetry != nullptr)
                impl_->visualTelemetry->recordExecutionObservation(
                    videowire::VisualExecutionKind::generator,
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count()));
            if (view.id == 0)
            {
                impl_->error = generator->second->log();
                return false;
            }
            generatedViews[layer.clipId] = view;
            return true;
        };
        for (int layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        {
            if (! renderGenerator (layers[layerIndex])) return 0;
            if (layers[layerIndex].fromLayer != nullptr
                && ! renderGenerator (*layers[layerIndex].fromLayer)) return 0;
        }
#endif

        // Apply the latest CPU decode for each source once in this sokol frame.
        // Upload calls may outnumber composites while the GL recovery path is
        // active; deferring here avoids sokol's one-update-per-image-per-frame
        // rule without adding extra commits or stalling the decode thread.
        for (auto& item : impl_->sources)
        {
            auto& source = item.second;
            if (! source.dirty) continue;
            sg_image_data data = {};
            data.mip_levels[0] = { source.pixels.data(), source.pixels.size() };
            sg_update_image (source.image, &data);
            source.dirty = false;
        }

        // Tier-1 retiming remains GPU-resident in Metal-only mode. Decoded
        // bracket frames are sampled directly into a reusable RGBA16F target;
        // no GL texture, framebuffer, or CPU blend is created.
        for (auto& item : impl_->sources)
        {
            auto& source = item.second;
            if (! source.frameBlend) continue;
            const auto earlier = impl_->sources.find (source.textureA);
            const auto later = impl_->sources.find (source.textureB);
            if (earlier == impl_->sources.end() || later == impl_->sources.end()
                || earlier->second.frameBlend || later->second.frameBlend)
            {
                impl_->error = "Metal frame blend has invalid source textures";
                return 0;
            }
            if (source.blendTarget.image.id == 0
                && ! impl_->makeTarget (source.blendTarget, source.width, source.height,
                                        "arbit-metal-frame-mix-target"))
            {
                impl_->error = "Metal frame blend target creation failed";
                return 0;
            }
            sg_pass pass = {};
            pass.attachments.colors[0] = source.blendTarget.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->frameMixPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = earlier->second.view;
            bindings.views[1] = later->second.view;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const MetalFrameMixParams params = {
                source.blendMix, { 0.0f, 0.0f, 0.0f } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();
        }

        auto sourceViewFor = [&] (unsigned handle) -> sg_view
        {
            const auto source = impl_->sources.find (handle);
            if (source == impl_->sources.end()) return {};
            return source->second.frameBlend
                ? source->second.blendTarget.texture : source->second.view;
        };

        int read = 0;
        sg_pass clearPass = {};
        clearPass.attachments.colors[0] = impl_->accum[read].attachment;
        clearPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
        clearPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        clearPass.action.colors[0].clear_value = {
            impl_->bg[0], impl_->bg[1], impl_->bg[2], impl_->bg[3] };
        sg_begin_pass (&clearPass);
        sg_end_pass();

        auto drawLayerTo = [&] (const LayerDesc& layer, Impl::Target& target,
                                float opacity, sg_view sourceOverride) -> bool
        {
            if ((layer.texture == 0 && sourceOverride.id == 0
                 && ! layer.particleSource && ! layer.shaderSource)
                || opacity <= 0.0f)
            {
                sg_pass clear = {};
                clear.attachments.colors[0] = target.attachment;
                clear.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                clear.action.colors[0].store_action = SG_STOREACTION_STORE;
                clear.action.colors[0].clear_value = { 0, 0, 0, 0 };
                sg_begin_pass (&clear);
                sg_end_pass();
                return true;
            }
            sg_view sourceView = sourceOverride;
            if (sourceView.id == 0)
            {
                if (layer.shaderSource)
                {
#if ARBIT_HAVE_METAL_GENERATORS
                    const auto generated = generatedViews.find (layer.clipId);
                    if (generated == generatedViews.end()) return false;
                    sourceView = generated->second;
#else
                    return false;
#endif
                }
                else if (layer.particleSource)
                {
                    ParticleParams params;
                    auto value = [&layer] (const char* key, double fallback)
                    {
                        const auto item = layer.genParams.find (key);
                        return item != layer.genParams.end() ? item->second : fallback;
                    };
                    params.count = static_cast<int> (value ("count", 512.0) + 0.5);
                    params.spawnTrack = static_cast<int> (value ("spawnTrack", 0.0) + 0.5);
                    params.size = static_cast<float> (value ("size", 2.0));
                    params.gravity = static_cast<float> (value ("gravity", 0.0));
                    params.force = static_cast<float> (value ("force", 1.0));
                    auto& engine = impl_->particles[layer.clipId];
                    if (engine == nullptr)
                        engine = std::make_unique<MetalParticleEngine>();
                    const auto started = std::chrono::steady_clock::now();
                    sourceView.id = engine->renderMetalViewUnlocked (
                        layer.shaderClock, impl_->width, impl_->height, params,
                        layer.notesPresent ? &layer.noteFeatures : nullptr);
                    if (impl_->visualTelemetry != nullptr)
                    {
                        const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count());
                        impl_->visualTelemetry->recordExecutionObservation(
                            videowire::VisualExecutionKind::particle, elapsed);
                        if (layer.particleNodeId != 0)
                            impl_->visualTelemetry->recordNodeEvaluation(layer.clipId,
                                layer.visualPlanStructuralRevision, layer.particleNodeId, elapsed,
                                layer.visualPlanTelemetryHold);
                    }
                    if (sourceView.id == 0)
                    {
                        impl_->particleBackend = "metal-rejected";
                        impl_->error = engine->log();
                        return false;
                    }
                    impl_->particleBackend = "metal-compute";
                }
                else
                    sourceView = sourceViewFor (layer.texture);
                if (sourceView.id == 0) return false;
            }

            MetalEffectParams effects = makeEffectParams (layer);
            float blurRadius = 0.0f;
            float sharpenAmount = 0.0f;
            bool blurOn = false, sharpenOn = false;
            bool orderedEffectPassOn = layer.lutTexture != 0;
            int orderedEffectBits = 0;
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
            {
                const auto& effect = layer.effects[fx];
                if (! effect.enabled) continue;
                if (effect.type == static_cast<int> (videofx::EffectType::Blur))
                {
                    blurRadius = effect.params[0];
                    blurOn = true;
                }
                else if (effect.type == static_cast<int> (videofx::EffectType::Sharpen))
                {
                    sharpenAmount = effect.params[0];
                    sharpenOn = true;
                }
                else if (metalUvEffectMode (effect.type) >= 0
                         || effect.type == static_cast<int> (videofx::EffectType::FeedbackTrail))
                {
                    orderedEffectPassOn = true;
                    orderedEffectBits |= videofx::kEffectBits[effect.type];
                }
            }

            sg_view processed = sourceView;
            int processedTarget = -1;
            auto nextEffectTarget = [&]() -> int
            {
                processedTarget = processedTarget < 0 ? 0 : processedTarget ^ 1;
                return processedTarget;
            };
            auto filterPass = [&] (sg_pipeline pipeline, const MetalFilterParams& params,
                                  sg_view input, sg_view lut, int targetIndex)
            {
                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (pipeline);
                sg_bindings bindings = {};
                bindings.views[0] = input;
                bindings.views[1] = lut;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
            };

            const int blurBits = videofx::kEffectBits[
                static_cast<int> (videofx::EffectType::Blur)]
                | videofx::kEffectBits[
                    static_cast<int> (videofx::EffectType::Sharpen)];
            const bool colorEffectsPreprocessed =
                (effects.mask & ~(blurBits | orderedEffectBits)) != 0
                && (blurOn || sharpenOn || orderedEffectPassOn);
            if (colorEffectsPreprocessed)
            {
                const int targetIndex = nextEffectTarget();
                const MetalGeometryParams identityGeometry = {
                    { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0 } };
                const MetalMaskParams fullMask = {
                    { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0 };
                effects.mask &= ~blurBits;
                effects.lutEnabled = 0.0f;
                effects.lutSize = 0.0f;
                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->layerPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.views[1] = impl_->identityLut.view;
                bindings.views[2] = processed;
                bindings.views[3] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range geometryRange = { &identityGeometry, sizeof (identityGeometry) };
                const sg_range maskRange = { &fullMask, sizeof (fullMask) };
                const sg_range effectsRange = { &effects, sizeof (effects) };
                sg_apply_uniforms (0, &geometryRange);
                sg_apply_uniforms (1, &maskRange);
                sg_apply_uniforms (2, &effectsRange);
                const MetalDepthFogParams noFog = {};
                const sg_range noFogRange = { &noFog, sizeof (noFog) };
                sg_apply_uniforms (3, &noFogRange);
                sg_draw (0, 6, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }
            if (blurOn && blurRadius > 0.0f)
            {
                for (int pass = 0; pass < 2; ++pass)
                {
                    const int targetIndex = nextEffectTarget();
                    const MetalFilterParams params = {
                        pass == 0 ? 1.0f / impl_->width : 0.0f,
                        pass == 0 ? 0.0f : 1.0f / impl_->height,
                        blurRadius, 0.0f };
                    filterPass (impl_->blurPipeline, params, processed, {}, targetIndex);
                    processed = impl_->effect[targetIndex].texture;
                }
            }
            if (sharpenOn && sharpenAmount > 0.0f)
            {
                const int targetIndex = nextEffectTarget();
                const MetalFilterParams params = {
                    1.0f / impl_->width, 1.0f / impl_->height,
                    sharpenAmount, 0.0f };
                filterPass (impl_->sharpenPipeline, params, processed, {}, targetIndex);
                processed = impl_->effect[targetIndex].texture;
            }
            if (layer.lutTexture != 0 && layer.lutSize >= 2)
            {
                const auto lut = impl_->luts.find (layer.lutTexture);
                if (lut == impl_->luts.end()) return false;
                const int targetIndex = nextEffectTarget();
                const MetalFilterParams params = {
                    0.0f, 0.0f, static_cast<float> (layer.lutSize), 0.0f };
                filterPass (impl_->lutPipeline, params, processed,
                            lut->second.view, targetIndex);
                processed = impl_->effect[targetIndex].texture;
            }

            // Stateless UV effects are intentionally separate, ordered passes.
            // This preserves rack slot order after color/blur/sharpen/LUT.
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
            {
                const auto& effect = layer.effects[fx];
                if (! effect.enabled) continue;
                const int mode = metalUvEffectMode (effect.type);
                if (mode < 0) continue;
                const int targetIndex = nextEffectTarget();
                MetalUvEffectParams params = {
                    static_cast<float> (impl_->width),
                    static_cast<float> (impl_->height),
                    static_cast<float> (layer.timeSec), mode, {} };
                const auto* definition = videofx::effectDefFor (effect.type);
                const int count = definition != nullptr
                    ? std::min (definition->paramCount, 4) : 0;
                for (int value = 0; value < count; ++value)
                    params.values[value] = effect.params[value];

                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->uvEffectPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }

            int feedbackSlot = -1;
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
                if (layer.effects[fx].enabled
                    && layer.effects[fx].type
                        == static_cast<int> (videofx::EffectType::FeedbackTrail))
                {
                    feedbackSlot = fx;
                    break;
                }
            if (feedbackSlot >= 0)
            {
                const int key = (layer.clipId << 8) | (feedbackSlot & 0xff);
                auto [historyIt, inserted] = impl_->feedback.try_emplace (key);
                auto& history = historyIt->second;
                if (inserted || layer.feedbackHistoryReset)
                {
                    if (! inserted)
                    {
                        impl_->destroyTarget(history.target[0]);
                        impl_->destroyTarget(history.target[1]);
                    }
                    history.current = 0;
                    history.ready = false;
                    if (! impl_->makeTarget (history.target[0], impl_->width, impl_->height,
                                             "arbit-metal-frame-feedback-a")
                        || ! impl_->makeTarget (history.target[1], impl_->width, impl_->height,
                                                "arbit-metal-frame-feedback-b"))
                    {
                        impl_->destroyTarget (history.target[0]);
                        impl_->destroyTarget (history.target[1]);
                        impl_->feedback.erase (historyIt);
                        impl_->error = "Metal feedback history creation failed";
                        return false;
                    }
                    for (int buffer = 0; buffer < 2; ++buffer)
                    {
                        sg_pass clear = {};
                        clear.attachments.colors[0] = history.target[buffer].attachment;
                        clear.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                        clear.action.colors[0].store_action = SG_STOREACTION_STORE;
                        clear.action.colors[0].clear_value = { 0, 0, 0, 0 };
                        sg_begin_pass (&clear);
                        sg_end_pass();
                    }
                }
                if (layer.feedbackHistoryHold && history.ready)
                {
                    processed = history.target[history.current].texture;
                }
                else
                {
                const int readHistory = history.ready ? history.current : 0;
                const int writeHistory = history.ready ? (history.current ^ 1) : 1;
                const auto& effect = layer.effects[feedbackSlot];
                const MetalFeedbackParams params = {
                    effect.params[0], effect.params[1], effect.params[2], 0.0f };
                sg_pass pass = {};
                pass.attachments.colors[0] = history.target[writeHistory].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->feedbackPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.views[1] = history.target[readHistory].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = history.target[writeHistory].texture;
                history.current = writeHistory;
                history.ready = true;
                }
            }

            const int displayWidth = impl_->presentWidth > 0 ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0 ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            const float refAspect = impl_->canvasWidth > 0 && impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasWidth) / impl_->canvasHeight : outAspect;
            const float videoAspect = layer.texHeight > 0
                ? static_cast<float> (layer.texWidth) / layer.texHeight : 1.0f;
            float lbx = 1.0f, lby = 1.0f;
            if (videoAspect > refAspect) lby = refAspect / videoAspect;
            else                         lbx = videoAspect / refAspect;

            MetalGeometryParams geometry = {};
            makeTransform (layer.translateX, layer.translateY, layer.rotationDeg,
                           layer.scale * lbx, layer.scale * lby, geometry.transform);
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            geometry.crop[0] = layer.cropLeft; geometry.crop[1] = layer.cropRight;
            geometry.crop[2] = layer.cropTop; geometry.crop[3] = layer.cropBottom;
            const MetalMaskParams mask = {
                { layer.maskCx, layer.maskCy, layer.maskW, layer.maskH },
                opacity, layer.maskType, layer.maskFeather, layer.maskInvert ? 1 : 0,
                { layer.matteBlack, layer.matteWhite, layer.matteErodeDilate,
                  layer.matteFeather },
                { layer.matteWidth > 0 ? 1.0f / layer.matteWidth : 0.0f,
                  layer.matteHeight > 0 ? 1.0f / layer.matteHeight : 0.0f },
                layer.matteChoke,
                (layer.matteApply ? 1 : 0) | (layer.matteInvert ? 2 : 0)
                    | ((layer.matteCombineMode + 1) << 2) };
            const MetalEffectParams neutralEffects = neutralEffectParams();
            const MetalEffectParams& layerEffects = colorEffectsPreprocessed
                ? neutralEffects : effects;
            const int depthMode = layer.depthEffect != 0 ? layer.depthEffect : (layer.depthFog ? 1 : 0);
            const MetalDepthFogParams fog = depthMode <= 1 ? MetalDepthFogParams {
                { layer.fogNear, layer.fogFar, layer.fogDensity, static_cast<float>(depthMode) },
                { layer.fogRed, layer.fogGreen, layer.fogBlue, layer.fogAlpha } }
                : MetalDepthFogParams {
                    { layer.depthParam0, layer.depthParam1, layer.depthParam2, static_cast<float>(depthMode) },
                    { layer.depthColorRed, layer.depthColorGreen, layer.depthColorBlue, 1.0f } };

            sg_pass layerPass = {};
            layerPass.attachments.colors[0] = target.attachment;
            layerPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            layerPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            layerPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&layerPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = processed;
            bindings.views[1] = impl_->identityLut.view;
            bindings.views[2] = layer.matteApply
                ? sourceViewFor(layer.matteTexture) : processed;
            if (bindings.views[2].id == 0) return false;
            bindings.views[3] = layer.matteCombineMode >= 0
                ? sourceViewFor(layer.matteTextureB) : processed;
            if (bindings.views[3].id == 0) return false;
            if (depthMode != 0)
            {
                const auto depth = impl_->depths.find (layer.depthTexture);
                if (depth == impl_->depths.end()
                    || depth->second.width != layer.depthWidth
                    || depth->second.height != layer.depthHeight)
                    return false;
                bindings.views[4] = depth->second.view;
            }
            else bindings.views[4] = processed;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &layerEffects, sizeof (layerEffects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const sg_range fogRange = { &fog, sizeof (fog) };
            sg_apply_uniforms (3, &fogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();
            if (&target == &impl_->layerTarget
                && layer.clipId == impl_->inspectionClipId
                && ! layer.inspectionDrawShapeOutput)
            {
                sg_pass retainPass = {};
                retainPass.attachments.colors[0] = impl_->inspectionTarget.attachment;
                retainPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                retainPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&retainPass);
                sg_apply_pipeline (impl_->blurPipeline);
                sg_bindings retainBindings = {};
                retainBindings.views[0] = impl_->layerTarget.texture;
                retainBindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&retainBindings);
                const MetalFilterParams retainParams = { 0, 0, 0, 0 };
                const sg_range retainRange = { &retainParams, sizeof (retainParams) };
                sg_apply_uniforms (0, &retainRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                impl_->inspectionRetainedHandle = impl_->inspectionTarget.texture.id;
                impl_->inspectionAdmitted = true;
            }
            return true;
        };

        std::vector<bool> overlayComposited (
            static_cast<size_t> (std::max (numOverlays, 0)), false);
        auto compositeOverlay = [&] (int index) -> bool
        {
            const auto& overlay = overlays[index];
            if (overlay.texture == 0 || overlay.opacity <= 0.0f) return true;
            const sg_view overlaySource = sourceViewFor (overlay.texture);
            if (overlaySource.id == 0) return false;

            const float refWidth = impl_->canvasWidth > 0
                ? static_cast<float> (impl_->canvasWidth)
                : static_cast<float> (impl_->width);
            const float refHeight = impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasHeight)
                : static_cast<float> (impl_->height);
            MetalGeometryParams geometry = {};
            makeTransform (overlay.posX, overlay.posY, 0.0f,
                           overlay.width / std::max (refWidth, 1.0f),
                           overlay.height / std::max (refHeight, 1.0f),
                           geometry.transform);
            const int displayWidth = impl_->presentWidth > 0
                ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0
                ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            const MetalMaskParams mask = {
                { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0 };
            const MetalEffectParams effects = neutralEffectParams();

            sg_pass overlayPass = {};
            overlayPass.attachments.colors[0] = impl_->layerTarget.attachment;
            overlayPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            overlayPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            overlayPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&overlayPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings overlayBindings = {};
            overlayBindings.views[0] = overlaySource;
            overlayBindings.views[1] = impl_->identityLut.view;
            overlayBindings.views[2] = overlaySource;
            overlayBindings.views[3] = overlaySource;
            // The shared fragment samples depth before checking its mode.
            overlayBindings.views[4] = overlaySource;
            overlayBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&overlayBindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &effects, sizeof (effects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const MetalDepthFogParams noFog = {};
            const sg_range noFogRange = { &noFog, sizeof (noFog) };
            sg_apply_uniforms (3, &noFogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { overlay.opacity, 0, { 0, 0 } };
            const sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;
            overlayComposited[static_cast<size_t> (index)] = true;
            return true;
        };

        for (int i = 0; i < numLayers; ++i)
        {
            const auto& layer = layers[i];
            if (layer.isAdjustment)
            {
                if (layer.opacity <= 0.0f) continue;
                for (int overlay = 0; overlay < numOverlays; ++overlay)
                    if (! overlayComposited[static_cast<size_t> (overlay)]
                        && overlays[overlay].ownerClipId == layer.clipId
                        && ! compositeOverlay (overlay))
                        return 0;
                LayerDesc adjustment = layer;
                adjustment.texWidth = impl_->width;
                adjustment.texHeight = impl_->height;
                adjustment.transitionType = 0;
                if (! drawLayerTo (adjustment, impl_->layerTarget, 1.0f,
                                   impl_->accum[read].texture))
                    return 0;

                sg_pass blendPass = {};
                blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&blendPass);
                sg_apply_pipeline (impl_->blendPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = impl_->layerTarget.texture;
                bindings.views[1] = impl_->accum[read].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const MetalBlendParams blend = {
                    layer.opacity, 0, { 0.0f, 0.0f } };
                const sg_range range = { &blend, sizeof (blend) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
                continue;
            }
            if (layer.transitionType != 0)
            {
                const LayerDesc empty;
                const LayerDesc& from = layer.fromLayer != nullptr ? *layer.fromLayer : empty;
                if (! drawLayerTo (from, impl_->transitionFrom, from.opacity, {})
                    || ! drawLayerTo (layer, impl_->layerTarget, layer.opacity, {}))
                    return 0;
                sg_pass transitionPass = {};
                transitionPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                transitionPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                transitionPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&transitionPass);
                sg_apply_pipeline (impl_->transitionPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = impl_->transitionFrom.texture;
                bindings.views[1] = impl_->layerTarget.texture;
                bindings.views[2] = impl_->accum[read].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const MetalTransitionParams params = {
                    std::clamp (layer.transitionProgress, 0.0f, 1.0f),
                    metalTransitionType (layer.transitionType), layer.blendMode, 0.0f };
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
                continue;
            }
            if ((layer.texture == 0 && ! layer.particleSource && ! layer.shaderSource)
                || layer.opacity <= 0.0f)
                continue;
            if (! drawLayerTo (layer, impl_->layerTarget, 1.0f, {})) return 0;

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { layer.opacity, layer.blendMode, { 0, 0 } };
            sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;

            if (layer.drawShape)
            {
                const auto drawShapeStarted = std::chrono::steady_clock::now();
                sg_pass shapePass = {};
                shapePass.attachments.colors[0] = impl_->layerTarget.attachment;
                shapePass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                shapePass.action.colors[0].store_action = SG_STOREACTION_STORE;
                shapePass.action.colors[0].clear_value = { 0, 0, 0, 0 };
                sg_begin_pass (&shapePass);
                sg_apply_pipeline (impl_->drawShapePipeline);
                const MetalDrawShapeParams shape = {
                    { layer.drawShapeCx, layer.drawShapeCy,
                      layer.drawShapeEllipse ? -std::max(layer.drawShapeW, 0.0f)
                                               : std::max(layer.drawShapeW, 0.0f),
                      std::max(layer.drawShapeH, 0.0f) },
                    { std::clamp(layer.drawShapeR, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeG, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeB, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeA, 0.0f, 1.0f) } };
                const sg_range shapeRange = { &shape, sizeof (shape) };
                sg_apply_uniforms (0, &shapeRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                if (impl_->visualTelemetry != nullptr && layer.drawShapeNodeId != 0)
                    impl_->visualTelemetry->recordNodeEvaluation(layer.clipId,
                        layer.visualPlanStructuralRevision, layer.drawShapeNodeId,
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - drawShapeStarted).count()),
                        layer.visualPlanTelemetryHold);
                if (layer.inspectionDrawShapeOutput
                    && layer.clipId == impl_->inspectionClipId)
                {
                    sg_pass retainPass = {};
                    retainPass.attachments.colors[0] = impl_->inspectionTarget.attachment;
                    retainPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                    retainPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                    sg_begin_pass (&retainPass);
                    sg_apply_pipeline (impl_->blurPipeline);
                    sg_bindings retainBindings = {};
                    retainBindings.views[0] = impl_->layerTarget.texture;
                    retainBindings.samplers[0] = impl_->sampler;
                    sg_apply_bindings (&retainBindings);
                    const MetalFilterParams retainParams = { 0, 0, 0, 0 };
                    const sg_range retainRange = { &retainParams, sizeof (retainParams) };
                    sg_apply_uniforms (0, &retainRange);
                    sg_draw (0, 3, 1);
                    sg_end_pass();
                    impl_->inspectionRetainedHandle = impl_->inspectionTarget.texture.id;
                    impl_->inspectionAdmitted = true;
                }

                sg_pass shapeBlendPass = {};
                shapeBlendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                shapeBlendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                shapeBlendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&shapeBlendPass);
                sg_apply_pipeline (impl_->blendPipeline);
                sg_bindings shapeBindings = {};
                shapeBindings.views[0] = impl_->layerTarget.texture;
                shapeBindings.views[1] = impl_->accum[read].texture;
                shapeBindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&shapeBindings);
                const MetalBlendParams normal = { 1.0f, 0, { 0, 0 } };
                const sg_range normalRange = { &normal, sizeof (normal) };
                sg_apply_uniforms (0, &normalRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
            }
        }

        // Remaining project-level and normal-clip overlays composite on top.
        // Adjustment-owned overlays were inserted immediately before their
        // owner so the adjustment transforms/effects them with the video.
        for (int i = 0; i < numOverlays; ++i)
        {
            if (overlayComposited[static_cast<size_t> (i)]) continue;
            const auto& overlay = overlays[i];
            if (overlay.texture == 0 || overlay.opacity <= 0.0f) continue;
            const sg_view overlaySource = sourceViewFor (overlay.texture);
            if (overlaySource.id == 0) return 0;

            const float refWidth = impl_->canvasWidth > 0
                ? static_cast<float> (impl_->canvasWidth)
                : static_cast<float> (impl_->width);
            const float refHeight = impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasHeight)
                : static_cast<float> (impl_->height);
            MetalGeometryParams geometry = {};
            makeTransform (overlay.posX, overlay.posY, 0.0f,
                           overlay.width / std::max (refWidth, 1.0f),
                           overlay.height / std::max (refHeight, 1.0f),
                           geometry.transform);
            const int displayWidth = impl_->presentWidth > 0
                ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0
                ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            const MetalMaskParams mask = {
                { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0 };
            const MetalEffectParams effects = neutralEffectParams();

            sg_pass overlayPass = {};
            overlayPass.attachments.colors[0] = impl_->layerTarget.attachment;
            overlayPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            overlayPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            overlayPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&overlayPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings overlayBindings = {};
            overlayBindings.views[0] = overlaySource;
            overlayBindings.views[1] = impl_->identityLut.view;
            overlayBindings.views[2] = overlaySource;
            overlayBindings.views[3] = overlaySource;
            // The shared fragment samples depth before checking its mode.
            overlayBindings.views[4] = overlaySource;
            overlayBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&overlayBindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &effects, sizeof (effects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const MetalDepthFogParams noFog = {};
            const sg_range noFogRange = { &noFog, sizeof (noFog) };
            sg_apply_uniforms (3, &noFogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { overlay.opacity, 0, { 0, 0 } };
            const sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;
        }

        sg_view finalView = impl_->accum[read].texture;
        const bool bloomOn = impl_->bloomIntensity > 0.0f
            && impl_->bloomRadius > 0.0f;
        const bool tonemapOn = impl_->tonemap != 0 || impl_->exposure != 1.0f;
        if (bloomOn)
        {
            const MetalPostParams threshold = {
                impl_->bloomThreshold, 0.0f, 1.0f, 0 };
            sg_pass pass = {};
            pass.attachments.colors[0] = impl_->effect[0].attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->bloomThresholdPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = finalView;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range range = { &threshold, sizeof (threshold) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();

            const float radius = std::min (impl_->bloomRadius, 20.0f);
            const int passes = std::max (1, std::min (5,
                static_cast<int> (std::ceil (impl_->bloomRadius / 10.0f))));
            for (int iteration = 0; iteration < passes; ++iteration)
            {
                for (int direction = 0; direction < 2; ++direction)
                {
                    const int source = direction == 0 ? 0 : 1;
                    const int target = direction == 0 ? 1 : 0;
                    const MetalFilterParams params = {
                        direction == 0 ? 1.0f / impl_->width : 0.0f,
                        direction == 0 ? 0.0f : 1.0f / impl_->height,
                        radius, 0.0f };
                    sg_pass blurPass = {};
                    blurPass.attachments.colors[0] = impl_->effect[target].attachment;
                    blurPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                    blurPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                    sg_begin_pass (&blurPass);
                    sg_apply_pipeline (impl_->blurPipeline);
                    sg_bindings blurBindings = {};
                    blurBindings.views[0] = impl_->effect[source].texture;
                    blurBindings.samplers[0] = impl_->sampler;
                    sg_apply_bindings (&blurBindings);
                    const sg_range blurRange = { &params, sizeof (params) };
                    sg_apply_uniforms (0, &blurRange);
                    sg_draw (0, 3, 1);
                    sg_end_pass();
                }
            }
        }
        if (bloomOn || tonemapOn)
        {
            const MetalPostParams params = {
                impl_->bloomThreshold, bloomOn ? impl_->bloomIntensity : 0.0f,
                impl_->exposure, impl_->tonemap };
            sg_pass pass = {};
            pass.attachments.colors[0] = impl_->layerTarget.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->postCombinePipeline);
            sg_bindings bindings = {};
            bindings.views[0] = finalView;
            bindings.views[1] = bloomOn ? impl_->effect[0].texture : finalView;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();
            finalView = impl_->layerTarget.texture;
        }

        Impl::DirectOutput* directOutput = direct
            ? impl_->directOutput (impl_->requestedDirectSurface,
                                   impl_->requestedDirectWidth,
                                   impl_->requestedDirectHeight)
            : nullptr;
        if (direct && directOutput == nullptr)
        {
            impl_->error = "Metal direct IOSurface target creation failed";
            return 0;
        }
        sg_view previewView = {};
        if (impl_->inspectionClipId >= 0 && impl_->inspectionAdmitted)
            previewView = impl_->inspectionTarget.texture;
        else if (impl_->inspectionRequestedHandle == 0xffffffffu)
            previewView = finalView;
        else if (impl_->inspectionRequestedHandle != 0)
            previewView = sourceViewFor (impl_->inspectionRequestedHandle);
        if (previewView.id != 0)
        {
            impl_->inspectionAdmitted = true;
            impl_->inspectionRetainedHandle = previewView.id;
        }

        sg_pass outputPass = {};
        outputPass.attachments.colors[0] = direct
            ? directOutput->attachment : impl_->outputAttachment;
        outputPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
        outputPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        sg_begin_pass (&outputPass);
        const bool showPreview = direct && impl_->inspectionAdmitted
            && impl_->inspectionPresentation.layout != videopreview::Layout::hidden;
        const bool frameCanvas = direct && ! showPreview
            && impl_->canvasWidth > 0 && impl_->canvasHeight > 0;
        sg_apply_pipeline (showPreview ? impl_->previewPipeline
                                      : (frameCanvas ? impl_->canvasPipeline : impl_->blitPipeline));
        sg_bindings outputBindings = {};
        outputBindings.views[0] = finalView;
        if (showPreview) outputBindings.views[1] = previewView;
        outputBindings.samplers[0] = impl_->sampler;
        sg_apply_bindings (&outputBindings);
        if (showPreview)
        {
            const auto& state = impl_->inspectionPresentation;
            const MetalPreviewParams params = {
                state.zoom, 0.0f, { state.panX, state.panY }, state.split,
                static_cast<int32_t> (state.layout),
                static_cast<int32_t> (state.background), 0.0f, { 0, 0 } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
        }
        else if (frameCanvas)
        {
            const float outAspect = static_cast<float> (impl_->requestedDirectWidth)
                                  / std::max (impl_->requestedDirectHeight, 1);
            float zx = impl_->zoom, zy = impl_->zoom;
            const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                     / impl_->canvasHeight;
            if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
            else                          zx *= canvasAspect / outAspect;
            const MetalCanvasParams params = {
                { (impl_->panX - zx + 1.0f) * 0.5f,
                  (impl_->panY - zy + 1.0f) * 0.5f,
                  (impl_->panX + zx + 1.0f) * 0.5f,
                  (impl_->panY + zy + 1.0f) * 0.5f },
                { 1.0f / impl_->requestedDirectWidth,
                  1.0f / impl_->requestedDirectHeight }, { 0.0f, 0.0f } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
        }
        sg_draw (0, 3, 1);
        sg_end_pass();
        sg_commit();

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
        id<MTLCommandBuffer> fence = [queue commandBuffer];
        [fence commit];
        [fence waitUntilCompleted];
        if (direct)
        {
            // The persistent consumer imports this IOSurface once and samples
            // it from another Metal device/command queue. A completed producer
            // command buffer orders execution, while IOSurfaceLock/Unlock is
            // the public cross-process cache-coherence barrier for the shared
            // allocation. No pixels are read or copied here.
            IOSurfaceRef sharedSurface = static_cast<IOSurfaceRef> (
                impl_->requestedDirectSurface);
            if (sharedSurface == nullptr
                || IOSurfaceLock (sharedSurface, kIOSurfaceLockReadOnly, nullptr) != 0)
            {
                impl_->error = "Metal IOSurface coherence lock failed";
                return 0;
            }
            IOSurfaceUnlock (sharedSurface, kIOSurfaceLockReadOnly, nullptr);
            impl_->error.clear();
            return 1;
        }
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->rectangleFbo);
        gl->BindFramebuffer (GL_DRAW_FRAMEBUFFER, impl_->outputFbo);
        gl->BlitFramebuffer (0, 0, impl_->width, impl_->height,
                             0, 0, impl_->width, impl_->height,
                             GL_COLOR_BUFFER_BIT, GL_NEAREST);
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        impl_->error.clear();
        return impl_->outputTexture;
    }
}

bool MetalFrameRenderer::renderCompositeToIOSurface (
    const arbitgl::GlFuncs* gl, void* ioSurface, int width, int height,
    const LayerDesc* layers, int numLayers,
    const ImageLayerDesc* overlays, int numOverlays)
{
    if (ioSurface == nullptr || width <= 0 || height <= 0) return false;
    impl_->requestedDirectSurface = ioSurface;
    impl_->requestedDirectWidth = width;
    impl_->requestedDirectHeight = height;
    const bool rendered = renderComposite (
        gl, layers, numLayers, overlays, numOverlays) != 0;
    impl_->requestedDirectSurface = nullptr;
    impl_->requestedDirectWidth = impl_->requestedDirectHeight = 0;
    return rendered;
}

void MetalFrameRenderer::clearDirectOutputs()
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    impl_->clearDirectOutputs();
}

bool MetalFrameRenderer::ready() const
{
    return impl_ != nullptr && impl_->programsReady
        && impl_->accum[0].image.id != 0
        && (impl_->directOnly || impl_->outputTexture != 0);
}

const std::string& MetalFrameRenderer::lastError() const
{ return impl_->error; }

const std::string& MetalFrameRenderer::particleBackend() const
{ return impl_->particleBackend; }

void MetalFrameRenderer::setVisualTelemetryOwner (videowire::VisualPlanTelemetry* owner)
{ impl_->visualTelemetry = owner; }

} // namespace videorender
#endif
