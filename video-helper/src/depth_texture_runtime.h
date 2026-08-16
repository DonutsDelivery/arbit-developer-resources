#pragma once

#include "renderer.h"
#include "visual_plan_executor.h"
#include "depth_texture_state.h"
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#include "depth_cache.h"
#endif

#include <string>
#include <cstdint>
#include <vector>

namespace videohelper
{
template <typename Renderer>
inline bool prepareDepthTexture(const std::string& canonicalRoot,
                                const std::vector<videowire::CompiledVisualLayerPlan>& plans,
                                int clipId, double exactSourcePts,
                                std::uint64_t helperGeneration,
                                Renderer& renderer,
                                DepthTextureState& state,
                                videorender::LayerDesc& layer,
                                std::string& error)
{
    state.beginHelperGeneration(renderer, helperGeneration);
    videowire::ExecutableDepthPayload payload;
    const bool hasDepth = videowire::visualPlanDepthBinding(plans, clipId, payload, error);
    if (!hasDepth)
        return error.empty();
    if (canonicalRoot.empty())
    {
        state.clear(renderer); error = "private depth cache root is unavailable"; return false;
    }
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    DepthCacheBinding binding;
    binding.key=payload.cacheKey; binding.receipt=payload.contentReceipt; binding.version=payload.version;
    binding.prefix=payload.prefix; binding.extension=payload.extension; binding.firstFrame=payload.firstFrame;
    binding.digits=payload.digits; binding.frames=payload.frames; binding.width=payload.width;
    binding.height=payload.height; binding.fps=payload.fps;
    const int wanted = depthFrameIndexForPts(exactSourcePts, binding.fps, binding.frames);
    if (wanted < 0) { state.clear(renderer); error="invalid depth frame PTS"; return false; }
    if (state.texture == 0 || state.frameIndex != wanted || state.receipt != binding.receipt)
    {
        // Receipt replacement is a resource identity change, not an in-place
        // texture update. Delete the old admitted handle before fresh admission
        // so stale content can never remain renderable after a failed reopen.
        state.invalidateReplacedReceipt(renderer, binding.receipt);
        auto frame = admitDepthFrame(canonicalRoot, binding, exactSourcePts, error);
        if (!frame) { state.clear(renderer); return false; }
        const unsigned uploaded = renderer.uploadR16(frame->pixels().data(), frame->width(), frame->height(),
                                                     state.texture);
        if (uploaded == 0) { state.clear(renderer); error="native R16 depth upload is unavailable"; return false; }
        state.texture=uploaded; state.width=frame->width(); state.height=frame->height();
        state.frameIndex=frame->index(); state.receipt=binding.receipt;
    }
    layer.depthTexture=state.texture; layer.depthWidth=state.width; layer.depthHeight=state.height;
    return true;
#else
    (void)exactSourcePts; (void)payload; (void)layer;
    state.clear(renderer); error="native depth fog execution is unavailable on this platform"; return false;
#endif
}
} // namespace videohelper
