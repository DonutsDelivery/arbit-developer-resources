#pragma once

#include <cstdint>
#include <string>

namespace videohelper
{
struct DepthTextureState
{
    unsigned texture = 0;
    int width = 0, height = 0, frameIndex = -1;
    std::string receipt;
    std::uint64_t helperGeneration = 0;

    template <typename Renderer>
    void clear(Renderer& renderer)
    {
        renderer.deleteTexture(texture);
        texture = 0; width = height = 0; frameIndex = -1; receipt.clear();
    }

    template <typename Renderer>
    void beginHelperGeneration(Renderer& renderer, std::uint64_t generation)
    {
        if (helperGeneration == generation)
            return;
        clear(renderer);
        helperGeneration = generation;
    }

    template <typename Renderer>
    void invalidateReplacedReceipt(Renderer& renderer, const std::string& nextReceipt)
    {
        if (texture != 0 && receipt != nextReceipt)
            clear(renderer);
    }
};
} // namespace videohelper
