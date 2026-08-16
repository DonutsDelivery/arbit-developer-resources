#pragma once

#include <string>

namespace videowire
{
enum class SourceKind
{
    Unspecified,
    Media,
    Shader,
    Particles,
    Adjustment,
};

inline SourceKind sourceKindFromWire (const std::string& value) noexcept
{
    if (value == "media")      return SourceKind::Media;
    if (value == "shader")     return SourceKind::Shader;
    if (value == "particles")  return SourceKind::Particles;
    if (value == "adjustment") return SourceKind::Adjustment;
    return SourceKind::Unspecified;
}

inline const char* sourceKindToWire (SourceKind value) noexcept
{
    switch (value)
    {
        case SourceKind::Media:      return "media";
        case SourceKind::Shader:     return "shader";
        case SourceKind::Particles:  return "particles";
        case SourceKind::Adjustment: return "adjustment";
        default:                     return "unspecified";
    }
}

inline SourceKind resolveSourceKind (SourceKind structuredKind,
                                     bool structuredAdjustment,
                                     const std::string& legacyPath) noexcept
{
    // Structured source identity is authoritative. Compatibility fields and
    // sentinels are consulted only when the typed field is absent.
    if (structuredKind != SourceKind::Unspecified)
        return structuredKind;
    if (structuredAdjustment)
        return SourceKind::Adjustment;
    if (legacyPath.rfind ("gen://adjustment", 0) == 0)
        return SourceKind::Adjustment;
    if (legacyPath.rfind ("gen://particles", 0) == 0)
        return SourceKind::Particles;
    if (legacyPath.rfind ("gen://shader", 0) == 0)
        return SourceKind::Shader;
    return SourceKind::Media;
}

template <typename SegmentRange, typename Segment>
const Segment* resolveTransitionFrom (const SegmentRange& segments,
                                      const Segment& current,
                                      double abutEpsilon = 1.0e-3) noexcept
{
    const bool hasBinding = current.transitionFromClipId >= 0
                         || current.transitionToClipId >= 0;
    const Segment* best = nullptr;
    if (hasBinding)
    {
        if (current.transitionToClipId >= 0
            && current.transitionToClipId != current.clipId)
            return nullptr;
        for (const auto& candidate : segments)
            if (&candidate != &current
                && candidate.clipId == current.transitionFromClipId
                && candidate.displayStartSec <= current.displayStartSec
                && (best == nullptr
                    || candidate.displayStartSec > best->displayStartSec))
                best = &candidate;
        return best;
    }

    for (const auto& candidate : segments)
        if (&candidate != &current && candidate.trackLayer == current.trackLayer
            && candidate.displayStartSec < current.displayStartSec
            && (best == nullptr
                || candidate.displayStartSec > best->displayStartSec))
            best = &candidate;
    if (best == nullptr)
        return nullptr;
    const double duration = (best->outSec - best->inSec)
                          / (best->rate > 1.0e-9 ? best->rate : 1.0e-9);
    return best->displayStartSec + duration >= current.displayStartSec - abutEpsilon
         ? best : nullptr;
}
} // namespace videowire
