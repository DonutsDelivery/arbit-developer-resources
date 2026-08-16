#pragma once

#include "render_snapshot.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace videohelper
{
struct SnapshotClipIdentity
{
    int clipId = -1;
    uint64_t revision = 0;
    std::string mode;
    bool compiledExportable = false;
};

inline bool validateSnapshotClipIdentities (
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    const std::vector<SnapshotClipIdentity>& records, std::string& error)
{
    if (plans.size() != records.size())
    {
        error = "snapshot clip revision records do not match compiled plans";
        return false;
    }
    for (const auto& plan : plans)
    {
        const auto record = std::find_if(records.begin(), records.end(), [&plan](const auto& value)
        { return value.clipId == plan.clipId; });
        if (record == records.end() || ! record->compiledExportable
            || record->mode != plan.identityMode || record->revision != plan.structuralRevision
            || (record->mode == "authoredGraph" && record->revision == 0)
            || (record->mode == "transientLegacyProjection" && record->revision != 0))
        {
            error = "snapshot clip revision record is invalid or not exportable";
            return false;
        }
    }
    return true;
}

inline bool validateCompositeProbeContract (
    const videowire::ResolvedVisualSnapshot& snapshot,
    uint64_t snapshotGeneration, double timelineSec, int width, int height,
    double fps, std::string& error)
{
    error.clear();
    if (! std::isfinite (timelineSec) || timelineSec < 0.0)
    {
        error = "timelineSec must be finite and non-negative";
        return false;
    }
    if (! std::isfinite (fps) || fps <= 0.0)
    {
        error = "fps must be finite and positive";
        return false;
    }
    if (width < 1 || height < 1 || width > 1920 || height > 1920)
    {
        error = "width/height must be between 1 and 1920";
        return false;
    }
    if (snapshotGeneration == 0)
    {
        error = "composite probe requires a processor-owned snapshot generation";
        return false;
    }
    if (snapshot.segments.empty() || snapshot.visualLayerPlans.empty())
    {
        error = "composite probe requires a compiled production snapshot";
        return false;
    }
    for (const auto& plan : snapshot.visualLayerPlans)
        if (! plan.producerValidated
            || (plan.identityMode != "authoredGraph" && plan.identityMode != "transientLegacyProjection")
            || (plan.identityMode == "authoredGraph" && plan.structuralRevision == 0)
            || (plan.identityMode == "transientLegacyProjection" && plan.structuralRevision != 0))
        {
            error = "current authored clip plan is invalid or not exportable in this snapshot";
            return false;
        }
    return true;
}
} // namespace videohelper
