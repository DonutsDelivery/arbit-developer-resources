#pragma once

#include "render_snapshot.h"
#include "depth_payload_schema.h"

#include <cstdint>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace videowire
{
template <typename Json, typename ShaderResolver>
bool parseSnapshotJson (const Json& object, ShaderResolver&& resolveShader,
                        ResolvedVisualSnapshot& result, std::string& error)
{
    std::vector<RawRenderSegment> raw;
    std::vector<CompiledVisualLayerPlan> plans;
    std::vector<VisualEventScheduleBinding> eventSchedules;
    std::unordered_map<std::string, std::string> matteAuthorities;
    if (object.contains ("segments") && object["segments"].is_array())
        for (const auto& value : object["segments"])
        {
            RawRenderSegment segment;
            segment.sourcePath = value.value ("sourcePath", "");
            segment.hasStructuredSourceKind = value.contains ("sourceKind");
            if (segment.hasStructuredSourceKind)
                segment.sourceKind = sourceKindFromWire (value.value ("sourceKind", ""));
            segment.clipId = value.value ("clipId", 0);
            segment.trackLayer = value.value ("trackLayer", 0);
            segment.isAdjustment = value.value ("isAdjustment", false);
            segment.inSec = value.value ("inSec", 0.0);
            segment.outSec = value.value ("outSec", 0.0);
            segment.rate = value.value ("rate", 1.0);
            segment.displayStartSec = value.value ("displayStartSec", 0.0);
            segment.clockStartSec = value.value ("clockStartSec", -1.0);
            segment.clockDurationSec = value.value ("clockDurationSec", 0.0);

            segment.retimeQuality = value.value ("retimeQuality", 0);
            segment.sourceFps = value.value ("sourceFps", 0.0);
            segment.seqStart = value.value ("seqStart", -1);
            segment.shaderSource = resolveShader (value);
            if (value.contains("genImages") && value["genImages"].is_object())
                for (auto it = value["genImages"].begin(); it != value["genImages"].end(); ++it)
                    if (it.value().is_string())
                        segment.genImages[it.key()] = it.value().template get<std::string>();
            segment.matteCacheKey = value.value ("matteCacheKey", std::string {});
            segment.matteAssetId = value.value ("matteAssetId", std::string {});
            segment.matteAssetVersion = value.value("matteAssetVersion", std::string {});
            segment.matteContentReceipt = value.value("matteContentReceipt", std::string {});
            segment.matteState = value.value ("matteState", std::string {});
            segment.matteFramePrefix = value.value("matteFramePrefix", std::string {});
            segment.matteFrameExtension = value.value("matteFrameExtension", std::string {});
            segment.matteFirstFrame = value.value("matteFirstFrame", 0);
            segment.matteFrameDigits = value.value("matteFrameDigits", 0);
            segment.matteFps = value.value ("matteFps", 0.0);
            segment.matteFrames = value.value ("matteFrames", 0);
            const bool hasMatteAddition = value.contains("matteCacheKey") || value.contains("matteAssetId")
                || value.contains("matteState") || value.contains("matteFps") || value.contains("matteFrames")
                || value.contains("matteAssetVersion") || value.contains("matteContentReceipt")
                || value.contains("matteFramePrefix") || value.contains("matteFrameExtension")
                || value.contains("matteFirstFrame") || value.contains("matteFrameDigits");
            if (hasMatteAddition)
            {
                const auto safeName = [](const std::string& text, bool extension)
                {
                    if (text.empty() || text.size() > 32 || (extension && text.front() != '.')) return false;
                    const size_t start = extension ? 1 : 0;
                    return start < text.size() && std::all_of(text.begin() + static_cast<std::ptrdiff_t>(start), text.end(),
                        [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; });
                };
                if (segment.matteAssetId.empty() || segment.matteAssetVersion.empty()
                    || segment.matteContentReceipt.size() != 64 || segment.matteState.empty()
                    || segment.matteCacheKey.size() != 64
                    || ! videohelper::safeCacheKey(segment.matteCacheKey)
                    || ! safeName(segment.matteFramePrefix, false)
                    || ! safeName(segment.matteFrameExtension, true)
                    || segment.matteFirstFrame < 0 || segment.matteFrameDigits <= 0
                    || segment.matteFrameDigits > 12 || ! std::isfinite(segment.matteFps)
                    || segment.matteFps <= 0.0 || segment.matteFrames <= 0
                    || segment.matteFrames > 1000000
                    || segment.matteFirstFrame > std::numeric_limits<int>::max() - (segment.matteFrames - 1)
                    || ! object.contains("matteContentRevision")
                    || ! object["matteContentRevision"].is_number_integer()
                    || object["matteContentRevision"].template get<int64_t>() <= 0)
                {
                    error = "malformed or unauthorised matte binding";
                    return false;
                }
                const auto authority = segment.matteAssetVersion + "|" + segment.matteContentReceipt + "|"
                    + segment.matteCacheKey + "|" + segment.matteState + "|" + segment.matteFramePrefix + "|"
                    + segment.matteFrameExtension + "|" + std::to_string(segment.matteFirstFrame) + "|"
                    + std::to_string(segment.matteFrameDigits) + "|" + std::to_string(segment.matteFps)
                    + "|" + std::to_string(segment.matteFrames);
                const auto [found, inserted] = matteAuthorities.emplace(segment.matteAssetId, authority);
                if (! inserted && found->second != authority)
                {
                    error = "duplicate matte asset authority";
                    return false;
                }
            }

            if (value.contains ("transition"))
            {
                const auto& transition = value["transition"];
                segment.transitionType = transition.value ("type", 0);
                segment.transitionDurationSec = transition.value ("durationSec", 0.0);
                segment.hasStructuredTransitionBinding =
                    transition.contains ("fromClipId") || transition.contains ("toClipId");
                segment.transitionFromClipId = transition.value ("fromClipId", -1);
                segment.transitionToClipId = transition.value ("toClipId", -1);
            }
            raw.push_back (std::move (segment));
        }

    const bool hasPlans = object.contains ("visualLayerPlans");
    if (hasPlans && object["visualLayerPlans"].is_array())
        for (const auto& value : object["visualLayerPlans"])
        {
            CompiledVisualLayerPlan plan;
            plan.clipId = value.value ("clipId", 0);
            plan.structuralRevision = value.value ("structuralRevision", uint64_t { 0 });
            plan.identityMode = value.value ("identityMode", std::string { "authoredGraph" });
            plan.producerValidated = value.value ("valid", false);
            plan.error = value.value ("error", std::string {});
            if (value.contains ("nodeKinds") && value["nodeKinds"].is_array())
                for (const auto& kind : value["nodeKinds"])
                    if (kind.is_string()) plan.nodeKinds.push_back(kind.template get<std::string>());
            if (value.contains ("nodeIds") && value["nodeIds"].is_array())
                for (const auto& nodeId : value["nodeIds"])
                    if (nodeId.is_number_integer()) plan.nodeIds.push_back(nodeId.template get<int>());
            if (value.contains("edges") && value["edges"].is_array())
                for (const auto& edge : value["edges"])
                    plan.edges.push_back({ edge.value("fromNodeId", 0), edge.value("fromPort", 0),
                                           edge.value("toNodeId", 0), edge.value("toPort", 0) });
            if (value.contains("ports") && value["ports"].is_array())
                for (const auto& port : value["ports"])
                    plan.ports.push_back({ port.value("nodeId", 0), port.value("port", 0),
                        port.value("channels", 1),
                        port.value("direction", std::string {}), port.value("carrier", std::string {}),
                        port.value("dataType", std::string {}), port.value("pixelFormat", std::string {}),
                        port.value("colorSpace", std::string {}) });
            if (value.contains("operations") && value["operations"].is_array())
                for (const auto& operation : value["operations"])
                {
                    const auto kind = operation.value("kind", std::string {});
                    const auto payload = operation.value("payloadXml", std::string {});
                    if (kind == "visual.depth.asset")
                    {
                        const auto capability = operation.value("backendCapability", std::string {});
                        const bool executable = capability == "source-decode"
                            && payload.rfind("<DepthAssetBinding ", 0) == 0
                            && payload.find("state=\"available\"") != std::string::npos
                            && payload.find("format=\"r16-unorm\"") != std::string::npos
                            && payload.find("sequencePath=\"") == std::string::npos;
                        if (! executable)
                        {
                            error = "depth payload does not match the executable receipt schema";
                            return false;
                        }
                    }
                    plan.operations.push_back({ operation.value("nodeId", 0), kind,
                        operation.value("backendCapability", std::string {}), payload });
                }
            plans.push_back (std::move (plan));
        }

    if (object.contains ("visualEventSchedules"))
    {
        if (! object["visualEventSchedules"].is_array()
            || object["visualEventSchedules"].size() > 256)
        {
            error = "visual Event schedule capacity exceeded";
            return false;
        }
        for (const auto& value : object["visualEventSchedules"])
        {
            VisualEventScheduleBinding schedule;
            schedule.clipId = value.value ("clipId", 0);
            schedule.nodeId = value.value ("nodeId", 0);
            schedule.portId = value.value ("portId", -1);
            schedule.sessionRevision = value.value ("sessionRevision", uint64_t { 0 });
            if (! value.contains ("triggers") || ! value["triggers"].is_array()
                || value["triggers"].size() > 256)
            {
                error = "visual Event schedule has invalid stable sink identity";
                return false;
            }
            for (const auto& item : value["triggers"])
                schedule.triggers.push_back ({ item.value ("sampleOffset", -1),
                    item.value ("timelineBeat", std::numeric_limits<double>::quiet_NaN()),
                    item.value ("strength", std::numeric_limits<float>::quiet_NaN()),
                    item.value ("sequence", uint64_t { 0 }) });
            eventSchedules.push_back (std::move (schedule));
        }
    }

    return normalizeSnapshot (
        std::move (raw), std::move (plans), std::move (eventSchedules), object.value ("authoringRevision",
                                      object.value ("structuralRevision", uint64_t { 0 })), hasPlans,
        result, error);
}
} // namespace videowire
