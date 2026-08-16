#pragma once

#include "depth_payload_schema.h"

#include "render_snapshot.h"
#include "visual_plan_telemetry.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

namespace videowire
{
inline constexpr int kMaxVisualParticles = 4096;

struct VisualTriggerConsumption
{
    int count = 0;
    float strongest = 0.0f;
};

class VisualEventTriggerCursor
{
public:
    explicit VisualEventTriggerCursor (bool replayFromStart) : replayFromStart_ (replayFromStart)
    {
        states_.reserve (256);
    }

    VisualTriggerConsumption consume (const std::vector<VisualEventScheduleBinding>& schedules,
                                      int clipId, int nodeId, int portId,
                                      uint64_t sessionRevision, double currentBeat)
    {
        VisualTriggerConsumption result;
        if (! std::isfinite (currentBeat) || sessionRevision == 0) return result;
        auto found = std::find_if (states_.begin(), states_.end(), [=] (const State& state)
            { return state.clipId == clipId && state.nodeId == nodeId && state.portId == portId; });
        if (found == states_.end())
        {
            if (states_.size() >= 256) return result;
            states_.push_back ({ clipId, nodeId, portId, sessionRevision, currentBeat });
            found = states_.end() - 1;
            if (! replayFromStart_) return result;
            found->priorBeat = -std::numeric_limits<double>::infinity();
        }
        else if (found->sessionRevision != sessionRevision || currentBeat < found->priorBeat)
        {
            found->sessionRevision = sessionRevision;
            found->priorBeat = replayFromStart_ ? -std::numeric_limits<double>::infinity() : currentBeat;
            if (! replayFromStart_) return result;
        }
        else if (currentBeat == found->priorBeat)
            return result;

        const double priorBeat = found->priorBeat;
        for (const auto& schedule : schedules)
            if (schedule.clipId == clipId && schedule.nodeId == nodeId
                && schedule.portId == portId && schedule.sessionRevision == sessionRevision)
                for (const auto& trigger : schedule.triggers)
                    if (trigger.timelineBeat > priorBeat && trigger.timelineBeat <= currentBeat)
                    {
                        ++result.count;
                        result.strongest = std::max (result.strongest, trigger.strength);
                    }
        found->priorBeat = currentBeat;
        return result;
    }

    void reset() noexcept { states_.clear(); }

private:
    struct State
    {
        int clipId = 0, nodeId = 0, portId = -1;
        uint64_t sessionRevision = 0;
        double priorBeat = 0.0;
    };
    bool replayFromStart_ = false;
    std::vector<State> states_;
};

// The first helper-side execution slice deliberately lowers only the ordered,
// single-frame production chain already implemented by FrameRenderer.  It does
// not render pixels itself: callers apply the returned switches to LayerDesc and
// continue through the existing native compositor.
struct VisualLayerExecution
{
    uint64_t structuralRevision = 0;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    bool transform = true;
    bool effects = true;
    bool mask = true;
    bool drawShape = false;
    bool drawShapeEllipse = false;
    bool feedback = false;
    bool matteApply = false;
    bool depthFog = false;
    int depthEffect = 0;
    bool particles = false;
    int particleSeed = 1, particleCount = 512;
    float particleLifetime = 1.8f, particleSize = 4.0f, particleSpeed = 1.0f;
    float particleRed = 0.2f, particleGreen = 0.7f, particleBlue = 1.0f, particleAlpha = 1.0f;
    bool matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float feedbackDecay = 0.92f, feedbackZoom = 0.99f, feedbackSwirl = 0.0f;
    float shapeCx = 0.5f, shapeCy = 0.5f, shapeW = 1.0f, shapeH = 1.0f;
    float shapeR = 1.0f, shapeG = 1.0f, shapeB = 1.0f, shapeA = 1.0f;
};

inline bool visualPlanUsesTypedMatte (const std::vector<CompiledVisualLayerPlan>& plans,
                                      int clipId)
{
    const auto plan = std::find_if(plans.begin(), plans.end(),
        [clipId](const CompiledVisualLayerPlan& candidate) { return candidate.clipId == clipId; });
    return plan != plans.end()
        && std::find(plan->nodeKinds.begin(), plan->nodeKinds.end(), "visual.matte.apply")
             != plan->nodeKinds.end();
}

inline bool visualPlanDepthBinding(const std::vector<CompiledVisualLayerPlan>& plans, int clipId,
                                   ExecutableDepthPayload& payload, std::string& error)
{
    const auto plan = std::find_if(plans.begin(), plans.end(), [clipId](const auto& candidate)
        { return candidate.clipId == clipId; });
    if (plan == plans.end()) return false;
    const auto operation = std::find_if(plan->operations.begin(), plan->operations.end(), [](const auto& op)
        { return op.kind == "visual.depth.asset"; });
    if (operation == plan->operations.end()) return false;
    if (!parseExecutableDepthPayload(operation->payloadXml, payload))
    {
        error = "typed depth asset receipt metadata is malformed or stale";
        return false;
    }
    return true;
}

inline float visualPayloadFloat (const std::string& xml, const char* name, float fallback)
{
    const std::string key = std::string (name) + "=\"";
    const auto begin = xml.find (key);
    if (begin == std::string::npos) return fallback;
    char* end = nullptr;
    const char* number = xml.c_str() + begin + key.size();
    const float value = std::strtof (number, &end);
    return end == number ? fallback : value;
}

inline std::string visualPayloadString(const std::string& xml, const char* name)
{
    const std::string key = std::string(name) + "=\"";
    const auto begin = xml.find(key);
    if (begin == std::string::npos) return {};
    const auto value = begin + key.size();
    const auto end = xml.find('"', value);
    return end == std::string::npos ? std::string{} : xml.substr(value, end - value);
}

inline bool visualPlanSecondaryMatte(const std::vector<CompiledVisualLayerPlan>& plans, int clipId,
                                     const RenderSegment& primary, RenderSegment& secondary)
{
    const auto plan = std::find_if(plans.begin(), plans.end(), [clipId](const auto& p)
        { return p.clipId == clipId; });
    if (plan == plans.end()) return false;
    std::vector<const CompiledVisualOperation*> assets;
    for (const auto& op : plan->operations)
        if (op.kind == "visual.matte.asset") assets.push_back(&op);
    if (assets.size() != 2) return false;
    secondary = primary;
    const auto& xml = assets[1]->payloadXml;
    secondary.matteAssetId = visualPayloadString(xml, "matteAssetId");
    secondary.matteAssetVersion = visualPayloadString(xml, "matteAssetVersion");
    secondary.matteContentReceipt = visualPayloadString(xml, "contentReceipt");
    secondary.matteState = visualPayloadString(xml, "state");
    secondary.matteCacheKey = visualPayloadString(xml, "cacheKey");
    secondary.matteFramePrefix = visualPayloadString(xml, "framePrefix");
    secondary.matteFrameExtension = visualPayloadString(xml, "frameExtension");
    secondary.matteFirstFrame = (int)visualPayloadFloat(xml, "firstFrame", -1.0f);
    secondary.matteFrameDigits = (int)visualPayloadFloat(xml, "frameDigits", -1.0f);
    secondary.matteFps = visualPayloadFloat(xml, "fps", 0.0f);
    secondary.matteFrames = (int)visualPayloadFloat(xml, "frames", 0.0f);
    return !secondary.matteAssetId.empty() && secondary.matteState == "available";
}

// Mutable temporal state belongs to one concrete viewport or export run, never
// to the immutable plan. Revisions and backward seeks explicitly reset an owner.
class VisualPlanExecutionState
{
public:
    static constexpr size_t kMaxAdmittedPlans = 256;
    struct Owner
    {
        uint64_t structuralRevision = 0;
        double lastTimeSec = 0.0;
        uint64_t evaluationSequence = 0;
        bool hasTime = false;
    };
    struct Slot
    {
        int clipId = -1;
        uint64_t structuralRevision = 0;
        Owner owner;
        VisualLayerExecution execution;
        bool lowered = false;
    };

    bool admitPlans (const std::vector<CompiledVisualLayerPlan>& plans,
                     std::string* diagnostic = nullptr);
    void reset() { for (size_t i = 0; i < slotCount_; ++i) slots_[i].owner = {}; }
    void reset (int clipId) { if (auto* slot = findSlot(clipId)) slot->owner = {}; }

    const VisualLayerExecution* compiled (int clipId, uint64_t revision) const
    {
        const auto* slot = findSlot(clipId);
        return slot != nullptr && slot->lowered && slot->structuralRevision == revision
            ? &slot->execution : nullptr;
    }
    void setTelemetryOwner (VisualPlanTelemetry& telemetry) { telemetryOwner_ = &telemetry; }
    VisualPlanTelemetry& telemetry() { return *telemetryOwner_; }
    const VisualPlanTelemetry& telemetry() const { return *telemetryOwner_; }

    const Owner* prepare (int clipId, uint64_t revision, double timeSec)
    {
        auto* slot = findSlot(clipId);
        if (slot == nullptr || slot->structuralRevision != revision) return nullptr;
        auto& owner = slot->owner;
        if (owner.structuralRevision != revision
            || (owner.hasTime && timeSec < owner.lastTimeSec))
            owner = Owner { revision };
        if (! owner.hasTime || timeSec > owner.lastTimeSec)
            ++owner.evaluationSequence;
        owner.lastTimeSec = timeSec;
        owner.hasTime = true;
        return &owner;
    }

    const Owner* owner (int clipId) const
    {
        const auto* slot = findSlot(clipId);
        return slot == nullptr ? nullptr : &slot->owner;
    }

    bool startsSequence (int clipId, uint64_t revision) const
    {
        const auto* value = owner(clipId);
        return value != nullptr && value->structuralRevision == revision && ! value->hasTime;
    }

    bool needsReset (int clipId, uint64_t revision, double timeSec) const
    {
        const auto* value = owner(clipId);
        return value == nullptr || value->structuralRevision != revision
            || (value->hasTime && timeSec < value->lastTimeSec);
    }

    bool isHold (int clipId, uint64_t revision, double timeSec) const
    {
        const auto* value = owner(clipId);
        return value != nullptr && value->structuralRevision == revision
            && value->hasTime && timeSec == value->lastTimeSec;
    }

private:
    Slot* findSlot (int clipId)
    {
        auto it = std::lower_bound(slots_.begin(), slots_.begin() + (ptrdiff_t) slotCount_, clipId,
            [](const Slot& slot, int id) { return slot.clipId < id; });
        return it != slots_.begin() + (ptrdiff_t) slotCount_ && it->clipId == clipId ? &*it : nullptr;
    }
    const Slot* findSlot (int clipId) const
    {
        auto it = std::lower_bound(slots_.begin(), slots_.begin() + (ptrdiff_t) slotCount_, clipId,
            [](const Slot& slot, int id) { return slot.clipId < id; });
        return it != slots_.begin() + (ptrdiff_t) slotCount_ && it->clipId == clipId ? &*it : nullptr;
    }
    std::array<Slot, kMaxAdmittedPlans> slots_ {};
    size_t slotCount_ = 0;
    VisualPlanTelemetry telemetry_;
    VisualPlanTelemetry* telemetryOwner_ = &telemetry_;
};

inline size_t countVisualPlanIntermediateImages (const CompiledVisualLayerPlan& plan)
{
    size_t count = 0;
    for (size_t i = 0; i < plan.edges.size(); ++i)
    {
        const auto& edge = plan.edges[i];
        const auto producer = std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const CompiledVisualOperation& operation) { return operation.nodeId == edge.fromNodeId; });
        const auto consumer = std::find_if(plan.operations.begin(), plan.operations.end(),
            [&](const CompiledVisualOperation& operation) { return operation.nodeId == edge.toNodeId; });
        if (producer == plan.operations.end() || consumer == plan.operations.end()
            || producer->backendCapability == "control-eval"
            || producer->backendCapability == "parked-metadata")
            continue;
        bool duplicate = false;
        for (size_t prior = 0; prior < i; ++prior)
            duplicate = duplicate || (plan.edges[prior].fromNodeId == edge.fromNodeId
                && plan.edges[prior].fromPort == edge.fromPort);
        if (! duplicate) ++count;
    }
    return count;
}

inline VisualTelemetryPlanAdmission makeVisualTelemetryAdmission (const CompiledVisualLayerPlan& plan)
{
    VisualTelemetryPlanAdmission admission;
    admission.clipId = plan.clipId;
    admission.structuralRevision = plan.structuralRevision;
    for (const auto& operation : plan.operations)
        if (operation.backendCapability == "native-gpu"
            || operation.backendCapability == "source-decode")
        {
            ++admission.executableNodeTotal;
            if (admission.nodeCount < kMaxCompiledNodesPerGraph)
                admission.stableNodeIds[admission.nodeCount++] = operation.nodeId;
        }
    admission.nodesTruncated = admission.executableNodeTotal > admission.nodeCount;
    admission.intermediateImageCount = static_cast<uint8_t>(std::min(
        countVisualPlanIntermediateImages(plan), kMaxIntermediateImagesPerGraph));
    return admission;
}

inline bool compileVisualLayerExecution (const CompiledVisualLayerPlan& plan,
                                         VisualLayerExecution& execution,
                                         std::string& error)
{
    execution = {};
    const auto isTrackingControl = [] (const std::string& kind)
    {
        return kind == "tracking.point.asset" || kind == "tracking.planar.asset"
            || kind == "tracking.correction" || kind == "tracking.point.apply.transform"
            || kind == "tracking.planar.apply.quad";
    };
    if (std::any_of(plan.operations.begin(), plan.operations.end(), [&] (const auto& operation)
        { return isTrackingControl(operation.kind); }))
    {
        CompiledVisualLayerPlan framePlan = plan;
        std::vector<int> removed;
        for (const auto& operation : plan.operations)
            if (isTrackingControl(operation.kind)) removed.push_back(operation.nodeId);
        const auto removedNode = [&] (int id)
        { return std::find(removed.begin(), removed.end(), id) != removed.end(); };
        framePlan.nodeKinds.clear(); framePlan.nodeIds.clear(); framePlan.operations.clear(); framePlan.ports.clear();
        for (size_t i = 0; i < plan.nodeIds.size(); ++i)
            if (! removedNode(plan.nodeIds[i])) { framePlan.nodeIds.push_back(plan.nodeIds[i]); framePlan.nodeKinds.push_back(plan.nodeKinds[i]); }
        for (const auto& operation : plan.operations)
            if (! removedNode(operation.nodeId)) framePlan.operations.push_back(operation);
        for (const auto& port : plan.ports)
            if (! removedNode(port.nodeId)) framePlan.ports.push_back(port);
        framePlan.edges.erase(std::remove_if(framePlan.edges.begin(), framePlan.edges.end(), [&] (const auto& edge)
            { return removedNode(edge.fromNodeId) || removedNode(edge.toNodeId); }), framePlan.edges.end());
        return compileVisualLayerExecution(framePlan, execution, error);
    }
    execution.structuralRevision = plan.structuralRevision;
    const bool typed = ! plan.nodeIds.empty() || ! plan.edges.empty()
                    || ! plan.ports.empty() || ! plan.operations.empty();
    std::vector<std::string> kinds;
    std::vector<int> ids;
    if (typed)
    {
        if (plan.operations.empty() || plan.operations.size() != plan.nodeIds.size())
        {
            error = "typed visual layer plan has no executable operation order";
            return false;
        }
        kinds.reserve (plan.operations.size());
        ids.reserve (plan.operations.size());
        for (const auto& operation : plan.operations)
        {
            if (operation.kind == "control.history")
            {
                error = "control.history temporal evaluation is not yet connected to native viewport/export image execution";
                return false;
            }
            // This slice admits only passes which the existing compositor runs on
            // the native GPU (source decode remains the established source path).
            const bool rectangleControl = (operation.kind == "visual.shape.rectangle"
                || operation.kind == "visual.shape.ellipse")
                && operation.backendCapability == "control-eval";
            if (! rectangleControl && operation.backendCapability != "native-gpu"
                && operation.backendCapability != "source-decode")
            {
                error = "visual layer plan requires unsupported execution capability";
                return false;
            }
            kinds.push_back (operation.kind);
            ids.push_back (operation.nodeId);
        }

        if (kinds.size() == 2 && kinds[0] == "visual.particles" && kinds[1] == "video.out")
        {
            const bool exact = plan.edges.size() == 1 && plan.edges[0].fromNodeId == ids[0]
                && plan.edges[0].fromPort == 1 && plan.edges[0].toNodeId == ids[1]
                && plan.edges[0].toPort == 0;
            if (! exact)
            {
                error = "visual.particles has unsupported production topology";
                return false;
            }
            const auto& payload = plan.operations[0].payloadXml;
            execution.transform = execution.effects = execution.mask = false;
            execution.particles = true;
            execution.particleNodeId = ids[0];
            execution.particleSeed = std::clamp((int) visualPayloadFloat(payload, "seed", 1.0f), 0, 65535);
            execution.particleCount = std::clamp((int) visualPayloadFloat(payload, "count", 512.0f), 1,
                                                 kMaxVisualParticles);
            execution.particleLifetime = std::clamp(visualPayloadFloat(payload, "lifetime", 1.8f), 0.1f, 10.0f);
            execution.particleSize = std::clamp(visualPayloadFloat(payload, "size", 4.0f), 1.0f, 32.0f);
            execution.particleSpeed = std::clamp(visualPayloadFloat(payload, "speed", 1.0f), 0.0f, 4.0f);
            execution.particleRed = std::clamp(visualPayloadFloat(payload, "red", 0.2f), 0.0f, 1.0f);
            execution.particleGreen = std::clamp(visualPayloadFloat(payload, "green", 0.7f), 0.0f, 1.0f);
            execution.particleBlue = std::clamp(visualPayloadFloat(payload, "blue", 1.0f), 0.0f, 1.0f);
            execution.particleAlpha = std::clamp(visualPayloadFloat(payload, "alpha", 1.0f), 0.0f, 1.0f);
            return true;
        }

        // The existing LayerDesc compositor represents one image value flowing
        // through a linear chain. Require every exact compiled edge to be the
        // corresponding consecutive image operation; branching/control/image
        // fan-in remains unsupported rather than being flattened or CPU-rendered.
        const bool linear = plan.edges.size() + 1 == ids.size()
            && std::all_of(ids.begin(), ids.end() - 1, [&](int id)
            {
                const auto index = (size_t) std::distance(ids.begin(),
                    std::find(ids.begin(), ids.end(), id));
                return std::any_of(plan.edges.begin(), plan.edges.end(),
                    [&](const CompiledVisualEdgeBinding& edge)
                    { return edge.fromNodeId == id && edge.toNodeId == ids[index + 1]; });
            });
        if (! linear)
        {
            const auto indexOf = [&](const char* kind)
            {
                const auto found = std::find(kinds.begin(), kinds.end(), kind);
                return found == kinds.end() ? -1 : (int) std::distance(kinds.begin(), found);
            };
            const int matteAsset = indexOf("visual.matte.asset");
            const int matteRefine = indexOf("visual.matte.refine");
            const int matteApply = indexOf("visual.matte.apply");
            const int matteCombine = indexOf("visual.matte.combine");
            const int matteSource = indexOf("video.source");
            const int matteOutput = indexOf("video.out");
            const int depthAsset = indexOf("visual.depth.asset");
            if (depthAsset >= 0)
            {
                int fog = indexOf("visual.depth.fog");
                if (fog < 0) fog = indexOf("visual.depth.blur");
                if (fog < 0) fog = indexOf("visual.depth.displace");
                if (fog < 0) fog = indexOf("visual.depth.relight");
                const auto edge = [&](int from, int fromPort, int to, int toPort)
                { return from >= 0 && to >= 0 && std::any_of(plan.edges.begin(), plan.edges.end(),
                    [&](const CompiledVisualEdgeBinding& e) { return e.fromNodeId == ids[(size_t)from]
                        && e.fromPort == fromPort && e.toNodeId == ids[(size_t)to] && e.toPort == toPort; }); };
                const bool exact = kinds.size() == 4 && ids.size() == 4 && kinds[0] == "video.source"
                    && kinds.back() == "video.out" && fog >= 0 && plan.edges.size() == 3
                    && edge(0, 0, fog, 0) && edge(depthAsset, 1, fog, 1)
                    && edge(fog, 2, 3, 0);
                const auto& binding = plan.operations[(size_t) depthAsset];
                const auto required = [&](const char* value) { return binding.payloadXml.find(value) != std::string::npos; };
                if (! exact || binding.backendCapability != "source-decode"
                    || binding.payloadXml.rfind("<DepthAssetBinding ", 0) != 0
                    || binding.payloadXml.size() < 3
                    || binding.payloadXml.compare(binding.payloadXml.size() - 2, 2, "/>") != 0
                    || ! required("state=\"available\"") || ! required("cacheKey=\"")
                    || ! required("contentReceipt=\"") || ! required("analysisReceipt=\"")
                    || ! required("framePrefix=\"") || ! required("frameExtension=\"")
                    || ! required("width=\"") || ! required("height=\"")
                    || ! required("fps=\"") || ! required("frames=\"")
                    || ! required("format=\"r16-unorm\"")
                    || binding.payloadXml.find("sequencePath=\"") != std::string::npos
                    || binding.payloadXml.find(" path=\"") != std::string::npos
                    || binding.payloadXml.find(" file=\"") != std::string::npos
                    || binding.payloadXml.find(" uri=\"") != std::string::npos
                    || binding.payloadXml.find(" decoder=\"") != std::string::npos
                    || binding.payloadXml.find(" upload=\"") != std::string::npos
                    || binding.payloadXml.find(" futureField=\"") != std::string::npos)
                {
                    error = "typed depth asset binding is missing, stale, or unsupported";
                    return false;
                }
                execution.transform = execution.effects = execution.mask = false;
                execution.depthFog = kinds[(size_t) fog] == "visual.depth.fog";
                execution.depthEffect = execution.depthFog ? 1
                    : kinds[(size_t) fog] == "visual.depth.blur" ? 2
                    : kinds[(size_t) fog] == "visual.depth.displace" ? 3 : 4;
                const auto& fogPayload = plan.operations[(size_t) fog].payloadXml;
                execution.fogNear = std::clamp(visualPayloadFloat(fogPayload, "near", 0.0f), 0.0f, 1.0f);
                execution.fogFar = std::clamp(visualPayloadFloat(fogPayload, "far", 1.0f), 0.0f, 1.0f);
                execution.fogDensity = std::clamp(visualPayloadFloat(fogPayload, "density", 1.0f), 0.0f, 32.0f);
                execution.fogRed = std::clamp(visualPayloadFloat(fogPayload, "red", 1.0f), 0.0f, 1.0f);
                execution.fogGreen = std::clamp(visualPayloadFloat(fogPayload, "green", 1.0f), 0.0f, 1.0f);
                execution.fogBlue = std::clamp(visualPayloadFloat(fogPayload, "blue", 1.0f), 0.0f, 1.0f);
                execution.fogAlpha = std::clamp(visualPayloadFloat(fogPayload, "alpha", 1.0f), 0.0f, 1.0f);
                if (execution.depthEffect == 2) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "radius", 4.0f), 0.0f, 32.0f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "focus", 0.5f), 0.0f, 1.0f);
                    execution.depthParam2 = std::clamp(visualPayloadFloat(fogPayload, "falloff", 0.25f), 0.001f, 1.0f);
                } else if (execution.depthEffect == 3) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "amountX", 0.0f), -0.25f, 0.25f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "amountY", 0.0f), -0.25f, 0.25f);
                    execution.depthParam2 = std::clamp(visualPayloadFloat(fogPayload, "center", 0.5f), 0.0f, 1.0f);
                } else if (execution.depthEffect == 4) {
                    execution.depthParam0 = std::clamp(visualPayloadFloat(fogPayload, "intensity", 1.0f), -2.0f, 2.0f);
                    execution.depthParam1 = std::clamp(visualPayloadFloat(fogPayload, "ambient", 1.0f), 0.0f, 2.0f);
                    execution.depthColorRed = std::clamp(visualPayloadFloat(fogPayload, "red", 1.0f), 0.0f, 2.0f);
                    execution.depthColorGreen = std::clamp(visualPayloadFloat(fogPayload, "green", 1.0f), 0.0f, 2.0f);
                    execution.depthColorBlue = std::clamp(visualPayloadFloat(fogPayload, "blue", 1.0f), 0.0f, 2.0f);
                }
                return true;
            }
            if (matteAsset >= 0 || matteRefine >= 0 || matteApply >= 0)
            {
                const auto hasExactEdge = [&](int from, int fromPort, int to, int toPort)
                {
                    return from >= 0 && to >= 0 && std::any_of(plan.edges.begin(), plan.edges.end(),
                        [&](const CompiledVisualEdgeBinding& edge)
                        { return edge.fromNodeId == ids[(size_t) from] && edge.fromPort == fromPort
                              && edge.toNodeId == ids[(size_t) to] && edge.toPort == toPort; });
                };
                std::vector<int> matteAssets;
                for (size_t i = 0; i < kinds.size(); ++i)
                    if (kinds[i] == "visual.matte.asset") matteAssets.push_back((int)i);
                const bool combined = matteCombine >= 0;
                const bool refined = matteRefine >= 0;
                const size_t expectedNodes = (refined ? 5u : 4u) + (combined ? 2u : 0u);
                const int maskSource = combined ? matteCombine : matteAsset;
                const bool exact = kinds.size() == expectedNodes && matteSource == 0
                    && matteAssets.size() == (combined ? 2u : 1u) && matteApply > matteAsset
                    && matteOutput == (int) kinds.size() - 1
                    && hasExactEdge(matteSource, 0, matteApply, 0)
                    && (!combined || (hasExactEdge(matteAssets[0], 0, matteCombine, 0)
                                   && hasExactEdge(matteAssets[1], 0, matteCombine, 1)))
                    && hasExactEdge(maskSource, combined ? 2 : 0,
                                    refined ? matteRefine : matteApply, refined ? 0 : 1)
                    && (! refined || hasExactEdge(matteRefine, 1, matteApply, 1))
                    && hasExactEdge(matteApply, 2, matteOutput, 0)
                    && plan.edges.size() == (refined ? 4u : 3u) + (combined ? 2u : 0u);
                if (! exact)
                {
                    error = "typed matte graph has unsupported production topology";
                    return false;
                }
                const auto validBinding = [&](int index)
                {
                    const auto& binding = plan.operations[(size_t)index];
                    return binding.backendCapability == "source-decode"
                        && binding.payloadXml.find("matteAssetId=\"") != std::string::npos
                        && binding.payloadXml.find("state=\"available\"") != std::string::npos
                        && binding.payloadXml.find("cacheKey=\"") != std::string::npos
                        && binding.payloadXml.find("contentReceipt=\"") != std::string::npos
                        && binding.payloadXml.find("framePrefix=\"") != std::string::npos
                        && binding.payloadXml.find("frameExtension=\"") != std::string::npos
                        && binding.payloadXml.find("firstFrame=\"") != std::string::npos
                        && binding.payloadXml.find("frameDigits=\"") != std::string::npos
                        && binding.payloadXml.find("sequencePath=\"") == std::string::npos
                        && binding.payloadXml.find("backend=\"rgba-cpu-decode-native-gpu-upload\"") != std::string::npos;
                };
                if (!std::all_of(matteAssets.begin(), matteAssets.end(), validBinding))
                {
                    error = "typed matte asset binding is missing, stale, or unsupported";
                    return false;
                }
                execution.transform = execution.effects = execution.mask = false;
                execution.matteApply = true;
                if (combined)
                    execution.matteCombineMode = std::clamp((int)visualPayloadFloat(
                        plan.operations[(size_t)matteCombine].payloadXml, "mode", 0.0f), 0, 3);
                if (refined)
                {
                    const auto& payload = plan.operations[(size_t) matteRefine].payloadXml;
                    execution.matteInvert = visualPayloadFloat(payload, "invert", 0.0f) >= 0.5f;
                    execution.matteBlack = std::clamp(visualPayloadFloat(payload, "black", 0.0f), 0.0f, 1.0f);
                    execution.matteWhite = std::clamp(visualPayloadFloat(payload, "white", 1.0f), 0.0f, 1.0f);
                    execution.matteErodeDilate = std::clamp(visualPayloadFloat(payload, "erodeDilate", 0.0f), -4.0f, 4.0f);
                    execution.matteFeather = std::clamp(visualPayloadFloat(payload, "feather", 0.0f), 0.0f, 4.0f);
                    execution.matteChoke = std::clamp(visualPayloadFloat(payload, "choke", 0.0f), -1.0f, 1.0f);
                }
                return true;
            }
            const int sourceIndex = indexOf("video.source");
            const int blendIndex = indexOf("video.blend");
            const int outputIndex = indexOf("video.out");
            if (sourceIndex != 0 || blendIndex < 0 || outputIndex != (int) kinds.size() - 1)
            {
                error = "visual layer plan is not an executable production composite";
                return false;
            }
            const auto allowed = [](const std::string& kind)
            {
                return kind == "video.source" || kind == "video.transform"
                    || kind == "video.effects" || kind == "video.mask.shape"
                    || kind == "video.text" || kind == "video.layer.source"
                    || kind == "visual.gradient" || kind == "visual.shape"
                    || kind == "visual.shape.rectangle" || kind == "visual.shape.ellipse"
                    || kind == "visual.draw.shape"
                    || kind == "video.blend" || kind == "video.out";
            };
            if (! std::all_of(kinds.begin(), kinds.end(), allowed))
            {
                error = "visual layer plan contains an unsupported ordered production operation";
                return false;
            }
            const auto hasEdge = [&](int from, int to)
            {
                return std::any_of(plan.edges.begin(), plan.edges.end(),
                    [&](const CompiledVisualEdgeBinding& edge)
                    { return edge.fromNodeId == ids[(size_t) from]
                          && edge.toNodeId == ids[(size_t) to]; });
            };
            std::vector<int> primary { sourceIndex };
            for (const char* kind : { "video.transform", "video.effects", "video.mask.shape" })
                if (const int index = indexOf(kind); index >= 0) primary.push_back(index);
            primary.push_back(blendIndex);
            size_t expectedEdges = primary.size(); // primary chain plus Blend -> Output
            bool topologyMatches = hasEdge(blendIndex, outputIndex);
            for (size_t i = 0; i + 1 < primary.size(); ++i)
                topologyMatches = topologyMatches && hasEdge(primary[i], primary[i + 1]);
            for (const char* kind : { "video.text", "video.layer.source", "visual.gradient", "visual.shape" })
                if (const int index = indexOf(kind); index >= 0)
                {
                    topologyMatches = topologyMatches && hasEdge(index, blendIndex);
                    ++expectedEdges;
                }
            const int rectangleIndex = indexOf("visual.shape.rectangle");
            const int ellipseIndex = indexOf("visual.shape.ellipse");
            const int primitiveIndex = rectangleIndex >= 0 ? rectangleIndex : ellipseIndex;
            const int drawShapeIndex = indexOf("visual.draw.shape");
            if (primitiveIndex >= 0 || drawShapeIndex >= 0)
            {
                topologyMatches = topologyMatches && primitiveIndex >= 0 && drawShapeIndex >= 0
                    && ! (rectangleIndex >= 0 && ellipseIndex >= 0)
                    && hasEdge(primitiveIndex, drawShapeIndex)
                    && hasEdge(drawShapeIndex, blendIndex);
                expectedEdges += 2;
            }
            if (! topologyMatches || plan.edges.size() != expectedEdges)
            {
                error = "visual layer plan is not an executable production composite";
                return false;
            }
            execution.transform = indexOf("video.transform") >= 0;
            execution.effects = indexOf("video.effects") >= 0;
            execution.mask = indexOf("video.mask.shape") >= 0;
            if (primitiveIndex >= 0 && drawShapeIndex >= 0)
            {
                const auto& rectangle = plan.operations[(size_t) primitiveIndex];
                const auto& draw = plan.operations[(size_t) drawShapeIndex];
                execution.drawShape = true;
                execution.drawShapeNodeId = ids[(size_t) drawShapeIndex];
                execution.drawShapeEllipse = ellipseIndex >= 0;
                execution.shapeCx = visualPayloadFloat(rectangle.payloadXml, "centerX", 0.5f);
                execution.shapeCy = visualPayloadFloat(rectangle.payloadXml, "centerY", 0.5f);
                execution.shapeW = visualPayloadFloat(rectangle.payloadXml, "width", 1.0f);
                execution.shapeH = visualPayloadFloat(rectangle.payloadXml, "height", 1.0f);
                execution.shapeR = visualPayloadFloat(draw.payloadXml, "red", 1.0f);
                execution.shapeG = visualPayloadFloat(draw.payloadXml, "green", 1.0f);
                execution.shapeB = visualPayloadFloat(draw.payloadXml, "blue", 1.0f);
                execution.shapeA = visualPayloadFloat(draw.payloadXml, "alpha", 1.0f);
            }
            return true;
        }
    }
    else
    {
        // Absence-only compatibility for snapshots produced before operation and
        // edge transport. Their already-admitted node order is the fixed chain.
        kinds = plan.nodeKinds;
    }

    if (kinds.empty() || kinds.back() != "video.out")
    {
        error = "visual layer plan has no executable output";
        return false;
    }
    const bool source = kinds.front() == "video.source"
                     || kinds.front() == "video.legacy.source"
                     || kinds.front() == "video.legacy.generator";
    if (! source)
    {
        error = "visual layer plan has no executable production source";
        return false;
    }

    size_t cursor = 1;
    const auto consume = [&](const char* kind, size_t& index)
    {
        if (index < kinds.size() && kinds[index] == kind) { ++index; return true; }
        return false;
    };
    const bool legacyChain = kinds.front().rfind ("video.legacy.", 0) == 0;
    if (consume ("video.legacy.retime", cursor)) {}
    execution.transform = consume (legacyChain ? "video.legacy.transform"
                                               : "video.transform", cursor);
    execution.effects = consume (legacyChain ? "video.legacy.effects"
                                             : "video.effects", cursor);
    if (! legacyChain && consume("visual.feedback", cursor))
    {
        const auto operation = std::find_if(plan.operations.begin(), plan.operations.end(),
            [](const CompiledVisualOperation& value) { return value.kind == "visual.feedback"; });
        execution.feedback = true;
        if (operation != plan.operations.end())
        {
            execution.feedbackDecay = std::clamp(visualPayloadFloat(operation->payloadXml, "decay", 0.92f), 0.0f, 0.999f);
            execution.feedbackZoom = std::clamp(visualPayloadFloat(operation->payloadXml, "zoom", 0.99f), 0.9f, 1.05f);
            execution.feedbackSwirl = std::clamp(visualPayloadFloat(operation->payloadXml, "swirl", 0.0f), -0.1f, 0.1f);
        }
    }
    execution.mask = legacyChain;
    if (! legacyChain)
        execution.mask = consume ("video.mask.shape", cursor);
    consume ("video.blend", cursor); // existing compositor owns the admitted pass

    if (cursor + 1 != kinds.size() || kinds[cursor] != "video.out")
    {
        error = "visual layer plan contains an unsupported ordered production operation";
        return false;
    }
    if (execution.feedback && execution.effects)
    {
        error = "visual.feedback cannot be combined with video.effects until LayerDesc owns a merged immutable rack";
        return false;
    }
    if (std::count(kinds.begin(), kinds.end(), "visual.feedback") > 1)
    {
        error = "visual.feedback supports exactly one production history pass per clip";
        return false;
    }
    return true;
}

inline bool VisualPlanExecutionState::admitPlans (
    const std::vector<CompiledVisualLayerPlan>& plans, std::string* diagnostic)
{
    if (plans.size() > kMaxAdmittedPlans)
    {
        const std::string reason = "visual execution admission exceeds fixed plan capacity";
        telemetry().recordFailedLowering(reason);
        if (diagnostic != nullptr) *diagnostic = reason;
        return false;
    }
    std::vector<VisualTelemetryPlanAdmission> telemetryAdmissions;
    telemetryAdmissions.reserve(plans.size());
    for (const auto& plan : plans)
        telemetryAdmissions.push_back(makeVisualTelemetryAdmission(plan));
    if (! telemetry().admitPlans(telemetryAdmissions, diagnostic))
        return false;

    std::array<Slot, kMaxAdmittedPlans> next {};
    size_t count = 0;
    bool complete = true;
    for (const auto& plan : plans)
    {
        auto duplicate = std::find_if(next.begin(), next.begin() + (ptrdiff_t) count,
            [&](const Slot& slot) { return slot.clipId == plan.clipId; });
        if (duplicate != next.begin() + (ptrdiff_t) count
            && duplicate->structuralRevision >= plan.structuralRevision)
            continue;
        if (duplicate == next.begin() + (ptrdiff_t) count)
        {
            if (count == kMaxAdmittedPlans)
            {
                complete = false;
                telemetry().recordFailedLowering("visual execution admission exceeds fixed plan capacity");
                if (diagnostic != nullptr && diagnostic->empty())
                    *diagnostic = "visual execution admission exceeds fixed plan capacity";
                continue;
            }
            duplicate = next.begin() + (ptrdiff_t) count++;
        }
        Slot slot;
        slot.clipId = plan.clipId;
        slot.structuralRevision = plan.structuralRevision;
        slot.owner.structuralRevision = plan.structuralRevision;
        std::string loweringError;
        const auto begin = std::chrono::steady_clock::now();
        slot.lowered = compileVisualLayerExecution(plan, slot.execution, loweringError);
        const auto loweringNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
        telemetry().recordPlanLowering(false, loweringNs, slot.lowered);
        if (! slot.lowered)
        {
            complete = false;
            telemetry().recordFailedLowering(loweringError);
            if (diagnostic != nullptr && diagnostic->empty()) *diagnostic = loweringError;
        }
        *duplicate = slot;
    }
    if (! complete) return false;
    std::sort(next.begin(), next.begin() + (ptrdiff_t) count,
        [](const Slot& a, const Slot& b) { return a.clipId < b.clipId; });
    slots_ = next;
    slotCount_ = count;
    return true;
}

inline const CompiledVisualLayerPlan* findVisualLayerPlan (
    const std::vector<CompiledVisualLayerPlan>& plans, int clipId)
{
    const auto it = std::find_if (plans.begin(), plans.end(),
        [clipId](const CompiledVisualLayerPlan& plan) { return plan.clipId == clipId; });
    return it == plans.end() ? nullptr : &*it;
}

struct VisualInspectionTarget
{
    int clipId = -1;
    uint64_t structuralRevision = 0;
    int nodeId = 0;
    int outputPort = -1;
};

// A borrowed backend-native image produced while lowering one layer. The
// renderer remains the owner; this record pins the exact handle for only the
// current frame/generation and never maps or copies its pixels.
struct VisualInspectionResource
{
    unsigned handle = 0;
    int width = 0;
    int height = 0;
    int nodeId = 0;
    int outputPort = -1;
};

enum class VisualInspectionSlice
{
    unsupported,
    decodedSource,
    parkedDepthMetadata,
    transformedLayer,
    retainedDrawShape,
    terminalComposite
};

inline VisualInspectionSlice classifyVisualInspectionTarget (
    const CompiledVisualLayerPlan& plan, const VisualInspectionTarget& target)
{
    const auto operation = std::find_if(plan.operations.begin(), plan.operations.end(),
        [&target](const CompiledVisualOperation& value) { return value.nodeId == target.nodeId; });
    if (operation == plan.operations.end()) return VisualInspectionSlice::unsupported;
    const auto output = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const CompiledVisualOperation& value) { return value.kind == "video.out"; });
    const bool terminal = output != plan.operations.end()
        && std::any_of(plan.edges.begin(), plan.edges.end(), [&](const CompiledVisualEdgeBinding& edge)
        { return edge.fromNodeId == target.nodeId && edge.toNodeId == output->nodeId; });
    // FrameRenderer exposes its genuine post-geometry layer target. Admit only
    // the exact source->transform->out shape: later operations would make the
    // fixed renderer boundary a different semantic resource.
    if (operation->kind == "video.transform" && operation->backendCapability == "native-gpu"
        && terminal && plan.operations.size() == 3
        && plan.operations.front().kind == "video.source"
        && plan.operations.back().kind == "video.out")
        return VisualInspectionSlice::transformedLayer;
    // visual.draw.shape materializes its image in a dedicated native render pass
    // immediately before Blend. This is the exact operation boundary, not a
    // reconstruction from the terminal composite.
    if (operation->kind == "visual.draw.shape"
        && operation->backendCapability == "native-gpu"
        && std::any_of(plan.edges.begin(), plan.edges.end(), [&](const CompiledVisualEdgeBinding& edge)
        {
            const auto consumer = std::find_if(plan.operations.begin(), plan.operations.end(),
                [&](const CompiledVisualOperation& value) { return value.nodeId == edge.toNodeId; });
            return edge.fromNodeId == target.nodeId && edge.fromPort == target.outputPort
                && consumer != plan.operations.end() && consumer->kind == "video.blend";
        }))
        return VisualInspectionSlice::retainedDrawShape;
    if (terminal) return VisualInspectionSlice::terminalComposite;
    // FrameRenderer receives decoded video.source as LayerDesc::texture. That
    // is the only exact non-terminal operation output exposed at this boundary.
    if (operation->kind == "video.source" && operation->backendCapability == "source-decode")
        return VisualInspectionSlice::decodedSource;
    if (operation->kind == "visual.depth.asset" && operation->backendCapability == "parked-metadata")
        return VisualInspectionSlice::parkedDepthMetadata;
    return VisualInspectionSlice::unsupported;
}

inline bool validateVisualInspectionTarget (
    const std::vector<CompiledVisualLayerPlan>& plans,
    const VisualInspectionTarget& target, std::string& error)
{
    const auto* plan = findVisualLayerPlan(plans, target.clipId);
    if (plan == nullptr)
    {
        error = "inspection target has no compiled visual plan";
        return false;
    }
    if (plan->structuralRevision != target.structuralRevision)
    {
        error = "inspection target revision is stale";
        return false;
    }
    const auto operation = std::find_if(plan->operations.begin(), plan->operations.end(),
        [&target](const CompiledVisualOperation& value) { return value.nodeId == target.nodeId; });
    if (operation == plan->operations.end())
    {
        error = "inspection target node is not executable";
        return false;
    }
    const auto port = std::find_if(plan->ports.begin(), plan->ports.end(),
        [&target](const CompiledVisualPortBinding& value)
        {
            return value.nodeId == target.nodeId && value.port == target.outputPort;
        });
    if (port == plan->ports.end() || port->direction != "out" || port->carrier != "frame")
    {
        error = "inspection target is not an image output port";
        return false;
    }
    const auto slice = classifyVisualInspectionTarget(*plan, target);
    if (slice == VisualInspectionSlice::parkedDepthMetadata)
    {
        error = "depth is parked metadata pending a depth-consuming execution seam; inspection is unavailable";
        return false;
    }
    if (slice == VisualInspectionSlice::unsupported)
    {
        error = "inspection target has no retainable GPU output resource";
        return false;
    }
    return true;
}

template <typename LayerDesc>
inline bool executeVisualLayerPlan (const std::vector<CompiledVisualLayerPlan>& plans,
                                    int clipId, LayerDesc& layer, std::string& error,
                                    const VisualInspectionTarget* inspection = nullptr,
                                    VisualInspectionResource* resource = nullptr,
                                    VisualPlanExecutionState* state = nullptr,
                                    double evaluationTimeSec = 0.0,
                                    const std::vector<VisualEventScheduleBinding>* eventSchedules = nullptr,
                                    VisualEventTriggerCursor* eventCursor = nullptr)
{
    const auto evaluationBegin = std::chrono::steady_clock::now();
    // Old snapshots which carry no plans retain their established compositor
    // path. Once any compiled plans are supplied, every rendered owner is strict.
    if (plans.empty()) return true;
    const auto* plan = findVisualLayerPlan (plans, clipId);
    if (plan == nullptr)
    {
        error = "render layer has no compiled visual execution plan";
        return false;
    }
    const bool pausedHold = state != nullptr
        && state->isHold(clipId, plan->structuralRevision, evaluationTimeSec);
    if (state != nullptr && state->needsReset(clipId, plan->structuralRevision, evaluationTimeSec))
    {
        state->reset(clipId);
        state->telemetry().resetOwner(clipId, plan->structuralRevision);
    }
    VisualLayerExecution uncachedExecution;
    const auto* execution = state != nullptr
        ? state->compiled(clipId, plan->structuralRevision) : nullptr;
    if (state != nullptr)
    {
        if (execution == nullptr)
        {
            error = "render layer has no pre-admitted visual execution";
            return false;
        }
        if (! pausedHold) state->telemetry().recordPlanLowering(true, 0, false);
    }
    else
    {
        if (! compileVisualLayerExecution (*plan, uncachedExecution, error)) return false;
        execution = &uncachedExecution;
    }
    if (execution->matteApply
        && (layer.matteTexture == 0 || layer.matteWidth <= 0 || layer.matteHeight <= 0))
    {
        error = "typed matte GPU texture is unavailable";
        return false;
    }
    if (execution->matteCombineMode >= 0
        && (layer.matteTextureB == 0 || layer.matteWidthB <= 0 || layer.matteHeightB <= 0))
    {
        error = "typed matte combine GPU texture is unavailable";
        return false;
    }
    if (state != nullptr)
    {
        layer.feedbackHistoryReset = state->startsSequence(clipId, plan->structuralRevision)
            || state->needsReset(clipId, plan->structuralRevision, evaluationTimeSec);
        layer.feedbackHistoryHold = state->isHold(clipId, plan->structuralRevision, evaluationTimeSec);
        state->prepare (clipId, plan->structuralRevision, evaluationTimeSec);
    }

    if (inspection != nullptr && resource != nullptr && inspection->clipId == clipId
        && inspection->structuralRevision == plan->structuralRevision
        && classifyVisualInspectionTarget(*plan, *inspection) == VisualInspectionSlice::decodedSource)
    {
        // Borrow the genuine source texture already uploaded/decoded for this
        // frame. Generated sources create their texture later inside FrameRenderer.
        if (! layer.shaderSource && ! layer.particleSource && layer.texture != 0
            && layer.texWidth > 0 && layer.texHeight > 0)
        {
            resource->handle = layer.texture;
            resource->width = layer.texWidth;
            resource->height = layer.texHeight;
            resource->nodeId = inspection->nodeId;
            resource->outputPort = inspection->outputPort;
        }
    }

    if (! execution->transform)
    {
        layer.scale = 1.0f;
        layer.translateX = layer.translateY = layer.rotationDeg = 0.0f;
        layer.cropLeft = layer.cropRight = layer.cropTop = layer.cropBottom = 0.0f;
    }
    if (! execution->effects)
    {
        layer.effects = nullptr;
        layer.effectCount = 0;
        layer.lutTexture = 0;
        layer.lutSize = 0;
    }
    if (execution->feedback)
    {
        layer.graphFeedbackEffect = {};
        layer.graphFeedbackEffect.enabled = true;
        layer.graphFeedbackEffect.type = 25; // stable FeedbackTrail wire value
        layer.graphFeedbackEffect.params[0] = execution->feedbackDecay;
        layer.graphFeedbackEffect.params[1] = execution->feedbackZoom;
        layer.graphFeedbackEffect.params[2] = execution->feedbackSwirl;
        layer.effects = &layer.graphFeedbackEffect;
        layer.effectCount = 1;
    }
    if (! execution->mask)
    {
        layer.maskType = 0;
        layer.maskInvert = false;
    }
    layer.matteApply = execution->matteApply;
    layer.depthFog = execution->depthFog;
    layer.depthEffect = execution->depthEffect;
    layer.fogNear = execution->fogNear; layer.fogFar = execution->fogFar;
    layer.fogDensity = execution->fogDensity;
    layer.fogRed = execution->fogRed; layer.fogGreen = execution->fogGreen;
    layer.fogBlue = execution->fogBlue; layer.fogAlpha = execution->fogAlpha;
    layer.depthParam0 = execution->depthParam0; layer.depthParam1 = execution->depthParam1;
    layer.depthParam2 = execution->depthParam2; layer.depthColorRed = execution->depthColorRed;
    layer.depthColorGreen = execution->depthColorGreen; layer.depthColorBlue = execution->depthColorBlue;
    layer.particleSource = execution->particles;
    layer.visualPlanStructuralRevision = execution->structuralRevision;
    layer.visualPlanTelemetryHold = pausedHold;
    layer.particleNodeId = execution->particleNodeId;
    layer.drawShapeNodeId = execution->drawShapeNodeId;
    if (execution->particles)
    {
        layer.particleTriggerConnected = false;
        layer.particleTriggerCount = 0;
        layer.particleTriggerStrength = 0.0f;
        if (eventSchedules != nullptr && eventCursor != nullptr)
        {
            const auto binding = std::find_if (eventSchedules->begin(), eventSchedules->end(),
                [&] (const auto& schedule)
                {
                    return schedule.clipId == clipId && schedule.nodeId == execution->particleNodeId
                        && schedule.portId == 0;
                });
            if (binding != eventSchedules->end())
            {
                layer.particleTriggerConnected = true;
                const auto consumed = eventCursor->consume (*eventSchedules, clipId,
                    execution->particleNodeId, 0, binding->sessionRevision, layer.shaderClock.beat);
                layer.particleTriggerCount = consumed.count;
                layer.particleTriggerStrength = consumed.strongest;
            }
        }
        layer.particleStateReset = state != nullptr && layer.feedbackHistoryReset;
        layer.shaderSource = false;
        layer.texture = 0;
        const auto update = [&](const char* name, double value)
        {
            const auto found = layer.genParams.find(name);
            if (found != layer.genParams.end()) found->second = value;
        };
        update("nativeBuiltin", 1.0); update("seed", execution->particleSeed);
        update("count", execution->particleCount); update("lifetime", execution->particleLifetime);
        update("size", execution->particleSize); update("speed", execution->particleSpeed);
        update("red", execution->particleRed); update("green", execution->particleGreen);
        update("blue", execution->particleBlue); update("alpha", execution->particleAlpha);
        // The caller already populated the canonical viewport/export ShaderClock
        // (project FPS, playing/hold and seek position). Never synthesize 60 Hz.
    }
    layer.matteInvert = execution->matteInvert;
    layer.matteCombineMode = execution->matteCombineMode;
    layer.matteBlack = execution->matteBlack;
    layer.matteWhite = execution->matteWhite;
    layer.matteErodeDilate = execution->matteErodeDilate;
    layer.matteFeather = execution->matteFeather;
    layer.matteChoke = execution->matteChoke;
    layer.drawShape = execution->drawShape;
    layer.inspectionDrawShapeOutput = inspection != nullptr
        && inspection->clipId == clipId
        && inspection->structuralRevision == plan->structuralRevision
        && classifyVisualInspectionTarget(*plan, *inspection)
            == VisualInspectionSlice::retainedDrawShape;
    layer.drawShapeEllipse = execution->drawShapeEllipse;
    if (execution->drawShape)
    {
        layer.drawShapeCx = execution->shapeCx; layer.drawShapeCy = execution->shapeCy;
        layer.drawShapeW = execution->shapeW; layer.drawShapeH = execution->shapeH;
        layer.drawShapeR = execution->shapeR; layer.drawShapeG = execution->shapeG;
        layer.drawShapeB = execution->shapeB; layer.drawShapeA = execution->shapeA;
    }
    if (state != nullptr)
    {
        const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - evaluationBegin).count());
        state->telemetry().recordEvaluation(clipId, plan->structuralRevision, elapsed, pausedHold);
    }
    return true;
}
} // namespace videowire
