#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace videowire
{
inline constexpr size_t kMaxTelemetryPlans = 64;
inline constexpr size_t kMaxCompiledNodesPerGraph = 8;
inline constexpr size_t kMaxIntermediateImagesPerGraph = 7;
inline constexpr size_t kMaxTelemetryNodes = kMaxTelemetryPlans * kMaxCompiledNodesPerGraph;
inline constexpr uint64_t kPlanLoweringBudgetNs = 2'000'000;

enum class VisualTransportMode : uint8_t { none, zeroCopy, readback };
enum class VisualBackend : uint8_t { openGL, metal, vulkan, software };
enum class VisualPresentationMode : uint8_t { none, windowPresent, encodedHandoff };
enum class VisualExecutionKind : uint8_t { generator, particle };
enum class VisualDropReason : uint8_t { noBuffer, transportFailure, renderFailure, dimensionMismatch,
                                        blitFailure, fenceFailure, socketFailure };
struct VisualDurationCounter { bool observed = false; uint64_t count = 0, totalNs = 0; double movingNs = 0.0; };

struct VisualTelemetryPlanAdmission
{
    int clipId = -1;
    uint64_t structuralRevision = 0;
    std::array<int, kMaxCompiledNodesPerGraph> stableNodeIds {};
    uint8_t nodeCount = 0;
    uint8_t intermediateImageCount = 0;
    size_t executableNodeTotal = 0;
    bool nodesTruncated = false;
};

struct VisualTelemetryNode
{
    int clipId = -1;
    uint64_t structuralRevision = 0;
    int stableNodeId = 0;
    bool available = false;
    uint64_t evaluations = 0;
    uint64_t measuredTotalNs = 0;
    double measuredMovingNs = 0.0;
};

struct VisualTelemetryLayer
{
    int clipId = -1;
    uint64_t structuralRevision = 0;
    uint64_t evaluations = 0;
    uint64_t measuredTotalNs = 0;
    double measuredMovingNs = 0.0;
};

struct VisualTelemetrySnapshot
{
    uint64_t graphEvaluations = 0;
    uint64_t graphMeasuredTotalNs = 0;
    double graphMeasuredMovingNs = 0.0;
    uint64_t planCacheHits = 0;
    uint64_t planCacheMisses = 0;
    uint64_t lastPlanLoweringNs = 0;
    uint64_t planLoweringBudgetNs = kPlanLoweringBudgetNs;
    bool lastPlanLoweringWithinBudget = true;
    uint64_t planInstalls = 0;
    uint64_t droppedSamples = 0;
    uint64_t rejectedPlans = 0;
    uint64_t failedLowerings = 0;
    std::string lastLoweringError;
    size_t executableNodeTotal = 0;
    bool executableNodeSubsetTruncated = false;
    VisualDurationCounter compositor, presentation, particles, generators;
    bool transportObserved = false, zeroCopyAllocationObserved = false;
    VisualTransportMode transportMode = VisualTransportMode::none;
    uint64_t zeroCopyFrames = 0, zeroCopyAllocationBytes = 0, readbackFrames = 0, readbackCopiedBytes = 0;
    uint64_t framesRendered = 0, framesPresented = 0, exportHandoffFrames = 0;
    bool presentationModeObserved = false;
    VisualPresentationMode presentationMode = VisualPresentationMode::none;
    uint64_t transportFramesHandedOff = 0;
    bool dropsObserved = false;
    uint64_t framesDropped = 0;
    uint64_t droppedNoBuffer = 0, droppedTransportFailure = 0, droppedRenderFailure = 0, droppedDimensionMismatch = 0;
    uint64_t droppedBlitFailure = 0, droppedFenceFailure = 0, droppedSocketFailure = 0;
    bool backendObserved = false;
    VisualBackend initialBackend = VisualBackend::openGL, currentBackend = VisualBackend::openGL;
    uint64_t fallbackCount = 0;
    bool resourcesObserved = false;
    uint64_t retainedFramesCurrent = 0, retainedFramesPeak = 0, intermediateImagesCurrent = 0, intermediateImagesPeak = 0;
    uint64_t retainedBytesCurrent = 0, retainedBytesPeak = 0;
    bool dimensionsObserved = false;
    int requestedWidth = 0, requestedHeight = 0, actualWidth = 0, actualHeight = 0;
    uint64_t dimensionMismatchCount = 0, halfResolutionMismatchCount = 0, recordingContentionDrops = 0;
    std::vector<VisualTelemetryLayer> layers;
    std::vector<VisualTelemetryNode> nodes;
};

inline uint64_t saturatingAdd (uint64_t value, uint64_t increment)
{
    return increment > std::numeric_limits<uint64_t>::max() - value
        ? std::numeric_limits<uint64_t>::max() : value + increment;
}

class VisualPlanTelemetry
{
public:
    // Session admission is a control event. Repeated admission is ignored; backend
    // changes are recorded only by the explicit transition API.
    bool admitSessionBackend (VisualBackend backend)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        if (! backendObserved_) { backendObserved_ = true; initialBackend_ = currentBackend_ = backend; }
        return true;
    }
    bool recordBackendTransition (VisualBackend backend)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        if (! backendObserved_) return false;
        if (currentBackend_ != backend)
        {
            currentBackend_ = backend;
            fallbackCount_ = saturatingAdd(fallbackCount_, 1);
        }
        return true;
    }
    bool recordCompositorObservation (uint64_t durationNs, bool succeeded)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        addDuration(compositor_, durationNs);
        if (succeeded) framesRendered_ = saturatingAdd(framesRendered_, 1);
        else recordDropUnlocked(VisualDropReason::renderFailure);
        return true;
    }
    bool recordPresentationObservation (uint64_t durationNs, bool succeeded,
                                        VisualPresentationMode mode = VisualPresentationMode::windowPresent)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        addDuration(presentation_, durationNs);
        presentationModeObserved_ = true; presentationMode_ = mode;
        if (succeeded)
        {
            framesPresented_ = saturatingAdd(framesPresented_, 1);
            if (mode == VisualPresentationMode::encodedHandoff)
                exportHandoffFrames_ = saturatingAdd(exportHandoffFrames_, 1);
        }
        else recordDropUnlocked(VisualDropReason::transportFailure);
        return true;
    }
    bool recordExecutionObservation (VisualExecutionKind kind, uint64_t durationNs)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        applyResetAndPending();
        addDuration(kind == VisualExecutionKind::generator ? generators_ : particles_, durationNs);
        return true;
    }
    bool recordRuntime (uint64_t compositorNs, uint64_t presentationNs, uint64_t,
                        uint64_t, VisualTransportMode mode, bool presented)
    {
        if (mode == VisualTransportMode::none) return false;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        addDuration(compositor_, compositorNs); addDuration(presentation_, presentationNs);
        transportObserved_ = true; transportMode_ = mode;
        framesRendered_ = saturatingAdd(framesRendered_, 1);
        if (presented) framesPresented_ = saturatingAdd(framesPresented_, 1);
        auto& frames = mode == VisualTransportMode::zeroCopy ? zeroCopyFrames_ : readbackFrames_;
        frames = saturatingAdd(frames, 1);
        return true;
    }
    bool recordReadbackCopiedBytes (uint64_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        transportObserved_ = true; readbackCopiedBytes_ = saturatingAdd(readbackCopiedBytes_, bytes); return true;
    }
    bool recordZeroCopyAllocationBytes (uint64_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        zeroCopyAllocationObserved_ = true;
        zeroCopyAllocationBytes_ = saturatingAdd(zeroCopyAllocationBytes_, bytes); return true;
    }
    // Producer transport handoff is distinct from local window presentation.
    bool recordTransportHandoff (VisualTransportMode mode, uint64_t copiedBytes = 0)
    {
        if (mode == VisualTransportMode::none || (mode == VisualTransportMode::zeroCopy && copiedBytes != 0))
            return false;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        transportObserved_ = true; transportMode_ = mode;
        transportFramesHandedOff_ = saturatingAdd(transportFramesHandedOff_, 1);
        if (mode == VisualTransportMode::readback)
        {
            readbackFrames_ = saturatingAdd(readbackFrames_, 1);
            readbackCopiedBytes_ = saturatingAdd(readbackCopiedBytes_, copiedBytes);
        }
        else
            zeroCopyFrames_ = saturatingAdd(zeroCopyFrames_, 1);
        return true;
    }
    bool recordReadbackTransfer (uint64_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        transportObserved_ = true; transportMode_ = VisualTransportMode::readback;
        readbackFrames_ = saturatingAdd(readbackFrames_, 1);
        readbackCopiedBytes_ = saturatingAdd(readbackCopiedBytes_, bytes);
        return true;
    }
    bool recordDrop (VisualDropReason reason)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        recordDropUnlocked(reason); return true;
    }
    bool recordResources (uint64_t frames, uint64_t images, uint64_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        resourcesObserved_ = true; retainedFramesCurrent_ = frames; intermediateImagesCurrent_ = images; retainedBytesCurrent_ = bytes;
        retainedFramesPeak_ = std::max(retainedFramesPeak_, frames); intermediateImagesPeak_ = std::max(intermediateImagesPeak_, images);
        retainedBytesPeak_ = std::max(retainedBytesPeak_, bytes); return true;
    }
    bool recordDimensions (int rw, int rh, int aw, int ah)
    {
        constexpr int maximumDimension = 32768;
        const auto valid = [](int value) { return value > 0 && value <= maximumDimension; };
        if (! valid(rw) || ! valid(rh) || ! valid(aw) || ! valid(ah)) return false;
        const uint64_t requestedPixels = uint64_t(rw) * uint64_t(rh);
        const uint64_t actualPixels = uint64_t(aw) * uint64_t(ah);
        if (requestedPixels > uint64_t(maximumDimension) * uint64_t(maximumDimension)
            || actualPixels > uint64_t(maximumDimension) * uint64_t(maximumDimension)) return false;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        dimensionsObserved_ = true; requestedWidth_ = rw; requestedHeight_ = rh; actualWidth_ = aw; actualHeight_ = ah;
        if (rw != aw || rh != ah)
        {
            dimensionMismatchCount_ = saturatingAdd(dimensionMismatchCount_, 1);
            if (rw % 2 == 0 && rh % 2 == 0 && rw / 2 == aw && rh / 2 == ah)
                halfResolutionMismatchCount_ = saturatingAdd(halfResolutionMismatchCount_, 1);
        }
        return true;
    }
    bool admitPlans (const std::vector<VisualTelemetryPlanAdmission>& requested,
                     std::string* diagnostic = nullptr)
    {
        if (diagnostic != nullptr)
            diagnostic->clear();
        std::lock_guard<std::mutex> lock (mutex_);
        std::array<Slot, kMaxTelemetryPlans> next {};
        size_t count = 0;
        uint64_t rejected = 0;
        for (const auto& candidate : requested)
        {
            const auto duplicate = std::find_if(next.begin(), next.begin() + (ptrdiff_t) count,
                [&](const Slot& slot) { return slot.clipId == candidate.clipId; });
            if (duplicate != next.begin() + (ptrdiff_t) count)
            {
                if (candidate.structuralRevision <= duplicate->revision) continue;
                *duplicate = makeSlot(candidate);
                continue;
            }
            if (count >= kMaxTelemetryPlans)
            {
                ++rejected;
                continue;
            }
            next[count++] = makeSlot(candidate);
        }
        std::sort(next.begin(), next.begin() + (ptrdiff_t) count,
            [](const Slot& a, const Slot& b) { return a.clipId < b.clipId; });
        slots_ = next;
        slotCount_ = count;
        rejectedPlans_ = saturatingAdd(rejectedPlans_, rejected);
        requestReset();
        if (rejected != 0 && diagnostic != nullptr)
            *diagnostic = "visual telemetry admission exceeds fixed plan capacity";
        return rejected == 0;
    }

    void resetOwner (int, uint64_t) { requestReset(); }

    // Graph/revision and backward-seek reset uses resetOwner(): it clears runtime
    // aggregates, dimensions, and resource current values while retaining peaks.
    // A new viewport/export session must call this control-thread operation to
    // additionally forget initial/current backend and fallback history.
    void resetSession()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        resetUnlocked();
        backendObserved_ = false;
        initialBackend_ = currentBackend_ = VisualBackend::openGL;
        fallbackCount_ = 0;
        appliedGeneration_ = resetGeneration_.load(std::memory_order_acquire);
    }

    bool recordFailedLowering (const std::string& error)
    {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (! lock.owns_lock()) return dropRecording();
        failedLowerings_ = saturatingAdd(failedLowerings_, 1);
        lastLoweringError_ = error;
        return true;
    }

    void recordPlanLowering (bool cacheHit, uint64_t durationNs, bool installed)
    {
        std::unique_lock<std::mutex> lock (mutex_, std::try_to_lock);
        if (! lock.owns_lock())
        {
            enqueuePending({ resetGeneration_.load(std::memory_order_acquire), cacheHit,
                             installed, ! cacheHit, durationNs });
            saturatingIncrementAtomic(droppedSamples_);
            return;
        }
        applyResetAndPending();
        planCacheHits_ = saturatingAdd(planCacheHits_, cacheHit ? 1 : 0);
        planCacheMisses_ = saturatingAdd(planCacheMisses_, cacheHit ? 0 : 1);
        planInstalls_ = saturatingAdd(planInstalls_, installed ? 1 : 0);
        if (! cacheHit)
        {
            lastPlanLoweringNs_ = durationNs;
            lastPlanLoweringWithinBudget_ = durationNs <= kPlanLoweringBudgetNs;
        }
    }

    void recordEvaluation (int clipId, uint64_t revision, uint64_t durationNs, bool pausedHold)
    {
        if (pausedHold) return;
        std::unique_lock<std::mutex> lock (mutex_, std::try_to_lock);
        if (! lock.owns_lock()) { saturatingIncrementAtomic(droppedSamples_); return; }
        applyResetAndPending();
        auto* slot = findSlot(clipId, revision);
        if (slot == nullptr) return;
        graphEvaluations_ = saturatingAdd(graphEvaluations_, 1);
        graphMeasuredTotalNs_ = saturatingAdd(graphMeasuredTotalNs_, durationNs);
        graphMeasuredMovingNs_ = moving(graphMeasuredMovingNs_, durationNs, slot->evaluations == 0);
        slot->evaluations = saturatingAdd(slot->evaluations, 1);
        slot->measuredTotalNs = saturatingAdd(slot->measuredTotalNs, durationNs);
        slot->measuredMovingNs = moving(slot->measuredMovingNs, durationNs, slot->evaluations == 1);
    }

    // Record only a duration measured around one concrete typed render operation.
    // Unwired admitted nodes remain unavailable; aggregate layer time is never split.
    bool recordNodeEvaluation (int clipId, uint64_t revision, int stableNodeId,
                               uint64_t durationNs, bool pausedHold = false)
    {
        if (pausedHold) return true;
        std::unique_lock<std::mutex> lock (mutex_, std::try_to_lock);
        if (! lock.owns_lock()) { saturatingIncrementAtomic(droppedSamples_); return false; }
        applyResetAndPending();
        auto* slot = findSlot(clipId, revision);
        if (slot == nullptr) return false;
        const auto found = std::lower_bound(slot->nodeIds.begin(),
            slot->nodeIds.begin() + (ptrdiff_t) slot->nodeCount, stableNodeId);
        if (found == slot->nodeIds.begin() + (ptrdiff_t) slot->nodeCount
            || *found != stableNodeId) return false;
        const size_t index = static_cast<size_t>(std::distance(slot->nodeIds.begin(), found));
        const bool first = ! slot->nodeAvailable[index];
        slot->nodeAvailable[index] = true;
        slot->nodeEvaluations[index] = saturatingAdd(slot->nodeEvaluations[index], 1);
        slot->nodeMeasuredTotalNs[index] = saturatingAdd(slot->nodeMeasuredTotalNs[index], durationNs);
        slot->nodeMeasuredMovingNs[index] = moving(slot->nodeMeasuredMovingNs[index], durationNs, first);
        return true;
    }

    VisualTelemetrySnapshot snapshot() const
    {
        std::lock_guard<std::mutex> lock (mutex_);
        const_cast<VisualPlanTelemetry*>(this)->applyResetAndPending();
        VisualTelemetrySnapshot out;
        out.graphEvaluations = graphEvaluations_;
        out.graphMeasuredTotalNs = graphMeasuredTotalNs_;
        out.graphMeasuredMovingNs = graphMeasuredMovingNs_;
        out.planCacheHits = planCacheHits_; out.planCacheMisses = planCacheMisses_;
        out.lastPlanLoweringNs = lastPlanLoweringNs_;
        out.lastPlanLoweringWithinBudget = lastPlanLoweringWithinBudget_;
        out.planInstalls = planInstalls_;
        out.droppedSamples = droppedSamples_.load(std::memory_order_relaxed);
        out.rejectedPlans = rejectedPlans_;
        out.failedLowerings = failedLowerings_;
        out.lastLoweringError = lastLoweringError_;
        out.layers.reserve(slotCount_); out.nodes.reserve(slotCount_ * kMaxCompiledNodesPerGraph);
        for (size_t s = 0; s < slotCount_; ++s)
        {
            const auto& slot = slots_[s];
            out.executableNodeTotal += slot.executableNodeTotal;
            out.executableNodeSubsetTruncated = out.executableNodeSubsetTruncated || slot.nodesTruncated;
            out.layers.push_back({ slot.clipId, slot.revision, slot.evaluations,
                                   slot.measuredTotalNs, slot.measuredMovingNs });
            for (size_t n = 0; n < slot.nodeCount; ++n)
                out.nodes.push_back({ slot.clipId, slot.revision, slot.nodeIds[n], slot.nodeAvailable[n],
                    slot.nodeEvaluations[n], slot.nodeMeasuredTotalNs[n], slot.nodeMeasuredMovingNs[n] });
        }
        out.compositor = compositor_; out.presentation = presentation_; out.particles = particles_; out.generators = generators_;
        out.transportObserved = transportObserved_; out.zeroCopyAllocationObserved = zeroCopyAllocationObserved_; out.transportMode = transportMode_; out.zeroCopyFrames = zeroCopyFrames_; out.zeroCopyAllocationBytes = zeroCopyAllocationBytes_; out.readbackFrames = readbackFrames_; out.readbackCopiedBytes = readbackCopiedBytes_; out.transportFramesHandedOff = transportFramesHandedOff_;
        out.framesRendered = framesRendered_; out.framesPresented = framesPresented_; out.exportHandoffFrames = exportHandoffFrames_; out.framesDropped = framesDropped_;
        out.presentationModeObserved = presentationModeObserved_; out.presentationMode = presentationMode_;
        out.dropsObserved = dropsObserved_;
        out.droppedNoBuffer = droppedNoBuffer_; out.droppedTransportFailure = droppedTransportFailure_; out.droppedRenderFailure = droppedRenderFailure_; out.droppedDimensionMismatch = droppedDimensionMismatch_;
        out.droppedBlitFailure = droppedBlitFailure_; out.droppedFenceFailure = droppedFenceFailure_; out.droppedSocketFailure = droppedSocketFailure_;
        out.backendObserved = backendObserved_; out.initialBackend = initialBackend_; out.currentBackend = currentBackend_; out.fallbackCount = fallbackCount_;
        out.resourcesObserved = resourcesObserved_;
        out.retainedFramesCurrent = retainedFramesCurrent_; out.retainedFramesPeak = retainedFramesPeak_; out.intermediateImagesCurrent = intermediateImagesCurrent_; out.intermediateImagesPeak = intermediateImagesPeak_;
        out.retainedBytesCurrent = retainedBytesCurrent_; out.retainedBytesPeak = retainedBytesPeak_; out.dimensionsObserved = dimensionsObserved_; out.requestedWidth = requestedWidth_; out.requestedHeight = requestedHeight_; out.actualWidth = actualWidth_; out.actualHeight = actualHeight_;
        out.dimensionMismatchCount = dimensionMismatchCount_; out.halfResolutionMismatchCount = halfResolutionMismatchCount_; out.recordingContentionDrops = recordingContentionDrops_.load();
        return out;
    }

    // Test-only contention seam; production never waits on this lock from render.
    std::unique_lock<std::mutex> lockForTesting() { return std::unique_lock<std::mutex>(mutex_); }
    void seedForSaturationTesting (uint64_t value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        graphEvaluations_ = graphMeasuredTotalNs_ = planCacheHits_ = planCacheMisses_ = planInstalls_ = value;
        compositor_.observed = presentation_.observed = particles_.observed = generators_.observed = true;
        compositor_.count = compositor_.totalNs = presentation_.count = presentation_.totalNs = value;
        particles_.count = particles_.totalNs = generators_.count = generators_.totalNs = value;
        zeroCopyFrames_ = zeroCopyAllocationBytes_ = readbackFrames_ = readbackCopiedBytes_ = transportFramesHandedOff_ = value;
        framesRendered_ = framesPresented_ = exportHandoffFrames_ = framesDropped_ = droppedNoBuffer_ = droppedTransportFailure_ = value;
        droppedRenderFailure_ = droppedDimensionMismatch_ = droppedBlitFailure_ = droppedFenceFailure_ = droppedSocketFailure_ = value;
        fallbackCount_ = dimensionMismatchCount_ = halfResolutionMismatchCount_ = value;
        retainedFramesPeak_ = intermediateImagesPeak_ = retainedBytesPeak_ = value;
        if (slotCount_ != 0)
        {
            slots_[0].evaluations = slots_[0].measuredTotalNs = slots_[0].nodeEvaluations[0] = value;
            slots_[0].nodeAvailable[0] = true;
            slots_[0].nodeMeasuredTotalNs[0] = value;
        }
    }

private:
    void recordDropUnlocked (VisualDropReason reason)
    {
        dropsObserved_ = true; framesDropped_ = saturatingAdd(framesDropped_, 1);
        uint64_t* c = reason == VisualDropReason::noBuffer ? &droppedNoBuffer_ : reason == VisualDropReason::transportFailure
            ? &droppedTransportFailure_ : reason == VisualDropReason::renderFailure ? &droppedRenderFailure_
            : reason == VisualDropReason::dimensionMismatch ? &droppedDimensionMismatch_
            : reason == VisualDropReason::blitFailure ? &droppedBlitFailure_
            : reason == VisualDropReason::fenceFailure ? &droppedFenceFailure_ : &droppedSocketFailure_;
        *c = saturatingAdd(*c, 1);
    }
    static void addDuration (VisualDurationCounter& d, uint64_t ns)
    { const bool first = ! d.observed; d.observed = true; d.count = saturatingAdd(d.count, 1); d.totalNs = saturatingAdd(d.totalNs, ns); d.movingNs = moving(d.movingNs, ns, first); }
    struct Slot
    {
        int clipId = -1;
        uint64_t revision = 0;
        std::array<int, kMaxCompiledNodesPerGraph> nodeIds {};
        std::array<bool, kMaxCompiledNodesPerGraph> nodeAvailable {};
        std::array<uint64_t, kMaxCompiledNodesPerGraph> nodeEvaluations {};
        std::array<uint64_t, kMaxCompiledNodesPerGraph> nodeMeasuredTotalNs {};
        std::array<double, kMaxCompiledNodesPerGraph> nodeMeasuredMovingNs {};
        size_t nodeCount = 0;
        size_t executableNodeTotal = 0;
        bool nodesTruncated = false;
        uint64_t evaluations = 0, measuredTotalNs = 0;
        double measuredMovingNs = 0.0;
    };
    static Slot makeSlot (const VisualTelemetryPlanAdmission& value)
    {
        Slot slot; slot.clipId = value.clipId; slot.revision = value.structuralRevision;
        slot.nodeCount = value.nodeCount;
        slot.executableNodeTotal = value.executableNodeTotal;
        slot.nodesTruncated = value.nodesTruncated;
        std::copy_n(value.stableNodeIds.begin(), value.nodeCount, slot.nodeIds.begin());
        std::sort(slot.nodeIds.begin(), slot.nodeIds.begin() + (ptrdiff_t) slot.nodeCount);
        slot.nodeCount = (size_t) std::distance(slot.nodeIds.begin(),
            std::unique(slot.nodeIds.begin(), slot.nodeIds.begin() + (ptrdiff_t) slot.nodeCount));
        return slot;
    }
    static double moving (double previous, uint64_t sample, bool first)
    { return first ? static_cast<double>(sample) : previous * 0.8 + static_cast<double>(sample) * 0.2; }
    Slot* findSlot (int clipId, uint64_t revision)
    {
        const auto it = std::lower_bound(slots_.begin(), slots_.begin() + (ptrdiff_t) slotCount_, clipId,
            [](const Slot& slot, int id) { return slot.clipId < id; });
        return it != slots_.begin() + (ptrdiff_t) slotCount_ && it->clipId == clipId && it->revision == revision
            ? &*it : nullptr;
    }
    void requestReset()
    {
        auto current = resetGeneration_.load(std::memory_order_relaxed);
        while (! resetGeneration_.compare_exchange_weak(current, current + 1,
                                                        std::memory_order_release)) {}
    }
    void resetUnlocked()
    {
        graphEvaluations_ = graphMeasuredTotalNs_ = 0; graphMeasuredMovingNs_ = 0.0;
        planCacheHits_ = planCacheMisses_ = lastPlanLoweringNs_ = planInstalls_ = 0;
        lastPlanLoweringWithinBudget_ = true;
        compositor_ = {}; presentation_ = {}; particles_ = {}; generators_ = {};
        presentationModeObserved_ = false; presentationMode_ = VisualPresentationMode::none;
        transportObserved_ = zeroCopyAllocationObserved_ = false; transportMode_ = VisualTransportMode::none;
        zeroCopyFrames_ = zeroCopyAllocationBytes_ = readbackFrames_ = readbackCopiedBytes_ = transportFramesHandedOff_ = 0;
        framesRendered_ = framesPresented_ = exportHandoffFrames_ = 0;
        dropsObserved_ = false; framesDropped_ = droppedNoBuffer_ = droppedTransportFailure_ = 0;
        droppedRenderFailure_ = droppedDimensionMismatch_ = droppedBlitFailure_ = droppedFenceFailure_ = droppedSocketFailure_ = 0;
        resourcesObserved_ = false; retainedFramesCurrent_ = intermediateImagesCurrent_ = retainedBytesCurrent_ = 0;
        dimensionsObserved_ = false; requestedWidth_ = requestedHeight_ = actualWidth_ = actualHeight_ = 0;
        dimensionMismatchCount_ = halfResolutionMismatchCount_ = 0;
        recordingContentionDrops_.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < slotCount_; ++i)
        {
            slots_[i].evaluations = slots_[i].measuredTotalNs = 0;
            slots_[i].measuredMovingNs = 0.0;
            slots_[i].nodeAvailable.fill(false);
            slots_[i].nodeEvaluations.fill(0);
            slots_[i].nodeMeasuredTotalNs.fill(0);
            slots_[i].nodeMeasuredMovingNs.fill(0.0);
        }
    }
    void applyResetAndPending()
    {
        const auto requested = resetGeneration_.load(std::memory_order_acquire);
        if (appliedGeneration_ != requested) { resetUnlocked(); appliedGeneration_ = requested; }
        for (auto& slot : pending_)
        {
            if (! slot.ready.exchange(false, std::memory_order_acq_rel)) continue;
            const auto sample = slot.sample;
            if (sample.generation == requested)
            {
                planCacheHits_ = saturatingAdd(planCacheHits_, sample.cacheHit ? 1 : 0);
                planCacheMisses_ = saturatingAdd(planCacheMisses_, sample.cacheHit ? 0 : 1);
                planInstalls_ = saturatingAdd(planInstalls_, sample.installed ? 1 : 0);
                if (sample.hasDuration)
                {
                    lastPlanLoweringNs_ = sample.durationNs;
                    lastPlanLoweringWithinBudget_ = sample.durationNs <= kPlanLoweringBudgetNs;
                }
            }
            slot.claimed.clear(std::memory_order_release);
        }
    }
    struct PendingSample
    {
        uint64_t generation = 0;
        bool cacheHit = false, installed = false, hasDuration = false;
        uint64_t durationNs = 0;
    };
    struct PendingSlot
    {
        std::atomic_flag claimed = ATOMIC_FLAG_INIT;
        std::atomic<bool> ready { false };
        PendingSample sample;
    };
    void enqueuePending (PendingSample sample)
    {
        for (auto& slot : pending_)
            if (! slot.claimed.test_and_set(std::memory_order_acquire))
            {
                slot.sample = sample;
                slot.ready.store(true, std::memory_order_release);
                return;
            }
    }
    static void saturatingIncrementAtomic (std::atomic<uint64_t>& value)
    {
        auto current = value.load(std::memory_order_relaxed);
        while (current != std::numeric_limits<uint64_t>::max()
               && ! value.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {}
    }
    bool dropRecording()
    {
        saturatingIncrementAtomic(recordingContentionDrops_);
        saturatingIncrementAtomic(droppedSamples_);
        return false;
    }

    mutable std::mutex mutex_;
    std::array<Slot, kMaxTelemetryPlans> slots_ {};
    size_t slotCount_ = 0;
    uint64_t graphEvaluations_ = 0, graphMeasuredTotalNs_ = 0;
    double graphMeasuredMovingNs_ = 0.0;
    uint64_t planCacheHits_ = 0, planCacheMisses_ = 0, lastPlanLoweringNs_ = 0, planInstalls_ = 0;
    bool lastPlanLoweringWithinBudget_ = true;
    uint64_t rejectedPlans_ = 0, failedLowerings_ = 0, appliedGeneration_ = 0;
    std::string lastLoweringError_;
    std::atomic<uint64_t> resetGeneration_ { 0 }, droppedSamples_ { 0 };
    std::array<PendingSlot, 64> pending_ {};
    VisualDurationCounter compositor_, presentation_, particles_, generators_;
    bool presentationModeObserved_ = false; VisualPresentationMode presentationMode_ = VisualPresentationMode::none;
    bool transportObserved_ = false, zeroCopyAllocationObserved_ = false;
    VisualTransportMode transportMode_ = VisualTransportMode::none;
    uint64_t zeroCopyFrames_=0, zeroCopyAllocationBytes_=0, readbackFrames_=0, readbackCopiedBytes_=0, transportFramesHandedOff_=0, framesRendered_=0, framesPresented_=0, exportHandoffFrames_=0;
    bool dropsObserved_ = false; uint64_t framesDropped_=0;
    uint64_t droppedNoBuffer_=0, droppedTransportFailure_=0, droppedRenderFailure_=0, droppedDimensionMismatch_=0;
    uint64_t droppedBlitFailure_=0, droppedFenceFailure_=0, droppedSocketFailure_=0;
    bool backendObserved_ = false; VisualBackend initialBackend_ = VisualBackend::openGL, currentBackend_ = VisualBackend::openGL; uint64_t fallbackCount_=0;
    bool resourcesObserved_ = false; uint64_t retainedFramesCurrent_=0, retainedFramesPeak_=0, intermediateImagesCurrent_=0, intermediateImagesPeak_=0, retainedBytesCurrent_=0, retainedBytesPeak_=0;
    bool dimensionsObserved_ = false; int requestedWidth_=0, requestedHeight_=0, actualWidth_=0, actualHeight_=0; uint64_t dimensionMismatchCount_=0, halfResolutionMismatchCount_=0;
    std::atomic<uint64_t> recordingContentionDrops_ { 0 };
};
} // namespace videowire
