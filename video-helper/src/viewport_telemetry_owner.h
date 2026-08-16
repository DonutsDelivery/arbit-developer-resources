#pragma once

#include "visual_plan_telemetry.h"

#include <chrono>
#include <limits>
#include <string>
#include <utility>

namespace videowire
{
struct SteadyTelemetryClock
{
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() { return std::chrono::steady_clock::now(); }
    static uint64_t elapsedNs (time_point begin, time_point end)
    {
        return static_cast<uint64_t> (
            std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count());
    }
};

template <typename Clock = SteadyTelemetryClock>
class ViewportTelemetryOwner
{
public:
    ViewportTelemetryOwner (VisualPlanTelemetry& telemetry, bool strictMetal)
        : telemetry_ (telemetry), strictMetal_ (strictMetal) {}
    ~ViewportTelemetryOwner() { exportedBufferPoolClosed(); }

    bool admitBackend (VisualBackend backend)
    {
        if (strictMetal_ && backend != VisualBackend::metal)
            return diagnoseStrictTransition (backend);
        return telemetry_.admitSessionBackend (backend);
    }

    bool transitionBackend (VisualBackend backend)
    {
        if (strictMetal_ && backend != VisualBackend::metal)
            return diagnoseStrictTransition (backend);
        return telemetry_.recordBackendTransition (backend);
    }

    template <typename RenderCall, typename ValidOutput>
    auto renderComposite (RenderCall&& call, ValidOutput&& validOutput)
    {
        const auto begin = Clock::now();
        auto output = std::forward<RenderCall> (call)();
        const auto end = Clock::now();
        telemetry_.recordCompositorObservation (
            Clock::elapsedNs (begin, end), std::forward<ValidOutput> (validOutput) (output));
        return output;
    }

    template <typename PresentCall>
    bool present (PresentCall&& call)
    {
        const auto begin = Clock::now();
        const bool succeeded = std::forward<PresentCall> (call)();
        const auto end = Clock::now();
        telemetry_.recordPresentationObservation (Clock::elapsedNs (begin, end), succeeded);
        return succeeded;
    }

    static bool checkedPayloadBytes (size_t stride, int height, uint64_t& bytes)
    {
        if (stride == 0 || height <= 0
            || stride > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t> (height))
            return false;
        bytes = static_cast<uint64_t> (stride) * static_cast<uint64_t> (height);
        return true;
    }

    bool readbackCopied (bool copied, size_t stride, int height,
                         int requestedWidth, int requestedHeight,
                         int actualWidth, int actualHeight)
    {
        uint64_t bytes = 0;
        if (! copied || ! checkedPayloadBytes (stride, height, bytes))
        {
            telemetry_.recordDrop (VisualDropReason::transportFailure);
            return false;
        }
        if (! telemetry_.recordDimensions (requestedWidth, requestedHeight, actualWidth, actualHeight))
        {
            telemetry_.recordDrop (VisualDropReason::dimensionMismatch);
            return false;
        }
        return telemetry_.recordTransportHandoff (VisualTransportMode::readback, bytes);
    }

    bool exportedBufferAllocated (uint64_t concreteAllocationBytes)
    {
        if (poolFrames_ == std::numeric_limits<uint64_t>::max()
            || concreteAllocationBytes > std::numeric_limits<uint64_t>::max() - poolBytes_)
            return false;
        ++poolFrames_; poolBytes_ += concreteAllocationBytes;
        if (concreteAllocationBytes != 0)
            telemetry_.recordZeroCopyAllocationBytes (concreteAllocationBytes);
        telemetry_.recordResources (poolFrames_, poolFrames_, poolBytes_);
        return true;
    }

    bool exportedBufferReleased (uint64_t concreteAllocationBytes)
    {
        if (poolFrames_ == 0 || concreteAllocationBytes > poolBytes_)
            return false;
        --poolFrames_; poolBytes_ -= concreteAllocationBytes;
        telemetry_.recordResources (poolFrames_, poolFrames_, poolBytes_);
        return true;
    }

    void exportedBufferPoolClosed()
    {
        poolFrames_ = poolBytes_ = 0;
        telemetry_.recordResources (0, 0, 0);
    }

    bool zeroCopyHandoff (bool blitSucceeded, bool fenceSucceeded, bool socketSucceeded)
    {
        if (! blitSucceeded) { telemetry_.recordDrop (VisualDropReason::blitFailure); return false; }
        if (! fenceSucceeded) { telemetry_.recordDrop (VisualDropReason::fenceFailure); return false; }
        if (! socketSucceeded) { telemetry_.recordDrop (VisualDropReason::socketFailure); return false; }
        return telemetry_.recordTransportHandoff (VisualTransportMode::zeroCopy);
    }

    void noFreeExportedBuffer() { telemetry_.recordDrop (VisualDropReason::noBuffer); }

private:
    bool diagnoseStrictTransition (VisualBackend backend)
    {
        const char* name = backend == VisualBackend::openGL ? "OpenGL"
            : backend == VisualBackend::vulkan ? "Vulkan" : "software";
        telemetry_.recordFailedLowering (
            std::string ("strict Metal viewport rejected backend transition to ") + name);
        return false;
    }

    VisualPlanTelemetry& telemetry_;
    bool strictMetal_ = false;
    uint64_t poolFrames_ = 0;
    uint64_t poolBytes_ = 0;
};
} // namespace videowire
