#pragma once

#include "visual_plan_telemetry.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <utility>

namespace videowire
{
struct ExportSteadyTelemetryClock
{
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() { return std::chrono::steady_clock::now(); }
    static uint64_t elapsedNs (time_point begin, time_point end)
    { return static_cast<uint64_t> (std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()); }
};

template <typename Clock = ExportSteadyTelemetryClock>
class ExportTelemetryOwner
{
public:
    ExportTelemetryOwner (VisualPlanTelemetry& telemetry, VisualBackend backend,
                          int requestedWidth, int requestedHeight)
        : telemetry_ (telemetry), requestedWidth_ (requestedWidth), requestedHeight_ (requestedHeight)
    {
        telemetry_.admitSessionBackend (backend);
        telemetry_.recordResources (0, 0, 0);
    }

    ~ExportTelemetryOwner() { releaseAll(); }

    ExportTelemetryOwner (const ExportTelemetryOwner&) = delete;
    ExportTelemetryOwner& operator= (const ExportTelemetryOwner&) = delete;

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

    bool observeFrame (int actualWidth, int actualHeight, uint64_t strideBytes,
                       uint64_t payloadBytes, uint64_t intermediateImages = 1)
    {
        if (! telemetry_.recordDimensions (requestedWidth_, requestedHeight_, actualWidth, actualHeight))
        {
            lastFrameFailure_ = VisualDropReason::dimensionMismatch;
            telemetry_.recordDrop (VisualDropReason::dimensionMismatch);
            return false;
        }
        if (actualWidth != requestedWidth_ || actualHeight != requestedHeight_)
        {
            lastFrameFailure_ = VisualDropReason::dimensionMismatch;
            telemetry_.recordDrop (VisualDropReason::dimensionMismatch);
            return false;
        }
        const auto height = static_cast<uint64_t> (actualHeight);
        const uint64_t expectedBytes = strideBytes > std::numeric_limits<uint64_t>::max() / height
            ? std::numeric_limits<uint64_t>::max() : strideBytes * height;
        if (payloadBytes != 0 && payloadBytes != expectedBytes)
        {
            lastFrameFailure_ = VisualDropReason::transportFailure;
            telemetry_.recordDrop (VisualDropReason::transportFailure);
            return false;
        }
        const uint64_t concreteBytes = payloadBytes != 0 ? payloadBytes : expectedBytes;
        frames_ = 1;
        images_ = intermediateImages;
        bytes_ = concreteBytes;
        telemetry_.recordResources (frames_, images_, bytes_);
        telemetry_.recordReadbackTransfer (concreteBytes);
        lastFrameFailure_ = VisualDropReason::noBuffer;
        return true;
    }

    template <typename HandoffCall>
    bool encodedHandoff (uint64_t payloadBytes, HandoffCall&& call)
    {
        const auto begin = Clock::now();
        const bool succeeded = std::forward<HandoffCall> (call)();
        const auto end = Clock::now();
        telemetry_.recordPresentationObservation (Clock::elapsedNs (begin, end), succeeded,
                                                   VisualPresentationMode::encodedHandoff);
        if (succeeded)
        {
            bytes_ = payloadBytes;
            telemetry_.recordResources (frames_, images_, bytes_);
        }
        return succeeded;
    }

    VisualPlanTelemetry& telemetry() { return telemetry_; }
    VisualDropReason lastFrameFailure() const { return lastFrameFailure_; }
    void releaseAll()
    {
        frames_ = images_ = bytes_ = 0;
        telemetry_.recordResources (0, 0, 0);
    }

private:
    VisualPlanTelemetry& telemetry_;
    int requestedWidth_ = 0, requestedHeight_ = 0;
    uint64_t frames_ = 0, images_ = 0, bytes_ = 0;
    VisualDropReason lastFrameFailure_ = VisualDropReason::noBuffer;
};
} // namespace videowire
