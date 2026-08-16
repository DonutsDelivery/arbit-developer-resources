#include "../src/export_telemetry_owner.h"

#include <cstdio>
#include <limits>
#include <vector>

namespace
{
struct FakeClock
{
    using time_point = uint64_t;
    static std::vector<uint64_t> ticks;
    static size_t cursor;
    static time_point now() { return ticks.at(cursor++); }
    static uint64_t elapsedNs(time_point begin, time_point end) { return end - begin; }
    static void set(std::initializer_list<uint64_t> values) { ticks = values; cursor = 0; }
};
std::vector<uint64_t> FakeClock::ticks;
size_t FakeClock::cursor = 0;
int failures = 0;
void check(bool value, const char* message)
{
    if (! value) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
}

int main()
{
    videowire::VisualPlanTelemetry telemetry;
    {
        videowire::ExportTelemetryOwner<FakeClock> owner(
            telemetry, videowire::VisualBackend::openGL, 1920, 1080);
        FakeClock::set({10, 10});
        check(owner.renderComposite([] { return true; }, [] (bool valid) { return valid; }),
              "1080p compositor succeeds");
        check(owner.observeFrame(1920, 1080, 1920u * 4u, 1920u * 1080u * 4u),
              "1080p checked readback admitted");
        FakeClock::set({20, 20});
        check(owner.encodedHandoff(123, [] { return true; }), "encoder handoff succeeds");
        const auto live = telemetry.snapshot();
        check(live.backendObserved && live.initialBackend == videowire::VisualBackend::openGL,
              "backend admitted on shared telemetry");
        check(live.dimensionsObserved && live.actualWidth == 1920 && live.actualHeight == 1080,
              "1080p dimensions observed");
        check(live.readbackFrames == 1 && live.readbackCopiedBytes == 1920u * 1080u * 4u,
              "actual checked readback bytes counted");
        check(live.framesRendered == 1 && live.framesPresented == 1
                  && live.exportHandoffFrames == 1,
              "render, presentation, and export handoff counters are distinct");
        check(live.compositor.count == 1 && live.compositor.totalNs == 0
                  && live.presentation.count == 1 && live.presentation.totalNs == 0,
              "zero-duration observations are counted");
        check(! live.particles.observed && ! live.generators.observed,
              "unsupported internal timings remain unavailable");
    }
    auto snapshot = telemetry.snapshot();
    check(snapshot.retainedFramesCurrent == 0 && snapshot.intermediateImagesCurrent == 0
              && snapshot.retainedBytesCurrent == 0,
          "RAII cleanup zeros resources on normal return");

    videowire::VisualPlanTelemetry fourK;
    {
        videowire::ExportTelemetryOwner<FakeClock> owner(
            fourK, videowire::VisualBackend::metal, 3840, 2160);
        check(owner.observeFrame(3840, 2160, 3840u * 4u, 3840ull * 2160ull * 4ull),
              "4K dimensions admitted");
    }
    snapshot = fourK.snapshot();
    check(snapshot.requestedWidth == 3840 && snapshot.requestedHeight == 2160,
          "4K requested dimensions retained");

    videowire::VisualPlanTelemetry mismatch;
    {
        videowire::ExportTelemetryOwner<FakeClock> owner(
            mismatch, videowire::VisualBackend::openGL, 3840, 2160);
        check(! owner.observeFrame(1920, 1080, 1920u * 4u, 1920u * 1080u * 4u),
              "half-resolution mismatch rejected");
    }
    snapshot = mismatch.snapshot();
    check(snapshot.dimensionMismatchCount == 1 && snapshot.halfResolutionMismatchCount == 1
              && snapshot.droppedDimensionMismatch == 1 && snapshot.readbackFrames == 0,
          "half mismatch has one exact failure and no readback success");

    videowire::VisualPlanTelemetry failed;
    {
        videowire::ExportTelemetryOwner<FakeClock> owner(
            failed, videowire::VisualBackend::openGL, 1920, 1080);
        FakeClock::set({1, 2});
        check(! owner.renderComposite([] { return false; }, [] (bool valid) { return valid; }),
              "render failure returned");
        FakeClock::set({3, 4});
        check(! owner.encodedHandoff(99, [] { return false; }), "encoder failure returned");
        owner.releaseAll(); // cancellation/early-return cleanup is idempotent
    }
    snapshot = failed.snapshot();
    check(snapshot.framesRendered == 0 && snapshot.framesPresented == 0
              && snapshot.exportHandoffFrames == 0,
          "failures never increment success counters");
    check(snapshot.droppedRenderFailure == 1 && snapshot.droppedTransportFailure == 1,
          "render and encoder failures use exact reasons");
    check(snapshot.retainedFramesCurrent == 0 && snapshot.intermediateImagesCurrent == 0
              && snapshot.retainedBytesCurrent == 0,
          "cancellation cleanup leaves zero resources");

    videowire::VisualPlanTelemetry overflow;
    {
        videowire::ExportTelemetryOwner<FakeClock> owner(
            overflow, videowire::VisualBackend::openGL, 1, 2);
        check(owner.observeFrame(1, 2, std::numeric_limits<uint64_t>::max(), 0),
              "overflow-sized stride is observed without wrapping");
    }
    snapshot = overflow.snapshot();
    check(snapshot.readbackCopiedBytes == std::numeric_limits<uint64_t>::max()
              && snapshot.retainedBytesPeak == std::numeric_limits<uint64_t>::max(),
          "overflow bytes saturate");

    std::printf("export telemetry owner: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
