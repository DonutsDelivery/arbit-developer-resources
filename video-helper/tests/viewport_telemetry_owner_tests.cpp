#include "../src/viewport_telemetry_owner.h"

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
    videowire::ViewportTelemetryOwner<FakeClock> owner(telemetry, false);
    check(owner.admitBackend(videowire::VisualBackend::openGL), "OpenGL admission");
    check(owner.admitBackend(videowire::VisualBackend::metal), "repeated admission ignored");

    FakeClock::set({10, 10});
    check(owner.renderComposite([] { return 7u; }, [] (unsigned value) { return value != 0; }) == 7,
          "valid compositor output returned");
    FakeClock::set({20, 20});
    check(owner.present([] { return true; }), "successful presentation returned");

    FakeClock::set({30, 35});
    owner.renderComposite([] { return 0u; }, [] (unsigned value) { return value != 0; });
    FakeClock::set({40, 49});
    check(! owner.present([] { return false; }), "presentation failure returned");

    auto snapshot = telemetry.snapshot();
    check(snapshot.compositor.count == 2 && snapshot.compositor.totalNs == 5,
          "zero compositor duration is observed and counted");
    check(snapshot.presentation.count == 2 && snapshot.presentation.totalNs == 9,
          "zero presentation duration is observed and counted");
    check(snapshot.framesRendered == 1 && snapshot.framesPresented == 1,
          "failures never increment success counters");
    check(snapshot.droppedRenderFailure == 1 && snapshot.droppedTransportFailure == 1,
          "renderer and presentation failures map exact drop reasons");
    check(snapshot.initialBackend == videowire::VisualBackend::openGL
          && snapshot.currentBackend == videowire::VisualBackend::openGL
          && snapshot.fallbackCount == 0,
          "backend identity is exact and repeated admission is not fallback");

    check(owner.transitionBackend(videowire::VisualBackend::vulkan), "actual backend transition");
    snapshot = telemetry.snapshot();
    check(snapshot.currentBackend == videowire::VisualBackend::vulkan && snapshot.fallbackCount == 1,
          "actual transition increments fallback once");

    videowire::VisualPlanTelemetry strictTelemetry;
    videowire::ViewportTelemetryOwner<FakeClock> strict(strictTelemetry, true);
    check(strict.admitBackend(videowire::VisualBackend::metal), "strict Metal admission");
    check(! strict.transitionBackend(videowire::VisualBackend::openGL),
          "strict Metal refuses degraded transition");
    const auto strictSnapshot = strictTelemetry.snapshot();
    check(strictSnapshot.initialBackend == videowire::VisualBackend::metal
          && strictSnapshot.currentBackend == videowire::VisualBackend::metal
          && strictSnapshot.fallbackCount == 0 && strictSnapshot.failedLowerings == 1,
          "strict Metal diagnoses without recording degraded pass");
    check(! snapshot.particles.observed && ! snapshot.generators.observed,
          "unowned generator and particle timings remain unavailable");

    videowire::VisualPlanTelemetry transportTelemetry;
    videowire::ViewportTelemetryOwner<FakeClock> transport(transportTelemetry, false);
    check(transport.readbackCopied(true, 32, 3, 8, 3, 8, 3),
          "successful readback payload is admitted");
    auto transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.readbackFrames == 1
          && transportSnapshot.readbackCopiedBytes == 96
          && transportSnapshot.transportFramesHandedOff == 1
          && transportSnapshot.actualWidth == 8 && transportSnapshot.actualHeight == 3,
          "readback records checked stride times height and actual dimensions");
    check(! transport.readbackCopied(false, 32, 3, 8, 3, 8, 3)
          && ! transport.readbackCopied(true, std::numeric_limits<size_t>::max(), 2,
                                        8, 3, 8, 3),
          "failed and overflowing copies are rejected");
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.readbackFrames == 1 && transportSnapshot.readbackCopiedBytes == 96
          && transportSnapshot.droppedTransportFailure == 2,
          "failed copies do not count successful readback");

    check(transport.exportedBufferAllocated(80)
          && transport.exportedBufferAllocated(80), "concrete exported pool allocations");
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyAllocationBytes == 160
          && transportSnapshot.retainedFramesCurrent == 2
          && transportSnapshot.retainedFramesPeak == 2
          && transportSnapshot.intermediateImagesCurrent == 2
          && transportSnapshot.retainedBytesCurrent == 160,
          "pool publishes absolute current and peak snapshots");
    check(! transport.zeroCopyHandoff(false, true, true)
          && ! transport.zeroCopyHandoff(true, false, true)
          && ! transport.zeroCopyHandoff(true, true, false),
          "zero-copy stage failures are rejected");
    transport.noFreeExportedBuffer();
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyFrames == 0
          && transportSnapshot.droppedBlitFailure == 1
          && transportSnapshot.droppedFenceFailure == 1
          && transportSnapshot.droppedSocketFailure == 1
          && transportSnapshot.droppedNoBuffer == 1,
          "exact zero-copy failure reasons never count success");
    check(transport.zeroCopyHandoff(true, true, true), "successful socket handoff");
    transport.exportedBufferReleased(80);
    transport.exportedBufferPoolClosed();
    transportSnapshot = transportTelemetry.snapshot();
    check(transportSnapshot.zeroCopyFrames == 1
          && transportSnapshot.transportFramesHandedOff == 2
          && transportSnapshot.framesPresented == 0,
          "transport handoff is distinct from window presentation");
    check(transportSnapshot.retainedFramesCurrent == 0
          && transportSnapshot.intermediateImagesCurrent == 0
          && transportSnapshot.retainedBytesCurrent == 0
          && transportSnapshot.retainedFramesPeak == 2,
          "pool teardown returns current resources to zero and preserves peak");

    videowire::VisualPlanTelemetry saturated;
    saturated.seedForSaturationTesting(std::numeric_limits<uint64_t>::max());
    saturated.recordTransportHandoff(videowire::VisualTransportMode::zeroCopy);
    saturated.recordDrop(videowire::VisualDropReason::blitFailure);
    saturated.recordDrop(videowire::VisualDropReason::fenceFailure);
    saturated.recordDrop(videowire::VisualDropReason::socketFailure);
    const auto saturatedSnapshot = saturated.snapshot();
    check(saturatedSnapshot.transportFramesHandedOff == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedBlitFailure == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedFenceFailure == std::numeric_limits<uint64_t>::max()
          && saturatedSnapshot.droppedSocketFailure == std::numeric_limits<uint64_t>::max(),
          "viewport transport and stage-specific drop counters saturate");

    std::printf("viewport telemetry owner: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
