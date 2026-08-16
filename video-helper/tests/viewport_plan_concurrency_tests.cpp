#include "../src/visual_plan_publication.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
int failures = 0;
void check(bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

videowire::CompiledVisualLayerPlan planFor(int clipId, uint64_t revision)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = clipId;
    plan.structuralRevision = revision;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.out" };
    plan.nodeIds = { 11, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}
}

int main()
{
    std::string diagnostic;
    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> published;
    check(videowire::makeVisualPlanExecutionSnapshot({ planFor(7, 1) }, published, diagnostic),
          "initial viewport execution snapshot admits synchronously");
    const auto initial = published;

    auto unsupported = planFor(7, 2);
    unsupported.operations[1].backendCapability = "cpu-fallback";
    check(! videowire::makeVisualPlanExecutionSnapshot({ unsupported }, published, diagnostic)
          && published == initial && ! diagnostic.empty()
          && initial->plans[0].structuralRevision == 1,
          "failed viewport admission rolls back without replacing the live snapshot");

    videowire::VisualPlanExecutionState exportAdmission;
    std::vector<videowire::CompiledVisualLayerPlan> tooMany;
    for (size_t i = 0; i <= videowire::VisualPlanExecutionState::kMaxAdmittedPlans; ++i)
        tooMany.push_back(planFor(static_cast<int>(i + 1), 1));
    diagnostic.clear();
    check(! exportAdmission.admitPlans(tooMany, &diagnostic)
          && diagnostic == "visual execution admission exceeds fixed plan capacity"
          && exportAdmission.compiled(1, 1) == nullptr
          && exportAdmission.telemetry().snapshot().failedLowerings == 1,
          "export admission refuses over-capacity plans before installing partial state");

    std::mutex publicationMutex;
    std::atomic<bool> stop { false };
    std::atomic<bool> staleLifetimeSafe { true };
    std::atomic<uint64_t> leasedRevision { 0 };
    std::atomic<uint64_t> leaseCount { 0 };
    std::thread render([&]
    {
        while (! stop.load(std::memory_order_acquire))
        {
            std::shared_ptr<videowire::VisualPlanExecutionSnapshot> frame;
            {
                std::lock_guard<std::mutex> lock(publicationMutex);
                frame = published;
            }
            const auto revision = frame->plans[0].structuralRevision;
            auto* owner = frame->state.prepare(7, revision, static_cast<double>(revision));
            if (owner == nullptr || frame->plans[0].structuralRevision != revision)
                staleLifetimeSafe.store(false, std::memory_order_release);
            leasedRevision.store(revision, std::memory_order_release);
            leaseCount.fetch_add(1, std::memory_order_release);
        }
    });
    while (leasedRevision.load(std::memory_order_acquire) != 1)
        std::this_thread::yield();
    for (uint64_t revision = 2; revision < 200; ++revision)
    {
        std::shared_ptr<videowire::VisualPlanExecutionSnapshot> next;
        check(videowire::makeVisualPlanExecutionSnapshot({ planFor(7, revision) }, next, diagnostic),
              "replacement snapshot admits off render thread");
        std::lock_guard<std::mutex> lock(publicationMutex);
        published = std::move(next);
    }
    while (leasedRevision.load(std::memory_order_acquire) < 2)
        std::this_thread::yield();
    stop.store(true, std::memory_order_release);
    render.join();
    check(leaseCount.load(std::memory_order_acquire) >= 2
          && staleLifetimeSafe.load(std::memory_order_acquire)
          && initial->plans[0].structuralRevision == 1
          && initial->state.compiled(7, 1) != nullptr,
          "concurrent timeline replacement preserves every leased stale frame lifetime");

    std::printf("viewport plan concurrency: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
