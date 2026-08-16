#include "../src/composite_probe_contract.h"
#include "../src/bounded_line.h"
#include "../src/compositor_ownership.h"

#include <cstdio>
#include <limits>
#include <new>
#include <sstream>
#include <string>

namespace
{
int checks = 0;
int failures = 0;
void check (bool condition, const char* message)
{
    ++checks;
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

videowire::ResolvedVisualSnapshot validSnapshot()
{
    videowire::ResolvedVisualSnapshot snapshot;
    snapshot.segments.resize (1);
    snapshot.segments[0].clipId = 4;
    snapshot.visualLayerPlans.resize (1);
    snapshot.visualLayerPlans[0].clipId = 4;
    snapshot.visualLayerPlans[0].structuralRevision = 9;
    snapshot.visualLayerPlans[0].producerValidated = true;
    return snapshot;
}
}

int main()
{
    std::string error;
    auto snapshot = validSnapshot();
    check (videohelper::validateCompositeProbeContract(snapshot, 9, 1.25, 1920, 1080, 30.0, error),
           "processor snapshot is admitted");
    snapshot.visualLayerPlans.push_back(snapshot.visualLayerPlans.front());
    snapshot.visualLayerPlans.back().clipId = 5;
    snapshot.visualLayerPlans.back().structuralRevision = 17;
    check (videohelper::validateCompositeProbeContract(snapshot, 10, 1.25, 1920, 1080, 30.0, error),
           "valid multi-clip snapshots may have unequal authored revisions");
    snapshot.visualLayerPlans[0].identityMode = "transientLegacyProjection";
    snapshot.visualLayerPlans[0].structuralRevision = 0;
    check (videohelper::validateCompositeProbeContract(snapshot, 10, 1.25, 1920, 1080, 30.0, error),
           "mixed legacy and authored plans with unequal revisions are admitted");
    std::vector<videohelper::SnapshotClipIdentity> identities {
        { 4, 0, "transientLegacyProjection", true }, { 5, 17, "authoredGraph", true } };
    check (videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "records match plans by clip id and identity mode");
    identities[0].mode = "authoredGraph";
    check (! videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "record identity-mode mismatch is rejected");
    identities[0] = { 4, 0, "transientLegacyProjection", true };
    identities[1].revision = 18;
    check (! videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "record revision mismatch is rejected");
    snapshot = validSnapshot();
    snapshot.visualLayerPlans[0].producerValidated = false;
    check (! videohelper::validateCompositeProbeContract(snapshot, 9, 1.25, 1920, 1080, 30.0, error),
           "invalid compiled plan is rejected");
    snapshot = validSnapshot();
    check (! videohelper::validateCompositeProbeContract(snapshot, 0, 1.25, 1920, 1080, 30.0, error),
           "unidentified processor snapshot is rejected");
    check (! videohelper::validateCompositeProbeContract(snapshot, 9,
               std::numeric_limits<double>::quiet_NaN(), 1920, 1080, 30.0, error),
           "non-finite time is rejected");
    check (! videohelper::validateCompositeProbeContract(snapshot, 9, 0.0, 1921, 1080, 30.0, error),
           "oversized canvas is rejected");

    std::vector<char> line;
    std::istringstream exact (std::string (64, 'x') + "\n");
    check (videohelper::readBoundedLine (exact, line, 64) == videohelper::BoundedLineResult::line
               && line.size() == 64, "exact-limit helper request is accepted");
    std::istringstream oversized (std::string (65, 'x'));
    check (videohelper::readBoundedLine (oversized, line, 64) == videohelper::BoundedLineResult::oversized
               && line.empty(), "newline-free oversized helper request is rejected at the cap");
    std::istringstream partial ("partial");
    check (videohelper::readBoundedLine (partial, line, 64) == videohelper::BoundedLineResult::line
               && std::string (line.begin(), line.end()) == "partial",
           "bounded parser preserves a final partial line at EOF");
    std::istringstream empty;
    check (videohelper::readBoundedLine (empty, line, 64) == videohelper::BoundedLineResult::end,
           "clean EOF is distinguished from a partial line");

    videohelper::CompositorOwnershipGate gate;
    auto exportLease = gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::exportJob);
    check (exportLease && ! gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::frameProbe)
               && ! gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::renderCache),
           "one atomic gate deterministically excludes probe and cache during export");
    exportLease.reset();
    check (gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::frameProbe) != nullptr,
           "RAII completion releases compositor ownership");

    videohelper::CompositorOwnershipGate allocationFailureGate;
    bool sawBadAlloc = false;
    try
    {
        allocationFailureGate.tryClaimWithFactory(
            videohelper::CompositorOwnershipGate::Owner::exportJob,
            [] (auto&, auto) -> std::shared_ptr<videohelper::CompositorOwnershipGate::Lease>
            { throw std::bad_alloc(); });
    }
    catch (const std::bad_alloc&) { sawBadAlloc = true; }
    check (sawBadAlloc
               && allocationFailureGate.owner() == videohelper::CompositorOwnershipGate::Owner::none
               && allocationFailureGate.tryClaim(
                      videohelper::CompositorOwnershipGate::Owner::renderCache) != nullptr,
           "lease allocation failure restores the gate for the next compositor owner");

    std::printf ("composite probe contract: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
