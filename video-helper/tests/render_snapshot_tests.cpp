#include "../src/render_snapshot_json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

using json = nlohmann::json;

namespace
{
int failures = 0;
int checks = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

std::string noShader (const json&) { return {}; }

json representativeSnapshot()
{
    return {
        { "authoringRevision", 12 },
        { "matteContentRevision", 7 },
        { "segments", {
            {
                { "sourceKind", "media" }, { "sourcePath", "/media/a.mov" },
                { "clipId", 10 }, { "trackLayer", 0 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },

                { "displayStartSec", 0.0 },
                { "matteAssetId", "matte-shared" }, { "matteState", "available" },
                { "matteAssetVersion", "content-v1" },
                { "matteContentReceipt", std::string(64, 'a') },
                { "matteFramePrefix", "mask-" }, { "matteFrameExtension", ".webp" },
                { "matteFirstFrame", 17 }, { "matteFrameDigits", 4 },
                { "matteCacheKey", std::string(64, 'a') }, { "matteFps", 30.0 }, { "matteFrames", 90 }
            },
            {
                { "sourceKind", "media" }, { "sourcePath", "/media/b.mov" },
                { "clipId", 20 }, { "trackLayer", 0 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },
                { "displayStartSec", 2.0 },
                { "matteAssetId", "matte-shared" }, { "matteState", "available" },
                { "matteAssetVersion", "content-v1" },
                { "matteContentReceipt", std::string(64, 'a') },
                { "matteFramePrefix", "mask-" }, { "matteFrameExtension", ".webp" },
                { "matteFirstFrame", 17 }, { "matteFrameDigits", 4 },
                { "matteCacheKey", std::string(64, 'a') }, { "matteFps", 30.0 },
                { "matteFrames", 90 },
                { "transition", {
                    { "type", 1 }, { "durationSec", 1.0 },
                    { "fromClipId", 10 }, { "toClipId", 20 }
                } }
            },
            {
                { "sourceKind", "adjustment" },
                { "sourcePath", "gen://adjustment" },
                { "clipId", 30 }, { "trackLayer", 1 },
                { "inSec", 0.0 }, { "outSec", 3.0 }, { "rate", 1.0 },
                { "displayStartSec", 0.0 }
            }
        } }
    };
}
}

int main()
{
    std::string error;
    videowire::ResolvedVisualSnapshot preview;
    videowire::ResolvedVisualSnapshot exportSnapshot;
    videowire::ResolvedVisualSnapshot rejected;
    const auto wire = representativeSnapshot();
    check (videowire::parseSnapshotJson (wire, noShader, preview, error),
           "preview snapshot normalizes");
    check (videowire::parseSnapshotJson (wire, noShader, exportSnapshot, error),
           "export snapshot normalizes");
    check (preview.authoringRevision == exportSnapshot.authoringRevision
           && preview.segments.size() == exportSnapshot.segments.size(),
           "preview and export receive equivalent snapshot identity");
    const auto previewB = std::find_if (preview.segments.begin(), preview.segments.end(),
                                        [] (const auto& segment) { return segment.clipId == 20; });
    const auto exportB = std::find_if (exportSnapshot.segments.begin(), exportSnapshot.segments.end(),
                                       [] (const auto& segment) { return segment.clipId == 20; });
    const auto adjustment = std::find_if (preview.segments.begin(), preview.segments.end(),
                                           [] (const auto& segment) { return segment.clipId == 30; });
    check (previewB != preview.segments.end() && exportB != exportSnapshot.segments.end()
           && previewB->transitionFromClipId == exportB->transitionFromClipId
           && previewB->matteCacheKey == exportB->matteCacheKey
           && adjustment != preview.segments.end()
           && adjustment->sourceKind == videowire::SourceKind::Adjustment,
           "owner inputs normalize identically");
    const auto previewA = std::find_if (preview.segments.begin(), preview.segments.end(),
                                        [] (const auto& segment) { return segment.clipId == 10; });
    check (previewA != preview.segments.end() && previewB != preview.segments.end()
           && previewA->matteAssetId == "matte-shared"
           && previewA->matteAssetVersion == "content-v1"
           && previewA->matteContentReceipt == std::string(64, 'a')
           && previewA->matteFramePrefix == "mask-" && previewA->matteFrameExtension == ".webp"
           && previewA->matteFirstFrame == 17 && previewA->matteFrameDigits == 4
           && previewA->matteAssetId == previewB->matteAssetId,
           "full matte identity and naming bind identically to two clips");
    auto rejectedDepthOperation = wire;
    rejectedDepthOperation["visualLayerPlans"] = json::array({ {
        { "clipId", 10 }, { "operations", json::array({ {
            { "nodeId", 61 }, { "kind", "visual.depth.asset" },
            { "backendCapability", "source-decode" },
            { "payloadXml", "<DepthAssetBinding sequencePath=\"/untrusted\"/>" }
        } }) }
    } });
    check (! videowire::parseSnapshotJson(rejectedDepthOperation, noShader, rejected, error)
           && error == "depth payload does not match the executable receipt schema",
           "helper rejects malformed executable depth capability and payload");

    const std::string validDepth = "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>";
    for (const char* extra : { "path", "file", "uri", "decoder", "upload", "futureField" })
    {
        auto injected = rejectedDepthOperation;
        injected["visualLayerPlans"][0]["operations"][0]["backendCapability"] = "parked-metadata";
        auto payload = validDepth;
        payload.insert(payload.size() - 2, std::string(" ") + extra + "=\"x\"");
        injected["visualLayerPlans"][0]["operations"][0]["payloadXml"] = payload;
        check (! videowire::parseSnapshotJson(injected, noShader, rejected, error),
               "snapshot admission rejects unknown depth payload attributes");
    }

    auto invalidTransition = wire;
    invalidTransition["segments"][1]["transition"]["fromClipId"] = 999;
    check (! videowire::parseSnapshotJson (invalidTransition, noShader, rejected, error),
           "explicit missing transition owner fails without legacy inference");

    auto invalidKind = wire;
    invalidKind["segments"][0]["sourceKind"] = "future-kind";
    check (! videowire::parseSnapshotJson (invalidKind, noShader, rejected, error),
           "explicit unknown source kind fails without path fallback");

    auto invalidMatte = wire;
    invalidMatte["segments"][1].erase ("matteFrames");
    check (! videowire::parseSnapshotJson (invalidMatte, noShader, rejected, error),
           "incomplete explicit matte binding fails");
    auto conflictingMatte = wire;
    conflictingMatte["segments"][1]["matteCacheKey"] = std::string(64, 'b');
    check (! videowire::parseSnapshotJson(conflictingMatte, noShader, rejected, error)
           && error == "duplicate matte asset authority",
           "duplicate matte IDs cannot add conflicting path authority");
    auto conflictingVersion = wire;
    conflictingVersion["segments"][1]["matteAssetVersion"] = "content-v2";
    check (! videowire::parseSnapshotJson(conflictingVersion, noShader, rejected, error)
           && error == "duplicate matte asset authority",
           "duplicate matte IDs cannot add conflicting version authority");
    auto invalidNaming = wire;
    invalidNaming["segments"][0]["matteFramePrefix"] = "../escape";
    check (! videowire::parseSnapshotJson(invalidNaming, noShader, rejected, error),
           "unsafe matte naming is rejected");
    auto nonFiniteMatte = wire;
    nonFiniteMatte["segments"][0]["matteFps"] = std::numeric_limits<double>::infinity();
    check (! videowire::parseSnapshotJson(nonFiniteMatte, noShader, rejected, error),
           "non-finite matte fps is rejected");
    auto duplicateClip = wire;
    duplicateClip["segments"][1]["clipId"] = 10;
    check (! videowire::parseSnapshotJson(duplicateClip, noShader, rejected, error)
           && error == "snapshot duplicates a stable clip owner",
           "duplicate stable clip owners are rejected");
    auto traversalMatte = wire;
    traversalMatte["segments"][0]["matteCacheKey"] = "../outside";
    check (! videowire::parseSnapshotJson(traversalMatte, noShader, rejected, error)
           && error == "malformed or unauthorised matte binding",
           "matte path additions reject traversal");
    auto staleMatte = wire;
    staleMatte["segments"][0]["matteState"] = "stale";
    staleMatte["segments"][1]["matteState"] = "stale";
    check (videowire::parseSnapshotJson (staleMatte, noShader, rejected, error)
           && ! videowire::validateSnapshotResources (rejected,
               [] (const std::string&) { return true; }, error)
           && error == "matte asset is stale: matte-shared",
           "stale asset is preserved and rejected without regeneration");

    json legacy = {
        { "segments", {{
            { "sourcePath", "gen://adjustment" }, { "clipId", 7 },
            { "inSec", 0.0 }, { "outSec", 1.0 }, { "rate", 1.0 }
        }} }
    };
    check (videowire::parseSnapshotJson (legacy, noShader, rejected, error)
           && rejected.segments[0].sourceKind == videowire::SourceKind::Adjustment,
           "legacy sentinel normalizes through absence-only compatibility path");

    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path)
               { return path == "/media/a.mov" || path == "/media/b.mov"
                     || path == "/mattes/b"; },
               error),
           "matte authority remains non-executable without a trusted cache root");
    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path) { return path != "/media/a.mov"; },
               error)
           && error == "media source is missing: /media/a.mov",
           "resource that became stale fails before matte execution admission");
    check (videowire::validateSnapshotResources (
               rejected,
               [] (const std::string&) { return false; },
               error),
           "absence-only generator compatibility does not require media resources");
    check (! videowire::validateSnapshotResources (
               preview,
               [] (const std::string& path)
               { return path == "/media/a.mov" || path == "/media/b.mov"
                     || path == "/mattes/b"; },
               error),
           "matte binding cannot bypass fail-closed execution admission");

    std::vector<videowire::RawRenderSegment> raw (1);
    raw[0].clipId = 44;
    raw[0].sourcePath = "/media/unchanged.wav";
    raw[0].inSec = 0.0;
    raw[0].outSec = 1.0;
    const auto originalPath = raw[0].sourcePath;
    videowire::CompiledVisualLayerPlan compiledPlan;
    compiledPlan.clipId = 44;
    compiledPlan.producerValidated = true;
    compiledPlan.nodeKinds = {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { compiledPlan }, 1, true, rejected, error)
           && rejected.visualLayerPlans.size() == 1
           && rejected.segments.size() == 1,
           "fixed visual layer plan lowers onto the existing render segment");
    auto typedPlan = compiledPlan;
    typedPlan.nodeKinds = { "video.source", "video.out" };
    typedPlan.nodeIds = { 101, 102 };
    typedPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    typedPlan.edges = { { 101, 0, 102, 0 } };
    typedPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, { typedPlan }, 2, true, rejected, error),
           "typed node, port, and edge bindings are admitted");
    auto particlePlan = typedPlan;
    particlePlan.nodeKinds = { "visual.particles", "video.out" };
    particlePlan.nodeIds = { 201, 202 };
    particlePlan.ports = {
        { 201, 0, 1, "in", "event", "event", "unspecified", "unspecified" },
        { 201, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 202, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    particlePlan.edges = { { 201, 1, 202, 0 } };
    particlePlan.operations = {
        { 201, "visual.particles", "native-gpu", "" },
        { 202, "video.out", "native-gpu", "" }
    };
    videowire::VisualEventScheduleBinding particleEvents;
    particleEvents.clipId = 44;
    particleEvents.nodeId = 201;
    particleEvents.portId = 0;
    particleEvents.sessionRevision = 9;
    particleEvents.triggers = { { 0, 4.0, 0.5f, 1 }, { 32, 4.01, 1.0f, 2 } };
    check (videowire::normalizeSnapshot (raw, { particlePlan }, { particleEvents },
                                         2, true, rejected, error)
           && rejected.visualEventSchedules.size() == 1,
           "particle Event schedules retain exact stable sink identity and order");
    auto connectedWithoutTriggers = particleEvents;
    connectedWithoutTriggers.triggers.clear();
    check (videowire::normalizeSnapshot (raw, { particlePlan }, { connectedWithoutTriggers },
                                         2, true, rejected, error)
           && rejected.visualEventSchedules.size() == 1
           && rejected.visualEventSchedules.front().triggers.empty(),
           "an active particle Event sink remains authoritative before its first trigger");
    particleEvents.portId = 1;
    check (! videowire::normalizeSnapshot (raw, { particlePlan }, { particleEvents },
                                           2, true, rejected, error)
           && error == "visual Event schedule has invalid stable sink identity",
           "particle Event schedules reject a Frame port masquerading as an Event sink");
    particleEvents.portId = 0;
    particleEvents.triggers[1].timelineBeat = 3.0;
    check (! videowire::normalizeSnapshot (raw, { particlePlan }, { particleEvents },
                                           2, true, rejected, error)
           && error == "visual Event schedule trigger ordering or bounds are invalid",
           "particle Event schedules reject non-deterministic timeline order");
    auto pointsPlan = typedPlan;
    pointsPlan.nodeKinds = { "video.source", "visual.points.grid",
                             "visual.points.set-color", "visual.clone-to-points", "video.out" };
    pointsPlan.nodeIds = { 101, 103, 104, 105, 102 };
    pointsPlan.ports.insert(pointsPlan.ports.begin() + 1, {
        { 103, 0, 1, "out", "control", "points", "unspecified", "unspecified" },
        { 104, 0, 1, "in", "control", "points", "unspecified", "unspecified" },
        { 104, 1, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 104, 2, 1, "out", "control", "points", "unspecified", "unspecified" },
        { 105, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 1, 1, "in", "control", "points", "unspecified", "unspecified" },
        { 105, 2, 1, "out", "control", "shape", "unspecified", "unspecified" }
    });
    pointsPlan.edges.push_back({ 103, 0, 104, 0 });
    pointsPlan.edges.push_back({ 104, 2, 105, 1 });
    pointsPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.points.grid", "control-eval", "<NodeParams columns=\"4\" rows=\"3\"/>" },
        { 104, "visual.points.set-color", "control-eval", "<NodeParams red=\"1\"/>" },
        { 105, "visual.clone-to-points", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    const bool pointsAdmitted = videowire::normalizeSnapshot(
        raw, { pointsPlan }, 2, true, rejected, error);
    if (! pointsAdmitted) std::fprintf(stderr, "points admission: %s\n", error.c_str());
    check (pointsAdmitted,
           "clone-to-points admits only as typed control geometry");
    pointsPlan.operations[0].backendCapability = "control-eval";
    check (! videowire::normalizeSnapshot (raw, { pointsPlan }, 2, true, rejected, error),
           "frame-producing operations cannot claim control evaluation");
    typedPlan.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::normalizeSnapshot (raw, { typedPlan }, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "unadmitted operation backend rejects before rendering");
    typedPlan.operations[1].backendCapability = "native-gpu";
    auto proceduralPlan = typedPlan;
    proceduralPlan.nodeKinds = {
        "video.source", "visual.noise", "visual.field.scalar", "visual.field.vector",
        "visual.field.color", "video.out"
    };
    proceduralPlan.nodeIds = { 101, 103, 104, 105, 106, 102 };
    proceduralPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 1, 1, "out", "control", "scalar", "unspecified", "unspecified" },
        { 104, 0, 1, "in", "control", "scalar", "unspecified", "unspecified" },
        { 104, 1, 1, "out", "control", "scalarField", "unspecified", "unspecified" },
        { 105, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 105, 1, 2, "out", "control", "vectorField", "unspecified", "unspecified" },
        { 106, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 106, 1, 4, "out", "control", "colorField", "unspecified", "linearSRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    proceduralPlan.edges = { { 101, 0, 102, 0 }, { 103, 1, 104, 0 } };
    proceduralPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.noise", "control-eval", "<NodeParams seed='17' frequency='2'/>" },
        { 104, "visual.field.scalar", "control-eval", "" },
        { 105, "visual.field.vector", "control-eval", "" },
        { 106, "visual.field.color", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, { proceduralPlan }, 2, true, rejected, error),
           "procedural noise and scalar/vector/color fields admit as control evaluation");
    proceduralPlan.operations[2].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, { proceduralPlan }, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "field operation cannot claim native renderer execution");

    auto valueMathPlan = typedPlan;
    valueMathPlan.nodeKinds = {
        "video.source", "visual.vec2.add", "visual.vec2.multiply", "visual.vec2.remap",
        "visual.color.mix", "visual.color.remap", "video.out"
    };
    valueMathPlan.nodeIds = { 101, 103, 104, 105, 106, 107, 102 };
    valueMathPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 1, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 103, 2, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 104, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 104, 1, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 104, 2, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 105, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 105, 5, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 106, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 106, 3, 4, "out", "control", "color", "unspecified", "linearSRGB" },
        { 107, 0, 4, "in", "control", "color", "unspecified", "linearSRGB" },
        { 107, 5, 4, "out", "control", "color", "unspecified", "linearSRGB" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    valueMathPlan.edges = { { 101, 0, 102, 0 }, { 103, 2, 104, 0 }, { 104, 2, 105, 0 } };
    valueMathPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.vec2.add", "control-eval", "" },
        { 104, "visual.vec2.multiply", "control-eval", "" },
        { 105, "visual.vec2.remap", "control-eval", "" },
        { 106, "visual.color.mix", "control-eval", "" },
        { 107, "visual.color.remap", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, { valueMathPlan }, 2, true, rejected, error),
           "typed vector and color math admit only as deterministic control evaluation");
    valueMathPlan.operations[4].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, { valueMathPlan }, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "color math cannot claim native image execution");

    auto shapeUvPlan = typedPlan;
    shapeUvPlan.nodeKinds = {
        "video.source", "visual.shape.rectangle", "visual.shape.ellipse",
        "visual.shape.union", "visual.uv.transform", "visual.uv.remap", "video.out"
    };
    shapeUvPlan.nodeIds = { 101, 103, 104, 105, 106, 107, 102 };
    shapeUvPlan.ports = {
        { 101, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 103, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 104, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 105, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 1, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 105, 2, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 106, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 106, 4, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 107, 0, 2, "in", "control", "vec2", "unspecified", "unspecified" },
        { 107, 5, 2, "out", "control", "vec2", "unspecified", "unspecified" },
        { 102, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    shapeUvPlan.edges = { { 101, 0, 102, 0 }, { 103, 0, 105, 0 }, { 104, 0, 105, 1 },
                          { 106, 4, 107, 0 } };
    shapeUvPlan.operations = {
        { 101, "video.source", "source-decode", "" },
        { 103, "visual.shape.rectangle", "control-eval", "<NodeParams width='1'/>" },
        { 104, "visual.shape.ellipse", "control-eval", "<NodeParams width='0.5'/>" },
        { 105, "visual.shape.union", "control-eval", "" },
        { 106, "visual.uv.transform", "control-eval", "<NodeParams pivotX='0.5'/>" },
        { 107, "visual.uv.remap", "control-eval", "" },
        { 102, "video.out", "native-gpu", "" }
    };
    check (videowire::normalizeSnapshot (raw, { shapeUvPlan }, 2, true, rejected, error),
           "shape geometry and UV operations admit as typed control evaluation");
    shapeUvPlan.operations[3].backendCapability = "native-gpu";
    check (! videowire::normalizeSnapshot (raw, { shapeUvPlan }, 2, true, rejected, error)
           && error == "visual layer plan operation identity or backend admission is invalid",
           "shape geometry cannot claim native image execution");

    typedPlan.nodeKinds[1] = "video.future";
    typedPlan.operations[1].kind = "video.future";
    check (! videowire::normalizeSnapshot (raw, { typedPlan }, 2, true, rejected, error)
           && error == "visual layer plan contains an unsupported typed operation",
           "unsupported typed operation rejects without a fixed-chain comparison");
    typedPlan.nodeKinds[1] = "video.out";
    typedPlan.operations[1].kind = "video.out";
    typedPlan.ports[1].dataType = "mask";
    check (! videowire::normalizeSnapshot (raw, { typedPlan }, 2, true, rejected, error)
           && error == "visual layer plan has an incompatible typed edge binding",
           "incompatible typed edge bindings reject before rendering");
    auto editablePlan = compiledPlan;
    editablePlan.nodeKinds = { "video.source", "video.transform", "video.out" };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 2, true, rejected, error)
           && rejected.visualLayerPlans.size() == 1
           && rejected.visualLayerPlans[0].nodeKinds == editablePlan.nodeKinds,
           "editable transform plan lowers onto existing production operations");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 3, true, rejected, error),
           "editable effects plan lowers onto the production effect rack");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 4, true, rejected, error),
           "editable shape-mask plan lowers onto production mask operations");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 5, true, rejected, error),
           "editable blend plan lowers onto production layer compositing");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 6, true, rejected, error),
           "editable text plan lowers onto production text compositing");
    editablePlan.nodeKinds = {
        "video.source", "video.transform", "video.effects", "video.mask.shape",
        "video.text", "video.layer.source", "video.blend", "video.out"
    };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 7, true, rejected, error),
           "editable second-source plan lowers onto production layer compositing");
    editablePlan.nodeKinds = { "video.source", "video.out" };
    check (videowire::normalizeSnapshot (raw, { editablePlan }, 8, true, rejected, error),
           "delete-healed direct plan lowers onto existing production operations");
    compiledPlan.producerValidated = false;
    compiledPlan.error = "malformed fixed topology";
    check (! videowire::normalizeSnapshot (raw, { compiledPlan }, 2, true, rejected, error)
           && error == "malformed fixed topology",
           "malformed visual layer plan rejects the current structural candidate");
    videowire::RevisionLedger planRevisions;
    check (planRevisions.requestAuthoring (1) && planRevisions.accept (1)
           && planRevisions.compileSucceeded (1, true)
           && planRevisions.requestAuthoring (2) && planRevisions.reject (2)
           && planRevisions.state().rejected == 2
           && planRevisions.state().lastGood == 1
           && planRevisions.state().exportable == 0,
           "fixed-topology rejection preserves last-good preview and clears exportability");
    check (videowire::normalizeSnapshot (raw, 1, rejected, error)
           && raw[0].sourcePath == originalPath
           && raw[0].sourceKind == videowire::SourceKind::Unspecified,
           "snapshot construction does not mutate canonical input");

    videowire::RevisionLedger revisions;
    check (revisions.requestAuthoring (1) && revisions.accept (1)
           && revisions.compileSucceeded (1, true),
           "accepted authoring becomes compiled last-good and exportable");
    const auto structuralBeforeRuntime = revisions.state();
    check (revisions.evaluate (1, 100) && revisions.evaluate (1, 101),
           "runtime evaluations advance monotonically");
    check (revisions.state().authoring == structuralBeforeRuntime.authoring
           && revisions.state().compiled == structuralBeforeRuntime.compiled
           && revisions.state().evaluation == 101,
           "runtime evaluation does not bump structural revisions");

    check (revisions.requestAuthoring (2), "new structural authoring advances");
    check (revisions.state().compiled == 1 && revisions.state().lastGood == 1
           && revisions.state().exportable == 0,
           "uncompiled authoring keeps last-good preview but clears exportability");
    check (! revisions.accept (1) && ! revisions.compileSucceeded (1, true),
           "stale acknowledgement cannot become current");
    check (revisions.reject (2)
           && revisions.state().rejected == 2
           && revisions.state().compiled == 1
           && revisions.state().lastGood == 1
           && revisions.state().exportable == 0,
           "rejection preserves last-good and refuses current export");

    videowire::RevisionLedger replay;
    replay.requestAuthoring (1);
    replay.accept (1);
    replay.compileSucceeded (1, true);
    check (replay.state().accepted == 1 && replay.state().compiled == 1
           && replay.state().lastGood == 1 && replay.state().exportable == 1,
           "restart replay recreates the accepted revision tuple");

    videowire::RevisionLedger race;
    check (race.requestAuthoring (10) && race.requestAuthoring (11)
           && race.requestAuthoring (12),
           "compile race publishes revisions 10, 11, and 12");
    check (race.accept (11) && race.compileSucceeded (11, true)
           && race.state().compiled == 11 && race.state().lastGood == 11
           && race.state().exportable == 0,
           "newest completed stale revision becomes last-good but not exportable");
    check (! race.accept (10) && ! race.compileSucceeded (10, true)
           && race.state().compiled == 11,
           "older completion cannot roll compiled state backward");
    check (race.accept (12) && race.compileSucceeded (12, true)
           && race.state().accepted == 12 && race.state().compiled == 12
           && race.state().lastGood == 12 && race.state().exportable == 12,
           "current completion becomes compiled and exportable");

    videowire::RevisionLedger rejectedRace;
    check (rejectedRace.requestAuthoring (10) && rejectedRace.requestAuthoring (11)
           && rejectedRace.requestAuthoring (12),
           "rejected race publishes revisions 10, 11, and 12");
    check (rejectedRace.accept (11) && rejectedRace.compileSucceeded (11, true)
           && ! rejectedRace.accept (10) && ! rejectedRace.compileSucceeded (10, true),
           "rejected race retains the newest usable out-of-order completion");
    check (rejectedRace.reject (12)
           && rejectedRace.state().authoring == 12
           && rejectedRace.state().accepted == 11
           && rejectedRace.state().rejected == 12
           && rejectedRace.state().compiled == 11
           && rejectedRace.state().lastGood == 11
           && rejectedRace.state().exportable == 0,
           "current rejection preserves revision 11 only as last-good");

    // Exact lazy-open boundary: custom naming is retained, and post-admission
    // frame/manifest mutation is rejected by the same validator used by viewport/export.
    const auto cacheRoot = std::filesystem::canonical(std::filesystem::temp_directory_path())
        / ("arbit-matte-cache-test-" + std::to_string(reinterpret_cast<uintptr_t>(&checks)));
    videohelper::MatteCacheBinding cacheBinding;
    cacheBinding.version = "content-v1";
    cacheBinding.prefix = "subject-";
    cacheBinding.extension = ".rgba";
    cacheBinding.firstFrame = 42;
    cacheBinding.digits = 3;
    cacheBinding.frames = 1;
    cacheBinding.fps = 24.0;
    cacheBinding.contentRevision = 99;
    const std::string frameBytes = "immutable-frame-bytes";
    const std::vector<std::string> frameHashes { videohelper::sha256Text(frameBytes) };
    const auto manifest = videohelper::bindingMaterial(cacheBinding, frameHashes);
    cacheBinding.key = cacheBinding.receipt = videohelper::sha256Text(manifest);
    const auto cacheDir = cacheRoot / cacheBinding.key;
    std::filesystem::create_directories(cacheDir);
    const auto framePath = cacheDir / videohelper::matteFrameName(cacheBinding, 0);
    { std::ofstream frame(framePath, std::ios::binary); frame << frameBytes; }
    { std::ofstream manifestFile(cacheDir / "manifest", std::ios::binary); manifestFile << manifest; }
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    videowire::RenderSegment lazySegment;
    lazySegment.matteAssetId = "matte-custom";
    lazySegment.matteAssetVersion = cacheBinding.version;
    lazySegment.matteContentReceipt = cacheBinding.receipt;
    lazySegment.matteCacheKey = cacheBinding.key;
    lazySegment.matteFramePrefix = cacheBinding.prefix;
    lazySegment.matteFrameExtension = cacheBinding.extension;
    lazySegment.matteFirstFrame = cacheBinding.firstFrame;
    lazySegment.matteFrameDigits = cacheBinding.digits;
    lazySegment.matteFrames = cacheBinding.frames;
    lazySegment.matteFps = cacheBinding.fps;
    lazySegment.matteContentRevision = cacheBinding.contentRevision;
    lazySegment.matteTrustedRoot = cacheRoot.string();
    std::string resolvedDir, customPattern;
    check (videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error)
           && customPattern == cacheDir.string() + "/subject-%03d.rgba",
           "viewport/export lazy-open validator derives the complete custom naming tuple");
    const auto originalSource = cacheRoot / "project-controlled-original.mov";
    { std::ofstream source(originalSource, std::ios::binary); source << "original-source-v1"; }
    { std::ofstream source(originalSource, std::ios::binary | std::ios::trunc); source << "mutated-source-v2"; }
    check (videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "original-source mutation cannot alter receipt-addressed cached decode authority");
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    { std::ofstream frame(framePath, std::ios::binary | std::ios::trunc); frame << "tampered"; }
    check (! videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "frame mutation after admission is rejected immediately before lazy open");
    { std::ofstream frame(framePath, std::ios::binary | std::ios::trunc); frame << frameBytes; }
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    { std::ofstream manifestFile(cacheDir / "manifest", std::ios::binary | std::ios::app); manifestFile << "tamper"; }
    check (! videowire::revalidateMatteForOpen(lazySegment, resolvedDir, customPattern, error),
           "manifest mutation after admission is rejected immediately before lazy open");
    std::filesystem::permissions(cacheDir / "manifest", std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::permissions(framePath, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::permissions(cacheDir, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::filesystem::remove_all(cacheRoot);

    std::printf ("render-snapshot: %d/%d checks passed\n", checks - failures, checks);
    return failures;
}
