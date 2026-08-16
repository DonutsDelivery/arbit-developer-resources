#include "../src/visual_plan_executor.h"
#include "../src/visual_plan_telemetry_json.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <string>

namespace
{
std::atomic<uint64_t> allocations { 0 };
}

void* operator new (std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete (void* value) noexcept { std::free(value); }
void operator delete (void* value, std::size_t) noexcept { std::free(value); }

namespace
{
int failures = 0;
void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

struct FakeLayer
{
    struct Clock { double time = 0.0, timeDelta = 0.0, beat = 0.0; int frame = 0; bool playing = false; };
    struct Effect { bool enabled = false; int type = -1; float params[9] {}; };
    Clock shaderClock;
    std::map<std::string, double> genParams;
    unsigned texture = 0;
    int texWidth = 0, texHeight = 0;
    bool shaderSource = false;
    bool particleSource = false;
    bool particleStateReset = false;
    bool particleTriggerConnected = false;
    int particleTriggerCount = 0;
    float particleTriggerStrength = 0.0f;
    uint64_t visualPlanStructuralRevision = 0;
    bool visualPlanTelemetryHold = false;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    bool drawShape = false;
    bool inspectionDrawShapeOutput = false;
    bool drawShapeEllipse = false;
    float drawShapeCx = 0.5f, drawShapeCy = 0.5f, drawShapeW = 1.0f, drawShapeH = 1.0f;
    float drawShapeR = 1.0f, drawShapeG = 1.0f, drawShapeB = 1.0f, drawShapeA = 1.0f;
    float scale = 2.0f, translateX = 1.0f, translateY = 1.0f, rotationDeg = 4.0f;
    float cropLeft = 0.1f, cropRight = 0.1f, cropTop = 0.1f, cropBottom = 0.1f;
    const Effect* effects = reinterpret_cast<const Effect*> (1);
    int effectCount = 2;
    Effect graphFeedbackEffect;
    bool feedbackHistoryReset = false;
    bool feedbackHistoryHold = false;
    unsigned lutTexture = 4;
    int lutSize = 17;
    int maskType = 2;
    bool maskInvert = true;
    unsigned matteTexture = 0;
    unsigned matteTextureB = 0;
    int matteWidth = 0, matteHeight = 0;
    int matteWidthB = 0, matteHeightB = 0;
    bool matteApply = false, matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    unsigned depthTexture = 0;
    int depthWidth = 0, depthHeight = 0;
    bool depthFog = false;
    int depthEffect = 0;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float opacity = 0.4f;
    int blendMode = 3;
};

videowire::CompiledVisualLayerPlan directPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
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

videowire::CompiledVisualLayerPlan mattePlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.refine",
                       "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 42, 43, 12 };
    plan.edges = { { 11, 0, 43, 0 }, { 41, 0, 42, 0 },
                   { 42, 1, 43, 1 }, { 43, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 41, "visual.matte.asset", "source-decode",
          "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"available\" cacheKey=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" framePrefix=\"subject-\" frameExtension=\".rgba\" firstFrame=\"42\" frameDigits=\"3\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>" },
        { 42, "visual.matte.refine", "native-gpu",
          "<NodeParams invert=\"1\" black=\"-2\" white=\"3\" erodeDilate=\"20\" feather=\"20\" choke=\"-2\"/>" },
        { 43, "visual.matte.apply", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan combinedMattePlan()
{
    auto plan = mattePlan();
    plan.nodeKinds = { "video.source", "visual.matte.asset", "visual.matte.asset",
                       "visual.matte.combine", "visual.matte.refine", "visual.matte.apply", "video.out" };
    plan.nodeIds = { 11, 41, 44, 45, 42, 43, 12 };
    plan.edges = { {11,0,43,0}, {41,0,45,0}, {44,0,45,1}, {45,2,42,0},
                   {42,1,43,1}, {43,2,12,0} };
    auto bindingB = plan.operations[1];
    bindingB.nodeId = 44;
    bindingB.payloadXml = "<MatteAssetBinding matteAssetId=\"matte-2\" matteAssetVersion=\"v2\" state=\"available\" cacheKey=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" contentReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"other-\" frameExtension=\".rgba\" firstFrame=\"7\" frameDigits=\"3\" fps=\"24\" frames=\"8\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    plan.operations = { plan.operations[0], plan.operations[1], bindingB,
        {45, "visual.matte.combine", "native-gpu", "<NodeParams mode=\"3\"/>"},
        plan.operations[2], plan.operations[3], plan.operations[4] };
    return plan;
}

videowire::CompiledVisualLayerPlan depthPlan()
{
    auto plan = directPlan();
    plan.nodeKinds = { "video.source", "visual.depth.asset", "visual.depth.fog", "video.out" };
    plan.nodeIds = { 11, 61, 62, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 61, 0, 0, "out", "none", "unspecified", "unspecified", "unspecified" },
        { 61, 1, 1, "out", "frame", "depth", "r16", "linearSRGB" },
        { 62, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 62, 1, 1, "in", "frame", "depth", "r16", "linearSRGB" },
        { 62, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 62, 0 }, { 61, 1, 62, 1 }, { 62, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 61, "visual.depth.asset", "source-decode",
          "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"depth-sequence-v1\" state=\"available\" cacheKey=\"depth-job-1-abc/published\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" analysisReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"depth_\" frameExtension=\".png\" firstFrame=\"1\" frameDigits=\"6\" width=\"2\" height=\"2\" fps=\"24\" frames=\"3\" format=\"r16-unorm\"/>" },
        { 62, "visual.depth.fog", "native-gpu",
          "<NodeParams near=\"-1\" far=\"2\" density=\"99\" red=\"0.2\" green=\"0.3\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}
}

int main()
{
    std::string error;
    videowire::VisualEventScheduleBinding eventSchedule;
    eventSchedule.clipId = 7;
    eventSchedule.nodeId = 70;
    eventSchedule.portId = 0;
    eventSchedule.sessionRevision = 3;
    eventSchedule.triggers = {
        { 0, 1.0, 0.25f, 1 }, { 32, 1.5, 0.75f, 2 }, { 64, 1.5, 0.5f, 3 }
    };
    const std::vector<videowire::VisualEventScheduleBinding> eventSchedules { eventSchedule };
    videowire::VisualEventTriggerCursor viewportCursor (false);
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.25).count == 0,
           "viewport cursor attaches without replaying historical triggers");
    const auto viewportAdvance = viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5);
    check (viewportAdvance.count == 2 && viewportAdvance.strongest == 0.75f
           && viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 0,
           "viewport cursor consumes equal-beat triggers once in deterministic order");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 0,
           "viewport backward seek resets without bursting retained history");
    check (viewportCursor.consume(eventSchedules, 7, 70, 0, 4, 2.0).count == 0,
           "viewport revision replacement resets independently");
    videowire::VisualEventTriggerCursor exportCursor (true);
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1
           && exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.5).count == 2,
           "export cursor deterministically replays schedules from timeline start");
    check (exportCursor.consume(eventSchedules, 7, 70, 1, 3, 2.0).count == 0
           && exportCursor.consume(eventSchedules, 8, 70, 0, 3, 2.0).count == 0,
           "cursor never retargets a stable sink by presentation identity");
    check (exportCursor.consume(eventSchedules, 7, 70, 0, 3, 1.0).count == 1,
           "export backward seek reconstructs its own trigger cursor");
    auto direct = directPlan();
    FakeLayer layer;

    auto particles = direct;
    particles.nodeKinds = { "visual.particles", "video.out" };
    particles.nodeIds = { 51, 12 };
    particles.edges = { { 51, 1, 12, 0 } };
    particles.operations = {
        { 51, "visual.particles", "native-gpu", "<NodeParams seed=\"70000\" count=\"9000\" lifetime=\"20\" size=\"0\" speed=\"9\" red=\"-1\" green=\"2\" blue=\"0.4\" alpha=\"2\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    FakeLayer particleLayer;
    for (const char* key : { "nativeBuiltin", "seed", "count", "lifetime", "size",
                             "speed", "red", "green", "blue", "alpha" })
        particleLayer.genParams.emplace(key, 0.0);
    particleLayer.shaderClock.time = 1.25;
    particleLayer.shaderClock.frame = 30;
    particleLayer.shaderClock.timeDelta = 1.0 / 24.0;
    particleLayer.shaderClock.playing = false;
    check (videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error,
                                             nullptr, nullptr, nullptr, 2.5)
           && particleLayer.particleSource && ! particleLayer.shaderSource
           && particleLayer.genParams["nativeBuiltin"] == 1.0
           && particleLayer.genParams["seed"] == 65535.0
           && particleLayer.genParams["count"] == 4096.0
           && particleLayer.genParams["lifetime"] == 10.0
           && particleLayer.genParams["size"] == 1.0
           && particleLayer.genParams["speed"] == 4.0
           && particleLayer.genParams["red"] == 0.0
           && particleLayer.genParams["green"] == 1.0
           && particleLayer.shaderClock.time == 1.25
           && particleLayer.shaderClock.frame == 30
           && particleLayer.shaderClock.timeDelta == 1.0 / 24.0
           && ! particleLayer.shaderClock.playing,
           "curated particle source clamps payload and preserves the canonical ShaderClock");
    videowire::VisualPlanExecutionState particleState;
    particleState.admitPlans({ particles });
    const std::vector<videowire::CompiledVisualLayerPlan> admittedParticles { particles };
    const auto particleMapSize = particleLayer.genParams.size();
    const auto particleAllocations = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,
                                             nullptr, nullptr, &particleState, 3.0)
           && particleLayer.genParams.size() == particleMapSize
           && allocations.load(std::memory_order_relaxed) == particleAllocations,
           "typed particle render updates only pre-existing parameters without allocation or insertion");
    auto particleSchedule = eventSchedule;
    particleSchedule.nodeId = 51;
    const std::vector<videowire::VisualEventScheduleBinding> particleSchedules { particleSchedule };
    videowire::VisualEventTriggerCursor particleCursor (true);
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,
                                             nullptr, nullptr, &particleState, 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected
           && particleLayer.particleTriggerCount == 3
           && particleLayer.particleTriggerStrength == 0.75f,
           "typed particle execution consumes the exact Event sink at the canonical beat");
    particleLayer.shaderClock.beat = 1.5;
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,
                                             nullptr, nullptr, &particleState, 3.0,
                                             &particleSchedules, &particleCursor)
           && particleLayer.particleTriggerConnected && particleLayer.particleTriggerCount == 0,
           "typed particle execution does not retrigger a held frame");
    particleSchedule.nodeId = 52;
    const std::vector<videowire::VisualEventScheduleBinding> staleSchedules { particleSchedule };
    check (videowire::executeVisualLayerPlan(admittedParticles, 7, particleLayer, error,
                                             nullptr, nullptr, &particleState, 3.0,
                                             &staleSchedules, &particleCursor)
           && ! particleLayer.particleTriggerConnected,
           "typed particle execution never retargets a stale visual NodeId");
    auto malformedParticles = particles;
    malformedParticles.edges[0].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedParticles }, 7, particleLayer, error)
           && error == "visual.particles has unsupported production topology",
           "particle source preserves exact image port and fails closed on malformed topology");
    particles.operations[0].backendCapability = "control-eval";
    check (! videowire::executeVisualLayerPlan({ particles }, 7, particleLayer, error),
           "particle image source rejects non-native backend classification");

    check (videowire::executeVisualLayerPlan ({ direct }, 7, layer, error),
           "ordered direct chain executes");
    check (layer.scale == 1.0f && layer.translateX == 0.0f
           && layer.effects == nullptr && layer.effectCount == 0
           && layer.lutTexture == 0 && layer.maskType == 0
           && layer.opacity == 0.4f && layer.blendMode == 3,
           "absent production operations are neutralized through LayerDesc");

    auto matte = mattePlan();
    FakeLayer matteLayer;
    matteLayer.matteTexture = 91;
    matteLayer.matteWidth = 640;
    matteLayer.matteHeight = 360;
    check (videowire::executeVisualLayerPlan({ matte }, 7, matteLayer, error)
           && matteLayer.matteApply && matteLayer.matteInvert
           && matteLayer.matteBlack == 0.0f && matteLayer.matteWhite == 1.0f
           && matteLayer.matteErodeDilate == 4.0f && matteLayer.matteFeather == 4.0f
           && matteLayer.matteChoke == -1.0f,
           "valid matte topology propagates bounded refinement parameters");
    auto combined = combinedMattePlan();
    matteLayer.matteTextureB = 92; matteLayer.matteWidthB = 640; matteLayer.matteHeightB = 360;
    check(videowire::executeVisualLayerPlan({combined}, 7, matteLayer, error)
          && matteLayer.matteCombineMode == 3,
          "two immutable matte receipts lower xor through retained GPU resources");
    const std::array<std::array<float, 4>, 4> oracles {{
        {{0.7f, 0.4f, 0.7f, 0.0f}}, {{0.7f, 0.4f, 0.4f, 1.0f}},
        {{0.7f, 0.4f, 0.3f, 2.0f}}, {{0.7f, 0.4f, 0.3f, 3.0f}} }};
    for (const auto& oracle : oracles)
    {
        const int mode = (int)oracle[3];
        const float actual = mode == 0 ? std::max(oracle[0], oracle[1])
                           : mode == 1 ? std::min(oracle[0], oracle[1])
                           : mode == 2 ? std::max(oracle[0] - oracle[1], 0.0f)
                                       : std::abs(oracle[0] - oracle[1]);
        check(std::abs(actual - oracle[2]) < 1.0e-6f, "tiny independent matte-combine pixel oracle");
    }
    FakeLayer missingSecond = matteLayer; missingSecond.matteTextureB = 0;
    check(!videowire::executeVisualLayerPlan({combined}, 7, missingSecond, error)
          && error == "typed matte combine GPU texture is unavailable",
          "combine rejects a missing retained second GPU texture");
    auto malformedMatte = matte;
    malformedMatte.edges[2].toPort = 0;
    check (! videowire::executeVisualLayerPlan({ malformedMatte }, 7, matteLayer, error)
           && error == "typed matte graph has unsupported production topology",
           "malformed matte topology fails closed");
    auto missingMatte = matte;
    missingMatte.operations[1].payloadXml =
        "<MatteAssetBinding state=\"available\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ missingMatte }, 7, matteLayer, error)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "missing matte binding fails closed");
    auto staleMatte = matte;
    staleMatte.operations[1].payloadXml =
        "<MatteAssetBinding matteAssetId=\"matte-1\" state=\"stale\" backend=\"rgba-cpu-decode-native-gpu-upload\"/>";
    check (! videowire::executeVisualLayerPlan({ staleMatte }, 7, matteLayer, error)
           && error == "typed matte asset binding is missing, stale, or unsupported",
           "stale matte binding fails closed");
    FakeLayer texturelessMatte;
    check (! videowire::executeVisualLayerPlan({ matte }, 7, texturelessMatte, error)
           && error == "typed matte GPU texture is unavailable",
           "unavailable matte texture fails closed");
    auto depth = depthPlan();
    videowire::ExecutableDepthPayload viewportPayload, exportPayload;
    std::string viewportDepthError, exportDepthError;
    check (videowire::visualPlanDepthBinding({ depth }, 7, viewportPayload, viewportDepthError)
           && videowire::visualPlanDepthBinding({ depth }, 7, exportPayload, exportDepthError)
           && viewportPayload.cacheKey == exportPayload.cacheKey
           && viewportPayload.contentReceipt == exportPayload.contentReceipt
           && viewportPayload.width == exportPayload.width
           && viewportPayload.height == exportPayload.height,
           "viewport/export shared depth preparation reads identical immutable descriptors");
    FakeLayer depthLayer;
    depthLayer.depthTexture = 9; depthLayer.depthWidth = 2; depthLayer.depthHeight = 2;
    check (videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error)
           && depthLayer.depthFog && depthLayer.fogNear == 0.0f && depthLayer.fogFar == 1.0f
           && depthLayer.fogDensity == 32.0f && depthLayer.fogAlpha == 1.0f,
           "depth fog exact topology executes with bounded parameters");
    const std::array<std::pair<const char*, const char*>, 3> depthKinds {{
        { "visual.depth.blur", "<NodeParams radius=\"99\" focus=\"0.25\" falloff=\"0\"/>" },
        { "visual.depth.displace", "<NodeParams amountX=\"1\" amountY=\"-1\" center=\"2\"/>" },
        { "visual.depth.relight", "<NodeParams intensity=\"3\" ambient=\"-1\" red=\"3\" green=\"0.5\" blue=\"1\"/>" } }};
    for (size_t i = 0; i < depthKinds.size(); ++i) {
        auto primitive = depthPlan();
        primitive.nodeKinds[2] = depthKinds[i].first;
        primitive.operations[2].kind = depthKinds[i].first;
        primitive.operations[2].payloadXml = depthKinds[i].second;
        FakeLayer primitiveLayer;
        primitiveLayer.depthTexture = 10 + static_cast<unsigned>(i);
        primitiveLayer.depthWidth = primitiveLayer.depthHeight = 2;
        check(videowire::executeVisualLayerPlan({ primitive }, 7, primitiveLayer, error)
              && primitiveLayer.depthEffect == static_cast<int>(i) + 2 && !primitiveLayer.depthFog,
              "depth primitive exact topology lowers to its stable native wire mode");
    }
    videowire::VisualInspectionTarget depthTarget { 7, 0, 61, 0 };
    check (! videowire::validateVisualInspectionTarget({ depth }, depthTarget, error)
           && error == "inspection target is not an image output port",
           "metadata-only depth node cannot be inspected as a materialized output");

    for (const char* extra : { "path", "file", "uri", "decoder", "upload", "futureField" })
    {
        auto injected = depth;
        injected.operations[1].payloadXml.insert(injected.operations[1].payloadXml.size() - 2,
                                                  std::string(" ") + extra + "=\"x\"");
        check (! videowire::executeVisualLayerPlan({ injected }, 7, depthLayer, error),
               "depth executor rejects every extra payload attribute");
    }
    for (const char* malformed : {
             "<Other depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetId=\"duplicate\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"/>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\"><child/></DepthAssetBinding>",
             "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"v1\" state=\"available\" contract=\"parked-metadata\" pendingExecutionSeam=\"depth-consuming-operation\">text</DepthAssetBinding>" })
    {
        auto rejectedPayload = depth;
        rejectedPayload.operations[1].payloadXml = malformed;
        check (! videowire::executeVisualLayerPlan({ rejectedPayload }, 7, depthLayer, error),
               "depth executor rejects malformed exact-schema payload");
    }
    depth.edges[1].fromPort = 0;
    check (! videowire::executeVisualLayerPlan({ depth }, 7, depthLayer, error)
           && error == "typed depth asset binding is missing, stale, or unsupported",
           "depth topology rejects any connection into composition");
    videowire::VisualInspectionTarget inspection { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ direct }, inspection, error),
           "exact executable image output is inspectable");
    inspection.structuralRevision = 1;
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target revision is stale",
           "stale inspection revisions fail closed");
    inspection = { 7, 0, 12, 0 };
    check (! videowire::validateVisualInspectionTarget({ direct }, inspection, error)
           && error == "inspection target is not an image output port",
           "input ports cannot be selected as intermediate outputs");

    auto sourceThenEffects = direct;
    sourceThenEffects.nodeKinds = { "video.source", "video.transform", "video.effects", "video.out" };
    sourceThenEffects.nodeIds = { 11, 14, 13, 12 };
    sourceThenEffects.operations = {
        { 11, "video.source", "source-decode", "" },
        { 14, "video.transform", "native-gpu", "" },
        { 13, "video.effects", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    sourceThenEffects.edges = { { 11, 0, 14, 0 }, { 14, 1, 13, 0 }, { 13, 1, 12, 0 } };
    sourceThenEffects.ports.push_back(
        { 14, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    inspection = { 7, 0, 11, 0 };
    check (videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error),
           "decoded source output is the admitted non-terminal inspection slice");
    FakeLayer sourceLayer;
    sourceLayer.texture = 77;
    sourceLayer.texWidth = 1920;
    sourceLayer.texHeight = 1080;
    videowire::VisualInspectionResource sourceResource;
    check (videowire::executeVisualLayerPlan({ sourceThenEffects }, 7, sourceLayer, error,
                                             &inspection, &sourceResource)
           && sourceResource.handle == 77 && sourceResource.width == 1920
           && sourceResource.height == 1080 && sourceResource.nodeId == 11,
           "executor retains the genuine decoded GPU texture without readback");
    inspection = { 7, 0, 14, 1 };
    check (! videowire::validateVisualInspectionTarget({ sourceThenEffects }, inspection, error)
           && error == "inspection target has no retainable GPU output resource",
           "transform followed by another operation is not misreported at the renderer boundary");
    auto sourceThenTransform = sourceThenEffects;
    sourceThenTransform.nodeKinds.erase(sourceThenTransform.nodeKinds.begin() + 2);
    sourceThenTransform.nodeIds.erase(sourceThenTransform.nodeIds.begin() + 2);
    sourceThenTransform.operations.erase(sourceThenTransform.operations.begin() + 2);
    sourceThenTransform.edges = { { 11, 0, 14, 0 }, { 14, 1, 12, 0 } };
    check (videowire::validateVisualInspectionTarget({ sourceThenTransform }, inspection, error)
           && videowire::classifyVisualInspectionTarget(sourceThenTransform, inspection)
                == videowire::VisualInspectionSlice::transformedLayer,
           "exact terminal transform output uses the genuine renderer resource seam");

    auto reversed = direct;
    std::reverse (reversed.operations.begin(), reversed.operations.end());
    check (! videowire::executeVisualLayerPlan ({ reversed }, 7, layer, error),
           "exact operation order must agree with exact edges");

    auto branch = direct;
    branch.nodeKinds.insert (branch.nodeKinds.begin() + 1, "video.effects");
    branch.nodeIds.insert (branch.nodeIds.begin() + 1, 13);
    branch.operations.insert (branch.operations.begin() + 1,
                              { 13, "video.effects", "native-gpu", "" });
    branch.edges = { { 11, 0, 13, 0 }, { 11, 0, 12, 0 } };
    videowire::VisualLayerExecution execution;
    check (! videowire::compileVisualLayerExecution (branch, execution, error),
           "branching image execution fails closed instead of flattening");

    auto composite = direct;
    composite.nodeKinds = { "video.source", "video.transform", "video.effects",
        "video.mask.shape", "video.text", "visual.gradient", "video.blend", "video.out" };
    composite.nodeIds = { 11, 21, 22, 23, 25, 26, 24, 12 };
    composite.operations.clear();
    for (size_t i = 0; i < composite.nodeIds.size(); ++i)
        composite.operations.push_back({ composite.nodeIds[i], composite.nodeKinds[i],
            i == 0 ? "source-decode" : "native-gpu", "" });
    composite.edges = {
        { 11, 0, 21, 0 }, { 21, 1, 22, 0 }, { 22, 1, 23, 0 },
        { 23, 1, 24, 0 }, { 25, 0, 24, 2 }, { 26, 0, 24, 3 }, { 24, 1, 12, 0 }
    };
    FakeLayer compositeLayer;
    check (videowire::executeVisualLayerPlan({ composite }, 7, compositeLayer, error)
           && compositeLayer.scale == 2.0f && compositeLayer.effects != nullptr
           && compositeLayer.maskType == 2,
           "established typed multi-source composite preserves production rendering");

    auto drawShape = direct;
    drawShape.nodeKinds = { "video.source", "visual.shape.rectangle",
                            "visual.draw.shape", "video.blend", "video.out" };
    drawShape.nodeIds = { 11, 31, 32, 24, 12 };
    drawShape.operations = {
        { 11, "video.source", "source-decode", "" },
        { 31, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 32, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 24, "video.blend", "native-gpu", "" },
        { 12, "video.out", "native-gpu", "" }
    };
    drawShape.edges = { { 11, 0, 24, 0 }, { 31, 0, 32, 0 },
                        { 32, 1, 24, 2 }, { 24, 1, 12, 0 } };
    drawShape.ports.push_back(
        { 32, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    FakeLayer shapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, shapeLayer, error)
           && shapeLayer.drawShape && shapeLayer.drawShapeCx == 0.25f
           && shapeLayer.drawShapeCy == 0.75f && shapeLayer.drawShapeW == 0.5f
           && shapeLayer.drawShapeH == 0.2f && shapeLayer.drawShapeR == 0.1f
           && shapeLayer.drawShapeG == 0.2f && shapeLayer.drawShapeB == 0.3f
           && shapeLayer.drawShapeA == 0.4f,
           "rectangle Draw Shape lowers canonical payload into the shared renderer input");
    inspection = { 7, 0, 32, 1 };
    check (videowire::validateVisualInspectionTarget({ drawShape }, inspection, error)
           && videowire::classifyVisualInspectionTarget(drawShape, inspection)
                == videowire::VisualInspectionSlice::retainedDrawShape,
           "Draw Shape output admits its exact native render-pass boundary");
    FakeLayer inspectedShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, inspectedShapeLayer, error,
                                             &inspection, nullptr)
           && inspectedShapeLayer.inspectionDrawShapeOutput,
           "executor marks only the exact selected Draw Shape output for retention");
    inspection.outputPort = 0;
    check (! videowire::validateVisualInspectionTarget({ drawShape }, inspection, error),
           "Draw Shape control input cannot alias the retained image output");
    inspection = { 7, 1, 32, 1 };
    FakeLayer staleShapeLayer;
    check (videowire::executeVisualLayerPlan({ drawShape }, 7, staleShapeLayer, error,
                                             &inspection, nullptr)
           && ! staleShapeLayer.inspectionDrawShapeOutput,
           "stale Draw Shape revision never arms backend retention");
    auto drawEllipse = drawShape;
    drawEllipse.nodeKinds[1] = "visual.shape.ellipse";
    drawEllipse.operations[1].kind = "visual.shape.ellipse";
    FakeLayer ellipseLayer;
    check (videowire::executeVisualLayerPlan({ drawEllipse }, 7, ellipseLayer, error)
           && ellipseLayer.drawShape && ellipseLayer.drawShapeEllipse,
           "ellipse Draw Shape lowers to the native shape renderer");
    drawShape.operations[2].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution(drawShape, execution, error),
           "Draw Shape rejects CPU fallback capability");

    auto cpu = direct;
    cpu.operations[1].backendCapability = "cpu-fallback";
    check (! videowire::compileVisualLayerExecution (cpu, execution, error)
           && error == "visual layer plan requires unsupported execution capability",
           "CPU image fallback is rejected");

    auto feedback = direct;
    feedback.structuralRevision = 4;
    feedback.nodeKinds.insert(feedback.nodeKinds.begin() + 1, "visual.feedback");
    feedback.nodeIds.insert(feedback.nodeIds.begin() + 1, 30);
    feedback.operations.insert(feedback.operations.begin() + 1,
        { 30, "visual.feedback", "native-gpu",
          "<NodeParams decay=\"2\" zoom=\"0.5\" swirl=\"-0.2\"/>" });
    feedback.ports.push_back({ 30, 0, 1, "in", "frame", "image", "rgba8", "sRGB" });
    feedback.ports.push_back({ 30, 1, 1, "out", "frame", "image", "rgba8", "sRGB" });
    feedback.edges = { { 11, 0, 30, 0 }, { 30, 1, 12, 0 } };
    videowire::VisualPlanExecutionState feedbackState;
    feedbackState.admitPlans({ feedback });
    FakeLayer feedbackLayer;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, feedbackLayer, error,
                                             nullptr, nullptr, &feedbackState, 1.0)
           && feedbackLayer.effects == &feedbackLayer.graphFeedbackEffect
           && feedbackLayer.effectCount == 1 && feedbackLayer.graphFeedbackEffect.type == 25
           && feedbackLayer.graphFeedbackEffect.params[0] == 0.999f
           && feedbackLayer.graphFeedbackEffect.params[1] == 0.9f
           && feedbackLayer.graphFeedbackEffect.params[2] == -0.1f
           && feedbackLayer.feedbackHistoryReset && ! feedbackLayer.feedbackHistoryHold,
           "graph feedback safely lowers clamped immutable payload into production FeedbackTrail");
    FakeLayer heldFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, heldFeedback, error,
                                             nullptr, nullptr, &feedbackState, 1.0)
           && heldFeedback.feedbackHistoryHold && ! heldFeedback.feedbackHistoryReset,
           "same-time feedback presentation holds production history");
    FakeLayer resetFeedback;
    check (videowire::executeVisualLayerPlan({ feedback }, 7, resetFeedback, error,
                                             nullptr, nullptr, &feedbackState, 0.5)
           && resetFeedback.feedbackHistoryReset && ! resetFeedback.feedbackHistoryHold,
           "backward feedback evaluation resets per-owner production history");

    videowire::VisualPlanExecutionState viewportState;
    videowire::VisualPlanExecutionState exportState;
    auto ownerPlan = directPlan(); ownerPlan.structuralRevision = 4;
    viewportState.admitPlans({ ownerPlan });
    check (viewportState.prepare(7, 4, 1.0)->evaluationSequence == 1,
           "temporal owner starts an explicit evaluation sequence");
    check (viewportState.prepare(7, 4, 1.0)->evaluationSequence == 1,
           "same-time presentation does not advance temporal state");
    check (viewportState.prepare(7, 4, 2.0)->evaluationSequence == 2,
           "forward evaluation advances exactly once");
    check (viewportState.prepare(7, 4, 0.5)->evaluationSequence == 1,
           "backward seek explicitly resets temporal state");
    ownerPlan.structuralRevision = 5;
    viewportState.admitPlans({ ownerPlan });
    exportState.admitPlans({ ownerPlan });
    check (viewportState.prepare(7, 5, 0.5)->evaluationSequence == 1,
           "structural revision replacement resets temporal ownership");
    check (exportState.prepare(7, 5, 0.5)->evaluationSequence == 1
           && viewportState.prepare(7, 5, 1.0)->evaluationSequence == 2
           && exportState.owner(7)->evaluationSequence == 1,
           "viewport and export owners share semantics without sharing state");
    viewportState.reset(7);
    check (viewportState.owner(7) != nullptr && viewportState.owner(7)->structuralRevision == 0,
           "explicit owner reset clears temporal state without removing the fixed slot");

    auto history = direct;
    history.nodeKinds.insert(history.nodeKinds.begin() + 1, "control.history");
    history.nodeIds.insert(history.nodeIds.begin() + 1, 31);
    history.operations.insert(history.operations.begin() + 1,
                              { 31, "control.history", "control-eval", "" });
    check (! videowire::compileVisualLayerExecution(history, execution, error)
           && error == "control.history temporal evaluation is not yet connected to native viewport/export image execution",
           "temporal operation retains a dedicated unsupported production diagnostic");

    auto full = direct;
    full.nodeKinds = { "video.source", "video.transform", "video.effects",
                       "video.mask.shape", "video.blend", "video.out" };
    full.nodeIds = { 11, 21, 22, 23, 24, 12 };
    full.operations.clear();
    full.edges.clear();
    full.ports.clear();
    for (size_t i = 0; i < full.nodeIds.size(); ++i)
    {
        full.operations.push_back ({ full.nodeIds[i], full.nodeKinds[i],
            i == 0 ? "source-decode" : "native-gpu", "" });
        if (i + 1 < full.nodeIds.size())
            full.edges.push_back ({ full.nodeIds[i], 0, full.nodeIds[i + 1], 0 });
    }
    FakeLayer preserved;
    check (videowire::executeVisualLayerPlan ({ full }, 7, preserved, error)
           && preserved.scale == 2.0f && preserved.effects != nullptr
           && preserved.maskType == 2 && preserved.opacity == 0.4f,
           "supported full production chain preserves existing rendering inputs");

    FakeLayer legacy;
    auto legacyPlan = direct;
    legacyPlan.nodeKinds = {
        "video.legacy.source", "video.legacy.retime", "video.legacy.transform",
        "video.legacy.effects", "video.out"
    };
    legacyPlan.nodeIds.clear();
    legacyPlan.ports.clear();
    legacyPlan.edges.clear();
    legacyPlan.operations.clear();
    check (videowire::executeVisualLayerPlan ({ legacyPlan }, 7, legacy, error)
           && legacy.scale == 2.0f && legacy.effects != nullptr && legacy.maskType == 2,
           "fixed legacy plan preserves the complete established renderer chain");

    FakeLayer noPlan;
    check (videowire::executeVisualLayerPlan ({}, 7, noPlan, error)
           && noPlan.scale == 2.0f && noPlan.effects != nullptr,
           "snapshots without compiled plans retain legacy production rendering");

    // Dependency-light production-path telemetry coverage.
    auto measuredPlan = directPlan();
    measuredPlan.structuralRevision = 9;
    videowire::VisualPlanExecutionState measuredState;
    videowire::VisualTelemetryPlanAdmission measuredAdmission;
    measuredAdmission.clipId = 7; measuredAdmission.structuralRevision = 9;
    measuredAdmission.nodeCount = 2; measuredAdmission.stableNodeIds[0] = 12;
    measuredAdmission.stableNodeIds[1] = 11; measuredAdmission.intermediateImageCount = 1;
    measuredState.telemetry().admitPlans({ measuredAdmission });
    measuredState.admitPlans({ measuredPlan });
    FakeLayer measuredLayerA, measuredLayerB;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerA, error,
                                             nullptr, nullptr, &measuredState, 1.0)
           && videowire::executeVisualLayerPlan({ measuredPlan }, 7, measuredLayerB, error,
                                                nullptr, nullptr, &measuredState, 2.0),
           "telemetry fixture executes two monotonic evaluations without sleeps");
    auto measured = measuredState.telemetry().snapshot();
    check (measured.graphEvaluations == 2 && measured.graphMeasuredTotalNs > 0
           && measured.planCacheMisses == 1 && measured.planCacheHits == 2
           && measured.planInstalls == 1,
           "graph accounting and plan lowering/install cache hit/miss are measured");
    check (measured.layers.size() == 1 && measured.nodes.size() == 2
           && measured.layers[0].clipId == 7 && measured.layers[0].structuralRevision == 9
           && measured.nodes[0].clipId == 7 && measured.nodes[0].structuralRevision == 9
           && measured.nodes[0].stableNodeId == 11 && measured.nodes[1].stableNodeId == 12,
           "telemetry preserves exact clip/revision/stable-node identities in deterministic order");
    FakeLayer seekLayer;
    check (videowire::executeVisualLayerPlan({ measuredPlan }, 7, seekLayer, error,
                                             nullptr, nullptr, &measuredState, 0.5),
           "backward seek re-evaluates the current plan");
    measured = measuredState.telemetry().snapshot();
    check (measured.graphEvaluations == 1 && measured.planCacheMisses == 0
           && measured.planCacheHits == 1 && measured.layers[0].evaluations == 1,
           "backward seek resets per-revision costs and compile cache accounting");
    auto revisedPlan = measuredPlan;
    revisedPlan.structuralRevision = 10;
    measuredAdmission.structuralRevision = 10;
    measuredState.telemetry().admitPlans({ measuredAdmission });
    measuredState.admitPlans({ revisedPlan });
    measured = measuredState.telemetry().snapshot();
    check (measured.graphEvaluations == 0 && measured.layers.size() == 1
           && measured.layers[0].structuralRevision == 10 && measured.layers[0].evaluations == 0
           && measured.nodes.size() == 2 && measured.nodes[0].evaluations == 0,
           "admitting a new revision immediately replaces stale identities with zeroed slots");
    FakeLayer revisedLayer;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, revisedLayer, error,
                                             nullptr, nullptr, &measuredState, 3.0),
           "new structural revision installs independently");
    measured = measuredState.telemetry().snapshot();
    check (measured.layers.size() == 1 && measured.layers[0].structuralRevision == 10
           && measured.nodes.size() == 2 && measured.nodes[0].structuralRevision == 10,
           "revision change removes stale identities");

    videowire::VisualPlanTelemetry bounded;
    videowire::VisualTelemetryPlanAdmission exact;
    exact.clipId = 2; exact.structuralRevision = 1; exact.nodeCount = 8;
    exact.intermediateImageCount = 7;
    for (int i = 0; i < 8; ++i) exact.stableNodeIds[(size_t) i] = 8 - i;
    auto duplicate = exact; duplicate.structuralRevision = 0;
    auto stale = exact; stale.clipId = 1; stale.structuralRevision = 1;
    check (bounded.admitPlans({ exact, duplicate, stale }), "exact 8-node/7-image plans admit");
    bounded.recordEvaluation(2, 1, 100, false);
    check(bounded.recordNodeEvaluation(2, 1, 8, 40)
          && bounded.recordNodeEvaluation(2, 1, 8, 60)
          && ! bounded.recordNodeEvaluation(2, 1, 999, 10),
          "node costs accept only measured samples for admitted stable identities");
    bounded.recordEvaluation(99, 1, 100, false);
    const auto boundedSnapshot = bounded.snapshot();
    check (boundedSnapshot.layers.size() == 2 && boundedSnapshot.nodes.size() == 16
           && boundedSnapshot.layers[0].clipId == 1 && boundedSnapshot.layers[1].clipId == 2
           && boundedSnapshot.nodes[8].stableNodeId == 1,
           "duplicate identities normalize and fixed-capacity snapshot ordering is deterministic");

    auto telemetryCapped = exact; telemetryCapped.clipId = 3;
    telemetryCapped.nodeCount = videowire::kMaxCompiledNodesPerGraph;
    telemetryCapped.intermediateImageCount = videowire::kMaxIntermediateImagesPerGraph;
    std::string admissionError;
    check (bounded.admitPlans({ exact, telemetryCapped }, &admissionError)
           && bounded.snapshot().rejectedPlans == 0,
           "telemetry admits a truthful fixed-size subset without rejecting the graph");

    auto largerComposite = composite;
    largerComposite.nodeKinds.insert(largerComposite.nodeKinds.end() - 2, "video.layer.source");
    largerComposite.nodeIds.insert(largerComposite.nodeIds.end() - 2, 99);
    largerComposite.operations.insert(largerComposite.operations.end() - 2,
        { 99, "video.layer.source", "native-gpu", "" });
    largerComposite.edges.insert(largerComposite.edges.end() - 1, { 99, 0, 24, 4 });
    FakeLayer largerLayer;
    check (largerComposite.operations.size() > videowire::kMaxCompiledNodesPerGraph
           && videowire::executeVisualLayerPlan({ largerComposite }, 7, largerLayer, error),
           "larger established typed graph remains executable beyond telemetry detail caps");

    const auto beforeHold = measuredState.telemetry().snapshot();
    FakeLayer heldMeasured;
    check (videowire::executeVisualLayerPlan({ revisedPlan }, 7, heldMeasured, error,
                                             nullptr, nullptr, &measuredState, 3.0),
           "paused hold executes");
    const auto afterHold = measuredState.telemetry().snapshot();
    check (afterHold.graphEvaluations == beforeHold.graphEvaluations
           && afterHold.graphMeasuredTotalNs == beforeHold.graphMeasuredTotalNs
           && afterHold.graphMeasuredMovingNs == beforeHold.graphMeasuredMovingNs
           && afterHold.planCacheHits == beforeHold.planCacheHits
           && afterHold.planCacheMisses == beforeHold.planCacheMisses
           && afterHold.planInstalls == beforeHold.planInstalls
           && afterHold.lastPlanLoweringNs == beforeHold.lastPlanLoweringNs,
           "paused hold freezes the complete telemetry cache and evaluation contract");

    videowire::VisualPlanTelemetry contended;
    contended.admitPlans({ exact });
    {
        auto heldLock = contended.lockForTesting();
        contended.recordEvaluation(2, 1, 10, false);
        contended.recordNodeEvaluation(2, 1, 8, 10);
        contended.resetOwner(2, 1);
        contended.recordPlanLowering(false, 123, true);
    }
    const auto contentionSnapshot = contended.snapshot();
    check (contentionSnapshot.droppedSamples == 3 && contentionSnapshot.planCacheMisses == 1
           && contentionSnapshot.planInstalls == 1 && contentionSnapshot.lastPlanLoweringNs == 123,
           "contention drops evaluation but preserves first post-reset plan-lowering/install sample");

    videowire::VisualPlanTelemetry staleContended;
    staleContended.admitPlans({ exact });
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 456, true);
        staleContended.resetOwner(2, 1);
    }
    const auto staleContention = staleContended.snapshot();
    check (staleContention.planCacheMisses == 0 && staleContention.planInstalls == 0
           && staleContention.lastPlanLoweringNs == 0,
           "reset generation discards a deferred pre-reset lowering sample");
    {
        auto heldLock = staleContended.lockForTesting();
        staleContended.recordPlanLowering(false, 0, true);
    }
    const auto zeroDuration = staleContended.snapshot();
    check (zeroDuration.planCacheMisses == 1 && zeroDuration.planInstalls == 1
           && zeroDuration.lastPlanLoweringNs == 0,
           "explicit pending presence preserves a legitimate zero-duration lowering sample");

    videowire::VisualPlanTelemetry mixedGeneration;
    mixedGeneration.admitPlans({ exact });
    {
        auto heldLock = mixedGeneration.lockForTesting();
        mixedGeneration.recordPlanLowering(false, 41, true);
        mixedGeneration.resetOwner(2, 1);
        mixedGeneration.recordPlanLowering(true, 0, false);
    }
    const auto mixedSnapshot = mixedGeneration.snapshot();
    check (mixedSnapshot.planCacheMisses == 0 && mixedSnapshot.planInstalls == 0
           && mixedSnapshot.planCacheHits == 1 && mixedSnapshot.lastPlanLoweringNs == 0,
           "one drain rejects pre-reset pending data while preserving post-reset data");

    auto filteredPlan = directPlan();
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 1,
        { 90, "control.constant", "control-eval", "" });
    filteredPlan.operations.insert(filteredPlan.operations.begin() + 2,
        { 91, "visual.depth.asset", "parked-metadata", "" });
    const auto filteredAdmission = videowire::makeVisualTelemetryAdmission(filteredPlan);
    check (filteredAdmission.executableNodeTotal == 2 && filteredAdmission.nodeCount == 2
           && filteredAdmission.stableNodeIds[0] == 11
           && filteredAdmission.stableNodeIds[1] == 12 && ! filteredAdmission.nodesTruncated,
           "telemetry reports only executable source-decode/native-gpu operations");

    FakeLayer invariantLayer;
    const std::vector<videowire::CompiledVisualLayerPlan> invariantPlans { revisedPlan };
    const auto mapSizeBefore = invariantLayer.genParams.size();
    const auto allocationsBefore = allocations.load(std::memory_order_relaxed);
    check (videowire::executeVisualLayerPlan(invariantPlans, 7, invariantLayer, error,
                                             nullptr, nullptr, &measuredState, 4.0),
           "pre-admitted execution renders through fixed lookup slots");
    check (allocations.load(std::memory_order_relaxed) == allocationsBefore
           && invariantLayer.genParams.size() == mapSizeBefore,
           "pre-admitted render lookup performs no allocation or map insertion");

    contended.seedForSaturationTesting(std::numeric_limits<uint64_t>::max());
    contended.recordEvaluation(2, 1, std::numeric_limits<uint64_t>::max(), false);
    contended.recordNodeEvaluation(2, 1, 1, 1);
    contended.recordPlanLowering(true, 0, false);
    contended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::zeroCopy, true);
    contended.recordZeroCopyAllocationBytes(1); contended.recordReadbackCopiedBytes(1);
    contended.recordDrop(videowire::VisualDropReason::noBuffer);
    contended.recordDimensions(2, 2, 1, 1);
    const auto saturated = contended.snapshot();
    check (saturated.graphEvaluations == std::numeric_limits<uint64_t>::max()
           && saturated.graphMeasuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].evaluations == std::numeric_limits<uint64_t>::max()
           && saturated.nodes[0].measuredTotalNs == std::numeric_limits<uint64_t>::max()
           && saturated.planCacheHits == std::numeric_limits<uint64_t>::max()
           && saturated.compositor.count == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyFrames == std::numeric_limits<uint64_t>::max()
           && saturated.zeroCopyAllocationBytes == std::numeric_limits<uint64_t>::max()
           && saturated.readbackCopiedBytes == std::numeric_limits<uint64_t>::max()
           && saturated.framesDropped == std::numeric_limits<uint64_t>::max()
           && saturated.dimensionMismatchCount == std::numeric_limits<uint64_t>::max(),
           "all telemetry counters and measured durations saturate");
    const auto telemetryJson = videowire::visualTelemetryJson(boundedSnapshot);
    videowire::VisualPlanTelemetry runtime;
    const auto unseenJson = videowire::visualTelemetryJson(runtime.snapshot());
    check(! unseenJson["compositor"]["available"].get<bool>() && unseenJson["compositor"]["movingNs"].is_null()
          && ! unseenJson["transport"]["available"].get<bool>() && unseenJson["transport"]["mode"].is_null()
          && ! unseenJson["backend"]["available"].get<bool>() && unseenJson["backend"]["current"].is_null()
          && ! unseenJson["resources"]["available"].get<bool>() && unseenJson["resources"]["retainedFramesCurrent"].is_null()
          && ! unseenJson["dimensions"]["available"].get<bool>() && unseenJson["dimensions"]["actualWidth"].is_null(),
          "unseen runtime families emit available false and null values, never production-looking zeroes");
    check(runtime.admitSessionBackend(videowire::VisualBackend::openGL)
          && runtime.admitSessionBackend(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal)
          && runtime.recordBackendTransition(videowire::VisualBackend::metal),
          "bounded backend admission happens once and only explicit transitions are events");
    check(runtime.recordDimensions(1920, 1080, 1920, 1080)
          && ! runtime.recordDimensions(0, 1080, 1920, 1080)
          && ! runtime.recordDimensions(std::numeric_limits<int>::max(), 1080, 1920, 1080),
          "dimensions reject non-positive and oversized values before arithmetic");
    check(! runtime.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::none, true)
          && runtime.recordRuntime(100, 40, 0, 0, videowire::VisualTransportMode::zeroCopy, true)
          && runtime.recordZeroCopyAllocationBytes(1920ull*1080ull*4ull)
          && runtime.recordReadbackCopiedBytes(123),
          "transport rejects none, zero durations count, and byte semantics use separate typed APIs");
    check(! runtime.snapshot().particles.observed && ! runtime.snapshot().generators.observed,
          "generic runtime transport does not fabricate generator or particle execution");
    check(runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 17)
          && runtime.recordExecutionObservation(videowire::VisualExecutionKind::particle, 23),
          "typed production execution hooks independently record generator and particle work");
    runtime.recordResources(3, 7, 4096); runtime.recordResources(0, 0, 0);
    runtime.recordDrop(videowire::VisualDropReason::noBuffer);
    runtime.recordDimensions(1000, 1000, 900, 900);
    runtime.recordDimensions(3840, 2160, 1920, 1080);
    const auto runtimeSnapshot = runtime.snapshot();
    const auto runtimeJson = videowire::visualTelemetryJson(runtimeSnapshot);
    check(runtimeSnapshot.compositor.count == 1 && runtimeSnapshot.presentation.totalNs == 40
          && runtimeSnapshot.particles.count == 1 && runtimeSnapshot.particles.totalNs == 23
          && runtimeSnapshot.generators.count == 1 && runtimeSnapshot.generators.totalNs == 17
          && runtimeSnapshot.zeroCopyFrames == 1 && runtimeSnapshot.zeroCopyAllocationBytes == 1920ull*1080ull*4ull
          && runtimeSnapshot.readbackCopiedBytes == 123 && runtimeSnapshot.framesRendered == 1
          && runtimeSnapshot.framesPresented == 1 && runtimeSnapshot.framesDropped == 1
          && runtimeSnapshot.droppedNoBuffer == 1 && runtimeSnapshot.fallbackCount == 1,
          "runtime schema records typed helper samples without claiming production owners");
    check(runtimeSnapshot.retainedFramesCurrent == 0 && runtimeSnapshot.retainedFramesPeak == 3
          && runtimeSnapshot.intermediateImagesCurrent == 0 && runtimeSnapshot.intermediateImagesPeak == 7
          && runtimeSnapshot.retainedBytesCurrent == 0 && runtimeSnapshot.retainedBytesPeak == 4096
          && runtimeSnapshot.requestedWidth == 3840 && runtimeSnapshot.actualWidth == 1920
          && runtimeSnapshot.dimensionMismatchCount == 2 && runtimeSnapshot.halfResolutionMismatchCount == 1,
          "resource snapshots preserve peaks/releases and dimensions count all mismatches plus exact halves");
    check(runtimeJson["transport"]["mode"] == "zero-copy" && runtimeJson["drops"]["reasons"]["noBuffer"] == 1
          && runtimeJson["resources"]["retainedFramesPeak"] == 3
          && runtimeJson["dimensions"]["halfResolutionMismatchCount"] == 1,
          "observed runtime JSON projects typed transport, drop, resource and dimension fields");
    runtime.resetOwner(0, 0);
    const auto graphReset = runtime.snapshot();
    check(! graphReset.transportObserved && ! graphReset.resourcesObserved && ! graphReset.dimensionsObserved
          && graphReset.retainedFramesCurrent == 0 && graphReset.retainedFramesPeak == 3
          && ! graphReset.particles.observed && ! graphReset.generators.observed
          && graphReset.backendObserved && graphReset.fallbackCount == 1,
          "graph/seek reset clears runtime/current values while preserving resource peaks and session backend");
    runtime.recordExecutionObservation(videowire::VisualExecutionKind::generator, 31);
    check(runtime.snapshot().generators.observed && runtime.snapshot().generators.count == 1
          && runtime.snapshot().generators.totalNs == 31,
          "the first real execution after a requested reset belongs to the new generation");
    runtime.resetSession();
    const auto sessionReset = runtime.snapshot();
    check(! sessionReset.backendObserved && sessionReset.fallbackCount == 0,
          "session reset separately clears initial/current backend and fallback history");
    videowire::VisualPlanTelemetry runtimeContended;
    {
        auto heldLock = runtimeContended.lockForTesting();
        runtimeContended.recordRuntime(1, 1, 1, 1, videowire::VisualTransportMode::readback, true);
        runtimeContended.recordReadbackCopiedBytes(1); runtimeContended.recordZeroCopyAllocationBytes(1);
        runtimeContended.recordDrop(videowire::VisualDropReason::renderFailure);
        runtimeContended.recordResources(1, 1, 1); runtimeContended.recordDimensions(1, 1, 1, 1);
        runtimeContended.admitSessionBackend(videowire::VisualBackend::openGL);
        runtimeContended.recordFailedLowering("contended");
        runtimeContended.recordExecutionObservation(videowire::VisualExecutionKind::particle, 1);
    }
    check(runtimeContended.snapshot().recordingContentionDrops == 9,
          "every runtime recording API is nonblocking and explicitly counts contention drops");
    check (telemetryJson.is_object() && telemetryJson["layers"].is_array()
           && telemetryJson["nodes"].is_array() && telemetryJson["nodes"].size() == 16
           && telemetryJson["nodes"][0]["clipId"] == 1
           && telemetryJson["nodes"][0]["structuralRevision"] == 1
           && telemetryJson["nodes"][0]["stableNodeId"] == 1
           && ! telemetryJson["nodes"][0]["available"].get<bool>()
           && telemetryJson["nodes"][0]["evaluations"].is_null()
           && telemetryJson["nodes"][0]["measuredTotalNs"].is_null()
           && telemetryJson["nodes"][0]["measuredMovingNs"].is_null()
           && telemetryJson["nodes"][15]["available"].get<bool>()
           && telemetryJson["nodes"][15]["evaluations"] == 2
           && telemetryJson["nodes"][15]["measuredTotalNs"] == 100
           && telemetryJson["nodes"][15]["measuredMovingNs"] == 44.0
           && telemetryJson.contains("planCacheHits")
           && telemetryJson.contains("lastPlanLoweringWithinBudget"),
           "viewport telemetry JSON has stable bounded arrays and exact identity/cache fields");

    std::printf ("visual plan executor: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
