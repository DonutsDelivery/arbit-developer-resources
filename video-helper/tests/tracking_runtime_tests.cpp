#include "../src/tracking_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
namespace
{
int failures = 0;
void check(bool condition, const char* message)
{
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

nlohmann::json pointAsset()
{
    return { {"assetType", "PointTrackAsset"}, {"domain", {{"width", 100}, {"height", 50}}},
             {"samples", nlohmann::json::array({{{"trajectory", nlohmann::json::array({
                 {{"seconds", 0.0}, {"state", "valid"}, {"x", 10.0}, {"y", 10.0}},
                 {{"seconds", 1.0}, {"state", "valid"}, {"x", 20.0}, {"y", 15.0}}
             })}}})} };
}

nlohmann::json planarAsset()
{
    return { {"assetType", "PlanarTrackAsset"}, {"domain", {{"width", 100}, {"height", 50}}},
             {"samples", nlohmann::json::array({
                 {{"seconds", 0.0}, {"state", "valid"}, {"corners", {{0.0,0.0},{99.0,0.0},{99.0,49.0},{0.0,49.0}}}},
                 {{"seconds", 1.0}, {"state", "valid"}, {"corners", {{10.0,5.0},{89.0,5.0},{89.0,44.0},{10.0,44.0}}}}
             })} };
}

std::string publish(const fs::path& trackingRoot, nlohmann::json asset)
{
    const auto samples = asset.at("samples");
    const auto payload = std::string("tracking-asset-receipt-v1\0", 26)
        + videohelper::trackingruntime::canonicalMetadata(asset).dump() + "\n" + samples.dump();
    const auto receipt = videohelper::sha256Text(payload);
    asset["contentReceipt"] = {{"value", receipt}};
    fs::create_directories(trackingRoot / receipt);
    const auto path = trackingRoot / receipt / "asset.json";
    std::ofstream(path) << asset.dump();
    fs::permissions(path, fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                    fs::perm_options::replace);
    return receipt;
}

videowire::CompiledVisualLayerPlan plan(int clipId, bool point, const std::string& receipt)
{
    videowire::CompiledVisualLayerPlan p; p.clipId = clipId; p.producerValidated = true;
    const std::string sourceKind = point ? "tracking.point.asset" : "tracking.planar.asset";
    const std::string applyKind = point ? "tracking.point.apply.transform" : "tracking.planar.apply.quad";
    p.operations = {
        {1, sourceKind, "control-eval", "<TrackingAssetBinding cacheKey=\"" + receipt + "\" contentReceipt=\"" + receipt + "\"/>"},
        {2, "tracking.correction", "control-eval", point
            ? "<TrackingCorrection><K t=\"0\" v0=\"0\" v1=\"0\"/><K t=\"2\" v0=\"0.2\" v1=\"-0.2\"/></TrackingCorrection>"
            : "<TrackingCorrection><K t=\"0\" v0=\"0\" v1=\"0\" v2=\"0\" v3=\"0\" v4=\"0\" v5=\"0\" v6=\"0\" v7=\"0\"/></TrackingCorrection>"},
        {3, applyKind, "native-gpu", ""}
    };
    p.edges = {{1,0,2,0},{2,1,3,0}};
    return p;
}
}

int main()
{
    const auto root = fs::canonical(fs::temp_directory_path()) / ("tracking-runtime-" + std::to_string(::getpid()));
    const auto trackingRoot = root / "tracking";
    fs::create_directories(trackingRoot);
    const auto pointReceipt = publish(trackingRoot, pointAsset());
    const auto planarReceipt = publish(trackingRoot, planarAsset());
    const std::string depthRoot = (root / "depth").string();
    const std::vector<videowire::CompiledVisualLayerPlan> plans { plan(7,true,pointReceipt), plan(8,false,planarReceipt) };
    std::string error;

    videorender::LayerDesc point;
    check(videohelper::trackingruntime::prepareTracking(depthRoot, plans, 7, 1.0, point, error)
          && std::abs(point.translateX - 0.3f) < 1.0e-6f
          && std::abs(point.translateY - 0.1f) < 1.0e-6f && !point.cornerPin,
          "point runtime applies trajectory plus interpolated correction");

    videorender::LayerDesc planar;
    error.clear();
    check(videohelper::trackingruntime::prepareTracking(depthRoot, plans, 8, 1.0, planar, error)
          && planar.cornerPin && std::abs(planar.corners[0] + 0.79f) < 1.0e-6f
          && std::abs(planar.corners[1] - 0.78f) < 1.0e-6f,
          "planar runtime produces deterministic corner-pin coordinates");

    videorender::LayerDesc secondPoint;
    error.clear();
    check(videohelper::trackingruntime::prepareTracking(depthRoot, plans, 7, 1.0, secondPoint, error)
          && secondPoint.translateX == point.translateX && secondPoint.translateY == point.translateY,
          "two-layer evaluation remains clip-local and deterministic");

    auto stalePlans = plans;
    stalePlans[0].operations[0].payloadXml.replace(stalePlans[0].operations[0].payloadXml.find(pointReceipt), pointReceipt.size(), std::string(64, 'a'));
    videorender::LayerDesc rejected;
    error.clear();
    check(!videohelper::trackingruntime::prepareTracking(depthRoot, stalePlans, 7, 1.0, rejected, error)
          && error == "tracking binding receipt is invalid", "stale binding is rejected");

    const auto replacedPath = trackingRoot / pointReceipt / "asset.json";
    fs::permissions(replacedPath, fs::perms::owner_write, fs::perm_options::add);
    std::ofstream(replacedPath, std::ios::trunc) << "{}";
    fs::permissions(replacedPath, fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                    fs::perm_options::replace);
    error.clear();
    check(!videohelper::trackingruntime::prepareTracking(depthRoot, plans, 7, 1.0, rejected, error)
          && error == "tracking receipt is malformed", "replaced receipt payload is rejected");

    fs::remove_all(root);
    if (failures == 0) std::puts("tracking runtime tests: PASS");
    return failures == 0 ? 0 : 1;
}
