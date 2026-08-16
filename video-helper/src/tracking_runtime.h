#pragma once

#include "depth_cache.h"
#include "render_snapshot.h"
#include "renderer.h"
#include "sha256.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace videohelper
{
namespace trackingruntime
{
inline std::string xmlAttribute (const std::string& xml, const char* name)
{
    const std::string key = std::string(name) + "=\"";
    const auto begin = xml.find(key);
    if (begin == std::string::npos) return {};
    const auto end = xml.find('"', begin + key.size());
    return end == std::string::npos ? std::string{} : xml.substr(begin + key.size(), end - begin - key.size());
}
inline bool finiteNumber (const nlohmann::json& value)
{ return value.is_number() && std::isfinite(value.get<double>()); }
inline nlohmann::json canonicalMetadata (nlohmann::json asset)
{
    asset.erase("samples"); asset.erase("contentReceipt"); return asset;
}
inline bool admitAsset (const std::string& depthRoot, const std::string& key,
                        const std::string& expectedReceipt, nlohmann::json& asset,
                        std::string& error)
{
#if defined(__linux__) || defined(__APPLE__)
    if (!videohelper::depthcache::lowercaseHash(key) || key != expectedReceipt)
    { error="tracking binding receipt is invalid"; return false; }
    const auto slash=depthRoot.find_last_of('/');
    if (slash==std::string::npos) { error="private tracking cache root is unavailable"; return false; }
    auto root=videohelper::depthcache::openAbsoluteDirectory(depthRoot.substr(0,slash)+"/tracking");
    if (!root) { error="private tracking cache root is unavailable"; return false; }
    videohelper::depthcache::Fd receipt(::openat(root.get(),key.c_str(),O_RDONLY|O_CLOEXEC|O_DIRECTORY|O_NOFOLLOW));
    if (!receipt || !videohelper::depthcache::exactEntries(receipt.get(),{"asset.json"}))
    { error="tracking receipt is missing or replaced"; return false; }
    std::vector<uint8_t> bytes; struct stat identity{};
    if (!videohelper::depthcache::readSealedFile(receipt.get(),"asset.json",256ull*1024*1024,bytes,identity))
    { error="tracking receipt is missing or replaced"; return false; }
    try { asset=nlohmann::json::parse(bytes.begin(),bytes.end()); }
    catch (...) { error="tracking receipt is malformed"; return false; }
    try
    {
        const auto receiptValue=asset.at("contentReceipt").at("value").get<std::string>();
        const auto samples=asset.at("samples");
        const auto payload=std::string("tracking-asset-receipt-v1\0",26)
            + canonicalMetadata(asset).dump() + "\n" + samples.dump();
        if (receiptValue!=expectedReceipt || videohelper::sha256Text(payload)!=expectedReceipt)
        { error="tracking receipt is stale or mutated"; return false; }
    }
    catch (...) { error="tracking receipt is malformed"; return false; }
    return true;
#else
    (void)depthRoot;(void)key;(void)expectedReceipt;(void)asset;
    error="native tracking execution is unavailable on this platform"; return false;
#endif
}
inline std::vector<double> correctionAt (const std::string& xml, double pts, int channels)
{
    std::vector<std::pair<double,std::vector<double>>> keys;
    size_t at=0;
    while ((at=xml.find("<K ",at))!=std::string::npos)
    {
        const auto end=xml.find("/>",at); if(end==std::string::npos) break;
        const auto row=xml.substr(at,end+2-at); const auto t=xmlAttribute(row,"t");
        try { std::vector<double> values; for(int i=0;i<channels;++i) values.push_back(std::stod(xmlAttribute(row,("v"+std::to_string(i)).c_str()))); keys.push_back({std::stod(t),values}); } catch (...) {}
        at=end+2;
    }
    std::sort(keys.begin(),keys.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    std::vector<double> out((size_t)channels,0.0); if(keys.empty()) return out;
    if (pts <= keys.front().first) return keys.front().second;
    if (pts >= keys.back().first) return keys.back().second;
    const auto b=std::upper_bound(keys.begin(),keys.end(),pts,[](double t,const auto&k){return t<k.first;}); const auto a=b-1;
    const double alpha=(pts-a->first)/(b->first-a->first); for(int i=0;i<channels;++i) out[(size_t)i]=a->second[(size_t)i]+(b->second[(size_t)i]-a->second[(size_t)i])*alpha; return out;
}
inline bool prepareTracking (const std::string& depthRoot,
                             const std::vector<videowire::CompiledVisualLayerPlan>& plans,
                             int clipId, double exactSourcePts,
                             videorender::LayerDesc& layer, std::string& error)
{
    const auto plan=std::find_if(plans.begin(),plans.end(),[&](const auto&p){return p.clipId==clipId;});
    if(plan==plans.end()) return true;
    const auto apply=std::find_if(plan->operations.begin(),plan->operations.end(),[](const auto&o){return o.kind=="tracking.point.apply.transform"||o.kind=="tracking.planar.apply.quad";});
    if(apply==plan->operations.end()) return true;
    const bool point=apply->kind=="tracking.point.apply.transform"; const int channels=point?2:8;
    const auto incoming=[&](int to,int port){return std::find_if(plan->edges.begin(),plan->edges.end(),[&](const auto&e){return e.toNodeId==to&&e.toPort==port;});};
    const auto correctionEdge=incoming(apply->nodeId,0); if(correctionEdge==plan->edges.end()){error="tracking apply has no correction";return false;}
    const auto sourceEdge=incoming(correctionEdge->fromNodeId,0); if(sourceEdge==plan->edges.end()){error="tracking correction has no source";return false;}
    const auto operation=[&](int id){return std::find_if(plan->operations.begin(),plan->operations.end(),[&](const auto&o){return o.nodeId==id;});};
    const auto correction=operation(correctionEdge->fromNodeId), source=operation(sourceEdge->fromNodeId);
    if(correction==plan->operations.end()||source==plan->operations.end()){error="tracking usage is incomplete";return false;}
    const auto key=xmlAttribute(source->payloadXml,"cacheKey"), receipt=xmlAttribute(source->payloadXml,"contentReceipt"); nlohmann::json asset;
    if(!admitAsset(depthRoot,key,receipt,asset,error)) return false;
    try
    {
        if(asset.at("assetType")!=(point?"PointTrackAsset":"PlanarTrackAsset")){error="tracking receipt kind was replaced";return false;}
        const int width=asset.at("domain").at("width"),height=asset.at("domain").at("height");
        auto correctionValues=correctionAt(correction->payloadXml,exactSourcePts,channels);
        if(point)
        {
            const auto& samples=asset.at("samples").at(0).at("trajectory");
            auto sample=samples.begin(); for(auto it=samples.begin();it!=samples.end()&&it->at("seconds").get<double>()<=exactSourcePts;++it) sample=it;
            if(sample==samples.end()||sample->at("state")!="valid"||!finiteNumber(sample->at("x"))||!finiteNumber(sample->at("y"))){error="point tracking sample is unavailable at source PTS";return false;}
            const auto& origin=samples.at(0); layer.translateX+=(float)(2.0*(sample->at("x").get<double>()-origin.at("x").get<double>())/width+correctionValues[0]); layer.translateY+=(float)(2.0*(sample->at("y").get<double>()-origin.at("y").get<double>())/height+correctionValues[1]);
        }
        else
        {
            const auto& samples=asset.at("samples"); auto sample=samples.begin(); for(auto it=samples.begin();it!=samples.end()&&it->at("seconds").get<double>()<=exactSourcePts;++it) sample=it;
            if(sample==samples.end()||sample->at("state")!="valid"||sample->at("corners").size()!=4){error="planar tracking sample is unavailable at source PTS";return false;}
            layer.cornerPin=true; for(int i=0;i<4;++i){const auto&c=sample->at("corners").at(i);layer.corners[(size_t)i*2]=(float)(2.0*(c.at(0).get<double>()+0.5)/width-1.0+correctionValues[(size_t)i*2]);layer.corners[(size_t)i*2+1]=(float)(1.0-2.0*(c.at(1).get<double>()+0.5)/height+correctionValues[(size_t)i*2+1]);}
        }
    }
    catch (...) { error="tracking receipt sample is malformed"; return false; }
    return true;
}
}
}
