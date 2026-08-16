#define PROGRAMMABLE_RUNTIME_ENTROPY_TESTING 1
#include "programmable_admission.h"
#include "../../shared/SecureEntropy.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <array>
#include <vector>
using namespace programmableruntime;

static Grant makeGrant(PayloadKind kind,const std::string& source,const SessionSecret& secret,
                       uint64_t generation,uint64_t nonce,uint64_t now)
{
    Grant g; g.kind=kind; g.fingerprint=fingerprint(kind,source); g.sessionGeneration=generation;
    g.nonce=nonce; g.issuedAtMs=now; g.approved=true; g.cpuMs=25; g.gpuMs=50; g.memoryMiB=64;
    g.mac=sign(secret,g,kind); return g;
}
static void appendU64(std::vector<uint8_t>& out,uint64_t value){for(int s=56;s>=0;s-=8)out.push_back(uint8_t(value>>s));}
static void append(std::vector<uint8_t>& out,std::string_view value){appendU64(out,value.size());out.insert(out.end(),value.begin(),value.end());}
static bool failEntropy(void*,size_t){return false;}
static int entropyCalls=0;
static bool failSecondEntropy(void* out,size_t size)
{
    ++entropyCalls;
    if(entropyCalls==2) return false;
    std::fill_n(static_cast<uint8_t*>(out),size,0xa5);
    return true;
}

int main()
{
    // RFC 4231 test case 1: fixed external HMAC-SHA256 known answer.
    SessionSecret rfcKey{}; std::fill(rfcKey.begin(),rfcKey.end(),0x0b);
    const std::string hi="Hi There";
    assert(hmacSha256(rfcKey,hi.data(),hi.size())=="198a607eb44bfbc69903a0f1cf2bbdc5ba0aa3f3d9ae3c1c7a3b1696a0b68cf7");

    // Manual framing oracle: does not call canonicalGrant, sign, or fingerprint.
    SessionSecret vectorKey{}; for(size_t i=0;i<vectorKey.size();++i)vectorKey[i]=uint8_t(i);
    std::vector<uint8_t> framed{3}; append(framed,"DonutStudio/RuntimeGrant/HMAC-SHA256");
    appendU64(framed,3); append(framed,"shader"); append(framed,std::string(64,'1'));
    append(framed,"pack"); append(framed,"program");
    for(uint64_t v:{7,9,123456,1,0,0,1,25,50,64}) appendU64(framed,v);
    append(framed,"shader");
    assert(hmacSha256(vectorKey,framed.data(),framed.size())=="417979296781323c0f1abbc48965ab1291c9b54b6da3e677340196b87f6f15ae");

    SessionSecret secret{}; for(size_t i=0;i<secret.size();++i)secret[i]=uint8_t(i*7+3);
    const uint64_t generation=42,now=100000;
    std::ifstream input(std::string(SHADER_CATALOG_SOURCE_ROOT)+"/arbit-essentials/shaders/aurora_drift.fs",std::ios::binary);
    const std::string source((std::istreambuf_iterator<char>(input)),{}); assert(!source.empty());
    Grant parsed; std::string error;
    auto g=makeGrant(PayloadKind::shader,source,secret,generation,1,now);
    g.catalogPackId="arbit-essentials-shaders-v1"; g.catalogProgramId="aurora_drift";
    g.verifiedBundledCurated=true; g.mac=sign(secret,g,PayloadKind::shader);
    assert(programmableadmission::identifyCatalogGpuPayload(g,source,error)==PayloadKind::shader);
    programmableadmission::SessionVerifier verifier(secret,generation);
    assert(verifier.admit(programmableadmission::toJson(g),PayloadKind::shader,source,parsed,error,now));
    assert(programmableadmission::admitsGpuPayload(parsed,PayloadKind::shader,error));

    auto tampered=g; tampered.nonce=2; const std::string changed=source+"\n// tampered";
    tampered.fingerprint=fingerprint(PayloadKind::shader,changed); tampered.mac=sign(secret,tampered,PayloadKind::shader);
    error.clear(); assert(programmableadmission::identifyCatalogGpuPayload(tampered,changed,error)==PayloadKind::invalid);

    auto substitution=g; substitution.nonce=3; substitution.catalogProgramId="prime_lattice";
    substitution.mac=sign(secret,substitution,PayloadKind::shader);
    error.clear(); assert(programmableadmission::identifyCatalogGpuPayload(substitution,source,error)==PayloadKind::invalid);

    auto weaker=g; weaker.nonce=4; weaker.kind=PayloadKind::shader; weaker.catalogPackId="arbit-isf-starters-v1";
    weaker.catalogProgramId="warp_tunnel"; weaker.mac=sign(secret,weaker,PayloadKind::shader);
    error.clear(); assert(programmableadmission::identifyCatalogGpuPayload(weaker,source,error)==PayloadKind::invalid);

    // Exercise the production raw-field route: caller runtimeKind is ignored and
    // substituting catalog identity cannot select a weaker semantic kind.
    auto rawGrantWire=programmableadmission::toJson(g); rawGrantWire["runtimeKind"]="shader";
    rawGrantWire["catalogProgramId"]="prime_lattice";
    nlohmann::json rawOwner={{"shaderSource",source},{"runtimeKind","shader"},
                             {"runtimeGrant",rawGrantWire}};
    programmableadmission::SessionVerifier rawSession(secret,generation); Grant rawGrant;
    error.clear(); assert(!programmableadmission::admitCatalogGpuField(
        rawSession,rawOwner,"shaderSource","runtimeGrant",rawGrant,error,now));

    rawOwner["runtimeGrant"]=programmableadmission::toJson(g);
    rawOwner["runtimeKind"]="raymarch";
    programmableadmission::SessionVerifier independentKindSession(secret,generation);
    error.clear(); assert(programmableadmission::admitCatalogGpuField(
        independentKindSession,rawOwner,"shaderSource","runtimeGrant",rawGrant,error,now));

    std::ifstream isfInput(std::string(SHADER_CATALOG_SOURCE_ROOT)+"/arbit-isf-starters/shaders/warp_tunnel.fs",std::ios::binary);
    const std::string isfSource((std::istreambuf_iterator<char>(isfInput)),{}); assert(!isfSource.empty());
    auto isfGrant=makeGrant(PayloadKind::isf,isfSource,secret,generation,7,now);
    isfGrant.catalogPackId="arbit-isf-starters-v1"; isfGrant.catalogProgramId="warp_tunnel";
    isfGrant.verifiedBundledCurated=true; isfGrant.mac=sign(secret,isfGrant,PayloadKind::isf);
    error.clear(); assert(programmableadmission::identifyCatalogGpuPayload(isfGrant,isfSource,error)==PayloadKind::isf);
    nlohmann::json isfOwner={{"shaderSource",isfSource},{"runtimeKind","shader"},
                             {"runtimeGrant",programmableadmission::toJson(isfGrant)}};
    programmableadmission::SessionVerifier isfSession(secret,generation);
    error.clear(); assert(!programmableadmission::admitCatalogGpuField(
        isfSession,isfOwner,"shaderSource","runtimeGrant",rawGrant,error,now));
    assert(error=="sandbox unavailable");

    auto forged=programmableadmission::toJson(g); forged["nonce"]=5; forged["cpuMs"]=26;
    error.clear(); assert(!verifier.admit(forged,PayloadKind::shader,source,parsed,error,now));
    auto stale=g; stale.nonce=6; stale.issuedAtMs=now-programmableadmission::SessionVerifier::maxAgeMs-1;
    stale.mac=sign(secret,stale,PayloadKind::shader); error.clear();
    assert(!verifier.admit(programmableadmission::toJson(stale),PayloadKind::shader,source,parsed,error,now));

    SessionSecret entropySecret{}; uint64_t entropyGeneration=99;
    assert(!secureentropy::generateSessionMaterialForTest(entropySecret,entropyGeneration,failEntropy));
    assert(entropyGeneration==0); for(auto b:entropySecret) assert(b==0);
    entropyCalls=0;
    assert(!secureentropy::generateSessionMaterialForTest(entropySecret,entropyGeneration,failSecondEntropy));
    assert(entropyGeneration==0); for(auto b:entropySecret) assert(b==0);
    assert(secureentropy::generateSessionMaterial(entropySecret,entropyGeneration));
    assert(entropyGeneration!=0);

    // Transport can deliver serialized issuance out of order. Both unique,
    // authenticated nonces remain valid; a nonce is consumed exactly once.
    auto high=makeGrant(PayloadKind::lua,"return 9",secret,generation,101,now);
    auto low=makeGrant(PayloadKind::lua,"return 8",secret,generation,100,now);
    programmableadmission::SessionVerifier outOfOrder(secret,generation);
    error.clear(); assert(outOfOrder.admit(programmableadmission::toJson(high),PayloadKind::lua,"return 9",parsed,error,now));
    error.clear(); assert(outOfOrder.admit(programmableadmission::toJson(low),PayloadKind::lua,"return 8",parsed,error,now));

    // Deterministic replay race: a barrier releases both contenders together;
    // the verifier mutex and consumed set permit exactly one winner.
    auto raced=makeGrant(PayloadKind::javascript,"1+1",secret,generation,102,now);
    programmableadmission::SessionVerifier replayRace(secret,generation);
    std::mutex gateMutex; std::condition_variable gateCv; int arrived=0; bool release=false;
    bool results[2]{};
    auto contender=[&](int index)
    {
        { std::unique_lock<std::mutex> lock(gateMutex); ++arrived; gateCv.notify_all();
          gateCv.wait(lock,[&]{return release;}); }
        Grant local; std::string localError;
        results[index]=replayRace.admit(programmableadmission::toJson(raced),
            PayloadKind::javascript,"1+1",local,localError,now);
    };
    std::thread first(contender,0),second(contender,1);
    { std::unique_lock<std::mutex> lock(gateMutex); gateCv.wait(lock,[&]{return arrived==2;});
      release=true; gateCv.notify_all(); }
    first.join(); second.join(); assert(results[0] != results[1]);

    // The replay window is fixed-capacity and fails closed while all entries are
    // fresh. Once time advances beyond freshness, expired entries are reclaimed.
    programmableadmission::SessionVerifier bounded(secret,generation);
    for (std::size_t i=0;i<programmableadmission::SessionVerifier::replayCapacity;++i)
    {
        auto item=makeGrant(PayloadKind::lua,"return 3",secret,generation,1000+i,now);
        error.clear(); assert(bounded.admit(programmableadmission::toJson(item),PayloadKind::lua,
                                            "return 3",parsed,error,now));
    }
    auto overflow=makeGrant(PayloadKind::lua,"return 3",secret,generation,999999,now);
    error.clear(); assert(!bounded.admit(programmableadmission::toJson(overflow),PayloadKind::lua,
                                         "return 3",parsed,error,now));
    assert(error=="programmable runtime replay window capacity exhausted");
    auto afterExpiry=makeGrant(PayloadKind::lua,"return 4",secret,generation,1000000,
                               now+programmableadmission::SessionVerifier::maxAgeMs+1);
    error.clear(); assert(bounded.admit(programmableadmission::toJson(afterExpiry),PayloadKind::lua,
                                        "return 4",parsed,error,afterExpiry.issuedAtMs));

    // Distinct fresh grants released at one barrier all succeed exactly once.
    constexpr int distinctCount=64;
    programmableadmission::SessionVerifier distinct(secret,generation);
    std::mutex distinctMutex; std::condition_variable distinctCv;
    int distinctArrived=0; bool distinctRelease=false;
    std::array<bool,distinctCount> distinctFirst{}, distinctReplay{};
    std::vector<std::thread> distinctThreads;
    for(int i=0;i<distinctCount;++i)
        distinctThreads.emplace_back([&,i]{
            auto item=makeGrant(PayloadKind::lua,"return 5",secret,generation,2000000+i,now);
            { std::unique_lock<std::mutex> lock(distinctMutex); ++distinctArrived; distinctCv.notify_all();
              distinctCv.wait(lock,[&]{return distinctRelease;}); }
            Grant local; std::string localError;
            distinctFirst[i]=distinct.admit(programmableadmission::toJson(item),PayloadKind::lua,
                                             "return 5",local,localError,now);
            localError.clear();
            distinctReplay[i]=distinct.admit(programmableadmission::toJson(item),PayloadKind::lua,
                                              "return 5",local,localError,now);
        });
    { std::unique_lock<std::mutex> lock(distinctMutex);
      distinctCv.wait(lock,[&]{return distinctArrived==distinctCount;});
      distinctRelease=true; distinctCv.notify_all(); }
    for(auto& thread:distinctThreads) thread.join();
    for(int i=0;i<distinctCount;++i) assert(distinctFirst[i] && !distinctReplay[i]);

    // Reclamation is mutex-atomic: concurrent fresh admission can reclaim stale
    // entries, while a replay of a still-fresh nonce remains rejected.
    programmableadmission::SessionVerifier reclaim(secret,generation);
    auto staleEntry=makeGrant(PayloadKind::lua,"return 6",secret,generation,3000000,now);
    auto retained=makeGrant(PayloadKind::lua,"return 7",secret,generation,3000001,
                            now+programmableadmission::SessionVerifier::maxAgeMs);
    error.clear(); assert(reclaim.admit(programmableadmission::toJson(staleEntry),PayloadKind::lua,
                                        "return 6",parsed,error,now));
    error.clear(); assert(reclaim.admit(programmableadmission::toJson(retained),PayloadKind::lua,
                                        "return 7",parsed,error,retained.issuedAtMs));
    std::mutex reclaimMutex; std::condition_variable reclaimCv; int reclaimArrived=0; bool reclaimRelease=false;
    bool freshAccepted=false, retainedReplayAccepted=true;
    auto reclaimContender=[&](bool replay){
        { std::unique_lock<std::mutex> lock(reclaimMutex); ++reclaimArrived; reclaimCv.notify_all();
          reclaimCv.wait(lock,[&]{return reclaimRelease;}); }
        Grant local; std::string localError;
        if(replay)
            retainedReplayAccepted=reclaim.admit(programmableadmission::toJson(retained),PayloadKind::lua,
                "return 7",local,localError,retained.issuedAtMs+1);
        else {
            auto fresh=makeGrant(PayloadKind::lua,"return 8",secret,generation,3000002,
                                 now+programmableadmission::SessionVerifier::maxAgeMs+1);
            freshAccepted=reclaim.admit(programmableadmission::toJson(fresh),PayloadKind::lua,
                "return 8",local,localError,fresh.issuedAtMs);
        }
    };
    std::thread reclaimer(reclaimContender,false), replayDuringReclaim(reclaimContender,true);
    { std::unique_lock<std::mutex> lock(reclaimMutex); reclaimCv.wait(lock,[&]{return reclaimArrived==2;});
      reclaimRelease=true; reclaimCv.notify_all(); }
    reclaimer.join(); replayDuringReclaim.join();
    assert(freshAccepted && !retainedReplayAccepted);

    // A replacement helper has a new generation and secret. Old grants cannot
    // authenticate there, and resetting the stale verifier cannot resume them.
    SessionSecret replacementSecret{}; std::fill(replacementSecret.begin(),replacementSecret.end(),0x5c);
    programmableadmission::SessionVerifier replacement(replacementSecret,generation+1);
    error.clear(); assert(!replacement.admit(programmableadmission::toJson(high),PayloadKind::lua,"return 9",parsed,error,now));
    replayRace.reset(replacementSecret,generation+1);
    error.clear(); assert(!replayRace.admit(programmableadmission::toJson(raced),PayloadKind::javascript,"1+1",parsed,error,now));
    std::cout<<"authenticated programmable admission tests passed\n";
}
