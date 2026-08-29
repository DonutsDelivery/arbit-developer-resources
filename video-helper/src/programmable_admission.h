#pragma once
#include "../../shared/ProgrammableRuntimeGrant.h"
#include "../../shared/ShaderCatalogManifest.h"
#include <chrono>
#include <mutex>
#include <deque>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace programmableadmission
{
using json = nlohmann::json;

inline bool parseGrant (const json& value, programmableruntime::Grant& out, std::string& error)
{
    if (! value.is_object()) { error = "programmable runtime grant is missing"; return false; }
    try
    {
        out.version = value.at("version").get<uint32_t>();
        out.kind = programmableruntime::kindFromWire(value.at("kind").get<std::string>());
        out.fingerprint = value.at("fingerprint").get<std::string>();
        out.catalogPackId = value.at("catalogPackId").get<std::string>();
        out.catalogProgramId = value.at("catalogProgramId").get<std::string>();
        out.sessionGeneration = value.at("sessionGeneration").get<uint64_t>();
        out.nonce = value.at("nonce").get<uint64_t>();
        out.issuedAtMs = value.at("issuedAtMs").get<uint64_t>();
        out.approved = value.at("approved").get<bool>();
        out.disk = value.at("disk").get<bool>();
        out.network = value.at("network").get<bool>();
        out.verifiedBundledCurated = value.at("verifiedBundledCurated").get<bool>();
        out.cpuMs = value.at("cpuMs").get<uint32_t>();
        out.gpuMs = value.at("gpuMs").get<uint32_t>();
        out.memoryMiB = value.at("memoryMiB").get<uint32_t>();
        out.mac = value.at("mac").get<std::string>();
    }
    catch (const std::exception& e)
    {
        error = std::string("malformed programmable runtime grant: ") + e.what();
        return false;
    }
    return true;
}

inline uint64_t nowMs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

class SessionVerifier
{
public:
    static constexpr uint64_t maxAgeMs = 30000;
    static constexpr std::size_t replayCapacity = 4096;
    SessionVerifier() = default;
    SessionVerifier (const programmableruntime::SessionSecret& secret, uint64_t generation)
        : secret_(secret), generation_(generation), ready_(true) {}

    SessionVerifier (programmableruntime::SessionSecret&& secret, uint64_t generation)
        : secret_(secret), generation_(generation), ready_(true)
    {
        secureClear(secret);
    }

    void reset (const programmableruntime::SessionSecret& secret, uint64_t generation)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        secureClear(secret_);
        secret_ = secret;
        generation_ = generation;
        consumed_.clear();
        consumedNonces_.clear();
        ready_ = true;
    }

    void reset (programmableruntime::SessionSecret&& secret, uint64_t generation)
    {
        reset(static_cast<const programmableruntime::SessionSecret&>(secret), generation);
        secureClear(secret);
    }

    bool admit (const json& wire, programmableruntime::PayloadKind authoritativeKind,
                const std::string& payload, programmableruntime::Grant& grant,
                std::string& error, uint64_t now = nowMs())
    {
       #if defined(_WIN32)
        error = "programmable runtime grants unsupported on Windows";
        return false;
       #endif
        if (! parseGrant(wire, grant, error)
            || ! programmableruntime::admits(grant, authoritativeKind, payload, error))
            return false;

        const std::lock_guard<std::mutex> lock(mutex_);
        if (! ready_ || grant.sessionGeneration != generation_)
        {
            error = "programmable runtime grant helper session mismatch";
            return false;
        }
        if (grant.issuedAtMs > now || now - grant.issuedAtMs > maxAgeMs)
        {
            error = "programmable runtime grant is stale";
            return false;
        }
        for (auto it = consumed_.begin(); it != consumed_.end();)
        {
            if (it->issuedAtMs > now || now - it->issuedAtMs > maxAgeMs)
            {
                consumedNonces_.erase(it->nonce);
                it = consumed_.erase(it);
            }
            else ++it;
        }
        if (consumedNonces_.count(grant.nonce) != 0)
        {
            error = "programmable runtime grant replay";
            return false;
        }
        if (! programmableruntime::constantEqual(
                grant.mac, programmableruntime::sign(secret_, grant, authoritativeKind)))
        {
            error = "programmable runtime grant authentication failed";
            return false;
        }
        if (consumed_.size() >= replayCapacity)
        {
            error = "programmable runtime replay window capacity exhausted";
            return false;
        }
        // Issuance is serialized by the plugin authority, but transport and
        // worker scheduling are not. The fixed-capacity freshness window admits
        // unique out-of-order delivery and fails closed rather than evicting a
        // still-fresh nonce under load.
        consumedNonces_.insert(grant.nonce);
        consumed_.push_back({ grant.nonce, grant.issuedAtMs });
        return true;
    }

private:
    static void secureClear (programmableruntime::SessionSecret& secret)
    {
        volatile uint8_t* cursor = secret.data();
        for (std::size_t i = 0; i < secret.size(); ++i) cursor[i] = 0;
    }

    std::mutex mutex_;
    programmableruntime::SessionSecret secret_ {};
    uint64_t generation_ = 0;
    struct Consumed { uint64_t nonce; uint64_t issuedAtMs; };
    std::deque<Consumed> consumed_;
    std::unordered_set<uint64_t> consumedNonces_;
    bool ready_ = false;
};

inline SessionVerifier& verifier() { static SessionVerifier value; return value; }
inline programmableruntime::PayloadKind identifyCatalogGpuPayload (
    const programmableruntime::Grant& grant, const std::string& payload, std::string& error)
{
    const auto* entry = shadercatalog::find(grant.catalogPackId, grant.catalogProgramId);
    if (entry == nullptr)
    {
        error = "programmable shader catalog identity is missing or unknown";
        return programmableruntime::PayloadKind::invalid;
    }
    const auto digest = programmableruntime::detail::hex(programmableruntime::detail::sha256(
        reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
    if (digest != entry->sourceSha256)
    {
        error = "programmable shader catalog source mismatch";
        return programmableruntime::PayloadKind::invalid;
    }
    const auto kind = programmableruntime::kindFromWire(entry->kind);
    if (kind == programmableruntime::PayloadKind::invalid)
        error = "programmable shader catalog kind is invalid";
    return kind;
}
inline bool admitsGpuPayload (const programmableruntime::Grant& grant,
                              programmableruntime::PayloadKind kind, std::string& error)
{
    if ((kind != programmableruntime::PayloadKind::shader
         && kind != programmableruntime::PayloadKind::isf)
        || ! grant.verifiedBundledCurated)
    {
        error = "sandbox unavailable";
        return false;
    }
    return true;
}
inline bool admit (const json& wire, programmableruntime::PayloadKind kind,
                   const std::string& payload, programmableruntime::Grant& grant,
                   std::string& error)
{
    return verifier().admit(wire, kind, payload, grant, error);
}
inline bool admitCatalogGpuField (SessionVerifier& session, const json& owner,
                                  const char* sourceField, const char* grantField,
                                  programmableruntime::Grant& grant, std::string& error,
                                  uint64_t now = nowMs())
{
    const auto source = owner.value(sourceField, std::string{});
    if (source.empty()) return true;
    const auto it = owner.find(grantField);
    if (it == owner.end()) { error=std::string("missing ")+grantField; return false; }
    if (! parseGrant(*it, grant, error)) return false;
    const auto kind = identifyCatalogGpuPayload(grant, source, error);
    return kind != programmableruntime::PayloadKind::invalid
        && session.admit(*it, kind, source, grant, error, now)
        && admitsGpuPayload(grant, kind, error);
}
inline bool admitCatalogGpuField (const json& owner, const char* sourceField,
                                  const char* grantField, programmableruntime::Grant& grant,
                                  std::string& error)
{
    return admitCatalogGpuField(verifier(), owner, sourceField, grantField, grant, error);
}
inline bool admitField (const json& owner, const char* field,
                        programmableruntime::PayloadKind kind, const std::string& payload,
                        programmableruntime::Grant& grant, std::string& error)
{
    if (payload.empty()) return true;
    const auto it = owner.find(field);
    if (it == owner.end()) { error = std::string("missing ") + field; return false; }
    return admit(*it, kind, payload, grant, error);
}
inline json toJson (const programmableruntime::Grant& grant)
{
    return {{"version",grant.version},{"kind",programmableruntime::kindToWire(grant.kind)},
            {"fingerprint",grant.fingerprint},{"sessionGeneration",grant.sessionGeneration},
            {"catalogPackId",grant.catalogPackId},{"catalogProgramId",grant.catalogProgramId},
            {"nonce",grant.nonce},{"issuedAtMs",grant.issuedAtMs},{"approved",grant.approved},
            {"disk",grant.disk},{"network",grant.network},
            {"verifiedBundledCurated",grant.verifiedBundledCurated},
            {"cpuMs",grant.cpuMs},{"gpuMs",grant.gpuMs},{"memoryMiB",grant.memoryMiB},
            {"mac",grant.mac}};
}
}
