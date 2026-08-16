#pragma once
// Dependency-light schema constants and exact receipt-payload framing for tracking assets.
#include <string>
#include <string_view>

namespace donutstudio::tracking {
inline constexpr std::string_view schema = "donutstudio-tracking-asset-v1";
inline constexpr std::string_view pointAssetType = "PointTrackAsset";
inline constexpr std::string_view planarAssetType = "PlanarTrackAsset";
inline constexpr std::string_view receiptDomain = "tracking-asset-receipt-v1";

enum class SampleState { valid, lost, invalid };

// Inputs must already be RFC-8259 JSON serialized with sorted object keys, no
// insignificant whitespace, finite numbers only, and ASCII escaping. SHA-256
// this byte string to obtain contentReceipt.value. Keeping hashing outside this
// header avoids adding a crypto/JSON dependency to native asset carriers.
inline std::string receiptPayload(std::string_view canonicalMetadata,
                                  std::string_view canonicalSamples) {
    std::string out;
    out.reserve(receiptDomain.size() + 1 + canonicalMetadata.size() + 1 + canonicalSamples.size());
    out.append(receiptDomain);
    out.push_back('\0');
    out.append(canonicalMetadata);
    out.push_back('\n');
    out.append(canonicalSamples);
    return out;
}
} // namespace donutstudio::tracking
