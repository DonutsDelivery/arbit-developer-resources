#pragma once

#include "sha256.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace videohelper
{
// Authority boundary: this cache defends against untrusted project/media data.
// A process already compromised under the application's OS principal can alter
// application state or race pathname opens and is outside application authority.
// Do not describe this protocol as protection against that process compromise.
struct MatteCacheBinding
{
    std::string key, receipt, version, prefix, extension;
    int firstFrame = 0, digits = 0, frames = 0;
    double fps = 0.0;
    uint64_t contentRevision = 0;
};

inline bool safeCacheKey (const std::string& key)
{
    return key.size() == 64 && std::all_of(key.begin(), key.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
inline std::string matteFrameName (const MatteCacheBinding& b, int index)
{
    std::ostringstream out; out << b.prefix << std::setw(b.digits) << std::setfill('0')
                                << b.firstFrame + index << b.extension; return out.str();
}
inline std::string hashFileBounded (const std::filesystem::path& path, uint64_t& aggregate,
                                    uint64_t perFrameLimit, uint64_t aggregateLimit)
{
    std::error_code ec;
    if (std::filesystem::is_symlink(path, ec) || !std::filesystem::is_regular_file(path, ec)) return {};
    const auto permissions = std::filesystem::status(path, ec).permissions();
    const auto writable = std::filesystem::perms::owner_write | std::filesystem::perms::group_write
        | std::filesystem::perms::others_write;
    if (ec || (permissions & writable) != std::filesystem::perms::none) return {};
    const auto bytes = std::filesystem::file_size(path, ec);
    if (ec || bytes == 0 || bytes > perFrameLimit || bytes > aggregateLimit - aggregate) return {};
    std::ifstream in(path, std::ios::binary); if (!in) return {};
    Sha256 hash; char buffer[65536]; uint64_t readBytes = 0;
    while (in) { in.read(buffer, sizeof(buffer)); const auto n=in.gcount();
        if(n>0){hash.update(buffer,(size_t)n);readBytes+=(uint64_t)n;} }
    if (!in.eof() || readBytes != bytes) return {};
    aggregate += bytes; return hash.finishHex();
}
inline std::string bindingMaterial (const MatteCacheBinding& b, const std::vector<std::string>& hashes)
{
    std::ostringstream out; out << "matte-cache-v1\n" << b.version << '\n' << b.prefix << '\n'
        << b.extension << '\n' << b.firstFrame << '\n' << b.digits << '\n' << b.frames << '\n'
        << std::setprecision(17) << b.fps << '\n';
    for (const auto& hash : hashes) out << hash << '\n';
    return out.str();
}
inline bool validateMatteCache (const std::filesystem::path& trustedRoot, MatteCacheBinding& b,
                                std::string& resolvedDirectory, std::string& error)
{
    constexpr uint64_t perFrameLimit = 256ull * 1024 * 1024;
    constexpr uint64_t aggregateLimit = 1024ull * 1024 * 1024;
    const auto safeNamePart = [] (const std::string& text, bool extension) {
        if (text.empty() || text.size() > (extension ? 12u : 128u)
            || (extension && (text.size() < 2 || text.front() != '.'))) return false;
        const auto first = extension ? 1u : 0u;
        return std::all_of(text.begin() + static_cast<std::ptrdiff_t>(first), text.end(),
            [extension] (unsigned char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
                    || (c >= 'A' && c <= 'Z') || (!extension && (c == '_' || c == '-'));
            });
    };
    if (!safeCacheKey(b.key) || b.receipt != b.key || !safeNamePart(b.prefix, false)
        || !safeNamePart(b.extension, true) || b.firstFrame < 0 || b.digits <= 0 || b.digits > 12
        || b.frames <= 0 || b.frames > 1000000
        || b.firstFrame > std::numeric_limits<int>::max() - (b.frames - 1)
        || !std::isfinite(b.fps) || b.fps <= 0.0) {
        error="invalid matte cache authority"; return false; }
    std::error_code ec;
    const auto root=std::filesystem::weakly_canonical(trustedRoot,ec); if(ec){error="matte cache root unavailable";return false;}
    const auto dir=std::filesystem::weakly_canonical(root / b.key,ec);
    const auto dirStatus = std::filesystem::status(dir, ec);
    const auto writable = std::filesystem::perms::owner_write | std::filesystem::perms::group_write
        | std::filesystem::perms::others_write;
    if(ec || dir.parent_path()!=root || std::filesystem::is_symlink(root / b.key,ec)
        || !std::filesystem::is_directory(dir,ec) || (dirStatus.permissions() & writable) != std::filesystem::perms::none)
        {error="matte cache key is not an immutable directory beneath trusted root";return false;}
    const auto manifestPath=dir / "manifest";
    const auto manifestPermissions = std::filesystem::status(manifestPath, ec).permissions();
    if(std::filesystem::is_symlink(manifestPath,ec) || !std::filesystem::is_regular_file(manifestPath,ec)
       || (manifestPermissions & writable) != std::filesystem::perms::none
       || std::filesystem::file_size(manifestPath,ec) > 80ull * 1000000ull)
       {error="matte cache manifest invalid";return false;}
    std::ifstream manifest(manifestPath, std::ios::binary);
    std::string manifestText((std::istreambuf_iterator<char>(manifest)),{});
    if(!manifest || manifestText.empty()){error="matte cache manifest missing";return false;}
    std::vector<std::string> hashes; hashes.reserve((size_t)b.frames); uint64_t aggregate=0;
    for(int i=0;i<b.frames;++i){const auto name=matteFrameName(b,i); const auto frame=dir/name;
        if(frame.parent_path()!=dir){error="invalid matte frame name";return false;}
        auto hash=hashFileBounded(frame,aggregate,perFrameLimit,aggregateLimit);
        if(hash.empty()){error="matte cache frame invalid or tampered";return false;} hashes.push_back(std::move(hash));}
    const auto material=bindingMaterial(b,hashes);
    if(sha256Text(material)!=b.receipt || manifestText!=material){error="matte cache manifest or receipt mismatch";return false;}
    resolvedDirectory=dir.string(); error.clear(); return true;
}
} // namespace videohelper
