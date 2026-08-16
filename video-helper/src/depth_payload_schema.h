#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

namespace videowire
{
struct ExecutableDepthPayload
{
    std::string assetId, version, cacheKey, contentReceipt, analysisReceipt;
    std::string prefix, extension;
    int firstFrame = 0, digits = 0, width = 0, height = 0, frames = 0;
    double fps = 0.0;
};

inline bool parseDepthAttributes(std::string_view xml,
                                 std::unordered_map<std::string, std::string>& attributes)
{
    size_t cursor = 0;
    const auto skip = [&] { while (cursor < xml.size()
        && std::isspace(static_cast<unsigned char>(xml[cursor]))) ++cursor; };
    const auto take = [&](std::string_view token) {
        if (xml.substr(cursor, token.size()) != token) return false;
        cursor += token.size();
        return true;
    };
    skip();
    if (!take("<DepthAssetBinding") || (cursor < xml.size()
        && !std::isspace(static_cast<unsigned char>(xml[cursor])))) return false;
    while (true)
    {
        skip();
        if (take("/>")) break;
        const size_t start = cursor;
        while (cursor < xml.size() && (std::isalnum(static_cast<unsigned char>(xml[cursor]))
            || xml[cursor] == '_')) ++cursor;
        if (cursor == start) return false;
        const std::string name(xml.substr(start, cursor - start));
        skip(); if (!take("=")) return false; skip();
        if (cursor >= xml.size() || xml[cursor] != '"') return false;
        const size_t valueStart = ++cursor;
        while (cursor < xml.size() && xml[cursor] != '"')
            if (xml[cursor++] == '<') return false;
        if (cursor >= xml.size()) return false;
        if (!attributes.emplace(name, std::string(xml.substr(valueStart, cursor - valueStart))).second)
            return false;
        ++cursor;
    }
    skip(); return cursor == xml.size();
}

inline bool parseExecutableDepthPayload(std::string_view xml, ExecutableDepthPayload& out)
{
    std::unordered_map<std::string, std::string> a;
    if (!parseDepthAttributes(xml, a) || a.size() != 15) return false;
    const auto get = [&](const char* key) -> const std::string* { const auto i=a.find(key);
        return i == a.end() ? nullptr : &i->second; };
    const auto integer = [&](const char* key, int& value) {
        const auto* text=get(key);
        if (text == nullptr || text->empty()) return false;
        char* end=nullptr; errno=0;
        const long parsed=std::strtol(text->c_str(), &end, 10);
        if (errno || end != text->c_str()+text->size() || parsed < 0
            || parsed > std::numeric_limits<int>::max()) return false;
        value=static_cast<int>(parsed); return true; };
    const auto* id=get("depthAssetId"), *version=get("depthAssetVersion"), *state=get("state"),
        *key=get("cacheKey"), *receipt=get("contentReceipt"), *analysis=get("analysisReceipt"),
        *prefix=get("framePrefix"), *extension=get("frameExtension"), *format=get("format"), *fps=get("fps");
    if (!id || id->empty() || !version || version->empty() || !key || !receipt || !analysis
        || !prefix || prefix->empty() || !extension || *extension != ".png"
        || !state || *state != "available" || !format || *format != "r16-unorm" || !fps)
        return false;
    const auto lowercaseReceipt = [](const std::string& value) {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }); };
    const auto slash = key->find('/');
    if (slash == std::string::npos || slash != key->rfind('/')
        || key->substr(0, slash).rfind("depth-job-", 0) != 0
        || key->substr(slash + 1) != "published" || key->find("..") != std::string::npos
        || !lowercaseReceipt(*receipt) || !lowercaseReceipt(*analysis))
        return false;
    char* end=nullptr; errno=0; const double parsedFps=std::strtod(fps->c_str(), &end);
    if (errno || end != fps->c_str()+fps->size() || !std::isfinite(parsedFps) || parsedFps <= 0.0)
        return false;
    out = {}; out.assetId=*id; out.version=*version; out.cacheKey=*key;
    out.contentReceipt=*receipt; out.analysisReceipt=*analysis; out.prefix=*prefix;
    out.extension=*extension; out.fps=parsedFps;
    if (!integer("firstFrame", out.firstFrame) || !integer("frameDigits", out.digits)
        || !integer("width", out.width) || !integer("height", out.height)
        || !integer("frames", out.frames)) return false;
    return out.firstFrame > 0 && out.digits >= 1 && out.digits <= 9
        && out.width >= 1 && out.width <= 16384 && out.height >= 1 && out.height <= 16384
        && out.frames >= 1 && out.frames <= 10000000;
}

inline bool admitParkedDepthPayload (std::string_view xml)
{
    size_t cursor = 0;
    const auto skipSpace = [&]
    {
        while (cursor < xml.size() && std::isspace(static_cast<unsigned char>(xml[cursor])))
            ++cursor;
    };
    const auto consume = [&] (std::string_view token)
    {
        if (xml.substr(cursor, token.size()) != token) return false;
        cursor += token.size();
        return true;
    };

    skipSpace();
    if (! consume("<DepthAssetBinding")) return false;
    if (cursor < xml.size() && ! std::isspace(static_cast<unsigned char>(xml[cursor]))
        && xml[cursor] != '/' && xml[cursor] != '>')
        return false;

    std::unordered_map<std::string, std::string> attributes;
    while (true)
    {
        skipSpace();
        if (consume("/>")) break;
        if (consume(">"))
        {
            // No character data (including whitespace), comments, or child elements are admitted.
            if (! consume("</DepthAssetBinding>")) return false;
            break;
        }

        const size_t nameStart = cursor;
        while (cursor < xml.size()
               && (std::isalnum(static_cast<unsigned char>(xml[cursor])) || xml[cursor] == '_'))
            ++cursor;
        if (cursor == nameStart) return false;
        const std::string name(xml.substr(nameStart, cursor - nameStart));
        skipSpace();
        if (! consume("=")) return false;
        skipSpace();
        if (cursor >= xml.size() || (xml[cursor] != '"' && xml[cursor] != '\'')) return false;
        const char quote = xml[cursor++];
        const size_t valueStart = cursor;
        while (cursor < xml.size() && xml[cursor] != quote)
        {
            if (xml[cursor] == '<' || xml[cursor] == '>') return false;
            ++cursor;
        }
        if (cursor >= xml.size()) return false;
        const std::string value(xml.substr(valueStart, cursor - valueStart));
        ++cursor;
        if (! attributes.emplace(name, value).second) return false;
    }

    skipSpace();
    if (cursor != xml.size() || attributes.size() != 5) return false;
    const auto exact = [&] (const char* name, const char* expected)
    {
        const auto found = attributes.find(name);
        return found != attributes.end() && found->second == expected;
    };
    const auto id = attributes.find("depthAssetId");
    const auto version = attributes.find("depthAssetVersion");
    return id != attributes.end() && ! id->second.empty()
        && version != attributes.end() && ! version->second.empty()
        && exact("state", "available")
        && exact("contract", "parked-metadata")
        && exact("pendingExecutionSeam", "depth-consuming-operation");
}
} // namespace videowire
