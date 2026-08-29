// shader_dialect.h — Arbit Media Machine shader dialect shims + ISF parser (pre-work A5)
//
// Takes a raw user/agent shader source in one of three dialects and produces a
// single contract-conformant GLSL 330 core fragment shader, plus (for ISF) the
// list of generator parameters its INPUTS declare. This is the import/authoring
// front door for shader generator clips (generative-visuals-research.md §3.1-3.2,
// §4.4): Shadertoy convention is the AUTHORING dialect (what LLMs and humans write
// fluently); ISF (MIT spec) is the IMPORT format (its INPUTS map 1:1 onto
// generator params, and Vidvox's ISF-Files pack is MIT-bundleable).
//
// Pure string/JSON transform — no GL, no JUCE, no compilation. The helper feeds
// wrapToContract()'s output straight into its existing runtime GLSL compiler
// (renderer.cpp); this module never touches a GL context, so it builds and is
// golden-file tested in isolation. nlohmann/json (header-only) parses the ISF
// header. Destined for video-helper/src/.
//
// Core principle (hardened after adversarial review June 13): NEVER report ok=true
// while emitting GLSL that cannot compile. Every case that would produce
// uncompilable source — an INPUT name that is not a legal/free GLSL identifier, a
// duplicate or contract-colliding name, a shader that already owns main() on the
// Shadertoy path — is reported as a WrapDiagnostic::Error (ok=false) so the
// deferred-RPC error log (§4.5) surfaces it, instead of failing silently at GL
// compile time during integration.
//
// What it pins (the §4.4 contract):
//   - #version 330 core is forced; any user #version line is stripped
//     (preprocessor-aware: "# version" too); #extension is REJECTED, also
//     preprocessor-aware ("# extension") (risk 1).
//   - one fixed Arbit prelude (Block A/B/C uniforms + helpers + texture2D shim).
//   - entry adapters: bare void main() as-is; Shadertoy mainImage(out,in) shimmed
//     (unless the shader already defines main()); ISF body keeps its own main()
//     with gl_FragColor shimmed + auto-vars provided.
//   - the NaN/Inf output guard (risk 4) is applied where this module owns the
//     final write (the generated Shadertoy main()).

#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <set>
#include <string>
#include <vector>

namespace arbitshader
{

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
enum class Dialect { Unknown, BareGlsl, Shadertoy, Isf };

enum class InputType { Bool, Long, Float, Point2D, Color, Image, Audio, AudioFFT, Event, Unknown };

inline const char* inputTypeToWire (InputType t) noexcept
{
    switch (t)
    {
        case InputType::Bool:     return "bool";
        case InputType::Long:     return "long";
        case InputType::Float:    return "float";
        case InputType::Point2D:  return "point2D";
        case InputType::Color:    return "color";
        case InputType::Image:    return "image";
        case InputType::Audio:    return "audio";
        case InputType::AudioFFT: return "audioFFT";
        case InputType::Event:    return "event";
        case InputType::Unknown:  return "unknown";
    }
    return "unknown";
}

inline InputType inputTypeFromWire (const std::string& s) noexcept
{
    if (s == "bool")     return InputType::Bool;
    if (s == "long")     return InputType::Long;
    if (s == "float")    return InputType::Float;
    if (s == "point2D")  return InputType::Point2D;
    if (s == "color")    return InputType::Color;
    if (s == "image")    return InputType::Image;
    if (s == "audio")    return InputType::Audio;
    if (s == "audioFFT") return InputType::AudioFFT;
    if (s == "event")    return InputType::Event;
    return InputType::Unknown;
}

inline const char* inputTypeToGlsl (InputType t) noexcept
{
    switch (t)
    {
        case InputType::Bool:     return "bool";
        case InputType::Long:     return "int";
        case InputType::Float:    return "float";
        case InputType::Point2D:  return "vec2";
        case InputType::Color:    return "vec4";
        case InputType::Image:    return "sampler2D";
        case InputType::Audio:    return "sampler2D";
        case InputType::AudioFFT: return "sampler2D";
        case InputType::Event:    return "bool";
        case InputType::Unknown:  return "float";
    }
    return "float";
}

// A generator parameter (an ISF INPUT). Registered later as clip<id>/gen/<name>.
struct GenInput
{
    std::string name;
    InputType   type = InputType::Unknown;
    std::string label;

    // Scalar inputs (bool 0/1, long, float):
    double defaultScalar = 0.0;
    double minScalar = 0.0;
    double maxScalar = 1.0;
    bool   hasMin = false;
    bool   hasMax = false;

    // Vector inputs (point2D uses .xy; color uses .rgba):
    std::array<double, 4> defaultVec { { 0.0, 0.0, 0.0, 0.0 } };
    std::array<double, 4> minVec { { 0.0, 0.0, 0.0, 0.0 } };
    std::array<double, 4> maxVec { { 0.0, 0.0, 0.0, 0.0 } };
    bool   hasMinVec = false;
    bool   hasMaxVec = false;

    bool   droppedEnum = false; // a long carried VALUES/LABELS (dropped in v1)

    // IMPORTED image (M7): set for an entry synthesized from the ISF IMPORTED map
    // (vs. a user-facing INPUT). It is still an `image`-type sampler param, but its
    // texture comes from this header-declared PATH (resolved by the host), not from
    // a user control. Empty for a normal INPUT.
    std::string importedPath;
};

// One ISF render PASS (multipass support, M7). A pass renders the shader body
// (branching on PASSINDEX) into its TARGET buffer; later passes / the final pass
// sample earlier targets by name. An empty target == the visible output pass.
// PERSISTENT targets survive across frames (double-buffered: a pass reads the
// previous frame's content while writing this frame's) — the ISF feedback path.
struct IsfPass
{
    std::string target;        // TARGET buffer name ("" = render to the output)
    bool        persistent = false;
};

struct IsfHeader
{
    bool valid = false;
    std::string isfVersion;
    std::string description;
    std::string credit;
    std::vector<std::string> categories;
    std::vector<GenInput> inputs;
    std::vector<IsfPass> passes;   // multipass (M7); empty = single implicit output pass
    int  passCount = 0;
    bool persistent = false;
    bool importsImages = false;
    bool singlePass = true;
    std::string parseError;

    // The character index just past the header comment's closing "*/" (set when
    // valid), so the body can be sliced without re-finding the terminator.
    size_t bodyStart = 0;
};

struct WrapDiagnostic
{
    enum Level { Error, Warning };
    Level level = Error;
    std::string message;
};

struct WrapResult
{
    bool ok = false;
    Dialect dialect = Dialect::Unknown;
    std::string glsl;
    std::vector<GenInput> params;
    std::vector<IsfPass> passes;     // ISF multipass (M7); empty = single implicit pass
    bool singlePassOk = true;
    bool outputGuarded = false;
    std::vector<WrapDiagnostic> diagnostics;

    bool hasError() const
    {
        for (const auto& d : diagnostics) if (d.level == WrapDiagnostic::Error) return true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// Source scanning helpers (comment/string aware)
// ---------------------------------------------------------------------------
namespace detail
{
    inline bool isIdentChar (char c)
    {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    }

    // True if `name` occurs in `src` (outside comments) as a whole identifier
    // followed — after optional whitespace — by '(' : i.e. a function reference.
    // This is how entry points are detected, so `mainImageCached`, `mymainImage`
    // and `mainImageScale = 1.0` do NOT register as `mainImage`/`main`.
    inline bool hasFunctionCode (const std::string& src, const std::string& name)
    {
        const size_t n = src.size();
        for (size_t i = 0; i < n; )
        {
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '/')
            { while (i < n && src[i] != '\n') ++i; continue; }
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '*')
            { i += 2; while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) ++i; i += 2; continue; }

            if (src.compare (i, name.size(), name) == 0)
            {
                const bool boundaryBefore = (i == 0) || !isIdentChar (src[i - 1]);
                const size_t after = i + name.size();
                const bool boundaryAfter = (after >= n) || !isIdentChar (src[after]);
                if (boundaryBefore && boundaryAfter)
                {
                    size_t j = after;
                    while (j < n && std::isspace (static_cast<unsigned char> (src[j]))) ++j;
                    if (j < n && src[j] == '(') return true;
                }
            }
            ++i;
        }
        return false;
    }

    // True when source code declares a function with the requested return type,
    // name, and first parameter type. Calls are deliberately not matched.
    inline bool hasFunctionDefinitionCode (const std::string& src,
                                           const std::string& returnType,
                                           const std::string& name,
                                           const std::string& firstParamType)
    {
        std::vector<std::string> tokens;
        const size_t n = src.size();
        for (size_t i = 0; i < n; )
        {
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '/')
            { while (i < n && src[i] != '\n') ++i; continue; }
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '*')
            { i += 2; while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) ++i; i += 2; continue; }
            if ((src[i] >= 'A' && src[i] <= 'Z') || (src[i] >= 'a' && src[i] <= 'z') || src[i] == '_')
            {
                const size_t start = i++;
                while (i < n && isIdentChar (src[i])) ++i;
                tokens.emplace_back (src.substr (start, i - start));
            }
            else
            {
                if (src[i] == '(') tokens.emplace_back ("(");
                ++i;
            }
        }
        for (size_t i = 0; i + 3 < tokens.size(); ++i)
            if (tokens[i] == returnType && tokens[i + 1] == name
                && tokens[i + 2] == "(" && tokens[i + 3] == firstParamType)
                return true;
        return false;
    }

    // True if a preprocessor directive `#<ws>?keyword` appears outside comments.
    // Preprocessor-aware: matches "#extension", "# extension", "#\textension"
    // (GLSL/C permit whitespace between '#' and the directive keyword), with a
    // word boundary after the keyword.
    inline bool hasPreprocessorDirective (const std::string& src, const std::string& keyword)
    {
        const size_t n = src.size();
        for (size_t i = 0; i < n; )
        {
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '/')
            { while (i < n && src[i] != '\n') ++i; continue; }
            if (i + 1 < n && src[i] == '/' && src[i + 1] == '*')
            { i += 2; while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) ++i; i += 2; continue; }

            if (src[i] == '#')
            {
                size_t j = i + 1;
                while (j < n && (src[j] == ' ' || src[j] == '\t')) ++j;
                if (src.compare (j, keyword.size(), keyword) == 0)
                {
                    const size_t after = j + keyword.size();
                    if (after >= n || !isIdentChar (src[after])) return true;
                }
            }
            ++i;
        }
        return false;
    }

    // Find the first balanced {...} JSON object in s[from, end), honoring string
    // literals and escapes. Returns the object substring, or "" if none. `end`
    // bounds the search so a comment's JSON can't reach into shader code.
    inline std::string extractJsonObject (const std::string& s, size_t from, size_t end)
    {
        if (end > s.size()) end = s.size();
        size_t start = s.find ('{', from);
        if (start == std::string::npos || start >= end) return {};
        int depth = 0;
        bool inStr = false, esc = false;
        for (size_t i = start; i < end; ++i)
        {
            const char c = s[i];
            if (inStr)
            {
                if (esc)            esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"')  inStr = false;
                continue;
            }
            if (c == '"')      inStr = true;
            else if (c == '{') ++depth;
            else if (c == '}') { if (--depth == 0) return s.substr (start, i - start + 1); }
        }
        return {};
    }

    // Locate the ISF header: a LEADING block comment (only whitespace before it)
    // whose first non-whitespace content is '{' (the ISF spec mandates `/*{ ... }*/`).
    // On success returns true and sets jsonOut to the balanced JSON object and
    // bodyStartOut to just past the real closing "*/" (found AFTER the JSON object,
    // so a "*/" inside a JSON string value does not terminate it early).
    inline bool locateIsfHeader (const std::string& src, std::string& jsonOut, size_t& bodyStartOut)
    {
        size_t open = src.find ("/*");
        if (open == std::string::npos) return false;
        for (size_t k = 0; k < open; ++k)
            if (!std::isspace (static_cast<unsigned char> (src[k]))) return false; // code precedes comment
        // First non-whitespace inside the comment must be '{'.
        size_t p = open + 2;
        while (p < src.size() && std::isspace (static_cast<unsigned char> (src[p]))) ++p;
        if (p >= src.size() || src[p] != '{') return false;

        const std::string obj = extractJsonObject (src, p, src.size());
        if (obj.empty()) return false;
        // Real comment terminator is the first "*/" at/after the object's end.
        const size_t objEnd = p + obj.size();
        const size_t close = src.find ("*/", objEnd);
        if (close == std::string::npos) return false;

        jsonOut = obj;
        bodyStartOut = close + 2;
        return true;
    }

    // Remove every line whose first non-space content is a "#<ws>?version" directive.
    inline std::string stripVersionLines (const std::string& src)
    {
        std::string out;
        out.reserve (src.size());
        size_t i = 0;
        while (i < src.size())
        {
            size_t eol = src.find ('\n', i);
            if (eol == std::string::npos) eol = src.size();
            size_t j = i;
            while (j < eol && std::isspace (static_cast<unsigned char> (src[j]))) ++j;
            bool isVersion = false;
            if (j < eol && src[j] == '#')
            {
                size_t k = j + 1;
                while (k < eol && (src[k] == ' ' || src[k] == '\t')) ++k;
                isVersion = src.compare (k, 7, "version") == 0;
            }
            if (!isVersion) out.append (src, i, eol - i + (eol < src.size() ? 1 : 0));
            i = eol + 1;
        }
        return out;
    }

    // A valid, unreserved GLSL identifier for a synthesized uniform.
    inline bool isValidGlslIdentifier (const std::string& s)
    {
        if (s.empty()) return false;
        if (!((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') || s[0] == '_')) return false;
        for (char c : s) if (!isIdentChar (c)) return false;
        if (s.rfind ("gl_", 0) == 0) return false; // gl_ prefix is reserved
        return true;
    }

    // Names the prelude/contract already occupies, plus GLSL keywords/builtins the
    // contract relies on. A synthesized ISF uniform with any of these names would
    // redefine an existing identifier.
    inline bool isReservedName (const std::string& n)
    {
        static const char* const kReserved[] = {
            // Block A/B/C uniforms + output
            "uResolution","uTime","uTimeDelta","uFrame","uBeat","uBPM","uBeatPhase",
            "uBarPhase","uBeatsPerBar","uClipBeat","uClipLength","uPlaying",
            "uRMS","uPeak","uOnset","uOnsetAge","uAudioBands","uAudioBands2D",
            "uNotes","uLinks","uNoteCount","uLinkCount","uRootFreq",
            "uScoreHistoryBeats","uScoreLookaheadBeats","fragColor",
            // Block D camera uniforms (P5)
            "uCamPos","uCamTarget","uCamUp","uCamFov","uCamNear","uCamFar",
            // helpers
            "arbitBand","hsv2rgb","beatPulse","arbitGuard","arbitCameraRay",
            "arbitDisplayFragCoord","arbitTextureCoord",
            // Block C accessors (M5)
            "arbitNoteTexel","arbitNoteMidi","arbitNoteVel","arbitNoteAge","arbitNoteRemain",
            "arbitNoteLength","arbitNoteStartOffset","arbitNoteEndOffset",
            "arbitNoteFreq","arbitNoteCents","arbitNoteTrack","arbitNoteIsRoot","arbitNotePrimesLo",
            "arbitNotePrimesHi","arbitNoteMasterRow","arbitNotePitch","arbitNoteBend",
            "arbitNoteActive","arbitLink","arbitLinkRatio",
            "arbitLinkTexel","arbitNote","lattice2","ArbitNote","ArbitLink",
            // ISF auto-var macro targets / Shadertoy alias targets
            "TIME","TIMEDELTA","FRAMEINDEX","RENDERSIZE","DATE","isf_FragNormCoord",
            "IMG_PIXEL","IMG_NORM_PIXEL","IMG_THIS_PIXEL","IMG_SIZE",
            "iTime","iTimeDelta","iFrame","iResolution","iChannel0","iMouse",
            // builtins the contract relies on
            "texture","texture2D","main",
            // GLSL keywords / types most likely to be used as a name
            "void","float","int","uint","bool","true","false","const","uniform",
            "attribute","varying","in","out","inout","layout","struct","precision",
            "lowp","mediump","highp","if","else","for","while","do","switch","case",
            "default","break","continue","return","discard","invariant",
            "vec2","vec3","vec4","ivec2","ivec3","ivec4","uvec2","uvec3","uvec4",
            "bvec2","bvec3","bvec4","mat2","mat3","mat4","sampler1D","sampler2D",
            "sampler3D","samplerCube",
        };
        for (const char* r : kReserved) if (n == r) return true;
        return false;
    }
} // namespace detail

// ---------------------------------------------------------------------------
// Dialect detection
// ---------------------------------------------------------------------------
inline Dialect detectDialect (const std::string& src)
{
    std::string json; size_t bodyStart = 0;
    if (detail::locateIsfHeader (src, json, bodyStart))
    {
        try
        {
            auto j = nlohmann::json::parse (json);
            if (j.is_object()
                && (j.contains ("ISFVSN") || j.contains ("INPUTS") || j.contains ("PASSES")
                    || j.contains ("IMPORTED") || j.contains ("CATEGORIES") || j.contains ("DESCRIPTION")))
                return Dialect::Isf;
        }
        catch (...) { /* not valid JSON -> not ISF */ }
    }
    if (detail::hasFunctionCode (src, "mainImage")) return Dialect::Shadertoy;
    if (detail::hasFunctionCode (src, "main"))      return Dialect::BareGlsl;
    return Dialect::Unknown;
}

// ---------------------------------------------------------------------------
// ISF header parsing
// ---------------------------------------------------------------------------
inline IsfHeader parseIsfHeader (const std::string& src)
{
    IsfHeader h;
    std::string json; size_t bodyStart = 0;
    if (!detail::locateIsfHeader (src, json, bodyStart))
    { h.parseError = "no ISF /*{ ... }*/ header"; return h; }

    nlohmann::json j;
    try { j = nlohmann::json::parse (json); }
    catch (const std::exception& e) { h.parseError = std::string ("JSON parse error: ") + e.what(); return h; }
    if (!j.is_object()) { h.parseError = "ISF header is not a JSON object"; return h; }

    h.bodyStart = bodyStart;
    if (j.contains ("ISFVSN") && j["ISFVSN"].is_string())      h.isfVersion = j["ISFVSN"].get<std::string>();
    if (j.contains ("DESCRIPTION") && j["DESCRIPTION"].is_string()) h.description = j["DESCRIPTION"].get<std::string>();
    if (j.contains ("CREDIT") && j["CREDIT"].is_string())      h.credit = j["CREDIT"].get<std::string>();
    if (j.contains ("CATEGORIES") && j["CATEGORIES"].is_array())
        for (const auto& c : j["CATEGORIES"]) if (c.is_string()) h.categories.push_back (c.get<std::string>());
    h.importsImages = j.contains ("IMPORTED") && !j["IMPORTED"].empty();

    if (j.contains ("PASSES") && j["PASSES"].is_array())
    {
        h.passCount = static_cast<int> (j["PASSES"].size());
        for (const auto& p : j["PASSES"])
        {
            IsfPass pass;
            if (p.is_object())
            {
                if (p.contains ("TARGET") && p["TARGET"].is_string())
                    pass.target = p["TARGET"].get<std::string>();
                if (p.contains ("PERSISTENT"))
                {
                    const auto& pv = p["PERSISTENT"];
                    if ((pv.is_boolean() && pv.get<bool>())
                        || (pv.is_string() && !pv.get<std::string>().empty())
                        || (pv.is_number() && pv.get<double>() != 0.0))
                        pass.persistent = true;
                }
            }
            if (pass.persistent) h.persistent = true;
            h.passes.push_back (std::move (pass));
        }
    }
    // singlePass is now informational only (multipass is supported): true when a
    // single implicit output pass with no persistent state — i.e. no extra work.
    h.singlePass = (h.passCount <= 1) && !h.persistent;

    auto readScalar = [] (const nlohmann::json& v, double fallback) -> double
    {
        if (v.is_number())  return v.get<double>();
        if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
        return fallback;
    };
    auto readVec = [] (const nlohmann::json& v, std::array<double, 4>& out)
    {
        if (!v.is_array()) return;
        for (size_t i = 0; i < v.size() && i < 4; ++i)
            if (v[i].is_number()) out[i] = v[i].get<double>();
    };

    if (j.contains ("INPUTS") && j["INPUTS"].is_array())
    {
        for (const auto& in : j["INPUTS"])
        {
            if (!in.is_object()) continue;
            GenInput gi;
            if (in.contains ("NAME") && in["NAME"].is_string()) gi.name = in["NAME"].get<std::string>();
            if (gi.name.empty()) continue;
            std::string typeStr = in.contains ("TYPE") && in["TYPE"].is_string() ? in["TYPE"].get<std::string>() : "";
            gi.type = inputTypeFromWire (typeStr);
            gi.label = in.contains ("LABEL") && in["LABEL"].is_string() ? in["LABEL"].get<std::string>() : gi.name;

            const bool isVec = (gi.type == InputType::Point2D || gi.type == InputType::Color);
            if (gi.type == InputType::Color) gi.defaultVec[3] = 1.0; // opaque unless overridden

            if (in.contains ("DEFAULT"))
            {
                if (isVec) readVec (in["DEFAULT"], gi.defaultVec);
                else       gi.defaultScalar = readScalar (in["DEFAULT"], 0.0);
            }
            // MIN/MAX: scalar types use scalar fields; vec types use the vec fields.
            // Never fabricate a scalar range from a vec MIN/MAX (the array would be
            // silently dropped while hasMin/hasMax claimed a 0..1 range existed).
            if (in.contains ("MIN"))
            {
                if (isVec) { readVec (in["MIN"], gi.minVec); gi.hasMinVec = true; }
                else       { gi.minScalar = readScalar (in["MIN"], gi.minScalar); gi.hasMin = true; }
            }
            if (in.contains ("MAX"))
            {
                if (isVec) { readVec (in["MAX"], gi.maxVec); gi.hasMaxVec = true; }
                else       { gi.maxScalar = readScalar (in["MAX"], gi.maxScalar); gi.hasMax = true; }
            }
            if (gi.type == InputType::Long && (in.contains ("VALUES") || in.contains ("LABELS")))
                gi.droppedEnum = true;

            h.inputs.push_back (std::move (gi));
        }
    }

    // IMPORTED images (M7): the ISF spec's `"IMPORTED": { "<name>": { "PATH": ... } }`
    // map (a value may also be a bare path string). Each becomes an `image`-type
    // sampler param carrying its PATH — declared in the prelude and bound exactly
    // like an image INPUT, except the texture comes from the header path (resolved
    // by the host) rather than a user control. Appended after INPUTS so a name that
    // collides with an INPUT surfaces as a duplicate at validation.
    if (j.contains ("IMPORTED") && j["IMPORTED"].is_object())
    {
        for (const auto& it : j["IMPORTED"].items())
        {
            GenInput gi;
            gi.name = it.key();
            if (gi.name.empty()) continue;
            gi.type  = InputType::Image;
            gi.label = gi.name;
            const auto& v = it.value();
            if (v.is_string())
                gi.importedPath = v.get<std::string>();
            else if (v.is_object() && v.contains ("PATH") && v["PATH"].is_string())
                gi.importedPath = v["PATH"].get<std::string>();
            h.inputs.push_back (std::move (gi));
        }
    }

    h.valid = true;
    return h;
}

// ---------------------------------------------------------------------------
// The fixed Arbit prelude (§4.1-4.4).
// ---------------------------------------------------------------------------
inline std::string arbitPrelude (Dialect dialect, const IsfHeader* isf, bool includeHsv2Rgb)
{
    std::string p;
    p += "#version 330 core\n";
    p += "// --- Arbit uniform contract (auto-prepended) ---\n";
    p += "uniform vec2  uResolution;\n";
    p += "uniform float uTime;\n";
    p += "uniform float uTimeDelta;\n";
    p += "uniform int   uFrame;\n";
    p += "uniform float uBeat;\n";
    p += "uniform float uBPM;\n";
    p += "uniform float uBeatPhase;\n";
    p += "uniform float uBarPhase;\n";
    p += "uniform float uBeatsPerBar;\n";
    p += "uniform float uClipBeat;\n";
    p += "uniform float uClipLength;\n";
    p += "uniform int   uPlaying;\n";
    p += "uniform float uRMS;\n";
    p += "uniform float uPeak;\n";
    p += "uniform float uOnset;\n";
    p += "uniform float uOnsetAge;\n";
    p += "uniform sampler1D uAudioBands;\n";
    p += "uniform sampler2D uAudioBands2D;   // 64x1 alias for ISF audioFFT / Shadertoy iChannel0\n";
    p += "uniform sampler2D uNotes;          // 128 rows x 4 RGBA32F texels\n";
    p += "uniform sampler2D uLinks;          // 256 x 1\n";
    p += "uniform int   uNoteCount;\n";
    p += "uniform int   uLinkCount;\n";
    p += "uniform float uRootFreq;          // Block C: song root frequency in Hz\n";
    p += "uniform float uScoreHistoryBeats; // Block C: visible beats behind playhead\n";
    p += "uniform float uScoreLookaheadBeats; // Block C: visible beats ahead\n";
    // --- Block D: camera (3D / SDF raymarch, P5) ---
    // Contract uniforms readable by ANY shader; a Raymarch generator clip ships
    // uCamPos/uCamTarget/uCamFov as genParams so the camera automates + mod-routes
    // like any other param. The helper sets them per-frame (shader_generator
    // render); unused ones optimise out (loc -1, skipped). uCamUp/uCamNear/uCamFar
    // use stable defaults unless a clip overrides them.
    p += "uniform vec3  uCamPos;            // Block D: eye position (default 0,0,5)\n";
    p += "uniform vec3  uCamTarget;         // look-at point (default 0,0,0)\n";
    p += "uniform vec3  uCamUp;             // up vector (default 0,1,0)\n";
    p += "uniform float uCamFov;            // vertical FOV in degrees (default 45)\n";
    p += "uniform float uCamNear;           // near plane (default 0.05)\n";
    p += "uniform float uCamFar;            // far plane / max march distance (default 60)\n";
    p += "// Block D depth-MRT reserved: v1 raymarch outputs only fragColor and is\n";
    p += "// ordered by zOrder; per-fragment depth output is a future slice.\n";
    p += "out vec4 fragColor;\n";
    p += "#define texture2D texture\n";
    p += "#define gl_FragColor fragColor   // legacy ES/120 output shim (330 core removed gl_FragColor)\n";
    p += "float arbitBand(float b){ return texture(uAudioBands, clamp((b+0.5)/64.0,0.0,1.0)).r; }\n";
    if (includeHsv2Rgb)
        p += "vec3 hsv2rgb(vec3 c){ vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);"
             " vec3 q=abs(fract(c.xxx+K.xyz)*6.0-K.www);"
             " return c.z*mix(K.xxx,clamp(q-K.xxx,0.0,1.0),c.y); }\n";
    p += "float beatPulse(float width){ float d=min(uBeatPhase,1.0-uBeatPhase);"
         " return 1.0-smoothstep(0.0,max(width,1e-4),d); }\n";
    p += "vec4 arbitGuard(vec4 c){ for(int i=0;i<4;i++){ if(isnan(c[i])||isinf(c[i])) c[i]=0.0; } return clamp(c,0.0,1.0); }\n";
    // Generator targets enter a compositor whose row zero is the displayed top
    // row. Present user shaders with the usual bottom-left GLSL/Shadertoy
    // coordinate system despite that storage convention. Keep texture helpers
    // in displayed-image space as well, so ISF image inputs stay upright.
    p += "vec4 arbitDisplayFragCoord(){ return vec4(gl_FragCoord.x,uResolution.y-gl_FragCoord.y,gl_FragCoord.zw); }\n";
    p += "vec2 arbitTextureCoord(vec2 q){ return vec2(q.x/uResolution.x,1.0-q.y/uResolution.y); }\n";
    // Block C symbolic accessors (M5): read the packed score. uNotes is a
    // 4-texel x 128-row RGBA32F texture (texelFetch by [texel,row]); uLinks is
    // 256x1. Loop `for(int i=0;i<uNoteCount;i++)` — holes (freed rows) read back
    // velocity 0 and gate to nothing via arbitNoteActive.
    p += "vec4  arbitNoteTexel(int row,int texel){ return texelFetch(uNotes, ivec2(texel,row), 0); }\n";
    p += "float arbitNoteMidi(int row){ return arbitNoteTexel(row,0).r; }\n";        // authored MIDI note (float)
    p += "float arbitNoteVel(int row){ return arbitNoteTexel(row,0).g; }\n";         // velocity 0..1
    p += "float arbitNoteAge(int row){ return arbitNoteTexel(row,0).b; }\n";         // beats since onset (<0 = upcoming)
    p += "float arbitNoteRemain(int row){ return arbitNoteTexel(row,0).a; }\n";      // beats until note end
    p += "float arbitNoteLength(int row){ return arbitNoteAge(row)+arbitNoteRemain(row); }\n";
    p += "float arbitNoteStartOffset(int row){ return -arbitNoteAge(row); }\n";
    p += "float arbitNoteEndOffset(int row){ return arbitNoteRemain(row); }\n";
    p += "float arbitNoteFreq(int row){ return arbitNoteTexel(row,1).r; }\n";        // frame-resolved Hz, including pitch bend
    p += "float arbitNoteCents(int row){ return arbitNoteTexel(row,1).g; }\n";       // frame-resolved cents from song root
    p += "float arbitNoteTrack(int row){ return arbitNoteTexel(row,1).b; }\n";       // track id
    p += "float arbitNoteIsRoot(int row){ return arbitNoteTexel(row,1).a; }\n";      // 1 = harmonic root
    p += "vec4  arbitNotePrimesLo(int row){ return arbitNoteTexel(row,2); }\n";      // exponents of 2,3,5,7
    p += "vec4  arbitNotePrimesHi(int row){ return arbitNoteTexel(row,3); }\n";      // 11,13,masterRow,bendSemitones
    p += "float arbitNoteMasterRow(int row){ return arbitNoteTexel(row,3).b; }\n";   // link master's row (-1 = none)
    p += "float arbitNoteBend(int row){ return arbitNoteTexel(row,3).a; }\n";        // frame-resolved pitch bend in semitones
    p += "float arbitNotePitch(int row){ return 69.0+12.0*log2(max(arbitNoteFreq(row),0.001)/440.0); }\n";
    p += "float arbitNoteActive(int row){ return (arbitNoteVel(row)>0.0 && arbitNoteAge(row)>=0.0 && arbitNoteRemain(row)>0.0) ? 1.0 : 0.0; }\n";
    p += "vec4  arbitLinkTexel(int edge){ return texelFetch(uLinks, ivec2(edge,0), 0); }\n";   // (slaveRow, masterRow, num, den)
    p += "float arbitLinkRatio(int edge){ vec4 l=arbitLinkTexel(edge); return l.w!=0.0 ? l.z/l.w : 1.0; }\n";
    // Struct view (M5 house shaders — Harmonic Strings / lattice_orbit / etc.):
    // wrap the packed texels so a shader reads note.t0.x directly. Field map
    // matches the flat accessors above — t0=(midi,vel,age,remain),
    // t1=(bentFreqHz,bentCents,track,isRoot), t2=primesLo(2,3,5,7),
    // t3=primesHi(11,13,masterRow,bendSemitones); link.t0=(slaveRow,masterRow,num,den).
    p += "struct ArbitNote { vec4 t0; vec4 t1; vec4 t2; vec4 t3; };\n";
    p += "struct ArbitLink { vec4 t0; };\n";
    p += "ArbitNote arbitNote(int row){ int r=clamp(row,0,127); ArbitNote n;"
         " n.t0=arbitNoteTexel(r,0); n.t1=arbitNoteTexel(r,1);"
         " n.t2=arbitNoteTexel(r,2); n.t3=arbitNoteTexel(r,3); return n; }\n";
    p += "ArbitLink arbitLink(int edge){ ArbitLink l; l.t0=arbitLinkTexel(clamp(edge,0,255)); return l; }\n";
    // Syntonic 2D lattice basis (e3=3-limit exponent, e5=5-limit) for note layout.
    p += "vec2 lattice2(float e3,float e5){ return vec2(e3*0.8660254 + e5*0.35, e5*0.72); }\n";
    // Block D camera helper (P5): primary ray for a fragCoord, aspect-correct
    // (divide by uResolution.y), look-at basis from uCamPos -> uCamTarget with
    // uCamUp, perspective from uCamFov. ro = eye, rd = normalised direction.
    // arbitDisplayFragCoord already maps the compositor's storage orientation to
    // the usual bottom-left shader coordinates, so positive uv.y is world-up.
    p += "void arbitCameraRay(vec2 fragCoord, out vec3 ro, out vec3 rd){"
         " vec2 uv=(fragCoord-0.5*uResolution)/uResolution.y;"
         " vec3 f=normalize(uCamTarget-uCamPos);"
         " vec3 r=normalize(cross(f,uCamUp));"
         " vec3 u=cross(r,f);"
         " float z=1.0/tan(radians(uCamFov)*0.5);"
         " ro=uCamPos; rd=normalize(uv.x*r+uv.y*u+z*f); }\n";

    if (dialect == Dialect::Shadertoy)
    {
        p += "// --- Shadertoy aliases ---\n";
        p += "#define iTime uTime\n";
        p += "#define iTimeDelta uTimeDelta\n";
        p += "#define iFrame uFrame\n";
        p += "#define iResolution vec3(uResolution, 1.0)\n";
        p += "#define iChannel0 uAudioBands2D\n";
        p += "#define iMouse vec4(0.0)\n";
    }
    else if (dialect == Dialect::Isf)
    {
        p += "// --- ISF auto-variables ---\n";
        p += "#define TIME uTime\n";
        p += "#define TIMEDELTA uTimeDelta\n";
        p += "#define FRAMEINDEX uFrame\n";
        p += "#define RENDERSIZE uResolution\n";
        p += "// Deterministic ISF date: fixed project epoch with timeline seconds in .w.\n";
        p += "#define DATE vec4(1970.0, 1.0, 1.0, uTime)\n";
        p += "#define isf_FragNormCoord (gl_FragCoord.xy / uResolution)\n";
        p += "#define IMG_PIXEL(img, q) texture(img, arbitTextureCoord(q))\n";
        p += "#define IMG_NORM_PIXEL(img, q) texture(img, vec2((q).x, 1.0-(q).y))\n";
        p += "#define IMG_THIS_PIXEL(img) texture(img, arbitTextureCoord(gl_FragCoord.xy))\n";
        p += "#define IMG_SIZE(img) uResolution\n";
        if (isf != nullptr)
        {
            p += "// --- ISF INPUTS ---\n";
            for (const auto& gi : isf->inputs)
            {
                p += "uniform ";
                p += inputTypeToGlsl (gi.type);
                p += ' ';
                p += gi.name;
                p += ";\n";
            }
            // ISF multipass (M7): PASSINDEX selects the current pass; each named
            // TARGET is a sampler2D the body reads (a PERSISTENT target reads the
            // previous frame). Declared once per distinct target name.
            if (isf->passCount > 1 || isf->persistent)
            {
                p += "uniform int PASSINDEX;\n";
                std::set<std::string> declared;
                for (const auto& ps : isf->passes)
                    if (! ps.target.empty() && declared.insert (ps.target).second)
                        p += "uniform sampler2D " + ps.target + ";\n";
            }
            else
            {
                p += "#define PASSINDEX 0\n";
            }
        }
    }
    // Apply the display-space coordinate contract to bare GLSL, ISF, and
    // Shadertoy shaders, including sources that provide their own main(). The
    // helper above was parsed before this macro and therefore still sees the
    // backend's native fragment coordinate.
    p += "#define gl_FragCoord arbitDisplayFragCoord()\n";
    p += "// --- user shader ---\n";
    return p;
}

// ---------------------------------------------------------------------------
// wrapToContract — the front door.
// ---------------------------------------------------------------------------
inline WrapResult wrapToContract (const std::string& src)
{
    WrapResult r;
    r.dialect = detectDialect (src);

    if (r.dialect == Dialect::Unknown)
    {
        r.diagnostics.push_back ({ WrapDiagnostic::Error,
            "no recognizable entry point: expected a bare `void main()`, a Shadertoy "
            "`void mainImage(out vec4, in vec2)`, or a leading ISF `/*{ ... }*/` header" });
        return r;
    }

    if (detail::hasPreprocessorDirective (src, "extension"))
    {
        r.diagnostics.push_back ({ WrapDiagnostic::Error,
            "`#extension` directives are not permitted; the compile target is pinned to "
            "`#version 330 core`" });
        return r;
    }

    IsfHeader isf;
    if (r.dialect == Dialect::Isf)
    {
        isf = parseIsfHeader (src);
        if (!isf.valid)
        {
            r.diagnostics.push_back ({ WrapDiagnostic::Error,
                std::string ("ISF header parse failed: ") + isf.parseError });
            return r;
        }
        r.params = isf.inputs;
        r.passes = isf.passes;
        r.singlePassOk = isf.singlePass;
        // Multipass / PERSISTENT ISF is now supported (M7): the body runs once per
        // PASS (branching on PASSINDEX), each TARGET is its own buffer later passes
        // sample, PERSISTENT targets are double-buffered across frames. Only the
        // common case is covered — all passes render at the output resolution
        // (per-pass WIDTH/HEIGHT expressions are evaluated as the render size).
        for (const auto& ps : isf.passes)
            if (! ps.target.empty() && ! detail::isValidGlslIdentifier (ps.target))
                r.diagnostics.push_back ({ WrapDiagnostic::Error,
                    "ISF PASS TARGET '" + ps.target + "' is not a valid GLSL identifier "
                    "(it becomes a sampler2D uniform name)" });
        if (isf.passCount > 1)
            r.diagnostics.push_back ({ WrapDiagnostic::Warning,
                "multipass ISF: all passes render at the output resolution (per-pass "
                "WIDTH/HEIGHT expressions are not evaluated)" });
        if (isf.importsImages)
            r.diagnostics.push_back ({ WrapDiagnostic::Warning,
                "ISF IMPORTED images are bound as image samplers from their declared PATHs "
                "(resolved by the host); an unresolved/undecodable path falls back to black" });

        // Validate INPUT names BEFORE we synthesize uniforms from them. Anything
        // that would produce uncompilable / colliding GLSL is an Error (ok=false),
        // never a silent bad uniform.
        std::set<std::string> seen;
        for (const auto& gi : isf.inputs)
        {
            if (!detail::isValidGlslIdentifier (gi.name))
                r.diagnostics.push_back ({ WrapDiagnostic::Error,
                    "ISF INPUT name '" + gi.name + "' is not a valid GLSL identifier "
                    "(must match [A-Za-z_][A-Za-z0-9_]* and not start with gl_)" });
            else if (detail::isReservedName (gi.name))
                r.diagnostics.push_back ({ WrapDiagnostic::Error,
                    "ISF INPUT name '" + gi.name + "' collides with a reserved Arbit/GLSL "
                    "identifier; rename the input to import this shader" });
            if (!seen.insert (gi.name).second)
                r.diagnostics.push_back ({ WrapDiagnostic::Error,
                    "duplicate ISF INPUT name '" + gi.name + "'" });
            if (gi.droppedEnum)
                r.diagnostics.push_back ({ WrapDiagnostic::Warning,
                    "long INPUT '" + gi.name + "' carries VALUES/LABELS enum metadata, "
                    "dropped in v1 (treated as a plain int range)" });
        }
        if (r.hasError()) return r;
    }

    // Body: strip user #version lines; for ISF, also drop the header comment.
    std::string body = src;
    if (r.dialect == Dialect::Isf && isf.bodyStart <= body.size())
        body = body.substr (isf.bodyStart);
    body = detail::stripVersionLines (body);

    const bool includeHsv2Rgb = ! detail::hasFunctionDefinitionCode (
        body, "vec3", "hsv2rgb", "vec3");
    std::string out = arbitPrelude (
        r.dialect, r.dialect == Dialect::Isf ? &isf : nullptr, includeHsv2Rgb);
    out += body;
    if (!out.empty() && out.back() != '\n') out += '\n';

    const bool bodyOwnsMain = detail::hasFunctionCode (body, "main");

    if (r.dialect == Dialect::Shadertoy && !bodyOwnsMain)
    {
        // We own the final write, so NaN-guard it (risk 4).
        out += "// --- Arbit Shadertoy entry adapter ---\n";
        out += "void main(){ vec4 c = vec4(0.0); mainImage(c, gl_FragCoord.xy); fragColor = arbitGuard(c); }\n";
        r.outputGuarded = true;
    }
    else
    {
        // bare/ISF own main(), or a Shadertoy paste that already defines main():
        // we cannot wrap the output write, so it is not auto-guarded.
        r.outputGuarded = false;
        if (r.dialect == Dialect::Shadertoy && bodyOwnsMain)
            r.diagnostics.push_back ({ WrapDiagnostic::Warning,
                "shader defines its own main() — using it as the entry point; the Shadertoy "
                "adapter is skipped and the output is not NaN/Inf-guarded" });
        else
            r.diagnostics.push_back ({ WrapDiagnostic::Warning,
                r.dialect == Dialect::Isf
                    ? "ISF shader owns main(); its gl_FragColor output is not NaN/Inf-guarded "
                      "(driver TDR is the backstop)"
                    : "bare void main() output is not NaN/Inf-guarded — author should clamp/guard "
                      "fragColor explicitly (driver TDR is the backstop)" });
    }

    r.glsl = std::move (out);
    r.ok = !r.hasError();
    return r;
}

} // namespace arbitshader
