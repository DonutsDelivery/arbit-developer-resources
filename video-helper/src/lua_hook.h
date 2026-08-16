#pragma once
//
// Per-frame Lua hook (Media Machine M8 — "agent-programmable media machine").
//
// A single user/agent-authored Lua script drives render-graph clip params as an
// arbitrary function of the frame's clock, the mix audio, and a coarse score
// summary. It is the *general case* of the mod matrix: where a routing maps one
// source through one curve to one destination, the hook is a Turing-complete
// per-frame function returning any set of param overrides.
//
//   -- the script defines a global `frame(ctx)` returning {paramId = value, ...}
//   function frame(ctx)
//     return { ["clip1/source/opacity"] = 0.5 + 0.5 * math.sin(ctx.beat * math.pi) }
//   end
//
// The returned param ids use the same `clip<id>/<node>/<param>` namespace as
// baked automation and the mod matrix (PROTOCOL.md §Param namespace). The hook
// is applied as the TOP layer in stateAt (static -> ISF default -> baked ->
// mod-matrix -> Lua), so a script always wins where it sets a value.
//
// Sandbox (mirrors plugin/Source/LuaScriptEngine.h): a fresh lua_State with only
// the base/table/string/math stdlibs; dofile/loadfile/load/require and the whole
// os/io surface are absent; math.random/randomseed are removed so re-export is
// byte-identical (the determinism contract every export test asserts). The state
// PERSISTS for the whole export, so a script may keep cross-frame state in
// globals — the exporter warms the hook from timeline frame 0 (like the M5/M6
// drivers) so a mid-range export sees the same global history as a full one.
//
// Header-only and guarded by ARBIT_HAVE_LUA: when Lua is not found at configure
// time the class compiles to no-ops and a job's luaScript is silently inert.
//
#include <string>
#include <map>
#include <chrono>
#include <cstdlib>
#include <limits>

#if ARBIT_HAVE_LUA
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#endif

namespace arbitlua
{
// One sounding note handed to the script as a ctx.notes[i] table. The JI data
// (cents, ratio) is what makes Arbit's scripting unique — no other tool's
// per-frame hook can read the project's microtonal notes.
struct HookNote
{
    double midi = 60.0;      // .midi     — MIDI note number (may be fractional)
    double freq = 0.0;       // .freq     — resolved frequency (Hz, through the link chain)
    double velocity = 0.0;   // .velocity — 0..127
    double cents = 0.0;      // .cents    — cents from the JI root (centsFromRoot)
    double age = 0.0;        // .age      — beats since this note's onset
    int    trackId = 0;      // .trackId
    int    ratioNum = 1;     // .ratioNum — reduced interval vs root (numerator)
    int    ratioDen = 1;     // .ratioDen — denominator
    bool   isRoot = false;   // .isRoot
    float  primes[6] = {0,0,0,0,0,0};  // .primes[1..6] — exponents of 2,3,5,7,11,13
};

// One harmonic link handed to the script as a ctx.links[i] table — the JI
// edge graph the shader's uLinks texture also carries (M5 Block C).
struct HookLink
{
    int    slaveNoteId = 0;   // .slave  — tuned note's id
    int    masterNoteId = 0;  // .master — reference note's id
    int    num = 1;           // .num    — slaveHarmonic (ratio numerator)
    int    den = 1;           // .den    — masterHarmonic (denominator)
    double ratio = 1.0;       // .ratio  — num/den
};

// The per-frame context handed to the script's frame(ctx). Built by the
// exporter each frame; POD so it exists regardless of ARBIT_HAVE_LUA.
struct FrameCtx
{
    double t = 0.0;            // ctx.t      — display seconds
    double beat = 0.0;         // ctx.beat   — absolute beat (t * bpm/60)
    double bpm = 120.0;        // ctx.bpm
    double beatsPerBar = 4.0;  // ctx.bar
    long   frame = 0;          // ctx.frame  — frame index from the timeline head
    // Block B audio (zero when unavailable, e.g. warm-up before the range start)
    float rms = 0.0f;          // ctx.rms
    float peak = 0.0f;         // ctx.peak
    float onset = 0.0f;        // ctx.onset
    float onsetAge = 0.0f;     // ctx.onsetAge
    const float* bands = nullptr;  // ctx.bands[1..bandCount] (1-indexed in Lua)
    int bandCount = 0;
    // Block C score: the notes SOUNDING at this frame's beat (ctx.notes[1..N],
    // 1-indexed), with their JI cents/ratio/prime-exponents — plus the root and
    // the full harmonic link graph (ctx.links[1..linkCount], score-global).
    const HookNote* notes = nullptr;
    int    noteCount = 0;      // ctx.noteCount == #ctx.notes (active count)
    double rootFreq = 0.0;     // ctx.rootFreq  — JI root frequency (Hz)
    const HookLink* links = nullptr;
    int    linkCount = 0;      // ctx.linkCount == #ctx.links
};

class LuaHook
{
public:
    LuaHook() = default;
    ~LuaHook() { close(); }
    LuaHook (const LuaHook&) = delete;
    LuaHook& operator= (const LuaHook&) = delete;

    // Compile the script and run its top-level once (setup). Returns true iff it
    // compiled, ran, and defined a global `frame` function. errOut carries the
    // Lua message on failure. A no-op returning false when built without Lua.
    bool compile (const std::string& source, std::string& errOut,
                  uint32_t cpuMs = 25, uint32_t memoryMiB = 64);

    bool valid() const
    {
       #if ARBIT_HAVE_LUA
        return L_ != nullptr && hasFrameFn_;
       #else
        return false;
       #endif
    }

    // Call frame(ctx); merge the returned {paramId=value} table into `out`
    // (string keys with number values only). Returns false + errOut on a Lua
    // error (logged once by the caller). No-op when !valid().
    bool runFrame (const FrameCtx& ctx, std::map<std::string, double>& out,
                   std::string& errOut);

    void close();

private:
   #if ARBIT_HAVE_LUA
    lua_State* L_ = nullptr;
    bool hasFrameFn_ = false;
public:
    struct AllocatorTestState
    {
        size_t used=0, maximum=0;
        bool accountingValid=true;
        uint32_t cpuMs=25;
        std::chrono::steady_clock::time_point deadline{};
    };
    static void* testAllocate (AllocatorTestState& state, void* ptr,
                               size_t oldSize, size_t newSize)
    { return allocate(&state, ptr, oldSize, newSize); }
private:
    using Limits = AllocatorTestState;
    Limits limits_;
    static void* allocate (void* opaque, void* ptr, size_t oldSize, size_t newSize)
    {
        auto& l = *static_cast<Limits*> (opaque);
        if (! l.accountingValid) return nullptr;
        // Lua documents oldSize as a type tag, not an allocation size, when ptr is null.
        const size_t accountedOld = ptr == nullptr ? 0 : oldSize;
        if (ptr != nullptr && accountedOld > l.used) { l.accountingValid = false; return nullptr; }
        if (newSize == 0)
        {
            if (ptr != nullptr) { l.used -= accountedOld; std::free (ptr); }
            return nullptr;
        }
        const size_t base = l.used - accountedOld;
        if (base > l.maximum || newSize > l.maximum - base) return nullptr;
        if (auto* replacement = std::realloc (ptr, newSize))
        {
            l.used = base + newSize;
            return replacement;
        }
        return nullptr;
    }
    static void deadlineHook (lua_State* state, lua_Debug*)
    {
        auto* self=*static_cast<LuaHook**>(lua_getextraspace(state));
        if(std::chrono::steady_clock::now() >= self->limits_.deadline) luaL_error(state,"Lua CPU deadline exceeded");
    }
    void armDeadline() { limits_.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(limits_.cpuMs); }
   #endif
};

#if ARBIT_HAVE_LUA

inline void LuaHook::close()
{
    if (L_ != nullptr) { lua_close (L_); L_ = nullptr; }
    hasFrameFn_ = false;
}

inline bool LuaHook::compile (const std::string& source, std::string& errOut,
                             uint32_t cpuMs, uint32_t memoryMiB)
{
    close();
    constexpr size_t bytesPerMiB = 1024u * 1024u;
    if (memoryMiB > std::numeric_limits<size_t>::max() / bytesPerMiB)
    {
        errOut = "Lua memory limit overflow";
        return false;
    }
    limits_ = {};
    limits_.cpuMs=cpuMs;
    limits_.maximum=static_cast<size_t>(memoryMiB) * bytesPerMiB;
#if LUA_VERSION_NUM >= 505
    // Lua 5.5 makes the string-table seed explicit. Keep it fixed because the
    // programmable export contract is byte-deterministic and random APIs are
    // removed from the sandbox below.
    L_ = lua_newstate(allocate, &limits_, 0);
#else
    L_ = lua_newstate(allocate, &limits_);
#endif
    if (L_ == nullptr) { errOut = "out of memory creating Lua state"; return false; }
    *static_cast<LuaHook**>(lua_getextraspace(L_)) = this;
    lua_sethook(L_, deadlineHook, LUA_MASKCOUNT, 1000);

    // Safe stdlibs only — no os/io, no package loader.
    luaL_requiref (L_, "_G",     luaopen_base,   1); lua_pop (L_, 1);
    luaL_requiref (L_, "table",  luaopen_table,  1); lua_pop (L_, 1);
    luaL_requiref (L_, "string", luaopen_string, 1); lua_pop (L_, 1);
    luaL_requiref (L_, "math",   luaopen_math,   1); lua_pop (L_, 1);

    // Strip the dangerous / non-deterministic globals.
    const char* kill[] = { "dofile", "loadfile", "load", "loadstring",
                           "require", "collectgarbage", "os", "io" };
    for (const char* g : kill) { lua_pushnil (L_); lua_setglobal (L_, g); }
    // math.random/randomseed break byte-identical re-export → remove them.
    lua_getglobal (L_, "math");
    if (lua_istable (L_, -1))
    {
        lua_pushnil (L_); lua_setfield (L_, -2, "random");
        lua_pushnil (L_); lua_setfield (L_, -2, "randomseed");
    }
    lua_pop (L_, 1);

    armDeadline();
    if (luaL_loadbuffer (L_, source.data(), source.size(), "=arbit_lua_hook") != LUA_OK
        || lua_pcall (L_, 0, 0, 0) != LUA_OK)
    {
        const char* msg = lua_tostring (L_, -1);
        errOut = msg != nullptr ? msg : "Lua load/run error";
        close();
        return false;
    }

    lua_getglobal (L_, "frame");
    hasFrameFn_ = (lua_isfunction (L_, -1) != 0);
    lua_pop (L_, 1);
    if (! hasFrameFn_)
    {
        errOut = "script defines no global function `frame(ctx)`";
        close();
        return false;
    }
    return true;
}

inline bool LuaHook::runFrame (const FrameCtx& ctx,
                               std::map<std::string, double>& out,
                               std::string& errOut)
{
    if (! valid()) return true;   // inert, not an error

    lua_settop (L_, 0);           // defensive: start each frame on a clean stack
    lua_getglobal (L_, "frame");

    // Build the ctx table.
    lua_createtable (L_, 0, 12);
    auto setNum = [&] (const char* k, double v)
    { lua_pushnumber (L_, v); lua_setfield (L_, -2, k); };
    setNum ("t", ctx.t);
    setNum ("beat", ctx.beat);
    setNum ("bpm", ctx.bpm);
    setNum ("bar", ctx.beatsPerBar);
    setNum ("frame", (double) ctx.frame);
    setNum ("rms", ctx.rms);
    setNum ("peak", ctx.peak);
    setNum ("onset", ctx.onset);
    setNum ("onsetAge", ctx.onsetAge);
    setNum ("noteCount", (double) ctx.noteCount);
    setNum ("linkCount", (double) ctx.linkCount);
    setNum ("rootFreq", ctx.rootFreq);
    // ctx.bands[1..N] (1-indexed, Lua convention).
    lua_createtable (L_, ctx.bandCount, 0);
    for (int i = 0; i < ctx.bandCount; ++i)
    {
        lua_pushnumber (L_, ctx.bands != nullptr ? ctx.bands[i] : 0.0);
        lua_rawseti (L_, -2, i + 1);
    }
    lua_setfield (L_, -2, "bands");
    // ctx.notes[1..noteCount] — the sounding notes with their JI data.
    lua_createtable (L_, ctx.noteCount, 0);
    for (int i = 0; i < ctx.noteCount; ++i)
    {
        const HookNote& n = ctx.notes[i];
        lua_createtable (L_, 0, 9);
        lua_pushnumber (L_, n.midi);      lua_setfield (L_, -2, "midi");
        lua_pushnumber (L_, n.freq);      lua_setfield (L_, -2, "freq");
        lua_pushnumber (L_, n.velocity);  lua_setfield (L_, -2, "velocity");
        lua_pushnumber (L_, n.cents);     lua_setfield (L_, -2, "cents");
        lua_pushnumber (L_, n.age);       lua_setfield (L_, -2, "age");
        lua_pushinteger (L_, n.trackId);  lua_setfield (L_, -2, "trackId");
        lua_pushinteger (L_, n.ratioNum); lua_setfield (L_, -2, "ratioNum");
        lua_pushinteger (L_, n.ratioDen); lua_setfield (L_, -2, "ratioDen");
        lua_pushboolean (L_, n.isRoot ? 1 : 0); lua_setfield (L_, -2, "isRoot");
        lua_createtable (L_, 6, 0);                  // note.primes[1..6] = exps of 2,3,5,7,11,13
        for (int p = 0; p < 6; ++p)
        { lua_pushnumber (L_, n.primes[p]); lua_rawseti (L_, -2, p + 1); }
        lua_setfield (L_, -2, "primes");
        lua_rawseti (L_, -2, i + 1);
    }
    lua_setfield (L_, -2, "notes");
    // ctx.links[1..linkCount] — the harmonic link graph (score-global).
    lua_createtable (L_, ctx.linkCount, 0);
    for (int i = 0; i < ctx.linkCount; ++i)
    {
        const HookLink& lk = ctx.links[i];
        lua_createtable (L_, 0, 5);
        lua_pushinteger (L_, lk.slaveNoteId);  lua_setfield (L_, -2, "slave");
        lua_pushinteger (L_, lk.masterNoteId); lua_setfield (L_, -2, "master");
        lua_pushinteger (L_, lk.num);          lua_setfield (L_, -2, "num");
        lua_pushinteger (L_, lk.den);          lua_setfield (L_, -2, "den");
        lua_pushnumber  (L_, lk.ratio);        lua_setfield (L_, -2, "ratio");
        lua_rawseti (L_, -2, i + 1);
    }
    lua_setfield (L_, -2, "links");

    armDeadline();
    if (lua_pcall (L_, 1, 1, 0) != LUA_OK)
    {
        const char* msg = lua_tostring (L_, -1);
        errOut = msg != nullptr ? msg : "Lua frame() error";
        lua_settop (L_, 0);
        return false;
    }

    // Merge the returned {paramId = value} table (string keys, number values).
    if (lua_istable (L_, -1))
    {
        lua_pushnil (L_);
        while (lua_next (L_, -2) != 0)
        {
            if (lua_type (L_, -2) == LUA_TSTRING && lua_isnumber (L_, -1))
                out[lua_tostring (L_, -2)] = lua_tonumber (L_, -1);
            lua_pop (L_, 1);   // pop value, keep key for lua_next
        }
    }
    lua_settop (L_, 0);
    return true;
}

#else   // ! ARBIT_HAVE_LUA — no-op stubs

inline void LuaHook::close() {}
inline bool LuaHook::compile (const std::string&, std::string& errOut, uint32_t, uint32_t)
{ errOut = "Lua support not compiled in"; return false; }
inline bool LuaHook::runFrame (const FrameCtx&, std::map<std::string, double>&, std::string&)
{ return true; }

#endif

}  // namespace arbitlua
