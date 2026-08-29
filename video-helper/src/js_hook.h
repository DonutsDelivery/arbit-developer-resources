#pragma once
//
// Per-frame JavaScript hook (visual-engine plan P2 — QuickJS-ng front-end).
//
// The exact analogue of the Lua hook (lua_hook.h): a user/agent-authored script
// defines a global `frame(ctx)` returning {paramId = value, ...}, applied as the
// TOP layer in stateAt. JavaScript is the creative-coding + AI-agent lingua
// franca, so this gives Shadertoy-JS / generative-JS authoring without a byte of
// browser. Only one hook (Lua OR JS) is active per export job.
//
//   function frame(ctx) {
//     return { "clip1/source/opacity": 0.5 + 0.5 * Math.sin(ctx.beat * Math.PI) };
//   }
//
// ctx mirrors the Lua ctx field-for-field, but with JS conventions: ctx.bands,
// ctx.notes, ctx.timelineNotes, and ctx.links are 0-indexed arrays. Lua uses
// 1-indexed arrays. The note JI
// data (cents, ratio, prime exponents) is the same Arbit-unique payload.
//
// Reuses lua_hook.h's FrameCtx / HookNote / HookLink PODs verbatim (the plan's
// "do not duplicate them") — they are engine-agnostic.
//
// Sandbox / determinism: a plain JS_NewContext exposes only the ECMAScript
// builtins (Object/Array/Math/JSON/String/...) — no module loader and no
// quickjs-libc (os/std) is ever linked, so file/IO/network are absent by
// omission, mirroring the Lua sandbox dropping os/io/require. A determinism
// prelude removes the non-deterministic surfaces (Math.random, Date) so
// re-export stays byte-identical — the contract every export test asserts.
// QuickJS-ng's refcount GC is itself deterministic.
//
// Header-only and guarded by ARBIT_HAVE_QUICKJS: without QuickJS at configure
// time the class compiles to no-ops and a job's jsScript is silently inert.
//
#include "lua_hook.h"   // arbitlua::FrameCtx / HookNote / HookLink (shared PODs)

#include <string>
#include <map>
#include <chrono>

#if ARBIT_HAVE_QUICKJS
extern "C" {
#include "quickjs.h"
}
#endif

namespace arbitjs
{
// Reuse the shared per-frame contract structs — no duplication (P2).
using arbitlua::FrameCtx;
using arbitlua::HookNote;
using arbitlua::HookLink;

class JsHook
{
public:
    JsHook() = default;
    ~JsHook() { close(); }
    JsHook (const JsHook&) = delete;
    JsHook& operator= (const JsHook&) = delete;

    // Compile the script and run its top-level once (setup). Returns true iff it
    // compiled, ran, and defined a global `frame` function. errOut carries the
    // JS message on failure. A no-op returning false when built without QuickJS.
    bool compile (const std::string& source, std::string& errOut,
                  uint32_t cpuMs = 25, uint32_t memoryMiB = 64);

    bool valid() const
    {
       #if ARBIT_HAVE_QUICKJS
        return cx_ != nullptr && hasFrameFn_;
       #else
        return false;
       #endif
    }

    // Call frame(ctx); merge the returned {paramId: value} object into `out`
    // (string keys with number values only). Returns false + errOut on a JS
    // error. No-op when !valid().
    bool runFrame (const FrameCtx& ctx, std::map<std::string, double>& out,
                   std::string& errOut);

    void close();

private:
   #if ARBIT_HAVE_QUICKJS
    JSRuntime* rt_ = nullptr;
    JSContext* cx_ = nullptr;   // named cx_ to avoid shadowing the JSContext type
    JSValue    frameFn_ {};
    bool hasFrameFn_ = false;
    uint32_t cpuMs_ = 25;
    std::chrono::steady_clock::time_point deadline_ {};
    static int interrupt (JSRuntime*, void* opaque)
    { return std::chrono::steady_clock::now() >= static_cast<JsHook*>(opaque)->deadline_; }
   #endif
};

#if ARBIT_HAVE_QUICKJS

inline void JsHook::close()
{
    if (cx_ != nullptr)
    {
        if (hasFrameFn_) { JS_FreeValue (cx_, frameFn_); }
        JS_FreeContext (cx_);
        cx_ = nullptr;
    }
    if (rt_ != nullptr) { JS_FreeRuntime (rt_); rt_ = nullptr; }
    hasFrameFn_ = false;
}

inline bool JsHook::compile (const std::string& source, std::string& errOut,
                            uint32_t cpuMs, uint32_t memoryMiB)
{
    close();
    rt_ = JS_NewRuntime();
    if (rt_ == nullptr) { errOut = "out of memory creating JS runtime"; return false; }
    cpuMs_ = cpuMs;
    JS_SetMemoryLimit (rt_, static_cast<size_t>(memoryMiB) * 1024u * 1024u);
    JS_SetInterruptHandler (rt_, interrupt, this);
    cx_ = JS_NewContext (rt_);   // ECMAScript builtins only; no libc/module loader
    if (cx_ == nullptr) { errOut = "out of memory creating JS context"; close(); return false; }

    auto grabException = [&] (const char* fallback)
    {
        JSValue e = JS_GetException (cx_);
        const char* s = JS_ToCString (cx_, e);
        errOut = s != nullptr ? s : fallback;
        if (s != nullptr) JS_FreeCString (cx_, s);
        JS_FreeValue (cx_, e);
    };

    // Determinism prelude: strip non-deterministic globals so re-export is
    // byte-identical. Math.random/Date removed (calling them then throws —
    // loud + deterministic, mirroring the Lua hook nilling math.random/os).
    static const char* kPrelude =
        "Math.random=undefined;"
        "globalThis.Date=undefined;";
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(cpuMs_);
    JSValue pv = JS_Eval (cx_, kPrelude, std::char_traits<char>::length (kPrelude),
                          "=arbit_js_prelude", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException (pv)) { JS_FreeValue (cx_, pv); grabException ("JS prelude error"); close(); return false; }
    JS_FreeValue (cx_, pv);

    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(cpuMs_);
    JSValue r = JS_Eval (cx_, source.data(), source.size(), "=arbit_js_hook",
                         JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException (r)) { JS_FreeValue (cx_, r); grabException ("JS load/run error"); close(); return false; }
    JS_FreeValue (cx_, r);

    JSValue global = JS_GetGlobalObject (cx_);
    frameFn_ = JS_GetPropertyStr (cx_, global, "frame");
    JS_FreeValue (cx_, global);
    hasFrameFn_ = JS_IsFunction (cx_, frameFn_);
    if (! hasFrameFn_)
    {
        JS_FreeValue (cx_, frameFn_);
        errOut = "script defines no global function `frame(ctx)`";
        close();
        return false;
    }
    return true;
}

inline bool JsHook::runFrame (const FrameCtx& ctx,
                              std::map<std::string, double>& out,
                              std::string& errOut)
{
    if (! valid()) return true;   // inert, not an error

    JSContext* c = cx_;
    auto setNum = [&] (JSValue o, const char* k, double v)
    { JS_SetPropertyStr (c, o, k, JS_NewFloat64 (c, v)); };

    JSValue cobj = JS_NewObject (c);
    setNum (cobj, "t", ctx.t);
    setNum (cobj, "beat", ctx.beat);
    setNum (cobj, "bpm", ctx.bpm);
    setNum (cobj, "bar", ctx.beatsPerBar);
    setNum (cobj, "frame", (double) ctx.frame);
    setNum (cobj, "rms", ctx.rms);
    setNum (cobj, "peak", ctx.peak);
    setNum (cobj, "onset", ctx.onset);
    setNum (cobj, "onsetAge", ctx.onsetAge);
    setNum (cobj, "noteCount", (double) ctx.noteCount);
    setNum (cobj, "timelineNoteCount", (double) ctx.timelineNoteCount);
    setNum (cobj, "linkCount", (double) ctx.linkCount);
    setNum (cobj, "rootFreq", ctx.rootFreq);
    setNum (cobj, "scoreHistoryBeats", ctx.scoreHistoryBeats);
    setNum (cobj, "scoreLookaheadBeats", ctx.scoreLookaheadBeats);

    // ctx.bands[0..N-1] (0-indexed — JS convention).
    JSValue bands = JS_NewArray (c);
    for (int i = 0; i < ctx.bandCount; ++i)
        JS_SetPropertyUint32 (c, bands, (uint32_t) i,
                              JS_NewFloat64 (c, ctx.bands != nullptr ? ctx.bands[i] : 0.0));
    JS_SetPropertyStr (c, cobj, "bands", bands);

    auto buildNotes = [&] (const HookNote* values, int count)
    {
        JSValue notes = JS_NewArray (c);
        for (int i = 0; i < count; ++i)
        {
            const HookNote& n = values[i];
            JSValue o = JS_NewObject (c);
            JS_SetPropertyStr (c, o, "id", JS_NewInt32 (c, n.id));
            setNum (o, "midi", n.midi);
            setNum (o, "freq", n.freq);
            setNum (o, "velocity", n.velocity);
            setNum (o, "cents", n.cents);
            setNum (o, "age", n.age);
            setNum (o, "remain", n.remain);
            setNum (o, "startBeat", n.startBeat);
            setNum (o, "lengthBeats", n.lengthBeats);
            JS_SetPropertyStr (c, o, "active", JS_NewBool (c, n.active));
            JS_SetPropertyStr (c, o, "trackId", JS_NewInt32 (c, n.trackId));
            JS_SetPropertyStr (c, o, "ratioNum", JS_NewInt32 (c, n.ratioNum));
            JS_SetPropertyStr (c, o, "ratioDen", JS_NewInt32 (c, n.ratioDen));
            JS_SetPropertyStr (c, o, "isRoot", JS_NewBool (c, n.isRoot));
            JSValue primes = JS_NewArray (c);
            for (int p = 0; p < 6; ++p)
                JS_SetPropertyUint32 (c, primes, (uint32_t) p, JS_NewFloat64 (c, n.primes[p]));
            JS_SetPropertyStr (c, o, "primes", primes);
            JS_SetPropertyUint32 (c, notes, (uint32_t) i, o);
        }
        return notes;
    };
    JS_SetPropertyStr (c, cobj, "notes", buildNotes (ctx.notes, ctx.noteCount));
    JS_SetPropertyStr (c, cobj, "timelineNotes",
                       buildNotes (ctx.timelineNotes, ctx.timelineNoteCount));

    // ctx.links[0..linkCount-1] — the harmonic link graph (score-global).
    JSValue links = JS_NewArray (c);
    for (int i = 0; i < ctx.linkCount; ++i)
    {
        const HookLink& lk = ctx.links[i];
        JSValue o = JS_NewObject (c);
        JS_SetPropertyStr (c, o, "slave", JS_NewInt32 (c, lk.slaveNoteId));
        JS_SetPropertyStr (c, o, "master", JS_NewInt32 (c, lk.masterNoteId));
        JS_SetPropertyStr (c, o, "num", JS_NewInt32 (c, lk.num));
        JS_SetPropertyStr (c, o, "den", JS_NewInt32 (c, lk.den));
        setNum (o, "ratio", lk.ratio);
        JS_SetPropertyUint32 (c, links, (uint32_t) i, o);
    }
    JS_SetPropertyStr (c, cobj, "links", links);

    JSValue argv[1] = { cobj };
    deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(cpuMs_);
    JSValue res = JS_Call (c, frameFn_, JS_UNDEFINED, 1, argv);
    JS_FreeValue (c, cobj);

    if (JS_IsException (res))
    {
        JS_FreeValue (c, res);
        JSValue e = JS_GetException (c);
        const char* s = JS_ToCString (c, e);
        errOut = s != nullptr ? s : "JS frame() error";
        if (s != nullptr) JS_FreeCString (c, s);
        JS_FreeValue (c, e);
        return false;
    }

    // Merge the returned {paramId: value} object (string keys, number values).
    if (JS_IsObject (res))
    {
        JSPropertyEnum* tab = nullptr;
        uint32_t len = 0;
        if (JS_GetOwnPropertyNames (c, &tab, &len, res,
                                    JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0)
        {
            for (uint32_t i = 0; i < len; ++i)
            {
                JSValue v = JS_GetProperty (c, res, tab[i].atom);
                double d = 0.0;
                if (JS_ToFloat64 (c, &d, v) == 0)
                {
                    const char* k = JS_AtomToCString (c, tab[i].atom);
                    if (k != nullptr) { out[k] = d; JS_FreeCString (c, k); }
                }
                JS_FreeValue (c, v);
            }
            JS_FreePropertyEnum (c, tab, len);
        }
    }
    JS_FreeValue (c, res);
    return true;
}

#else   // ! ARBIT_HAVE_QUICKJS — no-op stubs

inline void JsHook::close() {}
inline bool JsHook::compile (const std::string&, std::string& errOut, uint32_t, uint32_t)
{ errOut = "JavaScript (QuickJS) support not compiled in"; return false; }
inline bool JsHook::runFrame (const FrameCtx&, std::map<std::string, double>&, std::string&)
{ return true; }

#endif

}  // namespace arbitjs
