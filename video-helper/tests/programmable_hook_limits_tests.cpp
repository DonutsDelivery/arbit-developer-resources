#include "js_hook.h"
#include "lua_hook.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <string>

namespace
{
template <typename Hook>
void expectBoundedFailure (Hook& hook, const std::string& source,
                           uint32_t cpuMs, uint32_t memoryMiB)
{
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    assert (! hook.compile(source, error, cpuMs, memoryMiB));
    assert (! error.empty());
    assert (std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
}
}

int main()
{
#if ARBIT_HAVE_QUICKJS
    {
        arbitjs::JsHook hook;
        std::string error;
        assert(hook.compile(
            "function frame(ctx){const n=ctx.timelineNotes[0];return {probe:n.id+n.startBeat+"
            "n.lengthBeats+n.age+n.remain+(n.active?1:0)+ctx.scoreHistoryBeats+"
            "ctx.scoreLookaheadBeats+ctx.noteCount+ctx.timelineNoteCount};}", error, 50, 8));
        arbitlua::HookNote note;
        note.id = 7; note.startBeat = 2.0; note.lengthBeats = 3.0;
        note.age = 4.0; note.remain = -1.0; note.active = true;
        arbitlua::FrameCtx ctx;
        ctx.notes = &note; ctx.noteCount = 1;
        ctx.timelineNotes = &note; ctx.timelineNoteCount = 1;
        ctx.scoreHistoryBeats = 8.0; ctx.scoreLookaheadBeats = 16.0;
        std::map<std::string, double> output;
        assert(hook.runFrame(ctx, output, error));
        assert(std::abs(output["probe"] - 42.0) < 1.0e-9);
    }
    {
        arbitjs::JsHook hook;
        expectBoundedFailure(hook, "for (;;) {}", 10, 8);
    }
    {
        arbitjs::JsHook hook;
        expectBoundedFailure(hook,
            "let a=[]; for(let i=0;;++i) a.push(new ArrayBuffer(1048576));", 100, 4);
    }
#endif
#if ARBIT_HAVE_LUA
    {
        arbitlua::LuaHook hook;
        std::string error;
        assert(hook.compile(
            "function frame(ctx) local n=ctx.timelineNotes[1] return {probe=n.id+n.startBeat+"
            "n.lengthBeats+n.age+n.remain+(n.active and 1 or 0)+ctx.scoreHistoryBeats+"
            "ctx.scoreLookaheadBeats+ctx.noteCount+ctx.timelineNoteCount} end", error, 50, 8));
        arbitlua::HookNote note;
        note.id = 7; note.startBeat = 2.0; note.lengthBeats = 3.0;
        note.age = 4.0; note.remain = -1.0; note.active = true;
        arbitlua::FrameCtx ctx;
        ctx.notes = &note; ctx.noteCount = 1;
        ctx.timelineNotes = &note; ctx.timelineNoteCount = 1;
        ctx.scoreHistoryBeats = 8.0; ctx.scoreLookaheadBeats = 16.0;
        std::map<std::string, double> output;
        assert(hook.runFrame(ctx, output, error));
        assert(std::abs(output["probe"] - 42.0) < 1.0e-9);
    }
    {
        using State = arbitlua::LuaHook::AllocatorTestState;
        State state; state.maximum = 32;
        void* p = arbitlua::LuaHook::testAllocate(state, nullptr, 9, 16); // oldSize is a Lua type tag
        assert(p != nullptr && state.used == 16 && state.accountingValid);
        p = arbitlua::LuaHook::testAllocate(state, p, 16, 32);           // exact boundary
        assert(p != nullptr && state.used == 32);
        assert(arbitlua::LuaHook::testAllocate(state, p, 32, 33) == nullptr);
        assert(state.used == 32 && state.accountingValid);
        assert(arbitlua::LuaHook::testAllocate(state, p, 33, 0) == nullptr);
        assert(! state.accountingValid && state.used == 32);             // free underflow fails closed
        std::free(p);
    }
    {
        using State = arbitlua::LuaHook::AllocatorTestState;
        State state; state.maximum = std::numeric_limits<size_t>::max();
        void* p = arbitlua::LuaHook::testAllocate(state, nullptr, 0, 1);
        assert(p != nullptr && state.used == 1);
        assert(arbitlua::LuaHook::testAllocate(
            state, p, 1, std::numeric_limits<size_t>::max()) == nullptr);
        assert(state.used == 1 && state.accountingValid);                // realloc failure preserves accounting
        assert(arbitlua::LuaHook::testAllocate(state, p, 1, 0) == nullptr);
        assert(state.used == 0 && state.accountingValid);
    }
    {
        arbitlua::LuaHook hook;
        std::string error;
        if (sizeof(size_t) < sizeof(uint64_t))
            assert(! hook.compile("function frame() return {} end", error, 10,
                                  std::numeric_limits<uint32_t>::max()));
    }
    {
        arbitlua::LuaHook hook;
        expectBoundedFailure(hook, "while true do end", 10, 8);
    }
    {
        arbitlua::LuaHook hook;
        expectBoundedFailure(hook,
            "local a={} while true do a[#a+1]=string.rep('x',1048576) end", 100, 4);
    }
#endif
    std::cout << "programmable hook limit tests passed\n";
}
