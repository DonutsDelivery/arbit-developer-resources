#include "js_hook.h"
#include "lua_hook.h"
#include <cassert>
#include <chrono>
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
