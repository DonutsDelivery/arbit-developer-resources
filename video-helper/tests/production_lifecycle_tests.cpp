#include "../../plugin/Source/SidecarProcessManager.h"
#include "../../plugin/Source/ProgrammableRuntimeAuthority.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#if ! JUCE_WINDOWS
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#endif

static SidecarProcessManager::Config configFor(const juce::File& helper,
                                                const std::filesystem::path& matte,
                                                uint8_t marker, uint64_t generation)
{
    SidecarProcessManager::Config config;
    config.binaryPath = helper;
    config.args.add("--matte-cache-root");
    config.args.add(matte.string());
    config.args.add("--depth-cache-root");
    config.args.add(matte.string());
    config.requiresPrivateChannel = true;
    config.inheritedPrivateBytes.assign(40, marker);
    if (marker == 0x5a)
    {
        constexpr char privateMarker[] = "M6_PRIVATE_CHANNEL_MARKER";
        std::copy_n(reinterpret_cast<const uint8_t*>(privateMarker), sizeof(privateMarker) - 1,
                    config.inheritedPrivateBytes.begin());
    }
    for (int i = 0; i < 8; ++i)
        config.inheritedPrivateBytes[32 + i] = uint8_t(generation >> (56 - i * 8));
    config.healthCheckIntervalMs = 0;
    config.maxRestartAttempts = 0;
    config.requestTimeoutMs = 2000;
    config.terminateOnRequestTimeout = true;
#if defined(__linux__)
    config.requiresResourceLimits = true;
    config.maxAddressSpaceBytes = uint64_t { 4 } * 1024u * 1024u * 1024u;
    config.maxCpuSeconds = 60;
#endif
    return config;
}

#if defined(__linux__)
static void proveMeasuredResourceLimit(SidecarProcessManager& manager, const char* mode)
{
    SidecarProcessManager::Config config;
    config.binaryPath = juce::File(RESOURCE_LIMIT_FIXTURE);
    config.args.add(mode);
    config.healthCheckIntervalMs = 0;
    config.maxRestartAttempts = 0;
    config.requiresResourceLimits = true;
    if (std::strcmp(mode, "memory") == 0)
        config.maxAddressSpaceBytes = uint64_t { 64 } * 1024u * 1024u;
    else
        config.maxCpuSeconds = 1;

    assert(manager.launch(config));
    juce::String error;
    const auto prepared = manager.sendRequestSync("prepare", juce::var(), 2000, error);
    assert(error.isEmpty());
    const pid_t leader = manager.getOwnedProcessId();
    const pid_t descendant = static_cast<pid_t>((int)prepared.getProperty("descendantPid", 0));
    assert(leader > 0 && descendant > 0 && getpgid(leader) == leader
           && getpgid(descendant) == leader);

    const auto started = std::chrono::steady_clock::now();
    error.clear();
    manager.sendRequestSync("exhaust", juce::var(), 5000, error);
    assert(error.containsIgnoreCase("process exited"));
    assert(std::chrono::steady_clock::now() - started < std::chrono::seconds(6));
    assert(!manager.isRunning() && manager.getOwnedProcessId() == -1);
    for (int i = 0; i < 100 && (kill(descendant, 0) == 0 || errno != ESRCH); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(kill(descendant, 0) != 0 && errno == ESRCH);
    const auto reason = manager.getLastTerminationReason();
    assert(reason.length() < 160);
    assert(reason.containsIgnoreCase(mode));
}
#endif

int main()
{
    juce::MessageManager::getInstance();
    const juce::File helper(PRIVATE_LIFECYCLE_HELPER);
    const auto matte = std::filesystem::temp_directory_path() /
        ("arbit-private-lifecycle-" + std::to_string(juce::Time::getHighResolutionTicks()));
    std::filesystem::create_directories(matte);

    SidecarProcessManager manager;
    auto config = configFor(helper, matte, 0x5a, 11);
#if JUCE_WINDOWS
    assert(!manager.launch(config));
    assert(config.inheritedPrivateBytes.empty());
    assert(manager.getConfig().inheritedPrivateBytes.empty());
    assert(!manager.isRunning());
    std::filesystem::remove_all(matte);
    return 0;
#else
    programmableruntime::SessionSecret secretA {};
    std::copy_n(config.inheritedPrivateBytes.begin(), secretA.size(), secretA.begin());
#if ! defined(__linux__)
    // A platform without kernel-backed resource limits must reject a worker
    // that requires them. Then exercise the remaining production lifecycle
    // with that unsupported capability explicitly not requested.
    config.requiresResourceLimits = true;
    assert(!manager.launch(config));
    assert(config.inheritedPrivateBytes.empty());
    assert(!manager.isRunning());
    config = configFor(helper, matte, 0x5a, 11);
#endif
    assert(manager.launch(config));
    assert(config.inheritedPrivateBytes.empty());
    assert(manager.getConfig().inheritedPrivateBytes.empty());
    juce::String error;
    auto result = manager.sendRequestSync("ping", juce::var(), 2000, error);
    assert(error.isEmpty() && !result.isVoid());
    result = manager.sendRequestSync("test_private_startup_diagnostics", juce::var(), 2000, error);
    assert(error.isEmpty());
    for (const char* field : { "privateFdClosed", "markerAbsentArgvEnv", "packetTemporaryZero", "sourceTemporaryZero" })
        assert((bool)result.getProperty(field, false));

    HarmonicMIDI::ProgrammableRuntimeAuthority authority;
    authority.beginHelperSession(secretA, 11);
    const juce::String source("function frame(ctx) return 1 end");
    authority.approve(programmableruntime::PayloadKind::lua, source);
    const auto oldGrant = authority.project(programmableruntime::PayloadKind::lua, source, true);
    auto makeScriptParams = [&source](const programmableruntime::Grant& grant)
    {
        auto* object = new juce::DynamicObject();
        object->setProperty("source", source); object->setProperty("lang", "lua");
        object->setProperty("runtimeGrant", HarmonicMIDI::ProgrammableRuntimeAuthority::toVar(grant));
        return juce::var(object);
    };
    result = manager.sendRequestSync("viewport_set_script", makeScriptParams(oldGrant), 2000, error);
    assert(error.isEmpty() && (bool)result.getProperty("ok", false));

    manager.shutdown();
    assert(!manager.isRunning());

    for (const auto size : { std::size_t(0), std::size_t(39), std::size_t(41) })
    {
        auto malformed = configFor(helper, matte, 0x33, 12);
        malformed.inheritedPrivateBytes.resize(size);
        assert(!manager.launch(malformed));
        assert(malformed.inheritedPrivateBytes.empty());
        assert(!manager.isRunning());
    }

    auto missing = configFor(juce::File((matte / "missing-helper").string()), matte, 0x44, 13);
    assert(!manager.launch(missing));
    assert(missing.inheritedPrivateBytes.empty());

    const auto nonexecPath = matte / "nonexec-helper";
    { std::ofstream file(nonexecPath); file << "not executable\n"; }
    ::chmod(nonexecPath.c_str(), 0600);
    auto nonexec = configFor(juce::File(nonexecPath.string()), matte, 0x45, 14);
    assert(manager.launch(nonexec));
    for (int i = 0; i < 100 && manager.isRunning(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!manager.isRunning());

    auto crashed = configFor(helper, matte, 0x66, 15);
    assert(manager.launch(crashed));
    error.clear(); result = manager.sendRequestSync("ping", juce::var(), 2000, error);
    assert(error.isEmpty());
    assert(manager.crashChildForRestartTest());
    for (int i = 0; i < 100 && manager.isRunning(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!manager.isRunning());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(!manager.isRunning());

    auto replacement = configFor(helper, matte, 0x77, 16);
    assert(manager.launch(replacement));
    error.clear(); result = manager.sendRequestSync("viewport_set_script", makeScriptParams(oldGrant), 2000, error);
    assert(error.containsIgnoreCase("session") || error.containsIgnoreCase("authentication"));
    programmableruntime::SessionSecret secretB {}; secretB.fill(0x77);
    authority.beginHelperSession(secretB, 16);
    authority.approve(programmableruntime::PayloadKind::lua, source);
    const auto freshGrant = authority.project(programmableruntime::PayloadKind::lua, source, true);
    error.clear(); result = manager.sendRequestSync("viewport_set_script", makeScriptParams(freshGrant), 2000, error);
    assert(error.isEmpty() && (bool)result.getProperty("ok", false));
    manager.shutdown();
    assert(manager.getOwnedProcessId() == -1);

#if defined(__linux__)
    // These are measured over-budget workloads under limits installed by the
    // production fork path. No resource-limit signal is injected.
    proveMeasuredResourceLimit(manager, "memory");
    proveMeasuredResourceLimit(manager, "cpu");

    // Resource-failed sessions cannot authorize a replacement helper. Only a
    // grant projected from the replacement's fresh session is accepted.
    auto postLimit = configFor(helper, matte, 0x79, 19);
    assert(manager.launch(postLimit));
    error.clear(); result = manager.sendRequestSync("viewport_set_script", makeScriptParams(freshGrant), 2000, error);
    assert(error.containsIgnoreCase("session") || error.containsIgnoreCase("authentication"));
    programmableruntime::SessionSecret secretC {}; secretC.fill(0x79);
    authority.beginHelperSession(secretC, 19);
    authority.approve(programmableruntime::PayloadKind::lua, source);
    const auto newestGrant = authority.project(programmableruntime::PayloadKind::lua, source, true);
    error.clear(); result = manager.sendRequestSync("viewport_set_script", makeScriptParams(newestGrant), 2000, error);
    assert(error.isEmpty() && (bool)result.getProperty("ok", false));
    manager.shutdown();
#endif

    // A real RPC timeout is a lifecycle fault for this privileged helper. Stop
    // the actual helper process, then prove the production manager kills and
    // reaps its complete process group and clears its process ledger.
    auto timedOut = configFor(helper, matte, 0x78, 17);
    timedOut.requestTimeoutMs = 100;
    assert(manager.launch(timedOut));
    error.clear(); result = manager.sendRequestSync("ping", juce::var(), 2000, error);
    assert(error.isEmpty());
    const pid_t timedOutPid = manager.getOwnedProcessId();
    assert(timedOutPid > 0 && getpgid(timedOutPid) == timedOutPid);
    assert(kill(-timedOutPid, SIGSTOP) == 0);
    const auto timeoutStart = std::chrono::steady_clock::now();
    error.clear(); result = manager.sendRequestSync("ping", juce::var(), 100, error);
    assert(error.containsIgnoreCase("timed out"));
    assert(std::chrono::steady_clock::now() - timeoutStart < std::chrono::seconds(5));
    assert(!manager.isRunning());
    assert(manager.getOwnedProcessId() == -1);
    assert(kill(timedOutPid, 0) != 0 && errno == ESRCH);
    assert(manager.getLastTerminationReason().containsIgnoreCase("timed out"));

    auto stalled = configFor(helper, matte, 0x22, 18);
    stalled.testPrivatePayloadWriter = [] (int fd, const std::vector<uint8_t>& bytes)
    {
        const auto sent = ::write(fd, bytes.data(), 7);
        std::this_thread::sleep_for(std::chrono::milliseconds(2300));
        return sent > 0 ? static_cast<size_t>(sent) : size_t { 0 };
    };
    const auto stallStart = std::chrono::steady_clock::now();
    assert(!manager.launch(stalled));
    assert(!manager.isRunning());
    assert(std::chrono::steady_clock::now() - stallStart < std::chrono::seconds(5));
    std::filesystem::remove_all(matte);
#endif
}
