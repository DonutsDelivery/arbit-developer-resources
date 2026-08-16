#include "video_control_plan.h"

#include <cmath>
#include <iostream>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

videocontrol::Operation valueOp(int nodeId, float value, int slot)
{
    videocontrol::Operation operation;
    operation.nodeId = nodeId;
    operation.kind = "control.const";
    operation.params = { value };
    operation.outputSlots = { slot };
    return operation;
}

videocontrol::Operation sinkOp(int nodeId, int slot)
{
    videocontrol::Operation operation;
    operation.nodeId = nodeId;
    operation.kind = "sink";
    operation.inputs = { { slot } };
    operation.destination = "clip1/gen/level";
    return operation;
}
} // namespace

int main()
{
    arbitmod::Score score;
    arbitmod::Clock clock;
    arbitmod::Audio audio;

    {
        videocontrol::Plan plan;
        plan.numSlots = 3;
        plan.operations.push_back(valueOp(1, 0.25f, 0));
        plan.operations.push_back(valueOp(2, 0.5f, 1));
        videocontrol::Operation math;
        math.nodeId = 3;
        math.kind = "control.math";
        math.inputs = { { 0, 1 }, {} };
        math.params = { 0.0f };
        math.outputSlots = { 2 };
        plan.operations.push_back(math);
        auto sink = sinkOp(4, 2);
        sink.depth = 0.5f;
        plan.operations.push_back(sink);

        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "fan-in plan admits");
        const auto& values = executor.evaluate(score, clock, audio, 1.0f / 60.0f,
                                                1.0f / 60.0f);
        check(values.size() == 1, "fan-in plan emits one sink");
        check(values.size() == 1 && std::abs(values[0].value - 0.375f) < 1.0e-6f,
              "fan-in Math result reaches sink with depth");
    }

    {
        videocontrol::Plan plan;
        plan.numSlots = 2;
        plan.operations.push_back(valueOp(1, 0.8f, 0));
        videocontrol::Operation history;
        history.nodeId = 2;
        history.kind = "control.history";
        history.inputs = { { 0 } };
        history.params = { 0.2f };
        history.outputSlots = { 1 };
        plan.operations.push_back(history);
        plan.operations.push_back(sinkOp(3, 1));

        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "History plan admits");
        const auto first = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(first.size() == 1 && std::abs(first[0].value - 0.2f) < 1.0e-6f,
              "History emits initial value first");
        const auto second = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(second.size() == 1 && std::abs(second[0].value - 0.8f) < 1.0e-6f,
              "History emits previous input next");
        executor.reset();
        const auto reset = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(reset.size() == 1 && std::abs(reset[0].value - 0.2f) < 1.0e-6f,
              "History reset is deterministic");
    }

    {
        videocontrol::Plan invalid;
        invalid.numSlots = static_cast<int>(videocontrol::kMaxSlots + 1);
        videocontrol::Executor executor;
        std::string error;
        check(!executor.bind(invalid, error), "over-cap plan rejects atomically");
        check(!error.empty(), "over-cap rejection carries a diagnostic");
    }

    {
        videocontrol::Plan invalid;
        invalid.numSlots = 1;
        invalid.operations.push_back(valueOp(1, 0.5f, 0));
        auto sink = sinkOp(2, 0);
        sink.targetClipId = 9;
        invalid.operations.push_back(sink);
        videocontrol::Executor executor;
        std::string error;
        check(!executor.bind(invalid, error), "partial structured target rejects atomically");
    }

    {
        audio.rms = 0.9f;
        audio.sourceCount = 1;
        audio.sources[0] = { 42, 1, 0.35f, 0.6f, 1.0f };
        videocontrol::Plan plan;
        plan.numSlots = 1;
        videocontrol::Operation source;
        source.nodeId = 1;
        source.kind = "source";
        source.source.type = arbitmod::SourceType::AudioRms;
        source.source.trackId = 42;
        source.outputSlots = { 0 };
        plan.operations.push_back(source);
        plan.operations.push_back(sinkOp(2, 0));
        videocontrol::Executor executor;
        std::string error;
        check(executor.bind(plan, error), "typed track-audio plan admits");
        const auto values = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(values.size() == 1 && std::abs(values[0].value - 0.35f) < 1.0e-6f,
              "track audio source selects stable track id instead of master");
        source.source.trackId = 99;
        plan.operations[0] = source;
        check(executor.bind(plan, error), "missing track-audio plan remains executable");
        const auto missing = executor.evaluate(score, clock, audio, 0.1f, 0.05f);
        check(missing.size() == 1 && missing[0].value == 0.0f,
              "missing track audio source fails closed to zero");
    }

    if (failures != 0) return 1;
    std::cout << "video control plan: all checks passed\n";
    return 0;
}
