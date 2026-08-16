#pragma once

#include "mod_defs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace videocontrol
{

inline constexpr std::size_t kMaxOperations = 512;
inline constexpr std::size_t kMaxSlots = 2048;
inline constexpr std::size_t kMaxInputsPerOperation = 16;
inline constexpr std::size_t kMaxSourcesPerInput = 64;
inline constexpr std::size_t kMaxParamsPerOperation = 32;
inline constexpr std::size_t kMaxSinks = 128;

struct Operation
{
    int nodeId = -1;
    std::string kind;
    std::vector<std::vector<int>> inputs;
    std::vector<float> params;
    std::vector<int> outputSlots;
    arbitmod::ModSource source;

    std::string destination;
    int targetClipId = -1;
    int targetNodeId = -1;
    std::string targetParamId;
    float depth = 1.0f;
    arbitmod::Curve curve = arbitmod::Curve::Linear;
    float smoothingBeats = 0.0f;
    arbitmod::Mode mode = arbitmod::Mode::Add;
    bool enabled = true;
};

struct Plan
{
    int version = 1;
    int numSlots = 0;
    std::vector<Operation> operations;
};

struct SinkValue
{
    std::string destination;
    float value = 0.0f;
    arbitmod::Mode mode = arbitmod::Mode::Add;
    bool enabled = true;
};

inline bool supportedKind(const std::string& kind)
{
    return kind == "source" || kind == "sink" || kind == "control.const"
        || kind == "control.random.step" || kind == "score.root.freq"
        || kind == "control.curve" || kind == "control.math" || kind == "control.mix"
        || kind == "control.slew" || kind == "control.history"
        || kind == "control.compare" || kind == "control.logic" || kind == "control.map"
        || kind == "control.samplehold" || kind == "control.quantize"
        || kind == "control.function" || kind == "control.stepseq"
        || kind == "control.trigger" || kind == "score.harmonic.complexity"
        || kind == "score.note.match";
}

inline bool validatePlan(const Plan& plan, std::string& error)
{
    if (plan.version != 1)
    {
        error = "unsupported video control plan version";
        return false;
    }
    if (plan.numSlots < 0 || static_cast<std::size_t>(plan.numSlots) > kMaxSlots)
    {
        error = "video control plan slot count exceeds capacity";
        return false;
    }
    if (plan.operations.size() > kMaxOperations)
    {
        error = "video control plan operation count exceeds capacity";
        return false;
    }

    std::size_t sinks = 0;
    std::vector<bool> produced(static_cast<std::size_t>(plan.numSlots), false);
    for (const auto& op : plan.operations)
    {
        if (!supportedKind(op.kind))
        {
            error = "unsupported video control operation kind";
            return false;
        }
        if (op.inputs.size() > kMaxInputsPerOperation
            || op.params.size() > kMaxParamsPerOperation)
        {
            error = "video control plan operation exceeds input or parameter capacity";
            return false;
        }
        for (const auto& fanIn : op.inputs)
        {
            if (fanIn.size() > kMaxSourcesPerInput)
            {
                error = "video control plan fan-in exceeds capacity";
                return false;
            }
            for (const int slot : fanIn)
            {
                if (slot < 0 || slot >= plan.numSlots)
                {
                    error = "video control plan input slot is out of range";
                    return false;
                }
                if (!produced[static_cast<std::size_t>(slot)])
                {
                    error = "video control plan input is not topologically available";
                    return false;
                }
            }
        }
        for (const float parameter : op.params)
            if (!std::isfinite(parameter))
            {
                error = "video control plan parameter is not finite";
                return false;
            }
        for (const int slot : op.outputSlots)
        {
            if (slot < 0 || slot >= plan.numSlots)
            {
                error = "video control plan output slot is out of range";
                return false;
            }
            if (produced[static_cast<std::size_t>(slot)])
            {
                error = "video control plan output slot has multiple producers";
                return false;
            }
            produced[static_cast<std::size_t>(slot)] = true;
        }
        if (op.kind == "sink" && ++sinks > kMaxSinks)
        {
            error = "video control plan sink count exceeds capacity";
            return false;
        }
        if (!std::isfinite(op.depth) || !std::isfinite(op.smoothingBeats)
            || (op.kind == "sink" && op.destination.empty()))
        {
            error = "video control plan sink metadata is invalid";
            return false;
        }
        const bool anyStructuredTarget = op.targetClipId >= 0 || op.targetNodeId >= 0
            || !op.targetParamId.empty();
        if (op.kind == "sink" && anyStructuredTarget
            && (op.targetClipId < 0 || op.targetNodeId < 0 || op.targetParamId.empty()))
        {
            error = "video control plan structured sink identity is incomplete";
            return false;
        }
    }
    return true;
}

class Executor
{
public:
    bool bind(const Plan& plan, std::string& error)
    {
        if (!validatePlan(plan, error))
            return false;
        plan_ = plan;
        slots_.assign(static_cast<std::size_t>(plan_.numSlots), 0.0f);
        states_.assign(plan_.operations.size(), {});
        results_.reserve(kMaxSinks);
        reset();
        return true;
    }

    void reset() noexcept
    {
        std::fill(slots_.begin(), slots_.end(), 0.0f);
        for (auto& state : states_) state = {};
        results_.clear();
    }

    bool empty() const noexcept { return plan_.operations.empty(); }

    const std::vector<SinkValue>& evaluate(const arbitmod::Score& score,
                                           const arbitmod::Clock& clock,
                                           const arbitmod::Audio& audio,
                                           float dtBeats, float dtSeconds)
    {
        results_.clear();
        std::fill(slots_.begin(), slots_.end(), 0.0f);

        for (std::size_t index = 0; index < plan_.operations.size(); ++index)
        {
            const auto& op = plan_.operations[index];
            auto& state = states_[index];
            float in[16] = {};
            const std::size_t inputCount = std::min<std::size_t>(op.inputs.size(), 16);
            for (std::size_t input = 0; input < inputCount; ++input)
                for (const int slot : op.inputs[input])
                    in[input] += slots_[static_cast<std::size_t>(slot)];

            const auto param = [&op](std::size_t i, float fallback = 0.0f)
            {
                return i < op.params.size() && std::isfinite(op.params[i]) ? op.params[i] : fallback;
            };
            float value = 0.0f;

            if (op.kind == "source")
            {
                value = arbitmod::evaluateSource(op.source, score, clock, audio);
            }
            else if (op.kind == "control.const")
            {
                value = arbitmod::clamp01(param(0));
            }
            else if (op.kind == "control.random.step")
            {
                const double rate = std::max(static_cast<double>(param(0, 0.0625f)), 0.0625);
                const auto step = static_cast<std::int64_t>(std::floor(clock.beat * rate));
                const auto seed = static_cast<std::uint32_t>(std::lround(param(1)));
                std::uint32_t bits = static_cast<std::uint32_t>(step) ^ (seed * 0x9e3779b9u);
                bits ^= bits >> 16; bits *= 0x7feb352du;
                bits ^= bits >> 15; bits *= 0x846ca68bu; bits ^= bits >> 16;
                value = static_cast<float>(bits >> 8) / 16777215.0f;
            }
            else if (op.kind == "score.root.freq")
            {
                constexpr double low = 27.5, high = 4186.009;
                value = score.rootFreq > 0.0f
                    ? arbitmod::clamp01(static_cast<float>((std::log2(score.rootFreq)
                        - std::log2(low)) / (std::log2(high) - std::log2(low)))) : 0.0f;
            }
            else if (op.kind == "control.curve")
            {
                const float x = arbitmod::clamp01(in[0]);
                const float amount = std::clamp(param(0), -1.0f, 1.0f);
                const float shape = arbitmod::clamp01(param(1));
                const float gamma = std::pow(4.0f, amount);
                const float curved = std::pow(x, gamma);
                const float smooth = x * x * (3.0f - 2.0f * x);
                value = arbitmod::clamp01(curved + shape * (smooth - curved));
            }
            else if (op.kind == "control.math")
            {
                switch (static_cast<int>(std::lround(param(0))))
                {
                    case 1: value = in[0] - in[1]; break;
                    case 2: value = in[0] * in[1]; break;
                    case 3: value = std::min(in[0], in[1]); break;
                    case 4: value = std::max(in[0], in[1]); break;
                    case 5: value = 0.5f * (in[0] + in[1]); break;
                    default: value = in[0] + in[1]; break;
                }
            }
            else if (op.kind == "control.mix")
            {
                const float balance = arbitmod::clamp01(param(0));
                value = in[0] * (1.0f - balance) + in[1] * balance;
            }
            else if (op.kind == "control.slew")
            {
                const float target = arbitmod::clamp01(in[0]);
                if (!state.primed)
                {
                    state.value = target;
                    state.primed = true;
                }
                const float timeMs = std::max(0.0f, param(target >= state.value ? 0 : 1));
                const float maxDelta = timeMs <= 0.0f ? 1.0f
                    : std::max(0.0f, dtSeconds) / (timeMs * 0.001f);
                state.value += std::clamp(target - state.value, -maxDelta, maxDelta);
                value = arbitmod::clamp01(state.value);
            }
            else if (op.kind == "control.history")
            {
                if (!state.primed)
                {
                    state.value = arbitmod::clamp01(param(0));
                    state.primed = true;
                }
                value = state.value;
                state.value = arbitmod::clamp01(in[0]);
            }
            else if (op.kind == "control.compare")
            {
                const int operation = static_cast<int>(std::lround(param(0)));
                const bool result = operation == 1 ? in[0] < in[1]
                    : operation == 2 ? std::abs(in[0] - in[1]) <= 1.0e-4f : in[0] > in[1];
                value = result ? 1.0f : 0.0f;
            }
            else if (op.kind == "control.logic")
            {
                const bool a = in[0] > 0.5f, b = in[1] > 0.5f;
                const int operation = static_cast<int>(std::lround(param(0)));
                const bool result = operation == 1 ? (a || b)
                    : operation == 2 ? (a != b) : operation == 3 ? !a : (a && b);
                value = result ? 1.0f : 0.0f;
            }
            else if (op.kind == "control.map")
            {
                const float denominator = std::abs(param(1) - param(0)) < 1.0e-6f
                    ? 1.0f : param(1) - param(0);
                const float t = std::clamp((in[0] - param(0)) / denominator, 0.0f, 1.0f);
                value = param(2) + t * (param(3) - param(2));
            }
            else if (op.kind == "control.samplehold")
            {
                const bool gate = in[1] > 0.5f;
                if (gate && !state.flag) state.value = in[0];
                state.flag = gate;
                value = state.value;
            }
            else if (op.kind == "control.quantize")
            {
                const int steps = std::clamp(static_cast<int>(std::lround(param(0))), 1, 128);
                value = std::round(arbitmod::clamp01(in[0]) * steps) / static_cast<float>(steps);
            }
            else if (op.kind == "control.function")
            {
                value = std::pow(arbitmod::clamp01(in[0]),
                                 std::exp2(std::clamp(param(0), -1.0f, 1.0f) * 3.0f));
            }
            else if (op.kind == "control.stepseq")
            {
                const int count = std::clamp(static_cast<int>(std::lround(param(0))), 1, 8);
                const float rate = std::max(param(1), 0.0625f);
                const int step = (static_cast<int>(std::floor(clock.beat * rate)) % count + count) % count;
                value = arbitmod::clamp01(param(static_cast<std::size_t>(2 + step)));
            }
            else if (op.kind == "control.trigger")
            {
                const bool gate = in[0] > 0.5f;
                const int divisor = std::clamp(static_cast<int>(std::lround(param(0))), 1, 64);
                value = 0.0f;
                if (gate && !state.flag)
                {
                    ++state.count;
                    if (state.count % divisor == 0) value = 1.0f;
                }
                state.flag = gate;
            }
            else if (op.kind == "score.harmonic.complexity")
            {
                float sum = 0.0f;
                for (const auto& link : score.links)
                    sum += static_cast<float>(std::log2(
                        std::max(1, link.slaveHarmonic) * std::max(1, link.masterHarmonic)));
                value = arbitmod::clamp01(sum / 32.0f);
            }
            else if (op.kind == "score.note.match")
            {
                const int wanted = static_cast<int>(std::lround(param(0)));
                value = 0.0f;
                for (const auto& note : score.notes)
                    if (note.midiNote == wanted) { value = 1.0f; break; }
            }
            else if (op.kind == "sink")
            {
                float shaped = in[0];
                if (op.curve != arbitmod::Curve::Linear)
                    shaped = arbitmod::applyCurve(op.curve, arbitmod::clamp01(std::abs(shaped)))
                        * (shaped < 0.0f ? -1.0f : 1.0f);
                float routed = op.depth * shaped;
                if (op.smoothingBeats > 0.0f)
                {
                    if (!state.primed) { state.value = routed; state.primed = true; }
                    else state.value = arbitmod::onePoleBeats(
                        state.value, routed, dtBeats, op.smoothingBeats);
                    routed = state.value;
                }
                else
                {
                    state.value = routed;
                    state.primed = true;
                }
                results_.push_back({ op.destination, routed, op.mode, op.enabled });
                continue;
            }

            if (!std::isfinite(value)) value = 0.0f;
            for (const int slot : op.outputSlots)
                slots_[static_cast<std::size_t>(slot)] = value;
        }
        return results_;
    }

private:
    struct OperationState
    {
        float value = 0.0f;
        bool primed = false;
        bool flag = false;
        int count = 0;
    };

    Plan plan_;
    std::vector<float> slots_;
    std::vector<OperationState> states_;
    std::vector<SinkValue> results_;
};

} // namespace videocontrol
