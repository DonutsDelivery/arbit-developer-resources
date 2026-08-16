#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace videotime
{

struct TempoPoint
{
    double beat = 0.0;
    double bpm = 120.0;
    bool step = false;
};

struct MeterPoint
{
    double beat = 0.0;
    int numerator = 4;
    int denominator = 4;
};

struct Clock
{
    double beat = 0.0;
    double bpm = 120.0;
    double beatsPerBar = 4.0;
    double barPhase = 0.0;
};

class BeatTimeline
{
public:
    BeatTimeline()
    {
        reset(120.0, 4.0);
    }

    void reset(double bpm, double beatsPerBar)
    {
        tempoPoints_.clear();
        tempoPoints_.push_back({ 0.0, sanitizeBpm(bpm), true });
        const int numerator = std::max(1, static_cast<int>(std::llround(
            std::isfinite(beatsPerBar) ? beatsPerBar : 4.0)));
        meterPoints_.clear();
        meterPoints_.push_back({ 0.0, numerator, 4 });
    }

    void set(std::vector<TempoPoint> tempoPoints, std::vector<MeterPoint> meterPoints)
    {
        tempoPoints_ = std::move(tempoPoints);
        meterPoints_ = std::move(meterPoints);
        normalizeTempo();
        normalizeMeter();
    }

    const std::vector<TempoPoint>& tempoPoints() const noexcept { return tempoPoints_; }
    const std::vector<MeterPoint>& meterPoints() const noexcept { return meterPoints_; }

    double tempoAtBeat(double beat) const noexcept
    {
        if (tempoPoints_.empty())
            return 120.0;
        if (! std::isfinite(beat) || beat <= tempoPoints_.front().beat)
            return tempoPoints_.front().bpm;
        if (beat >= tempoPoints_.back().beat)
            return tempoPoints_.back().bpm;

        const auto next = std::upper_bound(tempoPoints_.begin(), tempoPoints_.end(), beat,
            [](double value, const TempoPoint& point) { return value < point.beat; });
        const auto& previous = *(next - 1);
        if (previous.step)
            return previous.bpm;
        const double length = next->beat - previous.beat;
        if (length <= 0.0)
            return previous.bpm;
        const double fraction = (beat - previous.beat) / length;
        return previous.bpm + fraction * (next->bpm - previous.bpm);
    }

    double beatToSeconds(double beat) const noexcept
    {
        if (! std::isfinite(beat) || beat <= 0.0)
            return 0.0;
        if (tempoPoints_.empty())
            return beat * 0.5;

        double seconds = 0.0;
        for (std::size_t i = 0; i + 1 < tempoPoints_.size(); ++i)
        {
            const auto& current = tempoPoints_[i];
            const auto& next = tempoPoints_[i + 1];
            const double segmentEnd = std::min(beat, next.beat);
            if (segmentEnd > current.beat)
            {
                const double fullLength = next.beat - current.beat;
                const double partialLength = segmentEnd - current.beat;
                const double partialEndBpm = current.step ? current.bpm
                    : current.bpm + (next.bpm - current.bpm) * partialLength / fullLength;
                seconds += segmentSeconds(partialLength, current.bpm, partialEndBpm, current.step);
            }
            if (beat <= next.beat)
                return seconds;
        }

        const auto& last = tempoPoints_.back();
        if (beat > last.beat)
            seconds += (beat - last.beat) * 60.0 / last.bpm;
        return seconds;
    }

    double secondsToBeat(double seconds) const noexcept
    {
        if (! std::isfinite(seconds) || seconds <= 0.0)
            return 0.0;
        if (tempoPoints_.empty())
            return seconds * 2.0;

        double elapsed = 0.0;
        for (std::size_t i = 0; i + 1 < tempoPoints_.size(); ++i)
        {
            const auto& current = tempoPoints_[i];
            const auto& next = tempoPoints_[i + 1];
            const double beatLength = next.beat - current.beat;
            const double duration = segmentSeconds(beatLength, current.bpm, next.bpm, current.step);
            if (seconds <= elapsed + duration)
                return current.beat + segmentBeatForSeconds(
                    seconds - elapsed, beatLength, current.bpm, next.bpm, current.step);
            elapsed += duration;
        }

        const auto& last = tempoPoints_.back();
        return last.beat + (seconds - elapsed) * last.bpm / 60.0;
    }

    MeterPoint meterAtBeat(double beat) const noexcept
    {
        if (meterPoints_.empty())
            return {};
        const auto next = std::upper_bound(meterPoints_.begin(), meterPoints_.end(), beat,
            [](double value, const MeterPoint& point) { return value < point.beat; });
        return next == meterPoints_.begin() ? meterPoints_.front() : *(next - 1);
    }

    Clock clockAtSeconds(double seconds) const noexcept
    {
        Clock result;
        result.beat = secondsToBeat(seconds);
        result.bpm = tempoAtBeat(result.beat);
        const auto meter = meterAtBeat(result.beat);
        result.beatsPerBar = static_cast<double>(meter.numerator) * 4.0
                           / static_cast<double>(meter.denominator);
        const double bars = result.beatsPerBar > 0.0
            ? (result.beat - meter.beat) / result.beatsPerBar : 0.0;
        result.barPhase = bars - std::floor(bars);
        return result;
    }

private:
    static double sanitizeBpm(double bpm) noexcept
    {
        return std::clamp(std::isfinite(bpm) ? bpm : 120.0, 20.0, 999.0);
    }

    static int sanitizeDenominator(int denominator) noexcept
    {
        if (denominator <= 1) return 1;
        if (denominator <= 2) return 2;
        if (denominator <= 4) return 4;
        if (denominator <= 8) return 8;
        if (denominator <= 16) return 16;
        return 32;
    }

    static double segmentSeconds(double beatLength, double bpmA, double bpmB, bool step) noexcept
    {
        if (beatLength <= 0.0)
            return 0.0;
        if (step || std::abs(bpmB - bpmA) < 1.0e-12)
            return beatLength * 60.0 / bpmA;
        return 60.0 * beatLength * std::log(bpmB / bpmA) / (bpmB - bpmA);
    }

    static double segmentBeatForSeconds(double seconds, double beatLength,
                                        double bpmA, double bpmB, bool step) noexcept
    {
        if (seconds <= 0.0 || beatLength <= 0.0)
            return 0.0;
        if (step || std::abs(bpmB - bpmA) < 1.0e-12)
            return std::min(beatLength, seconds * bpmA / 60.0);
        const double slope = (bpmB - bpmA) / beatLength;
        const double offset = bpmA * std::expm1(seconds * slope / 60.0) / slope;
        return std::clamp(offset, 0.0, beatLength);
    }

    void normalizeTempo()
    {
        tempoPoints_.erase(std::remove_if(tempoPoints_.begin(), tempoPoints_.end(),
            [](const auto& point) { return ! std::isfinite(point.beat) || point.beat < 0.0; }),
            tempoPoints_.end());
        for (auto& point : tempoPoints_)
            point.bpm = sanitizeBpm(point.bpm);
        std::stable_sort(tempoPoints_.begin(), tempoPoints_.end(),
            [](const auto& a, const auto& b) { return a.beat < b.beat; });
        tempoPoints_.erase(std::unique(tempoPoints_.begin(), tempoPoints_.end(),
            [](const auto& a, const auto& b) { return std::abs(a.beat - b.beat) < 1.0e-9; }),
            tempoPoints_.end());
        if (tempoPoints_.empty())
            tempoPoints_.push_back({ 0.0, 120.0, true });
        else if (tempoPoints_.front().beat > 1.0e-9)
            tempoPoints_.insert(tempoPoints_.begin(), { 0.0, tempoPoints_.front().bpm, true });
        else
            tempoPoints_.front().beat = 0.0;
    }

    void normalizeMeter()
    {
        meterPoints_.erase(std::remove_if(meterPoints_.begin(), meterPoints_.end(),
            [](const auto& point) { return ! std::isfinite(point.beat) || point.beat < 0.0; }),
            meterPoints_.end());
        for (auto& point : meterPoints_)
        {
            point.numerator = std::clamp(point.numerator, 1, 32);
            point.denominator = sanitizeDenominator(point.denominator);
        }
        std::stable_sort(meterPoints_.begin(), meterPoints_.end(),
            [](const auto& a, const auto& b) { return a.beat < b.beat; });
        meterPoints_.erase(std::unique(meterPoints_.begin(), meterPoints_.end(),
            [](const auto& a, const auto& b) { return std::abs(a.beat - b.beat) < 1.0e-9; }),
            meterPoints_.end());
        if (meterPoints_.empty())
            meterPoints_.push_back({ 0.0, 4, 4 });
        else if (meterPoints_.front().beat > 1.0e-9)
            meterPoints_.insert(meterPoints_.begin(), { 0.0,
                meterPoints_.front().numerator, meterPoints_.front().denominator });
        else
            meterPoints_.front().beat = 0.0;
    }

    std::vector<TempoPoint> tempoPoints_;
    std::vector<MeterPoint> meterPoints_;
};

} // namespace videotime
