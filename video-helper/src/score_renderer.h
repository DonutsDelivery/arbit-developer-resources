#pragma once

#include "mod_defs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace videorender
{
struct ScoreRenderResult
{
    std::vector<uint8_t> rgba;
    int width = 0;
    int height = 0;
};

struct ScoreClock
{
    float beat = 0.0f;
    float beatsPerBar = 4.0f;
};

namespace scoredetail
{
struct Canvas
{
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;

    Canvas(int w, int h, std::array<uint8_t, 4> background)
        : width(w), height(h), pixels(static_cast<size_t>(w * h * 4), 0)
    {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const auto offset = static_cast<size_t>((y * w + x) * 4);
                for (int c = 0; c < 4; ++c) pixels[offset + static_cast<size_t>(c)] = background[c];
            }
    }

    void pixel(int x, int y, std::array<uint8_t, 4> color)
    {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const auto offset = static_cast<size_t>((y * width + x) * 4);
        const int alpha = color[3];
        for (int c = 0; c < 3; ++c)
            pixels[offset + static_cast<size_t>(c)] = static_cast<uint8_t>(
                (color[c] * alpha + pixels[offset + static_cast<size_t>(c)] * (255 - alpha)) / 255);
        pixels[offset + 3] = 255;
    }

    void line(int x0, int y0, int x1, int y1, std::array<uint8_t, 4> color, int thickness = 1)
    {
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for (;;)
        {
            for (int oy = -thickness / 2; oy <= thickness / 2; ++oy)
                for (int ox = -thickness / 2; ox <= thickness / 2; ++ox)
                    pixel(x0 + ox, y0 + oy, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice = 2 * error;
            if (twice >= dy) { error += dy; x0 += sx; }
            if (twice <= dx) { error += dx; y0 += sy; }
        }
    }

    void ellipse(int cx, int cy, int rx, int ry, std::array<uint8_t, 4> color, bool filled)
    {
        for (int y = -ry; y <= ry; ++y)
            for (int x = -rx; x <= rx; ++x)
            {
                const float value = static_cast<float>(x * x) / std::max(1, rx * rx)
                                  + static_cast<float>(y * y) / std::max(1, ry * ry);
                if ((filled && value <= 1.0f) || (!filled && value >= 0.72f && value <= 1.18f))
                    pixel(cx + x, cy + y, color);
            }
    }
};

inline std::array<uint8_t, 4> trackColor(int track, bool root)
{
    static constexpr std::array<std::array<uint8_t, 4>, 8> colors {{
        {{230,230,234,255}}, {{143,212,255,255}}, {{181,232,160,255}}, {{255,179,193,255}},
        {{216,180,254,255}}, {{255,213,158,255}}, {{158,231,227,255}}, {{246,166,255,255}}
    }};
    return root ? std::array<uint8_t, 4>{{255,196,64,255}}
                : colors[static_cast<size_t>((track % 8 + 8) % 8)];
}

inline void accidental(Canvas& canvas, int x, int y, int value, std::array<uint8_t, 4> color, int scale)
{
    if (value > 0)
    {
        canvas.line(x - scale, y - 2 * scale, x - scale, y + 2 * scale, color);
        canvas.line(x + scale, y - 2 * scale, x + scale, y + 2 * scale, color);
        canvas.line(x - 2 * scale, y - scale, x + 2 * scale, y - scale, color);
        canvas.line(x - 2 * scale, y + scale, x + 2 * scale, y + scale, color);
    }
    else if (value < 0)
    {
        canvas.line(x - scale, y - 3 * scale, x - scale, y + 2 * scale, color);
        canvas.ellipse(x, y + scale, scale + 1, scale, color, false);
    }
}

inline void comma(Canvas& canvas, int x, int y, int prime, int exponent,
                  std::array<uint8_t, 4> color, int scale)
{
    const bool up = exponent > 0;
    canvas.line(x, y - 2 * scale, x, y + 2 * scale, color, std::max(1, scale / 2));
    const int tip = y + (up ? -3 * scale : 3 * scale);
    const int base = y + (up ? -scale : scale);
    canvas.line(x, tip, x - scale, base, color);
    canvas.line(x, tip, x + scale, base, color);
    if (prime != 5)
    {
        const int marks = prime >= 10 ? 2 : 1;
        for (int i = 0; i < marks; ++i)
            canvas.line(x + (i + 1) * scale, y - scale, x + (i + 1) * scale, y + scale,
                        color, std::max(1, scale / 2));
    }
    if (std::abs(exponent) > 1)
        canvas.line(x - scale, y + 3 * scale, x + scale, y + 3 * scale, color);
}
} // namespace scoredetail

inline ScoreRenderResult renderScore(const arbitmod::Score& score, const ScoreClock& clock,
                                     int width, int height,
                                     const std::map<std::string, double>& params)
{
    ScoreRenderResult result;
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384) return result;
    const auto value = [&params](const char* name, double fallback)
    {
        const auto found = params.find(name);
        return found == params.end() ? fallback : found->second;
    };
    const float history = std::max(0.0f, static_cast<float>(value("historyBeats", 2.0)));
    const float lookahead = std::max(0.25f, static_cast<float>(value("lookaheadBeats", 10.0)));
    const float span = history + lookahead;
    const float staffScale = std::clamp(static_cast<float>(value("staffScale", 1.0)), 0.5f, 2.0f);
    const int staffSpace = std::max(6, static_cast<int>(std::lround(height / 32.0f * staffScale)));
    const int middleY = height / 2;
    scoredetail::Canvas canvas(width, height, {{5, 6, 12, 255}});
    const auto staffColor = std::array<uint8_t, 4>{{190, 194, 210, 120}};
    const auto beatX = [&](float beat)
    {
        return static_cast<int>(std::lround((beat - (clock.beat - history)) / span * width));
    };
    const auto pitchY = [&](int diatonic)
    {
        return middleY - static_cast<int>(std::lround((diatonic - 28) * staffSpace * 0.5f));
    };

    for (int index : {30,32,34,36,38,26,24,22,20,18})
        canvas.line(0, pitchY(index), width - 1, pitchY(index), staffColor);
    const int beatsPerBar = std::max(1, static_cast<int>(std::lround(clock.beatsPerBar)));
    const int firstBar = static_cast<int>(std::floor((clock.beat - history) / beatsPerBar)) * beatsPerBar;
    for (int beat = firstBar; beat <= clock.beat + lookahead + beatsPerBar; beat += beatsPerBar)
        canvas.line(beatX(static_cast<float>(beat)), pitchY(38), beatX(static_cast<float>(beat)),
                    pitchY(18), {{120,124,145,90}});

    std::vector<const arbitmod::Note*> visible;
    for (const auto& note : score.notes)
        if (note.notationVisible && !note.muted
            && note.startBeat <= clock.beat + lookahead
            && note.endBeat() >= clock.beat - history)
            visible.push_back(&note);
    std::stable_sort(visible.begin(), visible.end(), [](const auto* a, const auto* b)
    {
        if (a->startBeat != b->startBeat) return a->startBeat < b->startBeat;
        if (a->diatonicIndex != b->diatonicIndex) return a->diatonicIndex < b->diatonicIndex;
        return a->id < b->id;
    });

    const int headW = std::max(7, staffSpace + staffSpace / 3);
    const int headH = std::max(5, staffSpace);
    for (const auto* note : visible)
    {
        const int x = beatX(note->startBeat);
        const int y = pitchY(note->diatonicIndex);
        const auto color = scoredetail::trackColor(note->trackId, note->isRoot);
        for (int ledger = 40; ledger <= note->diatonicIndex; ledger += 2)
            canvas.line(x - headW, pitchY(ledger), x + headW, pitchY(ledger), staffColor);
        for (int ledger = 16; ledger >= note->diatonicIndex; ledger -= 2)
            canvas.line(x - headW, pitchY(ledger), x + headW, pitchY(ledger), staffColor);
        if (note->diatonicIndex >= 27 && note->diatonicIndex <= 29)
            canvas.line(x - headW, pitchY(28), x + headW, pitchY(28), staffColor);
        int accidentalX = x - headW;
        if (note->baseAccidental != 0)
        {
            scoredetail::accidental(canvas, accidentalX, y, note->baseAccidental, color,
                                    std::max(2, staffSpace / 4));
            accidentalX -= 2 * staffSpace;
        }
        for (int i = 0; i < note->commaCount; ++i)
        {
            const auto& comma = note->commas[static_cast<size_t>(i)];
            scoredetail::comma(canvas, accidentalX, y, comma.prime, comma.exponent, color,
                               std::max(2, staffSpace / 4));
            accidentalX -= 2 * staffSpace;
        }
        if (note->edoActive && note->edoInflection != 0)
        {
            const bool up = note->edoInflection > 0;
            const int count = std::min(3, std::abs(note->edoInflection));
            for (int i = 0; i < count; ++i)
            {
                const int cy = y + (i - count / 2) * staffSpace;
                canvas.line(accidentalX - staffSpace / 2, cy + (up ? staffSpace / 2 : -staffSpace / 2),
                            accidentalX, cy + (up ? -staffSpace / 2 : staffSpace / 2), color);
                canvas.line(accidentalX, cy + (up ? -staffSpace / 2 : staffSpace / 2),
                            accidentalX + staffSpace / 2, cy + (up ? staffSpace / 2 : -staffSpace / 2), color);
            }
        }
        canvas.ellipse(x, y, headW / 2, headH / 2, color, note->lengthBeats < 2.0f);
        if (note->lengthBeats < 4.0f)
            canvas.line(x + headW / 2, y, x + headW / 2, y - 3 * staffSpace, color, 2);
    }

    const int playhead = beatX(clock.beat);
    canvas.line(playhead, 0, playhead, height - 1, {{255,164,58,170}}, 2);
    result.rgba = std::move(canvas.pixels);
    result.width = width;
    result.height = height;
    return result;
}
} // namespace videorender
