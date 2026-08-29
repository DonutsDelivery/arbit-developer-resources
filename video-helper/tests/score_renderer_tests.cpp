#include "../src/score_renderer.h"

#include <cstdio>

int main()
{
    int failures = 0;
    auto check = [&](bool condition, const char* message)
    {
        if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
    };

    arbitmod::Score score;
    score.edoStepsPerOctave = 31;
    arbitmod::Note root;
    root.id = 1; root.startBeat = 0.0f; root.lengthBeats = 4.0f;
    root.diatonicIndex = 28; root.isRoot = true;
    arbitmod::Note third;
    third.id = 2; third.startBeat = 1.0f; third.lengthBeats = 2.0f;
    third.diatonicIndex = 30; third.baseAccidental = 1; third.linked = true;
    third.commas[0] = { 5, -1 }; third.commaCount = 1;
    arbitmod::Note edo;
    edo.id = 3; edo.startBeat = 2.0f; edo.lengthBeats = 1.0f;
    edo.diatonicIndex = 32; edo.edoActive = true; edo.edoInflection = 2; edo.edoDegree = 17;
    score.notes = { root, third, edo };

    videorender::ScoreClock clock;
    clock.beat = 1.5f; clock.beatsPerBar = 4.0f;
    std::map<std::string, double> params {
        { "historyBeats", 2.0 }, { "lookaheadBeats", 6.0 }, { "staffScale", 1.0 }
    };
    const auto first = videorender::renderScore(score, clock, 640, 360, params);
    const auto repeated = videorender::renderScore(score, clock, 640, 360, params);
    check(first.rgba.size() == 640u * 360u * 4u, "score render has bounded RGBA output");
    check(first.rgba == repeated.rgba, "same score clock and size are deterministic");
    size_t lit = 0;
    for (size_t i = 0; i + 3 < first.rgba.size(); i += 4)
        if (first.rgba[i] > 20 || first.rgba[i + 1] > 20 || first.rgba[i + 2] > 20) ++lit;
    check(lit > 1000, "score render contains notation pixels");
    clock.beat = 2.5f;
    const auto moved = videorender::renderScore(score, clock, 640, 360, params);
    check(first.rgba != moved.rgba, "project beat moves the score window");

    std::printf("score-renderer: %d/4 checks passed\n", 4 - failures);
    return failures;
}
