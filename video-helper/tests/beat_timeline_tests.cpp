#include "beat_timeline.h"

#include <cmath>
#include <iostream>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void near(double actual, double expected, double tolerance, const char* message)
{
    check(std::abs(actual - expected) <= tolerance, message);
}
}

int main()
{
    videotime::BeatTimeline timeline;
    timeline.set({ { 0.0, 60.0, true }, { 4.0, 120.0, true }, { 8.0, 240.0, true } },
                 { { 0.0, 4, 4 }, { 6.0, 3, 4 } });

    near(timeline.beatToSeconds(4.0), 4.0, 1.0e-9, "step segment beat to seconds");
    near(timeline.beatToSeconds(8.0), 6.0, 1.0e-9, "second step segment beat to seconds");
    near(timeline.secondsToBeat(5.0), 6.0, 1.0e-9, "seconds to beat across step change");
    near(timeline.clockAtSeconds(5.0).bpm, 120.0, 1.0e-9, "tempo at mapped second");
    near(timeline.clockAtSeconds(5.0).beatsPerBar, 3.0, 1.0e-9, "meter at mapped beat");
    near(timeline.clockAtSeconds(5.0).barPhase, 0.0, 1.0e-9, "meter marker resets bar phase");

    timeline.set({ { 0.0, 60.0, false }, { 4.0, 120.0, true } },
                 { { 0.0, 7, 8 } });
    const double rampEndSeconds = timeline.beatToSeconds(4.0);
    near(rampEndSeconds, 4.0 * std::log(2.0), 1.0e-9, "linear ramp integral");
    near(timeline.secondsToBeat(rampEndSeconds), 4.0, 1.0e-9, "linear ramp inverse");
    near(timeline.clockAtSeconds(rampEndSeconds).bpm, 120.0, 1.0e-9, "ramp endpoint bpm");
    near(timeline.clockAtSeconds(rampEndSeconds).beatsPerBar, 3.5, 1.0e-9, "compound meter length");

    const double frameA = timeline.secondsToBeat(1.0);
    const double frameB = timeline.secondsToBeat(1.0 + 1.0 / 30.0);
    check(frameB > frameA, "frame beat delta remains positive through ramp");
    near(timeline.beatToSeconds(timeline.secondsToBeat(2.345)), 2.345, 1.0e-9,
         "seconds beat round trip");

    if (failures == 0)
        std::cout << "beat timeline: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
