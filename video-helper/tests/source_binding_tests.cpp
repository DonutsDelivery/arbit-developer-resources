#include "../src/source_binding.h"

#include <cstdio>
#include <vector>

struct Segment
{
    int clipId = 0;
    int trackLayer = 0;
    double inSec = 0.0;
    double outSec = 1.0;
    double rate = 1.0;
    double displayStartSec = 0.0;
    int transitionFromClipId = -1;
    int transitionToClipId = -1;
};

int main()
{
    int failures = 0;
    auto check = [&failures] (bool condition, const char* message)
    {
        if (! condition)
        {
            ++failures;
            std::fprintf (stderr, "FAIL: %s\n", message);
        }
    };

    using videowire::SourceKind;
    using videowire::resolveSourceKind;

    check (resolveSourceKind (SourceKind::Media, true, "gen://adjustment")
               == SourceKind::Media,
           "typed media overrides adjustment compatibility fields");
    check (resolveSourceKind (SourceKind::Shader, false, "/tmp/movie.mov")
               == SourceKind::Shader,
           "typed shader overrides media-looking path");
    check (resolveSourceKind (SourceKind::Unspecified, true, "/tmp/movie.mov")
               == SourceKind::Adjustment,
           "structured legacy adjustment precedes path fallback");
    check (resolveSourceKind (SourceKind::Unspecified, false, "gen://particles")
               == SourceKind::Particles,
           "legacy particle sentinel remains readable");
    check (resolveSourceKind (SourceKind::Unspecified, false, "gen://shader")
               == SourceKind::Shader,
           "legacy shader sentinel remains readable");
    check (resolveSourceKind (SourceKind::Unspecified, false, "/tmp/movie.mov")
               == SourceKind::Media,
           "ordinary legacy path resolves to media");
    check (videowire::sourceKindFromWire ("future") == SourceKind::Unspecified,
           "unknown wire values degrade to legacy fallback");

    std::vector<Segment> segments {
        { 10, 0, 0.0, 3.0, 1.0, 0.0, -1, -1 },
        { 20, 0, 0.0, 3.0, 1.0, 2.0, 10, 20 },
        { 30, 1, 0.0, 3.0, 1.0, 1.0, -1, -1 },
    };
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[0],
           "explicit transition A wins across timeline candidates");
    segments[1].transitionFromClipId = 30;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[2],
           "explicit transition A is not constrained by track ordering");
    segments[1].transitionToClipId = 99;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == nullptr,
           "conflicting explicit B fails closed instead of inferring");
    segments[1].transitionFromClipId = -1;
    segments[1].transitionToClipId = -1;
    check (videowire::resolveTransitionFrom (segments, segments[1]) == &segments[0],
           "legacy transition falls back to previous overlapping track segment");

    std::printf ("source-binding: %d/11 checks passed\n", 11 - failures);
    return failures;
}
