// mod_defs.h — Arbit Media Machine modulation engine core (pre-work A1)
//
// Pure, dependency-free C++17: no GL, no JUCE, no protocol, no I/O. This is the
// math behind the cross-domain modulation matrix (generative-visuals-research.md
// §5) — the engine that maps musical signals (note pitch/velocity/triggers, JI
// ratios, BPM-synced LFOs, retriggered ADSR envelopes) onto visual parameters.
//
// It is built and unit-tested in isolation BEFORE the helper integration exists
// (the workbench precedent). At integration time this header is destined for
// `video-helper/src/mod_defs.h` as the canonical definition shared by the Arbit
// plugin and the video-helper — the `effect_defs.h` pattern. It carries no GPL
// code, so it is safe under the CI gpl-hygiene scan.
//
// Load-bearing conventions (each has a unit test in tests/mod_tests.cpp):
//   - beatingRate is oriented by the LINK, never note order:
//       |f_slave*masterHarmonic - f_master*slaveHarmonic*2^octaveTranspose|
//     == 0 exactly for a pure JI link (HarmonicEngine.h:2407 orientation).
//   - smoothing is a one-pole filter specified in BEATS, so it is identical
//     under any BPM and any frame rate (beat-time, not milliseconds).
//   - the ADSR is a pure closed-form function of (beats-since-trigger, gate
//     length), reproducing VAModulation.h's per-segment exponential bend
//     f(u) = (1 - e^(-k u)) / (1 - e^(-k)), k = ±curve*CURVE_STRENGTH.
//   - generators (ADSR, LFO incl. seeded sample&hold) are pure functions of
//     beat time → bit-identical in the live loop and in export (parity free).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace arbitmod
{

// ---------------------------------------------------------------------------
// Constants (ported from VAModulation.h so live synth and visual envelopes
// share a shape).
// ---------------------------------------------------------------------------
constexpr float kCurveStrength = 10.0f;   // |k| at full bend (VAModulation CURVE_STRENGTH)
constexpr float kPi            = 3.14159265358979323846f;
constexpr int   kMaxSlots      = 16;       // mirrors the Sybil-16 matrix / NUM_MOD_SLOTS

// ---------------------------------------------------------------------------
// Tiny pure helpers
// ---------------------------------------------------------------------------
inline float clampf (float v, float lo, float hi) { return std::max (lo, std::min (hi, v)); }
inline float clamp01 (float v) { return clampf (v, 0.0f, 1.0f); }
inline float fracf (float v) { return v - std::floor (v); }
inline float lerpf (float a, float b, float t) { return a + (b - a) * t; }

// 32-bit integer hash → [-1, 1]; deterministic, used for seeded Sample&Hold so
// the value depends only on (seed, cycle index), never on call history.
inline float hashToBipolar (uint32_t seed, int32_t cycle)
{
    uint32_t h = seed ^ (static_cast<uint32_t> (cycle) * 2654435761u);
    h ^= h >> 15;
    h *= 1664525u;
    h += 1013904223u;
    h ^= h >> 13;
    return static_cast<float> (static_cast<int32_t> (h)) / 2147483648.0f;
}

// ---------------------------------------------------------------------------
// Score model (minimal). A2 (Block C packer) formalises the texture layout;
// this is the in-memory shape the source evaluators read, matching the
// viewport_set_notes payload fields (§4.3).
// ---------------------------------------------------------------------------
struct Clock
{
    float beat        = 0.0f;
    float bpm         = 120.0f;
    float beatsPerBar = 4.0f;
};

struct PitchBendPoint
{
    float position = 0.0f;
    float semitones = 0.0f;
    float tension = 0.0f;
    float sCurve = 0.0f;
    float vibratoDepthCents = 0.0f;
    float vibratoRateHz = 0.0f;
    int vibratoWaveform = 0;
    float vibratoFadeIn = 0.0f;
    float vibratoFadeOut = 0.0f;
};

struct PitchAnchor
{
    int id = -1;
    float position = 0.0f;
    float frequency = 0.0f;
};

struct Note
{
    struct NotationComma { int prime = 0; int exponent = 0; };
    int   id          = 0;
    int   trackId     = 0;
    float startBeat   = 0.0f;
    float lengthBeats = 1.0f;
    float midiNote    = 60.0f;
    float velocity    = 100.0f;     // 0..127
    float freqHz      = 261.625565f;
    float durationSeconds = 0.0f;
    std::vector<PitchBendPoint> pitchBendPoints;
    std::vector<PitchAnchor> pitchAnchors;
    int   ratioNum    = 1;          // reduced interval vs root
    int   ratioDen    = 1;
    float primes[6]   = {0,0,0,0,0,0};  // exponents of 2,3,5,7,11,13
    int   linkMasterId = -1;        // master note id, or -1
    bool  isRoot      = false;
    float centsOffset = 0.0f;
    int edoStep = -1;
    bool muted = false;
    bool notationVisible = true;
    int diatonicIndex = 28;
    int baseAccidental = 0;
    bool linked = false;
    std::array<NotationComma, 9> commas {};
    int commaCount = 0;
    bool hasUnmappedPrime = false;
    bool edoActive = false;
    int edoInflection = 0;
    int edoDegree = 0;

    float endBeat()  const { return startBeat + lengthBeats; }
    bool  activeAt (float b) const { return b >= startBeat && b < endBeat(); }
    bool  startedBy (float b) const { return startBeat <= b; }
};

inline float shapePitchBendProgress (float progress, float tension, float sCurve)
{
    float t = std::clamp (progress, 0.0f, 1.0f);
    if (sCurve > 0.001f)
    {
        const float k = 1.0f + sCurve * 6.0f;
        t = t <= 0.5f
            ? 0.5f * std::pow (2.0f * t, k)
            : 1.0f - 0.5f * std::pow (2.0f * (1.0f - t), k);
    }
    if (std::fabs (tension) > 0.001f)
    {
        if (tension > 0.0f)
            t = t * (1.0f - tension) + std::pow (t, 3.0f) * tension;
        else
            t = t * (1.0f + tension)
              + (1.0f - std::pow (1.0f - t, 3.0f)) * (-tension);
    }
    return t;
}

inline float pitchBendWave (int waveform, float phase)
{
    if (waveform == 1)
    {
        float cycle = std::fmod (phase / (2.0f * 3.14159265f), 1.0f);
        if (cycle < 0.0f) cycle += 1.0f;
        return 4.0f * std::fabs (cycle - 0.5f) - 1.0f;
    }
    return std::sin (phase);
}

inline float pitchBendEnvelope (float progress, float fadeIn, float fadeOut)
{
    float envelope = 1.0f;
    if (fadeIn > 0.001f && progress < fadeIn)
        envelope = progress / fadeIn;
    if (fadeOut > 0.001f && progress > 1.0f - fadeOut)
        envelope = std::min (envelope, (1.0f - progress) / fadeOut);
    return std::clamp (envelope, 0.0f, 1.0f);
}

inline float interpolatePitchBend (const std::vector<PitchBendPoint>& points,
                                   float position, float noteLengthSeconds = 0.0f)
{
    if (points.empty()) return 0.0f;
    position = std::clamp (position, 0.0f, 1.0f);
    float previousPosition = 0.0f;
    float previousSemitones = points.front().semitones;
    for (const auto& point : points)
    {
        if (position <= point.position)
        {
            float progress = point.position - previousPosition > 0.0001f
                ? (position - previousPosition) / (point.position - previousPosition)
                : 0.0f;
            progress = shapePitchBendProgress (progress, point.tension, point.sCurve);
            float semitones = previousSemitones
                + progress * (point.semitones - previousSemitones);
            if (point.vibratoDepthCents > 0.01f && point.vibratoRateHz > 0.01f
                && noteLengthSeconds > 0.0f)
            {
                const float segmentSeconds = (point.position - previousPosition)
                                           * noteLengthSeconds;
                const float phase = 2.0f * 3.14159265f * point.vibratoRateHz
                                  * progress * segmentSeconds;
                semitones += pitchBendWave (point.vibratoWaveform, phase)
                           * (point.vibratoDepthCents / 100.0f)
                           * pitchBendEnvelope (progress, point.vibratoFadeIn,
                                                point.vibratoFadeOut);
            }
            return semitones;
        }
        previousPosition = point.position;
        previousSemitones = point.semitones;
    }
    return points.back().semitones;
}

inline float pitchFrequencyAtBeat (const Note& note, float beat)
{
    const float position = note.lengthBeats > 0.0001f
        ? std::clamp ((beat - note.startBeat) / note.lengthBeats, 0.0f, 1.0f)
        : 0.0f;
    const float bend = interpolatePitchBend (
        note.pitchBendPoints, position, note.durationSeconds);
    const PitchAnchor* activeAnchor = nullptr;
    for (const auto& anchor : note.pitchAnchors)
        if (anchor.position <= position + 0.0001f
            && (activeAnchor == nullptr
                || anchor.position > activeAnchor->position
                || (anchor.position == activeAnchor->position && anchor.id > activeAnchor->id)))
            activeAnchor = &anchor;
    if (activeAnchor == nullptr || activeAnchor->frequency <= 0.0f
        || ! std::isfinite (activeAnchor->frequency))
        return note.freqHz * std::pow (2.0f, bend / 12.0f);
    const float anchorBend = interpolatePitchBend (
        note.pitchBendPoints, activeAnchor->position, note.durationSeconds);
    return activeAnchor->frequency * std::pow (2.0f, (bend - anchorBend) / 12.0f);
}

constexpr float kDefaultScoreHistoryBeats = 8.0f;
constexpr float kDefaultScoreLookaheadBeats = 16.0f;

// 3 = sounding, 2 = upcoming, 1 = recently ended, 0 = outside the
// score-visible timeline window. Block C and programmable hooks share this
// predicate so shaders and scripts see the same past/future span.
inline int timelineResidency (const Note& note, float beat,
                              float historyBeats, float lookaheadBeats)
{
    if (note.activeAt (beat)) return 3;
    if (note.startBeat > beat && note.startBeat <= beat + std::max (0.0f, lookaheadBeats))
        return 2;
    if (note.endBeat() <= beat && note.endBeat() >= beat - std::max (0.0f, historyBeats))
        return 1;
    return 0;
}

struct Link
{
    int id             = 0;
    int slaveNoteId    = 0;
    int masterNoteId   = 0;
    int slaveHarmonic  = 1;     // numerator   (ratio = slaveHarmonic / masterHarmonic)
    int masterHarmonic = 1;     // denominator
    int octaveTranspose = 0;    // f_slave = f_master * (sH/mH) * 2^octaveTranspose for pure JI
};

struct Score
{
    int notationVersion = 1;
    uint64_t scoreRevision = 0;
    int edoStepsPerOctave = 12;
    float rootFreq = 261.625565f;
    float historyBeats = kDefaultScoreHistoryBeats;
    float lookaheadBeats = kDefaultScoreLookaheadBeats;
    std::vector<Note> notes;
    std::vector<Link> links;

    const Note* noteById (int id) const
    {
        for (const auto& n : notes) if (n.id == id) return &n;
        return nullptr;
    }
    const Link* linkById (int id) const
    {
        for (const auto& l : links) if (l.id == id) return &l;
        return nullptr;
    }

    static bool inTrack (const Note& n, int trackId) { return trackId < 0 || n.trackId == trackId; }
    static bool inRange (const Note& n, float lo, float hi) { return n.midiNote >= lo && n.midiNote <= hi; }

    // Latest-onset note currently sounding on a track within a pitch range.
    const Note* latestActiveNote (int trackId, float beat, float lo, float hi) const
    {
        const Note* best = nullptr;
        for (const auto& n : notes)
            if (inTrack (n, trackId) && inRange (n, lo, hi) && n.activeAt (beat))
                if (best == nullptr || n.startBeat > best->startBeat
                    || (n.startBeat == best->startBeat && n.id > best->id)) best = &n;
        return best;
    }

    // Most recent onset at-or-before `beat` (used by trigger / age — survives
    // past note-off so a hit can decay after the note ends).
    float latestOnsetBeat (int trackId, float beat, float lo, float hi) const
    {
        float t = -1e30f;
        for (const auto& n : notes)
            if (inTrack (n, trackId) && inRange (n, lo, hi) && n.startedBy (beat))
                t = std::max (t, n.startBeat);
        return t;
    }

    int activeCount (int trackId, float beat, float lo = 0.0f, float hi = 127.0f) const
    {
        int c = 0;
        for (const auto& n : notes)
            if (inTrack (n, trackId) && inRange (n, lo, hi) && n.activeAt (beat)) ++c;
        return c;
    }
};

struct AudioFeatureSource
{
    int trackId = -1;
    int tapPoint = 0;
    float rms = 0.0f;
    float peak = 0.0f;
    float onset = 0.0f;
};

// Tier-3 audio features. Master comes from Block B; bounded per-track/group
// typed scalars come from transport shm without exposing raw audio rings.
struct Audio
{
    float rms   = 0.0f;
    float peak  = 0.0f;
    float onset = 0.0f;
    float bands[64] = {};
    std::array<AudioFeatureSource, 8> sources {};
    int sourceCount = 0;
    float band (int i) const { return (i >= 0 && i < 64) ? bands[i] : 0.0f; }
    const AudioFeatureSource* sourceFor(int trackId) const noexcept
    {
        for (int i = 0; i < sourceCount && i < (int) sources.size(); ++i)
            if (sources[(size_t) i].trackId == trackId) return &sources[(size_t) i];
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// Harmonic math
// ---------------------------------------------------------------------------
inline float centsFromRoot (float freqHz, float rootFreq)
{
    if (freqHz <= 0.0f || rootFreq <= 0.0f) return 0.0f;
    return 1200.0f * std::log2 (freqHz / rootFreq);
}

// THE pinned convention. Oriented by the link's own fields, never by note
// order. Returns 0 exactly when the link is pure JI.
inline float beatingRate (const Link& link, const Score& score)
{
    const Note* slave  = score.noteById (link.slaveNoteId);
    const Note* master = score.noteById (link.masterNoteId);
    if (slave == nullptr || master == nullptr) return 0.0f;
    const float octave = std::pow (2.0f, static_cast<float> (link.octaveTranspose));
    return std::fabs (slave->freqHz * static_cast<float> (link.masterHarmonic)
                      - master->freqHz * static_cast<float> (link.slaveHarmonic) * octave);
}

// Tenney height log2(num*den), normalised by a 13-limit reference so 3/2 is
// calm and high-limit intervals escalate toward 1.
inline float tenneyHeight (int num, int den, float normLog2 = 12.0f)
{
    const float prod = std::fabs (static_cast<float> (num) * static_cast<float> (den));
    if (prod <= 0.0f) return 0.0f;
    return clamp01 (std::log2 (prod) / std::max (1e-4f, normLog2));
}

// Sum of |prime exponent| of prime index p (0=2,1=3,2=5,3=7,4=11,5=13) over
// notes active at `beat` — "how p-limit is this chord".
inline float primeEnergy (const Score& score, int p, float beat,
                          int trackId = -1, float lo = 0.0f, float hi = 127.0f)
{
    if (p < 0 || p > 5) return 0.0f;
    float e = 0.0f;
    for (const auto& n : score.notes)
        if (Score::inTrack (n, trackId) && Score::inRange (n, lo, hi) && n.activeAt (beat))
            e += std::fabs (n.primes[p]);
    return e;
}

// Beat-integrated Lissajous phase for harmonograph visuals. The 2^k slowdown is
// PART of the source definition (raw audio-rate phase aliases at 60 fps); the
// factor cannot be left to the consumer. axis 0 = numerator rate, 1 = denom.
inline float lissajousPhase (const Note& n, int axis, float beat, float bpm, int k = 7)
{
    const int   harmonic = (axis == 0) ? n.ratioNum : n.ratioDen;
    const float slow      = std::pow (2.0f, static_cast<float> (k));
    const float rateHz    = (n.freqHz / slow) * static_cast<float> (std::max (1, harmonic));
    const float seconds   = beat * 60.0f / std::max (1e-4f, bpm);
    return fracf (rateHz * seconds);
}

// ---------------------------------------------------------------------------
// ADSR — pure closed form in beat time (ports VAModulation per-segment bend).
// ---------------------------------------------------------------------------
// Segment interpolation shape, u in [0,1]. Matches VAModulation::advanceSegment:
//   f(u) = (1 - e^(-k u)) / (1 - e^(-k)),  k = (rising ? curve : -curve)*STRENGTH
// linear when |k| < 0.05.
inline float segShape (float u, float curve, bool rising)
{
    u = clamp01 (u);
    const float k = (rising ? curve : -curve) * kCurveStrength;
    if (std::fabs (k) < 0.05f) return u;
    return (1.0f - std::exp (-k * u)) / (1.0f - std::exp (-k));
}

struct AdsrSpec
{
    float a    = 0.01f;   // attack  (beats)
    float d    = 0.20f;   // decay   (beats)
    float s    = 0.50f;   // sustain level 0..1
    float r    = 0.50f;   // release (beats)
    float bend = 0.0f;    // -1..1 per-segment exponential bulge (0 = linear)
};

// Level immediately before release begins (continuity when gate < a+d).
inline float adsrLevelBeforeRelease (float t, const AdsrSpec& e)
{
    if (t <= 0.0f) return 0.0f;
    if (t < e.a)  return lerpf (0.0f, 1.0f, segShape (e.a > 0 ? t / e.a : 1.0f, e.bend, true));
    if (t < e.a + e.d)
        return lerpf (1.0f, e.s, segShape (e.d > 0 ? (t - e.a) / e.d : 1.0f, e.bend, false));
    return e.s;
}

// Envelope value at `t` beats since the (re)trigger, with the gate held for
// `gateBeats`. Pure function → identical live and in export.
inline float adsrAt (float t, float gateBeats, const AdsrSpec& e)
{
    if (t < 0.0f) return 0.0f;
    if (t < gateBeats)                       // gate held: A → D → S
        return adsrLevelBeforeRelease (t, e);

    const float relStart = adsrLevelBeforeRelease (gateBeats, e);
    const float tr = t - gateBeats;          // release
    if (e.r <= 0.0f || tr >= e.r) return 0.0f;
    return lerpf (relStart, 0.0f, segShape (tr / e.r, e.bend, false));
}

// ---------------------------------------------------------------------------
// LFO — beat-time, deterministic. periodBeats is the cycle length in beats
// (BPM-synced by definition); use hzToPeriodBeats for free-running Hz rates.
// ---------------------------------------------------------------------------
enum class LFOShape { Sine, Triangle, Saw, Square, SampleHold };

inline float hzToPeriodBeats (float hz, float bpm)
{
    if (hz <= 0.0f) return 1.0f;
    return (bpm / 60.0f) / hz;               // beats per cycle
}

inline float lfoAt (LFOShape shape, float beat, float periodBeats, float phase0, uint32_t seed)
{
    if (periodBeats <= 0.0f) periodBeats = 1.0f;
    const float cyclePos = beat / periodBeats + phase0;
    const float phase    = fracf (cyclePos);     // 0..1
    switch (shape)
    {
        case LFOShape::Sine:     return std::sin (2.0f * kPi * phase);
        case LFOShape::Triangle:
            if (phase < 0.25f) return phase * 4.0f;
            if (phase < 0.75f) return 2.0f - phase * 4.0f;
            return phase * 4.0f - 4.0f;
        case LFOShape::Saw:      return 1.0f - 2.0f * phase;
        case LFOShape::Square:   return phase < 0.5f ? 1.0f : -1.0f;
        case LFOShape::SampleHold:
            return hashToBipolar (seed, static_cast<int32_t> (std::floor (cyclePos)));
    }
    return 0.0f;
}

// ---------------------------------------------------------------------------
// One-pole smoothing in BEATS (BPM- and frame-rate-independent).
// ---------------------------------------------------------------------------
inline float onePoleBeats (float prev, float target, float dtBeats, float tauBeats)
{
    if (tauBeats <= 0.0f) return target;     // no smoothing requested
    if (dtBeats  <= 0.0f) return prev;       // no time elapsed (paused / backward scrub) → hold
    const float a = 1.0f - std::exp (-dtBeats / tauBeats);
    return prev + (target - prev) * a;
}

// ---------------------------------------------------------------------------
// Routing curves (map a 0..1 source magnitude → 0..1).
// ---------------------------------------------------------------------------
enum class Curve { Linear, Exp, Log, SCurve };

inline float applyCurve (Curve c, float x)
{
    x = clamp01 (x);
    switch (c)
    {
        case Curve::Linear: return x;
        case Curve::Exp:    return x * x;                 // accelerating
        case Curve::Log:    return std::sqrt (x);         // decelerating
        case Curve::SCurve: return x * x * (3.0f - 2.0f * x);
    }
    return x;
}

enum class Mode { Add, Multiply, Replace };

inline float combine (Mode m, float base, float value)
{
    switch (m)
    {
        case Mode::Add:      return base + value;
        case Mode::Multiply: return base * value;
        case Mode::Replace:  return value;
    }
    return base;
}

// ---------------------------------------------------------------------------
// Source taxonomy (§5.1) + descriptor.
// ---------------------------------------------------------------------------
enum class SourceType
{
    // Tier 1 — symbolic
    NotePitch, NoteVelocity, NoteGate, NoteTrigger, NoteCount, NoteAge,
    CentsFromRoot, PrimeEnergy, RootTrigger,
    ClockBeatPhase, ClockBarPhase, ClockBeat,
    // Tier 1b — ratio-native
    HarmRatio, HarmRatioLog2, HarmRatioNum, HarmRatioDen,
    HarmLissajous, HarmBeatingRate, HarmTenney, HarmLinkRatio,
    // Tier 2 — generators
    Env, Lfo,
    // Tier 3 — audio-derived
    AudioRms, AudioPeak, AudioOnset, AudioBand
};

struct LfoSpec
{
    LFOShape shape       = LFOShape::Sine;
    float    periodBeats = 1.0f;   // cycle length in beats (BPM-synced)
    float    phase0      = 0.0f;
    uint32_t seed        = 1u;
    bool     hz          = false;  // if true, interpret rateHz instead of periodBeats
    float    rateHz      = 1.0f;
    bool     retrigger   = false;  // restart phase on the track's latest onset
};

struct ModSource
{
    SourceType type    = SourceType::ClockBeatPhase;
    int   trackId      = -1;       // -1 = any track
    float pitchLo      = 0.0f;     // pitch-range filter (e.g. kick vs snare)
    float pitchHi      = 127.0f;
    int   primeIndex   = 1;        // PrimeEnergy: INDEX not prime number (0=2,1=3,2=5,3=7,4=11,5=13)
    int   axis         = 0;        // Lissajous axis (0=num,1=den)
    int   linkId       = -1;       // HarmLinkRatio / HarmBeatingRate target
    int   band         = 0;        // AudioBand index
    int   lissajousK   = 7;        // Lissajous octave slowdown
    float triggerDecayBeats = 0.5f;// NoteTrigger / RootTrigger exp decay
    AdsrSpec adsr {};
    LfoSpec  lfo {};
};

// ---------------------------------------------------------------------------
// Source evaluation. Returns a value in the source's natural canonical form
// (mostly 0..1; LFO is -1..1; discrete ratio sources return the integer; Hz/
// cents sources return their physical value scaled by depth downstream).
// Pure for Tier 1/1b/2 — depends only on (Score, Clock); Tier 3 reads Audio.
// ---------------------------------------------------------------------------
inline float evaluateSource (const ModSource& s, const Score& score,
                             const Clock& clk, const Audio& audio)
{
    const float beat = clk.beat;
    switch (s.type)
    {
        case SourceType::NotePitch:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? clamp01 ((n->midiNote - 21.0f) / 87.0f) : 0.0f;  // piano range A0..C8
        }
        case SourceType::NoteVelocity:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? clamp01 (n->velocity / 127.0f) : 0.0f;
        }
        case SourceType::NoteGate:
            return score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi) ? 1.0f : 0.0f;
        case SourceType::NoteTrigger:
        {
            const float onset = score.latestOnsetBeat (s.trackId, beat, s.pitchLo, s.pitchHi);
            if (onset < -1e29f) return 0.0f;
            const float age = beat - onset;
            return clamp01 (std::exp (-age / std::max (1e-3f, s.triggerDecayBeats)));
        }
        case SourceType::NoteCount:
            return clamp01 (static_cast<float> (score.activeCount (s.trackId, beat, s.pitchLo, s.pitchHi)) / 16.0f);
        case SourceType::NoteAge:
        {
            const float onset = score.latestOnsetBeat (s.trackId, beat, s.pitchLo, s.pitchHi);
            return onset < -1e29f ? 0.0f : std::max (0.0f, beat - onset);  // beats (unnormalised)
        }
        case SourceType::CentsFromRoot:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? centsFromRoot (n->freqHz, score.rootFreq) : 0.0f;   // cents
        }
        case SourceType::PrimeEnergy:
            return primeEnergy (score, s.primeIndex, beat, s.trackId, s.pitchLo, s.pitchHi);
        case SourceType::RootTrigger:
        {
            // Impulse (exp decay) on the most recent root-note onset.
            float onset = -1e30f;
            for (const auto& n : score.notes)
                if (n.isRoot && Score::inTrack (n, s.trackId)
                    && Score::inRange (n, s.pitchLo, s.pitchHi) && n.startedBy (beat))
                    onset = std::max (onset, n.startBeat);
            if (onset < -1e29f) return 0.0f;
            return clamp01 (std::exp (-(beat - onset) / std::max (1e-3f, s.triggerDecayBeats)));
        }
        case SourceType::ClockBeatPhase: return fracf (beat);
        case SourceType::ClockBarPhase:  return fracf (beat / std::max (1e-4f, clk.beatsPerBar));
        case SourceType::ClockBeat:      return beat;

        case SourceType::HarmRatio:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return (n && n->ratioDen != 0) ? static_cast<float> (n->ratioNum) / n->ratioDen : 1.0f;
        }
        case SourceType::HarmRatioLog2:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            if (!n || n->ratioDen == 0 || n->ratioNum <= 0) return 0.0f;
            return std::log2 (static_cast<float> (n->ratioNum) / n->ratioDen);
        }
        case SourceType::HarmRatioNum:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? static_cast<float> (n->ratioNum) : 0.0f;   // discrete, no smoothing
        }
        case SourceType::HarmRatioDen:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? static_cast<float> (n->ratioDen) : 0.0f;   // discrete, no smoothing
        }
        case SourceType::HarmLissajous:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? lissajousPhase (*n, s.axis, beat, clk.bpm, s.lissajousK) : 0.0f;
        }
        case SourceType::HarmBeatingRate:
        {
            if (s.linkId >= 0) { const Link* l = score.linkById (s.linkId); return l ? beatingRate (*l, score) : 0.0f; }
            // track variant: max beating rate over links whose slave is on the track
            float maxB = 0.0f;
            for (const auto& l : score.links)
            {
                const Note* sl = score.noteById (l.slaveNoteId);
                if (sl && Score::inTrack (*sl, s.trackId)) maxB = std::max (maxB, beatingRate (l, score));
            }
            return maxB;   // Hz
        }
        case SourceType::HarmTenney:
        {
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            return n ? tenneyHeight (n->ratioNum, n->ratioDen) : 0.0f;
        }
        case SourceType::HarmLinkRatio:
        {
            const Link* l = score.linkById (s.linkId);
            return (l && l->masterHarmonic != 0)
                ? static_cast<float> (l->slaveHarmonic) / l->masterHarmonic : 1.0f;
        }

        case SourceType::Env:
        {
            const float onset = score.latestOnsetBeat (s.trackId, beat, s.pitchLo, s.pitchHi);
            if (onset < -1e29f) return 0.0f;
            const Note* n = score.latestActiveNote (s.trackId, beat, s.pitchLo, s.pitchHi);
            // gate = note length if a matching note carries this onset; else
            // gate up to "now" (sustained).
            float gate = beat - onset;
            for (const auto& cand : score.notes)
                if (Score::inTrack (cand, s.trackId) && Score::inRange (cand, s.pitchLo, s.pitchHi)
                    && std::fabs (cand.startBeat - onset) < 1e-4f)
                    { gate = cand.lengthBeats; break; }
            (void) n;
            return adsrAt (beat - onset, gate, s.adsr);
        }
        case SourceType::Lfo:
        {
            float phase0 = s.lfo.phase0;
            float refBeat = beat;
            if (s.lfo.retrigger)
            {
                const float onset = score.latestOnsetBeat (s.trackId, beat, s.pitchLo, s.pitchHi);
                if (onset > -1e29f) refBeat = beat - onset;   // phase restarts at the onset
            }
            const float period = s.lfo.hz ? hzToPeriodBeats (s.lfo.rateHz, clk.bpm) : s.lfo.periodBeats;
            return lfoAt (s.lfo.shape, refBeat, period, phase0, s.lfo.seed);
        }

        case SourceType::AudioRms:
            if (const auto* f = audio.sourceFor(s.trackId)) return clamp01(f->rms);
            return s.trackId < 0 ? clamp01(audio.rms) : 0.0f;
        case SourceType::AudioPeak:
            if (const auto* f = audio.sourceFor(s.trackId)) return clamp01(f->peak);
            return s.trackId < 0 ? clamp01(audio.peak) : 0.0f;
        case SourceType::AudioOnset:
            if (const auto* f = audio.sourceFor(s.trackId)) return clamp01(f->onset);
            return s.trackId < 0 ? clamp01(audio.onset) : 0.0f;
        case SourceType::AudioBand:
            return s.trackId < 0 ? clamp01(audio.band(s.band)) : 0.0f;
    }
    return 0.0f;
}

// True for sources that must NOT be smoothed (integer/discrete jumps read as
// intentional hits — §5.1 step-change policy).
inline bool isDiscreteSource (SourceType t)
{
    return t == SourceType::HarmRatioNum || t == SourceType::HarmRatioDen;
}

// ---------------------------------------------------------------------------
// Routing slot (§5.2). final = combine(base, depth * curve(source)).
// ---------------------------------------------------------------------------
struct Routing
{
    ModSource   source;
    std::string destination;        // any existing paramId
    float       depth          = 1.0f;   // -1..1, scaled to destination range
    Curve       curve          = Curve::Linear;
    float       smoothingBeats = 0.0f;   // one-pole, in beats (0 = none)
    Mode        mode           = Mode::Add;
    bool        enabled        = true;
};

// Per-routing running state (the only mutable piece — the smoothing memory).
struct RoutingState
{
    float smoothed = 0.0f;
    bool  primed   = false;
};

// Evaluate one routing against a base value. dtBeats is the beat delta since the
// previous evaluation (drives smoothing). Returns the modulated value; updates
// st.smoothed. Discrete sources bypass smoothing.
inline float evaluateRouting (const Routing& r, RoutingState& st, float base,
                              const Score& score, const Clock& clk, const Audio& audio,
                              float dtBeats)
{
    if (!r.enabled) return base;

    const float raw = evaluateSource (r.source, score, clk, audio);

    // Linear passes the source through at its native range (discrete ratioNum,
    // physical Hz/cents/age all scale via depth). The nonlinear curves are only
    // defined on a normalised control signal, so they shape |clamp01(raw)| and
    // reattach the sign.
    float shaped;
    if (r.curve == Curve::Linear)
        shaped = raw;
    else
        shaped = applyCurve (r.curve, clamp01 (std::fabs (raw))) * (raw < 0.0f ? -1.0f : 1.0f);
    float value = r.depth * shaped;

    if (r.smoothingBeats > 0.0f && !isDiscreteSource (r.source.type))
    {
        if (!st.primed) { st.smoothed = value; st.primed = true; }
        else            st.smoothed = onePoleBeats (st.smoothed, value, dtBeats, r.smoothingBeats);
        value = st.smoothed;
    }
    else
    {
        st.smoothed = value;
        st.primed   = true;
    }

    return combine (r.mode, base, value);
}

// ---------------------------------------------------------------------------
// Wire-string ↔ enum converters. Canonical home so the helper (main.cpp
// parseExportJob) and the Arbit plugin (VideoModDefs.h) can never disagree on
// the spelling of a source/curve/mode/LFO-shape name. Names match the enum
// identifiers exactly; an unknown string degrades to the safe default.
// ---------------------------------------------------------------------------
inline SourceType sourceTypeFromString (const std::string& s)
{
    using ST = SourceType;
    if (s == "NotePitch")       return ST::NotePitch;
    if (s == "NoteVelocity")    return ST::NoteVelocity;
    if (s == "NoteGate")        return ST::NoteGate;
    if (s == "NoteTrigger")     return ST::NoteTrigger;
    if (s == "NoteCount")       return ST::NoteCount;
    if (s == "NoteAge")         return ST::NoteAge;
    if (s == "CentsFromRoot")   return ST::CentsFromRoot;
    if (s == "PrimeEnergy")     return ST::PrimeEnergy;
    if (s == "RootTrigger")     return ST::RootTrigger;
    if (s == "ClockBeatPhase")  return ST::ClockBeatPhase;
    if (s == "ClockBarPhase")   return ST::ClockBarPhase;
    if (s == "ClockBeat")       return ST::ClockBeat;
    if (s == "HarmRatio")       return ST::HarmRatio;
    if (s == "HarmRatioLog2")   return ST::HarmRatioLog2;
    if (s == "HarmRatioNum")    return ST::HarmRatioNum;
    if (s == "HarmRatioDen")    return ST::HarmRatioDen;
    if (s == "HarmLissajous")   return ST::HarmLissajous;
    if (s == "HarmBeatingRate") return ST::HarmBeatingRate;
    if (s == "HarmTenney")      return ST::HarmTenney;
    if (s == "HarmLinkRatio")   return ST::HarmLinkRatio;
    if (s == "Env")             return ST::Env;
    if (s == "Lfo")             return ST::Lfo;
    if (s == "AudioRms")        return ST::AudioRms;
    if (s == "AudioPeak")       return ST::AudioPeak;
    if (s == "AudioOnset")      return ST::AudioOnset;
    if (s == "AudioBand")       return ST::AudioBand;
    return ST::ClockBeatPhase;
}

inline Curve curveFromString (const std::string& s)
{
    if (s == "Exp")    return Curve::Exp;
    if (s == "Log")    return Curve::Log;
    if (s == "SCurve") return Curve::SCurve;
    return Curve::Linear;
}

inline Mode modeFromString (const std::string& s)
{
    if (s == "Multiply") return Mode::Multiply;
    if (s == "Replace")  return Mode::Replace;
    return Mode::Add;
}

inline LFOShape lfoShapeFromString (const std::string& s)
{
    if (s == "Triangle")   return LFOShape::Triangle;
    if (s == "Saw")        return LFOShape::Saw;
    if (s == "Square")     return LFOShape::Square;
    if (s == "SampleHold") return LFOShape::SampleHold;
    return LFOShape::Sine;
}

} // namespace arbitmod
