// arbit-video-helper — JSON-RPC 2.0 sidecar over stdin/stdout.
//
// Speaks the line-delimited protocol expected by Arbit's SidecarProcessManager:
// requests arrive as {"jsonrpc":"2.0","id":N,"method":"...","params":{...}}\n
// and every request gets exactly one {"jsonrpc":"2.0","id":N,"result":...}\n
// or {"jsonrpc":"2.0","id":N,"error":{"message":...}}\n reply.
//
// Decoded frames travel through a shared-memory ring (VideoFrameSharedMemory.h)
// negotiated via the attach_shm method; request_frame replies carry the slot
// index instead of pixel data.

#include "media.h"
#include "capture_device.h"
#include "exporter.h"
#include "composite_probe_contract.h"
#include "bounded_line.h"
#include "compositor_ownership.h"
#include "render_snapshot_json.h"
#if ARBIT_HAVE_VIEWPORT
#include "viewport.h"
#endif
#include "visual_plan_telemetry_json.h"
#include "shader_dialect.h"   // shader_compile RPC (GL-free dialect front door)
#include "shader_compile/shader_lang.h"   // P3: Slang/SPIR-V → GLSL front door
#include "gpu_backend/backend.h"          // P6: native Metal backend seam
#include "lua_hook.h"         // P2 Scripts tab: script_compile validation (Lua hook)
#include "js_hook.h"          // P2 Scripts tab: script_compile validation (JS hook)
#include "programmable_admission.h"
#include "../../shared/PrivateInheritedPayload.h"
#include "VideoFrameSharedMemory.h"

// rife_selftest RPC: exercises whichever RIFE backend is actually compiled in,
// so CI can assert real GPU/EP interpolation engaged instead of just ping/version.
#if defined(ARBIT_HAVE_NCNN)
#include "rife_ncnn.h"
namespace arbitselftest { using RifeBackend = arbitrife::RifeEngineNcnn; }
#elif defined(ARBIT_HAVE_ONNX)
#include "rife.h"
namespace arbitselftest { using RifeBackend = arbitrife::RifeEngine; }
#endif

#include <algorithm>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <iostream>

#if defined(__APPLE__)
#include <poll.h>
#include <unistd.h>
#endif
#if ! defined(_WIN32) && ! defined(__APPLE__)
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace
{
std::map<uint32_t, std::unique_ptr<MediaContext>> g_media;
uint32_t g_nextMediaId = 1;
videoshm::Region g_shm;
uint32_t g_nextSlot = 0;
std::filesystem::path g_trustedMatteCacheRoot;
std::filesystem::path g_trustedDepthCacheRoot;

// Recorder sessions are independent: each owns the Arbit-provided shared-memory
// ring and its encoder. A close/cancel moves one entry out of the map before
// finalization, so it can never close another recording's ring.
struct RecorderCaptureSession
{
    videoshm::Region shm;
    RecorderSession recorder;
    std::string outPath;
};
std::map<uint32_t, std::unique_ptr<RecorderCaptureSession>> g_recorderSessions;
std::mutex g_recorderSessionsMutex;
uint32_t g_nextRecorderSessionId = 1;

struct CaptureEndpoint
{
    videoshm::Region shm;
    CaptureDeviceSession capture;
    std::mutex frameMutex;
    uint32_t nextSlot = 0;
    uint32_t latestSlot = 0;
};
std::map<uint32_t, std::shared_ptr<CaptureEndpoint>> g_captureSessions;
std::mutex g_captureSessionsMutex;
uint32_t g_nextCaptureSessionId = 1;

void closeAllCaptureSessions()
{
    std::map<uint32_t, std::shared_ptr<CaptureEndpoint>> sessions;
    {
        const std::lock_guard<std::mutex> lock(g_captureSessionsMutex);
        sessions.swap(g_captureSessions);
    }
    for (auto& [id, session] : sessions)
    {
        (void) id;
        session->capture.close();
        session->shm.close();
    }
}

void cancelAllRecorderSessions()
{
    std::map<uint32_t, std::unique_ptr<RecorderCaptureSession>> sessions;
    {
        std::lock_guard<std::mutex> lock (g_recorderSessionsMutex);
        sessions.swap (g_recorderSessions);
    }
    for (auto& [id, session] : sessions)
    {
        (void) id;
        session->recorder.close();
        session->shm.close();
        std::remove (session->outPath.c_str());
    }
}
#if ARBIT_HAVE_VIEWPORT
Viewport g_viewport;
bool g_testRejectNextSnapshot = false;
bool g_testDeferNextSnapshot = false;
std::optional<videowire::ResolvedVisualSnapshot> g_testDeferredSnapshot;

// RAII: suspend viewport RIFE for the lifetime of an export/render-cache job (a
// second GPU session would contend for VRAM) and ALWAYS restore on scope exit —
// even if runExport throws or returns early. Without this, a job that threw left
// interpolation permanently suspended for the rest of the session (the viewport
// silently stayed on Frame Blend / nearest, never re-enabling RIFE).
struct ViewportInterpSuspendGuard
{
    ViewportInterpSuspendGuard()  { g_viewport.setInterpolationSuspended (true); }
    ~ViewportInterpSuspendGuard() { g_viewport.setInterpolationSuspended (false); }
    ViewportInterpSuspendGuard (const ViewportInterpSuspendGuard&) = delete;
    ViewportInterpSuspendGuard& operator= (const ViewportInterpSuspendGuard&) = delete;
};
#endif

// stdout is shared between the RPC loop and the export worker (which sends
// the deferred `export` reply when the render finishes) — one line per write.
std::mutex g_stdoutMutex;

// Async export state (PROTOCOL.md §Export): `export` parses the jobSpec,
// spawns a worker and defers its reply until the render finishes; the RPC
// loop stays free to answer export_progress / export_cancel meanwhile.
struct ExportAsyncState
{
    ExportProgress progress;
    std::atomic<bool> active { false };
    std::atomic<bool> done { false };       // a run has finished since launch
    std::atomic<bool> cancelled { false };  // last run ended via export_cancel
    std::atomic<int> jobId { 0 };           // increments per export request
    std::mutex resultMutex;                 // guards error/result
    std::string error;
    json result;
    std::thread worker;
} g_export;

// Async proxy transcode state (PROTOCOL.md §Proxy media): same deferred-reply
// job pattern as `export` — `proxy_generate` spawns a worker and answers when
// the transcode finishes; `proxy_progress` / `proxy_cancel` are served by the
// RPC loop meanwhile. One proxy job at a time, independent of exports.
struct ProxyAsyncState
{
    ProxyProgress progress;
    std::atomic<bool> active { false };
    std::atomic<bool> done { false };
    std::atomic<bool> cancelled { false };
    std::atomic<int> jobId { 0 };
    std::mutex resultMutex;                 // guards error/result
    std::string error;
    json result;
    std::thread worker;
} g_proxy;

// Async render-cache build state (PROTOCOL.md §Render cache): same deferred-
// reply job pattern. `render_cache_build` takes an export-shaped jobSpec
// (single source segment + that clip's effects/LUT), forces intra-only
// output and renders through the SAME GL FrameRenderer as export. One build
// at a time; refused while an export runs (and an arriving export aborts an
// in-flight build) so two offscreen GLFW contexts are never created
// concurrently. workerMutex serializes abort+join between the RPC loop and
// the export worker.
struct RenderCacheAsyncState
{
    ExportProgress progress;
    std::atomic<bool> active { false };
    std::atomic<bool> done { false };
    std::atomic<bool> cancelled { false };
    std::atomic<int> jobId { 0 };
    std::mutex resultMutex;                 // guards error/result
    std::string error;
    json result;
    std::mutex workerMutex;                 // guards join() callers
    std::thread worker;
} g_renderCache;
videohelper::CompositorOwnershipGate g_compositorOwnership;

void reply (const json& idVal, const json& result)
{
    json msg = { { "jsonrpc", "2.0" }, { "id", idVal }, { "result", result } };
    std::lock_guard<std::mutex> lock (g_stdoutMutex);
    std::fputs ((msg.dump() + "\n").c_str(), stdout);
    std::fflush (stdout);
}

void replyError (const json& idVal, const std::string& message)
{
    json msg = { { "jsonrpc", "2.0" }, { "id", idVal },
                 { "error", { { "code", -32000 }, { "message", message } } } };
    std::lock_guard<std::mutex> lock (g_stdoutMutex);
    std::fputs ((msg.dump() + "\n").c_str(), stdout);
    std::fflush (stdout);
}

MediaContext* findMedia (const json& params, std::string& error)
{
    if (! params.contains ("mediaId")) { error = "missing mediaId"; return nullptr; }
    auto it = g_media.find (params["mediaId"].get<uint32_t>());
    if (it == g_media.end()) { error = "unknown mediaId"; return nullptr; }
    return it->second.get();
}

// jobSpec schema (PROTOCOL.md §Export):
//   outPath: string (codec "prores" requires a .mov path)
//   fps, width, height: numbers
//   codec: "h264" | "h265" | "vp9" | "prores" | "ffv1"
//   encoder: "auto" | "software" | "nvenc" | "videotoolbox"
//     (auto = try hardware encode, fall back to software; prores/vp9 are
//     always software: prores_ks / libvpx-vp9)
//   interpolation: "none" | "minterpolate" | "rife" | "auto"
//     ("rife"/"auto" = RIFE optical-flow interpolation for segments
//     that under-deliver frames: source_fps * rate < target_fps.
//     "rife" errors when the ONNX backend is unavailable, "auto"
//     degrades silently to nearest-frame. See PROTOCOL.md §RIFE.)
//   audioPath: string (master mix WAV rendered by Arbit)
//   startSec, endSec: export range on the display timeline (omitted/0 =
//     full timeline; an explicit endSec also trims the audio track to the
//     range — PROTOCOL.md §Export)
//   segments: [{sourcePath, clipId, trackLayer, inSec, outSec, rate,
//               displayStartSec, transition?{type, durationSec},
//               sourceFps?, seqStart?}]  (image-sequence pattern hints)
//   clips: [{clipId, scale, translateX, translateY, rotation,
//            cropLeft..cropBottom, opacity, visible, zOrder,
//            blendMode, effects:[{slot,type,enabled,params{}}],
//            mask?{type,cx,cy,w,h,feather,invert}, lutPath?,
//            genParams?{<isfInputName>: value}}]
//     (lutPath = .cube file read by the helper at export start —
//      PROTOCOL.md §LUT; genParams = static ISF INPUT overrides, M7)
//   paramTimeline: [{paramId, atSec, value}] (baked automation;
//     paramId namespace: clip<id>/<node>/<param> or text<id>/<param>,
//     where <node> includes gen for ISF INPUTS: clip<id>/gen/<name>)
//   score: {rootFreq, historyBeats?, lookaheadBeats?, notes:[{id, trackId, startBeat,
//     lengthBeats, midiNote, velocity, freqHz, durationSeconds, pitchBendPoints,
//     pitchAnchors,
//     ratioNum, ratioDen,
//     primes:[e2,e3,e5,e7,e11,e13], linkMasterId, isRoot}],
//     links:[{id, slaveNoteId, masterNoteId, slaveHarmonic, masterHarmonic,
//     octaveTranspose}]} — Block C (M5): packed per frame into uNotes/uLinks
//     for shader generators (PROTOCOL.md §Shader generators → Block C)
//   modMatrix: [{source:{type, trackId?, pitchLo?, pitchHi?, primeIndex?,
//     axis?, linkId?, band?, lissajousK?, triggerDecayBeats?, adsr?{a,d,s,r,
//     bend}, lfo?{shape,periodBeats,phase0,seed,hz,rateHz,retrigger}},
//     destination:"clip<id>/<node>/<param>", depth?, curve?(Linear|Exp|Log|
//     SCurve), smoothingBeats?, mode?(Add|Multiply|Replace), enabled?}]
//     — M6 cross-domain modulation matrix: a musical source modulates a
//     render-graph clip param per frame (PROTOCOL.md §Mod matrix)
//   texts: [{textId, startSec, durationSec, posX, posY, opacity,
//            zOrder, width, height, rgbaBase64}] — rgbaBase64 is the
//     straight-RGBA block (top row first, stride width*4) rendered
//     by Arbit, base64-encoded inline (export runs headless, so the
//     viewport's text_set_image shm transport is not used)
// ── M6 mod-matrix enum string ⇄ value maps (mirror arbitmod's enums; unknown
// strings fall back to the engine default so a typo can't crash an export) ──
static arbitmod::SourceType parseSourceType (const std::string& s)
{
    using ST = arbitmod::SourceType;
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
static arbitmod::Curve parseCurve (const std::string& s)
{
    if (s == "Exp")    return arbitmod::Curve::Exp;
    if (s == "Log")    return arbitmod::Curve::Log;
    if (s == "SCurve") return arbitmod::Curve::SCurve;
    return arbitmod::Curve::Linear;
}
static arbitmod::Mode parseMode (const std::string& s)
{
    if (s == "Multiply") return arbitmod::Mode::Multiply;
    if (s == "Replace")  return arbitmod::Mode::Replace;
    return arbitmod::Mode::Add;
}
static arbitmod::LFOShape parseLfoShape (const std::string& s)
{
    if (s == "Triangle")   return arbitmod::LFOShape::Triangle;
    if (s == "Saw")        return arbitmod::LFOShape::Saw;
    if (s == "Square")     return arbitmod::LFOShape::Square;
    if (s == "SampleHold") return arbitmod::LFOShape::SampleHold;
    return arbitmod::LFOShape::Sine;
}

// Parse a `score` object (M5 Block C wire schema, documented at the top of
// parseExportJob) into an arbitmod::Score. Shared by the export job
// parser and the viewport_set_score RPC so the export and live-preview paths
// can never disagree on the note/link schema.
static void parseScoreJson (const json& sc, arbitmod::Score& score)
{
    score.notationVersion = sc.value("notationVersion", 1);
    score.scoreRevision = sc.value("scoreRevision", uint64_t { 0 });
    score.edoStepsPerOctave = std::max(1, sc.value("edoStepsPerOctave", 12));
    score.rootFreq = sc.value ("rootFreq", 261.625565f);
    score.historyBeats = std::max (0.0f, sc.value (
        "historyBeats", arbitmod::kDefaultScoreHistoryBeats));
    score.lookaheadBeats = std::max (0.0f, sc.value (
        "lookaheadBeats", arbitmod::kDefaultScoreLookaheadBeats));
    if (sc.contains ("notes"))
        for (const auto& n : sc["notes"])
        {
            arbitmod::Note nt;
            nt.id           = n.value ("id", 0);
            nt.trackId      = n.value ("trackId", 0);
            nt.startBeat    = n.value ("startBeat", 0.0f);
            nt.lengthBeats  = n.value ("lengthBeats", 1.0f);
            nt.midiNote     = n.value ("midiNote", 60.0f);
            nt.velocity     = n.value ("velocity", 100.0f);
            nt.freqHz       = n.value ("freqHz", 261.625565f);
            nt.durationSeconds = n.value ("durationSeconds", 0.0f);
            if (n.contains ("pitchBendPoints") && n["pitchBendPoints"].is_array())
                for (const auto& point : n["pitchBendPoints"])
                {
                    arbitmod::PitchBendPoint bend;
                    bend.position = point.value ("position", 0.0f);
                    bend.semitones = point.value ("semitones", 0.0f);
                    bend.tension = point.value ("tension", 0.0f);
                    bend.sCurve = point.value ("sCurve", 0.0f);
                    bend.vibratoDepthCents = point.value ("vibratoDepthCents", 0.0f);
                    bend.vibratoRateHz = point.value ("vibratoRateHz", 0.0f);
                    bend.vibratoWaveform = point.value ("vibratoWaveform", 0);
                    bend.vibratoFadeIn = point.value ("vibratoFadeIn", 0.0f);
                    bend.vibratoFadeOut = point.value ("vibratoFadeOut", 0.0f);
                    nt.pitchBendPoints.push_back (bend);
                }
            if (n.contains ("pitchAnchors") && n["pitchAnchors"].is_array())
                for (const auto& anchor : n["pitchAnchors"])
                    nt.pitchAnchors.push_back ({
                        anchor.value ("id", -1),
                        anchor.value ("position", 0.0f),
                        anchor.value ("frequency", 0.0f) });
            nt.ratioNum     = n.value ("ratioNum", 1);
            nt.ratioDen     = n.value ("ratioDen", 1);
            nt.linkMasterId = n.value ("linkMasterId", -1);
            nt.isRoot       = n.value ("isRoot", false);
            nt.centsOffset = n.value("centsOffset", 0.0f);
            nt.edoStep = n.value("edoStep", -1);
            nt.muted = n.value("muted", false);
            nt.notationVisible = n.value("notationVisible", ! nt.muted);
            nt.diatonicIndex = n.value("diatonicIndex", 28);
            nt.baseAccidental = n.value("baseAccidental", 0);
            nt.linked = n.value("linked", false);
            nt.hasUnmappedPrime = n.value("hasUnmappedPrime", false);
            nt.edoActive = n.value("edoActive", false);
            nt.edoInflection = n.value("edoInflection", 0);
            nt.edoDegree = n.value("edoDegree", 0);
            if (n.contains("commas") && n["commas"].is_array())
                for (const auto& comma : n["commas"])
                {
                    if (nt.commaCount >= static_cast<int>(nt.commas.size())) break;
                    nt.commas[static_cast<size_t>(nt.commaCount++)] = {
                        comma.value("prime", 0), comma.value("exponent", 0) };
                }
            if (n.contains ("primes") && n["primes"].is_array())
                for (size_t i = 0; i < n["primes"].size() && i < 6; ++i)
                    if (n["primes"][i].is_number())
                        nt.primes[i] = n["primes"][i].get<float>();
            score.notes.push_back (nt);
        }
    if (sc.contains ("links"))
        for (const auto& l : sc["links"])
        {
            arbitmod::Link lk;
            lk.id              = l.value ("id", 0);
            lk.slaveNoteId     = l.value ("slaveNoteId", 0);
            lk.masterNoteId    = l.value ("masterNoteId", 0);
            lk.slaveHarmonic   = l.value ("slaveHarmonic", 1);
            lk.masterHarmonic  = l.value ("masterHarmonic", 1);
            lk.octaveTranspose = l.value ("octaveTranspose", 0);
            score.links.push_back (lk);
        }
}

// Parse a `modMatrix` array (M6 routing wire schema) into arbitmod::Routings.
// Shared by the export job parser and the viewport_set_mod_matrix RPC so the
// export and live-preview paths can never disagree on the routing schema.
static void parseRoutingsJson (const json& arr, std::vector<arbitmod::Routing>& out)
{
    for (const auto& r : arr)
    {
        arbitmod::Routing rt;
        rt.destination    = r.value ("destination", "");
        rt.depth          = r.value ("depth", 1.0f);
        rt.curve          = parseCurve (r.value ("curve", std::string ("Linear")));
        rt.smoothingBeats = r.value ("smoothingBeats", 0.0f);
        rt.mode           = parseMode (r.value ("mode", std::string ("Add")));
        rt.enabled        = r.value ("enabled", true);
        if (r.contains ("source") && r["source"].is_object())
        {
            const auto& sj = r["source"];
            arbitmod::ModSource ms;
            ms.type       = parseSourceType (sj.value ("type", std::string ("ClockBeatPhase")));
            ms.trackId    = sj.value ("trackId", -1);
            ms.pitchLo    = sj.value ("pitchLo", 0.0f);
            ms.pitchHi    = sj.value ("pitchHi", 127.0f);
            ms.primeIndex = sj.value ("primeIndex", 1);
            ms.axis       = sj.value ("axis", 0);
            ms.linkId     = sj.value ("linkId", -1);
            ms.band       = sj.value ("band", 0);
            ms.lissajousK = sj.value ("lissajousK", 7);
            ms.triggerDecayBeats = sj.value ("triggerDecayBeats", 0.5f);
            if (sj.contains ("adsr") && sj["adsr"].is_object())
            {
                const auto& a = sj["adsr"];
                ms.adsr.a    = a.value ("a", 0.01f);
                ms.adsr.d    = a.value ("d", 0.20f);
                ms.adsr.s    = a.value ("s", 0.50f);
                ms.adsr.r    = a.value ("r", 0.50f);
                ms.adsr.bend = a.value ("bend", 0.0f);
            }
            if (sj.contains ("lfo") && sj["lfo"].is_object())
            {
                const auto& l = sj["lfo"];
                ms.lfo.shape       = parseLfoShape (l.value ("shape", std::string ("Sine")));
                ms.lfo.periodBeats = l.value ("periodBeats", 1.0f);
                ms.lfo.phase0      = l.value ("phase0", 0.0f);
                ms.lfo.seed        = (uint32_t) l.value ("seed", 1);
                ms.lfo.hz          = l.value ("hz", false);
                ms.lfo.rateHz      = l.value ("rateHz", 1.0f);
                ms.lfo.retrigger   = l.value ("retrigger", false);
            }
            rt.source = ms;
        }
        if (! rt.destination.empty())
            out.push_back (std::move (rt));
    }
}

static bool parseVideoControlPlanJson(const json& value, videocontrol::Plan& plan,
                                      std::string& error)
{
    plan = {};
    if (value.is_null()) return true;
    if (!value.is_object())
    {
        error = "controlPlan must be an object";
        return false;
    }
    plan.version = value.value("version", 1);
    plan.numSlots = value.value("numSlots", 0);
    if (!value.contains("operations") || !value["operations"].is_array())
    {
        error = "controlPlan.operations must be an array";
        return false;
    }
    if (value["operations"].size() > videocontrol::kMaxOperations)
    {
        error = "video control plan operation count exceeds capacity";
        return false;
    }

    for (const auto& item : value["operations"])
    {
        if (!item.is_object())
        {
            error = "video control plan operation must be an object";
            return false;
        }
        videocontrol::Operation operation;
        operation.nodeId = item.value("nodeId", -1);
        operation.kind = item.value("kind", std::string{});
        operation.destination = item.value("destination", std::string{});
        operation.targetClipId = item.value("targetClipId", -1);
        operation.targetNodeId = item.value("targetNodeId", -1);
        operation.targetParamId = item.value("targetParamId", std::string{});
        operation.depth = item.value("depth", 1.0f);
        operation.curve = parseCurve(item.value("curve", std::string("Linear")));
        operation.smoothingBeats = item.value("smoothing", 0.0f);
        operation.mode = parseMode(item.value("mode", std::string("Add")));
        operation.enabled = item.value("enabled", true);

        if (item.contains("inputs") && item["inputs"].is_array())
            for (const auto& input : item["inputs"])
            {
                if (!input.is_array())
                {
                    error = "video control plan input must be an array";
                    return false;
                }
                std::vector<int> fanIn;
                for (const auto& slot : input)
                    if (slot.is_number_integer()) fanIn.push_back(slot.get<int>());
                    else
                    {
                        error = "video control plan input slot must be an integer";
                        return false;
                    }
                operation.inputs.push_back(std::move(fanIn));
            }

        if (item.contains("params") && item["params"].is_array())
            for (const auto& parameter : item["params"])
                if (parameter.is_number()) operation.params.push_back(parameter.get<float>());
                else
                {
                    error = "video control plan parameter must be numeric";
                    return false;
                }

        if (item.contains("outputSlots") && item["outputSlots"].is_array())
            for (const auto& slot : item["outputSlots"])
                if (slot.is_number_integer()) operation.outputSlots.push_back(slot.get<int>());
                else
                {
                    error = "video control plan output slot must be an integer";
                    return false;
                }

        if (operation.kind == "source")
        {
            operation.source.type = parseSourceType(
                item.value("sourceType", std::string("ClockBeatPhase")));
            const auto parameter = [&operation](std::size_t index, float fallback)
            {
                return index < operation.params.size() ? operation.params[index] : fallback;
            };
            operation.source.trackId = static_cast<int>(std::lround(parameter(0, -1.0f)));
            operation.source.pitchLo = parameter(1, 0.0f);
            operation.source.pitchHi = parameter(2, 127.0f);
            operation.source.primeIndex = static_cast<int>(std::lround(parameter(3, 1.0f)));
            operation.source.axis = static_cast<int>(std::lround(parameter(4, 0.0f)));
            operation.source.linkId = static_cast<int>(std::lround(parameter(5, -1.0f)));
            operation.source.band = static_cast<int>(std::lround(parameter(6, 0.0f)));
            operation.source.lissajousK = static_cast<int>(std::lround(parameter(7, 7.0f)));
            operation.source.triggerDecayBeats = parameter(8, 0.5f);
            operation.source.adsr = { parameter(9, 0.05f), parameter(10, 0.1f),
                                      parameter(11, 0.8f), parameter(12, 0.2f),
                                      parameter(13, 0.0f) };
            operation.source.lfo.periodBeats = parameter(14, 1.0f);
            operation.source.lfo.phase0 = parameter(15, 0.0f);
            operation.source.lfo.seed = static_cast<uint32_t>(
                std::lround(parameter(16, 1.0f)));
            operation.source.lfo.shape = static_cast<arbitmod::LFOShape>(std::clamp(
                static_cast<int>(std::lround(parameter(17, 0.0f))), 0, 4));
            operation.source.lfo.hz = parameter(18, 0.0f) >= 0.5f;
            operation.source.lfo.rateHz = parameter(19, 1.0f);
            operation.source.lfo.retrigger = parameter(20, 0.0f) >= 0.5f;
        }
        plan.operations.push_back(std::move(operation));
    }
    return videocontrol::validatePlan(plan, error);
}

static videotime::BeatTimeline parseBeatTimelineJson (const json& owner,
                                                      double fallbackBpm,
                                                      double fallbackBeatsPerBar)
{
    std::vector<videotime::TempoPoint> tempos;
    std::vector<videotime::MeterPoint> meters;
    if (owner.contains("tempoMap") && owner["tempoMap"].is_array())
        for (const auto& point : owner["tempoMap"])
            if (point.is_object())
                tempos.push_back({ point.value("beat", 0.0), point.value("bpm", fallbackBpm),
                    point.value("interpolation", std::string("linear")) == "step" });
    if (owner.contains("timeSignatureMap") && owner["timeSignatureMap"].is_array())
        for (const auto& point : owner["timeSignatureMap"])
            if (point.is_object())
                meters.push_back({ point.value("beat", 0.0), point.value("numerator", 4),
                    point.value("denominator", 4) });

    videotime::BeatTimeline result;
    if (tempos.empty() && meters.empty())
    {
        result.reset(fallbackBpm, fallbackBeatsPerBar);
        return result;
    }
    if (tempos.empty())
        tempos.push_back({ 0.0, fallbackBpm, true });
    if (meters.empty())
        meters.push_back({ 0.0,
            std::max(1, static_cast<int>(std::llround(fallbackBeatsPerBar))), 4 });
    result.set(std::move(tempos), std::move(meters));
    return result;
}

// P3: resolve a clip/segment spec's shaderSource through the language front
// door — GLSL passes through; Slang / SPIR-V (per the spec's `shaderLang`) are
// lowered to GLSL here so the renderer only ever sees GLSL. On a lowering
// failure logs once and returns "" (the clip renders without a broken shader
// rather than failing the whole job).
static std::string resolveShaderSource (const json& spec)
{
    const std::string raw = spec.value ("shaderSource", std::string());
    if (raw.empty()) return raw;
    const auto lang = arbitshadercompile::langFromWire (spec.value ("shaderLang", std::string()));
    if (lang == arbitshadercompile::Lang::Glsl) return raw;
    const auto low = arbitshadercompile::lowerToGlsl (raw, lang);
    if (low.ok) return low.glsl;
    std::fprintf (stderr, "[shader] lower (%s) failed: %s\n",
                  arbitshadercompile::langToWire (lang), low.error.c_str());
    return std::string();
}

static bool admitProgrammableField (const json& owner, const char* sourceField,
                                    const char* grantField, std::string& error)
{
    programmableruntime::Grant grant;
    return programmableadmission::admitCatalogGpuField(owner, sourceField, grantField,
                                                        grant, error);
}

static bool admitProgrammableJob (const json& params, std::string& error)
{
    if (! params.is_object()) { error = "programmable job params must be an object"; return false; }
    for (const char* collection : { "segments", "clips" })
    {
        const auto it = params.find(collection);
        if (it == params.end()) continue;
        if (! it->is_array()) { error = std::string(collection) + " must be an array"; return false; }
        for (const auto& member : *it)
        {
            if (! member.is_object()) { error = std::string(collection) + " members must be objects"; return false; }
            if (! admitProgrammableField(member,"shaderSource","runtimeGrant",error)) return false;
        }
    }
    const auto langIt = params.find("scriptLang");
    if (langIt != params.end() && ! langIt->is_string()) { error = "scriptLang must be a string"; return false; }
    const auto lang = langIt == params.end() ? std::string{} : langIt->get<std::string>();
    if (lang != "" && lang != "lua" && lang != "js") { error = "scriptLang must be lua or js"; return false; }
    for (const char* sourceField : { "luaScript", "jsScript" })
    {
        const auto it = params.find(sourceField);
        if (it != params.end() && ! it->is_string()) { error = std::string(sourceField) + " must be a string"; return false; }
    }
    const auto luaSource = params.value("luaScript",std::string{});
    const auto jsSource = params.value("jsScript",std::string{});
    if ((! luaSource.empty() && lang != "lua") || (! jsSource.empty() && lang != "js")
        || (! luaSource.empty() && ! jsSource.empty()))
    { error = "script language/source mismatch"; return false; }
    const auto source = lang == "lua" ? luaSource : jsSource;
    if(!source.empty())
    {
        if (lang == "lua")
        {
           #if ! ARBIT_HAVE_LUA
            error = "Lua runtime unavailable"; return false;
           #endif
        }
        else
        {
           #if ! ARBIT_HAVE_QUICKJS
            error = "QuickJS runtime unavailable"; return false;
           #endif
        }
        programmableruntime::Grant grant;
        const auto it=params.find("scriptRuntimeGrant");
        if(it==params.end()){error="missing scriptRuntimeGrant";return false;}
        const auto kind=lang=="lua"?programmableruntime::PayloadKind::lua:programmableruntime::PayloadKind::javascript;
        if(!programmableadmission::admit(*it,kind,source,grant,error)) return false;
    }
    return true;
}

std::string parseExportJob (const json& params, ExportJob& job)
{
    std::string admissionError;
    if(!admitProgrammableJob(params,admissionError)) return admissionError;
    // Process-owned launch capability; jobSpec can neither supply nor override it.
    job.depthCacheRoot = g_trustedDepthCacheRoot.string();
    const std::string encoder = params.value ("encoder", "auto");
    if (encoder != "auto" && encoder != "software" && encoder != "nvenc"
        && encoder != "videotoolbox")
        return "unknown encoder: " + encoder;
    const std::string interpolation = params.value ("interpolation", "none");
    if (interpolation != "none" && interpolation != "minterpolate"
        && interpolation != "rife" && interpolation != "auto")
        return "unknown interpolation: " + interpolation;
    const std::string codec = params.value ("codec", "h264");
    if (codec != "h264" && codec != "h265" && codec != "vp9" && codec != "prores"
        && codec != "ffv1"
        && codec != "dpx")
        return "unknown codec: " + codec;

    job.outPath = params.value ("outPath", "");
    job.width = params.value ("width", 1920);
    job.height = params.value ("height", 1080);
    job.fps = params.value ("fps", 30.0);
    job.codec = codec;
    job.proresProfile = params.value ("proresProfile", "hq");   // hq | 4444 | 4444xq
    job.encoder = encoder;
    job.interpolation = interpolation;
    job.audioPath = params.value ("audioPath", "");
    job.durationSec = params.value ("durationSec", 0.0);
    job.lufsTarget = params.value ("lufsTarget", 0.0);   // 0 = off; <0 = LUFS target (downward-only)
    job.startSec = params.value ("startSec", 0.0);
    job.endSec = params.value ("endSec", 0.0);
    job.intraOnly = params.value ("intraOnly", false);
    job.feedbackPreRollSec = std::max (0.0, params.value ("feedbackPreRollSec", 0.0));
    job.bpm = params.value ("bpm", 120.0);
    job.beatsPerBar = params.value ("beatsPerBar", 4.0);
    job.beatTimeline = parseBeatTimelineJson(params, job.bpm, job.beatsPerBar);
    job.luaScript = params.value ("luaScript", "");   // M8 per-frame hook (inert if no Lua)
    job.jsScript  = params.value ("jsScript", "");    // P2 JS per-frame hook (inert if no QuickJS)
    job.scriptLang = params.value ("scriptLang", ""); // "lua" | "js" | "" (infer)
    if (! job.luaScript.empty() || ! job.jsScript.empty())
    {
        programmableruntime::Grant grant;
        std::string ignored;
        programmableadmission::parseGrant(params.at("scriptRuntimeGrant"), grant, ignored);
        job.scriptCpuMs = grant.cpuMs;
        job.scriptMemoryMiB = grant.memoryMiB;
    }
    if (params.contains ("canvasBackground") && params["canvasBackground"].is_object())   // M1 canvas bg
    {
        const auto& bg = params["canvasBackground"];
        job.bgColor[0] = (float) bg.value ("r", 0.04);
        job.bgColor[1] = (float) bg.value ("g", 0.04);
        job.bgColor[2] = (float) bg.value ("b", 0.05);
        job.bgColor[3] = (float) bg.value ("a", 1.0);
    }
    if (params.contains ("post") && params["post"].is_object())   // HDR bloom + tonemap (increment 2)
    {
        const auto& p = params["post"];
        job.post.bloomIntensity = (float) p.value ("bloomIntensity", 0.0);
        job.post.bloomThreshold = (float) p.value ("bloomThreshold", 1.0);
        job.post.bloomRadius    = (float) p.value ("bloomRadius", 8.0);
        job.post.tonemap        = p.value ("tonemap", 0);
        job.post.exposure       = (float) p.value ("exposure", 1.0);
    }
    videowire::ResolvedVisualSnapshot snapshot;
    std::string snapshotError;
    if (! videowire::parseSnapshotJson (
            params, resolveShaderSource, snapshot, snapshotError))
        return snapshotError;
    if (! videowire::validateSnapshotResources(snapshot,
            [] (const std::string& path) { return std::filesystem::exists(path); },
            g_trustedMatteCacheRoot, params.value("matteContentRevision", uint64_t { 0 }), snapshotError))
        return snapshotError;
    job.authoringRevision = snapshot.authoringRevision;
    job.exportableRevision = params.value ("exportableRevision", job.authoringRevision);
    if (job.authoringRevision != 0
        && job.exportableRevision != job.authoringRevision)
        return "current authoring revision is not exportable";
    job.segments = std::move (snapshot.segments);
    job.visualLayerPlans = std::move (snapshot.visualLayerPlans);
    job.visualEventSchedules = std::move (snapshot.visualEventSchedules);

    if (params.contains ("clips"))
        for (const auto& c : params["clips"])
        {
            ExportClipState cs;
            cs.clipId = c.value ("clipId", 0);
            cs.scale = c.value ("scale", 1.0);
            cs.translateX = c.value ("translateX", 0.0);
            cs.translateY = c.value ("translateY", 0.0);
            cs.rotationDeg = c.value ("rotation", 0.0);
            cs.cropLeft = c.value ("cropLeft", 0.0);
            cs.cropRight = c.value ("cropRight", 0.0);
            cs.cropTop = c.value ("cropTop", 0.0);
            cs.cropBottom = c.value ("cropBottom", 0.0);
            cs.opacity = c.value ("opacity", 1.0);
            cs.visible = c.value ("visible", true);
            cs.zOrder = c.value ("zOrder", 0);
            cs.blendMode = c.value ("blendMode", 0);
            if (c.contains ("mask"))
            {
                const auto& m = c["mask"];
                cs.maskType = m.value ("type", 0);
                cs.maskCx = m.value ("cx", 0.5);
                cs.maskCy = m.value ("cy", 0.5);
                cs.maskW = m.value ("w", 0.8);
                cs.maskH = m.value ("h", 0.8);
                cs.maskFeather = m.value ("feather", 0.05);
                cs.maskInvert = m.value ("invert", false);
            }
            cs.lutPath = c.value ("lutPath", "");
            cs.shaderSource = resolveShaderSource (c);   // P3: GLSL/Slang/SPIR-V
            cs.isAdjustment = c.value ("isAdjustment", false);  // M8 "effect the world"
            if (c.contains ("genParams") && c["genParams"].is_object())
                for (auto it = c["genParams"].begin(); it != c["genParams"].end(); ++it)
                    if (it.value().is_number())
                        cs.genParams[it.key()] = it.value().get<double>();
            if (c.contains ("genImages") && c["genImages"].is_object())
                for (auto it = c["genImages"].begin(); it != c["genImages"].end(); ++it)
                    if (it.value().is_string())
                        cs.genImages[it.key()] = it.value().get<std::string>();
            if (c.contains ("effects"))
                for (const auto& e : c["effects"])
                {
                    ExportEffectSlot slot;
                    slot.slot = e.value ("slot", -1);
                    slot.type = e.value ("type", -1);
                    slot.enabled = e.value ("enabled", true);
                    if (e.contains ("params"))
                        for (auto it = e["params"].begin(); it != e["params"].end(); ++it)
                            slot.params.emplace_back (it.key(), it.value().get<double>());
                    cs.effects.push_back (std::move (slot));
                }
            job.clips.push_back (std::move (cs));
        }

    if (params.contains ("paramTimeline"))
        for (const auto& p : params["paramTimeline"])
        {
            ExportParamSample ps;
            ps.paramId = p.value ("paramId", "");
            ps.atSec = p.value ("atSec", 0.0);
            ps.value = p.value ("value", 0.0);
            if (! ps.paramId.empty())
                job.paramTimeline.push_back (std::move (ps));
        }

    // Block C symbolic score (M5). Parsed unconditionally (the GL frame loop is
    // the only consumer); mod_defs.h is plain C++17, no GL/GPL.
    if (params.contains ("score") && params["score"].is_object())
        parseScoreJson (params["score"], job.score);

    // Cross-domain modulation matrix (M6). Each routing maps a musical source
    // (Block A clock / Block B audio / Block C score) onto a render-graph clip
    // param (the same clip<id>/<node>/<param> namespace as paramTimeline) with
    // depth/curve/smoothingBeats/mode. The GL frame loop evaluates them per frame
    // via arbitmod::evaluateRouting. Parsed unconditionally; empty ⇒ no modulation.
    if (params.contains ("modMatrix"))
        parseRoutingsJson (params["modMatrix"], job.routings);

    if (params.contains("controlPlan"))
    {
        std::string controlPlanError;
        if (!parseVideoControlPlanJson(params["controlPlan"], job.controlPlan,
                                       controlPlanError))
            return controlPlanError;
    }

    if (params.contains ("texts"))
    {
        // Standard base64 ('+'/'/', '=' padding — what juce::Base64
        // emits). Returns false on any non-base64 character.
        auto base64Decode = [] (const std::string& in,
                                std::vector<uint8_t>& out) -> bool
        {
            auto val = [] (char c) -> int
            {
                if (c >= 'A' && c <= 'Z') return c - 'A';
                if (c >= 'a' && c <= 'z') return c - 'a' + 26;
                if (c >= '0' && c <= '9') return c - '0' + 52;
                if (c == '+') return 62;
                if (c == '/') return 63;
                return -1;
            };
            out.clear();
            out.reserve (in.size() / 4 * 3);
            int buf = 0, bits = 0;
            for (char c : in)
            {
                if (c == '=' || c == '\n' || c == '\r') continue;
                const int v = val (c);
                if (v < 0) return false;
                buf = (buf << 6) | v;
                bits += 6;
                if (bits >= 8)
                {
                    bits -= 8;
                    out.push_back ((uint8_t) ((buf >> bits) & 0xff));
                }
            }
            return true;
        };

        for (const auto& t : params["texts"])
        {
            ExportTextOverlay tx;
            tx.textId = t.value ("textId", 0);
            tx.ownerClipId = t.value ("ownerClipId", -1);
            tx.startSec = t.value ("startSec", 0.0);
            tx.durationSec = t.value ("durationSec", 0.0);
            tx.posX = t.value ("posX", 0.0);
            tx.posY = t.value ("posY", 0.0);
            tx.opacity = t.value ("opacity", 1.0);
            tx.zOrder = t.value ("zOrder", 0);
            tx.width = t.value ("width", 0);
            tx.height = t.value ("height", 0);
            if (t.contains ("rgbaBase64")
                && (! base64Decode (t["rgbaBase64"].get<std::string>(), tx.rgba)
                    || tx.rgba.size()
                           != (size_t) tx.width * (size_t) tx.height * 4))
                tx.rgba.clear(); // bad pixels: keep timing, draw nothing
            job.texts.push_back (std::move (tx));
        }
    }
    return {};
}

static std::string base64Encode (const std::vector<uint8_t>& bytes)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve ((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3)
    {
        const uint32_t a = bytes[i];
        const uint32_t b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
        const uint32_t c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
        const uint32_t word = (a << 16) | (b << 8) | c;
        out.push_back (alphabet[(word >> 18) & 63]);
        out.push_back (alphabet[(word >> 12) & 63]);
        out.push_back (i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=');
        out.push_back (i + 2 < bytes.size() ? alphabet[word & 63] : '=');
    }
    return out;
}

// `export` runs on a worker thread with a DEFERRED reply: the RPC loop keeps
// serving export_progress / export_cancel (and viewport traffic) while the
// render runs, and the original request id is answered when it finishes.
void handleExportAsync (const json& idVal, const json& params)
{
    if (g_export.active.load())
    {
        replyError (idVal, "export already running");
        return;
    }
    if (g_export.worker.joinable())
        g_export.worker.join();

    auto job = std::make_shared<ExportJob>();
    if (auto err = parseExportJob (params, *job); ! err.empty())
    {
        replyError (idVal, err);
        return;
    }

    // A user export outranks background cache work. Cancel and join before
    // claiming the shared compositor gate: joining while holding an export
    // lease would deadlock waiting for the cache worker to release its lease.
    {
        std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
        g_renderCache.progress.abort.store (true);
        if (g_renderCache.worker.joinable())
            g_renderCache.worker.join();
    }
    auto compositorLease = g_compositorOwnership.tryClaim (
        videohelper::CompositorOwnershipGate::Owner::exportJob);
    if (! compositorLease)
    {
        replyError (idVal, "busy: production GPU compositor already owned");
        return;
    }

    g_export.progress.frame.store (0);
    g_export.progress.totalFrames.store (0);
    g_export.progress.encodeFps.store (0.0);
    g_export.progress.phase.store (1);
    g_export.progress.abort.store (false);
    g_export.progress.visualTelemetry.resetSession();
    g_export.done.store (false);
    g_export.cancelled.store (false);
    g_export.jobId.fetch_add (1);
    g_export.active.store (true);

    try
    {
        g_export.worker = std::thread ([idVal, job, compositorLease]
        {
        std::string usedEncoder, interpolationBackend;
        bool glCompositing = false;
        // Don't run viewport RIFE while the exporter holds the GPU (two CUDA
        // sessions would contend for VRAM) — the viewport falls back to Frame
        // Blend for the duration.
#if ARBIT_HAVE_VIEWPORT
        ViewportInterpSuspendGuard interpGuard; // restores on scope exit (exception-safe)
#endif
        const std::string error = runExport (*job, usedEncoder, glCompositing,
                                             interpolationBackend,
                                             &g_export.progress);
        json result;
        if (error.empty())
            result = { { "outPath", job->outPath }, { "encoder", usedEncoder },
                       { "glCompositing", glCompositing },
                       { "interpolationBackend", interpolationBackend } };
#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
        if (error.empty() && (! job->luaScript.empty() || ! job->jsScript.empty()))
            result["testScriptLimitsAtOperation"] = {
                { "cpuMs", job->scriptCpuMs }, { "memoryMiB", job->scriptMemoryMiB } };
#endif
        {
            std::lock_guard<std::mutex> lock (g_export.resultMutex);
            g_export.error = error;
            g_export.result = result;
        }
        g_export.cancelled.store (error == "cancelled");
        g_export.progress.phase.store (0);
        g_export.done.store (true);
        g_export.active.store (false);
        if (error.empty())
            reply (idVal, result);
        else
            replyError (idVal, error);
        });
    }
    catch (const std::system_error& e)
    {
        const std::string error = std::string ("failed to start export worker: ") + e.what();
        {
            std::lock_guard<std::mutex> lock (g_export.resultMutex);
            g_export.error = error;
            g_export.result = json {};
        }
        g_export.progress.phase.store (0);
        g_export.done.store (true);
        g_export.active.store (false);
        compositorLease.reset();
        replyError (idVal, error);
    }
}

// `proxy_generate` runs on a worker thread with a DEFERRED reply, exactly
// like `export` (PROTOCOL.md §Proxy media). Resolution is set by `scale`
// (0.1..1.0 of source); the legacy `preset` ("native_intra" | "half_intra")
// is still accepted as a fallback when `scale` is absent. All-keyframe H.264.
void handleProxyAsync (const json& idVal, const json& params)
{
    if (g_proxy.active.load())
    {
        replyError (idVal, "proxy job already running");
        return;
    }
    if (g_proxy.worker.joinable())
        g_proxy.worker.join();

    const std::string sourcePath = params.value ("sourcePath", "");
    const std::string outPath = params.value ("outPath", "");
    if (sourcePath.empty() || outPath.empty())
    {
        replyError (idVal, "missing sourcePath/outPath");
        return;
    }

    // `scale` is the modern control; fall back to the legacy preset when absent.
    double scale = params.value ("scale", 0.0);
    if (scale <= 0.0)
    {
        const std::string preset = params.value ("preset", "native_intra");
        if (preset == "half_intra")            scale = 0.5;
        else if (preset == "native_intra")     scale = 1.0;
        else { replyError (idVal, "unknown preset: " + preset); return; }
    }
    scale = std::max (0.1, std::min (1.0, scale));

    g_proxy.progress.frame.store (0);
    g_proxy.progress.totalFrames.store (0);
    g_proxy.progress.abort.store (false);
    g_proxy.done.store (false);
    g_proxy.cancelled.store (false);
    g_proxy.jobId.fetch_add (1);
    g_proxy.active.store (true);

    g_proxy.worker = std::thread ([idVal, sourcePath, outPath, scale]
    {
        const std::string error = MediaContext::generateProxy (sourcePath, outPath,
                                                               scale,
                                                               &g_proxy.progress);
        json result;
        if (error.empty())
            result = { { "outPath", outPath }, { "scale", scale },
                       { "frames", g_proxy.progress.frame.load() } };
        {
            std::lock_guard<std::mutex> lock (g_proxy.resultMutex);
            g_proxy.error = error;
            g_proxy.result = result;
        }
        g_proxy.cancelled.store (error == "cancelled");
        g_proxy.done.store (true);
        g_proxy.active.store (false);
        if (error.empty())
            reply (idVal, result);
        else
            replyError (idVal, error);
    });
}

// `render_cache_build` (PROTOCOL.md §Render cache): renders ONE clip's
// per-layer effect pass (effect slots + LUT) over its source media into an
// intra-only H.264 file at source resolution, via the same offscreen-GL
// runExport path as `export`. Deferred reply like export/proxy. The jobSpec
// shape is parseExportJob's; intraOnly is forced. A CPU-fallback render
// (no GL) would silently DROP the effects — that is a hard failure here:
// the file is removed and the reply errors, so the plugin keeps rendering
// the clip live instead of substituting an unbaked copy.
void handleRenderCacheAsync (const json& idVal, const json& params)
{
    if (g_export.active.load())
    {
        replyError (idVal, "busy: export running");
        return;
    }
    if (g_renderCache.active.load())
    {
        replyError (idVal, "busy: render cache build already running");
        return;
    }
    {
        std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
        if (g_renderCache.worker.joinable())
            g_renderCache.worker.join();
    }

    auto job = std::make_shared<ExportJob>();
    if (auto err = parseExportJob (params, *job); ! err.empty())
    {
        replyError (idVal, err);
        return;
    }
    job->intraOnly = true;
    job->audioPath.clear(); // video-only bake
    auto compositorLease = g_compositorOwnership.tryClaim (
        videohelper::CompositorOwnershipGate::Owner::renderCache);
    if (! compositorLease)
    {
        replyError (idVal, "busy: production GPU compositor already owned");
        return;
    }

    g_renderCache.progress.frame.store (0);
    g_renderCache.progress.totalFrames.store (0);
    g_renderCache.progress.encodeFps.store (0.0);
    g_renderCache.progress.phase.store (1);
    g_renderCache.progress.abort.store (false);
    g_renderCache.done.store (false);
    g_renderCache.cancelled.store (false);
    g_renderCache.jobId.fetch_add (1);
    g_renderCache.active.store (true);

    std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
    try
    {
        g_renderCache.worker = std::thread ([idVal, job, compositorLease]
        {
        std::string usedEncoder, interpolationBackend;
        bool glCompositing = false;
#if ARBIT_HAVE_VIEWPORT
        g_viewport.setInterpolationSuspended (true);   // free the GPU for the cache build
#endif
        std::string error = runExport (*job, usedEncoder, glCompositing,
                                       interpolationBackend,
                                       &g_renderCache.progress);
#if ARBIT_HAVE_VIEWPORT
        g_viewport.setInterpolationSuspended (false);
#endif
        if (error.empty() && ! glCompositing)
        {
            std::remove (job->outPath.c_str());
            error = "render cache requires GL compositing (headless/no-GL fallback "
                    "would drop the baked effects)";
        }
        json result;
        if (error.empty())
            result = { { "outPath", job->outPath }, { "encoder", usedEncoder },
                       { "frames", (int64_t) g_renderCache.progress.frame.load() } };
#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
        if (error.empty() && (! job->luaScript.empty() || ! job->jsScript.empty()))
            result["testScriptLimitsAtOperation"] = {
                { "cpuMs", job->scriptCpuMs }, { "memoryMiB", job->scriptMemoryMiB } };
#endif
        {
            std::lock_guard<std::mutex> lock (g_renderCache.resultMutex);
            g_renderCache.error = error;
            g_renderCache.result = result;
        }
        g_renderCache.cancelled.store (error == "cancelled");
        g_renderCache.progress.phase.store (0);
        g_renderCache.done.store (true);
        g_renderCache.active.store (false);
        if (error.empty())
            reply (idVal, result);
        else
            replyError (idVal, error);
        });
    }
    catch (const std::system_error& e)
    {
        const std::string error = std::string ("failed to start render cache worker: ") + e.what();
        {
            std::lock_guard<std::mutex> lock (g_renderCache.resultMutex);
            g_renderCache.error = error;
            g_renderCache.result = json {};
        }
        g_renderCache.progress.phase.store (0);
        g_renderCache.done.store (true);
        g_renderCache.active.store (false);
        compositorLease.reset();
        replyError (idVal, error);
    }
}

json handle (const std::string& method, const json& params, std::string& error)
{
    if (method == "ping")
        return "pong";

    if (method == "version")
        return json { { "name", "arbit-video-helper" }, { "version", "0.1.0" },
                      { "avformat", LIBAVFORMAT_VERSION_MAJOR },
                      { "vidstab", MediaContext::vidstabAvailable() },
                      { "proxy", true },
                      { "renderCache", true },
#if ARBIT_HAVE_AUBIO
                      { "beatDetection", true } };
#else
                      { "beatDetection", false } };
#endif

    if (method == "gpu_backend_info" || method == "gpu_backend_selftest")
    {
        if (method == "gpu_backend_info")
        {
            const auto info = arbitgpu::queryNativeBackend();
            return json { { "available", info.available }, { "backend", info.backend },
                          { "device", info.device }, { "compute", info.compute },
                          { "error", info.error } };
        }

        const auto test = arbitgpu::runNativeBackendSelfTest();
        return json { { "ok", test.computePassed && test.renderPassed },
                      { "available", test.available }, { "backend", test.backend },
                      { "device", test.device }, { "compute", test.compute },
                      { "computePassed", test.computePassed },
                      { "renderPassed", test.renderPassed },
                      { "computeChecksum", test.computeChecksum },
                      { "renderChecksum", test.renderChecksum },
                      { "error", test.error } };
    }

    if (method == "rife_selftest")
    {
#if defined(ARBIT_HAVE_NCNN) || defined(ARBIT_HAVE_ONNX)
        arbitselftest::RifeBackend rife;
        std::string initErr = rife.init();
        if (! initErr.empty())
            return json { { "ok", false }, { "backend", "" }, { "error", initErr } };

        // Tiny synthetic frame pair (solid red -> solid blue); real inference,
        // not a representative image, but enough to exercise init+interpolate
        // end to end on whatever backend/GPU actually engaged.
        const int w = 64, h = 64;
        std::vector<uint8_t> frame0 (static_cast<size_t> (w) * h * 4);
        std::vector<uint8_t> frame1 (static_cast<size_t> (w) * h * 4);
        for (int i = 0; i < w * h; ++i)
        {
            frame0[i * 4 + 0] = 255; frame0[i * 4 + 1] = 0;   frame0[i * 4 + 2] = 0;   frame0[i * 4 + 3] = 255;
            frame1[i * 4 + 0] = 0;   frame1[i * 4 + 1] = 0;   frame1[i * 4 + 2] = 255; frame1[i * 4 + 3] = 255;
        }
        std::vector<uint8_t> out;
        std::string interpErr = rife.interpolate (frame0.data(), w * 4, frame1.data(), w * 4,
                                                   w, h, 0.5f, out);
        if (! interpErr.empty())
            return json { { "ok", false }, { "backend", rife.backend() }, { "error", interpErr } };

        return json { { "ok", true }, { "backend", rife.backend() },
                      { "outputBytes", static_cast<int> (out.size()) } };
#else
        return json { { "ok", false }, { "backend", "" }, { "error", "no RIFE backend compiled in" } };
#endif
    }

    if (method == "open" || method == "probe")
    {
        if (! params.contains ("path")) { error = "missing path"; return {}; }
        auto ctx = std::make_unique<MediaContext>();
        // Optional image-sequence hints (pattern paths only): fps sets the
        // image2 frame rate, startNumber the first frame number.
        error = ctx->open (params["path"].get<std::string>(), true,
                           params.value ("fps", 0.0),
                           params.value ("startNumber", -1));
        if (! error.empty()) return {};

        const auto& mi = ctx->info();
        json info = {
            { "durationSec", mi.durationSec }, { "fps", mi.fps },
            { "width", mi.width }, { "height", mi.height },
            { "hasVideo", mi.hasVideo }, { "hasAudio", mi.hasAudio },
            { "hasAlpha", mi.hasAlpha },
            { "audioSampleRate", mi.audioSampleRate },
            { "audioChannels", mi.audioChannels },
            { "container", mi.container },
            { "videoCodec", mi.videoCodec }, { "audioCodec", mi.audioCodec },
            { "hwaccel", mi.hwaccel },
        };
        if (method == "probe")
            return info;

        const uint32_t id = g_nextMediaId++;
        g_media[id] = std::move (ctx);
        info["mediaId"] = id;
        return info;
    }

    if (method == "close")
    {
        if (params.contains ("mediaId"))
            g_media.erase (params["mediaId"].get<uint32_t>());
        return json { { "ok", true } };
    }

    if (method == "attach_shm")
    {
        if (! params.contains ("name")) { error = "missing name"; return {}; }
        if (! g_shm.open (params["name"].get<std::string>()))
        {
            error = "cannot open shared memory region";
            return {};
        }
        auto* h = g_shm.header();
        return json { { "slotCount", h->slotCount }, { "slotBytes", h->slotBytes } };
    }

    if (method == "request_frame")
    {
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        if (! g_shm.isOpen()) { error = "attach_shm not called"; return {}; }

        const double timeSec = params.value ("timeSec", 0.0);
        int maxW = params.value ("maxW", 1920);
        int maxH = params.value ("maxH", 1080);

        // Never exceed what a slot can hold.
        auto* h = g_shm.header();
        while ((uint32_t) (maxW * maxH * 4) > h->slotBytes && maxW > 16)
        {
            maxW /= 2;
            maxH /= 2;
        }

        DecodedFrame df;
        error = media->getFrame (timeSec, maxW, maxH, df);
        if (! error.empty()) return {};

        const uint32_t slotIndex = g_nextSlot;
        g_nextSlot = (g_nextSlot + 1) % h->slotCount;
        auto* slot = g_shm.slot (slotIndex);
        uint8_t* payload = g_shm.slotPayload (slotIndex);
        if (slot == nullptr || payload == nullptr) { error = "bad shm slot"; return {}; }

        slot->generation.fetch_add (1, std::memory_order_acq_rel); // -> odd: writing
        slot->width = (uint32_t) df.width;
        slot->height = (uint32_t) df.height;
        slot->strideBytes = (uint32_t) df.strideBytes;
        slot->ptsSec = df.ptsSec;
        slot->mediaId = params["mediaId"].get<uint32_t>();
        std::memcpy (payload, df.rgba.data(), df.rgba.size());
        slot->generation.fetch_add (1, std::memory_order_acq_rel); // -> even: ready

        return json { { "slot", slotIndex }, { "ptsSec", df.ptsSec },
                      { "width", df.width }, { "height", df.height },
                      { "strideBytes", df.strideBytes } };
    }

    if (method == "capture_list_sources")
    {
        std::string captureError;
        const auto devices = listCaptureDevices (params.value ("kind", std::string()), captureError);
        json sources = json::array();
        for (const auto& device : devices)
            sources.push_back ({ { "id", device.id }, { "name", device.name },
                                 { "kind", device.kind }, { "backend", device.backend },
                                 { "availableFormats", json::array() } });
        return json { { "sources", sources }, { "enumerationError", captureError } };
    }

    if (method == "capture_preview_open")
    {
        for (const auto* required : { "sourceId", "backend", "shmName" })
            if (! params.contains(required)) { error = std::string("missing ") + required; return {}; }
        auto endpoint = std::make_shared<CaptureEndpoint>();
        if (! endpoint->shm.open(params["shmName"].get<std::string>()))
        {
            error = "cannot open capture shared memory region";
            return {};
        }
        const uint32_t captureId = g_nextCaptureSessionId++;
        const int width = params.value("width", 640);
        const int height = params.value("height", 360);
        const double fps = params.value("fps", 30.0);
        auto openError = endpoint->capture.open(params["sourceId"].get<std::string>(),
            params["backend"].get<std::string>(), width, height, fps,
            [endpoint, captureId] (const uint8_t* pixels, const CaptureFrameInfo& frame)
            {
                const std::lock_guard<std::mutex> lock(endpoint->frameMutex);
                auto* header = endpoint->shm.header();
                if (header == nullptr) return;
                const size_t bytes = static_cast<size_t>(frame.strideBytes) * static_cast<size_t>(frame.height);
                if (bytes == 0 || bytes > header->slotBytes) return;
                const uint32_t slotIndex = endpoint->nextSlot++ % header->slotCount;
                auto* slot = endpoint->shm.slot(slotIndex);
                auto* payload = endpoint->shm.slotPayload(slotIndex);
                if (slot == nullptr || payload == nullptr) return;
                slot->generation.fetch_add(1, std::memory_order_acq_rel);
                slot->width = static_cast<uint32_t>(frame.width);
                slot->height = static_cast<uint32_t>(frame.height);
                slot->strideBytes = static_cast<uint32_t>(frame.strideBytes);
                slot->ptsSec = frame.timestampSec;
                slot->mediaId = captureId;
                std::memcpy(payload, pixels, bytes);
                slot->generation.fetch_add(1, std::memory_order_acq_rel);
                endpoint->latestSlot = slotIndex;
            });
        if (! openError.empty())
        {
            endpoint->shm.close();
            error = openError;
            return {};
        }
        {
            const std::lock_guard<std::mutex> lock(g_captureSessionsMutex);
            g_captureSessions[captureId] = endpoint;
        }
        return json { { "captureId", captureId } };
    }

    if (method == "capture_preview_latest")
    {
        if (! params.contains("captureId")) { error = "missing captureId"; return {}; }
        std::shared_ptr<CaptureEndpoint> endpoint;
        {
            const std::lock_guard<std::mutex> lock(g_captureSessionsMutex);
            const auto found = g_captureSessions.find(params["captureId"].get<uint32_t>());
            if (found == g_captureSessions.end()) { error = "unknown captureId"; return {}; }
            endpoint = found->second;
        }
        const auto frame = endpoint->capture.latestFrame();
        if (frame.sequence == 0)
        {
            const auto captureError = endpoint->capture.lastError();
            if (! captureError.empty()) { error = captureError; return {}; }
            return json { { "pending", true } };
        }
        const std::lock_guard<std::mutex> lock(endpoint->frameMutex);
        return json { { "slot", endpoint->latestSlot }, { "sequence", frame.sequence },
                      { "timestampSec", frame.timestampSec }, { "width", frame.width },
                      { "height", frame.height }, { "strideBytes", frame.strideBytes } };
    }

    if (method == "capture_record_start" || method == "capture_record_stop")
    {
        if (! params.contains("captureId") || ! params.contains("recordingId"))
        { error = "missing captureId or recordingId"; return {}; }
        std::shared_ptr<CaptureEndpoint> endpoint;
        {
            const std::lock_guard<std::mutex> lock(g_captureSessionsMutex);
            const auto found = g_captureSessions.find(params["captureId"].get<uint32_t>());
            if (found == g_captureSessions.end()) { error = "unknown captureId"; return {}; }
            endpoint = found->second;
        }
        const uint32_t recordingId = params["recordingId"].get<uint32_t>();
        if (method == "capture_record_start")
        {
            if (! params.contains("outPath")) { error = "missing outPath"; return {}; }
            error = endpoint->capture.startRecording(recordingId, params["outPath"].get<std::string>(),
                                                       params.value("codec", std::string("h264")));
        }
        else
            error = endpoint->capture.stopRecording(recordingId, params.value("cancel", false));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "capture_close")
    {
        if (! params.contains("captureId")) { error = "missing captureId"; return {}; }
        std::shared_ptr<CaptureEndpoint> endpoint;
        {
            const std::lock_guard<std::mutex> lock(g_captureSessionsMutex);
            const auto found = g_captureSessions.find(params["captureId"].get<uint32_t>());
            if (found == g_captureSessions.end()) return json { { "ok", true } };
            endpoint = found->second;
            g_captureSessions.erase(found);
        }
        endpoint->capture.close();
        endpoint->shm.close();
        return json { { "ok", true } };
    }

    // Recorder protocol v2: every operation is scoped to an explicit recordingId.
    if (method == "record_open")
    {
        if (! params.contains ("shmName")) { error = "missing shmName"; return {}; }
        if (! params.contains ("outPath")) { error = "missing outPath"; return {}; }

        auto session = std::make_unique<RecorderCaptureSession>();
        if (! session->shm.open (params["shmName"].get<std::string>()))
        {
            error = "cannot open recorder shared memory region";
            return {};
        }
        session->outPath = params["outPath"].get<std::string>();
        error = session->recorder.open (session->outPath,
                                        params.value ("width", 1920),
                                        params.value ("height", 1080),
                                        params.value ("fps", 30.0),
                                        params.value ("codec", std::string ("h264")),
                                        params.value ("encoder", std::string ("auto")));
        if (! error.empty())
        {
            session->shm.close();
            return {};
        }

        std::lock_guard<std::mutex> lock (g_recorderSessionsMutex);
        const uint32_t recordingId = g_nextRecorderSessionId++;
        g_recorderSessions.emplace (recordingId, std::move (session));
        return json { { "ok", true }, { "recordingId", recordingId } };
    }

    if (method == "record_push_frame")
    {
        if (! params.contains ("recordingId")) { error = "missing recordingId"; return {}; }
        if (! params.contains ("slot")) { error = "missing slot"; return {}; }
        std::lock_guard<std::mutex> lock (g_recorderSessionsMutex);
        const auto it = g_recorderSessions.find (params["recordingId"].get<uint32_t>());
        if (it == g_recorderSessions.end()) { error = "unknown recordingId"; return {}; }
        auto& session = *it->second;
        if (! session.shm.isOpen() || ! session.recorder.isOpen())
        {
            error = "recording session not open";
            return {};
        }

        const uint32_t slotIndex = params["slot"].get<uint32_t>();
        auto* h = session.shm.header();
        if (h == nullptr) { error = "recorder shm not attached"; return {}; }
        std::vector<uint8_t> pixels (h->slotBytes);
        uint32_t width = 0, height = 0, strideBytes = 0;
        double ptsSec = 0.0;
        if (! session.shm.readSlot (slotIndex, width, height, strideBytes, ptsSec,
                                    pixels.data(), pixels.size()))
        {
            error = "recorder slot torn or invalid (dropped frame, caller may retry)";
            return {};
        }
        error = session.recorder.pushFrame (pixels.data(), (int) strideBytes, /*bgra*/ true);
        if (! error.empty()) return {};
        return json { { "ok", true }, { "framesEncoded", session.recorder.framesEncoded() } };
    }

    if (method == "record_close" || method == "record_cancel")
    {
        if (! params.contains ("recordingId")) { error = "missing recordingId"; return {}; }
        std::unique_ptr<RecorderCaptureSession> session;
        {
            std::lock_guard<std::mutex> lock (g_recorderSessionsMutex);
            const auto it = g_recorderSessions.find (params["recordingId"].get<uint32_t>());
            if (it == g_recorderSessions.end()) { error = "unknown recordingId"; return {}; }
            session = std::move (it->second);
            g_recorderSessions.erase (it);
        }
        error = session->recorder.close();
        session->shm.close();
        if (! error.empty()) return {};
        if (method == "record_cancel")
            std::remove (session->outPath.c_str());
        return json { { "ok", true } };
    }

    if (method == "extract_audio")
    {
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        if (! params.contains ("outPath")) { error = "missing outPath"; return {}; }

        double durationSec = 0.0;
        int sampleRate = 0, channels = 0;
        error = media->extractAudio (params["outPath"].get<std::string>(),
                                     durationSec, sampleRate, channels);
        if (! error.empty()) return {};
        return json { { "outPath", params["outPath"] }, { "durationSec", durationSec },
                      { "sampleRate", sampleRate }, { "channels", channels } };
    }

    if (method == "detect_beats")
    {
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        double bpm = 0.0, confidence = 0.0;
        std::vector<double> beats;
        error = media->detectBeats (bpm, confidence, beats);
        if (! error.empty()) return {};
        return json { { "bpm", bpm }, { "confidence", confidence }, { "beats", beats } };
    }

    if (method == "stabilize_detect")
    {
        // Pass 1 of 2 (PROTOCOL.md §Stabilization): vidstabdetect over
        // [inSec, outSec) writes per-frame motion transforms to trfPath.
        // Blocking — the caller caches the .trf and runs this off-thread.
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        if (! params.contains ("trfPath")) { error = "missing trfPath"; return {}; }
        int frames = 0;
        error = media->stabilizeDetect (params["trfPath"].get<std::string>(),
                                        params.value ("inSec", 0.0),
                                        params.value ("outSec", 0.0), frames);
        if (! error.empty()) return {};
        return json { { "trfPath", params["trfPath"] }, { "frames", frames } };
    }

    if (method == "stabilize_render")
    {
        // Pass 2 of 2: vidstabtransform (input=trfPath) over the SAME range,
        // h264-encoded intermediate at outPath. strength 0..1 -> smoothing.
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        if (! params.contains ("trfPath") || ! params.contains ("outPath"))
        {
            error = "missing trfPath/outPath";
            return {};
        }
        int frames = 0;
        error = media->stabilizeRender (params["trfPath"].get<std::string>(),
                                        params["outPath"].get<std::string>(),
                                        params.value ("inSec", 0.0),
                                        params.value ("outSec", 0.0),
                                        params.value ("strength", 0.5), frames);
        if (! error.empty()) return {};
        return json { { "outPath", params["outPath"] }, { "frames", frames } };
    }

    if (method == "recipe_preview")
    {
        if (g_export.active.load() || g_renderCache.active.load())
        {
            error = "production GPU compositor is busy";
            return {};
        }
        ExportJob job;
        if (auto parseError = parseExportJob(params, job); ! parseError.empty())
        {
            error = parseError;
            return {};
        }
        if (job.width < 1 || job.height < 1 || job.width > 640 || job.height > 360)
        {
            error = "recipe preview dimensions are outside the bounded GPU preview size";
            return {};
        }
        job.durationSec = 1.0 / std::max(1.0, job.fps);
        job.startSec = 0.0;
        job.endSec = job.durationSec;
        job.encoder = "software";
        job.codec = "h264";
        std::string encoder, interpolation;
        bool gpuComposited = false;
        error = runExport(job, encoder, gpuComposited, interpolation, nullptr);
        if (! error.empty()) return {};
        if (! gpuComposited)
        {
            std::filesystem::remove(job.outPath);
            error = "recipe preview requires the production GPU compositor";
            return {};
        }
        MediaContext rendered;
        if (auto openError = rendered.open(job.outPath, false); ! openError.empty())
        {
            std::filesystem::remove(job.outPath);
            error = openError;
            return {};
        }
        std::vector<std::string> paths;
        error = rendered.writeThumbnails({ 0.0 }, job.width, job.height,
            params.value("previewDirectory", std::string {}),
            params.value("previewBaseName", std::string { "visual-recipe" }), paths);
        std::filesystem::remove(job.outPath);
        if (! error.empty() || paths.empty())
        {
            if (error.empty()) error = "recipe preview produced no rendered image";
            return {};
        }
        return json { { "path", paths.front() }, { "gpuComposited", true } };
    }

    if (method == "thumbnails")
    {
        auto* media = findMedia (params, error);
        if (media == nullptr) return {};
        if (! params.contains ("times") || ! params.contains ("outDir")
            || ! params.contains ("baseName"))
        {
            error = "missing times/outDir/baseName";
            return {};
        }
        std::vector<double> times = params["times"].get<std::vector<double>>();
        std::vector<std::string> paths;
        error = media->writeThumbnails (times,
                                        params.value ("maxW", 320),
                                        params.value ("maxH", 180),
                                        params["outDir"].get<std::string>(),
                                        params["baseName"].get<std::string>(),
                                        paths);
        if (! error.empty()) return {};
        return json { { "paths", paths } };
    }

    // NOTE: "export" is handled in main() (deferred reply — handleExportAsync).
    // NOTE: "proxy_generate" too (deferred reply — handleProxyAsync).

    if (method == "composite_frame_probe")
    {
        auto compositorLease = g_compositorOwnership.tryClaim (
            videohelper::CompositorOwnershipGate::Owner::frameProbe);
        if (! compositorLease)
        {
            error = "production GPU compositor is busy";
            return {};
        }
        ExportJob job;
        if (error = parseExportJob (params, job); ! error.empty()) return {};
        const double timelineSec = params.value ("timelineSec", -1.0);
        videowire::ResolvedVisualSnapshot identity;
        identity.authoringRevision = job.authoringRevision;
        identity.segments = job.segments;
        identity.visualLayerPlans = job.visualLayerPlans;
        identity.visualEventSchedules = job.visualEventSchedules;
        const auto snapshotGeneration = params.value ("snapshotGeneration", uint64_t { 0 });
        if (! videohelper::validateCompositeProbeContract (
                identity, snapshotGeneration, timelineSec,
                job.width, job.height, job.fps, error)) return {};
        const auto clipRevisions = params.value ("clipRevisions", json::array());
        if (! clipRevisions.is_array())
        {
            error = "snapshot clip revision records do not match compiled plans";
            return {};
        }
        std::vector<videohelper::SnapshotClipIdentity> identityRecords;
        for (const auto& record : clipRevisions)
            identityRecords.push_back ({ record.value ("clipId", -1),
                record.value ("authoredStructuralRevision", uint64_t { 0 }),
                record.value ("identityMode", std::string {}),
                record.value ("snapshotCompiledExportable", false) });
        if (! videohelper::validateSnapshotClipIdentities (
                job.visualLayerPlans, identityRecords, error))
            return {};
        CompositeFrameResult frame;
        error = renderCompositeFrame (job, timelineSec, frame);
        if (! error.empty()) return {};
        return json {
            { "width", frame.width }, { "height", frame.height },
            { "format", "rgba8" }, { "strideBytes", frame.width * 4 },
            { "snapshotGeneration", snapshotGeneration },
            { "clipRevisions", clipRevisions },
            { "graphLayerCount", job.visualLayerPlans.size() },
            { "compositorBackend", frame.compositorBackend },
            { "presentationBackend", frame.presentationBackend },
            { "rgbaBase64", base64Encode (frame.rgba) }
#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
            , { "testScriptLimitsAtOperation", {
                { "cpuMs", job.scriptCpuMs }, { "memoryMiB", job.scriptMemoryMiB } } }
#endif
        };
    }

    if (method == "proxy_progress")
    {
        // Cheap poll (PROTOCOL.md §Proxy media), same contract as
        // export_progress: after completion the final result/error stays
        // readable until the next proxy job starts.
        json r = {
            { "active", g_proxy.active.load() },
            { "done", g_proxy.done.load() },
            { "cancelled", g_proxy.cancelled.load() },
            { "jobId", g_proxy.jobId.load() },
            { "frame", g_proxy.progress.frame.load() },
            { "totalFrames", g_proxy.progress.totalFrames.load() },
        };
        if (g_proxy.done.load())
        {
            std::lock_guard<std::mutex> lock (g_proxy.resultMutex);
            if (! g_proxy.error.empty())
                r["error"] = g_proxy.error;
            else if (! g_proxy.result.is_null())
                r["result"] = g_proxy.result;
        }
        return r;
    }

    if (method == "proxy_cancel")
    {
        // The transcode loop polls the abort flag once per packet; the worker
        // removes the partial file and the deferred reply errors "cancelled".
        g_proxy.progress.abort.store (true);
        return json { { "ok", true }, { "active", g_proxy.active.load() } };
    }

    // NOTE: "render_cache_build" is handled in main() (deferred reply —
    // handleRenderCacheAsync).

    if (method == "render_cache_progress")
    {
        // Cheap poll (PROTOCOL.md §Render cache), same contract as
        // export_progress/proxy_progress: after completion the final
        // result/error stays readable until the next build starts.
        json r = {
            { "active", g_renderCache.active.load() },
            { "done", g_renderCache.done.load() },
            { "cancelled", g_renderCache.cancelled.load() },
            { "jobId", g_renderCache.jobId.load() },
            { "frame", (int64_t) g_renderCache.progress.frame.load() },
            { "totalFrames", (int64_t) g_renderCache.progress.totalFrames.load() },
        };
        if (g_renderCache.done.load())
        {
            std::lock_guard<std::mutex> lock (g_renderCache.resultMutex);
            if (! g_renderCache.error.empty())
                r["error"] = g_renderCache.error;
            else if (! g_renderCache.result.is_null())
                r["result"] = g_renderCache.result;
        }
        return r;
    }

    if (method == "render_cache_cancel")
    {
        // The frame loop polls the abort flag; runExport removes the partial
        // file and the deferred reply errors "cancelled".
        g_renderCache.progress.abort.store (true);
        return json { { "ok", true }, { "active", g_renderCache.active.load() } };
    }

    if (method == "export_progress")
    {
        // Cheap poll (PROTOCOL.md §Export): answered by the RPC loop while
        // the export worker renders. After completion the final result/error
        // stays readable here until the next export starts.
        static const char* phaseNames[] = { "idle", "setup", "audio", "video",
                                            "finalize" };
        const int phase = std::clamp (g_export.progress.phase.load(), 0, 4);
        json r = {
            { "active", g_export.active.load() },
            { "done", g_export.done.load() },
            { "cancelled", g_export.cancelled.load() },
            { "jobId", g_export.jobId.load() },
            { "phase", phaseNames[phase] },
            { "frame", g_export.progress.frame.load() },
            { "totalFrames", g_export.progress.totalFrames.load() },
            { "fps", g_export.progress.encodeFps.load() },
            { "visualTelemetry", videowire::visualTelemetryJson(
                g_export.progress.visualTelemetry.snapshot()) },
        };
        if (g_export.done.load())
        {
            std::lock_guard<std::mutex> lock (g_export.resultMutex);
            if (! g_export.error.empty())
                r["error"] = g_export.error;
            else if (! g_export.result.is_null())
                r["result"] = g_export.result;
        }
        return r;
    }

    if (method == "export_cancel")
    {
        // Sets the abort flag; the frame loop notices within a frame, the
        // worker deletes the partial file and the deferred `export` reply
        // errors with "cancelled".
        g_export.progress.abort.store (true);
        return json { { "ok", true }, { "active", g_export.active.load() } };
    }

    // ---- Viewport (GPU preview window) — protocol in video-helper/PROTOCOL.md

#if ! ARBIT_HAVE_VIEWPORT
    if (method.rfind ("viewport_", 0) == 0 || method.rfind ("graph_", 0) == 0
        || method.rfind ("text_", 0) == 0 || method.rfind ("scope_", 0) == 0
        || method == "attach_transport" || method == "attach_audio"
        || method == "get_av_offset")
    {
        error = "helper built without viewport support";
        return {};
    }
#else
    if (method == "viewport_open")
    {
        error = g_viewport.open (params.value ("width", 960),
                                 params.value ("height", 540),
                                 params.value ("x", -1),
                                 params.value ("y", -1),
                                 params.value ("alwaysOnTop", false),
                                 params.value ("targetFps", 60.0),
                                 params.value ("clkFps", 0.0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_close")
    {
        g_viewport.close();
        return json { { "ok", true } };
    }

    // Live project value-grid frame rate (frame-perfect parity): the rate the
    // exporter steps at. Re-snaps the preview's modulation/shader-clock/score/
    // lua/source grid to the same grid the export uses. Fire-and-forget.
    if (method == "viewport_set_clkfps")
    {
        g_viewport.setValueFps (params.value ("clkFps", 0.0));
        return json { { "ok", true } };
    }

    // Project-wide playback frame-rate up-conversion target. 0 ⇒ legacy
    // slowed-only interpolation; > 0 ⇒ interpolate any clip whose effective
    // cadence (sourceFps * rate) is below this fps up to it. See
    // Viewport::setInterpTargetFps.
    if (method == "viewport_set_target_fps")
    {
        g_viewport.setInterpTargetFps (params.value ("targetFps", 0.0));
        return json { { "ok", true } };
    }

    // Realtime interpolation resolution cap (long side, px). 0 ⇒ legacy 360p.
    // Higher = crisper preview, slower; export is uncapped. See
    // Viewport::setInterpMaxLongSide.
    if (method == "viewport_set_interp_max_dim")
    {
        g_viewport.setInterpMaxLongSide ((int) params.value ("maxDim", 0));
        return json { { "ok", true } };
    }

    // ---- Zero-copy docked viewport — PROTOCOL.md §Zero-copy docked
    // viewport. Platform mechanism behind one RPC surface: dmabuf (Linux),
    // IOSurface (macOS), D3D11 keyed-mutex interop (Windows); gpuPath in the
    // open reply names which. Builds without a backend fail cleanly from
    // Viewport::openShared and Arbit demotes down the ladder.
    if (method == "viewport_open_shared")
    {
        Viewport::SharedOpenResult r;
        error = g_viewport.openShared (params.value ("width", 960),
                                       params.value ("height", 540),
                                       params.value ("targetFps", 60.0),
                                       params.value ("bufferCount", 3), r,
                                       params.value ("clkFps", 0.0));
        if (! error.empty()) return {};
        return json { { "socketPath", r.socketPath },
                      { "gpuPath", Viewport::sharedGpuPathTag() },
                      { "fourcc", r.fourcc }, { "modifier", r.modifier },
                      { "bufferCount", r.bufferCount }, { "device", r.device } };
    }

    if (method == "viewport_resize_shared")
    {
        error = g_viewport.resizeShared (params.value ("width", 0),
                                         params.value ("height", 0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_close_shared")
    {
        g_viewport.close();
        return json { { "ok", true } };
    }

    // ---- CPU-shm docked viewport — PROTOCOL.md §shm-docked. The portable
    // docked rung (GLX/X11, Windows, macOS, DAW-hosted): same offscreen
    // render graph, frames delivered through an Arbit-owned shm ring.
    if (method == "viewport_open_shm")
    {
        error = g_viewport.openShm (params.value ("width", 960),
                                    params.value ("height", 540),
                                    params.value ("targetFps", 60.0),
                                    params.value ("shmName", std::string()),
                                    params.value ("clkFps", 0.0));
        if (! error.empty()) return {};
        return json { { "ok", true }, { "gpuPath", "shm" } };
    }

    if (method == "viewport_resize_shm")
    {
        error = g_viewport.resizeShm (params.value ("width", 0),
                                      params.value ("height", 0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_close_shm")
    {
        g_viewport.close();
        return json { { "ok", true } };
    }

    if (method == "viewport_set_bounds")
    {
        g_viewport.setBounds (params.value ("x", 100), params.value ("y", 100),
                              params.value ("width", 960), params.value ("height", 540));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_fullscreen")
    {
        g_viewport.setFullscreen (params.value ("fullscreen", false));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_canvas")
    {
        // Project canvas (PROTOCOL.md §Project canvas & view transform).
        // Accepted while closed too — applies on the next viewport open.
        error = g_viewport.setCanvas (params.value ("width", 0),
                                      params.value ("height", 0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_set_view")
    {
        error = g_viewport.setView (params.value ("zoom", 1.0),
                                    params.value ("panX", 0.0),
                                    params.value ("panY", 0.0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_set_canvas_frame_enabled")
    {
        g_viewport.setCanvasFrameEnabled (params.value ("enabled", true));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_canvas_background")
    {
        // Live canvas background colour (preview == export for M1's bgColor).
        // Same {r,g,b,a} schema as the export job's "canvasBackground".
        error = g_viewport.setCanvasBackground (params.value ("r", 0.04),
                                                params.value ("g", 0.04),
                                                params.value ("b", 0.05),
                                                params.value ("a", 1.0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_set_post")
    {
        // Live bloom/tonemap post stack (preview == export). Same schema as the
        // export job's "post" object; neutral values skip the pass.
        error = g_viewport.setPostFx (params.value ("bloomIntensity", 0.0),
                                      params.value ("bloomThreshold", 1.0),
                                      params.value ("bloomRadius", 0.0),
                                      params.value ("tonemap", 0),
                                      params.value ("exposure", 1.0));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "attach_transport")
    {
        if (! params.contains ("name")) { error = "missing name"; return {}; }
        g_viewport.attachTransport (params["name"].get<std::string>());
        return json { { "ok", true } };
    }

    if (method == "attach_audio")
    {
        if (! params.contains ("name")) { error = "missing name"; return {}; }
        g_viewport.attachAudio (params["name"].get<std::string>());
        return json { { "ok", true } };
    }

    if (method == "viewport_set_audio_mix")
    {
        // Frame-perfect audio parity (sweep): a baked master-mix WAV the live
        // viewport offline-analyzes so a STOPPED/scrubbed preview reads the same
        // audio-reactive features the exporter does. Empty/absent path clears it.
        // Decode+analyze happens synchronously here (off the render thread); the
        // plugin calls this from its own background bake thread.
        g_viewport.setAudioMix (params.value ("path", std::string {}));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_timeline")
    {
        if(!admitProgrammableJob(params,error)) return {};
        videowire::ResolvedVisualSnapshot snapshot;
        if (! videowire::parseSnapshotJson (params, resolveShaderSource, snapshot, error))
        {
            g_viewport.rejectSnapshot (
                params.value ("authoringRevision",
                              params.value ("structuralRevision", uint64_t { 0 })));
            return {};
        }
        if (! videowire::validateSnapshotResources (
                snapshot,
                [] (const std::string& path) { return std::filesystem::exists (path); },
                g_trustedMatteCacheRoot,
                params.value("matteContentRevision", uint64_t { 0 }),
                error))
        {
            g_viewport.rejectSnapshot (snapshot.authoringRevision);
            return {};
        }
        if (g_testRejectNextSnapshot)
        {
            g_testRejectNextSnapshot = false;
            g_viewport.rejectSnapshot (snapshot.authoringRevision);
            error = "forced renderer admission rejection";
            return {};
        }
        if (g_testDeferNextSnapshot)
        {
            g_testDeferNextSnapshot = false;
            if (! g_viewport.beginSnapshot (snapshot.authoringRevision))
            {
                error = "stale deferred authoring revision";
                return {};
            }
            g_testDeferredSnapshot = std::move (snapshot);
            return json { { "deferred", true } };
        }
        if (! g_viewport.installSnapshot (std::move (snapshot)))
        {
            error = "stale authoring revision";
            return {};
        }
        const auto revisions = g_viewport.revisionState();
        return json { { "ok", true },
                      { "authoringRevision", revisions.authoring },
                      { "acceptedRevision", revisions.accepted },
                      { "rejectedRevision", revisions.rejected },
                      { "compiledRevision", revisions.compiled },
                      { "lastGoodRevision", revisions.lastGood },
                      { "exportableRevision", revisions.exportable },
                      { "evaluationSequence", revisions.evaluation } };
    }

    if (method == "viewport_set_inspection")
    {
        error = g_viewport.setInspectionTarget(params.value("clipId", -1),
            params.value("structuralRevision", uint64_t { 0 }),
            params.value("nodeId", 0), params.value("outputPort", -1));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_describe_inspection")
        return json::parse(g_viewport.describeInspection());

    if (method == "viewport_set_inspection_presentation")
    {
        videopreview::State state;
        state.layout = static_cast<videopreview::Layout>(params.value("layout", 0));
        state.background = static_cast<videopreview::Background>(params.value("background", 0));
        state.zoom = params.value("zoom", 1.0f);
        state.panX = params.value("panX", 0.0f);
        state.panY = params.value("panY", 0.0f);
        state.split = params.value("split", 0.5f);
        error = g_viewport.setInspectionPresentation(state);
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "test_reject_next_snapshot")
    {
        g_testRejectNextSnapshot = true;
        return json { { "ok", true } };
    }

    if (method == "test_defer_next_snapshot")
    {
        if (g_testDeferredSnapshot.has_value())
        {
            error = "a deferred snapshot is already pending";
            return {};
        }
        g_testDeferNextSnapshot = true;
        return json { { "ok", true } };
    }

    if (method == "test_release_deferred_snapshot")
    {
        if (! g_testDeferredSnapshot.has_value())
        {
            error = "no deferred snapshot is pending";
            return {};
        }
        auto snapshot = std::move (*g_testDeferredSnapshot);
        g_testDeferredSnapshot.reset();
        if (! g_viewport.completeSnapshot (std::move (snapshot)))
        {
            error = "deferred snapshot is no longer current";
            return {};
        }
        return json { { "ok", true } };
    }

    if (method == "viewport_set_score")
    {
        // M5 Block C live score: same wire schema as the export jobSpec's
        // `score`. Empty/absent ⇒ clears the live score (shaders zero-feed).
        arbitmod::Score score;
        if (params.contains ("score") && params["score"].is_object())
            parseScoreJson (params["score"], score);
        else if (params.contains ("notes") || params.contains ("rootFreq"))
            parseScoreJson (params, score); // flat form
        g_viewport.setScore (std::move (score));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_beat_timeline")
    {
        g_viewport.setBeatTimeline(parseBeatTimelineJson(
            params, params.value("bpm", 120.0), params.value("beatsPerBar", 4.0)));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_mod_matrix")
    {
        // M6 live mod matrix: same wire schema as the export jobSpec's
        // `modMatrix`. Empty/absent ⇒ clears the live routings (no overlay).
        std::vector<arbitmod::Routing> routings;
        if (params.contains ("modMatrix") && params["modMatrix"].is_array())
            parseRoutingsJson (params["modMatrix"], routings);
        g_viewport.setModMatrix (std::move (routings));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_control_plan")
    {
        videocontrol::Plan plan;
        std::string controlPlanError;
        if (!parseVideoControlPlanJson(params, plan, controlPlanError))
        {
            error = controlPlanError;
            return {};
        }
        g_viewport.setControlPlan(std::move(plan));
        return json { { "ok", true } };
    }

    if (method == "viewport_visual_event")
    {
        const std::string kind = params.value("kind", std::string {});
        const int clipId = params.value("clipId", -1);
        const auto parseId = [&params](const char* key) -> uint64_t
        {
            const auto found = params.find(key);
            if (found == params.end()) return 0;
            if (found->is_string())
            {
                try { return std::stoull(found->get<std::string>()); }
                catch (...) { return 0; }
            }
            return found->is_number_unsigned() ? found->get<uint64_t>() : 0;
        };
        const uint64_t sequence = parseId("sequence");
        const uint64_t epoch = parseId("epoch");
        if (sequence == 0 || !g_viewport.enqueueVisualEvent(kind, clipId, sequence, epoch))
        {
            error = "invalid, stale, or over-cap visual event receipt";
            return {};
        }
        return json { { "ok", true } };
    }

    if (method == "viewport_set_script")
    {
        // P2 Scripts-tab live preview: the project-global per-frame hook (same
        // source/lang the export jobSpec carries as luaScript/jsScript/scriptLang).
        // Empty/absent source ⇒ clears the live hook (un-scripted preview). The
        // viewport compiles + runs frame() on the render thread (see setScript).
        const std::string source = params.value ("source", std::string {});
        const std::string lang   = params.value ("lang",   std::string {});
        programmableruntime::Grant grant;
        if(!source.empty())
        {
            if (lang == "lua")
            {
               #if ! ARBIT_HAVE_LUA
                error = "Lua runtime unavailable"; return {};
               #endif
            }
            else
            {
               #if ! ARBIT_HAVE_QUICKJS
                error = "QuickJS runtime unavailable"; return {};
               #endif
            }
            const auto it=params.find("runtimeGrant");
            const auto kind=lang=="lua"?programmableruntime::PayloadKind::lua:programmableruntime::PayloadKind::javascript;
            if(it==params.end() || !programmableadmission::admit(*it,kind,source,grant,error))
            { if(error.empty()) error="missing runtimeGrant"; return {}; }
        }
        g_viewport.setScript (source, lang, grant.cpuMs, grant.memoryMiB);
        return json { { "ok", true } };
    }

    if (method == "graph_set_param")
    {
        if (! params.contains ("paramId") || ! params.contains ("value"))
        {
            error = "missing paramId/value";
            return {};
        }
        error = g_viewport.setParam (params["paramId"].get<std::string>(),
                                     params["value"].get<double>(),
                                     params.value ("atBeat", -1.0));
        if (! error.empty()) return {};
        const auto runtimeRevision = params.value ("runtimeRevision", uint64_t { 0 });
        g_viewport.acceptRuntimeRevision (runtimeRevision);
        g_viewport.advanceEvaluation (runtimeRevision);
        return json { { "ok", true } };
    }

    if (method == "graph_set_effects")
    {
        if (! params.contains ("clipId") || ! params.contains ("effects"))
        {
            error = "missing clipId/effects";
            return {};
        }
        std::vector<EffectSlotSpec> specs;
        for (const auto& e : params["effects"])
        {
            EffectSlotSpec spec;
            spec.slot = e.value ("slot", -1);
            spec.type = e.value ("type", -1);
            spec.enabled = e.value ("enabled", true);
            if (e.contains ("params"))
                for (auto it = e["params"].begin(); it != e["params"].end(); ++it)
                    spec.params.emplace_back (it.key(), it.value().get<double>());
            specs.push_back (std::move (spec));
        }
        error = g_viewport.setEffects (params["clipId"].get<int>(), specs);
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "graph_set_lut")
    {
        // Per-clip .cube LUT (PROTOCOL.md §LUT). path "" clears the LUT.
        if (! params.contains ("clipId"))
        {
            error = "missing clipId";
            return {};
        }
        error = g_viewport.setLut (params["clipId"].get<int>(),
                                   params.value ("path", ""));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "graph_describe")
        return json::parse (g_viewport.describeGraph());

    // ---- Shader generators (M3). shader_compile is the GL-free front door:
    // it wraps the source through the dialect detector (bare GLSL / Shadertoy /
    // ISF), reporting authoring errors (bad/missing entry, #extension, ISF
    // header/INPUT problems) and the discovered generator params WITHOUT a GL
    // context — so an agent can validate a shader before it ever reaches the
    // renderer. GL-level compile errors still surface when the shader is
    // actually run (the export result, or the live deferred compile log).
    if (method == "shader_compile")
    {
        const std::string source = params.value ("source", std::string());
        programmableruntime::Grant runtimeGrant;
        const auto grantIt=params.find("runtimeGrant");
        if (! source.empty())
        {
            if (grantIt == params.end()
                || ! programmableadmission::parseGrant(*grantIt, runtimeGrant, error))
            {
                if (error.empty()) error="missing or invalid runtimeGrant";
                return {};
            }
            if (runtimeGrant.verifiedBundledCurated)
            {
                if (! programmableadmission::admitCatalogGpuField(
                        params, "source", "runtimeGrant", runtimeGrant, error))
                    return {};
            }
            else if (! programmableadmission::admit(*grantIt,
                         programmableruntime::PayloadKind::shader, source, runtimeGrant, error)
                     || ! programmableadmission::admitsGpuPayload(
                         runtimeGrant, programmableruntime::PayloadKind::shader, error))
                return {};
        }
        // P3: lower Slang / SPIR-V to GLSL at the GL-free front door, then run
        // the existing dialect wrap on the lowered GLSL (a bare-GLSL shader).
        const auto lang = arbitshadercompile::langFromWire (params.value ("lang", std::string()));
        const auto lowered = arbitshadercompile::lowerToGlsl (source, lang);
        if (! lowered.ok)
            return json { { "ok", false },
                          { "acceptedLang", arbitshadercompile::langToWire (lang) },
                          { "diagnostics", json::array ({
                                { { "level", "error" }, { "message", lowered.error } } }) },
                          { "params", json::array() },
                          { "passes", json::array() } };

        const arbitshader::WrapResult wrap = arbitshader::wrapToContract (lowered.glsl);

        json diags = json::array();
        for (const auto& d : wrap.diagnostics)
            diags.push_back ({ { "level", d.level == arbitshader::WrapDiagnostic::Error
                                              ? "error" : "warning" },
                               { "message", d.message } });

        json paramsOut = json::array();
        for (const auto& p : wrap.params)
            paramsOut.push_back ({ { "name", p.name },
                                   { "type", arbitshader::inputTypeToWire (p.type) },
                                   { "default", p.defaultScalar },
                                   { "min", p.minScalar },
                                   { "max", p.maxScalar },
                                   { "importedPath", p.importedPath } });   // M7: IMPORTED image source ("" = a user INPUT)

        // M7 multipass: expose the ISF PASS list so an agent/producer can see a
        // shader's render structure (feedback targets, output pass) without GL.
        json passesOut = json::array();
        for (const auto& ps : wrap.passes)
            passesOut.push_back ({ { "target", ps.target },           // "" = the visible output pass
                                   { "persistent", ps.persistent } }); // double-buffered feedback target

        const char* dialect = wrap.dialect == arbitshader::Dialect::BareGlsl  ? "glsl"
                            : wrap.dialect == arbitshader::Dialect::Shadertoy ? "shadertoy"
                            : wrap.dialect == arbitshader::Dialect::Isf       ? "isf"
                                                                              : "unknown";
        return json { { "ok", wrap.ok },
                      { "dialect", dialect },
                      { "acceptedLang", arbitshadercompile::langToWire (lowered.lang) },
                      { "cacheHit", lowered.cacheHit },
                      { "outputGuarded", wrap.outputGuarded },
                      { "diagnostics", diags },
                      { "params", paramsOut },
                      { "passes", passesOut } };
    }

    // ---- Per-frame script hook (M8 Lua / P2 JS). script_compile is the GL-free
    // validation front door for the Scripts tab: it compiles the source on a
    // throwaway engine and reports {ok, engine, available, error} WITHOUT running
    // an export — the analogue of shader_compile for the frame(ctx) hook. The hook
    // itself only runs during export (exporter.cpp, stateAt top layer); this just
    // surfaces syntax errors so the editor's compile log mirrors the Shader tab.
    // `available` is false when the requested engine wasn't built into this helper
    // (Lua is an optional dep; QuickJS is vendored) so the UI can say so.
    if (method == "script_compile")
    {
        const std::string source = params.value ("source", std::string());
        const std::string lang   = params.value ("lang", std::string());
        const bool wantJs = (lang != "lua");   // default JS; Scripts tab sends it explicitly
        if (! source.empty())
        {
            if (wantJs)
            {
               #if ! ARBIT_HAVE_QUICKJS
                error = "QuickJS runtime unavailable"; return {};
               #endif
            }
            else
            {
               #if ! ARBIT_HAVE_LUA
                error = "Lua runtime unavailable"; return {};
               #endif
            }
        }
        programmableruntime::Grant runtimeGrant;
        const auto grantIt=params.find("runtimeGrant");
        const auto payloadKind=wantJs?programmableruntime::PayloadKind::javascript:programmableruntime::PayloadKind::lua;
        if(!source.empty() && (grantIt==params.end()
            || !programmableadmission::admit(*grantIt,payloadKind,source,runtimeGrant,error)))
        { if(error.empty()) error="missing runtimeGrant"; return {}; }
        std::string err;
        bool ok = false;
        bool available = false;
        if (wantJs)
        {
           #if ARBIT_HAVE_QUICKJS
            available = true;
           #endif
            arbitjs::JsHook hook;
            ok = hook.compile (source, err, runtimeGrant.cpuMs, runtimeGrant.memoryMiB);
        }
        else
        {
           #if ARBIT_HAVE_LUA
            available = true;
           #endif
            arbitlua::LuaHook hook;
            ok = hook.compile (source, err, runtimeGrant.cpuMs, runtimeGrant.memoryMiB);
        }
        return json { { "ok", ok },
                      { "engine", wantJs ? "js" : "lua" },
                      { "available", available },
                      { "error", err } };
    }

    // ---- Text overlays (PROTOCOL.md §Text overlays). The helper never
    // rasterizes text: Arbit ships straight-alpha RGBA pixels through a
    // named shm region, read once here and uploaded by the render thread.
    if (method == "text_set_image")
    {
        const int textId = params.value ("textId", -1);
        const std::string shmName = params.value ("shmName", "");
        const int width = params.value ("width", 0);
        const int height = params.value ("height", 0);
        if (textId < 0 || shmName.empty() || width <= 0 || height <= 0)
        {
            error = "missing/bad textId/shmName/width/height";
            return {};
        }

        videoshm::Region region;
        if (! region.open (shmName))
        {
            error = "cannot open text shm region: " + shmName;
            return {};
        }
        std::vector<uint8_t> scratch ((size_t) region.header()->slotBytes);
        uint32_t w = 0, h = 0, stride = 0;
        double pts = 0.0;
        if (! region.readSlot (0, w, h, stride, pts, scratch.data(), scratch.size())
            || (int) w != width || (int) h != height || stride < (uint32_t) width * 4)
        {
            error = "cannot read text shm slot (size mismatch?)";
            return {};
        }
        // Tighten the rows to width*4 (Arbit writes tight rows already).
        std::vector<uint8_t> rgba ((size_t) width * (size_t) height * 4);
        for (int y = 0; y < height; ++y)
            std::memcpy (rgba.data() + (size_t) y * width * 4,
                         scratch.data() + (size_t) y * stride,
                         (size_t) width * 4);

        error = g_viewport.setTextImage (textId, std::move (rgba), width, height,
                                         params.value ("startSec", 0.0),
                                         params.value ("durationSec", 0.0),
                                         params.value ("posX", 0.0),
                                         params.value ("posY", 0.0),
                                         params.value ("opacity", 1.0),
                                         params.value ("zOrder", 0.0),
                                         params.value ("ownerClipId", -1));
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "text_remove")
    {
        const int textId = params.value ("textId", -1);
        if (textId < 0) { error = "missing textId"; return {}; }
        error = g_viewport.removeText (textId);
        if (! error.empty()) return {};
        return json { { "ok", true } };
    }

    if (method == "viewport_info" || method == "get_av_offset")
    {
        const auto vi = g_viewport.info();
        return json {
            { "open", vi.open }, { "width", vi.width }, { "height", vi.height },
            { "measuredFps", vi.measuredFps }, { "avOffsetSec", vi.avOffsetSec },
            { "interpolationActive", vi.interpolationActive },
            { "interpolationBackend", vi.interpolationBackend },
            { "interpReady", vi.interpReady },
            { "interpWorkerBackend", vi.interpWorkerBackend },
            { "interpInferences", vi.interpInferences },
            { "interpCacheHits", vi.interpCacheHits },
            { "interpCacheMisses", vi.interpCacheMisses },
            { "interpTargetFps", vi.interpTargetFps },
            { "interpError", vi.interpError },
            { "gpuPath", vi.gpuPath },
            { "backendPolicy", vi.backendPolicy },
            { "compositorBackend", vi.compositorBackend },
            { "particleBackend", vi.particleBackend },
            { "rendererError", vi.rendererError },
            { "fallbackCount", vi.fallbackCount },
            { "framesPresented", vi.framesPresented },
            { "frameHash", vi.lastFrameHash },
            { "displaySec", vi.displaySec },
            { "sourcePtsSec", vi.sourcePtsSec },
            { "sourceIdealSec", vi.sourceIdealSec },
            { "transportOpen", vi.transportOpen },
            { "transportPlaying", vi.transportPlaying },
            { "transportGeneration", vi.transportGeneration },
            { "transportAgeSec", vi.transportAgeSec },
            { "transportPlayheadBeats", vi.transportPlayheadBeats },
            { "presentPath", vi.presentPath },
            { "dmabufCapable", vi.dmabufCapable },
            { "sharedFramesSent", vi.sharedFramesSent },
            { "sharedFramesDroppedNoBuffer", vi.sharedFramesDroppedNoBuffer },
            { "sharedFreeBuffers", vi.sharedFreeBuffers },
            { "sharedBusyBuffers", vi.sharedBusyBuffers },
            { "canvasWidth", vi.canvasWidth },
            { "canvasHeight", vi.canvasHeight },
            { "viewZoom", vi.viewZoom },
            { "viewPanX", vi.viewPanX },
            { "viewPanY", vi.viewPanY },
            { "graphTelemetry", videowire::visualTelemetryJson(vi.graphTelemetry) },
            { "gpu", {
                { "glMajor", vi.gpuCaps.glMajor },
                { "glMinor", vi.gpuCaps.glMinor },
                { "computeShaders", vi.gpuCaps.computeShaders },
                { "particles", vi.gpuCaps.particles },
                { "ssbo", vi.gpuCaps.ssbo },
                { "imageLoadStore", vi.gpuCaps.imageLoadStore },
                { "maxComputeWorkGroupCount", { vi.gpuCaps.maxComputeWorkGroupCount[0],
                                                vi.gpuCaps.maxComputeWorkGroupCount[1],
                                                vi.gpuCaps.maxComputeWorkGroupCount[2] } },
                { "maxComputeWorkGroupSize", { vi.gpuCaps.maxComputeWorkGroupSize[0],
                                               vi.gpuCaps.maxComputeWorkGroupSize[1],
                                               vi.gpuCaps.maxComputeWorkGroupSize[2] } },
                { "maxComputeWorkGroupInvocations", vi.gpuCaps.maxComputeWorkGroupInvocations },
            } },
        };
    }

    // ---- Video scopes (PROTOCOL.md §Scopes) — viewport-only diagnostics

    if (method == "scope_enable")
    {
        // types: array of "waveform" | "vectorscope" | "histogram".
        // Empty/missing disables all scope computation.
        uint32_t mask = 0;
        if (params.contains ("types"))
            for (const auto& t : params["types"])
            {
                const std::string s = t.get<std::string>();
                if (s == "waveform")         mask |= Viewport::kScopeWaveform;
                else if (s == "vectorscope") mask |= Viewport::kScopeVectorscope;
                else if (s == "histogram")   mask |= Viewport::kScopeHistogram;
                else { error = "unknown scope type: " + s; return {}; }
            }
        g_viewport.setScopeMask (mask);
        return json { { "ok", true }, { "mask", mask } };
    }

    if (method == "scope_data")
        return json::parse (g_viewport.scopeDataJson());
#endif // ARBIT_HAVE_VIEWPORT

    if (method == "shutdown")
    {
        // Abort + join any in-flight export/proxy/render-cache job so
        // std::exit doesn't tear down a joinable std::thread (terminate) or
        // leave a half-written file.
        g_export.progress.abort.store (true);
        g_proxy.progress.abort.store (true);
        g_renderCache.progress.abort.store (true);
        if (g_export.worker.joinable())
            g_export.worker.join();
        if (g_proxy.worker.joinable())
            g_proxy.worker.join();
        {
            std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
            if (g_renderCache.worker.joinable())
                g_renderCache.worker.join();
        }
        closeAllCaptureSessions();
        cancelAllRecorderSessions();
#if ARBIT_HAVE_VIEWPORT
        g_viewport.close();
#endif
        reply (json(), json { { "ok", true } });
        std::exit (0);
    }

    error = "unknown method: " + method;
    return {};
}
} // namespace

int main (int argc, char** argv)
{
#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
    bool testPrivateFdClosed = false, testMarkerInArgvOrEnv = false;
    bool testPacketZero = false, testSourceZero = false;
#endif
    std::array<uint8_t, programmableruntime::privatepayload::packetSize> sessionPacket {};
#if defined(_WIN32)
    const auto privateHandleText = std::getenv("ARBIT_PRIVATE_PAYLOAD_HANDLE");
    char* privateHandleEnd = nullptr;
    const auto privateHandleValue = privateHandleText != nullptr
        ? std::strtoull(privateHandleText, &privateHandleEnd, 10) : 0;
    const bool privateHandleValid = privateHandleText != nullptr
        && privateHandleEnd != privateHandleText && *privateHandleEnd == '\0'
        && privateHandleValue != 0;
    _putenv_s("ARBIT_PRIVATE_PAYLOAD_HANDLE", "");
    const bool privatePayloadRead = privateHandleValid
        && programmableruntime::privatepayload::readOneShot(
            reinterpret_cast<HANDLE>(static_cast<uintptr_t>(privateHandleValue)), sessionPacket);
#else
    const bool privatePayloadRead = programmableruntime::privatepayload::readOneShot(3, sessionPacket);
#endif
    if (! privatePayloadRead)
    {
        std::fprintf(stderr, "missing private programmable-runtime session secret\n");
        return 2;
    }
    programmableruntime::SessionSecret sessionSecret {};
    std::copy_n(sessionPacket.begin(), sessionSecret.size(), sessionSecret.begin());
    uint64_t sessionGeneration = 0;
    for (size_t i = sessionSecret.size(); i < sessionPacket.size(); ++i)
        sessionGeneration = (sessionGeneration << 8) | sessionPacket[i];
    programmableadmission::verifier().reset(std::move(sessionSecret), sessionGeneration);
    volatile uint8_t* packetBytes = sessionPacket.data();
    for (size_t i = 0; i < sessionPacket.size(); ++i) packetBytes[i] = 0;
#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
#if defined(_WIN32)
    testPrivateFdClosed = true;
#else
    testPrivateFdClosed = fcntl(3, F_GETFD) == -1 && errno == EBADF;
#endif
    testPacketZero = std::all_of(sessionPacket.begin(), sessionPacket.end(), [](uint8_t b) { return b == 0; });
    testSourceZero = std::all_of(sessionSecret.begin(), sessionSecret.end(), [](uint8_t b) { return b == 0; });
    const std::string marker = "M6_PRIVATE_CHANNEL_MARKER";
    for (int i = 0; i < argc; ++i) testMarkerInArgvOrEnv |= std::string(argv[i]).find(marker) != std::string::npos;
#if ! defined(_WIN32)
    extern char** environ;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry)
        testMarkerInArgvOrEnv |= std::string(*entry).find(marker) != std::string::npos;
#endif
#endif
    std::filesystem::path requestedMatteRoot, requestedDepthRoot;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument (argv[i]);
        if (argument == "--matte-cache-root" && ++i < argc && requestedMatteRoot.empty())
            requestedMatteRoot = std::filesystem::path(argv[i]);
        else if (argument == "--depth-cache-root" && ++i < argc && requestedDepthRoot.empty())
            requestedDepthRoot = std::filesystem::path(argv[i]);
        else
        {
            std::fprintf(stderr, "usage: %s --matte-cache-root <dir> --depth-cache-root <dir>\n", argv[0]);
            return 2;
        }
    }
    std::error_code matteRootError;
    if (! requestedMatteRoot.is_absolute()
        || std::filesystem::is_symlink(requestedMatteRoot, matteRootError)
        || ! std::filesystem::is_directory(requestedMatteRoot, matteRootError))
    {
        std::fprintf (stderr, "invalid or missing --matte-cache-root\n");
        return 2;
    }
    g_trustedMatteCacheRoot = std::filesystem::canonical(requestedMatteRoot, matteRootError);
    if (matteRootError)
    {
        std::fprintf (stderr, "cannot canonicalize --matte-cache-root\n");
        return 2;
    }
    std::error_code depthRootError;
    if (!requestedDepthRoot.is_absolute()
        || std::filesystem::is_symlink(requestedDepthRoot, depthRootError)
        || !std::filesystem::is_directory(requestedDepthRoot, depthRootError))
    {
        std::fprintf(stderr, "invalid or missing --depth-cache-root\n");
        return 2;
    }
    g_trustedDepthCacheRoot = std::filesystem::canonical(requestedDepthRoot, depthRootError);
    if (depthRootError)
    {
        std::fprintf(stderr, "cannot canonicalize --depth-cache-root\n");
        return 2;
    }
#if ARBIT_HAVE_VIEWPORT
    g_viewport.setDepthCacheRoot(g_trustedDepthCacheRoot.string());
#endif
    av_log_set_level (AV_LOG_ERROR);

    // Initialize the shared CUDA hardware-decode context ONCE, single-threaded,
    // before any worker/render thread spawns. Doing CUDA init up front here is
    // what makes hw decode safe on the live paths (viewport render thread +
    // InterpEngine worker): the old per-context creation deadlocked the NVIDIA
    // driver when two threads called cuInit() concurrently. No-op (returns
    // nullptr) on machines without NVIDIA or when ARBIT_DISABLE_HWDEC is set.
    sharedCudaDeviceCtx();

#if defined(_WIN32)
    // Binary-safe stdio on Windows.
    _setmode (_fileno (stdin), _O_BINARY);
    _setmode (_fileno (stdout), _O_BINARY);
#endif

    const auto processRequest = [&] (const char* lineData, size_t lineSize)
    {
        if (lineSize == 0) return;

        json req = json::parse (lineData, lineData + lineSize, nullptr, false);
        if (req.is_discarded())
        {
            replyError (json(), "parse error");
            return;
        }

        const json idVal = req.value ("id", json());
        const std::string method = req.value ("method", "");
        const json params = req.contains ("params") ? req["params"] : json::object();

#if defined(ARBIT_PROGRAMMABLE_TEST_MODE)
        if (method == "test_private_startup_diagnostics")
        {
            reply(idVal, json { { "privateFdClosed", testPrivateFdClosed },
                { "markerAbsentArgvEnv", ! testMarkerInArgvOrEnv },
                { "packetTemporaryZero", testPacketZero }, { "sourceTemporaryZero", testSourceZero } });
            return;
        }
#endif

        if (method == "export")
        {
            handleExportAsync (idVal, params); // deferred reply from the worker
            return;
        }

        if (method == "proxy_generate")
        {
            handleProxyAsync (idVal, params); // deferred reply from the worker
            return;
        }

        if (method == "render_cache_build")
        {
            handleRenderCacheAsync (idVal, params); // deferred reply
            return;
        }

        std::string error;
        json result = handle (method, params, error);
        if (! error.empty())
            replyError (idVal, error);
        else
            reply (idVal, result);
    };

    constexpr size_t kMaxStdinLineBytes = 16u * 1024u * 1024u;
    auto readBoundedLine = [&] (std::vector<char>& destination)
    {
        const auto result = videohelper::readBoundedLine (
            std::cin, destination, kMaxStdinLineBytes);
        if (result == videohelper::BoundedLineResult::oversized)
        {
            replyError (nullptr, "request line exceeds 16 MiB cap");
            return false;
        }
        return result == videohelper::BoundedLineResult::line;
    };
    std::vector<char> line;
#if defined(__APPLE__) && ARBIT_HAVE_VIEWPORT
    // Keep AppKit event routing on the process main thread while stdin remains
    // responsive. Read the raw descriptor directly: std::cin is buffered and
    // its filebuf can read past a newline into its own buffer, which leaves
    // poll(STDIN_FILENO) watching an empty pipe while a complete request line
    // sits unread in the stream buffer, stalling the loop until EOF.
    {
        std::string pending;
        char chunk[4096];
        while (true)
        {
            g_viewport.processWindowEvents();
            pollfd input { STDIN_FILENO, POLLIN, 0 };
            const int ready = poll (&input, 1, 8);
            if (ready < 0 || ready == 0)
                continue;
            if ((input.revents & (POLLIN | POLLHUP)) == 0)
                break;
            const ssize_t n = ::read (STDIN_FILENO, chunk, sizeof (chunk));
            if (n <= 0)
                break;
            pending.append (chunk, static_cast<size_t> (n));
            if (pending.size() > kMaxStdinLineBytes)
            {
                replyError (nullptr, "request line exceeds 16 MiB cap");
                break;
            }
            size_t newlinePos = 0;
            while ((newlinePos = pending.find ('\n')) != std::string::npos)
            {
                const std::string oneLine = pending.substr (0, newlinePos);
                pending.erase (0, newlinePos + 1);
                if (! oneLine.empty())
                    processRequest (oneLine.data(), oneLine.size());
            }
        }
    }
#else
    while (readBoundedLine (line))
        if (! line.empty()) processRequest (line.data(), line.size());
#endif

    // stdin closed (Arbit quit/killed us politely): stop any running export,
    // proxy or render-cache job before static teardown destroys the joinable
    // workers.
    g_export.progress.abort.store (true);
    g_proxy.progress.abort.store (true);
    g_renderCache.progress.abort.store (true);
    if (g_export.worker.joinable())
        g_export.worker.join();
    if (g_proxy.worker.joinable())
        g_proxy.worker.join();
    {
        std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
        if (g_renderCache.worker.joinable())
            g_renderCache.worker.join();
    }
    closeAllCaptureSessions();
    cancelAllRecorderSessions();
#if ARBIT_HAVE_VIEWPORT
    g_viewport.close();
#endif
    return 0;
}
