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
#if ARBIT_HAVE_VIEWPORT
#include "viewport.h"
#endif
#include "shader_dialect.h"   // shader_compile RPC (GL-free dialect front door)
#include "shader_compile/shader_lang.h"   // P3: Slang/SPIR-V → GLSL front door
#include "lua_hook.h"         // P2 Scripts tab: script_compile validation (Lua hook)
#include "js_hook.h"          // P2 Scripts tab: script_compile validation (JS hook)
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

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
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
//   codec: "h264" | "h265" | "vp9" | "prores"
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
//   score: {rootFreq, lookaheadBeats?, notes:[{id, trackId, startBeat,
//     lengthBeats, midiNote, velocity, freqHz, ratioNum, ratioDen,
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
// parseExportJob) into an arbitmod::Score + lookahead. Shared by the export job
// parser and the viewport_set_score RPC so the export and live-preview paths
// can never disagree on the note/link schema.
static void parseScoreJson (const json& sc, arbitmod::Score& score, double& lookaheadBeats)
{
    score.rootFreq = sc.value ("rootFreq", 261.625565f);
    lookaheadBeats = sc.value ("lookaheadBeats", 4.0);
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
            nt.ratioNum     = n.value ("ratioNum", 1);
            nt.ratioDen     = n.value ("ratioDen", 1);
            nt.linkMasterId = n.value ("linkMasterId", -1);
            nt.isRoot       = n.value ("isRoot", false);
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

std::string parseExportJob (const json& params, ExportJob& job)
{
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
    job.luaScript = params.value ("luaScript", "");   // M8 per-frame hook (inert if no Lua)
    job.jsScript  = params.value ("jsScript", "");    // P2 JS per-frame hook (inert if no QuickJS)
    job.scriptLang = params.value ("scriptLang", ""); // "lua" | "js" | "" (infer)
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
    if (params.contains ("segments"))
        for (const auto& s : params["segments"])
        {
            ExportSegment seg;
            seg.sourcePath = s.value ("sourcePath", "");
            seg.clipId = s.value ("clipId", 0);
            seg.trackLayer = s.value ("trackLayer", 0);
            seg.inSec = s.value ("inSec", 0.0);
            seg.outSec = s.value ("outSec", 0.0);
            seg.rate = s.value ("rate", 1.0);
            seg.displayStartSec = s.value ("displayStartSec", 0.0);
            seg.sourceFps = s.value ("sourceFps", 0.0);
            seg.seqStart = s.value ("seqStart", -1);
            seg.retimeQuality = s.value ("retimeQuality", 0);   // per-clip retime tier (frame-perfect parity)
            seg.matteDir = s.value ("matteDir", std::string {}); // AI roto alpha-matte sequence
            seg.matteFps = s.value ("matteFps", 0.0);
            seg.matteFrames = s.value ("matteFrames", 0);
            if (s.contains ("transition"))
            {
                const auto& t = s["transition"];
                seg.transitionType = t.value ("type", 0);
                seg.transitionDurationSec = t.value ("durationSec", 0.0);
            }
            job.segments.push_back (seg);
        }
    std::sort (job.segments.begin(), job.segments.end(),
               [] (const ExportSegment& a, const ExportSegment& b)
               { return a.displayStartSec < b.displayStartSec; });

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
        parseScoreJson (params["score"], job.score, job.scoreLookaheadBeats);

    // Cross-domain modulation matrix (M6). Each routing maps a musical source
    // (Block A clock / Block B audio / Block C score) onto a render-graph clip
    // param (the same clip<id>/<node>/<param> namespace as paramTimeline) with
    // depth/curve/smoothingBeats/mode. The GL frame loop evaluates them per frame
    // via arbitmod::evaluateRouting. Parsed unconditionally; empty ⇒ no modulation.
    if (params.contains ("modMatrix"))
        parseRoutingsJson (params["modMatrix"], job.routings);

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

    g_export.progress.frame.store (0);
    g_export.progress.totalFrames.store (0);
    g_export.progress.encodeFps.store (0.0);
    g_export.progress.phase.store (1);
    g_export.progress.abort.store (false);
    g_export.done.store (false);
    g_export.cancelled.store (false);
    g_export.jobId.fetch_add (1);
    g_export.active.store (true);

    g_export.worker = std::thread ([idVal, job]
    {
        // §Render cache: a user export outranks a background cache build —
        // abort + join any in-flight build so two offscreen GL contexts are
        // never alive at once (the plugin re-queues the build afterwards).
        {
            std::lock_guard<std::mutex> wlock (g_renderCache.workerMutex);
            g_renderCache.progress.abort.store (true);
            if (g_renderCache.worker.joinable())
                g_renderCache.worker.join();
        }
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
    g_renderCache.worker = std::thread ([idVal, job]
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
        std::vector<ViewportSegment> segs;
        if (params.contains ("segments"))
            for (const auto& s : params["segments"])
            {
                ViewportSegment seg;
                seg.sourcePath = s.value ("sourcePath", "");
                seg.clipId = s.value ("clipId", 0);
                seg.inSec = s.value ("inSec", 0.0);
                seg.outSec = s.value ("outSec", 0.0);
                seg.rate = s.value ("rate", 1.0);
                seg.retimeQuality = s.value ("retimeQuality", 0);
                seg.displayStartSec = s.value ("displayStartSec", 0.0);
                seg.trackLayer = s.value ("trackLayer", 0);
                seg.sourceFps = s.value ("sourceFps", 0.0);
                seg.seqStart = s.value ("seqStart", -1);
                seg.shaderSource = resolveShaderSource (s);   // P3: GLSL/Slang/SPIR-V
                if (s.contains ("transition"))
                {
                    const auto& t = s["transition"];
                    seg.transitionType = t.value ("type", 0);
                    seg.transitionDurationSec = t.value ("durationSec", 0.0);
                }
                segs.push_back (seg);
            }
        g_viewport.setTimeline (std::move (segs));
        return json { { "ok", true } };
    }

    if (method == "viewport_set_score")
    {
        // M5 Block C live score: same wire schema as the export jobSpec's
        // `score`. Empty/absent ⇒ clears the live score (shaders zero-feed).
        arbitmod::Score score;
        double lookahead = 4.0;
        if (params.contains ("score") && params["score"].is_object())
            parseScoreJson (params["score"], score, lookahead);
        else if (params.contains ("notes") || params.contains ("rootFreq"))
            parseScoreJson (params, score, lookahead); // flat form
        g_viewport.setScore (std::move (score), lookahead);
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

    if (method == "viewport_set_script")
    {
        // P2 Scripts-tab live preview: the project-global per-frame hook (same
        // source/lang the export jobSpec carries as luaScript/jsScript/scriptLang).
        // Empty/absent source ⇒ clears the live hook (un-scripted preview). The
        // viewport compiles + runs frame() on the render thread (see setScript).
        const std::string source = params.value ("source", std::string {});
        const std::string lang   = params.value ("lang",   std::string {});
        g_viewport.setScript (source, lang);
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
        std::string err;
        bool ok = false;
        bool available = false;
        if (wantJs)
        {
           #if ARBIT_HAVE_QUICKJS
            available = true;
           #endif
            arbitjs::JsHook hook;
            ok = hook.compile (source, err);
        }
        else
        {
           #if ARBIT_HAVE_LUA
            available = true;
           #endif
            arbitlua::LuaHook hook;
            ok = hook.compile (source, err);
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
        reply (json(), json { { "ok", true } });
        std::exit (0);
    }

    error = "unknown method: " + method;
    return {};
}
} // namespace

int main()
{
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

    std::string line;
    while (std::getline (std::cin, line))
    {
        if (line.empty()) continue;

        json req = json::parse (line, nullptr, false);
        if (req.is_discarded())
        {
            replyError (json(), "parse error");
            continue;
        }

        const json idVal = req.value ("id", json());
        const std::string method = req.value ("method", "");
        const json params = req.contains ("params") ? req["params"] : json::object();

        if (method == "export")
        {
            handleExportAsync (idVal, params); // deferred reply from the worker
            continue;
        }

        if (method == "proxy_generate")
        {
            handleProxyAsync (idVal, params); // deferred reply from the worker
            continue;
        }

        if (method == "render_cache_build")
        {
            handleRenderCacheAsync (idVal, params); // deferred reply
            continue;
        }

        std::string error;
        json result = handle (method, params, error);
        if (! error.empty())
            replyError (idVal, error);
        else
            reply (idVal, result);
    }

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
    return 0;
}
