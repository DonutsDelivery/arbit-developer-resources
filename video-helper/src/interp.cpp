// InterpEngine implementation — see interp.h. Compiled when EITHER RIFE backend is
// available (ARBIT_HAVE_NCNN, the portable default, OR ARBIT_HAVE_ONNX). When both
// are built the portable ncnn-Vulkan engine is preferred (one binary, all GPUs).

#include "interp.h"

#include "media.h"

// Pick the RIFE backend at compile time. Both engines expose the identical
// init()/backend()/interpolate() contract, so the worker is backend-agnostic.
#if defined(ARBIT_HAVE_NCNN)
  #include "rife_ncnn.h"
  namespace arbitinterp { using RifeBackend = arbitrife::RifeEngineNcnn; }
#elif defined(ARBIT_HAVE_ONNX)
  #include "rife.h"
  namespace arbitinterp { using RifeBackend = arbitrife::RifeEngine; }
#else
  #error "interp.cpp requires ARBIT_HAVE_NCNN or ARBIT_HAVE_ONNX (a RIFE backend)"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace arbitinterp
{
namespace
{
// One request fills the bracket containing srcSec plus this many ahead, so the
// worker stays ahead of playback walking forward through source frames. Deeper
// lead (3 vs 2) absorbs transient worker stalls at higher interp resolutions —
// at 1080p a single dropped lead surfaced as a brief Frame-Blend "downgrade".
constexpr int kPrefetchBrackets = 3;
// Per-clip LRU cap on cached interpolated frames. Sized to hold the full
// prefetch window without evicting frames playback still needs (higher now that
// frames can be 1080p ≈ 8 MB rather than the original 360p ≈ 0.9 MB).
constexpr int kMaxCachePerClip = 40;
// Safety ceiling on inferences per worker wakeup (bounds a single request's
// cost). Raised with the deeper prefetch so one wakeup can refill the lead.
constexpr int kMaxInferPerRequest = 16;
// Fetch tolerance: if the exact bucket isn't cached yet (the worker is filling
// the leading edge this very cycle), accept the nearest cached bucket within
// ±this many buckets. At 120 buckets/s, ±2 = ±16 ms of source time — visually
// identical during smooth motion, and it converts the steady leading-edge miss
// (~17% → blend-fallback stutter) into a hit on the adjacent just-synthesized
// frame. 0 would restore exact-only matching.
constexpr int64_t kFetchToleranceBuckets = 2;
// Default decode + inference clamp (the measured realtime point on RTX 4070).
constexpr int kDefaultPreviewW = 640;
constexpr int kDefaultPreviewH = 360;

// Copy a (possibly strided) decoded frame into a tightly packed w*4 buffer.
void packTight (const DecodedFrame& f, std::vector<uint8_t>& out)
{
    const int w = f.width, h = f.height;
    out.resize ((size_t) w * (size_t) h * 4);
    for (int y = 0; y < h; ++y)
        std::memcpy (out.data() + (size_t) y * (size_t) w * 4,
                     f.rgba.data() + (size_t) y * (size_t) f.strideBytes,
                     (size_t) w * 4);
}
} // namespace

struct InterpEngine::Impl
{
    RifeBackend rife;
    std::atomic<bool> available { false };
    std::atomic<int> tier { 0 };          // 2 cuda, 3 coreml, 4 vulkan, 0 none
    std::string backend;
    std::string lastError;                // reason RIFE is unavailable (init failure / CPU-EP)
    std::atomic<bool> initFailed { false }; // release flag: publishes lastError to readers

    std::thread worker;
    std::atomic<bool> started { false };
    std::atomic<bool> stop { false };

    // Request mailbox: latest-wins per clip.
    std::mutex reqMx;
    std::condition_variable reqCv;
    std::map<int, Request> pending;

    // Result cache: clipId -> bucket -> packed RGBA. Shared render/worker.
    struct CachedFrame { std::vector<uint8_t> rgba; int w = 0, h = 0; uint64_t lru = 0; };
    std::mutex cacheMx;
    std::map<int, std::map<int64_t, CachedFrame>> cache;
    uint64_t lruClock = 0;

    // Clips the render thread asked to forget; reaped on the worker thread.
    std::mutex forgetMx;
    std::vector<int> forgetQueue;

    // Worker-thread-only per-clip decoder + bracket cursor.
    struct ClipDecoder
    {
        std::string path;
        double fps = 0.0;
        int seqStart = -1;
        std::unique_ptr<MediaContext> media;
        bool openFailed = false;
        DecodedFrame f0, f1;
        bool pairValid = false;
    };
    std::map<int, ClipDecoder> decoders;

    std::atomic<uint64_t> inferCount { 0 }, hits { 0 }, misses { 0 };

    // --- temporary profiling: settle inference-bound vs decode-bound (ARBIT_INTERP_PROF=1) ---
    bool profEnabled = false;
    uint64_t profInferNs = 0, profInferN = 0, profDecodeNs = 0, profDecodeN = 0;
    std::chrono::steady_clock::time_point profLast {};

    void workerMain();
    void processRequest (const Request& r);
    void reapForgotten();

    bool cacheHas (int clipId, int64_t bucket)
    {
        std::lock_guard<std::mutex> lk (cacheMx);
        auto ci = cache.find (clipId);
        return ci != cache.end() && ci->second.count (bucket) > 0;
    }

    void cacheStore (int clipId, int64_t bucket, CachedFrame&& cf)
    {
        std::lock_guard<std::mutex> lk (cacheMx);
        auto& m = cache[clipId];
        cf.lru = ++lruClock;
        m[bucket] = std::move (cf);
        while ((int) m.size() > kMaxCachePerClip)
        {
            auto victim = m.begin();
            for (auto it = m.begin(); it != m.end(); ++it)
                if (it->second.lru < victim->second.lru) victim = it;
            m.erase (victim);
        }
    }
};

InterpEngine::InterpEngine() : impl_ (std::make_unique<Impl>()) {}

InterpEngine::~InterpEngine()
{
    impl_->stop.store (true);
    impl_->reqCv.notify_all();
    if (impl_->worker.joinable())
        impl_->worker.join();
}

void InterpEngine::start()
{
    if (impl_->started.exchange (true))
        return;
    impl_->worker = std::thread ([this] { impl_->workerMain(); });
}

void InterpEngine::Impl::workerMain()
{
    if (auto err = rife.init(); ! err.empty())
    {
        std::fprintf (stderr, "[interp] RIFE init failed (tier 2 disabled): %s\n", err.c_str());
        lastError = err;
        initFailed.store (true);   // publishes lastError (release) before available
        available.store (false);
        return;
    }
    backend = rife.backend();
    const int t = backend == "rife-cuda"   ? 2
                : backend == "rife-coreml" ? 3
                : backend == "rife-vulkan" ? 4   // ncnn-Vulkan: always a real GPU device
                : 0;
    tier.store (t);
    // t==0 is the ONNX CPU EP (not realtime) — refuse so the viewport stays on Frame
    // Blend. rife-vulkan is GPU-backed by construction, so it's a realtime tier.
    available.store (t != 0);
    if (t == 0)
    {
        std::fprintf (stderr, "[interp] RIFE on CPU EP — not realtime; tier 2 disabled.\n");
        lastError = "RIFE on CPU EP (not realtime)";
        initFailed.store (true);
        return;
    }
    std::fprintf (stderr, "[interp] ready: %s (realtime tier %d)\n", backend.c_str(), t);

    if (const char* p = std::getenv ("ARBIT_INTERP_PROF"); p && *p && *p != '0')
    {
        profEnabled = true;
        profLast = std::chrono::steady_clock::now();
    }

    while (! stop.load())
    {
        Request r;
        {
            std::unique_lock<std::mutex> lk (reqMx);
            reqCv.wait (lk, [this] { return stop.load() || ! pending.empty(); });
            if (stop.load()) break;
            auto it = pending.begin();           // process oldest queued clip
            r = it->second;
            pending.erase (it);
        }
        reapForgotten();
        if (stop.load()) break;
        processRequest (r);

        if (profEnabled)
        {
            const auto now = std::chrono::steady_clock::now();
            const double winSec = std::chrono::duration<double> (now - profLast).count();
            if (winSec >= 2.0)
            {
                const double infMs = profInferN ? (profInferNs / 1e6 / profInferN) : 0.0;
                const double decMs = profDecodeN ? (profDecodeNs / 1e6 / profDecodeN) : 0.0;
                const double busyMs = (profInferNs + profDecodeNs) / 1e6;
                std::fprintf (stderr,
                    "[interp-prof] %.1fs: infer %llu@%.1fms (%.0f/s ceil) | decode %llu@%.1fms | busy %.0f%% (%.0fms/%.0fms)\n",
                    winSec, (unsigned long long) profInferN, infMs,
                    infMs > 0 ? 1000.0 / infMs : 0.0,
                    (unsigned long long) profDecodeN, decMs,
                    100.0 * busyMs / (winSec * 1000.0), busyMs, winSec * 1000.0);
                profInferNs = profInferN = profDecodeNs = profDecodeN = 0;
                profLast = now;
            }
        }
    }
}

void InterpEngine::Impl::reapForgotten()
{
    std::vector<int> ids;
    {
        std::lock_guard<std::mutex> lk (forgetMx);
        ids.swap (forgetQueue);
    }
    for (int id : ids)
    {
        decoders.erase (id);
        std::lock_guard<std::mutex> lk (cacheMx);
        cache.erase (id);
    }
}

void InterpEngine::Impl::processRequest (const Request& r)
{
    ClipDecoder& d = decoders[r.clipId];
    if (d.path != r.sourcePath || d.fps != r.sourceFps || d.seqStart != r.seqStart)
    {
        d.media.reset();
        d.path = r.sourcePath;
        d.fps = r.sourceFps;
        d.seqStart = r.seqStart;
        d.openFailed = false;
        d.pairValid = false;
    }
    if (d.media == nullptr && ! d.openFailed)
    {
        d.media = std::make_unique<MediaContext>();
        // Hardware decode (NVDEC on NVIDIA). This worker and the viewport render
        // thread now share the ONE process-wide CUDA device context created
        // single-threaded at startup (sharedCudaDeviceCtx), so opening a hw
        // decoder here no longer triggers a concurrent CUDA init next to the
        // render thread's — the race that previously deadlocked the driver and
        // forced software decode. RIFE still needs CPU pixels; getFrame() ->
        // downloadIfHw() transfers hw frames back via av_hwframe_transfer_data.
        // Falls back to software automatically if hw decode is unavailable or
        // ARBIT_DISABLE_HWDEC is set.
        if (! d.media->open (r.sourcePath, true, r.sourceFps, r.seqStart).empty())
        {
            d.media.reset();
            d.openFailed = true;
        }
    }
    if (d.media == nullptr)
        return;

    const double fps = std::max (d.media->info().fps, 1.0);
    const double frameDur = 1.0 / fps;
    const int pw = r.previewW > 0 ? r.previewW : kDefaultPreviewW;
    const int ph = r.previewH > 0 ? r.previewH : kDefaultPreviewH;

    auto decodeAt = [&] (double t, DecodedFrame& o) -> bool
    {
        DecodedFrame f;
        const auto t0 = std::chrono::steady_clock::now();
        const bool failed = (! d.media->getFrame (t, pw, ph, f).empty() || f.width <= 0);
        if (profEnabled)
        {
            profDecodeNs += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds> (
                std::chrono::steady_clock::now() - t0).count();
            ++profDecodeN;
        }
        if (failed)
            return false;
        o = std::move (f);
        return true;
    };
    // Next distinct frame after `after` (widen step for pts jitter; reject a
    // size change so the bracket pair stays equally sized for RIFE).
    auto decodeNext = [&] (const DecodedFrame& after, DecodedFrame& o) -> bool
    {
        for (const double k : { 1.05, 1.6, 2.3 })
        {
            DecodedFrame f;
            if (! decodeAt (after.ptsSec + k * frameDur, f))
                return false;
            if (f.ptsSec > after.ptsSec + 1e-6)
            {
                if (f.width != after.width || f.height != after.height)
                    return false;
                o = std::move (f);
                return true;
            }
        }
        return false;
    };

    double walk = r.srcSec;
    // (Re)build the bracket pair when invalid or srcSec jumped outside it.
    if (! d.pairValid || walk < d.f0.ptsSec - 1e-6 || walk > d.f1.ptsSec + 8.0 * frameDur)
    {
        d.pairValid = false;
        if (! decodeAt (walk, d.f0) || ! decodeNext (d.f0, d.f1))
            return;
        d.pairValid = true;
    }

    int inferBudget = kMaxInferPerRequest;
    for (int bracket = 0; bracket < kPrefetchBrackets && inferBudget > 0 && ! stop.load(); ++bracket)
    {
        int guard = 0;
        while (walk >= d.f1.ptsSec && guard++ < 16)
        {
            d.f0 = std::move (d.f1);
            if (! decodeNext (d.f0, d.f1)) { d.pairValid = false; return; }
        }
        if (walk >= d.f1.ptsSec) { d.pairValid = false; return; }

        const double f0p = d.f0.ptsSec, f1p = d.f1.ptsSec;
        const double span = std::max (1e-9, f1p - f0p);
        // b0 must match fetch's rounding (llround = floor(x+0.5)), NOT ceil:
        // otherwise the leading-edge bucket the render thread asks for at srcSec≈f0p
        // (when frac(f0p*120) <= 0.5 it rounds DOWN to floor(f0p*120)) is never
        // stored -> a cache miss every time playback enters this bracket. The
        // (at most one) bucket of overlap with the previous bracket is absorbed by
        // the cacheHas() dedup below; alpha clamps it to the f0 frame either way.
        const int64_t b0 = (int64_t) std::llround (f0p * kBucketsPerSec);
        const int64_t b1 = (int64_t) std::floor ((f1p - 1e-9) * kBucketsPerSec);
        for (int64_t b = b0; b <= b1 && inferBudget > 0 && ! stop.load(); ++b)
        {
            if (cacheHas (r.clipId, b))
                continue;
            const double t = (double) b / kBucketsPerSec;
            const double alpha = std::clamp ((t - f0p) / span, 0.0, 1.0);
            CachedFrame cf;
            cf.w = d.f0.width;
            cf.h = d.f0.height;
            if (alpha <= 0.02)        packTight (d.f0, cf.rgba);
            else if (alpha >= 0.98)   packTight (d.f1, cf.rgba);
            else
            {
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = rife.interpolate (d.f0.rgba.data(), d.f0.strideBytes,
                                        d.f1.rgba.data(), d.f1.strideBytes,
                                        d.f0.width, d.f0.height, (float) alpha,
                                        cf.rgba).empty();
                if (profEnabled)
                {
                    profInferNs += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds> (
                        std::chrono::steady_clock::now() - t0).count();
                    ++profInferN;
                }
                if (! ok)
                    continue;  // inference failed: leave bucket empty (render falls back)
                --inferBudget;
                inferCount.fetch_add (1);
            }
            cacheStore (r.clipId, b, std::move (cf));
        }
        // Move into the next bracket to keep it warm.
        walk = f1p + 0.5 * frameDur;
    }
}

void InterpEngine::request (const Request& r)
{
    {
        std::lock_guard<std::mutex> lk (impl_->reqMx);
        impl_->pending[r.clipId] = r;
    }
    impl_->reqCv.notify_one();
}

bool InterpEngine::fetch (int clipId, double srcSec, std::vector<uint8_t>& out, int& w, int& h)
{
    const int64_t b = (int64_t) std::llround (srcSec * kBucketsPerSec);
    std::lock_guard<std::mutex> lk (impl_->cacheMx);
    auto ci = impl_->cache.find (clipId);
    if (ci == impl_->cache.end()) { impl_->misses.fetch_add (1); return false; }
    auto& buckets = ci->second;
    // Exact bucket, else the nearest cached one within ±kFetchToleranceBuckets —
    // the leading-edge frame the worker is filling THIS cycle. Probe closest-
    // first, preferring the just-behind bucket (more likely already synthesized).
    auto bi = buckets.find (b);
    for (int64_t d = 1; d <= kFetchToleranceBuckets && bi == buckets.end(); ++d)
    {
        bi = buckets.find (b - d);
        if (bi == buckets.end())
            bi = buckets.find (b + d);
    }
    if (bi == buckets.end()) { impl_->misses.fetch_add (1); return false; }
    out = bi->second.rgba;
    w = bi->second.w;
    h = bi->second.h;
    bi->second.lru = ++impl_->lruClock;
    impl_->hits.fetch_add (1);
    return true;
}

void InterpEngine::forget (int clipId)
{
    std::lock_guard<std::mutex> lk (impl_->forgetMx);
    impl_->forgetQueue.push_back (clipId);
}

bool InterpEngine::available() const { return impl_->available.load(); }
int InterpEngine::tierCode() const { return impl_->tier.load(); }
std::string InterpEngine::backend() const
{
    // backend is written by the worker BEFORE available.store(); gating the read
    // on available.load() gives a happens-before so the std::string is fully
    // published (and immutable thereafter) before any cross-thread read.
    return impl_->available.load() ? impl_->backend : std::string();
}
std::string InterpEngine::lastError() const
{
    // initFailed (acquire) pairs with the worker's release store after writing
    // lastError, so the string is fully published before we read it.
    return impl_->initFailed.load() ? impl_->lastError : std::string();
}
uint64_t InterpEngine::inferenceCount() const { return impl_->inferCount.load(); }
uint64_t InterpEngine::cacheHits() const { return impl_->hits.load(); }
uint64_t InterpEngine::cacheMisses() const { return impl_->misses.load(); }

} // namespace arbitinterp
