#include "exporter.h"
#include "lua_hook.h"   // M8: per-frame Lua hook (no-op without ARBIT_HAVE_LUA)
#include "js_hook.h"    // P2: per-frame JS hook (no-op without ARBIT_HAVE_QUICKJS)
#include "media.h"

#if ARBIT_HAVE_VIEWPORT
#include "renderer.h"
#include "lut_loader.h"
#include "block_b_analyzer.h"   // A3 Block B audio-feature analyzer (header-only)
#include "mix_analyze.h"        // shared decode+offline-analyze (live viewport reuses this)
#include "block_c_packer.h"     // A2 Block C note/link voice allocator (header-only)
#include "video_param_grammar.h" // Stage 6: shared render-graph param resolver (single source of truth)
#include <GLFW/glfw3.h>
#endif

#if ARBIT_HAVE_ONNX
#include "rife.h"
namespace { using RifeEnginePtr = arbitrife::RifeEngine*; }
#else
namespace { using RifeEnginePtr = void*; } // helper built without ONNX
#endif

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>

namespace
{

// A clip whose source is a procedural shader generator (M3) rather than a media
// file uses the `gen://` sentinel sourcePath; it has no MediaContext and must be
// skipped by every media-decode/probe loop (the GL path renders it from the
// clip's compiled ShaderGenerator; the no-GL fallback cannot render it at all).
inline bool isGeneratorPath (const std::string& p) { return p.rfind ("gen://", 0) == 0; }

// Does the project carry GL-composited content the minterpolate CPU path cannot
// render? That path streams raw per-segment source frames through an FFmpeg
// motion-interpolation filter graph — it never runs the GL composite, so text,
// baked automation, modulation, shader/particle generators, transitions,
// post-FX, masks, non-identity clip transforms and multi-layer compositing are
// ALL invisible to it. With any of these present, a minterpolate export would
// silently drop them and not match the preview, so the caller refuses
// minterpolate and uses the GL retime path instead (audit #6).
inline bool minterpolateWouldDropContent (const ExportJob& job)
{
    if (! job.texts.empty()) return true;          // text overlays (GL-rasterised)
    if (! job.paramTimeline.empty()) return true;  // baked automation
    if (! job.routings.empty()) return true;       // live modulation
    if (job.post.bloomIntensity > 0.0f || job.post.tonemap != 0
        || job.post.exposure != 1.0f) return true; // post-FX

    bool haveLayer = false; int firstLayer = 0;
    for (const auto& s : job.segments)
    {
        if (isGeneratorPath (s.sourcePath)) return true;                 // shader/particle clip
        if (s.transitionType != 0 || s.transitionDurationSec > 0.0) return true; // transition
        if (! haveLayer) { haveLayer = true; firstLayer = s.trackLayer; }
        else if (s.trackLayer != firstLayer) return true;               // >1 composited layer
    }
    for (const auto& c : job.clips)
    {
        if (! c.effects.empty() || ! c.visible || c.maskType != 0 || c.blendMode != 0)
            return true;
        if (std::abs (c.scale - 1.0) > 1e-9 || c.translateX != 0.0 || c.translateY != 0.0
            || c.rotationDeg != 0.0 || c.opacity < 1.0
            || c.cropLeft != 0.0 || c.cropRight != 0.0
            || c.cropTop != 0.0 || c.cropBottom != 0.0)
            return true;                                                // non-identity transform/crop
    }
    return false;
}

#if ARBIT_HAVE_VIEWPORT
// Decode an entire WAV to packed mono float32 for Block B analysis (M4). Models
// AudioTrack::setup/transcode but downmixes to ONE channel via swresample (so we
// read a single contiguous plane), and keeps the PCM rather than re-encoding.
// Arbit's master mix is range-scoped: WAV sample 0 == the export range start
// (see exporter.h §range), so the caller maps a frame at timeline t to feature
// time (t - rangeStartSec). Returns empty samples on any failure → zero-feed.
struct MonoPcm { std::vector<float> samples; int sampleRate = 0; };

MonoPcm decodeWavToMonoFloat (const std::string& wavPath)
{
    MonoPcm out;
    if (wavPath.empty())
        return out;

    AVFormatContext* in = nullptr;
    if (avformat_open_input (&in, wavPath.c_str(), nullptr, nullptr) < 0)
        return out;
    avformat_find_stream_info (in, nullptr);
    const int streamIdx = av_find_best_stream (in, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (streamIdx < 0) { avformat_close_input (&in); return out; }

    AVStream* st = in->streams[streamIdx];
    const AVCodec* decCodec = avcodec_find_decoder (st->codecpar->codec_id);
    AVCodecContext* dec = decCodec != nullptr ? avcodec_alloc_context3 (decCodec) : nullptr;
    if (dec == nullptr) { avformat_close_input (&in); return out; }
    avcodec_parameters_to_context (dec, st->codecpar);
    if (avcodec_open2 (dec, decCodec, nullptr) < 0)
    {
        avcodec_free_context (&dec); avformat_close_input (&in); return out;
    }

    const int sr = dec->sample_rate > 0 ? dec->sample_rate : 44100;

    AVChannelLayout monoOut;
    av_channel_layout_default (&monoOut, 1);
    SwrContext* swr = nullptr;
    if (swr_alloc_set_opts2 (&swr, &monoOut, AV_SAMPLE_FMT_FLT, sr,
                             &dec->ch_layout, dec->sample_fmt, sr,   // guarded sr in==out
                             0, nullptr) < 0
        || swr_init (swr) < 0)
    {
        swr_free (&swr); av_channel_layout_uninit (&monoOut);
        avcodec_free_context (&dec); avformat_close_input (&in);
        return out;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame*  decFrame = av_frame_alloc();
    std::vector<float> buf;

    auto pull = [&] (AVFrame* f /* nullptr = flush swr */)
    {
        const int inN = f != nullptr ? f->nb_samples : 0;
        const uint8_t* const* inData = f != nullptr ? (const uint8_t* const*) f->data : nullptr;
        const int maxOut = (int) av_rescale_rnd (swr_get_delay (swr, sr) + inN,
                                                 sr, sr, AV_ROUND_UP) + 16;
        if ((int) buf.size() < maxOut) buf.resize ((size_t) maxOut);
        uint8_t* outPtr = (uint8_t*) buf.data();
        const int got = swr_convert (swr, &outPtr, maxOut, inData, inN);
        if (got > 0)
            out.samples.insert (out.samples.end(), buf.data(), buf.data() + got);
    };

    while (av_read_frame (in, pkt) >= 0)
    {
        if (pkt->stream_index == streamIdx && avcodec_send_packet (dec, pkt) >= 0)
            while (avcodec_receive_frame (dec, decFrame) >= 0)
                pull (decFrame);
        av_packet_unref (pkt);
    }
    avcodec_send_packet (dec, nullptr);                       // drain decoder
    while (avcodec_receive_frame (dec, decFrame) >= 0)
        pull (decFrame);
    pull (nullptr);                                           // flush swr tail

    out.sampleRate = sr;
    av_frame_free (&decFrame);
    av_packet_free (&pkt);
    swr_free (&swr);
    av_channel_layout_uninit (&monoOut);
    avcodec_free_context (&dec);
    avformat_close_input (&in);
    return out;
}
#endif // ARBIT_HAVE_VIEWPORT

// ---------------------------------------------------------------- encoders

const char* hwEncoderName (const std::string& codec, const std::string& backend)
{
    if (codec != "h264" && codec != "h265")
        return nullptr; // vp9/prores have no hw encode path here
    if (backend == "nvenc")
        return codec == "h265" ? "hevc_nvenc" : "h264_nvenc";
    if (backend == "videotoolbox")
        return codec == "h265" ? "hevc_videotoolbox" : "h264_videotoolbox";
    return nullptr;
}

const char* softwareEncoderName (const std::string& codec)
{
    if (codec == "h265")   return "libx265";
    if (codec == "vp9")    return "libvpx-vp9";
    if (codec == "prores") return "prores_ks";
    if (codec == "dpx")    return "dpx";        // 10-bit RGB image sequence (image2 muxer)
    return "libx264";
}

const AVCodec* pickVideoEncoder (const ExportJob& job, std::string& nameOut)
{
    // Explicit hardware request: that encoder or nothing.
    if (job.encoder == "nvenc" || job.encoder == "videotoolbox")
    {
        if (const char* n = hwEncoderName (job.codec, job.encoder))
            if (const AVCodec* c = avcodec_find_encoder_by_name (n))
            {
                nameOut = n;
                return c;
            }
        return nullptr;
    }

    if (job.encoder == "auto" && (job.codec == "h264" || job.codec == "h265"))
    {
        for (const char* backend : { "nvenc", "videotoolbox" })
            if (const char* n = hwEncoderName (job.codec, backend))
                if (const AVCodec* c = avcodec_find_encoder_by_name (n))
                {
                    nameOut = n;
                    return c;
                }
    }

    const char* n = softwareEncoderName (job.codec);
    nameOut = n;
    return avcodec_find_encoder_by_name (n);
}

// ---------------------------------------------------------------- progress

void setPhase (ExportProgress* p, int phase)
{
    if (p != nullptr)
        p->phase.store (phase, std::memory_order_relaxed);
}

bool wantsAbort (const ExportProgress* p)
{
    return p != nullptr && p->abort.load (std::memory_order_relaxed);
}

// Wall-clock encode-rate tracker for the frame loops: frames / elapsed.
struct FpsClock
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    void update (ExportProgress* p, int64_t framesDone) const
    {
        if (p == nullptr)
            return;
        p->frame.store (framesDone, std::memory_order_relaxed);
        const double sec = std::chrono::duration<double> (
                               std::chrono::steady_clock::now() - t0).count();
        if (sec > 1e-3)
            p->encodeFps.store ((double) framesDone / sec, std::memory_order_relaxed);
    }
};

// ----------------------------------------------------------- colour tagging

// We composite and encode SDR in BT.709 space, limited (TV) range. Without
// these tags the container says "unspecified" and a colour-managed player or
// colourist has to guess the primaries/transfer/matrix — and guesses wrong
// (e.g. BT.601 for SD-sized frames). We do NOT render wide-gamut or HDR today,
// so BT.709 is the honest tag regardless of output resolution or source gamut.
static void tagColorBt709 (AVCodecContext* enc)
{
    if (enc == nullptr) return;
    enc->color_primaries        = AVCOL_PRI_BT709;
    enc->color_trc              = AVCOL_TRC_BT709;
    enc->colorspace             = AVCOL_SPC_BT709;
    enc->color_range            = AVCOL_RANGE_MPEG;   // limited/TV range
    enc->chroma_sample_location = AVCHROMA_LOC_LEFT;  // H.264/HEVC convention
}

// Pin an RGBA->YUV sws context to BT.709 coefficients with a full-range RGB
// source and a limited-range YUV destination, so the pixels actually written
// match the BT.709 / limited tag above. (sws otherwise auto-selects BT.601 vs
// BT.709 by frame size, which would silently mismatch the tag for small frames.)
static void setSws709 (SwsContext* sws)
{
    if (sws == nullptr) return;
    const int* coef = sws_getCoefficients (SWS_CS_ITU709);
    sws_setColorspaceDetails (sws, coef, /*srcRange (RGB) full*/ 1,
                                    coef, /*dstRange (YUV) limited*/ 0,
                                    0, 1 << 16, 1 << 16);
}

// ------------------------------------------------------------- frame paint

// Letterbox an RGBA DecodedFrame into the centre of a yuv420p AVFrame.
std::string paintFrame (const DecodedFrame& src, AVFrame* dst, SwsContext*& sws,
                        int& swsW, int& swsH)
{
    std::memset (dst->data[0], 16, (size_t) dst->linesize[0] * (size_t) dst->height);
    std::memset (dst->data[1], 128, (size_t) dst->linesize[1] * (size_t) (dst->height / 2));
    std::memset (dst->data[2], 128, (size_t) dst->linesize[2] * (size_t) (dst->height / 2));

    if (src.width <= 0 || src.height <= 0)
        return {};

    const double scale = std::min ((double) dst->width / src.width,
                                   (double) dst->height / src.height);
    const int w = std::max (2, (int) std::lround (src.width * scale)) & ~1;
    const int h = std::max (2, (int) std::lround (src.height * scale)) & ~1;
    const int x = ((dst->width - w) / 2) & ~1;
    const int y = ((dst->height - h) / 2) & ~1;

    if (sws == nullptr || swsW != w || swsH != h)
    {
        sws_freeContext (sws);
        sws = sws_getContext (src.width, src.height, AV_PIX_FMT_RGBA,
                              w, h, AV_PIX_FMT_YUV420P,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
        swsW = w;
        swsH = h;
        setSws709 (sws);
    }
    if (sws == nullptr)
        return "sws_getContext failed";

    const uint8_t* srcData[1] = { src.rgba.data() };
    const int srcStride[1] = { src.strideBytes };
    uint8_t* dstData[3] = {
        dst->data[0] + y * dst->linesize[0] + x,
        dst->data[1] + (y / 2) * dst->linesize[1] + x / 2,
        dst->data[2] + (y / 2) * dst->linesize[2] + x / 2,
    };
    sws_scale (sws, srcData, srcStride, 0, src.height, dstData, dst->linesize);
    return {};
}

// ------------------------------------------------------------------ muxing

std::string encodeAndWrite (AVFormatContext* fmt, AVStream* stream,
                            AVCodecContext* enc, AVFrame* frame)
{
    int ret = avcodec_send_frame (enc, frame);
    if (ret < 0)
        return "avcodec_send_frame failed";

    AVPacket* pkt = av_packet_alloc();
    while ((ret = avcodec_receive_packet (enc, pkt)) >= 0)
    {
        av_packet_rescale_ts (pkt, enc->time_base, stream->time_base);
        pkt->stream_index = stream->index;
        if (av_interleaved_write_frame (fmt, pkt) < 0)
        {
            av_packet_free (&pkt);
            return "write_frame failed";
        }
    }
    av_packet_free (&pkt);
    if (ret != AVERROR (EAGAIN) && ret != AVERROR_EOF)
        return "avcodec_receive_packet failed";
    return {};
}

// Master-mix WAV → AAC. setup() must run before avformat_write_header
// (registers the stream); transcode() after it (writes packets).
struct AudioTrack
{
    AVFormatContext* in = nullptr;
    AVCodecContext* dec = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* stream = nullptr;
    SwrContext* swr = nullptr;
    int streamIdx = -1;
    // Range trim: stop after this many output samples (0 = whole WAV). A
    // short linear fade is applied at the cut so a range edge that lands
    // mid-ring doesn't click.
    int64_t maxSamples = 0;
    // Constant linear gain applied to every output sample (LUFS normalization;
    // 1.0 = unity). Downward-only is enforced by the caller (gain <= 1.0).
    float gainLinear = 1.0f;

    ~AudioTrack()
    {
        swr_free (&swr);
        if (dec) avcodec_free_context (&dec);
        if (enc) avcodec_free_context (&enc);
        if (in) avformat_close_input (&in);
    }

    std::string setup (AVFormatContext* fmt, const std::string& wavPath)
    {
        if (avformat_open_input (&in, wavPath.c_str(), nullptr, nullptr) < 0)
            return "cannot open audio wav: " + wavPath;
        avformat_find_stream_info (in, nullptr);
        streamIdx = av_find_best_stream (in, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (streamIdx < 0)
            return "no audio stream in wav";

        AVStream* inStream = in->streams[streamIdx];
        const AVCodec* decCodec = avcodec_find_decoder (inStream->codecpar->codec_id);
        dec = avcodec_alloc_context3 (decCodec);
        avcodec_parameters_to_context (dec, inStream->codecpar);
        if (avcodec_open2 (dec, decCodec, nullptr) < 0)
            return "cannot open wav decoder";

        const AVCodec* encCodec = avcodec_find_encoder (AV_CODEC_ID_AAC);
        if (encCodec == nullptr)
            return "no AAC encoder";
        enc = avcodec_alloc_context3 (encCodec);
        enc->sample_rate = dec->sample_rate > 0 ? dec->sample_rate : 44100;
        av_channel_layout_default (&enc->ch_layout, 2);
        enc->sample_fmt = AV_SAMPLE_FMT_FLTP;
        enc->bit_rate = 256000;
        enc->time_base = { 1, enc->sample_rate };
        if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        if (avcodec_open2 (enc, encCodec, nullptr) < 0)
            return "cannot open AAC encoder";

        stream = avformat_new_stream (fmt, nullptr);
        avcodec_parameters_from_context (stream->codecpar, enc);
        stream->time_base = enc->time_base;

        if (swr_alloc_set_opts2 (&swr, &enc->ch_layout, enc->sample_fmt, enc->sample_rate,
                                 &dec->ch_layout, dec->sample_fmt, dec->sample_rate,
                                 0, nullptr) < 0
            || swr_init (swr) < 0)
            return "swr init failed";
        return {};
    }

    std::string transcode (AVFormatContext* fmt, ExportProgress* progress = nullptr)
    {
        std::vector<float> fifoL, fifoR;
        int64_t nextPts = 0;
        int64_t enqueued = 0;        // samples accepted into the fifo
        bool reachedCap = false;     // maxSamples hit: stop decoding
        const int frameSize = enc->frame_size > 0 ? enc->frame_size : 1024;

        auto flushChunks = [&] (bool final) -> std::string
        {
            while ((int) fifoL.size() >= frameSize || (final && ! fifoL.empty()))
            {
                const int n = std::min ((int) fifoL.size(), frameSize);
                AVFrame* af = av_frame_alloc();
                af->nb_samples = frameSize; // zero-padded final frame
                av_channel_layout_copy (&af->ch_layout, &enc->ch_layout);
                af->format = enc->sample_fmt;
                af->sample_rate = enc->sample_rate;
                av_frame_get_buffer (af, 0);
                std::memset (af->data[0], 0, sizeof (float) * (size_t) frameSize);
                std::memset (af->data[1], 0, sizeof (float) * (size_t) frameSize);
                std::memcpy (af->data[0], fifoL.data(), sizeof (float) * (size_t) n);
                std::memcpy (af->data[1], fifoR.data(), sizeof (float) * (size_t) n);
                af->pts = nextPts;
                nextPts += frameSize;
                fifoL.erase (fifoL.begin(), fifoL.begin() + n);
                fifoR.erase (fifoR.begin(), fifoR.begin() + n);
                auto err = encodeAndWrite (fmt, stream, enc, af);
                av_frame_free (&af);
                if (! err.empty()) return err;
            }
            return {};
        };

        AVPacket* pkt = av_packet_alloc();
        AVFrame* decFrame = av_frame_alloc();
        AVFrame* cvtFrame = av_frame_alloc();
        std::string error;

        while (error.empty() && ! reachedCap && av_read_frame (in, pkt) >= 0)
        {
            if (wantsAbort (progress))
            {
                error = "cancelled";
                av_packet_unref (pkt);
                break;
            }
            if (pkt->stream_index == streamIdx && avcodec_send_packet (dec, pkt) >= 0)
            {
                while (! reachedCap && avcodec_receive_frame (dec, decFrame) >= 0)
                {
                    cvtFrame->nb_samples = (int) av_rescale_rnd (
                        swr_get_delay (swr, dec->sample_rate) + decFrame->nb_samples,
                        enc->sample_rate, dec->sample_rate, AV_ROUND_UP);
                    av_channel_layout_copy (&cvtFrame->ch_layout, &enc->ch_layout);
                    cvtFrame->format = enc->sample_fmt;
                    cvtFrame->sample_rate = enc->sample_rate;
                    av_frame_get_buffer (cvtFrame, 0);
                    const int got = swr_convert (swr, cvtFrame->data, cvtFrame->nb_samples,
                                                 (const uint8_t**) decFrame->data,
                                                 decFrame->nb_samples);
                    if (got > 0)
                    {
                        int take = got;
                        if (maxSamples > 0 && enqueued + take >= maxSamples)
                        {
                            take = (int) std::max<int64_t> (0, maxSamples - enqueued);
                            reachedCap = true;
                        }
                        // Mono WAVs land in data[0] only; duplicate to both channels.
                        const float* l = (const float*) cvtFrame->data[0];
                        const float* r = (const float*) cvtFrame->data[1];
                        fifoL.insert (fifoL.end(), l, l + take);
                        fifoR.insert (fifoR.end(), r != nullptr ? r : l,
                                      (r != nullptr ? r : l) + take);
                        if (gainLinear != 1.0f)
                            for (size_t k = fifoL.size() - (size_t) take; k < fifoL.size(); ++k)
                            {
                                fifoL[k] *= gainLinear;
                                fifoR[k] *= gainLinear;
                            }
                        enqueued += take;
                        if (reachedCap)
                        {
                            // ~10 ms fade-out over the fifo tail (whatever of
                            // it is still unflushed — at least the last
                            // partial AAC frame).
                            const size_t fadeN = std::min<size_t> (
                                fifoL.size(), (size_t) enc->sample_rate / 100);
                            for (size_t k = 0; k < fadeN; ++k)
                            {
                                const float g = (float) k / (float) fadeN; // 0 at the last sample
                                fifoL[fifoL.size() - 1 - k] *= g;
                                fifoR[fifoR.size() - 1 - k] *= g;
                            }
                        }
                        error = flushChunks (false);
                        if (! error.empty()) break;
                    }
                    av_frame_unref (cvtFrame);
                }
            }
            av_packet_unref (pkt);
        }
        if (error.empty()) error = flushChunks (true);
        if (error.empty()) error = encodeAndWrite (fmt, stream, enc, nullptr); // flush

        av_packet_free (&pkt);
        av_frame_free (&decFrame);
        av_frame_free (&cvtFrame);
        return error;
    }
};

// ----------------------------------------------------------- loudness (LUFS)

// ITU-R BS.1770-4 integrated loudness of an audio file. Decodes, resamples to
// 48 kHz stereo float, applies the K-weighting pre-filter, mean-squares 400 ms
// blocks at 100 ms hops, and applies the two-stage gate (-70 LUFS absolute,
// then -10 LU relative to the absolute-gated mean). Returns -70.0 for
// silence/failure so the caller computes no (downward) gain.
static double measureIntegratedLufs (const std::string& path)
{
    AVFormatContext* in = nullptr;
    if (avformat_open_input (&in, path.c_str(), nullptr, nullptr) < 0)
        return -70.0;
    avformat_find_stream_info (in, nullptr);
    const int sidx = av_find_best_stream (in, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (sidx < 0) { avformat_close_input (&in); return -70.0; }
    AVStream* st = in->streams[sidx];
    const AVCodec* dc = avcodec_find_decoder (st->codecpar->codec_id);
    AVCodecContext* dec = dc ? avcodec_alloc_context3 (dc) : nullptr;
    if (dec == nullptr) { avformat_close_input (&in); return -70.0; }
    avcodec_parameters_to_context (dec, st->codecpar);
    if (avcodec_open2 (dec, dc, nullptr) < 0)
    { avcodec_free_context (&dec); avformat_close_input (&in); return -70.0; }

    const int FS = 48000;   // K-weighting coefficients below are the 48 kHz set
    SwrContext* swr = nullptr;
    AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
    if (swr_alloc_set_opts2 (&swr, &stereo, AV_SAMPLE_FMT_FLT, FS,
                             &dec->ch_layout, dec->sample_fmt, dec->sample_rate,
                             0, nullptr) < 0 || swr_init (swr) < 0)
    { swr_free (&swr); avcodec_free_context (&dec); avformat_close_input (&in); return -70.0; }

    // K-weighting biquads (BS.1770-4, 48 kHz): stage 1 high-shelf, stage 2 RLB HP.
    const double b1[3] = { 1.53512485958697, -2.69169618940638, 1.19839281085285 };
    const double a1[2] = { -1.69065929318241, 0.73248077421585 };
    const double b2[3] = { 1.0, -2.0, 1.0 };
    const double a2[2] = { -1.99004745483398, 0.99007225036621 };
    struct BQ {
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        double step (double x, const double* b, const double* a)
        {
            const double y = b[0]*x + b[1]*x1 + b[2]*x2 - a[0]*y1 - a[1]*y2;
            x2 = x1; x1 = x; y2 = y1; y1 = y; return y;
        }
    } s1L, s1R, s2L, s2R;

    const int subLen = FS / 10;                          // 100 ms sub-block
    double subSqL = 0, subSqR = 0; int subN = 0;
    std::vector<std::pair<double,double>> subs;          // last <=4 sub-block sumSqs
    std::vector<std::pair<double,double>> blocks;        // (loudness, msSum) per 400 ms block

    auto pushSub = [&]
    {
        subs.emplace_back (subSqL, subSqR);
        if (subs.size() > 4) subs.erase (subs.begin());
        if (subs.size() == 4)
        {
            double sL = 0, sR = 0;
            for (auto& s : subs) { sL += s.first; sR += s.second; }
            const double blockLen = 4.0 * subLen;
            const double msSum = (sL + sR) / blockLen;   // stereo channel weights = 1
            const double loud = -0.691 + 10.0 * std::log10 (msSum > 0 ? msSum : 1e-12);
            blocks.emplace_back (loud, msSum);
        }
        subSqL = subSqR = 0; subN = 0;
    };

    AVPacket* pkt = av_packet_alloc();
    AVFrame* fr = av_frame_alloc();
    std::vector<float> buf;
    while (av_read_frame (in, pkt) >= 0)
    {
        if (pkt->stream_index == sidx && avcodec_send_packet (dec, pkt) >= 0)
            while (avcodec_receive_frame (dec, fr) >= 0)
            {
                const int maxOut = (int) av_rescale_rnd (
                    swr_get_delay (swr, dec->sample_rate) + fr->nb_samples,
                    FS, dec->sample_rate, AV_ROUND_UP);
                buf.resize ((size_t) maxOut * 2);
                uint8_t* out[1] = { (uint8_t*) buf.data() };
                const int n = swr_convert (swr, out, maxOut,
                                           (const uint8_t**) fr->extended_data, fr->nb_samples);
                for (int i = 0; i < n; ++i)
                {
                    const double kl = s2L.step (s1L.step (buf[(size_t) i*2],   b1, a1), b2, a2);
                    const double kr = s2R.step (s1R.step (buf[(size_t) i*2+1], b1, a1), b2, a2);
                    subSqL += kl * kl; subSqR += kr * kr;
                    if (++subN >= subLen) pushSub();
                }
            }
        av_packet_unref (pkt);
    }
    av_frame_free (&fr);
    av_packet_free (&pkt);
    swr_free (&swr);
    avcodec_free_context (&dec);
    avformat_close_input (&in);

    double sumAbs = 0; int nAbs = 0;
    for (auto& b : blocks) if (b.first >= -70.0) { sumAbs += b.second; ++nAbs; }
    if (nAbs == 0) return -70.0;
    const double relGate = -0.691 + 10.0 * std::log10 (sumAbs / nAbs) - 10.0;
    double sumRel = 0; int nRel = 0;
    for (auto& b : blocks) if (b.first >= -70.0 && b.first >= relGate) { sumRel += b.second; ++nRel; }
    if (nRel == 0) return -70.0;
    return -0.691 + 10.0 * std::log10 (sumRel / nRel);
}

// --------------------------------------------------------- minterpolate

// Per-segment motion interpolation graph: unique source frames go in with
// display-time pts; minterpolate emits frames on the output fps grid.
struct InterpGraph
{
    AVFilterGraph* graph = nullptr;
    AVFilterContext* src = nullptr;
    AVFilterContext* sink = nullptr;

    ~InterpGraph() { avfilter_graph_free (&graph); }

    std::string init (int w, int h, double outFps, AVRational timeBase)
    {
        graph = avfilter_graph_alloc();
        char args[256];
        std::snprintf (args, sizeof (args),
                       "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=1/1",
                       w, h, (int) AV_PIX_FMT_YUV420P, timeBase.num, timeBase.den);
        if (avfilter_graph_create_filter (&src, avfilter_get_by_name ("buffer"),
                                          "in", args, nullptr, graph) < 0)
            return "buffersrc create failed";
        if (avfilter_graph_create_filter (&sink, avfilter_get_by_name ("buffersink"),
                                          "out", nullptr, nullptr, graph) < 0)
            return "buffersink create failed";

        char desc[256];
        std::snprintf (desc, sizeof (desc),
                       "minterpolate=fps=%.6f:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                       outFps);
        AVFilterInOut* outputs = avfilter_inout_alloc();
        AVFilterInOut* inputs = avfilter_inout_alloc();
        outputs->name = av_strdup ("in");
        outputs->filter_ctx = src;
        outputs->pad_idx = 0;
        outputs->next = nullptr;
        inputs->name = av_strdup ("out");
        inputs->filter_ctx = sink;
        inputs->pad_idx = 0;
        inputs->next = nullptr;
        const int ret = avfilter_graph_parse_ptr (graph, desc, &inputs, &outputs, nullptr);
        avfilter_inout_free (&inputs);
        avfilter_inout_free (&outputs);
        if (ret < 0)
            return "minterpolate graph parse failed";
        if (avfilter_graph_config (graph, nullptr) < 0)
            return "minterpolate graph config failed";
        return {};
    }
};

#if ARBIT_HAVE_VIEWPORT

// -------------------------------------------------- GL composited path (WP6)

// Hidden offscreen GL 3.3 context for the exporter. Plain GLFW hints (EGL is
// not required — nothing is dmabuf-exported here). The exporter never calls
// glfwTerminate(): the viewport thread may own other GLFW windows.
struct GlExportContext
{
    GLFWwindow* win = nullptr;
    arbitgl::GlFuncs gl {};
    arbitgl::GpuCaps gpuCaps {}; // P1: compute caps for this offscreen context
    videorender::FrameRenderer renderer;

    bool init (int outW, int outH, std::string& errorOut)
    {
        if (glfwInit() != GLFW_TRUE)
        {
            errorOut = "glfwInit failed (no display?)";
            return false;
        }
        // GL 4.3 core (4.1 on macOS); see viewport.cpp for the rationale. The
        // #version 330 shaders compile unchanged; the bump exposes compute.
#if defined (__APPLE__)
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
#else
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
        glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE); // offscreen: never mapped
        win = glfwCreateWindow (64, 64, "Arbit Export", nullptr, nullptr);
        glfwDefaultWindowHints(); // don't leak VISIBLE=false to a later viewport_open
        if (win == nullptr)
        {
            errorOut = "glfwCreateWindow failed (GL 4.3 core unavailable?)";
            return false;
        }
        glfwMakeContextCurrent (win);
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            errorOut = "GL core loader failed, missing: " + missing;
            shutdown();
            return false;
        }
        arbitgl::loadGl43Functions (gl);     // soft: compute caps where available
        arbitgl::queryGpuCaps (gl, gpuCaps);
        if (! renderer.initialize (&gl, std::max (outW, 2), std::max (outH, 2), errorOut))
        {
            shutdown();
            return false;
        }
        return true;
    }

    void shutdown()
    {
        if (renderer.ready())
            renderer.shutdown();
        if (win != nullptr)
        {
            glfwMakeContextCurrent (nullptr);
            glfwDestroyWindow (win);
            win = nullptr;
        }
    }
};

// Per-clip render-graph state — mirrors the viewport's ClipGraphParams so the
// exporter applies static params + baked automation through the same routing.
struct ClipRenderState
{
    float scale = 1.0f;
    float translateX = 0.0f, translateY = 0.0f;
    float rotationDeg = 0.0f;
    float cropLeft = 0.0f, cropRight = 0.0f, cropTop = 0.0f, cropBottom = 0.0f;
    float opacity = 1.0f;
    float visible = 1.0f;
    float zOrder = 0.0f;
    int blendMode = 0;
    videorender::EffectSlotState effects[videofx::kMaxEffectSlots];

    // Shape mask (mirrors ClipGraphParams — automatable via clip<id>/mask/*).
    int maskType = 0;
    float maskCx = 0.5f, maskCy = 0.5f;
    float maskW = 0.8f, maskH = 0.8f;
    float maskFeather = 0.05f;
    int maskInvert = 0;

    // ISF INPUTS (M7): generator param values keyed by name (the "clip<id>/gen/
    // <name>" sink). Seeded with the shader's ISF defaults, overlaid by static
    // genParams + baked automation + mod-matrix routings; copied to the shader
    // LayerDesc for upload. Only populated for shader-generator clips.
    std::map<std::string, double> genParams;
};

void resetSlotParams (videorender::EffectSlotState& e)
{
    std::memset (e.params, 0, sizeof (e.params));
    if (const auto* def = videofx::effectDefFor (e.type))
        for (int i = 0; i < def->paramCount; ++i)
            e.params[i] = def->params[i].defaultValue;
}

// Applies one render-graph param. The transform2d/source/mask/effect grammar is the
// shared videoparam resolver (video_param_grammar.h) — the SAME definition the viewport
// uses, so the two can no longer silently diverge. The "gen" namespace is exporter-only
// glue (the viewport's clip struct has no ISF-input map).
bool applyGraphParam (ClipRenderState& p, const std::string& node,
                      const std::string& param, double value)
{
    if (videoparam::applyGraphParam (p, node, param, value))
        return true;
    if (node == "gen")
    {
        // ISF INPUT (M7): "clip<id>/gen/<name>". Stored unclamped — the shader's
        // declared min/max are authoring hints; the generator/shader is free to
        // clamp, and a mod-matrix routing controls range via depth. The uniform
        // type (scalar vs vector) is resolved at upload time in ShaderGenerator.
        p.genParams[param] = value;
        return true;
    }
    return false;
}

// Splits a "clip<id>/<node>/<param>" destination into its parts. Returns false
// for text<id>/… and malformed ids — the M6 mod matrix only targets render-graph
// clip params (the namespace applyGraphParam / getGraphParam handle). Mirrors
// parseBakedTimeline's slicing.
bool parseClipParamId (const std::string& id, int& clipId,
                       std::string& node, std::string& param)
{
    if (id.rfind ("clip", 0) != 0) return false;
    const auto s1 = id.find ('/');
    if (s1 == std::string::npos) return false;
    const auto s2 = id.find ('/', s1 + 1);
    if (s2 == std::string::npos) return false;
    try { clipId = std::stoi (id.substr (4, s1 - 4)); }
    catch (...) { return false; }
    node  = id.substr (s1 + 1, s2 - s1 - 1);
    param = id.substr (s2 + 1);
    return ! node.empty() && ! param.empty();
}

// Global HDR post-stack params ("post/<node>/<param>" — bloom + grade). Unlike
// clip params these are FRAME-GLOBAL (not per-clip), so the frame loop resolves
// them separately and pushes them via FrameRenderer::setPostFx (the thesis:
// a DAW signal modulating a global look param). Only the continuous floats are
// mod destinations; tonemap is an int mode and stays static. Mirrors
// parseClipParamId / getGraphParam / applyGraphParam for the post namespace.
bool parsePostParamId (const std::string& id, std::string& node, std::string& param)
{
    if (id.rfind ("post/", 0) != 0) return false;
    const auto s1 = id.find ('/');                 // the "post/" slash
    const auto s2 = id.find ('/', s1 + 1);
    if (s2 == std::string::npos) return false;
    node  = id.substr (s1 + 1, s2 - s1 - 1);
    param = id.substr (s2 + 1);
    return ! node.empty() && ! param.empty();
}

bool getPostParam (const ExportJob::PostFx& p, const std::string& node,
                   const std::string& param, double& out)
{
    if (node == "bloom")
    {
        if (param == "intensity") { out = p.bloomIntensity; return true; }
        if (param == "threshold") { out = p.bloomThreshold; return true; }
        if (param == "radius")    { out = p.bloomRadius;    return true; }
    }
    else if (node == "grade")
    {
        if (param == "exposure")  { out = p.exposure; return true; }
    }
    return false;
}

void applyPostParam (ExportJob::PostFx& p, const std::string& node,
                     const std::string& param, double value)
{
    if (node == "bloom")
    {
        if (param == "intensity")      p.bloomIntensity = (float) std::max (value, 0.0);
        else if (param == "threshold") p.bloomThreshold = (float) value;
        else if (param == "radius")    p.bloomRadius    = (float) std::max (value, 0.0);
    }
    else if (node == "grade")
    {
        if (param == "exposure")       p.exposure = (float) std::max (value, 0.0);
    }
}

// Reads the current value of a render-graph param out of a ClipRenderState — the
// inverse of applyGraphParam, used by the M6 mod matrix to fetch the `base` a routing
// modulates (combine(mode, base, value)). The transform2d/source/mask/effect grammar is
// the shared videoparam resolver (so read and write can never disagree); "gen" is the
// exporter-only ISF-input glue.
bool getGraphParam (const ClipRenderState& p, const std::string& node,
                    const std::string& param, double& out)
{
    if (videoparam::getGraphParam (p, node, param, out))
        return true;
    if (node == "gen")
    {
        // ISF INPUT base (M7). stateAt seeds every scalar INPUT with its ISF
        // default before routings apply, so a mod-matrix routing to a gen param
        // finds a base here; an unseeded name (vector/image INPUT) returns false
        // and the routing is skipped rather than applied to a junk base.
        const auto it = p.genParams.find (param);
        if (it == p.genParams.end()) return false;
        out = it->second;
        return true;
    }
    return false;
}

ClipRenderState buildStaticState (const ExportClipState& c)
{
    ClipRenderState s;
    s.scale = (float) c.scale;
    s.translateX = (float) c.translateX;
    s.translateY = (float) c.translateY;
    s.rotationDeg = (float) c.rotationDeg;
    s.cropLeft = (float) std::clamp (c.cropLeft, 0.0, 1.0);
    s.cropRight = (float) std::clamp (c.cropRight, 0.0, 1.0);
    s.cropTop = (float) std::clamp (c.cropTop, 0.0, 1.0);
    s.cropBottom = (float) std::clamp (c.cropBottom, 0.0, 1.0);
    s.opacity = (float) std::clamp (c.opacity, 0.0, 1.0);
    s.visible = c.visible ? 1.0f : 0.0f;
    s.zOrder = (float) c.zOrder;
    s.blendMode = c.blendMode;
    s.maskType = std::clamp (c.maskType, 0, 2);
    s.maskCx = (float) c.maskCx;
    s.maskCy = (float) c.maskCy;
    s.maskW = (float) std::max (c.maskW, 0.0);
    s.maskH = (float) std::max (c.maskH, 0.0);
    s.maskFeather = (float) std::clamp (c.maskFeather, 0.0, 1.0);
    s.maskInvert = c.maskInvert ? 1 : 0;
    s.genParams = c.genParams;   // M7: static ISF INPUT overrides (ISF defaults seeded later in stateAt)
    for (const auto& fx : c.effects)
    {
        if (fx.slot < 0 || fx.slot >= videofx::kMaxEffectSlots)
            continue;
        auto& e = s.effects[fx.slot];
        e.type = (fx.type >= 0 && fx.type < videofx::kEffectTypeCount) ? fx.type : -1;
        e.enabled = fx.enabled;
        resetSlotParams (e);
        if (e.type >= 0)
            for (const auto& [name, value] : fx.params)
                if (const int idx = videofx::effectParamIndex (e.type, name.c_str()); idx >= 0)
                {
                    const auto& pd = videofx::kEffectDefs[e.type].params[idx];
                    e.params[idx] = std::clamp ((float) value, pd.minValue, pd.maxValue);
                }
    }
    return s;
}

// One baked automation track: "clip<id>/<node>/<param>" + time-sorted samples.
struct BakedTrack
{
    int clipId = 0;
    std::string node, param;
    std::vector<std::pair<double, double>> samples; // (atSec, value), sorted
};

// Linear interpolation between baked samples, constant extrapolation outside.
double sampleBakedTrack (const std::vector<std::pair<double, double>>& s, double t)
{
    if (t <= s.front().first) return s.front().second;
    if (t >= s.back().first)  return s.back().second;
    const auto it = std::upper_bound (s.begin(), s.end(), t,
        [] (double v, const std::pair<double, double>& p) { return v < p.first; });
    const auto& b = *it;
    const auto& a = *(it - 1);
    const double span = b.first - a.first;
    if (span <= 1e-12) return a.second;
    return a.second + (b.second - a.second) * (t - a.first) / span;
}

std::map<int, std::vector<BakedTrack>> parseBakedTimeline (const ExportJob& job)
{
    // Group samples by paramId, preserving per-param time order (Arbit emits
    // them sorted; sort defensively anyway).
    std::map<std::string, BakedTrack> byId;
    for (const auto& p : job.paramTimeline)
    {
        // paramId = "clip<id>/<node>/<param>"
        if (p.paramId.rfind ("clip", 0) != 0) continue;
        const auto slash1 = p.paramId.find ('/');
        if (slash1 == std::string::npos) continue;
        const auto slash2 = p.paramId.find ('/', slash1 + 1);
        if (slash2 == std::string::npos) continue;
        auto& tr = byId[p.paramId];
        if (tr.samples.empty())
        {
            try { tr.clipId = std::stoi (p.paramId.substr (4, slash1 - 4)); }
            catch (...) { byId.erase (p.paramId); continue; }
            tr.node = p.paramId.substr (slash1 + 1, slash2 - slash1 - 1);
            tr.param = p.paramId.substr (slash2 + 1);
        }
        tr.samples.emplace_back (p.atSec, p.value);
    }
    std::map<int, std::vector<BakedTrack>> byClip;
    for (auto& [id, tr] : byId)
    {
        if (tr.samples.empty() || tr.node.empty() || tr.param.empty())
            continue;
        std::stable_sort (tr.samples.begin(), tr.samples.end(),
                          [] (const auto& a, const auto& b) { return a.first < b.first; });
        byClip[tr.clipId].push_back (std::move (tr));
    }
    return byClip;
}

// Text overlay automatable state — mirrors the viewport's Impl::applyText
// (viewport.cpp); keep the two in sync.
struct TextRenderState
{
    float opacity = 1.0f;
    float posX = 0.0f, posY = 0.0f;
    float translateX = 0.0f, translateY = 0.0f;
    float visible = 1.0f;
    float zOrder = 0.0f;
};

bool applyTextParam (TextRenderState& t, const std::string& param, double value)
{
    if (param == "opacity")    { t.opacity = (float) std::clamp (value, 0.0, 1.0); return true; }
    if (param == "posX")       { t.posX = (float) value; return true; }
    if (param == "posY")       { t.posY = (float) value; return true; }
    if (param == "translateX") { t.translateX = (float) value; return true; }
    if (param == "translateY") { t.translateY = (float) value; return true; }
    if (param == "visible")    { t.visible = value >= 0.5 ? 1.0f : 0.0f; return true; }
    if (param == "zOrder")     { t.zOrder = (float) value; return true; }
    return false;
}

// One baked text automation track: "text<id>/<param>" + time-sorted samples.
struct TextBakedTrack
{
    std::string param;
    std::vector<std::pair<double, double>> samples; // (atSec, value), sorted
};

std::map<int, std::vector<TextBakedTrack>> parseTextTimeline (const ExportJob& job)
{
    std::map<std::string, std::pair<int, TextBakedTrack>> byId;
    for (const auto& p : job.paramTimeline)
    {
        // paramId = "text<id>/<param>" (clip params handled elsewhere)
        if (p.paramId.rfind ("text", 0) != 0) continue;
        const auto slash = p.paramId.find ('/');
        if (slash == std::string::npos) continue;
        auto& [textId, tr] = byId[p.paramId];
        if (tr.samples.empty())
        {
            try { textId = std::stoi (p.paramId.substr (4, slash - 4)); }
            catch (...) { byId.erase (p.paramId); continue; }
            tr.param = p.paramId.substr (slash + 1);
        }
        tr.samples.emplace_back (p.atSec, p.value);
    }
    std::map<int, std::vector<TextBakedTrack>> byText;
    for (auto& [id, entry] : byId)
    {
        auto& [textId, tr] = entry;
        if (tr.samples.empty() || tr.param.empty())
            continue;
        std::stable_sort (tr.samples.begin(), tr.samples.end(),
                          [] (const auto& a, const auto& b) { return a.first < b.first; });
        byText[textId].push_back (std::move (tr));
    }
    return byText;
}

// Per-clip decode stream — same texture-reuse policy as the viewport (skip
// re-upload while the source frame hasn't advanced half a source frame).
struct ExportClipStream
{
    std::string path;
    std::unique_ptr<MediaContext> media;
    unsigned tex = 0;
    int texW = 0, texH = 0;
    double lastSrcSec = -1.0e9;
    bool openFailed = false;

    // Frame Blend (retime tier 1) bracket pair: blendF0Pts <= srcSec < blendF1Pts
    // while valid. Advanced forward (decode order), rebuilt on a jump. Ported
    // from the live viewport (viewport.cpp ClipStream) so a clip set to Frame
    // Blend exports the same alpha-mixed in-betweens it previews (audit #2/#5).
    unsigned texA = 0, texB = 0;
    int blendW = 0, blendH = 0;
    double blendF0Pts = -1.0e9, blendF1Pts = -1.0e9;
    bool blendPairValid = false;

#if ARBIT_HAVE_ONNX
    // RIFE bracketing pair: rifeF0.ptsSec <= srcSec < rifeF1.ptsSec while
    // valid. Advanced forward only (decode order), rebuilt on jumps.
    DecodedFrame rifeF0, rifeF1;
    bool rifePairValid = false;
    std::vector<uint8_t> rifeScratch;
#endif

    // AI roto (Epic C): the clip's alpha-matte sequence opened as an
    // image-sequence MediaContext (frame N == source frame N). Decoded at the
    // same canvas fit as the clip frame so the two align pixel-for-pixel.
    std::unique_ptr<MediaContext> matteMedia;
    bool matteOpenFailed = false;
};

// AI roto: multiply a decoded clip frame's alpha by its matte frame (decoded at
// the same fit, so dimensions align). The matte luma (R channel) scales the
// clip's existing alpha — subject stays opaque, background goes transparent.
static void applyAlphaMatte (DecodedFrame& df, MediaContext& matte, double srcSec)
{
    DecodedFrame mf;
    if (! matte.getFrame (srcSec, df.width, df.height, mf).empty()
        || mf.width != df.width || mf.height != df.height || mf.rgba.empty())
        return;   // matte unavailable / mismatched -> leave clip alpha untouched
    for (int y = 0; y < df.height; ++y)
    {
        uint8_t* drow = df.rgba.data() + (size_t) y * df.strideBytes;
        const uint8_t* mrow = mf.rgba.data() + (size_t) y * mf.strideBytes;
        for (int x = 0; x < df.width; ++x)
            drow[x * 4 + 3] = (uint8_t) ((drow[x * 4 + 3] * mrow[x * 4]) / 255);
    }
}

// Renders the whole job through the FrameRenderer. Mirrors the viewport's
// per-frame gathering (viewport.cpp renderLoop): all segments active at the
// display time, leading-edge transitions with the previous segment on the
// same trackLayer as the A source, composite order (trackLayer, zOrder).
std::string runGlFrameLoop (const ExportJob& job, GlExportContext& glctx,
                            AVFormatContext* fmt, AVStream* vStream,
                            AVCodecContext* vEnc, double rangeStartSec,
                            double durationSec, RifeEnginePtr rife,
                            ExportProgress* progress)
{
    (void) rife; // unused when built without ONNX
    const auto bakedByClip = parseBakedTimeline (job);
    const auto bakedByText = parseTextTimeline (job);

    // Text overlays: upload each rasterized block once. textId order matches
    // the viewport's std::map iteration, so equal-zOrder texts stay stable.
    struct ExportTextLayer
    {
        const ExportTextOverlay* tx = nullptr;
        unsigned tex = 0;
    };
    std::vector<ExportTextLayer> textLayers;
    for (const auto& tx : job.texts)
        if (tx.width > 0 && tx.height > 0
            && tx.rgba.size() == (size_t) tx.width * (size_t) tx.height * 4)
            textLayers.push_back ({ &tx,
                glctx.renderer.uploadRgba (tx.rgba.data(), tx.width, tx.height,
                                           tx.width * 4, 0) });
    std::stable_sort (textLayers.begin(), textLayers.end(),
                      [] (const ExportTextLayer& a, const ExportTextLayer& b)
                      { return a.tx->textId < b.tx->textId; });

    std::map<int, ClipRenderState> staticState;
    for (const auto& c : job.clips)
        staticState[c.clipId] = buildStaticState (c);

    // Per-clip 3D LUTs: parse each clip's .cube at job start and upload once.
    // A bad/missing LUT file fails the export loudly (the user attached it).
    struct ClipLut { unsigned tex = 0; int size = 0; };
    std::map<int, ClipLut> clipLuts;
    for (const auto& c : job.clips)
    {
        if (c.lutPath.empty())
            continue;
        std::vector<float> rgb;
        int size = 0;
        const std::string err = loadCubeLut (c.lutPath, rgb, size);
        if (! err.empty())
        {
            for (auto& [id, l] : clipLuts)
                glctx.renderer.deleteTexture (l.tex);
            for (auto& tl : textLayers)
                glctx.renderer.deleteTexture (tl.tex);
            return "clip " + std::to_string (c.clipId) + " LUT: " + err;
        }
        clipLuts[c.clipId] = { glctx.renderer.uploadLut3D (rgb.data(), size, 0),
                               size };
    }

    // Shader-generator clips (M3): compile each clip's source up front (the GL
    // context is current on this worker thread). A failed compile aborts the
    // export with the diagnostics — mirroring the LUT path — so a broken shader
    // is surfaced loudly rather than silently rendering nothing.
    std::set<int> shaderClips;
    // M7: each shader clip's discovered ISF INPUT defaults (name -> defaultV).
    // stateAt seeds these beneath static/baked/mod-matrix so an unset INPUT
    // renders at its declared default and a routing to it has a base to read.
    std::map<int, std::map<std::string, double>> clipGenDefaults;
    for (const auto& c : job.clips)
    {
        if (c.shaderSource.empty())
            continue;
        std::string shaderLog;
        std::vector<videorender::GenParam> shaderParams;
        if (! glctx.renderer.setClipShader (c.clipId, c.shaderSource, shaderLog, shaderParams))
        {
            glctx.renderer.clearClipShader (c.clipId);
            for (auto& [id, l] : clipLuts)
                glctx.renderer.deleteTexture (l.tex);
            for (auto& tl : textLayers)
                glctx.renderer.deleteTexture (tl.tex);
            return "clip " + std::to_string (c.clipId) + " shader: " + shaderLog;
        }
        shaderClips.insert (c.clipId);
        auto& defs = clipGenDefaults[c.clipId];
        for (const auto& gp : shaderParams)
        {
            // Seed each component key (scalar: bare name; vec: name.x/.y or
            // name.r/.g/.b/.a) so getGraphParam finds a base for a mod-matrix
            // routing and render() assembles the uniform. Sampler INPUTS → 0 comps.
            const int n = videorender::genParamComponentCount (gp.type);
            for (int k = 0; k < n; ++k)
                defs[gp.name + videorender::genParamComponentSuffix (gp.type, k)] =
                    (n == 1) ? gp.defaultV : gp.defaultVec[k];
        }

        // Decode each `image`-type sampler once and register it with the renderer
        // (M7). The discovered param carries the name + type (5 = InputType::Image);
        // its texture path comes from c.genImages[name] when the producer supplied
        // one (a user INPUT control, or an override), else from the param's own
        // gp.importedPath — the ISF IMPORTED map's header PATH, which makes imported
        // images "just work" with no producer wiring. Software decode (no hw — see
        // MediaContext::open) of frame 0 covers stills (image2 demuxer) and a clip's
        // first frame, fitted within the canvas (getFrame preserves aspect) — plenty
        // for a sampler read at normalized coords, bounded VRAM. A missing/undecodable
        // path is a logged warning, not an abort: the sampler falls back to 1x1 black
        // (zero-feed), exactly like an unfed INPUT.
        for (const auto& gp : shaderParams)
        {
            if (gp.type != 5)
                continue;
            std::string path = gp.importedPath;                 // IMPORTED header PATH
            if (const auto pit = c.genImages.find (gp.name);    // producer override / INPUT control
                pit != c.genImages.end() && ! pit->second.empty())
                path = pit->second;
            if (path.empty())
                continue;
            MediaContext img;
            if (! img.open (path, /*allowHwDecode*/ false).empty())
            {
                std::fprintf (stderr, "[image-input] clip %d image '%s': cannot open %s\n",
                              c.clipId, gp.name.c_str(), path.c_str());
                continue;
            }
            DecodedFrame df;
            const int capW = job.width  > 0 ? job.width  : 1920;
            const int capH = job.height > 0 ? job.height : 1080;
            if (! img.getFrame (0.0, capW, capH, df).empty() || df.rgba.empty())
            {
                std::fprintf (stderr, "[image-input] clip %d image '%s': decode failed for %s\n",
                              c.clipId, gp.name.c_str(), path.c_str());
                continue;
            }
            glctx.renderer.setClipImage (c.clipId, gp.name, df.rgba.data(),
                                         df.width, df.height, df.strideBytes);
        }
    }

    // M8 adjustment clips ("effect the world"): clips with no media/shader that
    // run their effects rack over the composite beneath them. Like shaderClips,
    // their segments are dispatched to a dedicated layer-build branch (no decode)
    // so decodeLayer is never asked for the gen://adjustment sentinel source.
    std::set<int> adjustmentClips;
    for (const auto& c : job.clips)
        if (c.isAdjustment && c.shaderSource.empty())
            adjustmentClips.insert (c.clipId);

    // P4 particle clips: a GPU compute particle pool, no media/shader. Identified
    // by the gen://particles segment sentinel (ExportClipState carries no
    // generator type; the segment is the discriminator). Dispatched to a
    // dedicated no-decode layer-build branch, like shader/adjustment clips.
    std::set<int> particleClips;
    for (const auto& s : job.segments)
        if (s.sourcePath.rfind ("gen://particles", 0) == 0)
            particleClips.insert (s.clipId);

    // Block B audio features (M4): if any shader clip is present and a master
    // mix WAV was provided, analyze it ONCE up front (A3 block_b_analyzer) so the
    // generators can read uRMS/uPeak/uOnset/uOnsetAge/uAudioBands. The mix WAV is
    // range-scoped (sample 0 == rangeStartSec), so a frame at timeline t reads
    // the feature whose window ends at buffer-relative (t - rangeStartSec). A
    // fresh analyzer from sample 0 self-suppresses a fabricated edge onset (A3
    // startup gate); full live⇄export convergence for a mid-timeline range
    // additionally needs the mix WAV to carry warmupSamples() of pre-roll
    // (block_b README Guarantee 2) — prepending that is Arbit's job (PROTOCOL.md).
    std::vector<arbitblockb::FeatureFrame> audioFeats;
    double audioSampleRate = 0.0;
    if (! shaderClips.empty() && ! job.audioPath.empty())
        audioFeats = videohelper::analyzeMixWavOffline (job.audioPath, audioSampleRate);
    // "Latest feature whose window-END time <= bufRelSec" — frame k has
    // sampleIndex (k+1)*kHop, so idx = floor(t*sr/kHop) - 1, clamped.
    auto audioFeatureAt = [&] (double bufRelSec) -> videorender::AudioFeatures
    {
        videorender::AudioFeatures af;
        if (audioFeats.empty() || audioSampleRate <= 0.0)
            return af;
        const double targetSample = std::max (0.0, bufRelSec) * audioSampleRate;
        long idx = (long) std::floor (targetSample / (double) arbitblockb::kHop) - 1;
        // Sub-frame-0 (t < kHop/sr): no window has ended yet. Clamp to frame 0
        // rather than zero-feed — a PARITY CONTRACT the live path (M4 Slice B)
        // must mirror, or the first ~hop of a clip diverges preview-vs-export.
        if (idx < 0) idx = 0;
        if (idx >= (long) audioFeats.size()) idx = (long) audioFeats.size() - 1;
        const arbitblockb::FeatureFrame& f = audioFeats[(size_t) idx];
        af.rms = f.rms; af.peak = f.peak; af.onset = f.onset; af.onsetAge = f.onsetAge;
        af.bands.assign (f.bands.begin(), f.bands.end());
        return af;
    };

    // Block C symbolic score (M5): when a shader clip is present and the job
    // carries a score, run the A2 BlockCPacker over it. The packer is a STATEFUL
    // voice allocator — packed exactly once per frame in monotonic order. To give
    // a mid-range export (startSec > 0) the SAME row assignment as a from-the-top
    // export, the packer is WARMED from timeline frame 0: the first frame
    // requested catches it up frame-by-frame from 0, so it has seen every note's
    // true entry/exit before the range (a fresh packer at rangeStart would assign
    // rows as if all active notes entered at once). Catch-up is CPU-only (pack, no
    // GL), so spanning the whole pre-range is cheap. Empty score ⇒ shaders see the
    // zero-feed (uNoteCount 0, samplers black). See exporter.h §score.
    const bool haveScore = (! shaderClips.empty() || ! particleClips.empty())
                         && ! job.score.notes.empty();
    arbitblockc::BlockCPacker scorePacker;   // ctor reset() → deterministic from beat 0
    auto packNotesAtBeat = [&] (double beat) -> videorender::NoteFeatures
    {
        videorender::NoteFeatures nf;
        if (! haveScore) return nf;
        const arbitblockc::PackResult pr =
            scorePacker.pack (job.score, (float) beat, (float) job.scoreLookaheadBeats);
        nf.notesTex  = pr.notesTex;
        nf.linksTex  = pr.linksTex;
        nf.noteCount = pr.noteCount;
        nf.linkCount = pr.linkCount;
        nf.rootFreq  = job.score.rootFreq;
        return nf;
    };
    // Memoized, monotonic per-frame driver. g = the timeline frame index (t·fps,
    // snapped); the first call catches the packer up from frame 0 (warming a
    // mid-range export with the full pre-range history), later calls advance one
    // frame, and asking for the same frame twice re-uses the cache (buildFrame
    // never goes backward). The cached value is copied by fillDesc, so returning
    // a reference is safe. The packed beat matches makeShaderClock (t·bpm/60).
    long packedFrame = -1;
    videorender::NoteFeatures cachedNotes;
    auto notesForFrame = [&] (double t) -> const videorender::NoteFeatures&
    {
        if (! haveScore) return cachedNotes;
        const long g = (long) std::llround (t * job.fps);
        for (long f = packedFrame + 1; f <= g; ++f)
            cachedNotes = packNotesAtBeat ((double) f / job.fps * job.bpm / 60.0);
        if (g > packedFrame) packedFrame = g;
        return cachedNotes;
    };

    // ── M6: cross-domain modulation matrix (mod_defs.h evaluateRouting) ──
    // Like the score packer, each routing's one-pole smoothing carries BEAT
    // memory, so routings are advanced exactly once per frame in monotonic order
    // and warmed from timeline frame 0 (a frame-aligned mid-range export gets the
    // same smoothed state a from-the-top export would). The advance discards the
    // combined value and keeps only RoutingState::smoothed — which, in EVERY
    // evaluateRouting branch, equals the modulation value combine() applies — so
    // stateAt reads st.smoothed and lays combine(mode, base, st.smoothed) over
    // each clip's resolved (static+baked) param. The beat is the ABSOLUTE
    // timeline beat (t·bpm/60, == the packer), so score-sourced routings see the
    // same note onsets the packer does. Audio sources only have features inside
    // the export range (the mix WAV is range-scoped); before rangeStartSec their
    // Audio is zero-fed (matches a fresh analyzer). Empty routings ⇒ no work and
    // stateAt's loop is skipped ⇒ byte-identical to a no-modMatrix job.
    std::vector<arbitmod::RoutingState> routingStates (job.routings.size());
    long advancedRoutingFrame = -1;
    auto advanceRoutingsToFrame = [&] (double t)
    {
        if (job.routings.empty()) return;
        const long g = (long) std::llround (t * job.fps);
        const double dtBeats = (1.0 / job.fps) * job.bpm / 60.0;   // constant per frame
        for (long f = advancedRoutingFrame + 1; f <= g; ++f)
        {
            const double ft   = (double) f / job.fps;              // timeline seconds
            const double beat = ft * job.bpm / 60.0;               // absolute beat (== packer)
            arbitmod::Clock clk;
            clk.beat        = (float) beat;
            clk.bpm         = (float) job.bpm;
            clk.beatsPerBar = (float) job.beatsPerBar;
            arbitmod::Audio aud;
            if (ft >= rangeStartSec)        // range-scoped features; pre-range = zero-fed
            {
                const videorender::AudioFeatures vaf = audioFeatureAt (ft - rangeStartSec);
                aud.rms = vaf.rms; aud.peak = vaf.peak; aud.onset = vaf.onset;
                const int nb = std::min (64, (int) vaf.bands.size());
                for (int b = 0; b < nb; ++b) aud.bands[(size_t) b] = vaf.bands[(size_t) b];
            }
            for (size_t ri = 0; ri < job.routings.size(); ++ri)
            {
                const arbitmod::Routing& r = job.routings[ri];
                if (! r.enabled) continue;
                // base is irrelevant to state advancement (combine reads it, the
                // smoothing does not); we read st.smoothed in stateAt.
                arbitmod::evaluateRouting (r, routingStates[ri], 0.0f,
                                           job.score, clk, aud, (float) dtBeats);
            }
        }
        if (g > advancedRoutingFrame) advancedRoutingFrame = g;
    };

    // --- M8 per-frame Lua hook: compile once; each frame build the ctx (clock +
    // range-scoped audio + coarse score) and run frame(ctx). Like the routing
    // driver above, the persistent lua_State is WARMED from timeline frame 0 so a
    // script keeping cross-frame state in globals is deterministic across a
    // mid-range export. luaOverrides holds the CURRENT frame's {paramId->value};
    // stateAt applies the ones for its clip as the top layer.
    // P2: the hook engine is Lua OR JS (never both). Explicit scriptLang wins;
    // empty infers (a non-empty jsScript ⇒ js, else lua). Both objects exist;
    // only the selected one is compiled and run.
    arbitlua::LuaHook luaHook;
    arbitjs::JsHook   jsHook;
    const bool wantJs = job.scriptLang == "js"
                        || (job.scriptLang.empty() && ! job.jsScript.empty());
    const char* hookTag = wantJs ? "js" : "lua";
    bool luaActive = false;
    if (wantJs && ! job.jsScript.empty())
    {
        std::string err;
        luaActive = jsHook.compile (job.jsScript, err);
        if (! luaActive)
            std::fprintf (stderr, "[js] hook disabled: %s\n", err.c_str());
    }
    else if (! wantJs && ! job.luaScript.empty())
    {
        std::string err;
        luaActive = luaHook.compile (job.luaScript, err);
        if (! luaActive)
            std::fprintf (stderr, "[lua] hook disabled: %s\n", err.c_str());
    }
    const double luaRootFreq  = job.score.rootFreq;
    std::map<std::string, double> luaOverrides;
    long luaFrameDone = -1;
    bool luaErrLogged = false;
    std::vector<arbitlua::HookNote> luaNotes;    // active notes for the frame being built
    std::vector<arbitlua::HookLink> luaLinks;    // the link graph (score-global, built once)
    for (const auto& l : job.score.links)
    {
        arbitlua::HookLink hl;
        hl.slaveNoteId = l.slaveNoteId; hl.masterNoteId = l.masterNoteId;
        hl.num = l.slaveHarmonic; hl.den = l.masterHarmonic;
        hl.ratio = l.masterHarmonic != 0 ? (double) l.slaveHarmonic / l.masterHarmonic : 0.0;
        luaLinks.push_back (hl);
    }
    auto advanceLuaToFrame = [&] (double t)
    {
        if (! luaActive) return;
        const long g = (long) std::llround (t * job.fps);
        for (long f = luaFrameDone + 1; f <= g; ++f)
        {
            const double ft = (double) f / job.fps;
            arbitlua::FrameCtx ctx;
            ctx.t           = ft;
            ctx.beat        = ft * job.bpm / 60.0;
            ctx.bpm         = job.bpm;
            ctx.beatsPerBar = job.beatsPerBar;
            ctx.frame       = f;
            ctx.rootFreq    = luaRootFreq;
            videorender::AudioFeatures vaf;      // zero-fed before the range start
            if (ft >= rangeStartSec)
                vaf = audioFeatureAt (ft - rangeStartSec);
            ctx.rms = vaf.rms; ctx.peak = vaf.peak;
            ctx.onset = vaf.onset; ctx.onsetAge = vaf.onsetAge;
            ctx.bands = vaf.bands.empty() ? nullptr : vaf.bands.data();
            ctx.bandCount = (int) vaf.bands.size();
            // ctx.notes — the notes SOUNDING at this beat, with JI cents/ratio.
            luaNotes.clear();
            const float beatF = (float) ctx.beat;
            for (const auto& n : job.score.notes)
                if (n.activeAt (beatF))
                {
                    arbitlua::HookNote hn;
                    hn.midi = n.midiNote; hn.freq = n.freqHz; hn.velocity = n.velocity;
                    hn.cents = arbitmod::centsFromRoot (n.freqHz, job.score.rootFreq);
                    hn.age = ctx.beat - n.startBeat;
                    hn.trackId = n.trackId; hn.ratioNum = n.ratioNum; hn.ratioDen = n.ratioDen;
                    hn.isRoot = n.isRoot;
                    for (int p = 0; p < 6; ++p) hn.primes[p] = n.primes[p];
                    luaNotes.push_back (hn);
                }
            ctx.notes = luaNotes.empty() ? nullptr : luaNotes.data();
            ctx.noteCount = (int) luaNotes.size();
            ctx.links = luaLinks.empty() ? nullptr : luaLinks.data();
            ctx.linkCount = (int) luaLinks.size();
            luaOverrides.clear();                // keep only the latest frame's overrides
            std::string err;
            const bool ok = wantJs ? jsHook.runFrame (ctx, luaOverrides, err)
                                   : luaHook.runFrame (ctx, luaOverrides, err);
            if (! ok && ! luaErrLogged)
            {
                std::fprintf (stderr, "[%s] frame() error: %s\n", hookTag, err.c_str());
                luaErrLogged = true;
            }
        }
        if (g > luaFrameDone) luaFrameDone = g;
    };

    auto stateAt = [&] (int clipId, double t) -> ClipRenderState
    {
        ClipRenderState s;
        if (const auto it = staticState.find (clipId); it != staticState.end())
            s = it->second;
        // M7: seed ISF INPUT defaults under any static override already in
        // s.genParams (emplace never overwrites), so every scalar INPUT has a
        // base for baked automation / mod-matrix to modulate.
        if (const auto dit = clipGenDefaults.find (clipId); dit != clipGenDefaults.end())
            for (const auto& [k, v] : dit->second)
                s.genParams.emplace (k, v);
        if (const auto it = bakedByClip.find (clipId); it != bakedByClip.end())
            for (const auto& tr : it->second)
                applyGraphParam (s, tr.node, tr.param, sampleBakedTrack (tr.samples, t));
        // M6: layer modulation routings over static+baked. routingStates is
        // already advanced to this frame (buildFrame calls advanceRoutingsToFrame
        // before any stateAt). Each routing targeting THIS clip reads the running
        // base (so multiple routings to one param compose in declaration order)
        // and writes combine(mode, base, smoothed) back through the viewport's
        // sink. Routings to other clips / unreadable params are skipped.
        for (size_t ri = 0; ri < job.routings.size(); ++ri)
        {
            const arbitmod::Routing& r = job.routings[ri];
            if (! r.enabled) continue;
            int cid = 0; std::string node, param;
            if (! parseClipParamId (r.destination, cid, node, param) || cid != clipId)
                continue;
            double base = 0.0;
            if (! getGraphParam (s, node, param, base)) continue;
            applyGraphParam (s, node, param,
                             arbitmod::combine (r.mode, (float) base, routingStates[ri].smoothed));
        }
        // M8: the Lua hook is the TOP layer — apply this frame's overrides for
        // this clip after static/baked/mod-matrix. luaOverrides is already
        // computed for this frame (buildFrame calls advanceLuaToFrame first).
        for (const auto& [pid, value] : luaOverrides)
        {
            int cid = 0; std::string node, param;
            if (! parseClipParamId (pid, cid, node, param) || cid != clipId) continue;
            applyGraphParam (s, node, param, value);
        }
        return s;
    };

    std::map<int, ExportClipStream> streams;

#if ARBIT_HAVE_ONNX
    // RIFE path for an under-delivering segment: keep the two source frames
    // bracketing srcSec and synthesize the in-between at
    // alpha = (srcSec - pts0) / (pts1 - pts0) before compositing. Returns
    // false on any failure so the caller falls back to nearest-frame.
    auto rifeDecodeLayer = [&] (ExportClipStream& cs, double srcSec,
                                double srcFps) -> bool
    {
        const double frameDur = 1.0 / srcFps;
        auto decodeAt = [&] (double t, DecodedFrame& out) -> bool
        {
            DecodedFrame df;
            if (! cs.media->getFrame (t, vEnc->width, vEnc->height, df).empty()
                || df.width <= 0)
                return false;
            out = std::move (df);
            return true;
        };
        // Next distinct frame after `after` (steps widen for pts jitter).
        auto decodeNext = [&] (const DecodedFrame& after, DecodedFrame& out) -> bool
        {
            for (const double k : { 1.05, 1.6, 2.3 })
            {
                DecodedFrame df;
                if (! decodeAt (after.ptsSec + k * frameDur, df))
                    return false;
                if (df.ptsSec > after.ptsSec + 1e-6)
                {
                    if (df.width != after.width || df.height != after.height)
                        return false;
                    out = std::move (df);
                    return true;
                }
            }
            return false; // stuck at end of stream
        };

        // (Re)build the pair when invalid or srcSec jumped outside it.
        if (! cs.rifePairValid
            || srcSec < cs.rifeF0.ptsSec - 1e-6
            || srcSec > cs.rifeF1.ptsSec + 8.0 * frameDur)
        {
            cs.rifePairValid = false;
            if (! decodeAt (srcSec, cs.rifeF0)
                || ! decodeNext (cs.rifeF0, cs.rifeF1))
                return false;
            cs.rifePairValid = true;
        }
        int guard = 0;
        while (srcSec >= cs.rifeF1.ptsSec && guard++ < 16)
        {
            cs.rifeF0 = std::move (cs.rifeF1);
            if (! decodeNext (cs.rifeF0, cs.rifeF1))
            {
                cs.rifePairValid = false; // clip tail: nearest-frame handles it
                return false;
            }
        }
        if (srcSec >= cs.rifeF1.ptsSec)
        {
            cs.rifePairValid = false;
            return false;
        }

        const double span = std::max (1e-9, cs.rifeF1.ptsSec - cs.rifeF0.ptsSec);
        const double alpha = std::clamp ((srcSec - cs.rifeF0.ptsSec) / span, 0.0, 1.0);

        // On (or within 2% of) a real source frame: show it untouched.
        const DecodedFrame* exact = nullptr;
        if (alpha <= 0.02)      exact = &cs.rifeF0;
        else if (alpha >= 0.98) exact = &cs.rifeF1;

        if (exact != nullptr)
            cs.tex = glctx.renderer.uploadRgba (exact->rgba.data(), exact->width,
                                                exact->height, exact->strideBytes,
                                                cs.tex);
        else
        {
            if (! rife->interpolate (cs.rifeF0.rgba.data(), cs.rifeF0.strideBytes,
                                     cs.rifeF1.rgba.data(), cs.rifeF1.strideBytes,
                                     cs.rifeF0.width, cs.rifeF0.height,
                                     (float) alpha, cs.rifeScratch).empty())
                return false; // inference failed -> nearest-frame fallback
            cs.tex = glctx.renderer.uploadRgba (cs.rifeScratch.data(),
                                                cs.rifeF0.width, cs.rifeF0.height,
                                                cs.rifeF0.width * 4, cs.tex);
        }
        cs.texW = cs.rifeF0.width;
        cs.texH = cs.rifeF0.height;
        cs.lastSrcSec = -1.0e9; // force re-upload if nearest path takes over
        return true;
    };
#endif

    auto decodeLayer = [&] (const ExportSegment& seg, double srcSec) -> ExportClipStream*
    {
        auto& cs = streams[seg.clipId];
        if (cs.path != seg.sourcePath)
        {
            cs.media.reset();
            cs.path = seg.sourcePath;
            cs.openFailed = false;
            cs.texW = 0;
            cs.lastSrcSec = -1.0e9;
            cs.matteMedia.reset();          // source swap invalidates the matte too
            cs.matteOpenFailed = false;
        }
        if (cs.media == nullptr && ! cs.openFailed)
        {
            cs.media = std::make_unique<MediaContext>();
            // Software decode: hw sessions next to the GL context deadlock
            // intermittently in the NVIDIA driver (offline render anyway).
            if (! cs.media->open (seg.sourcePath, false, seg.sourceFps, seg.seqStart).empty())
            {
                cs.media.reset();
                cs.openFailed = true;
            }
        }
        if (cs.media == nullptr)
            return nullptr;

        const double fps = std::max (cs.media->info().fps, 1.0);

        // AI roto (Epic C): lazily open the clip's matte sequence as its own
        // image-sequence MediaContext (matte_%06d.png, frame 1..N). Decoded per
        // frame at the clip's fit and multiplied into the clip alpha below.
        if (! seg.matteDir.empty() && cs.matteMedia == nullptr && ! cs.matteOpenFailed)
        {
            cs.matteMedia = std::make_unique<MediaContext>();
            const std::string mpat = seg.matteDir + "/matte_%06d.png";
            const double mfps = seg.matteFps > 0.0 ? seg.matteFps : fps;
            if (! cs.matteMedia->open (mpat, false, mfps, 1).empty())
            {
                cs.matteMedia.reset();
                cs.matteOpenFailed = true;
            }
        }

        // Retime ladder — gated on the per-clip tier so the export reproduces the
        // exact interpolation the live preview showed (audit #2/#5/#9/#10). The
        // trigger (rate < 0.999) is identical to the viewport's (viewport.cpp:
        // 1400/1446); the OLD export gate (fps*rate < job.fps) ignored the tier
        // entirely, so Frame-Blend/Speed-Warp clips exported as nearest judder.
#if ARBIT_HAVE_ONNX
        // Speed Warp (tier 2): RIFE synthesises the in-between at srcSec exactly.
        // No measuredFps load-shed gate here — the export is offline, so it always
        // renders the full tier (the viewport's load-shed is preview-only). Any
        // failure falls through to Frame Blend / nearest below.
        if (rife != nullptr && seg.retimeQuality >= 2 && seg.rate < 0.999
            && rifeDecodeLayer (cs, srcSec, fps))
            return &cs;
#endif

        // Frame Blend (tier 1): GL alpha-mix the two source frames bracketing
        // srcSec (ported from viewport.cpp:1446). Decodes at the encoder size so
        // the bracket pair is equally sized; any decode/seek failure leaves
        // blendPairValid=false and falls through to the nearest path below.
        if (seg.retimeQuality >= 1 && seg.rate < 0.999)
        {
            const double frameDur = 1.0 / fps;
            DecodedFrame df;
            // Next distinct frame after afterPts (steps widen for pts jitter);
            // rejects a size change so the bracket pair stays equally sized.
            auto decodeNextAfter = [&] (double afterPts, DecodedFrame& out,
                                        int w, int h) -> bool
            {
                for (const double k : { 1.05, 1.6, 2.3 })
                    if (cs.media->getFrame (afterPts + k * frameDur, vEnc->width,
                                            vEnc->height, out).empty()
                        && out.width > 0 && out.ptsSec > afterPts + 1e-6
                        && (w <= 0 || (out.width == w && out.height == h)))
                        return true;
                return false;
            };
            if (! cs.blendPairValid || srcSec < cs.blendF0Pts - 1e-6
                || srcSec > cs.blendF1Pts + 8.0 * frameDur)
            {
                // (Re)build the bracket pair around srcSec.
                cs.blendPairValid = false;
                if (cs.media->getFrame (srcSec, vEnc->width, vEnc->height, df).empty()
                    && df.width > 0)
                {
                    const int w = df.width, h = df.height;
                    cs.texA = glctx.renderer.uploadRgba (df.rgba.data(), w, h,
                                                         df.strideBytes, cs.texA);
                    cs.blendF0Pts = df.ptsSec;
                    DecodedFrame df1;
                    if (decodeNextAfter (df.ptsSec, df1, w, h))
                    {
                        cs.texB = glctx.renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                             df1.height, df1.strideBytes, cs.texB);
                        cs.blendF1Pts = df1.ptsSec;
                        if (cs.blendW != w || cs.blendH != h)
                        {
                            if (cs.tex != 0) glctx.renderer.deleteTexture (cs.tex);
                            cs.tex = 0;
                            cs.blendW = w;
                            cs.blendH = h;
                        }
                        cs.texW = w;
                        cs.texH = h;
                        cs.blendPairValid = true;
                    }
                }
            }
            else
            {
                // Forward-advance the pair so it keeps straddling srcSec.
                int guard = 0;
                while (cs.blendPairValid && srcSec >= cs.blendF1Pts && guard++ < 16)
                {
                    cs.blendF0Pts = cs.blendF1Pts;
                    std::swap (cs.texA, cs.texB);   // old f1 becomes the new f0
                    DecodedFrame df1;
                    if (! decodeNextAfter (cs.blendF0Pts, df1, cs.blendW, cs.blendH))
                    {
                        cs.blendPairValid = false;  // clip tail: nearest covers it
                        break;
                    }
                    cs.texB = glctx.renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                         df1.height, df1.strideBytes, cs.texB);
                    cs.blendF1Pts = df1.ptsSec;
                }
            }

            if (cs.blendPairValid && srcSec < cs.blendF1Pts && cs.blendW > 0)
            {
                const double span = std::max (1e-9, cs.blendF1Pts - cs.blendF0Pts);
                const float mix = (float) std::clamp (
                    (srcSec - cs.blendF0Pts) / span, 0.0, 1.0);
                cs.tex = glctx.renderer.frameBlendInto (cs.texA, cs.texB, cs.tex,
                                                        cs.blendW, cs.blendH, mix);
                cs.lastSrcSec = srcSec;  // blended frame == srcSec exactly
                return &cs;
            }
        }

        // Tier 0 (Nearest): hold the source frame whose pts straddles srcSec.
        if (cs.texW == 0 || std::abs (srcSec - cs.lastSrcSec) >= 0.5 / fps)
        {
            DecodedFrame df;
            if (cs.media->getFrame (srcSec, vEnc->width, vEnc->height, df).empty()
                && df.width > 0)
            {
                if (cs.matteMedia != nullptr)            // AI roto: isolate the subject
                    applyAlphaMatte (df, *cs.matteMedia, srcSec);
                cs.tex = glctx.renderer.uploadRgba (df.rgba.data(), df.width, df.height,
                                                    df.strideBytes, cs.tex);
                cs.texW = df.width;
                cs.texH = df.height;
                cs.lastSrcSec = df.ptsSec;
                cs.blendPairValid = false;   // nearest took over; rebuild bracket on return
            }
        }
        return cs.texW > 0 ? &cs : nullptr;
    };

    // cs == nullptr + shaderClock set ⇒ a shader-generator layer (no decoded
    // texture; the generator fills it in renderComposite).
    auto fillDesc = [&clipLuts] (videorender::LayerDesc& d, const ClipRenderState& p,
                                 const ExportClipStream* cs, double displaySec,
                                 int clipId,
                                 const videorender::ShaderClock* shaderClock,
                                 const videorender::AudioFeatures* audio,
                                 const videorender::NoteFeatures* notes,
                                 bool isAdjustment = false,
                                 bool particleSource = false)
    {
        if (particleSource)
        {
            // P4 particle clip: texture stays 0 — the renderer's ParticleEngine
            // fills it. The Block A clock drives the sim, the packed score seeds
            // from the spawn track, and count/spawn/force ride genParams. Same
            // ParticleEngine + clock as the viewport ⇒ export == preview.
            d.particleSource = true;
            if (shaderClock != nullptr)
                d.shaderClock = *shaderClock;
            if (notes != nullptr)
            {
                d.notesPresent = true;
                d.noteFeatures = *notes;
            }
            d.genParams = p.genParams;
        }
        else if (shaderClock != nullptr)
        {
            d.shaderSource = true;
            d.shaderClock = *shaderClock;   // texture stays 0 — generator fills it
            if (audio != nullptr)           // Block B (M4): live audio uniforms
            {
                d.audioPresent = true;
                d.audioFeatures = *audio;
            }
            if (notes != nullptr)           // Block C (M5): packed score uniforms
            {
                d.notesPresent = true;
                d.noteFeatures = *notes;
            }
            d.genParams = p.genParams;      // ISF INPUTS (M7): values for this frame
        }
        else if (isAdjustment)
        {
            // M8: no source — texture stays 0. The effects rack (copied below)
            // is applied to the composite beneath this layer in renderComposite.
            d.isAdjustment = true;
        }
        else
        {
            d.texture = cs->tex;
            d.texWidth = cs->texW;
            d.texHeight = cs->texH;
        }
        d.clipId = clipId;             // keys the feedback-trail history texture
        d.scale = p.scale;
        d.translateX = p.translateX;
        d.translateY = p.translateY;
        d.rotationDeg = p.rotationDeg;
        d.cropLeft = p.cropLeft;
        d.cropRight = p.cropRight;
        d.cropTop = p.cropTop;
        d.cropBottom = p.cropBottom;
        d.opacity = p.opacity;
        d.blendMode = p.blendMode;
        d.effects = p.effects;
        d.effectCount = videofx::kMaxEffectSlots;
        d.timeSec = displaySec;
        d.maskType = p.maskType;
        d.maskCx = p.maskCx;
        d.maskCy = p.maskCy;
        d.maskW = p.maskW;
        d.maskH = p.maskH;
        d.maskFeather = p.maskFeather;
        d.maskInvert = p.maskInvert != 0;
        if (const auto it = clipLuts.find (clipId); it != clipLuts.end())
        {
            d.lutTexture = it->second.tex;
            d.lutSize = it->second.size;
        }
    };

    struct ActiveLayer
    {
        const ExportSegment* seg = nullptr;
        ClipRenderState params;
        bool transitionActive = false;
        double transitionProgress = 0.0;
        const ExportSegment* fromSeg = nullptr;
        ClipRenderState fromParams;
    };

    // RGBA (renderer output) -> encoder pixel format, full frame
    // (yuv420p for h264/h265/vp9, yuv422p10le for prores).
    SwsContext* sws = sws_getContext (vEnc->width, vEnc->height, AV_PIX_FMT_RGBA,
                                      vEnc->width, vEnc->height, vEnc->pix_fmt,
                                      SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws == nullptr)
        return "sws_getContext (RGBA->encoder) failed";
    setSws709 (sws);

    AVFrame* frame = av_frame_alloc();
    frame->format = vEnc->pix_fmt;
    frame->width = vEnc->width;
    frame->height = vEnc->height;
    av_frame_get_buffer (frame, 0);

    glctx.renderer.setOutputSize (vEnc->width, vEnc->height);
    glctx.renderer.setBackgroundColor (job.bgColor[0], job.bgColor[1],   // M1 canvas background
                                       job.bgColor[2], job.bgColor[3]);
    // HDR post stack: buildFrame() resolves and pushes per-frame post params
    // (static job.post overlaid with any "post/*" mod routings + Lua), so it is
    // set fresh before every renderComposite — no static pre-loop call needed.

    std::string error;
    std::vector<uint8_t> rgba;
    std::vector<videorender::LayerDesc> descs;
    std::vector<videorender::LayerDesc> fromDescs;
    std::vector<videorender::ImageLayerDesc> overlays; // active text layers
    // The per-frame active-layer list MUST outlive buildFrame: each LayerDesc's
    // `effects` raw pointer (fillDesc: d.effects = p.effects) aims into an
    // ActiveLayer's ClipRenderState here, and the descs are consumed by
    // renderComposite/renderToPixelsAsync AFTER buildFrame returns. Declaring it
    // at loop scope (cleared each frame) keeps those pointers valid through the
    // render call — a buildFrame-local vector dangled (use-after-free that only
    // bit the first frame, whose first-call allocations reuse the freed block).
    std::vector<ActiveLayer> act;
    constexpr double kAbutEps = 1e-3;

    const int64_t totalFrames = (int64_t) std::llround (durationSec * job.fps);
    if (progress != nullptr)
        progress->totalFrames.store (totalFrames, std::memory_order_relaxed);
    setPhase (progress, 3);
    FpsClock fpsClock;
    int64_t encodedFrames = 0; // pts counter — readback lags render by one frame

    // Build the renderer layer list + overlays for display time t into the
    // shared descs/fromDescs/overlays buffers. Used by both the feedback
    // pre-roll and the main encode loop so they composite identical frames.
    auto buildFrame = [&] (double t)
    {
        advanceRoutingsToFrame (t);   // M6: advance mod-matrix state to this frame
        advanceLuaToFrame (t);        // M8: advance/evaluate the Lua hook this frame

        // Global HDR post stack (increment 2b): resolve this frame's bloom/grade
        // from the static job.post + any "post/<node>/<param>" routings (the
        // thesis — a DAW signal modulating a global look param) + Lua overrides,
        // then push. Mirrors stateAt's layering for the frame-global namespace.
        // With no post routings this equals the static job.post every frame, so
        // a neutral job stays byte-identical to a no-post export.
        {
            ExportJob::PostFx pf = job.post;
            for (size_t ri = 0; ri < job.routings.size(); ++ri)
            {
                const arbitmod::Routing& r = job.routings[ri];
                if (! r.enabled) continue;
                std::string node, param;
                if (! parsePostParamId (r.destination, node, param)) continue;
                double base = 0.0;
                if (! getPostParam (pf, node, param, base)) continue;
                applyPostParam (pf, node, param,
                                arbitmod::combine (r.mode, (float) base, routingStates[ri].smoothed));
            }
            for (const auto& [pid, value] : luaOverrides)   // M8 top layer
            {
                std::string node, param;
                if (! parsePostParamId (pid, node, param)) continue;
                applyPostParam (pf, node, param, value);
            }
            glctx.renderer.setPostFx (pf.bloomIntensity, pf.bloomThreshold,
                                      pf.bloomRadius, pf.tonemap, pf.exposure);
        }

        // --- Gather every segment active at t (+ transition A sources).
        act.clear();   // loop-scoped (see declaration): descs point into it.
        for (const auto& s : job.segments)
        {
            const double segDur = (s.outSec - s.inSec) / std::max (s.rate, 1e-9);
            if (! (t >= s.displayStartSec && t < s.displayStartSec + segDur))
                continue;

            ActiveLayer al;
            al.seg = &s;
            al.params = stateAt (s.clipId, t);

            if (s.transitionType != 0 && s.transitionDurationSec > 0.0
                && t < s.displayStartSec + s.transitionDurationSec)
            {
                al.transitionActive = true;
                al.transitionProgress = std::clamp (
                    (t - s.displayStartSec) / s.transitionDurationSec, 0.0, 1.0);

                // A = previous segment on the same trackLayer when it
                // abuts/overlaps; otherwise A stays empty (= black).
                const ExportSegment* best = nullptr;
                for (const auto& c : job.segments)
                    if (&c != &s && c.trackLayer == s.trackLayer
                        && c.displayStartSec < s.displayStartSec
                        && (best == nullptr
                            || c.displayStartSec > best->displayStartSec))
                        best = &c;
                if (best != nullptr)
                {
                    const double bestDur =
                        (best->outSec - best->inSec) / std::max (best->rate, 1e-9);
                    if (best->displayStartSec + bestDur >= s.displayStartSec - kAbutEps)
                    {
                        al.fromSeg = best;
                        al.fromParams = stateAt (best->clipId, t);
                    }
                }
            }
            act.push_back (std::move (al));
        }

        // Segments serving as the A side of an active cross transition render
        // inside that transition — suppress their standalone layer.
        {
            std::set<int> fromIds;
            for (const auto& al : act)
                if (al.fromSeg != nullptr)
                    fromIds.insert (al.fromSeg->clipId);
            if (! fromIds.empty())
                act.erase (std::remove_if (act.begin(), act.end(),
                    [&] (const ActiveLayer& al)
                    {
                        return ! al.transitionActive
                            && fromIds.count (al.seg->clipId) > 0;
                    }), act.end());
        }

        std::sort (act.begin(), act.end(),
                   [] (const ActiveLayer& a, const ActiveLayer& b)
                   {
                       if (a.seg->trackLayer != b.seg->trackLayer)
                           return a.seg->trackLayer < b.seg->trackLayer;
                       if (a.params.zOrder != b.params.zOrder)
                           return a.params.zOrder < b.params.zOrder;
                       return a.seg->displayStartSec < b.seg->displayStartSec;
                   });

        // --- Decode visible layers and build the renderer's layer list.
        descs.clear();
        fromDescs.clear();
        descs.reserve (act.size());
        fromDescs.reserve (act.size()); // stable: fromLayer pointers held by descs

        // Block C (M5): the packed score for THIS frame — warmed from frame 0 on
        // first use, one pack per frame, monotonic (buildFrame is called once per
        // frame in increasing t, warm-up included). Shared by every shader layer
        // this frame; independent of which clips are visible so the stateful
        // allocator advances deterministically with the frame sequence.
        const videorender::NoteFeatures* frameNotes =
            haveScore ? &notesForFrame (t) : nullptr;

        for (const auto& al : act)
        {
            if (al.params.visible < 0.5f || al.params.opacity <= 0.001f)
                continue;

            videorender::LayerDesc d;

            if (shaderClips.count (al.seg->clipId) > 0)
            {
                // Shader-generator clip: no media decode; the clock is a pure
                // function of display time + tempo (== the viewport's formula).
                const double segDur = (al.seg->outSec - al.seg->inSec)
                                    / std::max (al.seg->rate, 1e-9);
                const int frameIdx = (int) std::llround (
                    (t - al.seg->displayStartSec) * job.fps);
                const videorender::ShaderClock clock = videorender::makeShaderClock (
                    t, al.seg->displayStartSec, segDur, job.bpm, job.beatsPerBar,
                    job.fps, true, frameIdx);
                // Block B (M4): the mix WAV is range-scoped (sample 0 ==
                // rangeStartSec), so sample its features at buffer-relative time.
                const videorender::AudioFeatures af = audioFeatureAt (t - rangeStartSec);
                fillDesc (d, al.params, nullptr, t, al.seg->clipId, &clock,
                          af.bands.empty() ? nullptr : &af,
                          frameNotes);   // Block C (M5)
            }
            else if (particleClips.count (al.seg->clipId) > 0)
            {
                // P4 particle clip: no decode; the compute pool is driven by the
                // SAME clock formula as the viewport, and the packed score
                // (frameNotes) seeds from the spawn track ⇒ export == preview.
                const double segDur = (al.seg->outSec - al.seg->inSec)
                                    / std::max (al.seg->rate, 1e-9);
                const int frameIdx = (int) std::llround (
                    (t - al.seg->displayStartSec) * job.fps);
                const videorender::ShaderClock clock = videorender::makeShaderClock (
                    t, al.seg->displayStartSec, segDur, job.bpm, job.beatsPerBar,
                    job.fps, true, frameIdx);
                fillDesc (d, al.params, nullptr, t, al.seg->clipId, &clock,
                          nullptr, frameNotes, /*isAdjustment=*/false,
                          /*particleSource=*/true);
            }
            else if (adjustmentClips.count (al.seg->clipId) > 0)
            {
                // M8 adjustment clip: no decode, no shader — its effects rack
                // runs over the composite beneath it in renderComposite.
                fillDesc (d, al.params, nullptr, t, al.seg->clipId,
                          nullptr, nullptr, nullptr, /*isAdjustment=*/true);
            }
            else
            {
                const double srcSec = al.seg->inSec
                    + (t - al.seg->displayStartSec) * al.seg->rate;
                ExportClipStream* cs = decodeLayer (*al.seg, srcSec);
                if (cs == nullptr)
                    continue;
                fillDesc (d, al.params, cs, t, al.seg->clipId, nullptr, nullptr, nullptr);
            }

            if (al.transitionActive)
            {
                d.transitionType = al.seg->transitionType;
                d.transitionProgress = (float) al.transitionProgress;
                if (al.fromSeg != nullptr && al.fromParams.visible >= 0.5f
                    && al.fromParams.opacity > 0.001f)
                {
                    double fromSrc = al.fromSeg->inSec
                        + (t - al.fromSeg->displayStartSec) * al.fromSeg->rate;
                    fromSrc = std::clamp (fromSrc, al.fromSeg->inSec,
                                          std::max (al.fromSeg->inSec, al.fromSeg->outSec));
                    if (ExportClipStream* fs = decodeLayer (*al.fromSeg, fromSrc))
                    {
                        videorender::LayerDesc fd;
                        fillDesc (fd, al.fromParams, fs, t, al.fromSeg->clipId, nullptr, nullptr, nullptr);
                        fromDescs.push_back (fd);
                        d.fromLayer = &fromDescs.back();
                    }
                }
            }
            descs.push_back (d);
        }

        // --- Text overlays active at t: static jobSpec values overlaid with
        // baked text<id>/<param> automation, same gathering/ordering as the
        // viewport's render loop (viewport.cpp).
        overlays.clear();
        for (const auto& tl : textLayers)
        {
            const auto& tx = *tl.tx;
            TextRenderState ts;
            ts.opacity = (float) std::clamp (tx.opacity, 0.0, 1.0);
            ts.posX = (float) tx.posX;
            ts.posY = (float) tx.posY;
            ts.zOrder = (float) tx.zOrder;
            if (const auto it = bakedByText.find (tx.textId); it != bakedByText.end())
                for (const auto& tr : it->second)
                    applyTextParam (ts, tr.param, sampleBakedTrack (tr.samples, t));
            if (ts.visible < 0.5f || ts.opacity <= 0.001f
                || t < tx.startSec || t >= tx.startSec + tx.durationSec)
                continue;
            videorender::ImageLayerDesc ov;
            ov.texture = tl.tex;
            ov.width = tx.width;
            ov.height = tx.height;
            ov.posX = ts.posX + ts.translateX;
            ov.posY = ts.posY + ts.translateY;
            ov.opacity = ts.opacity;
            ov.zOrder = (int) ts.zOrder;
            ov.ownerClipId = tx.ownerClipId;
            overlays.push_back (ov);
        }
        std::stable_sort (overlays.begin(), overlays.end(),
                          [] (const videorender::ImageLayerDesc& a,
                              const videorender::ImageLayerDesc& b)
                          { return a.zOrder < b.zOrder; });
    };  // buildFrame

    // Feedback pre-roll (generative-visuals-research.md risk 13): for a
    // mid-timeline range, cross-frame feedback trails depend on frames before
    // the range that are never encoded. Render feedbackPreRollSec of warm-up
    // frames before rangeStartSec — composite-only, no readback/encode — so
    // frameParity_ and the per-clip history accumulate straight into the main
    // loop and the exported range matches a full-clip export. Skipped when no
    // clip uses feedback (the only history-dependent effect) or the range
    // starts at the timeline head (history already seeds cleared = correct).
    bool hasFeedback = false;
    for (const auto& c : job.clips)
        for (const auto& fx : c.effects)
            if (fx.type == (int) videofx::EffectType::FeedbackTrail)
                hasFeedback = true;
    if (hasFeedback && job.feedbackPreRollSec > 1e-6 && rangeStartSec > 1e-6)
    {
        int64_t warm = (int64_t) std::llround (job.feedbackPreRollSec * job.fps);
        // Never precede the timeline head; align warm frames to the same
        // n/fps grid the main loop uses so parity holds frame-for-frame.
        warm = std::min (warm, (int64_t) std::floor (rangeStartSec * job.fps));
        const double warmStart = rangeStartSec - (double) warm / job.fps;
        for (int64_t w = 0; w < warm && error.empty(); ++w)
        {
            if (wantsAbort (progress)) { error = "cancelled"; break; }
            buildFrame (warmStart + (double) w / job.fps);
            glctx.renderer.renderComposite (descs.data(), (int) descs.size(),
                                            overlays.empty() ? nullptr : overlays.data(),
                                            (int) overlays.size());
        }
    }

    for (int64_t n = 0; n < totalFrames && error.empty(); ++n)
    {
        if (wantsAbort (progress))
        {
            error = "cancelled";
            break;
        }
        buildFrame (rangeStartSec + (double) n / job.fps);

        // Pipelined readback: this call kicks off frame n's GPU readback and
        // hands back frame n-1's pixels, so encode overlaps the next render.
        bool havePrev = false;
        if (! glctx.renderer.renderToPixelsAsync (descs.data(), (int) descs.size(),
                                                  rgba, havePrev, error,
                                                  overlays.empty() ? nullptr : overlays.data(),
                                                  (int) overlays.size()))
            break;

        if (havePrev)
        {
            if (av_frame_make_writable (frame) < 0)
            {
                error = "frame not writable";
                break;
            }
            const uint8_t* srcData[1] = { rgba.data() };
            const int srcStride[1] = { vEnc->width * 4 };
            sws_scale (sws, srcData, srcStride, 0, vEnc->height, frame->data, frame->linesize);

            frame->pts = encodedFrames++;
            error = encodeAndWrite (fmt, vStream, vEnc, frame);
        }
        fpsClock.update (progress, n + 1);
    }

    // The pipeline lags by one frame: collect and encode the final readback.
    if (error.empty() && glctx.renderer.drainAsyncReadback (rgba))
    {
        if (av_frame_make_writable (frame) < 0)
            error = "frame not writable";
        else
        {
            const uint8_t* srcData[1] = { rgba.data() };
            const int srcStride[1] = { vEnc->width * 4 };
            sws_scale (sws, srcData, srcStride, 0, vEnc->height, frame->data, frame->linesize);
            frame->pts = encodedFrames++;
            error = encodeAndWrite (fmt, vStream, vEnc, frame);
        }
    }

    for (auto& [clipId, cs] : streams)
        glctx.renderer.deleteTexture (cs.tex);
    for (auto& tl : textLayers)
        glctx.renderer.deleteTexture (tl.tex);
    for (auto& [clipId, l] : clipLuts)
        glctx.renderer.deleteTexture (l.tex);
    for (int clipId : shaderClips)
        glctx.renderer.clearClipShader (clipId);
    sws_freeContext (sws);
    av_frame_free (&frame);
    return error;
}

#endif // ARBIT_HAVE_VIEWPORT

} // namespace

#if ARBIT_HAVE_VIEWPORT
// Shared decode + offline Block B analysis (declared in mix_analyze.h). Reuses
// the file-local decodeWavToMonoFloat + the analyzer above so the LIVE viewport
// (stopped/scrubbing) reads byte-identical features to this exporter. Same TU
// ⇒ the anonymous-namespace helpers are visible here.
namespace videohelper
{
std::vector<arbitblockb::FeatureFrame> analyzeMixWavOffline (const std::string& wavPath,
                                                             double& srOut)
{
    srOut = 0.0;
    const MonoPcm pcm = decodeWavToMonoFloat (wavPath);
    if (pcm.samples.empty())
        return {};
    srOut = (double) pcm.sampleRate;
    arbitblockb::BlockBAnalyzer analyzer ((float) pcm.sampleRate);
    return analyzer.analyzeOffline (pcm.samples.data(), (int) pcm.samples.size());
}
} // namespace videohelper
#endif // ARBIT_HAVE_VIEWPORT

// ------------------------------------------------------------------ export

std::string runExport (const ExportJob& job, std::string& usedEncoderOut,
                       bool& glCompositingOut,
                       std::string& interpolationBackendOut,
                       ExportProgress* progress)
{
    glCompositingOut = false;

    // audit #6: refuse minterpolate when it would silently drop GL-composited
    // content (see minterpolateWouldDropContent). Everything downstream selects
    // the render path from effInterp, never job.interpolation directly.
    std::string effInterp = job.interpolation;
    if (effInterp == "minterpolate" && minterpolateWouldDropContent (job))
    {
        std::fprintf (stderr,
                      "[export] minterpolate disabled: project has composited "
                      "content (effects/text/generators/transitions/automation/"
                      "multi-layer) it cannot render — using GL retime to match "
                      "the preview\n");
        effInterp = "none";
    }

    interpolationBackendOut = effInterp == "minterpolate" ? "minterpolate"
                                                          : "nearest";
    setPhase (progress, 1);
    if (job.outPath.empty()) return "missing outPath";
    if (job.fps <= 0.0 || job.width <= 0 || job.height <= 0)
        return "invalid fps/width/height";
    if (job.codec == "prores")
    {
        // prores_ks writes QuickTime sample descriptions; mp4/mkv muxers
        // have no tag for it — require the .mov container up front.
        const auto& p = job.outPath;
        const bool isMov = p.size() > 4
            && (p.compare (p.size() - 4, 4, ".mov") == 0
                || p.compare (p.size() - 4, 4, ".MOV") == 0);
        if (! isMov)
            return "prores requires a .mov output file";
    }

    double timelineEndSec = job.durationSec;
    for (const auto& s : job.segments)
        timelineEndSec = std::max (timelineEndSec,
                                   s.displayStartSec + (s.outSec - s.inSec) / std::max (1e-9, s.rate));

    // Export range (PROTOCOL.md §Export): output frame 0 samples the display
    // timeline at rangeStart; an explicit endSec may extend past the content
    // (black + audio). Default = full timeline.
    const double rangeStart = std::max (0.0, job.startSec);
    const double rangeEnd = job.endSec > 0.0 ? job.endSec : timelineEndSec;
    const double durationSec = rangeEnd - rangeStart;
    if (durationSec <= 0.0)
        return "nothing to export (empty range)";

    // Validate every unique source upfront. Software-decode probes, destroyed
    // immediately: the GL path must never hold CUDA/VAAPI decode sessions
    // next to its GL context (see MediaContext::open).
    std::map<std::string, double> sourceFps;
    {
        // path -> (sequence fps hint, sequence start number); 0/-1 for
        // regular media (every segment of one path shares the clip's hints).
        std::map<std::string, std::pair<double, int>> uniquePaths;
        for (const auto& s : job.segments)
            if (! isGeneratorPath (s.sourcePath))   // generators have no media to probe
                uniquePaths.emplace (s.sourcePath, std::make_pair (s.sourceFps, s.seqStart));
        for (const auto& [p, hint] : uniquePaths)
        {
            MediaContext probe;
            auto err = probe.open (p, false, hint.first, hint.second);
            if (! err.empty()) return err;
            sourceFps[p] = probe.info().fps;
        }
    }

    // ---- RIFE ("speed warp") setup. Engaged per segment by the trigger
    // rule: source_fps * rate < target_fps (a warped segment no longer
    // delivers enough real frames). "rife" fails loudly when the backend is
    // unavailable; "auto" degrades silently to nearest-frame.
#if ARBIT_HAVE_VIEWPORT && ARBIT_HAVE_ONNX
    std::unique_ptr<arbitrife::RifeEngine> rife;
#endif
    {
        const bool jobWantsRife = (job.interpolation == "rife"
                                   || job.interpolation == "auto");
        // The job-global heuristic (under-delivers for the output grid)...
        bool jobTrigger = false;
        // ...and the per-clip Speed-Warp tier (audit #2/#5): the viewport runs
        // RIFE whenever a clip's retimeQuality>=2 and it is slowed, regardless of
        // any job-global flag (viewport.cpp:1400), and decodeLayer gates the same
        // way — so the export must set the engine up on the SAME condition, else
        // a Speed-Warp clip previews with RIFE but exports as frame-blend.
        bool tierTrigger = false;
        for (const auto& s : job.segments)
        {
            if (isGeneratorPath (s.sourcePath)) continue;  // generators emit every frame
            if (std::max (sourceFps[s.sourcePath], 1.0) * s.rate < job.fps * 0.999)
                jobTrigger = true;
            if (s.retimeQuality >= 2 && s.rate < 0.999)
                tierTrigger = true;
        }
        const bool needRife = (jobWantsRife && jobTrigger) || tierTrigger;
#if ARBIT_HAVE_VIEWPORT && ARBIT_HAVE_ONNX
        if (needRife)
        {
            rife = std::make_unique<arbitrife::RifeEngine>();
            if (auto rerr = rife->init(); ! rerr.empty())
            {
                rife.reset();
                // Only an explicit job-global "rife" fails loudly; "auto" and a
                // per-clip tier degrade silently (decodeLayer falls to Frame
                // Blend / nearest, preserving the composite).
                if (job.interpolation == "rife")
                    return "RIFE unavailable: " + rerr;
                std::fprintf (stderr, "[export] RIFE unavailable (%s) — falling "
                                      "back to Frame Blend / nearest\n",
                              rerr.c_str());
            }
        }
#else
        if (job.interpolation == "rife" && needRife)
            return "RIFE unavailable: helper built without ONNX + GL support";
#endif
    }

    std::string encName;
    const AVCodec* vCodec = pickVideoEncoder (job, encName);
    if (vCodec == nullptr)
        return "video encoder unavailable: " + encName;
    usedEncoderOut = encName;

    // DPX is a frame-numbered image sequence (image2 muxer): derive a %06d
    // pattern from the requested path so each output frame is its own .dpx file.
    const bool isDpx = (job.codec == "dpx");
    std::string outTarget = job.outPath;
    if (isDpx && outTarget.find ('%') == std::string::npos)
    {
        const std::string ext = ".dpx";
        if (outTarget.size() >= ext.size()
            && outTarget.compare (outTarget.size() - ext.size(), ext.size(), ext) == 0)
            outTarget = outTarget.substr (0, outTarget.size() - ext.size()) + "_%06d.dpx";
        else
            outTarget += "_%06d.dpx";
    }

    AVFormatContext* fmt = nullptr;
    if (avformat_alloc_output_context2 (&fmt, nullptr, isDpx ? "image2" : nullptr,
                                        outTarget.c_str()) < 0
        || fmt == nullptr)
        return "cannot create output context for " + outTarget;

    const AVRational fpsQ = av_d2q (job.fps, 100000);
    AVCodecContext* vEnc = avcodec_alloc_context3 (vCodec);
    vEnc->width = job.width & ~1;
    vEnc->height = job.height & ~1;
    const bool prores4444 = encName == "prores_ks"
        && (job.proresProfile == "4444" || job.proresProfile == "4444xq");
    vEnc->pix_fmt = isDpx                  ? AV_PIX_FMT_GBRP10LE       // 10-bit RGB DPX
                  : encName != "prores_ks" ? AV_PIX_FMT_YUV420P
                  : prores4444             ? AV_PIX_FMT_YUV444P10LE   // 4444 / 4444 XQ
                                           : AV_PIX_FMT_YUV422P10LE;  // 422 HQ
    vEnc->time_base = av_inv_q (fpsQ);
    vEnc->framerate = fpsQ;
    // §Render cache: intra-only output — every frame a keyframe, no B-frame
    // reordering, so the baked file random-accesses like an edit proxy.
    vEnc->gop_size = job.intraOnly ? 1 : (int) std::lround (job.fps * 2.0);
    if (job.intraOnly)
        vEnc->max_b_frames = 0;
    if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
        vEnc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (encName == "libx264" || encName == "libx265")
    {
        av_opt_set (vEnc->priv_data, "preset", job.intraOnly ? "veryfast" : "medium", 0);
        av_opt_set (vEnc->priv_data, "crf", "18", 0);
    }
    else if (encName == "prores_ks")
    {
        // prores_ks profiles: 0 proxy, 1 LT, 2 standard, 3 422 HQ, 4 4444, 5 4444 XQ
        const char* prof = job.proresProfile == "4444xq" ? "5"
                         : job.proresProfile == "4444"   ? "4"
                                                         : "3"; // 422 HQ (default)
        av_opt_set (vEnc->priv_data, "profile", prof, 0);
    }
    else
        vEnc->bit_rate = (int64_t) job.width * job.height * 8; // ~16 Mbps at 1080p

    tagColorBt709 (vEnc);

    if (avcodec_open2 (vEnc, vCodec, nullptr) < 0)
    {
        avcodec_free_context (&vEnc);
        avformat_free_context (fmt);
        // "auto" falls back to software when the hardware encoder won't open.
        if (job.encoder == "auto" && encName != softwareEncoderName (job.codec))
        {
            ExportJob sw = job;
            sw.encoder = "software";
            return runExport (sw, usedEncoderOut, glCompositingOut,
                              interpolationBackendOut, progress);
        }
        return "cannot open video encoder " + encName;
    }

    AVStream* vStream = avformat_new_stream (fmt, nullptr);
    avcodec_parameters_from_context (vStream->codecpar, vEnc);
    vStream->time_base = vEnc->time_base;

    std::string error;
    AudioTrack audio;
    const bool wantAudio = ! job.audioPath.empty() && ! isDpx;  // image2 carries no audio
    if (wantAudio)
    {
        error = audio.setup (fmt, job.audioPath);
        // Explicit range end: trim audio to exactly the range duration (the
        // range mix WAV carries a decay tail past endSec — see ExportJob).
        if (error.empty() && job.endSec > 0.0)
            audio.maxSamples = (int64_t) std::llround (durationSec * audio.enc->sample_rate);
        // §Loudness: measure the master's integrated loudness and attenuate to
        // the target. Downward-only — a master already at/under the target has
        // been through the mix/limiter chain and is never boosted.
        if (error.empty() && job.lufsTarget < 0.0)
        {
            const double measured = measureIntegratedLufs (job.audioPath);
            const double gainDb = std::min (0.0, job.lufsTarget - measured);
            audio.gainLinear = (float) std::pow (10.0, gainDb / 20.0);
        }
    }

    bool headerWritten = false;
    // image2/DPX opens each numbered frame file itself via the pattern — the
    // main pb stays null, so only open it for single-file muxers.
    if (error.empty() && ! isDpx
        && ! (fmt->oformat->flags & AVFMT_NOFILE)
        && avio_open (&fmt->pb, job.outPath.c_str(), AVIO_FLAG_WRITE) < 0)
        error = "cannot open output file " + job.outPath;

    if (error.empty())
    {
        if (avformat_write_header (fmt, nullptr) < 0)
            error = "write_header failed";
        else
            headerWritten = true;
    }

    if (error.empty() && wantAudio)
    {
        setPhase (progress, 2);
        error = audio.transcode (fmt, progress);
    }

    // ---- video frames: sample the display timeline at the output fps ----
    AVFrame* frame = av_frame_alloc();
    frame->format = vEnc->pix_fmt;
    frame->width = vEnc->width;
    frame->height = vEnc->height;
    av_frame_get_buffer (frame, 0);

    // The CPU paths (paintFrame, minterpolate) operate in yuv420p. When the
    // encoder wants something else (prores: yuv422p10le) they paint into a
    // yuv420p staging frame that encodeAt converts; otherwise work == frame.
    AVFrame* work = frame;
    SwsContext* workCvt = nullptr;
    if (vEnc->pix_fmt != AV_PIX_FMT_YUV420P)
    {
        work = av_frame_alloc();
        work->format = AV_PIX_FMT_YUV420P;
        work->width = vEnc->width;
        work->height = vEnc->height;
        av_frame_get_buffer (work, 0);
        workCvt = sws_getContext (vEnc->width, vEnc->height, AV_PIX_FMT_YUV420P,
                                  vEnc->width, vEnc->height, vEnc->pix_fmt,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (workCvt == nullptr && error.empty())
            error = "sws_getContext (yuv420p->encoder) failed";
    }

    SwsContext* sws = nullptr;
    int swsW = 0, swsH = 0;
    const int64_t totalFrames = (int64_t) std::llround (durationSec * job.fps);
    size_t segIdx = 0;
    FpsClock cpuFpsClock;

    auto paintBlack = [&]() -> std::string
    {
        if (av_frame_make_writable (work) < 0)
            return "frame not writable";
        return paintFrame (DecodedFrame {}, work, sws, swsW, swsH);
    };
    auto encodeAt = [&] (int64_t n) -> std::string
    {
        if (work != frame)
        {
            if (av_frame_make_writable (frame) < 0)
                return "frame not writable";
            sws_scale (workCvt, work->data, work->linesize, 0, vEnc->height,
                       frame->data, frame->linesize);
        }
        frame->pts = n;
        auto err = encodeAndWrite (fmt, vStream, vEnc, frame);
        cpuFpsClock.update (progress, n + 1);
        return err;
    };

    // ---- GL composited path (export parity with the live viewport). The
    // minterpolate path keeps the legacy per-segment filter graph (motion
    // interpolation needs the raw retimed source stream, not composited
    // output); headless / GL failure falls back too — never crash, just
    // report glCompositing = false.
    bool glDone = false;
#if ARBIT_HAVE_VIEWPORT
    if (error.empty() && effInterp != "minterpolate")
    {
        GlExportContext glctx;
        std::string glErr;
        if (glctx.init (vEnc->width, vEnc->height, glErr))
        {
#if ARBIT_HAVE_ONNX
            error = runGlFrameLoop (job, glctx, fmt, vStream, vEnc, rangeStart,
                                    durationSec, rife.get(), progress);
            if (error.empty() && rife != nullptr)
            {
                interpolationBackendOut = rife->backend();
                if (rife->inferenceCount() > 0)
                    std::fprintf (stderr,
                                  "[rife] synthesized %d frames, avg %.1f ms "
                                  "(%s)\n",
                                  rife->inferenceCount(),
                                  rife->totalInferenceMs()
                                      / rife->inferenceCount(),
                                  rife->backend().c_str());
            }
#else
            error = runGlFrameLoop (job, glctx, fmt, vStream, vEnc, rangeStart,
                                    durationSec, nullptr, progress);
#endif
            glctx.shutdown();
            glDone = true;
            glCompositingOut = error.empty();
        }
        else
            std::fprintf (stderr,
                          "[export] GL compositing unavailable (%s) — CPU "
                          "fallback, effects/compositing skipped\n",
                          glErr.c_str());
    }
#endif

    // Legacy paths share one decode context per source path (hw decode fine:
    // no GL context exists in-process on these paths).
    std::map<std::string, std::unique_ptr<MediaContext>> sources;
    if (error.empty() && ! glDone)
        for (const auto& s : job.segments)
        {
            if (isGeneratorPath (s.sourcePath)) continue;  // no-GL path can't render generators
            if (sources.count (s.sourcePath)) continue;
            auto ctx = std::make_unique<MediaContext>();
            error = ctx->open (s.sourcePath, true, s.sourceFps, s.seqStart);
            if (! error.empty()) break;
            sources[s.sourcePath] = std::move (ctx);
        }

    if (error.empty() && ! glDone && progress != nullptr)
    {
        progress->totalFrames.store (totalFrames, std::memory_order_relaxed);
        setPhase (progress, 3);
    }

    if (error.empty() && ! glDone && effInterp == "minterpolate")
    {
        // Stream each segment's unique source frames through a minterpolate
        // graph with display-time pts (rebased to the export range); the
        // filter emits frames on the output fps grid. Gaps (and any indices
        // the filter skips) render black.
        const AVRational tb { 1, 90000 };
        int64_t nextN = 0;

        auto blackFillTo = [&] (int64_t n) -> std::string
        {
            std::string err;
            while (nextN < n && err.empty())
            {
                if (wantsAbort (progress))
                    return "cancelled";
                err = paintBlack();
                if (err.empty())
                    err = encodeAt (nextN++);
            }
            return err;
        };

        for (const auto& seg : job.segments)
        {
            if (! error.empty())
                break;
            if (isGeneratorPath (seg.sourcePath))
                continue;   // no-GL minterpolate path cannot render a generator

            // Skip segments entirely outside the export range.
            const double rangeSegDur = (seg.outSec - seg.inSec) / std::max (1e-9, seg.rate);
            if (seg.displayStartSec >= rangeEnd
                || seg.displayStartSec + rangeSegDur <= rangeStart)
                continue;

            auto& media = sources[seg.sourcePath];
            double srcFps = media->info().fps;
            if (srcFps <= 1.0)
                srcFps = 30.0;

            InterpGraph ig;
            error = ig.init (vEnc->width, vEnc->height, job.fps, tb);
            if (! error.empty())
                break;

            auto drainSink = [&]() -> std::string
            {
                AVFrame* out = av_frame_alloc();
                std::string err;
                while (err.empty() && av_buffersink_get_frame (ig.sink, out) >= 0)
                {
                    const double sec = (double) out->pts
                                     * av_q2d (av_buffersink_get_time_base (ig.sink));
                    const int64_t n = (int64_t) std::llround (sec * job.fps);
                    if (n >= nextN && n < totalFrames)
                    {
                        err = blackFillTo (n);
                        if (err.empty())
                        {
                            if (av_frame_make_writable (work) < 0)
                                err = "frame not writable";
                            else if (av_frame_copy (work, out) < 0)
                                err = "filtered frame copy failed";
                            else
                            {
                                err = encodeAt (n);
                                nextN = n + 1;
                            }
                        }
                    }
                    av_frame_unref (out);
                }
                av_frame_free (&out);
                return err;
            };

            // Feed every distinct source frame across the segment.
            AVFrame* fed = av_frame_alloc();
            fed->format = AV_PIX_FMT_YUV420P;
            fed->width = vEnc->width;
            fed->height = vEnc->height;
            av_frame_get_buffer (fed, 0);

            double lastPts = -1.0;
            const double step = 1.0 / srcFps;
            for (double srcSec = seg.inSec;
                 srcSec < seg.outSec + step * 0.5 && error.empty();
                 srcSec += step)
            {
                if (wantsAbort (progress))
                {
                    error = "cancelled";
                    break;
                }
                DecodedFrame df;
                if (! media->getFrame (std::min (srcSec, seg.outSec), vEnc->width,
                                       vEnc->height, df).empty())
                    continue;
                if (df.ptsSec <= lastPts)
                    continue; // same frame as last feed
                lastPts = df.ptsSec;

                if (av_frame_make_writable (fed) < 0)
                {
                    error = "feed frame not writable";
                    break;
                }
                error = paintFrame (df, fed, sws, swsW, swsH);
                if (! error.empty())
                    break;

                const double clampedSrc = std::clamp (df.ptsSec, seg.inSec, seg.outSec);
                const double displaySec = seg.displayStartSec
                                        + (clampedSrc - seg.inSec) / std::max (1e-9, seg.rate);
                // Rebase onto the export range; frames slightly before the
                // range still feed the filter (interpolation context) and the
                // drain's n >= nextN check drops their (negative) indices.
                fed->pts = (int64_t) std::llround ((displaySec - rangeStart) / av_q2d (tb));

                // KEEP_REF: the graph takes its own reference, fed stays usable.
                if (av_buffersrc_add_frame_flags (ig.src, fed, AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
                {
                    error = "buffersrc add failed";
                    break;
                }
                error = drainSink();
            }
            av_frame_free (&fed);

            if (error.empty())
            {
                (void) av_buffersrc_add_frame (ig.src, nullptr); // flush this segment
                error = drainSink();
            }
        }

        if (error.empty())
            error = blackFillTo (totalFrames);
    }
    else if (! glDone) for (int64_t n = 0; n < totalFrames && error.empty(); ++n)
    {
        if (wantsAbort (progress))
        {
            error = "cancelled";
            break;
        }
        const double t = rangeStart + (double) n / job.fps;

        const ExportSegment* seg = nullptr;
        for (size_t i = segIdx; i < job.segments.size(); ++i)
        {
            const auto& s = job.segments[i];
            const double segDur = (s.outSec - s.inSec) / std::max (1e-9, s.rate);
            if (t < s.displayStartSec) break;
            if (t < s.displayStartSec + segDur)
            {
                seg = &s;
                segIdx = i;
                break;
            }
        }

        if (av_frame_make_writable (work) < 0)
        {
            error = "frame not writable";
            break;
        }

        DecodedFrame df; // empty = black frame
        if (seg != nullptr && ! isGeneratorPath (seg->sourcePath))
        {
            const double srcSec = seg->inSec + (t - seg->displayStartSec) * seg->rate;
            DecodedFrame decoded;
            if (sources[seg->sourcePath]->getFrame (srcSec, vEnc->width, vEnc->height, decoded).empty())
                df = std::move (decoded); // decode errors render black, export continues
        }
        error = paintFrame (df, work, sws, swsW, swsH);

        if (error.empty())
            error = encodeAt (n);
    }

    if (error.empty())
    {
        setPhase (progress, 4);
        error = encodeAndWrite (fmt, vStream, vEnc, nullptr); // flush video
    }

    if (error.empty() && headerWritten && av_write_trailer (fmt) < 0)
        error = "write_trailer failed";

    sws_freeContext (sws);
    if (work != frame)
        av_frame_free (&work);
    sws_freeContext (workCvt);
    av_frame_free (&frame);
    avcodec_free_context (&vEnc);
    if (! (fmt->oformat->flags & AVFMT_NOFILE) && fmt->pb != nullptr)
        avio_closep (&fmt->pb);
    avformat_free_context (fmt);

    // A cancelled export must not leave a truncated file behind.
    if (error == "cancelled")
        std::remove (job.outPath.c_str());
    return error;
}

//======================================================================
// RecorderSession — push-frame encode for the live piano-roll recorder
// (record_open / record_push_frame / record_close RPCs). See exporter.h for
// the "why": reuses pickVideoEncoder/tagColorBt709/encodeAndWrite verbatim so
// a recording gets the same hardware-encoder choice and colour tagging as a
// normal export, with none of runExport's timeline-compositing machinery.

struct RecorderSession::Impl
{
    AVFormatContext* fmt = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* stream = nullptr;
    AVFrame* yuvFrame = nullptr;   // encoder-ready, yuv420p
    SwsContext* sws = nullptr;     // bgra/rgba -> yuv420p
    int srcPixFmt = -1;            // AV_PIX_FMT of the sws source (rebuild if it changes)
    int width = 0, height = 0;
    int64_t frameCount = 0;
    bool headerWritten = false;
    std::string outPath;

    ~Impl() { teardown(); }

    void teardown()
    {
        if (sws != nullptr) { sws_freeContext (sws); sws = nullptr; }
        if (yuvFrame != nullptr) av_frame_free (&yuvFrame);
        if (enc != nullptr) avcodec_free_context (&enc);
        if (fmt != nullptr)
        {
            if (! (fmt->oformat->flags & AVFMT_NOFILE) && fmt->pb != nullptr)
                avio_closep (&fmt->pb);
            avformat_free_context (fmt);
            fmt = nullptr;
        }
        headerWritten = false;
        stream = nullptr;
    }
};

RecorderSession::RecorderSession() : impl_ (std::make_unique<Impl>()) {}
RecorderSession::~RecorderSession() = default;

bool RecorderSession::isOpen() const { return impl_->fmt != nullptr; }
int64_t RecorderSession::framesEncoded() const { return impl_->frameCount; }

std::string RecorderSession::open (const std::string& outPath, int width, int height,
                                   double fps, const std::string& codec,
                                   const std::string& encoder)
{
    if (impl_->fmt != nullptr)
        return "recorder session already open";
    if (width <= 0 || height <= 0 || fps <= 0.0)
        return "invalid recorder dimensions/fps";

    // pickVideoEncoder/softwareEncoderName only read job.codec/job.encoder —
    // a stub ExportJob with just those two set is a safe, honest reuse.
    ExportJob stub;
    stub.codec = codec;
    stub.encoder = encoder;

    std::string encName;
    const AVCodec* vCodec = pickVideoEncoder (stub, encName);
    if (vCodec == nullptr)
        return "video encoder unavailable: " + encName;

    AVFormatContext* fmt = nullptr;
    if (avformat_alloc_output_context2 (&fmt, nullptr, nullptr, outPath.c_str()) < 0
        || fmt == nullptr)
        return "cannot create output context for " + outPath;

    const AVRational fpsQ = av_d2q (fps, 100000);
    AVCodecContext* enc = avcodec_alloc_context3 (vCodec);
    enc->width = width & ~1;
    enc->height = height & ~1;
    enc->pix_fmt = AV_PIX_FMT_YUV420P; // recordings are always SDR h264/h265 — no
                                        // ProRes/DPX pixel formats to branch on
    enc->time_base = av_inv_q (fpsQ);
    enc->framerate = fpsQ;
    enc->gop_size = (int) std::lround (fps * 2.0);
    if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
        enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (encName == "libx264" || encName == "libx265")
    {
        av_opt_set (enc->priv_data, "preset", "veryfast", 0);
        av_opt_set (enc->priv_data, "crf", "20", 0);
    }
    else
        enc->bit_rate = (int64_t) width * height * 8; // ~16 Mbps at 1080p, same as export

    tagColorBt709 (enc);

    if (avcodec_open2 (enc, vCodec, nullptr) < 0)
    {
        avcodec_free_context (&enc);
        avformat_free_context (fmt);
        // "auto" falls back to software when the hardware encoder won't open —
        // same fallback runExport does.
        if (encoder == "auto" && encName != softwareEncoderName (codec))
            return open (outPath, width, height, fps, codec, "software");
        return "cannot open video encoder " + encName;
    }

    AVStream* stream = avformat_new_stream (fmt, nullptr);
    avcodec_parameters_from_context (stream->codecpar, enc);
    stream->time_base = enc->time_base;

    if (! (fmt->oformat->flags & AVFMT_NOFILE)
        && avio_open (&fmt->pb, outPath.c_str(), AVIO_FLAG_WRITE) < 0)
    {
        avcodec_free_context (&enc);
        avformat_free_context (fmt);
        return "cannot open output file " + outPath;
    }

    if (avformat_write_header (fmt, nullptr) < 0)
    {
        avcodec_free_context (&enc);
        if (! (fmt->oformat->flags & AVFMT_NOFILE) && fmt->pb != nullptr)
            avio_closep (&fmt->pb);
        avformat_free_context (fmt);
        return "write_header failed";
    }

    AVFrame* yuvFrame = av_frame_alloc();
    yuvFrame->format = enc->pix_fmt;
    yuvFrame->width = enc->width;
    yuvFrame->height = enc->height;
    av_frame_get_buffer (yuvFrame, 0);

    impl_->fmt = fmt;
    impl_->enc = enc;
    impl_->stream = stream;
    impl_->yuvFrame = yuvFrame;
    impl_->width = enc->width;
    impl_->height = enc->height;
    impl_->headerWritten = true;
    impl_->frameCount = 0;
    impl_->outPath = outPath;
    return {};
}

std::string RecorderSession::pushFrame (const uint8_t* pixels, int strideBytes, bool bgra)
{
    if (impl_->fmt == nullptr)
        return "recorder session not open";
    if (pixels == nullptr || strideBytes <= 0)
        return "invalid frame buffer";

    const AVPixelFormat srcFmt = bgra ? AV_PIX_FMT_BGRA : AV_PIX_FMT_RGBA;
    if (impl_->sws == nullptr || impl_->srcPixFmt != (int) srcFmt)
    {
        if (impl_->sws != nullptr) sws_freeContext (impl_->sws);
        impl_->sws = sws_getContext (impl_->width, impl_->height, srcFmt,
                                     impl_->width, impl_->height, AV_PIX_FMT_YUV420P,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);
        impl_->srcPixFmt = (int) srcFmt;
        if (impl_->sws == nullptr)
            return "sws_getContext (capture->encoder) failed";
    }

    if (av_frame_make_writable (impl_->yuvFrame) < 0)
        return "frame not writable";

    const uint8_t* srcSlices[1] = { pixels };
    const int srcStrides[1] = { strideBytes };
    sws_scale (impl_->sws, srcSlices, srcStrides, 0, impl_->height,
              impl_->yuvFrame->data, impl_->yuvFrame->linesize);

    impl_->yuvFrame->pts = impl_->frameCount;
    auto err = encodeAndWrite (impl_->fmt, impl_->stream, impl_->enc, impl_->yuvFrame);
    if (! err.empty())
        return err;
    ++impl_->frameCount;
    return {};
}

std::string RecorderSession::close()
{
    if (impl_->fmt == nullptr)
        return {}; // never opened / already closed — not an error, matches Region::close()

    std::string error = encodeAndWrite (impl_->fmt, impl_->stream, impl_->enc, nullptr); // flush
    if (error.empty() && impl_->headerWritten && av_write_trailer (impl_->fmt) < 0)
        error = "write_trailer failed";

    if (! error.empty())
        std::remove (impl_->outPath.c_str()); // no truncated file left behind, same as runExport

    impl_->teardown();
    return error;
}
