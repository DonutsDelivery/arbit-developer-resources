#include "media.h"

extern "C"
{
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavutil/opt.h>
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

#if ARBIT_HAVE_AUBIO
#include <aubio/aubio.h>
#endif

namespace
{
std::string avErr (int code)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror (code, buf, sizeof (buf));
    return std::string (buf);
}

double streamTimeToSec (const AVStream* st, int64_t ts)
{
    if (ts == AV_NOPTS_VALUE) return 0.0;
    return (double) ts * st->time_base.num / st->time_base.den;
}

// get_format callback for hardware decoding: prefer the negotiated hw pixel
// format, fall back to the first software format the decoder offers.
AVPixelFormat getHwFormat (AVCodecContext* ctx, const AVPixelFormat* fmts)
{
    const auto desired = (AVPixelFormat) (intptr_t) ctx->opaque;
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == desired)
            return *p;
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        if (av_pix_fmt_desc_get (*p) != nullptr
            && (av_pix_fmt_desc_get (*p)->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0)
            return *p;
    return fmts[0];
}
} // namespace

AVBufferRef* sharedCudaDeviceCtx()
{
    // Create exactly ONE CUDA device context for the whole process, lazily and
    // single-threaded (call_once). Sharing one hw_device_ctx across all decode
    // sessions — instead of av_hwdevice_ctx_create per MediaContext — avoids the
    // concurrent cuInit()/CUDA-context creation that deadlocked the driver when
    // two threads opened hw decoders at once. main() pokes this at startup so the
    // one-time init happens before any worker/render thread exists.
    static std::once_flag onceFlag;
    static AVBufferRef* ctx = nullptr;
    std::call_once (onceFlag, []
    {
        if (std::getenv ("ARBIT_DISABLE_HWDEC") != nullptr)
            return; // escape hatch: ctx stays nullptr -> software everywhere
        AVBufferRef* created = nullptr;
        if (av_hwdevice_ctx_create (&created, AV_HWDEVICE_TYPE_CUDA,
                                    nullptr, nullptr, 0) == 0)
            ctx = created; // permanent process-lifetime ref (never unref'd)
        // On failure (no NVIDIA / no CUDA in this FFmpeg build) ctx stays
        // nullptr and callers silently fall back to software decode.
    });
    return ctx;
}

bool MediaContext::tryOpenHwDecoder (const AVCodec* codec)
{
    // Platform preference order; each is tried only if this FFmpeg build and
    // the running machine actually support it (runtime fallback to software).
    static const AVHWDeviceType preferred[] = {
#if defined(__APPLE__)
        AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
#elif defined(_WIN32)
        AV_HWDEVICE_TYPE_D3D11VA,
        AV_HWDEVICE_TYPE_CUDA,
#else
        AV_HWDEVICE_TYPE_CUDA,        // NVDEC
        AV_HWDEVICE_TYPE_VAAPI,
#endif
        AV_HWDEVICE_TYPE_NONE
    };

    for (int t = 0; preferred[t] != AV_HWDEVICE_TYPE_NONE; ++t)
    {
        // Does this decoder expose a hw config for the device type?
        AVPixelFormat hwFmt = AV_PIX_FMT_NONE;
        for (int i = 0;; ++i)
        {
            const AVCodecHWConfig* cfg = avcodec_get_hw_config (codec, i);
            if (cfg == nullptr) break;
            if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0
                && cfg->device_type == preferred[t])
            {
                hwFmt = cfg->pix_fmt;
                break;
            }
        }
        if (hwFmt == AV_PIX_FMT_NONE) continue;

        AVBufferRef* deviceCtx = nullptr;
        if (preferred[t] == AV_HWDEVICE_TYPE_CUDA)
        {
            // Reference the ONE shared CUDA device context instead of creating a
            // new one here — av_buffer_ref only bumps a refcount, so no second
            // cuInit/CUDA-context creation runs on this thread (that concurrent
            // init is what deadlocked the driver). nullptr = unavailable or
            // ARBIT_DISABLE_HWDEC set -> fall through to the next device type
            // (then software).
            //
            // Thread-safety: FFmpeg's CUDA hwcontext makes the shared CUDA
            // context current per operation (cuCtxPushCurrent/PopCurrent) and
            // serializes access via the AVHWDeviceContext lock callbacks, so
            // multiple decode threads referencing the same hw_device_ctx is the
            // supported, recommended pattern — each MediaContext still has its
            // own AVCodecContext.
            AVBufferRef* shared = sharedCudaDeviceCtx();
            if (shared == nullptr) continue;
            deviceCtx = av_buffer_ref (shared);
            if (deviceCtx == nullptr) continue;
        }
        else if (av_hwdevice_ctx_create (&deviceCtx, preferred[t], nullptr, nullptr, 0) < 0)
            continue;

        hwDeviceCtx_ = deviceCtx;
        hwPixFmt_ = hwFmt;
        videoDec_->hw_device_ctx = av_buffer_ref (deviceCtx);
        videoDec_->opaque = (void*) (intptr_t) hwFmt;
        videoDec_->get_format = getHwFormat;
        info_.hwaccel = av_hwdevice_get_type_name (preferred[t]);
        return true;
    }
    return false;
}

MediaContext::~MediaContext()
{
    if (lastFrame_ != nullptr) av_frame_free (&lastFrame_);
    if (hwTransferFrame_ != nullptr) av_frame_free (&hwTransferFrame_);
    if (hwDeviceCtx_ != nullptr) av_buffer_unref (&hwDeviceCtx_);
    if (sws_ != nullptr) sws_freeContext (sws_);
    if (videoDec_ != nullptr) avcodec_free_context (&videoDec_);
    if (audioDec_ != nullptr) avcodec_free_context (&audioDec_);
    if (fmt_ != nullptr) avformat_close_input (&fmt_);
}

std::string MediaContext::open (const std::string& path, bool allowHwDecode,
                                double seqFps, int seqStartNumber)
{
    path_ = path;

    // Image-sequence pattern ("img_%04d.png"): open through the image2
    // demuxer with an explicit frame rate so pts/duration are correct.
    const AVInputFormat* inputFormat = nullptr;
    AVDictionary* openOpts = nullptr;
    if (path.find ('%') != std::string::npos)
    {
        inputFormat = av_find_input_format ("image2");
        char fpsBuf[32];
        std::snprintf (fpsBuf, sizeof (fpsBuf), "%.6f", seqFps > 0.0 ? seqFps : 30.0);
        av_dict_set (&openOpts, "framerate", fpsBuf, 0);
        if (seqStartNumber >= 0)
            av_dict_set_int (&openOpts, "start_number", seqStartNumber, 0);
    }

    int rc = avformat_open_input (&fmt_, path.c_str(), inputFormat, &openOpts);
    av_dict_free (&openOpts);
    if (rc < 0) return "open failed: " + avErr (rc);
    rc = avformat_find_stream_info (fmt_, nullptr);
    if (rc < 0) return "stream info failed: " + avErr (rc);

    videoStream_ = av_find_best_stream (fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    audioStream_ = av_find_best_stream (fmt_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (videoStream_ < 0 && audioStream_ < 0)
        return "no audio or video streams found";

    info_.container = fmt_->iformat != nullptr ? fmt_->iformat->name : "";
    if (fmt_->duration != AV_NOPTS_VALUE)
        info_.durationSec = (double) fmt_->duration / AV_TIME_BASE;

    if (videoStream_ >= 0)
    {
        auto* st = fmt_->streams[videoStream_];

        // Alpha detection. Two cases:
        //  - the stream pix_fmt itself carries alpha (yuva420p, yuva444p10,
        //    rgba/bgra, ProRes 4444, qtrle 32-bit, PNG, ...)
        //  - webm/mkv VP8/VP9 alpha: the alpha plane travels as block
        //    additional data; the stream metadata carries alpha_mode=1 and
        //    only the libvpx decoders reassemble it (-> yuva420p).
        bool fmtHasAlpha = false;
        if (const auto* desc = av_pix_fmt_desc_get ((AVPixelFormat) st->codecpar->format))
            fmtHasAlpha = (desc->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        bool vpxAlpha = false;
        if (st->codecpar->codec_id == AV_CODEC_ID_VP8
            || st->codecpar->codec_id == AV_CODEC_ID_VP9)
            if (const auto* e = av_dict_get (st->metadata, "alpha_mode", nullptr, 0))
                vpxAlpha = std::strcmp (e->value, "1") == 0;

        const AVCodec* codec = nullptr;
        if (vpxAlpha)
            codec = avcodec_find_decoder_by_name (
                st->codecpar->codec_id == AV_CODEC_ID_VP9 ? "libvpx-vp9" : "libvpx");
        if (codec == nullptr)
        {
            vpxAlpha = false; // native vp8/vp9 decoders drop the alpha plane
            codec = avcodec_find_decoder (st->codecpar->codec_id);
        }
        if (codec == nullptr) return "no video decoder for codec";
        info_.hasAlpha = fmtHasAlpha || vpxAlpha;

        videoDec_ = avcodec_alloc_context3 (codec);
        avcodec_parameters_to_context (videoDec_, st->codecpar);
        videoDec_->thread_count = 0; // auto

        // Never hardware-decode alpha sources: hw surfaces are NV12/P010 and
        // the alpha plane would be silently dropped on download.
        const bool wantHw = allowHwDecode && ! info_.hasAlpha && tryOpenHwDecoder (codec);
        rc = avcodec_open2 (videoDec_, codec, nullptr);
        if (rc < 0 && wantHw)
        {
            // Hardware path failed at open — rebuild as a software decoder.
            avcodec_free_context (&videoDec_);
            if (hwDeviceCtx_ != nullptr) av_buffer_unref (&hwDeviceCtx_);
            hwPixFmt_ = AV_PIX_FMT_NONE;
            info_.hwaccel.clear();
            videoDec_ = avcodec_alloc_context3 (codec);
            avcodec_parameters_to_context (videoDec_, st->codecpar);
            videoDec_->thread_count = 0;
            rc = avcodec_open2 (videoDec_, codec, nullptr);
        }
        if (rc < 0) return "video decoder open failed: " + avErr (rc);

        info_.hasVideo = true;
        info_.width = videoDec_->width;
        info_.height = videoDec_->height;
        info_.videoCodec = codec->name;
        auto fr = av_guess_frame_rate (fmt_, st, nullptr);
        info_.fps = fr.den != 0 ? (double) fr.num / fr.den : 0.0;
        if (info_.durationSec <= 0.0 && st->duration != AV_NOPTS_VALUE)
            info_.durationSec = streamTimeToSec (st, st->duration);
    }

    if (audioStream_ >= 0)
    {
        auto* par = fmt_->streams[audioStream_]->codecpar;
        info_.hasAudio = true;
        info_.audioSampleRate = par->sample_rate;
        info_.audioChannels = par->ch_layout.nb_channels;
        const AVCodec* c = avcodec_find_decoder (par->codec_id);
        info_.audioCodec = c != nullptr ? c->name : "";
    }

    lastFrame_ = av_frame_alloc();
    return {};
}

bool MediaContext::openAudioDecoder (std::string& error)
{
    if (audioDec_ != nullptr) return true;
    if (audioStream_ < 0) { error = "media has no audio stream"; return false; }

    auto* st = fmt_->streams[audioStream_];
    const AVCodec* codec = avcodec_find_decoder (st->codecpar->codec_id);
    if (codec == nullptr) { error = "no audio decoder for codec"; return false; }
    audioDec_ = avcodec_alloc_context3 (codec);
    avcodec_parameters_to_context (audioDec_, st->codecpar);
    int rc = avcodec_open2 (audioDec_, codec, nullptr);
    if (rc < 0)
    {
        avcodec_free_context (&audioDec_);
        error = "audio decoder open failed: " + avErr (rc);
        return false;
    }
    return true;
}

void MediaContext::downloadIfHw (AVFrame* frame)
{
    if (hwPixFmt_ == AV_PIX_FMT_NONE || frame->format != hwPixFmt_)
        return;
    if (hwTransferFrame_ == nullptr)
        hwTransferFrame_ = av_frame_alloc();
    av_frame_unref (hwTransferFrame_);
    if (av_hwframe_transfer_data (hwTransferFrame_, frame, 0) == 0)
    {
        av_frame_copy_props (hwTransferFrame_, frame);
        av_frame_unref (frame);
        av_frame_move_ref (frame, hwTransferFrame_);
    }
}

std::string MediaContext::decodeForwardUntil (double targetSec, AVFrame* frame, bool& gotFrame)
{
    gotFrame = false;
    AVPacket* pkt = av_packet_alloc();
    auto* st = fmt_->streams[videoStream_];
    const double frameDur = info_.fps > 0.0 ? 1.0 / info_.fps : 1.0 / 30.0;

    std::string error;
    while (true)
    {
        int rc = av_read_frame (fmt_, pkt);
        if (rc == AVERROR_EOF)
        {
            // flush
            avcodec_send_packet (videoDec_, nullptr);
            while (avcodec_receive_frame (videoDec_, frame) == 0)
            {
                downloadIfHw (frame);
                gotFrame = true;
                lastDecodedPts_ = streamTimeToSec (st, frame->best_effort_timestamp);
                if (lastDecodedPts_ + frameDur > targetSec) break;
            }
            avcodec_flush_buffers (videoDec_);
            break;
        }
        if (rc < 0) { error = "read frame failed: " + avErr (rc); break; }

        if (pkt->stream_index != videoStream_)
        {
            av_packet_unref (pkt);
            continue;
        }

        rc = avcodec_send_packet (videoDec_, pkt);
        av_packet_unref (pkt);
        if (rc < 0 && rc != AVERROR (EAGAIN)) { error = "send packet failed: " + avErr (rc); break; }

        bool reached = false;
        while (avcodec_receive_frame (videoDec_, frame) == 0)
        {
            downloadIfHw (frame);
            gotFrame = true;
            lastDecodedPts_ = streamTimeToSec (st, frame->best_effort_timestamp);
            if (lastDecodedPts_ + frameDur > targetSec) { reached = true; break; }
        }
        if (reached) break;
    }

    av_packet_free (&pkt);
    return error;
}

std::string MediaContext::scaleToRgba (AVFrame* frame, int maxW, int maxH, DecodedFrame& out)
{
    int dstW = frame->width;
    int dstH = frame->height;
    if (maxW > 0 && maxH > 0 && (dstW > maxW || dstH > maxH))
    {
        const double s = std::min ((double) maxW / dstW, (double) maxH / dstH);
        dstW = std::max (2, (int) std::lround (dstW * s) & ~1);
        dstH = std::max (2, (int) std::lround (dstH * s) & ~1);
    }

    sws_ = sws_getCachedContext (sws_, frame->width, frame->height,
                                 (AVPixelFormat) frame->format,
                                 dstW, dstH, AV_PIX_FMT_RGBA,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ == nullptr) return "swscale context failed";

    out.width = dstW;
    out.height = dstH;
    out.strideBytes = dstW * 4;
    out.rgba.resize ((size_t) out.strideBytes * dstH);

    uint8_t* dstData[4] = { out.rgba.data(), nullptr, nullptr, nullptr };
    int dstLines[4] = { out.strideBytes, 0, 0, 0 };
    sws_scale (sws_, frame->data, frame->linesize, 0, frame->height, dstData, dstLines);
    return {};
}

std::string MediaContext::getFrame (double timeSec, int maxW, int maxH, DecodedFrame& out)
{
    std::lock_guard<std::mutex> lock (mutex_);
    if (! info_.hasVideo) return "media has no video stream";

    timeSec = std::clamp (timeSec, 0.0, std::max (0.0, info_.durationSec - 0.001));
    const double frameDur = info_.fps > 0.0 ? 1.0 / info_.fps : 1.0 / 30.0;

    // Already showing the right frame?
    if (lastFrame_->width > 0 && timeSec >= lastDecodedPts_
        && timeSec < lastDecodedPts_ + frameDur)
    {
        out.ptsSec = lastDecodedPts_;
        return scaleToRgba (lastFrame_, maxW, maxH, out);
    }

    // Backward jump, or forward jump beyond ~1s: seek to keyframe before target.
    const bool sequential = timeSec > lastDecodedPts_ && timeSec < lastDecodedPts_ + 1.0;
    if (! sequential)
    {
        auto* st = fmt_->streams[videoStream_];
        const int64_t ts = (int64_t) std::llround (timeSec * st->time_base.den
                                                   / (double) st->time_base.num);
        int rc = av_seek_frame (fmt_, videoStream_, ts, AVSEEK_FLAG_BACKWARD);
        if (rc < 0) return "seek failed: " + avErr (rc);
        avcodec_flush_buffers (videoDec_);
        lastDecodedPts_ = -1.0e9;
    }

    bool gotFrame = false;
    auto error = decodeForwardUntil (timeSec, lastFrame_, gotFrame);
    if (! error.empty()) return error;
    if (! gotFrame) return "no frame decoded at requested time";
    out.ptsSec = lastDecodedPts_;
    return scaleToRgba (lastFrame_, maxW, maxH, out);
}

// ============================================================================
// Audio extraction (float32 WAV)
// ============================================================================

namespace
{
struct WavWriter
{
    FILE* f = nullptr;
    uint32_t dataBytes = 0;
    int channels = 0, sampleRate = 0;

    bool start (const std::string& path, int ch, int rate)
    {
        channels = ch; sampleRate = rate;
        f = std::fopen (path.c_str(), "wb");
        if (f == nullptr) return false;
        uint8_t hdr[44] = {};
        std::fwrite (hdr, 1, 44, f); // placeholder, patched in finish()
        return true;
    }

    void write (const float* interleaved, size_t frames)
    {
        const size_t bytes = frames * (size_t) channels * sizeof (float);
        std::fwrite (interleaved, 1, bytes, f);
        dataBytes += (uint32_t) bytes;
    }

    void finish()
    {
        auto put32 = [this] (uint32_t v) { std::fwrite (&v, 4, 1, f); };
        auto put16 = [this] (uint16_t v) { std::fwrite (&v, 2, 1, f); };
        std::fseek (f, 0, SEEK_SET);
        std::fwrite ("RIFF", 1, 4, f); put32 (36 + dataBytes);
        std::fwrite ("WAVE", 1, 4, f);
        std::fwrite ("fmt ", 1, 4, f); put32 (16);
        put16 (3 /*IEEE float*/); put16 ((uint16_t) channels);
        put32 ((uint32_t) sampleRate);
        put32 ((uint32_t) (sampleRate * channels * 4));
        put16 ((uint16_t) (channels * 4)); put16 (32);
        std::fwrite ("data", 1, 4, f); put32 (dataBytes);
        std::fclose (f);
        f = nullptr;
    }
};
} // namespace

std::string MediaContext::extractAudio (const std::string& outPath, double& durationSecOut,
                                        int& sampleRateOut, int& channelsOut)
{
    std::lock_guard<std::mutex> lock (mutex_);
    std::string error;
    if (! openAudioDecoder (error)) return error;

    auto* st = fmt_->streams[audioStream_];
    const int outChannels = std::min (2, std::max (1, audioDec_->ch_layout.nb_channels));
    const int outRate = audioDec_->sample_rate;

    SwrContext* swr = nullptr;
    AVChannelLayout outLayout;
    av_channel_layout_default (&outLayout, outChannels);
    int rc = swr_alloc_set_opts2 (&swr, &outLayout, AV_SAMPLE_FMT_FLT, outRate,
                                  &audioDec_->ch_layout, audioDec_->sample_fmt,
                                  audioDec_->sample_rate, 0, nullptr);
    if (rc < 0 || swr_init (swr) < 0) return "resampler init failed";

    WavWriter wav;
    if (! wav.start (outPath, outChannels, outRate))
    {
        swr_free (&swr);
        return "cannot write " + outPath;
    }

    // Decode the audio stream from the start with an independent read position:
    // use a second format context so the video decode state is untouched.
    AVFormatContext* afmt = nullptr;
    if (avformat_open_input (&afmt, path_.c_str(), nullptr, nullptr) < 0
        || avformat_find_stream_info (afmt, nullptr) < 0)
    {
        if (afmt != nullptr) avformat_close_input (&afmt);
        swr_free (&swr);
        wav.finish();
        return "reopen for audio failed";
    }

    avcodec_flush_buffers (audioDec_);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<float> buf;

    auto drainFrames = [&]()
    {
        while (avcodec_receive_frame (audioDec_, frame) == 0)
        {
            const int maxOut = swr_get_out_samples (swr, frame->nb_samples);
            buf.resize ((size_t) maxOut * outChannels);
            uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
            const int got = swr_convert (swr, &outPtr, maxOut,
                                         (const uint8_t**) frame->extended_data,
                                         frame->nb_samples);
            if (got > 0) wav.write (buf.data(), (size_t) got);
        }
    };

    while (av_read_frame (afmt, pkt) >= 0)
    {
        if (pkt->stream_index == audioStream_)
        {
            if (avcodec_send_packet (audioDec_, pkt) == 0)
                drainFrames();
        }
        av_packet_unref (pkt);
    }
    avcodec_send_packet (audioDec_, nullptr);
    drainFrames();
    // flush resampler
    {
        const int maxOut = swr_get_out_samples (swr, 0);
        if (maxOut > 0)
        {
            buf.resize ((size_t) maxOut * outChannels);
            uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
            const int got = swr_convert (swr, &outPtr, maxOut, nullptr, 0);
            if (got > 0) wav.write (buf.data(), (size_t) got);
        }
    }

    const uint32_t frames = wav.dataBytes / (uint32_t) (outChannels * sizeof (float));
    wav.finish();
    durationSecOut = outRate > 0 ? (double) frames / outRate : 0.0;
    sampleRateOut = outRate;
    channelsOut = outChannels;

    av_frame_free (&frame);
    av_packet_free (&pkt);
    avformat_close_input (&afmt);
    swr_free (&swr);
    avcodec_flush_buffers (audioDec_);
    (void) st;
    return {};
}

// ============================================================================
// Beat detection (aubio)
// ============================================================================

std::string MediaContext::decodeAudioMono (int targetRate, std::vector<float>& monoOut)
{
    std::string error;
    if (! openAudioDecoder (error)) return error;

    SwrContext* swr = nullptr;
    AVChannelLayout monoLayout = AV_CHANNEL_LAYOUT_MONO;
    int rc = swr_alloc_set_opts2 (&swr, &monoLayout, AV_SAMPLE_FMT_FLT, targetRate,
                                  &audioDec_->ch_layout, audioDec_->sample_fmt,
                                  audioDec_->sample_rate, 0, nullptr);
    if (rc < 0 || swr_init (swr) < 0) return "resampler init failed";

    AVFormatContext* afmt = nullptr;
    if (avformat_open_input (&afmt, path_.c_str(), nullptr, nullptr) < 0
        || avformat_find_stream_info (afmt, nullptr) < 0)
    {
        if (afmt != nullptr) avformat_close_input (&afmt);
        swr_free (&swr);
        return "reopen for audio failed";
    }

    avcodec_flush_buffers (audioDec_);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<float> buf;

    auto drain = [&]()
    {
        while (avcodec_receive_frame (audioDec_, frame) == 0)
        {
            const int maxOut = swr_get_out_samples (swr, frame->nb_samples);
            buf.resize ((size_t) maxOut);
            uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
            const int got = swr_convert (swr, &outPtr, maxOut,
                                         (const uint8_t**) frame->extended_data,
                                         frame->nb_samples);
            if (got > 0) monoOut.insert (monoOut.end(), buf.begin(), buf.begin() + got);
        }
    };

    while (av_read_frame (afmt, pkt) >= 0)
    {
        if (pkt->stream_index == audioStream_)
            if (avcodec_send_packet (audioDec_, pkt) == 0)
                drain();
        av_packet_unref (pkt);
    }
    avcodec_send_packet (audioDec_, nullptr);
    drain();

    av_frame_free (&frame);
    av_packet_free (&pkt);
    avformat_close_input (&afmt);
    swr_free (&swr);
    avcodec_flush_buffers (audioDec_);
    return {};
}

std::string MediaContext::detectBeats (double& bpmOut, double& confidenceOut,
                                       std::vector<double>& beatTimesOut)
{
#if ARBIT_HAVE_AUBIO
    std::lock_guard<std::mutex> lock (mutex_);
    constexpr int rate = 44100;
    std::vector<float> mono;
    auto error = decodeAudioMono (rate, mono);
    if (! error.empty()) return error;
    if (mono.size() < (size_t) rate) return "audio too short for beat detection";

    const uint_t winSize = 1024, hopSize = 512;
    aubio_tempo_t* tempo = new_aubio_tempo ("default", winSize, hopSize, (uint_t) rate);
    if (tempo == nullptr) return "aubio tempo init failed";

    fvec_t* in = new_fvec (hopSize);
    fvec_t* out = new_fvec (1);

    for (size_t pos = 0; pos + hopSize <= mono.size(); pos += hopSize)
    {
        std::memcpy (in->data, mono.data() + pos, hopSize * sizeof (float));
        aubio_tempo_do (tempo, in, out);
        if (out->data[0] != 0.0f)
            beatTimesOut.push_back ((double) aubio_tempo_get_last_s (tempo));
    }

    bpmOut = (double) aubio_tempo_get_bpm (tempo);
    confidenceOut = (double) aubio_tempo_get_confidence (tempo);

    del_fvec (in);
    del_fvec (out);
    del_aubio_tempo (tempo);
    return {};
#else
    (void) bpmOut; (void) confidenceOut; (void) beatTimesOut;
    return "helper built without aubio (beat detection unavailable)";
#endif
}

// ============================================================================
// Stabilization — two-pass vid.stab (PROTOCOL.md §Stabilization)
// ============================================================================

bool MediaContext::vidstabAvailable()
{
    return avfilter_get_by_name ("vidstabdetect") != nullptr
        && avfilter_get_by_name ("vidstabtransform") != nullptr;
}

namespace
{
// Sequentially software-decodes path's video frames in [inSec, outSec) and
// pushes them through `filterName` (options applied via av_opt_set before
// init). Both vid.stab passes share this so their frame sequences are
// IDENTICAL — vid.stab indexes the .trf by frame count, not timestamps.
// When toYuv420p, a "format" filter is inserted before the sink (encoder
// input). onFrame (may be null) receives each filtered frame + its time base.
std::string runStabPass (const std::string& path, double inSec, double outSec,
                         const char* filterName,
                         const std::vector<std::pair<std::string, std::string>>& filterOpts,
                         bool toYuv420p,
                         const std::function<std::string (AVFrame*, AVRational)>& onFrame,
                         int& framesOut)
{
    framesOut = 0;

    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVFilterGraph* graph = nullptr;
    AVFilterContext* srcCtx = nullptr;
    AVFilterContext* sinkCtx = nullptr;
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* filtered = av_frame_alloc();
    std::string error;

    auto fail = [&] (std::string msg)
    {
        error = std::move (msg);
        return false;
    };

    do
    {
        if (avformat_open_input (&fmt, path.c_str(), nullptr, nullptr) < 0
            || avformat_find_stream_info (fmt, nullptr) < 0)
        {
            fail ("open failed: " + path);
            break;
        }
        const int vIdx = av_find_best_stream (fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vIdx < 0) { fail ("no video stream"); break; }
        AVStream* st = fmt->streams[vIdx];

        const AVCodec* codec = avcodec_find_decoder (st->codecpar->codec_id);
        if (codec == nullptr) { fail ("no video decoder for codec"); break; }
        dec = avcodec_alloc_context3 (codec);
        avcodec_parameters_to_context (dec, st->codecpar);
        dec->thread_count = 0;
        if (avcodec_open2 (dec, codec, nullptr) < 0) { fail ("video decoder open failed"); break; }

        double durationSec = 0.0;
        if (fmt->duration != AV_NOPTS_VALUE)
            durationSec = (double) fmt->duration / AV_TIME_BASE;
        else if (st->duration != AV_NOPTS_VALUE)
            durationSec = streamTimeToSec (st, st->duration);
        inSec = std::max (0.0, inSec);
        if (outSec <= 0.0 || (durationSec > 0.0 && outSec > durationSec))
            outSec = durationSec > 0.0 ? durationSec : 1.0e9;

        if (inSec > 0.0)
        {
            const int64_t ts = av_rescale_q ((int64_t) std::llround (inSec * AV_TIME_BASE),
                                             AVRational { 1, AV_TIME_BASE }, st->time_base);
            if (av_seek_frame (fmt, vIdx, ts, AVSEEK_FLAG_BACKWARD) < 0)
            {
                fail ("seek failed");
                break;
            }
        }

        // Graph is built lazily from the first decoded frame so the buffer
        // source sees the true output format (10-bit, deinterlaced, ...).
        // Format negotiation auto-inserts conversions where needed.
        auto buildGraph = [&] (const AVFrame* first) -> std::string
        {
            const AVFilter* bufFlt = avfilter_get_by_name ("buffer");
            const AVFilter* sinkFlt = avfilter_get_by_name ("buffersink");
            const AVFilter* stabFlt = avfilter_get_by_name (filterName);
            if (stabFlt == nullptr)
                return std::string (filterName) + " unavailable (libavfilter built without vid.stab)";
            if (bufFlt == nullptr || sinkFlt == nullptr)
                return "buffer/buffersink filters unavailable";

            graph = avfilter_graph_alloc();
            if (graph == nullptr) return "filter graph alloc failed";

            AVRational sar = first->sample_aspect_ratio;
            if (sar.num == 0) sar = { 1, 1 };
            char args[256];
            std::snprintf (args, sizeof (args),
                           "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
                           first->width, first->height, first->format,
                           st->time_base.num, st->time_base.den, sar.num, sar.den);
            if (avfilter_graph_create_filter (&srcCtx, bufFlt, "in", args, nullptr, graph) < 0)
                return "buffer source create failed";

            AVFilterContext* stabCtx = avfilter_graph_alloc_filter (graph, stabFlt, "stab");
            if (stabCtx == nullptr) return "stab filter alloc failed";
            for (const auto& opt : filterOpts)
                if (av_opt_set (stabCtx, opt.first.c_str(), opt.second.c_str(),
                                AV_OPT_SEARCH_CHILDREN) < 0)
                    return "cannot set " + std::string (filterName) + " option "
                         + opt.first + "=" + opt.second;
            if (avfilter_init_str (stabCtx, nullptr) < 0)
                return std::string (filterName) + " init failed";

            AVFilterContext* tail = stabCtx;
            if (toYuv420p)
            {
                const AVFilter* fmtFlt = avfilter_get_by_name ("format");
                if (fmtFlt == nullptr) return "format filter unavailable";
                AVFilterContext* fmtCtx = avfilter_graph_alloc_filter (graph, fmtFlt, "tofmt");
                if (fmtCtx == nullptr
                    || av_opt_set (fmtCtx, "pix_fmts", "yuv420p", AV_OPT_SEARCH_CHILDREN) < 0
                    || avfilter_init_str (fmtCtx, nullptr) < 0)
                    return "format filter setup failed";
                if (avfilter_link (stabCtx, 0, fmtCtx, 0) < 0)
                    return "filter link failed";
                tail = fmtCtx;
            }

            if (avfilter_graph_create_filter (&sinkCtx, sinkFlt, "out", nullptr, nullptr, graph) < 0)
                return "buffer sink create failed";
            if (avfilter_link (srcCtx, 0, stabCtx, 0) < 0
                || avfilter_link (tail, 0, sinkCtx, 0) < 0)
                return "filter link failed";
            if (avfilter_graph_config (graph, nullptr) < 0)
                return "filter graph config failed";
            return {};
        };

        auto drainSink = [&]() -> std::string
        {
            while (true)
            {
                const int rc = av_buffersink_get_frame (sinkCtx, filtered);
                if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) return {};
                if (rc < 0) return "buffersink read failed: " + avErr (rc);
                ++framesOut;
                std::string e;
                if (onFrame != nullptr)
                    e = onFrame (filtered, av_buffersink_get_time_base (sinkCtx));
                av_frame_unref (filtered);
                if (! e.empty()) return e;
            }
        };

        bool rangeDone = false;
        bool eofInput = false;
        while (! rangeDone && error.empty())
        {
            int rc = av_read_frame (fmt, pkt);
            if (rc == AVERROR_EOF)
            {
                avcodec_send_packet (dec, nullptr);
                eofInput = true;
            }
            else if (rc < 0) { fail ("read frame failed: " + avErr (rc)); break; }
            else if (pkt->stream_index != vIdx) { av_packet_unref (pkt); continue; }
            else
            {
                rc = avcodec_send_packet (dec, pkt);
                av_packet_unref (pkt);
                if (rc < 0 && rc != AVERROR (EAGAIN))
                {
                    fail ("send packet failed: " + avErr (rc));
                    break;
                }
            }

            while (error.empty())
            {
                rc = avcodec_receive_frame (dec, frame);
                if (rc == AVERROR (EAGAIN)) break;
                if (rc == AVERROR_EOF) { rangeDone = true; break; }
                if (rc < 0) { fail ("decode failed: " + avErr (rc)); break; }

                const double ptsSec = frame->best_effort_timestamp == AV_NOPTS_VALUE
                                          ? 0.0
                                          : streamTimeToSec (st, frame->best_effort_timestamp);
                if (ptsSec < inSec - 1.0e-6)        // pre-roll from keyframe seek
                {
                    av_frame_unref (frame);
                    continue;
                }
                if (ptsSec >= outSec - 1.0e-6)
                {
                    av_frame_unref (frame);
                    rangeDone = true;
                    break;
                }
                frame->pts = frame->best_effort_timestamp;
                if (graph == nullptr)
                    error = buildGraph (frame);
                if (error.empty())
                {
                    if (av_buffersrc_add_frame_flags (srcCtx, frame,
                                                      AV_BUFFERSRC_FLAG_KEEP_REF) < 0)
                        fail ("buffersrc push failed");
                    else
                        error = drainSink();
                }
                av_frame_unref (frame);
            }
            if (eofInput) break;
        }

        // Flush: vidstabdetect finalizes the .trf when the graph closes;
        // vidstabtransform may hold buffered frames.
        if (error.empty() && graph != nullptr)
        {
            if (av_buffersrc_add_frame (srcCtx, nullptr) < 0)
                fail ("buffersrc flush failed");
            else
                error = drainSink();
        }
    } while (false);

    if (graph != nullptr) avfilter_graph_free (&graph);
    if (dec != nullptr) avcodec_free_context (&dec);
    if (fmt != nullptr) avformat_close_input (&fmt);
    av_frame_free (&filtered);
    av_frame_free (&frame);
    av_packet_free (&pkt);
    return error;
}
} // namespace

std::string MediaContext::stabilizeDetect (const std::string& trfPath, double inSec,
                                           double outSec, int& framesOut)
{
    std::lock_guard<std::mutex> lock (mutex_);
    if (! info_.hasVideo) return "media has no video stream";
    if (! vidstabAvailable()) return "libavfilter lacks vid.stab filters";

    auto error = runStabPass (path_, inSec, outSec, "vidstabdetect",
                              { { "result", trfPath }, { "shakiness", "5" } },
                              false, nullptr, framesOut);
    if (error.empty() && framesOut == 0)
        error = "no frames in range";
    if (! error.empty())
        std::remove (trfPath.c_str());
    return error;
}

std::string MediaContext::stabilizeRender (const std::string& trfPath, const std::string& outPath,
                                           double inSec, double outSec, double strength,
                                           int& framesOut)
{
    std::lock_guard<std::mutex> lock (mutex_);
    if (! info_.hasVideo) return "media has no video stream";
    if (! vidstabAvailable()) return "libavfilter lacks vid.stab filters";

    strength = std::clamp (strength, 0.0, 1.0);
    const int smoothing = std::max (1, (int) std::lround (strength * 60.0));

    AVFormatContext* ofmt = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* ostream = nullptr;
    AVPacket* opkt = av_packet_alloc();
    bool headerWritten = false;
    const AVRational encTb { 1, 90000 };
    const int64_t inOffset = (int64_t) std::llround (std::max (0.0, inSec) * encTb.den);

    auto drainEncoder = [&]() -> std::string
    {
        while (true)
        {
            const int rc = avcodec_receive_packet (enc, opkt);
            if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) return {};
            if (rc < 0) return "encode receive failed: " + avErr (rc);
            opkt->stream_index = ostream->index;
            av_packet_rescale_ts (opkt, encTb, ostream->time_base);
            if (av_interleaved_write_frame (ofmt, opkt) < 0)
                return "write frame failed";
        }
    };

    auto onFrame = [&] (AVFrame* f, AVRational tb) -> std::string
    {
        if (enc == nullptr)
        {
            const AVCodec* vCodec = avcodec_find_encoder_by_name ("libx264");
            if (vCodec == nullptr) vCodec = avcodec_find_encoder (AV_CODEC_ID_H264);
            if (vCodec == nullptr) vCodec = avcodec_find_encoder (AV_CODEC_ID_MPEG4);
            if (vCodec == nullptr) return "no h264/mpeg4 encoder available";

            if (avformat_alloc_output_context2 (&ofmt, nullptr, "mp4", outPath.c_str()) < 0
                || ofmt == nullptr)
                return "cannot create output context for " + outPath;

            enc = avcodec_alloc_context3 (vCodec);
            enc->width = f->width;
            enc->height = f->height;
            enc->pix_fmt = (AVPixelFormat) f->format; // yuv420p (format filter)
            enc->time_base = encTb;
            const double fps = info_.fps > 0.0 ? info_.fps : 30.0;
            enc->framerate = av_d2q (fps, 100000);
            enc->gop_size = (int) std::lround (fps * 2.0);
            if (ofmt->oformat->flags & AVFMT_GLOBALHEADER)
                enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            if (std::strcmp (vCodec->name, "libx264") == 0)
            {
                av_opt_set (enc->priv_data, "preset", "veryfast", 0);
                av_opt_set (enc->priv_data, "crf", "18", 0);
            }
            else
                enc->bit_rate = (int64_t) f->width * f->height * 8;
            if (avcodec_open2 (enc, vCodec, nullptr) < 0)
                return "cannot open stabilize encoder";

            ostream = avformat_new_stream (ofmt, nullptr);
            if (ostream == nullptr) return "cannot create output stream";
            avcodec_parameters_from_context (ostream->codecpar, enc);
            ostream->time_base = encTb;

            if (! (ofmt->oformat->flags & AVFMT_NOFILE)
                && avio_open (&ofmt->pb, outPath.c_str(), AVIO_FLAG_WRITE) < 0)
                return "cannot open output file " + outPath;
            if (avformat_write_header (ofmt, nullptr) < 0)
                return "write_header failed";
            headerWritten = true;
        }

        // Intermediate timeline = source time minus inSec (the plugin maps
        // segment times the same way — parity by construction).
        int64_t pts = f->pts == AV_NOPTS_VALUE ? 0 : av_rescale_q (f->pts, tb, encTb);
        f->pts = std::max<int64_t> (0, pts - inOffset);
        f->duration = 0;
        f->pict_type = AV_PICTURE_TYPE_NONE;
        const int rc = avcodec_send_frame (enc, f);
        if (rc < 0) return "encode failed: " + avErr (rc);
        return drainEncoder();
    };

    int frames = 0;
    auto error = runStabPass (path_, inSec, outSec, "vidstabtransform",
                              { { "input", trfPath },
                                { "smoothing", std::to_string (smoothing) },
                                { "optzoom", "1" } },
                              true, onFrame, frames);
    framesOut = frames;

    if (error.empty() && enc == nullptr)
        error = "no frames in range";
    if (error.empty())
    {
        avcodec_send_frame (enc, nullptr); // flush
        error = drainEncoder();
    }
    if (error.empty() && av_write_trailer (ofmt) < 0)
        error = "write_trailer failed";

    if (enc != nullptr) avcodec_free_context (&enc);
    if (ofmt != nullptr)
    {
        if (headerWritten && ! (ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb != nullptr)
            avio_closep (&ofmt->pb);
        avformat_free_context (ofmt);
    }
    av_packet_free (&opkt);
    if (! error.empty())
        std::remove (outPath.c_str());
    return error;
}

// ============================================================================
// Edit proxies — intra-only H.264 transcode (PROTOCOL.md §Proxy media)
// ============================================================================

std::string MediaContext::generateProxy (const std::string& sourcePath,
                                         const std::string& outPath,
                                         double scale, ProxyProgress* progress)
{
    scale = std::max (0.1, std::min (1.0, scale)); // 10%..100% of source
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVFormatContext* ofmt = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* ostream = nullptr;
    SwsContext* sws = nullptr;
    AVPacket* pkt = av_packet_alloc();
    AVPacket* opkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* yuv = av_frame_alloc();
    bool headerWritten = false;
    const AVRational encTb { 1, 90000 };
    std::string error;

    auto fail = [&] (std::string msg) { error = std::move (msg); };

    // Encode to a temp sibling and atomically rename on success. The consumer
    // (videoBuildSegments) keys proxy substitution off existsAsFile(), so a
    // half-written file at the FINAL path is referenced and opened mid-encode —
    // before the mp4 moov atom is written — which fails and permanently blacks
    // the clip (the viewport latches openFailed). Publishing only a complete
    // file via rename closes that race.
    const std::string tmpPath = outPath + ".part";

    auto drainEncoder = [&]() -> std::string
    {
        while (true)
        {
            const int rc = avcodec_receive_packet (enc, opkt);
            if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) return {};
            if (rc < 0) return "encode receive failed: " + avErr (rc);
            opkt->stream_index = ostream->index;
            av_packet_rescale_ts (opkt, encTb, ostream->time_base);
            if (av_interleaved_write_frame (ofmt, opkt) < 0)
                return "write frame failed";
        }
    };

    do
    {
        if (avformat_open_input (&fmt, sourcePath.c_str(), nullptr, nullptr) < 0
            || avformat_find_stream_info (fmt, nullptr) < 0)
        {
            fail ("open failed: " + sourcePath);
            break;
        }
        const int vIdx = av_find_best_stream (fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vIdx < 0) { fail ("no video stream"); break; }
        AVStream* st = fmt->streams[vIdx];

        // Software decode only: the proxy worker runs alongside the viewport's
        // GL (+ possibly NVDEC) sessions — same rationale as the exporter.
        const AVCodec* codec = avcodec_find_decoder (st->codecpar->codec_id);
        if (codec == nullptr) { fail ("no video decoder for codec"); break; }
        dec = avcodec_alloc_context3 (codec);
        avcodec_parameters_to_context (dec, st->codecpar);
        dec->thread_count = 0;
        if (avcodec_open2 (dec, codec, nullptr) < 0) { fail ("video decoder open failed"); break; }

        auto fr = av_guess_frame_rate (fmt, st, nullptr);
        const double fps = fr.den != 0 ? (double) fr.num / fr.den : 30.0;
        double durationSec = 0.0;
        if (fmt->duration != AV_NOPTS_VALUE)
            durationSec = (double) fmt->duration / AV_TIME_BASE;
        else if (st->duration != AV_NOPTS_VALUE)
            durationSec = streamTimeToSec (st, st->duration);
        if (progress != nullptr)
            progress->totalFrames.store ((int) std::lround (durationSec * fps));

        // yuv420p needs even dimensions; scale (0.1..1.0) multiplies them (min 2).
        const int srcW = dec->width, srcH = dec->height;
        const int dstW = std::max (2, (int) std::lround (srcW * scale) & ~1);
        const int dstH = std::max (2, (int) std::lround (srcH * scale) & ~1);

        const AVCodec* vCodec = avcodec_find_encoder_by_name ("libx264");
        if (vCodec == nullptr) vCodec = avcodec_find_encoder (AV_CODEC_ID_H264);
        if (vCodec == nullptr) { fail ("no h264 encoder available"); break; }

        if (avformat_alloc_output_context2 (&ofmt, nullptr, "mp4", tmpPath.c_str()) < 0
            || ofmt == nullptr)
        {
            fail ("cannot create output context for " + outPath);
            break;
        }

        enc = avcodec_alloc_context3 (vCodec);
        enc->width = dstW;
        enc->height = dstH;
        enc->pix_fmt = AV_PIX_FMT_YUV420P;
        enc->time_base = encTb;
        enc->framerate = av_d2q (fps, 100000);
        enc->gop_size = 1;       // every frame a keyframe: intra-only
        enc->max_b_frames = 0;   // no reordering — frame-independent seeks
        if (ofmt->oformat->flags & AVFMT_GLOBALHEADER)
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        if (std::strcmp (vCodec->name, "libx264") == 0)
        {
            av_opt_set (enc->priv_data, "preset", "veryfast", 0);
            // Downscaled proxies tolerate a slightly higher crf (smaller files);
            // a native-resolution proxy stays near-visually-lossless.
            av_opt_set (enc->priv_data, "crf", scale < 0.999 ? "20" : "18", 0);
        }
        else
            enc->bit_rate = (int64_t) dstW * dstH * 8;
        if (avcodec_open2 (enc, vCodec, nullptr) < 0)
        {
            fail ("cannot open proxy encoder");
            break;
        }

        ostream = avformat_new_stream (ofmt, nullptr);
        if (ostream == nullptr) { fail ("cannot create output stream"); break; }
        avcodec_parameters_from_context (ostream->codecpar, enc);
        ostream->time_base = encTb;

        if (! (ofmt->oformat->flags & AVFMT_NOFILE)
            && avio_open (&ofmt->pb, tmpPath.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            fail ("cannot open output file " + outPath);
            break;
        }
        if (avformat_write_header (ofmt, nullptr) < 0) { fail ("write_header failed"); break; }
        headerWritten = true;

        yuv->format = AV_PIX_FMT_YUV420P;
        yuv->width = dstW;
        yuv->height = dstH;
        if (av_frame_get_buffer (yuv, 0) < 0) { fail ("frame alloc failed"); break; }

        auto encodeOne = [&] (AVFrame* in) -> std::string
        {
            sws = sws_getCachedContext (sws, in->width, in->height,
                                        (AVPixelFormat) in->format,
                                        dstW, dstH, AV_PIX_FMT_YUV420P,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (sws == nullptr) return "swscale context failed";
            if (av_frame_make_writable (yuv) < 0) return "frame not writable";
            sws_scale (sws, in->data, in->linesize, 0, in->height, yuv->data, yuv->linesize);

            // Preserve source timing: the proxy is a 1:1 timeline stand-in.
            const int64_t pts = in->best_effort_timestamp == AV_NOPTS_VALUE
                                    ? 0
                                    : av_rescale_q (in->best_effort_timestamp,
                                                    st->time_base, encTb);
            yuv->pts = std::max<int64_t> (0, pts);
            yuv->pict_type = AV_PICTURE_TYPE_NONE;
            const int rc = avcodec_send_frame (enc, yuv);
            if (rc < 0) return "encode failed: " + avErr (rc);
            if (progress != nullptr)
                progress->frame.fetch_add (1);
            return drainEncoder();
        };

        bool eof = false;
        while (! eof && error.empty())
        {
            if (progress != nullptr && progress->abort.load())
            {
                fail ("cancelled");
                break;
            }
            int rc = av_read_frame (fmt, pkt);
            if (rc == AVERROR_EOF)
            {
                avcodec_send_packet (dec, nullptr);
                eof = true;
            }
            else if (rc < 0) { fail ("read frame failed: " + avErr (rc)); break; }
            else if (pkt->stream_index != vIdx) { av_packet_unref (pkt); continue; }
            else
            {
                rc = avcodec_send_packet (dec, pkt);
                av_packet_unref (pkt);
                if (rc < 0 && rc != AVERROR (EAGAIN))
                {
                    fail ("send packet failed: " + avErr (rc));
                    break;
                }
            }

            while (error.empty())
            {
                rc = avcodec_receive_frame (dec, frame);
                if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) break;
                if (rc < 0) { fail ("decode failed: " + avErr (rc)); break; }
                error = encodeOne (frame);
                av_frame_unref (frame);
            }
        }

        if (error.empty())
        {
            avcodec_send_frame (enc, nullptr); // flush encoder
            error = drainEncoder();
        }
        if (error.empty() && av_write_trailer (ofmt) < 0)
            error = "write_trailer failed";
    } while (false);

    if (sws != nullptr) sws_freeContext (sws);
    if (enc != nullptr) avcodec_free_context (&enc);
    if (ofmt != nullptr)
    {
        if (headerWritten && ! (ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb != nullptr)
            avio_closep (&ofmt->pb);
        avformat_free_context (ofmt);
    }
    if (dec != nullptr) avcodec_free_context (&dec);
    if (fmt != nullptr) avformat_close_input (&fmt);
    av_frame_free (&yuv);
    av_frame_free (&frame);
    av_packet_free (&opkt);
    av_packet_free (&pkt);
    // Publish atomically: a complete file appears at outPath in one step, or
    // not at all. A failed/cancelled run leaves no partial file behind.
    if (! error.empty())
        std::remove (tmpPath.c_str());
    else if (std::rename (tmpPath.c_str(), outPath.c_str()) != 0)
    {
        std::remove (tmpPath.c_str());
        error = "could not publish proxy: " + outPath;
    }
    return error;
}

// ============================================================================
// Thumbnails (JPEG via libavcodec mjpeg encoder)
// ============================================================================

std::string MediaContext::writeThumbnails (const std::vector<double>& times, int maxW, int maxH,
                                           const std::string& outDir, const std::string& baseName,
                                           std::vector<std::string>& outPaths)
{
    const AVCodec* jpeg = avcodec_find_encoder (AV_CODEC_ID_MJPEG);
    if (jpeg == nullptr) return "mjpeg encoder unavailable";

    for (size_t i = 0; i < times.size(); ++i)
    {
        DecodedFrame df;
        auto error = getFrame (times[i], maxW, maxH, df);
        if (! error.empty()) return error;

        AVCodecContext* enc = avcodec_alloc_context3 (jpeg);
        enc->width = df.width;
        enc->height = df.height;
        enc->pix_fmt = AV_PIX_FMT_YUVJ420P;
        enc->time_base = { 1, 25 };
        enc->flags |= AV_CODEC_FLAG_QSCALE;
        enc->global_quality = FF_QP2LAMBDA * 6;
        if (avcodec_open2 (enc, jpeg, nullptr) < 0)
        {
            avcodec_free_context (&enc);
            return "jpeg encoder open failed";
        }

        AVFrame* yuv = av_frame_alloc();
        yuv->format = AV_PIX_FMT_YUVJ420P;
        yuv->width = df.width;
        yuv->height = df.height;
        av_frame_get_buffer (yuv, 0);

        SwsContext* toYuv = sws_getContext (df.width, df.height, AV_PIX_FMT_RGBA,
                                            df.width, df.height, AV_PIX_FMT_YUVJ420P,
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);
        const uint8_t* srcData[4] = { df.rgba.data(), nullptr, nullptr, nullptr };
        int srcLines[4] = { df.strideBytes, 0, 0, 0 };
        sws_scale (toYuv, srcData, srcLines, 0, df.height, yuv->data, yuv->linesize);
        sws_freeContext (toYuv);

        AVPacket* pkt = av_packet_alloc();
        std::string outPath = outDir + "/" + baseName + "-" + std::to_string (i) + ".jpg";
        bool ok = avcodec_send_frame (enc, yuv) == 0
               && avcodec_receive_packet (enc, pkt) == 0;
        if (ok)
        {
            FILE* f = std::fopen (outPath.c_str(), "wb");
            if (f != nullptr)
            {
                std::fwrite (pkt->data, 1, (size_t) pkt->size, f);
                std::fclose (f);
                outPaths.push_back (outPath);
            }
            else ok = false;
        }

        av_packet_free (&pkt);
        av_frame_free (&yuv);
        avcodec_free_context (&enc);
        if (! ok) return "thumbnail encode/write failed: " + outPath;
    }
    return {};
}
