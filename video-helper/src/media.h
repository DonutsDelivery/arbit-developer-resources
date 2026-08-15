#pragma once

// MediaContext — persistent libav decoder for one media file.
// Owns format/codec contexts so repeated frame requests (scrubbing, playback)
// reuse open streams instead of re-probing the file.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

struct MediaInfo
{
    double durationSec = 0.0;
    double fps = 0.0;
    int width = 0;
    int height = 0;
    bool hasVideo = false;
    bool hasAudio = false;
    bool hasAlpha = false; // source pix_fmt carries alpha (yuva*, rgba, ...)
                           // or webm/mkv alpha_mode side-channel
    int audioSampleRate = 0;
    int audioChannels = 0;
    std::string container;
    std::string videoCodec;
    std::string audioCodec;
    std::string hwaccel; // active hardware decode backend ("" = software)
};

struct DecodedFrame
{
    std::vector<uint8_t> rgba; // tightly packed, strideBytes per row
    int width = 0;
    int height = 0;
    int strideBytes = 0;
    double ptsSec = 0.0;
};

// Progress/abort shared between the async proxy job (main.cpp) and the
// transcode loop (PROTOCOL.md §Proxy media).
struct ProxyProgress
{
    std::atomic<int> frame { 0 };
    std::atomic<int> totalFrames { 0 };   // estimate (duration × fps)
    std::atomic<bool> abort { false };
};

// One process-wide CUDA hardware-decode device context, shared by every
// MediaContext decode session (NVIDIA's guidance: allocate the minimum number
// of CUDA contexts and share them across sessions). Lazily created under
// std::call_once and returned as a borrowed pointer (callers av_buffer_ref it);
// nullptr when CUDA is unavailable or ARBIT_DISABLE_HWDEC is set, in which case
// callers fall back to software decode.
//
// IMPORTANT: main() calls this ONCE at startup, before any worker/render thread
// spawns, so the CUDA init runs single-threaded. Initializing CUDA concurrently
// from two threads (the viewport render-thread decoder + the InterpEngine
// worker) is what previously deadlocked the NVIDIA driver and motivated the
// software-only fallback — that was an app-side concurrent-init bug, not a
// hardware decode-session limit (GeForce NVDEC has none).
AVBufferRef* sharedCudaDeviceCtx();

class MediaContext
{
public:
    ~MediaContext();

    // Opens the file and the video decoder (audio decoder is opened lazily).
    // Returns empty string on success, error message on failure.
    // allowHwDecode = false forces a software decoder — the offline exporter
    // uses this so per-clip decode sessions never create CUDA/VAAPI contexts
    // alongside its GL context (multiple NVDEC sessions + GL in one process
    // intermittently deadlock inside the NVIDIA driver).
    // Alpha sources always decode in software (hw surfaces are NV12/P010 —
    // the alpha plane would be dropped on download).
    // Image sequences: a path containing a printf pattern ("img_%04d.png")
    // opens through the image2 demuxer. seqFps sets the sequence frame rate
    // (<= 0 = 30), seqStartNumber the first frame number (< 0 = ffmpeg's
    // default 0..4 scan). Both are ignored for non-pattern paths.
    std::string open (const std::string& path, bool allowHwDecode = true,
                      double seqFps = 0.0, int seqStartNumber = -1);

    const MediaInfo& info() const { return info_; }
    const std::string& path() const { return path_; }

    // Decode the frame displayed at timeSec, scaled to fit maxW x maxH
    // (aspect preserved). Seeks backward + decodes forward when needed;
    // decodes forward directly for sequential requests.
    std::string getFrame (double timeSec, int maxW, int maxH, DecodedFrame& out);

    // Decode the entire audio stream to a float32 WAV at outPath.
    // Returns empty string on success.
    std::string extractAudio (const std::string& outPath, double& durationSecOut,
                              int& sampleRateOut, int& channelsOut);

    // Two-pass vid.stab stabilization (PROTOCOL.md §Stabilization).
    // True when this libavfilter build has vidstabdetect/vidstabtransform.
    static bool vidstabAvailable();

    // Pass 1: sequential software decode of [inSec, outSec) through
    // vidstabdetect, writing per-frame motion transforms to trfPath.
    // outSec <= 0 means "to end of stream".
    std::string stabilizeDetect (const std::string& trfPath, double inSec, double outSec,
                                 int& framesOut);

    // Pass 2: the SAME frame range (vid.stab indexes the .trf by frame
    // count, so both passes must see identical sequences) through
    // vidstabtransform (input=trfPath), h264-encoded to outPath. The
    // intermediate's timeline equals source time minus inSec. strength
    // 0..1 maps to vidstabtransform smoothing 1..60.
    std::string stabilizeRender (const std::string& trfPath, const std::string& outPath,
                                 double inSec, double outSec, double strength,
                                 int& framesOut);

    // Decode audio to mono float 44.1k and run aubio tempo tracking.
    std::string detectBeats (double& bpmOut, double& confidenceOut,
                             std::vector<double>& beatTimesOut);

    // Edit-proxy transcode (PROTOCOL.md §Proxy media): re-encode sourcePath's
    // video stream as intra-only (all-keyframe) H.264 yuv420p mp4 at outPath
    // so every frame is independently seekable. scale (0.1..1.0) multiplies the
    // source dimensions (e.g. 0.5 = half-res; 1.0 = native). Video-only output;
    // source pts are preserved so the proxy is a 1:1 timeline stand-in. Static —
    // opens its own software decode/encode contexts so a long transcode never
    // holds an open mediaId's mutex (request_frame stays responsive). Blocking;
    // run on a worker thread (main.cpp's async proxy job). progress may be null.
    static std::string generateProxy (const std::string& sourcePath,
                                      const std::string& outPath,
                                      double scale, ProxyProgress* progress);

    // Decode frames at the given times and write JPEG thumbnails.
    // Returns empty string on success; fills outPaths.
    std::string writeThumbnails (const std::vector<double>& times, int maxW, int maxH,
                                 const std::string& outDir, const std::string& baseName,
                                 std::vector<std::string>& outPaths);

private:
    std::string decodeForwardUntil (double targetSec, AVFrame* frame, bool& gotFrame);
    std::string scaleToRgba (AVFrame* frame, int maxW, int maxH, DecodedFrame& out);
    std::string decodeAudioMono (int targetRate, std::vector<float>& monoOut);
    bool openAudioDecoder (std::string& error);

    bool tryOpenHwDecoder (const AVCodec* codec);
    void downloadIfHw (AVFrame* frame);

    std::string path_;
    MediaInfo info_;
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* videoDec_ = nullptr;
    AVBufferRef* hwDeviceCtx_ = nullptr;   // non-null when hardware decode is active
    AVPixelFormat hwPixFmt_ = AV_PIX_FMT_NONE;
    AVFrame* hwTransferFrame_ = nullptr;   // staging frame for hw->sw download
    AVCodecContext* audioDec_ = nullptr;
    SwsContext* sws_ = nullptr;
    int videoStream_ = -1;
    int audioStream_ = -1;
    double lastDecodedPts_ = -1.0e9;
    AVFrame* lastFrame_ = nullptr;   // most recently decoded video frame
    std::mutex mutex_;
};
