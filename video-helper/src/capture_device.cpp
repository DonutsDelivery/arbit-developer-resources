#include "capture_device.h"
#include "exporter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>

extern "C"
{
#include <libavdevice/avdevice.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace
{
std::string ffmpegError (int errorCode)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE] {};
    av_strerror (errorCode, buffer, sizeof (buffer));
    return buffer;
}

void appendDevices (const char* inputFormatName, const char* kind,
                    std::vector<CaptureDeviceInfo>& devices, std::string& error)
{
    const AVInputFormat* inputFormat = av_find_input_format (inputFormatName);
    if (inputFormat == nullptr)
        return;

    AVDeviceInfoList* list = nullptr;
    const int result = avdevice_list_input_sources (inputFormat, nullptr, nullptr, &list);
    if (result < 0)
    {
        if (error.empty())
            error = ffmpegError (result);
        return;
    }

    for (int index = 0; index < list->nb_devices; ++index)
    {
        const auto* device = list->devices[index];
        if (device == nullptr || device->device_name == nullptr)
            continue;
        devices.push_back ({ device->device_name,
                             device->device_description != nullptr ? device->device_description : device->device_name,
                             kind, inputFormatName });
    }
    avdevice_free_list_devices (&list);
}
} // namespace

std::vector<CaptureDeviceInfo> listCaptureDevices (const std::string& kind, std::string& error)
{
    error.clear();
    std::vector<CaptureDeviceInfo> devices;
    static std::once_flag registerOnce;
    std::call_once (registerOnce, [] { avdevice_register_all(); });

    if (kind == "test")
    {
        devices.push_back ({ "testsrc2=size=640x360:rate=30", "FFmpeg Test Pattern", "test", "lavfi" });
        return devices;
    }

#if defined(__linux__)
    if (kind.empty() || kind == "camera")
        appendDevices ("v4l2", "camera", devices, error);
#elif defined(_WIN32)
    if (kind.empty() || kind == "camera")
        appendDevices ("dshow", "camera", devices, error);
#elif defined(__APPLE__)
    if (kind.empty() || kind == "camera")
        appendDevices ("avfoundation", "camera", devices, error);
#else
    (void) kind;
#endif
    return devices;
}

struct CaptureDeviceSession::Impl
{
    AVFormatContext* format = nullptr;
    AVCodecContext* decoder = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* scaler = nullptr;
    int streamIndex = -1;
    int width = 0;
    int height = 0;
    double fps = 30.0;
    bool paceInput = false;
    FrameCallback callback;
    std::mutex recordingsMutex;
    std::map<uint32_t, std::unique_ptr<RecorderSession>> recordings;
    std::map<uint32_t, std::string> outputPaths;

    ~Impl()
    {
        if (scaler != nullptr) sws_freeContext(scaler);
        if (packet != nullptr) av_packet_free(&packet);
        if (frame != nullptr) av_frame_free(&frame);
        if (decoder != nullptr) avcodec_free_context(&decoder);
        if (format != nullptr) avformat_close_input(&format);
    }
};

CaptureDeviceSession::CaptureDeviceSession() = default;
CaptureDeviceSession::~CaptureDeviceSession() { close(); }

std::string CaptureDeviceSession::open(const std::string& sourceId, const std::string& backend,
                                       int width, int height, double fps, FrameCallback callback)
{
    close();
    impl_ = std::make_unique<Impl>();
    impl_->width = std::max(16, width);
    impl_->height = std::max(16, height);
    impl_->fps = fps > 0.0 ? fps : 30.0;
    impl_->paceInput = backend == "lavfi";
    impl_->callback = std::move(callback);

    const AVInputFormat* input = av_find_input_format(backend.c_str());
    if (input == nullptr)
        return "capture backend unavailable: " + backend;

    impl_->format = avformat_alloc_context();
    if (impl_->format == nullptr)
        return "could not allocate capture format context";
    impl_->format->interrupt_callback.callback = [] (void* opaque) -> int {
        return static_cast<CaptureDeviceSession*>(opaque)->running_.load(std::memory_order_acquire) ? 0 : 1;
    };
    impl_->format->interrupt_callback.opaque = this;

    AVDictionary* options = nullptr;
    const auto sizeText = std::to_string(impl_->width) + "x" + std::to_string(impl_->height);
    const auto fpsText = std::to_string(impl_->fps);
    if (backend != "lavfi")
    {
        av_dict_set(&options, "video_size", sizeText.c_str(), 0);
        av_dict_set(&options, "framerate", fpsText.c_str(), 0);
    }
    running_.store(true, std::memory_order_release);
    int result = avformat_open_input(&impl_->format, sourceId.c_str(), input, &options);
    av_dict_free(&options);
    if (result < 0)
    {
        running_.store(false, std::memory_order_release);
        return "capture open failed: " + ffmpegError(result);
    }
    result = avformat_find_stream_info(impl_->format, nullptr);
    if (result < 0)
    {
        running_.store(false, std::memory_order_release);
        return "capture stream probe failed: " + ffmpegError(result);
    }
    impl_->streamIndex = av_find_best_stream(impl_->format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (impl_->streamIndex < 0)
    {
        running_.store(false, std::memory_order_release);
        return "capture source has no video stream";
    }
    auto* parameters = impl_->format->streams[impl_->streamIndex]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(parameters->codec_id);
    if (codec == nullptr)
    {
        running_.store(false, std::memory_order_release);
        return "capture decoder unavailable";
    }
    impl_->decoder = avcodec_alloc_context3(codec);
    if (impl_->decoder == nullptr || avcodec_parameters_to_context(impl_->decoder, parameters) < 0
        || avcodec_open2(impl_->decoder, codec, nullptr) < 0)
    {
        running_.store(false, std::memory_order_release);
        return "could not open capture decoder";
    }
    impl_->frame = av_frame_alloc();
    impl_->packet = av_packet_alloc();
    impl_->scaler = sws_getContext(impl_->decoder->width, impl_->decoder->height, impl_->decoder->pix_fmt,
                                   impl_->width, impl_->height, AV_PIX_FMT_BGRA,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (impl_->frame == nullptr || impl_->packet == nullptr || impl_->scaler == nullptr)
    {
        running_.store(false, std::memory_order_release);
        return "could not allocate capture frame conversion";
    }

    worker_ = std::thread([this]
    {
        std::vector<uint8_t> bgra(static_cast<size_t>(impl_->width * impl_->height * 4));
        uint8_t* planes[] { bgra.data(), nullptr, nullptr, nullptr };
        int strides[] { impl_->width * 4, 0, 0, 0 };
        uint64_t sequence = 0;
        const auto started = std::chrono::steady_clock::now();
        while (running_.load(std::memory_order_acquire))
        {
            const int readResult = av_read_frame(impl_->format, impl_->packet);
            if (readResult == AVERROR(EAGAIN))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (readResult < 0)
            {
                if (running_.load(std::memory_order_acquire))
                {
                    const std::lock_guard<std::mutex> lock(stateMutex_);
                    error_ = "capture frame read failed: " + ffmpegError(readResult);
                }
                break;
            }
            if (impl_->packet->stream_index == impl_->streamIndex
                && avcodec_send_packet(impl_->decoder, impl_->packet) >= 0)
            {
                while (avcodec_receive_frame(impl_->decoder, impl_->frame) == 0)
                {
                    sws_scale(impl_->scaler, impl_->frame->data, impl_->frame->linesize, 0,
                              impl_->decoder->height, planes, strides);
                    if (impl_->paceInput)
                    {
                        const auto due = started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>((sequence + 1) / impl_->fps));
                        std::this_thread::sleep_until(due);
                    }
                    CaptureFrameInfo info;
                    info.sequence = ++sequence;
                    info.timestampSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    info.width = impl_->width;
                    info.height = impl_->height;
                    info.strideBytes = strides[0];
                    {
                        const std::lock_guard<std::mutex> lock(stateMutex_);
                        latest_ = info;
                    }
                    if (impl_->callback) impl_->callback(bgra.data(), info);
                    const std::lock_guard<std::mutex> lock(impl_->recordingsMutex);
                    for (auto& [id, recorder] : impl_->recordings)
                    {
                        const auto pushError = recorder->pushFrame(bgra.data(), strides[0], true);
                        if (! pushError.empty())
                        {
                            const std::lock_guard<std::mutex> stateLock(stateMutex_);
                            error_ = "capture recorder " + std::to_string(id) + ": " + pushError;
                        }
                    }
                }
            }
            av_packet_unref(impl_->packet);
        }
        running_.store(false, std::memory_order_release);
    });
    return {};
}

void CaptureDeviceSession::close()
{
    running_.store(false, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
    if (impl_ != nullptr)
    {
        const std::lock_guard<std::mutex> lock(impl_->recordingsMutex);
        for (auto& [id, recorder] : impl_->recordings) recorder->close();
        impl_->recordings.clear();
        impl_->outputPaths.clear();
    }
    impl_.reset();
}

CaptureFrameInfo CaptureDeviceSession::latestFrame() const
{
    const std::lock_guard<std::mutex> lock(stateMutex_);
    return latest_;
}

std::string CaptureDeviceSession::lastError() const
{
    const std::lock_guard<std::mutex> lock(stateMutex_);
    return error_;
}

std::string CaptureDeviceSession::startRecording(uint32_t recordingId, const std::string& outPath,
                                                 const std::string& codec)
{
    if (impl_ == nullptr || ! isOpen()) return "capture session is not open";
    const std::lock_guard<std::mutex> lock(impl_->recordingsMutex);
    if (impl_->recordings.count(recordingId) != 0) return "recordingId already active";
    auto recorder = std::make_unique<RecorderSession>();
    const auto error = recorder->open(outPath, impl_->width, impl_->height, impl_->fps, codec, "auto");
    if (! error.empty()) return error;
    impl_->outputPaths[recordingId] = outPath;
    impl_->recordings[recordingId] = std::move(recorder);
    return {};
}

std::string CaptureDeviceSession::stopRecording(uint32_t recordingId, bool cancel)
{
    if (impl_ == nullptr) return "capture session is not open";
    std::unique_ptr<RecorderSession> recorder;
    std::string path;
    {
        const std::lock_guard<std::mutex> lock(impl_->recordingsMutex);
        const auto found = impl_->recordings.find(recordingId);
        if (found == impl_->recordings.end()) return "unknown recordingId";
        recorder = std::move(found->second);
        impl_->recordings.erase(found);
        path = impl_->outputPaths[recordingId];
        impl_->outputPaths.erase(recordingId);
    }
    const auto error = recorder->close();
    if (cancel && ! path.empty()) std::remove(path.c_str());
    return error;
}
