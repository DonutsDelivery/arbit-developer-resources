#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct CaptureDeviceInfo
{
    std::string id;
    std::string name;
    std::string kind;
    std::string backend;
};

struct CaptureFrameInfo
{
    uint64_t sequence = 0;
    double timestampSec = 0.0;
    int width = 0;
    int height = 0;
    int strideBytes = 0;
};

std::vector<CaptureDeviceInfo> listCaptureDevices (const std::string& kind, std::string& error);

class CaptureDeviceSession
{
public:
    using FrameCallback = std::function<void(const uint8_t*, const CaptureFrameInfo&)>;

    CaptureDeviceSession();
    ~CaptureDeviceSession();
    CaptureDeviceSession(const CaptureDeviceSession&) = delete;
    CaptureDeviceSession& operator=(const CaptureDeviceSession&) = delete;

    std::string open(const std::string& sourceId, const std::string& backend,
                     int width, int height, double fps, FrameCallback callback);
    void close();
    bool isOpen() const { return running_.load(std::memory_order_acquire); }
    CaptureFrameInfo latestFrame() const;
    std::string lastError() const;

    std::string startRecording(uint32_t recordingId, const std::string& outPath,
                               const std::string& codec);
    std::string stopRecording(uint32_t recordingId, bool cancel);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> running_ { false };
    std::thread worker_;
    mutable std::mutex stateMutex_;
    CaptureFrameInfo latest_;
    std::string error_;
};
