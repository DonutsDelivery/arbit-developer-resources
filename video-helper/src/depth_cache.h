#pragma once

#include "sha256.h"

#include <nlohmann/json.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#else
#include <dirent.h>
#include <fcntl.h>
#endif
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <utility>
#include <vector>

namespace videohelper
{
struct DepthCacheBinding
{
    std::string key, receipt, version, prefix, extension;
    int firstFrame = 0, digits = 0, frames = 0, width = 0, height = 0;
    double fps = 0.0;
};

struct DepthFrameIdentity
{
    uint64_t device = 0, inode = 0, size = 0;
    int64_t modifiedSeconds = 0, modifiedNanoseconds = 0;
};

class DepthFrame final
{
public:
    DepthFrame (int widthIn, int heightIn, int indexIn, double ptsIn,
                std::vector<uint16_t> pixelsIn, std::string hashIn,
                DepthFrameIdentity identityIn)
        : width_ (widthIn), height_ (heightIn), index_ (indexIn), pts_ (ptsIn),
          pixels_ (std::move (pixelsIn)), hash_ (std::move (hashIn)), identity_ (identityIn) {}

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    int index() const noexcept { return index_; }
    double pts() const noexcept { return pts_; }
    const std::vector<uint16_t>& pixels() const noexcept { return pixels_; }
    const std::string& sha256() const noexcept { return hash_; }
    const DepthFrameIdentity& identity() const noexcept { return identity_; }
    float normalized (size_t index) const noexcept
    {
        return index < pixels_.size() ? static_cast<float> (pixels_[index]) / 65535.0f : 0.0f;
    }

private:
    int width_, height_, index_;
    double pts_;
    std::vector<uint16_t> pixels_;
    std::string hash_;
    DepthFrameIdentity identity_;
};

using DepthFramePtr = std::shared_ptr<const DepthFrame>;
#if defined(_WIN32)
using DepthDirectoryHandle = HANDLE;
#else
using DepthDirectoryHandle = int;
#endif
using DepthOpenBarrier = void (*) (DepthDirectoryHandle directory, const char* frameName);

namespace depthcache
{
constexpr uint64_t kManifestLimit = 1024 * 1024;
constexpr uint64_t kFrameLimit = 64 * 1024 * 1024;
constexpr uint64_t kAggregateLimit = 2ull * 1024 * 1024 * 1024;
constexpr int kFrameCountLimit = 10000;
constexpr int kDimensionLimit = 32768;
constexpr uint64_t kPixelLimit = 268435456;

#if defined(_WIN32)
struct WindowsFileIdentity
{
    uint64_t volume = 0, index = 0, size = 0;
    FILETIME created {}, written {};
    DWORD attributes = 0, links = 0;
};
using ReceiptFileIdentity = WindowsFileIdentity;

class Fd final
{
public:
    explicit Fd (HANDLE value = INVALID_HANDLE_VALUE) noexcept : value_ (value) {}
    ~Fd() { if (*this) ::CloseHandle (value_); }
    Fd (Fd&& other) noexcept : value_ (other.release()) {}
    Fd& operator= (Fd&& other) noexcept
    {
        if (this != &other) { if (*this) ::CloseHandle(value_); value_ = other.release(); }
        return *this;
    }
    Fd (const Fd&) = delete;
    Fd& operator= (const Fd&) = delete;
    HANDLE get() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ != nullptr && value_ != INVALID_HANDLE_VALUE; }
    HANDLE release() noexcept { const HANDLE result = value_; value_ = INVALID_HANDLE_VALUE; return result; }
private:
    HANDLE value_;
};

inline bool safePart (const std::string& text, bool extension = false)
{
    if (text.empty() || text.size() > 128 || (extension && (text.size() < 2 || text.front() != '.'))) return false;
    const size_t first = extension ? 1 : 0;
    return std::all_of (text.begin() + static_cast<std::ptrdiff_t> (first), text.end(),
        [] (unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z') || c == '_' || c == '-'; });
}

inline bool lowercaseHash (const std::string& value)
{
    return value.size() == 64 && std::all_of (value.begin(), value.end(), [] (unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

inline bool queryIdentity (HANDLE handle, WindowsFileIdentity& identity)
{
    BY_HANDLE_FILE_INFORMATION info {};
    if (! ::GetFileInformationByHandle(handle, &info)) return false;
    identity.volume = info.dwVolumeSerialNumber;
    identity.index = (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    identity.size = (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    identity.created = info.ftCreationTime;
    identity.written = info.ftLastWriteTime;
    identity.attributes = info.dwFileAttributes;
    identity.links = info.nNumberOfLinks;
    return true;
}

inline bool sameIdentity (const WindowsFileIdentity& a, const WindowsFileIdentity& b)
{
    return a.volume == b.volume && a.index == b.index && a.size == b.size
        && a.created.dwLowDateTime == b.created.dwLowDateTime && a.created.dwHighDateTime == b.created.dwHighDateTime
        && a.written.dwLowDateTime == b.written.dwLowDateTime && a.written.dwHighDateTime == b.written.dwHighDateTime
        && a.attributes == b.attributes && a.links == b.links;
}

inline bool plainDirectory (const WindowsFileIdentity& identity)
{
    return (identity.attributes & FILE_ATTRIBUTE_DIRECTORY) != 0
        && (identity.attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0;
}

inline bool sealedRegular (const WindowsFileIdentity& identity)
{
    return (identity.attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0
        && (identity.attributes & FILE_ATTRIBUTE_READONLY) != 0 && identity.links == 1;
}

inline Fd openRelative (HANDLE parent, const std::wstring& name, bool directory)
{
    if (name.empty() || name.size() > 32767) return Fd();
    using NtCreateFileFn = NTSTATUS (NTAPI*) (PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES,
        PIO_STATUS_BLOCK, PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);
    static const NtCreateFileFn ntCreateFile = [] {
        NtCreateFileFn function = nullptr;
        const FARPROC address = ::GetProcAddress(::GetModuleHandleW(L"ntdll.dll"), "NtCreateFile");
        static_assert(sizeof(function) == sizeof(address), "Windows function pointer size mismatch");
        std::memcpy(&function, &address, sizeof(function));
        return function;
    }();
    if (ntCreateFile == nullptr) return Fd();
    UNICODE_STRING unicode {};
    unicode.Buffer = const_cast<PWSTR>(name.data());
    unicode.Length = static_cast<USHORT>(name.size() * sizeof(wchar_t));
    unicode.MaximumLength = unicode.Length;
    OBJECT_ATTRIBUTES attributes {};
    InitializeObjectAttributes(&attributes, &unicode, OBJ_CASE_INSENSITIVE, parent, nullptr);
    IO_STATUS_BLOCK statusBlock {};
    HANDLE child = INVALID_HANDLE_VALUE;
    constexpr ULONG openExisting = 1;
    constexpr ULONG synchronous = 0x00000020;
    constexpr ULONG openReparsePoint = 0x00200000;
    constexpr ULONG directoryFile = 0x00000001;
    constexpr ULONG nonDirectoryFile = 0x00000040;
    const ACCESS_MASK access = FILE_READ_ATTRIBUTES | SYNCHRONIZE
        | (directory ? (FILE_LIST_DIRECTORY | FILE_TRAVERSE) : FILE_READ_DATA);
    const NTSTATUS status = ntCreateFile(&child, access, &attributes, &statusBlock, nullptr,
        FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, openExisting,
        synchronous | openReparsePoint | (directory ? directoryFile : nonDirectoryFile), nullptr, 0);
    if (status < 0) return Fd();
    WindowsFileIdentity identity {};
    if (!queryIdentity(child, identity) || (directory ? !plainDirectory(identity)
                                             : (identity.attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0))
    { ::CloseHandle(child); return Fd(); }
    return Fd(child);
}

inline Fd openAbsoluteDirectory (const std::string& path)
{
    if (path.size() < 3 || path.size() > 32767 || path[1] != ':'
        || (path[2] != '/' && path[2] != '\\')
        || !((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')))
        return Fd();
    const wchar_t driveRoot[] = { static_cast<wchar_t>(path[0]), L':', L'\\', L'\0' };
    Fd current(::CreateFileW(driveRoot, FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    WindowsFileIdentity rootIdentity {};
    if (!current || !queryIdentity(current.get(), rootIdentity) || !plainDirectory(rootIdentity)) return Fd();
    size_t at = 3;
    while (at < path.size())
    {
        const size_t slash = path.find_first_of("/\\", at);
        const std::string part = path.substr(at, slash == std::string::npos ? std::string::npos : slash - at);
        if (part.empty() || part == "." || part == "..") return Fd();
        const int needed = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, part.data(),
            static_cast<int>(part.size()), nullptr, 0);
        if (needed <= 0) return Fd();
        std::wstring wide(static_cast<size_t>(needed), L'\0');
        if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, part.data(),
            static_cast<int>(part.size()), wide.data(), needed) != needed) return Fd();
        Fd next = openRelative(current.get(), wide, true);
        if (!next) return Fd();
        current = std::move(next);
        if (slash == std::string::npos) break;
        at = slash + 1;
    }
    return current;
}

inline Fd openReceiptDirectory (HANDLE root, const std::string& relative)
{
    const auto slash = relative.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= relative.size()
        || relative.find('/', slash + 1) != std::string::npos) return Fd();
    const auto owner = relative.substr(0, slash);
    const auto receipt = relative.substr(slash + 1);
    if (owner.rfind("depth-job-", 0) != 0 || !safePart(owner) || receipt != "published") return Fd();
    Fd ownerHandle = openRelative(root, std::wstring(owner.begin(), owner.end()), true);
    return ownerHandle ? openRelative(ownerHandle.get(), L"published", true) : Fd();
}

inline std::string field (const std::string& value) { return std::to_string(value.size()) + ":" + value; }
inline std::string frameName (const DepthCacheBinding& b, int index)
{
    std::ostringstream out;
    out << b.prefix << std::setw (b.digits) << std::setfill ('0') << b.firstFrame + index << b.extension;
    return out.str();
}

inline bool readSealedFile (HANDLE directory, const std::string& name, uint64_t limit,
                            std::vector<uint8_t>& bytes, WindowsFileIdentity& identity)
{
    Fd file = openRelative(directory, std::wstring(name.begin(), name.end()), false);
    WindowsFileIdentity before {};
    if (!file || !queryIdentity(file.get(), before) || !sealedRegular(before)
        || before.size == 0 || before.size > limit) return false;
    bytes.resize(static_cast<size_t>(before.size));
    size_t offset = 0;
    while (offset < bytes.size())
    {
        const DWORD request = static_cast<DWORD>((std::min)(bytes.size() - offset,
            static_cast<size_t>((std::numeric_limits<DWORD>::max)())));
        DWORD count = 0;
        if (!::ReadFile(file.get(), bytes.data() + offset, request, &count, nullptr) || count == 0) return false;
        offset += count;
    }
    WindowsFileIdentity after {};
    if (!queryIdentity(file.get(), after) || !sameIdentity(before, after)) return false;
    identity = after;
    return true;
}

inline bool exactEntries (HANDLE directory, const std::set<std::string>& expected)
{
    std::set<std::string> actual;
    std::vector<uint8_t> buffer(64 * 1024);
    for (;;)
    {
        if (!::GetFileInformationByHandleEx(directory, FileIdBothDirectoryInfo,
                                             buffer.data(), static_cast<DWORD>(buffer.size())))
        {
            if (::GetLastError() == ERROR_NO_MORE_FILES) break;
            return false;
        }
        auto* entry = reinterpret_cast<FILE_ID_BOTH_DIR_INFO*>(buffer.data());
        for (;;)
        {
            const int needed = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, entry->FileName,
                static_cast<int>(entry->FileNameLength / sizeof(wchar_t)), nullptr, 0, nullptr, nullptr);
            if (needed <= 0) return false;
            std::string name(static_cast<size_t>(needed), '\0');
            if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, entry->FileName,
                static_cast<int>(entry->FileNameLength / sizeof(wchar_t)), name.data(), needed, nullptr, nullptr) != needed)
                return false;
            if (name != "." && name != "..") actual.insert(std::move(name));
            if (entry->NextEntryOffset == 0) break;
            entry = reinterpret_cast<FILE_ID_BOTH_DIR_INFO*>(
                reinterpret_cast<uint8_t*>(entry) + entry->NextEntryOffset);
        }
    }
    return actual == expected;
}
inline uint64_t identitySize (const WindowsFileIdentity& identity) { return identity.size; }
#else
using ReceiptFileIdentity = struct stat;
class Fd final
{
public:
    explicit Fd (int value = -1) noexcept : value_ (value) {}
    ~Fd() { if (value_ >= 0) ::close (value_); }
    Fd (Fd&& other) noexcept : value_ (other.release()) {}
    Fd& operator= (Fd&& other) noexcept { if (this != &other) { if (value_ >= 0) ::close(value_); value_ = other.release(); } return *this; }
    Fd (const Fd&) = delete;
    Fd& operator= (const Fd&) = delete;
    int get() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ >= 0; }
    int release() noexcept { const int result = value_; value_ = -1; return result; }
private:
    int value_;
};

inline bool safePart (const std::string& text, bool extension = false)
{
    if (text.empty() || text.size() > 128 || (extension && (text.size() < 2 || text.front() != '.'))) return false;
    const size_t first = extension ? 1 : 0;
    return std::all_of (text.begin() + static_cast<std::ptrdiff_t> (first), text.end(),
        [] (unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z') || c == '_' || c == '-'; });
}

inline bool lowercaseHash (const std::string& value)
{
    return value.size() == 64 && std::all_of (value.begin(), value.end(), [] (unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

inline bool sealedRegular (const struct stat& info)
{
    return S_ISREG (info.st_mode) && info.st_nlink == 1
        && (info.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0;
}

inline bool sameIdentity (const struct stat& a, const struct stat& b)
{
#if defined(__APPLE__)
    const auto& aModified = a.st_mtimespec;
    const auto& bModified = b.st_mtimespec;
#else
    const auto& aModified = a.st_mtim;
    const auto& bModified = b.st_mtim;
#endif
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size
        && aModified.tv_sec == bModified.tv_sec && aModified.tv_nsec == bModified.tv_nsec;
}

inline bool queryIdentity (int handle, struct stat& identity) { return ::fstat(handle, &identity) == 0; }
inline bool plainDirectory (const struct stat& identity) { return S_ISDIR(identity.st_mode); }
inline uint64_t identitySize (const struct stat& identity) { return static_cast<uint64_t>(identity.st_size); }

inline Fd openAbsoluteDirectory (const std::string& path)
{
    if (path.empty() || path.front() != '/') return Fd();
    Fd current (::open ("/", O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    size_t at = 1;
    while (current && at < path.size())
    {
        const size_t slash = path.find ('/', at);
        const std::string part = path.substr (at, slash == std::string::npos ? std::string::npos : slash - at);
        if (! part.empty())
        {
            if (part == "." || part == "..") return Fd();
            Fd next (::openat (current.get(), part.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
            if (! next) return Fd();
            current = std::move (next);
        }
        if (slash == std::string::npos) break;
        at = slash + 1;
    }
    return current;
}

inline Fd openReceiptDirectory (int rootFd, const std::string& relative)
{
    const auto slash = relative.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= relative.size()
        || relative.find('/', slash + 1) != std::string::npos) return Fd();
    const auto owner = relative.substr(0, slash);
    const auto receipt = relative.substr(slash + 1);
    if (owner.rfind("depth-job-", 0) != 0 || !safePart(owner) || receipt != "published") return Fd();
    Fd ownerFd (::openat(rootFd, owner.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    return ownerFd ? Fd(::openat(ownerFd.get(), receipt.c_str(),
        O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW)) : Fd();
}

inline std::string field (const std::string& value) { return std::to_string(value.size()) + ":" + value; }
inline std::string frameName (const DepthCacheBinding& b, int index)
{
    std::ostringstream out;
    out << b.prefix << std::setw (b.digits) << std::setfill ('0') << b.firstFrame + index << b.extension;
    return out.str();
}

inline bool readSealedFile (int directoryFd, const std::string& name, uint64_t limit,
                            std::vector<uint8_t>& bytes, struct stat& identity)
{
    struct stat before {};
    if (::fstatat (directoryFd, name.c_str(), &before, AT_SYMLINK_NOFOLLOW) != 0
        || ! sealedRegular (before) || before.st_size <= 0 || static_cast<uint64_t>(before.st_size) > limit) return false;
    Fd file (::openat (directoryFd, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (! file || ::fstat (file.get(), &identity) != 0 || ! sealedRegular(identity)
        || ! sameIdentity(before, identity)) return false;
    bytes.resize (static_cast<size_t>(identity.st_size));
    size_t offset = 0;
    while (offset < bytes.size())
    {
        const auto count = ::read (file.get(), bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += static_cast<size_t>(count);
    }
    struct stat after {};
    return ::fstat(file.get(), &after) == 0 && sameIdentity(identity, after);
}

inline bool exactEntries (int directoryFd, const std::set<std::string>& expected)
{
    const int duplicate = ::dup (directoryFd);
    if (duplicate < 0) return false;
    DIR* directory = ::fdopendir (duplicate);
    if (directory == nullptr) { ::close(duplicate); return false; }
    std::set<std::string> actual;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0)
            actual.emplace(entry->d_name);
    const bool ok = errno == 0 && ::closedir(directory) == 0;
    return ok && actual == expected;
}
#endif

inline bool parseManifest (const std::vector<uint8_t>& bytes, const DepthCacheBinding& b, std::string& error)
{
    try
    {
        const auto manifest = nlohmann::json::parse(bytes.begin(), bytes.end());
        const auto fpsString = manifest.at("fps").get<std::string>();
        size_t used = 0;
        const double fps = std::stod(fpsString, &used);
        if (! manifest.is_object() || manifest.at("schema") != 3 || manifest.at("version") != b.version
            || manifest.at("framePrefix") != b.prefix || manifest.at("frameExtension") != b.extension
            || manifest.at("firstFrame") != b.firstFrame || manifest.at("frameDigits") != b.digits
            || manifest.at("width") != b.width || manifest.at("height") != b.height
            || manifest.at("frameCount") != b.frames || manifest.at("frames") != b.frames
            || used != fpsString.size() || !std::isfinite(fps) || fps != b.fps
            || manifest.at("contentReceipt") != b.receipt
            || manifest.at("outputSemantics").at("publication")
                != "per-frame min/max normalized uint16 grayscale PNG; constant/non-finite frames are rejected"
            || manifest.at("outputSemantics").at("channel") != "red/single-channel")
        { error = "depth receipt manifest does not match binding"; return false; }
    }
    catch (...) { error = "malformed depth receipt manifest"; return false; }
    return true;
}

inline bool decodePng16 (const std::vector<uint8_t>& encoded, int width, int height,
                         std::vector<uint16_t>& pixels, std::string& error)
{
    struct ContextFree { void operator() (AVCodecContext* value) const { avcodec_free_context(&value); } };
    struct FrameFree { void operator() (AVFrame* value) const { av_frame_free(&value); } };
    struct PacketFree { void operator() (AVPacket* value) const { av_packet_free(&value); } };
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_PNG);
    if (codec == nullptr || encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    { error = "PNG decoder unavailable"; return false; }
    std::unique_ptr<AVCodecContext, ContextFree> context(avcodec_alloc_context3(codec));
    std::unique_ptr<AVFrame, FrameFree> frame(av_frame_alloc());
    std::unique_ptr<AVPacket, PacketFree> packet(av_packet_alloc());
    if (!context || !frame || !packet || avcodec_open2(context.get(), codec, nullptr) < 0
        || av_new_packet(packet.get(), static_cast<int>(encoded.size())) < 0)
    { error = "PNG decoder unavailable"; return false; }
    std::memcpy(packet->data, encoded.data(), encoded.size());
    if (avcodec_send_packet(context.get(), packet.get()) < 0 || avcodec_receive_frame(context.get(), frame.get()) < 0
        || frame->width != width || frame->height != height
        || (frame->format != AV_PIX_FMT_GRAY16BE && frame->format != AV_PIX_FMT_GRAY16LE))
    { error = "depth frame is not exact 16-bit grayscale PNG"; return false; }
    if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > kPixelLimit)
    { error = "depth PNG dimensions exceed bound"; return false; }
    pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    const bool bigEndian = frame->format == AV_PIX_FMT_GRAY16BE;
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* row = frame->data[0] + static_cast<ptrdiff_t>(y) * frame->linesize[0];
        for (int x = 0; x < width; ++x)
            pixels[static_cast<size_t>(y) * width + x] = bigEndian
                ? static_cast<uint16_t>((static_cast<uint16_t>(row[2*x]) << 8) | row[2*x+1])
                : static_cast<uint16_t>((static_cast<uint16_t>(row[2*x+1]) << 8) | row[2*x]);
    }
    return true;
}
} // namespace depthcache

inline int depthFrameIndexForPts (double pts, double fps, int frames) noexcept
{
    if (! std::isfinite(pts) || ! std::isfinite(fps) || fps <= 0.0 || frames <= 0) return -1;
    const long double raw = std::floor(static_cast<long double>(pts) * static_cast<long double>(fps));
    if (raw <= 0.0L) return 0;
    if (raw >= static_cast<long double>(frames - 1)) return frames - 1;
    return static_cast<int>(raw);
}

inline DepthFramePtr admitDepthFrame (const std::string& canonicalCacheRoot,
                                      const DepthCacheBinding& binding, double pts,
                                      std::string& error, DepthOpenBarrier barrier = nullptr)
{
    using namespace depthcache;
    error.clear();
    const uint64_t pixels = binding.width > 0 && binding.height > 0
        ? static_cast<uint64_t>(binding.width) * static_cast<uint64_t>(binding.height) : 0;
    if (!lowercaseHash(binding.receipt) || binding.version != "depth-sequence-v1"
        || !safePart(binding.prefix) || !safePart(binding.extension, true)
        || binding.firstFrame < 0 || binding.digits <= 0 || binding.digits > 12
        || binding.frames <= 0 || binding.frames > kFrameCountLimit
        || binding.width <= 0 || binding.height <= 0 || binding.width > kDimensionLimit
        || binding.height > kDimensionLimit || pixels == 0 || pixels > kPixelLimit
        || !std::isfinite(binding.fps) || binding.fps <= 0.0
        || binding.firstFrame > std::numeric_limits<int>::max() - binding.frames + 1)
    { error = "malformed depth cache binding"; return {}; }
    const int selected = depthFrameIndexForPts(pts, binding.fps, binding.frames);
    if (selected < 0) { error = "invalid depth frame PTS"; return {}; }

    Fd root = openAbsoluteDirectory(canonicalCacheRoot);
    Fd directory = root ? openReceiptDirectory(root.get(), binding.key) : Fd();
    ReceiptFileIdentity directoryBefore {};
    if (!directory || !queryIdentity(directory.get(), directoryBefore) || !plainDirectory(directoryBefore))
    { error = "depth receipt directory unavailable"; return {}; }

    std::vector<uint8_t> manifestBytes;
    ReceiptFileIdentity manifestIdentity {};
    if (!readSealedFile(directory.get(), "receipt.json", kManifestLimit, manifestBytes, manifestIdentity)
        || !parseManifest(manifestBytes, binding, error))
    { if (error.empty()) error = "depth receipt manifest is not sealed"; return {}; }
    std::vector<uint8_t> ownerMarker;
    ReceiptFileIdentity markerIdentity {};
    if (!readSealedFile(directory.get(), ".depth-output-owned", 64, ownerMarker, markerIdentity)
        || std::string(ownerMarker.begin(), ownerMarker.end()) != "depth-output-v1\n")
    { error = "depth receipt owner marker is invalid"; return {}; }

    std::set<std::string> expected { "receipt.json", ".depth-output-owned" };
    std::string material = "depth-receipt-v1";
    std::ostringstream fps; fps << std::setprecision(17) << binding.fps;
    for (const auto& value : {binding.version, binding.prefix, binding.extension,
         std::to_string(binding.firstFrame), std::to_string(binding.digits), std::to_string(binding.width),
         std::to_string(binding.height), std::to_string(binding.frames), fps.str()}) material += field(value);
    uint64_t aggregate = 0;
    std::string selectedReceiptHash;
    for (int index = 0; index < binding.frames; ++index)
    {
        const auto name = frameName(binding, index);
        expected.insert(name);
        std::vector<uint8_t> bytes;
        ReceiptFileIdentity identity {};
        if (!readSealedFile(directory.get(), name, kFrameLimit, bytes, identity)
            || identitySize(identity) > kAggregateLimit - aggregate)
        { error = "depth frame invalid, linked, unsealed, or oversized"; return {}; }
        aggregate += identitySize(identity);
        Sha256 hash; hash.update(bytes.data(), bytes.size());
        const auto frameHash = hash.finishHex();
        if (index == selected) selectedReceiptHash = frameHash;
        material += field(name) + field(std::to_string(bytes.size())) + field(frameHash);
    }
    if (!exactEntries(directory.get(), expected)) { error = "depth receipt contains missing or extra entries"; return {}; }
#if defined(_WIN32)
    ReceiptFileIdentity directoryAfterEntries {};
    if (!queryIdentity(directory.get(), directoryAfterEntries)
        || !sameIdentity(directoryBefore, directoryAfterEntries))
    { error = "depth receipt directory changed during validation"; return {}; }
#endif
    if (sha256Text(material) != binding.receipt) { error = "depth frame-set receipt mismatch"; return {}; }

    const auto selectedName = frameName(binding, selected);
    if (barrier != nullptr) barrier(directory.get(), selectedName.c_str());
    std::vector<uint8_t> selectedBytes;
    ReceiptFileIdentity selectedIdentity {};
    if (!readSealedFile(directory.get(), selectedName, kFrameLimit, selectedBytes, selectedIdentity))
    { error = "selected depth frame changed before decode"; return {}; }
    Sha256 selectedHash; selectedHash.update(selectedBytes.data(), selectedBytes.size());
    const auto selectedHashHex = selectedHash.finishHex();
    if (selectedHashHex != selectedReceiptHash)
    { error = "selected depth frame changed after receipt validation"; return {}; }
#if defined(_WIN32)
    ReceiptFileIdentity directoryAfterSelected {};
    if (!queryIdentity(directory.get(), directoryAfterSelected)
        || !sameIdentity(directoryAfterEntries, directoryAfterSelected))
    { error = "depth receipt directory changed before decode"; return {}; }
#endif
    std::vector<uint16_t> decoded;
    if (!decodePng16(selectedBytes, binding.width, binding.height, decoded, error)) return {};
#if defined(_WIN32)
    const uint64_t written = (static_cast<uint64_t>(selectedIdentity.written.dwHighDateTime) << 32)
        | selectedIdentity.written.dwLowDateTime;
    constexpr uint64_t windowsToUnix100ns = 116444736000000000ull;
    const uint64_t unix100ns = written >= windowsToUnix100ns ? written - windowsToUnix100ns : 0;
    DepthFrameIdentity frameIdentity { selectedIdentity.volume, selectedIdentity.index,
        selectedIdentity.size, static_cast<int64_t>(unix100ns / 10000000ull),
        static_cast<int64_t>((unix100ns % 10000000ull) * 100ull) };
#else
 #if defined(__APPLE__)
    const auto& selectedModified = selectedIdentity.st_mtimespec;
 #else
    const auto& selectedModified = selectedIdentity.st_mtim;
 #endif
    DepthFrameIdentity frameIdentity { static_cast<uint64_t>(selectedIdentity.st_dev),
        static_cast<uint64_t>(selectedIdentity.st_ino), static_cast<uint64_t>(selectedIdentity.st_size),
        static_cast<int64_t>(selectedModified.tv_sec), static_cast<int64_t>(selectedModified.tv_nsec) };
#endif
    return std::make_shared<const DepthFrame>(binding.width, binding.height, selected,
        static_cast<double>(selected) / binding.fps, std::move(decoded), selectedHashHex, frameIdentity);
}
} // namespace videohelper
