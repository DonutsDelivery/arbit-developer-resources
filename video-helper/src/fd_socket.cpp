#include "fd_socket.h"

#if ARBIT_HAVE_DMABUF || ARBIT_HAVE_IOSURFACE || ARBIT_HAVE_D3D

#include "SharedGpuSurfaceProtocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#if defined (_WIN32)
 #define WIN32_LEAN_AND_MEAN
 #include <winsock2.h>
 #include <windows.h>
 // afunix.h is present in current SDKs/MinGW but spell the struct locally so
 // older toolchains build too (AF_UNIX support itself is Win10 1803+).
 struct sockaddr_un
 {
     ADDRESS_FAMILY sun_family;
     char sun_path[108];
 };
 #ifndef AF_UNIX
  #define AF_UNIX 1
 #endif
#else
 #include <cerrno>
 #include <ctime>
 #include <fcntl.h>
 #include <poll.h>
 #include <sys/socket.h>
 #include <sys/stat.h>
 #include <sys/un.h>
 #include <unistd.h>
#endif

namespace
{
// Generous upper bound for one message: header + BUFFERS payload for
// kMaxBuffers, with headroom for future fields.
constexpr size_t kMaxMsgBytes = 4096;
constexpr int kMaxFdsPerMsg = 16;

#if defined (_WIN32)
bool ensureWinsock()
{
    static bool initialized = []
    {
        WSADATA wsa;
        return WSAStartup (MAKEWORD (2, 2), &wsa) == 0;
    }();
    return initialized;
}

void closeSock (intptr_t& s)
{
    if (s >= 0)
    {
        ::closesocket ((SOCKET) s);
        s = -1;
    }
}

bool setNonBlocking (intptr_t s)
{
    u_long mode = 1;
    return ::ioctlsocket ((SOCKET) s, FIONBIO, &mode) == 0;
}

bool wouldBlock() { return WSAGetLastError() == WSAEWOULDBLOCK; }
#else
void closeSock (intptr_t& s)
{
    if (s >= 0)
    {
        ::close ((int) s);
        s = -1;
    }
}

#if ! defined (__linux__)
bool setNonBlocking (intptr_t s)
{
    const int flags = ::fcntl ((int) s, F_GETFL, 0);
    return flags >= 0 && ::fcntl ((int) s, F_SETFL, flags | O_NONBLOCK) == 0;
}

bool wouldBlock() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR; }
#endif
#endif
} // namespace

FdSocketServer::~FdSocketServer()
{
    close();
}

bool FdSocketServer::listen (std::string& pathOut, std::string& errorOut)
{
    close();

#if defined (_WIN32)
    if (! ensureWinsock())
    {
        errorOut = "WSAStartup failed";
        return false;
    }
    char tmp[MAX_PATH] = {};
    const DWORD n = GetTempPathA (MAX_PATH, tmp);
    std::string dir = n > 0 ? std::string (tmp, n) : std::string (".\\");
    if (! dir.empty() && (dir.back() == '\\' || dir.back() == '/'))
        dir.pop_back();
    const unsigned pid = (unsigned) GetCurrentProcessId();
#else
    const char* runtimeDir = std::getenv ("XDG_RUNTIME_DIR");
    std::string dir = (runtimeDir != nullptr && runtimeDir[0] != '\0') ? runtimeDir : "/tmp";
    const unsigned pid = (unsigned) getpid();
#endif

    std::random_device rd;
    char path[108]; // sizeof(sockaddr_un::sun_path)
    std::snprintf (path, sizeof (path), "%s/arbit-vp-%u-%08x.sock",
                   dir.c_str(), pid, (unsigned) rd());
    path_ = path;

#if defined (__linux__)
    listenFd_ = ::socket (AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
#elif defined (_WIN32)
    listenFd_ = (intptr_t) ::socket (AF_UNIX, SOCK_STREAM, 0);
    if ((SOCKET) listenFd_ == INVALID_SOCKET)
        listenFd_ = -1;
#else
    listenFd_ = ::socket (AF_UNIX, SOCK_STREAM, 0);
#endif
    if (listenFd_ < 0)
    {
        errorOut = "socket(AF_UNIX) failed";
        return false;
    }
#if ! defined (__linux__)
    if (! setNonBlocking (listenFd_))
    {
        errorOut = "could not set socket non-blocking";
        close();
        return false;
    }
#endif

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy (addr.sun_path, path_.c_str(), sizeof (addr.sun_path) - 1);
#if defined (_WIN32)
    DeleteFileA (path_.c_str());
    if (::bind ((SOCKET) listenFd_, (const sockaddr*) &addr, (int) sizeof (addr)) != 0)
#else
    ::unlink (path_.c_str());
    if (::bind ((int) listenFd_, (const sockaddr*) &addr, sizeof (addr)) != 0)
#endif
    {
        errorOut = "bind(" + path_ + ") failed";
        close();
        return false;
    }
#if ! defined (_WIN32)
    ::chmod (path_.c_str(), 0600);
    if (::listen ((int) listenFd_, 1) != 0)
#else
    if (::listen ((SOCKET) listenFd_, 1) != 0)
#endif
    {
        errorOut = "listen() failed";
        close();
        return false;
    }

#if ! defined (__linux__)
    // poll() calls accept() directly on macOS and Windows. Keep the listening
    // socket non-blocking so the render loop can finish opening the shared
    // viewport before a consumer connects.
    setNonBlocking (listenFd_);
#endif

    pathOut = path_;
    return true;
}

bool FdSocketServer::poll()
{
    if (listenFd_ < 0 || clientFd_ >= 0)
        return false;

#if defined (__linux__)
    const int fd = ::accept4 ((int) listenFd_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (fd < 0)
        return false;
    clientFd_ = fd;
#elif defined (_WIN32)
    const SOCKET s = ::accept ((SOCKET) listenFd_, nullptr, nullptr);
    if (s == INVALID_SOCKET)
        return false;
    clientFd_ = (intptr_t) s;
    setNonBlocking (clientFd_);
    rxBuf_.clear();
#else
    pollfd readiness { (int) listenFd_, POLLIN, 0 };
    if (::poll (&readiness, 1, 0) <= 0 || (readiness.revents & POLLIN) == 0)
        return false;
    const int fd = ::accept ((int) listenFd_, nullptr, nullptr);
    if (fd < 0)
        return false;
    ::fcntl (fd, F_SETFD, FD_CLOEXEC);
    const int one = 1;
    ::setsockopt (fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof (one)); // macOS: no MSG_NOSIGNAL
    clientFd_ = fd;
    setNonBlocking (clientFd_);
    rxBuf_.clear();
#endif
    return true;
}

#if defined (__linux__)

bool FdSocketServer::sendMsg (uint32_t type, const void* payload, size_t payloadBytes,
                              const int* fds, int fdCount)
{
    if (clientFd_ < 0 || fdCount > kMaxFdsPerMsg)
        return false;

    gpusurf::MsgHeader header;
    header.type = type;
    header.fdCount = (uint32_t) fdCount;
    header.payloadBytes = (uint32_t) payloadBytes;

    iovec iov[2];
    iov[0].iov_base = &header;
    iov[0].iov_len = sizeof (header);
    iov[1].iov_base = const_cast<void*> (payload);
    iov[1].iov_len = payloadBytes;

    msghdr msg {};
    msg.msg_iov = iov;
    msg.msg_iovlen = payloadBytes > 0 ? 2 : 1;

    alignas (cmsghdr) char control[CMSG_SPACE (sizeof (int) * kMaxFdsPerMsg)];
    if (fdCount > 0)
    {
        std::memset (control, 0, sizeof (control));
        msg.msg_control = control;
        msg.msg_controllen = CMSG_SPACE (sizeof (int) * (size_t) fdCount);
        cmsghdr* cmsg = CMSG_FIRSTHDR (&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN (sizeof (int) * (size_t) fdCount);
        std::memcpy (CMSG_DATA (cmsg), fds, sizeof (int) * (size_t) fdCount);
    }

    const ssize_t sent = ::sendmsg ((int) clientFd_, &msg, MSG_NOSIGNAL);
    return sent == (ssize_t) (sizeof (header) + payloadBytes);
}

FdSocketServer::RecvResult FdSocketServer::recvMsg (Received& out)
{
    if (clientFd_ < 0)
        return RecvResult::NoData;

    uint8_t buffer[kMaxMsgBytes];
    iovec iov { buffer, sizeof (buffer) };

    alignas (cmsghdr) char control[CMSG_SPACE (sizeof (int) * kMaxFdsPerMsg)];
    msghdr msg {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof (control);

    const ssize_t n = ::recvmsg ((int) clientFd_, &msg, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    if (n < 0)
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
            return RecvResult::NoData;
        dropClient();
        return RecvResult::Disconnected;
    }
    if (n == 0) // orderly shutdown
    {
        dropClient();
        return RecvResult::Disconnected;
    }

    // Collect any fds first so they cannot leak on a malformed message.
    out.fds.clear();
    for (cmsghdr* cmsg = CMSG_FIRSTHDR (&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR (&msg, cmsg))
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
        {
            const size_t count = (cmsg->cmsg_len - CMSG_LEN (0)) / sizeof (int);
            const int* received = (const int*) CMSG_DATA (cmsg);
            for (size_t i = 0; i < count; ++i)
                out.fds.push_back (received[i]);
        }

    if ((size_t) n < sizeof (gpusurf::MsgHeader))
    {
        for (int fd : out.fds) ::close (fd);
        out.fds.clear();
        return RecvResult::NoData; // malformed; ignore
    }

    gpusurf::MsgHeader header;
    std::memcpy (&header, buffer, sizeof (header));
    if (header.magic != gpusurf::kMagic
        || (size_t) n != sizeof (header) + header.payloadBytes)
    {
        for (int fd : out.fds) ::close (fd);
        out.fds.clear();
        return RecvResult::NoData; // malformed; ignore
    }

    out.type = header.type;
    out.payload.assign (buffer + sizeof (header), buffer + n);
    return RecvResult::Message;
}

#else // stream transports (macOS / Windows): framed by MsgHeader.payloadBytes

bool FdSocketServer::sendMsg (uint32_t type, const void* payload, size_t payloadBytes,
                              const int* fds, int fdCount)
{
    (void) fds;
    if (clientFd_ < 0 || fdCount != 0) // fd passing is Linux-only
        return false;

    gpusurf::MsgHeader header;
    header.type = type;
    header.fdCount = 0;
    header.payloadBytes = (uint32_t) payloadBytes;

    std::vector<uint8_t> buf (sizeof (header) + payloadBytes);
    std::memcpy (buf.data(), &header, sizeof (header));
    if (payloadBytes > 0)
        std::memcpy (buf.data() + sizeof (header), payload, payloadBytes);

    // Messages are tiny (<= kMaxMsgBytes) against default-size socket
    // buffers, so partial sends are rare; bounded retry covers them.
    size_t off = 0;
    for (int attempt = 0; attempt < 200 && off < buf.size(); ++attempt)
    {
#if defined (_WIN32)
        const int n = ::send ((SOCKET) clientFd_, (const char*) buf.data() + off,
                              (int) (buf.size() - off), 0);
#else
        // macOS has no MSG_NOSIGNAL; SO_NOSIGPIPE is set on accept instead.
        const ssize_t n = ::send ((int) clientFd_, buf.data() + off,
                                  buf.size() - off, 0);
#endif
        if (n > 0)
        {
            off += (size_t) n;
            continue;
        }
        if (n < 0 && wouldBlock())
        {
#if defined (_WIN32)
            Sleep (1);
#else
            struct timespec ts { 0, 1000000 }; // 1 ms
            nanosleep (&ts, nullptr);
#endif
            continue;
        }
        return false; // hard error / orderly close
    }
    return off == buf.size();
}

FdSocketServer::RecvResult FdSocketServer::recvMsg (Received& out)
{
    if (clientFd_ < 0)
        return RecvResult::NoData;

    // Drain whatever is available into the reassembly buffer.
    for (;;)
    {
        uint8_t chunk[4096];
#if defined (_WIN32)
        const int n = ::recv ((SOCKET) clientFd_, (char*) chunk, (int) sizeof (chunk), 0);
#else
        const ssize_t n = ::recv ((int) clientFd_, chunk, sizeof (chunk), 0);
#endif
        if (n > 0)
        {
            rxBuf_.insert (rxBuf_.end(), chunk, chunk + n);
            if ((size_t) n < sizeof (chunk))
                break;
            continue;
        }
        if (n == 0) // orderly shutdown
        {
            dropClient();
            return RecvResult::Disconnected;
        }
        if (wouldBlock())
            break;
        dropClient();
        return RecvResult::Disconnected;
    }

    if (rxBuf_.size() < sizeof (gpusurf::MsgHeader))
        return RecvResult::NoData;

    gpusurf::MsgHeader header;
    std::memcpy (&header, rxBuf_.data(), sizeof (header));
    if (header.magic != gpusurf::kMagic || header.payloadBytes > kMaxMsgBytes)
    {
        // Stream desync is unrecoverable — drop the client; it reconnects.
        dropClient();
        return RecvResult::Disconnected;
    }
    const size_t total = sizeof (header) + header.payloadBytes;
    if (rxBuf_.size() < total)
        return RecvResult::NoData;

    out.type = header.type;
    out.fds.clear();
    out.payload.assign (rxBuf_.begin() + sizeof (header), rxBuf_.begin() + (long) total);
    rxBuf_.erase (rxBuf_.begin(), rxBuf_.begin() + (long) total);
    return RecvResult::Message;
}

#endif // transport split

void FdSocketServer::dropClient()
{
    closeSock (clientFd_);
#if ! defined (__linux__)
    rxBuf_.clear();
#endif
}

void FdSocketServer::close()
{
    dropClient();
    closeSock (listenFd_);
    if (! path_.empty())
    {
#if defined (_WIN32)
        DeleteFileA (path_.c_str());
#else
        ::unlink (path_.c_str());
#endif
        path_.clear();
    }
}

#endif // ARBIT_HAVE_DMABUF || ARBIT_HAVE_IOSURFACE || ARBIT_HAVE_D3D
