#pragma once

/*
  VideoFrameSharedMemory — cross-platform shared-memory ring used to transport
  decoded RGBA video frames from the arbit-video-helper sidecar into Arbit.

  This header is intentionally free of JUCE and FFmpeg includes: it is compiled
  into both the (proprietary) plugin and the (GPL) video-helper binary. Keep it
  dependency-free.

  Protocol: Arbit creates the region and tells the helper its name via the
  "attach_shm" JSON-RPC call. Frame delivery is request/response: the helper
  writes a frame into the next ring slot (seqlock-style generation counter:
  odd = write in progress) and returns the slot index in the "request_frame"
  RPC response. The reader validates the generation before and after copying.
*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace videoshm
{

static constexpr uint32_t kMagic = 0x41565348; // 'AVSH'
static constexpr uint32_t kVersion = 1;
static constexpr uint32_t kDefaultSlotCount = 8;
// Default slot payload: 1920x1080 RGBA. The helper never decodes above the
// negotiated slot size; larger sources are scaled down.
static constexpr uint32_t kDefaultSlotBytes = 1920u * 1080u * 4u;

struct Header
{
    uint32_t magic;
    uint32_t version;
    uint32_t slotCount;
    uint32_t slotBytes;     // payload capacity per slot
    uint32_t slotStride;    // bytes from one SlotHeader to the next (header + payload, aligned)
    uint32_t reserved[3];
};

struct SlotHeader
{
    std::atomic<uint32_t> generation; // seqlock: odd while helper writes
    uint32_t width;
    uint32_t height;
    uint32_t strideBytes;   // bytes per row in payload
    double   ptsSec;        // presentation time of this frame in media time
    uint32_t mediaId;       // helper media handle the frame belongs to
    uint32_t reserved[3];
};

static constexpr size_t kAlign = 64;
inline size_t alignUp (size_t v) { return (v + (kAlign - 1)) & ~(kAlign - 1); }

inline size_t regionSize (uint32_t slotCount, uint32_t slotBytes)
{
    return alignUp (sizeof (Header))
         + (size_t) slotCount * (alignUp (sizeof (SlotHeader)) + alignUp (slotBytes));
}

/** Maps (and optionally creates) the named shared-memory region. */
class Region
{
public:
    Region() = default;
    ~Region() { close(); }

    Region (const Region&) = delete;
    Region& operator= (const Region&) = delete;

    /** Create the region (Arbit side). Name should be short (macOS shm names
        must be < 31 chars including the leading slash, which is added here). */
    bool create (const std::string& name, uint32_t slotCount = kDefaultSlotCount,
                 uint32_t slotBytes = kDefaultSlotBytes)
    {
        close();
        name_ = name;
        size_ = regionSize (slotCount, slotBytes);

#if defined(_WIN32)
        mapping_ = CreateFileMappingA (INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                       (DWORD) ((uint64_t) size_ >> 32),
                                       (DWORD) (size_ & 0xffffffffu), name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, size_);
        if (base_ == nullptr) { close(); return false; }
#else
        const std::string posixName = "/" + name;
        shm_unlink (posixName.c_str()); // stale region from a crashed run
        fd_ = shm_open (posixName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd_ < 0) return false;
        if (ftruncate (fd_, (off_t) size_) != 0) { close(); return false; }
        base_ = mmap (nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
        owner_ = true;
#endif
        std::memset (base_, 0, size_);
        auto* h = header();
        h->magic = kMagic;
        h->version = kVersion;
        h->slotCount = slotCount;
        h->slotBytes = slotBytes;
        h->slotStride = (uint32_t) (alignUp (sizeof (SlotHeader)) + alignUp (slotBytes));
        return true;
    }

    /** Open an existing region (helper side). */
    bool open (const std::string& name)
    {
        close();
        name_ = name;
#if defined(_WIN32)
        mapping_ = OpenFileMappingA (FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (base_ == nullptr) { close(); return false; }
        // Read the header to learn the size (the view maps the whole section).
#else
        const std::string posixName = "/" + name;
        fd_ = shm_open (posixName.c_str(), O_RDWR, 0600);
        if (fd_ < 0) return false;
        struct stat st {};
        if (fstat (fd_, &st) != 0) { close(); return false; }
        size_ = (size_t) st.st_size;
        base_ = mmap (nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
#endif
        auto* h = header();
        if (h->magic != kMagic || h->version != kVersion) { close(); return false; }
        if (size_ == 0)
            size_ = regionSize (h->slotCount, h->slotBytes);
        return true;
    }

    void close()
    {
#if defined(_WIN32)
        if (base_ != nullptr) UnmapViewOfFile (base_);
        if (mapping_ != nullptr) CloseHandle (mapping_);
        mapping_ = nullptr;
#else
        if (base_ != nullptr) munmap (base_, size_);
        if (fd_ >= 0) ::close (fd_);
        if (owner_ && ! name_.empty()) shm_unlink (("/" + name_).c_str());
        fd_ = -1;
        owner_ = false;
#endif
        base_ = nullptr;
        size_ = 0;
    }

    bool isOpen() const { return base_ != nullptr; }

    Header* header() const { return reinterpret_cast<Header*> (base_); }

    SlotHeader* slot (uint32_t index) const
    {
        auto* h = header();
        if (h == nullptr || index >= h->slotCount) return nullptr;
        auto* p = reinterpret_cast<uint8_t*> (base_) + alignUp (sizeof (Header))
                + (size_t) index * h->slotStride;
        return reinterpret_cast<SlotHeader*> (p);
    }

    uint8_t* slotPayload (uint32_t index) const
    {
        auto* s = slot (index);
        if (s == nullptr) return nullptr;
        return reinterpret_cast<uint8_t*> (s) + alignUp (sizeof (SlotHeader));
    }

    /** Reader-side: copy a frame out of a slot, validating the seqlock.
        Returns false if the slot was overwritten mid-copy (caller re-requests). */
    bool readSlot (uint32_t index, uint32_t& width, uint32_t& height,
                   uint32_t& strideBytes, double& ptsSec,
                   uint8_t* dest, size_t destCapacity) const
    {
        auto* s = slot (index);
        if (s == nullptr) return false;

        const uint32_t genBefore = s->generation.load (std::memory_order_acquire);
        if ((genBefore & 1u) != 0) return false; // write in progress

        width = s->width; height = s->height;
        strideBytes = s->strideBytes; ptsSec = s->ptsSec;
        const size_t bytes = (size_t) strideBytes * height;
        if (bytes == 0 || bytes > destCapacity || bytes > header()->slotBytes)
            return false;

        std::memcpy (dest, slotPayload (index), bytes);
        std::atomic_thread_fence (std::memory_order_acquire);
        return s->generation.load (std::memory_order_acquire) == genBefore;
    }

private:
    std::string name_;
    void* base_ = nullptr;
    size_t size_ = 0;
#if defined(_WIN32)
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
    bool owner_ = false;
#endif
};

//==============================================================================
// Transport clock block — a tiny sibling shm region Arbit's audio thread
// updates once per audio block. The helper's viewport render loop reads it
// lock-free (seqlock) and extrapolates the playhead between updates against
// the wall clock; audio is master, video chases.

static constexpr uint32_t kTransportMagic = 0x41565443; // 'AVTC'
static constexpr uint32_t kTransportVersion = 2;
static constexpr int kMaxTrackFeatureSlots = 8;

struct TrackFeatureSlot
{
    int32_t trackId = -1;
    uint32_t tapPoint = 0; // 1=track post-FX/pre-fader, 2=group post-FX/pre-fader
    float rms = 0.0f;
    float peak = 0.0f;
    float onset = 0.0f;
};

struct TransportBlock
{
    uint32_t magic;
    uint32_t version;
    std::atomic<uint32_t> generation; // seqlock: odd while Arbit writes
    uint32_t playing;                 // 0/1
    double playheadBeats;             // at the start of the audio block
    double bpm;
    int64_t sampleTime;               // engine sample position
    double sampleRate;
    int64_t hostTimeNs;               // CLOCK_MONOTONIC at write time
    double beatsPerBar;               // time-signature numerator (uBarPhase / ClockBarPhase)
    uint32_t trackFeatureCount;
    TrackFeatureSlot trackFeatures[kMaxTrackFeatureSlots];
};

/** Maps (and optionally creates) the named transport region. Same ownership
    rules as Region: Arbit creates, helper opens. */
class TransportRegion
{
public:
    TransportRegion() = default;
    ~TransportRegion() { close(); }
    TransportRegion (const TransportRegion&) = delete;
    TransportRegion& operator= (const TransportRegion&) = delete;

    bool create (const std::string& name)
    {
        close();
        name_ = name;
#if defined(_WIN32)
        mapping_ = CreateFileMappingA (INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                       0, (DWORD) sizeof (TransportBlock), name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof (TransportBlock));
        if (base_ == nullptr) { close(); return false; }
#else
        const std::string posixName = "/" + name;
        shm_unlink (posixName.c_str());
        fd_ = shm_open (posixName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd_ < 0) return false;
        if (ftruncate (fd_, (off_t) sizeof (TransportBlock)) != 0) { close(); return false; }
        base_ = mmap (nullptr, sizeof (TransportBlock), PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
        owner_ = true;
#endif
        std::memset (base_, 0, sizeof (TransportBlock));
        block()->magic = kTransportMagic;
        block()->version = kTransportVersion;
        return true;
    }

    bool open (const std::string& name)
    {
        close();
        name_ = name;
#if defined(_WIN32)
        mapping_ = OpenFileMappingA (FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (base_ == nullptr) { close(); return false; }
#else
        const std::string posixName = "/" + name;
        fd_ = shm_open (posixName.c_str(), O_RDWR, 0600);
        if (fd_ < 0) return false;
        base_ = mmap (nullptr, sizeof (TransportBlock), PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
#endif
        if (block()->magic != kTransportMagic) { close(); return false; }
        return true;
    }

    void close()
    {
#if defined(_WIN32)
        if (base_ != nullptr) UnmapViewOfFile (base_);
        if (mapping_ != nullptr) CloseHandle (mapping_);
        mapping_ = nullptr;
#else
        if (base_ != nullptr) munmap (base_, sizeof (TransportBlock));
        if (fd_ >= 0) ::close (fd_);
        if (owner_ && ! name_.empty()) shm_unlink (("/" + name_).c_str());
        fd_ = -1;
        owner_ = false;
#endif
        base_ = nullptr;
    }

    bool isOpen() const { return base_ != nullptr; }
    TransportBlock* block() const { return reinterpret_cast<TransportBlock*> (base_); }

    /** Writer (Arbit audio thread): wait-free, no syscalls. */
    void write (bool playing, double playheadBeats, double bpm,
                int64_t sampleTime, double sampleRate, int64_t hostTimeNs,
                double beatsPerBar, const TrackFeatureSlot* features = nullptr,
                uint32_t featureCount = 0)
    {
        auto* b = block();
        if (b == nullptr) return;
        b->generation.fetch_add (1, std::memory_order_acq_rel); // -> odd
        b->playing = playing ? 1u : 0u;
        b->playheadBeats = playheadBeats;
        b->bpm = bpm;
        b->sampleTime = sampleTime;
        b->sampleRate = sampleRate;
        b->hostTimeNs = hostTimeNs;
        b->beatsPerBar = beatsPerBar;
        const uint32_t bounded = features != nullptr
            ? std::min<uint32_t>(featureCount, kMaxTrackFeatureSlots) : 0;
        b->trackFeatureCount = bounded;
        for (uint32_t i = 0; i < bounded; ++i) b->trackFeatures[i] = features[i];
        for (uint32_t i = bounded; i < kMaxTrackFeatureSlots; ++i) b->trackFeatures[i] = {};
        b->generation.fetch_add (1, std::memory_order_acq_rel); // -> even
    }

    /** Reader (helper render loop): returns false if torn (caller retries). */
    bool read (TransportBlock& out) const
    {
        auto* b = block();
        if (b == nullptr) return false;
        const uint32_t g1 = b->generation.load (std::memory_order_acquire);
        if ((g1 & 1u) != 0) return false;
        out.magic = b->magic;
        out.version = b->version;
        out.generation.store (g1, std::memory_order_relaxed);
        out.playing = b->playing;
        out.playheadBeats = b->playheadBeats;
        out.bpm = b->bpm;
        out.sampleTime = b->sampleTime;
        out.sampleRate = b->sampleRate;
        out.hostTimeNs = b->hostTimeNs;
        out.beatsPerBar = b->beatsPerBar;
        out.trackFeatureCount = std::min<uint32_t>(b->trackFeatureCount, kMaxTrackFeatureSlots);
        for (uint32_t i = 0; i < out.trackFeatureCount; ++i)
            out.trackFeatures[i] = b->trackFeatures[i];
        std::atomic_thread_fence (std::memory_order_acquire);
        const uint32_t g2 = b->generation.load (std::memory_order_acquire);
        if (g2 != g1) return false;
        out.generation.store (g2, std::memory_order_relaxed);
        return true;
    }

private:
    std::string name_;
    void* base_ = nullptr;
#if defined(_WIN32)
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
    bool owner_ = false;
#endif
};

//==============================================================================
// Audio sample ring (Block B live) — Arbit's audio thread pushes the mono
// master mix here once per audio block; the helper viewport drains it lock-free
// and runs the Block B analyzer to drive audio-reactive shaders in the live
// preview. Single-producer / single-consumer overwrite ring: the producer only
// ever appends, the consumer reads the trailing `capacity` window. Same
// ownership rules as TransportRegion: Arbit creates, helper opens.

static constexpr uint32_t kAudioMagic = 0x41564152;   // 'AVAR'
static constexpr uint32_t kAudioRingCapacity = 16384; // mono float samples (power of 2)

struct AudioRingBlock
{
    uint32_t magic;
    uint32_t version;
    uint32_t capacity;                  // == kAudioRingCapacity
    uint32_t sampleRate;                // device sample rate (Hz); 0 until first write
    std::atomic<uint64_t> writeCursor;  // total mono samples ever written (monotonic)
    float samples[kAudioRingCapacity];  // sample n lives at samples[n & (capacity-1)]
};

/** Maps (and optionally creates) the named audio-ring region. */
class AudioRingRegion
{
public:
    AudioRingRegion() = default;
    ~AudioRingRegion() { close(); }
    AudioRingRegion (const AudioRingRegion&) = delete;
    AudioRingRegion& operator= (const AudioRingRegion&) = delete;

    bool create (const std::string& name)
    {
        close();
        name_ = name;
#if defined(_WIN32)
        mapping_ = CreateFileMappingA (INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                       0, (DWORD) sizeof (AudioRingBlock), name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof (AudioRingBlock));
        if (base_ == nullptr) { close(); return false; }
#else
        const std::string posixName = "/" + name;
        shm_unlink (posixName.c_str());
        fd_ = shm_open (posixName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd_ < 0) return false;
        if (ftruncate (fd_, (off_t) sizeof (AudioRingBlock)) != 0) { close(); return false; }
        base_ = mmap (nullptr, sizeof (AudioRingBlock), PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
        owner_ = true;
#endif
        std::memset (base_, 0, sizeof (AudioRingBlock));
        block()->magic = kAudioMagic;
        block()->version = 1;
        block()->capacity = kAudioRingCapacity;
        return true;
    }

    bool open (const std::string& name)
    {
        close();
        name_ = name;
#if defined(_WIN32)
        mapping_ = OpenFileMappingA (FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (mapping_ == nullptr) return false;
        base_ = MapViewOfFile (mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (base_ == nullptr) { close(); return false; }
#else
        const std::string posixName = "/" + name;
        fd_ = shm_open (posixName.c_str(), O_RDWR, 0600);
        if (fd_ < 0) return false;
        base_ = mmap (nullptr, sizeof (AudioRingBlock), PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; close(); return false; }
#endif
        if (block()->magic != kAudioMagic) { close(); return false; }
        return true;
    }

    void close()
    {
#if defined(_WIN32)
        if (base_ != nullptr) UnmapViewOfFile (base_);
        if (mapping_ != nullptr) CloseHandle (mapping_);
        mapping_ = nullptr;
#else
        if (base_ != nullptr) munmap (base_, sizeof (AudioRingBlock));
        if (fd_ >= 0) ::close (fd_);
        if (owner_ && ! name_.empty()) shm_unlink (("/" + name_).c_str());
        fd_ = -1;
        owner_ = false;
#endif
        base_ = nullptr;
    }

    bool isOpen() const { return base_ != nullptr; }
    AudioRingBlock* block() const { return reinterpret_cast<AudioRingBlock*> (base_); }

    /** Writer (Arbit audio thread): wait-free. Downmixes stereo→mono inline;
        pass R == nullptr for a mono source. */
    void writeStereo (const float* L, const float* R, int n, double sampleRate)
    {
        auto* b = block();
        if (b == nullptr || L == nullptr || n <= 0) return;
        const uint64_t wc = b->writeCursor.load (std::memory_order_relaxed);
        for (int i = 0; i < n; ++i)
        {
            const float s = R != nullptr ? 0.5f * (L[i] + R[i]) : L[i];
            b->samples[(wc + (uint64_t) i) & (kAudioRingCapacity - 1)] = s;
        }
        b->sampleRate = (uint32_t) sampleRate;
        b->writeCursor.store (wc + (uint64_t) n, std::memory_order_release); // publish
    }

    /** Reader (helper render loop): copies up to maxOut new samples into out,
        advancing *cursor; returns the count copied (0 when drained). If the
        reader fell more than `capacity` behind, drops the oldest samples (a
        live preview can skip — continuity is best-effort). srOut, when non-null,
        receives the writer's reported sample rate. */
    int read (float* out, int maxOut, uint64_t* cursor, uint32_t* srOut) const
    {
        auto* b = block();
        if (b == nullptr || out == nullptr || maxOut <= 0 || cursor == nullptr) return 0;
        const uint64_t wc = b->writeCursor.load (std::memory_order_acquire);
        if (srOut != nullptr) *srOut = b->sampleRate;
        uint64_t rc = *cursor;
        if (rc > wc) rc = wc; // writer restarted (re-attach): resync forward
        uint64_t avail = wc - rc;
        if (avail > kAudioRingCapacity) { rc = wc - kAudioRingCapacity; avail = kAudioRingCapacity; }
        const uint64_t cap64 = (uint64_t) maxOut;
        const int n = (int) (avail < cap64 ? avail : cap64);
        for (int i = 0; i < n; ++i)
            out[i] = b->samples[(rc + (uint64_t) i) & (kAudioRingCapacity - 1)];
        *cursor = rc + (uint64_t) n;
        return n;
    }

    /** Current producer cursor (total samples written). The consumer can snap
        its read cursor to this to skip a stale backlog (e.g. when paused). */
    uint64_t writeCursor() const
    {
        auto* b = block();
        return b != nullptr ? b->writeCursor.load (std::memory_order_acquire) : 0;
    }

private:
    std::string name_;
    void* base_ = nullptr;
#if defined(_WIN32)
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
    bool owner_ = false;
#endif
};

} // namespace videoshm
