#include "rife_ncnn.h"

#ifdef ARBIT_HAVE_NCNN

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
 #include <unistd.h>          // readlink (/proc/self/exe)
#elif defined(__APPLE__)
 #include <mach-o/dyld.h>     // _NSGetExecutablePath
 #include <vector>
#elif defined(_WIN32)
 #include <windows.h>         // GetModuleFileNameW
#endif

#include "model_fetch.h"   // SHA-256 + libavformat HTTPS download (auto-fetch)

// ncnn
#include "gpu.h"
#include "mat.h"
#include "benchmark.h"

// vendored nihui RIFE net (MIT)
#include "rife_net/rife.h"

namespace arbitrife
{

namespace
{
    // Directory holding the running helper binary (for the bundled-model lookup).
    // Cross-platform so a model shipped next to the helper is found on every OS.
    std::filesystem::path exeDir()
    {
#if defined(__linux__)
        char buf[4096];
        const ssize_t n = ::readlink ("/proc/self/exe", buf, sizeof buf - 1);
        if (n > 0) { buf[(size_t) n] = '\0'; return std::filesystem::path (buf).parent_path(); }
#elif defined(__APPLE__)
        uint32_t sz = 0;
        _NSGetExecutablePath (nullptr, &sz);          // query length
        std::vector<char> buf (sz + 1, '\0');
        if (_NSGetExecutablePath (buf.data(), &sz) == 0)
        {
            std::error_code ec;
            auto p = std::filesystem::canonical (buf.data(), ec); // resolve symlinks
            return (ec ? std::filesystem::path (buf.data()) : p).parent_path();
        }
#elif defined(_WIN32)
        wchar_t buf[MAX_PATH];
        const DWORD n = ::GetModuleFileNameW (nullptr, buf, MAX_PATH);
        if (n > 0 && n < MAX_PATH) return std::filesystem::path (buf).parent_path();
#endif
        return {};
    }

    // Per-user data dir Arbit installs the helper model under, by OS convention.
    std::filesystem::path userDataModelRoot()
    {
#if defined(_WIN32)
        if (const char* p = std::getenv ("LOCALAPPDATA"); p && p[0])
            return std::filesystem::path (p) / "Arbit" / "video-helper" / "models";
#elif defined(__APPLE__)
        if (const char* h = std::getenv ("HOME"); h && h[0])
            return std::filesystem::path (h) / "Library" / "Application Support"
                       / "Arbit" / "video-helper" / "models";
#else
        if (const char* x = std::getenv ("XDG_DATA_HOME"); x && x[0])
            return std::filesystem::path (x) / "Arbit" / "video-helper" / "models";
        if (const char* h = std::getenv ("HOME"); h && h[0])
            return std::filesystem::path (h) / ".local/share/Arbit/video-helper/models";
#endif
        return {};
    }

    bool hasModel (const std::filesystem::path& dir)
    {
        std::error_code ec;
        return ! dir.empty()
            && std::filesystem::exists (dir / "flownet.param", ec)
            && std::filesystem::exists (dir / "flownet.bin", ec);
    }

    // Resolve a RIFE v4 ncnn model directory (one holding flownet.param +
    // flownet.bin). Order: explicit env override → bundled next to the helper
    // (models/rife-v4.6) → per-user data/cache dirs. "" if none found. This is
    // the no-env-var default path (1a): a normal launch finds the bundled model.
    std::string resolveRifeModelDir()
    {
        if (const char* env = std::getenv ("ARBIT_RIFE_NCNN_MODEL");
            env != nullptr && env[0] != '\0')
            return env; // override wins; an incomplete dir surfaces as a load error

        std::vector<std::filesystem::path> candidates;
        if (const auto ed = exeDir(); ! ed.empty())
        {
            // Bundled next to the helper — the canonical install layout on all OSes
            // (Linux dir / .app Resources / Windows install dir).
            candidates.push_back (ed / "models" / "rife-v4.6");
            candidates.push_back (ed / "models" / "rife");
        }
        if (const auto ud = userDataModelRoot(); ! ud.empty())
            candidates.push_back (ud / "rife-v4.6");
#if defined(__linux__)
        if (const char* home = std::getenv ("HOME"); home && home[0])
            candidates.push_back (std::filesystem::path (home)
                                  / ".cache/Arbit/video-helper/models/rife-v4.6"); // dev convenience
#endif
        for (const auto& c : candidates)
            if (hasModel (c))
                return c.string();
        return {};
    }

    // RIFE v4.6 ncnn model (nihui conversion of hzwer/Practical-RIFE, MIT),
    // hosted alongside the ONNX model. Auto-downloaded on first use when not
    // bundled — same proven libavformat path the ONNX engine uses (rife.cpp).
    struct RifeFile { const char* name; const char* url; const char* sha256; int64_t bytes; };
    const RifeFile kRifeFiles[] = {
        { "flownet.param",
          "https://raw.githubusercontent.com/nihui/rife-ncnn-vulkan/a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7/models/rife-v4.6/flownet.param",
          "724569596bcd1e7b9fa50455c604777ebed99746d2ef40aa86e31b5725f1053c", 16532 },
        { "flownet.bin",
          "https://raw.githubusercontent.com/nihui/rife-ncnn-vulkan/a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7/models/rife-v4.6/flownet.bin",
          "f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958", 10614320 },
    };

    // Fetch the model into the per-user data dir (the resolver's userDataModelRoot)
    // if absent/corrupt, verifying each file's SHA-256. Returns "" + sets dirOut on
    // success. Runs on the interp worker thread (off the render thread), so the
    // blocking download just delays RIFE availability — Frame Blend covers until then.
    std::string ensureRifeModel (std::string& dirOut)
    {
        namespace fs = std::filesystem;
        const fs::path dir = userDataModelRoot() / "rife-v4.6";
        if (dir.empty())
            return "no writable model dir (HOME/LOCALAPPDATA unset)";
        std::error_code ec;
        fs::create_directories (dir, ec);

        for (const auto& f : kRifeFiles)
        {
            const fs::path dest = dir / f.name;
            if (fs::exists (dest, ec) && arbitmodelfetch::sha256File (dest) == f.sha256)
                continue; // already have a good copy
            std::fprintf (stderr, "[rife-ncnn] downloading %s (%lld bytes)\n",
                          f.name, (long long) f.bytes);
            const fs::path tmp = dir / (std::string (f.name) + ".part");
            if (auto err = arbitmodelfetch::downloadFile (f.url, tmp, f.bytes * 2 + 4096);
                ! err.empty())
            {
                fs::remove (tmp, ec);
                return std::string ("RIFE model download failed: ") + err;
            }
            if (arbitmodelfetch::sha256File (tmp) != f.sha256)
            {
                fs::remove (tmp, ec);
                return std::string ("RIFE model sha256 mismatch (corrupt/tampered): ") + f.name;
            }
            fs::rename (tmp, dest, ec);
            if (ec) return "cannot move RIFE model into place: " + ec.message();
        }
        dirOut = dir.string();
        return "";
    }
} // namespace

// ncnn::create_gpu_instance() is process-global and not reference-counted, but
// the helper may construct more than one engine (interp worker + exporter). Guard
// with a refcount so the instance is created once and destroyed when the last
// engine goes away.
namespace
{
std::mutex g_gpuMutex;
int        g_gpuRefcount = 0;

bool acquireGpuInstance()
{
    std::lock_guard<std::mutex> lock (g_gpuMutex);
    if (g_gpuRefcount == 0)
    {
        if (ncnn::create_gpu_instance() != 0)
            return false;
    }
    ++g_gpuRefcount;
    return true;
}

void releaseGpuInstance()
{
    std::lock_guard<std::mutex> lock (g_gpuMutex);
    if (g_gpuRefcount > 0 && --g_gpuRefcount == 0)
        ncnn::destroy_gpu_instance();
}
} // namespace

#ifdef ARBIT_RIFE_ZEROCOPY
namespace
{
// A small pool of OPAQUE_FD-exportable, DEVICE_LOCAL VkBuffers on ncnn's VkDevice,
// each holding one interleaved-RGB8 frame. Allocated once per frame size; each
// slot's memory fd is exported once (the render thread imports each slot once and
// caches its GL buffer). Round-robin slot reuse; the InterpEngine cache tracks
// which (clip,bucket) currently owns each slot.
struct ExportableRing
{
    static constexpr int kSlots = 32;
    struct Slot { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; int memFd = -1; size_t allocSize = 0; };

    ncnn::VulkanDevice* vkdev = nullptr;
    Slot   slots[kSlots];
    int    next = 0;
    int    w = 0, h = 0;
    VkBuffer       stageBuf = VK_NULL_HANDLE;   // host-visible staging for uploadFrameToGpu
    VkDeviceMemory stageMem = VK_NULL_HANDLE;
    size_t         stageCap = 0;

    uint32_t pickMem (uint32_t bits, VkMemoryPropertyFlags want) const
    {
        const VkPhysicalDeviceMemoryProperties& mp = vkdev->info.physicalDeviceMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    }

    bool ready() const { return vkdev != nullptr && slots[0].buf != VK_NULL_HANDLE; }
    int  acquire()     { int i = next; next = (next + 1) % kSlots; return i; }

    bool ensure (ncnn::VulkanDevice* dev, int W, int H)
    {
        if (ready() && dev == vkdev && W == w && H == h) return true;
        destroy();
        vkdev = dev; w = W; h = H;
        VkDevice d = vkdev->vkdevice();
        auto pGetFd = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr (d, "vkGetMemoryFdKHR");
        if (! pGetFd) { vkdev = nullptr; return false; }
        const size_t bytes = (size_t) W * H * 3;
        for (int i = 0; i < kSlots; i++)
        {
            Slot& s = slots[i];
            VkExternalMemoryBufferCreateInfo eb {}; eb.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
            eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkBufferCreateInfo bci {}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.pNext = &eb; bci.size = bytes;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer (d, &bci, nullptr, &s.buf) != VK_SUCCESS) { destroy(); return false; }
            VkMemoryRequirements req; vkGetBufferMemoryRequirements (d, s.buf, &req);
            VkExportMemoryAllocateInfo em {}; em.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
            em.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryDedicatedAllocateInfo de {}; de.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            de.buffer = s.buf; de.pNext = &em;
            VkMemoryAllocateInfo mai {}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.pNext = &de; mai.allocationSize = req.size;
            mai.memoryTypeIndex = pickMem (req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory (d, &mai, nullptr, &s.mem) != VK_SUCCESS) { destroy(); return false; }
            vkBindBufferMemory (d, s.buf, s.mem, 0);
            s.allocSize = (size_t) req.size;
            VkMemoryGetFdInfoKHR gfi {}; gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            gfi.memory = s.mem; gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            if (pGetFd (d, &gfi, &s.memFd) != VK_SUCCESS || s.memFd < 0) { destroy(); return false; }
        }
        next = 0;
        return true;
    }

    bool ensureStage (size_t bytes)
    {
        if (stageBuf != VK_NULL_HANDLE && stageCap >= bytes) return true;
        VkDevice d = vkdev->vkdevice();
        if (stageBuf) { vkDestroyBuffer (d, stageBuf, nullptr); stageBuf = VK_NULL_HANDLE; }
        if (stageMem) { vkFreeMemory (d, stageMem, nullptr); stageMem = VK_NULL_HANDLE; }
        VkBufferCreateInfo bci {}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer (d, &bci, nullptr, &stageBuf) != VK_SUCCESS) return false;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements (d, stageBuf, &req);
        VkMemoryAllocateInfo mai {}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = pickMem (req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory (d, &mai, nullptr, &stageMem) != VK_SUCCESS) return false;
        vkBindBufferMemory (d, stageBuf, stageMem, 0);
        stageCap = (size_t) req.size;
        return true;
    }

    bool copyStageToSlot (int slot, size_t bytes)
    {
        VkDevice d = vkdev->vkdevice();
        const uint32_t qfi = vkdev->info.compute_queue_family_index();
        VkCommandPoolCreateInfo pci {}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; pci.queueFamilyIndex = qfi;
        VkCommandPool pool = VK_NULL_HANDLE;
        if (vkCreateCommandPool (d, &pci, nullptr, &pool) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo cbi {}; cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE; vkAllocateCommandBuffers (d, &cbi, &cb);
        VkCommandBufferBeginInfo bi {}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer (cb, &bi);
        VkBufferCopy region {}; region.size = bytes;
        vkCmdCopyBuffer (cb, stageBuf, slots[slot].buf, 1, &region);
        vkEndCommandBuffer (cb);
        VkQueue q = vkdev->acquire_queue (qfi);
        VkSubmitInfo si {}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit (q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle (q);
        vkdev->reclaim_queue (qfi, q);
        vkDestroyCommandPool (d, pool, nullptr);
        return true;
    }

    void destroy()
    {
        if (vkdev)
        {
            VkDevice d = vkdev->vkdevice();
            for (auto& s : slots)
            {
                if (s.buf) vkDestroyBuffer (d, s.buf, nullptr);
                if (s.mem) vkFreeMemory (d, s.mem, nullptr);
                // s.memFd ownership transfers to GL on import; not closed here (a
                // <=kSlots fd leak at process teardown, harmless).
                s = Slot {};
            }
            if (stageBuf) vkDestroyBuffer (d, stageBuf, nullptr);
            if (stageMem) vkFreeMemory (d, stageMem, nullptr);
        }
        stageBuf = VK_NULL_HANDLE; stageMem = VK_NULL_HANDLE; stageCap = 0;
        vkdev = nullptr; w = 0; h = 0; next = 0;
    }
};
} // namespace
#endif // ARBIT_RIFE_ZEROCOPY

struct RifeEngineNcnn::Impl
{
    std::unique_ptr<RIFE> rife;
    bool gpuAcquired = false;
#ifdef ARBIT_RIFE_ZEROCOPY
    ExportableRing ring;
#endif

    ~Impl()
    {
#ifdef ARBIT_RIFE_ZEROCOPY
        ring.destroy();            // free exportable buffers while the device is alive
#endif
        rife.reset();              // destroy the net (and its pipelines) first
        if (gpuAcquired)
            releaseGpuInstance();
    }
};

RifeEngineNcnn::RifeEngineNcnn() : impl_ (std::make_unique<Impl>()) {}
RifeEngineNcnn::~RifeEngineNcnn() = default;

std::string RifeEngineNcnn::init()
{
    // Test hook: simulate a machine with no usable Vulkan so the HUD's
    // distro-specific install hint can be verified without removing libvulkan.
    if (const char* t = std::getenv ("ARBIT_TEST_VULKAN_FAIL"); t && *t && *t != '0')
        return "ncnn: no Vulkan GPU available (test)";

    std::string modelDir = resolveRifeModelDir();
    if (modelDir.empty())
    {
        // Not present locally — auto-fetch it into the per-user data dir (like
        // the ONNX path). Blocks this worker thread; Frame Blend covers meanwhile.
        if (auto err = ensureRifeModel (modelDir); ! err.empty())
            return err; // surfaced in the HUD ("RIFE: <reason>")
    }

    if (! acquireGpuInstance())
        return "ncnn: failed to create Vulkan instance (no conformant driver?)";
    impl_->gpuAcquired = true;

    if (ncnn::get_gpu_count() <= 0)
        return "ncnn: no Vulkan GPU available";

    const int gpuid = ncnn::get_default_gpu_index();

    // RIFE v4: arbitrary-timestep, single flownet (no context/fusion nets).
    impl_->rife = std::make_unique<RIFE> (gpuid,
                                          /*tta*/ false,
                                          /*tta_temporal*/ false,
                                          /*uhd*/ false,
                                          /*num_threads*/ 1,
                                          /*rife_v2*/ false,
                                          /*rife_v4*/ true);

    // ncnn (and the vendored RIFE net) take wide paths on Windows; widen the
    // UTF-8 model dir via filesystem so non-ASCII user paths load correctly.
#if defined(_WIN32)
    const int loadRc = impl_->rife->load (std::filesystem::u8path (modelDir).wstring());
#else
    const int loadRc = impl_->rife->load (modelDir);
#endif
    if (loadRc != 0)
    {
        impl_->rife.reset();
        return std::string ("ncnn: failed to load RIFE model from ") + modelDir;
    }

    backend_ = "rife-vulkan";
    return "";
}

std::string RifeEngineNcnn::interpolate (const uint8_t* rgba0, int stride0,
                                         const uint8_t* rgba1, int stride1,
                                         int width, int height, float timestep,
                                         std::vector<uint8_t>& rgbaOut)
{
    if (impl_->rife == nullptr)
        return "ncnn: engine not initialized";
    if (width <= 0 || height <= 0)
        return "ncnn: invalid frame size";

    // The nihui RIFE net consumes/produces tightly-packed straight-RGB
    // (ncnn::Mat elemsize 3, elempack 3). Drop alpha on the way in, restore it
    // opaque on the way out. (This RGBA<->RGB copy is the only CPU touch the
    // zero-copy VkImage path will later remove.)
    const size_t rgbStride = (size_t) width * 3;
    std::vector<uint8_t> rgb0 ((size_t) height * rgbStride);
    std::vector<uint8_t> rgb1 ((size_t) height * rgbStride);

    for (int y = 0; y < height; ++y)
    {
        const uint8_t* s0 = rgba0 + (size_t) y * stride0;
        const uint8_t* s1 = rgba1 + (size_t) y * stride1;
        uint8_t* d0 = rgb0.data() + (size_t) y * rgbStride;
        uint8_t* d1 = rgb1.data() + (size_t) y * rgbStride;
        for (int x = 0; x < width; ++x)
        {
            d0[0] = s0[0]; d0[1] = s0[1]; d0[2] = s0[2];
            d1[0] = s1[0]; d1[1] = s1[1]; d1[2] = s1[2];
            d0 += 3; d1 += 3; s0 += 4; s1 += 4;
        }
    }

    ncnn::Mat in0 (width, height, (void*) rgb0.data(), (size_t) 3, 3);
    ncnn::Mat in1 (width, height, (void*) rgb1.data(), (size_t) 3, 3);
    ncnn::Mat out (width, height, (size_t) 3, 3);

    const double t0 = ncnn::get_current_time();
    const int ret = impl_->rife->process (in0, in1, timestep, out);
    inferMsTotal_ += ncnn::get_current_time() - t0;
    ++inferCount_;

    if (ret != 0 || out.empty())
        return "ncnn: RIFE process failed";

    rgbaOut.resize ((size_t) width * height * 4);
    const uint8_t* src = (const uint8_t*) out.data;
    uint8_t* dst = rgbaOut.data();
    for (int i = 0, n = width * height; i < n; ++i)
    {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = 255;
        dst += 4; src += 3;
    }
    return "";
}

#ifdef ARBIT_RIFE_ZEROCOPY
bool RifeEngineNcnn::gpuZeroCopyAvailable() const
{
    if (impl_->rife == nullptr) return false;
    ncnn::VulkanDevice* vkdev = impl_->rife->vulkanDevice();
    if (vkdev == nullptr) return false;
    return (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr (vkdev->vkdevice(), "vkGetMemoryFdKHR") != nullptr;
}

std::string RifeEngineNcnn::interpolateToGpu (const uint8_t* rgba0, int stride0,
                                              const uint8_t* rgba1, int stride1,
                                              int width, int height, float timestep,
                                              GpuFrameHandle& out)
{
    out = GpuFrameHandle {};
    if (impl_->rife == nullptr) return "ncnn: engine not initialized";
    if (width <= 0 || height <= 0) return "ncnn: invalid frame size";
    ncnn::VulkanDevice* vkdev = impl_->rife->vulkanDevice();
    if (vkdev == nullptr) return "ncnn: no vulkan device";
    if (! impl_->ring.ensure (vkdev, width, height)) return "ncnn: zero-copy ring init failed";

    // Same RGBA->RGB drop as interpolate() (decode/input stays CPU; only the OUTPUT
    // download is eliminated by the zero-copy path).
    const size_t rgbStride = (size_t) width * 3;
    std::vector<uint8_t> rgb0 ((size_t) height * rgbStride);
    std::vector<uint8_t> rgb1 ((size_t) height * rgbStride);
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* s0 = rgba0 + (size_t) y * stride0;
        const uint8_t* s1 = rgba1 + (size_t) y * stride1;
        uint8_t* d0 = rgb0.data() + (size_t) y * rgbStride;
        uint8_t* d1 = rgb1.data() + (size_t) y * rgbStride;
        for (int x = 0; x < width; ++x)
        {
            d0[0] = s0[0]; d0[1] = s0[1]; d0[2] = s0[2];
            d1[0] = s1[0]; d1[1] = s1[1]; d1[2] = s1[2];
            d0 += 3; d1 += 3; s0 += 4; s1 += 4;
        }
    }

    ncnn::Mat in0 (width, height, (void*) rgb0.data(), (size_t) 3, 3);
    ncnn::Mat in1 (width, height, (void*) rgb1.data(), (size_t) 3, 3);

    const int slot = impl_->ring.acquire();
    const ExportableRing::Slot& s = impl_->ring.slots[slot];
    int ow = 0, oh = 0;
    const double t0 = ncnn::get_current_time();
    const int ret = impl_->rife->process_v4_to_buffer (in0, in1, timestep, s.buf, s.allocSize, ow, oh);
    inferMsTotal_ += ncnn::get_current_time() - t0;
    ++inferCount_;
    if (ret != 0 || ow != width || oh != height) return "ncnn: process_v4_to_buffer failed";

    out.slot = slot; out.memFd = s.memFd; out.allocSize = s.allocSize;
    out.w = width; out.h = height; out.valid = true;
    return "";
}

std::string RifeEngineNcnn::uploadFrameToGpu (const uint8_t* rgba, int stride,
                                              int width, int height, GpuFrameHandle& out)
{
    out = GpuFrameHandle {};
    if (impl_->rife == nullptr) return "ncnn: engine not initialized";
    if (width <= 0 || height <= 0) return "ncnn: invalid frame size";
    ncnn::VulkanDevice* vkdev = impl_->rife->vulkanDevice();
    if (vkdev == nullptr) return "ncnn: no vulkan device";
    if (! impl_->ring.ensure (vkdev, width, height)) return "ncnn: zero-copy ring init failed";

    const size_t bytes = (size_t) width * height * 3;
    if (! impl_->ring.ensureStage (bytes)) return "ncnn: staging init failed";

    VkDevice d = vkdev->vkdevice();
    void* mp = nullptr;
    if (vkMapMemory (d, impl_->ring.stageMem, 0, bytes, 0, &mp) != VK_SUCCESS) return "ncnn: stage map failed";
    uint8_t* dst = (uint8_t*) mp;
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* src = rgba + (size_t) y * stride;
        for (int x = 0; x < width; ++x)
        {
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            dst += 3; src += 4;
        }
    }
    vkUnmapMemory (d, impl_->ring.stageMem);

    const int slot = impl_->ring.acquire();
    if (! impl_->ring.copyStageToSlot (slot, bytes)) return "ncnn: stage->slot copy failed";

    const ExportableRing::Slot& s = impl_->ring.slots[slot];
    out.slot = slot; out.memFd = s.memFd; out.allocSize = s.allocSize;
    out.w = width; out.h = height; out.valid = true;
    return "";
}
#endif // ARBIT_RIFE_ZEROCOPY

} // namespace arbitrife

#endif // ARBIT_HAVE_NCNN
