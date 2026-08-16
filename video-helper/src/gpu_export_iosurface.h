// gpu_export_iosurface.h — IOSurfaceExporter: allocates IOSurface-backed
// native Metal render targets and exposes their global IOSurfaceIDs
// for the zero-copy docked viewport (SharedGpuSurfaceProtocol.h, macOS).
//
// IOSurfaces created with kIOSurfaceIsGlobal can be opened in any process of
// the same user session via IOSurfaceLookup(id) — the ID is a plain uint32,
// so unlike dmabuf no fd/port passing is needed; it rides in
// BufferInfo.modifier (fourcc = gpusurf::kHandleIOSurface, planeCount = 0).
// kIOSurfaceIsGlobal is deprecated (the blessed replacement is mach-port
// transfer, which our unix-socket channel cannot carry) but remains the
// standard cross-process texture path and works through macOS 15.
//
// The compositor creates an MTLTexture view over each surface and renders into
// it directly. There is deliberately no CGL import or blit fallback.
#pragma once

#if ARBIT_HAVE_IOSURFACE

#include "gl_loader.h"

#include <cstdint>
#include <string>

namespace gpuexp
{

struct ExportedBuffer
{
    unsigned tex = 0, fbo = 0;    // unused on native Metal
    int width = 0, height = 0;

    uint32_t fourcc = 0;          // gpusurf::kHandleIOSurface
    uint64_t modifier = 0;        // global IOSurfaceID
    int planeCount = 0;           // always 0 (no fds on this transport)
    int fds[4] = { -1, -1, -1, -1 };
    uint32_t strides[4] = {}, offsets[4] = {};

    void* surface = nullptr;      // IOSurfaceRef (owned)

    // Announced via FRAME_READY and not yet FRAME_RELEASEd by the consumer.
    // Managed by the viewport render loop, not by the exporter.
    bool busy = false;
};

class IOSurfaceExporter
{
public:
    IOSurfaceExporter() = default;
    ~IOSurfaceExporter() { shutdown(); }
    IOSurfaceExporter (const IOSurfaceExporter&) = delete;
    IOSurfaceExporter& operator= (const IOSurfaceExporter&) = delete;

    // gl is retained in the cross-platform signature but unused on macOS.
    bool initialize (const arbitgl::GlFuncs* gl, std::string& errorOut);
    bool available() const { return ready_; }

    // Apple Silicon exposes one unified Metal device for this process.
    const std::string& devicePath() const { return devicePath_; }

    // Allocates one w*h BGRA IOSurface for direct Metal rendering.
    bool allocate (ExportedBuffer& b, int width, int height, std::string& errorOut);
    void destroy (ExportedBuffer& b);

    // Unsupported by design: the shared macOS path renders directly.
    bool blit (unsigned srcTexture, const ExportedBuffer& b);

    bool fenceSyncAvailable() const { return false; } // v1 sync only
    int createNativeFenceFd() { return -1; }

    void shutdown();

private:
    bool ready_ = false;
    std::string devicePath_;      // always empty
};

} // namespace gpuexp

#endif // ARBIT_HAVE_IOSURFACE
