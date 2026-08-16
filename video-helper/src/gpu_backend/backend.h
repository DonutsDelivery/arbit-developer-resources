// gpu_backend/backend.h -- language-neutral native GPU backend seam.
//
// The existing FrameRenderer remains on OpenGL while the visual-engine P6
// migration lands in slices.  This seam makes native-backend capability and
// verification available without leaking Objective-C types into the helper.
#pragma once

#include <cstdint>
#include <string>

namespace arbitgpu
{

struct BackendInfo
{
    bool available = false;
    bool compute = false;
    std::string backend;
    std::string device;
    std::string error;
};

struct BackendSelfTest : BackendInfo
{
    bool computePassed = false;
    bool renderPassed = false;
    uint32_t computeChecksum = 0;
    uint32_t renderChecksum = 0;
};

// Cheap device/capability query.  Does not initialize sokol_gfx or submit GPU
// work, so it is safe to expose in the regular version/capability RPCs.
BackendInfo queryNativeBackend();

// Native compute + offscreen-render validation. The P6 backend stays alive for
// the helper process because renderer-facing Metal resources now share it.
BackendSelfTest runNativeBackendSelfTest();

} // namespace arbitgpu
