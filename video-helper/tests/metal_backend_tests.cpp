#include "gpu_backend/backend.h"

#include <cstdio>

int main()
{
    const auto info = arbitgpu::queryNativeBackend();
    if (! info.available || info.backend != "metal" || ! info.compute)
    {
        std::fprintf (stderr, "Metal capability probe failed: %s\n", info.error.c_str());
        return 1;
    }

    const auto test = arbitgpu::runNativeBackendSelfTest();
    if (! test.available || ! test.computePassed || ! test.renderPassed)
    {
        std::fprintf (stderr, "Metal self-test failed: %s\n", test.error.c_str());
        return 2;
    }

    std::printf ("backend=%s device=%s compute=%u render=%u\n",
                 test.backend.c_str(), test.device.c_str(),
                 test.computeChecksum, test.renderChecksum);
    return 0;
}
