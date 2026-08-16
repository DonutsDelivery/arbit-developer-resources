#include "rife_ncnn.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main (int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: rife-ncnn-inference-tests <model-directory> [--zero-copy]\n";
        return 2;
    }
    const bool requireZeroCopy = argc == 3 && std::string (argv[2]) == "--zero-copy";
    if (argc == 3 && ! requireZeroCopy)
        return 2;

#if defined(_WIN32)
    if (_putenv_s ("ARBIT_RIFE_NCNN_MODEL", argv[1]) != 0)
#else
    if (setenv ("ARBIT_RIFE_NCNN_MODEL", argv[1], 1) != 0)
#endif
    {
        std::cerr << "failed to set model directory\n";
        return 2;
    }

    arbitrife::RifeEngineNcnn engine;
    if (const auto error = engine.init(); ! error.empty())
    {
        std::cerr << "init failed: " << error << '\n';
        return 1;
    }
    if (engine.backend() != "rife-vulkan")
    {
        std::cerr << "unexpected backend: " << engine.backend() << '\n';
        return 1;
    }

    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int stride = width * 4;
    std::vector<uint8_t> first (static_cast<size_t> (stride * height), 0);
    std::vector<uint8_t> second (first.size(), 0);
    for (size_t i = 0; i < first.size(); i += 4)
    {
        first[i] = 255;
        first[i + 3] = 255;
        second[i + 2] = 255;
        second[i + 3] = 255;
    }

    std::vector<uint8_t> output;
    if (const auto error = engine.interpolate (first.data(), stride,
                                                second.data(), stride,
                                                width, height, 0.5f, output);
        ! error.empty())
    {
        std::cerr << "inference failed: " << error << '\n';
        return 1;
    }
    if (output.size() != first.size() || engine.inferenceCount() != 1
        || engine.totalInferenceMs() <= 0.0)
    {
        std::cerr << "invalid inference receipt\n";
        return 1;
    }
    if (output == first || output == second)
    {
        std::cerr << "inference returned an endpoint frame\n";
        return 1;
    }
    for (size_t i = 3; i < output.size(); i += 4)
    {
        if (output[i] != 255)
        {
            std::cerr << "non-opaque output alpha\n";
            return 1;
        }
    }

#ifdef ARBIT_RIFE_ZEROCOPY
    if (requireZeroCopy)
    {
        if (! engine.gpuZeroCopyAvailable())
        {
            std::cerr << "zero-copy unavailable\n";
            return 1;
        }
        arbitrife::GpuFrameHandle gpuOutput;
        if (const auto error = engine.interpolateToGpu (first.data(), stride,
                                                        second.data(), stride,
                                                        width, height, 0.5f, gpuOutput);
            ! error.empty())
        {
            std::cerr << "zero-copy inference failed: " << error << '\n';
            return 1;
        }
        if (! gpuOutput.valid || gpuOutput.slot < 0 || gpuOutput.memFd < 0
            || gpuOutput.allocSize < static_cast<size_t> (width * height * 3)
            || gpuOutput.w != width || gpuOutput.h != height
            || engine.inferenceCount() != 2)
        {
            std::cerr << "invalid zero-copy inference receipt\n";
            return 1;
        }
    }
#else
    if (requireZeroCopy)
    {
        std::cerr << "test was built without zero-copy support\n";
        return 1;
    }
#endif

    std::cout << "RIFE ncnn inference: PASS (" << engine.totalInferenceMs() << " ms)\n";
    return 0;
}
