#pragma once

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND && ARBIT_HAVE_IOSURFACE

#include <memory>
#include <string>

class MacMetalViewportSurface
{
public:
    MacMetalViewportSurface();
    ~MacMetalViewportSurface();

    MacMetalViewportSurface (const MacMetalViewportSurface&) = delete;
    MacMetalViewportSurface& operator= (const MacMetalViewportSurface&) = delete;

    static bool available();

    bool initialize (void* glfwWindow, int width, int height,
                     bool onscreen, std::string& error);
    bool resize (int width, int height, std::string& error);
    bool present (std::string& error);

    void* ioSurface() const;
    int width() const;
    int height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
