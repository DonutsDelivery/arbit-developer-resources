#include "gpu_export_iosurface.h"

#if ARBIT_HAVE_IOSURFACE

#include "SharedGpuSurfaceProtocol.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>

namespace gpuexp
{

namespace
{
// CFDictionary helper for IOSurfaceCreate.
void dictSetInt (CFMutableDictionaryRef dict, CFStringRef key, int32_t value)
{
    CFNumberRef num = CFNumberCreate (kCFAllocatorDefault, kCFNumberSInt32Type, &value);
    CFDictionarySetValue (dict, key, num);
    CFRelease (num);
}
} // namespace

bool IOSurfaceExporter::initialize (const arbitgl::GlFuncs* gl, std::string& errorOut)
{
    (void) gl;
    ready_ = false;
    errorOut.clear();
    ready_ = true;
    return true;
}

bool IOSurfaceExporter::allocate (ExportedBuffer& b, int width, int height,
                                  std::string& errorOut)
{
    if (! ready_)
    {
        errorOut = "exporter not initialized";
        return false;
    }

    b = ExportedBuffer {};
    b.width = width;
    b.height = height;

    CFMutableDictionaryRef props = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    dictSetInt (props, kIOSurfaceWidth, width);
    dictSetInt (props, kIOSurfaceHeight, height);
    dictSetInt (props, kIOSurfaceBytesPerElement, 4);
    dictSetInt (props, kIOSurfacePixelFormat, (int32_t) 'BGRA');
    // Global surfaces are the only ones IOSurfaceLookup can resolve from
    // another process; deprecated but standard (see header).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CFDictionarySetValue (props, kIOSurfaceIsGlobal, kCFBooleanTrue);
#pragma clang diagnostic pop

    IOSurfaceRef surf = IOSurfaceCreate (props);
    CFRelease (props);
    if (surf == nullptr)
    {
        errorOut = "IOSurfaceCreate failed";
        return false;
    }
    b.surface = (void*) surf;
    b.modifier = (uint64_t) IOSurfaceGetID (surf);
    b.fourcc = gpusurf::kHandleIOSurface;
    b.planeCount = 0;

    errorOut.clear();
    return true;
}

void IOSurfaceExporter::destroy (ExportedBuffer& b)
{
    if (b.surface != nullptr)
        CFRelease ((IOSurfaceRef) b.surface);
    b = ExportedBuffer {};
}

bool IOSurfaceExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    (void) srcTexture;
    (void) b;
    return false; // macOS shared presentation is direct Metal-only.
}

void IOSurfaceExporter::shutdown()
{
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_IOSURFACE
