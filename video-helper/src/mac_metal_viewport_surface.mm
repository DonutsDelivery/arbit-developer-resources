#include "mac_metal_viewport_surface.h"

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND && ARBIT_HAVE_IOSURFACE

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Cocoa/Cocoa.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>

namespace
{
void setDictionaryInt (CFMutableDictionaryRef properties, CFStringRef key, int value)
{
    CFNumberRef number = CFNumberCreate (
        kCFAllocatorDefault, kCFNumberIntType, &value);
    CFDictionarySetValue (properties, key, number);
    CFRelease (number);
}
}

struct MacMetalViewportSurface::Impl
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLTexture> surfaceTexture = nil;
    CAMetalLayer* layer = nil;
    IOSurfaceRef surface = nullptr;
    int surfaceWidth = 0;
    int surfaceHeight = 0;
    bool onscreen = false;

    ~Impl()
    {
#if ! __has_feature(objc_arc)
        [surfaceTexture release];
#endif
        surfaceTexture = nil;
        if (surface != nullptr)
            CFRelease (surface);
#if ! __has_feature(objc_arc)
        [layer release];
        [queue release];
        [device release];
#endif
    }

    bool allocate (int width, int height, std::string& error)
    {
        width = std::max (width, 1);
        height = std::max (height, 1);
        if (surface != nullptr && surfaceWidth == width && surfaceHeight == height)
            return true;

#if ! __has_feature(objc_arc)
        [surfaceTexture release];
#endif
        surfaceTexture = nil;
        if (surface != nullptr)
        {
            CFRelease (surface);
            surface = nullptr;
        }

        CFMutableDictionaryRef properties = CFDictionaryCreateMutable (
            kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks);
        setDictionaryInt (properties, kIOSurfaceWidth, width);
        setDictionaryInt (properties, kIOSurfaceHeight, height);
        setDictionaryInt (properties, kIOSurfaceBytesPerElement, 4);
        setDictionaryInt (properties, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
        surface = IOSurfaceCreate (properties);
        CFRelease (properties);
        if (surface == nullptr)
        {
            error = "Metal viewport IOSurface allocation failed";
            return false;
        }

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:width height:height mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        surfaceTexture = [device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
        if (surfaceTexture == nil)
        {
            error = "Metal viewport IOSurface texture creation failed";
            return false;
        }
        surfaceWidth = width;
        surfaceHeight = height;
        if (layer != nil)
            layer.drawableSize = CGSizeMake (width, height);
        error.clear();
        return true;
    }
};

MacMetalViewportSurface::MacMetalViewportSurface() : impl_ (std::make_unique<Impl>()) {}
MacMetalViewportSurface::~MacMetalViewportSurface() = default;

bool MacMetalViewportSurface::available()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const bool result = device != nil;
#if ! __has_feature(objc_arc)
        [device release];
#endif
        return result;
    }
}

bool MacMetalViewportSurface::initialize (void* glfwWindow, int width, int height,
                                          bool onscreen, std::string& error)
{
    @autoreleasepool
    {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device == nil)
        {
            error = "Metal returned no default device";
            return false;
        }
        impl_->queue = [impl_->device newCommandQueue];
        if (impl_->queue == nil)
        {
            error = "Metal viewport command queue creation failed";
            return false;
        }
        impl_->onscreen = onscreen;

        if (onscreen)
        {
            auto* window = static_cast<GLFWwindow*> (glfwWindow);
            NSWindow* cocoaWindow = window != nullptr ? glfwGetCocoaWindow (window) : nil;
            NSView* contentView = cocoaWindow.contentView;
            if (contentView == nil)
            {
                error = "Metal viewport has no Cocoa content view";
                return false;
            }
            impl_->layer = [CAMetalLayer layer];
#if ! __has_feature(objc_arc)
            [impl_->layer retain];
#endif
            impl_->layer.device = impl_->device;
            impl_->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
            // Presentation copies the compositor's IOSurface texture into the
            // drawable with a Metal blit, so the drawable cannot be restricted
            // to render-pass attachment use.
            impl_->layer.framebufferOnly = NO;
            impl_->layer.opaque = YES;
            impl_->layer.contentsScale = cocoaWindow.backingScaleFactor;
            contentView.wantsLayer = YES;
            contentView.layer = impl_->layer;
        }
        return impl_->allocate (width, height, error);
    }
}

bool MacMetalViewportSurface::resize (int width, int height, std::string& error)
{
    @autoreleasepool { return impl_->allocate (width, height, error); }
}

bool MacMetalViewportSurface::present (std::string& error)
{
    @autoreleasepool
    {
        if (! impl_->onscreen)
            return true;
        if (impl_->layer == nil || impl_->surfaceTexture == nil)
        {
            error = "Metal viewport presentation surface is unavailable";
            return false;
        }
        id<CAMetalDrawable> drawable = [impl_->layer nextDrawable];
        if (drawable == nil)
        {
            error = "Metal viewport returned no drawable";
            return false;
        }
        if (drawable.texture.width != static_cast<NSUInteger> (impl_->surfaceWidth)
            || drawable.texture.height != static_cast<NSUInteger> (impl_->surfaceHeight))
        {
            impl_->layer.drawableSize = CGSizeMake (impl_->surfaceWidth, impl_->surfaceHeight);
            drawable = [impl_->layer nextDrawable];
            if (drawable == nil
                || drawable.texture.width != static_cast<NSUInteger> (impl_->surfaceWidth)
                || drawable.texture.height != static_cast<NSUInteger> (impl_->surfaceHeight))
            {
                error = "Metal viewport drawable size mismatch";
                return false;
            }
        }

        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        const MTLSize size = MTLSizeMake (impl_->surfaceWidth, impl_->surfaceHeight, 1);
        [blit copyFromTexture:impl_->surfaceTexture sourceSlice:0 sourceLevel:0
                sourceOrigin:MTLOriginMake (0, 0, 0) sourceSize:size
                   toTexture:drawable.texture destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake (0, 0, 0)];
        [blit endEncoding];
        [command presentDrawable:drawable];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError)
        {
            error = command.error.localizedDescription.UTF8String ?: "Metal viewport presentation failed";
            return false;
        }
        error.clear();
        return true;
    }
}

void* MacMetalViewportSurface::ioSurface() const { return impl_->surface; }
int MacMetalViewportSurface::width() const { return impl_->surfaceWidth; }
int MacMetalViewportSurface::height() const { return impl_->surfaceHeight; }

#endif
