#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

int main (int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: metal-iosurface-consumer ID WIDTH HEIGHT\n";
        return 2;
    }

    const auto surfaceId = static_cast<IOSurfaceID> (std::strtoul (argv[1], nullptr, 10));
    const auto width = static_cast<NSUInteger> (std::strtoul (argv[2], nullptr, 10));
    const auto height = static_cast<NSUInteger> (std::strtoul (argv[3], nullptr, 10));
    IOSurfaceRef surface = IOSurfaceLookup (surfaceId);
    if (surface == nullptr || width == 0 || height == 0)
    {
        std::cerr << "consumer could not look up IOSurface " << surfaceId << '\n';
        return 3;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor
                                                    iosurface:surface plane:0];
    id<MTLBuffer> readback = [device newBufferWithLength:256
                                                 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (width / 2, height / 2, 0)
               sourceSize:MTLSizeMake (1, 1, 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:256
 destinationBytesPerImage:256];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];

    const auto* bgra = static_cast<const uint8_t*> ([readback contents]);
    const uint8_t zero[4] = {};
    if (bgra == nullptr) bgra = zero;
    const bool red = bgra[0] < 20 && bgra[1] < 20
                  && bgra[2] > 220 && bgra[3] > 220;
    std::cout << "Cross-process IOSurface center BGRA="
              << static_cast<int> (bgra[0]) << ','
              << static_cast<int> (bgra[1]) << ','
              << static_cast<int> (bgra[2]) << ','
              << static_cast<int> (bgra[3]) << '\n';

    [readback release];
    [texture release];
    [queue release];
    [device release];
    CFRelease (surface);
    return red ? 0 : 4;
}
