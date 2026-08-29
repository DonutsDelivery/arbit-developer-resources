#import <Cocoa/Cocoa.h>

#include "helper_glfw.h"

void configureMacVideoHelperAsAccessoryApplication()
{
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}
