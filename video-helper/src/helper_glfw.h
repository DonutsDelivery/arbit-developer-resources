#pragma once

#include <GLFW/glfw3.h>

#if defined (__APPLE__)
void configureMacVideoHelperAsAccessoryApplication();
#endif

inline bool initializeHelperGlfw()
{
#if defined (__APPLE__)
    // The helper is launched by DonutStudio and must not claim a menu bar or
    // Dock icon when GLFW creates its Cocoa application object.
    glfwInitHint (GLFW_COCOA_MENUBAR, GLFW_FALSE);
#endif
    if (glfwInit() != GLFW_TRUE)
        return false;
#if defined (__APPLE__)
    configureMacVideoHelperAsAccessoryApplication();
#endif
    return true;
}
