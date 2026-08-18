// gl_loader.h — hand-rolled OpenGL 3.3 core function-pointer table.
//
// The helper links only the system GL library, which portably exports GL 1.x
// entry points; everything modern (shaders, VAOs, FBOs, PBOs) must be
// resolved at runtime from the driver. GLFW is already the platform layer,
// so the loader uses glfwGetProcAddress — no GLAD/GLEW vendoring (decision
// documented in artifacts/planning/missing-features/plans/video-parity/verify-gl.md). Function-pointer
// typedefs come from the Khronos <GL/glext.h> shipped with the GL headers.
//
// GL 1.1 functions (glTexImage2D, glClear, glDrawElements, glReadPixels, …)
// keep resolving at link time as before; only >1.1 entry points live here.
//
// Only compiled when ARBIT_HAVE_VIEWPORT (GLFW + OpenGL found at configure
// time); the headless build never includes this header.
#pragma once

#if ARBIT_HAVE_VIEWPORT

#if defined(_WIN32)
 #define WIN32_LEAN_AND_MEAN
 #include <windows.h>
 #include <GL/gl.h>
 #include <GL/glext.h>
#elif defined(__APPLE__)
 // Apple ships no Khronos <GL/glext.h>; the core-profile header declares
 // prototypes for every GL 3.2 core function, so GlFuncs derives the
 // pointer types with decltype below (loading still goes through
 // glfwGetProcAddress, same as the other platforms).
 #define GL_SILENCE_DEPRECATION 1
 #include <OpenGL/gl3.h>
#else
 #include <GL/gl.h>
 #include <GL/glext.h>
#endif

#include <string>

#include "gpu_caps.h"

namespace arbitgl
{

// X-macro list of every GL > 1.1 function the FrameRenderer needs.
// X(typedef, Name) — loaded as "gl" #Name via glfwGetProcAddress.
#define ARBIT_GL33_FUNCS(X)                                          \
    X (PFNGLACTIVETEXTUREPROC,           ActiveTexture)              \
    X (PFNGLBLENDFUNCSEPARATEPROC,       BlendFuncSeparate)          \
    X (PFNGLGENBUFFERSPROC,              GenBuffers)                 \
    X (PFNGLBINDBUFFERPROC,              BindBuffer)                 \
    X (PFNGLBUFFERDATAPROC,              BufferData)                 \
    X (PFNGLDELETEBUFFERSPROC,           DeleteBuffers)              \
    X (PFNGLMAPBUFFERRANGEPROC,          MapBufferRange)             \
    X (PFNGLUNMAPBUFFERPROC,             UnmapBuffer)                \
    X (PFNGLCREATESHADERPROC,            CreateShader)               \
    X (PFNGLSHADERSOURCEPROC,            ShaderSource)               \
    X (PFNGLCOMPILESHADERPROC,           CompileShader)              \
    X (PFNGLGETSHADERIVPROC,             GetShaderiv)                \
    X (PFNGLGETSHADERINFOLOGPROC,        GetShaderInfoLog)           \
    X (PFNGLDELETESHADERPROC,            DeleteShader)               \
    X (PFNGLCREATEPROGRAMPROC,           CreateProgram)              \
    X (PFNGLATTACHSHADERPROC,            AttachShader)               \
    X (PFNGLLINKPROGRAMPROC,             LinkProgram)                \
    X (PFNGLGETPROGRAMIVPROC,            GetProgramiv)               \
    X (PFNGLGETPROGRAMINFOLOGPROC,       GetProgramInfoLog)          \
    X (PFNGLDELETEPROGRAMPROC,           DeleteProgram)              \
    X (PFNGLUSEPROGRAMPROC,              UseProgram)                 \
    X (PFNGLGETUNIFORMLOCATIONPROC,      GetUniformLocation)         \
    X (PFNGLUNIFORM1IPROC,               Uniform1i)                  \
    X (PFNGLUNIFORM1FPROC,               Uniform1f)                  \
    X (PFNGLUNIFORM2FPROC,               Uniform2f)                  \
    X (PFNGLUNIFORM3FPROC,               Uniform3f)                  \
    X (PFNGLUNIFORM4FPROC,               Uniform4f)                  \
    X (PFNGLUNIFORMMATRIX4FVPROC,        UniformMatrix4fv)           \
    X (PFNGLGENVERTEXARRAYSPROC,         GenVertexArrays)            \
    X (PFNGLBINDVERTEXARRAYPROC,         BindVertexArray)            \
    X (PFNGLDELETEVERTEXARRAYSPROC,      DeleteVertexArrays)         \
    X (PFNGLVERTEXATTRIBPOINTERPROC,     VertexAttribPointer)        \
    X (PFNGLENABLEVERTEXATTRIBARRAYPROC, EnableVertexAttribArray)    \
    X (PFNGLGENFRAMEBUFFERSPROC,         GenFramebuffers)            \
    X (PFNGLBINDFRAMEBUFFERPROC,         BindFramebuffer)            \
    X (PFNGLFRAMEBUFFERTEXTURE2DPROC,    FramebufferTexture2D)       \
    X (PFNGLCHECKFRAMEBUFFERSTATUSPROC,  CheckFramebufferStatus)     \
    X (PFNGLDELETEFRAMEBUFFERSPROC,      DeleteFramebuffers)         \
    X (PFNGLBLITFRAMEBUFFERPROC,         BlitFramebuffer)            \
    X (PFNGLDRAWBUFFERSPROC,             DrawBuffers)                \
    X (PFNGLTEXIMAGE3DPROC,              TexImage3D)

// Optional GL 4.3 entry points for compute / SSBO / image load-store (P1 of
// the visual-engine plan). Loaded *soft*: a missing one just leaves the
// matching GpuCaps flag false, never a hard failure. macOS ships no desktop
// GL above 4.1 and none of these prototypes — so the table is empty there and
// the decltype-based GlFuncs declaration below stays well-formed.
#if ! defined (__APPLE__)
 #define ARBIT_GL43_FUNCS(X)                                          \
    X (PFNGLDISPATCHCOMPUTEPROC,           DispatchCompute)            \
    X (PFNGLMEMORYBARRIERPROC,             MemoryBarrier)              \
    X (PFNGLBINDIMAGETEXTUREPROC,          BindImageTexture)           \
    X (PFNGLBINDBUFFERBASEPROC,            BindBufferBase)             \
    X (PFNGLSHADERSTORAGEBLOCKBINDINGPROC, ShaderStorageBlockBinding)  \
    X (PFNGLGETPROGRAMRESOURCEINDEXPROC,   GetProgramResourceIndex)    \
    X (PFNGLGETINTEGERI_VPROC,             GetIntegeri_v)
#else
 #define ARBIT_GL43_FUNCS(X)
#endif

struct GlFuncs
{
#if defined(__APPLE__)
 #define ARBIT_GL_DECL(type, name) decltype (&::gl##name) name = nullptr;
#else
 #define ARBIT_GL_DECL(type, name) type name = nullptr;
#endif
    ARBIT_GL33_FUNCS (ARBIT_GL_DECL)
    ARBIT_GL43_FUNCS (ARBIT_GL_DECL)
#undef ARBIT_GL_DECL
};

// Resolves every table entry via glfwGetProcAddress. A GLFW GL context must
// be current on the calling thread. Returns false and fills missingOut with
// a comma-separated list of unresolved symbols on failure.
bool loadGlFunctions (GlFuncs& gl, std::string& missingOut);

// Soft-loads the optional GL 4.3 entry points into the same table. Never
// fails: returns true iff every 4.3 entry point resolved (always false on
// macOS, where the table is empty). Call after loadGlFunctions with a context
// current.
bool loadGl43Functions (GlFuncs& gl);

// Populates caps from the current context. A GL context must be current; the
// version comes from link-time glGetIntegerv, the indexed compute limits from
// the soft-loaded glGetIntegeri_v (only when computeShaders is true).
void queryGpuCaps (const GlFuncs& gl, GpuCaps& caps);

} // namespace arbitgl

#endif // ARBIT_HAVE_VIEWPORT
