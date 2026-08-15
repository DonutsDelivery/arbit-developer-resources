# arbit-video-helper — GPL sidecar binary for Arbit video support.
#
# Links FFmpeg (libav*) and aubio, which must never be linked into the Arbit
# plugin/standalone binaries. All video decode, audio extraction, beat
# detection, thumbnailing, and export happens in this process; Arbit talks to
# it over JSON-RPC (stdin/stdout) + a shared-memory frame ring.
#
# License of this target: GPL-2.0-or-later (source published in-repo).

cmake_minimum_required(VERSION 3.22)
project(arbit-video-helper VERSION 0.1.0 LANGUAGES C CXX)  # C for vendored QuickJS-ng

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hardware accel: the decoder tries NVDEC/VAAPI (Linux), D3D11VA/NVDEC
# (Windows), VideoToolbox (macOS) at runtime and falls back to software.
# When linking a system FFmpeg nothing extra is needed here. When CI builds
# FFmpeg from source (Windows), configure it with nv-codec-headers
# (https://git.videolan.org/git/ffmpeg/nv-codec-headers.git, MIT, headers
# only — NVIDIA libs are dlopen'd from the user's driver) so NVDEC/NVENC
# support is compiled in: --enable-ffnvcodec.

find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libavdevice libswscale libswresample libavfilter)
pkg_check_modules(AUBIO IMPORTED_TARGET aubio)

# Per-frame Lua hook (M8 "agent-programmable media machine"). Optional — without
# Lua the helper builds fine and a job's luaScript is silently inert (lua_hook.h
# compiles to no-ops). Lua is MIT, pure-C, no GPL entanglement. Distro packages
# name the .pc variously, so try the common ones.
pkg_check_modules(LUA IMPORTED_TARGET lua5.4)
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua-5.4)
endif()
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua5.3)
endif()
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua)
endif()

find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Viewport window (GPU-resident preview): GLFW + OpenGL. Optional — without
# it the helper builds headless and the viewport_* methods return an error.
pkg_check_modules(GLFW3 IMPORTED_TARGET glfw3)
find_package(OpenGL QUIET)

add_executable(arbit-video-helper
    src/main.cpp
    src/capture_device.cpp
    src/media.cpp
    src/exporter.cpp
)

if(GLFW3_FOUND AND OpenGL_FOUND)
    target_sources(arbit-video-helper PRIVATE
        src/viewport.cpp
        src/renderer.cpp
        src/shader_generator.cpp
        src/particle_engine.cpp
        src/gl_loader.cpp
        src/lut_loader.cpp)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::GLFW3 OpenGL::GL)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_VIEWPORT=1)

    # Zero-copy docked viewport: exports the composited frame as shared GPU
    # buffers over a unix socket (SharedGpuSurfaceProtocol.h). One backend
    # per platform (gpu_export_common.h); actual capability (driver
    # extensions, context flavor) is probed at runtime and
    # viewport_open_shared fails cleanly without it.
    if(APPLE)
        # IOSurface + CGLTexImageIOSurface2D (global IOSurfaceIDs — no fd
        # passing). OpenGL framework comes in via find_package above.
        target_sources(arbit-video-helper PRIVATE
            src/gpu_export_iosurface.cpp
            src/fd_socket.cpp)
        target_link_libraries(arbit-video-helper PRIVATE
            "-framework IOSurface" "-framework CoreFoundation")
        target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_IOSURFACE=1)
    elseif(WIN32)
        # D3D11 keyed-mutex shared textures bridged into GL via
        # WGL_NV_DX_interop2; socket is AF_UNIX SOCK_STREAM (Winsock).
        target_sources(arbit-video-helper PRIVATE
            src/gpu_export_d3d.cpp
            src/fd_socket.cpp)
        target_link_libraries(arbit-video-helper PRIVATE d3d11 ws2_32)
        target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_D3D=1)
    else()
        # Two Linux backends may compile; gpu_export_common.h prefers VkFd.
        #  - VkFd (gpu_export_vk.cpp): Vulkan external memory (OPAQUE_FD) imported
        #    into GL via GL_EXT_memory_object_fd. Works on the NVIDIA proprietary
        #    driver under GLX (where EGL_MESA dmabuf export is unavailable) and is
        #    windowing-agnostic. Links libvulkan (hard dep when enabled, like
        #    ARBIT_NCNN_ZEROCOPY) — set ARBIT_WITH_VKFD=OFF for a vulkan-free build.
        #  - EGL dmabuf (gpu_export.cpp): Mesa/Wayland fallback when VkFd is off.
        # fd_socket.cpp is shared by both.
        option(ARBIT_WITH_VKFD "Zero-copy docked viewport via Vulkan OPAQUE_FD (Linux)" ON)
        set(_arbit_shared_any OFF)
        find_library(ARBIT_VKFD_LOADER NAMES vulkan vulkan-1)
        if(ARBIT_WITH_VKFD AND ARBIT_VKFD_LOADER)
            # The system ships no Vulkan headers (only the loader); fetch the same
            # pinned headers RIFE uses. FetchContent honors the first declare, so
            # the later ncnn-zerocopy block reuses this populated copy.
            include(FetchContent)
            FetchContent_Declare(VulkanHeaders
                GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
                GIT_TAG d2d8ded679e53c9f27c0c7a18750e661745886eb) # pinned (Vulkan 1.4.x)
            FetchContent_MakeAvailable(VulkanHeaders)
            target_sources(arbit-video-helper PRIVATE src/gpu_export_vk.cpp)
            target_include_directories(arbit-video-helper PRIVATE ${vulkanheaders_SOURCE_DIR}/include)
            target_link_libraries(arbit-video-helper PRIVATE ${ARBIT_VKFD_LOADER})
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_VKFD=1)
            set(_arbit_shared_any ON)
        endif()
        pkg_check_modules(EGL IMPORTED_TARGET egl)
        if(EGL_FOUND)
            target_sources(arbit-video-helper PRIVATE src/gpu_export.cpp)
            target_link_libraries(arbit-video-helper PRIVATE PkgConfig::EGL)
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_DMABUF=1)
            set(_arbit_shared_any ON)
        endif()
        if(_arbit_shared_any)
            target_sources(arbit-video-helper PRIVATE src/fd_socket.cpp)
        endif()
    endif()
endif()

# RIFE optical-flow frame interpolation ("speed warp"): ONNX Runtime.
# Strictly optional and OFF by default — CI and packagers build without it
# and the helper degrades to nearest-frame (interpolation "auto") or reports
# "RIFE unavailable" (interpolation "rife"). The RIFE model binary is NOT
# part of the build; the helper downloads it on first use (see src/rife.cpp
# and PROTOCOL.md §RIFE).
option(ARBIT_WITH_ONNX "Enable RIFE frame interpolation via ONNX Runtime" OFF)
set(ARBIT_ONNXRUNTIME_ROOT "" CACHE PATH
    "Existing ONNX Runtime install (include/ + lib/); skips the prebuilt download")
if(ARBIT_WITH_ONNX)
    set(ARBIT_ORT_VERSION 1.26.0)
    if(ARBIT_ONNXRUNTIME_ROOT)
        set(ARBIT_ORT_ROOT "${ARBIT_ONNXRUNTIME_ROOT}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        # Official prebuilt, GPU flavor (CUDA 13 EP + CPU EP in one package).
        # The CUDA provider is dlopen'd at runtime; machines without
        # CUDA/cuDNN fall back to the CPU EP inside RifeEngine::init().
        include(FetchContent)
        FetchContent_Declare(onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ARBIT_ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ARBIT_ORT_VERSION}.tgz
            URL_HASH SHA256=aa619d5701bbe58046cc998b21e692d5b2aefac1479f375c4b988526cb80befa
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ARBIT_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        # Official prebuilt for Apple Silicon. The Core ML EP is compiled into
        # the dylib (no separate provider lib), so RifeEngine can append it at
        # runtime on __APPLE__. Realtime tier 2 on macOS runs on the ANE / GPU.
        include(FetchContent)
        FetchContent_Declare(onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ARBIT_ORT_VERSION}/onnxruntime-osx-arm64-${ARBIT_ORT_VERSION}.tgz
            URL_HASH SHA256=7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ARBIT_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "ARBIT_WITH_ONNX=ON: no prebuilt wired for this "
                            "platform — set ARBIT_ONNXRUNTIME_ROOT to an "
                            "ONNX Runtime install")
    endif()
    find_library(ARBIT_ORT_LIBRARY NAMES onnxruntime
                 HINTS "${ARBIT_ORT_ROOT}/lib" REQUIRED)
    # rife.cpp = the RIFE ONNX engine (shared by export + viewport). interp.cpp (the
    # live-viewport async RIFE worker, retime tier 2) is added below for EITHER backend.
    target_sources(arbit-video-helper PRIVATE src/rife.cpp)
    target_include_directories(arbit-video-helper PRIVATE "${ARBIT_ORT_ROOT}/include")
    target_link_libraries(arbit-video-helper PRIVATE "${ARBIT_ORT_LIBRARY}")
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_ONNX=1)
    # Dev/run-in-place: resolve libonnxruntime.so* from the fetched package.
    # Deployment ships the ORT libs next to the helper binary ($ORIGIN).
    set_target_properties(arbit-video-helper PROPERTIES
        BUILD_RPATH "${ARBIT_ORT_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN;${ARBIT_ORT_ROOT}/lib")
endif()

# --- Bundled Vulkan loader discovery (Linux) -------------------------------
# The Linux release bundle ships libvulkan.so.1 next to the helper so the helper
# starts even on machines with NO system Vulkan loader — it then finds no GPU and
# degrades to the shm present path + Frame Blend, so basic (non-interpolated)
# video editing always works (the core pipeline is GL + FFmpeg, not Vulkan).
# $ORIGIN makes the helper find that bundled loader; it is harmless when absent
# (the dynamic linker falls through to the system paths) and the loader is a thin
# dispatch shim that still loads the user's own GPU driver/ICD at runtime.
# Deployment copies the build-tree binary directly (no cmake --install step), so
# bake the rpath into the build tree via BUILD_WITH_INSTALL_RPATH. ncnn is static
# and the helper has no other build-tree shared deps besides ORT (whose path stays
# in INSTALL_RPATH above), so this is safe.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET arbit-video-helper APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    set_target_properties(arbit-video-helper PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# RIFE frame interpolation, portable path: ncnn + Vulkan ("rife-vulkan", tier 4).
# Unlike the ONNX/CUDA path above (NVIDIA-only, OFF by default), this runs on any
# conformant Vulkan driver (AMD/Intel/NVIDIA/Apple-MoltenVK) from one binary, so
# it ships ON by default — the flag exists only to allow a stripped build.
#
# Net source is vendored (src/rife_net/, nihui/rife-ncnn-vulkan, MIT) and ncnn
# (Tencent, BSD-3) is fetched. ncnn 20250503 is PINNED on purpose: it is the
# newest release that still defines the afpvec8 packed-fp16 shader type +
# Option::use_shader_pack8 that the 2022-era net source uses (both removed by
# ncnn 20251225). See artifacts/media-machine/rife-ncnn-vulkan-plan.md.
# ncnn's SIMPLEVK path expects a Vulkan loader symbol at link time on macOS.
# MoltenVK satisfies that (see Apple branch below). Default ON everywhere so
# the shipped helper includes Speed Warp; configure -DARBIT_WITH_NCNN=OFF to
# strip RIFE. macOS CI installs molten-vk via Homebrew and asserts rife-vulkan.
set(_arbit_ncnn_default ON)
option(ARBIT_WITH_NCNN "Enable portable RIFE frame interpolation via ncnn + Vulkan" ${_arbit_ncnn_default})
# Zero-copy GPU RIFE output (Phase 1b): keep the RIFE result on the GPU and export
# it to OpenGL via OPAQUE_FD instead of bouncing it through CPU. Needs the REAL
# Vulkan headers (ncnn's in-house simplevk.h lacks the external-memory structs) and
# the Vulkan loader at link time, and makes libvulkan a HARD runtime dependency
# (vs the SIMPLEVK soft-dlopen). The export path itself (OPAQUE_FD -> GL) is
# Linux-specific (macOS would need IOSurface, Windows DXGI — not built), so this
# defaults ON only on Linux and OFF elsewhere. Set OFF for a stripped Linux build
# that keeps only the portable CPU-readback RIFE path (no Vulkan headers/loader).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_arbit_zerocopy_default ON)
else()
    set(_arbit_zerocopy_default OFF)
endif()
option(ARBIT_NCNN_ZEROCOPY "Zero-copy GPU RIFE output (Vulkan->GL OPAQUE_FD, Linux)" ${_arbit_zerocopy_default})
if(ARBIT_WITH_NCNN)
    include(FetchContent)
    # ncnn build knobs: Vulkan on, static, no extras, bundle+build glslang.
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
    set(NCNN_SYSTEM_GLSLANG OFF CACHE BOOL "" FORCE)

    if(ARBIT_NCNN_ZEROCOPY)
        # SIMPLEVK off => ncnn includes <vulkan/vulkan.h> (real external-memory
        # structs) and links the loader. Provide both before configuring ncnn.
        set(NCNN_SIMPLEVK OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
            GIT_TAG d2d8ded679e53c9f27c0c7a18750e661745886eb) # pinned (Vulkan 1.4.x)
        FetchContent_MakeAvailable(VulkanHeaders)
        find_library(ARBIT_VULKAN_LOADER NAMES vulkan vulkan-1)
        if(NOT ARBIT_VULKAN_LOADER)
            message(FATAL_ERROR
                "ARBIT_NCNN_ZEROCOPY=ON needs the Vulkan loader (libvulkan). Install "
                "vulkan-loader / libvulkan-dev, or configure -DARBIT_NCNN_ZEROCOPY=OFF "
                "for the CPU-readback-only RIFE path.")
        endif()
        set(Vulkan_INCLUDE_DIR ${vulkanheaders_SOURCE_DIR}/include CACHE PATH "" FORCE)
        set(Vulkan_LIBRARY ${ARBIT_VULKAN_LOADER} CACHE FILEPATH "" FORCE)
    elseif(APPLE)
        # ncnn's SimpleVK uses STATIC link-time Vulkan linkage on Apple platforms
        # (unlike the dlopen path used on Linux/Windows/Android) — see ncnn's
        # src/CMakeLists.txt "simplevk always use static vulkan linkage on apple
        # platform". If Vulkan_LIBRARY isn't defined there, ncnn's own CMake only
        # WARNS and links nothing, so the helper would build fine but never find a
        # GPU. MoltenVK is itself a complete Vulkan-API implementation (not just a
        # loader), so linking it directly satisfies vkGetInstanceProcAddr et al.
        # with no separate loader / ICD-JSON discovery step needed.
        find_library(ARBIT_MOLTENVK_LIBRARY NAMES MoltenVK)
        if(NOT ARBIT_MOLTENVK_LIBRARY)
            message(FATAL_ERROR
                "ARBIT_WITH_NCNN=ON needs MoltenVK on macOS (brew install molten-vk), "
                "or configure -DARBIT_WITH_NCNN=OFF to build without RIFE support.")
        endif()
        set(Vulkan_LIBRARY ${ARBIT_MOLTENVK_LIBRARY} CACHE FILEPATH "" FORCE)
    endif()

    # ncnn 20250503 only enables base VK_KHR_external_memory; the PATCH_COMMAND adds
    # the fd-based variants needed to export to GL (idempotent, inert unless the ncnn
    # object is compiled with ARBIT_RIFE_ZEROCOPY — see below).
    FetchContent_Declare(ncnn
        URL https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-full-source.zip
        URL_HASH SHA256=b3190d1b6beae3aac30fa99f4d3fd89a2b47a010b888d2d2389189784743a690
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND ${CMAKE_COMMAND} -DNCNN_GPU_CPP=src/gpu.cpp
                      -P ${CMAKE_CURRENT_SOURCE_DIR}/patches/ncnn_enable_external_memory_fd.cmake)
    # ncnn's vendored glslang declares a pre-3.5 cmake_minimum_required; allow it
    # under CMake 4.x (harmless on older CMake).
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(ncnn)

    # GCC 13 can ICE in cfgcleanup while compiling ncnn's generated AVX packing
    # unit with ncnn's own -Ofast flags (GCC PR-style internal failure, not a
    # source diagnostic). Override that dependency target to -O2 only on the
    # affected compiler generation. GNU 12/14+, Clang, and DonutStudio's own
    # targets retain their existing release optimization settings.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
        target_compile_options(ncnn PRIVATE -O2)
    endif()

    if(ARBIT_NCNN_ZEROCOPY)
        # activates the gpu.cpp patch block
        target_compile_definitions(ncnn PRIVATE ARBIT_RIFE_ZEROCOPY)
    endif()

    # Hex-encode the vendored net's custom-layer GLSL shaders into headers the
    # net source #includes (ncnn compiles them to SPIR-V online via glslang).
    set(ARBIT_RIFE_NET ${CMAKE_CURRENT_SOURCE_DIR}/src/rife_net)
    set(ARBIT_RIFE_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/rife_shaders)
    file(MAKE_DIRECTORY ${ARBIT_RIFE_SHADER_DIR})
    set(ARBIT_RIFE_SHADER_HEADERS)
    foreach(s
        rife_preproc rife_postproc rife_preproc_tta rife_postproc_tta
        rife_flow_tta_avg rife_v2_flow_tta_avg rife_v4_flow_tta_avg
        rife_flow_tta_temporal_avg rife_v2_flow_tta_temporal_avg rife_v4_flow_tta_temporal_avg
        rife_out_tta_temporal_avg rife_v4_timestep rife_v4_timestep_tta
        warp warp_pack4 warp_pack8)
        set(_hdr ${ARBIT_RIFE_SHADER_DIR}/${s}.comp.hex.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND}
                    -DSHADER_SRC=${ARBIT_RIFE_NET}/${s}.comp
                    -DSHADER_COMP_HEADER=${_hdr}
                    -P ${ARBIT_RIFE_NET}/generate_shader_comp_header.cmake
            DEPENDS ${ARBIT_RIFE_NET}/${s}.comp
            WORKING_DIRECTORY ${ARBIT_RIFE_SHADER_DIR}
            COMMENT "hex shader ${s}.comp"
            VERBATIM)
        list(APPEND ARBIT_RIFE_SHADER_HEADERS ${_hdr})
    endforeach()
    add_custom_target(arbit-rife-shaders DEPENDS ${ARBIT_RIFE_SHADER_HEADERS})

    target_sources(arbit-video-helper PRIVATE
        ${ARBIT_RIFE_NET}/rife.cpp
        ${ARBIT_RIFE_NET}/warp.cpp
        src/rife_ncnn.cpp)
    add_dependencies(arbit-video-helper arbit-rife-shaders)
    target_include_directories(arbit-video-helper PRIVATE
        ${ARBIT_RIFE_NET} ${ARBIT_RIFE_SHADER_DIR})
    target_link_libraries(arbit-video-helper PRIVATE ncnn)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_NCNN=1)
    if(ARBIT_NCNN_ZEROCOPY)
        # rife_ncnn.cpp's ExportableRing + rife.cpp's process_v4_to_buffer include
        # <vulkan/vulkan.h> (via ncnn's PUBLIC Vulkan::Vulkan) and use the external-
        # memory API. The GL import side is loaded at runtime via the viewport's GL
        # context (no extra link).
        target_compile_definitions(arbit-video-helper PRIVATE ARBIT_RIFE_ZEROCOPY=1)
        target_link_libraries(arbit-video-helper PRIVATE Vulkan::Headers)
    endif()
endif()

# The live-viewport async RIFE worker is backend-agnostic (picks ncnn-Vulkan when
# built, else the ONNX engine), so it compiles whenever EITHER backend is enabled.
if(ARBIT_WITH_NCNN OR ARBIT_WITH_ONNX)
    target_sources(arbit-video-helper PRIVATE src/interp.cpp)
endif()

target_include_directories(arbit-video-helper PRIVATE
    # VideoFrameSharedMemory.h is shared with the plugin (dependency-free header)
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source
)

target_link_libraries(arbit-video-helper PRIVATE
    PkgConfig::AV
    nlohmann_json::nlohmann_json
)

if(AUBIO_FOUND)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::AUBIO)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_AUBIO=1)
endif()

if(LUA_FOUND)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::LUA)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_LUA=1)
    message(STATUS "arbit-video-helper: Lua per-frame hook enabled (lua ${LUA_VERSION})")
else()
    message(STATUS "arbit-video-helper: Lua not found — per-frame hook disabled")
endif()

# Per-frame JavaScript hook (visual-engine P2 — QuickJS-ng front-end). Vendored
# under third_party/quickjs/ (MIT, single-tree, no external deps) so it always
# builds where the source is present; js_hook.h compiles to no-ops when
# ARBIT_HAVE_QUICKJS is off. JavaScript is the creative-coding + AI-agent lingua
# franca — Shadertoy-JS / generative-JS authoring with no browser. The compiled
# library is third-party C: warnings silenced so it never trips the helper flags.
set(ARBIT_QUICKJS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs)
if(EXISTS ${ARBIT_QUICKJS_DIR}/quickjs.c)
    add_library(arbit-quickjs STATIC
        ${ARBIT_QUICKJS_DIR}/quickjs.c
        ${ARBIT_QUICKJS_DIR}/cutils.c
        ${ARBIT_QUICKJS_DIR}/libregexp.c
        ${ARBIT_QUICKJS_DIR}/libunicode.c
        ${ARBIT_QUICKJS_DIR}/xsum.c)
    target_include_directories(arbit-quickjs PUBLIC ${ARBIT_QUICKJS_DIR})
    set_target_properties(arbit-quickjs PROPERTIES
        C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        target_compile_options(arbit-quickjs PRIVATE -w)
    endif()
    if(NOT WIN32)
        target_link_libraries(arbit-quickjs PUBLIC m)
    endif()
    target_link_libraries(arbit-video-helper PRIVATE arbit-quickjs)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_QUICKJS=1)
    message(STATUS "arbit-video-helper: QuickJS-ng per-frame JS hook enabled")
else()
    message(STATUS "arbit-video-helper: QuickJS-ng not vendored — JS hook disabled")
endif()

# SPIR-V → GLSL ingestion (visual-engine P3 — the "universal adapter"). Vendored
# SPIRV-Cross (third_party/spirv_cross/, Apache-2.0, GLSL backend only). Slang,
# HLSL and GLSL all compile to SPIR-V (slangc/dxc/glslang), so this one backend
# makes the helper language-agnostic. Gated: shader_lang.h returns a clean
# diagnostic without it. Third-party C++ → warnings silenced.
set(ARBIT_SPVCROSS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spirv_cross)
if(EXISTS ${ARBIT_SPVCROSS_DIR}/spirv_glsl.cpp)
    add_library(arbit-spirv-cross STATIC
        ${ARBIT_SPVCROSS_DIR}/spirv_cross.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_glsl.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cfg.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_parser.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cross_parsed_ir.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cross_util.cpp)
    target_include_directories(arbit-spirv-cross PUBLIC ${ARBIT_SPVCROSS_DIR})
    set_target_properties(arbit-spirv-cross PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        target_compile_options(arbit-spirv-cross PRIVATE -w)
    endif()
    target_link_libraries(arbit-video-helper PRIVATE arbit-spirv-cross)
    target_include_directories(arbit-video-helper PRIVATE ${ARBIT_SPVCROSS_DIR})
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_SPIRV_CROSS=1)
    message(STATUS "arbit-video-helper: SPIR-V ingestion enabled (SPIRV-Cross)")
else()
    message(STATUS "arbit-video-helper: SPIRV-Cross not vendored — SPIR-V ingestion disabled")
endif()

if(WIN32)
    # windows.h (pulled in by VideoFrameSharedMemory.h / gl_loader.h) defines
    # min/max macros that break std::min/std::max.
    target_compile_definitions(arbit-video-helper PRIVATE NOMINMAX)

    # FFmpeg, zimg, and the rest of the DLL closure come from MSYS2. Keep the
    # helper's own GCC C++ and GCC support runtimes inside the executable so a
    # different libstdc++ ABI cannot be selected to satisfy both the Ubuntu
    # cross-compiler and the MSYS2 dependency graph at process startup.
    target_link_options(arbit-video-helper PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(arbit-video-helper PRIVATE rt)   # shm_open (Linux only; macOS has it in libSystem)
endif()
if(NOT WIN32)
    target_link_libraries(arbit-video-helper PRIVATE pthread)
endif()

# --- Tests (dependency-free) -------------------------------------------------
# The shared render-graph param grammar (Stage 6) is header-only and links nothing
# (no FFmpeg/GL/Lua), so its unit test builds and runs anywhere — the guard-rail that
# replaced the old "KEEP IN SYNC" comments across exporter.cpp / viewport.cpp.
enable_testing()
add_executable(arbit-video-param-tests tests/param_grammar_tests.cpp)
target_include_directories(arbit-video-param-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-video-param-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME video_param_grammar COMMAND arbit-video-param-tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND UNIX)
    add_test(NAME concurrent_recorder_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrent_recorder_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    add_test(NAME capture_source_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_source_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    set_tests_properties(concurrent_recorder_smoke capture_source_smoke PROPERTIES TIMEOUT 30)
endif()
