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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/generate_shader_catalog_manifest.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared/ShaderCatalogManifest.h --check
    RESULT_VARIABLE ARBIT_SHADER_CATALOG_CHECK_RESULT)
if(NOT ARBIT_SHADER_CATALOG_CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR "ShaderCatalogManifest.h is stale; regenerate it before building")
endif()
add_custom_target(arbit-shader-catalog-check ALL
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/generate_shader_catalog_manifest.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared/ShaderCatalogManifest.h --check
    COMMENT "Checking generated shader catalog manifest"
    VERBATIM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    option(ARBIT_WITH_METAL "Enable the native sokol_gfx Metal backend" ON)
    if(ARBIT_WITH_METAL)
        enable_language(OBJCXX)
    endif()
endif()

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
option(ARBIT_REQUIRE_VIEWPORT "Fail configure when the production viewport cannot be built" OFF)
if(ARBIT_REQUIRE_VIEWPORT AND NOT (GLFW3_FOUND AND OpenGL_FOUND))
    message(FATAL_ERROR
        "ARBIT_REQUIRE_VIEWPORT=ON requires GLFW3 and OpenGL; refusing a headless shipping helper")
endif()

add_executable(arbit-video-helper
    src/main.cpp
    src/capture_device.cpp
    src/media.cpp
    src/exporter.cpp
)
add_dependencies(arbit-video-helper arbit-shader-catalog-check)

# Visual-engine P6 foundation: a language-neutral native-backend seam plus a
# real compute/offscreen-render self-test.  FrameRenderer remains GL-backed
# until the subsequent migration slices route its passes through this seam.
if(APPLE AND ARBIT_WITH_METAL)
    target_sources(arbit-video-helper PRIVATE src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-video-helper PRIVATE third_party/sokol)
    target_link_libraries(arbit-video-helper PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_METAL_BACKEND=1)
    message(STATUS "arbit-video-helper: native sokol_gfx Metal backend enabled")
else()
    target_sources(arbit-video-helper PRIVATE src/gpu_backend/backend_stub.cpp)
endif()

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
        if(ARBIT_WITH_METAL)
            # IOSurface-backed native Metal targets (global IOSurfaceIDs — no
            # fd passing). The shared path creates no OpenGL context.
            target_sources(arbit-video-helper PRIVATE
                src/mac_metal_viewport_surface.mm
                src/gpu_export_iosurface.cpp
                src/fd_socket.cpp)
            target_link_libraries(arbit-video-helper PRIVATE
                "-framework IOSurface" "-framework CoreFoundation")
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_IOSURFACE=1)
        endif()
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
# Linux-specific (macOS would need IOSurface, Windows DXGI — not built). It remains
# fail-closed until ncnn exposes a legal asynchronous submission API; the portable
# synchronous CPU-readback RIFE path stays available.
set(_arbit_zerocopy_default OFF)
option(ARBIT_NCNN_ZEROCOPY "Zero-copy GPU RIFE output (Vulkan->GL OPAQUE_FD, Linux)" ${_arbit_zerocopy_default})
if(ARBIT_NCNN_ZEROCOPY)
    message(FATAL_ERROR
        "ARBIT_NCNN_ZEROCOPY is blocked: pinned ncnn 20250503 exposes only "
        "VkCompute::submit_and_wait(), so external wait/signal semaphores and "
        "per-submission completion cannot be attached legally. See "
        "artifacts/media-machine/rife-ncnn-async-submission-blocker.md.")
endif()
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

    add_executable(rife-ncnn-inference-tests
        tests/rife_ncnn_inference_tests.cpp
        ${ARBIT_RIFE_NET}/rife.cpp
        ${ARBIT_RIFE_NET}/warp.cpp
        src/rife_ncnn.cpp)
    add_dependencies(rife-ncnn-inference-tests arbit-rife-shaders)
    target_include_directories(rife-ncnn-inference-tests PRIVATE
        src ${ARBIT_RIFE_NET} ${ARBIT_RIFE_SHADER_DIR})
    target_link_libraries(rife-ncnn-inference-tests PRIVATE ncnn PkgConfig::AV)
    target_compile_definitions(rife-ncnn-inference-tests PRIVATE ARBIT_HAVE_NCNN=1)
    if(NOT MSVC)
        target_compile_options(rife-ncnn-inference-tests PRIVATE -Wall -Wextra)
    endif()
    if(ARBIT_NCNN_ZEROCOPY)
        target_compile_definitions(rife-ncnn-inference-tests PRIVATE ARBIT_RIFE_ZEROCOPY=1)
        target_link_libraries(rife-ncnn-inference-tests PRIVATE Vulkan::Headers)
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

# Strict macOS generator path: compile the same wrapped GLSL contract to SPIR-V
# with glslang, convert it to MSL in-process, and render into private Metal
# textures sampled directly by the compositor. MoltenVK's converter is static,
# so the installed helper does not launch or depend on a developer tool.
if(APPLE AND ARBIT_WITH_METAL AND GLFW3_FOUND AND OpenGL_FOUND)
    if(NOT ARBIT_WITH_NCNN)
        message(FATAL_ERROR "The native Metal generator path requires bundled glslang (ARBIT_WITH_NCNN=ON)")
    endif()
    find_path(ARBIT_MVK_SHADER_CONVERTER_INCLUDE
        NAMES MoltenVKShaderConverter/SPIRVConversion.h)
    find_library(ARBIT_MVK_SHADER_CONVERTER_LIBRARY
        NAMES MoltenVKShaderConverter
        PATH_SUFFIXES
            Frameworks/MoltenVKShaderConverter.xcframework/macos-arm64
            Frameworks/MoltenVKShaderConverter.xcframework/macos-x86_64)
    find_program(ARBIT_MVK_SHADER_CONVERTER_TOOL NAMES MoltenVKShaderConverter)
    if(ARBIT_MVK_SHADER_CONVERTER_TOOL)
        get_filename_component(ARBIT_MVK_SHADER_CONVERTER_TOOL_REAL
            ${ARBIT_MVK_SHADER_CONVERTER_TOOL} REALPATH)
        get_filename_component(ARBIT_MVK_BIN_DIR
            ${ARBIT_MVK_SHADER_CONVERTER_TOOL_REAL} DIRECTORY)
        get_filename_component(ARBIT_MVK_PREFIX ${ARBIT_MVK_BIN_DIR} DIRECTORY)
    endif()
    find_library(ARBIT_MVK_SPIRV_CROSS_LIBRARY NAMES SPIRVCross
        HINTS ${ARBIT_MVK_PREFIX}/libexec/lib
        PATH_SUFFIXES libexec/lib)
    find_library(ARBIT_MVK_SPIRV_TOOLS_LIBRARY NAMES SPIRVTools
        HINTS ${ARBIT_MVK_PREFIX}/libexec/lib
        PATH_SUFFIXES libexec/lib)
    if(NOT ARBIT_MVK_SHADER_CONVERTER_INCLUDE
       OR NOT ARBIT_MVK_SHADER_CONVERTER_LIBRARY
       OR NOT ARBIT_MVK_SPIRV_CROSS_LIBRARY
       OR NOT ARBIT_MVK_SPIRV_TOOLS_LIBRARY)
        message(FATAL_ERROR
            "Strict Metal generators require MoltenVKShaderConverter and its static SPIR-V libraries")
    endif()

    target_sources(arbit-video-helper PRIVATE src/gpu_backend/metal_shader_generator.mm)
    target_include_directories(arbit-video-helper PRIVATE
        ${ARBIT_MVK_SHADER_CONVERTER_INCLUDE}
        ${ncnn_SOURCE_DIR}/glslang)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_METAL_GENERATORS=1)
    target_link_libraries(arbit-video-helper PRIVATE
        glslang SPIRV glslang-default-resource-limits
        ${ARBIT_MVK_SHADER_CONVERTER_LIBRARY}
        ${ARBIT_MVK_SPIRV_CROSS_LIBRARY}
        ${ARBIT_MVK_SPIRV_TOOLS_LIBRARY}
        "-framework Foundation")
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

add_executable(arbit-beat-timeline-tests tests/beat_timeline_tests.cpp)
target_include_directories(arbit-beat-timeline-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-beat-timeline-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME beat_timeline COMMAND arbit-beat-timeline-tests)

add_executable(arbit-video-control-plan-tests tests/video_control_plan_tests.cpp)
target_include_directories(arbit-video-control-plan-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(arbit-video-control-plan-tests PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(arbit-video-control-plan-tests PRIVATE /W4)
else()
  target_compile_options(arbit-video-control-plan-tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME video_control_plan COMMAND arbit-video-control-plan-tests)

add_test(NAME depth_primitives_source_contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/depth_primitives_source_contract.py)
add_test(NAME rife_ncnn_async_submission_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rife_ncnn_async_submission_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-source-binding-tests tests/source_binding_tests.cpp)
target_include_directories(arbit-source-binding-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-source-binding-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME source_binding COMMAND arbit-source-binding-tests)

add_executable(arbit-render-snapshot-tests tests/render_snapshot_tests.cpp)
target_include_directories(arbit-render-snapshot-tests PRIVATE src)
target_link_libraries(arbit-render-snapshot-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-render-snapshot-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME render_snapshot COMMAND arbit-render-snapshot-tests)

add_executable(programmable-admission-tests tests/programmable_admission_tests.cpp)
add_dependencies(programmable-admission-tests arbit-shader-catalog-check)
target_include_directories(programmable-admission-tests PRIVATE src)
target_compile_definitions(programmable-admission-tests PRIVATE
    SHADER_CATALOG_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
target_link_libraries(programmable-admission-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(programmable-admission-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME programmable_admission COMMAND programmable-admission-tests)
set_tests_properties(programmable_admission PROPERTIES TIMEOUT 30)

option(ARBIT_PROGRAMMABLE_TEST_MODE "Enable private programmable-runtime test diagnostics" OFF)
if(ARBIT_PROGRAMMABLE_TEST_MODE)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_PROGRAMMABLE_TEST_MODE=1)
endif()

if(UNIX)
    add_executable(programmable-private-payload-fixture tests/private_payload_fixture.cpp)
    target_include_directories(programmable-private-payload-fixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    add_executable(programmable-private-payload-tests tests/private_payload_tests.cpp)
    target_include_directories(programmable-private-payload-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(programmable-private-payload-tests PRIVATE
        PRIVATE_PAYLOAD_FIXTURE="$<TARGET_FILE:programmable-private-payload-fixture>")
    add_dependencies(programmable-private-payload-tests programmable-private-payload-fixture)
    add_test(NAME programmable_private_payload COMMAND programmable-private-payload-tests)
    set_tests_properties(programmable_private_payload PROPERTIES TIMEOUT 30)
    # This matrix asserts compile-gated private diagnostics in addition to the
    # public grant/replay behavior.  Do not register it against production-mode
    # helpers, where those diagnostics intentionally do not exist.
    if(ARBIT_PROGRAMMABLE_TEST_MODE)
        add_test(NAME programmable_rpc_grant_matrix
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/programmable_rpc_grant_matrix.py
                    $<TARGET_FILE:arbit-video-helper>)
        set_tests_properties(programmable_rpc_grant_matrix PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)
    endif()

endif()

# Production lifecycle coverage deliberately links JUCE and drives the real
# process manager into the exact helper target rather than a fork-only fixture.
# It is also a Windows source/launch gate: that build exercises the production
# CreateProcess call site and proves required-private-channel launch is refused.
set(ARBIT_JUCE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE" CACHE PATH
    "JUCE source tree used by the production sidecar lifecycle test")
if(EXISTS "${ARBIT_JUCE_SOURCE_DIR}/CMakeLists.txt" AND NOT MINGW)
    add_subdirectory("${ARBIT_JUCE_SOURCE_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/JUCE EXCLUDE_FROM_ALL)
    add_executable(programmable-production-lifecycle-tests
        tests/production_lifecycle_tests.cpp)
    target_include_directories(programmable-production-lifecycle-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source)
    target_compile_definitions(programmable-production-lifecycle-tests PRIVATE
        PRIVATE_LIFECYCLE_HELPER="$<TARGET_FILE:arbit-video-helper>"
        ARBIT_PROGRAMMABLE_TEST_MODE=1 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(programmable-resource-limit-fixture
            tests/resource_limit_fixture.cpp)
        target_compile_definitions(programmable-production-lifecycle-tests PRIVATE
            RESOURCE_LIMIT_FIXTURE="$<TARGET_FILE:programmable-resource-limit-fixture>")
        add_dependencies(programmable-production-lifecycle-tests
            programmable-resource-limit-fixture)
    endif()
    target_link_libraries(programmable-production-lifecycle-tests PRIVATE
        juce::juce_core juce::juce_events)
    if(NOT MSVC)
        target_compile_options(programmable-production-lifecycle-tests PRIVATE -UNDEBUG)
    endif()
    add_dependencies(programmable-production-lifecycle-tests arbit-video-helper)
    if(ARBIT_PROGRAMMABLE_TEST_MODE)
        add_test(NAME programmable_production_lifecycle
            COMMAND programmable-production-lifecycle-tests)
        set_tests_properties(programmable_production_lifecycle PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)
    endif()
endif()

add_executable(programmable-platform-capability-tests tests/platform_capability_tests.cpp)
target_include_directories(programmable-platform-capability-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
add_test(NAME programmable_platform_capability COMMAND programmable-platform-capability-tests)
set_tests_properties(programmable_platform_capability PROPERTIES TIMEOUT 30)

add_executable(programmable-hook-limits-tests tests/programmable_hook_limits_tests.cpp)
target_include_directories(programmable-hook-limits-tests PRIVATE src)
if(LUA_FOUND)
    target_link_libraries(programmable-hook-limits-tests PRIVATE PkgConfig::LUA)
    target_compile_definitions(programmable-hook-limits-tests PRIVATE ARBIT_HAVE_LUA=1)
endif()
if(TARGET arbit-quickjs)
    target_link_libraries(programmable-hook-limits-tests PRIVATE arbit-quickjs)
    target_compile_definitions(programmable-hook-limits-tests PRIVATE ARBIT_HAVE_QUICKJS=1)
endif()
if(NOT MSVC)
    target_compile_options(programmable-hook-limits-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME programmable_hook_limits COMMAND programmable-hook-limits-tests)
set_tests_properties(programmable_hook_limits PROPERTIES TIMEOUT 30)

if(UNIX AND NOT WIN32)
    add_executable(arbit-depth-cache-tests tests/depth_cache_tests.cpp)
    target_include_directories(arbit-depth-cache-tests PRIVATE src)
    target_link_libraries(arbit-depth-cache-tests PRIVATE PkgConfig::AV nlohmann_json::nlohmann_json)
    target_compile_options(arbit-depth-cache-tests PRIVATE -Wall -Wextra -Werror)
    add_test(NAME depth_cache_admission COMMAND arbit-depth-cache-tests)
    set_tests_properties(depth_cache_admission PROPERTIES TIMEOUT 30)
endif()
add_test(NAME depth_cache_windows_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/depth_cache_windows_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-composite-probe-contract-tests tests/composite_probe_contract_tests.cpp)
target_include_directories(arbit-composite-probe-contract-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-composite-probe-contract-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME composite_probe_contract COMMAND arbit-composite-probe-contract-tests)

add_executable(arbit-visual-plan-executor-tests tests/visual_plan_executor_tests.cpp)
target_include_directories(arbit-visual-plan-executor-tests PRIVATE src)
target_link_libraries(arbit-visual-plan-executor-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-visual-plan-executor-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME visual_plan_executor COMMAND arbit-visual-plan-executor-tests)

if(UNIX AND NOT WIN32)
    add_executable(arbit-tracking-runtime-tests tests/tracking_runtime_tests.cpp)
    target_include_directories(arbit-tracking-runtime-tests PRIVATE src)
    target_compile_definitions(arbit-tracking-runtime-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
    target_link_libraries(arbit-tracking-runtime-tests PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(arbit-tracking-runtime-tests PRIVATE -Wall -Wextra -Werror)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        # GCC 16 misdiagnoses nlohmann::json's shared output adapter teardown
        # after inlining it through dump(); keep project warnings fatal.
        target_compile_options(arbit-tracking-runtime-tests PRIVATE -Wno-array-bounds)
    endif()
    add_test(NAME tracking_runtime COMMAND arbit-tracking-runtime-tests)
    set_tests_properties(tracking_runtime PROPERTIES TIMEOUT 30)
endif()

add_test(NAME generator_particle_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/generator_particle_telemetry_source_contract.py)
add_test(NAME node_cost_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/node_cost_telemetry_source_contract.py)

add_executable(arbit-viewport-telemetry-owner-tests tests/viewport_telemetry_owner_tests.cpp)
target_include_directories(arbit-viewport-telemetry-owner-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-viewport-telemetry-owner-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME viewport_telemetry_owner COMMAND arbit-viewport-telemetry-owner-tests)
add_test(NAME viewport_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/viewport_telemetry_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-export-telemetry-owner-tests tests/export_telemetry_owner_tests.cpp)
target_include_directories(arbit-export-telemetry-owner-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-export-telemetry-owner-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME export_telemetry_owner COMMAND arbit-export-telemetry-owner-tests)
add_test(NAME export_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/export_telemetry_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

# M7 physical Linux acceptance: drive the production helper/exporter/FrameRenderer
# and FFmpeg handoff at exact 1080p then 4K. This is intentionally not part of
# the default test set: it needs a real OpenGL device/display and ffprobe and may
# take up to five minutes. Run explicitly with `ctest -L native-export-runtime`.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND GLFW3_FOUND AND OpenGL_FOUND)
    add_test(NAME native_linux_export_runtime
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/export_runtime_fixture.py
                $<TARGET_FILE:arbit-video-helper>)
    set_tests_properties(native_linux_export_runtime PROPERTIES
        LABELS "native-export-runtime"
        TIMEOUT 300
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
endif()

add_executable(arbit-viewport-plan-concurrency-tests tests/viewport_plan_concurrency_tests.cpp)
target_include_directories(arbit-viewport-plan-concurrency-tests PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(arbit-viewport-plan-concurrency-tests PRIVATE Threads::Threads)
if(NOT MSVC)
    target_compile_options(arbit-viewport-plan-concurrency-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME viewport_plan_concurrency COMMAND arbit-viewport-plan-concurrency-tests)

if(GLFW3_FOUND AND OpenGL_FOUND)
    # Real Linux pixel acceptance for Depth Fog. This compiles the production
    # FrameRenderer and shader owners, creates a hidden native GL context, uploads
    # an actual GL_R16 gradient, and reads the composited pixels back. No CPU
    # renderer or replacement shader participates.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(arbit-depth-fog-gl-pixel-tests
            tests/depth_fog_gl_pixel_tests.cpp
            src/renderer.cpp
            src/shader_generator.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp)
        target_include_directories(arbit-depth-fog-gl-pixel-tests PRIVATE src)
        target_compile_definitions(arbit-depth-fog-gl-pixel-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
        target_link_libraries(arbit-depth-fog-gl-pixel-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
        target_compile_options(arbit-depth-fog-gl-pixel-tests PRIVATE -Wall -Wextra -Werror)
        add_test(NAME depth_fog_gl_pixels COMMAND arbit-depth-fog-gl-pixel-tests)
        set_tests_properties(depth_fog_gl_pixels PROPERTIES
            LABELS "native-depth-fog-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)
    endif()

    # The production particle GL path requires OpenGL 4.3 shader-storage
    # buffers. macOS exposes at most OpenGL 4.1 and uses the native Metal path.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(arbit-particle-engine-gl-tests
            tests/particle_engine_gl_tests.cpp src/particle_engine.cpp src/gl_loader.cpp)
        target_include_directories(arbit-particle-engine-gl-tests PRIVATE src)
        target_compile_definitions(arbit-particle-engine-gl-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
        target_link_libraries(arbit-particle-engine-gl-tests PRIVATE PkgConfig::GLFW3 OpenGL::GL)
        add_test(NAME particle_engine_gl COMMAND arbit-particle-engine-gl-tests)
        set_tests_properties(particle_engine_gl PROPERTIES
            LABELS "native-particle-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)
    endif()
endif()

add_executable(arbit-node-preview-presentation-tests tests/node_preview_presentation_tests.cpp)
target_include_directories(arbit-node-preview-presentation-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-node-preview-presentation-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME node_preview_presentation COMMAND arbit-node-preview-presentation-tests)

if(APPLE AND ARBIT_WITH_METAL)
    add_executable(arbit-metal-backend-tests
        tests/metal_backend_tests.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-backend-tests PRIVATE src third_party/sokol)
    target_link_libraries(arbit-metal-backend-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    add_test(NAME metal_backend COMMAND arbit-metal-backend-tests)

    if(GLFW3_FOUND AND OpenGL_FOUND)
        add_executable(arbit-metal-iosurface-consumer
            tests/metal_iosurface_consumer.mm)
        target_link_libraries(arbit-metal-iosurface-consumer PRIVATE
            "-framework Metal" "-framework IOSurface" "-framework Foundation")

        add_executable(arbit-metal-particle-smoke
            tests/metal_particle_smoke_tests.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/gpu_backend/backend_sokol.mm)
        target_include_directories(arbit-metal-particle-smoke PRIVATE src third_party/sokol)
        target_compile_definitions(arbit-metal-particle-smoke PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1)
        target_link_libraries(arbit-metal-particle-smoke PRIVATE
            PkgConfig::GLFW3 OpenGL::GL
            "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
            "-framework IOSurface")

        add_executable(arbit-metal-frame-compositor-smoke
            tests/metal_frame_compositor_smoke_tests.cpp
            src/gl_loader.cpp
            src/gpu_backend/metal_shader_generator.mm
            src/gpu_backend/backend_sokol.mm)
        target_include_directories(arbit-metal-frame-compositor-smoke PRIVATE
            src third_party/sokol
            ${ARBIT_MVK_SHADER_CONVERTER_INCLUDE}
            ${ncnn_SOURCE_DIR}/glslang)
        target_compile_definitions(arbit-metal-frame-compositor-smoke PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1
            ARBIT_HAVE_METAL_GENERATORS=1)
        target_link_libraries(arbit-metal-frame-compositor-smoke PRIVATE
            PkgConfig::GLFW3 OpenGL::GL
            "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
            "-framework IOSurface" "-framework Foundation"
            nlohmann_json::nlohmann_json
            glslang SPIRV glslang-default-resource-limits
            ${ARBIT_MVK_SHADER_CONVERTER_LIBRARY}
            ${ARBIT_MVK_SPIRV_CROSS_LIBRARY}
            ${ARBIT_MVK_SPIRV_TOOLS_LIBRARY})
        add_dependencies(arbit-metal-frame-compositor-smoke
            arbit-metal-iosurface-consumer)
        add_test(NAME metal_frame_compositor_smoke
                 COMMAND arbit-metal-frame-compositor-smoke)
    endif()
endif()

if(Python3_Interpreter_FOUND AND UNIX)
    add_test(NAME concurrent_recorder_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrent_recorder_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    add_test(NAME capture_source_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_source_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    set_tests_properties(concurrent_recorder_smoke capture_source_smoke PROPERTIES
        TIMEOUT 30
        RUN_SERIAL TRUE)
endif()
