# arbit-video-helper — the GPL video sidecar for DonutStudio

This directory is the **complete corresponding source** for the
`arbit-video-helper` executable that ships inside the DonutStudio
installers and zips (Windows, macOS, Linux). DonutStudio itself is a
closed-source application; all of its video functionality lives in this
separate GPL-licensed sidecar process, and publishing this directory in
the public DonutStudio repository is how we satisfy the GPL source-access
obligation for those bundled binaries.

The internal names (`arbit-video-helper`, `arbit_*` symbols, `ARBIT_*`
CMake options) come from Arbit, the product's former name, and are kept
as-is so the published source matches the shipped binaries exactly.

## What it does

The helper owns everything that touches FFmpeg and other GPL/LGPL media
libraries, which must never be linked into the proprietary DonutStudio
binaries:

- video decode and thumbnailing
- audio extraction and beat detection (aubio, optional)
- the OpenGL render/compositing engine used by the video editor
  (generators, shaders, adjustment clips, mod matrix, Lua/JS hooks)
- video export (encode via FFmpeg, including x264/x265 where enabled)
- optional RIFE frame interpolation (ncnn-Vulkan by default, ONNX Runtime
  optionally); model weights are downloaded at first use, not bundled —
  see `PROTOCOL.md` for provenance and pinned hashes

## How DonutStudio talks to it

- **JSON-RPC over stdin/stdout** for commands — the full protocol is
  documented in [`PROTOCOL.md`](PROTOCOL.md), with the ping/pong and
  lifecycle rules in [`PINGPONG.md`](PINGPONG.md).
- **A shared-memory frame ring** for decoded RGBA frames
  ([`shared/VideoFrameSharedMemory.h`](shared/VideoFrameSharedMemory.h)).
- **An optional zero-copy GPU surface channel** (dmabuf / IOSurface /
  D3D11 shared textures over a unix-domain socket,
  [`shared/SharedGpuSurfaceProtocol.h`](shared/SharedGpuSurfaceProtocol.h)).

The two headers in `shared/` are the process boundary: dependency-free
POD/IPC protocol definitions that are compiled into both the proprietary
plugin and this GPL helper. In the private tree they live at
`plugin/Source/`; they are vendored here (and the include path adjusted
in `CMakeLists.txt`) so this directory builds standalone.

## Building

The helper is a standalone CMake project — configure from this directory:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
# → build/arbit-video-helper
```

Key CMake options:

- `ARBIT_WITH_NCNN` (default **ON**) — RIFE frame interpolation via
  ncnn + Vulkan (ncnn is fetched and built from source via FetchContent,
  with the patch in `patches/`)
- `ARBIT_NCNN_ZEROCOPY` (default ON on Linux) — zero-copy Vulkan→GL
  RIFE output; needs `libvulkan-dev` at build time
- `ARBIT_WITH_ONNX` (default OFF) — alternative RIFE backend via a
  prebuilt ONNX Runtime download

### Linux

Official binaries are built in an Ubuntu 24.04 container (the helper
needs FFmpeg 6's channel-layout API, so Ubuntu 22.04's FFmpeg 4.4 is too
old). Build dependencies:

```sh
apt-get install cmake ninja-build pkg-config build-essential git \
    libavformat-dev libavcodec-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libaubio-dev nlohmann-json3-dev libvulkan-dev
```

then run the `cmake`/`ninja` commands above. Optional extras picked up
via pkg-config when present: `aubio` (beat detection), `lua5.4`/`lua5.3`
(Lua frame hooks), `glfw3` + OpenGL/EGL (render engine, viewport).

### Windows

Official Windows binaries are cross-compiled from Linux with
MinGW-w64 (POSIX threads variant) against the unmodified MSYS2 MINGW64
packages for FFmpeg, aubio, GLFW, etc.:

```sh
# toolchain: g++-mingw-w64-x86-64 (posix), cmake, ninja
cmake -S . -B build-win -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=<mingw-w64-x86_64 toolchain file> \
    -DCMAKE_BUILD_TYPE=Release
ninja -C build-win
```

with `CMAKE_PREFIX_PATH`/`PKG_CONFIG` pointed at an MSYS2 MINGW64
sysroot (packages from <https://packages.msys2.org>). A native MSYS2
MINGW64 shell build with the same packages works too. The DLLs shipped
next to the exe in the DonutStudio Windows bundle are those unmodified
MSYS2 packages (see `NOTICE.md` for source pointers).

### macOS

Native build with the dependencies from Homebrew
(`ffmpeg`, `aubio`, `glfw`, `nlohmann-json`, `lua`, `pkg-config`):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## License

`arbit-video-helper` is licensed **GPL-2.0-or-later** — see
[`COPYING`](COPYING) for the full GNU General Public License version 2
text.

Vendored third-party code in this tree (QuickJS-ng, SPIRV-Cross,
rife-ncnn-vulkan net code) is under MIT / dual Apache-2.0-OR-MIT
licenses; see [`NOTICE.md`](NOTICE.md) for details, including the note
that we elect the MIT option for SPIRV-Cross. Because the shipped
binaries link GPL-licensed FFmpeg builds (with x264/x265) and, where
enabled, aubio (GPL-3.0-or-later), the **effective license of the
distributed helper binaries is GPL-3.0-or-later**; the source in this
directory remains available under GPL-2.0-or-later as declared in
`CMakeLists.txt`.

## Release correspondence

This source corresponds to the video-helper bundled with **DonutStudio v0.7.1**. It was exported from private source commit `37c24598ebf52255562190b9340f9646c2c3d622` with helper-tree SHA-256 `b9f43d798fa5b351becdff5bf90cf6555afddec0bcc4cd67e968bb7e418b09a7`. A later app release may reference this same public source commit only when its distributed helper binary and helper-tree receipt are unchanged.
