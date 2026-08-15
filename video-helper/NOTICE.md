# Third-party notices — arbit-video-helper

`arbit-video-helper` is GPL-2.0-or-later (see `COPYING` and `README.md`).
This file lists the third-party code vendored in this tree, the libraries
fetched or linked at build time, and source pointers for the prebuilt
libraries bundled next to the helper binary in the DonutStudio
installers/zips.

## Vendored in this tree

| Component | Location | License |
|---|---|---|
| QuickJS-ng (Bellard, Gordon, Noordhuis, Ibarra Corretgé) | `third_party/quickjs/` | MIT (`third_party/quickjs/LICENSE`) |
| xsum (Radford M. Neal; sub-vendored by QuickJS-ng) | `third_party/quickjs/xsum.{c,h}` | MIT-style (per file headers) |
| dirent for Windows (Toni Rönkkö; sub-vendored by QuickJS-ng) | `third_party/quickjs/dirent_compat.h` | MIT (per file header) |
| SPIRV-Cross (Arm Limited) | `third_party/spirv_cross/` | Apache-2.0 **OR** MIT (dual, per-file SPDX) — see note below |
| Khronos SPIR-V headers | `third_party/spirv_cross/spirv.h(pp)`, `GLSL.std.450.h`, `NonSemanticShaderDebugInfo100.h` | MIT (Khronos Group) |
| rife-ncnn-vulkan net code + GLSL compute shaders (nihui) | `src/rife_net/` | MIT (`src/rife_net/LICENSE.nihui`) |

**SPIRV-Cross license election.** Every SPIRV-Cross source file in
`third_party/spirv_cross/` carries `SPDX-License-Identifier: Apache-2.0
OR MIT`. For compatibility with GPL version 2 we **elect the MIT
option**. The vendored `third_party/spirv_cross/LICENSE` file contains
only the Apache-2.0 text as shipped by upstream; the per-file SPDX
markers are authoritative for the dual licensing. Even if SPIRV-Cross
were treated as Apache-2.0-only, this project's GPL-2.0-or-later terms
permit distribution under GPL version 3, with which Apache-2.0 is
compatible — so the combined work is distributable either way.

**RIFE model weights** are not in this tree; they are downloaded at
first use from `donutsdelivery.online` with pinned SHA-256 hashes. They
are MIT-licensed (hzwer's Practical-RIFE, ONNX conversion via
AmusementClub/vs-mlrt) — provenance documented in `PROTOCOL.md` (RIFE
section).

## Fetched at build time (FetchContent / pkg-config)

- **ncnn** (Tencent) — BSD-3-Clause — fetched as the upstream
  full-source release and patched with
  `patches/ncnn_enable_external_memory_fd.cmake` (first-party patch)
- **Vulkan-Headers** (Khronos) — Apache-2.0 / MIT
- **nlohmann_json** — MIT (system package preferred, FetchContent fallback)
- **ONNX Runtime** (Microsoft, optional `ARBIT_WITH_ONNX`) — MIT (prebuilt release download)
- **FFmpeg** (libavformat/avcodec/avutil/swscale/swresample/avfilter) — LGPL-2.1+/GPL-2.0+ depending on build configuration (see bundled-libraries section)
- **aubio** (optional) — GPL-3.0-or-later — where linked, the effective license of the resulting binary is GPL-3.0-or-later
- **GLFW** (optional) — zlib/libpng
- **Lua** 5.3/5.4 (optional) — MIT

## Prebuilt libraries bundled with the shipped binaries

The DonutStudio Windows and macOS bundles ship prebuilt shared libraries
next to `arbit-video-helper`. None of them are modified by us. Sources:

- **FFmpeg** — <https://ffmpeg.org> — the shipped builds are configured
  with **x264** (<https://www.videolan.org/developers/x264.html>) and
  **x265** (<https://www.x265.org>), i.e. GPL-enabled FFmpeg builds.
- **libmp3lame** (LAME) — LGPL-2.1 — <https://lame.sourceforge.io>
- **libgnutls** (GnuTLS) — LGPL-2.1+ — <https://www.gnutls.org>
- **libgme** (game-music-emu) — LGPL-2.1 —
  <https://github.com/libgme/game-music-emu>
- **Windows DLLs** are unmodified MSYS2 MINGW64 packages; exact package
  versions and their corresponding source packages are available at
  <https://packages.msys2.org> (source archives at
  <https://repo.msys2.org>).
- **macOS dylibs** come from Homebrew bottles; the formulae (with
  upstream source URLs and patches) are at
  <https://github.com/Homebrew/homebrew-core>.

The complete corresponding source for the helper shipped with DonutStudio v0.6.0 is this repository tree. The exact app release, private source commit, and helper-tree digest are recorded in `SOURCE-CORRESPONDENCE.json`. Source locations for bundled third-party libraries are listed above.
