# arbit-video-helper — viewport & render-graph protocol

JSON-RPC 2.0 over stdio, line-delimited (same transport as all other helper
methods). This file documents the GPU viewport surface added Jun 2026.

## Native GPU backend diagnostics

`gpu_backend_info` performs a cheap native-device query and returns
`{available, backend, device, compute, error}`. On Apple builds with
`ARBIT_WITH_METAL=ON`, `backend` is `"metal"`; this query submits no GPU work.

`gpu_backend_selftest` initializes the P6 `sokol_gfx` Metal backend for one
isolated test, dispatches a deterministic compute kernel, renders a known colour
into an offscreen Metal texture, reads both results back, and returns
`computePassed`, `renderPassed`, and stable checksums.

The first renderer-facing P6 slice is enabled by default on Apple builds:
particle-layer simulation and point rasterization run natively in Metal. The
result remains on the GPU: an IOSurface
bridges the Metal BGRA8 target into the existing OpenGL compositor. If Metal,
pipeline creation, or the bridge is unavailable, the layer automatically uses
the existing macOS OpenGL 4.1 fragment/MRT particle path. Set
`ARBIT_VIDEO_METAL=0` only to force that recovery path. Other layer generation,
effects, compositing, and presentation remain OpenGL, so
`viewport_info.gpuPath` and `presentPath` still report that real end-to-end
path. `viewport_info.compositorBackend` reports the production compositor used
for the latest frame (`metal` or `opengl`), while `particleBackend` separately reports `metal`,
`opengl-fragment`, `pending`, or `none` for the latest composite frame.

Apple developer builds also produce `arbit-metal-particle-smoke`. Run it from
an ordinary macOS Terminal session to create a hidden CGL context, render the
same deterministic note-fed particle frame through both the GL fallback and
Metal/IOSurface bridge, and print hashes plus coverage statistics. It fails if
either path is blank, if Metal silently fell back to GL, or if particle coverage
diverges grossly. It is deliberately not registered as an automatic CTest
because headless CI and sandboxed service processes have no Aqua/WindowServer
connection.

## Clock: transport shared-memory block

Arbit creates a tiny shm region (`videoshm::TransportBlock`, seqlock) and
tells the helper its name via `attach_transport {name}`. Arbit's **audio
thread** writes it once per audio block: `{playing, playheadBeats, bpm,
sampleTime, sampleRate, hostTimeNs}`. Transport version 2 also carries at most
eight stable-track-ID feature slots (`tapPoint`, RMS, peak, onset) in the same
seqlock transaction. These are bounded typed scalars derived from admitted
Session Graph sources; no per-track raw-audio rings cross the process boundary.
Missing/stale track owners evaluate to zero. Per-track spectral bands are not
silently approximated in version 2: plans must use master `AudioBand` or a
track/group RMS, peak, or onset source. Export rejects plans containing these
live-only track/group sources until the job carries a deterministic per-track
feature stream; it never substitutes the master mix. The helper's render loop
reads it lock-free each frame and extrapolates the playhead against its monotonic
clock (extrapolation clamped to 0.5 s so a stalled writer freezes rather than
runs away). `viewport_set_beat_timeline` supplies the processor-owned
`tempoMap[]` (`beat`, `bpm`, `interpolation`) and `timeSignatureMap[]`
(`beat`, `numerator`, `denominator`). The shared-memory beat is converted
through that immutable map; no RPC is involved per frame. Audio is master;
video chases.

## Window management

| method | params | notes |
|---|---|---|
| `viewport_open` | `width, height, x, y, alwaysOnTop, targetFps` | Opens the helper-owned window (GLFW + OpenGL, vsync-paced). Blocking until the window is up; errors if no display. Idempotent. |
| `viewport_close` | — | Destroys the window and decoders. |
| `viewport_set_bounds` | `x, y, width, height` | Applied on the next render tick. |
| `viewport_set_fullscreen` | `fullscreen` | Primary monitor. |
| `viewport_set_canvas` | `width, height` | §Project canvas & view transform below. Accepted while closed (applies to the next open); survives close/reopen. |

Embedding/reparenting into the host editor is explicitly out of scope; the
window floats (optionally always-on-top), matching Arbit's undock pattern.

## Project canvas & view transform

`viewport_set_canvas {width, height}` declares the PROJECT CANVAS — the
export-frame coordinate space. When set (Arbit always sets it):

- Per-layer letterboxing and text-overlay sizing are computed against the
  canvas aspect/pixels instead of the surface, so the viewport composition
  matches an export at canvas resolution regardless of the surface shape.
- The canvas is fitted inside the viewport surface through a VIEWPORT-ONLY
  view transform, controlled via the param namespace: `view/zoom` (1 = the
  canvas exactly fits the surface, letterboxed), `view/panX`, `view/panY`
  (output NDC, x right / y down). Zooming out (< 1) really renders layer
  content that overflows the canvas — layers are not clipped to it.
- A post-composite pass dims everything outside the canvas rect and draws a
  crisp ~1.5 px violet border at the canvas bounds (constant thickness in
  surface pixels at any zoom). Scopes keep measuring the undecorated
  composite; `frameHash` hashes the PRESENTED (framed) output.

The EXPORTER never sees any of this: it runs its own `FrameRenderer` with
canvas unset (output == canvas) and identity view, so exported pixels are
bit-identical whatever `view/*` says. Arbit defaults the export resolution to
the project canvas so viewport framing and export agree.

`view/*` values and the canvas live on the helper's Viewport object: they
survive viewport close/reopen but reset with the helper process (Arbit
re-pushes them on restart replay). `viewport_info` reports `canvasWidth,
canvasHeight, viewZoom, viewPanX, viewPanY`; `graph_describe` mirrors them
under a top-level `canvas` object.

## Timeline

`viewport_set_timeline {structuralRevision, matteContentRevision, segments: [{sourceKind?, sourcePath,
clipId, isAdjustment?, inSec, outSec, rate, displayStartSec, matteAssetId?,
matteAssetVersion?, matteContentReceipt?, matteState?, matteCacheKey?, matteFramePrefix?,
matteFrameExtension?, matteFirstFrame?, matteFrameDigits?, matteFps?, matteFrames?,
transition?: {type, durationSec, fromClipId?, toClipId?}}]}` — the same constant-rate segment shape
the exporter uses. Matte bindings carry the complete project-owned identity/naming tuple,
and use only a processor-published immutable cache key. The helper accepts its trusted
absolute cache root exclusively from the launch-time `--matte-cache-root` argument,
canonicalizes it, and rejects wire/project-controlled roots. It validates the manifest,
receipt, containment, file type, byte bounds, and every named frame before admission.
`structuralRevision` is monotonic and reserved for topology,
resource, or compile-affecting publications; it is not advanced by slider,
automation, or modulation values.
`isAdjustment=true` explicitly declares a read-only `CompositeBelow` input;
helpers still recognize the legacy `gen://adjustment` source sentinel while old
projects migrate. Matte fields authoritatively identify the baked alpha sequence;
preview, composite probes, and export derive the decoder pattern and start number
from `matteFramePrefix`, `matteFrameExtension`, `matteFirstFrame`, and
`matteFrameDigits` after revalidating the immutable cache immediately before open.
This authority boundary protects against untrusted project and imported-media data.
A process already compromised under the same OS principal can alter application state
or race pathname opens and is outside application authority; this protocol does not
claim protection against same-principal process compromise.
`sourceKind` is one of `media|shader|particles|adjustment`. When present it is
authoritative; `isAdjustment` and `gen://` sentinels are fallback-only readers.
Likewise, transition `fromClipId`/`toClipId` are authoritative A/B owners.
Same-track timeline-order inference runs only when both structured IDs are absent.

Preview and export pass this wire shape through the same immutable
`ResolvedVisualSnapshot` normalizer before either render path sees it. Snapshot
construction is read-only: it does not promote legacy graphs, rewrite project
state, or publish undoable mutations. Explicit invalid source, transition, matte,
or timing bindings are rejected; compatibility inference is absence-only.

The timeline acknowledgement and `graph_describe` diagnostics expose:
`authoringRevision`, `acceptedRevision`, `rejectedRevision`, `compiledRevision`,
`lastGoodRevision`, `exportableRevision`, and `evaluationSequence`. A runtime
`graph_set_param` advances evaluation without advancing authoring/compiled state.
New authoring immediately clears current exportability; rejection preserves the
previous compiled/last-good revision but export must not silently use it.
`composite_frame_probe` accepts one immutable processor-owned offline-export
snapshot plus one `timelineSec` and bounded canvas dimensions. It returns the
production offscreen compositor's exact RGBA8 readback, processor snapshot
generation, per-clip authored structural revision and snapshot
compiled/exportable validity, compositor backend, and readback presentation
backend. It has no output/source path field and never opens a viewport. Every
current authored clip plan must be valid in the snapshot; valid clips may carry
unequal revisions. The live viewport helper's admission, last-good, rejection,
or generation state is not authority for this RPC and is not claimed. Export,
render-cache, or another probe makes the compositor deterministically busy. On
macOS the compositor must report a native Metal backend and the
presentation identity is `metal-iosurface-cpu-readback`; there is no OpenGL or
CPU-renderer fallback. Other platforms report the offscreen OpenGL async
readback explicitly.
Arbit computes segments from the beat-marker mapping (`VideoBeatMath.h` is the
single source of truth); the helper only evaluates
`sourceSec = inSec + (displaySec − displayStartSec) × rate` for the segment
containing the current display time. Re-send the whole list on any clip edit.

v2 additions per segment (all optional, defaults preserve v1 behavior):

- `trackLayer` (int, default 0) — compositing layer. The renderer gathers
  **all** segments active at the display time and composites them ordered by
  `(trackLayer, clip zOrder)` instead of drawing only the first match.
- `transition {type, durationSec}` — transition on this segment's **leading
  edge**. `type`: 0 none, 1 fade, 2 dissolve, 3 wipe-left, 4 wipe-right,
  5 wipe-up, 6 wipe-down (reference-editor enum order). When the previous
  segment on the same trackLayer abuts/overlaps, it is the A source (cross
  transition); otherwise A = black. Speed ramps never reach the helper:
  Arbit subdivides ramped clips into constant-rate segments
  (`VideoBeatMath::applySpeedRamp`).
- `sourceFps` (double, default 0) + `seqStart` (int, default -1) —
  image-sequence hints: when `sourcePath` is an image2 printf pattern
  ("img_%04d.png"), `sourceFps` sets the sequence frame rate and `seqStart`
  the first frame number. Ignored for regular media. The exporter's jobSpec
  segments accept the same two fields.

## Alpha-channel sources

The decode → shm → GL pipeline is RGBA end to end (the shm slot payload was
always 4-channel — no layout/version change), so per-pixel alpha flows from
the decoder into the compositor for free. Sources whose pix_fmt carries
alpha (yuva420p, yuva444p10, rgba/bgra, ProRes 4444, qtrle 32-bit, PNG, ...)
and webm/mkv VP8/VP9 alpha (`alpha_mode=1` stream metadata, decoded via the
libvpx decoders — the native vp8/vp9 decoders drop the alpha plane) convert
through swscale to **straight (non-premultiplied) RGBA**, the same
convention text-overlay images use. The blend pass multiplies the layer's
per-pixel alpha into the composite (`mix(back, front, front.a × opacity)`),
so lower layers show through transparent areas in both the live viewport
and the GL export path (export output itself stays opaque yuv420p — alpha
matters for blending, not for the file).

Alpha sources always decode in **software**: hardware surfaces are
NV12/P010 and would silently drop the alpha plane. Opaque sources keep the
hardware decode path. `probe`/`open` report `hasAlpha` per media. Caveats:
edit proxies and stabilized intermediates are opaque yuv420p H.264 — a
proxied/stabilized alpha clip composites without its alpha (use the
original source when transparency matters).

## Image sequences

`open`/`probe` accept optional `fps` (double) and `startNumber` (int)
params. When `path` contains a printf pattern ("…/img_%04d.png") the file
opens through FFmpeg's image2 demuxer with `framerate = fps` (default 30)
and `start_number = startNumber` (default: ffmpeg scans 0–4). The probe
reports the resulting fps and duration (frameCount / fps), `hasAudio:
false` (callers must treat a missing audio stream as "silent clip", not an
error) and `hasAlpha` from the image pix_fmt — PNG sequences with
transparency composite per-pixel like any other alpha source.

## Render graph

Fixed v1 topology per clip: `source → retime → transform2d → display`.
Every parameter has a stable string ID `clip<id>/<node>/<param>`:

| node | params | meaning |
|---|---|---|
| `source` | `opacity` (0–1), `visible` (0/1), `zOrder`, `blendMode` (0 normal, 1 add, 2 multiply, 3 screen, 4 overlay) | compositing controls; vocabulary mirrors the original editor's `CompositorLayer` |
| `transform2d` | `scale`, `translateX`, `translateY` (NDC, −2..2), `rotation` (deg), `cropLeft`, `cropRight`, `cropTop`, `cropBottom` (0–1 fraction removed) | crop remaps texture coordinates before the letterbox fit + transform |
| `effect<slot>` | `type` (videofx::EffectType, −1 empty), `enabled` (0/1), then per-effect params by wire name (see `src/effect_defs.h`, e.g. `clip7/effect2/radius`) | ordered rack, slots 0–7; single uber-shader pass keyed by effect bitmask, blur/sharpen as a neighborhood pass |
| `mask` | `type` (0 none, 1 rect, 2 ellipse), `cx`, `cy` (shape centre), `w`, `h` (full extents), `feather`, `invert` (0/1) | per-clip shape mask (Jun 2026): multiplies the layer alpha in the geometry pass, so the masked alpha feeds the blend/composite. Coordinates are normalized 0..1 over the clip's **displayed (post-crop) frame**, origin top-left, +y down — the mask travels with the clip through translate/rotate/scale. `feather` is an edge-soften width in the same units (rect: inward from the edge; ellipse: scaled by the mean extent so it visually matches). cx/cy/w/h/feather are automatable; type/invert are static. |
| `gen` | `<isfInputName>` (+ `.x`/`.y` for point2D, `.r`/`.g`/`.b`/`.a` for color) — one per ISF INPUT the clip's shader declares (M7) | a shader-generator clip's own INPUTS as automatable params: `clip<id>/gen/<name>`. Scalar INPUTS (`bool`/`long`/`float`/`event`) are one key; **vector INPUTS are addressed per component** — `clip<id>/gen/tint.r`, `clip<id>/gen/pos.x` — so a single channel can be automated/modulated independently. An unset INPUT/component renders at the INPUT's declared ISF default. Settable statically (`clips[].genParams`), by baked automation, and as a **mod-matrix destination** (so a musical source can drive a shader's inner parameter). `image` INPUTS are bound from a file instead — `clips[].genImages: {<name>: path}` decodes the image once and binds it to the `sampler2D` (static, not a value param). `audio`/`audioFFT` sampler INPUTS are bound to the mix spectrum (the same nx1 band texture as the contract `uAudioBands2D`), so a stock ISF audio-reactive shader's own named INPUT reacts to the audio; black in silence. |
| `retime` | (none yet) | rate comes from the timeline segments; interpolation params land here with RIFE |

Color-suite effects (Jun 2026, `src/effect_defs.h` appended — new types show up
in the rack UI automatically): `chroma_key` (keyR/keyG/keyB, tolerance,
softness, spillSuppress — alpha from CbCr-plane distance to the key color,
spill suppression clamps the dominant key channel), `luma_key` (low, high,
softness, invert — keeps luma in [low, high]) and `color_wheels` (lift/gamma/
gain × RGB, 9 params: `out = pow(clamp(in·gain + lift·(1−in)), 1/gamma)`).
Uber-shader order: **keying first** (on the source colors, so the produced
alpha feeds the blend pass), then the basic adjustments (exposure …hue),
then the wheels, then the grading looks/stylize/vignette. `kMaxEffectParams`
was bumped 4 → 9 for the wheels; both serializers, the jobSpec and the
automation subParam packing are count-driven and unaffected by the bump.

Text overlays are addressable as `text<id>/{opacity, posX, posY, translateX,
translateY, visible, zOrder}` once registered via `text_set_image`.

- `graph_set_param {paramId, value, atBeat?, interpolation?, runtimeRevision?}` — immediate
  when `atBeat` is omitted/negative; otherwise queued and applied when the
  playhead passes `atBeat`. `interpolation` accepts
  `linear|hold|ease_in|ease_out|ease_in_out` (keyframe vocabulary reserved;
  v1 applies values as steps). This is the hook Arbit automation lanes will
  drive. `runtimeRevision` is a separate monotonic stream: accepting it never
  requests graph validation or recompilation.
- `graph_describe` — returns `{nodes:[{clipId, topology, params:{id:value}}]}`;
  v2 adds per-clip `effects:[{slot,type,enabled,params}]` and a top-level
  `texts:[{textId, params}]`; it also reports the latest accepted
  `structuralRevision` and `runtimeRevision` for restart/debug inspection.
- `graph_set_effects {clipId, effects:[{slot, type, enabled, params:{name:value}}]}`
  — bulk (re)configure a clip's whole rack in one call (add/remove/reorder).
  Individual sliders and automation use per-param `graph_set_param`.
- `graph_set_lut {clipId, path}` — attach a `.cube` 3D LUT to a clip
  (path-carrying RPC: the helper parses the file on the RPC thread and the
  render thread uploads a `GL_RGB32F` 3D texture, sampled trilinearly with
  texel-centre remap `c·(N−1)/N + 0.5/N` so an identity cube is bit-stable).
  `path: ""` clears the LUT. Errors on unreadable/malformed files and
  unsupported sizes (`LUT_3D_SIZE` 2..129 accepted; 17/33/65 are the common
  sizes; 1D LUTs rejected; `DOMAIN_MIN/MAX` parsed but the domain is assumed
  0..1). The LUT applies **after** the effects rack (color pass +
  blur/sharpen), before the geometry pass. `graph_describe` reports
  `lutPath`/`lutSize` per clip.

Adding effect nodes or a user-editable DAG later extends `topology` and the
param namespace without breaking existing IDs.

## Text overlays

The helper never rasterizes text — Arbit renders each overlay with JUCE
(fonts, outline, shadow, background) into **straight (non-premultiplied)**
RGBA (top row first, stride width×4 — the compositor's blend pass computes
`mix(back, front, front.a × opacity)`, i.e. straight-alpha) and ships the
pixels; preview and export therefore match exactly.

- `text_set_image {textId, ownerClipId?, shmName, width, height, startSec,
  durationSec, posX, posY (NDC −1..1, +posY = down like translateY), opacity,
  zOrder}` —
  register/update an overlay. Pixels are read once (while handling the RPC)
  from slot 0 of the named shm region (same mechanism as the frame ring) and
  uploaded to a texture by the render thread; the region can be reused or
  unlinked afterwards. Re-sending a textId replaces its image. The image
  composites at natural pixel size, centred at (posX, posY); whole-overlay
  opacity lives in the `opacity` param (not baked into pixels) so opacity
  automation needs no re-rasterize. Texts survive timeline resends and are
  cleared on viewport close.
- `text_remove {textId}` — drop the overlay and its texture (idempotent).

## Zero-copy docked viewport

`viewport_open_shared {width, height, targetFps, bufferCount (default 3,
clamped 2..4)}` — instead of opening an OS window, the helper renders the
graph offscreen (hidden GLFW window) into a ring of GPU buffers that the
Arbit process maps directly — no pixel ever crosses through CPU memory. The
buffer mechanism is per-platform (one backend compiled per OS,
`src/gpu_export_common.h`):

- **Linux (`gpuPath:"dmabuf"`)** — EGL context, FBO-attached RGBA8 textures
  exported once as dmabufs (`eglCreateImage(EGL_GL_TEXTURE_2D_KHR)` +
  `eglExportDMABUFImageMESA`); fds ride the socket via SCM_RIGHTS. Needs
  `EGL_MESA_image_dma_buf_export` (Mesa, NVIDIA ≥ 555).
- **macOS (`gpuPath:"iosurface"`)** — global IOSurfaces bound as
  `GL_TEXTURE_RECTANGLE` render targets via `CGLTexImageIOSurface2D`;
  `BufferInfo.modifier` carries the IOSurfaceID (a plain uint32 —
  `IOSurfaceLookup` resolves it cross-process, nothing to attach),
  `fourcc = 'IOSF'`, `planeCount = 0`.
- **Windows (`gpuPath:"d3d11"`)** — D3D11 keyed-mutex shared textures
  (`D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`, legacy global handles) bridged
  into GL on both sides with `WGL_NV_DX_interop2`;
  `BufferInfo.modifier` carries the shared HANDLE value (opened with
  `ID3D11Device::OpenSharedResource`, no handle duplication),
  `fourcc = 'D3DH'`, `planeCount = 0`. Keyed-mutex discipline: producer
  AcquireSync(0) → blit → ReleaseSync(1); consumer AcquireSync(1) → sample
  (held while displayed) → ReleaseSync(0) before FRAME_RELEASE.

Returns `{socketPath, gpuPath, fourcc, modifier, bufferCount, device}`
(`device` = exporting DRM render node on Linux, empty elsewhere); errors
with `"no shared-GPU export support: …"` when the context/driver can't
(GLX-backed context, missing export extension, no WGL interop) — Arbit then
demotes down the ladder (shm-docked first). Mutually exclusive with
`viewport_open`; timeline/params/transport state is shared by both modes.

Buffer announcement and frame signalling happen on the returned unix socket
(helper-owned, `$XDG_RUNTIME_DIR/arbit-vp-<pid>-<rand>.sock`; `%TEMP%` on
Windows). Transport: `SOCK_SEQPACKET` on Linux (one datagram == one
message); `SOCK_STREAM` on macOS/Windows (no SEQPACKET there), framed by
`MsgHeader.payloadBytes`, `fdCount` always 0. Message shapes in
`plugin/Source/SharedGpuSurfaceProtocol.h`: helper sends HELLO, then BUFFERS
(on Linux with one SCM_RIGHTS fd per plane, buffer order, planes within a
buffer in plane order; on macOS/Windows the handles are inline in
`modifier`), then per-frame FRAME_READY `{bufferIndex, flags, frameSeq,
displaySec, frameHash}`; the consumer sends FRAME_RELEASE when done sampling
a buffer. A buffer is never rewritten between FRAME_READY and its
FRAME_RELEASE; with no free buffer the frame is dropped. v1 sync =
`glFinish` before FRAME_READY (`kFlagFenceSync` fence fds are the Linux v2;
Windows additionally orders through the keyed mutex). Buffer rows are
top-down: texel row 0 (GL `t = 0`) is the image's top row. `frameHash` in
FRAME_READY is the same lagging composite hash as `viewport_info.frameHash`.
Helper restart/close closes the socket (EOF ⇒ all announced buffers
invalid); consumer disconnect frees all buffers and the helper re-accepts,
re-sending HELLO + BUFFERS to the next client.

`viewport_resize_shared {width, height}` — reallocates the ring: BUFFERS_GONE
on the socket (consumer drops its imports; its handles stay valid until
released), then a fresh BUFFERS at the new size. `viewport_close_shared {}`
— ends shared mode (BUFFERS_GONE, socket unlinked).

Arbit-side import (`SharedGpuSurfaceBackends.h` picks the backend): Linux
needs an EGL-backed JUCE context (native Wayland, `--wayland`) +
`EGL_EXT_image_dma_buf_import`; macOS imports into the JUCE NSOpenGL/CGL
context (`CGLTexImageIOSurface2D`, rectangle textures); Windows needs
`WGL_NV_DX_interop2` on the JUCE WGL context plus a hardware D3D11 device.
Anything missing demotes the present path.

`viewport_info` v2 adds `presentPath`: `"dmabuf-docked" |
"iosurface-docked" | "d3d11-docked" | "shm-docked" | "glfw-window" | "none"`
— the active rung as the helper sees it (the zero-copy name states the
platform mechanism; `"cpu-shm"` is reported Arbit-side when it falls back to
the `request_frame` ring) — and `dmabufCapable` (true once a shared-GPU
export probe succeeded for the current viewport — the field name is
historical and covers all three mechanisms).

## Docked viewport, CPU-shm rung (shm-docked)

`viewport_open_shm {width, height, targetFps, shmName}` — the rung between
dmabuf and glfw-window: the helper renders the SAME composited graph on a
hidden offscreen GL context (platform-default context API — GLX, WGL, CGL or
EGL all work; not gated on dmabuf support) and delivers frames through an
**Arbit-owned** `VideoFrameSharedMemory` ring named `shmName`, via pipelined
PBO readback (one frame of latency, one CPU copy per frame). This is the
docked preview everywhere zero-copy is unavailable: GLX/X11 sessions,
Windows, macOS, and DAW-hosted VST3/CLAP (host GL is never EGL there).

Ring usage (same `Header`/`SlotHeader` layout as the decode ring, distinct
region): Arbit creates the region (3 slots, slotBytes = width×height×4 at
open) and passes its name; the helper round-robins slots under the per-slot
seqlock (generation odd = write in progress) and self-describes each frame:
`width/height/strideBytes/ptsSec` as usual, plus `reserved[0]/[1]` =
frameSeq lo/hi (monotonic) and `reserved[2]` = 1 (payload format BGRA8, top
row first — memcpy-compatible with juce ARGB rows). The reader polls: scan
all slots, take the highest frameSeq with an even generation, re-validate
the generation after copying. The helper clamps its render size so a frame
always fits one slot.

`viewport_resize_shm {width, height}` — render-size change only (frames
self-describe their dims, no renegotiation). When the new size outgrows a
slot, Arbit instead closes, recreates the ring, and reopens.
`viewport_close_shm {}` — ends shm mode; the region stays Arbit-owned (it
survives helper restarts, so restart replay just re-issues
`viewport_open_shm`). Mutually exclusive with `viewport_open` /
`viewport_open_shared`; timeline/params/transport/canvas state is shared by
all modes.

Zero-copy is implemented on all three platforms (see §Zero-copy docked
viewport above); shm-docked remains the rung beneath it for GLX/X11
sessions, DAW-hosted plugins, drivers without export/interop support, and
helper builds without a shared-GPU backend.

## Recorder (piano-roll live capture → encode)

The exact reverse of the decode ring (`attach_shm`/`request_frame` above):
there, Arbit creates the shm region and the HELPER writes decoded frames for
Arbit to read. Here, Arbit creates the shm region and Arbit ITSELF writes
already-rendered piano-roll frames (its own `Component::createComponentSnapshot`
capture, not anything decoded or GL-composited by the helper) for the helper
to read and encode. Same `videoshm::Region` class, same seqlock ring, opposite
producer/consumer roles — no new shared-memory mechanism, just the existing
one used the other way.

This is a separate, much smaller code path than `export` (§Export below): no
GL compositing, no render graph, no audio, no ProRes/DPX — just raw BGRA
frames pushed in one at a time and muxed straight to a file. The encoder
selection (hardware nvenc/videotoolbox preference, falling back to software)
and BT.709 colour tagging are shared with `export` via the same
`pickVideoEncoder`/`tagColorBt709`/`encodeAndWrite` functions (`exporter.cpp`),
so a recording gets the same encoder choice an export would — the only new
code is the class that opens/feeds/closes them per-frame instead of from a
composited timeline (`RecorderSession`, `exporter.h`/`exporter.cpp`).

`record_open {width, height, fps, shmName, outPath, codec, encoder}` — Arbit
creates and names the shm region (sized for `width`×`height` BGRA8 frames)
and calls this with its name; the helper opens the region AND opens the
output file's encoder (`codec`: `"h264"` default | `"h265"`; `encoder`:
`"auto"` default (prefer hw) | `"software"`). Returns
`{ok: true, recordingId: N}`. `recordingId` uniquely owns this encoder and
shared-memory ring until it is closed or cancelled; multiple recording IDs
may be active concurrently.

`record_push_frame {recordingId, slot}` — Arbit has already written one BGRA8
frame into `slot` of that recording's shm ring (same seqlock write protocol
viewport-shm's helper→Arbit direction uses, just performed by Arbit here) and
calls this to have the helper read, scale (bilinear BGRA→YUV420P, matching
`export`'s sws pipeline), and encode that slot. Returns `{ok: true,
framesEncoded: N}`, or an error if the ID is unknown or the slot is torn.

`record_close {recordingId}` — removes only that recording from the helper's
session map, flushes its encoder, writes its trailer, and releases its shm
ring. Other recording IDs remain active.

`record_cancel {recordingId}` — performs the same isolated teardown, then
removes the partial output file.

## Live capture sources

`capture_list_sources {kind?}` enumerates devices without opening them. It
returns `{sources:[{id,name,kind,backend,availableFormats}], enumerationError}`.
Persist `id` verbatim: an unplugged device may later return with the same ID.
Passing `kind:"test"` exposes the deterministic FFmpeg `lavfi` test pattern
used by smoke tests when no physical camera is available.

`capture_preview_open {sourceId,backend,width,height,fps,shmName}` opens the
source on a helper worker thread and returns `{captureId}`. Arbit creates the
BGRA8 shared-memory ring named by `shmName`; the helper continuously publishes
the newest decoded frame there using the standard seqlock protocol.

`capture_preview_latest {captureId}` returns `{pending:true}` until the first
frame, then `{slot,sequence,timestampSec,width,height,strideBytes}`. Pixel data
never travels over JSON. Device-open, permission, unplug, decode, and timeout
errors are returned as structured RPC errors.

`capture_record_start {captureId,recordingId,outPath,codec}` attaches an
independent `RecorderSession` to that decoded source. Any number of recording
IDs may consume one capture source. `capture_record_stop
{captureId,recordingId,cancel?}` finalizes only that recording; `cancel:true`
also removes its partial output. `capture_close {captureId}` stops the worker,
finalizes attached recorders, and releases the preview ring.

## Introspection

`viewport_info` (alias `get_av_offset`) returns:
`{open, width, height, measuredFps, avOffsetSec, interpolationActive,
interpolationBackend ("nearest"|"rife-cuda"), gpuPath, particleBackend,
framesPresented, frameHash, gpu}`.

- `avOffsetSec` = presented frame pts − ideal source time (target: < 1 frame).
- `frameHash` = FNV-1a of the COMPOSITED output (first ~1 MB, read back
  asynchronously via PBO every few frames), so effect/param/transform changes
  alter the hash, not just new decoded frames. The value lags the visible
  frame by a few frames — poll after a short settle when testing.
- `gpu` = the GPU capability snapshot taken once after context creation (P1):
  `{glMajor, glMinor, computeShaders, ssbo, imageLoadStore,
  maxComputeWorkGroupCount:[x,y,z], maxComputeWorkGroupSize:[x,y,z],
  maxComputeWorkGroupInvocations}`. The context is requested at GL 4.3 core
  (4.1 on macOS, which caps desktop GL there → `computeShaders:false`).
  `graph_describe` mirrors the same object under its top-level `gpu` key.
  OpenGL compute features gate on `computeShaders`; where false they degrade
  or report unavailable rather than crashing. Apple particle layers are the
  exception: they use the native Metal backend and report that independently
  through `particleBackend`, even though the compositor's GL 4.1 capability
  snapshot correctly says `computeShaders:false`.

## Export (GL-composited, baked timeline)

`export {jobSpec}` renders offline through the SAME GL 3.3 FrameRenderer the
live viewport uses, so preview and export agree: multi-layer compositing
ordered by `(trackLayer, zOrder)`, per-clip effects rack, crop/transform,
blend modes and leading-edge transitions. The viewport is driven by live
`graph_set_param` pushes; the exporter is driven by the jobSpec's static clip
state plus a **baked automation timeline**. Both feed identical `LayerDesc`s.

**Codecs / containers.** `codec`: `h264` | `h265` | `vp9` | `prores`.
h264/h265 try hardware first under `encoder: "auto"` (`h264_nvenc` /
`hevc_nvenc`, `*_videotoolbox` on macOS) and fall back to software
(`libx264` / `libx265`) when the hw encoder is missing or fails to open.
vp9 is always `libvpx-vp9`; prores is always `prores_ks` (profile 3 =
ProRes 422 HQ, `yuv422p10le`) and **requires a `.mov` outPath** — mp4/mkv
muxers have no ProRes tag, so the helper rejects other extensions up front.
The result's `encoder` field reports what was actually used.

**Export range.** `startSec` / `endSec` (display-timeline seconds; both
optional) select a sub-range: output frame 0 samples the timeline at
`startSec`, the output ends at `endSec` (exclusive; may extend past the
content — black + audio). Omitted/0 = full timeline. When `endSec` is
explicit the audio track is also trimmed to exactly `endSec - startSec`
with a ~10 ms fade at the cut (Arbit renders the range mix WAV with its
usual decay tail, which full-timeline exports keep — "audio is
authoritative when longer" — but range exports must cut). Arbit's
`videoExportTimeline` maps the arranger time selection (beats × preview
BPM) onto these fields and renders the offline mix over the same beats, so
viewport playback of the selection and the exported range agree.

**Feedback pre-roll.** `feedbackPreRollSec` (optional, seconds; default 0)
warms cross-frame feedback (the `feedback_trail` effect) for a mid-range
export. Output frame 0 normally starts from a cleared feedback history, so a
range that begins mid-clip (`startSec > 0`) would show no accumulated trail
where a full-clip export has one. When set, the helper renders this many
seconds of warm-up frames before `startSec` — composited but not encoded —
so the trail at the range start matches a full-clip export (PINGPONG.md §4;
generative-visuals-research.md risk 13). Ignored unless some clip uses
`feedback_trail` and `startSec > 0`. Arbit supplies a beats-derived value
(default 8 beats).

**Tempo and meter.** `tempoMap[]` and `timeSignatureMap[]` drive every video
clock consumer: shader uniforms, score packing, modulation, scripts, seeks,
warm-up, and pending beat-stamped parameters. Linear tempo ramps use the exact
integral/inverse mapping. `bpm` (default 120) and `beatsPerBar` (default 4)
remain compatibility fallbacks when the arrays are absent. Export and viewport
use the same evaluator, so a beat-synced result exports exactly as it previews.

### Session Graph video Control plan

`controlPlan` is the bounded executable projection of every supported Session
Graph Control closure ending at a video parameter sink. The same object is
embedded in export jobs and sent live with `viewport_set_control_plan`.

- `version`: currently `1`.
- `numSlots`: fixed scalar slot count.
- `operations`: topologically ordered operations carrying exact input fan-in,
  resolved parameters, output slots, and sink shaping metadata.
- `diagnostics`: producer-side closure rejection messages.

The helper rejects the complete plan for unknown kinds, non-finite parameters,
out-of-range or multiply-owned slots, forward references, and capacity
violations. It never truncates or approximates topology. Legacy flat
`modMatrix` entries remain accepted for old projects; Graph-owned routes are
sent through `controlPlan` only, preventing double application.

**Async + progress + cancel.** `export` replies are DEFERRED: the helper
parses the jobSpec, spawns a render worker and answers the request id only
when the render finishes (the RPC loop keeps serving other methods
meanwhile; one export at a time — a second `export` while active errors
with `"export already running"`). While it runs:

- `export_progress {}` → `{active, done, cancelled, jobId, phase, frame,
  totalFrames, fps}` — cheap poll; `phase` ∈ `idle | setup | audio | video |
  finalize`, `frame`/`totalFrames` count output frames, `fps` is the
  wall-clock encode rate. After completion `done` is true and the final
  `result` (or `error`) rides along until the next export starts.
- `export_cancel {}` → `{ok, active}` — sets an abort flag the frame loop
  (and the audio transcode loop) polls; the worker stops within a frame,
  **removes the partial output file**, and the deferred `export` reply
  errors with exactly `"cancelled"`.

jobSpec fields (on top of the original outPath/width/height/fps/codec/
encoder/interpolation/audioPath/durationSec/startSec/endSec):

- `segments: [{sourcePath, clipId, trackLayer, inSec, outSec, rate,
  displayStartSec, transition?: {type, durationSec}}]` — the same
  constant-rate spans `viewport_set_timeline` receives (speed ramps arrive
  already subdivided by Arbit; `transition` sits on the clip's FIRST slice).
- `clips: [{clipId, scale, translateX, translateY, rotation, cropLeft,
  cropRight, cropTop, cropBottom, opacity, visible, zOrder, blendMode,
  effects: [{slot, type, enabled, params: {name: value}}],
  mask?: {type, cx, cy, w, h, feather, invert}, lutPath?, shaderSource?,
  genParams?, genImages?, isAdjustment?}]`
  — static render-graph state per clip (what `pushClipGraphState` streams
  live). `shaderSource` (optional GLSL string) makes the clip a **shader
  generator** — see §Shader generators below; its segments carry timing only
  (`sourcePath: "gen://shader"`), no media is decoded, and the effects rack /
  transform / blend still apply on top of the generated layer. When any shader
  generator is present and `audioPath` is set, the exporter also analyzes that
  mix WAV for the shader **Block B** audio uniforms (§Shader generators) — the
  same field that feeds the muxed audio track, reused.
  `mask` mirrors the `clip<id>/mask/*` node (cx/cy/w/h/feather also bake
  through `paramTimeline`). `lutPath` ships the `.cube` **as a path, not an
  inline table** — export runs on the same machine as the viewport, the
  helper already receives source-media paths the same way, and re-parsing
  the file at export start keeps the jobSpec small; the exporter fails
  loudly when the file is missing/malformed (the user attached it). Both
  the viewport and the exporter parse with the same `lut_loader.cpp` and
  render through the same LUT pass, preserving export == viewport parity.
  `genParams: {<isfInputName>: value}` (M7) sets static values for a
  shader-generator clip's ISF INPUTS (`clip<id>/gen/<name>`); keys not present
  fall back to the INPUT's declared ISF default, and baked automation /
  mod-matrix routings layer on top.
  `genImages: {<isfInputName>: "<path>"}` (M7 image follow-up) binds a file to a
  shader-generator clip's `"TYPE": "image"` INPUT — the named `sampler2D` reads
  that image. The helper decodes each path **once** (FFmpeg image2, software) at
  export start and binds the texture while the clip's shader renders; `texture(
  <name>, isf_FragNormCoord)` samples it upright (top row first). Unlike
  `genParams`, an image is **static** — it does not change over the clip, so it
  is not baked-automation / mod-matrix / Lua driven. A missing or undecodable
  path is a logged warning, not an error: the sampler falls back to 1×1 black
  (reads return 0 — the zero-feed contract), so the export never aborts on a bad
  image. Only meaningful for a clip with `shaderSource`.
  `isAdjustment: true` (M8) makes the clip an **adjustment clip** ("effect the
  world") — it has no source of its own (no media, no `shaderSource`): its
  `effects` rack runs over the **composite of every layer beneath it** and the
  result is mixed back over that composite at the clip's `opacity` (1 = full
  adjustment, &lt;1 = partial strength). Its segments carry timing/zOrder/
  trackLayer only (`sourcePath: "gen://adjustment"`, no media decoded); stack it
  above the layers it should grade via a higher `trackLayer`/`zOrder`. An
  adjustment clip with an empty/all-disabled rack is a pass-through. The clip's
  `mask`/`transform2d`/crop apply to the adjustment, so a masked or transformed
  adjustment grades only its **region** (outside the mask the composite shows
  through unchanged) — a local adjustment; a full-frame clip (no mask, identity
  transform) grades the whole picture. Mutually exclusive with `shaderSource`.
- `paramTimeline: [{paramId, atSec, value}]` — automation lanes baked by
  Arbit: sampled at every lane point, with curved spans (tension/sCurve,
  AutomationClip overrides) subdivided to ≤ 1/30 s. `paramId` is the usual
  `clip<id>/<node>/<param>` namespace, or `text<id>/<param>` for text
  overlay lanes (opacity, translateX, translateY); `value` is denormalized
  (native range). The helper interpolates **linearly** between samples of
  the same paramId and holds the first/last value outside them, overlaying
  the static clip/text state per output frame.
- `texts: [{textId, ownerClipId?, startSec, durationSec, posX, posY, opacity, zOrder,
  width, height, rgbaBase64}]` — text overlays. `rgbaBase64` is the SAME
  straight (non-premultiplied) RGBA block `text_set_image` ships for the
  live viewport (top row first, stride width×4, rasterized by Arbit),
  base64-encoded **inline in the jobSpec** instead of via shm: exports must
  work headless with no viewport open, and the text shm transport is
  serviced by the viewport's render thread. Project overlays and text owned by
  ordinary clips composite above all video layers in zOrder. Text owned by an
  adjustment clip is inserted immediately before that adjustment, so its
  transform/crop/mask/effects process the text and accumulated video together.
  Ties are broken by textId, matching the viewport. Each renders at its natural
  pixel size centred at `(posX + translateX,
  posY + translateY)` NDC, for `startSec ≤ t < startSec + durationSec`.
  Entries with missing/undecodable pixels keep their timing but draw
  nothing. Text overlays render on the GL path only — the CPU fallback
  (`glCompositing: false`) skips compositing, including texts.
- `score: {rootFreq, lookaheadBeats?, notes: [{id, trackId, startBeat,
  lengthBeats, midiNote, velocity, freqHz, ratioNum, ratioDen,
  primes: [e2,e3,e5,e7,e11,e13], linkMasterId, isRoot}], links: [{id,
  slaveNoteId, masterNoteId, slaveHarmonic, masterHarmonic, octaveTranspose}]}`
  — the symbolic note/link timeline for shader generators' **Block C**
  (§Shader generators). The exporter runs the A2 packer over it once per frame
  to fill `uNotes`/`uLinks`/`uNoteCount`/`uLinkCount`/`uRootFreq`. Omitted or
  `notes: []` ⇒ Block C stays zero-fed. GL path only.
- `modMatrix: [{source: {type, trackId?, pitchLo?, pitchHi?, primeIndex?, axis?,
  linkId?, band?, lissajousK?, triggerDecayBeats?, adsr?: {a,d,s,r,bend},
  lfo?: {shape, periodBeats, phase0, seed, hz, rateHz, retrigger}}, destination,
  depth?, curve?, smoothingBeats?, mode?, enabled?}]` — the **M6 cross-domain
  modulation matrix** (§Mod matrix). Each routing maps a musical `source`
  (Block A clock / Block B audio / Block C score — the same data the shader
  uniforms read) onto a `destination` render-graph param in the
  `clip<id>/<node>/<param>` namespace, evaluated per output frame. Omitted or
  `[]` ⇒ no modulation, output byte-identical to a no-`modMatrix` job. GL path
  only (the CPU fallback skips compositing entirely, so it skips routings too).
- `luaScript: "<lua source>"` — the **M8 per-frame Lua hook** (§Lua hook). A
  single script defining a global `frame(ctx)` that returns a table of
  `clip<id>/<node>/<param>` → value overrides — the Turing-complete general case
  of the mod matrix. Applied as the TOP layer (static → ISF default → baked →
  mod-matrix → hook). `""`/omitted ⇒ no hook. Inert (no-op) when the helper is
  built without Lua. GL path only.
- `jsScript: "<javascript source>"` — the **P2 per-frame JavaScript hook**
  (QuickJS-ng), the exact analogue of `luaScript`: `function frame(ctx)`
  returning a `{ "clip<id>/<node>/<param>": value }` object, same top-of-stack
  application and same `ctx` contract (but JS arrays are 0-indexed:
  `ctx.notes[0]`, `ctx.bands[0]`). Sandboxed (no module loader / IO) and
  deterministic (`Math.random`/`Date` removed → byte-identical re-export).
  Inert when the helper is built without QuickJS.
- `scriptLang: "lua" | "js" | ""` — picks the hook engine. Empty infers (a
  non-empty `jsScript` ⇒ js, else lua). **Only one hook is active per job.**

The GL pass runs in a hidden offscreen GLFW window (plain GL 3.3 core, no
EGL requirement); composited RGBA feeds the existing sws → encoder-format →
encoder path (yuv420p, or yuv422p10le for prores; the CPU fallbacks paint
yuv420p and convert). Fallbacks — all reported as `"glCompositing": false` in the result and
never fatal: headless/no display, GL 3.3 unavailable, helper built without
viewport support, or `interpolation == "minterpolate"` (motion interpolation
needs the raw retimed source stream, so that path keeps the legacy
single-segment CPU letterbox renderer; effects/compositing are skipped).

Result: `{outPath, encoder, glCompositing, interpolationBackend}`.
`interpolationBackend` reports the slow-motion backend actually used:
`"nearest"` (frame repeat), `"minterpolate"`, `"rife-cuda"` or `"rife-cpu"`.

## Shader generators (M3, Layer 2)

A clip can be a **procedural shader generator** instead of decoded media: its
pixels come from a user/agent-authored GLSL fragment shader the helper runs
into the clip's layer texture every frame. A generator IS a first-class layer
— the effects rack, transform2d, blend mode, mask and LUT all apply on top —
so "spiral shader clip → kaleidoscope on the result" is just an effect slot.

**Wiring.** A generator clip's segments use `sourcePath: "gen://shader"`
(the sentinel skips media decode in every probe/decode loop), and its
`clips[]` entry carries `shaderSource` (the raw GLSL). Both the live viewport
and the export jobSpec carry the same source; the helper compiles it through
the dialect front door (see below) and renders it with the Block A clock.

**The uniform contract** (declared in the auto-prepended prelude; M3 feeds the
Block A clock live, **M4 feeds Block B audio and M5 feeds Block C symbolic in the
export path** — all without any shader change beyond new prelude accessors):

- **Block A — clock**: `uResolution` (vec2), `uTime`, `uTimeDelta`, `uFrame`
  (wall-clock, clip-relative); `uBeat` (timeline-absolute beat), `uBeatPhase`,
  `uBarPhase`, `uBPM`, `uBeatsPerBar`, `uClipBeat` (beats since clip start),
  `uClipLength`, `uPlaying`. Derived from display time through the jobSpec
  tempo/meter maps — the SAME evaluator in the viewport and exporter.
- **Block B — audio features** (M4): `uRMS`, `uPeak`, `uOnset` (0..1 transient
  strength), `uOnsetAge` (seconds since the last onset), and the 64 log-spaced
  magnitudes `uAudioBands` (`sampler1D`, read via `arbitBand(b)`) plus its
  `uAudioBands2D` 64×1 alias (the ISF audioFFT / Shadertoy `iChannel0` channel).
  In the **export path** the helper analyzes the master-mix WAV (`audioPath`)
  once with the in-house A3 analyzer (`block_b_analyzer.h`, fixed 2048/512 hop,
  the same constants the live audio thread uses) and feeds each frame the latest
  feature whose analysis window ended at or before its display time. The mix WAV
  is **range-scoped** (sample 0 = `startSec`), so a frame at timeline `t` reads
  buffer-relative `t − startSec`. Live⇄export parity for a *mid-timeline* range
  additionally requires the mix WAV to carry `warmupSamples()` of real pre-roll
  before the range (A3 README Guarantee 2) — prepending that is Arbit's job. A
  silent or absent `audioPath` ⇒ every Block B feature is exactly 0 (silence is
  defined; the onset detector never fires). Live-viewport Block B (audio shm) is
  M4 Slice B; the export path is live now.
- **Block C — symbolic score** (M5): `uNotes` (a 4-texel × 128-row `RGBA32F`
  texture — per row: texel0 `midiNote, vel/127, ageBeats, remainBeats`; texel1
  `freqHz, centsFromRoot, trackId, isRoot`; texel2 `e2,e3,e5,e7`; texel3
  `e11,e13, masterRow, –`), `uLinks` (256×1 `RGBA32F`: `slaveRow, masterRow,
  num, den` per edge), `uNoteCount` (highest occupied row + 1 — the loop bound,
  covers freed-row holes), `uLinkCount`, `uRootFreq` (Hz). In the **export
  path** the helper runs the in-house A2 packer (`block_c_packer.h`, the SAME
  voice allocator the live path will use) over the jobSpec `score` once per
  frame at the frame's mapped beat (== the Block A clock), in monotonic frame
  order. The packer is **stateful**: a note keeps its
  row for its whole residency (so age-/row-keyed effects don't jump), rows free
  and reuse on exit. To keep row assignment consistent across export modes the
  packer is **warmed from timeline frame 0** (a CPU-only catch-up on the first
  frame requested), so a frame-aligned *mid-timeline* range export gets the SAME
  rows as a from-the-top export at the same timeline instant; a non-frame-aligned
  `startSec` snaps the warm-up grid to the nearest frame. Per-note CONTENT
  (midi/freq/cents/age) is always beat-correct. An empty/absent `score` ⇒
  `uNoteCount`/`uLinkCount` 0
  and the note/link samplers read black. Live-viewport Block C (live score) is a
  later slice; the export path is live now. (`uChord`/`uLastOnsetBeat` are
  reserved — declared in the generator, not in the v1 prelude.)
- Helpers: `beatPulse(width)`, `hsv2rgb()`, `arbitGuard()`; `arbitBand(b)` for
  Block B; and for Block C `arbitNoteMidi/Vel/Age/Remain/Freq/Cents/Track/IsRoot
  (row)`, `arbitNotePrimesLo/Hi(row)`, `arbitNoteMasterRow(row)`,
  `arbitNoteActive(row)`, `arbitLink(edge)`, `arbitLinkRatio(edge)` — so a shader
  reads the packed score by note/link without touching raw texel layout. The
  full contract is live in the export path through M5.

**Dialects.** Raw source may be bare `void main()` (writes `fragColor`),
Shadertoy `void mainImage(out vec4, in vec2)` (the convention LLMs are most
fluent in), or **ISF** (the JSON header's `INPUTS` become generator params —
**controllable end-to-end (M7)**: each INPUT is exposed as `clip<id>/gen/<name>`
(vector INPUTS per component, `…/<name>.r`, `…/<name>.x`) and uploaded to its
uniform from the clip's static `genParams`, baked automation, or a mod-matrix
routing, falling back to the INPUT's declared default; `image` INPUTS bind a
decoded file via `genImages` to their `sampler2D`, and `audio`/`audioFFT`
sampler INPUTS bind to the mix spectrum so a shader's own named audio INPUT is
reactive — all on texture units from TEXTURE4 up).
All are wrapped to one `#version 330 core` shader against the contract prelude.

**`shader_compile {source, lang?}`** → `{ok, dialect, acceptedLang, cacheHit,
outputGuarded, diagnostics:[{level, message}], params:[{name, type, default,
min, max}]}` — the **GL-free** front door: it detects the dialect and reports
authoring errors (missing/ambiguous entry point, `#extension`, ISF
header/INPUT problems) and the discovered generator params WITHOUT a GL
context, so an agent can validate a shader before it ever reaches the renderer.
`dialect` ∈ `glsl | shadertoy | isf | unknown`. GL-level compile errors (e.g. an
undeclared identifier that passes the textual wrap) surface when the shader is
actually run.

*Language ingestion (P3).* `lang` ∈ `glsl | slang | spirv` (default `glsl`).
`spirv` accepts a **base64-encoded SPIR-V module** and lowers it to GLSL via
SPIRV-Cross — the **universal adapter**: Slang, HLSL and GLSL all compile to
SPIR-V (`slangc`/`dxc`/`glslang`), so any of them ingests through this one path.
Lowered GLSL then runs through the same dialect wrap as a hand-written shader.
`slang` source ingestion is gated (`ARBIT_HAVE_SLANG`); when the in-process
compiler is not built the call returns guidance to compile offline with
`slangc -target spirv` and import as `spirv`. `acceptedLang` echoes the engine
used; lowered blobs are cached on disk (`cacheHit` true on repeat). The same
`lang`/`shaderLang` field is honored on clip/segment `shaderSource`. Author
Slang against `shaders/arbit_contract.slang` (the shared clock/audio/score
uniform contract module).

**Compile-on-use + last-good.** The renderer compiles a clip's program when its
source is set; a failed GL compile keeps the clip's **last good program** (a
broken live edit never blanks the composite). In the export path a clip whose
source fails to compile aborts the export with `"clip <id> shader: <log>"`
(loud, like a bad LUT) rather than silently rendering nothing. Generators run on
the GL path only — the CPU fallback skips them.

## Mod matrix (M6, cross-domain modulation)

*The music moves the picture.* The mod matrix routes the SAME musical signals the
shader uniforms expose — Block A clock, Block B audio, Block C score — onto
ordinary **render-graph params** (transform/opacity/effect/mask), so a note's
velocity can drive a clip's opacity, a beat phase can sweep a crop, an audio band
can rotate a layer. It is the `modMatrix` jobSpec field (§Export); the engine is
`mod_defs.h` (`arbitmod::evaluateSource` / `evaluateRouting`), vendored for M5 and
reused here with **no new per-frame analysis** — the export loop already builds the
clock, audio features, and packed score every frame.

**A routing** is `source → destination` with `depth`, `curve`
(`Linear|Exp|Log|SCurve`), `smoothingBeats` (one-pole, in beats), `mode`
(`Add|Multiply|Replace`), `enabled`. The `destination` is any param in the
`clip<id>/<node>/<param>` namespace `paramTimeline` writes (`transform2d/*`,
`source/{opacity,visible,zOrder,blendMode}`, `mask/*`, `effect<slot>/*`). Per
frame the modulated value is `combine(mode, base, depth·curve(source))`, where
`base` is the param's resolved **static + baked-automation** value — so routings
**layer on top of** keyframed automation rather than replacing it. Multiple
routings to one param compose in declaration order (each reads the running base).

**Sources** (`source.type`) span the three tiers: symbolic
(`NotePitch/Velocity/Gate/Trigger/Count/Age`, `CentsFromRoot`, `PrimeEnergy`,
`RootTrigger`), ratio-native (`HarmRatio[Log2|Num|Den]`, `HarmLissajous`,
`HarmBeatingRate`, `HarmTenney`, `HarmLinkRatio`), clock
(`ClockBeat[Phase]`, `ClockBarPhase`), generators (`Env`, `Lfo` — BPM-synced or
Hz, optional note-retrigger), and audio (`AudioRms/Peak/Onset/Band`). `trackId`
(-1 = any) and `pitchLo/pitchHi` filter which notes a symbolic source listens to.

**Determinism & parity.** Routings are evaluated **once per output frame in
monotonic order**, and — because the one-pole smoothing carries beat memory, like
the score packer — routing state is **warmed from timeline frame 0**: a
frame-aligned mid-range export renders a given instant byte-identically to a
from-the-top export (the same contract M5 guarantees, regression-guarded by the
mod-matrix test's smoothed mid-range parity check). The clock beat is the
**absolute mapped timeline beat** (== the packer), so score-sourced routings
see the same onsets the packed `uNotes` do. Symbolic/clock/score sources warm
exactly from frame 0; **audio**-derived sources only have features inside the
export range (the mix WAV is range-scoped), so before `startSec` their audio is
zero-fed. The per-routing mutable state lives in the export frame loop, one set
per export. Discrete sources (`HarmRatioNum/Den`) bypass smoothing (integer jumps
read as intentional). GL path only.

## Lua hook (M8, agent-programmable per-frame)

### Programmable-runtime session boundary

Non-empty programmable source fields are accepted only with an authenticated,
single-use `runtimeGrant`.  This applies to `shader_compile`, `script_compile`,
`viewport_set_script`, `viewport_set_timeline`, `export`,
`render_cache_build`, and `composite_frame_probe`, including programmable
fields nested in both `segments` and `clips`.  Missing, forged, wrong-kind,
stale-session, stale-time, and replayed grants fail closed before execution.

On POSIX the plugin sends exactly 40 private bytes (32-byte secret followed by
an eight-byte big-endian generation) over inherited descriptor 3.  The helper
reads it once and closes the descriptor; it is never an argument, environment
variable, RPC field, persisted state, or log value.  Unique authenticated
nonces may arrive out of issue order, while each nonce is consumed at most
once.  Helper replacement always creates new session material and invalidates
all old grants.  Automatic blind helper restart is disabled.

Windows intentionally reports this capability as unavailable and refuses a
helper launch configured with private grant material.  There is no argv,
environment, or ordinary inherited-handle fallback.

*The general case of the mod matrix.* Where a routing maps one source through one
curve to one destination, the `luaScript` jobSpec field (§Export) is a
Turing-complete per-frame function. A single script defines a global
`frame(ctx)` that returns a table of `clip<id>/<node>/<param>` → number
overrides:

```lua
function frame(ctx)
  -- ctx.t, ctx.beat, ctx.bpm, ctx.bar, ctx.frame                  (Block A clock)
  -- ctx.rms, ctx.peak, ctx.onset, ctx.onsetAge, ctx.bands[1..64]  (Block B audio)
  -- ctx.notes[1..noteCount], ctx.noteCount, ctx.rootFreq          (Block C score)
  --   each note: .midi .freq .velocity .cents .age .trackId .ratioNum .ratioDen
  --              .isRoot .primes[1..6]  (exponents of 2,3,5,7,11,13)
  -- ctx.links[1..linkCount], ctx.linkCount                         (harmonic edge graph)
  --   each link: .slave .master .num .den .ratio
  local lvl = ctx.rms
  if ctx.noteCount > 0 then lvl = ctx.notes[1].cents / 1200.0 end   -- read the score
  return {
    ["clip1/source/opacity"] = 0.5 + 0.5 * math.sin(ctx.beat * math.pi),
    ["clip1/gen/level"]      = lvl,
  }
end
```

The returned ids use the same `clip<id>/<node>/<param>` namespace as
`paramTimeline` and the mod matrix; each is applied via the same param sink.
**Layering:** the hook is the TOP layer — static → ISF default → baked →
mod-matrix → **Lua** — so a script wins wherever it sets a value (and leaves
everything else to the layers below). `ctx` is exactly the mod-matrix's own
inputs — Block A clock + Block B audio + Block C score — except the score is
the **full sounding-note list** (`ctx.notes`, the notes active at this frame's
beat, each with its JI cents/ratio and prime-exponent lattice coords
`.primes[1..6]`) plus the **harmonic link graph** (`ctx.links`), not just one
source's filtered pick: the unique capability is a script reading the project's
actual microtonal notes and their JI lattice.

**Sandbox:** a fresh `lua_State` with only the `base`/`table`/`string`/`math`
stdlibs; `dofile`/`loadfile`/`load`/`require` and the whole `os`/`io` surface are
absent, and `math.random`/`math.randomseed` are removed so re-export is
byte-identical (the determinism contract). **State** persists for the whole
export, so a script may keep cross-frame state in globals; like the mod-matrix
and score-packer drivers the hook is **warmed from timeline frame 0**, so a
mid-range export sees the same global history as a from-the-top one (audio is
zero-fed before `startSec`, same range-scoping caveat as the mod matrix).

**Errors are contained:** a script that fails to compile or lacks a `frame`
function disables the hook (logged once to stderr) and the export renders without
it; a per-frame `frame()` error yields empty overrides for the run (logged once)
rather than aborting. `""`/omitted ⇒ no hook. GL path only; **inert (no-op) when
the helper is built without Lua** (optional `lua5.4` dep — CMake prints whether
it is enabled).

**JavaScript variant (P2):** `jsScript` + `scriptLang:"js"` runs the identical
hook in QuickJS-ng (vendored, MIT) instead of Lua — same `frame(ctx)` contract,
same top-of-stack layering, same `clip<id>/<node>/<param>` return object. The
only surface difference is JS array indexing: `ctx.notes[0]`, `ctx.bands[0]`,
`note.primes[0..5]` (Lua's are 1-indexed). Same sandbox posture by construction
— a plain `JS_NewContext` exposes only the ECMAScript builtins (no module
loader, no `os`/`io`; quickjs-libc is never linked), and a determinism prelude
removes `Math.random` and `Date` so re-export is byte-identical. Only one engine
runs per job. Inert (no-op) when built without QuickJS. A JS and the equivalent
Lua script produce identical override maps (hence identical frame hashes).

## Stabilization (two-pass vid.stab)

FFmpeg's libvidstab filters, run as two explicit helper jobs over an open
`mediaId` (both blocking — Arbit runs them on a worker thread like
`detect_beats`):

- `stabilize_detect {mediaId, trfPath, inSec?, outSec?}` → `{trfPath,
  frames}` — pass 1: sequential software decode of `[inSec, outSec)`
  (`outSec ≤ 0` = end of stream) through `vidstabdetect` (shakiness 5),
  writing per-frame motion transforms to `trfPath`.
- `stabilize_render {mediaId, trfPath, outPath, inSec?, outSec?, strength?}`
  → `{outPath, frames}` — pass 2: the SAME frame range through
  `vidstabtransform` (`input=trfPath`, `optzoom=1`), h264-encoded (crf 18,
  veryfast; mpeg4 fallback) to `outPath`. `strength` 0..1 maps linearly to
  `smoothing` 1..60. Both passes must use the same range: the `.trf` is
  indexed by frame count, not timestamps. The intermediate's timeline equals
  source time minus `inSec`.
- `version` reports `vidstab: true/false` (runtime
  `avfilter_get_by_name` check) — Arbit-side callers should treat `false`
  as "feature unavailable", not an error.

**Apply strategy — pre-rendered intermediate, not an in-path filter.**
`vidstabtransform` consumes frames strictly sequentially and addresses the
`.trf` by frame index, which is fundamentally incompatible with the
seek-driven decoders used by the viewport (`MediaContext::getFrame`) and the
exporter's per-segment decode sessions: any seek desynchronizes the filter
from the transform list. So Arbit pre-renders a stabilized intermediate per
clip (same pattern as the stretched-audio cache) and substitutes it as the
segments' `sourcePath` in `videoBuildSegments` — the one segment builder
feeding BOTH `viewport_set_timeline` and the export jobSpec, so viewport and
export parity holds by construction. The exporter itself needed no changes.

Arbit-side caching (`VideoSidecar::getVideoCacheDirectory()`):
`stab-<hash>.trf` and `stab-<hash>-r<strength%>.mp4`, where `<hash>` covers
source path + size + mtime + trim (`[videoInPointSec, source end]` — tail
edits, markers and ramps never invalidate; moving the in-point does).
Changing the strength re-renders pass 2 but reuses the cached `.trf`.

Per-clip UI state is `VideoClipParams::stabilizeEnabled/stabilizeStrength`
(ClipPropertiesPanel toggle + slider; serialized in both StateSerializer XML
and the `.arbit` `videoStabilize:` line). The toggle is **non-automatable by
design**: strength changes require an offline re-render of the cached
intermediate, so it cannot vary along the timeline. While analysis hasn't
completed (or the cache is missing), segments fall back to the unstabilized
source — the panel shows "analyzing…" / "ready" / "not analyzed".

## Proxy media (intra-frame edit proxies)

Ported from the reference editor's `proxy_media.py`: long-GOP sources (h264/
h265 with inter-frames) seek slowly because every random access decodes from
the previous keyframe. A proxy is a one-time transcode of the source into an
**all-keyframe (intra-only) H.264 mp4** — every frame independently
decodable, so scrubbing and reverse/random seeks are cheap.

**Presets** (the reference's `NATIVE_INTRA` plus a half-res variant; DNxHR
is deliberately skipped — intra H.264 already gives frame-independence and
mp4 needs no extra muxer/profile plumbing):

- `native_intra` — source resolution, libx264 `crf 18 / veryfast / g=1 /
  bf=0`, yuv420p.
- `half_intra` — half the source dimensions (even-rounded), `crf 20`,
  otherwise identical.

Proxies are **video-only**: Arbit's audio pipeline already extracts/stretches
clip audio from the original into its own WAV cache, so the proxy never
carries an audio stream. Source pts are preserved (rescaled), making the
proxy a **1:1 timeline stand-in** — segment `inSec/outSec/rate` need no
rebasing (unlike the stabilized intermediate).

**Helper methods** — same async job pattern as `export` (one proxy job at a
time, independent of exports):

- `proxy_generate {sourcePath, outPath, preset}` — DEFERRED reply
  `{outPath, preset, frames}` when the transcode finishes; errors with
  `"proxy job already running"` while one is active. Sequential software
  decode (never NVDEC — same GL-coexistence rationale as the exporter).
- `proxy_progress {}` → `{active, done, cancelled, jobId, frame,
  totalFrames}` (+ final `result`/`error` after completion, until the next
  job starts). `totalFrames` is the duration × fps estimate.
- `proxy_cancel {}` → `{ok, active}` — abort flag polled once per packet;
  the worker **removes the partial file** and the deferred reply errors
  with exactly `"cancelled"`.
- `version` reports `proxy: true`.

**Arbit-side cache** (`VideoSidecar::getVideoCacheDirectory()`):
`proxy-<hash>-native.mp4` / `proxy-<hash>-half.mp4`, where `<hash>` covers
source path + size + mtime only — **no trim/marker state** (the proxy spans
the whole source 1:1, so clip edits never invalidate it; replacing the
source file does). Generation: per-clip "Generate Proxy (Native Intra /
Half Resolution)" in the arranger Video submenu, or the `video_generate_proxy`
test command.

**Viewport-only substitution — explicit parity decision.** The per-project
"Use Proxies" setting (arranger Video submenu; serialized in the
StateSerializer XML only — the portable `.arbit` format skips it because
proxies are machine-local cache artifacts and the flag would be meaningless
on a machine without the cache) makes `videoBuildSegments` substitute the
cached proxy as segment `sourcePath` **only on the viewport timeline path**
(`videoViewportSyncTimeline`). When both presets are cached the half-res one
wins; stabilized clips never substitute — their segments already reference
the pre-rendered stabilized intermediate, and swapping in a proxy of the
raw source would silently drop the stabilization.

**EXPORT ALWAYS USES ORIGINALS**: the export jobSpec is built with
substitution off regardless of the project setting, so exported pixels are
identical with proxies on or off. This is a deliberate, documented exception
to the export == viewport parity rule: a proxy is a *lower-fidelity preview
stand-in* of the same frames (re-encoded, possibly half-res), not authored
content — rendering the final output from a degraded copy would be the
actual parity violation. Geometry, timing, effects, transforms and
automation still go through the one shared segment builder and render graph,
so everything except decode fidelity stays parity-by-construction.

## Render cache (baked-effects viewport substitution)

Edit proxies make *decode* cheap; the render cache makes *effects* cheap. A
cache entry is one clip's **per-layer effect pass — effect slots + LUT —
baked into the pixels** of an intra-only H.264 mp4 at source resolution,
rendered through the SAME offscreen-GL `runExport` path as export (single
source segment at rate 1, default transform/opacity/no mask, the clip's
effects rack + lutPath as the only graph state). The viewport substitutes it
as segment `sourcePath` so heavy effect chains play smoothly; the layer's
live effect pass is then **cleared (graph_set_effects with an empty rack +
empty LUT) so effects are never applied twice**.

**Bake scope (what invalidates vs what stays live).** Only the per-layer
effect pass is baked. Transform/crop, opacity, zOrder, blend mode, shape
masks and transitions stay LIVE graph state — moving, fading, masking or
re-stacking a clip never invalidates its cache. Uncacheable clips simply
render live, with no behavior change:

- effects that produce transparency (`chroma_key`, `luma_key`) — H.264
  carries no alpha, so baking would flatten the keyed hole (v1 limitation);
- clips whose effect params are targeted by an automation lane WITH data —
  the param animates over time, a static bake can't represent it;
- stabilized clips whose stabilized intermediate isn't rendered yet (when it
  is, the bake reads the *intermediate*, so stabilization is included).

**Helper methods** — deferred-reply job pattern like `export`/`proxy`:

- `render_cache_build {jobSpec}` — jobSpec is the export shape (§Export)
  with one segment + one clip state; `intraOnly` is forced (libx264 g=1,
  bf=0, veryfast, crf 18) and any `audioPath` ignored. DEFERRED reply
  `{outPath, encoder, frames}`. Refused with `"busy: …"` while an export or
  another build runs; a *starting export aborts an in-flight build* (the
  plugin re-queues it later) so two offscreen GL contexts never coexist.
  A CPU-fallback render (no GL) is a HARD FAILURE — the file is removed and
  the reply errors, because a fallback render would silently drop the very
  effects being baked.
- `render_cache_progress {}` / `render_cache_cancel {}` — same contract as
  the proxy job's poll/cancel (cancel removes the partial file, deferred
  reply errors `"cancelled"`).
- `version` reports `renderCache: true`.

**Arbit-side cache** (`VideoSidecar::getVideoCacheDirectory()`):
`rcache-<hash>.mp4` where `<hash>` covers everything baked: input identity
(source path + size + mtime — or the stabilized intermediate's when
stabilization is enabled), every occupied effect slot (type/enabled/params),
LUT file identity (path + size + mtime) and stabilization state. A param
edit changes the hash, so the entry self-invalidates: segments fall back to
the original source + live effects immediately, and the background builder
(message-thread timer, ~2 s debounce after the last edit, build on a worker
thread) re-bakes once the params settle.

**EXPORT ALWAYS USES ORIGINALS + LIVE GRAPH** — same parity decision as
§Proxy media: the export jobSpec is built with substitution off and carries
the clip's real effects/LUT, so exported pixels are identical with the
render cache on or off and never inherit a generation of H.264 loss.

## RIFE frame interpolation ("speed warp")

Footage slowed below 1× by beat-marker rubber-band warp or speed ramps
(both arrive as constant-rate `rate < 1` segments from
`videoBuildSegments`) repeats source frames on the output grid. The
`"rife"` / `"auto"` export interpolation modes synthesize the missing
in-betweens with RIFE v4 optical-flow interpolation instead.

Trigger rule (ported from the reference editor, timeline_engine.py): a
segment engages interpolation only when it under-delivers real frames —
`source_fps × rate < target_fps`. Segments at or above the output rate
pass through untouched, and output timestamps within 2% of a real source
frame show that frame unmodified.

Export modes:

- `"rife"` — RIFE on triggering segments inside the **GL frame loop**
  (synthesized frame replaces the layer texture *before* compositing, so
  effects/transforms/blends/transitions apply identically). Fails the
  export with `"RIFE unavailable: …"` when the ONNX backend or the model
  download is unavailable — the user chose RIFE explicitly. Per-frame
  inference errors and unreachable bracketing pairs (e.g. clip tail)
  degrade that frame to nearest, never fail the job.
- `"auto"` — same trigger rule, but degrades **silently** to nearest-frame
  when the helper is built without ONNX or the backend/model cannot be
  initialized. Intended default for programmatic callers.

Backend: ONNX Runtime, an **optional** build dependency
(`-DARBIT_WITH_ONNX=ON`, default OFF — CI and packagers without it build
exactly as before; `CMakeLists.txt` fetches the official
`onnxruntime-linux-x64-gpu_cuda13` prebuilt on Linux x64, other platforms
supply `ARBIT_ONNXRUNTIME_ROOT`). Sessions try the CUDA execution provider
first and fall back to the CPU EP when CUDA/cuDNN are missing
(`ARBIT_RIFE_DISABLE_CUDA=1` forces CPU for diagnostics).

Model (NOT committed, NOT shipped — downloaded on first use):

- `rife_v4.18.onnx` (21,532,608 bytes, "v2" graph: input `[1,7,H,W]` =
  frame0 RGB + frame1 RGB + timestep plane in 0–1 floats, output
  `[1,3,H,W]`, arbitrary H×W via internal mod-32 padding).
- License: RIFE / Practical-RIFE code and trained weights are **MIT**
  (Copyright (c) 2021 hzwer, github.com/hzwer/Practical-RIFE). The ONNX
  conversion is published by the vs-mlrt project
  (github.com/AmusementClub/vs-mlrt, release `external-models`, file
  `rife_v4.18.7z`, `rife_v2/rife_v4.18.onnx`), which credits the
  Practical-RIFE "trained model" distribution as its source. MIT permits
  redistribution; provenance recorded here.
- Hosted at
  `https://donutsdelivery.online/download-arbit/files/video-helper/models/rife_v4.18.onnx`
  (+ `.sha256`). The helper downloads it on first use via the libavformat
  https client into `~/.cache/Arbit/video-helper/models/` (Linux;
  `%LOCALAPPDATA%\Arbit\…` Windows, `~/Library/Caches/Arbit/…` macOS) and
  verifies the pinned SHA-256
  `097a83ef9024d7a340a5a987f130870f80d9e056633537e08694ca3204c0c7a8`
  before *and* after download (a corrupt cache re-fetches; a mismatched
  download is rejected). `ARBIT_RIFE_MODEL=<path>` overrides the cache for
  testing.

Measured (RTX 4070, ORT 1.26.0, June 2026; CUDA numbers taken while the
GPU was ~99% busy with the host session, so treat them as worst-case):

| resolution | CUDA EP | CPU EP |
|------------|---------|--------|
| 640×360    | 15.3 ms/frame | 156 ms/frame |
| 1920×1080  | 140 ms/frame  | 1761 ms/frame |

Viewport (explicit decision): RIFE stays **export-only** for now. The
trigger rule and degrade chain below remain the contract for the realtime
path, but the rife-cuda rung is not wired into the viewport render loop:
at 1080p the measured inference cost (~140 ms/frame on a contended GPU)
cannot hold the frame budget, and the viewport must never block its
render thread on inference. Offline export uses the same GL compositing
pipeline, so exported RIFE frames are the live viewport frames plus the
synthesized in-betweens — parity for everything except the added frames,
which is the feature.

## Interpolation (realtime viewport — future increment)

Trigger rule (ported from the original editor): interpolate only when
`effective_fps = source_fps × rate < target_fps`; pass through otherwise.
Degrade chain: RIFE-CUDA → nearest-frame. Never a CPU interpolator in the
realtime path. v1 ships the nearest-frame rung; `interpolationBackend`
reports which rung is active.

## Scopes (waveform / vectorscope / histogram)

Video scopes are measurement diagnostics computed by the helper from the
**composited viewport frame** — the exact pixels being presented (window or
dmabuf mode), after effects, masks, LUTs, transitions, blending and text
overlays. Computation is off by default and runs ONLY while at least one
scope type is enabled.

- `scope_enable {types: ["waveform" | "vectorscope" | "histogram", ...]}` —
  selects which scopes the render loop computes. An empty/missing `types`
  array disables all scope work. Returns `{ok, mask}`. The selection
  survives viewport close/reopen (it lives outside the per-open state);
  computation itself only happens while a viewport is open.
- `scope_data {}` — returns the most recent scope arrays:
  - `open`: viewport open flag; `seq`: update counter (0 = no data yet —
    poll again; the first sample lands ~200 ms after enabling).
  - `sampleWidth` × `sampleHeight` (480×135) and `samples` (64 800): the
    downscaled analysis frame all scopes are computed from.
  - `waveform`: base64 of 480×64 u8 (row-major, `waveformWidth` ×
    `waveformHeight`). Column x = horizontal image position; row 0 = 100%
    luma (Rec.709, integer weights 54/183/19 ÷ 256), bottom row = 0%.
    Saturating density intensity (`min(255, count × 40)`).
  - `histogram`: `{r, g, b}`, each 256 u16 counts of 8-bit channel values
    over the analysis frame. Each channel's bins sum to `samples` (the
    64 800-sample frame keeps even a solid colour within u16).
  - `vectorscope`: base64 of 128×128 u8 (`vectorscopeSize`). BT.709 Cb/Cr
    density: +Cb right, +Cr up (row 0 = top), neutral at the centre.
    Saturating intensity (`min(255, count × 32)`).

Mechanics: the render loop reuses the frameHash pattern — a double-buffered
async PBO readback, but of a fixed 480×135 downscale blit of the composite,
issued at most ~10 Hz. Returned data therefore lags the visible frame by one
readback interval (~100 ms). CPU accumulation is ~65k pixels per pass.

**Viewport-only (explicit decision, no export semantics):** scopes render
nothing and write no pixels — they are a read-only measurement of the live
composited preview, so the offline-export parity rule does not apply. There
is no exporter counterpart to `scope_enable`/`scope_data`, and enabling
scopes changes neither the presented frame, the `frameHash`, nor exported
output.
