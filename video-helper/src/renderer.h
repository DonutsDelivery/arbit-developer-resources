// renderer.h — FrameRenderer: the GL 3.3 core render pipeline shared by the
// viewport (realtime preview) and, later, the exporter (WP6 offline path).
//
// Faithful C++/GLSL-330 port of the reference editor's GPU compositor
// (artifacts/planning/missing-features/plans/video-parity/compositor-spec.md, effects-spec.md,
// overlay-fx-spec.md §transitions). Per output frame:
//
//   for each layer (ordered by caller: trackLayer, then zOrder):
//     1. effects chain  — verbatim uber-shader color pass (bitmask gated,
//        videofx::kEffectBits), then separable gaussian blur + unsharp-mask
//        sharpen (Arbit extensions; the reference registered but never
//        implemented blur/sharpen — see effects-spec §5)
//     2. geometry pass  — crop TexCoord remap, letterbox fit against the
//        output aspect, then user transform M = T(tx,ty)·R(deg)·S(scale)
//        into a transparent-cleared per-layer FBO at output resolution
//     3. transition     — when the layer carries a leading-edge transition,
//        the A side (previous segment or transparent/black) is rendered the
//        same way and the verbatim transition shader blends A→B
//     4. composite      — verbatim blendNormal/Add/Multiply/Screen/Overlay +
//        opacity mix into the ping-pong accumulator
//   image overlays (text — WP5) composite last, ordered by zOrder.
//
// Coordinate convention: decoded frames upload in raster order (top row
// first), and all offscreen passes preserve "texture/FBO row 0 = image top".
// glReadPixels of the composite therefore returns top-row-first RGBA — the
// DecodedFrame convention the exporter consumes. Only presentToWindow flips
// vertically (window row 0 is the bottom scanline). A consequence is that
// the reference transform matrix applied verbatim reproduces the previous
// fixed-function semantics: +translateY moves the image down on screen and
// +rotation turns clockwise.
//
// Exporter integration (WP6): create a hidden GLFW window
// (glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE)), loadGlFunctions(), then
// FrameRenderer::initialize() at the job's output size. Per output frame:
// uploadRgba() each active segment's decoded frame, fill LayerDescs from the
// shipped ClipGraphParams, and renderToPixels() — the returned RGBA feeds
// the existing paintFrame()/sws RGBA→YUV path unchanged.
//
// All programs are compiled once at initialize(); per-frame work is uniforms
// and draws only (no shader recompiles).

#pragma once

#if ARBIT_HAVE_VIEWPORT

#include "gl_loader.h"
#include "effect_defs.h"
#include "node_preview_presentation.h"
#include "shader_generator.h"   // ShaderGenerator, ShaderClock, GenParam
#include "particle_engine.h"    // ParticleEngine, ParticleParams (P4)
#include "visual_plan_telemetry.h"
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
#include "gpu_backend/frame_renderer_metal.h"
#endif

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace videorender
{

// One effects-rack slot (mirrors PROTOCOL.md clip<id>/effect<slot>/*).
// params are stored in the videofx::kEffectDefs param-table order.
struct EffectSlotState
{
    int type = -1;                                  // videofx::EffectType, -1 = empty
    bool enabled = false;
    float params[videofx::kMaxEffectParams] = {};
};

// Per-clip render state driving one composited layer. Plain data so the
// viewport (live params) and the exporter (shipped params) can both fill it.
struct LayerDesc
{
    unsigned texture = 0;          // RGBA source texture (top row first)
    int texWidth = 0, texHeight = 0;

    // Stable clip identity (-1 = none). Used to key the cross-frame feedback
    // history texture so each clip's trail is independent across frames, and
    // (when shaderSource) the per-clip ShaderGenerator program.
    int clipId = -1;

    // Shader-generator source (M3): when true, this layer's pixels come from
    // the clip's compiled ShaderGenerator (set via setClipShader) rendered with
    // shaderClock — `texture` is ignored. The composite loop substitutes the
    // generated texture before the effects/transform chain, so a shader clip is
    // a full first-class layer (effects rack, transform, blend all apply).
    bool shaderSource = false;
    ShaderClock shaderClock;

    // Particle-generator source (P4): when true, this layer's pixels come from
    // the clip's compute-driven ParticleEngine (keyed by clipId) rendered with
    // shaderClock — `texture` and `shaderSource` are ignored. The v1 particle
    // params (count/spawnTrack/size/gravity/force) ride `genParams` like a
    // shader's ISF INPUTs; the spawn-source track's notes (noteFeatures, gated by
    // notesPresent) seed/colour/force the pool. Composited identically to a
    // shader layer (effects rack, transform, blend all apply).
    bool particleSource = false;
    bool particleStateReset = false;
    bool particleTriggerConnected = false;
    int particleTriggerCount = 0;
    float particleTriggerStrength = 0.0f;
    // Stable typed-plan identities for operation-local measured telemetry.
    uint64_t visualPlanStructuralRevision = 0;
    bool visualPlanTelemetryHold = false;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;

    // Typed visual.draw.shape overlay. The helper executor fills this only for
    // an admitted rectangle/ellipse -> Draw Shape -> Blend branch. Both native
    // compositors produce it directly on-GPU; no CPU composition/readback exists.
    bool drawShape = false;
    // Set only by exact plan inspection admission. The compositor retains the
    // native texture produced by this draw pass before Blend consumes it.
    bool inspectionDrawShapeOutput = false;
    bool drawShapeEllipse = false;
    float drawShapeCx = 0.5f, drawShapeCy = 0.5f;
    float drawShapeW = 1.0f, drawShapeH = 1.0f;
    float drawShapeR = 1.0f, drawShapeG = 1.0f, drawShapeB = 1.0f, drawShapeA = 1.0f;

    // Block B audio features (M4): when audioPresent, a shader-generator layer
    // uploads these to uRMS/uPeak/uOnset/uOnsetAge + the uAudioBands sampler
    // instead of the zeroed defaults. Plain copyable data carried by the same
    // `local = layer` copy that carries shaderClock; the viewport (live shm) and
    // the exporter (mix-WAV analysis) both fill it from block_b_defs.h ⇒ parity.
    // Ignored unless shaderSource is true.
    bool audioPresent = false;
    AudioFeatures audioFeatures;

    // Block C symbolic score (M5): when notesPresent, a shader-generator layer
    // uploads the packed note/link textures (uNotes/uLinks + uNoteCount/
    // uLinkCount/uRootFreq) instead of the black defaults. Same plain-copyable
    // posture as audioFeatures (carried by the `local = layer` copy); the
    // viewport (live score) and the exporter (packed timeline) both fill it from
    // the SAME packer ⇒ parity. Ignored unless shaderSource is true.
    bool notesPresent = false;
    NoteFeatures noteFeatures;

    // ISF INPUTS (M7): per-frame values for this shader layer's generator params
    // ("clip<id>/gen/<name>"), keyed by param name. Filled by the exporter
    // (static + baked + mod matrix) — empty ⇒ every INPUT at its ISF default.
    // Ignored unless shaderSource is true; rides the `local = layer` copy.
    std::map<std::string, double> genParams;

    // transform2d
    float scale = 1.0f;
    float translateX = 0.0f;       // NDC, +1 shifts half the viewport right
    float translateY = 0.0f;       // NDC, + moves down on screen
    float rotationDeg = 0.0f;      // + turns clockwise on screen
    float cropLeft = 0.0f, cropRight = 0.0f, cropTop = 0.0f, cropBottom = 0.0f;
    bool cornerPin = false;
    std::array<float, 8> corners { -1.0f, 1.0f, 1.0f, 1.0f,
                                   1.0f,-1.0f,-1.0f,-1.0f };

    // source compositing
    float opacity = 1.0f;
    int blendMode = 0;             // videofx::BlendMode wire value

    // Per-clip shape mask (Arbit extension, Jun 2026): multiplies the layer's
    // alpha in the geometry pass. Coordinates are normalized 0..1 over the
    // clip's displayed (post-crop) frame, origin top-left, +y down — the mask
    // therefore travels with the clip through translate/rotate/scale.
    int maskType = 0;              // 0 none, 1 rect, 2 ellipse
    float maskCx = 0.5f, maskCy = 0.5f;   // shape centre
    float maskW = 0.8f, maskH = 0.8f;     // full extents
    float maskFeather = 0.05f;     // edge feather width (same 0..1 units)
    bool maskInvert = false;

    // Typed project MatteAsset upload and bounded native-GPU refinement. The
    // decoder may populate matteTexture, but composition never maps pixels back.
    unsigned matteTexture = 0;
    unsigned matteTextureB = 0;
    int matteWidth = 0, matteHeight = 0;
    int matteWidthB = 0, matteHeightB = 0;
    bool matteApply = false, matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;

    // Exact R16 depth resource and native fog pass. The caller owns the texture
    // for the duration of renderComposite; FrameRenderer never reopens a path.
    unsigned depthTexture = 0;
    int depthWidth = 0, depthHeight = 0;
    bool depthFog = false;
    int depthEffect = 0; // 0 none, 1 fog, 2 blur, 3 displacement, 4 relighting
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;


    // Per-clip 3D LUT (Arbit extension, Jun 2026): GL_TEXTURE_3D owned by the
    // caller (uploadLut3D), sampled trilinearly after the effects chain.
    // 0 = no LUT.
    unsigned lutTexture = 0;
    int lutSize = 0;               // grid size N (17/33/65), for texel centring

    // effects rack (slot order = application order for blur/sharpen passes;
    // the color uber-shader applies its fixed in-shader order regardless)
    const EffectSlotState* effects = nullptr;
    int effectCount = 0;
    // Immutable graph payload lowering storage. The pointer above may target
    // this slot for the production FeedbackTrail pass.
    EffectSlotState graphFeedbackEffect;
    bool feedbackHistoryReset = false;
    bool feedbackHistoryHold = false;

    double timeSec = 0.0;          // uTime (animates film grain)

    // Adjustment layer (M8 "effect the world"): when true, this layer has NO
    // source of its own (texture == 0, no shaderSource). Instead its effects
    // rack runs over the composite of every layer beneath it, and the result is
    // mixed back over that composite at `opacity` (1 = full adjustment, <1 =
    // partial). Transform/crop/mask apply to the accumulated composite, so an
    // adjustment can also be used as an animated group transform. Handled by
    // a dedicated branch in renderComposite.
    bool isAdjustment = false;

    // Leading-edge transition: when transitionType != 0 this layer's content
    // becomes transition(A=fromLayer, B=this). fromLayer == nullptr means
    // A = transparent black (composites as the background, the closest
    // sensible reading of "fade from black" that does not blot out lower
    // track layers).
    int transitionType = 0;        // videofx::TransitionType wire value
    float transitionProgress = 0.0f; // 0..1 across the transition window
    const LayerDesc* fromLayer = nullptr; // A side (must not itself transition)
};

// Generic image overlay (WP5 text overlays feed this; nothing does yet).
// The texture composites in normal blend mode at its natural pixel size.
struct ImageLayerDesc
{
    unsigned texture = 0;          // RGBA, top row first
    int width = 0, height = 0;     // texture pixel size
    float posX = 0.0f, posY = 0.0f; // NDC center offset, same sign convention
                                    // as LayerDesc::translateX/translateY
    float opacity = 1.0f;
    int zOrder = 0;                // caller sorts; renderer draws in order given
    // Clip-owned text attached to an adjustment clip is inserted immediately
    // before that adjustment, so its accumulated-composite transform/effects
    // include the text. -1 keeps the overlay at project/top level.
    int ownerClipId = -1;
};

class FrameRenderer
{
public:
    FrameRenderer() = default;
    ~FrameRenderer() = default;
    FrameRenderer (const FrameRenderer&) = delete;
    FrameRenderer& operator= (const FrameRenderer&) = delete;

    // gl must outlive the renderer; the GL context must be current on the
    // calling thread for this and every other method.
    bool initialize (const arbitgl::GlFuncs* gl, int outWidth, int outHeight,
                     std::string& error, bool metalOnly = false);
    void shutdown();
    bool ready() const
    {
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
        return metalOnly_ ? metalRenderer_ != nullptr : gl_ != nullptr;
#else
        return gl_ != nullptr;
#endif
    }

    // Reallocates the FBO chain when the size changes (no-op otherwise).
    void setOutputSize (int outWidth, int outHeight);
    int outputWidth() const  { return outW_; }
    int outputHeight() const { return outH_; }

    // Project canvas (PROTOCOL.md §Project canvas & view transform): the
    // export-frame coordinate space. 0/0 (default) = follow the output size
    // — the exporter never sets it, so export behavior is unchanged. When
    // set, per-layer letterboxing and overlay sizing are computed against
    // the canvas aspect, and the canvas is mapped into the output through
    // the view transform (zoom 1 = canvas fits the output exactly).
    void setCanvas (int width, int height);
    // Viewport-only view transform: zoom (1 = fit) + pan in output NDC
    // (x right, y down — matching translateX/translateY). Never set by the
    // exporter, so exported pixels ignore it by construction.
    void setView (float zoom, float panX, float panY);

    // HDR post stack (visuals-quality increment 2): bloom + tonemap applied to
    // the final HDR composite each frame (inside renderComposite, before any
    // readback), so preview and export match by construction. Default-neutral
    // (intensity 0, tonemap 0, exposure 1) ⇒ the whole pass is skipped and the
    // composite is returned unchanged — byte-identical to a no-post job, which
    // is why the HDR float buffers (increment 1) cause no visible change until a
    // caller opts in here. tonemap: 0 clamp, 1 Reinhard, 2 ACES filmic.
    void setPostFx (float bloomIntensity, float bloomThreshold, float bloomRadius,
                    int tonemap, float exposure);

    // Canvas background colour (Media Machine M1 "canvas background param"): the
    // colour the composite is cleared to beneath all layers (shown wherever no
    // opaque layer covers — gaps, transparent/partial clips, letterbox). Default
    // is the editor charcoal (0.04, 0.04, 0.05, 1); the exporter/viewport set it
    // from the project's canvas background. RGBA in 0..1.
    void setBackgroundColor (float r, float g, float b, float a);

    // Display (present) size, distinct from the composite output size, so the
    // viewport can composite at the CANVAS resolution (pixel effects — blur,
    // bloom, feedback, particle size — then match the export exactly; audit #16)
    // while the framed result is presented into the panel/window at its own
    // resolution. 0 (default) ⇒ applyCanvasFrame outputs at the composite size,
    // the legacy behavior. Set to the framebuffer size in the viewport.
    void setPresentSize (int width, int height);

    // Post-composite canvas frame: dims everything outside the canvas rect
    // and draws a crisp ~1.5 px border at the canvas bounds. Maps the (canvas-
    // resolution) composite into the present target with the view transform
    // (letterbox + zoom/pan), at the present resolution when one is set, else at
    // the composite resolution. Returns the framed texture or the input
    // unchanged when no canvas is set. Call after renderComposite; viewport-only.
    unsigned applyCanvasFrame (unsigned compositeTexture);
    unsigned applyNodePreview (unsigned finalTexture, unsigned previewTexture,
                               const videopreview::State& state);

    // Creates (existingTexture == 0) or updates an RGBA texture from
    // raster-order pixels. strideBytes must be a multiple of 4.
    unsigned uploadRgba (const uint8_t* rgba, int width, int height,
                         int strideBytes, unsigned existingTexture);
    unsigned uploadR16 (const uint16_t* pixels, int width, int height,
                        unsigned existingTexture);
    void deleteTexture (unsigned texture);

    // Frame Blend (retime tier 1): alpha-mix two equally-sized bracket frames
    // (mix 0=texA, 1=texB) into outTex and return it. Binds fbo_ + sets its own
    // viewport (renderComposite re-establishes both afterward); creates/sizes
    // outTex (RGBA8 linear/clamp) when outTex==0. Called from the render thread
    // (decodeLayer) before renderComposite.
    unsigned frameBlendInto (unsigned texA, unsigned texB, unsigned outTex,
                             int width, int height, float mix);

    // Creates (existingTexture == 0) or replaces a 3D LUT texture from
    // size^3 RGB float triples in .cube order (red fastest). GL_RGB32F,
    // GL_LINEAR (hardware trilinear), CLAMP_TO_EDGE. Delete via
    // deleteTexture.
    unsigned uploadLut3D (const float* rgbTriples, int size,
                          unsigned existingTexture);

    // Composites layers (ordered back→front by the caller) and overlays into
    // the internal ping-pong chain. Returns the composited texture (owned by
    // the renderer, overwritten on the next call).
    unsigned renderComposite (const LayerDesc* layers, int numLayers,
                              const ImageLayerDesc* overlays = nullptr,
                              int numOverlays = 0);
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    bool renderCompositeToIOSurface (void* ioSurface, int width, int height,
                                     const LayerDesc* layers, int numLayers,
                                     const ImageLayerDesc* overlays = nullptr,
                                     int numOverlays = 0);
    void clearMetalDirectOutputs();
#endif

    // Backend used by a particle layer in the most recent composite frame.
    // This is deliberately separate from the compositor/presentation path.
    const std::string& particleBackend() const { return particleBackend_; }
    const std::string& compositorBackend() const { return compositorBackend_; }
    void setVisualTelemetryOwner (videowire::VisualPlanTelemetry* owner)
    {
        visualTelemetry_ = owner;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
        if (metalRenderer_ != nullptr) metalRenderer_->setVisualTelemetryOwner(owner);
#endif
    }

    // Exact intermediate-resource seam. When requested, renderComposite retains
    // the matching clip's post-transform layer frame in an owned GPU texture.
    // It is never mapped/read back and remains valid until resize or shutdown.
    void setTransformInspectionClip (int clipId)
    {
        transformInspectionClip_ = clipId;
        transformInspectionReady_ = false;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
        if (metalOnly_ && metalRenderer_ != nullptr)
            metalRenderer_->setInspection (clipId, metalInspectionRequestedHandle_,
                                           metalInspectionPresentation_);
#endif
    }
    unsigned transformInspectionTexture() const
    {
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
        if (metalOnly_ && metalRenderer_ != nullptr)
            return metalRenderer_->inspectionHandle();
#endif
        return transformInspectionReady_ ? transformInspectionTex_ : 0;
    }
    void setMetalInspectionPresentation (unsigned requestedHandle,
                                         const videopreview::State& state);
    bool metalInspectionResourceAdmitted() const;

    // Draws a composited texture 1:1 to the currently bound default
    // framebuffer, flipped upright for window presentation.
    bool presentToWindow (unsigned compositeTexture, int fbWidth, int fbHeight);

    // Offscreen path (exporter seam): composite + synchronous readback.
    // rgbaOut is resized to outWidth*outHeight*4, top row first, stride
    // outWidth*4.
    bool renderToPixels (const LayerDesc* layers, int numLayers,
                         std::vector<uint8_t>& rgbaOut, std::string& error,
                         const ImageLayerDesc* overlays = nullptr,
                         int numOverlays = 0);

    // Pipelined variant for the export frame loop: composites this frame and
    // issues an async PBO readback of it, then maps the PREVIOUS call's
    // readback into prevFrameOut (havePrevOut=false on the first call). The
    // GPU works on frame N while the caller encodes frame N-1. Call
    // drainAsyncReadback() after the last frame to collect the final one.
    bool renderToPixelsAsync (const LayerDesc* layers, int numLayers,
                              std::vector<uint8_t>& prevFrameOut, bool& havePrevOut,
                              std::string& error,
                              const ImageLayerDesc* overlays = nullptr,
                              int numOverlays = 0);
    bool drainAsyncReadback (std::vector<uint8_t>& rgbaOut);

    // Asynchronous composite hash (double-buffered PBO readback of the first
    // ~1 MB of the composited frame). Each call issues a new readback and
    // hashes the previously issued one, so the returned hash lags by one
    // call interval. Returns false until a previous readback is available.
    bool hashComposite (unsigned compositeTexture, uint64_t& hashOut);

    // shm-docked viewport seam: pipelined full-frame readback of an already-
    // composited texture. beginMappedReadback() maps the PREVIOUS call's
    // pixels (BGRA8, top row first, stride widthOut*4 — BGRA so the consumer
    // can memcpy straight into juce::Image ARGB rows) and issues a new async
    // readback of `texture` at the current output size. Returns nullptr until
    // a previous readback is available (first call, and right after a
    // resize). The caller copies the pixels out and MUST call
    // endMappedReadback() before the next begin.
    const uint8_t* beginMappedReadback (unsigned texture, int& widthOut, int& heightOut);
    void endMappedReadback();

    // Downscaled async readback of the composited frame for the video scopes
    // (PROTOCOL.md §Scopes). Blits compositeTexture into a fixed
    // kScopeW x kScopeH target and issues a double-buffered PBO read (same
    // pattern as hashComposite); fills rgbaOut (kScopeW*kScopeH*4, stride
    // kScopeW*4) with the PREVIOUSLY issued readback, so data lags by one
    // call interval. The blit pass flips vertically — irrelevant to scopes,
    // which treat pixels as samples (waveform columns are preserved).
    // Returns false until a previous readback is available.
    static constexpr int kScopeW = 480, kScopeH = 135;
    bool readbackScope (unsigned compositeTexture, std::vector<uint8_t>& rgbaOut);

    // Shader-generator sources (M3 — Layer 2). Compile/replace a clip's
    // procedural generator program from raw GLSL (any supported dialect — the
    // dialect front door wraps it against the Arbit uniform contract). Returns
    // true on success; logOut always carries the diagnostics, paramsOut the
    // discovered generator INPUTS. A failed compile keeps the clip's last good
    // program (renderComposite keeps drawing it), so a broken edit never blanks
    // the composite. The GL context must be current on the calling thread.
    bool setClipShader (int clipId, const std::string& source,
                        std::string& logOut, std::vector<GenParam>& paramsOut);
    void clearClipShader (int clipId);
    bool hasClipShader (int clipId) const;

    // Decode-once image for a shader clip's `image`-type ISF INPUT (M7 image
    // follow-up). Uploads `rgba` (straight RGBA, top row first, strideBytes per
    // row) to a GL texture owned by the renderer, keyed by (clipId, inputName);
    // re-uploads into the existing texture when the same key is set again. While
    // the clip's shader renders, the texture is bound to its `sampler2D <name>`
    // uniform (renderClipShaderToTexture). Freed by clearClipShader (a clip's
    // images live and die with its program) and on shutdown. The GL context must
    // be current.
    void setClipImage (int clipId, const std::string& name,
                       const uint8_t* rgba, int width, int height, int strideBytes);

private:
    struct ProgramSet;

    unsigned createOutputTexture() const;
    // HDR bloom + tonemap over the final composite (increment 2). Returns the
    // post-processed texture (a free accum target) when enabled, or
    // compositeTexture unchanged when neutral. Always leaves FBO 0 bound.
    unsigned applyPostFx (unsigned compositeTexture);
    void attachTarget (unsigned texture) const;
    void drawQuad() const;
    void bindTexture (int unit, unsigned texture, int samplerLocation) const;
    void setPassthrough (int transformLoc, int cropLoc) const;
    unsigned runEffectChain (const LayerDesc& layer);
    void drawLayerGeometry (unsigned sourceTexture, const LayerDesc& layer,
                            unsigned targetTexture, float opacity);
    unsigned buildLayerFrame (const LayerDesc& layer, float& blendOpacityOut);
    void blendOnto (unsigned frontTexture, unsigned backTexture,
                    unsigned targetTexture, float opacity, int blendMode);
    void drawImageOverlay (const ImageLayerDesc& overlay, unsigned targetTexture);
    unsigned compileProgram (const char* vertexSrc, const char* fragmentSrc,
                             const char* name, std::string& error) const;
    void releaseTargets();

    // Maps canvas NDC -> output NDC: scale (zx, zy) then translate (cx, cy).
    // Identity when no canvas is set (export path).
    void viewMap (float& zx, float& zy, float& cx, float& cy) const;

    const arbitgl::GlFuncs* gl_ = nullptr;
    videowire::VisualPlanTelemetry* visualTelemetry_ = nullptr;
    bool metalOnly_ = false;
    unsigned nextMetalHandle_ = 0x80000000u;
    int outW_ = 0, outH_ = 0;
    int canvasW_ = 0, canvasH_ = 0;     // 0 = follow output (export path)
    float viewZoom_ = 1.0f, viewPanX_ = 0.0f, viewPanY_ = 0.0f;
    // Present (display) target, distinct from the composite size (audit #16):
    // 0 ⇒ applyCanvasFrame outputs at the composite size (legacy). When set, the
    // composite runs at the canvas resolution and is framed into presentTex_ at
    // this size — owned, RGBA16F, recreated on size change.
    int presentW_ = 0, presentH_ = 0;
    unsigned presentTex_ = 0;

    // HDR post stack state (increment 2). All-neutral default ⇒ applyPostFx is
    // a no-op (returns the composite unchanged).
    float bloomIntensity_ = 0.0f;       // 0 = no bloom
    float bloomThreshold_ = 1.0f;       // extract luminance above this
    float bloomRadius_    = 8.0f;       // gaussian blur radius (px; blur shader clamps 20)
    int   tonemapMode_    = 0;          // 0 clamp, 1 Reinhard, 2 ACES
    float exposure_       = 1.0f;       // linear exposure multiply pre-tonemap

    // Canvas background clear colour (M1). Default = editor charcoal.
    float bgR_ = 0.04f, bgG_ = 0.04f, bgB_ = 0.05f, bgA_ = 1.0f;

    unsigned vao_ = 0, vbo_ = 0, ebo_ = 0, fbo_ = 0;
    unsigned accumTex_[2] = { 0, 0 };   // ping-pong accumulator
    unsigned layerTexA_ = 0;            // transition A frame
    unsigned layerTexB_ = 0;            // per-layer geometry frame
    unsigned fxTex_[2] = { 0, 0 };      // effects/blur ping-pong + transition out
    int transformInspectionClip_ = -1;
    unsigned transformInspectionTex_ = 0;
    bool transformInspectionReady_ = false;
    unsigned metalInspectionRequestedHandle_ = 0;
    videopreview::State metalInspectionPresentation_;

    // Cross-frame feedback history (FeedbackTrail effect). Per-(clipId,slot)
    // double-buffered persistent texture: each frame reads one buffer and
    // writes the other (frameParity_), so trails accumulate across frames.
    // Key = (clipId << 8) | slot. Allocated lazily and cleared (trail starts
    // from black at clip start); freed + recreated on resize/shutdown via
    // releaseTargets (PINGPONG.md §3 reset rules).
    struct FeedbackHistory { unsigned tex[2] = { 0, 0 }; int w = 0, h = 0; unsigned current = 0; bool ready = false; };
    std::map<int, FeedbackHistory> feedbackHistory_;
    unsigned frameParity_ = 0;          // advances once per renderComposite (frame)

    // Per-clip procedural shader generators (M3). Keyed by clipId; rendered into
    // the layer's source texture by renderComposite when LayerDesc::shaderSource
    // is set. Compiled out-of-band by setClipShader; freed by clearClipShader
    // and on shutdown.
    unsigned renderClipShaderToTexture (int clipId, const ShaderClock& clock,
                                        const AudioFeatures* audio,
                                        const NoteFeatures* notes,
                                        const std::map<std::string, double>* genValues = nullptr);
    std::map<int, std::unique_ptr<ShaderGenerator>> shaderGens_;

    // Per-clip compute particle engines (P4). Keyed by clipId; lazily created on
    // first render (no out-of-band compile step — the fixed compute+draw programs
    // build on first use). Rendered into the layer's source texture when
    // LayerDesc::particleSource is set; freed on shutdown. Returns 0 when compute
    // is unavailable on this context (the layer then renders nothing).
    unsigned renderClipParticlesToTexture (int clipId, uint64_t structuralRevision,
                                           int stableNodeId, bool telemetryHold,
                                           const ShaderClock& clock,
                                           const ParticleParams& params,
                                           const NoteFeatures* notes);
    std::map<int, std::unique_ptr<ParticleEngine>> particleGens_;
    std::string particleBackend_ = "none";
    std::string compositorBackend_ = "opengl";

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    // Production macOS compositor. Backend selection is whole-session: this
    // exists only for a strict Metal renderer, so a rejected frame can never
    // slide into the legacy OpenGL implementation.
    std::unique_ptr<MetalFrameRenderer> metalRenderer_;
#endif

    // Decoded image textures for shader clips' `image`-type ISF INPUTS (M7 image
    // follow-up): clipId -> (inputName -> GL texture). Populated by setClipImage,
    // passed to the clip's generator render() to bind its sampler2D INPUTs, freed
    // by clearClipShader / shutdown.
    std::map<int, std::map<std::string, unsigned>> clipImageTex_;

    // Async hash PBO pair.
    unsigned hashPbo_[2] = { 0, 0 };
    bool hashPending_[2] = { false, false };
    int hashIdx_ = 0;
    int hashBytes_ = 0;

    unsigned exportPbo_[2] = { 0, 0 };
    bool exportPending_[2] = { false, false };
    int exportIdx_ = 0;
    int exportPboBytes_ = 0;
    int hashRows_ = 0;

    // Full-frame mapped readback PBO pair (shm-docked viewport; lazily
    // created on first beginMappedReadback call). Per-PBO dims so a frame
    // issued before a resize still maps with its own size.
    unsigned framePbo_[2] = { 0, 0 };
    bool framePending_[2] = { false, false };
    int framePboW_[2] = { 0, 0 }, framePboH_[2] = { 0, 0 };
    int frameIdx_ = 0;
    bool frameMapped_ = false;

    // Scope readback: fixed-size downscale target + async PBO pair
    // (lazily created on first readbackScope call).
    unsigned scopeTex_ = 0;
    unsigned scopePbo_[2] = { 0, 0 };
    bool scopePending_[2] = { false, false };
    int scopeIdx_ = 0;

    // Programs + cached uniform locations.
    struct LayerProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uCornerPin = -1, uCorners = -1,
                       uTexture = -1, uOpacity = -1, uBlendMode = -1,
                       uMaskType = -1, uMaskRect = -1, uMaskFeather = -1, uMaskInvert = -1,
                       uMatteTexture = -1, uMatteTextureB = -1, uMatteApply = -1, uMatteTexel = -1,
                       uMatteCombineMode = -1,
                       uMatteRefine = -1, uMatteChoke = -1, uMatteInvert = -1,
                       uDepthTexture = -1, uDepthFog = -1, uFogRangeDensity = -1,
                       uFogColor = -1; } layer_;
    struct DrawShapeProg { unsigned prog = 0; int uTransform = -1, uCrop = -1,
                           uRect = -1, uColor = -1; } drawShape_;
    struct BlendProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uFrontTex = -1, uBackTex = -1, uOpacity = -1, uBlendMode = -1; } blend_;
    struct PreviewProg { unsigned prog = 0; int uFinal = -1, uPreview = -1, uLayout = -1,
                         uBackground = -1, uZoom = -1, uPan = -1, uSplit = -1; } preview_;
    struct TransProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uFromTex = -1, uToTex = -1, uProgress = -1, uType = -1; } trans_;
    struct FxProg    { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uTime = -1, uEffectMask = -1; int uValue[40] = {}; } fx_;
    struct BlurProg  { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uTexelStep = -1, uRadius = -1; } blur_;
    struct SharpProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uTexelSize = -1, uAmount = -1; } sharpen_;
    // Geometric/UV effect passes (Jun 2026): one program per effect, indexed by
    // geom_[] is indexed by position in the kGeomEffects table (renderer.cpp),
    // not by EffectType — the stateless geom effects need not be contiguous.
    // uParam[j] holds the location of the j-th param (its uniform name equals
    // the videofx::kEffectDefs param wire name).
    static constexpr int kGeomEffectCount = 8;   // stateless geom effects (see kGeomEffects table)
    struct GeomProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1,
                      uResolution = -1, uTime = -1; int uParam[videofx::kMaxEffectParams] = {}; } geom_[kGeomEffectCount];
    // Cross-frame feedback pass (FeedbackTrail): also reads uFeedbackTex (the
    // persistent previous-frame history). uParam[j] = j-th param wire name.
    struct FeedbackProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1,
                          uFeedbackTex = -1; int uParam[videofx::kMaxEffectParams] = {}; } feedback_;
    struct LutProg   { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uLut = -1, uLutSize = -1; } lut_;
    struct BlitProg  { unsigned prog = 0; int uTexture = -1; } blit_;
    struct CanvasProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uCanvasRect = -1, uTexel = -1; } canvas_;
    // HDR post stack (increment 2): bright-pass extract + combine/tonemap.
    // The blur between them reuses blur_. Both use kVertexShader, so uTransform/
    // uCrop are driven by setPassthrough like every other fullscreen pass.
    struct BloomThreshProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uThreshold = -1; } bloomThresh_;
    struct BloomCombineProg { unsigned prog = 0; int uTransform = -1, uCrop = -1, uTexture = -1, uBloomTex = -1,
                              uIntensity = -1, uExposure = -1, uTonemap = -1; } bloomCombine_;
    // Frame Blend (retime tier 1): two-sampler alpha-mix of bracket frames.
    struct FrameBlendProg { unsigned prog = 0; int uTransform = -1, uCrop = -1,
                            uTexA = -1, uTexB = -1, uMix = -1; } frameBlend_;
};

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
