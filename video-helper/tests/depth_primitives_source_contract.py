#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
frame_nodes = (root.parent / "plugin/Source/graph/FrameNodes.h").read_text()
executor = (root / "src/visual_plan_executor.h").read_text()
gl = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
renderer = (root / "src/renderer.h").read_text()
smoke = (root / "tests/metal_frame_compositor_smoke_tests.cpp").read_text()
for kind in ("visual.depth.blur", "visual.depth.displace", "visual.depth.relight"):
    assert kind in frame_nodes and kind in executor
for mode in ("depthMode == 2", "depthMode == 3", "depthMode == 4"):
    assert mode in metal
for mode in ("uDepthFog == 2", "uDepthFog == 3", "uDepthFog == 4"):
    assert mode in gl
assert "uploadR16" in renderer and "deleteTexture" in renderer
assert "for (int mode = 2; mode <= 4; ++mode)" in smoke
assert "depths.find (layer.depthTexture)" in metal
print("depth primitives source contract: PASS")
