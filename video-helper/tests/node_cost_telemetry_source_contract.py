#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
telemetry = (root / "src/visual_plan_telemetry.h").read_text()
executor = (root / "src/visual_plan_executor.h").read_text()
renderer = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
hud = (root.parent / "plugin/Source/VideoEditor/VideoEditorWindow.h").read_text()

assert "recordNodeEvaluation" in telemetry
assert "aggregate layer time is never split" in telemetry
assert "slot->nodeEvaluations[i]" not in telemetry
assert "particleNodeId = ids[0]" in executor
assert "drawShapeNodeId = ids[(size_t) drawShapeIndex]" in executor
assert "recordNodeEvaluation(use->clipId" in renderer
assert "recordNodeEvaluation(layer.clipId" in metal
assert 'TRANS("Node %ID%: Not observed")' in hud
assert "measuredMovingNs" in hud
print("node cost telemetry source contract: PASS")