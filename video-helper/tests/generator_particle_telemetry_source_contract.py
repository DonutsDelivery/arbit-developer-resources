#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
renderer = (root / "src/renderer.cpp").read_text()
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
telemetry = (root / "src/visual_plan_telemetry.h").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
hud = (root.parent / "plugin/Source/VideoEditor/VideoEditorWindow.h").read_text()

assert "recordExecutionObservation(videowire::VisualExecutionKind::generator" in renderer
assert "recordExecutionObservation(videowire::VisualExecutionKind::particle" in renderer
assert "renderViewUnlocked" in metal and "VisualExecutionKind::generator" in metal
assert "renderMetalViewUnlocked" in metal and "VisualExecutionKind::particle" in metal
assert "renderer.setVisualTelemetryOwner(&im.visualTelemetry)" in viewport
assert "glctx.renderer.setVisualTelemetryOwner(&telemetryOwner.telemetry())" in exporter
assert "addDuration(particles_, particleNs)" not in telemetry
assert "addDuration(generators_, generatorNs)" not in telemetry
assert "Generators: Not observed" in hud and "Particles: Not observed" in hud
print("generator/particle telemetry source contract: PASS")
