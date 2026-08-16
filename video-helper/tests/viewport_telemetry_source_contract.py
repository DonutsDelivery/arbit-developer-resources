#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
renderer = (root / "src/renderer.cpp").read_text()
required = {
    "production telemetry owner": "ViewportTelemetryOwner<> viewportTelemetry (im.visualTelemetry, metalOnly)",
    "shared RPC/HUD telemetry": "vi.graphTelemetry = impl_->visualTelemetry.snapshot()",
    "Metal compositor owner": "viewportTelemetry.renderComposite ([&]",
    "OpenGL compositor callsite": "return renderer.renderComposite (descs.data()",
    "Metal present owner": "localMetalSurface->present (presentError)",
    "renderer present owner": "renderer.presentToWindow (presentTex, fbW, fbH)",
    "GLFW swap owner": "glfwSwapBuffers (win)",
    "backend admission": "viewportTelemetry.admitBackend (admittedBackend)",
    "IOSurface SHM copied payload": "viewportTelemetry.readbackCopied (true, rowBytes, ph",
    "mapped PBO SHM copied payload": "static_cast<size_t> (slot->strideBytes), ph",
    "failed SHM copy branch": "viewportTelemetry.readbackCopied (false",
    "checked SHM payload bytes": "viewportTelemetry.checkedPayloadBytes (",
    "concrete IOSurface allocation bytes": "IOSurfaceGetAllocSize",
    "exported allocation snapshot": "viewportTelemetry.exportedBufferAllocated (allocationBytes)",
    "exported release snapshot": "viewportTelemetry.exportedBufferReleased (allocationBytes)",
    "exported teardown zero": "viewportTelemetry.exportedBufferPoolClosed()",
    "blit result checked": "blitSucceeded = im.exporter.blit (presentTex, *target)",
    "fence result checked": "const bool fenceSucceeded = ! fenceRequired || fenceFd >= 0",
    "socket handoff owner": "viewportTelemetry.zeroCopyHandoff (",
    "no-buffer owner": "viewportTelemetry.noFreeExportedBuffer()",
}
missing = [name for name, needle in required.items() if needle not in viewport]
if "bool FrameRenderer::presentToWindow" not in renderer:
    missing.append("renderer presentation success contract")
if missing:
    print("missing viewport telemetry source contracts: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("viewport telemetry source contract: PASS")
