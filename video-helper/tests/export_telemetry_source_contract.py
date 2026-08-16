#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1])
exporter = (root / "src/exporter.cpp").read_text()
main = (root / "src/main.cpp").read_text()
required_exporter = {
    "shared admission telemetry": "exportAdmission.setTelemetryOwner(progress->visualTelemetry)",
    "single export owner": "ExportTelemetryOwner<> telemetryOwner(",
    "execution-state telemetry": "visualPlanState.setTelemetryOwner(telemetryOwner->telemetry())",
    "backend admission": "glctx.renderer.compositorBackend()",
    "compositor success boundary": "telemetryOwner->renderComposite(renderCall",
    "checked readback dimensions": "telemetryOwner->observeFrame(vEnc->width, vEnc->height",
    "actual readback payload bytes": "rgba.size()))",
    "encoder handoff success boundary": "telemetryOwner->encodedHandoff(rgba.size()",
    "cancellation check": "wantsAbort (progress)",
}
required_main = {
    "session reset": "g_export.progress.visualTelemetry.resetSession()",
    "helper status telemetry": "g_export.progress.visualTelemetry.snapshot()",
}
missing = [name for name, needle in required_exporter.items() if needle not in exporter]
missing += [name for name, needle in required_main.items() if needle not in main]
if missing:
    print("missing export telemetry source contracts: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("export telemetry source contract: PASS")
