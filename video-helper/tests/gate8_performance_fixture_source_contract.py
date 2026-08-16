#!/usr/bin/env python3
from pathlib import Path

fixture = (Path(__file__).resolve().parent / "export_runtime_fixture.py").read_text()

for resolution in ("1920, 1080", "3840, 2160"):
    assert resolution in fixture
for field in ("compositor", "physicalDisplay", "generator", "particle", "zeroCopy",
              "fallback", "frameDrops", "cache", "perNodeCost", "compileLatency",
              "retention", "throughputFps", "decodedWidth", "decodedHeight"):
    assert f'"{field}"' in fixture
assert 'unavailable("Not observed by the headless export owner")' in fixture
assert '"reason": (None if telemetry["transport"]["mode"] == "zero-copy"' in fixture
assert "os.replace(temporary, destination)" in fixture
assert '"scene": "two-layer-particle-mask-keyframe"' in fixture
print("Gate-8 performance fixture source contract: PASS")