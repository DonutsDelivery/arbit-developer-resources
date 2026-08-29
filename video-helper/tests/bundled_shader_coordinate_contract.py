#!/usr/bin/env python3
"""Reject stale framebuffer-orientation compensation in first-party ISF shaders."""

import json
from pathlib import Path
import re
import sys


ALIAS_PATTERN = re.compile(r"vec2\s+(\w+)\s*=\s*isf_FragNormCoord\s*;")
DIRECT_FLIP_PATTERN = re.compile(
    r"vec2\s+\w+\s*=\s*vec2\(\s*isf_FragNormCoord\.x\s*,"
    r"\s*1\.0\s*-\s*isf_FragNormCoord\.y\s*\)\s*;"
)


def stale_coordinate_flip(source: str) -> bool:
    if DIRECT_FLIP_PATTERN.search(source):
        return True
    for match in ALIAS_PATTERN.finditer(source):
        alias = re.escape(match.group(1))
        alias_flip = re.compile(
            rf"vec2\s+\w+\s*=\s*vec2\(\s*{alias}\.x\s*,"
            rf"\s*1\.0\s*-\s*{alias}\.y\s*\)\s*;"
        )
        if alias_flip.search(source, match.end()):
            return True
    return False


def main() -> int:
    shader_root = Path(sys.argv[1])
    failures = []
    checked_files = set()
    for manifest_path in sorted(shader_root.glob("*/pack.json")):
        manifest = json.loads(manifest_path.read_text())
        pack = manifest.get("pack", {})
        pack_id = pack.get("id", "")
        if pack.get("format") != "isf" or not pack_id.startswith(("arbit-", "donutstudio-")):
            continue
        for shader in manifest.get("shaders", []):
            shader_path = manifest_path.parent / shader["file"]
            if shader_path in checked_files:
                continue
            checked_files.add(shader_path)
            if stale_coordinate_flip(shader_path.read_text()):
                failures.append(shader_path.relative_to(shader_root))

    if failures:
        print("First-party ISF shaders must use the display-space coordinate contract directly.")
        print("The renderer already maps top-down framebuffer storage to y=0 at the bottom.")
        for path in failures:
            print(f"  stale vertical compensation: {path}")
        return 1

    print(f"bundled shader coordinate contract passed ({len(checked_files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
