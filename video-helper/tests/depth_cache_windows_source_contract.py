#!/usr/bin/env python3
"""Guard the fail-closed Windows depth receipt admission boundary."""
from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
cache = (root / "src" / "depth_cache.h").read_text(encoding="utf-8")
runtime = (root / "src" / "depth_texture_runtime.h").read_text(encoding="utf-8")

required = (
    'GetProcAddress(::GetModuleHandleW(L"ntdll.dll"), "NtCreateFile")',
    "FILE_FLAG_OPEN_REPARSE_POINT",
    "openReparsePoint",
    "FILE_ATTRIBUTE_REPARSE_POINT",
    "FILE_ATTRIBUTE_READONLY",
    "GetFileInformationByHandle",
    "GetFileInformationByHandleEx",
    "FILE_SHARE_READ",
    "sameIdentity(directoryBefore, directoryAfterEntries)",
    "sameIdentity(directoryAfterEntries, directoryAfterSelected)",
)
missing = [token for token in required if token not in cache]
if missing:
    raise SystemExit("missing Windows depth admission contract(s): " + ", ".join(missing))
if "secure depth receipt admission is unavailable on Windows" in cache:
    raise SystemExit("Windows depth admission still contains the temporary fail-closed hold")
if "defined(__linux__) || defined(__APPLE__) || defined(_WIN32)" not in runtime:
    raise SystemExit("Windows depth texture runtime does not use secure receipt admission")
print("Windows depth receipt source contract passed")
