#!/usr/bin/env python3
"""Dependency-light guard for the intentionally blocked ncnn zero-copy path."""

from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
cmake = (root / "CMakeLists.txt").read_text()
rife = (root / "src/rife_net/rife.cpp").read_text()
engine = (root / "src/rife_ncnn.cpp").read_text()
blocker = (root.parent / "artifacts/planning/research-only/media-machine/rife-ncnn-async-submission-blocker.md").read_text()

assert 'set(_arbit_zerocopy_default OFF)' in cmake
assert 'if(ARBIT_NCNN_ZEROCOPY)\n    message(FATAL_ERROR' in cmake
assert 'cmd.submit_and_wait();' in rife
assert 'vkQueueWaitIdle(zq);' in rife
assert 'vkQueueWaitIdle (q);' in engine

for required in (
    'VkCompute::submit_and_wait()', 'VkSemaphore', 'VkFence',
    'VkCompute lifetime', 'slot reuse', 'Smallest upstream patch',
):
    assert required in blocker, required

print('PASS: unsafe zero-copy activation is fail-closed and the ncnn API blocker is documented')
