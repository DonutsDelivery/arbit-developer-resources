#!/usr/bin/env python3
import argparse
import os
import tempfile
import time

from rpc_smoke_common import FrameRing, RpcHelper


def wait_for_frame(helper, capture_id, after=0, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = helper.call("capture_preview_latest", {"captureId": capture_id})
        if not result.get("pending") and result.get("sequence", 0) > after:
            return result
        time.sleep(0.03)
    raise TimeoutError(f"capture {capture_id} produced no new frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper")
    args = parser.parse_args()
    rings = [FrameRing(width=160, height=90), FrameRing(width=160, height=90)]
    helper = RpcHelper(args.helper)
    capture_ids = []
    try:
        listed = helper.call("capture_list_sources", {"kind": "test"})
        source = listed["sources"][0]
        for ring in rings:
            result = helper.call("capture_preview_open", {
                "sourceId": source["id"], "backend": source["backend"],
                "width": ring.width, "height": ring.height, "fps": 30.0,
                "shmName": ring.name,
            })
            capture_ids.append(result["captureId"])
        first_frames = [wait_for_frame(helper, capture_id) for capture_id in capture_ids]
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, "capture-a.mp4"), os.path.join(directory, "capture-b.mp4")]
            for capture_id, recording_id, path in zip(capture_ids, (301, 302), paths):
                helper.call("capture_record_start", {
                    "captureId": capture_id, "recordingId": recording_id,
                    "outPath": path, "codec": "h264",
                })
            time.sleep(0.5)
            helper.call("capture_record_stop", {"captureId": capture_ids[0], "recordingId": 301})
            advanced = wait_for_frame(helper, capture_ids[1], first_frames[1]["sequence"])
            assert advanced["sequence"] > first_frames[1]["sequence"]
            time.sleep(0.25)
            helper.call("capture_record_stop", {"captureId": capture_ids[1], "recordingId": 302})
            for path in paths:
                assert os.path.getsize(path) > 256, f"empty capture recording: {path}"
        print("capture source smoke: PASS")
    finally:
        for capture_id in capture_ids:
            try:
                helper.call("capture_close", {"captureId": capture_id})
            except Exception:
                pass
        helper.close()
        for ring in rings:
            ring.close()


if __name__ == "__main__":
    main()
