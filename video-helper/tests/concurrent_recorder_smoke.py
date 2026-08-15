#!/usr/bin/env python3
import argparse
import os
import tempfile

from rpc_smoke_common import FrameRing, RpcHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper")
    args = parser.parse_args()
    rings = [FrameRing(), FrameRing()]
    helper = RpcHelper(args.helper)
    try:
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, "first.mp4"), os.path.join(directory, "second.mp4")]
            ids = []
            for ring, path in zip(rings, paths):
                result = helper.call("record_open", {
                    "width": ring.width, "height": ring.height, "fps": 30.0,
                    "shmName": ring.name, "outPath": path,
                    "codec": "h264", "encoder": "software",
                })
                ids.append(result["recordingId"])
            assert ids[0] != ids[1]
            for frame in range(6):
                for ring, recording_id in zip(rings, ids):
                    slot = frame % ring.slots
                    ring.write_bgra(slot, frame)
                    helper.call("record_push_frame", {"recordingId": recording_id, "slot": slot})
            helper.call("record_close", {"recordingId": ids[0]})
            for frame in range(6, 12):
                slot = frame % rings[1].slots
                rings[1].write_bgra(slot, frame)
                helper.call("record_push_frame", {"recordingId": ids[1], "slot": slot})
            helper.call("record_close", {"recordingId": ids[1]})
            for path in paths:
                assert os.path.getsize(path) > 256, f"empty recording: {path}"
        print("concurrent recorder smoke: PASS")
    finally:
        helper.close()
        for ring in rings:
            ring.close()


if __name__ == "__main__":
    main()
