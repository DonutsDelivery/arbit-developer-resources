#!/usr/bin/env python3
"""Bounded native Linux production-export acceptance fixture.

Drives the real helper JSON-RPC export owner, production visual-plan executor,
FrameRenderer OpenGL compositor, FFmpeg encoder, and ffprobe.  No renderer or
readback is replaced by test code.
"""

import argparse
import json
import math
import os
import pathlib
import queue
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

WATCHDOG_SECONDS = 300
FPS = 24.0



class Failure(RuntimeError):
    pass


class NoGlDevice(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def is_no_gl(message):
    text = message.lower()
    return any(token in text for token in (
        "glfwinit failed", "glfwcreatewindow failed", "gl core loader failed",
        "native gpu visual plan compositor unavailable", "opengl unavailable",
        "helper built without viewport support"))


def particle_plan(clip_id, revision):
    node_id = 500 + clip_id
    return {
        "clipId": clip_id,
        "structuralRevision": revision,
        "identityMode": "authoredGraph",
        "valid": True,
        "nodeKinds": ["visual.particles", "video.out"],
        "nodeIds": [node_id, node_id + 1],
        "ports": [
            {"nodeId": node_id, "port": 1, "channels": 1, "direction": "out",
             "carrier": "frame", "dataType": "image", "pixelFormat": "rgba8",
             "colorSpace": "sRGB"},
            {"nodeId": node_id + 1, "port": 0, "channels": 1, "direction": "in",
             "carrier": "frame", "dataType": "image", "pixelFormat": "rgba8",
             "colorSpace": "sRGB"},
        ],
        "edges": [{"fromNodeId": node_id, "fromPort": 1,
                   "toNodeId": node_id + 1, "toPort": 0}],
        "operations": [
            {"nodeId": node_id, "kind": "visual.particles", "backendCapability": "native-gpu",
             "payloadXml": "<NodeParams seed=\"17\" count=\"512\" lifetime=\"1.8\" size=\"4\" speed=\"1\" red=\"0.2\" green=\"0.7\" blue=\"1\" alpha=\"1\"/>"},
            {"nodeId": node_id + 1, "kind": "video.out", "backendCapability": "native-gpu",
             "payloadXml": ""},
        ],
    }


def canonical_job(out_path, width, height, frames):
    # Export owns frames at PTS 0 through (frames - 1) / FPS. Keep the requested
    # out-point strictly beyond that final PTS so binary floating-point rounding
    # cannot truncate the last frame when the production owner floors duration*fps.
    duration = (frames + 0.25) / FPS
    segments = []
    clips = []
    plans = []
    for clip_id, layer in ((101, 0), (202, 1)):
        segments.append({
            "sourceKind": "particles", "sourcePath": "gen://particles", "clipId": clip_id,
            "trackLayer": layer, "inSec": 0.0, "outSec": duration,
            "rate": 1.0, "displayStartSec": 0.0,
        })
        clips.append({
            "clipId": clip_id, "zOrder": layer,
            "opacity": 0.78 if layer else 1.0,
            "translateX": 0.08 if layer else -0.04,
            "rotation": 3.0 if layer else -2.0,
            "mask": {"type": 2 if layer else 1, "cx": 0.5, "cy": 0.5,
                     "w": 0.72 if layer else 0.92, "h": 0.74,
                     "feather": 0.04, "invert": False},
            "genParams": {"nativeBuiltin": 1.0, "seed": 17.0 + layer,
                          "count": 512.0, "lifetime": 1.8, "size": 4.0,
                          "speed": 1.0, "red": 0.2, "green": 0.7,
                          "blue": 1.0, "alpha": 1.0},
        })
        plans.append(particle_plan(clip_id, 700 + layer))
    return {
        "outPath": str(out_path), "width": width, "height": height,
        "fps": FPS, "durationSec": duration, "codec": "h264",
        "encoder": "software", "interpolation": "none", "intraOnly": True,
        "authoringRevision": 91, "exportableRevision": 91,
        "canvasBackground": {"r": 0.015, "g": 0.02, "b": 0.04, "a": 1.0},
        "segments": segments, "clips": clips, "visualLayerPlans": plans,
        # Real baked keyframes: mask position and generator input both vary.
        "paramTimeline": [
            {"paramId": "clip202/mask/cx", "atSec": 0.0, "value": 0.32},
            {"paramId": "clip202/mask/cx", "atSec": duration, "value": 0.68},
            {"paramId": "clip101/gen/speed", "atSec": 0.0, "value": 0.3},
            {"paramId": "clip101/gen/speed", "atSec": duration, "value": 1.4},
        ],
    }


class Helper:
    def __init__(self, executable, file_limit=None):
        secret_read, secret_write = os.pipe()
        os.write(secret_write, os.urandom(32) + (1).to_bytes(8, "big"))
        os.close(secret_write)
        self.matte_root = tempfile.mkdtemp(prefix="arbit-export-matte-")
        self.depth_root = tempfile.mkdtemp(prefix="arbit-export-depth-")
        try:
            saved_fd3 = os.dup(3)
        except OSError:
            saved_fd3 = None
        if secret_read != 3:
            os.dup2(secret_read, 3)
            os.close(secret_read)
        def setup():
            os.setsid()
            if file_limit is not None:
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        self.proc = subprocess.Popen(
            [executable, "--matte-cache-root", self.matte_root,
             "--depth-cache-root", self.depth_root],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, preexec_fn=setup,
            pass_fds=(3,))
        if saved_fd3 is None:
            os.close(3)
        else:
            os.dup2(saved_fd3, 3)
            os.close(saved_fd3)
        self.responses = {}
        self.lines = queue.Queue()
        self.next_id = 1
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self.lines.put(line)

    def send(self, method, params=None):
        request_id = self.next_id
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method,
                   "params": params or {}}
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        return request_id

    def response(self, request_id, timeout=10.0):
        deadline = time.monotonic() + timeout
        while request_id not in self.responses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Failure(f"RPC {request_id} timed out")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise Failure(f"RPC {request_id} timed out")
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Failure(f"helper emitted non-JSON stdout: {line!r}: {exc}")
            if "id" in message:
                self.responses[message["id"]] = message
        return self.responses.pop(request_id)

    def call(self, method, params=None, timeout=10.0):
        response = self.response(self.send(method, params), timeout)
        if "error" in response:
            raise Failure(response["error"].get("message", str(response["error"])))
        return response["result"]

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait()
        stderr = self.proc.stderr.read()
        shutil.rmtree(self.matte_root, ignore_errors=True)
        shutil.rmtree(self.depth_root, ignore_errors=True)
        return stderr


def poll_export(helper, export_id, cancel_after_frame=None, timeout=180.0):
    deadline = time.monotonic() + timeout
    cancelled = False
    samples = []
    while time.monotonic() < deadline:
        status = helper.call("export_progress")
        samples.append(status)
        if export_id in helper.responses:
            return status, samples, helper.responses.pop(export_id)
        if cancel_after_frame is not None and not cancelled and status["frame"] >= cancel_after_frame:
            helper.call("export_cancel")
            cancelled = True
        if status["done"]:
            return status, samples, helper.response(export_id, 10.0)
        time.sleep(0.01)
    last = samples[-1] if samples else None
    raise Failure(f"export row exceeded bounded per-row timeout; last status={last!r}")


def assert_zero_resources(telemetry, row):
    resources = telemetry["resources"]
    require(resources["available"], f"{row}: resource telemetry unavailable")
    for key in ("retainedFramesCurrent", "intermediateImagesCurrent", "retainedBytesCurrent"):
        require(resources[key] == 0, f"{row}: {key} is {resources[key]}, expected zero")


def assert_success_telemetry(status, width, height, frames, row):
    telemetry = status["visualTelemetry"]
    backend = telemetry["backend"]
    dimensions = telemetry["dimensions"]
    transport = telemetry["transport"]
    require(backend["available"] and backend["initial"] == "opengl"
            and backend["current"] == "opengl", f"{row}: backend is not OpenGL: {backend}")
    require(backend["fallbackCount"] == 0, f"{row}: backend fallback observed")
    require(dimensions["available"] and
            (dimensions["requestedWidth"], dimensions["requestedHeight"]) == (width, height) and
            (dimensions["actualWidth"], dimensions["actualHeight"]) == (width, height),
            f"{row}: requested/actual dimensions differ: {dimensions}")
    require(dimensions["dimensionMismatchCount"] == 0 and
            dimensions["halfResolutionMismatchCount"] == 0,
            f"{row}: dimension mismatch telemetry is nonzero")
    require(telemetry["compositor"]["available"] and telemetry["compositor"]["count"] == frames,
            f"{row}: compositor count is not exact")
    require(telemetry["presentation"]["available"] and
            telemetry["presentationSemantic"] == "encoded/handoff",
            f"{row}: export handoff was not observed")
    require(transport["available"] and transport["mode"] == "readback",
            f"{row}: production readback transport was not observed")
    for key in ("readbackFrames", "framesRendered", "framesPresented", "exportHandoffFrames"):
        require(transport[key] == frames, f"{row}: {key}={transport[key]}, expected {frames}")
    drops = telemetry["drops"]
    require((not drops["available"]) or drops["framesDropped"] == 0,
            f"{row}: unexpected drops: {drops}")
    require(telemetry["recordingContentionDrops"] == 0, f"{row}: telemetry contention drops")
    resources = telemetry["resources"]
    assert_zero_resources(telemetry, row)
    frame_bytes = width * height * 4
    require(resources["retainedFramesPeak"] <= 1 and resources["intermediateImagesPeak"] <= 1
            and resources["retainedBytesPeak"] <= frame_bytes,
            f"{row}: resource peak exceeded one bounded frame: {resources}")
    require(telemetry["graphEvaluations"] >= frames * 2,
            f"{row}: two real visual-plan layers were not evaluated per frame: "
            f"graphEvaluations={telemetry['graphEvaluations']}, frames={frames}, "
            f"nodes={telemetry['executableNodeTotal']}")
    return telemetry


def ffprobe_verify(path, width, height, expected_frames):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries",
               "stream=width,height,nb_read_frames:frame=best_effort_timestamp_time",
               "-of", "json", str(path)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    require(result.returncode == 0, f"ffprobe failed: {result.stderr.strip()}")
    probe = json.loads(result.stdout)
    require(len(probe.get("streams", [])) == 1, "ffprobe did not find exactly one video stream")
    stream = probe["streams"][0]
    require((stream["width"], stream["height"]) == (width, height),
            f"decoded dimensions differ: {stream}")
    require(int(stream["nb_read_frames"]) == expected_frames,
            f"decoded frame count differs: {stream}")
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in probe.get("frames", [])]
    require(len(timestamps) == expected_frames, f"decoded timestamp count={len(timestamps)}")
    for index, timestamp in enumerate(timestamps):
        require(math.isclose(timestamp, index / FPS, abs_tol=0.00001),
                f"frame {index} timestamp {timestamp} != {index / FPS}")
    return timestamps


def unavailable(reason="Not observed"):
    return {"available": False, "value": None, "reason": reason}


def measured_duration(value):
    return {
        "available": value["available"],
        "count": value["count"],
        "totalNs": value["totalNs"],
        "movingNs": value["movingNs"],
    }


def run_success(helper_path, temp, width, height, frames, label):
    helper = Helper(helper_path)
    output = temp / f"{label}.mp4"
    started = time.monotonic()
    try:
        export_id = helper.send("export", canonical_job(output, width, height, frames))
        status, _, response = poll_export(helper, export_id)
        elapsed = time.monotonic() - started
        if "error" in response:
            message = response["error"].get("message", str(response["error"]))
            if is_no_gl(message):
                raise NoGlDevice(message)
            raise Failure(f"{label}: export failed: {message}")
        result = response["result"]
        require(result["glCompositing"] is True, f"{label}: helper did not report GL compositing")
        telemetry = assert_success_telemetry(status, width, height, frames, label)
        timestamps = ffprobe_verify(output, width, height, frames)
        require(telemetry["particles"]["available"],
                f"{label}: real particle owner timing was not observed")
        # The production executor measures the particle operators. video.out is
        # a graph sink, not an independently executed renderer operation; keep
        # its timing explicitly unavailable rather than inventing a zero-cost
        # observation merely to make every admitted node look measured.
        measured_node_ids = {500 + clip_id for clip_id in (101, 202)}
        measured_nodes = [node for node in telemetry["nodes"]
                          if node["stableNodeId"] in measured_node_ids]
        sink_nodes = [node for node in telemetry["nodes"]
                      if node["stableNodeId"] not in measured_node_ids]
        require(len(measured_nodes) == 2 and all(node["available"] for node in measured_nodes),
                f"{label}: particle per-node cost was not observed: {telemetry['nodes']}")
        require(all(not node["available"] for node in sink_nodes),
                f"{label}: graph sink reported fabricated execution timing: {sink_nodes}")
        receipt = {
            "row": label, "result": "PASS", "scene": "two-layer-particle-mask-keyframe",
            "requestedFrames": frames,
            "export": {
                "requestedWidth": width, "requestedHeight": height,
                "decodedWidth": width, "decodedHeight": height,
                "decodedFrames": len(timestamps), "decodedFirstPts": timestamps[0],
                "decodedLastPts": timestamps[-1], "elapsedSeconds": round(elapsed, 6),
                "throughputFps": round(frames / elapsed, 6),
                "reportedEncodeFps": status["fps"], "output": str(output),
            },
            "compositor": measured_duration(telemetry["compositor"]),
            "presentation": {
                "exportHandoff": measured_duration(telemetry["presentation"]),
                "semantic": telemetry["presentationSemantic"],
                "physicalDisplay": unavailable("Not observed by the headless export owner"),
            },
            "generator": measured_duration(telemetry["generators"]),
            "particle": measured_duration(telemetry["particles"]),
            "transport": telemetry["transport"],
            "zeroCopy": {
                "available": telemetry["transport"]["mode"] == "zero-copy",
                "frames": telemetry["transport"]["zeroCopyFrames"],
                "allocationAvailable": telemetry["transport"]["allocationAvailable"],
                "allocationBytes": telemetry["transport"]["zeroCopyAllocationBytes"],
                "reason": (None if telemetry["transport"]["mode"] == "zero-copy"
                           else "Not observed; production export used readback"),
            },
            "fallback": telemetry["backend"], "frameDrops": telemetry["drops"],
            "cache": {"planHits": telemetry["planCacheHits"],
                      "planMisses": telemetry["planCacheMisses"]},
            "compileLatency": {"available": telemetry["planInstalls"] > 0,
                               "lastPlanLoweringNs": telemetry["lastPlanLoweringNs"],
                               "budgetNs": telemetry["planLoweringBudgetNs"],
                               "withinBudget": telemetry["lastPlanLoweringWithinBudget"]},
            "perNodeCost": telemetry["nodes"], "perLayerCost": telemetry["layers"],
            "retention": telemetry["resources"],
        }
        print(json.dumps(receipt, sort_keys=True))
        return receipt
    finally:
        stderr = helper.close()
        if helper.proc.returncode not in (0, None):
            raise Failure(f"{label}: helper exited {helper.proc.returncode}: {stderr[-2000:]}")


def run_cancel(helper_path, temp):
    helper = Helper(helper_path)
    output = temp / "cancelled.mp4"
    try:
        export_id = helper.send("export", canonical_job(output, 1920, 1080, 24))
        status, samples, response = poll_export(helper, export_id, cancel_after_frame=1)
        require("error" in response and response["error"].get("message") == "cancelled",
                f"cancel: expected cancelled error, got {response}")
        require(status["cancelled"] is True, f"cancel: status not marked cancelled: {status}")
        require(not output.exists(), "cancel: partial output was not removed")
        telemetry = status["visualTelemetry"]
        assert_zero_resources(telemetry, "cancel")
        # Cancellation is not a dropped rendered frame. Do not fabricate a drop.
        drops = telemetry["drops"]
        require((not drops["available"]) or drops["framesDropped"] == 0,
                f"cancel: fabricated drop reason: {drops}")
        require(any(s["visualTelemetry"]["backend"]["available"] for s in samples),
                "cancel: native compositor was never admitted before cancellation")
        print(json.dumps({"row": "cancel", "result": "PASS",
                          "renderedBeforeCancel": telemetry["transport"]["framesRendered"],
                          "drops": 0, "resourcesCurrent": 0}, sort_keys=True))
    finally:
        helper.close()


def run_encoder_failure(helper_path, temp):
    # A real process file-size limit lets FFmpeg open/write its real muxer, then
    # forces an encoder/mux handoff write failure. SIGXFSZ is ignored so the
    # production error path and telemetry remain observable instead of killing
    # the helper.
    # Leave enough room for the MP4 header and initial packets, then fail while
    # a real intra-frame packet is handed to the muxer.
    helper = Helper(helper_path, file_limit=32768)
    output = temp / "forced-encoder-failure.mp4"
    try:
        export_id = helper.send("export", canonical_job(output, 1920, 1080, 12))
        status, _, response = poll_export(helper, export_id)
        require("error" in response, f"encoder-failure: unexpectedly succeeded: {response}")
        telemetry = status["visualTelemetry"]
        assert_zero_resources(telemetry, "encoder-failure")
        drops = telemetry["drops"]
        transport = telemetry["transport"]
        require(drops["available"] and drops["framesDropped"] >= 1
                and drops["reasons"]["transportFailure"] >= 1,
                f"encoder-failure: transport drop not honestly recorded: {drops}; "
                f"error={response.get('error')}; telemetry={telemetry}")
        require(transport["framesRendered"] >= transport["exportHandoffFrames"],
                f"encoder-failure: handoffs exceed rendered frames: {transport}")
        print(json.dumps({"row": "forced-encoder-failure", "result": "PASS",
                          "error": response["error"].get("message"),
                          "framesRendered": transport["framesRendered"],
                          "handoffs": transport["exportHandoffFrames"],
                          "transportDrops": drops["reasons"]["transportFailure"],
                          "resourcesCurrent": 0}, sort_keys=True))
    finally:
        helper.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    parser.add_argument("--keep-output", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path,
                        help="atomically write the complete Gate-8 JSON receipt")
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "native OpenGL fixture is Linux-only")
    require(args.helper.is_file() and os.access(args.helper, os.X_OK), "helper is not executable")
    require(shutil.which("ffprobe") is not None, "ffprobe is required")

    def watchdog(_signum, _frame):
        raise Failure(f"global watchdog exceeded {WATCHDOG_SECONDS} seconds")
    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(WATCHDOG_SECONDS)
    temp_owner = None
    if args.keep_output:
        temp = args.keep_output.resolve()
        temp.mkdir(parents=True, exist_ok=True)
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="arbit-export-runtime-")
        temp = pathlib.Path(temp_owner.name)
    try:
        rows = [run_success(args.helper, temp, 1920, 1080, 3, "1080p"),
                run_success(args.helper, temp, 3840, 2160, 2, "4k")]
        run_cancel(args.helper, temp)
        run_encoder_failure(args.helper, temp)
        final = {"schemaVersion": 1, "fixture": "gate-8-native-linux-export-performance",
                 "result": "PASS", "recordedAt": datetime.now(timezone.utc).isoformat(),
                 "watchdogSeconds": WATCHDOG_SECONDS, "rows": rows,
                 "performancePolicy": "measurements reported; no arbitrary speed threshold"}
        if args.receipt:
            destination = args.receipt.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, destination)
        print(json.dumps(final, sort_keys=True))
        return 0
    except NoGlDevice as exc:
        print(f"SKIP: NO_GL_DEVICE: {exc}", file=sys.stderr)
        return 77
    except Failure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
        if temp_owner:
            temp_owner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
