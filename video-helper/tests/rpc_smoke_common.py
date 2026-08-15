#!/usr/bin/env python3
# pyright: reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportArgumentType=false
import json
import struct
import subprocess
import uuid
from multiprocessing import shared_memory

MAGIC = 0x41565348
VERSION = 1
ALIGN = 64


def align(value):
    return (value + ALIGN - 1) & ~(ALIGN - 1)


class FrameRing:
    def __init__(self, slots=3, width=64, height=64):
        self.slots = slots
        self.width = width
        self.height = height
        self.slot_bytes = width * height * 4
        self.slot_stride = ALIGN + align(self.slot_bytes)
        size = ALIGN + slots * self.slot_stride
        self.name = "arbtest-" + uuid.uuid4().hex[:12]
        self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
        self.shm.buf[:] = b"\0" * size
        struct.pack_into("<8I", self.shm.buf, 0, MAGIC, VERSION, slots,
                         self.slot_bytes, self.slot_stride, 0, 0, 0)

    def write_bgra(self, slot, frame_index):
        base = ALIGN + slot * self.slot_stride
        generation = frame_index * 2 + 1
        struct.pack_into("<I", self.shm.buf, base, generation)
        struct.pack_into("<III", self.shm.buf, base + 4,
                         self.width, self.height, self.width * 4)
        struct.pack_into("<dI3I", self.shm.buf, base + 16,
                         frame_index / 30.0, 0, 0, 0, 0)
        payload = base + ALIGN
        colour = bytes(((frame_index * 17) & 255, 80, 180, 255))
        self.shm.buf[payload:payload + self.slot_bytes] = colour * (self.width * self.height)
        struct.pack_into("<I", self.shm.buf, base, generation + 1)

    def close(self):
        self.shm.close()
        self.shm.unlink()


class RpcHelper:
    def __init__(self, executable):
        self.process = subprocess.Popen([executable], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.next_id = 1

    def call(self, method, params=None):
        request_id = self.next_id
        self.next_id += 1
        request = {"jsonrpc": "2.0", "id": request_id,
                   "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                error = self.process.stderr.read()
                raise RuntimeError(f"helper exited during {method}: {error}")
            reply = json.loads(line)
            if reply.get("id") != request_id:
                continue
            if "error" in reply:
                message = reply["error"]
                if isinstance(message, dict):
                    message = message.get("message", message)
                raise RuntimeError(f"{method}: {message}")
            return reply.get("result", {})

    def close(self):
        if self.process.poll() is None:
            try:
                self.call("shutdown")
            except Exception:
                self.process.terminate()
        self.process.wait(timeout=10)
