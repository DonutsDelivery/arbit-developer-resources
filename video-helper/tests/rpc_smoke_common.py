#!/usr/bin/env python3
# pyright: reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportArgumentType=false
import json
import os
import struct
import subprocess
import tempfile
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
        # Python 3.14 on macOS exposes the page-rounded mapping through buf,
        # which can be larger than the logical size requested above. Clear only
        # the protocol-owned extent so both sides of the assignment match.
        self.shm.buf[:size] = b"\0" * size
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
        self.matte_cache = tempfile.TemporaryDirectory(prefix="arbit-matte-cache-")
        self.depth_cache = tempfile.TemporaryDirectory(prefix="arbit-depth-cache-")
        secret_read, secret_write = os.pipe()
        session_packet = os.urandom(32) + struct.pack(">Q", 1)
        try:
            saved_fd3 = os.dup(3)
        except OSError:
            saved_fd3 = None
        if secret_read != 3:
            os.dup2(secret_read, 3)
            os.close(secret_read)
        self.process = subprocess.Popen([executable, "--matte-cache-root", self.matte_cache.name,
                                         "--depth-cache-root", self.depth_cache.name], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, pass_fds=(3,))
        if saved_fd3 is None:
            os.close(3)
        else:
            os.dup2(saved_fd3, 3)
            os.close(saved_fd3)
        try:
            os.write(secret_write, session_packet)
        finally:
            os.close(secret_write)
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
        self.matte_cache.cleanup()
        self.depth_cache.cleanup()
