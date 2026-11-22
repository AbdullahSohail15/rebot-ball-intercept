"""Client for scripts/real/rs_server.py: newest RealSense colour frame + hardware timestamp.

    cam = RealSenseClient()            # connects to 127.0.0.1:8770
    frame = cam.read()                 # Frame(img BGR, frame_number, hw_ts_ms, t_host); blocks for the next one
    cam.K, cam.dist, cam.width, cam.height
`t_host` is time.monotonic() when the frame reached the server (same clock in every process on
macOS), so `time.monotonic() - frame.t_host` is the age of the frame on the host side.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
from dataclasses import dataclass

import numpy as np

HDR = struct.Struct("<QddII")


@dataclass
class Frame:
    img: np.ndarray
    frame_number: int
    hw_ts_ms: float
    t_host: float


class RealSenseClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8770, timeout: float = 3.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
        info = json.loads(self._readline())
        self.width, self.height, self.fps = info["width"], info["height"], info["fps"]
        self.K = np.array(info["K"], float)
        self.dist = np.array(info["dist"], float)
        self.serial = info["serial"]
        self._latest: Frame | None = None
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _readline(self) -> bytes:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("camera server closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def _recv_exact(self, n: int) -> bytes:
        out = bytearray(self._buf[:n])
        self._buf = self._buf[n:]
        while len(out) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(out)))
            if not chunk:
                raise ConnectionError("camera server closed the connection")
            out += chunk
        return bytes(out)

    def _loop(self):
        self.sock.settimeout(None)
        try:
            while not self._closed:
                fn, ts, th, w, h = HDR.unpack(self._recv_exact(HDR.size))
                img = np.frombuffer(self._recv_exact(w * h * 3), np.uint8).reshape(h, w, 3)
                with self._cond:
                    self._latest = Frame(img, fn, ts, th)
                    self._cond.notify_all()
        except (ConnectionError, OSError):
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def read(self, timeout: float = 1.0, newer_than: int | None = None) -> Frame:
        """Newest frame; waits for one newer than `newer_than` (frame number) if given."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._closed or (self._latest is not None and (
                newer_than is None or self._latest.frame_number != newer_than)), timeout)
            if self._closed:
                raise ConnectionError("camera server closed the connection")
            if not ok:
                raise TimeoutError("no frame from the camera server")
            return self._latest

    def stop_server(self) -> None:
        try:
            self.sock.sendall(b"quit\n")
        except OSError:
            pass

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
