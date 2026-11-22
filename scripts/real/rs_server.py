"""RealSense colour stream server for macOS (runs as root, serves frames to user processes).

librealsense on macOS needs root, and root cannot read ~/Documents, so this file is copied to
~/realsense_tools and started once per session with a password dialog (scripts/real/start_camera.sh).
Everything else (tracker, arm control) runs as the normal user and reads frames from this server
via eaiproject/realsense_client.py.

Protocol (TCP, 127.0.0.1 only):
    server -> client, once:  one JSON line: {"width","height","fps","K","dist","serial"}
    server -> client, per frame: header struct "<QddII" (frame_number, hw_timestamp_ms,
        host_monotonic_s at arrival, width, height) followed by width*height*3 bytes BGR.
    client -> server: b"quit\n" stops the server.
Slow clients just get the newest frame (older ones are dropped). The server exits after
--idle-exit minutes without clients.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import threading
import time

import numpy as np
import pyrealsense2 as rs

HDR = struct.Struct("<QddII")


class Latest:
    def __init__(self):
        self.cond = threading.Condition()
        self.item = None
        self.seq = 0
        self.stop = False

    def put(self, item):
        with self.cond:
            self.item, self.seq = item, self.seq + 1
            self.cond.notify_all()


def serve_client(conn: socket.socket, latest: Latest, header: bytes, clients: list):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(None)

    def reader():
        try:
            while True:
                data = conn.recv(64)
                if not data:
                    return
                if b"quit" in data:
                    latest.stop = True
                    with latest.cond:
                        latest.cond.notify_all()
                    return
        except OSError:
            return

    threading.Thread(target=reader, daemon=True).start()
    try:
        conn.sendall(header)
        seen = 0
        while not latest.stop:
            with latest.cond:
                latest.cond.wait_for(lambda: latest.seq != seen or latest.stop, timeout=1.0)
                if latest.stop or latest.item is None or latest.seq == seen:
                    continue
                seen, (meta, img) = latest.seq, latest.item
            conn.sendall(meta + img)
    except OSError:
        pass
    finally:
        clients.remove(conn)
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--idle-exit", type=float, default=30.0, help="minutes without clients before exiting")
    ap.add_argument("--exposure", type=float, default=0, help="manual colour exposure in us (0 = auto)")
    args = ap.parse_args()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rs_server.pid"), "w") as fh:
        fh.write(str(os.getpid()))

    def start():
        p, c = rs.pipeline(), rs.config()
        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)   # macOS: colour needs depth on
        c.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pr = p.start(c)
        p.wait_for_frames(5000)                  # make sure frames actually arrive
        return p, pr

    try:
        pipe, prof = start()
    except RuntimeError as e:
        # A crashed earlier session can leave the camera stuck: hardware-reset it and retry once.
        print(f"start failed ({str(e).strip().splitlines()[0]}); resetting the camera", flush=True)
        try:
            pipe.stop()  # noqa: F821  (only if start() got that far)
        except Exception:  # noqa: BLE001
            pass
        devs = rs.context().query_devices()
        if len(devs):
            devs[0].hardware_reset()
        time.sleep(6.0)
        pipe, prof = start()
    dev = prof.get_device()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    color_sensor = [s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
    if args.exposure > 0:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, args.exposure)
    header = (json.dumps({"width": intr.width, "height": intr.height, "fps": args.fps,
                          "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                          "dist": list(intr.coeffs), "serial": dev.get_info(rs.camera_info.serial_number)})
              + "\n").encode()

    latest, clients = Latest(), []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(4)
    srv.settimeout(0.5)

    def acceptor():
        while not latest.stop:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            clients.append(conn)
            threading.Thread(target=serve_client, args=(conn, latest, header, clients), daemon=True).start()

    threading.Thread(target=acceptor, daemon=True).start()
    print(f"serving {intr.width}x{intr.height}@{args.fps} on 127.0.0.1:{args.port} (pid {os.getpid()})", flush=True)
    last_client = time.monotonic()
    try:
        while not latest.stop:
            frames = pipe.wait_for_frames(5000)
            c = frames.get_color_frame()
            if not c:
                continue
            t_host = time.monotonic()
            img = np.asanyarray(c.get_data())
            latest.put((HDR.pack(c.get_frame_number(), c.get_timestamp(), t_host, img.shape[1], img.shape[0]),
                        img.tobytes()))
            if clients:
                last_client = t_host
            elif t_host - last_client > args.idle_exit * 60:
                print("idle, exiting", flush=True)
                break
    finally:
        latest.stop = True
        pipe.stop()
        srv.close()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
