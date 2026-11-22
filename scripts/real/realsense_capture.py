"""Capture RGB frames and a video from the RealSense D435i with the librealsense SDK.

macOS notes (Apple Silicon, tested with pyrealsense2-macosx==2.54.2; 2.56.5 segfaults):
  - needs root to claim the USB device;
  - root cannot read ~/Documents (TCC), so run from a venv + copy outside Documents, e.g.
        osascript -e 'do shell script "cd ~/realsense_tools && .venv/bin/python realsense_capture.py --out ~/realsense_tools/out" with administrator privileges'
  - colour only streams while depth is enabled too (done below).
Outputs (default outputs/realsense_capture/):
    frame_XX.png        stills spread over the recording
    stream.mp4          the colour stream
    timestamps.csv      frame number, hardware timestamp (ms), timestamp domain, host arrival time
    intrinsics.json     factory colour intrinsics (K, distortion) and capture stats
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--stills", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "outputs" / "realsense_capture"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("No RealSense device found")
    dev = devs[0]
    info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in
            ("name", "serial_number", "firmware_version", "usb_type_descriptor")}
    print("device:", info, flush=True)

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(info["serial_number"])
    # macOS (librealsense 2.54 macosx build): the colour sensor delivers no frames unless the depth
    # stream runs too, so always enable depth at the same rate (depth frames are just discarded).
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    print(f"starting colour stream {args.width}x{args.height} @ {args.fps} fps", flush=True)
    profile = pipe.start(cfg)
    vprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vprof.get_intrinsics()

    for _ in range(30):                     # let auto-exposure settle
        pipe.wait_for_frames(5000)

    writer = cv2.VideoWriter(str(out / "stream.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    n_total = int(args.seconds * args.fps)
    still_at = set(np.linspace(0, n_total - 1, args.stills).astype(int).tolist())
    rows, hw_ts, n = [], [], 0
    t0 = time.perf_counter()
    try:
        while n < n_total:
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            writer.write(img)
            ts = color.get_timestamp()
            hw_ts.append(ts)
            rows.append(f"{color.get_frame_number()},{ts:.3f},{color.get_frame_timestamp_domain().name},"
                        f"{time.perf_counter() - t0:.4f}")
            if n in still_at:
                cv2.imwrite(str(out / f"frame_{len(list(out.glob('frame_*.png'))):02d}.png"), img)
            n += 1
    finally:
        writer.release()
        pipe.stop()
    wall = time.perf_counter() - t0

    dts = np.diff(hw_ts)
    stats = {
        "frames": n, "wall_seconds": round(wall, 3), "fps_wall": round(n / wall, 2),
        "fps_hw": round(1000.0 / float(np.median(dts)), 2) if len(dts) else None,
        "dt_ms_max": round(float(dts.max()), 2) if len(dts) else None,
        "dropped_estimate": int(np.sum(dts > 1.5 * np.median(dts))) if len(dts) else None,
    }
    (out / "timestamps.csv").write_text("frame_number,hw_timestamp_ms,domain,host_s\n" + "\n".join(rows) + "\n")
    (out / "intrinsics.json").write_text(json.dumps({
        "device": info,
        "width": intr.width, "height": intr.height,
        "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        "dist_model": str(intr.model), "dist": list(intr.coeffs),
        "requested_fps": args.fps, "stats": stats,
    }, indent=2))
    print("stats:", stats)
    print("saved to", out)

    # Files are written as root under sudo; give them back to the invoking user.
    uid, gid = int(os.environ.get("SUDO_UID", -1)), int(os.environ.get("SUDO_GID", -1))
    if uid >= 0:
        for p in [out, *out.iterdir()]:
            os.chown(p, uid, gid)


if __name__ == "__main__":
    main()
