"""Camera pose for this session from the table markers alone (no arm, no touches).

Averages the table-marker corners over N frames, then ONE PnP on all visible corners against
config/rig_map.json. Markers the detector could not read (small, far, blurred) are then looked for
where that first pose predicts them, and the PnP is repeated with their corners included. The result
is written to outputs/calibration/camera_extrinsics.json, and writes outputs/calibration/camera_extrinsics.json in the format
real_tracking.py and the planner already read. Run it at the start of every session and whenever
the camera was moved or bumped, with the camera server at the TRACKING resolution: the file is
valid only for the stream it was made on.

    scripts/real/start_camera.sh                       # 640x480 @ 60, as for tracking
    .venv/bin/python scripts/real/relocalize.py

Refuses (exit 1, old file untouched) with fewer than 3 map markers in view, with no marker far
from the arm, or when the fit is bad (> 3 px; --force writes it anyway).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.rig_draw import GREEN, MAGENTA, STATUS_COLOR, draw_map, draw_table_grid, outline, put  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=30, help="frames to average the corners over")
    ap.add_argument("--force", action="store_true", help="write the extrinsics even if the fit is bad (FAIL)")
    args = ap.parse_args()
    from eaiproject.real_intercept import TABLE_XY
    from eaiproject.realsense_client import RealSenseClient

    if not rc.MAP_PATH.exists():
        print(f"{rc.MAP_PATH} not found: do the static calibration first (docs/calibration_protocol.md)")
        return 2
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    try:
        frames, grays, fn, g = [], [], None, None
        t_end = time.monotonic() + 6.0
        while len(frames) < args.frames and time.monotonic() < t_end:
            g = cam.read(timeout=2.0, newer_than=fn)
            fn = g.frame_number
            frames.append(rc.detect_table(g.img))
            grays.append(cv2.cvtColor(g.img, cv2.COLOR_BGR2GRAY))
        mean, jitter = rc.average_corners(frames)
        print(f"camera {cam.serial} {cam.width}x{cam.height}; {len(frames)} frames; map of {rig_map.time} "
              f"with {sorted(rig_map.markers)}")
        if jitter and max(jitter.values()) > 0.5:
            print(f"WARNING: corners moved up to {max(jitter.values()):.2f} px while averaging; is the camera fixed firmly?")
        try:
            res = rc.relocalize(rig_map, cam.K, cam.dist, mean)
            # second pass: predict the markers that were not read from the first pose, find their edges
            extra, _ = rc.average_corners([rc.guided_corners(gr, rig_map, cam.K, cam.dist, res.rvec, res.tvec, have=mean)
                                           for gr in grays])
            if extra:
                res = rc.relocalize(rig_map, cam.K, cam.dist, mean, guided=extra)
                mean = {**mean, **{k: extra[k] for k in res.guided}}
            ext = rc.extrinsics_dict(res, rig_map, cam.K, cam.dist, cam.width, cam.height,
                                     table_ref_xy=(sum(TABLE_XY[0]) / 2, sum(TABLE_XY[1]) / 2), serial=cam.serial)
        except rc.CalibrationError as e:
            print(f"\nREFUSED: {e}")
            return 1
        print(f"\nstatus {res.status}: reprojection RMS {res.reproj_rms_px:.2f} px, {res.table_rms_mm:.1f} mm on the table "
              f"(accept < {rc.OK_PX} px, < {rc.OK_MM} mm)")
        for k in res.used:
            print(f"  {k:9s} {res.per_marker_px[k]:5.2f} px  {res.per_marker_mm[k]:5.1f} mm  {'far' if rig_map.is_far(k) else 'near'}"
                  f"{'  (found by prediction)' if k in res.guided else ''}")
        if res.rejected:
            print(f"WARNING: {res.rejected} disagree with the map and were left out: marker moved on the table? "
                  "If so, redo the static calibration.")
        (x0, x1), (y0, y1) = TABLE_XY
        dz = (abs(rig_map.plane[1]) * (x1 - x0) + abs(rig_map.plane[2]) * (y1 - y0)) / 2 * 1000
        if dz > 1.5:
            print(f"NOTE: the table is tilted in the arm frame; the tracker assumes it level at its mean height, which is "
                  f"up to {dz:.1f} mm off at the corners of the tracked area (a similar error in ball position there).")
        c = res.camera_position
        print(f"camera position (arm frame): x {c[0]:.3f}  y {c[1]:.3f}  z {c[2]:.3f} m")

        img = g.img.copy()
        draw_table_grid(img, cam.K, cam.dist, res.rvec, res.tvec, rig_map.plane, rc.GATE_X)
        for k, cpx in mean.items():
            outline(img, cpx, GREEN, 1)
        draw_map(img, rig_map, cam.K, cam.dist, res.rvec, res.tvec, MAGENTA,
                 {k: f"{k} {v:.1f}px" for k, v in res.per_marker_px.items()})
        put(img, f"{res.status}  {res.reproj_rms_px:.2f} px  {res.table_rms_mm:.1f} mm   green = detected, magenta = map",
            (10, 22), STATUS_COLOR[res.status])
        rc.CAL_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rc.CAL_DIR / "relocalize_overlay.png"), img)

        if res.status == "FAIL" and not args.force:
            print(f"\nNOT WRITTEN: the fit is too bad to use. See {rc.CAL_DIR / 'relocalize_overlay.png'}; check that the "
                  "markers did not move and that the camera server runs the stream whose intrinsics it reports.")
            return 1
        rc.write_json_with_backup(rc.CAL_DIR / "camera_extrinsics.json", ext)
        print(f"wrote {rc.CAL_DIR / 'camera_extrinsics.json'} (valid for {cam.width}x{cam.height} only) and relocalize_overlay.png")
        return 0
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
