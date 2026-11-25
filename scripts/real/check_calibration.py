"""Live visual check of the calibration (read-only on CAN; works without the arm too).

Uses config/rig_map.json and outputs/calibration/camera_extrinsics.json (from relocalize.py):

  markers    map outlines (magenta) projected over the image next to the detected ones (green), with
             the error per marker in px and in mm on the table
  fingertip  a magenta dot at the projected FK fingertip: it should sit on the real fingertip as you
             move the unpowered arm by hand. Rest the closed fingertips on a marker corner: the error
             to that corner is shown; SPACE records it
  ball (b)   put the ball on a known point (a marker centre, or a point added with t): tracked
             position vs expected, in mm; SPACE records it
  status     OK / WARN / FAIL (OK: < 1.5 px reprojection and < 5 mm position)

Keys: b ball mode on/off   SPACE record the current touch / ball error   t add the fingertip position
      as a known ball point   a auto-tune the ball colour   s screenshot   u undo last record   q quit
Recorded rows are appended to outputs/calibration/validation.csv (with --label as the camera name).

    .venv/bin/python scripts/real/relocalize.py
    .venv/bin/python scripts/real/check_calibration.py --label cam1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.rig_draw import (GREEN, GREY, MAGENTA, RED, STATUS_COLOR, WHITE, YELLOW, draw_table_grid, ipt,  # noqa: E402
                                 outline, put, window_closed)

NEAR_CORNER = 0.020          # m: the fingertip is "on" a corner when closer than this
NEAR_POINT = 0.050           # m: the ball is "on" a known point when closer than this
FIELDS = ["time", "camera", "kind", "point", "expected_x", "expected_y", "measured_x", "measured_y",
          "dx_mm", "dy_mm", "dz_mm", "err_mm", "reproj_px"]
WORST = {"OK": 0, "WARN": 1, "FAIL": 2}


def open_arm():
    from eaiproject.arm_reader import ArmReader
    try:
        return ArmReader()
    except Exception as e:  # noqa: BLE001
        print(f"Arm not available on CAN ({e}): fingertip check disabled")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="cam", help="camera-position name written to validation.csv")
    ap.add_argument("--no-arm", action="store_true", help="do not open the CAN bus")
    args = ap.parse_args()
    from eaiproject.real_intercept import auto_hsv, load_calibration, make_tracker, table_z
    from eaiproject.realsense_client import RealSenseClient

    if not rc.MAP_PATH.exists() or not (rc.CAL_DIR / "camera_extrinsics.json").exists():
        print("need config/rig_map.json and outputs/calibration/camera_extrinsics.json: run rig_solve.py, then relocalize.py")
        return 2
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    calib, plane, cal = load_calibration()
    K, dist, rvec, tvec = calib.K, calib.dist, calib.rvec, calib.tvec
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    if (cam.width, cam.height) != (calib.width, calib.height):
        print(f"camera {cam.width}x{cam.height} != extrinsics {calib.width}x{calib.height}: run scripts/real/relocalize.py")
        cam.close()
        return 2
    arm = None if args.no_arm else open_arm()
    trk, zt = make_tracker(calib, plane)
    proj = lambda P: rc.project(K, dist, rvec, tvec, np.asarray(P, float).reshape(-1, 3))

    pts_path = rc.CAL_DIR / "check_points.json"
    extra = json.loads(pts_path.read_text()) if pts_path.exists() else {}
    corners = [(k, j, P[j]) for k, P in rig_map.markers.items() for j in range(4)]
    rows: list[dict] = []
    win = "check calibration (b ball, SPACE record, t add point, a colour, s screenshot, u undo, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    kk = 2 if cam.width < 900 else 1
    cv2.resizeWindow(win, cam.width * kk, cam.height * kk)
    ball_mode, msg, last_fn, shots = False, "", None, 0
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        found, guided = rc.detect_with_map(f.img, rig_map, K, dist, (rvec, tvec))    # + markers found by prediction
        px, mm = rc.marker_errors(rig_map, K, dist, rvec, tvec, found)
        draw_table_grid(img, K, dist, rvec, tvec, rig_map.plane, rc.GATE_X)
        for k, c in found.items():
            outline(img, c, GREEN, 1)
        for k, P in rig_map.markers.items():
            lab = (f"{k.split(':')[1]}{'~' if k in guided else ''} {px[k]:.1f}px {mm[k]:.1f}mm" if k in px
                   else f"{k.split(':')[1]} not seen")
            outline(img, proj(P), MAGENTA if k in px else GREY, 1, lab)
        med_px = float(np.median(list(px.values()))) if px else None
        med_mm = float(np.median(list(mm.values()))) if mm else None
        st_markers = rc.status(med_px, med_mm) if len(px) >= 2 else "WARN"

        # fingertip
        s = arm.latest() if arm is not None else None
        current = None                                      # what SPACE would record
        if s is not None:
            cv2.circle(img, ipt(proj(s.p)[0]), 6, MAGENTA, 2, cv2.LINE_AA)
            cv2.circle(img, ipt(proj(s.p)[0]), 1, MAGENTA, -1)
            k, j, P = min(corners, key=lambda c: np.linalg.norm(c[2] - s.p))
            d = s.p - P
            if np.linalg.norm(d) < NEAR_CORNER and not ball_mode:
                e = float(np.linalg.norm(d) * 1000)
                cv2.line(img, ipt(proj(s.p)[0]), ipt(proj(P)[0]), YELLOW, 1)
                put(img, f"fingertip vs {k} {rc.CORNER_NAMES[j]}: {e:.1f} mm (dx {d[0] * 1000:+.1f} dy {d[1] * 1000:+.1f} "
                         f"dz {d[2] * 1000:+.1f})  SPACE", (10, 70), STATUS_COLOR[rc.status(mm=e)], 0.5)
                current = {"kind": "touch", "point": f"{k} {rc.CORNER_NAMES[j]}", "expected_x": P[0], "expected_y": P[1],
                           "measured_x": s.p[0], "measured_y": s.p[1], "dx_mm": d[0] * 1000, "dy_mm": d[1] * 1000,
                           "dz_mm": d[2] * 1000, "err_mm": e}
            else:
                put(img, f"fingertip x {s.p[0] * 100:5.1f} y {s.p[1] * 100:+5.1f} cm z {s.p[2] * 1000:5.1f} mm "
                         f"({(s.p[2] - table_z(plane, s.p[0], s.p[1])) * 1000:+.0f} mm above the table plane)", (10, 70), WHITE, 0.5)
        elif arm is not None:
            put(img, "no joint readings", (10, 70), RED)

        # ball
        if ball_mode:
            known = {**{f"{k} centre": rig_map.centre(k)[:2] for k in rig_map.markers},
                     **{n: np.array(p[:2]) for n, p in extra.items()}}
            for n, p in known.items():
                c = proj([p[0], p[1], table_z(plane, p[0], p[1]) + rc.BALL_R])[0]
                cv2.drawMarker(img, ipt(c), YELLOW, cv2.MARKER_TILTED_CROSS, 10, 1)
            det = trk.detect(f.img)
            if det is None:
                put(img, "BALL: not found (a = auto-tune the colour, ball alone on the table)", (10, 94), RED, 0.5)
            else:
                cv2.circle(img, ipt(det.uv), int(max(det.radius_px, 4)) + 3, GREEN, 2)
                exact = rc.pixel_to_ball_xy(K, dist, rvec, tvec, plane, [det.uv], rc.BALL_R)[0]
                n, p = min(known.items(), key=lambda kv: np.linalg.norm(kv[1] - det.xy))
                d = det.xy - p
                e = float(np.linalg.norm(d) * 1000)
                if e < NEAR_POINT * 1000:
                    put(img, f"BALL at {n}: {e:.1f} mm off  (dx {d[0] * 1000:+.1f} dy {d[1] * 1000:+.1f})   SPACE records",
                        (10, 94), STATUS_COLOR[rc.status(mm=e)])
                    put(img, f"tracker ({det.xy[0]:.3f}, {det.xy[1]:+.3f})  expected ({p[0]:.3f}, {p[1]:+.3f})  "
                             f"[tilted-plane projection: {np.linalg.norm(exact - p) * 1000:.1f} mm]", (10, 114), WHITE, 0.42)
                    current = {"kind": "ball", "point": n, "expected_x": p[0], "expected_y": p[1], "measured_x": det.xy[0],
                               "measured_y": det.xy[1], "dx_mm": d[0] * 1000, "dy_mm": d[1] * 1000, "dz_mm": "", "err_mm": e}
                else:
                    put(img, f"BALL at ({det.xy[0]:.3f}, {det.xy[1]:+.3f}) m: put it on a yellow cross", (10, 94), WHITE, 0.5)

        # overall status
        errs = {kind: [r["err_mm"] for r in rows if r["kind"] == kind] for kind in ("touch", "ball")}
        parts = [st_markers] + [rc.status(mm=float(np.mean(v))) for v in errs.values() if v]
        overall = max(parts, key=WORST.get)
        txt = f"{overall}   markers {len(px)}/{len(rig_map.markers)}"
        if med_px is not None:
            txt += f"  median {med_px:.2f} px  {med_mm:.1f} mm"
        put(img, txt, (10, 24), STATUS_COLOR[overall], 0.65, 2)
        rec = "   ".join(f"{kind} {np.mean(v):.1f} mm (n={len(v)}, max {max(v):.1f})" for kind, v in errs.items() if v)
        put(img, f"'{args.label}' {'BALL mode' if ball_mode else 'markers + fingertip'}   recorded: {rec or 'nothing yet'}",
            (10, 46), WHITE, 0.45)
        if overall != "OK" and st_markers != "OK":
            put(img, "markers off the map: rerun relocalize.py (camera moved?)", (10, img.shape[0] - 34), RED, 0.5)
        if msg:
            put(img, msg, (10, img.shape[0] - 12), YELLOW, 0.5)
        cv2.imshow(win, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or window_closed(win):
            break
        if key == ord("b"):
            ball_mode, msg = not ball_mode, ""
        if key == ord("a"):
            r = auto_hsv(f.img, trk.table_mask)
            if r is None:
                msg = "auto-tune needs exactly ONE orange blob on the table (ball still, hands away)"
            else:
                trk.lo, trk.hi = np.array(r[0], np.uint8), np.array(r[1], np.uint8)
                msg = f"ball colour tuned for this check: lo {r[0]} hi {r[1]}"
        if key == ord("t"):
            if s is None:
                msg = "no joint readings: cannot add a point"
            else:
                name = f"point {len(extra) + 1}"
                extra[name] = [float(s.p[0]), float(s.p[1])]
                rc.CAL_DIR.mkdir(parents=True, exist_ok=True)
                pts_path.write_text(json.dumps(extra, indent=1))
                msg = f"added '{name}' at ({s.p[0]:.3f}, {s.p[1]:+.3f}) m: mark the spot, then put the ball there (b)"
        if key == ord("u") and rows:
            r = rows.pop()
            msg = f"removed {r['kind']} at {r['point']}"
        if key == ord(" "):
            if current is None:
                msg = "nothing to record: rest the fingertip on a marker corner, or (b) put the ball on a yellow cross"
            else:
                rows.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": args.label, **current,
                             "reproj_px": "" if med_px is None else round(med_px, 3)})
                msg = f"recorded {current['kind']} at {current['point']}: {current['err_mm']:.1f} mm ({len(rows)} rows)"
                print(msg, flush=True)
        if key == ord("s"):
            shots += 1
            p = rc.CAL_DIR / f"check_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(p), img)
            msg = f"saved {p.name}"
    cv2.destroyAllWindows()
    cam.close()
    if arm is not None:
        arm.close()
    if rows:
        path = rc.CAL_DIR / "validation.csv"
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        for kind in ("touch", "ball"):
            v = [r["err_mm"] for r in rows if r["kind"] == kind]
            if v:
                print(f"{kind}: n={len(v)}  mean {np.mean(v):.1f} mm  max {max(v):.1f} mm  -> {rc.status(mm=float(np.mean(v)))}")
        print(f"appended {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
