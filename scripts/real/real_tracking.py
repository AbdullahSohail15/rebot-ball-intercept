"""Live ball tracking on the real rig, arm in SHADOW mode (nothing is sent to the motors).

Shows the RealSense image with: detection, track, velocity, predicted path, the intercept line and
the predicted crossing point, and where the gripper WOULD go (fingertip target). A side panel renders
the MuJoCo arm in the pose the planner would command. Each roll is scored: the predicted crossing
point at 0.5 / 0.3 / 0.15 s before the ball reaches the line vs where it actually crossed.

    scripts/real/start_camera.sh
    .venv/bin/python scripts/real/real_tracking.py

The camera pose is found at start from the table markers and config/rig_map.json (hands and arm away
from the markers for the first second). It is re-checked every few seconds in the background; a red
warning means the camera was moved: press c.

Keys: c  locate the camera again (after it was moved)
      a  auto-tune the ball colour (ball resting alone on the table)
      w  save the colour thresholds        h  show/hide the colour-mask inset
      r  reset the tracker                 q  quit (prints the roll statistics)
The HSV window has sliders for manual tuning.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject.real_intercept import (HostClock, InterceptPlanner, auto_hsv, camera_calib, load_hsv,  # noqa: E402
                                       locate_camera, make_tracker, save_hsv, table_z)
from eaiproject import rig_calib  # noqa: E402
from eaiproject.realsense_client import RealSenseClient  # noqa: E402
from eaiproject.rig_draw import put  # noqa: E402

WIN, HSV_WIN = "real tracking (shadow arm)", "HSV"
PRED_AT = (0.5, 0.3, 0.15)          # seconds before crossing at which the prediction is scored


class Roll:
    """One roll: prediction history and the actual crossing."""

    def __init__(self):
        self.preds: list[tuple[float, float]] = []      # (time to cross, predicted y)
        self.actual = None
        self.speed = None


def setup(cam, rig_map):
    """Locate the camera on the marker map; returns (calibration, tracker, pose monitor)."""
    res = locate_camera(cam, rig_map)
    print(f"camera located: {res.status}, {res.reproj_rms_px:.2f} px / {res.table_rms_mm:.1f} mm on {len(res.used)} markers "
          f"{res.used}; position x {res.camera_position[0]:.2f} y {res.camera_position[1]:.2f} z {res.camera_position[2]:.2f} m",
          flush=True)
    if res.rejected:
        print(f"WARNING: marker {res.rejected} does not fit the map (moved on the table?)", flush=True)
    calib = camera_calib(cam, res)
    return calib, make_tracker(calib, rig_map.plane), rig_calib.PoseMonitor(rig_map, calib.K, calib.dist, calib.rvec, calib.tvec)


def main() -> int:
    if not rig_calib.MAP_PATH.exists():
        print("config/rig_map.json not found: run scripts/real/calibrate.py first (docs/calibration_protocol.md)")
        return 2
    rig_map = rig_calib.RigMap.load(rig_calib.MAP_PATH)
    plane = rig_map.plane
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    try:
        calib, trk, monitor = setup(cam, rig_map)
    except rig_calib.CalibrationError as e:
        print(f"CANNOT LOCATE THE CAMERA: {e}")
        cam.close()
        return 1
    planner = InterceptPlanner(plane)
    gx = planner.gate_x
    clock = HostClock()
    mon_thread, mon_warned = None, False

    # MuJoCo shadow render
    m, d = planner.env.model, mujoco.MjData(planner.env.model)
    rend = mujoco.Renderer(m, 360, 360)
    vcam = mujoco.MjvCamera()
    vcam.lookat[:], vcam.azimuth, vcam.elevation, vcam.distance = (0.3, 0, 0.1), -150, -25, 1.1
    arm_adr = planner.env.arm_qadr

    lo, hi = load_hsv()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(HSV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HSV_WIN, 420, 160)
    for name, val, mx in (("H lo", lo[0], 179), ("H hi", hi[0], 179), ("S lo", lo[1], 255), ("V lo", lo[2], 255)):
        cv2.createTrackbar(name, HSV_WIN, int(val), mx, lambda _v: None)

    def set_sliders(l, h):
        for name, v in (("H lo", l[0]), ("H hi", h[0]), ("S lo", l[1]), ("V lo", l[2])):
            cv2.setTrackbarPos(name, HSV_WIN, int(v))

    def proj(P):
        return calib.project(np.atleast_2d(P)).round().astype(int)

    ball_z = lambda x, y: table_z(plane, x, y) + 0.022          # height of the ball centre
    show_mask, last_fn, rolls, roll = True, None, [], None
    frame_i, sim = 0, None
    fps, t_prev = 0.0, time.monotonic()
    msg, msg_t = "", 0.0
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        t_cap = clock(f.hw_ts_ms, f.t_host)
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now

        l = (cv2.getTrackbarPos("H lo", HSV_WIN), cv2.getTrackbarPos("S lo", HSV_WIN), cv2.getTrackbarPos("V lo", HSV_WIN))
        h = (cv2.getTrackbarPos("H hi", HSV_WIN), 255, 255)
        trk.lo, trk.hi = np.array(l, np.uint8), np.array(h, np.uint8)

        det = trk.process(f.img, t_cap)
        if monitor.due(now) and (mon_thread is None or not mon_thread.is_alive()):
            monitor.t_last = now
            mon_thread = threading.Thread(target=lambda im=f.img, t=now: monitor.update(monitor.detect(im), t), daemon=True)
            mon_thread.start()
        if bool(monitor.warning) != mon_warned:
            mon_warned = bool(monitor.warning)
            print(monitor.warning or "camera pose agrees with the map again", flush=True)
        if trk.x is not None and now - (trk.t_last_meas or now) > 0.6:
            trk.reset()                                   # lost for a while: start over
        p = planner.plan(trk, now)

        # ---- roll bookkeeping
        st = trk.state_at(now) if trk.ready else None
        moving = st is not None and st.vel[0] < -0.15
        if moving and roll is None:
            roll = Roll()
            roll.t0 = now
        if roll is not None and now - roll.t0 > 8.0:          # never crossed (stopped / rolled off): drop
            roll = None
        if roll is not None:
            if p.phase == "tracking" and p.crossing_y is not None:
                roll.preds.append((p.t_cross, p.crossing_y))
                roll.speed = float(np.linalg.norm(st.vel))
            hist = trk.history
            if len(hist) >= 2 and hist[-1][1][0] < gx <= hist[-2][1][0]:     # crossed between the last 2 measurements
                (t1, a), (t2, b) = hist[-2], hist[-1]
                s = (a[0] - gx) / (a[0] - b[0])
                roll.actual = float(a[1] + s * (b[1] - a[1]))
            if roll.actual is not None or (not moving and trk.x is None):
                if roll.actual is not None and roll.preds:
                    errs = {}
                    for tb in PRED_AT:
                        cand = [yy for tc, yy in roll.preds if tc <= tb]
                        if cand:
                            errs[tb] = (cand[0] - roll.actual) * 1000
                    rolls.append({"actual_y": roll.actual, "speed": roll.speed, "err_mm": errs})
                    e = "  ".join(f"{tb:.2f}s:{v:+.0f}mm" for tb, v in errs.items())
                    msg, msg_t = f"roll {len(rolls)}: crossed y={roll.actual*100:+.1f} cm at {roll.speed:.2f} m/s; pred err {e}", now
                    print(msg, flush=True)
                roll = None

        # ---- drawing
        img = f.img.copy()
        line_pts = proj([[gx, y, table_z(plane, gx, y)] for y in (-0.30, 0.30)])
        cv2.line(img, tuple(line_pts[0]), tuple(line_pts[1]), (255, 255, 255), 1)
        if det is not None:
            cv2.circle(img, tuple(np.round(det.uv).astype(int)), int(max(det.radius_px, 4)) + 3, (0, 255, 0), 2)
        if len(trk.history) > 1:
            pts = proj([[xy[0], xy[1], ball_z(*xy)] for _, xy in trk.history[-60:]])
            cv2.polylines(img, [pts], False, (0, 200, 255), 1)
        if st is not None:
            c = proj([[*st.pos, ball_z(*st.pos)]])[0]
            tipv = proj([[*(st.pos + st.vel * 0.25), ball_z(*st.pos)]])[0]
            cv2.arrowedLine(img, tuple(c), tuple(tipv), (255, 0, 255), 2)
            if p.crossing_y is not None:
                cp = proj([[gx, p.crossing_y, table_z(plane, gx, p.crossing_y) + 0.022]])[0]
                cv2.line(img, tuple(c), tuple(cp), (255, 0, 255), 1)
                cv2.drawMarker(img, tuple(cp), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        jw = proj([p.jaw])[0]
        jl = proj([p.jaw + [0, -0.05, 0]])[0]
        jr = proj([p.jaw + [0, 0.05, 0]])[0]
        cv2.line(img, tuple(jl), tuple(jr), (0, 255, 255), 3)          # open jaw, 10 cm
        cv2.circle(img, tuple(jw), 3, (0, 255, 255), -1)
        status = ("TRACKING" if det is not None else "COASTING") if trk.ready else ("ACQUIRING" if trk.x is not None else "NO BALL")
        col = {"TRACKING": (0, 255, 0), "COASTING": (0, 200, 255), "ACQUIRING": (0, 200, 255), "NO BALL": (160, 160, 160)}[status]
        put(img, f"{status}  {fps:4.1f} fps  frame age {(now - f.t_host)*1000:3.0f} ms  CV {np.mean(trk.timing_ms[-30:]):.1f} ms",
            (8, 18), col)
        if st is not None:
            put(img, f"ball x {st.pos[0]*100:5.1f} y {st.pos[1]*100:+5.1f} cm  v {np.linalg.norm(st.vel):.2f} m/s", (8, 38))
        if p.crossing_y is not None:
            put(img, f"crossing y {p.crossing_y*100:+5.1f} cm in {p.t_cross*1000:4.0f} ms  -> jaw y {p.jaw[1]*100:+5.1f} cm",
                (8, 58), (255, 0, 255))
        put(img, f"SHADOW: arm not powered | jaw target z {(p.jaw[2]-table_z(plane, gx, p.jaw[1]))*1000:.0f} mm above table"
                 f" | IK err {p.ik_err*1000:.1f} mm", (8, img.shape[0] - 10), (0, 255, 255), 0.45)
        if monitor.warning:
            put(img, monitor.warning, (8, 82), (0, 0, 255), 0.5, 2)
        elif monitor.median_px is not None:
            put(img, f"camera pose check: {monitor.median_px:.1f} px on {monitor.n_markers} markers", (8, 78), (160, 160, 160), 0.4)
        if msg and now - msg_t < 6:
            put(img, msg, (8, img.shape[0] - 30), (255, 255, 0), 0.45)
        if show_mask and trk.last_mask is not None:
            mk, (ox, oy) = trk.last_mask
            full = np.zeros(img.shape[:2], np.uint8)
            full[oy:oy + mk.shape[0], ox:ox + mk.shape[1]] = mk
            inset = cv2.resize(cv2.cvtColor(full, cv2.COLOR_GRAY2BGR), (img.shape[1] // 4, img.shape[0] // 4))
            img[-inset.shape[0] - 40:-40, -inset.shape[1]:] = inset
        # shadow arm render: every 3rd frame (it is the slow part of the loop)
        frame_i += 1
        if sim is None or frame_i % 3 == 0:
            d.qpos[:] = m.qpos0
            d.qpos[arm_adr] = p.q
            for jn in ("joint7", "joint_left", "joint_right"):
                d.qpos[m.joint(jn).qposadr[0]] = 0.05
            mujoco.mj_kinematics(m, d)
            rend.update_scene(d, vcam)
            sim = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
            put(sim, f"shadow arm: {p.phase}", (8, 20), (0, 255, 255))
            sim = cv2.resize(sim, (img.shape[0], img.shape[0]))
        cv2.imshow(WIN, np.hstack([img, sim]))

        k = cv2.waitKey(1) & 0xFF
        try:
            closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if k == ord("q") or closed:
            break
        if k == ord("a"):
            r = auto_hsv(f.img, trk.table_mask)
            if r is None:
                msg, msg_t = "auto-tune needs exactly ONE orange blob on the table (ball still, hands away)", now
            else:
                set_sliders(*r)
                msg, msg_t = f"colour tuned: lo {r[0]} hi {r[1]} (w to save)", now
        if k == ord("w"):
            save_hsv(l, h)
            msg, msg_t = f"saved HSV lo {l} hi {h}", now
        if k == ord("c"):
            try:
                hsv_now = (trk.lo, trk.hi)
                calib, trk, monitor = setup(cam, rig_map)
                trk.lo, trk.hi = hsv_now
                planner.reset(); roll = None
                msg, msg_t = "camera located again", now
            except rig_calib.CalibrationError as e:
                msg, msg_t = f"cannot locate the camera: {e}"[:110], now
                print(msg, flush=True)
        if k == ord("h"):
            show_mask = not show_mask
        if k == ord("r"):
            trk.reset(); planner.reset(); roll = None
    cv2.destroyAllWindows()
    cam.close()
    if rolls:
        print(f"\n{len(rolls)} rolls scored")
        for tb in PRED_AT:
            e = np.array([r["err_mm"][tb] for r in rolls if tb in r["err_mm"]])
            if len(e):
                print(f"  crossing prediction {tb:.2f} s before the line: mean |err| {np.abs(e).mean():.1f} mm, "
                      f"max {np.abs(e).max():.1f} mm (gripper margin +-28 mm)")
        out = ROOT / "outputs" / "hardware" / f"tracking_rolls_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rolls, indent=2))
        print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
