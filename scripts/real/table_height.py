"""Measure the table height where the arm works, in the arm's OWN kinematic frame.

The fingertip heights the arm reports are biased (joint zeros / fingertip model): on the plate it
reads +9 mm, on the table +6 mm. Rather than chase the cause, measure the table where the blocking
happens and command heights relative to that: any constant or slowly varying model error cancels.

For each target (yellow circle on the camera image, along the intercept line x = 0.40 m):
    rest the CLOSED fingertips on the table at the circle, with the gripper tilted roughly like
    the blocking pose (fingers forward and ~30 deg down; the window shows the tilt), SPACE.
Targets: (0.40, -0.20), (0.40, 0), (0.40, +0.20) required; (0.30, 0) and (0.55, 0) optional (s skips).

Output: outputs/calibration/table_height.json with a plane z = a + b x + c y (robot frame, metres),
fitted to these touches plus the two table-marker touches from the camera calibration.
Read-only on CAN.

    .venv/bin/python scripts/real/table_height.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAL = ROOT / "outputs" / "calibration"
TARGETS = [(0.40, -0.20, True), (0.40, 0.0, True), (0.40, 0.20, True), (0.30, 0.0, False), (0.55, 0.0, False)]


def put(img, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_plane(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
    coef = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    return coef, P[:, 2] - A @ coef


def main() -> int:
    import mujoco
    from eaiproject.kinematics import Fingertip
    from eaiproject.realsense_client import RealSenseClient
    from eaiproject.robstride import ARM_IDS, P_MECHPOS, RobStrideBus

    cal = json.loads((CAL / "camera_extrinsics.json").read_text())
    K, dist = np.array(cal["K"]), np.array(cal["dist"])
    rvec, tvec = np.array(cal["rvec"]), np.array(cal["tvec"])
    z0 = cal["table_z"]
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    if (cam.width, cam.height) != (cal["width"], cal["height"]):
        print(f"camera is {cam.width}x{cam.height} but the calibration is {cal['width']}x{cal['height']}; "
              "restart the server at the calibration resolution")
        return 2
    rs = RobStrideBus(movable=())
    tip = Fingertip()
    m, d = tip.m, tip.d

    def read_q():
        q = [rs.read_param_f32(mm, P_MECHPOS, timeout=0.03) for mm in ARM_IDS]
        return None if any(v is None for v in q) else np.array(q)

    def proj(P):
        p, _ = cv2.projectPoints(np.asarray(P, float).reshape(-1, 3), rvec, tvec, K, dist)
        return p.reshape(-1, 2).round().astype(int)

    if read_q() is None:
        print("Arm not answering on CAN")
        return 2
    win = "table height (SPACE record, s skip optional, u undo, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cam.width * 2 if cam.width < 900 else cam.width, cam.height * 2 if cam.width < 900 else cam.height)
    todo = list(TARGETS)
    done: list[tuple[tuple, list]] = []
    msg, last_fn, saved = "", None, False
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        q = read_q()
        p = tilt = None
        if q is not None:
            p = tip(q)
            R = d.xmat[tip.body].reshape(3, 3)              # gripper_end x = finger direction
            tilt = float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))))   # + = pointing down
        cur = todo[0] if todo else None
        for (x, y, req) in TARGETS:
            st = "done" if any(t[0][:2] == (x, y) for t in done) else ("now" if cur and (x, y) == cur[:2] else "todo")
            col = {"done": (0, 255, 0), "now": (0, 220, 255), "todo": (160, 160, 160)}[st]
            c = proj([x, y, z0])[0]
            cv2.circle(img, tuple(c), 10 if st == "now" else 6, col, 2)
            if not req:
                put(img, "opt", (c[0] + 10, c[1] + 4), col, 0.4)
        if p is not None:
            c = proj([p[0], p[1], z0])[0]
            cv2.drawMarker(img, tuple(c), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if cur:
            put(img, f"touch the table at the YELLOW circle ({cur[0]:.2f}, {cur[1]:+.2f}) m"
                     + ("" if cur[2] else "  [optional: s skips]"), (10, 22), (0, 220, 255))
        if p is not None:
            dxy = np.hypot(p[0] - cur[0], p[1] - cur[1]) * 100 if cur else 0
            put(img, f"fingertip x {p[0]*100:5.1f} y {p[1]*100:+5.1f} cm  z {p[2]*1000:5.1f} mm   "
                     f"{dxy:4.1f} cm from target   tilt {tilt:4.0f} deg (aim ~30)", (10, 46),
                (0, 255, 0) if dxy < 2.0 and 15 <= tilt <= 50 else (0, 165, 255))
        put(img, "magenta cross = where the fingertip is now (projected on the table)", (10, 70), (255, 0, 255), 0.45)
        if msg:
            put(img, msg, (10, img.shape[0] - 12), (0, 0, 255))
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        try:
            closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if k == ord("q") or closed:
            break
        if k == ord("u") and done:
            t, _ = done.pop()
            todo.insert(0, t)
            msg = "undone"
        if k == ord("s") and cur and not cur[2]:
            todo.pop(0)
            msg = "skipped"
        if k == ord(" ") and cur:
            msg = ""
            if p is None:
                msg = "no joint readings"
            elif np.hypot(p[0] - cur[0], p[1] - cur[1]) > 0.04:
                msg = "fingertip is more than 4 cm from the target circle"
            else:
                done.append((todo.pop(0), p.tolist()))
                print(f"touch near ({cur[0]:.2f}, {cur[1]:+.2f}): fingertip ({p[0]:.4f}, {p[1]:+.4f}) "
                      f"z {p[2]*1000:.1f} mm, tilt {tilt:.0f} deg", flush=True)
        n_req = sum(1 for (x, y, r) in TARGETS if r)
        if done and not todo and sum(1 for t, _ in done if t[2]) == n_req:
            P = np.array([pp for _, pp in done]
                         + [cal["touches_robot"][k] for k in ("orig:880", "orig:107") if k in cal["touches_robot"]])
            coef, res = fit_plane(P)
            a, b, c = coef
            z_gate = {f"{y:+.2f}": round((a + b * 0.40 + c * y) * 1000, 1) for y in (-0.24, -0.12, 0.0, 0.12, 0.24)}
            out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "plane_z=a+bx+cy": coef.tolist(),
                   "points_robot": P.tolist(), "residual_mm": (res * 1000).round(1).tolist(),
                   "z_at_gate_x0.40_mm": z_gate,
                   "tilt_deg_x_y": [round(float(np.degrees(np.arctan(b))), 2), round(float(np.degrees(np.arctan(c))), 2)]}
            (CAL / "table_height.json").write_text(json.dumps(out, indent=2))
            print("\n=== table height (arm's own frame) ===")
            print("along the intercept line x = 0.40 m (mm):", z_gate)
            print("plane residuals (mm):", out["residual_mm"])
            print(f"apparent slope: {out['tilt_deg_x_y'][0]:+.2f} deg along x, {out['tilt_deg_x_y'][1]:+.2f} deg along y")
            print("saved", CAL / "table_height.json")
            saved = True
            break
    cv2.destroyAllWindows()
    cam.close()
    rs.close()
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
