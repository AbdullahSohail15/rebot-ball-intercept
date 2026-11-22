"""Camera-to-robot calibration with the ball and the (unpowered) arm. No markers needed.

Each point:
  1. SEE    put the ball on the table, arm and hands out of the camera's view of it. SPACE records
            the ball's pixel position (averaged over 10 frames; needs exactly one ball blob).
  2. TOUCH  guide the arm by hand until the CLOSED fingertips rest gently on top of the ball
            (centred, without pushing it). SPACE records the fingertip position from the joint
            readings (MuJoCo forward kinematics): the ball centre is right below it, at height r.
            Check: the fingertip height should read ~44 mm (ball diameter) above the table.
Hands in the picture only matter in step 1, so hold the arm only in step 2.
Do 6-10 points spread over the area the arm can reach (near/far, left/right). Then press c.

Result: outputs/calibration/camera_extrinsics.json (K, dist, rvec, tvec: robot frame -> camera),
used by the real tracker, plus an overlay image to eyeball the result.

Keys: SPACE record, u undo last point, c compute (>= 6 points), q quit.
Read-only on the CAN bus: no motor is ever enabled.

    scripts/real/start_camera.sh      # camera server, once per session
    .venv/bin/python scripts/real/legacy/calibrate_ball.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eaiproject.ball_tracker import CameraCalib  # noqa: E402
from eaiproject.realsense_client import RealSenseClient  # noqa: E402
from eaiproject.robstride import ARM_IDS, P_MECHPOS, RobStrideBus  # noqa: E402

SCENE = ROOT / "assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml"
BALL_R = 0.022
HSV_LO, HSV_HI = (5, 120, 120), (25, 255, 255)
OUT = ROOT / "outputs" / "calibration"


class Fingertip:
    """Fingertip point (gripper_end origin, as in BallInterceptEnv) from real joint readings."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.adr = [self.m.joint(f"joint{i}").qposadr[0] for i in ARM_IDS]
        self.body = self.m.body("gripper_end").id

    def __call__(self, q) -> np.ndarray:
        self.d.qpos[:] = self.m.qpos0
        for a, v in zip(self.adr, q):
            self.d.qpos[a] = v
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.body].copy()


def find_ball(img: np.ndarray):
    """All plausible orange round blobs: list of (centroid uv, area)."""
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        a, bw, bh = st[i, 4], st[i, 2], st[i, 3]
        if a >= 20 and max(bw, bh) <= 2.0 * min(bw, bh) and a >= 0.45 * bw * bh:
            out.append((cent[i], int(a)))
    return out, mask


def read_q(rs: RobStrideBus):
    q = [rs.read_param_f32(m, P_MECHPOS, timeout=0.03) for m in ARM_IDS]
    return None if any(v is None for v in q) else np.array(q)


def put(img, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def solve(K, dist, pts_world, pts_img, w, h):
    W = np.asarray(pts_world, np.float64)
    U = np.asarray(pts_img, np.float64)
    ok, rvec, tvec = cv2.solvePnP(W, U, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    ok, rvec, tvec = cv2.solvePnP(W, U, K, dist, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    calib = CameraCalib(K, dist, rvec.reshape(3), tvec.reshape(3), w, h)
    reproj = np.linalg.norm(calib.project(W) - U, axis=1)
    Hinv = np.linalg.inv(calib.plane_homography(BALL_R))
    und = cv2.undistortPoints(U.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    back = (Hinv @ np.c_[und, np.ones(len(und))].T).T
    back = back[:, :2] / back[:, 2:3]
    table_err = np.linalg.norm(back - W[:, :2], axis=1)
    cam_pos = (-calib.R.T @ calib.tvec).tolist()
    return calib, reproj, table_err, cam_pos


def overlay(img, calib: CameraCalib, pts_world, pts_img):
    out = img.copy()
    for x in np.arange(0.0, 1.21, 0.1):                       # table grid on z = 0, 10 cm spacing
        seg = calib.project(np.array([[x, -0.35, 0], [x, 0.35, 0]])).astype(int)
        cv2.line(out, tuple(seg[0]), tuple(seg[1]), (255, 200, 0), 1)
    for y in np.arange(-0.3, 0.31, 0.1):
        seg = calib.project(np.array([[0, y, 0], [1.2, y, 0]])).astype(int)
        cv2.line(out, tuple(seg[0]), tuple(seg[1]), (255, 200, 0), 1)
    o, ex, ey, ez = calib.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])).astype(int)
    for e, c in ((ex, (0, 0, 255)), (ey, (0, 255, 0)), (ez, (255, 0, 0))):
        cv2.arrowedLine(out, tuple(o), tuple(e), c, 2)
    for wpt, u in zip(pts_world, pts_img):
        p = calib.project(np.array([wpt]))[0]
        cv2.circle(out, tuple(np.round(u).astype(int)), 6, (0, 255, 255), 1)
        cv2.drawMarker(out, tuple(np.round(p).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
    put(out, "grid 10 cm on the table (z=0); x red, y green, z blue at the arm base", (10, 20))
    put(out, "yellow = detected ball, magenta = reprojected", (10, 42))
    return out


def main() -> int:
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    rs = RobStrideBus(movable=())
    tip = Fingertip()
    if read_q(rs) is None:
        print("Arm not answering on CAN (powered? cable?)")
        return 2
    K, dist = cam.K, cam.dist
    pts_w, pts_i = [], []
    stage, pending, msg = "SEE", None, ""
    win = "calibration (SPACE record, u undo, c compute, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cam.width * 2 if cam.width < 900 else cam.width, cam.height * 2 if cam.width < 900 else cam.height)
    recent = []
    last_fn = None
    result = None
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        q = read_q(rs)
        p = tip(q) if q is not None else None
        balls, _ = find_ball(f.img)
        for uv, a in balls:
            cv2.circle(img, tuple(np.round(uv).astype(int)), 9, (0, 255, 0), 2)
        recent = (recent + [balls[0][0]])[-10:] if len(balls) == 1 else []
        for u in pts_i:
            cv2.drawMarker(img, tuple(np.round(u).astype(int)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        put(img, f"point {len(pts_w) + 1}  stage {stage}  ({len(pts_w)} recorded, need >= 6)", (10, 22), (0, 220, 255))
        if stage == "TOUCH":
            put(img, "now rest the CLOSED fingertips gently on top of the ball; SPACE", (10, 44))
            if p is not None:
                zmm = p[2] * 1000
                col = (0, 255, 0) if abs(zmm - 44) < 12 else (0, 165, 255)
                put(img, f"fingertip x {p[0]*100:6.1f}  y {p[1]*100:6.1f} cm   height {zmm:5.1f} mm (expect ~44)", (10, 66), col)
        else:
            put(img, "place the ball, hands + arm clear of it; SPACE when the ball is green", (10, 44))
            put(img, f"ball blobs: {len(balls)} (need exactly 1)", (10, 66), (0, 255, 0) if len(balls) == 1 else (0, 0, 255))
        if msg:
            put(img, msg, (10, img.shape[0] - 12), (0, 0, 255))
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        try:
            closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if k == ord("q") or closed:
            if pts_w and not result:
                print(f"quit with {len(pts_w)} points recorded but NOT computed (press c to compute)")
            break
        if k == ord("u") and stage == "TOUCH":
            stage, msg = "SEE", "back to SEE (nothing saved for this point)"
        elif k == ord("u") and pts_w:
            pts_w.pop(); pts_i.pop(); stage, msg = "SEE", "removed last point"
        if k == ord(" "):
            msg = ""
            if stage == "SEE":
                if len(recent) < 10:
                    msg = "need one steady ball detection (hands/arm out of the way?)"
                else:
                    pending, stage = np.mean(recent, axis=0), "TOUCH"
            else:
                if p is None:
                    msg = "no joint readings"
                elif abs(p[2] - 2 * BALL_R) > 0.025:
                    msg = f"fingertip {p[2]*1000:.0f} mm high: should be ~44 mm on top of the ball"
                else:
                    pts_w.append([p[0], p[1], BALL_R])
                    pts_i.append(pending.tolist())
                    print(f"point {len(pts_w)}: pixel ({pending[0]:.1f}, {pending[1]:.1f}) -> robot "
                          f"({p[0]:.3f}, {p[1]:.3f}) m, tip z {p[2]*1000:.0f} mm", flush=True)
                    stage, msg = "SEE", f"point {len(pts_w)} saved; move the arm away, then place the ball somewhere new"
        if k == ord("c"):
            if len(pts_w) < 6:
                msg = "need at least 6 points"
                continue
            calib, reproj, table_err, cam_pos = solve(K, dist, pts_w, pts_i, cam.width, cam.height)
            result = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera_serial": cam.serial,
                "width": cam.width, "height": cam.height, "K": K.tolist(), "dist": dist.tolist(),
                "rvec": calib.rvec.tolist(), "tvec": calib.tvec.tolist(), "camera_position_robot_frame": cam_pos,
                "ball_r": BALL_R, "points_world": pts_w, "points_pixel": pts_i,
                "reproj_err_px": reproj.round(2).tolist(), "table_err_mm": (table_err * 1000).round(1).tolist(),
            }
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "camera_extrinsics.json").write_text(json.dumps(result, indent=2))
            cv2.imwrite(str(OUT / "calibration_overlay.png"), overlay(f.img, calib, pts_w, pts_i))
            print(f"\nreprojection error: mean {reproj.mean():.2f} px, max {reproj.max():.2f} px")
            print(f"table error (pixel -> table vs arm): mean {table_err.mean()*1000:.1f} mm, max {table_err.max()*1000:.1f} mm")
            print(f"camera position in robot frame: x {cam_pos[0]:.3f}  y {cam_pos[1]:.3f}  z {cam_pos[2]:.3f} m")
            print(f"saved {OUT / 'camera_extrinsics.json'} and calibration_overlay.png")
            cv2.imshow(win, overlay(f.img, calib, pts_w, pts_i))
            cv2.waitKey(0)
            break
    cv2.destroyAllWindows()
    cam.close()
    rs.close()
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
