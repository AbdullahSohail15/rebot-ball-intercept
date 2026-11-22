"""Camera-to-robot calibration with the printed ChArUco board + 4 fingertip touches.

1. CAMERA  Tape the board flat on the table in front of the arm (inside its reach, around the
           intercept line), hands and arm out of the camera's view of it. SPACE averages the board
           pose over 30 frames.
2. TOUCH   Guide the unpowered arm by hand and rest the CLOSED fingertips on the circled corners
           A, B, C, D in turn, SPACE at each. Your hands can be anywhere in the picture now.
3. Result  The board pose in the robot frame (from the touches) joined with the board pose in the
           camera (from step 1) gives the camera pose in the robot frame. The touches also measure
           the table plane in the robot frame (height and tilt): the arm is on a plate, so the table
           is not at z = 0, and every height command must use the measured value.

Checks printed: touch residuals and the measured side lengths vs the printed 245 x 175 mm (a test of
the arm's joint zeros / kinematics), the table height and tilt, and whether the board moved.
Output: outputs/calibration/camera_extrinsics.json + calibration_overlay.png

Keys: SPACE record, u undo last, q quit. Read-only on CAN: no motor is enabled.

    .venv/bin/python scripts/real/make_calib_board.py    # print outputs/calibration/charuco_board_A4.pdf at 100%
    scripts/real/start_camera.sh                         # camera server, once per session
    .venv/bin/python scripts/real/calibrate_board.py
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
from eaiproject import calib_board as cb  # noqa: E402

OUT = ROOT / "outputs" / "calibration"
BALL_R = 0.022


def average_pose(poses):
    """Mean of (rvec, tvec) samples: chordal mean of rotations, mean of translations."""
    Rs = np.array([cv2.Rodrigues(np.asarray(r, float))[0] for r, _ in poses])
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, np.mean([t for _, t in poses], axis=0)


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| (P, Q: N x 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def compute(R_cb, t_cb, touches_robot: dict) -> dict:
    """Board->camera (R_cb, t_cb) and robot-frame touches of A..D -> robot->camera extrinsics + checks."""
    names = [n for n in "ABCD" if n in touches_robot]
    P = np.array([cb.TOUCH_POINTS[n] for n in names])
    Q = np.array([touches_robot[n] for n in names])
    R_rb, t_rb = kabsch(P, Q)                         # board -> robot
    resid = np.linalg.norm((R_rb @ P.T).T + t_rb - Q, axis=1)
    R_cr = R_cb @ R_rb.T                              # robot -> camera
    t_cr = t_cb - R_cr @ t_rb
    normal = R_rb @ np.array([0, 0, -1.0])            # board z points into the table -> up = -z
    # table plane z = a + b x + c y, through the touched points (fingertip on paper = table surface)
    A = np.c_[np.ones(len(Q)), Q[:, 0], Q[:, 1]]
    a, b, c = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
    sides = {}
    for u, v, nominal in (("A", "B", cb.W), ("B", "C", cb.H), ("C", "D", cb.W), ("D", "A", cb.H), ("A", "C", np.hypot(cb.W, cb.H))):
        if u in touches_robot and v in touches_robot:
            sides[u + v] = {"measured_mm": round(float(np.linalg.norm(np.subtract(touches_robot[u], touches_robot[v]))) * 1000, 1),
                            "printed_mm": round(nominal * 1000, 1)}
    return {
        "rvec": cv2.Rodrigues(R_cr)[0].reshape(3).tolist(), "tvec": t_cr.tolist(),
        "camera_position_robot_frame": (-R_cr.T @ t_cr).tolist(),
        "board_to_robot": {"R": R_rb.tolist(), "t": t_rb.tolist()},
        "touch_residual_mm": dict(zip(names, (resid * 1000).round(1).tolist())),
        "side_lengths": sides,
        "table_plane": {"z = a + b*x + c*y": [float(a), float(b), float(c)],
                        "z_at_board_centre_mm": round(float(a + b * Q[:, 0].mean() + c * Q[:, 1].mean()) * 1000, 1),
                        "tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(normal[2]))))), 2)},
    }


def put(img, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay(img, K, dist, rvec, tvec, plane, touches):
    out = img.copy()
    a, b, c = plane

    def proj(pts):
        p, _ = cv2.projectPoints(np.asarray(pts, float), np.asarray(rvec, float), np.asarray(tvec, float), K, dist)
        return p.reshape(-1, 2).round().astype(int)

    zt = lambda x, y: a + b * x + c * y
    for x in np.arange(0.1, 1.21, 0.1):
        s = proj([[x, -0.35, zt(x, -0.35)], [x, 0.35, zt(x, 0.35)]])
        cv2.line(out, tuple(s[0]), tuple(s[1]), (255, 200, 0), 1)
    for y in np.arange(-0.3, 0.31, 0.1):
        s = proj([[0.1, y, zt(0.1, y)], [1.2, y, zt(1.2, y)]])
        cv2.line(out, tuple(s[0]), tuple(s[1]), (255, 200, 0), 1)
    o, ex, ey, ez = proj([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    for e, col in ((ex, (0, 0, 255)), (ey, (0, 255, 0)), (ez, (255, 0, 0))):
        cv2.arrowedLine(out, tuple(o), tuple(e), col, 2)
    for n, p in touches.items():
        q = proj([p])[0]
        cv2.circle(out, tuple(q), 6, (255, 0, 255), 2)
        put(out, n, (q[0] + 6, q[1] - 6), (255, 0, 255))
    put(out, "10 cm grid on the measured table plane; arm base axes x red, y green, z blue", (10, 20))
    put(out, "magenta = fingertip touches reprojected (should sit on the board corners)", (10, 42))
    return out


def main() -> int:
    from eaiproject.realsense_client import RealSenseClient
    from eaiproject.robstride import ARM_IDS, P_MECHPOS, RobStrideBus
    from eaiproject.kinematics import Fingertip

    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh")
        return 2
    rs = RobStrideBus(movable=())
    tip = Fingertip()

    def read_q():
        q = [rs.read_param_f32(m, P_MECHPOS, timeout=0.03) for m in ARM_IDS]
        return None if any(v is None for v in q) else np.array(q)

    if read_q() is None:
        print("Arm not answering on CAN (powered? cable?)")
        return 2
    K, dist = cam.K, cam.dist
    win = "board calibration (SPACE record, u undo, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = 2 if cam.width < 900 else 1
    cv2.resizeWindow(win, cam.width * scale, cam.height * scale)
    stage, msg = "CAMERA", ""
    samples, cam_pose, cam_corners, touches = [], None, None, {}
    last_fn, result = None, None
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        det = cb.estimate_pose(f.img, K, dist)
        if det is not None:
            rv, tv, n, err, corners = det
            cv2.drawFrameAxes(img, K, dist, rv, tv, 0.05, 2)
            for p in corners:
                cv2.circle(img, tuple(np.round(p).astype(int)), 2, (0, 255, 0), -1)
        if stage == "CAMERA":
            put(img, "1/2 CAMERA: board flat on the table, hands and arm clear; SPACE", (10, 22), (0, 220, 255))
            if det is None:
                put(img, "board not detected", (10, 44), (0, 0, 255))
            else:
                ok = n >= 20 and err < 1.0
                put(img, f"corners {n}/24  reprojection {err:.2f} px", (10, 44), (0, 255, 0) if ok else (0, 165, 255))
        else:
            nxt = next((c for c in "ABCD" if c not in touches), None)
            q = read_q()
            p = tip(q) if q is not None else None
            put(img, f"2/2 TOUCH: rest the CLOSED fingertips on corner {nxt}; SPACE   (done: {''.join(touches) or '-'})",
                (10, 22), (0, 220, 255))
            if p is not None:
                put(img, f"fingertip x {p[0]*100:6.1f}  y {p[1]*100:6.1f}  z {p[2]*1000:6.1f} mm", (10, 44))
            if cam_pose is not None:            # show where the corner is, from the camera's board pose
                R_cb, t_cb = cam_pose
                pc, _ = cv2.projectPoints(np.array([cb.TOUCH_POINTS[nxt]]), cv2.Rodrigues(R_cb)[0], t_cb, K, dist)
                pc = tuple(np.round(pc.reshape(2)).astype(int))
                cv2.circle(img, pc, 10, (0, 220, 255), 2)
                put(img, nxt, (pc[0] + 12, pc[1] - 8), (0, 220, 255))
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
        if k == ord("u"):
            if touches:
                touches.pop(list(touches)[-1])
                msg = "removed last touch"
            elif stage == "TOUCH":
                stage, cam_pose, msg = "CAMERA", None, "back to the camera step"
        if k == ord(" "):
            msg = ""
            if stage == "CAMERA":
                if det is None or det[2] < 20 or det[3] > 1.0:
                    msg = "need >= 20 corners and < 1 px error (lighting? glare? board flat?)"
                    continue
                samples = []
                t_end = time.monotonic() + 3.0
                fn = f.frame_number
                while len(samples) < 30 and time.monotonic() < t_end:
                    g = cam.read(timeout=2.0, newer_than=fn)
                    fn = g.frame_number
                    d = cb.estimate_pose(g.img, K, dist)
                    if d is not None and d[2] >= 20:
                        samples.append((d[0], d[1]))
                        cam_corners = d[4]
                if len(samples) < 20:
                    msg = "board detection unstable; try again"
                    continue
                cam_pose = average_pose(samples)
                spread = np.std([t for _, t in samples], axis=0) * 1000
                print(f"camera: board at {np.linalg.norm(cam_pose[1]):.3f} m from the camera, "
                      f"pose jitter {spread.max():.2f} mm over {len(samples)} frames", flush=True)
                stage = "TOUCH"
            else:
                q = read_q()
                if q is None:
                    msg = "no joint readings"
                    continue
                nxt = next(c for c in "ABCD" if c not in touches)
                touches[nxt] = tip(q).tolist()
                p = touches[nxt]
                print(f"touch {nxt}: robot ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m", flush=True)
                if len(touches) == 4:
                    # did the board move during the touches? (compare with a fresh detection, arm in view is ok)
                    moved = None
                    d = cb.estimate_pose(cam.read(timeout=2.0).img, K, dist, min_corners=6)
                    if d is not None:
                        _, _, _, _, now_c = d
                        if len(now_c) == len(cam_corners):
                            moved = float(np.linalg.norm(now_c - cam_corners, axis=1).max())
                    R_cb, t_cb = cam_pose
                    res = compute(R_cb, t_cb, touches)
                    result = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera_serial": cam.serial,
                              "width": cam.width, "height": cam.height, "K": K.tolist(), "dist": dist.tolist(),
                              "ball_r": BALL_R, "touches_robot": touches, "board_moved_px": moved,
                              "board_to_camera": {"R": R_cb.tolist(), "t": t_cb.tolist()}, **res}
                    OUT.mkdir(parents=True, exist_ok=True)
                    (OUT / "camera_extrinsics.json").write_text(json.dumps(result, indent=2))
                    ov = overlay(f.img, K, dist, res["rvec"], res["tvec"], res["table_plane"]["z = a + b*x + c*y"], touches)
                    cv2.imwrite(str(OUT / "calibration_overlay.png"), ov)
                    tp = res["table_plane"]
                    print("\n=== calibration ===")
                    print("touch residuals (mm):", res["touch_residual_mm"])
                    for s, v in res["side_lengths"].items():
                        print(f"  side {s}: measured {v['measured_mm']} mm vs printed {v['printed_mm']} mm")
                    print(f"table surface at the board: z = {tp['z_at_board_centre_mm']} mm in the robot frame, tilt {tp['tilt_deg']} deg")
                    cp = res["camera_position_robot_frame"]
                    print(f"camera position (robot frame): x {cp[0]:.3f}  y {cp[1]:.3f}  z {cp[2]:.3f} m")
                    print("board moved during touches:", "unknown (corners hidden)" if moved is None else f"{moved:.1f} px")
                    print(f"saved {OUT / 'camera_extrinsics.json'}  and calibration_overlay.png")
                    cv2.imshow(win, ov)
                    cv2.waitKey(0)
                    break
    cv2.destroyAllWindows()
    cam.close()
    rs.close()
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
