"""Camera-to-robot calibration with the ArUco markers on the rig + one fingertip touch per marker.

OLDER SINGLE-VIEW METHOD: valid for one camera position only. The current flow is rig_capture.py +
rig_solve.py once, then relocalize.py every session (docs/calibration_protocol.md).

1. CAMERA  Arm and hands clear of the markers. SPACE averages every marker's corners over 30 frames.
2. TOUCH   For each marker in turn (yellow circle in the window), guide the unpowered arm and rest the
           CLOSED fingertips on the CENTRE of the marker, SPACE. Hands may be anywhere in view now.
3. Solve   Unknowns: camera pose (6) + each marker's rotation on the table (1 each) + small touch
           errors (2 each, with a 2 mm prior). Fitted to all marker corners by least squares, using
           the marker sizes (70 mm table / 50 mm base). The touch heights give the table and plate
           height in the robot frame.

Output: outputs/calibration/camera_extrinsics.json (+ calibration_overlay.png), read by the tracker.
Keys: SPACE record, u undo, s skip this marker, q quit. Read-only on CAN.

    scripts/real/start_camera.sh --width 1280 --height 720 --fps 30
    .venv/bin/python scripts/real/calibrate_markers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import markers  # noqa: E402

OUT = ROOT / "outputs" / "calibration"
ORDER = ["4x4:0", "4x4:2", "orig:880", "orig:107", "orig:467", "orig:718"]   # touch order (base first)
TOUCH_SIGMA = 0.002          # m, prior on how well a marker centre can be touched by eye


def local_corners(size: float) -> np.ndarray:
    """ArUco corner order TL, TR, BR, BL in the marker frame (x right, y up, z out of the marker)."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def marker_world(center, yaw, size):
    return center + (rotz(yaw) @ local_corners(size).T).T


def solve(K, dist, obs: dict, touches: dict) -> dict:
    """obs: key -> 4x2 mean corner pixels; touches: key -> robot xyz of the marker centre."""
    keys = [k for k in ORDER if k in obs and k in touches]
    if len(keys) < 3:
        raise ValueError("need at least 3 touched markers")
    C = np.array([touches[k] for k in keys])
    U = np.array([obs[k].mean(axis=0) for k in keys])
    # init: PnP on the marker centres (+ corners of the first marker to avoid a degenerate 3-point case)
    ok, rv, tv = cv2.solvePnP(C, U, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    R = cv2.Rodrigues(rv)[0]

    def backproject(uv, z):
        d = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
        d_w = R.T @ d
        o_w = -R.T @ tv.reshape(3)
        s = (z - o_w[2]) / d_w[2]
        return o_w + s * d_w

    yaw0 = []
    for k, c in zip(keys, C):
        a, b = backproject(obs[k][0], c[2]), backproject(obs[k][1], c[2])     # TL -> TR = marker +x
        yaw0.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
    n = len(keys)
    x0 = np.r_[rv.reshape(3), tv.reshape(3), yaw0, np.zeros(2 * n)]

    def residuals(x):
        rvec, tvec, yaw, dxy = x[:3], x[3:6], x[6:6 + n], x[6 + n:].reshape(n, 2)
        res = []
        for i, k in enumerate(keys):
            ctr = C[i] + np.r_[dxy[i], 0.0]
            P = marker_world(ctr, yaw[i], markers.SIZES[k])
            p, _ = cv2.projectPoints(P, rvec, tvec, K, dist)
            res.append((p.reshape(-1, 2) - obs[k]).ravel())
        res.append((dxy / TOUCH_SIGMA).ravel())                           # 1 px ~ 1 sigma of touch error
        return np.concatenate(res)

    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0)
    x = sol.x
    rvec, tvec, yaw, dxy = x[:3], x[3:6], x[6:6 + n], x[6 + n:].reshape(n, 2)
    per_marker = {}
    for i, k in enumerate(keys):
        P = marker_world(C[i] + np.r_[dxy[i], 0.0], yaw[i], markers.SIZES[k])
        p, _ = cv2.projectPoints(P, rvec, tvec, K, dist)
        per_marker[k] = {"reproj_px": round(float(np.linalg.norm(p.reshape(-1, 2) - obs[k], axis=1).mean()), 2),
                         "touch_correction_mm": (dxy[i] * 1000).round(1).tolist(),
                         "yaw_deg": round(float(np.degrees(yaw[i])), 1),
                         "centre_robot": (C[i] + np.r_[dxy[i], 0.0]).round(4).tolist()}
    table = [touches[k][2] for k in keys if k.startswith("orig")]
    plate = [touches[k][2] for k in keys if k.startswith("4x4")]
    R = cv2.Rodrigues(rvec)[0]
    return {"rvec": rvec.tolist(), "tvec": tvec.tolist(), "camera_position_robot_frame": (-R.T @ tvec).tolist(),
            "markers": per_marker,
            "table_z": float(np.mean(table)) if table else None, "plate_z": float(np.mean(plate)) if plate else None,
            "cost": float(sol.cost)}


def visible_table(K, dist, rvec, tvec, w, h, z, y_half=0.25):
    """Range of x (robot frame) where the whole strip |y| <= y_half of the table plane is in the image."""
    xs = []
    for x in np.arange(-0.2, 2.01, 0.02):
        P = np.array([[x, -y_half, z], [x, 0, z], [x, y_half, z]])
        cam = (cv2.Rodrigues(np.asarray(rvec))[0] @ P.T).T + tvec
        if np.any(cam[:, 2] <= 0):
            continue
        p, _ = cv2.projectPoints(P, np.asarray(rvec), np.asarray(tvec), K, dist)
        p = p.reshape(-1, 2)
        if np.all((p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)):
            xs.append(x)
    return (min(xs), max(xs)) if xs else None


def put(img, text, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay(img, K, dist, rvec, tvec, z_table, res):
    out = img.copy()
    rv, tv = np.asarray(rvec), np.asarray(tvec)

    def proj(P):
        p, _ = cv2.projectPoints(np.asarray(P, float), rv, tv, K, dist)
        return p.reshape(-1, 2).round().astype(int)

    for x in np.arange(0.0, 1.61, 0.1):
        s = proj([[x, -0.3, z_table], [x, 0.3, z_table]])
        cv2.line(out, tuple(s[0]), tuple(s[1]), (255, 200, 0), 1)
        put(out, f"{x:.1f}", tuple(s[1] + [4, 0]), (255, 200, 0), 0.4)
    for y in np.arange(-0.3, 0.31, 0.1):
        s = proj([[0.0, y, z_table], [1.6, y, z_table]])
        cv2.line(out, tuple(s[0]), tuple(s[1]), (255, 200, 0), 1)
    o, ex, ey, ez = proj([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    for e, col in ((ex, (0, 0, 255)), (ey, (0, 255, 0)), (ez, (255, 0, 0))):
        cv2.arrowedLine(out, tuple(o), tuple(e), col, 2)
    for k, m in res["markers"].items():
        P = marker_world(np.array(m["centre_robot"]), np.radians(m["yaw_deg"]), markers.SIZES[k])
        cv2.polylines(out, [proj(P)], True, (255, 0, 255), 1)
    put(out, "10 cm grid on the table (labels = x in m from the arm base); base axes x red y green z blue", (10, 22))
    put(out, "magenta = markers as fitted (should sit exactly on the real markers)", (10, 46))
    return out


def main() -> int:
    from eaiproject.realsense_client import RealSenseClient
    from eaiproject.robstride import ARM_IDS, P_MECHPOS, RobStrideBus
    from eaiproject.kinematics import Fingertip

    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh --width 1280 --height 720 --fps 30")
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
    win = "marker calibration (SPACE record, u undo, s skip, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(cam.width, 1280), min(cam.height, 720))
    stage, msg, obs, touches, todo = "CAMERA", "", {}, {}, []
    last_fn, result = None, None
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        found = markers.detect(f.img)
        if stage == "CAMERA":
            markers.draw(img, found)
            put(img, "1/2 CAMERA: arm + hands away from the markers; SPACE", (10, 24), (0, 220, 255))
            put(img, f"markers seen: {sorted(found)}  (need >= 3, ideally both base + both near table markers)",
                (10, 50), (0, 255, 0) if len(found) >= 3 else (0, 0, 255))
        else:
            cur = next((k for k in todo if k not in touches), None)
            q = read_q()
            p = tip(q) if q is not None else None
            put(img, f"2/2 TOUCH: rest CLOSED fingertips on the CENTRE of {cur}; SPACE (s = skip)", (10, 24), (0, 220, 255))
            put(img, f"done: {list(touches)}", (10, 50))
            if p is not None:
                put(img, f"fingertip x {p[0]*100:6.1f}  y {p[1]*100:6.1f} cm  z {p[2]*1000:6.1f} mm", (10, 76))
            for k, c in obs.items():
                col = (0, 220, 255) if k == cur else (0, 255, 0) if k in touches else (160, 160, 160)
                cv2.polylines(img, [c.round().astype(int)], True, col, 2)
                ctr = tuple(c.mean(axis=0).round().astype(int))
                cv2.circle(img, ctr, 5 if k != cur else 9, col, 2)
                put(img, k, (ctr[0] + 10, ctr[1] - 10), col, 0.5)
        if msg:
            put(img, msg, (10, img.shape[0] - 14), (0, 0, 255))
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        try:
            closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if k == ord("q") or closed:
            if touches and not result:
                print(f"quit with {len(touches)} touches, not solved")
            break
        if k == ord("u"):
            if touches:
                touches.pop(list(touches)[-1]); msg = "removed last touch"
            elif stage == "TOUCH":
                stage, obs, msg = "CAMERA", {}, "back to the camera step"
        do_solve = False
        if k == ord("s") and stage == "TOUCH":
            cur = next((kk for kk in todo if kk not in touches), None)
            if cur:
                todo.remove(cur); msg = f"skipped {cur}"
                do_solve = all(kk in touches for kk in todo)
        if k == ord(" "):
            msg = ""
            if stage == "CAMERA":
                if len(found) < 3:
                    msg = "need at least 3 markers in view"
                    continue
                acc: dict[str, list] = {}
                fn = f.frame_number
                t_end = time.monotonic() + 4.0
                n_frames = 0
                while n_frames < 30 and time.monotonic() < t_end:
                    g = cam.read(timeout=2.0, newer_than=fn)
                    fn = g.frame_number
                    n_frames += 1
                    for kk, c in markers.detect(g.img).items():
                        acc.setdefault(kk, []).append(c)
                obs = {kk: np.mean(v, axis=0) for kk, v in acc.items() if len(v) >= 0.7 * n_frames}
                jitter = {kk: float(np.std(v, axis=0).max()) for kk, v in acc.items() if kk in obs}
                print("camera: markers", {kk: f"jitter {j:.2f} px" for kk, j in jitter.items()}, flush=True)
                todo = [kk for kk in ORDER if kk in obs]
                if len(todo) < 3:
                    msg = "markers not stable enough; check lighting"
                    continue
                stage = "TOUCH"
            else:
                cur = next((kk for kk in todo if kk not in touches), None)
                q = read_q()
                if q is None or cur is None:
                    msg = "no joint readings"
                    continue
                touches[cur] = tip(q).tolist()
                pp = touches[cur]
                print(f"touch {cur}: robot ({pp[0]:.4f}, {pp[1]:.4f}, {pp[2]:.4f}) m", flush=True)
                do_solve = all(kk in touches for kk in todo)
        if do_solve:
            if len(touches) < 3:
                msg = "need at least 3 touched markers"
                continue
            res = solve(K, dist, obs, touches)
            z_tab = res["table_z"] if res["table_z"] is not None else res["plate_z"]
            vis = visible_table(K, dist, res["rvec"], np.array(res["tvec"]), cam.width, cam.height, z_tab)
            result = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera_serial": cam.serial,
                      "width": cam.width, "height": cam.height, "K": K.tolist(), "dist": dist.tolist(),
                      "ball_r": 0.022, "observed_corners_px": {kk: v.tolist() for kk, v in obs.items()},
                      "touches_robot": touches, "visible_x_range_m": vis, **res}
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "camera_extrinsics.json").write_text(json.dumps(result, indent=2))
            ov = overlay(f.img, K, dist, res["rvec"], res["tvec"], z_tab, res)
            cv2.imwrite(str(OUT / "calibration_overlay.png"), ov)
            print("\n=== calibration ===")
            for kk, m in res["markers"].items():
                print(f"  {kk}: reprojection {m['reproj_px']} px, touch correction {m['touch_correction_mm']} mm")
            if res["table_z"] is not None:
                print(f"table surface: z = {res['table_z']*1000:.1f} mm in the robot frame")
            if res["plate_z"] is not None:
                print(f"plate surface: z = {res['plate_z']*1000:.1f} mm")
            cp = res["camera_position_robot_frame"]
            print(f"camera position (robot frame): x {cp[0]:.3f}  y {cp[1]:.3f}  z {cp[2]:.3f} m")
            print("camera sees the full |y|<=25 cm table strip for x in",
                  "nothing?" if vis is None else f"[{vis[0]:.2f}, {vis[1]:.2f}] m")
            print(f"saved {OUT / 'camera_extrinsics.json'} and calibration_overlay.png")
            cv2.imshow(win, ov)
            cv2.waitKey(0)
            break
    cv2.destroyAllWindows()
    cam.close()
    rs.close()
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
