"""Real-rig glue: calibration loading, the ball tracker on RealSense frames, and the intercept planner.

The planner is the sim's InterceptPolicy logic (straight-line crossing prediction at the intercept
line, jaw facing the ball, rate limits on the jaw pose, IK in the MuJoCo model) driven by the tracker's
estimate and wall-clock time instead of sim state. Heights are relative to the MEASURED table height
along the intercept line (outputs/calibration/table_height.json, in the arm's own kinematic frame).
It only computes targets; whether they are sent to the motors is the caller's business.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ball_intercept_env import BallInterceptEnv
from .ball_tracker import BallTracker, CameraCalib

ROOT = Path(__file__).resolve().parents[1]
CAL_DIR = ROOT / "outputs" / "calibration"
BALL_R = 0.022
DEFAULT_HSV = ((5, 120, 120), (25, 255, 255))
TABLE_XY = ((0.12, 1.00), (-0.36, 0.38))     # tracked region of the real table, robot frame (m)


def load_calibration():
    cal = json.loads((CAL_DIR / "camera_extrinsics.json").read_text())
    calib = CameraCalib(np.array(cal["K"]), np.array(cal["dist"]), np.array(cal["rvec"]), np.array(cal["tvec"]),
                        cal["width"], cal["height"])
    th_path = CAL_DIR / "table_height.json"
    if th_path.exists():
        plane = np.array(json.loads(th_path.read_text())["plane_z=a+bx+cy"])
    else:
        plane = np.array([cal["table_z"], 0.0, 0.0])
    return calib, plane, cal


def table_z(plane, x, y) -> float:
    a, b, c = plane
    return float(a + b * x + c * y)


def shifted(calib: CameraCalib, dz: float) -> CameraCalib:
    """Same camera, world frame moved up by dz (so a table at z = dz becomes z' = 0 for the tracker)."""
    t = calib.tvec + calib.R @ np.array([0.0, 0.0, dz])
    return CameraCalib(calib.K, calib.dist, calib.rvec, t, calib.width, calib.height)


def load_hsv():
    p = CAL_DIR / "ball_hsv.json"
    if p.exists():
        d = json.loads(p.read_text())
        return tuple(d["lo"]), tuple(d["hi"])
    return DEFAULT_HSV


def save_hsv(lo, hi):
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    (CAL_DIR / "ball_hsv.json").write_text(json.dumps({"lo": list(map(int, lo)), "hi": list(map(int, hi))}))


def make_tracker(calib, plane, hsv=None) -> tuple[BallTracker, float]:
    """Tracker working in a frame where the table (mean height over the tracked area) is z' = 0."""
    (x0, x1), (y0, y1) = TABLE_XY
    zt = table_z(plane, 0.5 * (x0 + x1), 0.5 * (y0 + y1))
    lo, hi = hsv or load_hsv()
    trk = BallTracker(shifted(calib, zt), ball_r=BALL_R, hsv_lo=lo, hsv_hi=hi, table_xy=TABLE_XY,
                      roi_half=50, bgr=True)
    return trk, zt


class HostClock:
    """Maps camera hardware timestamps (ms) to time.monotonic(): offset = lower envelope of
    (arrival - hw), so frame intervals come from the camera clock and transport jitter is removed."""

    def __init__(self):
        self.off = None

    def __call__(self, hw_ms: float, t_host: float) -> float:
        o = t_host - hw_ms / 1000.0
        self.off = o if self.off is None else min(self.off + 1e-5, o)   # slow upward drift allowance
        return hw_ms / 1000.0 + self.off


def auto_hsv(img_bgr, table_mask=None):
    """HSV range from the single orange-ish blob in view (ball resting on the table), or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 70, 70), (35, 255, 255))
    if table_mask is not None:
        m &= table_mask
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cands = [i for i in range(1, n) if st[i, 4] >= 30 and max(st[i, 2], st[i, 3]) <= 2 * min(st[i, 2], st[i, 3])]
    if len(cands) != 1:
        return None
    px = hsv[lab == cands[0]]
    h = np.median(px[:, 0])
    lo = (int(max(0, h - 9)), int(max(60, np.percentile(px[:, 1], 5) - 35)), int(max(60, np.percentile(px[:, 2], 5) - 45)))
    hi = (int(min(179, h + 9)), 255, 255)
    return lo, hi


@dataclass
class Plan:
    q: np.ndarray            # arm joint targets (rad)
    jaw: np.ndarray          # fingertip target (x, y, z) robot frame
    yaw: float
    crossing_y: float | None
    t_cross: float | None    # seconds until the ball reaches the line
    ik_err: float
    phase: str               # "home" | "tracking" | "arrived"


class InterceptPlanner:
    """Sim InterceptPolicy, real-time version. Call `plan(tracker, now)` every control tick."""

    def __init__(self, plane, mode: str = "forward", max_lateral_speed: float = 0.6, max_yaw_rate: float = 1.5,
                 max_y: float = 0.24, max_yaw: float = 0.6, hover: float = 0.015):
        self.env = BallInterceptEnv(mode)
        self.plane = plane
        self.gate_x, self.jaw_height = self.env.gate_x, self.env.jaw_height
        self.v_lat, self.w_yaw = max_lateral_speed, max_yaw_rate
        self.max_y, self.max_yaw, self.hover = max_y, max_yaw, hover
        self.q_home = self.env.q_home.copy()
        self.reset()

    def reset(self):
        self.q_cmd = self.q_home.copy()
        self.y_cmd, self.yaw_cmd, self.t_last = 0.0, 0.0, None
        self.env.data.qpos[self.env.arm_qadr] = self.q_home

    def home_jaw(self) -> np.ndarray:
        return self.jaw_target(0.0, 0.0, far=True)

    def jaw_target(self, y, dist, far=False):
        z_tab = table_z(self.plane, self.gate_x, y)
        hover = self.hover if far else self.hover * float(np.clip((dist - 0.10) / 0.20, 0.0, 1.0))
        return np.array([self.gate_x, y, z_tab + self.jaw_height + hover])

    def plan(self, trk: BallTracker, now: float) -> Plan:
        dt = 0.0 if self.t_last is None else min(now - self.t_last, 0.1)
        self.t_last = now
        st = trk.state_at(now) if trk.ready else None
        cross, t_cross, phase = None, None, "home"
        y_goal, yaw_goal, dist = self.y_cmd, self.yaw_cmd, 1.0
        if st is not None and st.vel[0] < -0.05:
            x, y = st.pos
            vx, vy = st.vel
            dist = x - self.gate_x
            if dist < -0.05:
                phase = "arrived"
            else:
                phase = "tracking"
                t_cross = max(dist / -vx, 0.0)
                cross = float(y + vy * t_cross)
                y_goal = float(np.clip(cross, -self.max_y, self.max_y))
                yaw_goal = float(np.clip(np.arctan2(-vy, -vx), -self.max_yaw, self.max_yaw))
        if phase != "arrived":
            self.y_cmd += float(np.clip(y_goal - self.y_cmd, -self.v_lat * dt, self.v_lat * dt))
            self.yaw_cmd += float(np.clip(yaw_goal - self.yaw_cmd, -self.w_yaw * dt, self.w_yaw * dt))
            jaw = self.jaw_target(self.y_cmd, dist, far=(phase == "home"))
            self.q_cmd = self.env.ik_jaw(*jaw, yaw=self.yaw_cmd, q0=self.q_cmd, iters=40)
        jaw = self.jaw_target(self.y_cmd, dist, far=(phase == "home"))
        err = self.env.jaw_error(self.q_cmd, *jaw)
        return Plan(self.q_cmd.copy(), jaw, self.yaw_cmd, cross, t_cross, err, phase)
