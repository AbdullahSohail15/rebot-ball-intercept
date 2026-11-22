"""Closed loop with vision in the sim: external camera -> BallTracker -> InterceptPolicy.

The policy sees only the tracker's estimate (no ground-truth ball state). Ground truth is used
for scoring and for the comparison marker in the overlay.

SimCamera models a real camera: frames at `fps` (rendered at the control step where sim time
passes the next frame time, timestamped with the true capture time), optional Gaussian pixel
noise, and a fixed `latency` between capture and when the frame reaches the tracker.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .ball_intercept_env import GRIPPER_OPEN, BallInterceptEnv, InterceptPolicy, Launch
from .ball_tracker import BallTracker, camera_from_mujoco

CAMERA = "ext_cam"


class SimCamera:
    def __init__(self, env: BallInterceptEnv, cam: str = CAMERA, width: int = 640, height: int = 480,
                 fps: float = 60.0, latency: float = 0.0, noise_std: float = 0.0, seed: int = 0):
        self.env, self.cam = env, cam
        self.width, self.height, self.fps, self.latency, self.noise_std = width, height, fps, latency, noise_std
        self.renderer = mujoco.Renderer(env.model, height, width)
        mujoco.mj_forward(env.model, env.data)
        self.calib = camera_from_mujoco(env.model, env.data, cam, width, height)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.next_t = self.env.data.time
        self.queue: deque = deque()   # (deliver_time, capture_time, image, true_ball_xyz)

    def poll(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Call once per control step. Captures a frame if one is due, returns frames that have
        arrived (capture_time, rgb image, true ball position at capture)."""
        d = self.env.data
        if d.time + 1e-9 >= self.next_t:
            self.renderer.update_scene(d, self.cam)
            img = self.renderer.render()
            if self.noise_std > 0:
                img = np.clip(img + self.rng.normal(0, self.noise_std, img.shape), 0, 255).astype(np.uint8)
            truth = d.qpos[self.env.ball_q:self.env.ball_q + 3].copy()
            self.queue.append((d.time + self.latency, d.time, img, truth))
            self.next_t += 1.0 / self.fps
            if self.next_t < d.time:                     # never fall behind
                self.next_t = d.time + 1.0 / self.fps
        out = []
        while self.queue and self.queue[0][0] <= d.time + 1e-9:
            _, tc, img, truth = self.queue.popleft()
            out.append((tc, img, truth))
        return out


class TrackedInterceptPolicy(InterceptPolicy):
    """InterceptPolicy driven by the tracker estimate. Holds the home pose until the tracker has
    a few measurements, coasts on the filter if the ball is lost (e.g. hidden by the gripper), and
    holds still once the *estimated* ball position reaches the intercept line."""

    def __call__(self, est_obs: dict | None) -> np.ndarray:
        env = self.env
        if est_obs is None or est_obs["ball_pos"][0] < env.gate_x - 0.02:
            return np.r_[self.q_cmd, GRIPPER_OPEN]
        y = float(np.clip(self.predict_crossing_y(est_obs), -self.max_y, self.max_y))
        self.y_cmd += float(np.clip(y - self.y_cmd, -self.dy_max, self.dy_max))
        self.yaw_cmd += float(np.clip(self.yaw(est_obs) - self.yaw_cmd, -self.dyaw_max, self.dyaw_max))
        dist = est_obs["ball_pos"][0] - env.gate_x
        z = env.jaw_height + self.hover * float(np.clip((dist - 0.10) / 0.20, 0.0, 1.0))
        self.q_cmd = env.ik_jaw(env.gate_x, self.y_cmd, z, yaw=self.yaw_cmd, q0=self.q_cmd, iters=30)
        return np.r_[self.q_cmd, GRIPPER_OPEN]


def estimate_obs(tracker: BallTracker, now: float) -> dict | None:
    """Tracker estimate extrapolated to `now` (compensates the camera latency), as a policy obs."""
    if not tracker.ready:
        return None
    s = tracker.state_at(now)
    return dict(ball_pos=np.r_[s.pos, tracker.r], ball_vel=np.r_[s.vel, 0.0])


@dataclass
class TrackStats:
    det_err_mm: list = field(default_factory=list)      # |detected - true| at capture (mm)
    n_frames: int = 0
    n_detected: int = 0
    pred_cross_y: list = field(default_factory=list)    # (time, predicted crossing y)
    true_cross_y: float | None = None                   # true ball y when it reached x = gate_x
    cv_ms: list = field(default_factory=list)

    def final_pred_error_mm(self, before: float = 0.0) -> float | None:
        """|predicted - true| crossing y, using the last prediction made >= `before` s ahead."""
        if self.true_cross_y is None or not self.pred_cross_y:
            return None
        return abs(self.pred_cross_y[-1][1] - self.true_cross_y) * 1000


class TrackedRunner:
    """Runs vision-in-the-loop trials. `on_frame(runner, img, capture_t, truth)` is called for every
    frame the tracker processes (used by the live viewer / video writer)."""

    def __init__(self, env: BallInterceptEnv, camera: SimCamera, tracker: BallTracker,
                 policy: TrackedInterceptPolicy):
        self.env, self.camera, self.tracker, self.policy = env, camera, tracker, policy

    def reset(self, launch: Launch):
        obs = self.env.reset(launch)
        self.camera.reset()
        self.tracker.reset()
        self.policy.reset()
        self.stats = TrackStats()
        self._prev_ball_x = obs["ball_pos"][0]
        return obs

    def step(self, on_frame=None):
        env, tr, st = self.env, self.tracker, self.stats
        for tc, img, truth in self.camera.poll():
            det = tr.process(img, tc)
            st.n_frames += 1
            st.cv_ms.append(tr.timing_ms[-1])
            if det is not None:
                st.n_detected += 1
                st.det_err_mm.append(float(np.linalg.norm(det.xy - truth[:2]) * 1000))
            if on_frame is not None:
                on_frame(self, img, tc, truth)
        now = env.data.time
        est = estimate_obs(tr, now)
        if est is not None and est["ball_pos"][0] > env.gate_x - 0.02:
            pc = tr.predict_crossing(env.gate_x)
            if pc is not None:
                st.pred_cross_y.append((now, pc[0]))
        obs = env.step(self.policy(est))
        bx = obs["ball_pos"][0]
        if st.true_cross_y is None and self._prev_ball_x > env.gate_x >= bx:
            st.true_cross_y = float(obs["ball_pos"][1])
        self._prev_ball_x = bx
        return obs

    def run(self, launch: Launch, t_max: float = 2.0, settle: float = 0.3, on_frame=None):
        self.reset(launch)
        t_end = None
        while self.env.data.time < t_max:
            self.step(on_frame)
            if t_end is None and (self.env.result.crossed or self.env.data.qpos[self.env.ball_q] < -0.1):
                t_end = self.env.data.time + settle
            if t_end is not None and self.env.data.time >= t_end:
                break
        return self.env.result, self.stats


def make_tracked(mode: str = "forward", width: int = 640, height: int = 480, fps: float = 60.0,
                 latency: float = 0.0, noise_std: float = 0.0, seed: int = 0, **tracker_kw) -> TrackedRunner:
    env = BallInterceptEnv(mode)
    env.reset()
    cam = SimCamera(env, width=width, height=height, fps=fps, latency=latency, noise_std=noise_std, seed=seed)
    tracker = BallTracker(cam.calib, ball_r=env.ball_r, **tracker_kw)
    return TrackedRunner(env, cam, tracker, TrackedInterceptPolicy(env))
