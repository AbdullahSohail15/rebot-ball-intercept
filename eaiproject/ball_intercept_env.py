"""MuJoCo environment: reBot Arm B601-RS on a table, intercepting a rolling ball.

The arm is mounted at the -x end of the table (world origin, tabletop at z=0).
A ball is launched from the +x end and rolls toward the arm. The arm keeps its
gripper wide open with the fingers spread sideways, and moves it so the ball rolls
*between* the two fingers. Two gripper modes:

  "forward" (default)  the gripper faces the incoming ball: fingers point toward it,
                       pitched down 30 deg so the fingertips are near the table (the
                       gripper housing is taller than the fingers, so a level gripper
                       would let the ball roll under them). The ball rolls into the
                       jaw and stops against the palm.
  "down"               fingers point straight down like two goal posts; the ball rolls
                       through the jaw and out the back.

In both modes the jaw is yawed to face the ball's direction of travel.

Control interface
-----------------
The model's actuators are torque motors (same as Seeed's own MuJoCo sim).
`step(action)` takes joint-position targets and runs a PD + gravity-compensation
loop at the physics rate, with the gains Seeed uses in rebotarm_mujoco_rs:
    action[:6]  arm joint targets (rad)
    action[6]   gripper opening target (joint7, m, 0 = closed, 0.05 = fully open)
Use `ik_jaw(x, y, z, yaw)` to turn a desired jaw pose into joint targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
ARM_KP = np.array([80.0, 100.0, 100.0, 35.0, 25.0, 18.0])  # Seeed rebotarm_mujoco_rs defaults
ARM_KD = np.array([8.0, 10.0, 10.0, 4.0, 3.0, 2.5])
GRIPPER_KP, GRIPPER_KD = 1800.0, 18.0
GRIPPER_OPEN = 0.05

# gripper_end frame: local x = finger direction (origin at the fingertips), local y = opening
# direction (fingers at +-y), local z = finger thickness.
MODES = {
    # pitch: rotation about world y applied to the identity pose (x forward = +x world).
    # "forward": fingers toward the ball (+x), tipped down 30 deg. "down": fingers to -z.
    # gate_x: intercept line. Forward mode uses 0.40 because closer in, joint3 sits near its
    # -0.2 rad limit (0.32 -> -0.06 rad) and the IK clamps and drifts.
    # pitch 30 deg, fingertips 20 mm up (the fingertip is the finger's lowest point): fingers reach
    # just below the ball's equator (22 mm) and the palm stops the ball. With the fingertips above
    # ~30 mm, or at pitch >= 40 deg, the ball rolls under the palm and out the back; below ~16 mm the
    # housing/fingers touch the table during fast moves.
    "forward": dict(pitch=np.radians(30.0), jaw_height=0.020, gate_x=0.40,
                    seeds=[[0, 1.2, 1.6, -0.6, 0, 0], [0, 1.8, 1.16, -0.93, 0, 0], [0, 0.8, 1.2, 0.2, 0, 0],
                           [0, 1.5, 2.2, 0.5, 0, 0], [0, 0.5, 0.8, -0.2, 0, 0]]),
    "down": dict(pitch=np.pi / 2, jaw_height=0.015, gate_x=0.32, seeds=[[0, 1.8, 1.16, -0.93, 0, 0]]),
}


def jaw_rotation(pitch: float, yaw: float = 0.0) -> np.ndarray:
    """Target gripper_end orientation: tip the fingers down by `pitch`, then turn by `yaw`
    about world z. yaw = 0 means the jaw faces +x, i.e. toward a ball rolling in along -x."""
    c, s = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry


@dataclass
class Launch:
    """Ball launch parameters. Heading 0 = straight toward the arm (-x)."""
    x0: float = 0.95
    y0: float = 0.0
    speed: float = 0.8          # m/s
    heading: float = 0.0        # rad, positive turns the velocity toward +y

    @staticmethod
    def random(rng: np.random.Generator, gate_x: float = 0.32, y_gate_max: float = 0.22) -> "Launch":
        """Random launch whose straight-line crossing of the gate line lies in |y| <= y_gate_max."""
        x0 = 0.95
        y0 = rng.uniform(-0.25, 0.25)
        y_gate = rng.uniform(-y_gate_max, y_gate_max)
        heading = np.arctan2(y_gate - y0, x0 - gate_x)
        return Launch(x0=x0, y0=y0, speed=rng.uniform(0.6, 1.0), heading=float(heading))


@dataclass
class TrialResult:
    crossed: bool = False
    crossing_time: float | None = None
    lateral_offset: float | None = None   # ball centre in jaw frame (m), 0 = centred
    gap_half_width: float | None = None   # max |offset| that clears both fingers (m)
    touched_arm: bool = False             # ball touched the arm before/while passing the jaw
    touched_bodies: set = field(default_factory=set)
    arm_env_contacts: set = field(default_factory=set)  # arm geoms hitting table/rails (should stay empty)
    min_jaw_height: float = float("inf")                # lowest fingertip-origin z during the trial
    reached_palm: bool = False            # forward mode: ball rolled in and hit the palm

    @property
    def passed_between(self) -> bool:
        return bool(self.crossed and abs(self.lateral_offset) < self.gap_half_width)

    @property
    def clean_pass(self) -> bool:
        return self.passed_between and not self.touched_arm

    def summary(self) -> str:
        if not self.crossed:
            return "ball did not reach the gripper"
        tag = "CLEAN PASS" if self.clean_pass else ("PASS (touched)" if self.passed_between else "MISS")
        if self.reached_palm and self.passed_between:
            tag = "BALL IN GRIPPER" + ("" if self.clean_pass else " (touched fingertip)")
        return f"{tag}: offset {self.lateral_offset*1000:+.1f} mm (clearance limit +-{self.gap_half_width*1000:.1f} mm)"


class BallInterceptEnv:
    def __init__(self, mode: str = "forward", xml_path: str | Path = SCENE_XML, control_dt: float = 0.01,
                 jaw_height: float | None = None):
        assert mode in MODES, f"mode must be one of {list(MODES)}"
        self.mode = mode
        self.pitch = MODES[mode]["pitch"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = max(1, round(control_dt / m.opt.timestep))
        self.control_dt = self.n_sub * m.opt.timestep
        # Height of the fingertip point (gripper_end origin) above the table.
        self.jaw_height = MODES[mode]["jaw_height"] if jaw_height is None else jaw_height

        jid = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.arm_lo, self.arm_hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.arm_act = np.array([m.actuator(f"{n}_motor").id for n in ARM_JOINTS])
        self.grip_act = m.actuator("joint7_motor").id
        self.grip_joints = [m.joint(n) for n in ("joint7", "joint_left", "joint_right")]
        self.j7_q, self.j7_d = m.joint("joint7").qposadr[0], m.joint("joint7").dofadr[0]

        self.ball_q = m.joint("ball_free").qposadr[0]
        self.ball_d = m.joint("ball_free").dofadr[0]
        self.ball_geom = m.geom("ball_geom").id
        self.ball_r = float(m.geom_size[self.ball_geom, 0])
        self.jaw_body = m.body("gripper_end").id
        self.gate_x = MODES[mode]["gate_x"]
        m.geom_pos[m.geom("intercept_line").id, 0] = self.gate_x  # move the visual marker
        m.cam_pos[m.camera("gate_view").id, 0] = self.gate_x + 0.40  # keep the close-up on the line

        # Arm geoms = everything in the subtree of base_link.
        base = m.body("base_link").id
        arm_bodies = {b for b in range(m.nbody) if self._is_descendant(b, base)}
        self.arm_geom_mask = np.isin(m.geom_bodyid, list(arm_bodies))
        self.finger_geoms = {
            side: [g for g in range(m.ngeom)
                   if m.geom_bodyid[g] == m.body(f"gripper_{side}").id and m.geom_contype[g]]
            for side in ("left", "right")
        }
        self._ik_data = mujoco.MjData(m)
        # IK is seed-sensitive (a bad seed runs into the joint limits): try several seeds.
        sols = [self.ik_jaw(self.gate_x, 0.0, self.jaw_height, q0=np.array(sd, float), iters=400)
                for sd in MODES[mode]["seeds"]]
        errs = [self.jaw_error(q, self.gate_x, 0.0, self.jaw_height) for q in sols]
        self.q_home = sols[int(np.argmin(errs))]
        assert min(errs) < 1e-3, f"home IK did not converge (error {min(errs):.4f} m)"
        self.result = TrialResult()
        self.target = np.r_[self.q_home, GRIPPER_OPEN]

    # ------------------------------------------------------------------ helpers
    def _is_descendant(self, b: int, root: int) -> bool:
        while b != 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    def ik_jaw(self, x: float, y: float, z: float, yaw: float = 0.0, q0: np.ndarray | None = None,
               iters: int = 200, tol: float = 1e-5) -> np.ndarray:
        """Damped-least-squares IK: put the fingertip point (gripper_end origin) at (x, y, z) with
        the mode's pitch and the given yaw (see `jaw_rotation`). Respects joint limits.
        Seed-sensitive: warm-start from the current or home pose, and check with `jaw_error`."""
        R_t = jaw_rotation(self.pitch, yaw)
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        q = (self.data.qpos[self.arm_qadr] if q0 is None else q0).copy()
        p_t = np.array([x, y, z])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.jaw_body].reshape(3, 3)
            e = np.r_[p_t - d.xpos[self.jaw_body],
                      0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))]
            if e @ e < tol**2:
                break
            mujoco.mj_jacBody(m, d, jp, jr, self.jaw_body)
            J = np.vstack([jp[:, self.arm_dadr], jr[:, self.arm_dadr]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.arm_lo, self.arm_hi)
        return q

    def jaw_error(self, q: np.ndarray, x: float, y: float, z: float) -> float:
        """Position error (m) of the jaw for arm configuration q; use it to check IK convergence."""
        d = self._ik_data
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return float(np.linalg.norm(d.xpos[self.jaw_body] - [x, y, z]))

    def jaw_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.jaw_body].copy(), self.data.xmat[self.jaw_body].reshape(3, 3).copy()

    def _jaw_axes(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal unit vectors: `fwd` points from the jaw toward the incoming ball, `lat`
        along the opening (local y)."""
        f = R[:, 0] if self.mode == "forward" else R[:, 2]
        f = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
        return f, np.array([-f[1], f[0], 0.0])

    def finger_inner_faces(self) -> tuple[float, float]:
        """Inner finger faces along the jaw's local y axis (m, relative to gripper_end origin),
        measured from the finger collision meshes in the current sim state."""
        m, d = self.model, self.data
        p0, R = self.jaw_pose()
        ys = {}
        for side, geoms in self.finger_geoms.items():
            pts = []
            for g in geoms:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                pts.append((world - p0) @ R)  # into jaw frame
            ys[side] = np.concatenate(pts)[:, 1]
        # left fingers sit at -y, right at +y
        return float(ys["left"].max()), float(ys["right"].min())

    # --------------------------------------------------------------- core API
    def reset(self, launch: Launch | None = None, q_arm: np.ndarray | None = None) -> dict:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        launch = launch or Launch()
        q_arm = self.q_home if q_arm is None else q_arm
        d.qpos[self.arm_qadr] = q_arm
        for j in self.grip_joints:
            d.qpos[j.qposadr[0]] = GRIPPER_OPEN
        # Ball rolling without slipping from the first step: w = v x n / r.
        v = launch.speed * np.array([-np.cos(launch.heading), np.sin(launch.heading), 0.0])
        d.qpos[self.ball_q:self.ball_q + 7] = [launch.x0, launch.y0, self.ball_r, 1, 0, 0, 0]
        d.qvel[self.ball_d:self.ball_d + 3] = v
        d.qvel[self.ball_d + 3:self.ball_d + 6] = np.cross([0, 0, 1], v) / self.ball_r
        mujoco.mj_forward(m, d)
        self.target = np.r_[q_arm, GRIPPER_OPEN]
        self.launch = launch
        self.result = TrialResult()
        self._prev_s = self._ball_in_jaw()[0]
        return self.obs()

    def obs(self) -> dict:
        d = self.data
        return dict(
            time=d.time,
            arm_q=d.qpos[self.arm_qadr].copy(), arm_qd=d.qvel[self.arm_dadr].copy(),
            gripper_q=float(d.qpos[self.j7_q]),
            ball_pos=d.qpos[self.ball_q:self.ball_q + 3].copy(),
            ball_vel=d.qvel[self.ball_d:self.ball_d + 3].copy(),
            jaw_pos=self.jaw_pose()[0],
        )

    def _ball_in_jaw(self) -> np.ndarray:
        """(s, l): ball centre ahead of the fingertip point along `fwd`, and sideways along `lat`."""
        p0, R = self.jaw_pose()
        f, lat = self._jaw_axes(R)
        rel = self.data.qpos[self.ball_q:self.ball_q + 3] - p0
        return np.array([rel @ f, rel @ lat])

    def _control(self):
        d = self.data
        q, qd = d.qpos[self.arm_qadr], d.qvel[self.arm_dadr]
        tau = d.qfrc_bias[self.arm_dadr] + ARM_KP * (self.target[:6] - q) - ARM_KD * qd
        d.ctrl[self.arm_act] = tau
        d.ctrl[self.grip_act] = (GRIPPER_KP * (self.target[6] - d.qpos[self.j7_q])
                                 - GRIPPER_KD * d.qvel[self.j7_d] + d.qfrc_bias[self.j7_d])

    def _track_outcome(self):
        m, d, r = self.model, self.data, self.result
        r.min_jaw_height = min(r.min_jaw_height, float(d.xpos[self.jaw_body][2]))
        for c in d.contact[:d.ncon]:
            a1, a2 = self.arm_geom_mask[c.geom1], self.arm_geom_mask[c.geom2]
            if a1 != a2 and self.ball_geom not in (c.geom1, c.geom2):
                arm_g, env_g = (c.geom1, c.geom2) if a1 else (c.geom2, c.geom1)
                r.arm_env_contacts.add((m.body(m.geom_bodyid[arm_g]).name, m.geom(env_g).name or "table"))
        s, lat = self._ball_in_jaw()
        # "touched" window: forward mode, until the ball enters the jaw mouth (after that it is
        # supposed to hit the palm); down mode, until it is fully behind the ~4 cm thick fingers
        # (later collisions with the arm base are not part of the task).
        window = 0.0 if self.mode == "forward" else -(0.02 + self.ball_r + 0.01)
        for c in d.contact[:d.ncon]:
            g = c.geom1 if c.geom2 == self.ball_geom else (c.geom2 if c.geom1 == self.ball_geom else -1)
            if g < 0 or not self.arm_geom_mask[g]:
                continue
            body = m.body(m.geom_bodyid[g]).name
            if s > window and (self.mode == "down" or not r.crossed):
                r.touched_arm = True
                r.touched_bodies.add(body)
            if r.crossed and body == "gripper_end":
                r.reached_palm = True
        # The ball crosses the fingertip plane (forward) / finger plane (down).
        if not r.crossed and self._prev_s > 0.0 >= s:
            r.crossed, r.crossing_time = True, d.time
            left, right = self.finger_inner_faces()
            r.lateral_offset = float(lat - 0.5 * (left + right))
            # The ball fits if its centre is within (half gap - radius) of the jaw centre.
            r.gap_half_width = 0.5 * (right - left) - self.ball_r
        self._prev_s = s

    def step(self, action: np.ndarray | None = None) -> dict:
        if action is not None:
            self.target = np.asarray(action, float).copy()
            self.target[:6] = np.clip(self.target[:6], self.arm_lo, self.arm_hi)
        for _ in range(self.n_sub):
            self._control()
            mujoco.mj_step(self.model, self.data)
            self._track_outcome()
        return self.obs()


class InterceptPolicy:
    """Scripted baseline: predict where the ball crosses the gate line (straight-line
    extrapolation of its current velocity), and IK the open jaw there."""

    def __init__(self, env: BallInterceptEnv, max_lateral_speed: float = 1.2, max_yaw_rate: float = 3.0,
                 max_y: float = 0.24, max_yaw: float = 0.6, face_ball: bool = True, hover: float = 0.015):
        self.env = env
        self.max_yaw, self.face_ball = max_yaw, face_ball
        # Rate limits are applied to the jaw pose, not the joints: joint-space rate limiting makes
        # the jaw sag on its way sideways and the fingertips scrape the table.
        self.dy_max = max_lateral_speed * env.control_dt
        self.dyaw_max = max_yaw_rate * env.control_dt
        self.max_y = max_y  # open fingers reach the side rails beyond |y| ~ 0.255
        # Hold the jaw `hover` m higher while the ball is far away and lower it as the ball comes in:
        # the soft PD lets the jaw sag ~2 cm during the big sideways moves early in a trial.
        self.hover = hover
        self.reset()

    def reset(self):
        env = self.env
        self.q_cmd = env.data.qpos[env.arm_qadr].copy()
        self.y_cmd, self.yaw_cmd = float(env.data.xpos[env.jaw_body][1]), 0.0

    def predict_crossing_y(self, obs: dict) -> float:
        p, v = obs["ball_pos"], obs["ball_vel"]
        if v[0] > -1e-3:  # not moving toward the arm
            return float(p[1])
        t = (self.env.gate_x - p[0]) / v[0]
        return float(p[1] + v[1] * max(t, 0.0))

    def yaw(self, obs: dict) -> float:
        """Face the jaw back along the ball's velocity (0 when it rolls straight along -x)."""
        v = obs["ball_vel"]
        if not self.face_ball or v[0] > -1e-3:
            return 0.0
        return float(np.clip(np.arctan2(-v[1], -v[0]), -self.max_yaw, self.max_yaw))

    def __call__(self, obs: dict) -> np.ndarray:
        env = self.env
        if env.result.crossed or obs["ball_pos"][0] < env.gate_x - 0.05:  # ball arrived: hold still
            return np.r_[self.q_cmd, GRIPPER_OPEN]
        y = float(np.clip(self.predict_crossing_y(obs), -self.max_y, self.max_y))
        self.y_cmd += float(np.clip(y - self.y_cmd, -self.dy_max, self.dy_max))
        self.yaw_cmd += float(np.clip(self.yaw(obs) - self.yaw_cmd, -self.dyaw_max, self.dyaw_max))
        dist = obs["ball_pos"][0] - env.gate_x
        z = env.jaw_height + self.hover * float(np.clip((dist - 0.10) / 0.20, 0.0, 1.0))
        self.q_cmd = env.ik_jaw(env.gate_x, self.y_cmd, z, yaw=self.yaw_cmd, q0=self.q_cmd, iters=30)
        return np.r_[self.q_cmd, GRIPPER_OPEN]
