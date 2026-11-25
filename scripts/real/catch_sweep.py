"""Powered test after the calibration: the arm moves its jaw along the intercept line, left to
right and back, at the blocking pose, as it would to catch balls anywhere in the catching area.

It goes down in steps: a pass 60 mm above the table, one at 35 mm, one at the blocking height
(20 mm). Heights are above the table plane in config/rig_map.json, so this is also the test that
the calibrated table height is right where the arm will work.

    .venv/bin/python scripts/real/catch_sweep.py --dry-run     # plan and checks only, nothing powered
    .venv/bin/python scripts/real/catch_sweep.py --passes 60   # first time: one pass high above the table
    .venv/bin/python scripts/real/catch_sweep.py               # 60, 35, 20 mm

THIS POWERS ALL SIX ARM JOINTS, including joints 2 and 3 that carry the arm's weight.

How it is kept safe
  start     You hold the unpowered arm by hand near the start pose (over the middle of the line,
            a few cm above the table). ENTER engages the motors at exactly the pose they are in, so
            nothing jumps. Keep your hands on the arm until it prints HOLDING.
  limits    Motor position mode with a low firmware speed limit and a current limit per joint, set
            before enabling. The whole path is checked in the model first: joint limits, IK error,
            and the lowest point of the arm stays above the measured table.
  contact   The arm STOPS AND HOLDS (it does not go limp) when
              - a joint's current differs from the same point of the previous, higher pass by more
                than the contact threshold (touching the table or anything else),
              - a joint lags its target (blocked), or its current stays at the limit,
              - readings are lost, the gripper motor moves, or you press Ctrl+C.
            It then waits: support the arm and press ENTER to release it.
  the end   It returns to the start pose and holds. Support the arm, ENTER releases.
  last resort  A second Ctrl+C, or any unexpected error, disables all motors at once. The arm then
            drops; during the sweep the fingertips are at most 6 cm above the table.

Keep one hand near the arm's power switch. Nobody else within reach of the arm.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.ball_intercept_env import BallInterceptEnv  # noqa: E402
from eaiproject.robstride import ARM_IDS, GRIPPER_ID, JOINT_NAMES, P_MECHPOS, RobStrideBus  # noqa: E402

P_RUN_MODE, P_LOC_REF, P_LIMIT_SPD, P_LIMIT_CUR, P_IQF = 0x7005, 0x7016, 0x7017, 0x7018, 0x701A
RUN_MODE_POSITION = 1
CUR_CAP = {1: 6.0, 2: 18.0, 3: 14.0, 4: 5.0, 5: 4.0, 6: 4.0}          # A, firmware current limit per joint (the model
#                                                                       needs ~8 Nm on joint 2, ~5 Nm on joint 3 to hold the arm)
CONTACT_A = {1: 1.2, 2: 1.2, 3: 1.2, 4: 0.8, 5: 0.8, 6: 0.8}          # A, deviation from the expected current = contact
TRACK_ERR, TRACK_S = 0.04, 0.25          # rad behind the target for this long = blocked
CAP_FRAC, CAP_S = 0.9, 0.2               # current at the limit for this long = blocked
CONTACT_TICKS = 3                        # consecutive control ticks over the contact threshold
START_TOL = 0.30                         # rad: how close to the start pose the hand-held arm must be
MIN_CLEAR = 0.008                        # m: the model's lowest arm point must stay this far above the table
JOINT_SPEED = 0.8                        # rad/s firmware speed limit (the path itself stays below 80 % of it)
VERTICAL_SPEED = 0.03                    # m/s when changing height
GRIPPER_WATCH = math.radians(5.0)

LOG_DIR = ROOT / "outputs" / "hardware"
_now, _sleep = time.monotonic, time.sleep     # replaced by a simulated clock in the tests


class Hold(RuntimeError):
    """Stop moving and hold position; the user then releases the arm."""


@dataclass
class Segment:
    name: str
    Q: np.ndarray                      # (n, 6) joint targets, one per control tick
    compare: int | None = None         # index of the earlier segment whose currents are the contact baseline
    gravity: np.ndarray | None = None  # (n, 6) model gravity torque along the segment, Nm
    iq: np.ndarray | None = None       # filtered currents recorded while running this segment
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------- planning
class Model:
    """IK and clearance checks in the MuJoCo model, against the MEASURED table plane."""

    def __init__(self, plane):
        self.env = BallInterceptEnv("forward")
        self.plane = np.asarray(plane, float)
        m = self.env.model
        self.d = mujoco.MjData(m)
        bodies = {m.body(n).id for n in ("link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")}
        self.geoms = []                                    # (geom id, local mesh vertices) of the arm's collision shapes
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in bodies and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                self.geoms.append((g, m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))

    def pose(self, q):
        self.d.qpos[:] = self.env.model.qpos0
        self.d.qpos[self.env.arm_qadr] = q
        mujoco.mj_kinematics(self.env.model, self.d)

    def fingertip(self, q) -> np.ndarray:
        self.pose(q)
        return self.d.xpos[self.env.jaw_body].copy()

    def clearance(self, q) -> float:
        """Height of the arm's lowest point above the table plane (m)."""
        self.pose(q)
        low = np.inf
        for g, v in self.geoms:
            w = v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            low = min(low, float((w[:, 2] - rc.plane_z(self.plane, w[:, 0], w[:, 1])).min()))
        return low

    def gravity(self, Q) -> np.ndarray:
        """Joint torques (Nm) that hold the arm still at each row of Q, from the model."""
        m, out = self.env.model, []
        for q in Q:
            self.d.qpos[:] = m.qpos0
            self.d.qpos[self.env.arm_qadr] = q
            self.d.qvel[:] = 0
            mujoco.mj_forward(m, self.d)
            out.append(self.d.qfrc_bias[self.env.arm_dadr].copy())
        return np.array(out)

    def jaw(self, y: float, h: float, q0) -> np.ndarray:
        """Joint angles for the jaw on the intercept line at lateral position y, h above the table."""
        x = self.env.gate_x
        z = float(rc.plane_z(self.plane, x, y)) + h
        q = self.env.ik_jaw(x, y, z, q0=np.asarray(q0, float), iters=300)
        err = self.env.jaw_error(q, x, y, z)
        if err > 1e-3:
            raise ValueError(f"no IK solution for the jaw at y = {y:+.2f} m, {h * 1000:.0f} mm above the table (error {err * 1000:.1f} mm)")
        return q


def ease(n: int) -> np.ndarray:
    """n samples from 0 to 1 that start and end at rest."""
    u = np.linspace(0.0, 1.0, max(n, 2))
    return u * u * (3 - 2 * u)


def plan(model: Model, q_start: np.ndarray, heights: list[float], y_max: float, speed: float, rate: float) -> list[Segment]:
    """Start pose -> over the centre -> for each height: centre, left end, right end, centre -> start pose."""
    def cart(y0, h0, y1, h1):
        n = int(max(abs(y1 - y0) / speed, abs(h1 - h0) / min(speed, VERTICAL_SPEED)) * 1.5 * rate)   # eased: peak = the speed
        return [(y0 + s * (y1 - y0), h0 + s * (h1 - h0)) for s in ease(n)]

    def solve(points, q0):
        Q, q = [], q0
        for y, h in points:
            q = model.jaw(y, h, q)
            Q.append(q)
        return np.array(Q)

    q_ready = model.jaw(0.0, heights[0], model.env.q_home)
    n0 = int(np.abs(q_ready - q_start).max() / 0.15 * 1.5 * rate) + int(0.5 * rate)   # hand-held pose -> ready, <= 0.15 rad/s
    segs = [Segment("to the start pose", q_start + ease(n0)[:, None] * (q_ready - q_start))]
    prev, q, h_prev = None, q_ready, heights[0]
    for h in heights:
        if h != h_prev:
            segs.append(Segment(f"down to {h * 1000:.0f} mm", solve(cart(0.0, h_prev, 0.0, h), q)))
            segs[-1].gravity = model.gravity(segs[-1].Q)
            q = segs[-1].Q[-1]
        pts = cart(0.0, h, -y_max, h) + cart(-y_max, h, y_max, h) + cart(y_max, h, 0.0, h)
        segs.append(Segment(f"pass at {h * 1000:.0f} mm", solve(pts, q), compare=prev))
        segs[-1].gravity = model.gravity(segs[-1].Q)
        prev, q, h_prev = len(segs) - 1, segs[-1].Q[-1], h
    if h_prev != heights[0]:
        segs.append(Segment("back up", solve(cart(0.0, h_prev, 0.0, heights[0]), q)))
        segs[-1].gravity = model.gravity(segs[-1].Q)
    return segs


def check_plan(model: Model, segs: list[Segment], rate: float) -> tuple[list[str], dict]:
    """Problems that forbid running the plan (empty = fine), and numbers for the printout."""
    env, problems = model.env, []
    Q = np.concatenate([s.Q for s in segs])
    lo, hi = env.arm_lo + 0.05, env.arm_hi - 0.05
    for j in range(6):
        if Q[:, j].min() < lo[j] or Q[:, j].max() > hi[j]:
            problems.append(f"joint {j + 1} would go to [{Q[:, j].min():+.2f}, {Q[:, j].max():+.2f}] rad, outside its limits "
                            f"[{lo[j]:+.2f}, {hi[j]:+.2f}] with margin")
    step = np.abs(np.diff(Q, axis=0)).max(axis=0) * rate
    if step.max() > 0.8 * JOINT_SPEED:
        problems.append(f"joint {int(step.argmax()) + 1} would have to move at {step.max():.2f} rad/s (limit {0.8 * JOINT_SPEED:.2f}): lower --speed")
    clear = {}
    for s in segs:
        c = min(model.clearance(q) for q in s.Q[::5])
        clear[s.name] = c
        floor = MIN_CLEAR if s is not segs[0] else min(MIN_CLEAR, model.clearance(s.Q[0]) - 0.002)
        if c < floor:
            problems.append(f"'{s.name}': the model's lowest arm point comes within {c * 1000:.0f} mm of the table "
                            f"(need {MIN_CLEAR * 1000:.0f} mm)")
    return problems, {"ticks": len(Q), "seconds": len(Q) / rate, "max_joint_speed": step, "clearance": clear}


# ----------------------------------------------------------------------------------- running
class Arm:
    """The six arm motors in position mode. Every method that could move a motor is here."""

    def __init__(self, rs: RobStrideBus, caps: dict):
        self.rs, self.caps = rs, caps
        self.orig: dict[int, dict] = {}
        self.enabled = False
        self.miss = dict.fromkeys(ARM_IDS, 0)
        self.pos = np.zeros(6)
        self.iq = np.zeros(6)

    def read(self) -> bool:
        """Refresh positions and currents; False if a joint has not answered three times in a row."""
        for i, m in enumerate(ARM_IDS):
            p = self.rs.read_param_f32(m, P_MECHPOS, timeout=0.01)
            c = self.rs.read_param_f32(m, P_IQF, timeout=0.01)
            if p is None or c is None:
                self.miss[m] += 1
                if self.miss[m] >= 3:
                    return False
            else:
                self.miss[m], self.pos[i], self.iq[i] = 0, p, c
        return True

    def configure(self) -> None:
        """Position mode, speed and current limits, target = where each joint is now. Motors disabled."""
        for i, m in enumerate(ARM_IDS):
            self.orig[m] = {"run_mode": self.rs.read_param_u8(m, P_RUN_MODE), "limit_spd": self.rs.read_param_f32(m, P_LIMIT_SPD),
                            "limit_cur": self.rs.read_param_f32(m, P_LIMIT_CUR)}
            if None in self.orig[m].values():
                raise RuntimeError(f"could not read the parameters of {JOINT_NAMES[m]}")
            self.rs.stop(m)
            _sleep(0.01)
            self.rs.write_param_u8(m, P_RUN_MODE, RUN_MODE_POSITION)
            _sleep(0.01)
            if self.rs.read_param_u8(m, P_RUN_MODE) != RUN_MODE_POSITION:
                raise RuntimeError(f"{JOINT_NAMES[m]} did not switch to position mode")
            for idx, val in ((P_LIMIT_SPD, JOINT_SPEED), (P_LIMIT_CUR, self.caps[m]), (P_LOC_REF, float(self.pos[i]))):
                self.rs.write_param_f32(m, idx, val)
                _sleep(0.005)
                back = self.rs.read_param_f32(m, idx)
                if back is None or abs(back - val) > 1e-3:
                    raise RuntimeError(f"{JOINT_NAMES[m]}: parameter 0x{idx:04X} reads back {back}, wrote {val}")

    def engage(self) -> None:
        if not self.read():
            raise RuntimeError("lost readings just before enabling")
        self.command(self.pos)                       # the hand-held arm may have moved since configure()
        self.enabled = True
        for m in (6, 5, 4, 1, 3, 2):
            self.rs.enable(m)
            _sleep(0.01)

    def command(self, q) -> None:
        for m, v in zip(ARM_IDS, q):
            self.rs.write_param_f32(m, P_LOC_REF, float(v))

    def freeze(self) -> None:
        """Hold where the arm is now (this also takes the push off whatever it touched)."""
        self.read()
        self.command(self.pos)

    def release(self) -> None:
        for _ in range(3):
            for m in ARM_IDS:
                try:
                    self.rs.stop(m)
                except Exception:  # noqa: BLE001  (keep stopping the others)
                    pass
            _sleep(0.01)
        self.enabled = False

    def restore(self) -> None:
        for m, o in self.orig.items():
            try:
                self.rs.write_param_f32(m, P_LIMIT_SPD, o["limit_spd"])
                self.rs.write_param_f32(m, P_LIMIT_CUR, o["limit_cur"])
                self.rs.write_param_u8(m, P_RUN_MODE, o["run_mode"])
            except Exception:  # noqa: BLE001  (RAM parameters reset at power-off anyway)
                pass


def amps_per_nm(seg: Segment) -> float:
    """How many amps joint 2 draws per Nm of gravity load, fitted on a finished pass (joint 2's load
    changes by ~3 Nm along the line). Used to predict how the holding currents of joints 2 and 3
    change between two heights; 0 if the fit is not believable (then nothing is predicted)."""
    g, i = seg.gravity[:, 1], seg.iq[:, 1]
    if np.ptp(g) < 1.0:
        return 0.0
    a, _ = np.polyfit(g, i, 1)
    r = np.corrcoef(g, i)[0, 1]
    return float(abs(a)) * float(np.sign(a)) if abs(r) > 0.8 and 0.05 < abs(a) < 5.0 else 0.0


def run_segment(arm: Arm, seg: Segment, base: Segment | None, rate: float, contact: dict, grip0: float | None, log: list,
                t_start: float, a: float | None = None) -> None:
    """Stream one segment. Raises Hold on contact, blockage, lost readings or Ctrl+C.

    Contact = a joint's current leaves what is expected. For a pass that is the pass above it at the
    same point of the line; for a height change, the currents at its start. Both are corrected for
    the model's change in gravity load on joints 2 and 3, scaled by `a` amps per Nm (measured on the
    first pass). The first pass and the move to it have no expectation: only the blockage checks."""
    expect = None
    if a is not None and seg.gravity is not None:
        if base is not None and base.iq is not None:
            expect = base.iq.copy()
            expect[:, 1:3] += a * (seg.gravity[:, 1:3] - base.gravity[:, 1:3])       # joints 2 and 3 (same motor type)
        else:
            expect = np.tile(arm.iq, (len(seg.Q), 1))
            expect[:, 1:3] += a * (seg.gravity[:, 1:3] - seg.gravity[0, 1:3])
    dt = 1.0 / rate
    thr = np.array([contact[m] for m in ARM_IDS])
    caps = np.array([arm.caps[m] for m in ARM_IDS])
    filt = arm.iq.copy()
    rec = np.zeros_like(seg.Q)
    over = 0
    err_since = cap_since = None
    dev_max, err_max = np.zeros(6), 0.0
    for k, q in enumerate(seg.Q):
        t0 = _now()
        arm.command(q)
        if not arm.read():
            raise Hold("lost readings from a joint")
        filt = 0.7 * filt + 0.3 * arm.iq
        rec[k] = filt
        prev_target = seg.Q[max(k - 2, 0)]                               # allow for two ticks of delay
        err = np.minimum(np.abs(q - arm.pos), np.abs(prev_target - arm.pos))
        err_max = max(err_max, float(err.max()))
        err_since = (err_since or t0) if err.max() > TRACK_ERR else None
        cap_since = (cap_since or t0) if np.any(np.abs(arm.iq) > CAP_FRAC * caps) else None
        j = int(err.argmax())
        if err_since and t0 - err_since > TRACK_S:
            raise Hold(f"{JOINT_NAMES[ARM_IDS[j]]} is {math.degrees(err[j]):.1f} deg behind its target: blocked")
        if cap_since and t0 - cap_since > CAP_S:
            j = int((np.abs(arm.iq) / caps).argmax())
            raise Hold(f"{JOINT_NAMES[ARM_IDS[j]]} current {arm.iq[j]:+.1f} A is at its limit {caps[j]:.0f} A: blocked or overloaded")
        if expect is not None:
            dev = filt - expect[k]
            dev_max = np.maximum(dev_max, np.abs(dev))
            over = over + 1 if np.any(np.abs(dev) > thr) else 0
            if over >= CONTACT_TICKS:
                j = int((np.abs(dev) / thr).argmax())
                tip = seg.stats.get("tip", lambda: None)()
                raise Hold(f"CONTACT during '{seg.name}': {JOINT_NAMES[ARM_IDS[j]]} current is {dev[j]:+.1f} A away from the expected "
                           f"(threshold {thr[j]:.1f} A)" + (f", fingertip at y = {tip[1]:+.3f} m" if tip is not None else ""))
        if grip0 is not None and k % 10 == 0:
            g = arm.rs.read_param_f32(GRIPPER_ID, P_MECHPOS, timeout=0.01)
            if g is not None and abs(g - grip0) > GRIPPER_WATCH:
                raise Hold(f"the unpowered gripper motor moved {math.degrees(g - grip0):+.0f} deg: something pushed the jaw")
        log.append((round(t0 - t_start, 4), seg.name, k, *np.round(q, 5), *np.round(arm.pos, 5), *np.round(arm.iq, 3)))
        _sleep(max(0.0, dt - (_now() - t0)))
    seg.iq = rec
    seg.stats.update(err_max=err_max, dev_max=dev_max)


def wait_enter(prompt: str) -> threading.Event:
    ev = threading.Event()

    def ask():
        try:
            input(prompt)
        except EOFError:
            pass
        ev.set()

    threading.Thread(target=ask, daemon=True).start()
    return ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="PCAN_USBBUS1")
    ap.add_argument("--passes", default="60,35,20", help="heights of the fingertips above the table, mm, highest first")
    ap.add_argument("--y-max", type=float, default=0.24, help="half width of the catching area, m (planner: 0.24)")
    ap.add_argument("--speed", type=float, default=0.08, help="peak jaw speed along the line, m/s (max 0.14)")
    ap.add_argument("--rate", type=float, default=40.0, help="control rate, Hz")
    ap.add_argument("--contact", type=float, default=1.0, help="scale of the contact thresholds (1 = 1.2 A big joints, 0.8 A wrist)")
    ap.add_argument("--cap", type=float, default=1.0, help="scale of the current limits (1 = 6/18/14/5/4/4 A)")
    ap.add_argument("--dry-run", action="store_true", help="plan and check only; nothing is written or enabled")
    args = ap.parse_args()
    heights = [float(h) / 1000 for h in args.passes.split(",")]
    if heights != sorted(heights, reverse=True) or min(heights) < 0.015 or max(heights) > 0.12:
        print("--passes: heights in mm, highest first, between 15 and 120")
        return 2
    speed = float(np.clip(args.speed, 0.02, 0.14))
    y_max = float(np.clip(args.y_max, 0.05, 0.24))
    if not rc.MAP_PATH.exists():
        print("config/rig_map.json not found: run scripts/real/calibrate.py first")
        return 2
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    model = Model(rig_map.plane)
    caps = {m: v * args.cap for m, v in CUR_CAP.items()}
    contact = {m: v * args.contact for m, v in CONTACT_A.items()}

    rs = RobStrideBus(args.channel, movable=() if args.dry_run else ARM_IDS)
    arm = Arm(rs, caps)
    log, segs, code, configured = [], [], 1, False
    try:
        if not (arm.read() and arm.read() and not any(arm.miss.values())):
            print("Arm not answering on CAN (powered? cable?)")
            return 2
        if np.abs(arm.iq).max() > 0.3:
            print(f"joint currents read {arm.iq.round(2)} A: the motors must be disabled before this starts")
            return 2
        z_line = [float(rc.plane_z(rig_map.plane, model.env.gate_x, y)) * 1000 for y in (-y_max, 0.0, y_max)]
        print(f"map of {rig_map.time}: table on the intercept line is at z = {z_line[0]:.1f} / {z_line[1]:.1f} / {z_line[2]:.1f} mm "
              f"(right / centre / left) in the arm's frame")
        q_ready = model.jaw(0.0, heights[0], model.env.q_home)

        # ---- 1. the hand-held arm near the start pose
        print(f"\nHold the arm by hand over the MIDDLE of the intercept line (x = {model.env.gate_x:.2f} m), fingers forward and "
              f"tilted ~30 deg down, fingertips about {heights[0] * 1000:.0f} mm above the table.")
        ev = wait_enter("" if args.dry_run else "ENTER when it is there (the motors will then hold it) ... ")
        t_print = 0.0
        while True:
            if not arm.read():
                print("lost joint readings")
                return 2
            d = arm.pos - q_ready
            j = int(np.abs(d).argmax())
            tip = model.fingertip(arm.pos)
            ok = abs(d[j]) < START_TOL and model.clearance(arm.pos) > 0.005
            if _now() - t_print > 0.5:
                t_print = _now()
                print(f"  fingertip x {tip[0]:.2f} y {tip[1]:+.2f} z {tip[2] * 1000:4.0f} mm | furthest joint: {JOINT_NAMES[ARM_IDS[j]]} "
                      f"{math.degrees(d[j]):+4.0f} deg {'| OK' if ok else '| move closer'}      ", end="\r", flush=True)
            if args.dry_run or ev.is_set():
                print()
                if ok or args.dry_run:
                    break
                print(f"not close enough yet ({JOINT_NAMES[ARM_IDS[j]]} is {math.degrees(d[j]):+.0f} deg from the start pose, "
                      f"need < {math.degrees(START_TOL):.0f})")
                ev = wait_enter("ENTER when it is there ... ")
            _sleep(0.05)

        # ---- 2. plan from this pose, and check it in the model
        q_start = q_ready.copy() if args.dry_run and not ok else arm.pos.copy()
        segs = plan(model, q_start, heights, y_max, speed, args.rate)
        problems, info = check_plan(model, segs, args.rate)
        print(f"plan: {len(segs)} segments, {info['seconds']:.0f} s, jaw speed <= {speed:.2f} m/s, fastest joint "
              f"{info['max_joint_speed'].max():.2f} rad/s (firmware limit {JOINT_SPEED})")
        for name, c in info["clearance"].items():
            print(f"  {name:22s} lowest arm point {c * 1000:5.1f} mm above the table (model)")
        if problems:
            print("NOT RUNNING:\n  " + "\n  ".join(problems))
            return 2
        if args.dry_run:
            print("dry run: nothing written, nothing enabled.")
            return 0

        # ---- 3. engage and hold
        configured = True
        arm.configure()
        arm.engage()
        hold = arm.pos.copy()
        t_end = _now() + 1.5
        while _now() < t_end:
            if not arm.read():
                raise RuntimeError("lost readings while engaging")
            _sleep(0.02)
        sag = np.abs(arm.pos - hold)
        load = np.abs(arm.iq) / np.array([caps[m] for m in ARM_IDS])
        if sag.max() > 0.05 or load.max() > 0.75:
            raise Hold(f"cannot hold the arm safely here: sag {math.degrees(sag.max()):.1f} deg, currents {arm.iq.round(1)} A "
                       f"(limits {[caps[m] for m in ARM_IDS]}). Raise --cap only if the currents are steady and well below the motors' rating")
        print(f"HOLDING (currents {arm.iq.round(1)} A). Let go of the arm and step back. Ctrl+C = stop and hold.")
        _sleep(1.0)
        grip0 = rs.read_param_f32(GRIPPER_ID, P_MECHPOS, timeout=0.02)

        # ---- 4. the passes
        t_start, a = _now(), None
        for i, seg in enumerate(segs):
            seg.stats["tip"] = lambda: model.fingertip(arm.pos)
            print(f"-> {seg.name}", flush=True)
            run_segment(arm, seg, segs[seg.compare] if seg.compare is not None else None, args.rate, contact, grip0, log, t_start, a)
            dv = seg.stats["dev_max"]
            if a is not None and seg.gravity is not None:
                print(f"   no contact: currents within {dv.max():.2f} A of the expected (largest on "
                      f"{JOINT_NAMES[ARM_IDS[int(dv.argmax())]]}; thresholds {contact[1]:.1f} / {contact[4]:.1f} A); "
                      f"tracking error {math.degrees(seg.stats['err_max']):.1f} deg")
            else:
                print(f"   tracking error {math.degrees(seg.stats['err_max']):.1f} deg, currents now {arm.iq.round(1)} A")
            if a is None and seg.name.startswith("pass"):
                a = amps_per_nm(seg)
                print(f"   joint 2 draws {a:.2f} A per Nm of load" if a else
                      "   could not relate joint 2's current to its load: contact detection uses the raw currents")
        print(f"\nDONE: the jaw covered y = -{y_max:.2f} .. +{y_max:.2f} m at {', '.join(f'{h * 1000:.0f}' for h in heights)} mm "
              "above the table without touching anything.")
        code = 0
    except Hold as e:
        print(f"\nSTOPPED, HOLDING POSITION: {e}")
    except KeyboardInterrupt:
        print("\nCtrl+C: STOPPED, HOLDING POSITION")
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR, DISABLING THE MOTORS: {e}")
        if arm.enabled:
            arm.release()
    finally:
        try:
            if arm.enabled:
                arm.freeze()
                try:
                    wait_enter("SUPPORT THE ARM with your hand, then press ENTER to release it ... ").wait()
                except KeyboardInterrupt:
                    print("\nreleasing now")
                arm.release()
                print("motors disabled")
        finally:
            if arm.enabled:
                arm.release()
            if configured:
                arm.restore()
            rs.close()
    if log:
        out = LOG_DIR / f"catch_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "segment", "k"] + [f"{n}{j}" for n in ("target", "pos", "iq") for j in ARM_IDS])
            w.writerows(log)
        print(f"log: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
