"""First powered motion of the reBot B601-RS: move ONE joint a small angle and bring it back.

Default: joint 1 (base yaw) +15 deg and back to where it started, at 0.25 rad/s.

Only the joint being moved is powered; every other motor stays disabled and is watched. Gravity-
loaded joints (2, 3) are refused: disabling them on an abort would let the arm drop, and that needs a
gravity-compensated hold first. Allowed joints: 1 (yaw) and the wrist joints 4, 5, 6.

How the motor is driven: RobStride position mode (run_mode 1, the mode Seeed's POS_VEL uses). The
script streams a smooth position profile into loc_ref (0x7016); the motor's own firmware enforces the
speed limit limit_spd (0x7017) and the current limit limit_cur (0x7018). Both are set low before the
motor is enabled. All parameter writes are RAM-only (lost at power-off) and are restored at the end.

Aborts (motor disabled at once) on: tracking error > 0.15 rad for 0.4 s, current near the cap for
0.4 s (stalled / collision), any other joint moving > 3 deg, lost readings, Ctrl+C or any error.

Usage
    .venv/bin/python scripts/real/arm_motion_test.py --dry-run          # all checks, nothing written or enabled
    .venv/bin/python scripts/real/arm_motion_test.py                    # joint 1, +15 deg and back (asks first)
    .venv/bin/python scripts/real/arm_motion_test.py --joint 5 --delta -10
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject.robstride import ARM_IDS, GRIPPER_ID, JOINT_NAMES, P_MECHPOS, RobStrideBus  # noqa: E402

SCENE = ROOT / "assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml"
ALLOWED = (1, 4, 5, 6)
MAX_DELTA_DEG = 20.0
P_RUN_MODE, P_LOC_REF, P_LIMIT_SPD, P_LIMIT_CUR, P_IQF = 0x7005, 0x7016, 0x7017, 0x7018, 0x701A
RUN_MODE_POSITION = 1
CUR_CAP = {1: 10.0, 4: 4.0, 5: 4.0, 6: 4.0}        # A (RS06 ~0.63 Nm/A -> ~6 Nm; RS00 ~0.9 Nm/A -> ~3.5 Nm)
TRACK_ERR, WATCH_DEG, HOLD_S = 0.15, 3.0, 0.4


class Abort(RuntimeError):
    pass


def smooth_profile(q0: float, q1: float, speed: float, rate: float) -> list[float]:
    dist = abs(q1 - q0)
    T = max(math.pi * dist / (2 * speed), 0.5)
    n = max(2, int(T * rate))
    return [q0 + (q1 - q0) * 0.5 * (1 - math.cos(math.pi * k / (n - 1))) for k in range(n)]


def sim_check(j: int, q_now: np.ndarray, target: float) -> str | None:
    """Joint limits (with margin) and arm self/table collisions along the move, in the MuJoCo model."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[m.joint(f"joint{j}").id]
    if not (lo + 0.05 <= target <= hi - 0.05):
        return f"target {math.degrees(target):+.1f} deg is outside the model limits [{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}]"
    adr = [m.joint(f"joint{i}").qposadr[0] for i in ARM_IDS]
    arm_bodies = {m.body(n).id for n in ("link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")}
    table_or_base = {m.body(n).id for n in ("world", "base_link", "link1")}
    for s in np.linspace(0, 1, 30):
        q = q_now.copy()
        q[j - 1] = q_now[j - 1] + s * (target - q_now[j - 1])
        d.qpos[:] = m.qpos0
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if c.dist < -0.002 and ((b1 in arm_bodies and b2 in table_or_base) or (b2 in arm_bodies and b1 in table_or_base)):
                return f"model predicts contact {m.body(b1).name} / {m.body(b2).name} at {s:.0%} of the move"
    return None


def read_all(rs: RobStrideBus) -> dict[int, float | None]:
    return {m: rs.read_param_f32(m, P_MECHPOS, timeout=0.03) for m in (*ARM_IDS, GRIPPER_ID)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="PCAN_USBBUS1")
    ap.add_argument("--joint", type=int, default=1)
    ap.add_argument("--delta", type=float, default=15.0, help="degrees, |delta| <= 20")
    ap.add_argument("--speed", type=float, default=0.25, help="peak rad/s (max 0.5)")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true", help="run every check, write and enable nothing")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    j = args.joint
    if j not in ALLOWED:
        print(f"joint {j} not allowed here (allowed {ALLOWED}); joints 2/3 need a gravity-compensated hold first.")
        return 2
    if abs(args.delta) > MAX_DELTA_DEG:
        print(f"|delta| must be <= {MAX_DELTA_DEG} deg")
        return 2
    speed = min(max(args.speed, 0.05), 0.5)

    rs = RobStrideBus(args.channel, movable=() if args.dry_run else (j,))
    name = JOINT_NAMES[j]
    orig, start, end = {}, {}, {}
    enabled = wrote = False
    log = []
    try:
        start = read_all(rs)
        if any(v is None for v in start.values()):
            print("missing readings:", [JOINT_NAMES[m] for m, v in start.items() if v is None])
            return 2
        q_now = np.array([start[m] for m in ARM_IDS])
        q0 = start[j]
        target = q0 + math.radians(args.delta)
        problem = sim_check(j, q_now, target)
        if problem:
            print("model check failed:", problem)
            return 2
        for key, idx in (("run_mode", P_RUN_MODE), ("limit_spd", P_LIMIT_SPD), ("limit_cur", P_LIMIT_CUR), ("loc_ref", P_LOC_REF)):
            orig[key] = rs.read_param_u8(j, idx) if key == "run_mode" else rs.read_param_f32(j, idx)
        if None in orig.values():
            print("could not read the motor parameters:", orig)
            return 2
        iq0 = rs.read_param_f32(j, P_IQF)
        if iq0 is None or abs(iq0) > 0.2:
            print(f"{name} current reads {iq0}: expected ~0 A (motor should be disabled). Not continuing.")
            return 2

        print(f"{name} now {math.degrees(q0):+.2f} deg -> {math.degrees(target):+.2f} deg -> back to {math.degrees(q0):+.2f} deg")
        print(f"speed <= {speed} rad/s (firmware limit {speed * 1.3:.2f}), current cap {CUR_CAP[j]} A; "
              f"other joints stay unpowered. Model check: within limits, no collision.")
        print(f"motor params now: {orig} (restored at the end)")
        if args.dry_run:
            print("dry run: nothing written, nothing enabled.")
            return 0
        if not args.yes and input("Workspace clear? Type 'yes' to move: ").strip().lower() != "yes":
            print("cancelled")
            return 1

        # ---------------- configure while disabled ----------------
        wrote = True
        rs.stop(j)
        time.sleep(0.02)
        rs.write_param_u8(j, P_RUN_MODE, RUN_MODE_POSITION)
        time.sleep(0.02)
        if rs.read_param_u8(j, P_RUN_MODE) != RUN_MODE_POSITION:
            raise Abort("run_mode did not switch to position mode")
        for idx, val in ((P_LIMIT_SPD, speed * 1.3), (P_LIMIT_CUR, CUR_CAP[j]), (P_LOC_REF, q0)):
            rs.write_param_f32(j, idx, val)
            time.sleep(0.01)
            back = rs.read_param_f32(j, idx)
            if back is None or abs(back - val) > 1e-3:
                raise Abort(f"param 0x{idx:04X} readback {back} != {val}")

        # ---------------- move ----------------
        rs.enable(j)
        enabled = True
        path = ([q0] * int(0.5 * args.rate) + smooth_profile(q0, target, speed, args.rate) + [target] * int(0.8 * args.rate)
                + smooth_profile(target, q0, speed, args.rate) + [q0] * int(1.0 * args.rate))
        dt = 1.0 / args.rate
        err_since = cur_since = None
        t_start = time.monotonic()
        for k, q in enumerate(path):
            t0 = time.monotonic()
            rs.write_param_f32(j, P_LOC_REF, q)
            pos = rs.read_param_f32(j, P_MECHPOS, timeout=0.01)
            iq = rs.read_param_f32(j, P_IQF, timeout=0.01)
            if pos is None or iq is None:
                raise Abort("lost readings from the moving joint")
            err = q - pos
            err_since = (err_since or t0) if abs(err) > TRACK_ERR else None
            cur_since = (cur_since or t0) if abs(iq) > 0.9 * CUR_CAP[j] else None
            if err_since and t0 - err_since > HOLD_S:
                raise Abort(f"tracking error {math.degrees(err):+.1f} deg for > {HOLD_S} s")
            if cur_since and t0 - cur_since > HOLD_S:
                raise Abort(f"current {iq:+.2f} A at the cap for > {HOLD_S} s (blocked?)")
            if k % 5 == 0:                                   # other joints at 10 Hz
                for m in (*ARM_IDS, GRIPPER_ID):
                    if m == j:
                        continue
                    v = rs.read_param_f32(m, P_MECHPOS, timeout=0.01)
                    if v is not None and abs(math.degrees(v - start[m])) > WATCH_DEG:
                        raise Abort(f"unpowered {JOINT_NAMES[m]} moved {math.degrees(v - start[m]):+.1f} deg")
            log.append((round(t0 - t_start, 4), round(q, 5), round(pos, 5), round(iq, 3)))
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
        print("motion done")
    except Abort as e:
        print("ABORT:", e)
    except KeyboardInterrupt:
        print("Ctrl+C")
    finally:
        if enabled:
            for _ in range(3):
                rs.stop(j)
                time.sleep(0.01)
            print(f"{name} disabled")
        if wrote:
            try:
                time.sleep(0.05)
                rs.write_param_f32(j, P_LIMIT_SPD, orig["limit_spd"])
                rs.write_param_f32(j, P_LIMIT_CUR, orig["limit_cur"])
                rs.write_param_u8(j, P_RUN_MODE, orig["run_mode"])
                print("motor params restored")
            except Exception as e:  # noqa: BLE001
                print("could not restore params (they reset at power-off anyway):", e)
        try:
            end = read_all(rs)
        finally:
            rs.close()

    if log:
        out = ROOT / "outputs" / "hardware" / f"arm_motion_{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "target", "pos", "iq"])
            w.writerows(log)
        errs = [abs(r[1] - r[2]) for r in log]
        print(f"log: {out}\n  peak reached {math.degrees(max(abs(r[2] - log[0][1]) for r in log)):.2f} deg from start, "
              f"max tracking error {math.degrees(max(errs)):.2f} deg, max |current| {max(abs(r[3]) for r in log):.2f} A")
    if end and start:
        print("final vs start (deg): " + "  ".join(
            f"{JOINT_NAMES[m]} {math.degrees(end[m] - start[m]):+.2f}" for m in ARM_IDS if end.get(m) is not None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
