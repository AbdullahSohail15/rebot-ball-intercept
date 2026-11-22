"""Safe hardware check for the reBot B601-RS: read all joint states, then open/close ONLY the gripper.

The six arm motors (CAN IDs 1-6) are only ever *read* (RobStride type-17 parameter reads). They are
never enabled, so the arm stays exactly where it is, unpowered. Only the gripper motor (ID 7) is
enabled, in MIT mode with deliberately low gains, and moved along a slow, smooth profile.

Transport: PEAK PCAN-USB at 1 Mbit/s through python-can + the PCBUSB library (macOS). Install the
library once by copying libPCBUSB.dylib into ~/lib (see README). No sudo needed.

Usage
    .venv/bin/python scripts/real/gripper_test.py                 # read-only: print joint states for 3 s
    .venv/bin/python scripts/real/gripper_test.py --watch 0       # read-only, until Ctrl+C
    .venv/bin/python scripts/real/gripper_test.py --move          # open to 2.5 rad, close to 0.2 rad (asks first)
    .venv/bin/python scripts/real/gripper_test.py --move --open 5.0 --cycles 2

Safety layers in --move
    - aborts before enabling anything if the gripper is not at a plausible position (0..5 rad travel,
      zero must already be calibrated), if any motor reports a fault, or if the feedback-frame position
      disagrees with the mechPos parameter (catches a wrong scaling / wrong motor);
    - targets are clamped to [0.1, 5.0] rad; speed limited to --speed (default 1.0 rad/s);
    - low MIT gains (kp 5, kd 0.5 vs Seeed's 50 / 4); zero torque feed-forward;
    - aborts if tracking error > 0.8 rad, |torque| > --max-torque, feedback is lost, or a fault appears;
    - the gripper is disabled (type-4 stop) on exit, on any error and on Ctrl+C;
    - the arm joints are re-read at the end to confirm they did not move.

Protocol details: eaiproject/robstride.py
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eaiproject.robstride import (  # noqa: E402
    ARM_IDS, GRIPPER_ID, JOINT_NAMES, GRIP_CLOSE, GRIP_OPEN, TARGET_MIN, TARGET_MAX, P_RUN_MODE,
    RobStrideBus, read_joint_states, print_states,
)


def smooth_profile(q0: float, q1: float, speed: float, rate: float) -> list[float]:
    """Cosine (zero velocity at both ends) profile whose peak speed is `speed`."""
    dist = abs(q1 - q0)
    if dist < 1e-4:
        return [q1]
    T = max(math.pi * dist / (2 * speed), 0.3)
    n = max(2, int(T * rate))
    return [q0 + (q1 - q0) * 0.5 * (1 - math.cos(math.pi * k / (n - 1))) for k in range(n)]


class Abort(RuntimeError):
    pass


def move_gripper(rs: RobStrideBus, path: list[float], args, log: list) -> None:
    dt = 1.0 / args.rate
    err_since = None
    for q in path:
        t0 = time.monotonic()
        rs.mit(GRIPPER_ID, q, args.kp, args.kd)
        rs.drain(0.004)
        fb = rs.last_fb.get(GRIPPER_ID)
        if fb is None or t0 - fb.t > 0.2:
            raise Abort("lost gripper feedback for > 0.2 s")
        if fb.faults:
            raise Abort(f"gripper fault bits 0x{fb.faults:02x}")
        if abs(fb.tau) > args.max_torque:
            raise Abort(f"gripper torque {fb.tau:+.2f} Nm exceeds {args.max_torque} Nm (blocked?)")
        err = q - fb.pos
        if abs(err) > 0.8:
            err_since = err_since or t0
            if t0 - err_since > 0.3:
                raise Abort(f"tracking error {err:+.2f} rad for > 0.3 s (blocked or wrong scaling)")
        else:
            err_since = None
        log.append((round(t0, 4), round(q, 4), round(fb.pos, 4), round(fb.vel, 4), round(fb.tau, 4), fb.temp_c))
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))


def hold(rs: RobStrideBus, q: float, seconds: float, args, log: list) -> None:
    move_gripper(rs, [q] * max(1, int(seconds * args.rate)), args, log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="PCAN_USBBUS1")
    ap.add_argument("--watch", type=float, default=3.0, help="read-only: seconds to print states (0 = until Ctrl+C)")
    ap.add_argument("--move", action="store_true", help="enable the gripper and open/close it")
    ap.add_argument("--open", type=float, default=2.5, help="open target, motor rad (max 5.0)")
    ap.add_argument("--close", type=float, default=0.2, help="close target, motor rad (min 0.1)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="peak gripper speed, rad/s (max 3)")
    ap.add_argument("--kp", type=float, default=5.0)
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=125.0)
    ap.add_argument("--max-torque", type=float, default=2.0, help="abort above this |torque|, Nm")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()
    args.open = min(max(args.open, TARGET_MIN), TARGET_MAX)
    args.close = min(max(args.close, TARGET_MIN), TARGET_MAX)
    args.speed = min(max(args.speed, 0.1), 3.0)
    args.kp = min(max(args.kp, 0.0), 20.0)
    args.kd = min(max(args.kd, 0.0), 2.0)

    rs = RobStrideBus(args.channel)
    print(f"CAN open on {args.channel} @ 1 Mbit/s")
    try:
        before = read_joint_states(rs)
        print_states(before)
        missing = [JOINT_NAMES[m] for m, s in before.items() if s["pos"] is None]
        if len(missing) == 7:
            print("\nNo motor answered. Is the arm's power supply on, and is the CAN cable plugged into the arm?")
            return 2
        if missing:
            print(f"\nNo answer from: {', '.join(missing)}")

        if not args.move:
            t_end = None if args.watch <= 0 else time.monotonic() + args.watch
            while t_end is None or time.monotonic() < t_end:
                time.sleep(0.5)
                print()
                print_states(read_joint_states(rs))
            return 0

        # ---------------- pre-flight checks (nothing enabled yet) ----------------
        g = before[GRIPPER_ID]["pos"]
        if g is None:
            print("Gripper (ID 7) did not answer; not moving.")
            return 2
        if not (GRIP_CLOSE - 0.3 <= g <= GRIP_OPEN + 0.3):
            print(f"Gripper reads {g:.3f} rad, outside its 0..5 rad travel. Its zero is probably not set; "
                  "not moving. (Zeroing is a separate, deliberate step.)")
            return 2
        rs.stop(GRIPPER_ID)                       # disabled; also returns a feedback frame
        rs.drain(0.05)
        fb = rs.last_fb.get(GRIPPER_ID)
        if fb is None:
            print("No feedback frame from the gripper; not moving.")
            return 2
        if fb.faults:
            print(f"Gripper reports fault bits 0x{fb.faults:02x}; not moving.")
            return 2
        if abs(fb.pos - g) > 0.1:
            print(f"Feedback position {fb.pos:.3f} disagrees with mechPos {g:.3f}; scaling/ID mismatch, not moving.")
            return 2

        print(f"\nPlan: arm joints 1-6 stay UNPOWERED. Gripper: {g:.2f} -> open {args.open:.2f} -> close "
              f"{args.close:.2f} rad, x{args.cycles}, <= {args.speed} rad/s, kp {args.kp}, kd {args.kd}.")
        print("Keep fingers and objects clear of the gripper jaws. Ctrl+C stops and disables it at any time.")
        if not args.yes and input("Type 'yes' to proceed: ").strip().lower() != "yes":
            print("Cancelled.")
            return 1

        log: list = []
        try:
            rs.write_param_u8(GRIPPER_ID, P_RUN_MODE, 0)   # 0 = MIT (operation) mode; only while disabled
            time.sleep(0.02)
            mode = rs.read_param_u8(GRIPPER_ID, P_RUN_MODE)
            if mode != 0:
                raise Abort(f"could not set gripper run_mode to MIT (reads {mode})")
            rs.mit(GRIPPER_ID, g, args.kp, args.kd)       # hold target queued before enabling
            rs.enable(GRIPPER_ID)
            hold(rs, g, 0.5, args, log)
            q = g
            for c in range(args.cycles):
                print(f"cycle {c + 1}: opening to {args.open:.2f}")
                move_gripper(rs, smooth_profile(q, args.open, args.speed, args.rate), args, log)
                hold(rs, args.open, 0.8, args, log)
                print(f"cycle {c + 1}: closing to {args.close:.2f}")
                move_gripper(rs, smooth_profile(args.open, args.close, args.speed, args.rate), args, log)
                hold(rs, args.close, 0.8, args, log)
                q = args.close
            print("done")
        except Abort as e:
            print(f"ABORT: {e}")
        except KeyboardInterrupt:
            print("Ctrl+C")
        finally:
            for _ in range(3):
                rs.stop(GRIPPER_ID)
                time.sleep(0.01)
            print("gripper disabled")

        if log:
            out = Path(__file__).resolve().parents[2] / "outputs" / "hardware" / f"gripper_test_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "target", "pos", "vel", "torque", "temp_c"])
                w.writerows(log)
            errs = [abs(r[1] - r[2]) for r in log]
            print(f"log: {out}  (max tracking error {max(errs):.3f} rad, max |torque| "
                  f"{max(abs(r[4]) for r in log):.2f} Nm)")

        after = read_joint_states(rs)
        print("\nafter:")
        print_states(after)
        moved = [f"{JOINT_NAMES[m]} {math.degrees(after[m]['pos'] - before[m]['pos']):+.2f} deg"
                 for m in ARM_IDS if before[m]["pos"] is not None and after[m]["pos"] is not None
                 and abs(after[m]["pos"] - before[m]["pos"]) > math.radians(0.5)]
        print("arm joints unchanged" if not moved else f"WARNING arm joints changed: {', '.join(moved)}")
        return 0
    finally:
        rs.close()


if __name__ == "__main__":
    sys.exit(main())
