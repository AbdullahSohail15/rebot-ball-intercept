"""Read-only check that the real reBot B601-RS joints match the MuJoCo model (direction, zero, range).

NOTHING IS POWERED. The script only sends parameter reads (RobStride type 17); the bus wrapper refuses
enable/MIT/write commands to every motor (movable=()). You move the arm by hand.

For each joint, a window shows the sim arm BEFORE and AFTER a positive move of that joint (in the
sim's convention), plus a live render of the sim posed at the real joint readings. You move the real
joint the way the picture shows; the script measures which way the motor angle went:
    + change  -> real direction matches the model (sign +1)
    - change  -> reversed (sign -1)
It also flags cross-talk (another joint changed: wrong CAN ID / wiring) and, with --ranges, records
the range you sweep and compares it to the model's joint limits. The first step is a zero check: pose
the arm like the q = 0 picture and the readings are the zero offsets.

Usage (run in your own terminal; it is interactive):
    .venv/bin/python scripts/real/joint_check.py              # zero check + direction check, joints 1-6
    .venv/bin/python scripts/real/joint_check.py --ranges     # also sweep each joint through its range
    .venv/bin/python scripts/real/joint_check.py --gripper    # also check the gripper (open the jaws by hand)
    .venv/bin/python scripts/real/joint_check.py --live       # just mirror the real arm in the sim, no checks
    .venv/bin/python scripts/real/joint_check.py --joints 2 3 # only some joints

Keys in the window: SPACE accept step, r re-take the baseline, s skip, q quit.
Report: outputs/hardware/joint_check_<time>.json

Careful: the arm is unpowered. Joints 2 and 3 fall under gravity, so support the arm when you lift
it and lower it back gently. Never force a joint against its hard stop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject.robstride import ARM_IDS, GRIPPER_ID, JOINT_NAMES, P_MECHPOS, RobStrideBus  # noqa: E402

SCENE = ROOT / "assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml"
GRIP_MOTOR_OPEN = 5.0          # motor rad at fully open (Seeed config)
GRIP_SIM_OPEN = 0.05           # joint7 metres at fully open (sim)
PLUS_STEP = 0.5                # rad, size of the "AFTER" move in the pictures
MIN_MOVE_DEG = 8.0             # a step needs at least this much movement to count
CROSSTALK_DEG = 5.0            # another joint changing more than this is flagged
PANEL_W, PANEL_H = 420, 315


# ------------------------------------------------------------------------------------- sim side
class SimArm:
    def __init__(self) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
        self.grip_adr = [self.m.joint(n).qposadr[0] for n in ("joint7", "joint_left", "joint_right")]
        self.ranges = {i: tuple(self.m.jnt_range[self.m.joint(f"joint{i}").id]) for i in range(1, 7)}
        self.r = mujoco.Renderer(self.m, PANEL_H, PANEL_W)
        self.cams = []
        for az, el, dist, look in ((-125, -20, 0.72, (0.12, 0, 0.16)), (-90, -5, 0.85, (0.12, 0, 0.18))):
            c = mujoco.MjvCamera()
            c.lookat[:], c.azimuth, c.elevation, c.distance = look, az, el, dist
            self.cams.append(c)
        self.tcp = self.m.site("tcp").id

    def pose(self, q_arm, grip_motor: float | None = None) -> None:
        self.d.qpos[:] = self.m.qpos0
        for a, q in zip(self.qadr, q_arm):
            self.d.qpos[a] = q
        g = 0.0 if grip_motor is None else GRIP_SIM_OPEN * float(np.clip(grip_motor / GRIP_MOTOR_OPEN, 0, 1))
        for a in self.grip_adr:
            self.d.qpos[a] = g
        mujoco.mj_kinematics(self.m, self.d)

    def render(self, q_arm, grip_motor=None, cam: int = 0, label: str = "") -> np.ndarray:
        self.pose(q_arm, grip_motor)
        self.r.update_scene(self.d, self.cams[cam])
        img = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        if label:
            put(img, label, (10, 26), 0.7, (255, 255, 255), 2)
        return img

    def tcp_pos(self, q_arm) -> np.ndarray:
        self.pose(q_arm)
        return self.d.site_xpos[self.tcp].copy()


# Point that is easiest to watch for each joint: the elbow for the shoulder, the wrist for the elbow,
# the gripper tip for the wrist joints.
WATCH = {1: ("tcp", "the gripper tip"), 2: ("link3", "the elbow"), 3: ("link5", "the wrist"),
         4: ("tcp", "the gripper tip"), 5: ("tcp", "the gripper tip"), 6: ("tcp", "the gripper tip")}


def describe_plus(sim: SimArm, j: int, q_ref: np.ndarray) -> str:
    """Words for what a + move of joint j does, in the table frame (x toward the table, y left, z up)."""
    obj, label = WATCH[j]

    def point(q):
        sim.pose(q)
        return (sim.d.site_xpos[sim.tcp] if obj == "tcp" else sim.d.body(obj).xpos).copy()

    q1 = q_ref.copy()
    q1[j - 1] += PLUS_STEP
    dp = point(q1) - point(q_ref)
    words = []
    for k, (pos, neg) in enumerate((("toward the table", "back toward the base"),
                                    ("to the arm's left", "to the arm's right"), ("up", "down"))):
        if abs(dp[k]) > 0.015:
            words.append(pos if dp[k] > 0 else neg)
    if not words:
        return "the gripper rolls in place: follow the picture"
    return f"{label} moves " + " and ".join(words)


def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------------------------------- real side
def read_all(rs: RobStrideBus) -> dict[int, float | None]:
    return {mid: rs.read_param_f32(mid, P_MECHPOS, timeout=0.03) for mid in (*ARM_IDS, GRIPPER_ID)}


def q_arm_of(read: dict[int, float | None]) -> np.ndarray:
    return np.array([read[m] if read[m] is not None else 0.0 for m in ARM_IDS])


# ----------------------------------------------------------------------------------------- UI
WIN = "joint check (SPACE accept, r rebaseline, s skip, q quit)"


def compose(top: list[np.ndarray], live: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    row1 = np.hstack(top)
    info = np.full((PANEL_H, row1.shape[1] - live.shape[1], 3), 30, np.uint8)
    for k, (text, color) in enumerate(lines):
        put(info, text, (14, 32 + 30 * k), 0.62, color, 1)
    return np.vstack([row1, np.hstack([live, info])])


def run_step(rs, sim, title, pictures, lines_fn, accept_fn, grip=False):
    """Generic interactive step. Returns ('ok'|'skip'|'quit', baseline, last, history)."""
    base = read_all(rs)
    hist = []
    msg = ""
    while True:
        cur = read_all(rs)
        hist.append(cur)
        live = sim.render(q_arm_of(cur), cur[GRIPPER_ID] if grip else None, 0, "LIVE: sim at real readings")
        lines = [(title, (0, 220, 255))] + lines_fn(base, cur, hist) + ([(msg, (0, 0, 255))] if msg else [])
        cv2.imshow(WIN, compose(pictures, live, lines))
        k = cv2.waitKey(30) & 0xFF
        try:
            closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = True
        if k == ord("q") or closed:
            return "quit", base, cur, hist
        if k == ord("s"):
            return "skip", base, cur, hist
        if k == ord("r"):
            base, hist, msg = read_all(rs), [], ""
        if k == ord(" "):
            ok, msg = accept_fn(base, cur, hist)
            if ok:
                return "ok", base, cur, hist


def deg(x):
    return math.degrees(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="PCAN_USBBUS1")
    ap.add_argument("--joints", type=int, nargs="+", default=list(ARM_IDS))
    ap.add_argument("--ranges", action="store_true", help="also sweep each joint through its range")
    ap.add_argument("--gripper", action="store_true", help="also check the gripper direction")
    ap.add_argument("--live", action="store_true", help="only mirror the real arm in the sim")
    ap.add_argument("--no-zero", action="store_true", help="skip the zero-pose step")
    args = ap.parse_args()

    rs = RobStrideBus(args.channel, movable=())      # read-only: nothing may be enabled
    sim = SimArm()
    first = read_all(rs)
    missing = [JOINT_NAMES[m] for m, v in first.items() if v is None]
    if len(missing) == 7:
        print("No motor answered. Is the arm powered and the CAN cable connected?")
        return 2
    if missing:
        print("No answer from:", ", ".join(missing))
    print("Read-only: no motor is powered. Support the arm when moving joints 2/3; never force a hard stop.")
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    if args.live:
        while (cv2.waitKey(30) & 0xFF) != ord("q"):
            cur = read_all(rs)
            q = q_arm_of(cur)
            views = [sim.render(q, cur[GRIPPER_ID], c, "sim at real readings") for c in (0, 1)]
            txt = "  ".join(f"j{m}:{'--' if cur[m] is None else f'{deg(cur[m]):+.1f}'}" for m in ARM_IDS)
            g = cur[GRIPPER_ID]
            img = np.hstack(views)
            put(img, txt + (f"  grip:{g:.2f}rad" if g is not None else ""), (10, PANEL_H - 12), 0.5)
            cv2.imshow(WIN, img)
        rs.close()
        return 0

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "assumed_mapping": "q_sim = q_motor (Seeed default)",
              "joints": {}}
    quit_all = False

    # ---------------------------------------------------------------- zero check
    if not args.no_zero:
        q0 = np.zeros(6)
        pics = [sim.render(q0, None, 0, "TARGET: q = 0"), sim.render(q0, None, 1, "TARGET: q = 0 (side)")]

        def zero_lines(base, cur, hist):
            out = [("Pose the arm like the TARGET pictures:", (255, 255, 255)),
                   ("folded, pointing straight down the table,", (255, 255, 255)),
                   ("gripper level, jaws opening sideways.", (255, 255, 255))]
            out += [(f"{JOINT_NAMES[m]}: {deg(cur[m]):+7.2f} deg" if cur[m] is not None else f"{JOINT_NAMES[m]}: --",
                     (0, 255, 0) if cur[m] is not None and abs(deg(cur[m])) < 5 else (0, 200, 255)) for m in ARM_IDS]
            return out

        st, base, cur, _ = run_step(rs, sim, "ZERO CHECK (SPACE when posed)", pics, zero_lines,
                                    lambda b, c, h: (True, ""))
        if st == "quit":
            quit_all = True
        elif st == "ok":
            report["zero_offsets_deg"] = {JOINT_NAMES[m]: None if cur[m] is None else round(deg(cur[m]), 2)
                                          for m in ARM_IDS}

    # ---------------------------------------------------------------- per-joint direction (+ range)
    for j in args.joints:
        if quit_all:
            break
        if j not in ARM_IDS:
            continue
        name = JOINT_NAMES[j]
        q_ref = q_arm_of(read_all(rs))
        q_plus = q_ref.copy()
        q_plus[j - 1] += PLUS_STEP
        lo, hi = sim.ranges[j]
        words = describe_plus(sim, j, q_ref)
        pics = [sim.render(q_ref, None, 0, f"{name}: BEFORE"), sim.render(q_plus, None, 0, f"{name}: AFTER (+)")]
        entry = {"sim_range_deg": [round(deg(lo), 1), round(deg(hi), 1)], "plus_move": words}

        def dir_lines(base, cur, hist, j=j, words=words):
            if cur[j] is None or base[j] is None:
                return [("no reading from this joint", (0, 0, 255))]
            dj = deg(cur[j] - base[j])
            others = {m: deg(cur[m] - base[m]) for m in ARM_IDS if m != j and cur[m] is not None and base[m] is not None}
            worst = max(others.items(), key=lambda kv: abs(kv[1])) if others else (0, 0.0)
            col = (0, 255, 0) if dj > MIN_MOVE_DEG else (0, 0, 255) if dj < -MIN_MOVE_DEG else (255, 255, 255)
            return [(f"Move {name} by hand ~20-30 deg as in AFTER:", (255, 255, 255)),
                    (f"  {words}", (255, 255, 255)),
                    (f"{name} change: {dj:+7.2f} deg", col),
                    (f"largest other change: j{worst[0]} {worst[1]:+.1f} deg",
                     (0, 0, 255) if abs(worst[1]) > CROSSTALK_DEG else (180, 180, 180)),
                    ("SPACE when held there (then move it back)", (180, 180, 180))]

        def dir_accept(base, cur, hist, j=j):
            if cur[j] is None or base[j] is None:
                return False, "no reading from this joint"
            if abs(deg(cur[j] - base[j])) < MIN_MOVE_DEG:
                return False, f"move it at least {MIN_MOVE_DEG:.0f} deg first"
            return True, ""

        st, base, cur, hist = run_step(rs, sim, f"DIRECTION: {name} (motor ID {j})", pics, dir_lines, dir_accept)
        if st == "quit":
            quit_all = True
        if st != "ok":
            entry["status"] = "skipped"
            report["joints"][name] = entry
            continue
        dj = deg(cur[j] - base[j])
        cross = {JOINT_NAMES[m]: round(max(abs(deg(h[m] - base[m])) for h in hist if h[m] is not None), 1)
                 for m in ARM_IDS if m != j and base[m] is not None}
        cross = {k: v for k, v in cross.items() if v > CROSSTALK_DEG}
        entry.update({"moved_deg": round(dj, 1), "sign": 1 if dj > 0 else -1, "crosstalk_deg": cross})

        if args.ranges and not quit_all:
            def range_lines(base, cur, hist, j=j, lo=lo, hi=hi):
                vals = [deg(h[j]) for h in hist if h[j] is not None]
                if not vals:
                    return [("no reading", (0, 0, 255))]
                return [(f"Gently sweep {name} through its full range,", (255, 255, 255)),
                        ("stopping BEFORE the hard stops. SPACE when done.", (255, 255, 255)),
                        (f"now {vals[-1]:+7.1f}   min {min(vals):+7.1f}   max {max(vals):+7.1f} deg", (0, 255, 0)),
                        (f"sim limits:  {deg(lo):+7.1f} .. {deg(hi):+7.1f} deg", (180, 180, 180))]

            st, base, cur, hist = run_step(rs, sim, f"RANGE: {name}", pics, range_lines,
                                           lambda b, c, h: (True, ""))
            if st == "quit":
                quit_all = True
            vals = [h[j] for h in hist if h[j] is not None]
            if st == "ok" and vals:
                rmin, rmax = deg(min(vals)), deg(max(vals))
                entry["swept_deg"] = [round(rmin, 1), round(rmax, 1)]
                entry["outside_sim_limits"] = rmin < deg(lo) - 5 or rmax > deg(hi) + 5
        entry["status"] = "ok"
        report["joints"][name] = entry

    # ---------------------------------------------------------------- gripper
    if args.gripper and not quit_all:
        q_ref = q_arm_of(read_all(rs))
        pics = [sim.render(q_ref, 0.3, 0, "gripper: BEFORE"), sim.render(q_ref, 3.0, 0, "gripper: AFTER (open)")]

        def grip_lines(base, cur, hist):
            if cur[GRIPPER_ID] is None:
                return [("no gripper reading", (0, 0, 255))]
            d = cur[GRIPPER_ID] - base[GRIPPER_ID]
            return [("Open the jaws by hand (slowly).", (255, 255, 255)),
                    (f"gripper motor: {cur[GRIPPER_ID]:+.3f} rad  change {d:+.3f} rad", (0, 255, 0) if d > 0.3 else (255, 255, 255)),
                    ("expected: opening = increasing (0 closed .. 5 open)", (180, 180, 180))]

        def grip_accept(base, cur, hist):
            if cur[GRIPPER_ID] is None or abs(cur[GRIPPER_ID] - base[GRIPPER_ID]) < 0.3:
                return False, "open it a bit more first"
            return True, ""

        st, base, cur, hist = run_step(rs, sim, "DIRECTION: gripper (ID 7)", pics, grip_lines, grip_accept, grip=True)
        if st == "ok":
            d = cur[GRIPPER_ID] - base[GRIPPER_ID]
            report["joints"]["gripper"] = {"moved_rad": round(d, 3), "sign": 1 if d > 0 else -1, "status": "ok",
                                           "max_seen_rad": round(max(h[GRIPPER_ID] for h in hist if h[GRIPPER_ID] is not None), 3)}

    cv2.destroyAllWindows()
    rs.close()

    # ---------------------------------------------------------------- report
    print("\n=== joint check ===")
    if "zero_offsets_deg" in report:
        print("zero offsets (pose matched by eye, so +-3 deg is normal):")
        for k, v in report["zero_offsets_deg"].items():
            print(f"  {k}: {v:+.2f} deg" if v is not None else f"  {k}: --")
    problems = []
    for name, e in report["joints"].items():
        if e.get("status") != "ok":
            print(f"{name}: skipped")
            continue
        line = f"{name}: sign {e['sign']:+d}"
        if "moved_deg" in e:
            line += f" (moved {e['moved_deg']:+.1f} deg)"
        if e.get("crosstalk_deg"):
            line += f"  CROSS-TALK {e['crosstalk_deg']}"
            problems.append(f"{name} cross-talk")
        if "swept_deg" in e:
            line += f"  swept {e['swept_deg']} vs sim {e['sim_range_deg']}"
            if e["outside_sim_limits"]:
                line += "  OUTSIDE SIM LIMITS"
                problems.append(f"{name} range")
        if e["sign"] < 0:
            problems.append(f"{name} reversed")
        print(line)
    for k, v in report.get("zero_offsets_deg", {}).items():
        if v is not None and abs(v) > 5:
            problems.append(f"{k} zero offset {v:+.1f} deg")
    report["problems"] = problems
    wanted = [JOINT_NAMES[j] for j in args.joints if j in ARM_IDS] + (["gripper"] if args.gripper else [])
    checked = [n for n in wanted if report["joints"].get(n, {}).get("status") == "ok"]
    unchecked = [n for n in wanted if n not in checked]
    report["checked"], report["unchecked"] = checked, unchecked
    if not args.no_zero and "zero_offsets_deg" not in report:
        unchecked = ["zero pose"] + unchecked
    if problems:
        verdict = "PROBLEMS: " + "; ".join(problems)
    elif not checked:
        verdict = "NOTHING CHECKED (quit or window closed before any joint was tested)"
    elif unchecked:
        verdict = f"no problems in {', '.join(checked)}; NOT checked: {', '.join(unchecked)}"
    else:
        verdict = "all checked joints match the model"
    if unchecked and checked:
        print("not checked:", ", ".join(unchecked))
    print("RESULT:", verdict)
    out = ROOT / "outputs" / "hardware" / f"joint_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
