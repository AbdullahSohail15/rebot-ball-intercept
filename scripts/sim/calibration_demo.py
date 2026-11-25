"""Record a demonstration video of the calibration and the tracking, on the simulated rig.

The REAL scripts run (scripts/real/calibrate.py, check_calibration.py, real_tracking.py) and what
their windows show is recorded. Only the hardware is simulated (tests/sim_rig.py): a camera that
renders the table with its ArUco markers, the ball and the gripper, and an unpowered arm whose
readings include a model error of a few mm. Keys are pressed by a script; the caption under the
picture says what a person would be doing.

    .venv/bin/python scripts/sim/calibration_demo.py            # -> docs/media/calibration_demo.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sim_rig  # noqa: E402
from eaiproject import real_intercept  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402

W, H, BAR, FPS, DT = 1280, 720, 96, 20, 0.05
BIAS = np.array([0.002, -0.001, 0.009])           # the arm model's error: what it reports minus where it is
VIEWS = [(1.45, 0.25, 0.75), (1.40, -0.35, 0.60), (1.30, 0.05, 1.00), (1.50, 0.55, 0.85), (1.25, -0.55, 0.90),
         (1.60, -0.05, 0.65)]
REST = np.array([0.22, 0.0, 0.16])                # where the hand keeps the fingertips between touches
KEY_NAMES = {" ": "SPACE", "\r": "ENTER"}


class Director:
    """Stands in for the OpenCV window and keyboard: records every shown frame, plays scripted keys."""

    def __init__(self):
        self.frames: list[tuple[float, np.ndarray]] = []
        self.steps: list = []
        self.caption, self.script = "", ""
        self.key, self.key_until, self.t_last = "", 0.0, 0.0

    def card(self, lines: list[str], seconds: float) -> None:
        img = np.full((H + BAR, W, 3), 24, np.uint8)
        for i, (text, scale) in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(img, text, ((W - tw) // 2, 300 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235),
                        2 if scale > 0.9 else 1, cv2.LINE_AA)
        t0 = self.frames[-1][0] if self.frames else time.monotonic()
        self.frames += [(t0 + k / FPS, img) for k in range(1, int(seconds * FPS) + 1)]
        self.shift = time.monotonic() - self.frames[-1][0]          # cards take no real time

    def imshow(self, win, img) -> None:
        k = min(W / img.shape[1], H / img.shape[0])
        pic = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        out = np.full((H + BAR, W, 3), 24, np.uint8)
        x0, y0 = (W - pic.shape[1]) // 2, (H - pic.shape[0]) // 2
        out[y0:y0 + pic.shape[0], x0:x0 + pic.shape[1]] = pic
        cv2.putText(out, self.script, (16, H + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(self.caption.split("\n")[:2]):
            cv2.putText(out, line, (16, H + 58 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 1, cv2.LINE_AA)
        if time.monotonic() < self.key_until:
            (tw, _), _ = cv2.getTextSize(self.key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(out, (W - tw - 44, H + 12), (W - 16, H + 54), (0, 200, 255), 2)
            cv2.putText(out, self.key, (W - tw - 30, H + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        self.frames.append((time.monotonic() - getattr(self, "shift", 0.0), out))

    def waitkey(self, delay=0) -> int:
        time.sleep(max(0.0, DT - (time.monotonic() - self.t_last)))
        self.t_last = time.monotonic()
        if not self.steps:
            return ord("q")
        key, fn = self.steps.pop(0)
        if fn is not None:
            fn()
        if key is None:
            return -1
        self.key, self.key_until = "key: " + KEY_NAMES.get(key, key), time.monotonic() + 0.7
        return ord(key)

    def run(self, name: str, argv: list, steps: list, cam, arm) -> str:
        self.steps, self.script = list(steps), f"scripts/real/{name}.py"
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "real" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        out = io.StringIO()
        with mock.patch.object(sys, "argv", [name, *argv]), \
                mock.patch("eaiproject.realsense_client.RealSenseClient", lambda *a, **k: cam), \
                mock.patch("eaiproject.arm_reader.ArmReader", lambda *a, **k: arm), \
                mock.patch.multiple(cv2, imshow=self.imshow, waitKey=self.waitkey, namedWindow=lambda *a: None,
                                    resizeWindow=lambda *a: None, destroyAllWindows=lambda: None,
                                    getWindowProperty=lambda *a: 1.0, createTrackbar=lambda *a: None,
                                    getTrackbarPos=lambda n, w: {"H lo": 5, "H hi": 25, "S lo": 120, "V lo": 120}[n],
                                    setTrackbarPos=lambda *a: None), \
                redirect_stdout(out):
            spec.loader.exec_module(mod)
            code = mod.main()
        if code != 0:
            raise RuntimeError(f"{name} exited with {code}:\n{out.getvalue()}")
        return out.getvalue()

    def save(self, path: Path) -> float:
        """Write the frames at their real timing, resampled to FPS."""
        t = np.array([f[0] for f in self.frames])
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, macro_block_size=16,
                                output_params=["-crf", "27", "-preset", "slow", "-movflags", "+faststart"]) as wr:
            for ts in np.arange(t[0], t[-1], 1.0 / FPS):
                wr.append_data(cv2.cvtColor(self.frames[int(np.searchsorted(t, ts, side="right")) - 1][1], cv2.COLOR_BGR2RGB))
        return float(t[-1] - t[0])


def build(d: Director, world, cam, arm):
    """Step builders: each returns a list of (key, action) played one per shown frame."""
    n = lambda sec: max(1, int(sec / DT))
    wait = lambda sec: [(None, None)] * n(sec)
    say = lambda text: [(None, lambda: setattr(d, "caption", text))]
    key = lambda k: [(k, None)]
    state = {"eye": np.array(VIEWS[0], float), "tip": REST.copy()}

    def move_cam(c, eye, sec=0.8):
        eye = np.asarray(eye, float)

        def step(u):
            def f():
                a = state["eye"]
                c.set_pose(a + (eye - a) * u)
                if u >= 1:
                    state["eye"] = eye
            return f
        return [(None, step(u)) for u in np.linspace(0, 1, n(sec))[1:] ** 0.7]

    def move_arm(point, sec=0.7, lift=0.05):
        """Fingertips from where they are to a physical point, over an arc."""
        point = np.asarray(point, float)

        def step(u):
            def f():
                a = state["tip"]
                p = a + (point - a) * u + [0, 0, lift * np.sin(np.pi * u)]
                world.tip = p
                arm.p = p + BIAS
                if u >= 1:
                    state["tip"] = point
            return f
        return [(None, step(u)) for u in np.linspace(0, 1, n(sec))[1:]]

    def hide_arm():
        def f():
            world.tip = None
            state["tip"] = REST.copy()
        return [(None, f)]

    return wait, say, key, move_cam, move_arm, hide_arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "calibration_demo.mp4"))
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    patches = [mock.patch.multiple(rc, SESSIONS_DIR=tmp / "rig_calib", MAP_PATH=tmp / "rig_map.json", OUT_DIR=tmp / "out"),
               mock.patch.object(real_intercept, "CAL_DIR", tmp / "out")]
    for p in patches:
        p.start()
    world = sim_rig.World()
    cam = sim_rig.FakeCamera(world)
    arm = sim_rig.FakeArm(world, BIAS)
    d = Director()
    wait, say, key, move_cam, move_arm, hide_arm = build(d, world, cam, arm)
    T = world.truth

    d.card([("Camera-to-arm calibration", 1.3), ("the real scripts, running on a simulated rig", 0.8),
            ("camera, markers, ball and the unpowered arm are simulated; the arm's readings are 2-9 mm off", 0.6)], 3.5)

    # ------------------------------------------------------------------ calibrate.py
    s = say("Step 1 of 3 - VIEWS. Put the camera somewhere, hands off the markers, press SPACE.\n"
            "Then move the camera and repeat: 5 or more positions, different angles and distances.") + wait(2.5)
    for i, eye in enumerate(VIEWS):
        s += (move_cam(cam, eye) if i else []) + wait(0.9) + key(" ") + wait(0.5)
    s += say("Enough views. ENTER goes to the next step.") + wait(1.2) + key("\r")
    s += move_cam(cam, VIEWS[2], 0.8)
    s += say("Step 2 of 3 - TOUCH. Guide the unpowered arm by hand: closed fingertips on the corner\n"
             "circled in yellow, hold still, SPACE. The next corner is highlighted by itself.") + wait(2.0)
    for mk, skip in (("107", 0), ("880", 3)):
        if skip:
            s += move_arm(REST, 0.5) + say("That marker is done. n = next marker; markers out of reach are passed with n.") + wait(1.0)
            for _ in range(skip):
                s += key("n") + wait(0.5)
            s += say("All 4 corners of every marker within reach: at least 2 markers.")
        for c in range(4):
            s += move_arm(T[mk][c] + np.random.default_rng(c).normal(0, 0.0007, 3) * [1, 1, 0]) + wait(0.35) + key(" ") + wait(0.35)
    s += move_arm(REST, 0.5) + say("ENTER: next step.") + wait(1.0) + key("\r")
    t_sw = {}

    def start_sweep():
        t_sw["t0"] = time.monotonic()

        def path(t):
            u = t - t_sw["t0"]
            return 0.30 + 0.22 * (0.5 - 0.5 * np.cos(u * 2 * np.pi / 9.0)), 0.275 * np.sin(u * 2 * np.pi / 1.6)
        arm.path = path

    def follow():                                      # keep the drawn gripper on the sweeping fingertips
        world.tip = world.on_table([arm.path(time.monotonic())])[0]

    s += say("Step 3 of 3 - SWEEP. Gripper tilted like the blocking pose (30 deg). Hold SPACE and slide the\n"
             "fingertips over the table. Green = covered. Cover the yellow box around the intercept line.") + wait(2.5)
    s += [(None, start_sweep)] + [(" ", follow)] * int(9.2 / DT)
    s += [(None, lambda: (setattr(arm, "path", None), setattr(world, "tip", None)))] + wait(1.3)
    s += say("ENTER: solve. Marker positions, table plane and all camera poses are fitted together.") + wait(1.0) + key("\r")
    s += say("Result: the marker map in the arm's frame, written to config/rig_map.json.\n"
             "The three numbers must be under their limits; warnings would be listed here.") + wait(6.0) + key("\r")
    out = d.run("calibrate", ["--session", "demo"], s, cam, arm)
    rig_map = rc.RigMap.load(rc.MAP_PATH)
    err = max(float(np.linalg.norm(rig_map.markers[k] - (T[k][:4] + BIAS), axis=1).max()) for k in T) * 1000
    print(f"calibrate.py: {'no warnings' if 'no warnings' in out else 'WARNINGS'}; largest map corner error {err:.2f} mm")

    # ------------------------------------------------------------- check_calibration.py
    d.card([("The camera is moved to a new place", 1.1), ("640x480 tracking stream. No arm touches, no extra step:", 0.75),
            ("the tools find the camera pose from the markers when they start", 0.75)], 3.5)
    cam2 = sim_rig.FakeCamera(world, 640, 480)
    cam2.set_pose((1.50, -0.28, 0.92))
    spot = world.on_table([(0.42, 0.10)])[0]
    s = say("Live check. Magenta = the map projected with the pose just found, green = detected markers.\n"
            "They must coincide: error per marker in px and in mm on the table.") + wait(3.5)
    s += say("The magenta ring is where the arm REPORTS its fingertips. Moved by hand, it must follow them.\n"
             "On a marker corner the error to that corner appears; SPACE records it.")
    s += move_arm(T["107"][1], 1.4) + wait(1.2) + key(" ") + wait(0.8) + move_arm(T["880"][3], 1.2) + wait(1.0) + key(" ") + wait(0.8)
    s += say("t marks the spot under the fingertips as a known point, to put the ball on later.")
    s += move_arm(spot, 1.0) + wait(0.8) + key("t") + wait(1.0) + move_arm(REST, 0.6) + hide_arm()
    s += say("b = ball mode. The ball is put on known points (yellow crosses): a marker centre ...\n"
             "Tracked position against the expected one, in mm; SPACE records it.") + key("b") + wait(0.8)
    s += [(None, lambda: setattr(world, "ball", tuple(T["467"][4][:2])))] + wait(2.2) + key(" ") + wait(1.0)
    s += say("... and the spot marked with the fingertips. Under 5 mm = OK.\n"
             "Recorded rows go to outputs/calibration/validation.csv.")
    s += [(None, lambda: setattr(world, "ball", (0.42, 0.10)))] + wait(2.2) + key(" ") + wait(2.0)
    out = d.run("check_calibration", ["--label", "demo"], s, cam2, arm)
    print("check_calibration.py:", "; ".join(l for l in out.splitlines() if l.startswith(("touch", "ball"))))

    # ------------------------------------------------------------------ real_tracking.py
    d.card([("Ball tracking", 1.1), ("real_tracking.py: the arm is not powered; the right panel shows", 0.75),
            ("the pose the planner would command to block the ball on the intercept line", 0.75)], 3.5)
    world.ball = None

    def roll(y_start, y_cross, speed=0.55):
        t0 = {}

        def f():
            t0.setdefault("t", time.monotonic())
            x = 0.97 - speed * (time.monotonic() - t0["t"])
            world.ball = (x, y_start + (y_cross - y_start) * (0.97 - x) / 0.57) if x > 0.30 else None
        return [(None, f)] * int((0.70 / speed + 0.3) / DT)

    s = say("The camera pose was found from the markers at start. A ball is rolled towards the arm:\n"
            "detection, track, predicted crossing of the intercept line, and where the jaw would go.") + wait(2.0)
    s += roll(0.10, -0.12) + wait(1.5) + roll(-0.20, 0.15) + wait(1.5) + roll(0.02, 0.02, 0.8) + wait(1.5)
    s += say("Somebody bumps the camera. The pose is re-checked against the markers every 3 seconds ...") + wait(1.0)
    s += [(None, lambda: cam2.set_pose((1.50, -0.25, 0.92), (0.56, 0.01, 0.0)))] + wait(7.5)
    s += say("... and the red warning appears. c locates the camera again from the markers. Nothing else to redo.") + wait(2.5)
    s += key("c") + wait(2.0)
    s += say("Tracking continues from the new camera position.") + roll(-0.10, -0.18) + wait(1.0) + roll(0.18, 0.05) + wait(2.0)
    out = d.run("real_tracking", [], s, cam2, arm)
    print("real_tracking.py:", "; ".join(l for l in out.splitlines() if l.startswith(("roll", "CAMERA", "camera located"))))

    d.card([("Calibrate once: scripts/real/calibrate.py", 0.9), ("Afterwards the camera can stand anywhere", 0.9),
            ("docs/calibration_protocol.md", 0.7)], 3.0)
    for p in patches:
        p.stop()
    secs = d.save(Path(args.out))
    print(f"{len(d.frames)} frames, {secs:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
