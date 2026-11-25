"""Run the calibration scripts end to end on the simulated rig (tests/sim_rig.py), headless:
capture (views, touch, sweep) -> solve -> relocalise from a new camera pose -> live check.

The OpenCV window functions are replaced by a scripted key sequence, the camera server and the CAN
arm by fakes, and every output path points into a temporary folder. Set RIG_SIM_SHOTS=<dir> to keep
the last frame each window showed (to look at the UI).

    .venv/bin/python -m unittest tests.test_scripts_sim -v
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sim_rig  # noqa: E402
from eaiproject import markers, real_intercept  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402

BIAS = np.array([0.002, -0.001, 0.009])
VIEW_EYES = [(1.45, 0.25, 0.75), (1.40, -0.35, 0.60), (1.30, 0.05, 1.00), (1.50, 0.55, 0.85), (1.25, -0.55, 0.90),
             (1.60, -0.05, 0.65)]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "real" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Keys:
    """Scripted keyboard: each step is (key or None, callable run just before it or None)."""

    def __init__(self, steps, name):
        self.steps, self.name, self.last = list(steps), name, None

    def imshow(self, win, img):
        self.last = img

    def waitkey(self, delay=0):
        if not self.steps:
            return ord("q")
        key, fn = self.steps.pop(0)
        if fn is not None:
            fn()
        if not self.steps and os.environ.get("RIG_SIM_SHOTS") and self.last is not None:
            cv2.imwrite(str(Path(os.environ["RIG_SIM_SHOTS"]) / f"{self.name}.png"), self.last)
        return -1 if key is None else ord(key)


class ScriptsOnSimulatedRig(unittest.TestCase):
    def run_script(self, name, argv, steps=(), cam=None):
        keys = Keys(steps, f"{name}_{argv[0] if argv else 'main'}")
        out = io.StringIO()
        with mock.patch.object(sys, "argv", [name, *argv]), \
                mock.patch("eaiproject.realsense_client.RealSenseClient", lambda *a, **k: cam or self.cam), \
                mock.patch("eaiproject.arm_reader.ArmReader", lambda *a, **k: self.arm), \
                mock.patch.multiple(cv2, imshow=keys.imshow, waitKey=keys.waitkey, namedWindow=lambda *a: None,
                                    resizeWindow=lambda *a: None, destroyAllWindows=lambda: None,
                                    getWindowProperty=lambda *a: 1.0, createTrackbar=lambda *a: None,
                                    getTrackbarPos=lambda n, w: {"H lo": 5, "H hi": 25, "S lo": 120, "V lo": 120}[n],
                                    setTrackbarPos=lambda *a: None), \
                redirect_stdout(out):
            code = load_script(name).main()         # loaded here: real_tracking.py imports the camera client at the top
        return code, out.getvalue()

    def test_full_pipeline(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        out_dir = tmp / "calibration"
        for target, value in ((rc, {"SESSIONS_DIR": tmp / "rig_calib", "MAP_PATH": tmp / "config" / "rig_map.json",
                                    "OUT_DIR": out_dir}), (real_intercept, {"CAL_DIR": out_dir})):
            self.enterContext(mock.patch.multiple(target, **value))
        world = sim_rig.World()
        self.cam, self.arm = sim_rig.FakeCamera(world), sim_rig.FakeArm(world, BIAS)
        kin = {k: P[:4] + BIAS for k, P in world.truth.items()}            # what the map should contain
        idle = [(None, None)] * 2
        enter = [("\r", None)]

        # ---- calibrate.py, one run: views -> touch -> sweep -> solve
        steps = []
        for eye in VIEW_EYES:                                               # 1. six camera positions
            steps += [(None, lambda e=eye: self.cam.set_pose(e)), (None, None), (" ", None)]
        steps += enter + idle
        # 2. all corners of the two markers within reach: 107, then n x3 past the far ones to 880
        for key, skip in (("107", 0), ("880", 3)):
            steps += [("n", None)] * skip
            for c in range(4):
                steps += [(None, lambda k=key, c=c: self.arm.touch(world.truth[k][c])), (" ", None)]
        steps += [("c", lambda: self.arm.touch(world.truth["880"][4]))] + enter
        # 3. hold SPACE while the fingertip rasters over the table; then a stroke at the wrong tilt
        t0 = time.monotonic()

        def start_sweep():
            self.arm.path = lambda t: (0.28 + 0.26 * (((t - t0) * 0.9) % 1.0), 0.27 * np.sin((t - t0) * 40.0))

        hold = [(" ", lambda: time.sleep(0.02))] * 60
        release = [(None, lambda: time.sleep(0.4))] * 4
        steps += [(None, start_sweep)] + idle + hold + release
        steps += [(None, lambda: setattr(self.arm, "tilt", 50.0))] + hold[:10] + release + enter
        steps += idle + enter                                               # the result screen, then done
        code, out = self.run_script("calibrate", ["--session", "sim"], steps)
        self.arm.path, self.arm.tilt = None, 30.0
        self.assertEqual(code, 0, out)
        sess = tmp / "rig_calib/sim"
        self.assertEqual(len(json.loads((sess / "views.json").read_text())["views"]), 6)
        touches = json.loads((sess / "touches.json").read_text())["touches"]
        self.assertEqual([(t["key"], t["corner"]) for t in touches],
                         [("107", c) for c in range(4)] + [("880", c) for c in range(5)])
        self.assertNotIn("CHECK", out)
        self.assertGreater(len(np.load(sess / "sweep.npz")["p"]), 500)
        self.assertIn("no warnings", out)
        rig_map = rc.RigMap.load(rc.MAP_PATH)
        err = np.concatenate([np.linalg.norm(rig_map.markers[k] - kin[k], axis=1) for k in kin]) * 1000
        self.assertLess(err.max(), 2.0, f"map corner errors (mm): {err.round(2)}")
        self.assertAlmostEqual(rig_map.plane[0], world.plane[0] + BIAS[2], delta=0.001)
        self.assertGreater(json.loads((sess / "report/report.json").read_text())["sweep_filter"]["n_tilt_rejected"], 50)
        for f in ("report.txt", "layout.png", "sweep_heatmap.png", "reproj_view_00.png"):
            self.assertTrue((sess / "report" / f).exists(), f)

        # ---- the offline re-solve gives the same map
        code, out = self.run_script("calibrate", ["--solve"])
        self.assertEqual(code, 0, out)
        again = rc.RigMap.load(rc.MAP_PATH)
        for k in kin:
            np.testing.assert_allclose(again.markers[k], rig_map.markers[k], atol=1e-5)

        # ---- check_calibration.py from a NEW camera position at the tracking resolution: it locates the
        # camera itself; fingertip on a corner, then the ball on a marker centre and on a touched point
        cam2 = sim_rig.FakeCamera(world, 640, 480)
        cam2.set_pose((1.55, 0.30, 0.95), (0.55, 0.0, 0.0))
        centre = world.truth["467"][4]
        steps = idle + [(None, lambda: self.arm.touch(world.truth["107"][1])), (None, None), (" ", None),
                        (None, lambda: self.arm.touch(world.on_table([(0.45, 0.05)])[0])), ("t", None),
                        ("b", lambda: setattr(world, "ball", tuple(centre[:2]))), (None, None), (" ", None),
                        (None, lambda: setattr(world, "ball", (0.45, 0.05))), (None, None), (" ", None), ("s", None),
                        ("c", lambda: cam2.set_pose((1.50, -0.30, 0.90))), (None, None), (None, None), (" ", None)]
        code, out = self.run_script("check_calibration", ["--label", "sim1"], steps + idle, cam=cam2)
        self.assertEqual(code, 0, out)
        pose = json.loads((out_dir / "camera_pose.json").read_text())
        self.assertEqual((pose["width"], pose["height"]), (640, 480))
        self.assertLess(np.linalg.norm(np.array(pose["camera_position"]) - (np.array([1.50, -0.30, 0.90]) + BIAS)) * 1000, 8.0)
        with (out_dir / "validation.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["kind"] for r in rows], ["touch", "ball", "ball", "ball"], out)   # the last one after the camera moved
        self.assertEqual(rows[2]["point"], "point 1")
        for r in rows:
            self.assertLess(float(r["err_mm"]), 4.0, r)
        self.assertEqual(len(list(out_dir.glob("check_sim1_*.png"))), 1)

        # ---- too few markers in view: refuse, with a reason
        cam3 = sim_rig.FakeCamera(world, 640, 480)
        cam3.set_pose((0.9, 0.0, 0.5), (0.30, 0.0, 0.0))
        code, out = self.run_script("check_calibration", [], cam=cam3)
        self.assertEqual(code, 1, out)
        self.assertIn("CANNOT LOCATE THE CAMERA", out)

        # ---- real_tracking.py locates the camera itself, its watchdog notices a bumped camera, c fixes it
        world.ball = (0.80, 0.05)
        cam2.set_pose((1.55, 0.30, 0.95))
        rc_PoseMonitor = rc.PoseMonitor
        fast = lambda *a, **k: rc_PoseMonitor(*a, **{**k, "period": 0.05})
        wait = [(None, lambda: time.sleep(0.03))] * 8
        steps = wait + [(None, lambda: cam2.set_pose((1.55, 0.32, 0.95), (0.56, 0.0, 0.0)))] + wait * 2 + [("c", None)] + wait * 2
        with mock.patch.object(rc, "PoseMonitor", fast):
            code, out = self.run_script("real_tracking", [], steps, cam=cam2)
        self.assertEqual(code, 0, out)
        self.assertIn("CAMERA MOVED", out)
        self.assertEqual(out.count("camera located:"), 2)


class GuidedCornersTest(unittest.TestCase):
    """Markers the ArUco detector cannot read are found where a pose from the others predicts them."""

    def setUp(self):
        self.world = sim_rig.World()
        self.map = rc.RigMap({k: P[:4] for k, P in self.world.truth.items()}, self.world.plane, dict(markers.SIZES))
        self.cam = sim_rig.FakeCamera(self.world, 640, 480)
        self.cam.set_pose((1.55, 0.30, 0.95))             # 107 is too small here for the detector

    def truth_px(self, k):
        return rc.project(self.cam.K, self.cam.dist, self.cam.rvec, self.cam.tvec, self.world.truth[k][:4])

    def test_recovers_an_unread_marker(self):
        img = cv2.GaussianBlur(self.cam.read().img, (0, 0), 0.8)
        plain = markers.detect(img)
        self.assertNotIn("107", plain)
        found, guided = rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)       # pose from the other three
        self.assertIn("107", guided)
        for k in guided:
            self.assertLess(np.abs(found[k] - self.truth_px(k)).max(), 0.6, k)
        res = rc.relocalize(self.map, self.cam.K, self.cam.dist, plain, guided={k: found[k] for k in guided},
                            min_markers=len(plain))
        self.assertEqual(res.guided, guided)
        self.assertLess(res.reproj_rms_px, 0.5)
        self.assertLess(np.linalg.norm(res.camera_position - [1.55, 0.30, 0.95]) * 1000, 5.0)

    def test_guided_markers_do_not_relax_the_rules(self):
        img = self.cam.read().img
        plain = markers.detect(img)
        found, guided = rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)
        two = {k: plain[k] for k in sorted(plain)[:2]}
        with self.assertRaisesRegex(rc.CalibrationError, "need at least 3"):
            rc.relocalize(self.map, self.cam.K, self.cam.dist, two, guided={k: found[k] for k in found if k not in two})

    def test_covered_marker_is_not_invented(self):
        img = self.cam.read().img
        c = self.truth_px("107").astype(np.int32)
        cv2.fillConvexPoly(img, cv2.convexHull(c + np.array([[-6, -6], [6, -6], [6, 6], [-6, 6]])), (120, 140, 170))   # a hand
        self.assertEqual(rc.detect_with_map(img, self.map, self.cam.K, self.cam.dist)[1], [])

    def test_wrong_pose_finds_nothing(self):
        img = self.cam.read().img
        rv, tv = self.cam.rvec + [0.0, 0.1, 0.0], self.cam.tvec       # every outline predicted > 3 px away
        self.assertEqual(rc.guided_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.map, self.cam.K, self.cam.dist, rv, tv), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

