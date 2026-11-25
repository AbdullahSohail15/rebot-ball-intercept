"""scripts/real/catch_sweep.py against simulated motors (tests/sim_rig.py FakeMotors): the powered
sweep along the intercept line, and above all that it stops and holds when something is wrong.

    .venv/bin/python -m unittest tests.test_catch_sweep -v
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sim_rig  # noqa: E402
from eaiproject import markers  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402

MAP_PLANE = np.array([0.008, 0.002, -0.001])          # what the calibration says


class CatchSweepTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(rc, "MAP_PATH", tmp / "rig_map.json"))
        rc.RigMap({k: np.zeros((4, 3)) for k in markers.SIZES}, MAP_PLANE, dict(markers.SIZES), time="test").save(rc.MAP_PATH)
        spec = importlib.util.spec_from_file_location("catch_sweep", ROOT / "scripts/real/catch_sweep.py")
        self.cs = importlib.util.module_from_spec(spec)
        sys.modules["catch_sweep"] = self.cs
        spec.loader.exec_module(self.cs)
        self.clock = sim_rig.SimClock()
        self.cs._now, self.cs._sleep = self.clock.now, self.clock.sleep
        self.model = self.cs.Model(MAP_PLANE)
        self.q_ready = self.model.jaw(0.0, 0.060, self.model.env.q_home)
        self.cs.LOG_DIR = tmp / "hardware"
        self.prompts = []

    def run_sweep(self, argv, true_plane=MAP_PLANE, q0=None, on_prompt=None, **motor_kw):
        """Run main() on simulated motors. Returns (exit code, output, motors)."""
        q0 = self.q_ready + [0.05, -0.08, 0.06, 0.04, 0.0, 0.03] if q0 is None else q0      # held by hand, roughly there
        holder = {}

        def make_bus(channel, movable=()):
            holder["m"] = sim_rig.FakeMotors(self.cs.Model(MAP_PLANE), true_plane, q0, self.clock, movable, **motor_kw)
            return holder["m"]

        def fake_input(prompt=""):
            self.prompts.append((prompt, set(holder["m"].on)))
            if on_prompt:
                on_prompt(prompt, holder["m"])
            return ""

        out = io.StringIO()
        with mock.patch.object(self.cs, "RobStrideBus", make_bus), mock.patch("builtins.input", fake_input), \
                mock.patch.object(sys, "argv", ["catch_sweep", *argv]), redirect_stdout(out):
            code = self.cs.main()
        return code, out.getvalue(), holder["m"]

    def assert_safe_end(self, motors):
        self.assertEqual(motors.on, set(), "motors left enabled")
        self.assertTrue(motors.closed)
        self.assertNotIn(7, motors.ever_on)
        for m in range(1, 7):
            self.assertEqual(motors.par[m][motors.P_RUN_MODE], 0, "run mode not restored")
            self.assertEqual(motors.par[m][motors.P_LIMIT_CUR], 23.0, "current limit not restored")

    def test_dry_run_touches_nothing(self):
        code, out, motors = self.run_sweep(["--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("dry run", out)
        self.assertEqual((motors.writes, motors.ever_on, motors.movable), (0, set(), ()))

    def test_full_sweep_without_contact(self):
        code, out, motors = self.run_sweep([])
        self.assertIn("DONE", out, out)
        self.assertEqual(out.count("no contact"), 5)                      # 2 lower passes, 2 descents, the way back up
        self.assertNotIn("STOPPED", out)
        self.assertGreater(motors.min_clearance, 0.015)                   # 20 mm pass over a table that is where the map says
        self.assertEqual(motors.ever_on, {1, 2, 3, 4, 5, 6})
        self.assertEqual(self.prompts[-1][1], {1, 2, 3, 4, 5, 6}, "the arm must still be held when it asks to release")
        self.assert_safe_end(motors)

    def test_table_higher_than_calibrated_stops_and_holds(self):
        true_plane = MAP_PLANE + [0.024, 0.0, 0.0]                         # the real table is 24 mm above the map
        code, out, motors = self.run_sweep([], true_plane=true_plane)
        self.assertEqual(code, 1, out)
        self.assertIn("STOPPED, HOLDING POSITION: CONTACT during 'down to 20 mm'", out)
        self.assertEqual(out.count("no contact"), 2)                       # the descent to 35 mm and the pass there went through
        self.assertNotIn("DONE", out)
        self.assertGreater(motors.min_clearance, -0.004, "pressed into the table")   # the simulated table yields up to 6 mm
        self.assertIn("SUPPORT THE ARM", self.prompts[-1][0])
        self.assertEqual(self.prompts[-1][1], {1, 2, 3, 4, 5, 6}, "it must HOLD (stay powered) until the user releases")
        self.assert_safe_end(motors)

    def test_table_higher_on_one_side_only(self):
        true_plane = MAP_PLANE + [0.0, 0.0, 0.10]                          # +24 mm at y = +0.24 (the left end), level at the centre
        code, out, motors = self.run_sweep([], true_plane=true_plane)
        self.assertIn("CONTACT", out, out)
        self.assertRegex(out, r"fingertip at y = \+0\.\d+ m")
        self.assertGreater(motors.min_clearance, -0.004)
        self.assert_safe_end(motors)

    def test_blocked_joint_stops_and_holds(self):
        code, out, motors = self.run_sweep(["--passes", "60"], block_j1=0.35)        # something in the way of the swing
        self.assertEqual(code, 1, out)
        self.assertRegex(out, "STOPPED, HOLDING POSITION: joint1 (is .* behind its target|current .* at its limit)")
        self.assertLess(motors.pos[0], 0.36)
        self.assertEqual(self.prompts[-1][1], {1, 2, 3, 4, 5, 6})
        self.assert_safe_end(motors)

    def test_arm_must_be_near_the_start_pose(self):
        far = self.q_ready + [0.9, 0, 0, 0, 0, 0]

        def on_prompt(prompt, motors):
            if len(self.prompts) == 2:                                     # second ENTER: now the user has moved it there
                motors.pos[:6] = self.q_ready
        code, out, motors = self.run_sweep(["--passes", "60"], q0=far, on_prompt=on_prompt)
        self.assertIn("not close enough yet", out)
        self.assertEqual(self.prompts[0][1], set(), "nothing may be powered before the pose is accepted")
        self.assertIn("DONE", out)
        self.assert_safe_end(motors)

    def test_cannot_hold_releases_only_after_the_prompt(self):
        code, out, motors = self.run_sweep(["--cap", "0.3"])               # current limits too low to carry the arm
        self.assertIn("cannot hold the arm safely", out)
        self.assertEqual(self.prompts[-1][1], {1, 2, 3, 4, 5, 6})
        self.assert_safe_end(motors)

    def test_plan_is_refused_when_the_model_says_it_scrapes(self):
        m = self.cs.Model(MAP_PLANE)
        segs = self.cs.plan(m, self.q_ready, [0.060, 0.020], 0.24, 0.08, 40.0)
        self.assertEqual(self.cs.check_plan(m, segs, 40.0)[0], [])
        m.plane = MAP_PLANE + [0.015, 0, 0]                                # same joint path over a table 15 mm higher
        self.assertTrue(any("lowest arm point" in p for p in self.cs.check_plan(m, segs, 40.0)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
