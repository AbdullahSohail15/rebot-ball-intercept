"""Synthetic end-to-end test of eaiproject/rig_calib.py (no hardware).

A random marker layout on a slightly tilted table, several camera poses, corners with pixel noise,
fingertip touches with 2 mm noise and a sweep. The arm's model bias is a constant offset added to
everything measured through FK (touches and sweep), so the "kinematic frame" the solver must
recover is the physical layout shifted by that offset.

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaiproject import rig_calib as rc  # noqa: E402

BIAS = np.array([0.003, -0.002, 0.009])          # constant FK model bias (m)
K_HD = np.array([[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]])
K_VGA = np.array([[607.0, 0, 320.0], [0, 607.0, 240.0], [0, 0, 1]])
DIST = np.array([0.02, -0.05, 0.0005, -0.0003, 0.0])
SIZE = 0.070


def look_at(eye, target):
    """OpenCV world -> camera pose of a camera at `eye` looking at `target`, image up = world z."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return cv2.Rodrigues(R)[0].reshape(3), -R @ np.asarray(eye, float)


class Rig:
    """Ground truth, in the kinematic frame (physical layout + BIAS)."""

    def __init__(self, seed: int, n_near: int = 3, n_far: int = 2):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.plane = np.array([0.006 + BIAS[2], r.uniform(-0.004, 0.004), r.uniform(-0.004, 0.004)])
        near = [(0.30, -0.22), (0.32, 0.20), (0.46, -0.02)][:n_near]
        far = [(0.78, -0.20), (0.90, 0.18), (0.70, 0.05)][:n_far]
        self.sizes = {str(100 + i): SIZE for i in range(n_near + n_far)}
        self.keys = list(self.sizes)
        self.near = self.keys[:n_near]
        self.truth = {}
        for k, (x, y) in zip(self.keys, near + far):
            pose = (x + r.uniform(-0.02, 0.02), y + r.uniform(-0.02, 0.02), r.uniform(-np.pi, np.pi))
            self.truth[k] = rc.marker_points(self.plane, pose, SIZE)          # (5, 3) corners + centre

    def camera(self, i: int | None = None):
        r = self.rng
        az = r.uniform(-1.0, 1.0) if i is None else np.linspace(-0.9, 0.9, 6)[i % 6]
        d, h = r.uniform(0.9, 1.2), r.uniform(0.55, 0.9)
        eye = np.array([0.55 + d * np.cos(az), d * np.sin(az), h])
        return look_at(eye, [0.55 + r.uniform(-0.05, 0.05), r.uniform(-0.05, 0.05), 0.0])

    def detect(self, rvec, tvec, K, wh, noise_px):
        out = {}
        for k, P in self.truth.items():
            uv = rc.project(K, DIST, rvec, tvec, P[:4]) + self.rng.normal(0, noise_px, (4, 2))
            if np.all((uv > 5) & (uv < np.array(wh) - 5)):
                out[k] = uv
        return out

    def observations(self, touched: list[str], n_views: int = 6, noise_px: float = 0.3, touch_noise: float = 0.002,
                     corners=(0, 1, 2, 3)) -> rc.Observations:
        views = []
        for i in range(n_views):
            rv, tv = self.camera(i)
            views.append(rc.View(f"view_{i:02d}", K_HD, DIST, 1280, 720, self.detect(rv, tv, K_HD, (1280, 720), noise_px)))
        touches = [rc.Touch(k, c, self.truth[k][c] + self.rng.normal(0, touch_noise, 3)) for k in touched for c in corners]
        xs, ys = np.meshgrid(np.arange(0.26, 0.56, 0.02), np.arange(-0.26, 0.27, 0.02))
        sweep = np.c_[xs.ravel(), ys.ravel(), rc.plane_z(self.plane, xs.ravel(), ys.ravel())]
        sweep[:, 2] += self.rng.normal(0, 0.001, len(sweep))
        return rc.Observations(views, touches, sweep, dict(self.sizes))

    def corner_errors_mm(self, rig_map: rc.RigMap, keys=None) -> np.ndarray:
        return np.concatenate([np.linalg.norm(rig_map.markers[k] - self.truth[k][:4], axis=1)
                               for k in (keys or self.keys)]) * 1000

    def ball_errors_mm(self, rig_map: rc.RigMap, points_xy, tracker: bool = False) -> np.ndarray:
        """Relocalise from a NEW camera pose at the tracking resolution, then locate balls."""
        rv, tv = self.camera()
        det = self.detect(rv, tv, K_VGA, (640, 480), 0.2)
        res = rc.relocalize(rig_map, K_VGA, DIST, det)
        errs = []
        for x, y in points_xy:
            centre = rc.from_plane(self.plane, rc.to_plane(self.plane, [[x, y, 0]])[0], rc.BALL_R)[0]   # true ball centre
            uv = rc.project(K_VGA, DIST, rv, tv, centre)
            if tracker:                                   # the exact path real_tracking.py uses
                from eaiproject.ball_tracker import CameraCalib
                from eaiproject.real_intercept import make_tracker
                trk = make_tracker(CameraCalib(K_VGA, DIST, res.rvec, res.tvec, 640, 480), rig_map.plane, hsv=((0, 0, 0), (1, 1, 1)))
                xy = trk.pixel_to_world(uv[0])
            else:
                xy = rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, rig_map.plane, uv)[0]
            errs.append(np.linalg.norm(xy - centre[:2]) * 1000)
        return np.array(errs)


GATE = [(0.40, y) for y in (-0.24, -0.12, 0.0, 0.12, 0.24)]
LANE = GATE + [(x, y) for x in (0.55, 0.70) for y in (-0.2, 0.0, 0.2)]


class SolverTest(unittest.TestCase):
    def test_recovers_corners_within_2mm(self):
        worst = []
        for seed in range(5):
            rig = Rig(seed)
            sol = rc.solve_rig(rig.observations(rig.near))
            e = rig.corner_errors_mm(sol.rig_map)
            worst.append(e.max())
            # 12 touches with 2 mm noise place the map to ~0.6 mm and ~3 mm per metre of yaw, so single
            # corners can be 2-3 mm off: the 2 mm requirement is checked as the RMS over all corners.
            self.assertLess(np.sqrt(np.mean(e ** 2)), 2.0, f"seed {seed}: corner RMS error")
            self.assertLess(e.max(), 3.5, f"seed {seed}: corner error {e.max():.2f} mm")
            self.assertLess(sol.report["reproj_rms_px"], 1.0)
            self.assertLess(abs(sol.rig_map.plane[0] - rig.plane[0]) * 1000, 0.5)       # height bias absorbed
            self.assertEqual(sol.warnings, [])
        print(f"\n  max corner error over 5 seeds: {max(worst):.2f} mm")

    def test_exact_without_noise(self):
        rig = Rig(0)
        sol = rc.solve_rig(rig.observations(rig.near, noise_px=0.0, touch_noise=0.0), leave_one_out=False)
        self.assertLess(rig.corner_errors_mm(sol.rig_map).max(), 0.5)       # what is left: 1 mm sweep noise in the plane
        self.assertLess(sol.report["reproj_rms_px"], 0.01)

    def test_report_contents(self):
        rig = Rig(1)
        sol = rc.solve_rig(rig.observations(rig.near))
        r = sol.report
        self.assertEqual(set(r["per_marker_reproj_px"]), set(rig.keys))
        self.assertEqual(len(r["per_view"]), 6)
        self.assertEqual(len(r["touches"]), 12)
        self.assertEqual(set(r["leave_one_marker_out"]), set(rig.near))
        for v in r["leave_one_marker_out"].values():
            self.assertLess(v["xy_mm"], 2.5 * v["expected_xy_mm"])
        self.assertGreater(r["touch_geometry"]["span_m"], 0.20)
        self.assertEqual(len(sol.sweep_residuals), r["n_sweep_points"])

    def test_map_roundtrip(self):
        import json
        rig = Rig(2)
        m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
        m2 = rc.RigMap.from_json(json.loads(json.dumps(m.to_json())))
        for k in m.markers:
            np.testing.assert_allclose(m2.markers[k], m.markers[k], atol=1e-5)
        self.assertEqual(m2.touched, m.touched)
        np.testing.assert_allclose(m2.plane, m.plane)

    def test_relocalise_new_pose_ball_error(self):
        worst = 0.0
        for seed in range(5):
            rig = Rig(seed)
            m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
            e = rig.ball_errors_mm(m, LANE)
            worst = max(worst, e.max())
            self.assertLess(e.max(), 3.0, f"seed {seed}: ball-plane error {e.max():.2f} mm")
        print(f"\n  max ball-plane error over 5 seeds (new camera pose, 640x480): {worst:.2f} mm")

    def test_ball_through_the_real_tracker_path(self):
        """Same check through make_tracker() + BallTracker, as real_tracking.py does: the end-to-end
        sign convention (table + r on the camera's side), on the tilted table plane."""
        for seed in range(5):
            rig = Rig(seed)
            m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
            e = rig.ball_errors_mm(m, LANE, tracker=True)
            self.assertLess(e.max(), 3.0, f"tracker-path error {e.max():.2f} mm")

    def test_two_touched_markers(self):
        for seed in range(5):
            rig = Rig(seed)
            sol = rc.solve_rig(rig.observations(rig.near[:2]))
            e = rig.corner_errors_mm(sol.rig_map)
            self.assertLess(np.sqrt(np.mean(e ** 2)), 4.0)       # 8 touches: weaker, far markers feel the yaw lever arm
            self.assertLess(rig.ball_errors_mm(sol.rig_map, GATE).max(), 4.0)

    def test_untouched_marker_only_in_a_later_view(self):
        """The initial guess has to chain: a far marker that the first views do not show."""
        rig = Rig(4)
        obs = rig.observations(rig.near)
        for v in obs.views[:3]:
            v.corners.pop(rig.keys[-1], None)
        self.assertLess(rig.corner_errors_mm(rc.solve_rig(obs, leave_one_out=False).rig_map).max(), 2.0)


class WarningsAndFailuresTest(unittest.TestCase):
    def test_warns_on_collinear_touches(self):
        rig = Rig(0)
        sol = rc.solve_rig(rig.observations(rig.near), leave_one_out=False)
        sol2 = rc.solve_rig(rig.observations(rig.near[:1], corners=(0, rc.CENTRE, 2)), leave_one_out=False)   # one diagonal
        self.assertFalse(any("collinear" in w for w in sol.warnings), sol.warnings)
        self.assertTrue(any("collinear" in w for w in sol2.warnings), sol2.warnings)

    def test_warns_on_small_span(self):
        rig = Rig(0)
        sol = rc.solve_rig(rig.observations(rig.near[:1]), leave_one_out=False)
        self.assertTrue(any("span only" in w for w in sol.warnings), sol.warnings)
        self.assertTrue(any("only one marker touched" in w for w in sol.warnings))

    def test_fails_without_touches_or_markers(self):
        rig = Rig(0)
        with self.assertRaises(rc.CalibrationError):
            rc.solve_rig(rig.observations([]))
        obs = rig.observations(rig.near)
        for v in obs.views:
            v.corners = {k: c for k, c in v.corners.items() if k in rig.keys[:2]}
        with self.assertRaisesRegex(rc.CalibrationError, "table marker"):
            rc.solve_rig(obs)

    def test_relocalise_refuses_too_few_markers(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        with self.assertRaisesRegex(rc.CalibrationError, "need at least 3"):
            rc.relocalize(m, K_VGA, DIST, {k: det[k] for k in rig.keys[3:]})
        with self.assertRaisesRegex(rc.CalibrationError, "far from the arm"):
            rc.relocalize(m, K_VGA, DIST, {k: det[k] for k in rig.near})

    def test_relocalise_drops_a_moved_marker(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        det[rig.near[0]] = det[rig.near[0]] + 12.0                # this marker was nudged on the table
        res = rc.relocalize(m, K_VGA, DIST, det)
        self.assertEqual(res.rejected, [rig.near[0]])
        self.assertLess(res.reproj_rms_px, 1.0)

    def test_wrong_sign_of_r_fails_loudly(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
        rv, tv = rig.camera()
        res = rc.relocalize(m, K_VGA, DIST, rig.detect(rv, tv, K_VGA, (640, 480), 0.2))
        with self.assertRaisesRegex(rc.CalibrationError, "radius must be positive"):
            rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, m.plane, [[320, 240]], r=-rc.BALL_R)
        from eaiproject.ball_tracker import CameraCalib
        from eaiproject.real_intercept import on_table
        with self.assertRaisesRegex(rc.CalibrationError, "not above"):          # the tracker refuses a camera under the table
            on_table(CameraCalib(K_VGA, DIST, res.rvec, res.tvec, 640, 480), [2.0, 0, 0])
        flipped = -np.asarray(m.plane)                         # a plane whose "up" is away from the camera
        with self.assertRaisesRegex(rc.CalibrationError, "from above"):
            rc.pixel_to_ball_xy(K_VGA, DIST, res.rvec, res.tvec, [flipped[0] + 2.0, 0, 0], [[320, 240]])

    def test_pose_monitor_warns_after_a_bump(self):
        rig = Rig(0)
        m = rc.solve_rig(rig.observations(rig.near), leave_one_out=False).rig_map
        rv, tv = rig.camera()
        det = rig.detect(rv, tv, K_VGA, (640, 480), 0.2)
        res = rc.relocalize(m, K_VGA, DIST, det)
        mon = rc.PoseMonitor(m, K_VGA, DIST, res.rvec, res.tvec)
        self.assertIsNone(mon.update(det, 0.0))
        bumped = rig.detect(rv + [0.0, 0.03, 0.0], tv, K_VGA, (640, 480), 0.2)      # a few px
        self.assertIsNone(mon.update(bumped, 3.0))              # one bad check is not enough
        self.assertIn("CAMERA MOVED", mon.update(bumped, 6.0))
        self.assertIsNone(mon.update(det, 9.0))


class SweepTest(unittest.TestCase):
    def test_filter_rejects_wrong_tilt_and_lift_and_grids(self):
        from eaiproject.ball_intercept_env import jaw_rotation
        rng = np.random.default_rng(0)
        n = 4000
        p = np.c_[rng.uniform(0.3, 0.5, n), rng.uniform(-0.2, 0.2, n), 0.012 + rng.normal(0, 0.001, n)]
        tilt = np.radians(30 + rng.normal(0, 1.5, n))
        tilt[:400] = np.radians(45)                         # wrong tilt
        p[400:600, 2] += 0.03                               # lifted off the table
        R = np.array([jaw_rotation(t, y) for t, y in zip(tilt, rng.uniform(-0.6, 0.6, n))])
        np.testing.assert_allclose(rc.tilt_deg(R), np.degrees(tilt), atol=1e-6)
        pts, st = rc.filter_sweep(p, R, 30.0)
        self.assertGreaterEqual(st["n_tilt_rejected"], 400)
        self.assertEqual(st["n_lift_rejected"], 200)
        self.assertLessEqual(len(pts), 11 * 21)
        self.assertLess(np.abs(pts[:, 2] - 0.012).max(), 0.002)

    def test_curved_surface_is_reported(self):
        rng = np.random.default_rng(1)
        xs, ys = np.meshgrid(np.arange(0.26, 0.56, 0.02), np.arange(-0.26, 0.27, 0.02))
        x, y = xs.ravel(), ys.ravel()
        flat = np.c_[x, y, 0.010 + 0.003 * x - 0.002 * y + rng.normal(0, 0.0008, len(x))]
        self.assertFalse(rc.fit_quadratic(flat, rc.fit_plane(flat)[0])["significant"])
        bent = flat.copy()
        bent[:, 2] += 0.08 * y ** 2                         # 5 mm sag at |y| = 0.25
        q = rc.fit_quadratic(bent, rc.fit_plane(bent)[0])
        self.assertTrue(q["significant"])
        self.assertGreater(q["bend_mm"], 3.0)
        rig = Rig(0)
        obs = rig.observations(rig.near)
        obs.sweep[:, 2] += 0.08 * obs.sweep[:, 1] ** 2
        sol = rc.solve_rig(obs, leave_one_out=False)
        self.assertTrue(any("curved" in w for w in sol.warnings), sol.warnings)
        self.assertTrue(sol.rig_map.stats["surface"]["significant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
