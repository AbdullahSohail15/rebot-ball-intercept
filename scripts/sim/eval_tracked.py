"""Evaluate the vision-in-the-loop policy (external camera + BallTracker) on random launches.

    MUJOCO_GL=egl .venv/bin/python scripts/sim/eval_tracked.py --n 100            # default sweep
    MUJOCO_GL=egl .venv/bin/python scripts/sim/eval_tracked.py --n 100 --latency 0.05 --noise 6 --fps 60

Uses the same launches as eval_scripted.py (Launch.random with the same seed), so the numbers are
directly comparable with the ground-truth (oracle) policy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eaiproject.ball_intercept_env import Launch  # noqa: E402
from eaiproject.tracked_intercept import make_tracked  # noqa: E402

SWEEP = [  # (name, fps, latency s, pixel noise std)
    ("ideal 60 fps", 60, 0.000, 0.0),
    ("60 fps, noise 6, latency 33 ms", 60, 0.033, 6.0),
    ("60 fps, noise 6, latency 66 ms", 60, 0.066, 6.0),
    ("30 fps, noise 6, latency 66 ms", 30, 0.066, 6.0),
    ("30 fps, noise 10, latency 100 ms", 30, 0.100, 10.0),
]


def evaluate(n, seed, mode, fps, latency, noise, verbose=False):
    runner = make_tracked(mode, fps=fps, latency=latency, noise_std=noise, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        launch = Launch.random(rng, gate_x=runner.env.gate_x)
        res, st = runner.run(launch)
        rows.append(dict(
            ok=res.reached_palm and res.passed_between if mode == "forward" else res.passed_between,
            clean=res.clean_pass, contacts=bool(res.arm_env_contacts),
            off=abs(res.lateral_offset) * 1000 if res.crossed else np.nan,
            pred=st.final_pred_error_mm() if st.final_pred_error_mm() is not None else np.nan,
            det=np.mean(st.det_err_mm) if st.det_err_mm else np.nan,
            det_rate=st.n_detected / max(st.n_frames, 1), cv=np.mean(st.cv_ms) if st.cv_ms else np.nan,
            cv_max=np.max(st.cv_ms) if st.cv_ms else np.nan))
        if verbose:
            print(f"{i:3d} {res.summary()}  pred err {rows[-1]['pred']:.1f} mm")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--mode", default="forward", choices=["forward", "down"])
    ap.add_argument("--fps", type=float)
    ap.add_argument("--latency", type=float)
    ap.add_argument("--noise", type=float)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    sweep = SWEEP if a.fps is None and a.latency is None and a.noise is None else \
        [("custom", a.fps or 60, a.latency or 0.0, a.noise or 0.0)]
    print(f"mode={a.mode}, {a.n} launches, seed {a.seed}\n")
    hdr = (f"{'condition':34s} {'in gripper':>10s} {'clean':>6s} {'arm hit':>7s} {'|offset|':>9s} "
           f"{'pred err':>9s} {'det err':>8s} {'detected':>8s} {'CV ms':>11s}")
    print(hdr + "\n" + "-" * len(hdr))
    for name, fps, lat, noise in sweep:
        r = evaluate(a.n, a.seed, a.mode, fps, lat, noise, a.verbose)
        g = lambda k: np.array([x[k] for x in r], float)  # noqa: E731
        print(f"{name:34s} {int(g('ok').sum()):>6d}/{a.n:<3d} {int(g('clean').sum()):>6d} {int(g('contacts').sum()):>7d} "
              f"{np.nanmean(g('off')):>6.1f} mm {np.nanmean(g('pred')):>6.1f} mm {np.nanmean(g('det')):>5.1f} mm "
              f"{100 * np.nanmean(g('det_rate')):>7.0f}% {np.nanmean(g('cv')):>5.2f}/{np.nanmax(g('cv_max')):>4.1f}", flush=True)
    print("\n|offset| = ball centre vs jaw centre at the fingertip line; pred err = last predicted crossing y vs"
          "\ntrue crossing y; det err = detected vs true ball centre per frame; detected = frames with an accepted"
          "\ndetection (the gripper hides the ball at the end); CV ms = mean/max tracker time per frame.")


if __name__ == "__main__":
    main()
