"""Evaluate the scripted interceptor on N random launches (no rendering).

Usage:  .venv/bin/python scripts/sim/eval_scripted.py [--n 50] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eaiproject.ball_intercept_env import BallInterceptEnv, InterceptPolicy, Launch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="forward", choices=["forward", "down"])
args = ap.parse_args()

env, rng = BallInterceptEnv(args.mode), np.random.default_rng(args.seed)
pol = InterceptPolicy(env)
clean = between = bad_arm = 0
lines, lows = [], []
for k in range(args.n):
    obs = env.reset(Launch.random(rng, gate_x=env.gate_x)); pol.reset()
    while env.data.time < 3.0 and not (env.result.crossed and env.data.time > env.result.crossing_time + 0.3):
        obs = env.step(pol(obs))
    r = env.result
    clean += r.clean_pass; between += r.passed_between
    bad_arm += bool(r.arm_env_contacts)
    lows.append(r.min_jaw_height); lines.append(r.summary())
    print(f"{k:3d}  y0={env.launch.y0:+.2f}  v={env.launch.speed:.2f}  {r.summary()}  touched={sorted(r.touched_bodies)}"
          f"  arm-table/rail={sorted(r.arm_env_contacts)}  min jaw z={r.min_jaw_height*1000:.1f} mm")
print(f"[{args.mode}] ball in gripper (stopped at palm) {sum('IN GRIPPER' in l for l in lines)}/{args.n}, lowest fingertip point "
      f"{min(lows)*1000:.1f} mm")
print(f"[{args.mode}] passed between fingers {between}/{args.n}, clean (no fingertip contact) {clean}/{args.n}, "
      f"trials with arm hitting table/rails {bad_arm}/{args.n}")
