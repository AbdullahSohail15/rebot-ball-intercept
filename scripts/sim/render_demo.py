"""Render a demo video of the scripted interceptor on a few ball launches.

Usage:  MUJOCO_GL=egl .venv/bin/python scripts/sim/render_demo.py [--out outputs/demo.mp4]
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eaiproject.ball_intercept_env import BallInterceptEnv, InterceptPolicy, Launch  # noqa: E402

# (start y, y where the straight path crosses the gripper line, speed m/s)
DEMO_LAUNCHES = [(0.00, 0.00, 0.7), (0.20, -0.18, 0.8), (-0.22, 0.16, 0.9), (0.08, 0.22, 0.75), (-0.05, -0.20, 1.0)]


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="default: outputs/demo_ball_intercept_<mode>.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", default="forward", choices=["forward", "down"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    env = BallInterceptEnv(args.mode)
    pol = InterceptPolicy(env)
    W, H = args.width, args.height
    renderer = mujoco.Renderer(env.model, H, W)
    opt = mujoco.MjvOption()
    big, small = font(24), font(17)
    steps_per_frame = round(1.0 / args.fps / env.control_dt)

    def frame(title, status, color):
        views = []
        for cam in ("overview", "gate_view", "topdown"):
            renderer.update_scene(env.data, camera=cam, scene_option=opt)
            views.append(renderer.render())
        img = Image.fromarray(np.concatenate(views, axis=1))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, 3 * W, 44], fill=(0, 0, 0))
        dr.text((14, 8), title, font=big, fill=(255, 255, 255))
        dr.text((W + 14, 12), status, font=small, fill=color)
        for i in range(3):
            dr.rectangle([i * W, H - 34, i * W + 330, H], fill=(0, 0, 0))
        dr.text((14, H - 30), f"reBot Arm B601-RS  |  {env.mode} mode", font=small, fill=(230, 230, 230))
        dr.text((W + 14, H - 30), "close-up at the intercept line", font=small, fill=(230, 230, 230))
        dr.text((2 * W + 14, H - 30), "top view (arm at left)", font=small, fill=(230, 230, 230))
        return np.asarray(img)

    args.out = args.out or f"outputs/demo_ball_intercept_{args.mode}.mp4"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=8)
    results = []
    for k, (y0, yg, v) in enumerate(DEMO_LAUNCHES, 1):
        heading = float(np.arctan2(yg - y0, 0.95 - env.gate_x))
        obs = env.reset(Launch(y0=y0, speed=v, heading=heading))
        pol.reset()
        title = f"Trial {k}/{len(DEMO_LAUNCHES)}: ball {v:.2f} m/s, heading {np.degrees(heading):+.0f} deg"
        for _ in range(int(0.5 * args.fps)):  # short pause at launch
            writer.append_data(frame(title, "ball launched", (255, 255, 255)))
        n = 0
        while env.data.time < 3.0:
            obs = env.step(pol(obs))
            n += 1
            r = env.result
            if n % steps_per_frame == 0:
                if r.crossed:
                    status = r.summary()
                    col = (90, 255, 90) if r.clean_pass else (255, 90, 90)
                else:
                    status = f"tracking: predicted crossing y = {pol.predict_crossing_y(obs)*100:+.1f} cm"
                    col = (255, 220, 90)
                writer.append_data(frame(title, status, col))
            if r.crossed and env.data.time > r.crossing_time + 0.8:
                break
        for _ in range(int(0.7 * args.fps)):
            writer.append_data(frame(title, env.result.summary(), (90, 255, 90) if env.result.clean_pass else (255, 90, 90)))
        results.append(env.result)
        print(f"trial {k}: {env.result.summary()}  touched={sorted(env.result.touched_bodies)}")
    writer.close()
    print(f"clean passes {sum(r.clean_pass for r in results)}/{len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
