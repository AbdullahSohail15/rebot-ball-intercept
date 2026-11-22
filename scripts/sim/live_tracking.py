"""Live ball-tracking viewer: external camera view with the CV tracking overlay, next to an
overview of the arm, while the arm intercepts the ball using only the tracker's estimate.

Three ways to watch (any combination):
  --window        OpenCV window (needs a display: run locally, or over `ssh -X` / `ssh -Y`)
  --port 8765     MJPEG stream in a browser. On a headless server, tunnel it from your laptop:
                      ssh -L 8765:localhost:8765 <server>     then open http://localhost:8765
  --save out.mp4  write a video

With no option it opens a window if $DISPLAY is set, otherwise it serves on port 8765.

    MUJOCO_GL=egl .venv/bin/python scripts/sim/live_tracking.py --port 8765                 # runs until Ctrl-C
    MUJOCO_GL=egl .venv/bin/python scripts/sim/live_tracking.py --trials 5 --demo \
        --save outputs/demo_ball_tracking_forward.mp4 --no-realtime                     # demo video
    ... --latency 0.05 --noise 6 --fps 60                                               # realistic camera
"""
import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eaiproject.ball_intercept_env import Launch  # noqa: E402
from eaiproject.tracked_intercept import make_tracked  # noqa: E402

# Same launches as scripts/sim/render_demo.py: (start y, crossing y at the intercept line, speed)
DEMO_LAUNCHES = [(0.00, 0.00, 0.7), (0.20, -0.18, 0.8), (-0.22, 0.16, 0.9), (0.08, 0.22, 0.75), (-0.05, -0.20, 1.0)]

LOST_AFTER = 0.12   # s without a detection before the overlay stops drawing the extrapolated estimate

# RGB colours (frames are RGB until written/shown)
GREEN, CYAN, RED, MAGENTA, YELLOW, WHITE, BLUE = (40, 230, 60), (0, 220, 255), (255, 50, 40), (255, 0, 255), \
    (255, 220, 0), (255, 255, 255), (60, 120, 255)


# ------------------------------------------------------------------------- MJPEG server
class Stream:
    def __init__(self):
        self.jpg, self.cond = None, threading.Condition()

    def push(self, rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.cond:
            self.jpg = buf.tobytes()
            self.cond.notify_all()


def serve(stream: Stream, port: int):
    page = (b"<html><head><title>Ball tracking</title></head><body style='margin:0;background:#111'>"
            b"<img src='/stream' style='width:100%;max-width:1280px;display:block;margin:auto'></body></html>")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/frame.jpg"):
                with stream.cond:
                    stream.cond.wait_for(lambda: stream.jpg is not None, timeout=5)
                    jpg = stream.jpg
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(jpg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with stream.cond:
                            stream.cond.wait(timeout=1.0)
                            jpg = stream.jpg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(page)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ----------------------------------------------------------------------------- overlay
def text(img, s, org, color=WHITE, scale=0.5, thick=1):
    # Outline = offset copies at the same thickness (OpenCV 5 spaces thicker text differently).
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def P(uv):
    return int(round(uv[0])), int(round(uv[1]))


def draw_tracking(runner, img, tc, truth, info):
    env, tr = runner.env, runner.tracker
    out = img.copy()
    gx = env.gate_x
    # intercept line (on the table) and the tracker's search region
    a, b = tr.world_to_pixel([[gx, -0.33], [gx, 0.33]], z=0.0)
    cv2.line(out, P(a), P(b), WHITE, 1, cv2.LINE_AA)
    if tr.last_roi is not None and tr.last_roi != (0, 0, out.shape[1], out.shape[0]):
        x0, y0, x1, y1 = tr.last_roi
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), YELLOW, 1)
    # trail of accepted measurements
    if len(tr.history) > 1:
        pts = tr.world_to_pixel(np.array([h[1] for h in tr.history[-90:]]))
        cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)], False, CYAN, 2, cv2.LINE_AA)
    det = tr.last_det
    status, scol = "SEARCHING", YELLOW
    if det is not None:
        cv2.circle(out, P(det.uv), int(det.radius_px + 4), GREEN, 2, cv2.LINE_AA)
        cv2.drawMarker(out, P(det.uv), GREEN, cv2.MARKER_CROSS, 8, 1)
        status, scol = "TRACKING", GREEN
    lost = tr.x is not None and det is None and tr.state_at(tc).lost_for > LOST_AFTER
    if lost:
        status, scol = "LOST (ball hidden, e.g. by the gripper)", (255, 140, 0)
    elif det is None and tr.x is not None:
        status, scol = "COASTING on the filter (no detection)", (255, 140, 0)
    if tr.x is not None and not lost:
        s = tr.state_at(tc)
        c = tr.world_to_pixel(s.pos)[0]
        if det is None:
            cv2.circle(out, P(c), 10, (255, 140, 0), 1, cv2.LINE_AA)
        # velocity arrow (0.15 s ahead)
        cv2.arrowedLine(out, P(c), P(tr.world_to_pixel(s.pos + 0.15 * s.vel)[0]), BLUE, 2, cv2.LINE_AA, tipLength=0.3)
        pc = tr.predict_crossing(gx)
        if pc is not None and s.pos[0] > gx:
            end = tr.world_to_pixel([gx, pc[0]])[0]
            n = 12
            for i in range(0, n, 2):   # dashed predicted path
                p0 = tr.world_to_pixel(s.pos + (np.array([gx, pc[0]]) - s.pos) * i / n)[0]
                p1 = tr.world_to_pixel(s.pos + (np.array([gx, pc[0]]) - s.pos) * (i + 1) / n)[0]
                cv2.line(out, P(p0), P(p1), RED, 2, cv2.LINE_AA)
            cv2.circle(out, P(end), 7, RED, 2, cv2.LINE_AA)
            info["pred_y"] = pc[0]
    # ground truth (for comparison only, the policy never sees it)
    gt = tr.world_to_pixel(truth[:2])[0]
    cv2.drawMarker(out, P(gt), MAGENTA, cv2.MARKER_TILTED_CROSS, 7, 1)
    # mask inset (what the colour threshold sees)
    if tr.last_mask is not None:
        m, _ = tr.last_mask
        inset = cv2.resize(m, (120, 120), interpolation=cv2.INTER_NEAREST)
        out[8:128, -128:-8] = np.repeat(inset[:, :, None], 3, axis=2)
        cv2.rectangle(out, (out.shape[1] - 129, 7), (out.shape[1] - 8, 128), WHITE, 1)
        text(out, "HSV mask (search window)", (out.shape[1] - 230, 142), WHITE, 0.4)
    # status text
    cv_ms = tr.timing_ms[-1] if tr.timing_ms else 0.0
    text(out, f"EXTERNAL CAMERA  {runner.camera.fps:.0f} fps  latency {1000 * runner.camera.latency:.0f} ms"
              f"  noise {runner.camera.noise_std:.0f}", (8, 20), WHITE, 0.5)
    text(out, status, (8, 42), scol, 0.55, 2)
    y = 62
    if tr.x is not None and not lost:
        s = tr.state_at(tc)
        err = np.linalg.norm(s.pos - truth[:2]) * 1000
        text(out, f"ball  x {s.pos[0]:+.3f}  y {s.pos[1]:+.3f} m   v {np.linalg.norm(s.vel):.2f} m/s", (8, y)); y += 18
        text(out, f"estimate vs truth: {err:.1f} mm", (8, y)); y += 18
        if info.get("pred_y") is not None:
            text(out, f"predicted crossing y {info['pred_y']:+.3f} m", (8, y), RED); y += 18
    text(out, f"CV {cv_ms:.2f} ms/frame", (8, y)); y += 18
    legend = [("detection", GREEN), ("track", CYAN), ("velocity", BLUE), ("predicted path", RED), ("ground truth", MAGENTA)]
    x = 8
    for name, col in legend:
        cv2.circle(out, (x + 5, out.shape[0] - 12), 5, col, -1)
        text(out, name, (x + 14, out.shape[0] - 7), WHITE, 0.42)
        x += 22 + 8 * len(name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="forward", choices=["forward", "down"])
    ap.add_argument("--window", action="store_true", help="OpenCV window (needs $DISPLAY)")
    ap.add_argument("--port", type=int, default=None, help="serve an MJPEG stream on this port")
    ap.add_argument("--save", default=None, help="write the composite view to this mp4")
    ap.add_argument("--trials", type=int, default=0, help="number of launches (0 = until Ctrl-C)")
    ap.add_argument("--demo", action="store_true", help="use the 5 launches of render_demo.py (cycled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=60.0, help="camera frame rate")
    ap.add_argument("--latency", type=float, default=0.0, help="camera-to-tracker latency (s)")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian pixel noise std (0-255)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    ap.add_argument("--no-realtime", action="store_true", help="run as fast as possible (for --save)")
    ap.add_argument("--no-overview", action="store_true", help="show only the external camera")
    a = ap.parse_args()

    if not (a.window or a.port or a.save):
        if os.environ.get("DISPLAY"):
            a.window = True
        else:
            a.port = 8765
    if a.window and not os.environ.get("DISPLAY"):
        sys.exit("--window needs a display ($DISPLAY is empty). Use ssh -X, or --port 8765 and a browser.")

    runner = make_tracked(a.mode, width=a.width, height=a.height, fps=a.fps, latency=a.latency,
                          noise_std=a.noise, seed=a.seed)
    env = runner.env
    ov = None if a.no_overview else mujoco.Renderer(env.model, a.height, a.width)

    stream = None
    if a.port:
        stream = Stream()
        serve(stream, a.port)
        print(f"MJPEG stream: http://localhost:{a.port}   (from your laptop: ssh -L {a.port}:localhost:{a.port} <server>)")
    writer = None
    if a.save:
        import imageio.v2 as imageio
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(a.save, fps=a.fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=8)
    if a.window:
        cv2.namedWindow("ball tracking", cv2.WINDOW_NORMAL)

    rng = np.random.default_rng(a.seed)
    state = dict(trial=0, banner="", info={}, quit=False)

    def on_frame(r, img, tc, truth):
        cam_view = draw_tracking(r, img, tc, truth, state["info"])
        if ov is not None:
            ov.update_scene(env.data, "overview")
            o = ov.render().copy()
            text(o, f"Trial {state['trial']}: {state['title']}", (8, 22), WHITE, 0.55, 1)
            text(o, "arm uses ONLY the camera estimate", (8, 44), CYAN, 0.5)
            if state["banner"]:
                text(o, state["banner"], (8, o.shape[0] - 14), state["bcol"], 0.6, 2)
            frame = np.hstack([cam_view, o])
        else:
            frame = cam_view
        if writer is not None:
            writer.append_data(frame)
        if stream is not None:
            stream.push(frame)
        if a.window:
            cv2.imshow("ball tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                state["quit"] = True

    try:
        k = 0
        while not state["quit"] and (a.trials == 0 or k < a.trials):
            if a.demo:
                y0, yg, v = DEMO_LAUNCHES[k % len(DEMO_LAUNCHES)]
                launch = Launch(y0=y0, speed=v, heading=float(np.arctan2(yg - y0, 0.95 - env.gate_x)))
            else:
                launch = Launch.random(rng, gate_x=env.gate_x)
            k += 1
            state.update(trial=k, banner="", info={},
                         title=f"ball {launch.speed:.2f} m/s, heading {np.degrees(launch.heading):+.0f} deg")
            runner.reset(launch)
            t_wall0, t_end = time.perf_counter(), None
            while not state["quit"]:
                runner.step(on_frame)
                res = env.result
                if t_end is None and (res.crossed or env.data.qpos[env.ball_q] < -0.1 or env.data.time > 2.0):
                    t_end = env.data.time + 0.6
                if t_end is not None:   # keep updating: the ball reaches the palm just after crossing
                    ok = res.passed_between and (res.reached_palm or a.mode == "down")
                    state["banner"], state["bcol"] = res.summary(), (GREEN if ok else RED)
                if t_end is not None and env.data.time >= t_end:
                    pe = runner.stats.final_pred_error_mm()
                    print(f"trial {k}: {res.summary()} | pred err {float('nan') if pe is None else pe:.1f} mm | "
                          f"CV {np.mean(runner.stats.cv_ms):.2f} ms/frame", flush=True)
                    break
                if not a.no_realtime:
                    lag = env.data.time / a.speed - (time.perf_counter() - t_wall0)
                    if lag > 0:
                        time.sleep(lag)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
            print("wrote", a.save)
        if a.window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
