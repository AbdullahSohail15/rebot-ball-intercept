# Simulation guide

MuJoCo model of the reBot Arm B601-RS intercepting a ball rolled down a table. The arm holds its
gripper wide open across the ball's path and moves it so the ball rolls between the two fingers.

| mode | gripper pose | what happens to the ball |
|---|---|---|
| `forward` (default) | Faces the incoming ball: fingers point at it, tilted down 30°. The jaw also turns (yaw) to line up with the ball's direction of travel | Rolls into the jaw and stops against the palm, between the fingers |
| `down` | Fingers point straight down like two goal posts | Rolls through the jaw and out the back |

The tilt in forward mode is needed because the gripper housing is taller than the fingers (±4.1 cm
vs ±2 cm). With a level gripper the fingers would sit above the ball, and it would roll underneath.

## Commands

```bash
.venv/bin/python scripts/sim/render_demo.py --mode forward      # outputs/demo_ball_intercept_forward.mp4 (or --mode down)
.venv/bin/python scripts/sim/eval_scripted.py --n 100 --mode forward
.venv/bin/python scripts/sim/eval_tracked.py --n 100             # vision in the loop, sweep over camera rate/latency/noise
.venv/bin/python scripts/sim/live_tracking.py --window           # live tracking overlay (or --port 8765 to stream to a browser)
.venv/bin/python -m mujoco.viewer --mjcf assets/rebot_b601_rs/mjcf/ball_intercept_scene.xml
```

On a headless Linux server, prefix the rendering commands with `MUJOCO_GL=egl`. Keep the scene XML
next to `rs_arm.xml`: MuJoCo resolves `meshdir="../meshes"` relative to the top-level file.

## Scene

| item | value |
|---|---|
| world frame | origin at the arm base; z = 0 is the tabletop; the ball rolls toward −x |
| table | x ∈ [−0.15, 1.05] m, y ∈ [−0.35, 0.35] m; low rails on the sides and the back |
| ball | 4.4 cm diameter, 45 g. Launched from x = 0.95 m (blue line) with ω = v/r, so it rolls from the first step |
| intercept line (white) | **forward:** x = 0.40 m. Closer in, joint 3 sits near its −0.2 rad limit and the IK drifts. **down:** x = 0.32 m. The env moves the marker and the close-up camera to match the mode |
| jaw height (fingertip) | **forward:** 20 mm, so the fingers reach just below the ball's centre (22 mm). With the fingertips above ~30 mm, or tilted 40° or more, the ball rolls under the palm. **down:** 15 mm. The policy holds the jaw 15 mm higher while the ball is far away and lowers it as the ball comes in, because the soft PD lets the jaw sag during big sideways moves |
| gripper | joint7 = 0.05 (fully open). The finger gap measured in the sim is 10.0 cm, so the ball centre clears both fingers within ±2.8 cm |
| reach | Beyond about \|y\| = 0.255 m the open fingers hit the side rails. The policy clips the jaw to \|y\| ≤ 0.24 m |
| physics | dt = 1 ms, Newton, implicitfast. Self-collision excludes copied from Seeed's `rs_grasp_scene.xml` |
| cameras | `overview`, `gate_view` (close-up), `topdown`, and the arm's own `wrist_rgb`. The upstream model shows a D405 wrist camera mesh by default |

## Env API

```python
from eaiproject.ball_intercept_env import BallInterceptEnv, Launch, InterceptPolicy
env = BallInterceptEnv("forward")              # or "down"; 100 Hz control, 1 kHz physics
obs = env.reset(Launch(y0=0.1, speed=0.8, heading=-0.2))  # or Launch.random(rng, gate_x=env.gate_x)
obs = env.step(np.r_[q_arm_targets(6), gripper_target])   # PD + gravity comp to joint targets
env.ik_jaw(x, y, z, yaw)                       # joint targets for a fingertip position + yaw
env.result  # crossed, lateral_offset, touched_arm, passed_between, clean_pass, reached_palm,
            # arm_env_contacts (arm hitting table/rails), min_jaw_height
```

- **Actuation:** the model's actuators are torque motors. `step` runs PD plus gravity compensation using the gains from Seeed's `rebotarm_mujoco_rs` (arm kp 80/100/100/35/25/18, gripper kp 1800).
- **Success:** when the ball crosses the fingertip line (forward mode) or the finger plane (down mode), the env reads its lateral offset from the measured inner faces of the fingers. Any ball–arm contact before then marks the trial as "touched". In down mode the check runs until the ball is behind the fingers. In forward mode, `reached_palm` records that the ball then hit the palm.

## Current results (scripted baseline)

`InterceptPolicy` reads the ground-truth ball position and velocity. It extrapolates the ball in a
straight line to the intercept line and turns the jaw to face the ball. It rate-limits the jaw's
own motion (1.2 m/s sideways, 3 rad/s yaw) rather than the joints, then IKs. Results on 100
random launches (seed 2):

| mode | between fingers, no fingertip contact | stopped at palm | arm touched table/rails | lowest fingertip |
|---|---|---|---|---|
| forward | 100/100 | 100/100 | 0/100 | 10.8 mm |
| down | 100/100 | — | 0/100 | 14.8 mm |

- Launch ranges: start y ±0.25 m, crossing y ±0.22 m, speed 0.6–1.0 m/s.
- This baseline is an upper bound. It has perfect state and perfect prediction, and the ball is slow relative to the arm.

## Vision in the loop: external camera + classic CV tracker

This mirrors the real rig. A fixed external camera (`ext_cam`) on a stand at the side of the table,
1.25 m up, sees the whole tabletop. A colour-blob tracker turns each frame into a ball estimate, and
the arm acts **only** on that estimate. Ground truth is used just for scoring and for the magenta
marker in the overlay.

The tracker (`eaiproject/ball_tracker.py`) uses no learned detector:

| step | what it does |
|---|---|
| threshold | HSV threshold for the orange ball, restricted to the tabletop, inside a 120 px window around the prediction (whole table if the ball is lost) |
| blob | morphological open, then `connectedComponentsWithStats`. Keeps the blob whose area matches the ball size the calibration predicts at that spot, nearest the prediction |
| centroid → table | undistort the centroid, then back-project it onto the plane of the ball **centre** (z = r, not z = 0; at the table edge the difference is about 9 mm) |
| filter | constant-velocity Kalman filter (x, y, vx, vy) with real frame timestamps and a 4σ outlier gate |
| output | estimate extrapolated to *now*, which compensates camera latency. Crossing point = straight line to the intercept line |

- **CV cost:** about 0.6–1 ms per 640×480 frame in Python, and about 2 ms when it has to search the whole table.
- **Policy:** `TrackedInterceptPolicy` holds the home pose until the tracker has 4 measurements. It coasts on the filter when the gripper hides the ball, and holds still once the estimate reaches the line.
- **Sim camera model:** `SimCamera` adds the frame rate, Gaussian pixel noise and a capture-to-tracker latency queue.
- **Calibration:** in the sim it comes exactly from the MuJoCo camera (`camera_from_mujoco`). On the real rig, use a checkerboard for the intrinsics and `calibrate_extrinsics` (solvePnP) on 4+ table points with known robot-frame coordinates. Retune `hsv_lo`/`hsv_hi` on real frames, and pass `bgr=True` for OpenCV camera frames.

### Watching it live

`scripts/sim/live_tracking.py` shows the external camera with the tracking overlay next to the arm overview:
- detection circle and centroid
- track
- velocity arrow
- predicted path and crossing point
- ground truth
- the HSV mask inset
- TRACKING / COASTING / LOST status, estimate error, CV time per frame

```bash
# headless server: stream to a browser through an SSH tunnel
MUJOCO_GL=egl .venv/bin/python scripts/sim/live_tracking.py --port 8765        # runs until Ctrl-C
#   on your laptop: ssh -L 8765:localhost:8765 <server>, then open http://localhost:8765
# desktop, or ssh -X: OpenCV window (q / Esc to quit)
MUJOCO_GL=egl .venv/bin/python scripts/sim/live_tracking.py --window
# realistic camera
... --fps 60 --latency 0.05 --noise 6
# demo video
MUJOCO_GL=egl .venv/bin/python scripts/sim/live_tracking.py --demo --trials 5 --no-realtime \
    --save outputs/demo_ball_tracking_forward.mp4
# evaluation sweep over camera rate, latency and noise
MUJOCO_GL=egl .venv/bin/python scripts/sim/eval_tracked.py --n 100
```

Video: `outputs/demo_ball_tracking_forward.mp4` (generated by the command above) (the same 5 launches as the oracle demo).

### Results with the tracker (same 100 launches as the oracle, seed 2, forward mode)

| camera | in gripper | clean | arm hit table | \|offset\| | crossing prediction error | detection error | CV ms/frame (mean / max) |
|---|---|---|---|---|---|---|---|
| ideal, 60 fps | 100/100 | 100 | 0 | 1.4 mm | 1.6 mm | 1.7 mm | 0.44 / 25.6 (first-call warm-up) |
| 60 fps, noise 6, latency 33 ms | 100/100 | 100 | 0 | 1.6 mm | 1.8 mm | 1.8 mm | 0.47 / 4.0 |
| 60 fps, noise 6, latency 66 ms | 100/100 | 100 | 0 | 1.9 mm | 1.8 mm | 1.7 mm | 0.41 / 1.9 |
| 30 fps, noise 6, latency 66 ms | 100/100 | 100 | 0 | 2.5 mm | 1.8 mm | 1.8 mm | 0.47 / 3.1 |
| 30 fps, noise 10, latency 100 ms | 100/100 | 100 | 0 | 2.9 mm | 1.9 mm | 1.7 mm | 0.45 / 2.4 |

Down mode (60 fps, noise 6, latency 33 ms): 100/100 clean, 0 arm–table contacts.

The sim is the easy case, so treat these numbers as a ceiling:
- perfect calibration
- uniform lighting and constant colours
- no motion blur, no rolling shutter
- a straight-rolling ball

The margin is large: about 2–3 mm of error against ±28 mm of clearance. On the real rig, expect
calibration and HSV tuning to be the main work.

Natural next steps:
- real camera: calibration, HSV tuning, timestamps from the driver
- faster balls or a shorter reaction distance
- bounces or curved paths
- closing the gripper on the ball (forward mode)
- a learned policy
