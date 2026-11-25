# Real rig guide

Everything needed to run the ball-interception pipeline on the physical reBot Arm B601-RS with an
Intel RealSense D435i. The steps go from read-only checks to powered motion, one level of risk at a
time. Do them in order the first time.

- [Hardware](#hardware)
- [Software setup](#software-setup)
- [1. Check the arm (read-only)](#1-check-the-arm-read-only)
- [2. First powered motions](#2-first-powered-motions)
- [3. Camera](#3-camera)
- [4. Calibration](#4-calibration)
- [5. Live tracking (shadow mode)](#5-live-tracking-shadow-mode)
- [Safety rules](#safety-rules)
- [Troubleshooting](#troubleshooting)
- [Protocol notes](#protocol-notes)

## Hardware

| item | details |
|---|---|
| Arm | Seeed Studio reBot Arm B601-RS: RobStride RS06 motors on joints 1-3, RS00 on joints 4-6 and the gripper (CAN IDs 1-7, host ID 0xFD), 48 V supply |
| CAN adapter | PEAK PCAN-USB (shows up as "XCAN-USB", USB 0c72:000c), 1 Mbit/s |
| Camera | Intel RealSense D435i, USB 3, fixed on a stand looking at the table |
| Markers | ArUco markers taped flat on the table (70 mm, `DICT_ARUCO_ORIGINAL`, IDs 107, 880, 467, 718). Defined in `eaiproject/markers.py`. The markers on the arm's mounting plate are not used. |
| Ball | orange ping-pong ball, 44 mm |

Tested on an Apple Silicon Mac. Linux should be simpler (no root needed for the camera, SocketCAN for
the adapter), but the CAN transport in `eaiproject/robstride.py` currently opens the PCAN interface
through python-can; on Linux change the interface in `RobStrideBus` if you use SocketCAN.

## Software setup

```bash
python3.12 -m venv .venv                     # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
```

**PCAN driver on macOS (no sudo).** python-can needs the free PCBUSB library from
[mac-can.com](https://www.mac-can.com) (non-commercial use). Copy the dylib into `~/lib`, which macOS
searches by default:

```bash
curl -LO https://github.com/mac-can/PCBUSB-Library/releases/download/v0.13/macOS_Library_for_PCANUSB_v0.13.tar.gz
tar xzf macOS_Library_for_PCANUSB_v0.13.tar.gz
mkdir -p ~/lib && cp PCBUSB/libPCBUSB.0.13.dylib ~/lib/ && ln -sf ~/lib/libPCBUSB.0.13.dylib ~/lib/libPCBUSB.dylib
```

**RealSense on macOS.** What we found:

- Use `pyrealsense2-macosx==2.54.2`. Version 2.56.5 segfaults when it lists devices.
- librealsense must run as **root** on macOS. Root processes can't read `~/Documents` (macOS privacy
  protection), so the camera runs as a small frame server from `~/realsense_tools` with its own venv.
  `scripts/real/start_camera.sh` sets that up and starts it with a macOS password dialog, once per session.
- The colour stream only delivers frames while the depth stream is also enabled. The server does this.
- Your terminal app (e.g. VS Code) needs macOS **Camera** permission for the plain-webcam fallback.

## 1. Check the arm (read-only)

Nothing is powered in this section. The CAN wrapper refuses enable / move / write commands to every
motor that isn't explicitly allowed.

```bash
.venv/bin/python scripts/real/gripper_test.py            # prints all 7 joint states (position, velocity, current, bus voltage)
.venv/bin/python scripts/real/joint_check.py --ranges --gripper
```

`joint_check.py` opens a window with the MuJoCo arm BEFORE / AFTER a positive move of each joint and a
live render of the sim posed at the real readings. Move each real joint by hand the way the picture
shows: it reports the direction (sign), zero offset, cross-talk (a wrong CAN ID) and range against the
model. Seeed's convention is q_model = q_motor, and the arm we tested matched it.
`--live` just mirrors the real arm in the sim.

## 2. First powered motions

```bash
.venv/bin/python scripts/real/gripper_test.py --move                # gripper only: open 2.5 rad, close 0.2 rad
.venv/bin/python scripts/real/arm_motion_test.py --dry-run          # all checks, nothing enabled
.venv/bin/python scripts/real/arm_motion_test.py                    # joint 1: +15 deg and back
```

- `gripper_test.py --move` powers only the gripper: MIT mode, low gains (kp 5, kd 0.5), at most 1 rad/s,
  aborts on tracking error, torque, faults or lost feedback. It asks before moving.
- `arm_motion_test.py` powers ONE joint (1, 4, 5 or 6) in the motor's position mode, with the firmware
  speed limit (0.33 rad/s) and current cap set before enabling. It checks limits and collisions in the
  model first, aborts on tracking error, stall, or any other joint moving, and restores the motor
  parameters afterwards. Joints 2 and 3 are refused: they carry the arm's weight, so they need a
  gravity-compensated hold first (not written yet).

Measured on our arm: joint 1 reached +15.2 deg (target 15), returned within 0.01 deg, 1.9 A peak.

## 3. Camera

```bash
scripts/real/start_camera.sh                  # 640x480 @ 60 fps; password dialog once per session
.venv/bin/python scripts/real/camera_view.py  # live view: grid, ball blobs, detected markers; s = snapshot
```

Aim the camera so it sees the arm, the area in front of it (the intercept line is 0.40 m from the
base) and the table out to where the ball is launched, with all markers detected. Then fix the camera
firmly. If it moves later, press `c` in the tracking window: the pose is found again from the markers.

Stop the server with
`.venv/bin/python -c "from eaiproject.realsense_client import RealSenseClient as C; C().stop_server()"`.
It also stops after 30 minutes without clients.

## 4. Calibration

```bash
scripts/real/start_camera.sh --width 1280 --height 720 --fps 30
.venv/bin/python scripts/real/calibrate.py            # once: one guided window, about 20 min
.venv/bin/python scripts/real/check_calibration.py    # any time: live check
```

`calibrate.py` builds a map of the table markers in the arm's own frame, in three steps:

1. the camera looks at the markers from 5 or more positions,
2. the unpowered arm's fingertips touch the corners of the markers within reach,
3. the fingertips are slid over the table at the blocking tilt.

It then solves for the marker positions and the table plane and writes `config/rig_map.json`
(commit it). From then on the camera can be moved: `real_tracking.py` and `check_calibration.py`
find the camera pose from the markers when they start, and again when you press `c`.

The arm's model reads heights 8-15 mm off. Because the map and the table plane are measured with
the arm's own readings, and the planner commands the arm through the same model, this error cancels.

What the windows look like, on a simulated rig: [media/calibration_demo.mp4](media/calibration_demo.mp4).

The step-by-step lab protocol, acceptance thresholds, what to do when a check fails and the
validation experiment are in [calibration_protocol.md](calibration_protocol.md).

**Powered sweep of the catching area** (after the calibration):

```bash
.venv/bin/python scripts/real/catch_sweep.py --dry-run     # plan and checks only
.venv/bin/python scripts/real/catch_sweep.py --passes 60   # first: one pass 60 mm above the table
.venv/bin/python scripts/real/catch_sweep.py               # 60, 35 and 20 mm
```

The arm moves its jaw along the intercept line, left to right and back, stepping down to the
blocking height. This powers all six arm joints. You hand the arm over to the motors at the start
pose and take it back at the end; in between it stops and holds if it touches the table or anything
else. Read [calibration_protocol.md](calibration_protocol.md#powered-sweep-of-the-catching-area) first.

## 5. Live tracking (shadow mode)

```bash
.venv/bin/python scripts/real/real_tracking.py
```

The arm is not used at all. The camera pose is found from the markers at start (`c` finds it again
after the camera was moved). The window shows:

- the detection, track and velocity,
- the intercept line and the predicted crossing,
- where the open gripper would go,
- a MuJoCo render of the pose the planner would command.

Each roll is scored: the predicted crossing 0.5 / 0.3 / 0.15 s before the line against where the ball
really crossed (gripper margin ±28 mm).

Keys: `c` locates the camera again, `a` auto-tunes the ball colour (ball alone on the table), `w` saves it, `h` toggles the mask
inset, `r` resets the tracker. The HSV window has sliders.

The planner (`eaiproject/real_intercept.py`) is the sim policy in real time:

- straight-line crossing prediction,
- the jaw facing the ball,
- rate limits on the jaw pose (0.6 m/s sideways, slower than in the sim),
- IK in the MuJoCo model,
- heights from the table plane in `config/rig_map.json`.

**Next (not done yet):** closed-loop blocking with slow rolls. `catch_sweep.py` already holds the arm
with the motors' position mode and moves it along the intercept line.

## Safety rules

- Run the read-only checks after any change to wiring, firmware or zero positions.
- Joints 2 and 3 drop when disabled. Support the arm whenever it is unpowered and away from rest.
- `catch_sweep.py` is the only script that powers joints 2 and 3. Hand the arm to it and take it back
  by hand, as it asks.
- Every powered script enables only the joints it moves. It aborts and disables on tracking error,
  stall/torque, faults, lost feedback, unexpected motion of other joints, Ctrl+C and exceptions.
- Motor parameter writes are RAM-only (type 18). Nothing is saved to the motors' flash.
- Keep a hand on the arm's power switch during powered tests.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `No motor answered` / `transmit queue is full` | arm power off, or CAN cable not connected (no node ACKs the frames) |
| `PcanCanInitializationError ... handle is invalid` | the PCAN-USB adapter is unplugged |
| camera server: `failed to set power state` | another (stuck) server holds the camera. `start_camera.sh` kills the old one by PID; if it persists, unplug and replug the camera |
| camera server: `Frame didn't arrive` | the camera re-enumerated. Replug it and restart the server |
| colour stream gives no frames | depth must be enabled too (the server does this) |
| small markers not detected | too small / too edge-on in the image. Move the camera closer or higher. |
| fingertip reads ~50-60 mm on top of the 44 mm ball | known model height bias; the calibration measures the table with the arm itself, so it cancels |
| `CANNOT LOCATE THE CAMERA` | fewer than 3 markers read, or none far from the arm: see [calibration_protocol.md](calibration_protocol.md#every-session) |

## Protocol notes

RobStride private CAN protocol, 29-bit extended IDs, comm type in bits 24-28 (`eaiproject/robstride.py`):

| type | meaning | id layout | data |
|---|---|---|---|
| 1 | MIT command | `1<<24 \| torque_u16<<8 \| motor` | pos, vel, kp, kd (u16 big-endian) |
| 2 | feedback | `2<<24 \| mode/faults<<16 \| motor<<8 \| host` | pos, vel, torque, temp×10 |
| 3 / 4 | enable / stop | `t<<24 \| host<<8 \| motor` | zeros |
| 17 / 18 | read / write parameter | `t<<24 \| host<<8 \| motor` | index (u16 LE), value at bytes 4-7 |

Scaling: position ±4π rad. RS06: ±50 rad/s, ±36 Nm, kp 0-5000, kd 0-100. RS00: ±33 rad/s, ±14 Nm,
kp 0-500, kd 0-5. The torque ranges match the motors' own `limit_torque` parameter.

Useful parameters: 0x7005 run_mode (0 MIT, 1 position), 0x7016 loc_ref, 0x7017 limit_spd,
0x7018 limit_cur, 0x7019 mechPos, 0x701A iqf, 0x701B mechVel, 0x701C VBUS.
The gripper travel is 0 (closed) to 5.0 rad (open), from Seeed's config.
