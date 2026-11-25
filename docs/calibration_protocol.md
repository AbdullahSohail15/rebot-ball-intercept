# Calibration protocol (camera to arm)

The arm touches the table markers **once**. After that the camera can stand anywhere: every tool
finds the camera pose from the markers by itself when it starts.

- **Calibrate** (once, about 20 min, of which 15 with the arm): `scripts/real/calibrate.py`, one
  guided window. Result: `config/rig_map.json`, the marker corners and the table plane in the arm's
  frame. Commit it.
- **Use** (every session, nothing to do): `real_tracking.py` and `check_calibration.py` locate the
  camera from the markers at start, and again when you press `c` after moving it.

A 2-minute video of the whole flow on a simulated rig (the real scripts, simulated camera and arm):
[media/calibration_demo.mp4](media/calibration_demo.mp4). `scripts/sim/calibration_demo.py` makes it.

Redo the calibration only if a marker, the arm or the table has moved. Everything is read-only on
CAN: no script here enables or moves a motor.

"The arm's frame" means positions as the arm's own model reports them (`Fingertip`, MuJoCo FK from
the joint readings). That model reads heights 8-15 mm off. Calibrating with the arm's own readings
absorbs this, because the planner commands the arm through the same model.

## Before the lab

| check | why |
|---|---|
| 4 or more table markers (70 mm, `DICT_ARUCO_ORIGINAL`) taped flat, at least 2 within the arm's reach, at least 1 further than 0.50 m from the base | reach: needed for the touches. Far marker: the camera cannot be located without one in view |
| Measure one printed marker's black square with a ruler: 70.0 ± 0.3 mm | the camera measures every distance in units of the marker size |
| Each marker has a white margin of at least 1 cm, no tape over the black border, no curled edges | corners are found from the border's edges |
| Markers that are not in a line, as far apart as the arm can reach (20 cm or more) | the direction (yaw) of the whole map comes only from the touched points |
| Laptop with the repo, `.venv`, PCAN adapter, RealSense on USB 3, camera stand, orange ball | |
| `git checkout calibration-v2` and `.venv/bin/python -m unittest discover -s tests` passes | 22 tests, 15 s, no hardware |

New marker: add one line to `SIZES` in `eaiproject/markers.py` (`"<id>": 0.070`), then calibrate again.

## Calibrate

```bash
scripts/real/start_camera.sh --width 1280 --height 720 --fps 30
.venv/bin/python scripts/real/gripper_test.py        # joint states print: the arm answers on CAN
.venv/bin/python scripts/real/calibrate.py
```

| step in the window | arm | time |
|---|---|---|
| 1 Views: the camera at 5 or more positions | not needed | 5 min |
| 2 Touch: fingertips on marker corners | by hand, unpowered | 8 min |
| 3 Sweep: fingertips slid over the table | by hand, unpowered | 5 min |
| Solve: result on screen, map and report written | not needed | 10 s |

ENTER goes to the next step, BACKSPACE to the previous one, `1` `2` `3` jump to a step, `q` quits.
Everything is saved after every record, so you can quit at any time and run the script again: it
continues where you stopped. If arm time is short, press `2` and do steps 2 and 3 first; step 1
needs no arm. `calibrate.py --solve` repeats the solve from the saved data without any hardware.

### Step 1: views (camera only)

Arm and hands away from the markers. Put the camera somewhere, wait until it is still, press
SPACE (30 frames are averaged). Move it and repeat.

- **How many:** at least 5, better 6-8.
- **Vary:** left, right and front of the table; low (about 30° above the table) to high (about
  60°); near and far. Each view must show at least 3 markers; most should show all of them.
- A view is refused if the corners moved more than 0.5 px while averaging: the stand was still
  swinging. `u` removes the last view.

### Step 2: corner touches (arm unpowered, moved by hand)

The camera can stay anywhere that sees the markers. Support the arm: joints 2 and 3 drop when
they are not powered.

- The window circles **one corner in yellow** with its name (TL, TR, BR, BL; TL has the filled
  dot). Rest the **closed** fingertips on exactly that corner, hold still, SPACE.
- After the 4 corners press `n` for the next marker (`p` goes back). Pass the markers you cannot
  reach with `n`. `c` records the marker centre (optional), `s` skips one corner, `u` undoes.
- **How many:** all 4 corners of every marker within reach, **at least 2 markers**, 3 if possible.
- A red `CHECK:` line means the new point does not fit the others (wrong corner, or the fingertip
  was not on the table). Press `u` and repeat that corner.

### Step 3: surface sweep (arm unpowered, moved by hand)

Hold the gripper at the blocking tilt: fingers forward and about 30° down. The window shows the
tilt in green while it is within ±5°.

- **Hold SPACE** and slide the closed fingertips slowly over the table. Samples are taken only
  while the key is held. Release before lifting the arm. If holding the key does not record, `r`
  switches recording on and off instead.
- **Where:** the yellow box around the intercept line (x = 0.40 m, y = ±0.26 m) first, then as
  much of the table around it as the arm reaches.
- **How much:** the box at least 70 % green (the window shows the percentage), 150 or more cells
  in total. Two or three slow passes are enough.
- `u` removes the last stroke. Samples at the wrong tilt are kept in the file and left out by the solve.

### Solve

After step 3, ENTER solves and shows the result: the map from above, the three headline numbers and
any warnings. It also writes `config/rig_map.json` and the report folder
`outputs/rig_calib/<date>/report/`. To fix something, go back with BACKSPACE or `1` `2` `3`, redo
that part, and ENTER through to the solve again.

| quantity in the report | accept | if not |
|---|---|---|
| reprojection RMS | < 1.5 px (expect 0.1-0.5) | look at `report/reproj_view_XX.png`. One bad view: step 1, record the views again. All bad: a marker lifted off the table, or the camera stream changed |
| corner touches RMS | < 5 mm (expect 2-3) | `report.txt` lists every touch; redo the ones marked `<-- check` in step 2 (select the marker with `n`, `u`, SPACE) |
| leave-one-marker-out | within about 2.5× the "touch noise explains" value | one marker's touches do not fit the others: touch that marker again |
| touched points span / width | span ≥ 20 cm, width ≥ 5 cm | touch another marker further away, or all 4 corners |
| scale check | within the ± printed next to it | the printed marker is not 70 mm |
| sweep RMS | < 1.5 mm | look at `report/sweep_heatmap.png`: random speckle = lifted fingertips, sweep again. A smooth trend = model error; a warning says so |
| warnings | none | each warning says what to do |

`git add config/rig_map.json && git commit` when the result is clean.

## Every session

```bash
scripts/real/start_camera.sh                                # 640x480 @ 60
.venv/bin/python scripts/real/check_calibration.py          # optional, 2 min
.venv/bin/python scripts/real/real_tracking.py
```

Both tools locate the camera when they start: keep hands and arm off the markers for the first
second. After moving the camera, press `c` in the window.

| message | meaning | do |
|---|---|---|
| `camera located: OK` | < 1.5 px and < 5 mm on the table | go on |
| `camera located: WARN` | 1.5-3 px | usable; check the lighting and that the camera is fixed, press `c` |
| `CANNOT LOCATE THE CAMERA: only N map marker(s) visible` | fewer than 3 markers were read | move the camera closer or higher. At 640x480 a 70 mm marker is read reliably up to about 1.2 m |
| `CANNOT LOCATE THE CAMERA: no marker far from the arm` | only the markers next to the arm are seen | aim further down the table |
| `marker [...] does not fit the map` | one marker moved on the table | fine for today (it is left out); calibrate again before relying on it |
| red **CAMERA MOVED?** in `real_tracking.py` | the markers no longer sit where the pose says (checked every 3 s) | press `c` |

## Live check

`check_calibration.py` shows, on the live image:

- **Markers:** the map (magenta) over the detected outlines (green), with px and mm per marker. A
  `~` after the ID means the marker was found from its predicted position, not read by the detector.
- **Fingertip:** a magenta ring at the arm's reported fingertip. Move the unpowered arm by hand: the
  ring must stay on the real fingertips. Rest them on a marker corner: the error to that corner
  appears; SPACE records it.
- **Ball** (`b`): yellow crosses mark the known points. Put the ball on one: tracked position against
  expected, in mm; SPACE records it. `a` tunes the ball colour if it is not found.
- **Status** top left: OK / WARN / FAIL.

| check | accept | if not |
|---|---|---|
| markers, median | < 1.5 px, < 5 mm | press `c` |
| fingertip on a corner | < 5 mm | if the markers are fine but the fingertip is off everywhere by the same amount, the arm or the markers moved since the calibration: calibrate again |
| ball on a known point | < 5 mm | markers fine but the ball is off: retune the colour (`a`); a ball seen half in shadow shifts by a few mm |

## Powered sweep of the catching area

`catch_sweep.py` moves the jaw along the intercept line, left to right and back, at the blocking
pose: first 60 mm above the calibrated table, then 35 mm, then at the blocking height of 20 mm. It
shows that the arm reaches the whole catching area and that the calibrated table height is right
where it will work. **It powers all six arm joints**, including joints 2 and 3 that carry the
arm's weight. It has run only against simulated motors so far; follow the order below.

```bash
.venv/bin/python scripts/real/catch_sweep.py --dry-run       # 1. plan and checks, nothing powered
.venv/bin/python scripts/real/catch_sweep.py --passes 60     # 2. one pass high above the table
.venv/bin/python scripts/real/catch_sweep.py                 # 3. 60, 35 and 20 mm
```

1. Hold the unpowered arm by hand over the middle of the line, fingers forward and tilted down,
   fingertips about 6 cm above the table. The terminal shows how far it is from the start pose.
2. ENTER: the motors engage at the pose they are in. **Keep your hands on the arm until it prints
   `HOLDING`**, then let go and step back.
3. It moves by itself. `Ctrl+C` stops it; it then holds its position.
4. At the end, or after any stop, it holds and waits. **Support the arm with your hand**, then
   ENTER releases the motors.

It stops and holds, without going limp, when a joint's current leaves what is expected (touching
the table or anything else), when a joint lags its target or sits at its current limit (blocked),
when readings are lost, or when the jaw is pushed. Only a second `Ctrl+C` or an unexpected error
disables the motors at once; the fingertips are then at most 6 cm above the table.

| it prints | meaning | do |
|---|---|---|
| `NOT RUNNING: ... lowest arm point` | in the model the path would come too close to the calibrated table | check the table plane in the calibration report |
| `cannot hold the arm safely here` | the arm sags or a current is near its limit right after engaging | keep holding the arm, ENTER to release. Read the printed currents before raising `--cap` |
| `CONTACT during 'pass at 20 mm' ... fingertip at y = ...` | the arm touched something there. If nothing is in the way, the table is higher there than the map says | look at where the fingertips are. Sweep that area again in `calibrate.py` (step 3) and solve |
| `CONTACT` although nothing is touched | the threshold is too tight for this arm | each pass prints how close the currents came; rerun with `--contact 1.5` |
| `... behind its target: blocked` | something stops a joint | clear the workspace |

The run is logged to `outputs/hardware/catch_sweep_<time>.csv` (targets, positions, currents).

## Validation experiment (course evidence)

3 camera positions × 5 ball positions with known coordinates, error in mm in the arm frame.

1. Once, with the arm: in `check_calibration.py`, rest the fingertips on a spot on the intercept
   line, press `t`, and mark the spot with a pen dot. Do this for 2 spots (for example y = −0.15
   and y = +0.15). They are saved in `outputs/calibration/check_points.json`.
2. The 5 ball positions are these 2 spots and the centres of 3 markers.
3. For each camera position (A: front, B: left, C: right and higher):
   1. move the camera
   2. `check_calibration.py --label camA` (or press `c` in the open window and restart with the new label)
   3. touch 2 marker corners with the fingertips, SPACE each time (checks the arm frame)
   4. `b`, then put the ball on each of the 5 points in turn, SPACE each time, `s` for a screenshot
4. All rows go to `outputs/calibration/validation.csv`:
   `camera, kind, point, expected_x/y, measured_x/y, dx_mm, dy_mm, err_mm, reproj_px`.

What each row tests: a ball on a **marker centre** tests locating the camera and the ball projection
(the marker positions come from the map itself). A ball on a **pen dot** and the **corner touches**
also test that the map sits correctly in the arm frame. Report them separately.

Pass: mean error < 5 mm for every camera position, and no systematic shift between positions.

## What to keep

| file | what it shows |
|---|---|
| `config/rig_map.json` (in git) | the map and its fit statistics |
| `outputs/rig_calib/<date>/report/report.txt`, `layout.png`, `sweep_heatmap.png` | the solve report |
| `outputs/calibration/check_cam*.png` | live-check screenshots, one per camera position |
| `outputs/calibration/validation.csv` | the validation table |

`outputs/` is not in git, and the camera frames in it can show people. Copy only the files above
into the course report.
