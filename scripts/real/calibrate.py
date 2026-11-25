"""Static calibration of the rig, in one guided window: where the table markers and the table
surface are in the arm's frame. Do it once; redo it only if a marker, the arm or the table moved.

    scripts/real/start_camera.sh --width 1280 --height 720 --fps 30
    .venv/bin/python scripts/real/calibrate.py

Three steps, then the solve. ENTER = next step, BACKSPACE = previous, 1 / 2 / 3 = jump to a step,
q = quit. Everything is saved after every record (outputs/rig_calib/<session>/), so you can quit
at any point and run the script again to continue where you stopped.

  1 VIEWS   Move the camera to >= 5 different positions (angle and distance). At each one, with arm
            and hands clear of the markers, SPACE averages every marker corner over 30 frames.
            No arm needed.                                             SPACE record, u undo
  2 TOUCH   Guide the UNPOWERED arm by hand and rest the CLOSED fingertips on the corner circled
            in yellow, SPACE. Corners go TL, TR, BR, BL (TL has the filled dot).
                        SPACE record, n / p next / previous marker, s skip corner, c centre, u undo
  3 SWEEP   Hold the gripper at the blocking tilt (~30 deg down, the window shows it) and slide the
            closed fingertips slowly over the table while HOLDING SPACE (r toggles recording
            instead, if holding the key does not record).     u undo last stroke, l re-lock overlay
  SOLVE     Fits everything, writes config/rig_map.json and a report, and shows the result.

Read-only on CAN: RobStrideBus(movable=()) - nothing is ever enabled or moved.

    .venv/bin/python scripts/real/calibrate.py --solve        # solve again from the saved data, no hardware
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import markers  # noqa: E402
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject import rig_report  # noqa: E402
from eaiproject.rig_draw import (GREEN, GREY, MAGENTA, RED, WHITE, YELLOW, TopDown, ipt, outline, put,  # noqa: E402
                                 window_closed)

MAX_JITTER_PX = 0.5          # a view is rejected if the corners moved more than this while averaging
STILL_MM = 1.5               # a touch is rejected if the fingertip moved more than this in 0.25 s
HOLD_GAP = 1.0               # s; SPACE repeats arriving within this gap count as "key held"
GRID = 0.02                  # m, sweep coverage cell
TARGET_X, TARGET_Y = (rc.GATE_X - 0.06, rc.GATE_X + 0.06), (-0.26, 0.26)      # box to cover around the intercept line


def open_camera():
    from eaiproject.realsense_client import RealSenseClient
    try:
        return RealSenseClient()
    except OSError:
        print("Camera server not running: scripts/real/start_camera.sh --width 1280 --height 720 --fps 30")
        return None


def open_arm():
    from eaiproject.arm_reader import ArmReader
    try:
        return ArmReader()
    except Exception as e:  # noqa: BLE001  (python-can raises several types)
        print(f"Arm not available on CAN: {e}")
        return None


WIN = "rig calibration"
NAV = "ENTER next step   BACKSPACE previous   1/2/3 jump   q quit (everything is saved)"


def nav(key: int) -> str | None:
    """Step navigation shared by all steps: where to go for this key, or None."""
    if key == ord("q") or window_closed(WIN):
        return "quit"
    if key in (13, 10):
        return "next"
    if key in (8, 127):
        return "prev"
    return {ord("1"): "views", ord("2"): "touch", ord("3"): "sweep"}.get(key)


def footer(img, msg: str = "") -> None:
    if msg:
        put(img, msg, (10, img.shape[0] - 34), RED, 0.5)
    put(img, NAV, (10, img.shape[0] - 10), GREY, 0.45)


def grab_average(cam, last_fn, n: int = 30, timeout: float = 4.0):
    """Average table-marker corners over n fresh frames. Returns (mean, jitter, n_frames, last frame)."""
    frames, fn, t_end, g = [], last_fn, time.monotonic() + timeout, None
    while len(frames) < n and time.monotonic() < t_end:
        g = cam.read(timeout=2.0, newer_than=fn)
        fn = g.frame_number
        frames.append(markers.detect(g.img))
    mean, jitter = rc.average_corners(frames)
    return mean, jitter, len(frames), g


# ----------------------------------------------------------------------------------- views
def run_views(cam, sess: Path, n_frames: int) -> str:
    path = sess / "views.json"
    views = json.loads(path.read_text())["views"] if path.exists() else []
    msg, last_fn = "", None
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        found = markers.detect(f.img)
        for k, c in found.items():
            outline(img, c, GREEN, 2, k)
        n = len(views)
        put(img, f"1 VIEWS: {n} recorded (need {rc.MIN_VIEWS}, 6-8 is better). Hands + arm off the markers, camera still, SPACE",
            (10, 24), YELLOW if n < rc.MIN_VIEWS else GREEN)
        put(img, f"markers in view: {len(found)}  {sorted(found)}", (10, 48), GREEN if len(found) >= 3 else RED)
        seen = {k: sum(k in v["corners"] for v in views) for k in markers.SIZES}
        put(img, "views per marker: " + "  ".join(f"{k}: {c}" for k, c in seen.items()), (10, 72))
        put(img, "then move the camera: left / right / front, low (30 deg) to high (60 deg), near and far   (u = undo)",
            (10, 96), WHITE, 0.45)
        if cam.width < 1280:
            put(img, f"camera is {cam.width}x{cam.height}: start it at 1280x720 for this calibration (more accurate)",
                (10, 118), RED, 0.45)
        footer(img, msg)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if nav(key):
            return nav(key)
        if key == ord("u") and views:
            v = views.pop()
            path.write_text(json.dumps({"views": views}, indent=1))
            msg = f"removed {v['name']}"
        if key == ord(" "):
            mean, jitter, nf, g = grab_average(cam, last_fn, n_frames)
            last_fn = g.frame_number
            if len(mean) < 3:
                msg = f"only {len(mean)} table markers stable in view; need at least 3"
                continue
            if max(jitter.values()) > MAX_JITTER_PX:
                msg = f"corners moved {max(jitter.values()):.2f} px while averaging: let the camera settle, SPACE again"
                continue
            name = f"view_{int(views[-1]['name'].split('_')[1]) + 1 if views else 0:02d}"
            cv2.imwrite(str(sess / f"{name}.png"), g.img)
            views.append({"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "serial": cam.serial,
                          "width": cam.width, "height": cam.height, "K": cam.K.tolist(), "dist": cam.dist.tolist(),
                          "n_frames": nf, "corners": {k: c.tolist() for k, c in mean.items()},
                          "jitter_px": {k: round(j, 3) for k, j in jitter.items()}, "frame": f"{name}.png"})
            path.write_text(json.dumps({"views": views}, indent=1))
            msg = ""
            print(f"{name}: {sorted(mean)}  jitter max {max(jitter.values()):.2f} px", flush=True)


# ----------------------------------------------------------------------------------- touch
def touch_sanity(touches: list, new: dict, size: float) -> str:
    """Cheap checks that catch a wrong corner right away (the touch is still recorded; u undoes it)."""
    out = []
    zs = [t["p"][2] for t in touches]
    if len(zs) >= 2 and abs(new["p"][2] - np.median(zs)) > 0.012:
        out.append(f"height differs {1000 * (new['p'][2] - np.median(zs)):+.0f} mm from the other touches")
    loc = rc.local_corners(size)
    for t in touches:
        if t["key"] == new["key"] and t["corner"] != new["corner"]:
            want = np.linalg.norm(loc[t["corner"]] - loc[new["corner"]])
            got = np.linalg.norm(np.array(t["p"]) - new["p"])
            if abs(got - want) > 0.008:
                out.append(f"{1000 * got:.0f} mm from {rc.CORNER_NAMES[t['corner']]} (expected {1000 * want:.0f})")
    return ("CHECK: " + "; ".join(out) + "  (u = undo)") if out else ""


def run_touch(cam, arm, sess: Path) -> str:
    path = sess / "touches.json"
    touches = json.loads(path.read_text())["touches"] if path.exists() else []
    sizes = markers.SIZES
    seen: dict[str, np.ndarray] = {}                # last known corners (the hand hides markers while touching)
    cur, skipped, msg, last_fn = None, set(), "", None

    def done(k):
        return {t["corner"] for t in touches if t["key"] == k}

    def next_corner(k):
        return next((c for c in range(4) if c not in done(k) and (k, c) not in skipped), None)

    def save():
        path.write_text(json.dumps({"touches": touches}, indent=1))

    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        img = f.img.copy()
        for k, c in markers.detect(f.img).items():
            near = k in seen and np.abs(seen[k] - c).max() < 3.0
            seen[k] = 0.7 * seen[k] + 0.3 * c if near else c
        keys = sorted(seen)
        if cur not in keys:
            cur = next((k for k in keys if next_corner(k) is not None), keys[0] if keys else None)
        cc = next_corner(cur) if cur else None
        s = arm.latest()
        for k in keys:
            c = seen[k]
            outline(img, c, YELLOW if k == cur else GREY, 2 if k == cur else 1, k)
            for j in range(4):
                if j in done(k):
                    cv2.circle(img, ipt(c[j]), 5, GREEN, -1)
                elif k == cur:
                    cv2.circle(img, ipt(c[j]), 5, YELLOW, 1)
                    put(img, rc.CORNER_NAMES[j], c[j] + 0.25 * (c[j] - c.mean(axis=0)) + [-8, 4], GREY, 0.4)
            if rc.CENTRE in done(k):
                cv2.circle(img, ipt(c.mean(axis=0)), 5, GREEN, -1)
        if cur and cc is not None:
            tgt = seen[cur][cc]
            cv2.circle(img, ipt(tgt), 16, YELLOW, 2, cv2.LINE_AA)
            cv2.drawMarker(img, ipt(tgt), YELLOW, cv2.MARKER_CROSS, 44, 1, cv2.LINE_AA)
            put(img, f"touch here: {rc.CORNER_NAMES[cc]}", tgt + [20, -14], YELLOW, 0.6, 2)
            put(img, f"2 TOUCH: closed fingertips on corner {rc.CORNER_NAMES[cc]} of marker {cur}, hold still, SPACE", (10, 24), YELLOW)
        elif cur:
            put(img, f"2 TOUCH: marker {cur} done. n = next marker you can reach (p = previous), ENTER when finished", (10, 24), GREEN)
        else:
            put(img, "2 TOUCH: no marker seen yet: aim the camera at the markers", (10, 24), RED)
        y = 50
        for k in keys:
            d = done(k)
            txt = " ".join(n if i in d else "--" for i, n in enumerate(rc.CORNER_NAMES))
            put(img, f"{'>' if k == cur else ' '} {k:4s} {txt}", (10, y), YELLOW if k == cur else (GREEN if len(d) >= 4 else WHITE), 0.45)
            y += 20
        n_mk = sum(1 for k in keys if len(done(k) - {rc.CENTRE}) >= 2)
        put(img, f"{len(touches)} touches on {n_mk} marker(s). All 4 corners of every marker within reach, at least 2 markers",
            (10, y + 4), GREEN if n_mk >= 2 else WHITE, 0.45)
        put(img, "s = skip this corner   c = record the marker centre (optional)   u = undo", (10, y + 24), GREY, 0.42)
        if s is not None:
            put(img, f"fingertip x {s.p[0] * 100:6.1f}  y {s.p[1] * 100:+6.1f} cm  z {s.p[2] * 1000:6.1f} mm",
                (10, img.shape[0] - 58))
        else:
            put(img, "no joint readings", (10, img.shape[0] - 58), RED)
        footer(img, msg)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if nav(key):
            return nav(key)
        if key in (ord("n"), ord("p")) and keys:
            cur = keys[(keys.index(cur) + (-1 if key == ord("p") else 1)) % len(keys)]
            msg = ""
        if key == ord("s") and cur and cc is not None:
            skipped.add((cur, cc))
            msg = f"skipped {rc.CORNER_NAMES[cc]} of {cur}"
        if key == ord("u") and touches:
            t = touches.pop()
            skipped.discard((t["key"], t["corner"]))
            cur = t["key"]
            save()
            msg = f"removed {t['key']} {rc.CORNER_NAMES[t['corner']]}"
        if key in (ord(" "), ord("c")) and cur:
            corner = rc.CENTRE if key == ord("c") else cc
            st = arm.steady()
            if corner is None:
                msg = "all 4 corners of this marker are done: n = next marker, ENTER = next step"
            elif st is None:
                msg = "no joint readings"
            elif st[1] * 1000 > STILL_MM:
                msg = f"the fingertip moved {st[1] * 1000:.1f} mm: hold still and press again"
            else:
                new = {"key": cur, "corner": corner, "p": st[0].p.tolist(), "q": st[0].q.tolist(),
                       "R": st[0].R.ravel().tolist(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                msg = touch_sanity(touches, {**new, "p": st[0].p}, sizes[cur])
                touches[:] = [t for t in touches if (t["key"], t["corner"]) != (cur, corner)] + [new]
                save()
                print(f"touch {cur} {rc.CORNER_NAMES[corner]}: ({st[0].p[0]:.4f}, {st[0].p[1]:+.4f}, {st[0].p[2]:.4f}) m  {msg}",
                      flush=True)


# ----------------------------------------------------------------------------------- sweep
def provisional_pose(cam, last_fn, sess: Path):
    """A camera pose in the arm frame for the live overlay only: from the committed map if there is
    one, else a quick solve of this single view with the session's corner touches."""
    mean, _, _, g = grab_average(cam, last_fn, 10, 2.0)
    try:
        if rc.MAP_PATH.exists():
            res = rc.relocalize(rc.RigMap.load(rc.MAP_PATH), cam.K, cam.dist, mean, require_far=False)
            if res.status != "FAIL":
                return (res.rvec, res.tvec), "config/rig_map.json", g.frame_number
        obs, _ = rc.load_observations(sess)
        obs.views = [rc.View("live", cam.K, cam.dist, cam.width, cam.height, mean)]
        obs.sweep = np.zeros((0, 3))
        sol = rc.solve_rig(obs, leave_one_out=False)
        return sol.cams["live"], "this session's touches", g.frame_number
    except (rc.CalibrationError, cv2.error) as e:
        return None, str(e), g.frame_number


def run_sweep(cam, arm, sess: Path, ref_tilt: float, max_dev: float) -> str:
    path = sess / "sweep.npz"
    strokes: list[list] = []                       # each stroke: list of ArmSample-like tuples (t, q, p, R)
    if path.exists():
        z = np.load(path)
        strokes.append([(t, q, p, R) for t, q, p, R in zip(z["t"], z["q"], z["p"], z["R"])])
    top = TopDown(x=(0.10, 0.75), y=(-0.40, 0.40), scale=600.0)
    pose, pose_src, last_fn = provisional_pose(cam, None, sess)
    msg = "" if pose else f"no camera pose for the image overlay ({pose_src}); use the top view on the right"
    toggled, t_key, t_commit, stroke_open = False, -1e9, None, False

    def save():
        s = [x for st in strokes for x in st]
        if s:
            np.savez(path, t=np.array([x[0] for x in s]), q=np.array([x[1] for x in s]),
                     p=np.array([x[2] for x in s]), R=np.array([x[3] for x in s]), ref_tilt_deg=ref_tilt)

    def commit(t0, t1):
        new = [(s.t, s.q, s.p, s.R) for s in arm.between(t0, t1)]
        if new:
            strokes[-1].extend(new)

    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        now = time.monotonic()
        img = f.img.copy()
        s = arm.latest()
        holding = toggled or now - t_key < HOLD_GAP
        if toggled and stroke_open:                # toggle mode: commit continuously
            commit(t_commit, now)
            t_commit = now
        if not holding and stroke_open:            # key released: samples after the last key event are dropped
            stroke_open = False
            if not strokes[-1]:
                strokes.pop()
            save()

        allp = np.array([x[2] for st in strokes for x in st]).reshape(-1, 3)
        allR = np.array([x[3] for st in strokes for x in st]).reshape(-1, 3, 3)
        ok = rc.tilt_deviation_deg(allR, ref_tilt) <= max_dev if len(allp) else np.zeros(0, bool)
        cells = {(int(np.floor(p[0] / GRID)), int(np.floor(p[1] / GRID))) for p in allp[ok]}
        tx = range(int(np.floor(TARGET_X[0] / GRID)), int(np.ceil(TARGET_X[1] / GRID)))
        ty = range(int(np.floor(TARGET_Y[0] / GRID)), int(np.ceil(TARGET_Y[1] / GRID)))
        cover = sum((i, j) in cells for i in tx for j in ty) / (len(tx) * len(ty))

        # top view (always available: it needs no camera pose)
        tv = top.canvas(rc.GATE_X)
        top.rect(tv, TARGET_X, TARGET_Y, YELLOW, 1)
        for (i, j) in cells:
            top.cell(tv, ((i + 0.5) * GRID, (j + 0.5) * GRID), GRID, (0, 170, 0))
        for p in allp[~ok][::5]:
            cv2.circle(tv, ipt(top.px(p[:2])[0]), 1, GREY, -1)
        if s is not None:
            cv2.drawMarker(tv, ipt(top.px(s.p[:2])[0]), MAGENTA, cv2.MARKER_CROSS, 18, 2)
        put(tv, "top view, arm frame (m). green = covered, grey = wrong tilt", (8, 16), WHITE, 0.42)
        put(tv, "yellow box = cover this well", (8, 34), YELLOW, 0.42)

        # camera image overlay
        if pose is not None:
            pr = lambda P: rc.project(cam.K, cam.dist, *pose, np.asarray(P, float).reshape(-1, 3))
            z0 = float(np.median(allp[:, 2])) if len(allp) else (float(s.p[2]) if s is not None else 0.0)
            box = pr([[TARGET_X[0], TARGET_Y[0], z0], [TARGET_X[1], TARGET_Y[0], z0],
                      [TARGET_X[1], TARGET_Y[1], z0], [TARGET_X[0], TARGET_Y[1], z0]])
            cv2.polylines(img, [box.round().astype(np.int32)], True, YELLOW, 1, cv2.LINE_AA)
            a, b = pr([[rc.GATE_X, -0.30, z0], [rc.GATE_X, 0.30, z0]])
            cv2.line(img, ipt(a), ipt(b), WHITE, 1, cv2.LINE_AA)
            if cells:
                cc = np.array([[(i + 0.5) * GRID, (j + 0.5) * GRID, z0] for i, j in cells])
                for c in pr(cc):
                    cv2.circle(img, ipt(c), 3, (0, 200, 0), -1)
            if s is not None:
                cv2.drawMarker(img, ipt(pr(s.p)[0]), MAGENTA, cv2.MARKER_CROSS, 18, 2)
        if s is not None:
            tilt = float(rc.tilt_deg(s.R)[0])
            dev = float(rc.tilt_deviation_deg(s.R, ref_tilt)[0])
            good = dev <= max_dev
            put(img, f"tilt {tilt:4.0f} deg (keep {ref_tilt:.0f} +- {max_dev:.0f}); off by {dev:3.0f} deg   "
                     f"z {s.p[2] * 1000:5.1f} mm", (10, 50), GREEN if good else RED, 0.6, 2 if not good else 1)
        else:
            put(img, "no joint readings", (10, 50), RED)
        put(img, ("3 SWEEP: RECORDING" if holding else "3 SWEEP: hold SPACE and slide the closed fingertips over the table"),
            (10, 24), RED if holding else YELLOW, 0.65, 2 if holding else 1)
        put(img, f"{len(allp)} samples ({int(ok.sum())} at the right tilt)   {len(cells)} cells   "
                 f"box around the intercept line: {100 * cover:.0f} % covered (aim >= 70 %)",
            (10, 76), GREEN if cover >= 0.7 else WHITE, 0.5)
        if pose is not None:
            put(img, f"overlay pose from {pose_src} (l = re-lock with hands away)", (10, 98), GREY, 0.42)
        put(img, "r = recording on/off without holding   u = undo the last stroke", (10, 118), GREY, 0.42)
        footer(img, msg)
        h = img.shape[0]
        tv = cv2.resize(tv, (int(tv.shape[1] * h / tv.shape[0]), h))
        cv2.imshow(WIN, np.hstack([img, tv]))

        key = cv2.waitKey(1) & 0xFF
        if nav(key) and not holding:
            if stroke_open and not strokes[-1]:
                strokes.pop()
            save()
            return nav(key)
        if key == ord(" ") and not toggled:
            tk = time.monotonic()
            if not stroke_open:
                strokes.append([])
                stroke_open = True
            else:
                commit(t_commit, tk)               # everything up to this key event was with the key down
            t_key = t_commit = tk
        if key == ord("r"):
            toggled = not toggled
            if toggled:
                strokes.append([])
                stroke_open, t_commit = True, time.monotonic()
            msg = "recording ON (r stops)" if toggled else ""
        if key == ord("u") and strokes and not holding:
            n = len(strokes.pop())
            if not strokes and path.exists():
                path.unlink()
            save()
            msg = f"removed the last stroke ({n} samples)"
        if key == ord("l") and not holding:
            pose, pose_src, last_fn = provisional_pose(cam, last_fn, sess)
            msg = "" if pose else f"no camera pose: {pose_src}"


# ----------------------------------------------------------------------------------- solve
def solve(sess: Path, args) -> tuple[rc.RigSolution | None, str]:
    """Solve from the saved observations; write the report and config/rig_map.json."""
    ref = None if args.ref_tilt == "median" else float(args.ref_tilt)
    obs, sweep_stats = rc.load_observations(sess, ref, args.max_dev)
    obs.sources["session"] = str(sess.relative_to(ROOT)) if sess.is_relative_to(ROOT) else str(sess)
    try:
        sol = rc.solve_rig(obs, far_dist=args.far_dist)
    except rc.CalibrationError as e:
        return None, f"CANNOT SOLVE: {e}"
    if sweep_stats.get("n_raw") and sweep_stats["n_tilt_rejected"] > 0.5 * sweep_stats["n_raw"]:
        sol.warnings.append(f"{sweep_stats['n_tilt_rejected']} of {sweep_stats['n_raw']} sweep samples were more than "
                            f"{args.max_dev:.0f} deg from the {sweep_stats['ref_tilt_deg']:.0f} deg tilt and are not used "
                            "(--ref-tilt median uses the tilt that was actually held)")
    out, text = rig_report.save_report(sol, obs, sweep_stats, sess)
    sol.rig_map.save(rc.MAP_PATH)
    text += f"\n\nreport: {out}\nwrote {rc.MAP_PATH}  (commit it)\nnext: scripts/real/check_calibration.py"
    return sol, text


def show_result(sess: Path, sol, text: str) -> str:
    """The solve result in the window: the layout picture, the headline numbers and the warnings."""
    if sol is None:
        img = np.full((720, 1280, 3), 32, np.uint8)
        put(img, text[:150], (20, 60), RED, 0.6)
        lines = [text[150:300]]
    else:
        img = cv2.imread(str(sess / "report" / "layout.png"))
        r = sol.report
        ok = not sol.warnings
        put(img, "CALIBRATION OK - config/rig_map.json written" if ok else "MAP WRITTEN, BUT CHECK THE WARNINGS",
            (20, 70), GREEN if ok else YELLOW, 0.8, 2)
        lines = [f"reprojection {r['reproj_rms_px']} px (< {rc.OK_PX})   touches {r['touch_rms_mm']} mm (< {rc.OK_MM})   "
                 f"sweep {r['sweep_rms_mm']} mm (< 1.5)"] + [f"! {w}" for w in sol.warnings]
    y = 100
    for line in lines:
        for k in range(0, len(line), 125):                  # wrap long warnings
            put(img, line[k:k + 125], (20, y), WHITE if not line.startswith("!") else YELLOW, 0.45)
            y += 20
    put(img, f"full report: {sess / 'report' / 'report.txt'}", (60, img.shape[0] - 64), GREY, 0.45)
    put(img, "BACKSPACE or 1 / 2 / 3 = go back and fix      ENTER or q = done", (60, img.shape[0] - 42), WHITE, 0.5)
    while True:
        cv2.imshow(WIN, img)
        go = nav(cv2.waitKey(30) & 0xFF)
        if go:
            return "quit" if go == "next" else go


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="folder under outputs/rig_calib/ (default: today; with --solve: the newest)")
    ap.add_argument("--solve", action="store_true", help="only solve from the saved observations (no camera, no arm)")
    ap.add_argument("--frames", type=int, default=30, help="frames averaged per view")
    ap.add_argument("--ref-tilt", default=str(rc.BLOCK_TILT_DEG),
                    help="blocking tilt of the gripper in deg, or 'median' = the tilt actually held in the sweep")
    ap.add_argument("--max-dev", type=float, default=5.0, help="sweep: allowed orientation deviation, deg")
    ap.add_argument("--far-dist", type=float, default=rc.FAR_DIST,
                    help="markers farther than this from the base count as 'far' when locating the camera (m)")
    args = ap.parse_args()
    if args.solve:
        sess = rc.SESSIONS_DIR / args.session if args.session else rc.latest_session()
        if sess is None or not sess.exists():
            print(f"no saved session under {rc.SESSIONS_DIR}")
            return 2
        sol, text = solve(sess, args)
        print(text)
        return 0 if sol is not None else 1

    sess = rc.SESSIONS_DIR / (args.session or time.strftime("%Y%m%d"))
    sess.mkdir(parents=True, exist_ok=True)
    cam = open_camera()
    if cam is None:
        return 2
    print(f"session: {sess}")
    tilt = rc.BLOCK_TILT_DEG if args.ref_tilt == "median" else float(args.ref_tilt)
    n_views = len(json.loads((sess / "views.json").read_text())["views"]) if (sess / "views.json").exists() else 0
    step = "views" if n_views < rc.MIN_VIEWS else "touch" if not (sess / "touches.json").exists() else "sweep"
    order = ["views", "touch", "sweep", "solve"]
    arm, code = None, 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    try:
        while step != "quit":
            if step in ("touch", "sweep") and arm is None:
                arm = open_arm()
                if arm is None:
                    print("Steps 2 and 3 need the arm on CAN (powered, not enabled). The views are saved; run this again.")
                    code = 2
                    break
            if step == "views":
                go = run_views(cam, sess, args.frames)
            elif step == "touch":
                go = run_touch(cam, arm, sess)
            elif step == "sweep":
                go = run_sweep(cam, arm, sess, tilt, args.max_dev)
            else:
                sol, text = solve(sess, args)
                print("\n" + text, flush=True)
                code = 0 if sol is not None else 1
                go = show_result(sess, sol, text)
            i = order.index(step)
            step = go if go in order or go == "quit" else order[min(i + 1, 3)] if go == "next" else order[max(i - 1, 0)]
    finally:
        cv2.destroyAllWindows()
        cam.close()
        if arm is not None:
            arm.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
