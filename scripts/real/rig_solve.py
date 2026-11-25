"""Offline solve of the static rig calibration from saved observations (no hardware).

Reads outputs/rig_calib/<session>/ (views.json, touches.json, sweep.npz from rig_capture.py), fits
the table plane, every table marker and one camera pose per view in the arm's kinematic frame, and
writes:

    config/rig_map.json                       marker corners in the arm frame (commit this file)
    outputs/calibration/table_height.json     table plane from the sweep, in the format the planner reads
    outputs/rig_calib/<session>/report/       report.json, report.txt, layout.png, sweep_heatmap.png,
                                              reproj_<view>.png (the fitted markers over each view)

    .venv/bin/python scripts/real/rig_solve.py                   # newest session
    .venv/bin/python scripts/real/rig_solve.py --session 20261126 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import rig_calib as rc  # noqa: E402
from eaiproject.rig_draw import (GREEN, GREY, MAGENTA, WHITE, YELLOW, TopDown, colorbar, diverging, ipt,  # noqa: E402
                                 outline, put)


def report_text(sol: rc.RigSolution, sweep_stats: dict, th: dict | None) -> str:
    r = sol.report
    g = r["touch_geometry"]
    L = [f"=== rig calibration ({sol.rig_map.time}) ===",
         f"{r['n_views']} views, {r['n_markers']} table markers, {r['n_touches']} corner touches, "
         f"{r['n_sweep_points']} sweep cells",
         "",
         f"reprojection: RMS {r['reproj_rms_px']} px, max {r['reproj_max_px']} px   (accept: RMS < {rc.OK_PX} px)",
         "  per marker: " + "  ".join(f"{k} {v}" for k, v in r["per_marker_reproj_px"].items()),
         "  per view:   " + "  ".join(f"{k} {v['reproj_px']}" for k, v in r["per_view"].items()),
         "",
         f"corner touches: RMS {r['touch_rms_mm']} mm (in the table plane {r['touch_xy_rms_mm']} mm), "
         f"mean height offset {r['touch_dz_mean_mm']:+} mm   (accept: RMS < {rc.OK_MM} mm)"]
    for t in r["touches"]:
        flag = "   <-- check" if t["err_mm"] > 2 * rc.OK_MM else ""
        L.append(f"  {t['marker']:9s} {t['corner']:2s}  dx {t['dx_mm']:+6.2f}  dy {t['dy_mm']:+6.2f}  dz {t['dz_mm']:+6.2f}  "
                 f"|{t['err_mm']:5.2f}| mm{flag}")
    if r.get("leave_one_marker_out"):
        L += ["", "leave-one-marker-out (marker predicted without its own touches):"]
        for k, v in r["leave_one_marker_out"].items():
            L.append(f"  {k:9s} in-plane {v['xy_mm']} mm (touch noise explains ~{v['expected_xy_mm']}), worst corner "
                     f"{v['max_mm']} mm, height {v['dz_mm']:+} mm")
    L += ["",
          f"touched points: span {g['span_m'] * 100:.0f} cm, width {g['width_m'] * 100:.0f} cm, {g['n_points']} points "
          f"-> map yaw 1-sigma {g['yaw_sigma_deg']:.2f} deg = {g['sigma_mm_per_m']:.1f} mm per metre from the touches",
          f"scale check: touched points are {r['touch_scale_percent']:+.2f} % wider than the camera's layout "
          f"(noise allows about +-{g['sigma_mm_per_m'] / 10:.2f} %)",
          f"table plane: z = {r['plane_z=a+bx+cy'][0] * 1000:.1f} mm at the base, tilt {r['plane_tilt_deg_x_y'][0]:+.2f} deg "
          f"along x, {r['plane_tilt_deg_x_y'][1]:+.2f} deg along y",
          f"sweep: RMS {r['sweep_rms_mm']} mm, max {r['sweep_max_mm']} mm   {sweep_stats or ''}",
          f"far markers (needed for relocalisation): {r['far_markers']}"]
    if th is not None:
        c = th["curvature"]
        L += [f"table height along the intercept line (mm): {th['z_at_gate_mm']}",
              f"curvature: plane RMS {c['rms_plane_mm']:.2f} mm, quadratic RMS "
              f"{c['rms_quadratic_mm'] if c['rms_quadratic_mm'] is None else round(c['rms_quadratic_mm'], 2)} mm, bend "
              f"{c['bend_mm']:.1f} mm -> " + ("SMOOTH CURVATURE: quadratic saved as an extra field (planner still uses the plane)"
                                             if c["significant"] else "plane is enough")]
    L += ["", "WARNINGS:" if sol.warnings else "no warnings"] + [f"  - {w}" for w in sol.warnings]
    ok = not sol.warnings and r["reproj_rms_px"] < rc.OK_PX and r["touch_rms_mm"] < rc.OK_MM
    L += ["", f"RESULT: {'OK' if ok else 'CHECK THE WARNINGS'}"]
    return "\n".join(L)


def layout_image(sol: rc.RigSolution, obs: rc.Observations) -> np.ndarray:
    """Top view: fitted markers, touches (errors drawn 20x), camera positions, swept area."""
    m = sol.rig_map
    top = TopDown()
    img = top.canvas(rc.GATE_X)
    for p in obs.sweep:
        top.cell(img, p[:2], 0.02, (60, 80, 60))
    top.markers(img, m)
    for p, r in zip(sol.touch_points, sol.touch_residuals):
        a = top.px(p[:2] - r[:2])[0]                      # fitted corner
        b = top.px(p[:2] - r[:2] + 20 * r[:2])[0]
        cv2.circle(img, ipt(a), 3, GREEN, -1)
        cv2.line(img, ipt(a), ipt(b), MAGENTA, 1, cv2.LINE_AA)
    for name, (rv, tv) in sol.cams.items():
        C = rc.camera_centre(rv, tv)
        c = top.px(np.clip(C[:2], [top.x[0], top.y[0]], [top.x[1], top.y[1]]))[0]
        cv2.drawMarker(img, ipt(c), WHITE, cv2.MARKER_TRIANGLE_UP, 10, 1)
        put(img, f"{name.split('_')[-1]} h{C[2]:.2f}", c + [6, 4], GREY, 0.35)
    put(img, "rig map, arm frame (m). yellow = touched markers, magenta = touch error x20, triangles = camera views",
        (8, 16), WHITE, 0.42)
    put(img, f"reproj {sol.report['reproj_rms_px']} px   touches {sol.report['touch_rms_mm']} mm   "
             f"sweep {sol.report['sweep_rms_mm']} mm", (8, 34), WHITE, 0.42)
    return img


def heatmap_image(sol: rc.RigSolution, obs: rc.Observations, vmax_mm: float = 3.0) -> np.ndarray:
    """Sweep height residuals against the fitted plane, over the table."""
    top = TopDown(x=(0.10, 0.75), y=(-0.40, 0.40), scale=900.0)
    img = top.canvas(rc.GATE_X)
    for p, r in zip(obs.sweep, sol.sweep_residuals):
        top.cell(img, p[:2], 0.02, diverging(r * 1000, vmax_mm))
    top.markers(img, sol.rig_map)
    colorbar(img, vmax_mm, (img.shape[1] - 80, 50))
    put(img, f"sweep height minus the table plane (mm); RMS {sol.report['sweep_rms_mm']} mm, "
             f"{len(obs.sweep)} cells of 2 cm", (8, 16), WHITE, 0.45)
    put(img, "a smooth trend (one side red, one blue, or a bowl) = model error, not noise", (8, 34), WHITE, 0.42)
    return img


def reproj_images(sol: rc.RigSolution, obs: rc.Observations, sess: Path, out: Path) -> int:
    n = 0
    for v in obs.views:
        f = sess / f"{v.name}.png"
        if not f.exists():
            continue
        img = cv2.imread(str(f))
        rv, tv = sol.cams[v.name]
        for k, c in v.corners.items():
            outline(img, c, GREEN, 1)
        for k, P in sol.rig_map.markers.items():
            px = rc.project(v.K, v.dist, rv, tv, P)
            outline(img, px, MAGENTA, 1, f"{k} {sol.report['per_marker_reproj_px'][k]}px" if k in v.corners else k)
        put(img, f"{v.name}: green = detected, magenta = fitted map; RMS {sol.report['per_view'][v.name]['reproj_px']} px",
            (10, 24), YELLOW)
        cv2.imwrite(str(out / f"reproj_{v.name}.png"), img)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="folder under outputs/rig_calib/ (default: the newest one)")
    ap.add_argument("--ref-tilt", default=str(rc.BLOCK_TILT_DEG),
                    help="sweep reference tilt in deg, or 'median' to use the sweep's own median tilt")
    ap.add_argument("--max-dev", type=float, default=5.0, help="sweep: allowed orientation deviation, deg")
    ap.add_argument("--far-dist", type=float, default=rc.FAR_DIST,
                    help="markers farther than this from the base count as 'far' for relocalisation (m)")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write the map or table_height.json")
    args = ap.parse_args()
    sess = rc.SESSIONS_DIR / args.session if args.session else rc.latest_session()
    if sess is None or not sess.exists():
        print(f"no capture session found under {rc.SESSIONS_DIR}; run scripts/real/rig_capture.py first")
        return 2
    ref = None if args.ref_tilt == "median" else float(args.ref_tilt)
    obs, sweep_stats = rc.load_observations(sess, ref, args.max_dev)
    rel = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    obs.sources["session"] = rel(sess)
    print(f"session {sess}: {len(obs.views)} views, {len(obs.touches)} touches, sweep {sweep_stats or 'none'}")
    if sweep_stats and sweep_stats["n_raw"] and sweep_stats["n_tilt_rejected"] > 0.5 * sweep_stats["n_raw"]:
        print(f"NOTE: {sweep_stats['n_tilt_rejected']} of {sweep_stats['n_raw']} sweep samples are more than "
              f"{args.max_dev:.0f} deg from the {sweep_stats['ref_tilt_deg']:.0f} deg tilt (try --ref-tilt median to see "
              "what tilt was actually held)")
    try:
        sol = rc.solve_rig(obs, far_dist=args.far_dist)
    except rc.CalibrationError as e:
        print(f"\nCANNOT SOLVE: {e}")
        return 1
    th = None
    if len(obs.sweep) >= 10 or len(sol.touch_points) >= 3:
        th = rc.table_height_dict(obs.sweep, rc.GATE_X, sol.touch_points)
        th["rig_map_time"] = sol.rig_map.time
        d = [abs(float(rc.plane_z(th["plane_z=a+bx+cy"], rc.GATE_X, y) - rc.plane_z(sol.rig_map.plane, rc.GATE_X, y)))
             for y in (-0.24, 0.0, 0.24)]
        th["max_diff_to_map_plane_at_gate_mm"] = round(max(d) * 1000, 2)
    text = report_text(sol, sweep_stats, th)
    print("\n" + text)

    out = sess / "report"
    out.mkdir(exist_ok=True)
    (out / "report.txt").write_text(text + "\n")
    (out / "report.json").write_text(json.dumps({**sol.report, "sweep_filter": sweep_stats, "table_height": th}, indent=2))
    cv2.imwrite(str(out / "layout.png"), layout_image(sol, obs))
    if len(obs.sweep):
        cv2.imwrite(str(out / "sweep_heatmap.png"), heatmap_image(sol, obs))
    n = reproj_images(sol, obs, sess, out)
    print(f"\nreport: {out} (report.txt, report.json, layout.png, sweep_heatmap.png, {n} reprojection overlays)")
    if args.dry_run:
        print("dry run: map and table height NOT written")
        return 0
    sol.rig_map.save(rc.MAP_PATH)
    print(f"wrote {rel(rc.MAP_PATH)}  (commit it)")
    if th is not None:
        rc.write_json_with_backup(rc.CAL_DIR / "table_height.json", th)
        print(f"wrote {rel(rc.CAL_DIR / 'table_height.json')}")
    print("next: scripts/real/relocalize.py (camera at the tracking resolution), then scripts/real/check_calibration.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
