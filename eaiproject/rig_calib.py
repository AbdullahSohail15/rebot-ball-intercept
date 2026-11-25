"""Camera-to-arm calibration that survives camera moves: a static map of the table markers in the
arm's KINEMATIC base frame, plus relocalisation of the camera against that map.

    static (once):   multi-view marker photos + fingertip touches on marker corners + a fingertip
                     sweep of the table  ->  solve_rig()  ->  RigMap (config/rig_map.json)
    online (always): one PnP on all visible table-marker corners against the map -> relocalize()

"Kinematic frame" means the frame of eaiproject.kinematics.Fingertip (MuJoCo FK from joint
readings). The model reads heights 8-15 mm off, and measuring with the arm's own FK absorbs that,
because the planner commands targets through the same model.

Only the TABLE markers are used (keys starting with TABLE_PREFIX in eaiproject/markers.py, the
single source of truth for dictionaries, IDs and sizes). The plate markers are ignored everywhere.

Geometry conventions
    table plane      z = a + b x + c y in the base frame; the normal n points up (+z side)
    plane frame      origin (0, 0, a), e1 = base x projected into the plane, e2 = n x e1
    marker           (u, v, yaw) in the plane frame; corners in OpenCV ArUco order TL, TR, BR, BL
                     (marker x right, y up, z out of the marker = n); corner index 4 = centre
    camera           OpenCV: X_cam = R(rvec) X_base + tvec
    ball             centre on the plane table + r, on the camera's side (r > 0 along n)

Pure functions, no hardware imports.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import markers as _markers

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "config" / "rig_map.json"                 # committed: marker corners in the arm frame
CAL_DIR = ROOT / "outputs" / "calibration"                  # camera_extrinsics.json, table_height.json
SESSIONS_DIR = ROOT / "outputs" / "rig_calib"               # raw observations per capture session (git-ignored)
GATE_X = 0.40                                               # intercept line of the "forward" mode
BLOCK_TILT_DEG = 30.0                                       # gripper tilt of the blocking pose

TABLE_PREFIX = "orig:"            # table markers; "4x4:" are the plate markers (not used)
CORNER_NAMES = ("TL", "TR", "BR", "BL", "C")
CENTRE = 4
BALL_R = 0.022

SIGMA_PX = 0.5                    # averaged corner, incl. intrinsics / sub-pixel bias
SIGMA_TOUCH = 0.002               # fingertip resting on a marker corner (m)
SIGMA_SWEEP = 0.0015              # fingertip height while sliding on the table (m)
F_SCALE = 2.0                     # robust-loss scale, in sigmas

MIN_VIEWS = 5
MIN_TOUCH_SPAN = 0.20             # m; yaw of the whole map comes only from the touches
MIN_TOUCH_WIDTH = 0.05            # m; narrower than this across the main axis = "collinear"
FAR_DIST = 0.50                   # m from the base: a marker beyond this counts as "far from the arm"
OK_PX, FAIL_PX = 1.5, 3.0
OK_MM, FAIL_MM = 5.0, 10.0


class CalibrationError(ValueError):
    """The data cannot give a trustworthy result (too few markers, wrong sign, ...)."""


# --------------------------------------------------------------------------------- markers
def is_table(key: str) -> bool:
    return key.startswith(TABLE_PREFIX)


def table_sizes(sizes: dict | None = None) -> dict[str, float]:
    """Table markers and their side length (m), from eaiproject/markers.py."""
    return {k: float(s) for k, s in (_markers.SIZES if sizes is None else sizes).items() if is_table(k)}


def table_only(found: dict, sizes: dict | None = None) -> dict:
    """Keep only known table markers from a markers.detect() result."""
    known = table_sizes(sizes)
    return {k: np.asarray(v, float).reshape(4, 2) for k, v in found.items() if k in known}


def local_corners(size: float) -> np.ndarray:
    """(5, 2): TL, TR, BR, BL, centre in the marker's own frame (x right, y up)."""
    h = size / 2
    return np.array([[-h, h], [h, h], [h, -h], [-h, -h], [0.0, 0.0]])


def _refine(gray: np.ndarray, corners, modules: int = 7, max_shift: float = 3.0, passes: int = 2) -> np.ndarray | None:
    """Marker corners from straight lines fitted to the marker's four outer edges.

    The detector's cornerSubPix pulls the corner of a blurred black square inwards (0.3-0.8 px in
    our images), so markers look 1-3 % too small and a map built from them is stretched by several
    mm at the far markers. Edge positions found across each side (gradient centroid) do not have
    that bias; the corners are the intersections of the four fitted lines. `modules`: marker width
    in cells incl. the black border (7 for DICT_ARUCO_ORIGINAL), which limits the search band so it
    never reaches the first inner cell. Returns None if anything looks wrong.
    """
    c0 = np.asarray(corners, np.float64).reshape(4, 2)
    c = c0
    for _ in range(passes):
        ctr, lines = c.mean(axis=0), []
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            if L < 10:
                return None
            t = (b - a) / L
            nrm = np.array([-t[1], t[0]])
            if nrm @ (ctr - a) > 0:
                nrm = -nrm                                             # pointing out of the marker
            depth = 2 * abs(float(nrm @ (ctr - a)))                    # marker extent across this side (foreshortened)
            half = float(np.clip(0.6 * depth / modules, 1.0, 4.0))
            offs = np.arange(-half - 1, half + 1.01, 0.5)              # samples across the edge
            base = a + np.linspace(0.15, 0.85, max(6, int(L * 0.7 / 1.5)))[:, None] * (b - a)
            pts = (base[:, None, :] + offs[None, :, None] * nrm).astype(np.float32)
            prof = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR).astype(np.float64)
            g = np.clip(np.diff(prof, axis=1), 0, None)                # black -> white going outwards
            wsum = g.sum(axis=1)
            ok = wsum > 0.25 * np.median(wsum)
            if np.median(wsum) <= 0 or ok.sum() < 4:
                return None
            e = (g[ok] * (0.5 * (offs[:-1] + offs[1:]))).sum(axis=1) / wsum[ok]
            vx, vy, x0, y0 = cv2.fitLine((base[ok] + e[:, None] * nrm).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
        out = np.empty((4, 2))
        for i in range(4):
            (p1, d1), (p2, d2) = lines[i - 1], lines[i]
            A = np.column_stack([d1, -d2])
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            out[i] = p1 + np.linalg.solve(A, p2 - p1)[0] * d1
        if not np.all(np.isfinite(out)) or np.abs(out - c0).max() > max_shift:
            return None
        c = out
    return c


def refine_corners(gray: np.ndarray, corners, **kw) -> np.ndarray:
    """Edge-refined corners (see _refine); the input unchanged when the refinement fails."""
    out = _refine(gray, corners, **kw)
    return np.asarray(corners, np.float64).reshape(4, 2) if out is None else out


def _edge_contrast(gray: np.ndarray, c: np.ndarray, modules: int = 7) -> float:
    """Mean grey level just outside the quad minus just inside it: large and positive for a real
    marker (white margin around a black border), small when something covers it."""
    ctr, diff = c.mean(axis=0), []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        t = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        nrm = np.array([-t[1], t[0]])
        if nrm @ (ctr - a) > 0:
            nrm = -nrm
        d = max(1.0, abs(float(nrm @ (ctr - a))) / modules)          # half a border width
        base = a + np.linspace(0.2, 0.8, 8)[:, None] * (b - a)
        pts = np.stack([base + d * nrm, base - d * nrm]).astype(np.float32)
        v = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR).astype(float)
        diff.append(v[0].mean() - v[1].mean())
    return float(min(diff))


def guided_corners(gray: np.ndarray, rig_map: "RigMap", K, dist, rvec, tvec, have=(), min_side_px: float = 10.0,
                   max_shift: float = 3.0, min_contrast: float = 35.0) -> dict[str, np.ndarray]:
    """Corners of map markers the detector did NOT report, found from a pose we already have.

    The pose (from the markers that were detected, or from the last relocalisation) predicts where
    every other marker's outline is; the edge fit then locks onto the real edges near that
    prediction. This recovers small, far or blurred markers whose bit pattern the ArUco detector
    cannot read. A marker is accepted only if all four edges are found within `max_shift` px of
    the prediction with a clear white-outside / black-inside contrast (so a hand, the ball or the
    arm over a marker gives nothing).
    """
    h, w = gray.shape[:2]
    out = {}
    for k, P in rig_map.markers.items():
        if k in have:
            continue
        if np.any((cv2.Rodrigues(np.asarray(rvec, float))[0] @ P.T).T[:, 2] + np.asarray(tvec, float).reshape(3)[2] <= 0):
            continue                                                   # behind the camera
        pred = project(K, dist, rvec, tvec, P)
        sides = np.linalg.norm(pred - np.roll(pred, -1, axis=0), axis=1)
        if sides.min() < min_side_px or pred.min() < 6 or pred[:, 0].max() > w - 7 or pred[:, 1].max() > h - 7:
            continue
        c = _refine(gray, pred, max_shift=max_shift)
        if c is not None and _edge_contrast(gray, c) >= min_contrast:
            out[k] = c
    return out


def detect_table(img_bgr: np.ndarray, sizes: dict | None = None) -> dict[str, np.ndarray]:
    """Table markers in an image with edge-refined corners: key -> (4, 2). Use this (not the raw
    markers.detect) for everything that builds or uses the rig map, so both see the same corners."""
    found = table_only(_markers.detect(img_bgr), sizes)
    if not found:
        return found
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {k: refine_corners(gray, c) for k, c in found.items()}


def detect_with_map(img_bgr: np.ndarray, rig_map: "RigMap", K, dist, pose=None) -> tuple[dict, list]:
    """detect_table() plus guided_corners() for the map markers it missed. `pose` = (rvec, tvec) to
    predict from; None = a PnP on the markers detected in this image (needs 2 of them).
    Returns (key -> corners, keys that were found by prediction)."""
    found = {k: c for k, c in detect_table(img_bgr, rig_map.sizes).items() if k in rig_map.markers}
    if pose is None:
        if len(found) < 2:
            return found, []
        try:
            pose = _pnp(np.concatenate([rig_map.markers[k] for k in found]), np.concatenate(list(found.values())), K, dist)
        except (CalibrationError, cv2.error):
            return found, []
    extra = guided_corners(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), rig_map, K, dist, *pose, have=found)
    return {**found, **extra}, sorted(extra)


def average_corners(frames: list[dict], min_frac: float = 0.7) -> tuple[dict, dict]:
    """Mean corners over frames for markers seen in >= min_frac of them, and their jitter (max std, px)."""
    acc: dict[str, list] = {}
    for found in frames:
        for k, c in found.items():
            acc.setdefault(k, []).append(np.asarray(c, float).reshape(4, 2))
    keep = {k: v for k, v in acc.items() if len(v) >= max(1, min_frac * len(frames))}
    return ({k: np.mean(v, axis=0) for k, v in keep.items()},
            {k: float(np.std(v, axis=0).max()) for k, v in keep.items()})


# ----------------------------------------------------------------------------------- plane
def plane_z(plane, x, y):
    a, b, c = plane
    return a + b * np.asarray(x, float) + c * np.asarray(y, float)


def plane_frame(plane) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Origin, in-plane axes e1, e2 and the upward unit normal n of z = a + b x + c y."""
    a, b, c = (float(v) for v in plane)
    n = np.array([-b, -c, 1.0])
    n /= np.linalg.norm(n)
    e1 = np.array([1.0, 0.0, b])
    e1 /= np.linalg.norm(e1)
    return np.array([0.0, 0.0, a]), e1, np.cross(n, e1), n


def to_plane(plane, P) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points -> in-plane coordinates (N, 2) and signed height above the plane (N,)."""
    o, e1, e2, n = plane_frame(plane)
    d = np.asarray(P, float).reshape(-1, 3) - o
    return np.c_[d @ e1, d @ e2], d @ n


def from_plane(plane, uv, h=0.0) -> np.ndarray:
    o, e1, e2, n = plane_frame(plane)
    uv = np.asarray(uv, float).reshape(-1, 2)
    return o + uv[:, :1] * e1 + uv[:, 1:] * e2 + np.reshape(h, (-1, 1)) * n


def marker_points(plane, pose, size: float) -> np.ndarray:
    """(5, 3): the 4 corners + centre of a marker with in-plane pose (u, v, yaw), base frame."""
    u, v, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return from_plane(plane, np.array([u, v]) + local_corners(size) @ np.array([[c, s], [-s, c]]))


def rigid_fit_2d(local: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(u, v, yaw) of the rotation + translation taking `local` points onto `uv` (least squares)."""
    lm, um = local.mean(axis=0), uv.mean(axis=0)
    l, w = local - lm, uv - um
    yaw = np.arctan2(np.sum(l[:, 0] * w[:, 1] - l[:, 1] * w[:, 0]), np.sum(l[:, 0] * w[:, 0] + l[:, 1] * w[:, 1]))
    c, s = np.cos(yaw), np.sin(yaw)
    t = um - np.array([[c, -s], [s, c]]) @ lm
    return np.array([t[0], t[1], yaw])


def fit_plane(P: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Robust z = a + b x + c y (trimmed least squares). Returns (coef, residuals of all points)."""
    P = np.asarray(P, float).reshape(-1, 3)
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        coef = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)[0]
        res = P[:, 2] - A @ coef
        sig = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 5e-4)
        new = np.abs(res) < 3 * sig
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    return coef, res


def fit_quadratic(P: np.ndarray, plane) -> dict:
    """z = a + bx + cy + d x^2 + e xy + f y^2 on the sweep, and whether it is worth using:
    the curved part must be smooth model error (clearly lower residual, > 1 mm of bend)."""
    P = np.asarray(P, float).reshape(-1, 3)
    x, y = P[:, 0], P[:, 1]
    res_p = P[:, 2] - plane_z(plane, x, y)
    keep = np.abs(res_p) < 3 * max(1.4826 * float(np.median(np.abs(res_p - np.median(res_p)))), 5e-4)
    out = {"coef": None, "rms_plane_mm": float(np.sqrt(np.mean(res_p[keep] ** 2)) * 1000), "rms_quadratic_mm": None,
           "bend_mm": 0.0, "significant": False}
    if keep.sum() < 30:
        return out
    A = np.c_[np.ones(len(P)), x, y, x * x, x * y, y * y]
    coef = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)[0]
    res_q = P[:, 2] - A @ coef
    bend = A[keep, 3:] @ coef[3:]
    out.update(coef=coef.tolist(), rms_quadratic_mm=float(np.sqrt(np.mean(res_q[keep] ** 2)) * 1000),
               bend_mm=float((bend.max() - bend.min()) * 1000))
    out["significant"] = bool(out["rms_quadratic_mm"] < 0.7 * out["rms_plane_mm"] and out["bend_mm"] > 1.0)
    return out


# ----------------------------------------------------------------------------------- sweep
def tilt_deg(R) -> np.ndarray:
    """Downward tilt of the fingers (gripper_end x axis) in degrees, as shown by table_height.py."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    return np.degrees(np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0)))


def tilt_deviation_deg(R, ref_tilt_deg: float) -> np.ndarray:
    """Angle between the gripper orientation and the blocking orientation at `ref_tilt_deg`,
    ignoring yaw about the vertical (the jaw may turn while it slides)."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    p = np.radians(ref_tilt_deg)
    ref = np.array([-np.sin(p), 0.0, np.cos(p)])          # base z in the gripper frame at the blocking pose
    return np.degrees(np.arccos(np.clip(R[:, 2, :] @ ref, -1.0, 1.0)))


def filter_sweep(p, R, ref_tilt_deg: float | None, max_dev_deg: float = 5.0, grid: float = 0.02,
                 max_lift: float = 0.008) -> tuple[np.ndarray, dict]:
    """Raw sweep samples -> one point per `grid` cell.

    Drops samples whose orientation is more than max_dev_deg from the reference tilt (None = the
    median tilt of the sweep) and samples more than max_lift off the table (arm lifted mid-sweep).
    """
    p = np.asarray(p, float).reshape(-1, 3)
    stats = {"n_raw": len(p), "ref_tilt_deg": None, "n_tilt_rejected": 0, "n_lift_rejected": 0, "n_cells": 0}
    if len(p) == 0:
        return np.zeros((0, 3)), stats
    R = np.asarray(R, float).reshape(-1, 3, 3)
    ref = float(np.median(tilt_deg(R))) if ref_tilt_deg is None else float(ref_tilt_deg)
    ok = tilt_deviation_deg(R, ref) <= max_dev_deg
    stats.update(ref_tilt_deg=ref, n_tilt_rejected=int((~ok).sum()))
    p = p[ok]
    if len(p) >= 3:
        _, res = fit_plane(p)
        lifted = np.abs(res) > max_lift
        stats["n_lift_rejected"] = int(lifted.sum())
        p = p[~lifted]
    cells: dict[tuple, list] = {}
    for q in p:
        cells.setdefault((int(np.floor(q[0] / grid)), int(np.floor(q[1] / grid))), []).append(q)
    pts = np.array([np.median(v, axis=0) for v in cells.values()]).reshape(-1, 3)
    stats["n_cells"] = len(pts)
    return pts, stats


# ------------------------------------------------------------------------------ projection
def undistort_px(K, dist, uv) -> np.ndarray:
    """Distorted pixels -> pixels of the ideal pinhole camera with the same K."""
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(uv, np.asarray(K, float), np.asarray(dist, float), P=np.asarray(K, float)).reshape(-1, 2)


def project(K, dist, rvec, tvec, P) -> np.ndarray:
    p, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), np.asarray(rvec, float),
                             np.asarray(tvec, float), np.asarray(K, float), np.asarray(dist, float))
    return p.reshape(-1, 2)


def camera_centre(rvec, tvec) -> np.ndarray:
    return -cv2.Rodrigues(np.asarray(rvec, float))[0].T @ np.asarray(tvec, float).reshape(3)


def backproject_to_plane(K, dist, rvec, tvec, plane, uv, offset: float = 0.0) -> np.ndarray:
    """Pixels -> base-frame points on the table plane raised by `offset` along its normal."""
    K = np.asarray(K, float)
    und = undistort_px(K, dist, uv)
    rays = np.c_[(und[:, 0] - K[0, 2]) / K[0, 0], (und[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(und))]
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    d = rays @ R                                      # rows: R^T ray
    C = camera_centre(rvec, tvec)
    o, _, _, n = plane_frame(plane)
    s = (offset - n @ (C - o)) / (d @ n)
    return C + s[:, None] * d


def pixel_to_ball_xy(K, dist, rvec, tvec, plane, uv, r: float = BALL_R) -> np.ndarray:
    """Ball-centre (x, y) in the base frame for blob-centre pixels: the ray meets the plane of the
    ball centre, table + r on the camera's side. Fails loudly if the sign convention is broken."""
    if not r > 0:
        raise CalibrationError(f"ball radius must be positive (got {r}): the ball centre is ABOVE the table, "
                               "on the camera's side")
    o, _, _, n = plane_frame(plane)
    height = float(n @ (camera_centre(rvec, tvec) - o))
    if height <= r:
        raise CalibrationError(f"the camera is {height * 1000:.0f} mm above the table plane along +n: it must look "
                               "at the table from above (wrong plane sign or a mirrored pose)")
    return backproject_to_plane(K, dist, rvec, tvec, plane, uv, offset=r)[:, :2]


def _pnp(obj, img, K, dist):
    """SQPNP start, then Levenberg-Marquardt refinement on all points."""
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 1, 2)
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise CalibrationError("solvePnP failed")
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
    return rv.reshape(3), tv.reshape(3)


# ------------------------------------------------------------------------------ data model
@dataclass
class View:
    name: str
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    corners: dict                 # key -> (4, 2) averaged corner pixels


@dataclass
class Touch:
    key: str
    corner: int                   # 0..3 = TL, TR, BR, BL; 4 = centre
    p: np.ndarray                 # Fingertip(q), base frame


@dataclass
class Observations:
    views: list
    touches: list
    sweep: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))    # filtered, gridded
    sizes: dict = field(default_factory=table_sizes)
    sources: dict = field(default_factory=dict)


@dataclass
class RigMap:
    markers: dict                 # key -> (4, 3) corners in the base frame (TL, TR, BR, BL)
    plane: np.ndarray             # table plane z = a + b x + c y
    sizes: dict
    touched: dict = field(default_factory=dict)          # key -> touched corner indices
    far_dist: float = FAR_DIST
    time: str = ""
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)

    def centre(self, key: str) -> np.ndarray:
        return self.markers[key].mean(axis=0)

    def is_far(self, key: str) -> bool:
        return float(np.linalg.norm(self.centre(key)[:2])) >= self.far_dist

    def to_json(self) -> dict:
        mk = {}
        for k, c in self.markers.items():
            yaw = np.degrees(np.arctan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0]))
            mk[k] = {"size": self.sizes[k], "corners": np.round(c, 5).tolist(),
                     "centre": np.round(c.mean(axis=0), 5).tolist(), "yaw_deg": round(float(yaw), 2),
                     "touched_corners": [CORNER_NAMES[i] for i in self.touched.get(k, [])],
                     "far": self.is_far(k)}
        return {"version": 1, "time": self.time,
                "frame": "arm kinematic base frame (eaiproject.kinematics.Fingertip), metres",
                "corner_order": "TL, TR, BR, BL (OpenCV ArUco)",
                "plane_z=a+bx+cy": [float(v) for v in self.plane], "far_dist": self.far_dist, "markers": mk,
                "stats": self.stats, "warnings": self.warnings, "sources": self.sources}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @staticmethod
    def from_json(d: dict) -> "RigMap":
        mk = {k: np.array(m["corners"], float) for k, m in d["markers"].items() if is_table(k)}
        return RigMap(markers=mk, plane=np.array(d["plane_z=a+bx+cy"], float),
                      sizes={k: float(d["markers"][k]["size"]) for k in mk},
                      touched={k: [CORNER_NAMES.index(n) for n in d["markers"][k]["touched_corners"]] for k in mk
                               if d["markers"][k].get("touched_corners")},
                      far_dist=float(d.get("far_dist", FAR_DIST)), time=d.get("time", ""),
                      stats=d.get("stats", {}), warnings=d.get("warnings", []), sources=d.get("sources", {}))

    @staticmethod
    def load(path) -> "RigMap":
        return RigMap.from_json(json.loads(Path(path).read_text()))


def load_observations(session_dir, ref_tilt_deg: float | None = None, max_dev_deg: float = 5.0,
                      grid: float = 0.02) -> tuple[Observations, dict]:
    """Read what scripts/real/rig_capture.py saved (views.json, touches.json, sweep.npz)."""
    d = Path(session_dir)
    sizes = table_sizes()
    views, touches, sweep, stats = [], [], np.zeros((0, 3)), {}
    if (d / "views.json").exists():
        for v in json.loads((d / "views.json").read_text())["views"]:
            views.append(View(v["name"], np.array(v["K"], float), np.array(v["dist"], float), v["width"], v["height"],
                              table_only({k: np.array(c, float) for k, c in v["corners"].items()}, sizes)))
    if (d / "touches.json").exists():
        for t in json.loads((d / "touches.json").read_text())["touches"]:
            if t["key"] in sizes:
                touches.append(Touch(t["key"], int(t["corner"]), np.array(t["p"], float)))
    if (d / "sweep.npz").exists():
        z = np.load(d / "sweep.npz")
        sweep, stats = filter_sweep(z["p"], z["R"], ref_tilt_deg, max_dev_deg, grid)
    src = {"session": str(d), "views": "views.json", "touches": "touches.json", "sweep": "sweep.npz"}
    return Observations(views, touches, sweep, sizes, src), stats


# ---------------------------------------------------------------------------------- solver
def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(V, 3) rotation vectors -> (V, 3, 3) matrices."""
    th = np.linalg.norm(r, axis=1)
    k = r / np.where(th < 1e-12, 1.0, th)[:, None]
    Kx = np.zeros((len(r), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


class _Problem:
    """Index arrays for the joint problem. x = [plane (3) | marker u, v, yaw (3M) | rvec, tvec (6V)]."""

    def __init__(self, obs: Observations, keys: list[str], touches: list[Touch]):
        self.keys, self.views = keys, obs.views
        self.M, self.V = len(keys), len(obs.views)
        ki = {k: i for i, k in enumerate(keys)}
        self.local = np.array([local_corners(obs.sizes[k]) for k in keys])          # (M, 5, 2)
        vi, mi, ci, und, f, c = [], [], [], [], [], []
        for v_idx, v in enumerate(obs.views):
            for k, px in v.corners.items():
                if k not in ki:
                    continue
                u = undistort_px(v.K, v.dist, px)
                for j in range(4):
                    vi.append(v_idx); mi.append(ki[k]); ci.append(j); und.append(u[j])
                    f.append([v.K[0, 0], v.K[1, 1]]); c.append([v.K[0, 2], v.K[1, 2]])
        self.vi, self.mi, self.ci = np.array(vi), np.array(mi), np.array(ci)
        self.und, self.f, self.c = np.array(und), np.array(f), np.array(c)
        self.t_mi = np.array([ki[t.key] for t in touches], int)
        self.t_ci = np.array([t.corner for t in touches], int)
        self.t_p = np.array([t.p for t in touches], float).reshape(-1, 3)
        self.sweep = np.asarray(obs.sweep, float).reshape(-1, 3)

    def unpack(self, x):
        return x[:3], x[3:3 + 3 * self.M].reshape(self.M, 3), x[3 + 3 * self.M:].reshape(self.V, 6)

    def pack(self, plane, poses, cams):
        return np.r_[plane, np.ravel(poses), np.ravel(cams)]

    def points(self, plane, poses) -> np.ndarray:
        """(M, 5, 3) corners + centre of every marker in the base frame."""
        o, e1, e2, _ = plane_frame(plane)
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        u = poses[:, 0:1] + c * self.local[:, :, 0] - s * self.local[:, :, 1]
        v = poses[:, 1:2] + s * self.local[:, :, 0] + c * self.local[:, :, 1]
        return o + u[..., None] * e1 + v[..., None] * e2

    def parts(self, x):
        """Unweighted residuals: reprojection (N, 2) px, touches (T, 3) m, sweep heights (S,) m."""
        plane, poses, cams = self.unpack(x)
        P = self.points(plane, poses)
        R = _rodrigues(cams[:, :3])
        Xc = np.einsum("nij,nj->ni", R[self.vi], P[self.mi, self.ci]) + cams[self.vi, 3:]
        reproj = self.f * Xc[:, :2] / Xc[:, 2:3] + self.c - self.und
        touch = self.t_p - P[self.t_mi, self.t_ci] if len(self.t_p) else np.zeros((0, 3))
        sweep = self.sweep[:, 2] - plane_z(plane, self.sweep[:, 0], self.sweep[:, 1])
        return reproj, touch, sweep

    def residuals(self, x):
        reproj, touch, sweep = self.parts(x)
        return np.r_[reproj.ravel() / SIGMA_PX, touch.ravel() / SIGMA_TOUCH, sweep / SIGMA_SWEEP]


def touch_geometry(xy: np.ndarray) -> dict:
    """In-plane spread of the touched points: span along the main axis, width across it, and the
    1-sigma yaw uncertainty of the map that follows from SIGMA_TOUCH."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    d = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    along, across = d @ Vt[0], d @ Vt[-1] if len(xy) > 1 else np.zeros(len(xy))
    sig_yaw = SIGMA_TOUCH / max(float(np.sqrt(np.sum(d ** 2))), 1e-9)
    return {"n_points": len(xy), "span_m": float(np.ptp(along)), "width_m": float(np.ptp(across)),
            "centroid": xy.mean(axis=0).round(4).tolist(), "yaw_sigma_deg": float(np.degrees(sig_yaw)),
            "sigma_mm_per_m": float(sig_yaw * 1000)}


def _initial_guess(obs: Observations, prob: _Problem, touches: list[Touch]):
    """Plane from the sweep (else the touches), touched markers from their touched points, untouched
    markers by back-projection through views whose pose comes from PnP on already-known markers."""
    if len(obs.sweep) >= 10:
        plane, _ = fit_plane(obs.sweep)
    else:
        tp = np.array([t.p for t in touches])
        ok = len(tp) >= 3 and touch_geometry(tp[:, :2])["width_m"] > 0.02
        plane = fit_plane(tp)[0] if ok else np.array([float(np.median(tp[:, 2])), 0.0, 0.0])
    poses: dict[str, np.ndarray] = {}
    for k in prob.keys:
        tk = [t for t in touches if t.key == k]
        if len(tk) >= 2:
            uv, _ = to_plane(plane, [t.p for t in tk])
            poses[k] = rigid_fit_2d(local_corners(obs.sizes[k])[[t.corner for t in tk]], uv)
    if not poses:
        raise CalibrationError("no marker has at least two touched points: the map cannot be placed in the arm frame")

    def view_pose(v):
        known = [k for k in v.corners if k in poses]
        obj = np.concatenate([marker_points(plane, poses[k], obs.sizes[k])[:4] for k in known])
        return _pnp(obj, np.concatenate([v.corners[k] for k in known]), v.K, v.dist)

    def place(todo_only: bool):
        est: dict[str, list] = {}
        for v in obs.views:
            if v.name not in cams:
                continue
            for k, px in v.corners.items():
                if k in prob.keys and (k not in poses if todo_only else k not in touched):
                    est.setdefault(k, []).append(to_plane(plane, backproject_to_plane(v.K, v.dist, *cams[v.name], plane, px))[0])
        for k, l in est.items():
            poses[k] = rigid_fit_2d(local_corners(obs.sizes[k])[:4], np.mean(l, axis=0))
        return bool(est)

    touched = set(poses)
    cams: dict[str, tuple] = {}
    progressed = True
    while progressed:
        progressed = False
        todo = [v for v in obs.views if v.name not in cams and any(k in poses for k in v.corners)]
        for v in sorted(todo, key=lambda v: -sum(k in poses for k in v.corners)):
            cams[v.name] = view_pose(v)
            progressed = True
        progressed |= place(todo_only=True)
    lost_v = [v.name for v in obs.views if v.name not in cams]
    lost_m = [k for k in prob.keys if k not in poses]
    if lost_v or lost_m:
        raise CalibrationError(f"cannot link everything to the touched markers: views {lost_v}, markers {lost_m}. "
                               "Each view must show a touched marker, or share a marker with a view that does.")
    for _ in range(2):                                   # polish: all markers in every PnP, all views per marker
        cams = {v.name: view_pose(v) for v in obs.views}
        place(todo_only=False)
    return prob.pack(plane, [poses[k] for k in prob.keys], [np.r_[cams[v.name]] for v in obs.views])


@dataclass
class RigSolution:
    rig_map: RigMap
    cams: dict                    # view name -> (rvec, tvec)
    report: dict
    warnings: list
    touch_points: np.ndarray      # (T, 3) touched positions
    touch_residuals: np.ndarray   # (T, 3) touch - fitted corner (m)
    sweep_residuals: np.ndarray   # (S,) sweep height - plane (m)
    x: np.ndarray


def _run(prob: _Problem, x0: np.ndarray) -> np.ndarray:
    sol = least_squares(prob.residuals, x0, loss="soft_l1", f_scale=F_SCALE, x_scale="jac", max_nfev=300)
    return sol.x


def solve_rig(obs: Observations, far_dist: float = FAR_DIST, leave_one_out: bool = True) -> RigSolution:
    """Joint least squares for the table plane, every table marker's pose in it and one camera pose
    per view, in the arm's kinematic frame. Raises CalibrationError when the data cannot work."""
    warn: list[str] = []
    if not obs.views:
        raise CalibrationError("no camera views")
    keys = sorted({k for v in obs.views for k in v.corners if k in obs.sizes})
    if len(keys) < 3:
        raise CalibrationError(f"only {len(keys)} table marker(s) in the views ({keys}); the rig needs 4 or more")
    if len(keys) < 4:
        warn.append(f"only {len(keys)} table markers in the views; 4 or more are expected")
    unseen = sorted({t.key for t in obs.touches} - set(keys))
    if unseen:
        warn.append(f"touched markers never seen by the camera are ignored: {unseen}")
    touches = [t for t in obs.touches if t.key in keys]
    if len(touches) < 2:
        raise CalibrationError("fewer than 2 touched points: nothing ties the markers to the arm frame")
    if len(obs.views) < MIN_VIEWS:
        warn.append(f"only {len(obs.views)} views (at least {MIN_VIEWS} wanted)")
    n_seen = {k: sum(k in v.corners for v in obs.views) for k in keys}
    if min(n_seen.values()) < 2:
        warn.append(f"markers seen in a single view: {[k for k, n in n_seen.items() if n < 2]}")
    if len(obs.sweep) < 10:
        warn.append("no surface sweep: the table plane comes from the corner touches only")

    geo = touch_geometry(to_plane(fit_plane(np.array([t.p for t in touches]))[0] if len(touches) >= 3
                                  else [0, 0, 0], [t.p for t in touches])[0])
    touched_keys = sorted({t.key for t in touches})
    if len(touched_keys) < 2:
        warn.append("only one marker touched: touch at least 2")
    if geo["span_m"] < MIN_TOUCH_SPAN:
        warn.append(f"touched points span only {geo['span_m'] * 100:.0f} cm (< {MIN_TOUCH_SPAN * 100:.0f} cm): "
                    f"the yaw of the map is weak, about {geo['sigma_mm_per_m']:.1f} mm per metre from the touches")
    if geo["width_m"] < MIN_TOUCH_WIDTH:
        warn.append(f"touched points are close to collinear (only {geo['width_m'] * 100:.1f} cm wide): "
                    "touch all 4 corners, on markers that are not in one line")

    prob = _Problem(obs, keys, touches)
    x = _run(prob, _initial_guess(obs, prob, touches))
    plane, poses, cams = prob.unpack(x)
    reproj, touch, sweep = prob.parts(x)
    P = prob.points(plane, poses)
    o, _, _, n = plane_frame(plane)
    for v, cam in zip(obs.views, cams):
        if n @ (camera_centre(cam[:3], cam[3:]) - o) <= 0:
            raise CalibrationError(f"view {v.name} ends up below the table: mirrored solution, check the touches")

    err = np.linalg.norm(reproj, axis=1)
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else 0.0
    per_marker = {k: round(rms(err[prob.mi == i]), 3) for i, k in enumerate(keys)}
    per_view = {v.name: {"reproj_px": round(rms(err[prob.vi == i]), 3), "markers": sorted(k for k in v.corners if k in keys)}
                for i, v in enumerate(obs.views)}
    touch_rows = [{"marker": t.key, "corner": CORNER_NAMES[t.corner], "dx_mm": round(float(r[0]) * 1000, 2),
                   "dy_mm": round(float(r[1]) * 1000, 2), "dz_mm": round(float(r[2]) * 1000, 2),
                   "err_mm": round(float(np.linalg.norm(r)) * 1000, 2)} for t, r in zip(touches, touch)]
    report = {
        "n_views": len(obs.views), "n_markers": len(keys), "n_touches": len(touches), "n_sweep_points": len(obs.sweep),
        "reproj_rms_px": round(rms(err), 3), "reproj_max_px": round(float(err.max()), 3),
        "per_marker_reproj_px": per_marker, "per_view": per_view,
        "touch_rms_mm": round(rms(np.linalg.norm(touch, axis=1)) * 1000, 2),
        "touch_xy_rms_mm": round(rms(np.linalg.norm(touch[:, :2], axis=1)) * 1000, 2),
        "touch_dz_mean_mm": round(float(touch[:, 2].mean()) * 1000, 2),
        "touches": touch_rows, "touch_geometry": geo,
        "sweep_rms_mm": round(rms(sweep) * 1000, 2),
        "sweep_max_mm": round(float(np.abs(sweep).max()) * 1000, 2) if len(sweep) else 0.0,
        "plane_z=a+bx+cy": [float(v) for v in plane],
        "plane_tilt_deg_x_y": [round(float(np.degrees(np.arctan(plane[1]))), 3), round(float(np.degrees(np.arctan(plane[2]))), 3)],
        "far_markers": [k for i, k in enumerate(keys) if np.linalg.norm(P[i, 4, :2]) >= far_dist],
    }
    # Scale check: the camera sees distances in units of the marker size, the arm measures them
    # directly. If the touched points are spread differently from the fitted corners, the printed
    # marker size (or the corner detection) is off.
    fit_xy, t_xy = (prob.t_p - touch)[:, :2], prob.t_p[:, :2]
    fc, tc = fit_xy - fit_xy.mean(axis=0), t_xy - t_xy.mean(axis=0)
    scale = float(np.sum(fc * tc) / max(np.sum(fc * fc), 1e-12))
    report["touch_scale_percent"] = round((scale - 1) * 100, 3)
    if abs(scale - 1) > max(0.004, 3 * geo["sigma_mm_per_m"] / 1000) and geo["span_m"] >= MIN_TOUCH_SPAN:
        warn.append(f"the touched points are {abs(scale - 1) * 100:.1f} % {'further apart' if scale > 1 else 'closer together'} "
                    "than the camera's layout: measure a printed marker side (must match markers.py) and check the "
                    "camera intrinsics / resolution")
    if report["reproj_rms_px"] > OK_PX:
        warn.append(f"reprojection RMS {report['reproj_rms_px']} px is above {OK_PX} px")
    if report["touch_rms_mm"] > OK_MM:
        warn.append(f"touch residual RMS {report['touch_rms_mm']} mm is above {OK_MM} mm: a corner was touched wrongly?")
    if abs(report["touch_dz_mean_mm"]) > 3.0:
        warn.append(f"corner touches sit {report['touch_dz_mean_mm']:+.1f} mm from the swept plane on average: "
                    "the model's height bias differs between the touch pose and the blocking tilt")
    if not report["far_markers"]:
        warn.append(f"no marker is farther than {far_dist:.2f} m from the base: relocalisation will refuse "
                    "(it needs one far marker); lower far_dist or add a marker further out")

    if leave_one_out:
        loo = {}
        for k in touched_keys:
            rest = [t for t in touches if t.key != k]
            mine = [t for t in touches if t.key == k]
            if len(rest) < 2:
                continue
            p2 = _Problem(obs, keys, rest)
            pl2, po2, _ = p2.unpack(_run(p2, x))
            d = np.array([t.p for t in mine]) - p2.points(pl2, po2)[keys.index(k), [t.corner for t in mine]]
            # what touch noise alone explains: each touch (2 axes) + placing the map with the other touches
            rxy = np.array([t.p[:2] for t in rest])
            lever = np.linalg.norm(P[keys.index(k), 4, :2] - rxy.mean(axis=0))
            expect = SIGMA_TOUCH * np.sqrt(2 + 2 / len(rest) + lever ** 2 / np.sum((rxy - rxy.mean(axis=0)) ** 2)) * 1000
            loo[k] = {"xy_mm": round(rms(np.linalg.norm(d[:, :2], axis=1)) * 1000, 2),
                      "max_mm": round(float(np.linalg.norm(d, axis=1).max()) * 1000, 2),
                      "dz_mm": round(float(d[:, 2].mean()) * 1000, 2), "expected_xy_mm": round(float(expect), 2),
                      "markers_left": len({t.key for t in rest})}
            if loo[k]["xy_mm"] > max(OK_MM, 2.5 * expect):
                warn.append(f"leave-one-out: {k} is predicted {loo[k]['xy_mm']} mm off from the other markers' touches "
                            f"(touch noise explains about {expect:.1f} mm)")
        report["leave_one_marker_out"] = loo

    touched = {k: sorted({t.corner for t in touches if t.key == k}) for k in touched_keys}
    rig_map = RigMap(markers={k: P[i, :4].copy() for i, k in enumerate(keys)}, plane=np.array(plane, float),
                     sizes={k: obs.sizes[k] for k in keys}, touched=touched, far_dist=far_dist,
                     time=time.strftime("%Y-%m-%d %H:%M:%S"),
                     stats={k: report[k] for k in ("n_views", "n_markers", "n_touches", "n_sweep_points", "reproj_rms_px",
                                                   "per_marker_reproj_px", "touch_rms_mm", "touch_xy_rms_mm",
                                                   "touch_dz_mean_mm", "touch_scale_percent", "sweep_rms_mm", "touch_geometry")}
                     | ({"leave_one_marker_out": report["leave_one_marker_out"]} if leave_one_out else {}),
                     warnings=list(warn), sources=dict(obs.sources))
    report["warnings"] = warn
    return RigSolution(rig_map, {v.name: (c[:3].copy(), c[3:].copy()) for v, c in zip(obs.views, cams)}, report, warn,
                       np.array([t.p for t in touches]), touch, sweep, x)


def table_height_dict(sweep: np.ndarray, gate_x: float = 0.40, extra_points: np.ndarray | None = None) -> dict:
    """Contents of table_height.json in the format real_intercept.load_calibration() reads
    ("plane_z=a+bx+cy"), fitted to the sweep (or to `extra_points` when there is no sweep).
    A quadratic surface is added as an extra field only when the residuals show smooth curvature."""
    src = "sweep" if len(sweep) >= 10 else "corner touches"
    P = np.asarray(sweep if len(sweep) >= 10 else extra_points, float).reshape(-1, 3)
    coef, res = fit_plane(P)
    a, b, c = coef
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": f"rig_solve.py ({src})",
           "plane_z=a+bx+cy": coef.tolist(), "n_points": len(P),
           "residual_rms_mm": round(float(np.sqrt(np.mean(res ** 2))) * 1000, 2),
           "z_at_gate_mm": {f"{y:+.2f}": round(float(a + b * gate_x + c * y) * 1000, 1) for y in (-0.24, -0.12, 0.0, 0.12, 0.24)},
           "gate_x": gate_x,
           "tilt_deg_x_y": [round(float(np.degrees(np.arctan(b))), 2), round(float(np.degrees(np.arctan(c))), 2)]}
    quad = fit_quadratic(P, coef)
    out["curvature"] = {k: quad[k] for k in ("rms_plane_mm", "rms_quadratic_mm", "bend_mm", "significant")}
    if quad["significant"]:
        out["quadratic_z=a+bx+cy+dxx+exy+fyy"] = quad["coef"]       # NOT read by the planner (yet)
    return out


# -------------------------------------------------------------------------- relocalisation
@dataclass
class Reloc:
    rvec: np.ndarray
    tvec: np.ndarray
    used: list
    rejected: list
    per_marker_px: dict
    per_marker_mm: dict
    reproj_rms_px: float
    table_rms_mm: float
    camera_position: np.ndarray
    guided: list = field(default_factory=list)      # markers located by prediction (guided_corners)

    @property
    def status(self) -> str:
        return status(self.reproj_rms_px, self.table_rms_mm)


def status(px: float | None = None, mm: float | None = None) -> str:
    """OK / WARN / FAIL from a reprojection error (px) and / or a position error (mm)."""
    px, mm = px or 0.0, mm or 0.0
    if px > FAIL_PX or mm > FAIL_MM:
        return "FAIL"
    return "OK" if px < OK_PX and mm < OK_MM else "WARN"


def marker_errors(rig_map: RigMap, K, dist, rvec, tvec, detections: dict) -> tuple[dict, dict]:
    """Per-marker error of a pose against detected corners: mean reprojection (px) and mean
    distance on the table between the detected corners and the map (mm)."""
    px, mm = {}, {}
    for k, c in detections.items():
        if k not in rig_map.markers:
            continue
        c = np.asarray(c, float).reshape(4, 2)
        px[k] = float(np.linalg.norm(project(K, dist, rvec, tvec, rig_map.markers[k]) - c, axis=1).mean())
        on_table = backproject_to_plane(K, dist, rvec, tvec, rig_map.plane, c)
        mm[k] = float(np.linalg.norm(on_table - rig_map.markers[k], axis=1).mean() * 1000)
    return px, mm


def relocalize(rig_map: RigMap, K, dist, detections: dict, min_markers: int = 3, require_far: bool = True,
               guided: dict | None = None) -> Reloc:
    """Camera pose in the arm frame from ONE PnP over all visible table-marker corners.

    `detections`: key -> (4, 2) corners, ideally averaged over ~30 frames (average corners, not
    poses). Refuses (CalibrationError) with fewer than `min_markers` map markers, or when none of
    them is far from the arm: near markers alone leave the pose loose across the table.
    `guided`: extra corners from guided_corners() (markers located by prediction, not identified by
    their bit pattern). They join the fit but do not count towards the two rules above.
    """
    seen = {k: np.asarray(c, float).reshape(4, 2) for k, c in detections.items() if k in rig_map.markers}
    extra = {k: np.asarray(c, float).reshape(4, 2) for k, c in (guided or {}).items() if k in rig_map.markers and k not in seen}
    unknown = sorted(k for k in detections if is_table(k) and k not in rig_map.markers)

    def check_ids(keys):
        if len(keys) < min_markers:
            raise CalibrationError(
                f"only {len(keys)} map marker(s) visible ({sorted(keys)}); need at least {min_markers} of "
                f"{sorted(rig_map.markers)}. Move the camera so it sees more of the table, check lighting / occlusion."
                + (f" Markers not in the map: {unknown}." if unknown else ""))
        if require_far and not any(rig_map.is_far(k) for k in keys):
            far = sorted(k for k in rig_map.markers if rig_map.is_far(k))
            raise CalibrationError(
                f"no marker far from the arm is visible (seen: {sorted(keys)}; far markers, beyond "
                f"{rig_map.far_dist:.2f} m: {far or 'none in the map'}). Aim the camera further down the table.")

    def fit(keys):
        rv, tv = _pnp(np.concatenate([rig_map.markers[k] for k in keys]), np.concatenate([seen[k] for k in keys]), K, dist)
        return rv, tv, marker_errors(rig_map, K, dist, rv, tv, {k: seen[k] for k in keys})

    keys = sorted(seen)
    check_ids(keys)
    check = lambda keys: check_ids([k for k in keys if k not in extra])          # rules apply to identified markers
    seen.update(extra)
    keys = sorted(seen)
    rvec, tvec, (px, mm) = fit(keys)
    rejected = []
    if len([k for k in keys if k not in extra]) > min_markers and np.mean(list(px.values())) > OK_PX:
        # A bad fit with markers to spare: does dropping ONE marker (moved on the table, misread) fix it?
        best = None
        for k in keys:
            rest = [j for j in keys if j != k]
            try:
                check(rest)
            except CalibrationError:
                continue
            cand = fit(rest)
            if best is None or np.mean(list(cand[2][0].values())) < np.mean(list(best[1][2][0].values())):
                best = (k, cand, rest)
        if best is not None and np.mean(list(best[1][2][0].values())) < min(OK_PX, 0.4 * np.mean(list(px.values()))):
            (bad, (rvec, tvec, (px, mm)), keys), rejected = best, [best[0]]
    C = camera_centre(rvec, tvec)
    o, _, _, n = plane_frame(rig_map.plane)
    if n @ (C - o) <= 0:
        raise CalibrationError("the pose puts the camera below the table: mirrored PnP solution (camera too flat / "
                               "markers too small in the image). Raise the camera and retry.")
    allc = np.concatenate([seen[k] for k in keys])
    e_px = np.linalg.norm(project(K, dist, rvec, tvec, np.concatenate([rig_map.markers[k] for k in keys])) - allc, axis=1)
    e_mm = np.linalg.norm(backproject_to_plane(K, dist, rvec, tvec, rig_map.plane, allc)
                          - np.concatenate([rig_map.markers[k] for k in keys]), axis=1) * 1000
    return Reloc(rvec, tvec, keys, rejected, px, mm, float(np.sqrt(np.mean(e_px ** 2))),
                 float(np.sqrt(np.mean(e_mm ** 2))), C, sorted(k for k in keys if k in extra))


def extrinsics_dict(res: Reloc, rig_map: RigMap, K, dist, width: int, height: int, table_ref_xy=(0.56, 0.01),
                    serial: str | None = None, ball_r: float = BALL_R) -> dict:
    """camera_extrinsics.json in the format real_intercept.load_calibration() reads. K / dist must be
    those of the stream the tracker runs on (the file is only valid at this resolution)."""
    pixel_to_ball_xy(K, dist, res.rvec, res.tvec, rig_map.plane, [[width / 2, height / 2]], ball_r)   # sign check
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "relocalize.py (PnP on config/rig_map.json)",
            "rig_map_time": rig_map.time, "camera_serial": serial, "width": int(width), "height": int(height),
            "K": np.asarray(K, float).tolist(), "dist": np.asarray(dist, float).ravel().tolist(), "ball_r": ball_r,
            "rvec": res.rvec.tolist(), "tvec": res.tvec.tolist(),
            "camera_position_robot_frame": res.camera_position.tolist(),
            "table_z": float(plane_z(rig_map.plane, *table_ref_xy)),
            "plane_z=a+bx+cy": [float(v) for v in rig_map.plane],
            "markers_used": res.used, "markers_guided": res.guided, "markers_rejected": res.rejected,
            "reproj_rms_px": round(res.reproj_rms_px, 3), "table_rms_mm": round(res.table_rms_mm, 2),
            "per_marker_reproj_px": {k: round(v, 3) for k, v in res.per_marker_px.items()},
            "status": res.status,
            "touches_robot": {}}          # kept so the older table_height.py still loads this file


class PoseMonitor:
    """Re-check a camera pose against the map while tracking: feed it marker detections every few
    seconds; it reports the median per-marker reprojection error and warns after two bad checks in a
    row (one check can be spoiled by a hand or the ball over a marker)."""

    def __init__(self, rig_map: RigMap, K, dist, rvec, tvec, warn_px: float = OK_PX, period: float = 3.0):
        self.map, self.K, self.dist, self.rvec, self.tvec = rig_map, K, dist, rvec, tvec
        self.warn_px, self.period = warn_px, period
        self.t_last, self.bad, self.median_px, self.n_markers = None, 0, None, 0

    def due(self, now: float) -> bool:
        return self.t_last is None or now - self.t_last >= self.period

    def detect(self, img_bgr: np.ndarray) -> dict:
        """Markers for update(): detected ones plus those found where the pose predicts them."""
        return detect_with_map(img_bgr, self.map, self.K, self.dist, (self.rvec, self.tvec))[0]

    def update(self, detections: dict, now: float) -> str | None:
        """`detections` from self.detect(img). Returns a warning text, or None while the pose still agrees with the map."""
        self.t_last = now
        px, _ = marker_errors(self.map, self.K, self.dist, self.rvec, self.tvec, detections)
        self.n_markers = len(px)
        if len(px) < 2:
            self.median_px = None
            return None                                  # not enough markers in view to judge
        self.median_px = float(np.median(list(px.values())))
        self.bad = self.bad + 1 if self.median_px > self.warn_px else 0
        return self.warning

    @property
    def warning(self) -> str | None:
        if self.bad >= 2:
            return (f"CAMERA MOVED? markers are {self.median_px:.1f} px off the map (limit {self.warn_px} px): "
                    "run scripts/real/relocalize.py")
        return None


def latest_session() -> Path | None:
    ds = [d for d in SESSIONS_DIR.glob("*") if d.is_dir() and any((d / f).exists() for f in ("views.json", "touches.json"))]
    return max(ds, key=lambda d: d.stat().st_mtime) if ds else None


def write_json_with_backup(path, data: dict) -> None:
    """Write JSON; an existing file is kept next to it as <name>.prev.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.replace(path.with_suffix(".prev.json"))
    path.write_text(json.dumps(data, indent=2) + "\n")
