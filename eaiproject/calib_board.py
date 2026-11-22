"""The printed ChArUco calibration board (A4 landscape). Shared by the PDF generator and the calibration.

Board frame: origin at the OUTER corner of the checker pattern marked A, x along A->B (7 squares),
y along A->D (5 squares), z INTO the table (OpenCV board convention: x right, y down on the page). Touch points A, B, C, D are the four outer corners of
the checker pattern (exact, printed with circles and labels).
"""
from __future__ import annotations

import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 7, 5
SQUARE = 0.035        # m
MARKER = 0.026        # m
DICT = cv2.aruco.DICT_4X4_50

W, H = SQUARES_X * SQUARE, SQUARES_Y * SQUARE          # 0.245 x 0.175 m
TOUCH_POINTS = {"A": (0.0, 0.0, 0.0), "B": (W, 0.0, 0.0), "C": (W, H, 0.0), "D": (0.0, H, 0.0)}


def board() -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE, MARKER, cv2.aruco.getPredefinedDictionary(DICT))


def detector() -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board())


def estimate_pose(img_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, min_corners: int = 8):
    """Camera pose of the board: (rvec, tvec, n_corners, reproj_px) or None. Board -> camera."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector().detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj = board().getChessboardCorners()[ids.reshape(-1)]
    img = corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    return rvec.reshape(3), tvec.reshape(3), len(ids), err, img
