"""The physical ArUco markers on the real rig (two dictionaries) and their detection.

    table markers: DICT_ARUCO_ORIGINAL, 70 mm  (107, 880 near the arm; 467, 718 mid-table)
    base markers:  DICT_4X4_50,         50 mm  (0, 2) flat on the arm's mounting plate
Keys are "<dict>:<id>", e.g. "orig:107", "4x4:0".
"""
from __future__ import annotations

import cv2
import numpy as np

DICTS = {"orig": cv2.aruco.DICT_ARUCO_ORIGINAL, "4x4": cv2.aruco.DICT_4X4_50}
SIZES = {"orig:107": 0.070, "orig:880": 0.070, "orig:467": 0.070, "orig:718": 0.070,
         "4x4:0": 0.050, "4x4:2": 0.050}


def _detector(dict_id):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), p)


_DETECTORS = {k: _detector(v) for k, v in DICTS.items()}


def detect(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Known markers in the image: key -> 4x2 corner pixels (OpenCV order: TL, TR, BR, BL of the marker)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = {}
    for dname, det in _DETECTORS.items():
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            key = f"{dname}:{int(i)}"
            if key in SIZES:
                out[key] = c.reshape(4, 2).astype(np.float64)
    return out


def draw(img: np.ndarray, found: dict[str, np.ndarray]) -> None:
    for key, c in found.items():
        pts = c.round().astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.circle(img, tuple(pts[0]), 3, (0, 0, 255), -1)
        cv2.putText(img, key, tuple(pts[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
