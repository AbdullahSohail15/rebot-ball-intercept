"""Render the ChArUco calibration board to an A4 PDF at exact physical size.

    .venv/bin/python scripts/real/make_calib_board.py      # -> outputs/calibration/charuco_board_A4.pdf

Print at 100% / "Actual size" (no "fit to page"), then check the 100 mm ruler with a real ruler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import calib_board as cb  # noqa: E402

DPI = 600
PAGE_W_MM, PAGE_H_MM = 297.0, 210.0      # A4 landscape


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def main() -> None:
    page = np.full((mm(PAGE_H_MM), mm(PAGE_W_MM)), 255, np.uint8)
    bw, bh = cb.W * 1000, cb.H * 1000                 # 245 x 175 mm
    ox, oy = (PAGE_W_MM - bw) / 2, 12.0              # board top-left on the page (mm)
    img = cb.board().generateImage((mm(bw), mm(bh)), marginSize=0, borderBits=1)
    page[mm(oy):mm(oy) + img.shape[0], mm(ox):mm(ox) + img.shape[1]] = img

    # touch points: A top-left, B top-right, C bottom-right, D bottom-left (board frame: A origin,
    # x towards B, y towards D)
    pts = {"A": (ox, oy), "B": (ox + bw, oy), "C": (ox + bw, oy + bh), "D": (ox, oy + bh)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, (x, y) in pts.items():
        c = (mm(x), mm(y))
        cv2.circle(page, c, mm(4.0), 0, mm(0.4))
        cv2.line(page, (c[0] - mm(6), c[1]), (c[0] + mm(6), c[1]), 0, mm(0.25))
        cv2.line(page, (c[0], c[1] - mm(6)), (c[0], c[1] + mm(6)), 0, mm(0.25))
        dx = -mm(9) if name in "AD" else mm(5)
        dy = -mm(3) if name in "AB" else mm(8)
        cv2.putText(page, name, (c[0] + dx, c[1] + dy), font, mm(1.6) / 22, 0, mm(0.6))

    # scale check ruler (100 mm) and instructions at the bottom
    y = mm(oy + bh + 10)
    x0 = mm(ox)
    cv2.line(page, (x0, y), (x0 + mm(100), y), 0, mm(0.4))
    for k in range(11):
        tick = mm(3 if k % 5 else 5)
        cv2.line(page, (x0 + mm(10 * k), y), (x0 + mm(10 * k), y - tick), 0, mm(0.3))
    lines = ["100 mm: check with a ruler (print at 100% / Actual size)",
             f"ChArUco {cb.SQUARES_X}x{cb.SQUARES_Y}, square {cb.SQUARE*1000:.0f} mm, marker {cb.MARKER*1000:.0f} mm, DICT_4X4_50",
             "Touch A, B, C, D (circled outer checker corners) with the closed fingertips"]
    for k, t in enumerate(lines):
        cv2.putText(page, t, (x0 + mm(110), y - mm(2) + k * mm(5)), font, mm(1.0) / 22, 0, mm(0.25))

    out = ROOT / "outputs" / "calibration" / "charuco_board_A4.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page).save(out, "PDF", resolution=DPI)
    Image.fromarray(cv2.resize(page, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)).save(out.with_suffix(".preview.png"))
    print(out)


if __name__ == "__main__":
    main()
