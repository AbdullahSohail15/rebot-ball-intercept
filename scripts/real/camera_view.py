"""Live RealSense preview for aiming the camera (reads from the camera server, scripts/real/rs_server.py).

Shows the colour stream with a rule-of-thirds grid, the frame rate and frame age, and outlines any
orange blob (the ball) so you can check it stays visible across the table.

    scripts/real/start_camera.sh                      # once per session (password dialog)
    .venv/bin/python scripts/real/camera_view.py      # q / Esc to quit, s to save a snapshot
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eaiproject import markers  # noqa: E402
from eaiproject.realsense_client import RealSenseClient  # noqa: E402

HSV_LO, HSV_HI = (5, 120, 120), (25, 255, 255)   # orange ping-pong ball (rough; tuned later)


def main() -> int:
    try:
        cam = RealSenseClient()
    except OSError:
        print("Camera server not running. Start it with: scripts/real/start_camera.sh")
        return 2
    win = "RealSense live (q quit, s snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cam.width * (2 if cam.width < 900 else 1), cam.height * (2 if cam.width < 900 else 1))
    last_fn, t_prev, fps = None, time.monotonic(), 0.0
    out_dir = ROOT / "outputs" / "realsense_scene"
    while True:
        f = cam.read(timeout=2.0, newer_than=last_fn)
        last_fn = f.frame_number
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        img = f.img.copy()
        h, w = img.shape[:2]
        for k in (1, 2):
            cv2.line(img, (w * k // 3, 0), (w * k // 3, h), (200, 200, 200), 1)
            cv2.line(img, (0, h * k // 3), (w, h * k // 3), (200, 200, 200), 1)
        mask = cv2.inRange(cv2.cvtColor(f.img, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, 4] >= 15:
                x, y, bw, bh = stats[i, :4]
                cv2.rectangle(img, (x - 4, y - 4), (x + bw + 4, y + bh + 4), (0, 255, 0), 2)
                cv2.putText(img, f"ball? {stats[i, 4]}px", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        found = markers.detect(f.img)
        markers.draw(img, found)
        cv2.putText(img, f"markers: {len(found)}  {sorted(found)}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        age_ms = (now - f.t_host) * 1000
        cv2.putText(img, f"{fps:4.1f} fps  age {age_ms:3.0f} ms  {w}x{h}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, f"{fps:4.1f} fps  age {age_ms:3.0f} ms  {w}x{h}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"snap_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(p), f.img)
            print("saved", p)
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cam.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
