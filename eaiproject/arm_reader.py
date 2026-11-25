"""Read-only joint stream of the UNPOWERED arm for the calibration scripts.

A background thread polls the six arm joints (parameter reads only, RobStrideBus(movable=()), so
the CAN wrapper refuses any enable / move / write) and keeps the last few seconds of samples:
joint angles, Fingertip(q) and the gripper orientation, all in the arm's kinematic frame.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ArmSample:
    t: float                # time.monotonic()
    q: np.ndarray           # (6,) joint angles, rad
    p: np.ndarray           # (3,) fingertip, base frame
    R: np.ndarray           # (3, 3) gripper_end orientation


class ArmReader:
    def __init__(self, hz: float = 50.0, history_s: float = 20.0):
        from .kinematics import Fingertip
        from .robstride import ARM_IDS, P_MECHPOS, RobStrideBus

        self._ids, self._param = ARM_IDS, P_MECHPOS
        self.rs = RobStrideBus(movable=())               # read-only: nothing can be enabled or moved
        self._tip, self._tip_main = Fingertip(), Fingertip()
        self.period = 1.0 / hz
        self.history: deque[ArmSample] = deque(maxlen=int(hz * history_s))
        self._lock = threading.Lock()
        self._stop = False
        first = self._read()
        if first is None:
            self.rs.close()
            raise ConnectionError("arm not answering on CAN (powered? cable?)")
        self.history.append(first)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read(self) -> ArmSample | None:
        q = [self.rs.read_param_f32(m, self._param, timeout=0.03) for m in self._ids]
        if any(v is None for v in q):
            return None
        p = self._tip(q)
        return ArmSample(time.monotonic(), np.array(q), p, self._tip.d.xmat[self._tip.body].reshape(3, 3).copy())

    def _loop(self) -> None:
        nxt = time.monotonic()
        while not self._stop:
            try:
                s = self._read()
            except Exception:                            # bus closed / adapter unplugged: stop quietly
                break
            if s is not None:
                with self._lock:
                    self.history.append(s)
            nxt = max(nxt + self.period, time.monotonic() - self.period)
            time.sleep(max(0.0, nxt - time.monotonic()))

    def latest(self, max_age: float = 0.3) -> ArmSample | None:
        with self._lock:
            s = self.history[-1] if self.history else None
        return s if s is not None and time.monotonic() - s.t <= max_age else None

    def between(self, t0: float, t1: float) -> list[ArmSample]:
        with self._lock:
            return [s for s in self.history if t0 < s.t <= t1]

    def steady(self, window: float = 0.25) -> tuple[ArmSample, float] | None:
        """Mean of the last `window` seconds as one sample, and how far the fingertip moved in it (m)."""
        now = time.monotonic()
        ss = self.between(now - window, now)
        if len(ss) < 3:
            return None
        q = np.mean([s.q for s in ss], axis=0)
        P = np.array([s.p for s in ss])
        p = self._tip_main(q)
        R = self._tip_main.d.xmat[self._tip_main.body].reshape(3, 3).copy()
        return ArmSample(now, q, p, R), float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.rs.close()
