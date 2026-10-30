"""
Pedestrian tracking and predictive (time-to-collision style) braking for the offline replay.

Tracker: one constant-velocity Kalman filter per person, in WORLD coordinates (x, y in meters), so the person's own
velocity is estimated independently of the forklift's motion. Detections are associated greedily by Mahalanobis
distance with a chi-square gate; unmatched detections start new tracks; a track is confirmed after `min_hits`
associated detections and deleted after `max_miss` consecutive frames without one (it coasts on its prediction in
between, which bridges single-frame detection drop-outs).

Predictive check (used by the 'ttc' and 'hybrid' controllers in replay_braking.py): for every confirmed track, the
person is extrapolated at constant velocity and the forklift is assumed to keep its current speed and heading until
it brakes. The question asked each frame is "if I do NOT brake now but only at the next perception frame, will the
person be inside the corridor and closer than the safety margin to the fork tips at some moment before the forklift
stops?" If yes, brake now. This is the predictive counterpart of the position-only zone: same corridor, margin,
latency, dead time and deceleration, but it uses where the person will be instead of where the person is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHI2_2DOF_99 = 9.21


@dataclass
class TrackerParams:
    meas_sigma: float = 0.20        # m, position noise of one observation
    accel_sigma: float = 2.0        # m/s^2, white-acceleration process noise (people start, stop, turn)
    init_vel_sigma: float = 1.5     # m/s, velocity uncertainty of a new track
    gate: float = CHI2_2DOF_99      # Mahalanobis gate (squared)
    min_hits: int = 2               # associated detections before a track counts
    max_miss: int = 2               # frames a track may coast without a detection


@dataclass
class Track:
    tid: int
    x: np.ndarray                   # [x, y, vx, vy] world
    P: np.ndarray
    t: float
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)

    def confirmed(self, p: TrackerParams) -> bool:
        return self.hits >= p.min_hits


def _F(dt):
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def _Q(dt, sa):
    g = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], float)
    return g @ g.T * sa * sa


H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


class Tracker:
    def __init__(self, params: TrackerParams):
        self.p = params
        self.tracks: list[Track] = []
        self.next_id = 0

    def _predict(self, tr: Track, t: float):
        dt = max(0.0, t - tr.t)
        if dt > 0:
            F = _F(dt)
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + _Q(dt, self.p.accel_sigma)
            tr.t = t

    def step(self, t: float, meas: list[tuple[float, float, float]]) -> list[Track]:
        """meas: list of (x_world, y_world, confidence). Returns the live tracks after the update."""
        for tr in self.tracks:
            self._predict(tr, t)
        R = np.eye(2) * self.p.meas_sigma ** 2
        pairs = []
        for ti, tr in enumerate(self.tracks):
            S = H @ tr.P @ H.T + R
            Si = np.linalg.inv(S)
            for mi, (mx, my, _) in enumerate(meas):
                r = np.array([mx, my]) - H @ tr.x
                d2 = float(r @ Si @ r)
                if d2 <= self.p.gate:
                    pairs.append((d2, ti, mi))
        pairs.sort()
        used_t, used_m = set(), set()
        for d2, ti, mi in pairs:
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            tr = self.tracks[ti]
            S = H @ tr.P @ H.T + R
            K = tr.P @ H.T @ np.linalg.inv(S)
            z = np.array(meas[mi][:2])
            tr.x = tr.x + K @ (z - H @ tr.x)
            tr.P = (np.eye(4) - K @ H) @ tr.P
            tr.hits += 1
            tr.misses = 0
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.p.max_miss]
        for mi, (mx, my, _) in enumerate(meas):
            if mi in used_m:
                continue
            P0 = np.diag([self.p.meas_sigma ** 2, self.p.meas_sigma ** 2,
                          self.p.init_vel_sigma ** 2, self.p.init_vel_sigma ** 2])
            self.tracks.append(Track(self.next_id, np.array([mx, my, 0.0, 0.0]), P0, t))
            self.next_id += 1
        for tr in self.tracks:
            tr.history.append((t, *tr.x, tr.misses))
        return self.tracks


def predicted_conflict(p_fwd: float, p_lat: float, v_fwd: float, v_lat: float, v: float, decel: float,
                       wait_delay: float, half_width: float, margin: float, dt: float = 0.05,
                       min_horizon: float = 0.0) -> tuple[bool, float]:
    """Person at (p_fwd ahead of the fork tips, p_lat to the side) moving at (v_fwd, v_lat), both in the forklift's
    current frame (person's ground velocity, not relative). Forklift at speed v keeps going for wait_delay seconds,
    then decelerates at decel until it stops. Returns (conflict, smallest predicted gap inside the corridor)."""
    v = max(0.0, v)
    t_stop = wait_delay + (v / decel if decel > 0 else 0.0)
    horizon = max(t_stop, min_horizon)
    tau = np.arange(0.0, horizon + dt, dt)
    tb = np.clip(tau - wait_delay, 0.0, v / decel if decel > 0 else 0.0)
    s = v * np.minimum(tau, wait_delay) + v * tb - 0.5 * decel * tb * tb
    pf = p_fwd + v_fwd * tau
    pl = p_lat + v_lat * tau
    gap = pf - s
    inside = (np.abs(pl) <= half_width) & (gap >= -0.5)
    if not inside.any():
        return False, math.inf
    g = float(gap[inside].min())
    return g < margin, g
