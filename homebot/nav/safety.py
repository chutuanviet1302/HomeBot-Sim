"""Safety layer (50 Hz, highest priority): stop / slow down near moving obstacles.

* moving obstacle surface distance < 0.6 m  -> stop (and, if it keeps closing in and the
  space behind is free, back off slowly: a stopped robot can still be walked into)
* 0.6 - 1.2 m                               -> cap speed to 0.2 m/s
* time to contact (closing speed of robot and obstacle) < 1.0 s -> stop, < 2.0 s -> slow: the walking
  policy needs ~0.5 s to brake, a person crossing at 1 m/s closes 0.6 m in that time
It overrides every other command.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..apartment import ROBOT_RADIUS


@dataclass
class SafetyConfig:
    stop_dist: float = 0.6
    slow_dist: float = 1.2
    slow_speed: float = 0.2
    evade: bool = True
    evade_speed: float = 0.25
    ttc_stop: float = 1.0
    ttc_slow: float = 2.0


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig | None = None, enabled=True):
        self.cfg = cfg or SafetyConfig()
        self.enabled = enabled
        self.state = "clear"

    def filter(self, cmd, pose, dyn_tracks, t_now, static_pts=None, robot_vel=None):
        if not self.enabled:
            self.state = "off"
            return cmd
        c = self.cfg
        x, y, th = pose
        p = np.array([x, y])
        nearest, nt = np.inf, None
        ttc = np.inf
        v_r = np.zeros(2) if robot_vel is None else np.asarray(robot_vel, float)[:2]
        for tr in dyn_tracks:
            rel = tr.predict(t_now) - p
            gap = np.linalg.norm(rel) - ROBOT_RADIUS - tr.radius
            closing = -np.dot(tr.vel - v_r, rel) / (np.linalg.norm(rel) + 1e-6)
            if closing > 0.05:
                ttc = min(ttc, max(gap, 0.0) / closing)
            # distance now and in 0.3 s (constant velocity), whichever is smaller
            d = min(np.linalg.norm(tr.predict(t_now + dt) - p) for dt in (0.0, 0.15, 0.3))
            d -= ROBOT_RADIUS + tr.radius
            if d < nearest:
                nearest, nt = d, tr
        cmd = np.array(cmd, float)
        if nearest < c.stop_dist or ttc < c.ttc_stop:
            self.state = "stop"
            out = np.zeros(3)
            if c.evade and nt is not None:
                rel = nt.predict(t_now) - p
                closing = -np.dot(nt.vel, rel) / (np.linalg.norm(rel) + 1e-6)
                if closing > 0.2 and nearest < 0.35:
                    away = -rel / (np.linalg.norm(rel) + 1e-6)
                    # robot-frame direction, limited to what SONIC accepts
                    bx = np.cos(th) * away[0] + np.sin(th) * away[1]
                    by = -np.sin(th) * away[0] + np.cos(th) * away[1]
                    v = np.array([bx, by]) * c.evade_speed
                    if static_pts is not None and len(static_pts):
                        probe = p + away * 0.45
                        if np.linalg.norm(static_pts - probe, axis=1).min() < ROBOT_RADIUS + 0.05:
                            v[:] = 0.0
                    out[:2] = [np.clip(v[0], -0.3, 0.3), np.clip(v[1], -0.2, 0.2)]
                    self.state = "evade"
            return out
        if nearest < c.slow_dist or ttc < c.ttc_slow:
            self.state = "slow"
            sp = np.hypot(cmd[0], cmd[1])
            if sp > c.slow_speed:
                cmd[:2] *= c.slow_speed / sp
            return cmd
        self.state = "clear"
        return cmd
