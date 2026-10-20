"""Local planner: Dynamic Window Approach with constant-velocity prediction of moving obstacles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..apartment import CLEARANCE, ROBOT_RADIUS
from ..sonic_bridge import VelLimits


@dataclass
class DWAConfig:
    horizon: float = 2.0       # s, also the prediction horizon for dynamic obstacles
    dt: float = 0.2
    n_vx: int = 9
    n_vy: int = 5
    n_wz: int = 11
    w_goal: float = 1.0
    w_head: float = 0.35
    w_clear: float = 0.25
    w_speed: float = 0.25
    w_dyn: float = 3.0
    safe_margin: float = CLEARANCE  # extra clearance to static points (m)
    dyn_margin: float = 0.35   # extra clearance to predicted dynamic obstacles (m)
    window: float = 0.4        # s of acceleration reachable from current velocity
    vy_track: float = 0.08     # m/s: lateral speed the walking policy really reaches (it under-tracks larger
                               # side-step commands, measured 0.08 m/s for a 0.2 m/s command)


class DWA:
    def __init__(self, limits: VelLimits | None = None, cfg: DWAConfig | None = None):
        self.lim = limits or VelLimits()
        self.cfg = cfg or DWAConfig()

    def plan(self, pose, v_now, target, obstacles, dyn_tracks, t_now, carrying=False, final=False):
        """Pick (vx, vy, wz) for the next 0.1 s.

        pose: (x, y, yaw) estimate; v_now: current body velocity; target: lookahead xy;
        obstacles: (N,2) LiDAR points in world; dyn_tracks: list of Track.
        Returns (cmd, info) where info['ok'] is False if no admissible trajectory.
        """
        c, L = self.cfg, self.lim
        vmax = L.vx_max_carry if carrying else L.vx_max
        w = c.window
        vx_s = np.linspace(max(L.vx_min, v_now[0] - L.acc * w), min(vmax, v_now[0] + L.acc * w), c.n_vx)
        vy_max = min(L.vy_max, c.vy_track)
        vy_s = np.linspace(max(-vy_max, min(v_now[1], vy_max) - L.acc * w),
                           min(vy_max, max(v_now[1], -vy_max) + L.acc * w), c.n_vy)
        wz_s = np.linspace(max(-L.wz_max, v_now[2] - L.alpha * w), min(L.wz_max, v_now[2] + L.alpha * w), c.n_wz)
        VX, VY, WZ = [a.reshape(-1) for a in np.meshgrid(vx_s, vy_s, wz_s, indexing="ij")]
        # always include "stop" and pure rotations so the robot can wait or turn in place
        VX = np.concatenate([VX, [0, 0, 0]])
        VY = np.concatenate([VY, [0, 0, 0]])
        WZ = np.concatenate([WZ, [0, L.wz_max * 0.6, -L.wz_max * 0.6]])

        n_t = int(round(c.horizon / c.dt))
        ts = np.arange(1, n_t + 1) * c.dt
        x0, y0, th0 = pose
        th = th0 + WZ[:, None] * ts[None]                       # (K,T)
        # integrate exactly-ish with small steps
        cth, sth = np.cos(th), np.sin(th)
        dx = (cth * VX[:, None] - sth * VY[:, None]) * c.dt
        dy = (sth * VX[:, None] + cth * VY[:, None]) * c.dt
        X = x0 + np.cumsum(dx, 1)
        Y = y0 + np.cumsum(dy, 1)
        P = np.stack([X, Y], -1)                                 # (K,T,2)

        # static/unknown clearance from LiDAR points: Euclidean distance field on a local grid
        d = self._distance(P, obstacles, (x0, y0))                # (K,T)
        clear = d.min(1) - ROBOT_RADIUS
        # only the first ~1 s must be strictly collision free (the rest is re-planned)
        hard = d[:, : max(2, n_t // 2)].min(1) - ROBOT_RADIUS
        admissible = hard > c.safe_margin
        # current-pose escape: allow trajectories that increase clearance when already too close
        d0 = self._distance(np.array([[[x0, y0]]]), obstacles, (x0, y0))[0, 0] - ROBOT_RADIUS
        if d0 <= c.safe_margin:
            admissible |= d[:, 1] > d0 + ROBOT_RADIUS + 1e-3

        # dynamic obstacles: constant-velocity prediction over the horizon
        dyn_pen = np.zeros(len(VX))
        for tr in dyn_tracks:
            pred = tr.pos[None] + tr.vel[None] * (ts[:, None] + (t_now - tr.t))   # (T,2)
            dd = np.linalg.norm(P - pred[None], axis=-1) - ROBOT_RADIUS - tr.radius   # (K,T)
            viol = np.clip(c.dyn_margin - dd, 0, None)
            dyn_pen += (viol * np.exp(-0.5 * ts[None])).sum(1)
            admissible &= dd[:, :3].min(1) > 0.05  # never drive into a person in the next 0.6 s

        tgt = np.asarray(target)
        end = P[:, -1]
        goal_d = np.linalg.norm(end - tgt, axis=1)
        mid = P[:, n_t // 2]
        goal_d = 0.5 * goal_d + 0.5 * np.linalg.norm(mid - tgt, axis=1)
        desired = np.arctan2(tgt[1] - Y[:, -1], tgt[0] - X[:, -1])
        head_err = np.abs((desired - th[:, -1] + np.pi) % (2 * np.pi) - np.pi)
        if final:
            head_err *= 0.2
        cost = (c.w_goal * goal_d + c.w_head * head_err + c.w_clear / np.clip(clear, 0.05, None) * 0.1
                + c.w_speed * (vmax - VX) / vmax + c.w_dyn * dyn_pen + 0.3 * np.abs(VY))
        if not admissible.any():
            return np.zeros(3), {"ok": False}
        cost[~admissible] = np.inf
        k = int(np.argmin(cost))
        return np.array([VX[k], VY[k], WZ[k]]), {"ok": True, "traj": P[k], "clear": float(clear[k])}

    @staticmethod
    def _distance(P, obstacles, center, half=3.5, res=0.04):
        """Distance (m) from each point of P (...,2) to the nearest obstacle point."""
        if not len(obstacles):
            return np.full(P.shape[:-1], 5.0)
        n = int(2 * half / res)
        o = np.array(center) - half
        ij = np.floor((obstacles - o) / res).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        if not ok.any():
            return np.full(P.shape[:-1], 5.0)
        grid = np.ones((n, n), bool)
        grid[ij[ok, 1], ij[ok, 0]] = False
        edt = distance_transform_edt(grid) * res
        q = np.clip(np.floor((P - o) / res).astype(int), 0, n - 1)
        out = edt[q[..., 1], q[..., 0]]
        outside = ((P - o) < 0).any(-1) | ((P - o) >= 2 * half).any(-1)
        return np.where(outside, np.minimum(out, 0.5), out)
