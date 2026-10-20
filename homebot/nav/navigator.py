"""Three-layer navigation: A* (1 Hz) -> DWA (10 Hz) -> safety layer (50 Hz)."""
from __future__ import annotations

import numpy as np

from .. import apartment as apt
from ..lidar import Lidar2D, ObstacleTracker
from .astar import astar, inflate, path_to_waypoints
from .dwa import DWA
from .safety import SafetyLayer

INFLATE = apt.ROBOT_RADIUS + apt.CLEARANCE + apt.PLAN_MARGIN + 0.03
HIT, MISS, OCC_MAX = 1.0, 0.35, 4.0   # evidence model of the obstacle memory grid


class Navigator:
    def __init__(self, env, safety=True, lidar_rays=180):
        self.env = env
        self.lidar = Lidar2D(env.m, env.robot_body, n_rays=lidar_rays)
        self.static_occ = apt.static_map()
        self.tracker = ObstacleTracker(inflate(self.static_occ, 0.10))
        self.dwa = DWA(env.sonic.lim)
        self.safety = SafetyLayer(enabled=safety)
        self.cmd = np.zeros(3)
        self.goal = None
        self.path = None
        self.wp_idx = 0
        self.last_global = -1e9
        self.last_local = -1e9
        self.blocked_since = None
        # memory of obstacles that are not on the map (displaced chairs, ...): evidence grid,
        # raised by LiDAR hits, cleared by low-ring rays passing through free space
        self.mem = np.zeros_like(self.static_occ, dtype=np.float32)
        self.pts = np.zeros((0, 2))
        self.est = env.pose.copy()
        self.failed_plans = 0
        self.log = []

    def set_goal(self, xy, tol=0.15):
        self.goal = np.asarray(xy, float)
        self.tol = tol
        self.path = None
        self.last_global = -1e9
        self.blocked_since = None

    def _global_map(self, t, radius=INFLATE):
        occ = self.static_occ.copy()
        seen = self.mem >= 1.0
        if t < getattr(self, "avoid_agents_until", -1.0):
            # dead-lock recovery: plan around the people / pet currently blocking the way
            rr, cc = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
            ctr = apt.cell_to_world(rr, cc)
            for tr in self.tracker.dynamic_tracks():
                occ |= np.linalg.norm(ctr - tr.pos, axis=-1) < tr.radius + 0.1
        # LiDAR memory: a hit marks its whole cell, whose centre lies up to half a cell in front of the surface,
        # so remembered obstacles are inflated half a cell less (else a displaced chair looked ~3 cm larger than
        # it is and could close a walkway the robot fits through)
        inf = inflate(occ, radius) | inflate(seen, radius - 0.5 * apt.GRID_RES)
        occ |= seen
        # never block the cells under the robot itself
        rr, cc = apt.world_to_cell(self.est[:2])
        k = int(radius / apt.GRID_RES)                 # (start never inside the inflation)
        inf[max(rr - k, 0):rr + k + 1, max(cc - k, 0):cc + k + 1] &= occ[max(rr - k, 0):rr + k + 1, max(cc - k, 0):cc + k + 1]
        return inf

    def _mark_static_tracks(self):
        """A cluster sitting on cells that have been occupied for a long time is furniture."""
        for tr in self.tracker.tracks:
            r, c = apt.world_to_cell(tr.pos)
            if 0 <= r < self.mem.shape[0] and 0 <= c < self.mem.shape[1]:
                win = self.mem[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
                tr.static_evidence = bool(win.max() >= OCC_MAX - 0.01) and np.linalg.norm(tr.vel) < 0.6

    def _update_memory(self, ranges):
        H, W = self.mem.shape
        x, y, th = self.est
        # 1) free space along the low ring (it sees every floor-standing obstacle)
        r = np.minimum(ranges[-1], 3.5)
        steps = np.arange(0.3, 3.5, apt.GRID_RES)
        a = self.lidar.angles + th
        D = steps[None, :]
        free = D < (r[:, None] - 0.12)
        px = x + D * np.cos(a)[:, None]
        py = y + D * np.sin(a)[:, None]
        rr, cc = np.floor(py[free] / apt.GRID_RES).astype(int), np.floor(px[free] / apt.GRID_RES).astype(int)
        ok = (rr >= 0) & (cc >= 0) & (rr < H) & (cc < W)
        self.mem[rr[ok], cc[ok]] = np.maximum(self.mem[rr[ok], cc[ok]] - MISS, 0)
        # 2) hits on unknown obstacles that are not moving tracks
        unk = self.tracker.unknown_pts
        if len(unk):
            unk = unk[np.linalg.norm(unk - self.est[:2], axis=1) < 3.5]
            keep = np.ones(len(unk), bool)
            for tr in self.tracker.tracks:
                if tr.agent or (tr.hits < 3 and tr.moving_frames > 0):
                    keep &= np.linalg.norm(unk - tr.pos, axis=1) > tr.radius + 0.15
            rr, cc = apt.world_to_cell(unk[keep])
            ok = (rr >= 0) & (cc >= 0) & (rr < H) & (cc < W)
            self.mem[rr[ok], cc[ok]] = np.minimum(self.mem[rr[ok], cc[ok]] + HIT, OCC_MAX)

    def _polyline_len(self, wps):
        pts = np.vstack([self.est[:2], np.asarray(wps)])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def _path_still_free(self, t):
        """Is the remaining current path still collision-free in the latest map (minimum inflation)?"""
        occ = self._global_map(t, apt.ROBOT_RADIUS + apt.CLEARANCE)
        pts = np.vstack([self.est[:2], self.path[self.wp_idx:]])
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.1))
            for s_ in np.linspace(0, 1, n)[1:]:
                if apt.occupied(occ, a + s_ * (b - a)):
                    return False
        return True

    def _replan(self, t, force=False):
        path = None
        # tight spots (displaced chairs): retry with smaller inflation; DWA still enforces clearance
        # (never below the DWA's hard clearance: a gap A* accepts must be one the DWA can drive through)
        for radius in (INFLATE, apt.ROBOT_RADIUS + apt.CLEARANCE + apt.PLAN_MARGIN + 0.015,
                       apt.ROBOT_RADIUS + apt.CLEARANCE + apt.PLAN_MARGIN):
            path = astar(self._global_map(t, radius), self.est[:2], self.goal)
            if path is not None:
                break
        if path is None:
            # recovery (like ROS "clear costmap"): forget remembered obstacles farther than
            # 1.5 m - they will be re-observed if they are really there
            rr, cc = np.mgrid[0:self.mem.shape[0], 0:self.mem.shape[1]]
            far = np.linalg.norm(apt.cell_to_world(rr, cc) - self.est[:2], axis=-1) > 1.5
            self.mem[far] = 0.0
            path = astar(self._global_map(t, apt.ROBOT_RADIUS + apt.CLEARANCE + apt.PLAN_MARGIN), self.est[:2], self.goal)
        self.last_global = t
        if path is None:
            self.failed_plans += 1
            return False
        new = path_to_waypoints(path, 0.5, self.goal)
        # hysteresis: keep following the current route unless it got blocked or the new one is clearly shorter
        # (two near-equal routes around an obstacle otherwise flip at every 1 Hz replan and the robot dithers)
        if (not force and self.path is not None and self.wp_idx < len(self.path)
                and self._polyline_len(new) > 0.8 * self._polyline_len(self.path[self.wp_idx:]) - 0.3
                and self._path_still_free(t)):
            return True
        self.path = new
        self.wp_idx = 0
        return True

    def _lookahead(self, dist=0.9):
        p = self.est[:2]
        while self.wp_idx < len(self.path) - 1 and np.linalg.norm(self.path[self.wp_idx] - p) < dist:
            self.wp_idx += 1
        return self.path[self.wp_idx]

    def at_goal(self):
        return self.goal is not None and np.linalg.norm(self.env.pose[:2] - self.goal) < self.tol

    def step(self, carrying=False):
        """Compute and apply one 50 Hz command. Returns True when the goal is reached."""
        env = self.env
        t = env.t
        self.est = env.est_pose()
        # LiDAR at 10 Hz -> tracker + local planner
        if t - self.last_local >= 0.1 - 1e-9:
            ranges, _ = self.lidar.scan(env.d, env.pose)
            self.pts = self.lidar.points_world(ranges, self.est)
            self.tracker.update(self.pts, t)
            self._mark_static_tracks()
            self._update_memory(ranges)
            if self.goal is not None:
                if self.path is None or t - self.last_global >= 1.0:
                    self._replan(t)
                if self.path is not None:
                    final = np.linalg.norm(self.goal - self.est[:2]) < 0.9
                    tgt = self._lookahead()
                    cmd, info = self.dwa.plan(self.est, env.vel_body(), tgt, self.pts,
                                              self.tracker.dynamic_tracks(), t, carrying, final)
                    if final:
                        # slow, precise final approach (holonomic P-controller in robot frame)
                        e = self.goal - self.est[:2]
                        c, s = np.cos(self.est[2]), np.sin(self.est[2])
                        ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
                        if np.linalg.norm(e) < 0.45:
                            pc = np.array([np.clip(1.2 * ex, -0.2, 0.3), np.clip(1.2 * ey, -0.2, 0.2), 0.0])
                            if self._clear_ahead(pc) > 0.03:
                                cmd, info = pc, {"ok": True}
                    # blind corner: slow near a doorway (someone may step out of the occluded side)
                    if min(np.linalg.norm(self.est[:2] - dw) for dw in apt.DOORWAYS) < apt.BLIND_RADIUS:
                        sp = np.hypot(cmd[0], cmd[1])
                        if sp > apt.BLIND_SPEED:
                            cmd = np.array([cmd[0] * apt.BLIND_SPEED / sp, cmd[1] * apt.BLIND_SPEED / sp, cmd[2]])
                    self.cmd = cmd
                    moving = np.linalg.norm(cmd[:2]) > 0.05 or abs(cmd[2]) > 0.1
                    if not info["ok"] or not moving:
                        self.blocked_since = self.blocked_since or t
                        if t - self.blocked_since > 3.0:      # blocked > 3 s -> replan now
                            if any(np.linalg.norm(tr.pos - self.est[:2]) < 1.5 for tr in self.tracker.dynamic_tracks()):
                                self.avoid_agents_until = t + 8.0     # someone stands in the way: go around
                            self._replan(t, force=True)
                            self.blocked_since = t
                    else:
                        self.blocked_since = None
                else:
                    self.cmd = np.zeros(3)
            self.last_local = t
        cmd = self.safety.filter(self.cmd, self.est, self.tracker.dynamic_tracks(), t, self.pts,
                                 env._robot_world_vel())
        if self.at_goal():
            cmd = np.zeros(3)
        env.step(cmd, carrying)
        return self.at_goal()

    def _clear_ahead(self, cmd):
        if not len(self.pts):
            return 9.0
        x, y, th = self.est
        v = np.array([np.cos(th) * cmd[0] - np.sin(th) * cmd[1], np.sin(th) * cmd[0] + np.cos(th) * cmd[1]])
        p = np.array([x, y]) + v * 0.5
        return np.linalg.norm(self.pts - p, axis=1).min() - apt.ROBOT_RADIUS

    def turn_to(self, yaw, tol=0.06, max_t=8.0):
        t0 = self.env.t
        while self.env.t - t0 < max_t:
            err = (yaw - self.env.pose[2] + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < tol and abs(self.env.vel_body()[2]) < 0.08:
                return True
            cmd = np.array([0.0, 0.0, np.clip(1.5 * err, -0.6, 0.6)])
            self.env.step(cmd)
        return False

    def straight_to(self, xy, timeout=30.0, carrying=False, tol=0.04, speed=0.2):
        """Holonomic straight-line move with fixed heading (final approach / backing out of
        tight spots). Uses LiDAR clearance along the motion and the safety layer."""
        env = self.env
        t0 = env.t
        goal = np.asarray(xy, float)
        while env.t - t0 < timeout:
            self.est = env.est_pose()
            if env.t - self.last_local >= 0.1 - 1e-9:
                ranges, _ = self.lidar.scan(env.d, env.pose)
                self.pts = self.lidar.points_world(ranges, self.est)
                self.tracker.update(self.pts, env.t)
                self.last_local = env.t
            e = goal - self.est[:2]
            if np.linalg.norm(goal - env.pose[:2]) < tol:
                for _ in range(10):
                    env.step(np.zeros(3), carrying)
                return {"success": True, "time": env.t - t0}
            c, s = np.cos(self.est[2]), np.sin(self.est[2])
            ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
            v = np.array([ex, ey]) * 1.5
            n = np.linalg.norm(v)
            if n > speed:
                v *= speed / n
            cmd = np.array([v[0], np.clip(v[1], -0.2, 0.2), 0.0])
            # blocked only if the move brings the footprint closer to an obstacle it already (nearly) touches;
            # backing away from the table / counter edge is always allowed
            ahead = self._clear_ahead(cmd)
            if ahead < 0.0 and ahead < self._clear_ahead(np.zeros(3)) + 1e-3:
                cmd[:] = 0
            cmd = self.safety.filter(cmd, self.est, self.tracker.dynamic_tracks(), env.t, self.pts,
                                     env._robot_world_vel())
            env.step(cmd, carrying)
        return {"success": False, "time": env.t - t0}

    def go(self, xy, timeout=90.0, carrying=False, tol=0.15):
        """Blocking navigation. Returns dict with success and timing."""
        self.set_goal(xy, tol)
        t0 = self.env.t
        while self.env.t - t0 < timeout:
            if self.step(carrying):
                # let the robot come to rest
                for _ in range(15):
                    self.env.step(np.zeros(3), carrying)
                return {"success": True, "time": self.env.t - t0}
            if self.env.fallen():
                return {"success": False, "time": self.env.t - t0, "fell": True}
        return {"success": False, "time": self.env.t - t0}
