"""Simulated 2D LiDAR (mj_multiRay), clustering and dynamic obstacle tracking."""
from __future__ import annotations

import mujoco
import numpy as np

from .apartment import world_to_cell


def occupied_many(occ, pts):
    r, c = world_to_cell(pts)
    inside = (r >= 0) & (c >= 0) & (r < occ.shape[0]) & (c < occ.shape[1])
    out = np.ones(len(pts), bool)
    out[inside] = occ[r[inside], c[inside]]
    return out


class Lidar2D:
    """Two horizontal rings (chest 1.0 m, low 0.25 m), 180 rays each over 360 deg, 6 m range."""

    def __init__(self, model, robot_body, n_rays=180, rmax=6.0, heights=(1.0, 0.25)):
        self.m = model
        self.robot_body = robot_body
        self.n = n_rays
        self.rmax = rmax
        self.heights = heights
        self.angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        # groups 0 (furniture), 1 (people/pet/objects); robot geoms are group 3
        self.geomgroup = np.array([1, 1, 0, 0, 0, 0], np.uint8)
        nr = n_rays
        self._geomid = np.zeros(nr, np.int32)
        self._dist = np.zeros(nr, np.float64)

    def scan(self, data, pose):
        """Return (ranges[n_rings, n], hit_geom[n_rings, n]) from true sensor pose."""
        x, y, th = pose
        ranges, gids = [], []
        dirs = np.stack([np.cos(self.angles + th), np.sin(self.angles + th), np.zeros(self.n)], 1)
        for h in self.heights:
            mujoco.mj_multiRay(self.m, data, np.array([x, y, h]), dirs.reshape(-1), self.geomgroup, 1,
                               -1, self._geomid, self._dist, None, self.n, self.rmax)
            r = self._dist.copy()
            r[(r < 0) | (r > self.rmax)] = np.inf
            ranges.append(r)
            gids.append(self._geomid.copy())
        return np.array(ranges), np.array(gids)

    def points_world(self, ranges, est_pose):
        """Hit points in world frame using the (noisy) estimated robot pose."""
        x, y, th = est_pose
        pts = []
        for r in ranges:
            ok = np.isfinite(r)
            a = self.angles[ok] + th
            pts.append(np.stack([x + r[ok] * np.cos(a), y + r[ok] * np.sin(a)], 1))
        return np.concatenate(pts, 0) if pts else np.zeros((0, 2))


def cluster_points(pts, gap=0.18, min_pts=2):
    """Greedy euclidean clustering (single linkage, O(N^2) on a few hundred points)."""
    n = len(pts)
    if n == 0:
        return []
    labels = -np.ones(n, int)
    d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1) < gap * gap
    k = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = k
        while stack:
            j = stack.pop()
            nb = np.where(d2[j] & (labels < 0))[0]
            labels[nb] = k
            stack.extend(nb.tolist())
        k += 1
    out = []
    for c in range(k):
        p = pts[labels == c]
        if len(p) >= min_pts:
            ctr = p.mean(0)
            out.append((ctr, float(np.max(np.linalg.norm(p - ctr, axis=1))), p))
    return out


AGENT_MEMORY = 5.0     # s


class Track:
    _next = 0

    def __init__(self, pos, radius, t):
        self.id = Track._next
        Track._next += 1
        self.pos = pos.copy()
        self.vel = np.zeros(2)
        self.radius = radius
        self.t = t
        self.hits = 1
        self.moving_frames = 0
        self.hist = [(t, pos.copy())]
        self.static_evidence = False   # set by the navigator if it sits on long-known obstacle cells
        self.last_dynamic_t = None     # last time it was seen moving (a person who stops is still a person)

    @property
    def dynamic(self):
        """Moving consistently: fast for >= 2 scans AND displaced >= 0.15 m within ~1 s
        (view-dependent centroid jitter of furniture fails the second test)."""
        if self.moving_frames < 2 or self.static_evidence:
            return False
        t_old, p_old = self.hist[0]
        return np.linalg.norm(self.pos - p_old) > 0.15

    @property
    def agent(self):
        """Moving now, or seen moving within the last AGENT_MEMORY s (people stop, then walk on)."""
        return self.dynamic or (self.last_dynamic_t is not None and self.t - self.last_dynamic_t < AGENT_MEMORY)

    def predict(self, t):
        return self.pos + self.vel * (t - self.t)


class ObstacleTracker:
    """Separate unknown obstacles (not on the static map) and track their centroids.

    Velocities are estimated purely from consecutive LiDAR scans (centroid finite
    difference, low-pass filtered) - nothing is read from the simulator.
    """

    def __init__(self, static_occ_dilated, gate=0.6, v_thresh=0.25, max_age=0.6):
        self.occ = static_occ_dilated
        self.gate = gate
        self.v_thresh = v_thresh
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.unknown_pts = np.zeros((0, 2))

    def update(self, pts, t):
        keep = ~occupied_many(self.occ, pts) if len(pts) else np.zeros(0, bool)
        unk = pts[keep]
        self.unknown_pts = unk
        clusters = [c for c in cluster_points(unk) if c[1] < 0.8]  # big blobs = displaced furniture/walls
        used = set()
        for tr in self.tracks:
            pred = tr.predict(t)
            best, bd = None, self.gate
            for i, (ctr, rad, _) in enumerate(clusters):
                if i in used:
                    continue
                dd = np.linalg.norm(ctr - pred)
                if dd < bd:
                    best, bd = i, dd
            if best is not None:
                used.add(best)
                ctr, rad, _ = clusters[best]
                tr.hist.append((t, ctr.copy()))
                while len(tr.hist) > 2 and t - tr.hist[0][0] > 1.0:
                    tr.hist.pop(0)
                # velocity over a ~0.5 s window (less sensitive to centroid jitter)
                t_ref, p_ref = next(((ht, hp) for ht, hp in tr.hist if t - ht <= 0.5), tr.hist[0])
                v_meas = (ctr - p_ref) / max(t - t_ref, 0.1) if t > t_ref else np.zeros(2)
                tr.vel = 0.5 * tr.vel + 0.5 * v_meas
                tr.pos = ctr
                tr.radius = 0.7 * tr.radius + 0.3 * max(rad, 0.1)
                tr.t = t
                tr.hits += 1
                if np.linalg.norm(tr.vel) > self.v_thresh:
                    tr.moving_frames += 1
                else:
                    tr.moving_frames = max(0, tr.moving_frames - 1)
                if tr.dynamic:
                    tr.last_dynamic_t = t
        self.tracks = [tr for tr in self.tracks if t - tr.t <= self.max_age]
        for i, (ctr, rad, _) in enumerate(clusters):
            if i not in used:
                self.tracks.append(Track(ctr, max(rad, 0.1), t))
        return self.tracks

    def dynamic_tracks(self):
        """Moving obstacles and recently moving ones (people who paused): the safety layer and the DWA keep
        their larger clearance to both."""
        return [t for t in self.tracks if t.agent]
