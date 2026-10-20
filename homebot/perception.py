"""Object perception from the robot's head RGB-D camera: detection -> recognition -> 6D pose -> tracking.

Only sensor data is used: the rendered colour + depth image of the head camera and the camera pose from the robot's
own kinematics (no simulator segmentation, no object state).

1. Detection: the depth image is back-projected to a world point cloud; RANSAC finds the dominant horizontal
   support plane (table / counter); points 1-30 cm above it are grouped into objects by connected components
   on a 1 cm grid.
2. Recognition: every cluster is registered (ICP) against each model of the YCB library (points sampled from
   the YCB meshes); the label is the model with the lowest ICP residual, rejected if no model fits (rmse) or
   the cluster size disagrees with the model.
3. 6D pose: ICP gives the full rotation + translation of the model (body frame = the simulator's object frame:
   axis centre, half height); it is initialised upright on the plane at several yaw angles.
4. Tracking: a Kalman filter per object (position, random-walk model: objects rest until moved) with
   nearest-neighbour gating, track IDs, confirmation after 2 hits and deletion after missed frames.
5. Receptacles: a sink basin is found as a hole in a horizontal support plane with surface points below it
   (a depression, not an occlusion); its rim is fitted as a rectangle (centre, size, orientation, depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label as cc_label
from scipy.spatial import cKDTree

from . import apartment as apt
from .env import mat_to_quat


# ----------------------------------------------------------------------------- model library
def _load_obj_vertices(path: Path) -> np.ndarray:
    v = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.asarray(v, float)


def _voxel_down(p: np.ndarray, s: float) -> np.ndarray:
    k = np.floor(p / s).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return p[np.sort(idx)]


@dataclass
class ObjModel:
    name: str
    pts: np.ndarray                     # model points in the object body frame
    r: float
    h: float
    tree: cKDTree = field(repr=False, default=None)


def load_library(names=None) -> dict[str, ObjModel]:
    lib = {}
    for name, c in apt.YCB_LIBRARY.items():
        if names is not None and name not in names:
            continue
        f = apt.YCB_DIR / c["ycb"] / "google_16k" / "textured.obj"
        if not f.exists():
            continue
        v = _load_obj_vertices(f) - np.array([c["centre"][0], c["centre"][1], c["h"] / 2])
        v = _voxel_down(v, 0.004)
        lib[name] = ObjModel(name, v, c["r"], c["h"], cKDTree(v))
    return lib


# ----------------------------------------------------------------------------- geometry helpers
def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _kabsch(a, b):
    """R, t minimising |R a + t - b|."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def icp(obs, model: ObjModel, R, t, iters=25, trim=0.9):
    """Point-to-point ICP of the observed points onto the model (observation -> model frame), trimmed.
    Returns the object pose in the world (R, t) and the inlier RMSE."""
    Ri, ti = R.T, -R.T @ t                # world -> model
    rmse = np.inf
    for _ in range(iters):
        q = obs @ Ri.T + ti
        d, j = model.tree.query(q)
        keep = d <= np.quantile(d, trim)
        dR, dt = _kabsch(q[keep], model.pts[j[keep]])
        Ri, ti = dR @ Ri, dR @ ti + dt
        new = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(rmse - new) < 1e-6:
            rmse = new
            break
        rmse = new
    R = Ri.T
    return R, -R @ ti, rmse


@dataclass
class Detection:
    label: str
    pos: np.ndarray                     # object body origin in the world
    R: np.ndarray
    rmse: float
    n_pts: int
    bbox: tuple                         # image bbox (u0, v0, u1, v1)

    @property
    def quat(self):
        return mat_to_quat(self.R)


@dataclass
class Basin:
    center: np.ndarray                  # centre of the opening at rim height (world)
    half: np.ndarray                    # half size of the opening along (axis u, axis v)
    yaw: float                          # direction of axis u (long side)
    depth: float                        # rim height - lowest visible floor
    n_below: int                        # points seen inside the depression

    def contains(self, p, margin=0.0):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(p)[:2] - self.center[:2]
        uv = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        return bool(np.all(np.abs(uv) <= self.half - margin))


# ----------------------------------------------------------------------------- tracker
class Track:
    """One object track: position filtered with a random-walk Kalman filter (objects on a table or counter are
    static unless something moves them), latest 6D rotation from ICP, ID, hit / miss counts."""
    _next = 0

    def __init__(self, det: Detection, t: float):
        Track._next += 1
        self.id = Track._next
        self.label = det.label
        self.x = det.pos.copy()
        self.P = np.eye(3) * 0.01 ** 2
        self.R = det.R
        self.hits, self.misses, self.t = 1, 0, t

    def predict(self, t, q=0.05 ** 2):
        self.P = self.P + np.eye(3) * q * max(t - self.t, 0.0)       # (m^2/s) random walk
        self.t = t

    def update(self, det: Detection, sigma=0.004):
        K = self.P @ np.linalg.inv(self.P + np.eye(3) * sigma ** 2)
        self.x = self.x + K @ (det.pos - self.x)
        self.P = (np.eye(3) - K) @ self.P
        self.R = det.R
        self.hits += 1
        self.misses = 0

    @property
    def pos(self):
        return self.x

    @property
    def confirmed(self):
        return self.hits >= 2


class Tracker:
    def __init__(self, gate=0.15, max_misses=10):
        self.tracks: list[Track] = []
        self.gate, self.max_misses = gate, max_misses

    def step(self, dets: list[Detection], t: float):
        for tr in self.tracks:
            tr.predict(t)
        free = list(range(len(dets)))
        for tr in sorted(self.tracks, key=lambda x: -x.hits):     # greedy nearest-neighbour association
            cand = [(np.linalg.norm(dets[i].pos - tr.pos), i) for i in free if dets[i].label == tr.label]
            cand = [c for c in cand if c[0] < self.gate]
            if cand:
                _, i = min(cand)
                tr.update(dets[i])
                free.remove(i)
            else:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        self.tracks += [Track(dets[i], t) for i in free]
        return [tr for tr in self.tracks if tr.confirmed]


# ----------------------------------------------------------------------------- perception
class Perception:
    def __init__(self, env, cam="head", width=320, height=240, library=None):
        self.env = env
        self.m = env.m
        self.cam = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        self.W, self.H = width, height
        self.r = mujoco.Renderer(env.m, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1                          # everything a camera sees (people, objects, robot)
        self.f = height / 2 / np.tan(np.deg2rad(env.m.cam_fovy[self.cam]) / 2)
        self.lib = library or load_library()
        # robot self-filter (as robot_self_filter in ROS): points inside the arms' / hands' geoms (oriented bounding
        # boxes + 1 cm) at the current joint state are removed (robot model + kinematics only)
        m = env.m
        self.self_geoms = np.array([g for g in range(m.ngeom)
                                    if m.geom_bodyid[g] in env.ctrl.arm_bodies_all and m.geom_rbound[g] < 0.15])
        self.tracker = Tracker()
        self.rng = np.random.default_rng(0)
        self.depth_noise = 0.002                           # m, sensor noise (sigma, grows with distance^2)
        self.max_range = 2.0                               # m, objects are recognised up to this depth
        self.last_rgb = None
        self.last_dets: list[Detection] = []
        self.last_clusters = []
        self.stats = {"frames": 0, "detections": 0}

    # -------------------------------------------------------------- sensing
    def _render(self):
        e = self.env
        self.r.update_scene(e.d, self.cam, self.opt)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(e.d, self.cam, self.opt)
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        depth = depth + self.rng.normal(0, 1, depth.shape) * self.depth_noise * np.maximum(depth, 0.3) ** 2
        return rgb, depth

    def _cloud(self, depth, stride=1):
        """World points (N,3) + their pixel coordinates, from the camera pose given by the robot kinematics."""
        v, u = np.mgrid[0:self.H:stride, 0:self.W:stride]
        d = depth[::stride, ::stride]
        ok = (d > 0.15) & (d < self.max_range)
        u, v, d = u[ok], v[ok], d[ok]
        pc = np.stack([(u - self.W / 2 + 0.5) / self.f * d, -(v - self.H / 2 + 0.5) / self.f * d, -d], 1)
        R = self.env.d.cam_xmat[self.cam].reshape(3, 3)
        return pc @ R.T + self.env.d.cam_xpos[self.cam], np.stack([u, v], 1)

    def _self_filter(self, pts, pix, pad=0.01):
        if not len(self.self_geoms) or not len(pts):
            return pts, pix
        d = self.env.d
        tree = cKDTree(pts)
        drop = np.zeros(len(pts), bool)
        for g in self.self_geoms:
            idx = tree.query_ball_point(d.geom_xpos[g], self.m.geom_rbound[g] + pad)
            if not idx:
                continue
            idx = np.asarray(idx)
            # tight test: inside the geom's oriented bounding box (+ pad), in the geom frame
            local = (pts[idx] - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)
            c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            drop[idx[np.all(np.abs(local - c) <= h + pad, axis=1)]] = True
        return pts[~drop], pix[~drop]

    @staticmethod
    def _support_plane(p, rng, iters=150, tol=0.008):
        """RANSAC for the dominant horizontal plane (table / counter top) between 0.4 and 1.2 m."""
        cand = p[(p[:, 2] > 0.4) & (p[:, 2] < 1.2)]
        if len(cand) < 50:
            return None
        best, best_n = None, 0
        for _ in range(iters):
            s = cand[rng.choice(len(cand), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n = n / np.linalg.norm(n)
            if abs(n[2]) < 0.97:                          # horizontal planes only
                continue
            k = int(np.sum(np.abs((cand - s[0]) @ n) < tol))
            if k > best_n:
                best, best_n = (n * np.sign(n[2]), float(s[0] @ n * np.sign(n[2]))), k
        if best is None:
            return None
        n, d0 = best
        inl = cand[np.abs(cand @ n - d0) < tol]
        return float(np.median(inl[:, 2])), inl          # height, inlier points (extent of the support)

    def detect(self):
        rgb, depth = self._render()
        self.last_rgb = rgb
        self.last_clusters = []                       # (points, height above support, extent): diagnostics
        pts, pix = self._cloud(depth)
        pts, pix = self._self_filter(pts, pix)
        dets = []
        # every horizontal support in view (table, counter, chair seats...): the largest one is not necessarily
        # the one the object stands on
        sub = pts[::4]
        for _ in range(3):
            plane = self._support_plane(sub, self.rng)
            if plane is None or len(plane[1]) < 150:
                break
            dets += self._objects_on(pts, pix, *plane)
            sub = sub[np.abs(sub[:, 2] - plane[0]) > 0.02]
        self.last_dets = dets
        self.last_t = self.env.t
        self.stats["frames"] += 1
        self.stats["detections"] += len(dets)
        return dets

    def _objects_on(self, pts, pix, z0, inl):
        """Objects resting on the support plane at height z0 (inliers `inl` give its extent)."""
        dets = []
        lo, hi = inl[:, :2].min(0) - 0.02, inl[:, :2].max(0) + 0.02
        sel = ((pts[:, 2] > z0 + 0.008) & (pts[:, 2] < z0 + 0.30) & np.all(pts[:, :2] > lo, 1)
               & np.all(pts[:, :2] < hi, 1))
        obj, opix = pts[sel], pix[sel]
        if len(obj) <= 20:
            return dets
        cell = 0.01
        ij = np.floor((obj[:, :2] - lo) / cell).astype(int)
        grid = np.zeros(ij.max(0) + 2, bool)
        grid[ij[:, 0], ij[:, 1]] = True
        lab, n = cc_label(grid, structure=np.ones((3, 3)))
        pl = lab[ij[:, 0], ij[:, 1]]
        below = pts[(pts[:, 2] < z0 - 0.03) & (pts[:, 2] > z0 - 0.5)]
        for k in range(1, n + 1):
            c = obj[pl == k]
            if len(c) < 40:
                continue
            # resting on the support: right below the object there is no surface lower than the plane
            # (a chair corner next to the table continues downwards)
            fp = cKDTree(c[:, :2])
            if len(below) and np.sum(fp.query(below[:, :2], distance_upper_bound=0.01)[0] < 0.01) > 15:
                continue
            det = self._recognise(c, z0)
            if det is not None:
                uv = opix[pl == k]
                det.bbox = (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max()))
                dets.append(det)
        return dets

    def _recognise(self, c, z0, max_rmse=0.004):
        c = _voxel_down(c, 0.004)
        top = c[:, 2].max() - z0
        ext = np.ptp(c[:, :2], 0).max()
        cam = self.env.d.cam_xpos[self.cam]
        best = None
        self.last_clusters.append((len(c), round(float(top), 3), round(float(ext), 3)))
        for name, mdl in self.lib.items():
            if abs(top - mdl.h) > 0.015 or ext > 2.5 * mdl.r + 0.06:       # height of the object above the support
                continue
            # upright on the support, axis behind the visible surface (seen from the camera)
            view = c[:, :2].mean(0) - cam[:2]
            view = view / (np.linalg.norm(view) + 1e-9)
            t0 = np.r_[c[:, :2].mean(0) + view * mdl.r * 0.5, z0 + mdl.h / 2]
            for yaw in np.linspace(-np.pi, np.pi, 4, endpoint=False):
                R, t, e = icp(c, mdl, _rotz(yaw), t0, iters=6)       # coarse: a few iterations per hypothesis
                if e > 4 * max_rmse:
                    break                                            # far off for this model: next model
                R, t, e = icp(c, mdl, R, t, iters=20)               # refine
                if best is None or e < best[3]:
                    best = (name, R, t, e)
                if e < 0.6 * max_rmse:
                    break                                            # good fit (symmetric objects: any yaw)
        if best is None or best[3] > max_rmse:
            return None
        name, R, t, e = best
        return Detection(name, t, R, e, len(c), (0, 0, 0, 0))

    # -------------------------------------------------------------- receptacles
    def detect_basin(self, cell=0.02, min_side=0.12, max_side=0.8):
        """Sink basins in view: holes in a horizontal support plane with points below it."""
        _, depth = self._render()
        pts, pix = self._cloud(depth)
        pts, _ = self._self_filter(pts, pix)
        sub = pts[::3]
        found = []
        for _ in range(3):
            plane = self._support_plane(sub, self.rng)
            if plane is None or len(plane[1]) < 150:
                break
            z0, inl_sub = plane
            inl = pts[np.abs(pts[:, 2] - z0) < 0.008]
            found += self._basins_in(pts, z0, inl, cell, min_side, max_side)
            sub = sub[np.abs(sub[:, 2] - z0) > 0.02]
        self.last_basins = found
        return found

    @staticmethod
    def _basins_in(pts, z0, inl, cell, min_side, max_side):
        lo = inl[:, :2].min(0) - 2 * cell
        ij = np.floor((inl[:, :2] - lo) / cell).astype(int)
        mask = np.zeros(ij.max(0) + 3, bool)
        mask[ij[:, 0], ij[:, 1]] = True
        mask = binary_closing(mask, np.ones((3, 3)))            # sparse sampling far away
        holes = binary_fill_holes(mask) & ~mask                  # enclosed by the plane on every side
        lab, n = cc_label(holes)
        below = pts[(pts[:, 2] < z0 - 0.03) & (pts[:, 2] > z0 - 0.5)]
        bij = np.floor((below[:, :2] - lo) / cell).astype(int)
        okb = np.all((bij >= 0) & (bij < np.array(mask.shape)), 1)
        out = []
        for k in range(1, n + 1):
            cells = np.argwhere(lab == k)
            # surface below the plane inside the hole or right at its rim (from far away only the far inner
            # wall of a basin is visible, and it lies on the rim line)
            near_hole = binary_dilation(lab == k, np.ones((3, 3)))
            inside = okb.copy()
            inside[okb] = near_hole[bij[okb, 0], bij[okb, 1]]
            if len(cells) < (min_side / cell) ** 2 or inside.sum() < 30:
                continue                                         # too small, or nothing below: an occlusion
            xy = lo + (cells + 0.5) * cell
            c0 = xy.mean(0)
            w, V = np.linalg.eigh(np.cov((xy - c0).T))
            u = V[:, 1]                                          # long side
            yaw = float(np.arctan2(u[1], u[0]))
            R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            huv = (xy - c0) @ R.T
            lo_uv, hi_uv = huv.min(0) - cell / 2, huv.max(0) + cell / 2
            # refine every edge on the rim: the nearest counter-top point beyond the hole, within its span
            iuv = (inl[:, :2] - c0) @ R.T
            for a in (0, 1):
                b = 1 - a
                span = (iuv[:, b] > lo_uv[b] + cell) & (iuv[:, b] < hi_uv[b] - cell)
                lo_side = iuv[span & (iuv[:, a] < lo_uv[a] + 1.5 * cell) & (iuv[:, a] > lo_uv[a] - 3 * cell), a]
                hi_side = iuv[span & (iuv[:, a] > hi_uv[a] - 1.5 * cell) & (iuv[:, a] < hi_uv[a] + 3 * cell), a]
                if len(lo_side) > 5:
                    lo_uv[a] = lo_side.max()
                if len(hi_side) > 5:
                    hi_uv[a] = hi_side.min()
            half = (hi_uv - lo_uv) / 2
            if np.any(2 * half < min_side) or np.any(2 * half > max_side):
                continue
            center = c0 + R.T @ ((hi_uv + lo_uv) / 2)
            floor = below[inside, 2]
            out.append(Basin(np.r_[center, z0], half, yaw, float(z0 - np.percentile(floor, 5)), int(inside.sum())))
        return out

    def locate_basin(self, frames=3, near=None):
        """Sink basin estimate from `frames` views (median of the per-frame rim fits); None if not seen."""
        seen = []
        for _ in range(frames):
            bs = self.detect_basin()
            if bs:
                ref = near if near is not None else self.env.pose[:2]
                seen.append(min(bs, key=lambda b: np.linalg.norm(b.center[:2] - np.asarray(ref)[:2])))
        if not seen:
            return None
        k = int(np.argsort([b.center[0] for b in seen])[len(seen) // 2])
        best = seen[k]
        return Basin(np.median([b.center for b in seen], 0), np.median([b.half for b in seen], 0), best.yaw,
                     float(np.median([b.depth for b in seen])), best.n_below)

    # -------------------------------------------------------------- API
    def reset(self):
        self.tracker = Tracker()
        self.last_dets = []

    def update(self):
        """One perception frame: detect + recognise + 6D pose, then track. Returns the confirmed tracks."""
        return self.tracker.step(self.detect(), self.env.t)

    def locate(self, label=None, frames=3, near=None, max_misses=0):
        """Object pose estimate (position, rotation) of the best confirmed track of `label` (default: the
        manipulation target), after `frames` perception updates; None if it is not seen."""
        label = label or self.env.cup_type
        tracks = []
        for _ in range(frames):
            tracks = [t for t in self.update() if t.label == label and t.misses <= max_misses]
        if not tracks:
            return None
        if near is not None:
            tr = min(tracks, key=lambda t: np.linalg.norm(t.pos[:2] - np.asarray(near)[:2]))
        else:
            tr = max(tracks, key=lambda t: t.hits)
        return tr.pos.copy(), tr.R.copy()
