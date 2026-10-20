"""Global planner: 8-connected A* on the inflated occupancy grid."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation

from ..apartment import GRID_RES, cell_to_world, world_to_cell


def inflate(occ, radius):
    k = int(np.ceil(radius / GRID_RES))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (xx ** 2 + yy ** 2) * GRID_RES ** 2 <= radius ** 2 + 1e-9
    return binary_dilation(occ, structure=disk)


def _nearest_free(occ, rc, max_r=20):
    r0, c0 = rc
    H, W = occ.shape
    for rad in range(0, max_r):
        best = None
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad) if abs(dr) != rad else range(-rad, rad + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < H and 0 <= c < W and not occ[r, c]:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, (r, c))
        if best:
            return best[1]
    return None


def astar(occ, start_xy, goal_xy):
    """Return list of (r, c) cells or None. Start/goal are snapped to the nearest free cell."""
    H, W = occ.shape
    s = tuple(int(v) for v in world_to_cell(np.asarray(start_xy)))
    g = tuple(int(v) for v in world_to_cell(np.asarray(goal_xy)))
    s = s if (0 <= s[0] < H and 0 <= s[1] < W and not occ[s]) else _nearest_free(occ, s)
    g = g if (0 <= g[0] < H and 0 <= g[1] < W and not occ[g]) else _nearest_free(occ, g)
    if s is None or g is None:
        return None
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    gcost = np.full((H, W), np.inf)
    parent = -np.ones((H, W, 2), int)
    closed = np.zeros((H, W), bool)
    gcost[s] = 0.0
    h = lambda a: np.hypot(a[0] - g[0], a[1] - g[1])
    pq = [(h(s), 0.0, s)]
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == g:
            break
        r, c = cur
        for dr, dc, w in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not occ[nr, nc] and not closed[nr, nc]:
                ng = gc + w
                if ng < gcost[nr, nc]:
                    gcost[nr, nc] = ng
                    parent[nr, nc] = (r, c)
                    heapq.heappush(pq, (ng + h((nr, nc)), ng, (nr, nc)))
    if not closed[g]:
        return None
    path = [g]
    while path[-1] != s:
        r, c = parent[path[-1]]
        path.append((int(r), int(c)))
    return path[::-1]


def path_to_waypoints(path, spacing=0.5, goal_xy=None):
    """Cells -> world waypoints spaced ~`spacing` m apart (last = exact goal if given)."""
    pts = cell_to_world(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
    out = [pts[0]]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        acc += np.linalg.norm(b - a)
        if acc >= spacing:
            out.append(b)
            acc = 0.0
    if goal_xy is not None:
        out.append(np.asarray(goal_xy, float))
    elif np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return np.array(out)
