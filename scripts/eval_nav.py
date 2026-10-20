"""E1 / E4: random A -> B navigation at 3 obstacle levels (E4 = level 2 with the safety layer off).

python scripts/eval_nav.py --n 100 --levels 1 2 3 --e4 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot import apartment as apt  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.nav.astar import inflate  # noqa: E402
from homebot.nav.navigator import Navigator  # noqa: E402


def sample_pair(env, rng, min_dist=3.0):
    raw = apt.rasterize(env.items)
    stand = inflate(raw, apt.ROBOT_RADIUS + 0.13)          # start / goal: room to stand and turn
    from scipy.ndimage import label
    lab, _ = label(~inflate(raw, apt.ROBOT_RADIUS + apt.CLEARANCE + apt.PLAN_MARGIN))  # what the robot fits through
    while True:
        a = np.array([rng.uniform(0.4, 7.6), rng.uniform(0.4, 5.6)])
        b = np.array([rng.uniform(0.4, 7.6), rng.uniform(0.4, 5.6)])
        if apt.occupied(stand, a) or apt.occupied(stand, b):
            continue
        la, lb = lab[apt.world_to_cell(a)], lab[apt.world_to_cell(b)]
        if la and la == lb and np.linalg.norm(a - b) >= min_dist:
            return a, b


def run_one(args):
    level, seed, safety, timeout = args
    env = HomeEnv(level=level, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    a, b = sample_pair(env, rng)
    env.reset([*a, rng.uniform(-np.pi, np.pi)], robot_route=(a, b))
    nav = Navigator(env, safety=safety)
    t0 = time.time()
    path_len, prev = 0.0, env.pose[:2].copy()
    nav.set_goal(b, 0.15)
    ok, fell = False, False
    while env.t < timeout:
        if nav.step():
            ok = True
            break
        path_len += np.linalg.norm(env.pose[:2] - prev)
        prev = env.pose[:2].copy()
        if env.fallen():
            fell = True
            break
    st = {k: v for k, v in env.stats.items() if not k.startswith("_")}
    return {"level": level, "seed": seed, "safety": safety, "success": ok, "time_s": round(env.t, 2),
            "path_m": round(path_len, 2), "straight_m": round(float(np.linalg.norm(b - a)), 2), "fell": fell,
            **{k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in st.items()},
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--e4", action="store_true", help="also run level 2 with the safety layer off")
    ap.add_argument("--workers", type=int, default=3)   # ~600 MB per G1 + Inspire environment
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_e4_nav.csv"))
    a = ap.parse_args()
    jobs = [(lvl, s, True, a.timeout) for lvl in a.levels for s in range(a.n)]
    if a.e4:
        jobs += [(2, s, False, a.timeout) for s in range(a.n)]
    rows = []
    with Pool(a.workers, maxtasksperchild=10) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["safety", "level", "seed"], ascending=[False, True, True])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(summarize(df).to_string())


def summarize(df):
    df = df.assign(time_ok=df.time_s.where(df.success))
    g = df.groupby(["safety", "level"])
    return pd.DataFrame({
        "runs": g.size(),
        "success_rate": g.success.mean().round(3),
        "person_contacts": g.person_contacts.sum(),
        "person_contacts_robot_moving": g.person_contacts_moving.sum(),
        "pet_contacts": g.pet_contacts.sum(),
        "pet_contacts_robot_moving": g.pet_contacts_moving.sum(),
        "furniture_contacts": g.furniture_contacts.sum(),
        "min_person_dist_m": g.min_person_dist.min().round(3),
        "median_min_person_dist_m": g.min_person_dist.median().round(3),
        "mean_time_s(success)": g.time_ok.mean().round(1),
        "falls": g.fell.sum(),
    })


if __name__ == "__main__":
    main()
