"""E3: full task "bring the can to the sink" with a person walking around (level 2).

python scripts/eval_full.py --n 30 --level 2 --policy expert
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot import apartment as apt  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.nav.astar import inflate  # noqa: E402
from homebot.tasks import TaskRunner  # noqa: E402

COMMAND = "bring the can to the sink"


def living_room_start(env, rng):
    occ = inflate(apt.rasterize(env.items), 0.45)
    while True:
        p = np.array([rng.uniform(0.6, 4.0), rng.uniform(0.6, 5.4)])
        if not apt.occupied(occ, p):
            return np.array([*p, rng.uniform(-np.pi, np.pi)])


def run_one(args):
    seed, level, policy, host, sink_range = args
    rng = np.random.default_rng(70_000 + seed)
    sink_off = float(np.random.default_rng(90_000 + seed).uniform(-sink_range, sink_range)) if sink_range else 0.0
    env = HomeEnv(level=level, seed=1000 + seed, sink_offset=sink_off)
    start = living_room_start(env, rng)
    env.reset(start, robot_route=(start[:2], np.array(apt.PLACES["table"]["pos"])))
    pol = None
    if policy != "expert":
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_manip import make_policy
        pol = make_policy(policy, env, rng, host)
    r = TaskRunner(env, policy=pol).run(COMMAND)
    env.close()
    return {"seed": seed, "level": level, "policy": policy, "success": r["success"],
            "failed_at": r["failed_at"] or "", "time_s": round(r["time"], 1),
            "person_contacts": r["person_contacts"], "person_contacts_robot_moving": r["person_contacts_moving"],
            "pet_contacts": r["pet_contacts"], "furniture_contacts": r["furniture_contacts"],
            "min_person_dist": round(float(r["min_person_dist"]), 3), "fell": r["fell"],
            "arm_contacts": r["arm_contacts"], "arm_stops": r["arm_stops"],
            "left_arm_contacts": r["left_arm_contacts"], "sink_offset": round(sink_off, 3),
            "basin_err_mm": r.get("basin_err_mm", float("nan")), "trace": json.dumps(r["trace"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--workers", type=int, default=3)   # ~600 MB per G1 + Inspire environment
    ap.add_argument("--sink-range", type=float, default=0.0,
                    help="move the sink along the counter by up to +-this (m) per run (generalisation test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(s, a.level, a.policy, a.host, a.sink_range) for s in range(a.n)]
    rows = []
    with Pool(a.workers if a.policy == "expert" else 1, maxtasksperchild=3) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.append(r)
            print(f"{i + 1}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else ROOT / "results" / f"e3_full_level{a.level}_{a.policy.split(':')[0]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"success {df.success.mean():.2%} ({df.success.sum()}/{len(df)})")
    print("failures by step:\n", df[~df.success].failed_at.value_counts().to_string())
    print("person contacts:", df.person_contacts.sum(), "(robot moving:", df.person_contacts_robot_moving.sum(), ")")
    print("furniture contacts:", df.furniture_contacts.sum(), " falls:", int(df.fell.sum()),
          " arm contacts (right / left):", df.arm_contacts.sum(), "/", df.left_arm_contacts.sum(), " arm stops:", df.arm_stops.sum())


if __name__ == "__main__":
    main()
