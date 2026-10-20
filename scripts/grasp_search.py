"""Grasp-only search for the handshake power wrap on this G1 + Inspire model.

For each candidate (cup centre in the palm frame, thumb pre-rotation) the robot stands at the table, the
expert's pick motion runs with the candidate and success = the cup is lifted >= 5 cm and held by contact.
python scripts/grasp_search.py --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot import apartment as apt  # noqa: E402
from homebot import expert as X  # noqa: E402
from homebot import g1 as g1mod  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from eval_manip import perturbed  # noqa: E402


def trial(args):
    (dx, gap, dz, thumb_yaw), cup, seed = args
    dy = apt.CUP_TYPES[cup]["r"] + gap          # palm normal offset = object radius + gap for the open fingers
    X.OBJ_IN_PALM = {k: np.array([dx, dy, dz]) for k in apt.CUP_TYPES}
    g1mod.HAND_PRESHAPE = np.array([thumb_yaw, 0, 0, 0, 0, 0])
    g1mod.HAND_CLOSED = np.array([thumb_yaw, 0.6, 1.47, 1.47, 1.47, 1.47])
    rng = np.random.default_rng(70_000 + seed)
    env = HomeEnv(level=1, seed=seed, cup_type=cup)
    env.reset(perturbed("table", rng), arm_ready=True)              # robust score: perturbed stance + perception noise
    ok = X.Expert(env, rng=rng).pick_cup()
    return {"dx": dx, "gap": gap, "dz": dz, "thumb_yaw": thumb_yaw, "cup": cup, "seed": seed, "success": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    ap_ = a
    cands = list(itertools.product((-0.035, -0.02, -0.005), (0.025, 0.035, 0.045), (-0.03, -0.02, -0.01), (1.3,)))
    jobs = [(c, cup, s) for c in cands for cup in apt.CUP_TYPES for s in range(ap_.seeds)]
    with Pool(a.workers) as p:
        rows = p.map(trial, jobs)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "grasp_search.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    s = df.groupby(["dx", "gap", "dz", "thumb_yaw"]).success.sum().sort_values(ascending=False)
    print(s.head(15).to_string())


if __name__ == "__main__":
    main()
