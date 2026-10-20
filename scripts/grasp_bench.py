"""Fast grasp-only bench (like HomeHand's top_grasp_search): the robot is put directly at the stance ALIGN
reaches (+-3 cm, +-8 deg residual error), the base is locked at once and only the grasp + lift runs (~6 s of
simulation instead of ~35 s for a full E2 trial). Used to tune / rank grasp parameters, then E2 confirms.

python scripts/grasp_bench.py --n 6                       # current parameters
python scripts/grasp_bench.py --grid --n 3                # parameter grid, ranked
"""
from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot import apartment as apt  # noqa: E402
from homebot import expert as X  # noqa: E402
from homebot import g1 as g1mod  # noqa: E402
from homebot.env import HomeEnv, quat_to_mat  # noqa: E402


def stance_for(obj_xy, rel, rng):
    """Base pose that sees the object at `rel` in the pelvis frame (facing it), plus a residual error."""
    rel = np.asarray(rel, float)
    d0 = np.asarray(obj_xy) - (np.asarray(apt.PLACES["table"]["pos"]))
    yaw = np.arctan2(d0[1], d0[0]) - np.arctan2(rel[1], rel[0])
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.asarray(obj_xy) - np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    return np.array([*(base + rng.uniform(-0.03, 0.03, 2)), yaw + rng.uniform(-0.14, 0.14)])


def trial(args):
    params, cup, seed, *full = args
    dx, gap, dz, thumb, *ty = params
    if ty:
        X.THUMB_YAW_APPROACH = ty[0]
    X.OBJ_IN_PALM = {k: np.array([dx, v["r"] + gap, dz]) for k, v in apt.CUP_TYPES.items()}
    g1mod.HAND_PRESHAPE = np.array([thumb, 0, 0, 0, 0, 0])
    g1mod.HAND_CLOSED = np.array([thumb, 0.6, 1.47, 1.47, 1.47, 1.47])
    if full and full[0]:
        # the real E2 pick: perturbed start (+-10 cm, +-15 deg), ALIGN by walking, stand + lock, grasp, carry
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_manip import perturbed
        rng = np.random.default_rng(50_000 + seed)
        env = HomeEnv(level=1, seed=seed, cup_type=cup, relaxed_arms=False)
        env.reset(perturbed("table", rng))
        ok = X.Expert(env, rng=rng).pick_cup()
        for _ in range(50):
            env.step(np.zeros(3), carrying=True)
        held = bool(env.grasped and env.cup_pose()[0][2] > apt.TABLE_TOP + env.cup_h / 2 + 0.05)
        tilt = float(np.degrees(np.arccos(np.clip(quat_to_mat(env.cup_pose()[1])[2, 2], -1, 1))))
        return {"dx": dx, "gap": gap, "dz": dz, "thumb": thumb, "thumb_app": X.THUMB_YAW_APPROACH, "cup": cup,
                "seed": seed, "success": bool(ok and held), "tilt_deg": round(tilt, 1),
                "arm_contacts": env.stats["arm_contacts"], "arm_stops": env.stats["arm_stops"],
                "fell": env.stats["fell"]}
    rng = np.random.default_rng(90_000 + seed)
    env = HomeEnv(level=1, seed=seed, cup_type=cup, relaxed_arms=False)
    cup_xy = apt.CUP_HOME + rng.uniform(-0.10, 0.10, 2) * np.array([1.0, 0.3])
    env.reset(stance_for(cup_xy, X.OBJ_IN_BASE, rng), cup_xy=cup_xy)
    env.stand_mode = True
    env.walker.lock()                       # straight to the locked stance
    env.hand_goal_world = env.robot_to_world(env.wrist_pos, env.wrist_quat)
    for _ in range(10):
        env.step(np.zeros(3))
    ok = X.Expert(env, rng=rng).grasp_object(carry=False)
    # hold test: 1 s in the air
    for _ in range(50):
        env.step(np.zeros(3))
    held = bool(env.grasped and env.cup_pose()[0][2] > apt.TABLE_TOP + env.cup_h / 2 + 0.05)
    return {"dx": dx, "gap": gap, "dz": dz, "thumb": thumb, "cup": cup, "seed": seed, "success": bool(ok and held),
            "arm_contacts": env.stats["arm_contacts"], "arm_stops": env.stats["arm_stops"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="trials per object and candidate")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="real E2 pick (walk + align + lock) instead of the stance")
    ap.add_argument("--dz", type=float, nargs="+", default=None, help="grid over GRASP_DZ only")
    ap.add_argument("--dx", type=float, nargs="+", default=None)
    ap.add_argument("--gap", type=float, nargs="+", default=None)
    ap.add_argument("--thumb-app", type=float, nargs="+", default=None)
    a = ap.parse_args()
    cur = (X.GRASP_DX, X.GRASP_GAP, X.GRASP_DZ, float(g1mod.HAND_CLOSED[0]))
    cands = list(itertools.product((-0.03, -0.02), (0.015, 0.02, 0.025), (-0.025, -0.02, -0.015),
                                   (1.3,))) if a.grid else [cur]
    if a.dz or a.dx or a.gap or a.thumb_app:
        cands = list(itertools.product(a.dx or [cur[0]], a.gap or [cur[1]], a.dz or [cur[2]], [cur[3]],
                                       a.thumb_app or [X.THUMB_YAW_APPROACH]))
    jobs = [(c, cup, s, a.full) for c in cands for cup in apt.CUP_TYPES for s in range(a.n)]
    with Pool(a.workers) as p:
        rows = p.map(trial, jobs)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / (("grasp_bench_full" if a.full else "grasp_bench") + ("_grid" if len(cands) > 1 else "") + ".csv")
    df.to_csv(out, index=False)
    keys = [k for k in ("dx", "gap", "dz", "thumb", "thumb_app") if k in df]
    agg = dict(success=("success", "mean"), n=("success", "size"), arm_contacts=("arm_contacts", "sum"))
    if "tilt_deg" in df:
        agg["tilt_deg"] = ("tilt_deg", "median")
    g = df.groupby(keys).agg(**agg)
    print(g.sort_values("success", ascending=False).head(12).to_string())
    print(df.groupby("cup").success.mean().to_string())


if __name__ == "__main__":
    main()
