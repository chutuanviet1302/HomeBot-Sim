"""E2: pick / place success, robot start perturbed +-10 cm / +-15 deg from the ALIGN pose.

python scripts/eval_manip.py --n 50 --policy expert
python scripts/eval_manip.py --n 50 --policy gr00t --host <server>      (Kaggle / GPU box)
python scripts/eval_manip.py --n 50 --policy torch:checkpoints/act.pt
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot import apartment as apt  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.expert import CARRY, Expert  # noqa: E402

POS_NOISE, YAW_NOISE = 0.10, np.deg2rad(15)


def perturbed(place, rng):
    p = apt.PLACES[place]
    return np.array([*p["pos"], p["yaw"]]) + [*rng.uniform(-POS_NOISE, POS_NOISE, 2),
                                              rng.uniform(-YAW_NOISE, YAW_NOISE)]


def make_policy(name, env, rng, host="localhost"):
    if name == "expert":
        return Expert(env, rng=rng)
    from homebot.policy import LearnedSkillPolicy, gr00t_client, torch_policy
    if name == "gr00t":
        return LearnedSkillPolicy(env, gr00t_client(host))
    if name.startswith("torch:"):
        return LearnedSkillPolicy(env, torch_policy(name.split(":", 1)[1]))
    raise ValueError(name)


def run_one(args):
    skill, seed, policy, host, sink_range = args
    rng = np.random.default_rng(50_000 + seed)
    # generalisation test: the sink moved along the counter (the robot still starts at the usual spot and has to
    # find the basin with its camera)
    sink_off = float(np.random.default_rng(90_000 + seed).uniform(-sink_range, sink_range)) if sink_range else 0.0
    env = HomeEnv(level=1, seed=seed, sink_offset=sink_off)
    start = perturbed("table" if skill == "pick" else "sink", rng)
    if skill == "place":
        # the carried object is ~25 cm ahead of the pelvis: start 12 cm further back so that a +10 cm
        # perturbation does not put it inside the counter
        start[:2] -= 0.12 * np.array([np.cos(start[2]), np.sin(start[2])])
    env.reset(start, arm_ready=True)        # trial starts at the table / counter, hand ready
    pol = make_policy(policy, env, rng, host)
    if skill == "place":
        # start from the holding pose: cup placed between the fingers, fingers closed
        held = put_cup_in_hand(env, rng, start)
        ok = bool(held and pol.place_cup())
    else:
        ok = pol.pick_cup()
    st = env.stats
    cup_z = float(env.cup_pose()[0][2])
    env.close()
    return {"skill": skill, "seed": seed, "policy": policy, "success": bool(ok), "cup": env.cup_type,
            "arm_contacts": st["arm_contacts"], "arm_stops": st["arm_stops"], "fell": st["fell"],
            "arm_paths_rejected": st["arm_paths_rejected"], "arm_paths_blocked": st.get("arm_paths_blocked", 0),
            "perception_err_mm": st.get("perc_err_mm", float("nan")),
            "align_tries": st.get("align_tries", 0), "not_seen": st.get("not_seen", 0),
            "unreachable": st.get("unreachable", 0), "not_seen_info": st.get("not_seen_info", ""),
            "cup_dropped": bool(cup_z < 0.5), "sink_offset": round(sink_off, 3),
            "basin_err_mm": st.get("basin_err_mm", float("nan")), "sink_not_seen": st.get("sink_not_seen", 0),
            "dx": round(start[0] - apt.PLACES["table" if skill == "pick" else "sink"]["pos"][0], 3),
            "dyaw_deg": round(float(np.rad2deg(start[2] - apt.PLACES["table" if skill == "pick" else "sink"]["yaw"])), 1)}


def put_cup_in_hand(env, rng=None, start=None):
    """Initial state "holding the cup" for the isolated place test: a real (contact) pick at the table, then the
    robot is moved to `start` together with the cup, keeping the cup's pose in the hand and the arm/finger joint
    state (test setup only - the full task E3 walks there instead)."""
    import mujoco
    from homebot import expert as X
    from homebot.env import mat_to_quat, quat_to_mat, yaw_quat
    a = apt.PLACES["table"]
    for attempt in range(3):
        env.reset([*a["pos"], a["yaw"]], arm_ready=True)
        if X.Expert(env, rng=rng).pick_cup():
            break
    else:
        return False
    if start is None:
        return env.grasped
    hp, hq = env.hand_world()
    Rh = quat_to_mat(hq)
    cp, cq = env.cup_pose()
    rel_p, rel_R = Rh.T @ (cp - hp), Rh.T @ quat_to_mat(cq)
    keep = env.d.qpos.copy()
    env.walker.place(*start)
    arm = [env.m.jnt_qposadr[j] for j in env.ctrl.ik_joints + env.ctrl.l_arm_j]
    hand = [env.m.jnt_qposadr[j] for j in range(env.m.njnt) if env.m.joint(j).name.startswith(("R_", "L_"))]
    env.d.qpos[arm + hand] = keep[arm + hand]
    env.d.qvel[:] = 0
    mujoco.mj_forward(env.m, env.d)
    hp, hq = env.hand_world()
    Rh = quat_to_mat(hq)
    env.set_cup(hp + Rh @ rel_p, mat_to_quat(Rh @ rel_R))
    env.pose = env._true_pose()
    env.hand_goal_world = None
    env.stand_mode = False
    env.wrist_pos = X.CARRY_HIGH.copy()
    for _ in range(25):
        env.step(np.zeros(3), carrying=True)
    return env.grasped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--workers", type=int, default=3)   # ~600 MB per G1 + Inspire environment
    ap.add_argument("--out", default=None)
    ap.add_argument("--skills", nargs="+", default=["pick", "place"])
    ap.add_argument("--sink-range", type=float, default=0.0,
                    help="move the sink along the counter by up to +-this (m) per trial (generalisation test)")
    a = ap.parse_args()
    jobs = [(sk, s, a.policy, a.host, a.sink_range) for sk in a.skills for s in range(a.n)]
    workers = min(a.workers, 3) if a.policy == "expert" else 1     # ~600 MB per G1 + Inspire env
    rows = []
    # fresh worker processes every few trials: each trial builds a G1 model, a walking policy and renderers
    with Pool(workers, maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["skill"] != "pick", r["seed"]))
    df = pd.DataFrame(rows)
    tag = a.policy.replace(":", "_").replace("/", "_").replace("\\", "_")
    out = a.out or str(ROOT / "results" / f"e2_manip_{tag}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby(["policy", "skill"]).agg(n=("success", "size"), success=("success", "mean"),
                                              arm_contacts=("arm_contacts", "sum"), arm_stops=("arm_stops", "sum"),
                                              dropped=("cup_dropped", "sum"), falls=("fell", "sum"),
                                              perc_err_mm=("perception_err_mm", "median")).to_string())


if __name__ == "__main__":
    main()
