"""Carry bench: real contact pick at the table, then carry the object for a while (walking, turning, stopping)
and measure how it moves in the hand (slip) and whether it is lost.

python scripts/carry_bench.py --n 6
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_manip import perturbed  # noqa: E402
from homebot import expert as X  # noqa: E402
from homebot.env import HomeEnv, quat_to_mat, yaw_quat  # noqa: E402

# (duration s, vx, vy, wz) - backing out, turning, walking, stopping: a carry like the one to the sink
PROGRAM = [(3, -0.15, 0, 0), (4, 0, 0, 0.35), (6, 0.25, 0, 0), (3, 0, 0, 0), (4, 0, 0, -0.35), (6, 0.3, 0, 0),
           (2, 0, 0, 0), (5, 0.2, 0, 0.2), (4, 0, 0, 0), (5, -0.1, 0.08, 0)]


def rel_in_hand(env):
    hp, hq = env.hand_world()
    cp, _ = env.cup_pose()
    return quat_to_mat(hq).T @ (cp - hp)


def steady_hand_target(self):
    """Variant under test: the carry pose follows the gait-averaged base pose (advanced by the averaged velocity
    to remove the filter lag) instead of the swaying pelvis, so the arm absorbs the sway."""
    if self.hand_goal_world is not None:
        return self.hand_goal_world
    x, y, th = self.pose_f
    vw = self._robot_world_vel()
    x, y, th = x + 0.5 * vw[0], y + 0.5 * vw[1], th + 0.5 * self.vel_f[2]
    R = quat_to_mat(yaw_quat(th))
    return np.array([x, y, 0.0]) + R @ self.wrist_pos, R @ quat_to_mat(self.wrist_quat)


def run_one(args):
    seed, closed, dz, steady, noslip = args
    if steady:
        HomeEnv.hand_target_world = steady_hand_target
    if dz is not None:                        # grasp height variant: object centre this far along the hand z axis
        for k in X.OBJ_IN_PALM:
            X.OBJ_IN_PALM[k][2] = dz
    if closed is not None:                    # grip variant under test: [thumb yaw, thumb pitch, 4 fingers]
        from homebot import g1 as g1mod
        g1mod.HAND_CLOSED[:] = closed
    rng = np.random.default_rng(50_000 + seed)
    env = HomeEnv(level=1, seed=seed)
    env.reset(perturbed("table", rng), arm_ready=True)
    if noslip is not None:
        env.m.opt.noslip_iterations = noslip
    ex = X.Expert(env, rng=rng)
    if not ex.pick_cup():
        env.close()
        return {"seed": seed, "dz": dz, "picked": False}
    r0 = rel_in_hand(env)
    lost_t, drift = None, 0.0
    t0 = env.t
    for dur, vx, vy, wz in PROGRAM:
        for _ in range(int(dur / 0.02)):
            env.step(np.array([vx, vy, wz]), carrying=True)
            if not env.grasped and lost_t is None:
                lost_t = env.t - t0
            drift = max(drift, float(np.linalg.norm(rel_in_hand(env) - r0)))
    r1 = rel_in_hand(env)
    env.close()
    return {"seed": seed, "grip": closed, "dz": dz, "steady": steady, "noslip": noslip, "picked": True, "lost_t": lost_t, "max_drift_cm": round(100 * drift, 1),
            "final_shift_cm": np.round(100 * (r1 - r0), 1).tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--thumb", type=float, nargs="*", default=[], help="thumb pitch values to compare")
    ap.add_argument("--fingers", type=float, nargs="*", default=[], help="finger closing values to compare")
    ap.add_argument("--dz", type=float, nargs="*", default=[], help="grasp heights (GRASP_DZ) to compare")
    ap.add_argument("--steady", action="store_true", help="carry pose follows the gait-averaged base pose")
    ap.add_argument("--noslip", type=int, default=None, help="MuJoCo noslip solver iterations")
    a = ap.parse_args()
    from homebot import g1 as g1mod
    grips = [None] + [[g1mod.HAND_CLOSED[0], t, f, f, f, f] for t in (a.thumb or [g1mod.HAND_CLOSED[1]])
                      for f in (a.fingers or [g1mod.HAND_CLOSED[2]])
                      if (t, f) != (g1mod.HAND_CLOSED[1], g1mod.HAND_CLOSED[2])]
    # (worker processes are spawned: the variant travels with the job)
    jobs = ([(s, g, None, a.steady, a.noslip) for g in grips for s in range(a.n)]
            + [(s, None, z, a.steady, a.noslip) for z in a.dz for s in range(a.n)])
    if a.dz:
        jobs = [j for j in jobs if j[2] is not None]
    with Pool(a.workers, maxtasksperchild=2) as pool:
        for r in pool.imap(run_one, jobs):
            print(r, flush=True)
