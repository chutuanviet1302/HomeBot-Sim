"""Simulate one "bring the can to the sink" episode headless and save the recording like scripts/view.py does
(logs/replay_l{level}_s{seed}[_elderly][_sink+0.xxx][_pol-<policy>].npz), without opening the viewer.

python web/backend/run_episode.py --level 1 --seed 0 [--sink-range 0.15] [--person elderly] [--policy act:act_homebot.pt]
Prints progress lines ("  simulated  12.0 s ...") and finally "SAVED <file>" / "RESULT <text>".
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import view  # noqa: E402  (scripts/view.py)
from eval_full import COMMAND, living_room_start  # noqa: E402
from homebot import apartment as apt  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.tasks import TaskRunner  # noqa: E402


def policy_slug(policy: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", policy).strip("-")


def make_env(a):
    """scripts/view.py make_env plus a manipulation policy (as in scripts/eval_full.py)."""
    rng = np.random.default_rng(70_000 + a.seed)
    env = HomeEnv(level=a.level, seed=1000 + a.seed, sink_offset=a.sink_offset, person_profile=a.person)
    start = living_room_start(env, rng)
    env.reset(start, robot_route=(start[:2], np.array(apt.PLACES["table"]["pos"])))
    pol = None
    if a.policy != "expert":
        from eval_manip import make_policy
        pol = make_policy(a.policy, env, rng)
    return env, TaskRunner(env, policy=pol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sink-range", type=float, default=0.0, help="sink moved by U(-r, r) m (as eval_full.py)")
    ap.add_argument("--person", default="adult", choices=["adult", "elderly"])
    ap.add_argument("--policy", default="expert")
    a = ap.parse_args()
    a.command = COMMAND
    a.sink_offset = (float(np.random.default_rng(90_000 + a.seed).uniform(-a.sink_range, a.sink_range))
                     if a.sink_range else 0.0)
    print(f'START "{a.command}" level {a.level} seed {a.seed} person {a.person} policy {a.policy} '
          f"sink offset {a.sink_offset:+.3f} m", flush=True)
    view.make_env = make_env
    env, frames, res = view.record(a)
    orig = view.cache_path
    path = orig(a)
    if a.policy != "expert":
        path = path.with_name(f"{path.stem}_pol-{policy_slug(a.policy)}.npz")
    view.cache_path = lambda _a: path
    view.save_cache(a, frames, res)
    view.cache_path = orig
    print(f"SAVED {path.name}", flush=True)
    print(f"RESULT {res}", flush=True)


if __name__ == "__main__":
    main()
