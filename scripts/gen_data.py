"""Generate manipulation demos with the scripted expert, in LeRobot v2.1 format.

python scripts/gen_data.py --pick 200 --place 150 --out data/homebot_can
Only successful demos are kept (failures are logged).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_manip import perturbed, put_cup_in_hand  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.expert import Expert  # noqa: E402
from homebot.policy import TASKS  # noqa: E402
from homebot.record import Recorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, default=200)
    ap.add_argument("--place", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "data" / "homebot_can"))
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--seed0", type=int, default=200_000)   # disjoint from the E2 evaluation seeds (0..N)
    ap.add_argument("--workers", type=int, default=1, help="parallel generator processes (merged at the end)")
    a = ap.parse_args()
    if a.workers > 1:
        return parallel(a)
    rec = Recorder(a.out, images=not a.no_images)
    fails = 0
    t0 = time.time()
    for skill, n in (("pick_cup", a.pick), ("place_cup", a.place)):
        made, seed = 0, a.seed0 + (0 if skill == "pick_cup" else 100_000)
        while made < n:
            seed += 1
            rng = np.random.default_rng(seed)
            env = HomeEnv(level=1, seed=seed)
            start = perturbed("table" if skill == "pick_cup" else "sink", rng)
            if skill == "place_cup":
                start[:2] -= 0.12 * np.array([np.cos(start[2]), np.sin(start[2])])
            env.reset(start, arm_ready=True)
            if skill == "place_cup" and not put_cup_in_hand(env, rng, start):
                env.close()
                continue
            ex = Expert(env, recorder=rec, rng=rng)
            rec.start(env, TASKS[skill])
            ok = getattr(ex, skill)()
            if ok:
                rec.end(True)
                made += 1
            else:
                rec._reset_ep()
                fails += 1
            env.close()
            if made and made % 25 == 0 and ok:
                print(f"{skill}: {made}/{n}  ({time.time() - t0:.0f}s, expert failures so far {fails})", flush=True)
    rec.finalize()
    print(f"done: {len(rec.episodes)} episodes, {rec.total} frames, expert failures {fails}, "
          f"{time.time() - t0:.0f}s -> {a.out}")


def parallel(a):
    """Split the demos over `a.workers` processes (disjoint seeds), then merge their datasets."""
    import shutil
    import subprocess
    from homebot.record import merge_parts
    out = Path(a.out)
    parts = [out.parent / f"_{out.name}_part{i}" for i in range(a.workers)]
    split = lambda n, i: n // a.workers + (1 if i < n % a.workers else 0)
    procs = []
    for i, part in enumerate(parts):
        shutil.rmtree(part, ignore_errors=True)
        cmd = [sys.executable, __file__, "--pick", str(split(a.pick, i)), "--place", str(split(a.place, i)),
               "--out", str(part), "--seed0", str(a.seed0 + 10_000 * i)] + (["--no-images"] if a.no_images else [])
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"generator failed: {codes}")
    shutil.rmtree(out, ignore_errors=True)
    n_ep, n_fr = merge_parts(parts, out)
    for part in parts:
        shutil.rmtree(part, ignore_errors=True)
    print(f"merged: {n_ep} episodes, {n_fr} frames -> {out}")


if __name__ == "__main__":
    main()
