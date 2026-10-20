"""Interactive MuJoCo viewer: watch the robot do a command (default: the full task with a person walking around).

The episode is simulated first (same physics, perception and planners as the evaluations; ~2-3 min) and the
recorded states are then played back at exact real time, so the view is smooth even though perception and arm
path planning sometimes take longer than one control step. `--live` shows the simulation while it runs instead.

python scripts/view.py                         # "bring the can to the sink", level 2, seed 2
python scripts/view.py --level 3 --seed 7 --speed 0.5
python scripts/view.py --live
Keys (playback): space pause / resume, + / - speed, R restart, F follow the robot on / off.
Mouse: rotate / zoom / pan as in the MuJoCo simulate app. Close the window to stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_full import COMMAND, living_room_start  # noqa: E402
from homebot import apartment as apt  # noqa: E402
from homebot.env import CTRL_DT, HomeEnv  # noqa: E402
from homebot.tasks import TaskRunner  # noqa: E402


def make_env(a):
    rng = np.random.default_rng(70_000 + a.seed)
    env = HomeEnv(level=a.level, seed=1000 + a.seed)
    start = living_room_start(env, rng)
    env.reset(start, robot_route=(start[:2], np.array(apt.PLACES["table"]["pos"])))
    return env, TaskRunner(env)


def setup_cam(viewer):
    viewer.opt.geomgroup[:] = 1
    viewer.cam.lookat[:] = [4.0, 3.0, 0.5]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 6.0, -90, -45


def record(a):
    """Simulate the episode headless; keep qpos / mocap and a status line for every control step."""
    env, runner = make_env(a)
    frames = []
    step = env.step
    t_wall = time.time()

    def step_rec(*args, **kw):
        step(*args, **kw)
        frames.append((env.d.qpos.copy(), env.d.mocap_pos.copy(), env.d.mocap_quat.copy(),
                       f"{runner.state}  safety: {runner.nav.safety.state}"))
        if len(frames) % 500 == 0:
            print(f"  simulated {env.t:5.1f} s ({time.time() - t_wall:4.0f} s wall)", flush=True)

    env.step = step_rec
    r = runner.run(a.command)
    res = "success" if r["success"] else f"failed at {r['failed_at']}"
    print(f"{res} ({r['time']:.1f} s simulated, person contacts {r['person_contacts']}, "
          f"furniture contacts {r['furniture_contacts']})")
    env.close()
    return env, frames, res


def play(env, frames, res, speed):
    m, d = env.m, env.d
    st = {"speed": speed, "pause": False, "restart": False, "follow": True}

    def key(k):
        if k == 32:
            st["pause"] = not st["pause"]
        elif k in (61, 334):            # '=' / keypad +
            st["speed"] = min(8.0, st["speed"] * 2)
        elif k in (45, 333):            # '-' / keypad -
            st["speed"] = max(0.125, st["speed"] / 2)
        elif k == 82:                   # R
            st["restart"] = True
        elif k == 70:                   # F
            st["follow"] = not st["follow"]

    with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
        setup_cam(viewer)
        t_sim, last = 0.0, time.time()
        while viewer.is_running():
            now = time.time()
            if not st["pause"]:
                t_sim += (now - last) * st["speed"]
            last = now
            if st["restart"]:
                t_sim, st["restart"] = 0.0, False
            i = min(int(t_sim / CTRL_DT), len(frames) - 1)
            q, mp, mq, txt = frames[i]
            d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = q, mp, mq
            mujoco.mj_forward(m, d)
            if st["follow"]:
                viewer.cam.lookat[:2] = 0.95 * viewer.cam.lookat[:2] + 0.05 * q[:2]
            done = f"   [{res}]" if i == len(frames) - 1 else ""
            try:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  f"t = {i * CTRL_DT:5.1f} s   x{st['speed']:g}{'  (paused)' if st['pause'] else ''}",
                                  txt + done))
            except AttributeError:          # older mujoco without overlay texts
                pass
            viewer.sync()
            time.sleep(1 / 60)


def live(a):
    env, runner = make_env(a)
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        setup_cam(viewer)
        step = env.step
        wall0, sim0 = time.time(), env.t

        def step_and_show(*args, **kw):
            step(*args, **kw)
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.cam.lookat[:2] = 0.97 * viewer.cam.lookat[:2] + 0.03 * env.pose[:2]
            viewer.sync()
            lag = (env.t - sim0) / a.speed - (time.time() - wall0)
            if lag > 0:
                time.sleep(min(lag, CTRL_DT))

        env.step = step_and_show
        try:
            r = runner.run(a.command)
            print("success" if r["success"] else f"failed at {r['failed_at']}", f"({r['time']:.1f} s simulated)")
            while viewer.is_running():           # keep the final scene open
                viewer.sync()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--command", default=COMMAND)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--live", action="store_true", help="show the simulation while it runs (may stutter)")
    a = ap.parse_args()
    if a.live:
        live(a)
        return
    print(f'simulating "{a.command}" (level {a.level}, seed {a.seed}) ...', flush=True)
    env, frames, res = record(a)
    print("playing back: space pause, +/- speed, R restart, F follow")
    play(env, frames, res, a.speed)


if __name__ == "__main__":
    main()
