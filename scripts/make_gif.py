"""Animated GIF of a recorded episode for the README (from the replay cache written by scripts/view.py).

python scripts/view.py --level 2 --seed 2      # once: simulates and caches the episode
python scripts/make_gif.py --out report/demo.gif
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import view  # noqa: E402
from homebot.env import CTRL_DT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--speed", type=float, default=7.0, help="playback speed-up")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--out", default=str(ROOT / "report" / "demo.gif"))
    a = ap.parse_args()
    a.command = view.COMMAND
    cached = view.load_cache(a)
    if cached is None:
        sys.exit("no up-to-date recording: run  python scripts/view.py --level %d --seed %d  first" % (a.level, a.seed))
    frames, res = cached
    env, _ = view.make_env(a)                     # original meshes, shadows on (offline: speed does not matter)
    m, d = env.m, env.d
    h = int(a.width * 9 / 16) // 2 * 2
    r = mujoco.Renderer(m, h, a.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.geomgroup[3:] = 0
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [4.0, 3.0, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 3.6, -90, -58
    step = max(1, int(round(a.speed / (a.fps * CTRL_DT))))
    out = []
    for i in range(0, len(frames), step):
        q, mp, mq, txt = frames[i]
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = q, mp, mq
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        cam.lookat[:2] = q[:2] if i == 0 else 0.3 * cam.lookat[:2] + 0.7 * q[:2]     # follow the robot
        r.update_scene(d, cam, opt)
        img = np.ascontiguousarray(r.render())
        label = f"t={i * CTRL_DT:5.1f}s  x{a.speed:g}  {txt.split('  safety')[0]}"
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img)
    out += [out[-1]] * a.fps                     # hold the last frame for a second
    imageio.mimsave(a.out, out, duration=1000 / a.fps, loop=0)
    print(f"{len(out)} frames ({res}) -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
