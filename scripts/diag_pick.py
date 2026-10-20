"""Debug helper: phase-by-phase log (+ optional snapshots) of one expert pick.
python scripts/diag_pick.py --seed 0 [--png out.png] [--nonoise]"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_manip import perturbed  # noqa: E402
from homebot import expert as X  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--png", default=None)
ap.add_argument("--nonoise", action="store_true")
ap.add_argument("--noperturb", action="store_true")
ap.add_argument("--cup", default=None)
a = ap.parse_args()
if a.nonoise:
    X.OBJ_NOISE = 0.0
rng = np.random.default_rng(50_000 + a.seed)
env = HomeEnv(level=1, seed=a.seed, cup_type=a.cup)
from homebot import apartment as apt  # noqa: E402
start = np.array([*apt.PLACES["table"]["pos"], apt.PLACES["table"]["yaw"]]) if a.noperturb else perturbed("table", rng)
env.reset(start, arm_ready=True)
ex = X.Expert(env, rng=rng)
imgs = []
if a.png:
    import mujoco
    r = mujoco.Renderer(env.m, 240, 320)
    cam = mujoco.MjvCamera()


def snap():
    if a.png:
        cam.lookat[:] = env.cup_pose()[0]
        cam.distance, cam.azimuth, cam.elevation = 0.5, np.degrees(env.pose[2]) - 90, -35
        r.update_scene(env.d, cam)
        imgs.append(r.render().copy())


om, oh, oj, oa = ex.move_hand, ex.hand, ex.move_joint, ex.align


def mj(t, R, **kw):
    r = oj(t, R, **kw)
    p, _ = env.hand_world()
    print(f"joint palm err {np.linalg.norm(p - t):.3f} cup(robot) {env.world_to_robot(env.cup_pose()[0]).round(3)} contacts {env.cup_contacts()}")
    snap()
    return r


def al(*a_, **k):
    r = oa(*a_, **k)
    print("align", r, "pose", env.pose.round(3), "cup(robot)", env.world_to_robot(env.cup_pose()[0]).round(3))
    return r


def mh(t, R, **kw):
    om(t, R, **kw)
    p, _ = env.hand_world()
    rel = env.world_to_robot(env.cup_pose()[0])
    print(f"move  palm err {np.linalg.norm(p - t):.3f}  ik {env.ik_err:.3f}  cup(robot) {rel.round(3)}  contacts {env.cup_contacts()}")
    snap()


def hh(v, d=0.5):
    oh(v, d)
    print(f"hand  {np.round(v, 2)} contacts {env.cup_contacts()}")
    snap()


ex.move_hand, ex.hand, ex.move_joint, ex.align = mh, hh, mj, al
print("cup", env.cup_type, "start offset", (start - [*apt.PLACES['table']['pos'], apt.PLACES['table']['yaw']]).round(3))
print("pick", ex.pick_cup(), {k: v for k, v in env.stats.items() if "arm" in k})
if a.png:
    import imageio
    while len(imgs) % 4:
        imgs.append(imgs[-1])
    rows = [np.concatenate(imgs[i:i + 4], 1) for i in range(0, len(imgs), 4)]
    imageio.imwrite(a.png, np.concatenate(rows, 0))
