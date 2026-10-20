"""Interactive MuJoCo viewer: watch the robot do a command (default: the full task with a person walking around).

The episode is simulated first (same physics, perception and planners as the evaluations; ~2-3 min) and the
recorded states are then played back at exact real time, so the view is smooth even though perception and arm
path planning sometimes take longer than one control step. `--live` shows the simulation while it runs instead.

python scripts/view.py                         # "bring the can to the sink", level 2, seed 2
python scripts/view.py --level 3 --seed 7 --speed 0.5
python scripts/view.py --live
The recorded episode is cached in logs/ and replayed directly next time (re-simulated when the code changed;
--fresh forces it).
The replay draws reduced-detail copies of the robot meshes (the originals have ~0.9 M triangles: a few frames per
second on an integrated GPU) and hides the hands' collision meshes; physics and perception are not affected
(--full-detail draws the originals, --shadows turns shadows on).
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
from homebot import g1 as g1mod  # noqa: E402
from homebot.env import CTRL_DT, HomeEnv  # noqa: E402
from homebot.tasks import TaskRunner  # noqa: E402


LOWPOLY_RATIO = 0.25                      # fraction of the triangles kept per robot mesh in the viewer
LOWPOLY_MIN = 4000                        # meshes smaller than this are kept as they are
LOWPOLY_DIR = ROOT / "logs" / "lowpoly"


def _find_mesh(f):
    p = Path(f)
    if p.is_file():
        return p
    for d in (g1mod.G1_DIR / "assets", g1mod.INSPIRE_DIR / "meshes"):
        if (d / p.name).is_file():
            return d / p.name
    return None


def _lowpoly(path, ratio=None):
    """(vertices, faces) of a decimated copy of a mesh file (quadric decimation, cached), or None to keep the
    original. Every triangle keeps the orientation of the original surface it replaces (decimation can flip
    triangles, which then render black)."""
    ratio = LOWPOLY_RATIO if ratio is None else ratio
    out = LOWPOLY_DIR / f"{path.stem}_r{int(100 * ratio)}.npz"
    if out.exists():
        z = np.load(out)
        return (z["v"], z["f"]) if len(z["f"]) else None
    import open3d as o3d
    from scipy.spatial import cKDTree
    src = o3d.io.read_triangle_mesh(str(path))
    v0, f0 = np.asarray(src.vertices), np.asarray(src.triangles)
    LOWPOLY_DIR.mkdir(parents=True, exist_ok=True)
    if len(f0) < LOWPOLY_MIN:
        np.savez(out, v=np.zeros((0, 3)), f=np.zeros((0, 3), np.int32))
        return None
    src.remove_duplicated_vertices()                 # STL: shared corners merged, so decimation keeps the surface closed
    mesh = src.simplify_quadric_decimation(target_number_of_triangles=int(ratio * len(f0)))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    v, f = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.triangles, np.int32).copy()
    # orientation check against the nearest original triangle
    n0 = np.cross(v0[f0[:, 1]] - v0[f0[:, 0]], v0[f0[:, 2]] - v0[f0[:, 0]])
    n1 = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    _, k = cKDTree(v0[f0].mean(1)).query(v[f].mean(1))
    flip = np.einsum("ij,ij->i", n0[k], n1) < 0
    f[flip] = f[flip][:, ::-1]
    LOWPOLY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, v=v, f=f)
    return v, f


def use_lowpoly_meshes():
    """Build the scene with reduced-detail robot meshes (viewer only)."""
    orig = g1mod.build_spec

    def build_spec(world):
        orig(world)
        for mesh in world.meshes:
            path = _find_mesh(mesh.file) if mesh.file else None
            if path is None or path.suffix.lower() not in (".stl", ".obj"):
                continue
            lp = _lowpoly(path)
            if lp is None:
                continue
            v, f = lp
            mesh.file = ""
            mesh.uservert = v.reshape(-1).tolist()
            mesh.userface = f.reshape(-1).tolist()
        return world
    g1mod.build_spec = build_spec


def make_env(a):
    rng = np.random.default_rng(70_000 + a.seed)
    env = HomeEnv(level=a.level, seed=1000 + a.seed)
    start = living_room_start(env, rng)
    env.reset(start, robot_route=(start[:2], np.array(apt.PLACES["table"]["pos"])))
    return env, TaskRunner(env)


def setup_cam(viewer):
    viewer.opt.geomgroup[:] = 1
    viewer.opt.geomgroup[3:] = 0              # the hands' collision meshes (duplicates of the visual ones)
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


def cache_path(a):
    return ROOT / "logs" / f"replay_l{a.level}_s{a.seed}.npz"


def code_mtime():
    return max(p.stat().st_mtime for p in (ROOT / "homebot").rglob("*.py"))


def load_cache(a):
    p = cache_path(a)
    if not p.exists() or p.stat().st_mtime < code_mtime():
        return None
    z = np.load(p, allow_pickle=False)
    if str(z["command"]) != a.command:
        return None
    frames = list(zip(z["qpos"], z["mocap_pos"], z["mocap_quat"], [str(t) for t in z["text"]]))
    return frames, str(z["res"])


def save_cache(a, frames, res):
    p = cache_path(a)
    p.parent.mkdir(exist_ok=True)
    q, mp, mq, txt = zip(*frames)
    np.savez_compressed(p, qpos=np.array(q), mocap_pos=np.array(mp), mocap_quat=np.array(mq),
                        text=np.array(txt), res=res, command=a.command)


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
            mujoco.mj_kinematics(m, d)          # poses only (replay): no dynamics, no contacts
            mujoco.mj_camlight(m, d)
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
    ap.add_argument("--fresh", action="store_true", help="simulate again even if a recording is cached")
    ap.add_argument("--full-detail", action="store_true", help="draw the original robot meshes")
    ap.add_argument("--shadows", action="store_true")
    a = ap.parse_args()
    if a.live:
        live(a)
        return
    cached = None if a.fresh else load_cache(a)
    if cached is not None:
        frames, res = cached
        if not a.full_detail:
            use_lowpoly_meshes()
        env, _ = make_env(a)                 # same scene (layout, objects) for the recorded states
        print(f"replaying the recording in {cache_path(a).name} ({res})")
    else:
        print(f'simulating "{a.command}" (level {a.level}, seed {a.seed}) ...', flush=True)
        env, frames, res = record(a)
        save_cache(a, frames, res)
        if not a.full_detail:                # replay on a light copy of the same scene
            use_lowpoly_meshes()
            env, _ = make_env(a)
    print("playing back: space pause, +/- speed, R restart, F follow")
    if not a.shadows:
        env.m.light_castshadow[:] = 0
    play(env, frames, res, a.speed)


if __name__ == "__main__":
    main()
