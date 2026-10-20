"""Render a demo video: perspective view of the apartment + robot head camera, with a HUD.

python scripts/make_video.py --level 2 --seed 3 --out report/demo_level2.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_full import COMMAND, living_room_start  # noqa: E402
from homebot import apartment as apt  # noqa: E402
from homebot.env import HomeEnv  # noqa: E402
from homebot.tasks import TaskRunner  # noqa: E402

W, H = 960, 540
FPS = 25


class VideoHook:
    """Wraps env.step to grab frames every 2 control steps."""

    def __init__(self, env, runner, writer):
        self.env, self.runner, self.w = env, runner, writer
        self.r_main = mujoco.Renderer(env.m, H, W)
        self.r_head = mujoco.Renderer(env.m, 240, 320)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.k = 0
        self._step = env.step
        env.step = self.step
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [4.0, 3.0, 0.3]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 8.2, -90, -58

    def _head_view(self):
        """Head camera with the perception output (detections: box, label, ICP fit) of the last frame."""
        e = self.env
        per = e._perception
        self.r_head.update_scene(e.d, "head")
        img = np.ascontiguousarray(self.r_head.render())
        if per is not None and e.t - getattr(per, "last_t", -9.0) < 0.6:     # only fresh detections
            sx, sy = img.shape[1] / per.W, img.shape[0] / per.H
            for d in per.last_dets:
                u0, v0, u1, v1 = d.bbox
                p0, p1 = (int(u0 * sx) - 3, int(v0 * sy) - 3), (int(u1 * sx) + 3, int(v1 * sy) + 3)
                col = (40, 220, 60) if d.label == e.cup_type else (240, 180, 40)
                cv2.rectangle(img, p0, p1, col, 2)
                cv2.putText(img, f"{d.label} {1000 * d.rmse:.1f}mm", (p0[0], max(12, p0[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return img

    def step(self, *a, **kw):
        self._step(*a, **kw)
        self.k += 1
        if self.k % 2:
            return
        e = self.env
        # follow the robot loosely
        self.cam.lookat[:2] = 0.85 * self.cam.lookat[:2] + 0.15 * (0.5 * e.pose[:2] + 0.5 * np.array([4.0, 3.0]))
        self.r_main.update_scene(e.d, self.cam, self.opt)
        frame = self.r_main.render().copy()
        head = self._head_view()
        frame[10:10 + head.shape[0], W - 10 - head.shape[1]:W - 10] = head
        nav = self.runner.nav
        st = e.stats
        txt = [f'"{COMMAND}"  t={e.t:5.1f}s',
               f"state: {self.runner.state}   safety: {nav.safety.state}",
               f"cmd vx={nav.cmd[0]:+.2f} vy={nav.cmd[1]:+.2f} wz={nav.cmd[2]:+.2f}",
               f"person contacts: {st['person_contacts']}   min dist: {min(st['min_person_dist'], 9):.2f} m",
               f"arm contacts: {st['arm_contacts']}   perception err: {st.get('perc_err_mm', float('nan')):.1f} mm"]
        for i, t in enumerate(txt):
            cv2.putText(frame, t, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, t, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.w.append_data(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "report" / "demo.mp4"))
    a = ap.parse_args()
    rng = np.random.default_rng(70_000 + a.seed)
    env = HomeEnv(level=a.level, seed=1000 + a.seed)
    start = living_room_start(env, rng)
    env.reset(start, robot_route=(start[:2], np.array(apt.PLACES["table"]["pos"])))
    runner = TaskRunner(env)
    runner.state = "START"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(a.out, fps=FPS, macro_block_size=1, quality=7) as w:
        VideoHook(env, runner, w)
        r = runner.run(COMMAND)
    print(r["success"], r["failed_at"], round(r["time"], 1), "s ->", a.out)


if __name__ == "__main__":
    main()
