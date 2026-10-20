"""Learned manipulation policies (ACT / GR00T N1.7) plugged into the same skill interface
as the scripted expert: `pick_cup()` / `place_cup()` -> bool.

A policy maps an observation dict
    {"video.head": HxWx3 uint8, "video.wrist": HxWx3 uint8, "state": (44,), "task": str}
to an action chunk (T x 17) in the layout of homebot/record.py, which is executed at 25 Hz
(each action held for 2 control steps of 20 ms).
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import apartment as apt
from .env import yaw_quat
from .expert import Expert, place_success
from .record import IMG, state_vector

TASKS = {"pick_cup": "pick up the can from the table", "place_cup": "drop the can into the sink"}


def rot6d_to_mat(r6):
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    x = a / (np.linalg.norm(a) + 1e-9)
    b = b - np.dot(x, b) * x
    y = b / (np.linalg.norm(b) + 1e-9)
    return np.stack([x, y, np.cross(x, y)], 1)


def rot6d_to_quat(r6):
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    x = a / (np.linalg.norm(a) + 1e-9)
    b = b - np.dot(x, b) * x
    y = b / (np.linalg.norm(b) + 1e-9)
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).reshape(-1))
    return q


class LearnedSkillPolicy:
    def __init__(self, env, predict_fn, max_steps=600, grasp_close=0.7):
        self.env = env
        self.predict = predict_fn
        self.max_steps = max_steps
        self.grasp_close = grasp_close
        self.renderer = mujoco.Renderer(env.m, IMG, IMG)

    def _obs(self, task):
        imgs = {}
        for cam in ("head", "wrist"):
            self.renderer.update_scene(self.env.d, cam)
            imgs[f"video.{cam}"] = self.renderer.render().copy()
        return {**imgs, "state": state_vector(self.env).astype(np.float32), "task": TASKS[task]}

    def _apply(self, a):
        """17-D action -> palm target (pelvis frame -> world, held by the IK) + Inspire finger targets."""
        e = self.env
        p_w, Rb = e.robot_to_world(np.asarray(a[0:3]) + np.array([0, 0, 0.793]))
        R_w = Rb @ rot6d_to_mat(a[3:9])
        e.hand_goal_world = (p_w, R_w)
        e.fingers = np.asarray(a[11:17], float)
        for _ in range(2):
            e.step(np.zeros(3), carrying=e.grasped)

    def _run(self, task, done):
        Expert(self.env).stand()
        steps = 0
        while steps < self.max_steps:
            chunk = np.atleast_2d(self.predict(self._obs(task)))
            for a in chunk:
                self._apply(a)
                steps += 1
                if done():
                    return True
        return False

    def pick_cup(self):
        e = self.env

        def done():
            cp, _ = e.cup_pose()
            return e.grasped and cp[2] > apt.TABLE_TOP + e.cup_h / 2 + 0.08
        ok = self._run("pick_cup", done)
        Expert(e).to_carry()
        return ok and e.grasped

    def place_cup(self):
        e = self.env
        ok = self._run("place_cup", lambda: not e.grasped and place_success(e))
        Expert(e).to_carry()
        return ok and place_success(e)


def gr00t_client(host="localhost", port=5555):  # pragma: no cover - needs Isaac-GR00T
    """predict_fn backed by an Isaac-GR00T inference server (scripts/inference_service.py)."""
    from gr00t.eval.robot import RobotInferenceClient
    client = RobotInferenceClient(host=host, port=port)

    def predict(obs):
        o = {"video.head": obs["video.head"][None], "video.wrist": obs["video.wrist"][None],
             "state.body_joints": obs["state"][None, 0:29], "state.right_hand": obs["state"][None, 29:35],
             "state.right_wrist_pos": obs["state"][None, 35:38], "state.right_wrist_rot6d": obs["state"][None, 38:44],
             "annotation.human.task_description": [obs["task"]]}
        a = client.get_action(o)
        return np.concatenate([a["action.right_wrist_pos"], a["action.right_wrist_rot6d"], a["action.base_height"],
                               a["action.stand"], a["action.right_hand"]], -1)
    return predict


def torch_policy(path, device="cpu"):  # pragma: no cover - for a TorchScript export of ACT
    import torch
    model = torch.jit.load(path, map_location=device).eval()

    def predict(obs):
        with torch.no_grad():
            img = lambda k: torch.from_numpy(obs[k]).permute(2, 0, 1)[None].float().div(255).to(device)
            out = model(img("video.head"), img("video.wrist"), torch.from_numpy(obs["state"])[None].to(device))
        return out[0].cpu().numpy()
    return predict


__all__ = ["LearnedSkillPolicy", "gr00t_client", "torch_policy", "yaw_quat"]
