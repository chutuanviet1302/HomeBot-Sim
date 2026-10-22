"""Learned manipulation policies (ACT / GR00T N1.7) plugged into the same skill interface
as the scripted expert: `pick_cup()` / `place_cup()` -> bool.

Split of the work: the classical controller walks the robot up to the table / counter from the perceived object
or basin pose and stands (base lock); from there the learned policy drives the arm and the fingers.

A policy maps an observation dict
    {"video.head": HxWx3 uint8, "video.wrist": HxWx3 uint8, "state": (44,), "task": str}
to an action chunk (T x 17) in the layout of homebot/record.py, which is executed at 25 Hz
(each action held for 2 control steps of 20 ms). Psi0 (`psi0_client`) returns T x 34 joint-space actions in the
layout of scripts/export_psi0.py instead; those drive the arm joints directly.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import apartment as apt
from .env import yaw_quat
from .expert import OBJ_IN_BASE, Expert, place_success
from .record import IMG, state_vector

TASKS = {"pick_cup": "pick up the can from the table", "place_cup": "drop the can into the sink"}

# Psi0 layout (scripts/export_psi0.py): left hand 6 | right hand 6 | left arm 7 | right arm 7 | torso rpy 3 |
# torso height 1 (state, 30) [| base vx, vy, vyaw, target yaw (action, 34)]
PSI0_ACTION_DIM = 34
PSI0_R_HAND = slice(6, 12)
PSI0_R_ARM = slice(19, 26)


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
        return {**imgs, "state": state_vector(self.env).astype(np.float32), "task": TASKS[task],
                "pelvis_height": self.env.pelvis_height()}

    def _apply(self, a):
        """17-D action -> palm target (pelvis frame -> world, held by the IK) + Inspire finger targets.
        34-D (Psi0) action -> right arm joint targets (rate limited by the env, no IK) + Inspire finger targets."""
        e = self.env
        if len(a) >= PSI0_ACTION_DIM:
            e.hand_goal_world = None
            e.arm_q_goal = np.asarray(a[PSI0_R_ARM], float)
            e.fingers = np.asarray(a[PSI0_R_HAND], float)
        else:
            p_w, Rb = e.robot_to_world(np.asarray(a[0:3]) + np.array([0, 0, 0.793]))
            R_w = Rb @ rot6d_to_mat(a[3:9])
            e.hand_goal_world = (p_w, R_w)
            e.fingers = np.asarray(a[11:17], float)
        for _ in range(2):
            e.step(np.zeros(3), carrying=e.grasped)

    def _run(self, task, done):
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
        ex = Expert(e, rng=e.rng)
        if not ex.align_and_stand(OBJ_IN_BASE):          # classical approach to the perceived can
            e.stand_mode = False
            return False

        def done():
            cp, _ = e.cup_pose()
            return e.grasped and cp[2] > apt.TABLE_TOP + e.cup_h / 2 + 0.08
        ok = self._run("pick_cup", done)
        Expert(e).to_carry()
        return ok and e.grasped

    def place_cup(self):
        e = self.env
        if Expert(e, rng=e.rng).approach_sink() is None:  # classical approach to the perceived basin
            e.stand_mode = False
            return False
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


def psi0_state(obs):
    """HomeBot observation -> Psi0 30-D state, the mapping of scripts/export_psi0.py (left hand: its fixed relaxed
    pose, torso upright, torso height = pelvis height)."""
    from .g1 import RELAXED_HAND
    s = np.asarray(obs["state"], np.float32)
    return np.concatenate([RELAXED_HAND, s[29:35], s[15:29], [0.0, 0.0, 0.0, obs["pelvis_height"]]]).astype(np.float32)


def psi0_client(host="localhost", port=21074, timeout=120.0):
    """predict_fn backed by a Psi0 inference server (Psi0 `serve_psi0_simple`: HTTP POST /act, numpy arrays as
    base64). `host` is a host name (port `port`) or a full URL, e.g. an https tunnel to a Kaggle / Colab server.
    Returns the served action rows (action_exec_horizon x 34, joint space) for LearnedSkillPolicy."""
    import base64
    import time
    import uuid

    import requests
    from numpy.lib.format import descr_to_dtype, dtype_to_descr

    url = (host if host.startswith("http") else f"http://{host}:{port}").rstrip("/")
    session, client_id, first = requests.Session(), uuid.uuid4().hex[:8], [True]

    def enc(a):
        a = np.ascontiguousarray(a)
        return {"__numpy__": base64.b64encode(a.tobytes()).decode(), "dtype": dtype_to_descr(a.dtype),
                "shape": list(a.shape)}

    def dec(o):
        a = np.frombuffer(base64.b64decode(o["__numpy__"]), descr_to_dtype(o["dtype"]))
        return a.reshape(o["shape"])

    def predict(obs):
        history = {"client_id": client_id, **({"reset": True} if first[0] else {})}
        msg = {"image": {"observation.images.egocentric": enc(np.asarray(obs["video.head"], np.uint8))},
               "instruction": obs["task"], "history": history, "state": {"states": enc(psi0_state(obs))},
               "condition": {}, "gt_action": None, "dataset_name": "real", "timestamp": ""}
        for attempt in range(4):                # tunnels drop connections now and then
            try:
                r = session.post(f"{url}/act", json=msg, timeout=timeout)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        out = r.json()
        if not isinstance(out, dict) or "action" not in out:     # the server reports errors as {"status": ...}
            raise RuntimeError(f"Psi0 server error: {out}")
        first[0] = False
        a = np.atleast_2d(dec(out["action"]) if isinstance(out["action"], dict) else np.asarray(out["action"]))
        return a[:, :PSI0_ACTION_DIM].astype(np.float64)        # drop the padding to 36
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


def act_policy(path, device="cpu", n_exec=10):
    """predict_fn backed by an ACT checkpoint (homebot/act.py, trained with scripts/train_act.py)."""
    from .act import make_predict
    return make_predict(path, device, n_exec)


__all__ = ["LearnedSkillPolicy", "act_policy", "gr00t_client", "psi0_client", "psi0_state", "torch_policy",
           "yaw_quat"]
