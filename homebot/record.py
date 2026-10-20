"""Write demonstrations in LeRobot v2.1 layout (+ GR00T meta/modality.json).

dataset/
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/modality.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.head/episode_000000.mp4
  videos/chunk-000/observation.images.wrist/episode_000000.mp4
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from .env import quat_to_mat

FPS = 25                 # record every 2nd 50 Hz control step
STATE_DIM = 44           # 29 G1 joints | 6 fingers | 9 wrist pose (pos + rot6d)
ACTION_DIM = 17          # 9 palm pose | pelvis height, stand flag | 6 fingers
IMG = 224


def state_vector(env):
    """44-D proprioception: 29 G1 body joint angles | 6 Inspire actuated joints | palm pose in the pelvis frame
    (position 3 + rot6d 6)."""
    from .expert import rot6d_mat
    m, d = env.m, env.d
    body = [j for j in range(m.njnt) if m.jnt_type[j] == 3 and not m.joint(j).name.startswith(("R_", "L_"))]
    q = d.qpos[[m.jnt_qposadr[j] for j in body]]
    fingers = d.qpos[[m.jnt_qposadr[j] for j in env.ctrl.finger_j]]
    p, R = env.ctrl.site_pose(d)
    import numpy as _np
    from .env import quat_to_mat, yaw_quat
    Rb = quat_to_mat(yaw_quat(env.pose[2]))
    p_r = env.world_to_robot(p) - _np.array([0, 0, 0.793])
    return _np.concatenate([q, fingers, p_r, rot6d_mat(Rb.T @ R)])


class Recorder:
    def __init__(self, root, images=True):
        self.root = Path(root)
        self.images = images
        self.tasks = {}
        self.episodes = []
        self.total = 0
        self.renderer = None
        self._reset_ep()

    def _reset_ep(self):
        self.frames, self.head, self.wrist, self._k = [], [], [], 0

    def start(self, env, task):
        self._reset_ep()
        self.env = env
        self.task = task
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        if self.images and (self.renderer is None or self.renderer.model is not env.m):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(env.m, IMG, IMG)

    def add(self, env, action):
        self._k += 1
        if self._k % 2:
            return
        self.frames.append((state_vector(env), np.asarray(action, np.float32)))
        if self.images:
            for cam, buf in (("head", self.head), ("wrist", self.wrist)):
                self.renderer.update_scene(env.d, cam)
                buf.append(self.renderer.render().copy())

    def end(self, success: bool):
        import imageio.v2 as imageio
        ep = len(self.episodes)
        n = len(self.frames)
        if n == 0:
            return
        df = pd.DataFrame({
            "observation.state": [f[0].astype(np.float32) for f in self.frames],
            "action": [f[1] for f in self.frames],
            "timestamp": np.arange(n, dtype=np.float32) / FPS,
            "frame_index": np.arange(n),
            "episode_index": np.full(n, ep),
            "index": np.arange(self.total, self.total + n),
            "task_index": np.full(n, self.tasks[self.task]),
            "next.success": np.r_[np.zeros(n - 1, bool), [success]],
        })
        p = self.root / "data/chunk-000" / f"episode_{ep:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        if self.images:
            for key, buf in (("observation.images.head", self.head), ("observation.images.wrist", self.wrist)):
                v = self.root / "videos/chunk-000" / key / f"episode_{ep:06d}.mp4"
                v.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(v, buf, fps=FPS, macro_block_size=1)
        self.episodes.append({"episode_index": ep, "tasks": [self.task], "length": n, "success": success})
        self.total += n
        self._reset_ep()

    def finalize(self):
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        with open(meta / "episodes.jsonl", "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(meta / "tasks.jsonl", "w") as f:
            for t, i in self.tasks.items():
                f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        vid = lambda: {"dtype": "video", "shape": [IMG, IMG, 3], "names": ["height", "width", "channel"],
                       "info": {"video.fps": FPS, "video.codec": "h264"}}
        features = {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }
        if self.images:
            features["observation.images.head"] = vid()
            features["observation.images.wrist"] = vid()
        info = {"codebase_version": "v2.1", "robot_type": "unitree_g1_inspire_rh56", "fps": FPS,
                "total_episodes": len(self.episodes), "total_frames": self.total, "total_tasks": len(self.tasks),
                "chunks_size": 1000, "total_chunks": 1, "splits": {"train": f"0:{len(self.episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features}
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        modality = {
            "state": {"body_joints": {"start": 0, "end": 29}, "right_hand": {"start": 29, "end": 35},
                      "right_wrist_pos": {"start": 35, "end": 38}, "right_wrist_rot6d": {"start": 38, "end": 44}},
            "action": {"right_wrist_pos": {"start": 0, "end": 3}, "right_wrist_rot6d": {"start": 3, "end": 9},
                       "base_height": {"start": 9, "end": 10}, "stand": {"start": 10, "end": 11},
                       "right_hand": {"start": 11, "end": 17}},
            "video": {"head": {"original_key": "observation.images.head"},
                      "wrist": {"original_key": "observation.images.wrist"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }
        (meta / "modality.json").write_text(json.dumps(modality, indent=2))
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def merge_parts(parts, out):
    """Merge LeRobot datasets written by parallel generator processes into one (episodes renumbered)."""
    import shutil
    out = Path(out)
    tasks, episodes, total = {}, [], 0
    for part in parts:
        part = Path(part)
        ptasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
                  for l in (part / "meta/tasks.jsonl").read_text().splitlines() if l.strip()}
        for line in (part / "meta/episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            ep = len(episodes)
            df = pd.read_parquet(part / "data/chunk-000" / f"episode_{e['episode_index']:06d}.parquet")
            task = ptasks[int(df.task_index.iloc[0])]
            tasks.setdefault(task, len(tasks))
            df["episode_index"] = ep
            df["index"] = np.arange(total, total + len(df))
            df["task_index"] = tasks[task]
            p = out / "data/chunk-000" / f"episode_{ep:06d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p)
            for key in ("observation.images.head", "observation.images.wrist"):
                src = part / "videos/chunk-000" / key / f"episode_{e['episode_index']:06d}.mp4"
                if src.exists():
                    dst = out / "videos/chunk-000" / key / f"episode_{ep:06d}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            episodes.append({**e, "episode_index": ep})
            total += len(df)
    rec = Recorder(out, images=(out / "videos").exists())
    rec.tasks, rec.episodes, rec.total = tasks, episodes, total
    rec.finalize()
    return len(episodes), total
