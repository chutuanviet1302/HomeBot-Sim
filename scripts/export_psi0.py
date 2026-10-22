"""Convert the HomeBot LeRobot dataset (scripts/gen_data.py) to the Psi0 real-robot fine-tuning format.

python scripts/export_psi0.py --src data/homebot_can --dst data/psi0/real/homebot_can

Psi0 (github.com/physical-superintelligence-lab/Psi0, scripts/data/raw_to_lerobot.py) uses, per frame:
  states (30) = left hand 6 | right hand 6 | 14 arm joints (left 7, right 7) | torso rpy 3 | torso height 1
  action (34) = left hand 6 | right hand 6 | 14 arm joint targets | torso rpy 3 | torso height 1 |
                torso vx, vy, vyaw, target yaw
and one egocentric camera (observation.images.egocentric). Mapping from HomeBot:
  * right hand: the 6 Inspire actuated joints (state) / finger targets (action); left hand: its fixed relaxed pose
  * arm joints: the G1 body joints 15:29 (left arm, right arm, same order as Psi0's sol_q[15:29]); the action is
    the next frame's joint positions (the arm follows its targets within a control step)
  * torso: upright (rpy 0), height = pelvis height of the HomeBot action; base commands 0 (the robot stands -
    only the standing part of each episode is exported, walking up to the table / counter is classical control)
  * camera: the head camera video (224 x 224, 25 fps)
Psi0 pads states / actions to 36 and computes its own statistics (scripts/data/calc_modality_stats.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot.g1 import RELAXED_HAND  # noqa: E402

STAND, HEIGHT = 10, 9                   # HomeBot action indices: stand flag, pelvis height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "homebot_can"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "psi0" / "real" / "homebot_can"))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = [json.loads(l) for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "meta").mkdir(parents=True)
    episodes, index, out_ep = [], 0, 0
    left = list(np.asarray(RELAXED_HAND, float))
    for i in range(info["total_episodes"]):
        chunk = i // info["chunks_size"]
        df = pd.read_parquet(src / info["data_path"].format(episode_chunk=chunk, episode_index=i))
        S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        A = np.stack(df["action"].to_numpy()).astype(np.float64)
        keep = np.nonzero(A[:, STAND] > 0.5)[0]
        if len(keep) < 10:
            continue
        s, e = keep[0], keep[-1] + 1
        rows = []
        for t in range(s, e):
            nxt = min(t + 1, e - 1)
            height = float(A[t, HEIGHT])
            state = left + list(S[t, 29:35]) + list(S[t, 15:29]) + [0.0, 0.0, 0.0, height]
            action = left + list(A[t, 11:17]) + list(S[nxt, 15:29]) + [0.0, 0.0, 0.0, height] + [0.0] * 4
            rows.append({"states": np.asarray(state, np.float32), "action": np.asarray(action, np.float32),
                         "timestamp": np.float32((t - s) / info["fps"]), "frame_index": t - s,
                         "episode_index": out_ep, "index": index + t - s,
                         "task_index": int(df["task_index"].iloc[0]), "next.done": t == e - 1})
        (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(dst / "data" / "chunk-000" / f"episode_{out_ep:06d}.parquet")
        # video: the standing part of the head camera clip
        import cv2
        cap = cv2.VideoCapture(str(src / info["video_path"].format(
            episode_chunk=chunk, video_key="observation.images.head", episode_index=i)))
        frames, k = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if s <= k < e:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            k += 1
        cap.release()
        import imageio.v2 as imageio
        vdir = dst / "videos" / "chunk-000" / "observation.images.egocentric"
        vdir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(vdir / f"episode_{out_ep:06d}.mp4", frames, fps=info["fps"], codec="libx264",
                         pixelformat="yuv420p", macro_block_size=1)
        task = next(t["task"] for t in tasks if t["task_index"] == int(df["task_index"].iloc[0]))
        episodes.append({"episode_index": out_ep, "tasks": [task], "length": len(rows)})
        index += len(rows)
        out_ep += 1
        print(f"episode {i} -> {out_ep - 1}: {len(rows)} frames", flush=True)
    H, W = 224, 224
    out_info = {
        "codebase_version": "v2.1", "robot_type": "g1", "fps": info["fps"], "total_episodes": out_ep,
        "total_frames": index, "total_tasks": len(tasks), "total_videos": out_ep, "total_chunks": 1,
        "chunks_size": 1000, "splits": {"train": f"0:{out_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {"dtype": "video", "shape": [H, W, 3],
                                              "names": ["height", "width", "channel"],
                                              "info": {"video.fps": info["fps"], "video.codec": "h264",
                                                       "video.pix_fmt": "yuv420p", "has_audio": False}},
            "states": {"dtype": "float32", "shape": [30]},
            "action": {"dtype": "float32", "shape": [34]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]}}}
    (dst / "meta" / "info.json").write_text(json.dumps(out_info, indent=2))
    (dst / "meta" / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    print(f"{out_ep} episodes, {index} frames -> {dst}")


if __name__ == "__main__":
    main()
