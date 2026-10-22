"""Train ACT (homebot/act.py) on the LeRobot v2.1 dataset written by scripts/gen_data.py.

Kaggle (1x T4):   python scripts/train_act.py --data /kaggle/input/homebot-can --out act_homebot.pt --pretrained
Local smoke test: python scripts/train_act.py --data data/homebot_can --out logs/act_smoke.pt --max-episodes 2 \
                      --steps 20 --batch 4 --img 112

Only the frames where the robot stands (action "stand" flag = 1) are used: walking up to the table / counter is
done by the classical controller from the perceived object pose, the learned policy does the manipulation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homebot.act import ACT, TASK_IDS, act_loss, n_params, save_checkpoint  # noqa: E402

STAND = 10                          # index of the "stand" flag in the 17-D action


def read_video(path, size):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if f.shape[0] != size:
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)


def load_dataset(root, size, max_episodes=None, only_standing=True):
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tasks[t["task_index"]] = TASK_IDS[t["task"]]
    eps = []
    n = info["total_episodes"] if max_episodes is None else min(max_episodes, info["total_episodes"])
    for i in range(n):
        chunk = i // info["chunks_size"]
        df = pd.read_parquet(root / info["data_path"].format(episode_chunk=chunk, episode_index=i))
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        vids = [read_video(root / info["video_path"].format(episode_chunk=chunk, video_key=f"observation.images.{k}",
                                                            episode_index=i), size) for k in ("head", "wrist")]
        T = min(len(state), len(vids[0]), len(vids[1]))
        keep = np.arange(T)
        if only_standing:
            keep = keep[action[:T, STAND] > 0.5]
        if len(keep) < 5:
            continue
        s, e = keep[0], keep[-1] + 1                  # the standing part is one contiguous block
        eps.append({"state": state[s:e], "action": action[s:e], "head": vids[0][s:e], "wrist": vids[1][s:e],
                    "task": tasks[int(df["task_index"].iloc[0])]})
        print(f"episode {i}: {e - s} frames (task {eps[-1]['task']})", flush=True)
    return eps


def norm_stats(eps):
    S = np.concatenate([e["state"] for e in eps])
    A = np.concatenate([e["action"] for e in eps])
    return {"state_mean": S.mean(0), "state_std": np.maximum(S.std(0), 1e-2),
            "action_mean": A.mean(0), "action_std": np.maximum(A.std(0), 1e-2)}


class Sampler:
    def __init__(self, eps, norm, chunk, rng):
        self.eps, self.chunk, self.rng = eps, chunk, rng
        self.idx = np.array([(k, t) for k, e in enumerate(eps) for t in range(len(e["state"]))])
        self.n = {k: torch.as_tensor(v) for k, v in norm.items()}

    def batch(self, B, device, shift=8):
        pick = self.idx[self.rng.integers(len(self.idx), size=B)]
        imgs, st, act, mask, task = [], [], [], [], []
        for k, t in pick:
            e = self.eps[k]
            a = e["action"][t:t + self.chunk]
            m = np.ones(self.chunk, np.float32)
            if len(a) < self.chunk:                     # past the end: repeat the last action, masked out
                m[len(a):] = 0
                a = np.concatenate([a, np.repeat(a[-1:], self.chunk - len(a), 0)])
            pair = []
            for cam in ("head", "wrist"):
                im = e[cam][t]
                if shift:                               # random shift augmentation (as in ACT / DrQ)
                    H = im.shape[0]
                    p = np.pad(im, ((shift, shift), (shift, shift), (0, 0)), mode="edge")
                    dy, dx = self.rng.integers(0, 2 * shift + 1, 2)
                    im = p[dy:dy + H, dx:dx + H]
                pair.append(im)
            imgs.append(np.stack(pair))
            st.append(e["state"][t])
            act.append(a)
            mask.append(m)
            task.append(e["task"])
        imgs = torch.from_numpy(np.stack(imgs)).permute(0, 1, 4, 2, 3).float().div(255)
        st = (torch.from_numpy(np.stack(st)) - self.n["state_mean"]) / self.n["state_std"]
        act = (torch.from_numpy(np.stack(act)) - self.n["action_mean"]) / self.n["action_std"]
        return (imgs.to(device, non_blocking=True), st.to(device), torch.tensor(task, device=device),
                act.to(device), torch.from_numpy(np.stack(mask)).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "homebot_can"))
    ap.add_argument("--out", default=str(ROOT / "logs" / "act_homebot.pt"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--kl", type=float, default=10.0)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet ResNet-18 weights (needs torchvision)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    eps = load_dataset(a.data, a.img, a.max_episodes)
    norm = norm_stats(eps)
    print(f"{len(eps)} episodes, {sum(len(e['state']) for e in eps)} standing frames, loaded in "
          f"{time.time() - t0:.0f} s, device {device}", flush=True)
    model = ACT(chunk=a.chunk).to(device)
    if a.pretrained:
        import torchvision
        w = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
        missing = model.backbone.load_state_dict({k: v for k, v in w.items() if not k.startswith("fc.")})
        print("ImageNet ResNet-18 loaded", missing, flush=True)
    print(f"ACT: {n_params(model) / 1e6:.1f} M parameters", flush=True)
    back = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": back, "lr": a.lr_backbone}], weight_decay=1e-4)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    sampler = Sampler(eps, norm, a.chunk, rng)
    log, t1 = [], time.time()
    model.train()
    for step in range(1, a.steps + 1):
        imgs, st, task, act, mask = sampler.batch(a.batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred, mu, logvar = model(imgs, st, task, act)
            loss, l1, kl = act_loss(pred.float(), act, mask, mu.float(), logvar.float(), a.kl)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        log.append((float(l1), float(kl)))
        if step % 100 == 0 or step == a.steps:
            l1m, klm = np.mean(log, 0)
            log = []
            print(f"step {step}/{a.steps}  l1 {l1m:.4f}  kl {klm:.4f}  {(time.time() - t1) / step:.3f} s/step",
                  flush=True)
        if step % 2000 == 0 or step == a.steps:
            save_checkpoint(a.out, model.eval(), norm, {"steps": step, "args": vars(a)})
            model.train()
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
