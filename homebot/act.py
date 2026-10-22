"""ACT (Action Chunking with Transformers, Zhao et al. 2023) in plain PyTorch - no torchvision / LeRobot needed
at run time, so a checkpoint trained on a Kaggle GPU runs on any machine with torch.

Inputs: head + wrist RGB (224 x 224), 44-D proprioceptive state, task id (pick / drop).
Output: a chunk of `chunk` future 17-D actions (layout of homebot/record.py).
Architecture as in the paper: ResNet-18 features of both cameras (ImageNet initialised when trained), a
transformer encoder over [latent, state, task, image tokens] and a decoder with one query per future step; a
CVAE encoder over the action chunk gives the style latent during training (zero latent at inference).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- ResNet-18 (torchvision layout)
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet18(nn.Module):
    """Same parameter names as torchvision.models.resnet18 (its ImageNet weights load directly)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))      # (B, 512, H/32, W/32)


def sine_pos_2d(h, w, dim, device):
    """2-D sine position embedding (as in DETR / ACT), (h*w, dim)."""
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device) / d))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ys, xs = ys.reshape(-1, 1) * omega, xs.reshape(-1, 1) * omega
    return torch.cat([ys.sin(), ys.cos(), xs.sin(), xs.cos()], 1)


# ----------------------------------------------------------------------------- ACT
class ACT(nn.Module):
    def __init__(self, state_dim=44, action_dim=17, chunk=30, dim=256, heads=8, enc_layers=4, dec_layers=6,
                 latent=32, n_tasks=2, n_cams=2, dropout=0.1):
        super().__init__()
        self.cfg = dict(state_dim=state_dim, action_dim=action_dim, chunk=chunk, dim=dim, heads=heads,
                        enc_layers=enc_layers, dec_layers=dec_layers, latent=latent, n_tasks=n_tasks,
                        n_cams=n_cams, dropout=dropout)
        self.chunk, self.latent, self.dim = chunk, latent, dim
        self.backbone = ResNet18()
        self.img_proj = nn.Conv2d(512, dim, 1)
        self.cam_embed = nn.Embedding(n_cams, dim)
        self.state_proj = nn.Linear(state_dim, dim)
        self.task_embed = nn.Embedding(n_tasks, dim)
        self.latent_proj = nn.Linear(latent, dim)
        self.tok_embed = nn.Embedding(3, dim)                # latent / state / task tokens
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers)
        dlayer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers)
        self.queries = nn.Embedding(chunk, dim)
        self.head = nn.Linear(dim, action_dim)
        # CVAE encoder (training only): [cls, state, actions] -> latent mean / log-variance
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.vae_state = nn.Linear(state_dim, dim)
        self.vae_action = nn.Linear(action_dim, dim)
        self.vae_pos = nn.Embedding(chunk + 2, dim)
        self.vae_encoder = nn.TransformerEncoder(layer(), enc_layers)
        self.vae_out = nn.Linear(dim, 2 * latent)

    def _images(self, imgs):
        """imgs: (B, n_cams, 3, H, W) float in [0, 1] -> tokens (B, n_cams*h*w, dim)."""
        B, C = imgs.shape[:2]
        x = (imgs.flatten(0, 1) - IMG_MEAN.to(imgs.device)) / IMG_STD.to(imgs.device)
        f = self.img_proj(self.backbone(x))                       # (B*C, dim, h, w)
        h, w = f.shape[-2:]
        f = f.flatten(2).transpose(1, 2) + sine_pos_2d(h, w, self.dim, f.device)
        f = f.view(B, C, h * w, self.dim) + self.cam_embed.weight[None, :C, None]
        return f.flatten(1, 2)

    def forward(self, imgs, state, task, actions=None):
        """Returns (predicted chunk (B, chunk, action_dim), mu, logvar); mu/logvar are None without `actions`."""
        B = state.shape[0]
        mu = logvar = None
        if actions is not None:
            seq = torch.cat([self.vae_cls.expand(B, -1, -1), self.vae_state(state)[:, None],
                             self.vae_action(actions)], 1) + self.vae_pos.weight[None]
            stats = self.vae_out(self.vae_encoder(seq)[:, 0])
            mu, logvar = stats[:, :self.latent], stats[:, self.latent:]
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = torch.zeros(B, self.latent, device=state.device)
        toks = torch.stack([self.latent_proj(z), self.state_proj(state), self.task_embed(task)], 1)
        toks = toks + self.tok_embed.weight[None]
        memory = self.encoder(torch.cat([toks, self._images(imgs)], 1))
        out = self.decoder(self.queries.weight[None].expand(B, -1, -1), memory)
        return self.head(out), mu, logvar


def act_loss(pred, target, mask, mu, logvar, kl_weight=10.0):
    """L1 on the valid steps of the chunk + KL of the CVAE latent (as in the ACT paper)."""
    l1 = ((pred - target).abs().mean(-1) * mask).sum() / mask.sum().clamp(min=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


def save_checkpoint(path, model, norm, extra=None):
    torch.save({"cfg": model.cfg, "state_dict": model.state_dict(), "norm": norm, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = ACT(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, {k: torch.as_tensor(v, dtype=torch.float32) for k, v in ck["norm"].items()}, ck


TASK_IDS = {"pick up the can from the table": 0, "drop the can into the sink": 1}


def make_predict(path, device="cpu", n_exec=10):
    """predict_fn for homebot.policy.LearnedSkillPolicy: obs dict -> the first `n_exec` actions of the chunk
    (receding horizon: the policy looks again after n_exec steps)."""
    model, norm, _ = load_checkpoint(path, device)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    def predict(obs):
        with torch.no_grad():
            img = lambda k: torch.from_numpy(obs[k]).permute(2, 0, 1).float().div(255)
            imgs = torch.stack([img("video.head"), img("video.wrist")])[None].to(device)
            s = (torch.from_numpy(obs["state"]).float() - norm["state_mean"]) / norm["state_std"]
            task = torch.tensor([TASK_IDS[obs["task"]]], device=device)
            a, _, _ = model(imgs, s[None].to(device), task)
            a = a[0].cpu() * norm["action_std"] + norm["action_mean"]
        return a[:n_exec].numpy()
    return predict


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = ACT()
    print(f"ACT {n_params(m) / 1e6:.1f} M parameters")
    out, _, _ = m(torch.rand(2, 2, 3, 224, 224), torch.zeros(2, 44), torch.zeros(2, dtype=torch.long),
                  torch.zeros(2, 30, 17))
    print(out.shape, math.isfinite(out.sum().item()))
