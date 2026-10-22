"""Generate the Kaggle notebooks (python notebooks/make_notebooks.py)."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n"),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True}},
            "nbformat": 4, "nbformat_minor": 5}


COMMON = ("code", """
import os, subprocess
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
DATA = "/kaggle/input/homebot-can"          # Kaggle Dataset made from data/homebot_can
!nvidia-smi
""")

gr00t = nb([
    ("markdown", "# HomeBot-Sim: fine-tune GR00T N1.7 (frozen VLM, fp16, 2x T4)\n"
                 "Day-1 gate: run the 20-step VRAM probe first. If it OOMs, ACT is the primary policy."),
    COMMON,
    ("code", """
!git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T /kaggle/working/Isaac-GR00T
%cd /kaggle/working/Isaac-GR00T
!pip install -q -e . && pip install -q flash-attn --no-build-isolation || true
!git clone --depth 1 https://github.com/<your-user>/homebot-sim /kaggle/working/homebot-sim
"""),
    ("markdown", "## 20-step VRAM probe"),
    ("code", """
import sys; sys.path.insert(0, "/kaggle/working/homebot-sim")
from configs.gr00t_g1_sonic_inspire import TRAIN
cmd = f'''python scripts/gr00t_finetune.py --dataset-path {DATA} --num-gpus 1
  --output-dir /kaggle/working/probe --max-steps 20 --batch-size {TRAIN["batch_size"]}
  --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model
  --data-config /kaggle/working/homebot-sim/configs/gr00t_g1_sonic_inspire.py:make_config
  --embodiment-tag new_embodiment --report-to none'''
!{cmd} 2>&1 | tail -20
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
"""),
    ("markdown", "## Full fine-tune (checkpoint every 500 steps -> HF Hub, private)"),
    ("code", """
cmd = cmd.replace("--max-steps 20", f"--max-steps {TRAIN['max_steps']} --save-steps 500").replace("/probe", "/n17_homebot")
!{cmd} 2>&1 | tail -50
!huggingface-cli upload --private <your-user>/homebot-n17 /kaggle/working/n17_homebot .
"""),
])

act = nb([
    ("markdown", """# HomeBot-Sim: train ACT on 1x T4 (Kaggle)

1. Data: `data/homebot_can` (LeRobot v2.1, 60 episodes, 22 MB) comes with the cloned repo; a Kaggle Dataset with
   `meta/info.json` added as an input is used instead if present.
2. *Settings -> Accelerator*: **GPU T4 x2** (one GPU is used), *Internet*: on.
3. *Save Version -> Save & Run All* (runs in the background, about 2 h). 4. Download `act_homebot.pt` from the *Output*
   panel (~100 MB) and evaluate it locally:
   `python scripts/eval_manip.py --policy act:act_homebot.pt --n 50` (the same E2 protocol as the expert).

The model is plain PyTorch (`homebot/act.py`); torchvision is used here only for the ImageNet ResNet-18 weights."""),
    ("code", """
import glob, os, subprocess
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone --depth 1 -b homebot-sim-final https://github.com/chutuanviet1302/HomeBot-Sim /kaggle/working/homebot-sim
%cd /kaggle/working/homebot-sim
!python -c "import torch, torchvision, cv2, pandas, pyarrow; print(torch.__version__, torchvision.__version__, cv2.__version__)"
infos = glob.glob("/kaggle/input/**/meta/info.json", recursive=True)
DATA = os.path.dirname(os.path.dirname(infos[0])) if infos else "/kaggle/working/homebot-sim/data/homebot_can"
print("dataset:", DATA)
"""),
    ("markdown", "## Train (20k steps, batch 32, fp16; checkpoint every 2000 steps)"),
    ("code", """
!python scripts/train_act.py --data {DATA} --out /kaggle/working/act_homebot.pt --pretrained     --steps 20000 --batch 32 --chunk 30 --img 224 2>&1 | grep -v Warning
"""),
    ("code", """
!ls -la /kaggle/working/act_homebot.pt
import torch
ck = torch.load("/kaggle/working/act_homebot.pt", map_location="cpu", weights_only=False)
print("steps:", ck["steps"], "| config:", ck["cfg"])
"""),
])

# Psi0's scripts/train/psi0/finetune-real-amo-psi0.sh (the 30-D state / 34-D action G1 recipe, pad 36) without RTC;
# paths, precision, batch and schedule are set in the notebook
PSI0_ARGS = " ".join("""
finetune_real_psi0_config --seed=292285 --exp=homebot --train.name=finetune --train.data_parallel=ddp
--train.learning_rate=1e-4 --train.warmup_ratio=None --train.validation_steps=100000 --train.val_num_batches=1
--train.max_grad_norm=1.0 --train.lr_scheduler_type=cosine --train.lr_scheduler_kwargs.weight_decay=1e-6
--train.lr_scheduler_kwargs.betas 0.95 0.999 --train.optimizer-foreach=False --log.report_to=tensorboard
--data.train_repo_ids=homebot_can
--data.transform.repack.pad-action-dim=36 --data.transform.repack.pad-state-dim=36
--data.transform.field.stat-path=meta/stats_psi0.json --data.transform.field.stat-action-key=action
--data.transform.field.stat-state-key=states --data.transform.field.action_norm_type=bounds
--data.transform.field.no-use-norm-mask --data.transform.field.normalize-state
--data.transform.field.pad-action-dim=36 --data.transform.field.pad-state-dim=36
--data.transform.model.img-aug --data.transform.model.resize.size 240 320
--data.transform.model.center_crop.size 240 320
--model.noise-scheduler=flow --model.train-diffusion-steps=1000 --model.n_conditions=0
--model.action-chunk-size=30 --model.action-dim=36 --model.action-exec-horizon=30 --model.observation-horizon=1
--model.odim=36 --model.view_feature_dim=2048 --model.no-tune-vlm --model.no-use_film --model.no-combined_temb
""".split())

def skip_in_batch(src):
    """Serving cells: skipped in a committed run (Save & Run All), which trains and saves the checkpoint only."""
    import textwrap
    return "if not BATCH_RUN:   # Save & Run All: train only; serve from an interactive session\n" + \
        textwrap.indent(src.strip("\n"), "    ") + "\nelse:\n    print('committed run: serving skipped')\n"


psi0 = nb([
    ("markdown", """# HomeBot-Sim: fine-tune and serve Psi0 on Kaggle T4 x2 (best effort)

[Psi0](https://github.com/physical-superintelligence-lab/Psi0) (checked against commit `cda3fdf`, 2026-10-14):
Qwen3-VL-2B (frozen) + ~500M flow-matching action expert, warm-started from the public "AMO" G1 checkpoints
(30-D state, 34-D action, padded to 36). The authors fine-tune on 8 GPUs in bf16 with flash-attn and a global batch
of 128. **This is a best-effort port to 2x T4 (15 GB each, sm75: no native bf16, no flash-attn 2): sdpa attention,
bf16 (emulated; fp16 gave a nan loss), batch 2 per GPU x 4 accumulation = 16, a schedule cut to `TRAIN_HOURS`.
Not validated by the authors on this hardware.**

**Input**: one Kaggle Dataset holding the Psi0-format HomeBot data made locally by `scripts/export_psi0.py`
(the folder with `meta/`, `data/`, `videos/`; `states` 30, `action` 34, `observation.images.egocentric` 224x224).
**Settings**: Accelerator *GPU T4 x2*, Internet *on*. Then *Run All*.

Rough timing: install ~10 min, checkpoints (6.3 GB) ~5 min, probe ~5 min (prints the measured step time and the
projected training time), training `STEPS` optimizer steps (estimate 2-4 h for 2000), then the cell *Serve* starts
the inference server (`serve_psi0_simple`, port 21074) with a public cloudflared URL for `--host` of
`scripts/eval_manip.py --policy psi0`, kept alive `SERVE_HOURS`. The checkpoint is copied to
`/kaggle/working/psi0_homebot` (~6.5 GB): *Save Version* keeps it; to serve it later, add that output as an input
and set `SERVE_ONLY` to its path."""),
    ("code", r'''
STEPS = 300             # optimizer steps (~3.5 h at bf16 on 2x T4, ~0.2 epoch; the probe can cut it to TRAIN_HOURS)
PROBE = False           # 20-step probe (+~25 min): measured 2026-10-21, bf16 ~7 s per micro-step at BATCH 2
TRAIN_HOURS = 3.5       # used only with PROBE
BATCH, ACCUM = 1, 8     # per-GPU batch: 2 peaks at 14.2 of 14.6 GiB and OOMs in training; global batch 1 x 2 x 8 = 16
HALF = "bf16"           # fp16 gives a nan loss from the first step (Qwen3-VL overflows); bf16 is emulated on T4
SERVE_ONLY = None       # None: train. "auto" / a path: serve a saved run (add this notebook's output as an input)
SERVE_HOURS = 6         # keep the inference server and the tunnel alive this long
PORT = 21074
EXEC_HORIZON, FLOW_STEPS = 20, 5   # served chunk rows / flow steps (authors 10); ~3 s per call with bf16 on a T4

import glob, json, os, re, subprocess, time
from pathlib import Path
BATCH_RUN = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"   # Save & Run All (survives a closed browser)
if SERVE_ONLY == "auto":                    # a saved run (this notebook's output) added as an input
    SERVE_ONLY = str(Path(glob.glob("/kaggle/input/**/psi0_homebot/run_config.json", recursive=True)[0]).parent)
    print("serving", SERVE_ONLY)
from pathlib import Path
BIG = Path("/tmp/psi0")                     # repo, venv, caches, runs (large, not saved)
PSI, HOME = BIG / "Psi0", BIG / "home"      # PSI_HOME must not be the repo: it has its own real/ folder
OUT = Path("/kaggle/working/psi0_homebot")  # the final checkpoint (saved with the notebook version)
CKPT = HOME / "cache/checkpoints"
ACT = f"cd {PSI} && source .venv-psi/bin/activate && "


def sh(cmd, log=None, keep=lambda l: True, every=0.0, check=True):
    """Run a bash command; print the lines `keep` selects (at most one per `every` s, errors always), log all."""
    f, tail, last = open(log, "a") if log else None, [], 0.0
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if f:
            f.write(line)
        tail = (tail + [line])[-40:]
        err = any(k in line for k in ("Error", "Traceback", "error:", "Killed", "out of memory"))
        if err or (keep(line) and time.time() - last >= every):
            print(line.rstrip()[:400], flush=True)
            last = time.time()
    p.wait()
    if f:
        f.close()
    if check and p.returncode:
        print("".join(tail))
        raise RuntimeError(f"exit code {p.returncode}")
    return p.returncode


if not SERVE_ONLY:
    infos = [i for i in glob.glob("/kaggle/input/**/meta/info.json", recursive=True)
             if "states" in json.load(open(i))["features"]]
    assert infos, "add the Psi0-format HomeBot dataset (scripts/export_psi0.py output) as an input"
    DATA = Path(infos[0]).parent.parent
    print("dataset:", DATA, {k: json.load(open(infos[0]))[k] for k in ("total_episodes", "total_frames")})
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2; df -h /tmp | tail -1")
'''),
    ("markdown", "## Install Psi0 (uv, Python 3.11, torch 2.7) and patch it for T4"),
    ("code", r'''
COMMIT = "cda3fdfafe47cb6fa75400d013de1bd36e3cea05"
sh(f"""set -e
mkdir -p {BIG} && cd {BIG}
if [ ! -d Psi0 ]; then git init -q Psi0 && cd Psi0 && git remote add origin https://github.com/physical-superintelligence-lab/Psi0 \
  && GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin {COMMIT} && GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD && cd ..; fi
pip install -q uv
# uv.lock at this commit has a duplicated [package.optional-dependencies] table (TOML error) and points at a
# mirror: resolve from PyPI instead (pyproject pins torch, transformers, accelerate, lerobot)
cd Psi0 && rm -f uv.lock && ([ -d .venv-psi ] || uv venv -q .venv-psi --python 3.11) && source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync -q --group psi --group serve --index-strategy unsafe-best-match --active
# --python: on Kaggle a bare `uv pip` targets the system Python, not the active venv
uv pip install -q --python .venv-psi/bin/python tensorboard pyarrow==17.0.0   # newer pyarrow needs numpy 2, Psi0 pins 1.x
python -c "import torch, numpy, pyarrow; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), '| numpy', numpy.__version__, '| pyarrow', pyarrow.__version__)"
""")

# T4 (sm75): no flash-attn 2 wheel and no bf16 tensor cores -> sdpa attention; with HALF = "fp16" also fp16 for the
# frozen VLM and autocast. Start from pristine files so switching HALF between runs does not stack patches.
sh(f"cd {PSI} && git checkout -q -- src/psi")

def patch(rel, old, new):
    f = PSI / rel
    s = f.read_text(encoding="utf-8")
    assert old in s or new in s, f"{rel}: '{old}' not found (Psi0 changed?)"
    f.write_text(s.replace(old, new), encoding="utf-8")

for rel in ("src/psi/trainers/finetune.py", "src/psi/trainers/qwen3vl_mixin.py"):
    patch(rel, 'attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
if HALF == "fp16":
    for rel in ("src/psi/models/psi0.py", "src/psi/trainers/finetune.py", "src/psi/trainers/qwen3vl_mixin.py"):
        patch(rel, "torch.bfloat16", "torch.float16")
    # fp16 GradScaler (never hit by the authors' bf16 runs): clip_grad_norms calls accelerator.clip_grad_norm_ once per
    # param group and each call unscales -> "unscale_() has already been called". Unscale once, clip with torch.
    F = "src/psi/trainers/finetune.py"
    patch(F, "        norms: dict[str, float] = {}\n        for group in self.optimizer.param_groups:",
          "        norms: dict[str, float] = {}\n        self.accelerator.unscale_gradients()\n"
          "        for group in self.optimizer.param_groups:  # (unscaled once above)")
    patch(F, "                self.accelerator.clip_grad_norm_(params, max_norm)",
          "                torch.nn.utils.clip_grad_norm_(params, max_norm)")
    # skipping optimizer.step() on an inf grad leaves the scaler "unscaled" (same error next step); the scaler skips
    # inf steps by itself
    patch(F, "            if not skip_step:\n                self.optimizer.step()",
          "            if not skip_step or self.accelerator.scaler is not None:\n                self.optimizer.step()")

env = {"PSI_HOME": HOME, "DATA_HOME": HOME / "data", "HF_HOME": BIG / "hf", "TORCH_HOME": BIG / "torch",
       "HF_LEROBOT_HOME": HOME / "data/lerobot", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
       "OMP_NUM_THREADS": 2, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "AV_LOG_LEVEL": "quiet",
       "NO_ALBUMENTATIONS_UPDATE": 1, "TQDM_MININTERVAL": 10}
(PSI / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))   # train.py / serve assert load_dotenv()
os.environ.update({k: str(v) for k, v in env.items()})
'''),
    ("markdown", "## Data (statistics) and the public checkpoints: pre-trained VLM + pad-36 action header"),
    ("code", r'''
def episodes_stats(root):
    """meta/episodes_stats.jsonl (LeRobot v2.1 needs it; without it LeRobot looks the dataset up on the Hub)."""
    import cv2
    import numpy as np
    import pandas as pd
    st = lambda a: {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
                    "std": a.std(0).tolist(), "count": [len(a)]}
    lines = []
    for f in sorted((root / "data/chunk-000").glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        s = {k: st(np.stack(df[k].to_numpy()).astype(np.float64)) for k in ("states", "action")}
        s.update({k: st(df[k].to_numpy(np.float64)[:, None])
                  for k in ("timestamp", "frame_index", "episode_index", "index", "task_index", "next.done")})
        cap, frames = cv2.VideoCapture(str(root / f"videos/chunk-000/observation.images.egocentric/{f.stem}.mp4")), []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im[..., ::-1])
        px = np.stack(frames[::10]).reshape(-1, 3) / 255.0          # per channel, in [0, 1], shape (3, 1, 1)
        s["observation.images.egocentric"] = {**{k: np.reshape(v, (3, 1, 1)).tolist() for k, v in st(px).items()
                                                 if k != "count"}, "count": [len(frames[::10])]}
        lines.append(json.dumps({"episode_index": int(df["episode_index"].iloc[0]), "stats": s}))
    (root / "meta/episodes_stats.jsonl").write_text("".join(l + chr(10) for l in lines))
    print(len(lines), "episodes -> meta/episodes_stats.jsonl")


if not SERVE_ONLY:
    dst = HOME / "data/real/homebot_can"
    sh(f"rm -rf {dst} && mkdir -p {dst.parent} && cp -r {DATA} {dst}")
    if not (dst / "meta/episodes_stats.jsonl").exists():
        episodes_stats(dst)
    sh(ACT + f"python scripts/data/calc_modality_stats.py --work-dir={dst.parent} --task=homebot_can"
       f" && cp {dst}/meta/stats.json {dst}/meta/stats_psi0.json"
       f" && python scripts/data/patch_lerobot_meta.py {dst} > /dev/null", keep=lambda l: "write to" in l)
    sh(ACT + f"""python -c "from huggingface_hub import snapshot_download as s; s('USC-PSI-Lab/psi-model', \
local_dir='{CKPT}', allow_patterns=['psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/*', \
'psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/*'])" && du -sh {CKPT}/psi0/*""", keep=lambda l: "psi0/" in l)
'''),
    ("markdown", "## Probe: 20 steps (memory, speed, finite loss)"),
    ("code", f'''
ARGS = "{PSI0_ARGS}"
''' + r'''
def train_cmd(steps, accum, extra=""):
    return (ACT + f"CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 scripts/train.py {ARGS}"
            f" --train.mixed_precision={HALF} --train.train_batch_size={BATCH} --train.gradient_accumulation_steps={accum}"
            f" --train.max_training_steps={steps} --train.num_workers=2 --data.root_dir={HOME}/data/real"
            f" --model.model_name_or_path={CKPT}/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
            f" --model.pretrained-action-header-path={CKPT}/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k {extra}")

def gpu_peak(fn):
    mon = subprocess.Popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > /tmp/mem.csv",
                           shell=True)
    try:
        fn()
    finally:
        mon.terminate()
    m = [int(x) for x in open("/tmp/mem.csv").read().split() if x.isdigit()]
    return max(m) if m else 0

is_loss = lambda l: "loss" in l.lower()
if not SERVE_ONLY and PROBE:
    log = "/kaggle/working/probe.log"
    open(log, "w").close()                  # sh() appends; a nan from an earlier run must not fail this one
    t0 = time.time()
    peak = gpu_peak(lambda: sh(train_cmd(20, 1, "--train.warmup_steps=0 --train.checkpointing_steps=100000"),
                               log, keep=is_loss, every=5))
    text = open(log).read()
    # seconds per micro-step: median gap of tqdm's elapsed time between consecutive steps (its s/it is smoothed and
    # dominated by the ~10 min first step; the last step includes saving and teardown)
    ticks = dict((int(n), 60 * int(m) + int(s)) for n, m, s in re.findall(r"\| (\d+)/\d+ \[(\d+):(\d+)<", text))
    gaps = sorted(ticks[n + 1] - ticks[n] for n in ticks if n + 1 in ticks)
    rates = [gaps[len(gaps) // 2]] if gaps else []
    print(f"probe: {time.time() - t0:.0f} s, peak GPU memory {peak} MiB of ~15100")
    assert "nan" not in " ".join(l for l in text.splitlines() if is_loss(l)).lower(), "nan loss: try HALF = 'bf16'"
    if rates:
        s_it = rates[0]
        STEPS = max(50, min(STEPS, int(TRAIN_HOURS * 3600 / (s_it * ACCUM))))
        print(f"~{s_it:.2f} s per micro-step -> STEPS = {STEPS} (x {ACCUM} accumulation, global batch "
              f"{BATCH * 2 * ACCUM}, ~{STEPS * BATCH * 2 * ACCUM / 22087:.1f} epochs) ~ {s_it * ACCUM * STEPS / 3600:.1f} h")
    sh(f"rm -rf {PSI}/.runs/finetune")      # drop the probe run
'''),
    ("markdown", "## Fine-tune"),
    ("code", r'''
if not SERVE_ONLY:
    sh(train_cmd(STEPS, ACCUM, f"--train.warmup_steps={STEPS // 20} --train.checkpointing_steps={max(STEPS // 4, 1)}"
                 " --train.max_checkpoints_to_keep=1"), "/kaggle/working/train.log", keep=is_loss, every=120)
    run = max(Path(PSI, ".runs/finetune").iterdir(), key=lambda p: p.stat().st_mtime)
    step = max(int(c.name.split("_")[-1]) for c in (run / "checkpoints").glob("ckpt_*"))
    # what serve_psi0_simple needs: argv.txt, run_config.json (+ stats json) and checkpoints/ckpt_N/model.safetensors
    sh(f"rm -rf {OUT} && mkdir -p {OUT}/checkpoints/ckpt_{step} && cp {run}/*.json {run}/*.txt {OUT}/ "
       f"&& cp {run}/checkpoints/ckpt_{step}/model.safetensors {OUT}/checkpoints/ckpt_{step}/ && du -sh {OUT}")
'''),
    ("markdown", "## Serve: `serve_psi0_simple` on GPU 0 + public URL (cloudflared quick tunnel)"),
    ("code", skip_in_batch(r'''
import base64, urllib.request
import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr
RUN = Path(SERVE_ONLY or OUT)
if str(RUN).startswith("/kaggle/input"):     # mmap page reads over the dataset mount stall loading for > 45 min;
    sh(f"rm -rf /tmp/run && mkdir -p /tmp/run && cp -r {RUN}/. /tmp/run/")   # a sequential copy takes ~2 min
    RUN = Path("/tmp/run")
STEP =max(int(c.name.split("_")[-1]) for c in (RUN / "checkpoints").glob("ckpt_*"))
srv = subprocess.Popen(ACT + f"CUDA_VISIBLE_DEVICES=0 serve_psi0_simple --host 0.0.0.0 --port {PORT} --policy psi0"
                       f" --run-dir={RUN} --ckpt-step={STEP} --action-exec-horizon={EXEC_HORIZON}"
                       f" --num-inference-steps={FLOW_STEPS}"
                       " > /kaggle/working/serve.log 2>&1", shell=True, executable="/bin/bash")
health = lambda: urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5).status == 200
t0 = time.time()                             # loading 6 GB from a mounted dataset takes 10-20 min
while True:
    try:
        if health():
            break
    except Exception:
        pass
    assert srv.poll() is None, open("/kaggle/working/serve.log").read()[-3000:]
    assert time.time() - t0 < 1800, "server not up after 30 min, see /kaggle/working/serve.log"
    time.sleep(10)
print(f"server up after {time.time() - t0:.0f} s:", RUN, "step", STEP)

# self-test with the client's wire format (homebot/policy.py psi0_client)
enc = lambda a: {"__numpy__": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(),
                 "dtype": dtype_to_descr(a.dtype), "shape": list(a.shape)}
msg = {"image": {"observation.images.egocentric": enc(np.zeros((224, 224, 3), np.uint8))},
       "instruction": "pick up the can from the table", "history": {"client_id": "selftest"},
       "state": {"states": enc(np.zeros(30, np.float32))}, "condition": {}, "gt_action": None,
       "dataset_name": "real", "timestamp": ""}
t0 = time.time()
req = urllib.request.Request(f"http://localhost:{PORT}/act", json.dumps(msg).encode(), {"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(req, timeout=300))
assert isinstance(out, dict) and "action" in out, f"server error: {out} (see /kaggle/working/serve.log)"
print(f"/act: {time.time() - t0:.2f} s, action {out['action']['shape']} (exec horizon x 36)")

sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared"
   " && chmod +x /tmp/cloudflared")
tun = subprocess.Popen(f"/tmp/cloudflared tunnel --no-autoupdate --url http://localhost:{PORT} > /tmp/tunnel.log 2>&1",
                       shell=True)
URL = None
for _ in range(60):
    log = Path("/tmp/tunnel.log")              # created by the shell a moment after Popen returns
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read_text() if log.exists() else "")
    if m:
        URL = m.group(0)
        break
    time.sleep(2)
print("PUBLIC URL:", URL)
print(f"local: python scripts/eval_manip.py --policy psi0 --host {URL} --n 20 --workers 2")
''')),
    ("code", skip_in_batch(r'''
# keep the session (server + tunnel) alive; interrupt to stop
for i in range(int(SERVE_HOURS * 6)):
    if srv.poll() is not None or tun.poll() is not None:
        print("server or tunnel stopped"); sh("tail -20 /kaggle/working/serve.log /tmp/tunnel.log", check=False); break
    n = open("/kaggle/working/serve.log").read().count("Return Action")
    print(time.strftime("%H:%M"), "alive,", n, "requests served", URL, flush=True)
    time.sleep(600)
''')),
])

ev = nb([
    ("markdown", "# HomeBot-Sim: headless E2 eval (MuJoCo, EGL) for ACT / N1.7"),
    COMMON,
    ("code", """
%env MUJOCO_GL=egl
!pip install -q mujoco==3.11.0 scipy pandas pyarrow imageio imageio-ffmpeg opencv-python-headless
!git clone --depth 1 https://github.com/<your-user>/homebot-sim /kaggle/working/homebot-sim
%cd /kaggle/working/homebot-sim
!bash scripts/setup_assets.sh        # Menagerie G1, Inspire hand meshes, YCB soup can (~230 MB)
"""),
    ("markdown", "N1.7: start the Isaac-GR00T inference server on GPU 0 in the background, then evaluate."),
    ("code", """
import subprocess
srv = subprocess.Popen("cd /kaggle/working/Isaac-GR00T && python scripts/inference_service.py --server "
                       "--model-path /kaggle/working/n17_homebot --embodiment-tag new_embodiment "
                       "--data-config /kaggle/working/homebot-sim/configs/gr00t_g1_sonic_inspire.py:make_config",
                       shell=True)
!python scripts/eval_manip.py --n 50 --policy gr00t --host localhost
!python scripts/eval_manip.py --n 50 --policy torch:/kaggle/working/act_homebot/act_traced.pt
!python scripts/plots.py
"""),
])

for name, n in (("kaggle_finetune_gr00t", gr00t), ("kaggle_train_act", act), ("kaggle_finetune_psi0", psi0),
                ("kaggle_eval", ev)):
    (HERE / f"{name}.ipynb").write_text(json.dumps(n, indent=1), encoding="utf8")
print("ok")
