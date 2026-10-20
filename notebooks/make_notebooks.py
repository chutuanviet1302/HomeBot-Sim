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
    ("markdown", "# HomeBot-Sim: ACT baseline (LeRobot) on 1x T4"),
    COMMON,
    ("code", """
!pip install -q lerobot
!git clone --depth 1 https://github.com/<your-user>/homebot-sim /kaggle/working/homebot-sim
!lerobot-train --config_path /kaggle/working/homebot-sim/configs/act.yaml
!huggingface-cli upload --private <your-user>/homebot-act /kaggle/working/act_homebot .
"""),
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

for name, n in (("kaggle_finetune_gr00t", gr00t), ("kaggle_train_act", act), ("kaggle_eval", ev)):
    (HERE / f"{name}.ipynb").write_text(json.dumps(n, indent=1), encoding="utf8")
print("ok")
