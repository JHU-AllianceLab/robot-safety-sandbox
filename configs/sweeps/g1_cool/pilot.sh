#!/bin/bash
# COOLING-POLICY PILOT (the user's request, 2026-11-18; configs/g1_cool_thermal70_ppo.yaml):
# smoke checks of the task (examples/check_g1_cool.py: spawn rows, hot start temperatures,
# handover action seeding, reward terms, pushes), then one PPO seed for up to ~3.7 h.
# Checkpoints (+ vecnormalize) every 10M steps under runs/g1_cool/pilot_s0.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p a100,h100,h200,rtx6000 --gres=gpu:1 --cpus-per-task=16 --time=04:00:00 --job-name=g1_cool_pilot --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/pilot.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ -f data/g1_cool_handover_bank_dv1p5.pt ]] || python examples/build_cool_bank.py
python examples/check_g1_cool.py --num-envs 64 --steps 300 --device cuda:0
python examples/train.py --config configs/g1_cool_thermal70_ppo.yaml --seed 0 \
  --steps ${PILOT_STEPS:-500000000} --out runs/g1_cool/pilot_s0 --no-wandb
echo "[job] g1_cool pilot done"
