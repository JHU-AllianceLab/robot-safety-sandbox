#!/bin/bash
# Fallback -> cooling switch (the user, 2026-11-19: "run fallback policy for awhile and transit
# to this cooling policy"): from a SONIC handover with all four ankle windings at 65 C (housing
# 35 C), the thermal70 SHAC fallback acts for SWITCH_S (10 s), then the first cooling policy
# (pilot_s0, 500M); the task's minor pushes on. examples/cool_video.py:
#   1. smoke (2 envs, 2 s, switch at 1 s);
#   2. video of 4 robots + stats over 512 envs (same run) -> runs/g1_cool/video/switch<S>s_seed0*;
#   3. references on the same seed, stats only: fallback throughout, cooling from the handover
#      -> runs/g1_cool/switch/{fallback_only,cooling_only}_seed0*.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,a100,h100,h200 --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --job-name=cool_switch --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/cool_switch.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
COOL=runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip
FB=runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
S=${SWITCH_S:-10}
SEED=${SEED:-0}
python examples/cool_video.py --policy $COOL --fallback $FB --switch-s 1 --duration-s 2 \
  --panels 2 --num-envs 8 --hold-s 0.2 --out runs/g1_cool/video/switch_smoke.mp4
python examples/cool_video.py --policy $COOL --fallback $FB --switch-s $S --duration-s 30 \
  --panels 4 --num-envs 512 --seed $SEED --out runs/g1_cool/video/switch${S}s_seed${SEED}.mp4
python examples/cool_video.py --policy $COOL --fallback $FB --switch-s 1000 --duration-s 30 \
  --num-envs 512 --no-video --seed $SEED --out runs/g1_cool/switch/fallback_only_seed${SEED}
python examples/cool_video.py --policy $COOL --duration-s 30 \
  --num-envs 512 --no-video --seed $SEED --out runs/g1_cool/switch/cooling_only_seed${SEED}
echo "[job] cool_switch done"
