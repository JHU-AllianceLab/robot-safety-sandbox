#!/bin/bash
# Cooling curves (the user, 2026-11-18): the first cooling policy (pilot_s0, 500M) and the
# thermal70 SHAC fallback, alone and unpushed for 10 min from all four ankle windings at
# 65 C (housing 35 C "fresh" / 58 C "soaked"), on the same starts; plus heat-model
# references (passive, and each policy's late heat held out to 4 h). examples/cool_curve.py.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,h100,a100,h200 --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --job-name=cool_curve --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/cool_curve.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export PYTHONFAULTHANDLER=1
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
POL=(--policy cooling runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip
     --policy fallback runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip)
python examples/cool_curve.py "${POL[@]}" --num-envs 16 --duration-s 5 --extrapolate-s 60 \
  --out runs/g1_cool/cool_curve/smoke
python examples/cool_curve.py "${POL[@]}" --num-envs 512 --duration-s 600 --out runs/g1_cool/cool_curve
echo "[job] cool_curve done"
