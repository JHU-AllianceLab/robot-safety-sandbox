#!/bin/bash
# What do the handover pilots cost on the push bench? (the user, 2026-11-18, after the
# training-rollout failure rates rose 15% -> 36% / 57% with the handover-jump penalty).
# Each fine-tuned thermal70 SHAC fallback (runs/shac_thermal70/handover_pilot, 150M steps
# from shac_s0) on cross_eval.sh's 1024 kicks (kick at 0.02 s, |dv| ~ U(0, 4) m/s):
#   sonic      the anchor (policy-independent; same across the three tasks)
#   fallback   SHAC alone, deterministic: the plain safety cost of the penalty
#   rollout:1  SONIC under the exact-sim 1 s switch: real SONIC -> fallback handovers;
#              intervention dev_max_mean_post ~ ||pi - u_SONIC|| at the switch (the jump)
# Array task N: 1 = control (c=1e-6), 2 = c=0.01, 3 = c=0.05.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,h100,a100,h200 --array=1-3 --gres=gpu:1 --cpus-per-task=32 --mem=96G --time=00:45:00 --job-name=handover_eval --output=logs/slurm/%x_%A_%a.out configs/sweeps/shac_thermal70/handover_eval.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
N=${SLURM_ARRAY_TASK_ID:?an array task}
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) task $N"
C=(x c0p000001 c0p01 c0p05); C=${C[$N]}
P=runs/shac_thermal70/handover_pilot/${C}_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
[[ -f $P ]] || { echo "[handover-eval] FAILED: no $P"; exit 1; }
python examples/eval_sonic_qcbf.py --policy $P --tag handover_$C --sonic-dir $HOME/gear_sonic_hf \
  --task g1_recover_lafan_thermal70 --hip mjlab --suites push --episodes 1024 --num-envs 256 \
  --onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint \
  --no-obs-noise --rollout-workers 30 --conditions sonic fallback rollout:1 \
  --out runs/shac_thermal70/handover_pilot/eval/$C
echo "[job] handover eval $C done"
