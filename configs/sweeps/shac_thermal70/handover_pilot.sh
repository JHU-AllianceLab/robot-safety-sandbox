#!/bin/bash
# HANDOVER-SMOOTHNESS PILOT (the user's request, 2026-11-18): fine-tune the trained 70 C
# margin SHAC fallback (thermal70 seed 0, 1.5B steps) with the new handover term,
# safety_sb3 SHAC handover_rate_coef: + c * mean ||mu(s0) - clamp(a_SONIC_prev(s0))||^2
# over replay EPISODE-START states -- every g1_recover episode starts at a handover
# row of the bank, whose last action is SONIC's last applied one -- on top of the
# unchanged recipe (action_rate 0.05 / action_mag 0.005). Warm start: actor + critic +
# world model + obs stats from the checkpoint; fresh replay, so 100 model-only and 200
# critic-only iterations first; low LR (5e-5, cosine to 3% over the run) so the trained
# policy is nudged, not retrained. train/handover_rate_loss starts at the trained
# policy's handover jump^2 (before the actor moves) -- the baseline.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p interactive_gpu --gres=gpu:1 --cpus-per-task=16 --time=03:55:00 --job-name=handover_pilot --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70/handover_pilot.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
C=${HANDOVER_COEF:-0.2}
STEPS=${PILOT_STEPS:-300000000}
ITERS=$(( STEPS / 10240 ))     # 4096 envs x 2.5 steps per update
OUT=runs/shac_thermal70/handover_pilot/c${C/./p}_s0
python examples/train.py --config configs/g1_recover_sonic_thermal70_safetyshac1p.yaml --seed 0 \
  --load $P --handover-rate-coef $C --handover-batch 1024 \
  --steps $STEPS --lr 5e-5 --lr-decay-iters $ITERS --wm-warmup-iters 100 --actor-delay-iters 200 \
  --checkpoint-every 50000000 --video-interval 0 --no-wandb --out $OUT
echo "[job] handover_pilot done"
