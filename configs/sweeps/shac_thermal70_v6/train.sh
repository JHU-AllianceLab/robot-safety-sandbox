#!/bin/bash
# Hybrid Beat-It on sysID v0.6, step 3: the training (1.5B steps, one seed per job).
#   SEED=0 sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h100,l40s --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=24:00:00 --job-name=v6_hybrid --output=logs/slurm/%x_%j.out --export=ALL configs/sweeps/shac_thermal70_v6/train.sh
# Smoke 1998752 (L40S): 25.8k env-steps/s at 4096 envs (old non-hybrid thermal70 ~31k there,
# 33.7k on RTX PRO 6000) -> ~16 h on L40S, ~14-15 h on RTX PRO 6000. The trainer cannot resume:
# after a node failure, resubmit (checkpoints every 50M steps stay in the run dir).
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
SEED=${SEED:-0}
echo "[slurm] $(hostname) job ${SLURM_JOB_ID} seed ${SEED} $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import safety_sb3; print('[train] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q ssb-wt-hybrid
OUT=runs/shac_thermal70_v6/hybrid_s${SEED}
mkdir -p $OUT
# the MuJoCo-Warp solver-iteration warnings are ~750 MB per run: keep them out of the log
python examples/train.py --config configs/g1_recover_sonic_thermal70_v6_hybrid_safetyshac1p.yaml \
  --seed $SEED --out $OUT --no-wandb 2>&1 | grep -v --line-buffered -E "linesearch|solver iterations limit|warn_overflow"
