#!/bin/bash
#SBATCH --job-name=t65_lambdareach
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 65 C comparison, arm 3's fallback: lambda-Reach PPO (no temperatures, falls only) in the v6 world
# (configs/g1_recover_noheight_v6_lambdareach1p.yaml). SMOKE=1: 30M steps into runs/thermal65/
# lambdareach_smoke (throwaway) to check the wiring; otherwise 1.5B steps (~5-6 h) into
# runs/thermal65/lambdareach_s$SEED.
#   sbatch configs/sweeps/thermal65/lambdareach_train.sh        [SEED=0 SMOKE=1]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/safety-stable-baselines-baselines${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
SEED=${SEED:-0}
echo "[slurm] $(hostname) job ${SLURM_JOB_ID} seed ${SEED} smoke ${SMOKE:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import safety_sb3; print('[train] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q safety-stable-baselines-baselines
if [ -n "${SMOKE:-}" ]; then OUT=runs/thermal65/lambdareach_smoke; EXTRA="--steps 30000000"; rm -rf $OUT
else OUT=runs/thermal65/lambdareach_s${SEED}; EXTRA=""; fi
mkdir -p $OUT
python examples/train.py --config configs/g1_recover_noheight_v6_lambdareach1p.yaml \
  --seed $SEED --out $OUT --no-wandb $EXTRA 2>&1 | grep -v --line-buffered -E "linesearch|solver iterations limit|warn_overflow"
ls -la $OUT/*/ 2>/dev/null | head
echo "[train] done $(date -Is)"
