#!/bin/bash
#SBATCH --job-name=v061_rldh
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# RL-DH (learned half-space filter of SONIC, paper reward) on the identified G1 without
# temperatures (configs/g1_recover_noheight_v6_rldh1p.yaml: task g1_recover_noheight_v6, the v6
# SONIC handover bank, SONIC on the robot's own hip pitch). SMOKE=1: 30M steps into
# runs/v061_study/rldh_smoke (throwaway) to check the wiring; otherwise 1.5B steps into
# runs/v061_study/rldh_noheight_s$SEED. rtx6000 / h100 only: an earlier RL-DH run failed at
# startup on a100 / l40s in SonicReference.build's reference-vs-library check (since relaxed).
#   sbatch configs/sweeps/v061_study/rldh_train.sh        [SEED=0 SMOKE=1]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/safety-stable-baselines-baselines${PYTHONPATH:+:$PYTHONPATH}
export SONIC_DIR=${SONIC_DIR:-$HOME/gear_sonic_hf}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
SEED=${SEED:-0}
echo "[slurm] $(hostname) job ${SLURM_JOB_ID} seed ${SEED} smoke ${SMOKE:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# (not `tee /dev/stderr`: with stderr in the log file it reopens the file and truncates it)
SSB=$(python -c "import safety_sb3; assert hasattr(safety_sb3, 'HyperplanePolicy'); print(safety_sb3.__file__)")
echo "[train] safety_sb3 from $SSB"; [[ $SSB == *safety-stable-baselines-baselines* ]]
if [ -n "${SMOKE:-}" ]; then OUT=runs/v061_study/rldh_smoke; EXTRA="--steps 30000000"; rm -rf $OUT
else OUT=runs/v061_study/rldh_noheight_s${SEED}; EXTRA=""; fi
mkdir -p $OUT
python examples/train.py --config configs/g1_recover_noheight_v6_rldh1p.yaml \
  --seed $SEED --out $OUT --sonic-dir $SONIC_DIR --no-wandb $EXTRA 2>&1 \
  | grep -v --line-buffered -E "linesearch|solver iterations limit|warn_overflow"
ls -la $OUT/*/ 2>/dev/null | head
echo "[train] done $(date -Is)"
