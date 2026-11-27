#!/bin/bash
# The v0.6.1 filter study's trainings (the user, 2026-11-23), one model per job, seed $SEED (default 0):
#   ARM=beatit_noheight     Beat-It, temperature-agnostic   configs/g1_recover_noheight_v6_safetyshac1p.yaml       ~15 h
#   ARM=safetyppo_noheight  SafetyPPO, temperature-agnostic configs/g1_recover_noheight_v6_safetyppo1p.yaml        ~5 h
#   ARM=beatit_thermal      Beat-It thermal on v0.6.1       configs/g1_recover_thermal70_v61_hybrid_safetyshac1p.yaml ~15.5 h
# (RL-DH, temperature-agnostic: configs/sweeps/v061_study/rldh_train.sh; lambda-Reach, temperature-agnostic:
# reused, runs/thermal65/lambdareach_s0.) SMOKE=1: 30M steps into runs/v061_study/<arm>_smoke (throwaway).
#   ARM=beatit_noheight sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h100,l40s --gres=gpu:1 \
#     --cpus-per-task=16 --mem=96G --time=24:00:00 --job-name=v061_<arm> --output=logs/slurm/%x_%j.out \
#     --export=ALL configs/sweeps/v061_study/train.sh
# The trainers cannot resume: after a node failure, resubmit (checkpoints every 50M steps stay in the run dir).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
case "${ARM:?ARM=beatit_noheight|safetyppo_noheight|beatit_thermal}" in
  beatit_noheight)    CFG=configs/g1_recover_noheight_v6_safetyshac1p.yaml;            SSB=$HOME/ssb-wt-hybrid ;;
  safetyppo_noheight) CFG=configs/g1_recover_noheight_v6_safetyppo1p.yaml;             SSB=$HOME/safety-stable-baselines-baselines ;;
  beatit_thermal)     CFG=configs/g1_recover_thermal70_v61_hybrid_safetyshac1p.yaml;   SSB=$HOME/ssb-wt-hybrid ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$SSB${PYTHONPATH:+:$PYTHONPATH}
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
SEED=${SEED:-0}
echo "[slurm] $(hostname) job ${SLURM_JOB_ID:-local} arm $ARM seed $SEED smoke ${SMOKE:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import safety_sb3; print('[train] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q "$(basename $SSB)"
if [ -n "${SMOKE:-}" ]; then OUT=runs/v061_study/${ARM}_smoke; EXTRA="--steps 30000000"; rm -rf $OUT
else OUT=runs/v061_study/${ARM}_s${SEED}; EXTRA=""; fi
mkdir -p $OUT
# the MuJoCo-Warp solver-iteration warnings are ~750 MB per run: keep them out of the log
python examples/train.py --config $CFG --seed $SEED --out $OUT --no-wandb $EXTRA 2>&1 \
  | grep -v --line-buffered -E "linesearch|solver iterations limit|warn_overflow"
ls -la $OUT/*/ 2>/dev/null | head
