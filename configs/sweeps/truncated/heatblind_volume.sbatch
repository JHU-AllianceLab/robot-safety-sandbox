#!/bin/bash
#SBATCH --job-name=trunc_volume
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,b200,a100
#SBATCH --exclude=gh123,gh204,gh205
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Truncated Beat-It monitor (k imagined steps + a terminal value) vs the full 1 s monitor: calibrated
# safe-set volume and soundness like the v0.6.1 study's baselines (the user, 2026-11-26).
# Heat-blind Beat-It (runs/v061_study/beatit_noheight_s0), temperature-agnostic world g1_recover_noheight_v6.
#   1. label_start_states --dump-wm (records each episode's world-model inputs):
#        TEST  the held-out dv 3.5 m/s bank, K 4 (= the study's nt labelling, configs/sweeps/v061_study/eval.sh)
#        TRAIN the SONIC push training bank, K 1, MAX_ROWS rows (the terminal value's starts)
#   2. examples/truncated_monitor_volume.py: terminal value on TRAIN, every k scored on TEST.
#   sbatch configs/sweeps/truncated/heatblind_volume.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
E=runs/v061_study/eval; O=${OUT:-$E/truncated}; mkdir -p $O
POL=$(ls runs/v061_study/beatit_noheight_s0/*/final_model.zip | head -1)
TUBE=(--wm-tube wmmin:1:0.95 --tube-calib $E/calib/beatit_noheight/min_tube.json --tube-tag beatit_noheight_s0)
COMMON=(--policy $POL --task g1_recover_noheight_v6 --no-push --no-obs-noise "${TUBE[@]}" --dump-wm)
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is) sandbox $(git rev-parse --short HEAD) policy $POL"
[ -f $O/test_nt.episodes.pt ] || python examples/label_start_states.py "${COMMON[@]}" \
  --bank data/g1_sonic_push_gfall_sysid316v6_dv35_test_bank_net.pt --k 4 --out $O/test_nt.pt > $O/test_nt.log 2>&1 &
[ -f $O/train_nt.episodes.pt ] || python examples/label_start_states.py "${COMMON[@]}" \
  --bank data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt --k 1 --max-rows ${MAX_ROWS:-64000} --seed 7 \
  --out $O/train_nt.pt > $O/train_nt.log 2>&1 &
wait
for f in test_nt train_nt; do tail -2 $O/$f.log | cut -c1-250; [ -f $O/$f.episodes.pt ] || { echo "labelling $f FAILED"; tail -30 $O/$f.log; exit 1; }; done
python examples/truncated_monitor_volume.py --train $O/train_nt.episodes.pt --test $O/test_nt.episodes.pt --out $O ${EXTRA:-}
rc=$?; echo "done $(date -Is) rc $rc"; exit $rc
