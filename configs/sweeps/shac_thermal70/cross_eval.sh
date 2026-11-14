#!/bin/bash
# Cross-evaluation of the two 70 C fallbacks: each arm's SHAC policy in the OTHER arm's env,
# on the report's 1024 kicks per seed (same flags as wmreport_jobs.txt's eval lines, so the
# same episodes and start temperatures as the two reports). Array task N (1..6):
#   1-3: the barrier-trained SHAC (runs/shac_thermal_cbf, seed N-1) on g1_recover_lafan_thermal70
#        (no barrier; 70 C in the margin and the termination) -- does it overheat without
#        the barrier under it?
#   4-6: the margin-trained SHAC (runs/shac_thermal70, seed N-4) on g1_recover_lafan_thermal_cbf
#        (the 70 C barrier under every controller; fall margin only) -- how much does the
#        barrier have to clip it?
# Conditions: sonic (an anchor: policy-independent, so it should match the arm's report up to
# the simulator's non-determinism -- 6-13 of 1024 episodes flip between identical runs, net
# <= 0.5 points), fallback (SHAC alone) and
# rollout:1 (the exact-sim 1 s filter, which needs no world-model calibration).
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100,a100,h200 --array=1-6 --gres=gpu:1 --cpus-per-task=32 --mem=96G --time=00:45:00 --job-name=thermal_cross --output=logs/slurm/%x_%A_%a.out configs/sweeps/shac_thermal70/cross_eval.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
N=${SLURM_ARRAY_TASK_ID:?an array task}
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) task $N"

if (( N <= 3 )); then
  S=$((N - 1)); P=runs/shac_thermal_cbf/shac_s$S/g1_recover_thermal_cbf_safetyshac1p/final_model.zip
  TASK=g1_recover_lafan_thermal70; TAG=barrierpi_on_margin_s$S
else
  S=$((N - 4)); P=runs/shac_thermal70/shac_s$S/g1_recover_thermal70_safetyshac1p/final_model.zip
  TASK=g1_recover_lafan_thermal_cbf; TAG=marginpi_on_barrier_s$S
fi
[[ -f $P ]] || { echo "[cross] FAILED: no $P"; exit 1; }
python examples/eval_sonic_qcbf.py --policy $P --tag $TAG --sonic-dir $HOME/gear_sonic_hf --task $TASK \
  --hip mjlab --suites push --episodes 1024 --num-envs 256 --onset-s 0.02 0.02 --dv-range 0 4 \
  --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise --rollout-workers 30 \
  --conditions sonic fallback rollout:1 --out runs/shac_thermal_cross/eval/$TAG
echo "[job] cross $TAG done"
