#!/bin/bash
# Smoke of hybrid_step0_jobs.txt on SHAC seed 0, small: hybrid calibration (256 episodes, with
# the heat model's one-step check against the simulator), its table, a 256-episode eval of
# sonic + wmmin:1:0.95; and a 128-episode hybrid-temps calibration (its code path).
#   sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=00:45:00 --job-name=hybrid_step0_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70/hybrid_step0_smoke.sh
#SBATCH --gres=gpu:1
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
R=runs/shac_thermal70/hybrid_step0/smoke
T=thermal70_s0
BASE="python examples/eval_sonic_qcbf.py --policy runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip --tag $T --sonic-dir $HOME/gear_sonic_hf --task g1_recover_lafan_thermal70 --hip mjlab --suites push --num-envs 128 --onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise"
$BASE --episodes 256 --seed 1 --conditions sonic wmcalib:clean --wm-thermal hybrid --thermal-diag --out $R/calib/hybrid/$T
python examples/calibrate_wm_min_tube.py --calib $R/calib/hybrid/$T/push.pt --tag $T
$BASE --episodes 256 --conditions sonic wmmin:1:0.95 --wm-thermal hybrid --tube-calib $R/calib/hybrid/$T/min_tube.json --out $R/eval/hybrid/$T
$BASE --episodes 128 --seed 1 --conditions sonic wmcalib:clean --wm-thermal hybrid-temps --out $R/calib/hybrid-temps/$T
python examples/calibrate_wm_min_tube.py --calib $R/calib/hybrid-temps/$T/push.pt --tag $T
echo "[smoke] PASSED"
