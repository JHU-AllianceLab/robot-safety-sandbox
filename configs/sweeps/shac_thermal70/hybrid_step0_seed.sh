#!/bin/bash
# Array task N (1..3): SHAC seed N-1 of hybrid_step0_jobs.txt -- nn calibration, then for
# hybrid and hybrid-temps: calibration, conformal table, eval (jobs 8N-7 .. 8N); the seed
# that finishes last writes the comparison (job 25). See hybrid_step0_jobs.txt for the
# sbatch line. Conda env mjlab316 (MuJoCo 3.16).
#SBATCH --gres=gpu:1
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
J=configs/sweeps/shac_thermal70/hybrid_step0_jobs.txt
N=${SLURM_ARRAY_TASK_ID:?an array task}
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) seed $((N - 1))"
for j in $(seq $((8 * N - 7)) $((8 * N))); do
  examples/run_jobs.sh $J --only $j || { echo "[job] job $j FAILED"; exit 1; }
done
echo "[job] seed $((N - 1)) done"
R=runs/shac_thermal70/hybrid_step0/eval
if ls $R/hybrid/thermal70_s{0,1,2}/push.pt $R/hybrid-temps/thermal70_s{0,1,2}/push.pt >/dev/null 2>&1; then
  examples/run_jobs.sh $J --only 25 --force
fi
