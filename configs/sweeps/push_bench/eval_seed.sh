#!/bin/bash
# Array task N (1..3): seed N-1 of eval_jobs.txt -- SHAC's calibration, tube and evaluations, then
# SafePPO's evaluations, in order (jobs 6N-5 .. 6N). See eval_jobs.txt for the sbatch line.
#SBATCH --gres=gpu:1
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
N=${SLURM_ARRAY_TASK_ID:?an array task}
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) seed $((N - 1))"
for j in $(seq $((6 * N - 5)) $((6 * N))); do
  examples/run_jobs.sh configs/sweeps/push_bench/eval_jobs.txt --only $j || { echo "[job] job $j FAILED"; exit 1; }
done
echo "[job] seed $((N - 1)) done"
