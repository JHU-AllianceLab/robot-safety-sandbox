#!/bin/bash
# Array task N (1..3): SHAC seed N-1 of wmreport_jobs.txt -- its calibration, tube fit and
# evaluation, in order (jobs 3N-2 .. 3N); the seed that finishes last also writes the report
# (job 10). See wmreport_jobs.txt for the sbatch line. Conda env mjlab316 (MuJoCo 3.16).
#SBATCH --gres=gpu:1
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
N=${SLURM_ARRAY_TASK_ID:?an array task}
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) seed $((N - 1))"
for j in $((3 * N - 2)) $((3 * N - 1)) $((3 * N)); do
  examples/run_jobs.sh configs/sweeps/shac_thermal/wmreport_jobs.txt --only $j || { echo "[job] job $j FAILED"; exit 1; }
done
echo "[job] seed $((N - 1)) done"
if ls runs/shac_thermal/wmreport/eval/thermal_s{0,1,2}/push.pt >/dev/null 2>&1; then
  examples/run_jobs.sh configs/sweeps/shac_thermal/wmreport_jobs.txt --only 10 --force
fi
