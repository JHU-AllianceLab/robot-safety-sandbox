#!/bin/bash
# Run a jobs file from examples/make_sweep.py as a SLURM array: array task N
# runs job N (examples/run_jobs.sh --only N), one GPU each. Submit from the
# repository root; the account and partition go on the command line:
#
#   mkdir -p logs/slurm
#   sbatch -A hhu49_alliance1 -p rtx6000 --array=1-20 \
#       examples/slurm_jobs.sbatch configs/sweeps/lafan_recovery/jobs.txt
#
# A job whose outputs exist is skipped, so resubmitting the same array after a
# failure or a preemption reruns only what is missing (from scratch: the
# trainer does not resume). Per-job logs: logs/<jobs dir>_<jobs file>/.
#
#SBATCH --job-name=lafan_recovery
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out

set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: sbatch [...] $0 JOBS_FILE" >&2; exit 2; }
cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${ZOO_CONDA_ENV:-mjlab316}"   # this branch needs the MuJoCo 3.16 stack

# Video clips render at step 0 and every video_interval: use the GPU's EGL,
# not a software rasterizer (see envs/velocity/g1.py).
export MUJOCO_GL=egl
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f "$NV_EGL" ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="$NV_EGL"

echo "[slurm] $(hostname) job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-1}" \
     "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
examples/run_jobs.sh "$1" --only "${SLURM_ARRAY_TASK_ID:-1}"
