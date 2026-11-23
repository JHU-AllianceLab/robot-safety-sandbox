#!/bin/bash
# Video of the first cooling policy (pilot_s0, 500M) taking over from SONIC with all four ankle
# windings at 65 C (housing 35 C), the task's minor pushes on, 4 robots for 30 s
# (the user, 2026-11-19). examples/cool_video.py; a 3 s smoke first.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,a100,h100,h200 --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --job-name=cool_video --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/cool_video.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip
python examples/cool_video.py --policy $P --duration-s 3 --hold-s 0.2 --out runs/g1_cool/video/smoke.mp4
python examples/cool_video.py --policy $P --seed ${SEED:-0} --out runs/g1_cool/video/cooling_pilot_seed${SEED:-0}.mp4
echo "[job] cool_video done"
