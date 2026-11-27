#!/bin/bash
# A held-out SONIC hand-over bank on the sysID v0.6 robot, then the temperature sweep on it (the user, 2026-11-23:
# "build the v0.6 bank with our up-to-date robot model and run the calibration and evaluation there").
# The recipe of the temperature-agnostic bank (docs/experiments/shac-ppo-sonic.md section 5: LAFAN TEST clips,
# 8000 episodes, harvest seed 1, thinned with the TRAIN bank's weights) on the v0.6 robot exactly as
# configs/sweeps/shac_thermal70_v6/data_jobs.txt built the v6 TRAIN bank (task g1_recover_lafan_sysid_v6,
# SONIC on the robot's own hip pitch, |dv| ~ U(0, 5) m/s, rows 0 / 0.1 / 0.2 / 0.3 s after the kick).
# Then configs/sweeps/thermal_volume/sweep.sh with BANK= the new bank (K = 4 as the reference).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,a100,h100 --gres=gpu:1 --cpus-per-task=16 \
#     --mem=96G --time=03:30:00 --job-name=thermal_volume_v6bank --output=logs/slurm/%x_%j.out \
#     configs/sweeps/thermal_volume/v6_bank_sweep.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)"
D=data/g1_sonic_push_gfall_sysid316v6_test
[ -f $D.pt ] || python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan_sysid_v6 --hip mjlab \
  --library data/g1_lafan1_clips_test.pt --sonic-dir $HOME/gear_sonic_hf --episodes 8000 --seed 1 --bank-delays-s --out $D.pt
[ -f ${D}_bank.pt ] || python examples/harvest_g1_sonic_trajectories.py --from-traj $D.pt \
  --bank-delays-s 0 0.1 0.2 0.3 --bank-out ${D}_bank.pt
[ -f ${D}_bank_net.pt ] || python examples/dedup_start_bank.py --bank ${D}_bank.pt \
  --W-from data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt
echo "[bank] $(date +%T) ${D}_bank_net.pt ready"
set +e
BANK=${D}_bank_net.pt K=${K:-4} OUT=${OUT:-runs/thermal_volume/v6bank_${SLURM_JOB_ID:-local}} \
  bash configs/sweeps/thermal_volume/sweep.sh
