#!/bin/bash
# The v0.6.1 filter study's held-out evaluation bank (the user, 2026-11-23): SONIC kicked on the LAFAN TEST clips
# with |dv| ~ U(0, 3.5) m/s (the cap where the current Beat-It's Wang volume at eps 1% is 83-86 % on the
# 5 m/s v0.6 bank, runs/thermal_volume/dv_cut), a fresh harvest seed (2), 8000 episodes, rows 0 / 0.1 / 0.2 / 0.3 s
# after the kick, thinned with the v6 TRAIN bank's weights. Recipe otherwise = configs/sweeps/thermal_volume/
# v6_bank_sweep.sh. The harvest task has no heat model; its body files hash the same in v0.6.0 and v0.6.1.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,a100,h100 --gres=gpu:1 --cpus-per-task=16 \
#     --mem=96G --time=01:00:00 --job-name=v061_eval_bank --output=logs/slurm/%x_%j.out configs/sweeps/v061_study/eval_bank.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)"
D=data/g1_sonic_push_gfall_sysid316v6_dv35_test
[ -f $D.pt ] || python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan_sysid_v6 --hip mjlab \
  --library data/g1_lafan1_clips_test.pt --sonic-dir $HOME/gear_sonic_hf --episodes 8000 --seed 2 \
  --dv-range 0 3.5 --bank-delays-s --out $D.pt
[ -f ${D}_bank.pt ] || python examples/harvest_g1_sonic_trajectories.py --from-traj $D.pt \
  --bank-delays-s 0 0.1 0.2 0.3 --bank-out ${D}_bank.pt
[ -f ${D}_bank_net.pt ] || python examples/dedup_start_bank.py --bank ${D}_bank.pt \
  --W-from data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt
echo "[bank] $(date -Is) ${D}_bank_net.pt ready"
