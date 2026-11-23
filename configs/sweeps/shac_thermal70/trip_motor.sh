#!/bin/bash
# Which motor reaches 70 C first when SONIC alone (no fallback, no filter) tracks held-out
# LAFAN clips (test library; fallAndGetUp and sprint excluded) from a robot at room
# temperature: every motor's winding, housing and structure start at their own
# U(28, 32) C draw (--thermal-start-c; the user's "real case", 2026-11-16), no kick
# (--dv-range 0 0), up to 60 s of tracking per episode, on g1_recover_lafan_thermal70
# (the 70 C overheat termination ends the episode at the first motor over it).
# eval_sonic_qcbf.py --thermal-per-motor records each motor's winding at the spawn, its
# peak and at the episode's end (just before the respawn when the env ended it);
# examples/thermal_trip_report.py counts, per motor, the episodes it ended.
# The SHAC policy is only loaded (SONIC alone never runs it).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p interactive_gpu --gres=gpu:1 --cpus-per-task=16 --time=02:00:00 --job-name=thermal_trip --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70/trip_motor.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
OUT=runs/shac_thermal70/trip_motor/cold30_track60
python examples/eval_sonic_qcbf.py --policy $P --tag trip_cold30 --sonic-dir $HOME/gear_sonic_hf \
  --task g1_recover_lafan_thermal70 --hip mjlab --suites push --episodes 2048 --num-envs 512 \
  --onset-s 0.02 0.02 --window-s 60 --dv-range 0 0 --dv-bins 0 1 \
  --exclude-categories fallAndGetUp sprint --no-obs-noise --conditions sonic \
  --thermal-per-motor --thermal-start-c 28 32 --out $OUT/eval
python examples/thermal_trip_report.py $OUT/eval/push.pt --out $OUT/report.json | tee $OUT/report.txt
echo "[job] trip_motor done"
