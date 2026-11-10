#!/bin/bash
# Smoke test of the 98-D (nolinvel) student before array tasks 7-9 of jobs.txt: a 3M-step
# student on teacher seed 0 (state = exactly the actor terms), check_frozen.py, and a
# 64-episode calibration + tube fit + comparison.
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100,a100,h200 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=01:00:00 --job-name=nolinvel_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_student/smoke_nolinvel.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
T=runs/shac_noheight/shac_s0/g1_recover_noheight_safetyshac1p/final_model.zip
CFG=configs/g1_recover_sonic_noheight_student_wm.yaml
S=runs/shac_student/smoke_nolinvel
rm -rf $S && mkdir -p $S
python examples/train.py --config $CFG --task g1_recover_noheight_nofoot_nolinvel --seed 0 --load $T \
  --no-load-world-model --steps 3000000 --num-envs 1024 --lr-decay-iters 1200 \
  --checkpoint-every 1000000000 --no-wandb --out $S/nolinvel 2>&1 | tee $S/nolinvel.log
grep -q "state = the 'critic' group (98 dims); actor obs 98 dims" $S/nolinvel.log \
  || { echo "[smoke] FAILED: the student state is not the 98-D critic group"; exit 1; }
grep -q "world model NOT restored" $S/nolinvel.log \
  || { echo "[smoke] FAILED: the world model was not started fresh"; exit 1; }
grep -q "actor_learning_rate\|actor_loss" $S/nolinvel.log \
  && { echo "[smoke] FAILED: the actor was updated"; exit 1; }
P=$S/nolinvel/g1_recover_noheight_nofoot_nolinvel_safetyshac1p/final_model.zip
python configs/sweeps/shac_student/check_frozen.py $T $P --state-dim 98 --max-fold 1024
python examples/eval_sonic_qcbf.py --policy $P --tag smoke --task g1_recover_lafan_noheight_nofoot_nolinvel \
  --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 64 --num-envs 32 \
  --onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint \
  --no-obs-noise --seed 1 --conditions sonic wmcalib:clean --out $S/calib
python examples/calibrate_wm_min_tube.py --calib $S/calib/push.pt --tag smoke
python examples/compare_wm_tubes.py --arm teacher runs/shac_noheight/wmreport/calib/noheight_s0 \
  --arm nolinvel $S/calib --out $S/tube_compare.json
echo "[job] smoke ok"
