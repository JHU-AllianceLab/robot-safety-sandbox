#!/bin/bash
# Smoke test of the student pipeline on teacher seed 0, before the array (jobs.txt):
# a 3M-step 101-D student and a 2M-step 113-D control (fresh world model despite the
# matching shapes), check_frozen.py on both, and a 64-episode calibration + tube fit +
# comparison for the student.
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=01:00:00 --job-name=student_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_student/smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
T=runs/shac_noheight/shac_s0/g1_recover_noheight_safetyshac1p/final_model.zip
CFG=configs/g1_recover_sonic_noheight_student_wm.yaml
S=runs/shac_student/smoke
rm -rf $S && mkdir -p $S
SHORT="--num-envs 1024 --lr-decay-iters 1200 --checkpoint-every 1000000000 --no-wandb"

# 1. the 101-D student
python examples/train.py --config $CFG --seed 0 --load $T --no-load-world-model --steps 3000000 \
  $SHORT --out $S/nofoot 2>&1 | tee $S/nofoot.log
grep -q "state = the 'critic' group (101 dims); actor obs 98 dims" $S/nofoot.log \
  || { echo "[smoke] FAILED: the student state is not the 101-D critic group"; exit 1; }
grep -q "actor_learning_rate\|actor_loss" $S/nofoot.log \
  && { echo "[smoke] FAILED: the actor was updated"; exit 1; }
P=$S/nofoot/g1_recover_noheight_nofoot_safetyshac1p/final_model.zip
python configs/sweeps/shac_student/check_frozen.py $T $P --state-dim 101

# 2. the 113-D control: the teacher's shapes, but --no-load-world-model
python examples/train.py --config $CFG --task g1_recover_noheight --seed 0 --load $T \
  --no-load-world-model --steps 2000000 $SHORT --out $S/full 2>&1 | tee $S/full.log
grep -q "world model NOT restored" $S/full.log \
  || { echo "[smoke] FAILED: the control did not start a fresh world model"; exit 1; }
python configs/sweeps/shac_student/check_frozen.py $T \
  $S/full/g1_recover_noheight_safetyshac1p/final_model.zip --state-dim 113

# 3. calibration + tube fit of the student, and the comparison against the teacher's
python examples/eval_sonic_qcbf.py --policy $P --tag smoke --task g1_recover_lafan_noheight_nofoot \
  --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 64 --num-envs 32 \
  --onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint \
  --no-obs-noise --seed 1 --conditions sonic wmcalib:clean --out $S/calib/nofoot
python examples/calibrate_wm_min_tube.py --calib $S/calib/nofoot/push.pt --tag smoke
python examples/compare_wm_tubes.py --arm teacher runs/shac_noheight/wmreport/calib/noheight_s0 \
  --arm student $S/calib/nofoot --out $S/tube_compare.json
echo "[job] smoke ok"
