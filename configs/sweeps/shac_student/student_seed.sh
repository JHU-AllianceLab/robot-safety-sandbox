#!/bin/bash
# Array task N (1..9) of configs/sweeps/shac_student/jobs.txt: train student N, check its
# actor is still the teacher's (check_frozen.py), calibrate its tube (calib_jobs.txt lines
# 2N-1, 2N); the task that finishes last of 1-6 writes runs/shac_student/tube_compare.*,
# of 7-9 runs/shac_student/tube_compare_nolinvel.*.
# See jobs.txt for the sbatch line.
#SBATCH --gres=gpu:1
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
N=${SLURM_ARRAY_TASK_ID:?an array task}
D=configs/sweeps/shac_student
if (( N <= 3 )); then ARM=nofoot; S=$((N - 1)); RUN=g1_recover_noheight_nofoot_safetyshac1p; DIM=101
elif (( N <= 6 )); then ARM=full; S=$((N - 4)); RUN=g1_recover_noheight_safetyshac1p; DIM=113
else ARM=nolinvel; S=$((N - 7)); RUN=g1_recover_noheight_nofoot_nolinvel_safetyshac1p; DIM=98; fi
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $ARM seed $S"
examples/run_jobs.sh $D/jobs.txt --only $N || { echo "[job] training FAILED"; exit 1; }
python $D/check_frozen.py runs/shac_noheight/shac_s$S/g1_recover_noheight_safetyshac1p/final_model.zip \
  runs/shac_student/${ARM}_s$S/$RUN/final_model.zip --state-dim $DIM || exit 1
for j in $((2 * N - 1)) $((2 * N)); do
  examples/run_jobs.sh $D/calib_jobs.txt --only $j || { echo "[job] calib job $j FAILED"; exit 1; }
done
echo "[job] $ARM seed $S done"
if (( N <= 6 )) && ls runs/shac_student/calib/{nofoot,full}_s{0,1,2}/min_tube.json >/dev/null 2>&1; then
  python examples/compare_wm_tubes.py \
    --arm teacher runs/shac_noheight/wmreport/calib/noheight_s{0,1,2} \
    --arm control runs/shac_student/calib/full_s{0,1,2} \
    --arm student runs/shac_student/calib/nofoot_s{0,1,2} \
    --out runs/shac_student/tube_compare.json 2>&1 | tee runs/shac_student/tube_compare.log
fi
if (( N >= 7 )) && ls runs/shac_student/calib/{nofoot,full,nolinvel,teacher_rerun_a}_s{0,1,2}/min_tube.json \
    >/dev/null 2>&1; then
  python examples/compare_wm_tubes.py \
    --arm teacher runs/shac_noheight/wmreport/calib/noheight_s{0,1,2} \
    --arm teacher_rerun runs/shac_student/calib/teacher_rerun_a_s{0,1,2} \
    --arm full_113 runs/shac_student/calib/full_s{0,1,2} \
    --arm nofoot_101 runs/shac_student/calib/nofoot_s{0,1,2} \
    --arm nolinvel_98 runs/shac_student/calib/nolinvel_s{0,1,2} \
    --out runs/shac_student/tube_compare_nolinvel.json 2>&1 | tee runs/shac_student/tube_compare_nolinvel.log
fi
