#!/bin/bash
#SBATCH --job-name=fast_beatit_values
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,a100
#SBATCH --exclude=gl105
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# The robot-ready Beat-It filter's value functions and thresholds (the user, 2026-11-26), for the teacher (200-D) and the
# no-foot student (188-D) world model:
#   a. the student's labels on each task's calibration bank (coolmimic scripts/beatit_student_label.py, --dump-wm);
#   b. PUSH-BANK V (fitted once, task-agnostic, shipped): V_(50-k) imitates the full monitor's imagination from push
#      states (the generic train bank + the three tasks' seed-4 SONIC banks, fallback proposals);
#   c. TASK-TUNED V: the push-bank V fine-tuned on one task's tracker states (pushed, the tracker's own proposals;
#      runs/fast_beatit/tracker_states from 3 checkpoints);
#   d. thresholds per task (examples/fast_beatit_calibrate.py) into runs/sailr_df/calib/<task>/thresholds.json:
#      beatit_fast / beatit_fast_tuned (teacher), beatit_student / beatit_student_tuned (student; "full" = its full monitor).
#   sbatch configs/sweeps/fast_beatit/values.sbatch
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 SANDBOX=$PWD PYTHONPATH=$(realpath $HOME/ssb-wt-hybrid)
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
COOL=$HOME/coolmimic
O=runs/fast_beatit; REC=$O/tracker_states
STU=runs/beatit_student_v061/nofoot_s0/g1_recover_thermal70_v61_safetyshac1p/final_model.zip
STUTUBE=runs/beatit_student_v061/calib/nofoot_s0/min_tube.json
GEN=runs/beatit_student_v061/trunc_grid/real/real_rollouts.pt
REAL="$GEN $O/real/dance/real_rollouts.pt $O/real/fight/real_rollouts.pt $O/real/bentknee/real_rollouts.pt"
TASKS="dance fight bentknee"
F='linesearch|solver iterations limit|warn_overflow|FutureWarning|warnings.warn'
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)"
tests() { local d=$1 x=""; for T in $TASKS; do x="$x $T:$O/$d/$T/test.episodes.pt"; done; echo $x; }

echo "== a. student labels $(date -Is)"
for T in $TASKS; do
  mkdir -p $O/test_student/$T
  [ -f $O/test_student/$T/test.episodes.pt ] || python $COOL/scripts/beatit_student_label.py --policy $STU \
    --task g1_recover_thermal70_v61 --no-push --no-obs-noise --thermal-start-c 30 60 \
    --bank data/g1_sonic_push_gfall_sysid316v6_dv25_${T}_bank_net.pt --k 4 --wm-tube wmmin:1:0.95 \
    --tube-calib $STUTUBE --tube-tag nofoot_s0 --dump-wm --out $O/test_student/$T/test.pt 2>&1 | grep -v -E "$F" \
    > $O/test_student/$T/test.log &
done
wait
for T in $TASKS; do [ -f $O/test_student/$T/test.episodes.pt ] || { echo "[$T] student labels FAILED"; tail -20 $O/test_student/$T/test.log; exit 1; }; done

fitv() {  # model(teacher|student) out extra...
  local m=$1 o=$2; shift 2
  mkdir -p $o
  local d=test x=""; [ $m = student ] && { d=test_student; x="--nofoot --layout-from $GEN"; }
  python examples/fast_beatit_distill.py $x "$@" --out $o 2>&1 | grep -v -E "$F" | tee $o/fit.log | grep -E "^\[distill\]|V_" | cut -c1-220
  [ -f $o/values.pt ] || { echo "fit $o FAILED"; return 1; }
}
for m in teacher student; do
  d=test; [ $m = student ] && d=test_student
  echo "== b. $m push-bank V $(date -Is)"
  fitv $m $O/v_${m}_pushbank --real $REAL --tests $(tests $d) --variant-name pushbank || exit 1
  for T in $TASKS; do
    echo "== c. $m tuned V, $T $(date -Is)"
    fitv $m $O/v_${m}_tuned_$T --tracker $REC/${T}_it*.pt --tests $T:$O/$d/$T/test.episodes.pt \
      --init-values $O/v_${m}_pushbank/values.pt --init-variant pushbank --variant-name tuned_$T --lr 3e-4 --epochs 5 || exit 1
  done
done
echo "== d. thresholds $(date -Is)"
for T in $TASKS; do
  C=runs/sailr_df/calib/$T/thresholds.json
  cal() { python examples/fast_beatit_calibrate.py --ks 10 15 25 --thresholds $C "$@" 2>&1 | grep "^\[fast-calib\]" | sed "s/^/[$T] /"; }
  cal --labels $O/test/$T/test.episodes.pt --values $O/v_teacher_pushbank/values.pt --key beatit_fast
  cal --labels $O/test/$T/test.episodes.pt --values $O/v_teacher_tuned_$T/values.pt --key beatit_fast_tuned
  cal --labels $O/test_student/$T/test.episodes.pt --values $O/v_student_pushbank/values.pt --key beatit_student
  cal --labels $O/test_student/$T/test.episodes.pt --values $O/v_student_tuned_$T/values.pt --key beatit_student_tuned
done
echo "done $(date -Is)"
