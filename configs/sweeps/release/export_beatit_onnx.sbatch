#!/bin/bash
#SBATCH --job-name=beatit_onnx
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,b200,a100
#SBATCH --exclude=gh204,gh205
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Heat-aware Beat-It v0.6.1 for the robot (the user, 2026-11-25): filter only (fallback + 1 s
# monitor) as ONNX, two variants:
#   teacher_200d        runs/v061_study/beatit_thermal_s0 (the released one), Wang eps 1 % threshold 0.665
#   student_nofoot_188d the frozen teacher fallback + the no-foot student world model
#                       (runs/beatit_student_v061/nofoot_s0, coolmimic slurm/beatit_student.sbatch);
#                       no Wang threshold yet (spec.json records its 1 s / 95 % tube eps)
# examples/export_beatit_onnx.py, then examples/beatit_onnx_check.py with onnxruntime (CPU, 1 and 4
# threads) in ~/scratch_hhu49/yzeng52/venvs/ort.
#   sbatch configs/sweeps/release/export_beatit_onnx.sbatch
set -uo pipefail
SB=$(realpath "${SLURM_SUBMIT_DIR:-$PWD}")
cd "$SB"
SSB=$(realpath "${SSB_HYBRID:-$HOME/ssb-wt-hybrid}")
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$SSB${PYTHONPATH:+:$PYTHONPATH}
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
ORT=${ORT:-$HOME/scratch_hhu49/yzeng52/venvs/ort/bin/python}
OUT=${OUT:-$HOME/scratch_hhu49/yzeng52/deploy/beatit_v061_filter_2026-11-25}
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Is) sandbox $(git rev-parse --short HEAD) ssb $(git -C $SSB rev-parse --short HEAD)"
nvidia-smi -L | head -1
rc=0
run() {  # name run_dir tube tag threshold-args...
  local name=$1 rd=$2 tube=$3 tag=$4; shift 4
  echo "################ $name $(date -Is)"
  python examples/export_beatit_onnx.py --run-dir $rd --tube $tube --tag $tag --out $OUT/$name "$@" \
    || { echo "[onnx] $name export FAILED"; rc=1; return; }
  cp examples/beatit_onnx_check.py $OUT/$name/check_onnx.py
  for th in 1 4; do
    $ORT $OUT/$name/check_onnx.py --folder $OUT/$name --threads $th || { echo "[onnx] $name check t$th FAILED"; rc=1; }
  done
}
run teacher_200d runs/v061_study/beatit_thermal_s0/g1_recover_thermal70_v61_safetyshac1p \
  runs/v061_study/eval/calib/beatit_thermal/min_tube.json beatit_thermal_s0 \
  --threshold 0.665 \
  --threshold-note "Wang et al. Prop. 1, eps 1 %, beta 1e-10, on 31,780 held-out SONIC hand-over states, every motor at one T0 ~ U(30, 60) C, failure = a fall or a winding at 70 C within 3 s (runs/v061_study/eval/report.json hot30_60). Covers 83.0 % of the states, 99.4 % of rollouts inside succeed." \
  --what "heat-aware Beat-It v0.6.1 (seed 0, SysID v0.6.1 hybrid world model, 200-D state): fallback + 1 s monitor"
run student_nofoot_188d runs/beatit_student_v061/nofoot_s0/g1_recover_thermal70_v61_safetyshac1p \
  runs/beatit_student_v061/calib/nofoot_s0/min_tube.json nofoot_s0 \
  --threshold-note "NOT calibrated: no Wang threshold yet for this world model. tube_eps_1s_95 is its 1 s / 95 % world-model error bound (teacher's: 0.808); check_onnx.py uses it as the threshold." \
  --what "heat-aware Beat-It v0.6.1 with the no-foot STUDENT world model (188-D state: no foot height / air time / contact / contact forces; same fallback as the teacher): fallback + 1 s monitor"
echo "done $(date -Is) rc $rc"
ls -la $OUT/*
exit $rc
