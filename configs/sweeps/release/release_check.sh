#!/bin/bash
# The release of the three G1 thermal safety filters: build it and check it end to end.
# Every step runs even if an earlier one failed; exit 1 if any failed.
#   mkdir -p logs/slurm
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p l40s,rtx6000,h100,h200,a100 \
#       --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --job-name=release_check \
#       --output=logs/slurm/%x_%j.out configs/sweeps/release/release_check.sh
#
#   a. CPU tests: tests/test_wm_tube_filter.py (latch, release round trip) and the eval
#      parity (tests/test_wm_tube_parity.py, ZOO_SLOW_TESTS=1)
#   b. the barrier world's layout reference, if missing
#   c. examples/export_safety_filters.py -> runs/release/g1-thermal-safety-filters
#   d. live checks FROM THE RELEASE (filters/pretrained.py paths, manifest provenance):
#      the train-time filter's verdicts torch.equal to the eval's on live states, executed
#      action, holds (dwell 0 / 1 s / latch), for joint, separate (hybrid heat model) and
#      fall_only_cbf on the fallbacks' own tasks, and separate on the squat task env
#   e. sha256sum -c of the release
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

Q=configs/sweeps/track_squat/quiet.sh
REL=runs/release/g1-thermal-safety-filters
S=runs/release/check
rm -rf "$S"
mkdir -p "$S"
FAILED=()
step() {   # step NAME CMD...: run, log, record a failure, carry on
  local name="$1"; shift
  echo "=== [release] $name: $*"
  if "$@" 2>&1 | tee "$S/$name.log"; then
    echo "=== [release] $name OK"
  else
    echo "=== [release] $name FAILED"
    FAILED+=("$name")
  fi
}

# a. tests
step pytest_filter env OMP_NUM_THREADS=1 python -m pytest tests/test_wm_tube_filter.py -q
step pytest_parity env OMP_NUM_THREADS=1 ZOO_SLOW_TESTS=1 python -m pytest \
  tests/test_wm_tube_parity.py -q
# b. layout of the barrier world (the 70 C one exists; the squat project owns both)
[[ -f configs/track_squat/fallback_layout_thermal_cbf.json ]] || \
  step layout_cbf $Q python examples/smoke_wm_tube.py --dump-layout --world thermal_cbf \
    --out $S/layout
# c. export
step export python examples/export_safety_filters.py --out $REL --force
# d. live, from the release
L="python examples/smoke_wm_tube.py --root $REL --seed 0 --num-envs 256 --steps 300 --out $S"
step live_joint $Q $L --world thermal70 --task g1_recover_thermal70 --holds 0 1 latch
step live_separate $Q $L --world thermal70 --task g1_recover_thermal70 --wm-thermal hybrid \
  --holds 0 1 latch
step live_fall_only_cbf $Q $L --world thermal_cbf --task g1_recover_thermal_cbf --holds 0 1 latch
step live_separate_task $Q python examples/smoke_wm_tube.py --root $REL --seed 2 \
  --num-envs 256 --steps 300 --out $S/task --world thermal70 \
  --task g1_track_squat_thermal70 --wm-thermal hybrid --holds 1 latch
# e. checksums
step sha256 bash -c "cd $REL && sha256sum -c --quiet SHA256SUMS"

du -sh "$REL"
if ((${#FAILED[@]})); then
  echo "[release] FAILED: ${FAILED[*]}"
  exit 1
fi
echo "[release] all steps passed"
