#!/bin/bash
#SBATCH --job-name=fast_beatit_onnx
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
# Export the fast Beat-It monitors (k + V_(50-k)), the full monitor and the fallback as ONNX with per-task thresholds
# (examples/export_fast_beatit_onnx.py), then the onnxruntime parity + CPU latency check (fast_beatit_onnx_check.py,
# in the onnxruntime venv). Needs runs/fast_beatit/volume (configs/sweeps/fast_beatit/volume.sbatch).
#   sbatch configs/sweeps/fast_beatit/export.sbatch   [KS="10 15 25" OUT=runs/fast_beatit/onnx]
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 SANDBOX=$PWD PYTHONPATH=$(realpath $HOME/ssb-wt-hybrid)
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
OUT=${OUT:-runs/fast_beatit/onnx}
ORTPY=${ORTPY:-$HOME/scratch_hhu49/yzeng52/venvs/ort/bin/python}
F='linesearch|solver iterations limit|warn_overflow|FutureWarning|warnings.warn|TracerWarning|torch.jit'
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)"
python examples/export_fast_beatit_onnx.py --volume runs/fast_beatit/volume --ks ${KS:-10 15 25} --out $OUT 2>&1 \
  | grep -v -E "$F" | tail -20
[ -f $OUT/spec.json ] || { echo "export FAILED"; exit 1; }
$ORTPY examples/fast_beatit_onnx_check.py --dir $OUT 2>&1 | tail -20
ls -la $OUT
echo "done $(date -Is)"
