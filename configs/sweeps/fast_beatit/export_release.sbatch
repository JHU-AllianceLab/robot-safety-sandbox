#!/bin/bash
#SBATCH --job-name=fast_beatit_release
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,a100
#SBATCH --exclude=gl105
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
# ONNX for the Hugging Face release (the user, 2026-11-26): teacher (200-D) and no-foot student (188-D), each the
# fallback + the full monitor + fast monitors k 10 / 15 / 25 with the PUSH-BANK value functions (task-agnostic);
# thresholds per task from runs/sailr_df/calib/<task>/thresholds.json. Then the onnxruntime parity + CPU latency check.
#   sbatch configs/sweeps/fast_beatit/export_release.sbatch
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 SANDBOX=$PWD PYTHONPATH=$(realpath $HOME/ssb-wt-hybrid)
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
ORTPY=${ORTPY:-$HOME/scratch_hhu49/yzeng52/venvs/ort/bin/python}
O=runs/fast_beatit
F='linesearch|solver iterations limit|warn_overflow|FutureWarning|warnings.warn|TracerWarning|torch.jit|DeprecationWarning|torch.onnx.export|exact_index'
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)"
python examples/export_fast_beatit_onnx.py --values $O/v_teacher_pushbank/values.pt --variant pushbank \
  --calib-key beatit_fast --full-key beatit_thermal --ks 10 15 25 --out $O/release/teacher_200d \
  --what "Beat-It v0.6.1 heat-aware, teacher world model (200-D state)" 2>&1 | grep -v -E "$F" | tail -4
python examples/export_fast_beatit_onnx.py --values $O/v_student_pushbank/values.pt --variant pushbank \
  --calib-key beatit_student --full-key beatit_student --ks 10 15 25 --out $O/release/student_nofoot_188d \
  --run-dir runs/beatit_student_v061/nofoot_s0/g1_recover_thermal70_v61_safetyshac1p \
  --tube runs/beatit_student_v061/calib/nofoot_s0/min_tube.json --tag nofoot_s0 \
  --what "Beat-It v0.6.1 heat-aware, no-foot student world model (188-D state)" 2>&1 | grep -v -E "$F" | tail -4
for m in teacher_200d student_nofoot_188d; do
  [ -f $O/release/$m/spec.json ] || { echo "export $m FAILED"; continue; }
  $ORTPY examples/fast_beatit_onnx_check.py --dir $O/release/$m 2>&1 | tail -7 | sed "s/^/[$m] /"
done
echo "done $(date -Is)"
