#!/bin/bash
# Squat tracking, step 0: the GPU smoke of the whole pipeline (plan step 3, AMENDMENTS).
# Every step runs even if an earlier one failed (so one submission shows every problem);
# the exit status is 1 if any step failed, so the pilot can wait on it with afterok.
#   mkdir -p logs/slurm
#   sbatch -A hhu49_alliance1 -p rtx6000,h100,h200,a100,l40s --gres=gpu:1 --cpus-per-task=8 \
#       --mem=64G --time=05:00:00 --job-name=track_smoke --output=logs/slurm/%x_%j.out \
#       configs/sweeps/track_squat/smoke.sh
#
#   a. the fallbacks' layout references (configs/track_squat/fallback_layout_<world>.json)
#   b. the reference audit, both worlds: kinematics, open-loop playback, the oracle replay
#      through all six tubes (runs/track_squat/audit/<world>/audit.json: the engagement
#      floor the report subtracts)
#   c. the task env, both worlds: dims, spawn, amp features vs the reference, the frame
#      cursor, the bus vs the env's failure counters, the fallback alone from every phase bin
#   d. the train-time filter live on the task env: parity with the eval's verdicts, the
#      dwell invariants, and the throughput bench (thermal70)
#   e. 6 arms x 2M steps x 1024 envs, with the trainer's --smoke self-checks
#   f. eval (64 episodes; bare / wm_d1 / wmcalib at fallback seed 2) + the SONIC anchor +
#      the report, on the smoke runs
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

Q=configs/sweeps/track_squat/quiet.sh
S=runs/track_squat/smoke
rm -rf "$S" runs/track_squat/eval/smoke runs/track_squat/report/smoke
mkdir -p "$S"
FAILED=()
step() {   # step NAME CMD...: run, log, record a failure, carry on
  local name="$1"; shift
  echo "=== [smoke] $name: $*"
  if "$@" 2>&1 | tee "$S/$name.log"; then
    echo "=== [smoke] $name OK"
  else
    echo "=== [smoke] $name FAILED"
    FAILED+=("$name")
  fi
}

[[ -f data/g1_amp_expert_lafan_train_v1.pt ]] || step amp_expert $Q python examples/build_amp_expert.py

# a. layouts
step layout_t70 $Q python examples/smoke_wm_tube.py --dump-layout --world thermal70 --out $S/wm_tube
step layout_cbf $Q python examples/smoke_wm_tube.py --dump-layout --world thermal_cbf --out $S/wm_tube
# b. audits (the full segment)
step audit_t70 $Q python examples/audit_track_reference.py --task g1_track_squat_thermal70 \
  --out runs/track_squat/audit/thermal70
step audit_cbf $Q python examples/audit_track_reference.py --task g1_track_squat_thermal_cbf \
  --out runs/track_squat/audit/thermal_cbf
# c. the task env (the fallback-alone survival is REPORTED per phase bin, not gated: a low
#    number is a finding about the fallback on this segment, not a pipeline bug)
step task_t70 $Q python examples/smoke_g1_track.py --task g1_track_squat_thermal70 --fb-seed 2 \
  --min-survival 0.0 --out $S/task_t70.json
step task_cbf $Q python examples/smoke_g1_track.py --task g1_track_squat_thermal_cbf --fb-seed 2 \
  --min-survival 0.0 --out $S/task_cbf.json
# d. the filter on the task env (+ bench in thermal70)
step filter_t70 $Q python examples/smoke_wm_tube.py --world thermal70 --task g1_track_squat_thermal70 \
  --bench --out $S/wm_tube
step filter_cbf $Q python examples/smoke_wm_tube.py --world thermal_cbf --task g1_track_squat_thermal_cbf \
  --out $S/wm_tube
# e. every arm trains (2M steps)
for ARM in t70_bare t70_wm_d0 t70_wm_d1 cbf_bare cbf_wm_d0 cbf_wm_d1; do
  step train_$ARM $Q python examples/train.py --config configs/g1_track_squat_ampppo.yaml \
    --arm $ARM --seed 2 --stage smoke --num-envs 1024 --steps 2000000 --checkpoint-every 1000000 \
    --no-wandb --smoke --out runs/track_squat/smoke/$ARM/s2
done
# f. eval + anchor + report
for ARM in t70_wm_d1 cbf_bare; do
  W=thermal70; [[ $ARM == cbf_* ]] && W=thermal_cbf
  step eval_$ARM $Q python examples/eval_track.py --stage smoke --arm $ARM --seed 2 --world $W \
    --conditions bare wm_d1 wmcalib --fb-seeds 2 --episodes 64
done
step eval_t70_bare_ckpt $Q python examples/eval_track.py --stage smoke --arm t70_bare --seed 2 \
  --world thermal70 --curve-every 1000000 --episodes 64
step eval_sonic $Q python examples/eval_track.py --sonic --world thermal70 --conditions bare wm_d1 \
  --fb-seeds 2 --episodes 64 --out runs/track_squat/eval/smoke/sonic/thermal70
step report $Q python examples/track_report.py --stage smoke

echo
if (( ${#FAILED[@]} )); then
  echo "[smoke] FAILED steps: ${FAILED[*]}"
  exit 1
fi
echo "[smoke] all steps passed"
