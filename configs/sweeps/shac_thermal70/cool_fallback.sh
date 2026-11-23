#!/bin/bash
# Does the thermal70 SHAC fallback (trained to avoid falling AND a 70 C winding) cool hot
# ankles when it runs alone, with no task? (the user's question, 2026-11-18, before
# designing a cooling mode.) The fallback alone ("fallback" condition) from a LAFAN spawn
# (held-out library; fallAndGetUp and sprint excluded), one pelvis kick |dv| ~ U(0, 4) m/s
# at U(5, 15) s, then 45 s more (episodes up to 60 s), on g1_recover_lafan_thermal70 (the
# 70 C overheat termination ends the episode). Every motor starts at room temperature
# (U(28, 32) C, housing = structure = winding) except the ankles, hot as after the task
# tripped (ankle roll windings U(60, 68), ankle pitch U(45, 60) -- the trip_motor run's
# picture), in two housing scenarios:
#   fresh  : housing ~room (gap U(25, 35) / U(15, 30) C) -- right after a short hot burst
#   soaked : housing near the winding (gap U(3, 10) C) -- after long operation
# eval_sonic_qcbf.py --thermal-trace-s 1 records every motor's winding / housing / mean
# tau_m^2 every 1 s; examples/thermal_cool_report.py summarizes.
# Another policy (e.g. the g1_cool cooling policy, a plain PPO zip): POLICY=<zip> OUT_ROOT=<dir> TAG=<name>.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p interactive_gpu --gres=gpu:1 --cpus-per-task=16 --time=01:30:00 --job-name=thermal_cool --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70/cool_fallback.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=${POLICY:-runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip}
ROOT=${OUT_ROOT:-runs/shac_thermal70/cool_fallback}
COMMON=(--policy $P --tag ${TAG:-thermal70_s0} --sonic-dir $HOME/gear_sonic_hf
  --task g1_recover_lafan_thermal70 --hip mjlab --suites push --episodes 1024 --num-envs 1024
  --onset-s 5 15 --window-s 45 --dv-range 0 4 --dv-bins 0 1 2 3 4
  --exclude-categories fallAndGetUp sprint --no-obs-noise --conditions fallback
  --thermal-per-motor --thermal-trace-s 1 --thermal-start-c 28 32)
declare -A GAPS=([fresh]="25 35 15 30" [soaked]="3 10 3 10")
for S in fresh soaked; do
  read -r RG0 RG1 PG0 PG1 <<< "${GAPS[$S]}"
  python examples/eval_sonic_qcbf.py "${COMMON[@]}" \
    --thermal-start-group ankle_roll 60 68 $RG0 $RG1 \
    --thermal-start-group ankle_pitch 45 60 $PG0 $PG1 \
    --out $ROOT/$S/eval
  python examples/thermal_cool_report.py $ROOT/$S/eval/push.pt --dv-bins 0 1 2 3 4 \
    --out $ROOT/$S/report.json | tee $ROOT/$S/report.txt
done
echo "[job] cool_fallback done"
