#!/bin/bash
# Safe-set volume of the thermal Beat-It vs the start temperature (the user, 2026-11-23).
# Beat-It v6 hybrid s1 in its own world (g1_recover_thermal70_v6: fail = a fall or a winding at 70 C within 3 s),
# every motor soaked at T0 (winding = housing = structure), the 31,888 start states of the temperature-agnostic
# Wang calibration (runs/shac_vs_ppo_sonic/eval/ppo_s0.pt; BANK= another bank), no push, no obs noise, K rollouts per row;
# score = Beat-It's 1 s world-model bound (wmmin:1:0.95, tube v6hyb_s1). Then examples/thermal_safe_volume.py.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,a100,h100 --gres=gpu:1 --cpus-per-task=16 \
#     --mem=96G --time=03:00:00 --job-name=thermal_volume --output=logs/slurm/%x_%j.out \
#     configs/sweeps/thermal_volume/sweep.sh            [TEMPS="30 40 ..." K=2 MAX_ROWS= PAR=3 NENV=4096]
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
J=${SLURM_JOB_ID:-local}
OUT=${OUT:-runs/thermal_volume/$J}
TEMPS=${TEMPS:-30 40 50 55 58 60 62 63 64 65 66 67 68 69}
mkdir -p $OUT
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) temps: $TEMPS K ${K:-2} max_rows ${MAX_ROWS:-all} bank ${BANK:-runs/shac_vs_ppo_sonic/eval/ppo_s0.pt}"
python -c "import safety_sb3, os; print('[sweep] safety_sb3', os.path.dirname(safety_sb3.__file__))"
B=runs/shac_thermal70_v6/hybrid_s1/g1_recover_thermal70_v6_safetyshac1p/final_model.zip
one() {
  T=$1
  python examples/label_start_states.py --policy $B --wm-tube wmmin:1:0.95 \
    --tube-calib runs/shac_thermal70_v6/wmreport/calib/v6hyb_s1/min_tube.json --tube-tag v6hyb_s1 \
    --task g1_recover_thermal70_v6 --no-push --no-obs-noise --bank ${BANK:-runs/shac_vs_ppo_sonic/eval/ppo_s0.pt} \
    --k ${K:-2} --num-envs ${NENV:-4096} --thermal-start-c $T ${MAX_ROWS:+--max-rows $MAX_ROWS} \
    --out $OUT/T$T.pt > $OUT/T$T.log 2>&1 && echo "[sweep] T0 $T done $(date +%T)" || echo "[sweep] T0 $T FAILED (see $OUT/T$T.log)"
}
export -f one; export B OUT K NENV MAX_ROWS BANK
printf '%s\n' $TEMPS | xargs -P ${PAR:-3} -I{} bash -c 'one {}'
for T in $TEMPS; do grep -E "\[label\] done|Traceback|Error" $OUT/T$T.log | tail -2 | sed "s/^/  T$T /"; done
python examples/thermal_safe_volume.py --dir $OUT
echo "[sweep] done $(date -Is)"
