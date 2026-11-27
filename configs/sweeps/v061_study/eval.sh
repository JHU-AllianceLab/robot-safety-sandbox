#!/bin/bash
# The v0.6.1 filter study's evaluation (the user, 2026-11-23). Every step skips outputs that exist, so the
# script can be re-run as models arrive. Output root runs/v061_study/eval/.
#   1. world-model tubes of the two new Beat-Its (eval_sonic_qcbf wmcalib:clean + calibrate_wm_min_tube, the
#      recipe of configs/sweeps/shac_thermal70_v6/wmreport_jobs.txt jobs 1-2) on their SONIC-driven twins;
#   2. labeling (examples/label_start_states.py, K rollouts per row, no push, no obs noise) on the held-out
#      3.5 m/s bank, every model x condition:
#        nt        temperature-agnostic world (g1_recover_noheight_v6, fail = a fall): the four heat-blind models
#        hot30_60  soaked T0 ~ U(30, 60), fail = a fall or a winding at 70 C: thermal Beat-It on
#                  g1_recover_thermal70_v61, the heat-blind models on g1_recover_noheight_v61_heat
#        hot60_65  the same with T0 ~ U(60, 65)
#      and the temperature sweep of both Beat-Its (T0 = 30 .. 69 C, K_SWEEP rollouts);
#   3. examples/v061_report.py (soundness of {score >= 0}, Wang volumes at eps 1 % / 5 %).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,h100 --gres=gpu:1 --cpus-per-task=16 \
#     --mem=96G --time=08:00:00 --job-name=v061_eval --output=logs/slurm/%x_%j.out configs/sweeps/v061_study/eval.sh
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
HYB=$HOME/ssb-wt-hybrid; BAS=$HOME/safety-stable-baselines-baselines
R=runs/v061_study; E=$R/eval; mkdir -p $E/calib $E/label $E/sweep
BANK=${BANK:-data/g1_sonic_push_gfall_sysid316v6_dv35_test_bank_net.pt}
K=${K:-4}; K_SWEEP=${K_SWEEP:-2}; PAR=${PAR:-3}
RLDH_ARGS=${RLDH_ARGS:---sonic-history applied --dh-box sonic}  # the run config's (rldh smoke 2042353)
SONIC=$HOME/gear_sonic_hf
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is) bank $BANK K $K"
zipof() { ls $R/$1_s0/*/final_model.zip 2>/dev/null | head -1; }
declare -A POL=( [beatit_noheight]="$(zipof beatit_noheight)" [safetyppo]="$(zipof safetyppo_noheight)"
  [lambdareach]="$(ls runs/thermal65/lambdareach_s0/*/final_model.zip | head -1)" [rldh]="$(zipof rldh_noheight)"
  [beatit_thermal]="$(zipof beatit_thermal)" )
for m in "${!POL[@]}"; do echo "[eval] $m: ${POL[$m]:-MISSING}"; done

# --- 1. tubes ------------------------------------------------------------------------------------------
calib() {  # name task
  local n=$1 task=$2 d=$E/calib/$1
  [ -n "${POL[$n]}" ] || return 0
  [ -f $d/min_tube.json ] && return 0
  PYTHONPATH=$HYB python examples/eval_sonic_qcbf.py --policy ${POL[$n]} --tag ${n}_s0 --sonic-dir $SONIC \
    --task $task --hip mjlab --suites push --episodes 2048 --num-envs 256 --onset-s 0.02 0.02 --dv-range 0 4 \
    --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise --seed 1 \
    --conditions sonic wmcalib:clean --out $d > $d.log 2>&1 \
  && PYTHONPATH=$HYB python examples/calibrate_wm_min_tube.py --calib $d/push.pt --tag ${n}_s0 >> $d.log 2>&1 \
  && echo "[eval] tube $n: $(python -c "import json; print(json.load(open('$d/min_tube.json'))['min_eps']['0.95']['50'])")" \
  || echo "[eval] tube $n FAILED ($d.log)"
}
mkdir -p $E/calib/beatit_noheight $E/calib/beatit_thermal
calib beatit_noheight g1_recover_lafan_noheight_v6 &
calib beatit_thermal g1_recover_lafan_thermal70_v61 &
wait

# --- 2. labeling ---------------------------------------------------------------------------------------
# one labeling run: model condition out [extra label args...]
label() {
  local m=$1 cond=$2 out=$3; shift 3
  local pol=${POL[$m]}
  [ -n "$pol" ] || { echo "[eval] skip $m $cond (no model)"; return 0; }
  [ -f ${out%.pt}.episodes.pt ] && return 0
  local task ssb=$BAS args=()
  case $m in beatit_*) ssb=$HYB; args+=(--wm-tube wmmin:1:0.95 --tube-calib $E/calib/$m/min_tube.json --tube-tag ${m}_s0)
                       [ -f $E/calib/$m/min_tube.json ] || { echo "[eval] skip $m $cond (no tube)"; return 0; } ;;
             rldh) args+=(--rl-dh --sonic-dir $SONIC ${RLDH_ARGS:-}) ;; esac
  if [ $cond = nt ]; then task=g1_recover_noheight_v6
  elif [ $m = beatit_thermal ]; then task=g1_recover_thermal70_v61
  else task=g1_recover_noheight_v61_heat; fi
  PYTHONPATH=$ssb python examples/label_start_states.py --policy $pol --task $task --no-push --no-obs-noise \
    --bank $BANK "${args[@]}" "$@" --out $out > ${out%.pt}.log 2>&1 \
    && echo "[eval] $m $cond done $(date +%T)" || echo "[eval] $m $cond FAILED (${out%.pt}.log)"
}
export -f label; export E BANK K HYB BAS SONIC RLDH_ARGS
export POL_beatit_noheight="${POL[beatit_noheight]}" POL_safetyppo="${POL[safetyppo]}" \
  POL_lambdareach="${POL[lambdareach]}" POL_rldh="${POL[rldh]}" POL_beatit_thermal="${POL[beatit_thermal]}"
JOBS=$E/jobs.txt; : > $JOBS
for m in beatit_noheight safetyppo lambdareach rldh; do
  echo "$m nt $E/label/${m}_nt.pt --k $K" >> $JOBS
done
for m in beatit_thermal beatit_noheight safetyppo lambdareach rldh; do
  echo "$m hot30_60 $E/label/${m}_hot30_60.pt --k $K --thermal-start-c 30 60" >> $JOBS
  echo "$m hot60_65 $E/label/${m}_hot60_65.pt --k $K --thermal-start-c 60 65" >> $JOBS
done
for T in ${SWEEP_TEMPS:-30 40 50 55 58 60 62 63 64 65 66 67 68 69}; do
  for m in beatit_thermal beatit_noheight; do
    echo "$m hot$T $E/sweep/${m}_T$T.pt --k $K_SWEEP --thermal-start-c $T" >> $JOBS
  done
done
# xargs runs each line in a fresh bash: rebuild POL there from the exported POL_<model> variables
cat $JOBS | xargs -P $PAR -L 1 bash -c 'declare -A POL=( [beatit_noheight]="$POL_beatit_noheight" \
  [safetyppo]="$POL_safetyppo" [lambdareach]="$POL_lambdareach" [rldh]="$POL_rldh" \
  [beatit_thermal]="$POL_beatit_thermal" ); label "$@"' _

# --- 3. report -----------------------------------------------------------------------------------------
python examples/v061_report.py --root $E
echo "[eval] done $(date -Is)"
