#!/bin/bash
# Per-task recalibration for SAILR on dance / fight-and-sport (the user, 2026-11-25; memory sailr-dance-fight):
# a fresh SONIC dataset that tracks exactly the task's clips with pushes |dv| ~ U(0, 2.5) m/s, then
#   1. the heat-aware Beat-It's world-model tube on those clips (eval_sonic_qcbf wmcalib:clean +
#      calibrate_wm_min_tube; v0.6.1 study recipe, configs/sweeps/v061_study/eval.sh calib(), with the
#      task library and dv 0-2.5);
#   2. labels on the new bank (examples/label_start_states.py, K rollouts per row, no push, no obs noise):
#        beatit_thermal hot30_60  soaked T0 ~ U(30, 60) C, fail = fall or 70 C (as the walk threshold 0.665)
#        lambdareach    nt        temperature-agnostic world, fail = fall
#   3. examples/v061_report.py -> Wang eps 1 % / 5 % thresholds (report.json).
#   4. (2026-11-26) the FAST monitor's thresholds (k imagined steps + V_(50-k)): examples/fast_beatit_calibrate.py on
#      the same labels (labelled with --dump-wm) -> thresholds.json "beatit_fast". Value functions: FAST_VALUES
#      (default runs/fast_beatit/v_teacher_pushbank/values.pt: the push-bank V, task-agnostic; also on Hugging Face).
#   5. the robot-ready STUDENT filter (no foot inputs): labels with the student world model (--no-critic-foot) and its
#      full + fast thresholds -> "beatit_student" (STUDENT_VALUES, STUDENT_POLICY, STUDENT_TUBE).
# Outside this cluster: BANK_W = the Hugging Face release's bank_W.pt (the train bank's feature scales for the
# thinning step), and build data/g1_lafan1_clips.pt from the LAFAN1 CSVs (examples/build_g1_recover_mocap_states.py --library).
# Recipe of the bank = configs/sweeps/v061_study/eval_bank.sh (8000 episodes, rows 0/0.1/0.2/0.3 s after the
# kick, thinned with the v6 TRAIN bank's weights), library cut to the task clips, dv U(0, 2.5).
#   TASK_NAME=dance CLIPS="dance1_subject1 dance2_subject1" sbatch -A hhu49_alliance1 --comment=accept_cost \
#     -p rtx6000,l40s,h100,h200,b200 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=04:00:00 \
#     --job-name=sdf_calib_dance --output=logs/slurm/%x_%j.out --export=ALL configs/sweeps/sailr_df/calib.sh
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
: "${TASK_NAME:?dance|fight}" "${CLIPS:?space-separated clip names}"
HYB=$HOME/ssb-wt-hybrid; BAS=$HOME/safety-stable-baselines-baselines; SONIC=$HOME/gear_sonic_hf
E=runs/sailr_df/calib/$TASK_NAME; mkdir -p $E/calib/beatit_thermal $E/label
LIB=data/g1_lafan1_clips_$TASK_NAME.pt
D=data/g1_sonic_push_gfall_sysid316v6_dv25_$TASK_NAME
K=${K:-4}; EPISODES=${EPISODES:-8000}; SEED=${SEED:-3}
BT=runs/v061_study/beatit_thermal_s0/g1_recover_thermal70_v61_safetyshac1p/final_model.zip
LR=runs/thermal65/lambdareach_s0/g1_recover_noheight_v6/final_model.zip
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is) task $TASK_NAME clips $CLIPS"

# --- 0. library + SONIC dataset ---------------------------------------------------------------------------
[ -f $LIB ] || python examples/subset_clip_library.py --library data/g1_lafan1_clips.pt --clips $CLIPS --out $LIB
[ -f $D.pt ] || python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan_sysid_v6 --hip mjlab \
  --library $LIB --sonic-dir $SONIC --episodes $EPISODES --seed $SEED --dv-range 0 2.5 --bank-delays-s --out $D.pt
[ -f ${D}_bank.pt ] || python examples/harvest_g1_sonic_trajectories.py --from-traj $D.pt \
  --bank-delays-s 0 0.1 0.2 0.3 --bank-out ${D}_bank.pt
[ -f ${D}_bank_net.pt ] || python examples/dedup_start_bank.py --bank ${D}_bank.pt \
  --W-from ${BANK_W:-data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt}
[ -f ${D}_bank_net.pt ] || { echo "[calib] no bank"; exit 1; }
echo "[calib] $(date -Is) bank ${D}_bank_net.pt ready"

# --- 1. heat-aware Beat-It tube on the task clips ---------------------------------------------------------
d=$E/calib/beatit_thermal
if [ ! -f $d/min_tube.json ]; then
  PYTHONPATH=$HYB python examples/eval_sonic_qcbf.py --policy $BT --tag beatit_thermal_s0 --sonic-dir $SONIC \
    --task g1_recover_lafan_thermal70_v61 --hip mjlab --library $LIB --suites push --episodes 2048 --num-envs 256 \
    --onset-s 0.02 0.02 --dv-range 0 2.5 --dv-bins 0 1 2 2.5 --no-obs-noise --seed 1 \
    --conditions sonic wmcalib:clean --out $d > $d.log 2>&1 \
  && PYTHONPATH=$HYB python examples/calibrate_wm_min_tube.py --calib $d/push.pt --tag beatit_thermal_s0 >> $d.log 2>&1 \
  || { echo "[calib] tube FAILED ($d.log)"; tail -20 $d.log; exit 1; }
fi
echo "[calib] tube 1 s/95%: $(python -c "import json; print(json.load(open('$d/min_tube.json'))['min_eps']['0.95']['50'])")"

# --- 2. labels ----------------------------------------------------------------------------------------------
lab() {  # name task out ssb [args...]
  local n=$1 task=$2 out=$3 ssb=$4; shift 4
  [ -f ${out%.pt}.episodes.pt ] && return 0
  PYTHONPATH=$ssb python examples/label_start_states.py --policy ${!n} --task $task --no-push --no-obs-noise \
    --bank ${D}_bank_net.pt --k $K "$@" --out $out > ${out%.pt}.log 2>&1 \
    && echo "[calib] label $(basename $out) done $(date +%T)" || echo "[calib] label $out FAILED (${out%.pt}.log)"
}
lab BT g1_recover_thermal70_v61 $E/label/beatit_thermal_hot30_60.pt $HYB --thermal-start-c 30 60 \
  --wm-tube wmmin:1:0.95 --tube-calib $d/min_tube.json --tube-tag beatit_thermal_s0 --dump-wm &
lab LR g1_recover_noheight_v6 $E/label/lambdareach_nt.pt $BAS &
wait

# --- 3. report ----------------------------------------------------------------------------------------------
python examples/v061_report.py --root $E || echo "[calib] report FAILED"
python - <<EOF
import json
r = json.load(open("$E/report.json"))
t = json.load(open("$d/min_tube.json"))["min_eps"]["0.95"]["50"]
bt, lr = r["label"]["hot30_60"]["beatit_thermal"], r["label"]["nt"]["lambdareach"]
out = {"task": "$TASK_NAME", "clips": "$CLIPS".split(), "bank": "${D}_bank_net.pt", "tube_eps_1s_95": t,
       "beatit_thermal": {"wang_0.01": bt["wang_0.01"], "wang_0.05": bt["wang_0.05"],
                          "raw_threshold_eps1": None if bt["wang_0.01"]["delta"] is None else bt["wang_0.01"]["delta"] + t},
       "lambdareach": {"wang_0.01": lr["wang_0.01"], "wang_0.05": lr["wang_0.05"]}}
json.dump(out, open("$E/thresholds.json", "w"), indent=1)
print("[calib] thresholds:", json.dumps(out, indent=1))
EOF
# --- 4. the fast monitor's thresholds (needs the --dump-wm labels and its push-bank value functions) ----------
FV=${FAST_VALUES:-runs/fast_beatit/v_teacher_pushbank/values.pt}
if [ -f "$FV" ]; then
  PYTHONPATH=$HYB python examples/fast_beatit_calibrate.py --labels $E/label/beatit_thermal_hot30_60.episodes.pt \
    --values $FV --ks 10 15 25 --thresholds $E/thresholds.json --key beatit_fast || echo "[calib] fast-monitor calibration FAILED"
else
  echo "[calib] no fast-monitor value functions ($FV): fast thresholds skipped"
fi
# --- 5. the robot-ready STUDENT filter (no foot inputs): its labels and its full + fast thresholds -------------
SV=${STUDENT_VALUES:-runs/fast_beatit/v_student_pushbank/values.pt}
STU=${STUDENT_POLICY:-runs/beatit_student_v061/nofoot_s0/g1_recover_thermal70_v61_safetyshac1p/final_model.zip}
STUTUBE=${STUDENT_TUBE:-runs/beatit_student_v061/calib/nofoot_s0/min_tube.json}
if [ -f "$SV" ] && [ -f "$STU" ]; then
  [ -f $E/label/beatit_student_hot30_60.episodes.pt ] || PYTHONPATH=$HYB python examples/label_start_states.py \
    --policy $STU --task g1_recover_thermal70_v61 --no-critic-foot --no-push --no-obs-noise --thermal-start-c 30 60 \
    --bank ${D}_bank_net.pt --k $K --wm-tube wmmin:1:0.95 --tube-calib $STUTUBE --tube-tag nofoot_s0 --dump-wm \
    --out $E/label/beatit_student_hot30_60.pt > $E/label/beatit_student_hot30_60.log 2>&1
  PYTHONPATH=$HYB python examples/fast_beatit_calibrate.py --labels $E/label/beatit_student_hot30_60.episodes.pt \
    --values $SV --ks 10 15 25 --thresholds $E/thresholds.json --key beatit_student || echo "[calib] student calibration FAILED"
else
  echo "[calib] no student policy / value functions ($STU, $SV): student thresholds skipped"
fi
echo "[calib] done $(date -Is)"
