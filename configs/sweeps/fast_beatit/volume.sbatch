#!/bin/bash
#SBATCH --job-name=fast_beatit_vol
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,a100,h100,b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Fast Beat-It monitor (k imagined steps + V_(50-k) from real rollouts) vs the full 50-step monitor, calibrated on
# each SAILR task's own bank (the user, 2026-11-26). Per task (dance, fight, bentknee), in parallel:
#   1. a TRAIN bank of the task clips: SONIC harvest, pushes U(0, 2.5) m/s, 8000 episodes, seed 4 (the calibration
#      bank is seed 3), delays 0 / 0.1 / 0.2 / 0.3 s, thinned like the calibration bank;
#   2. the calibration (test) bank relabelled as in configs/sweeps/sailr_df/calib.sh (hot30_60, K 4, the task's 1 s
#      tube) with --dump-wm, so the monitor can be re-scored offline;
#   3. the fallback's REAL rollouts from the train bank (coolmimic scripts/real_margin_collect.py, 4 s cap).
# Then examples/fast_beatit_volume.py: V_h fitted on the generic push train bank's real rollouts (job 2140882) +
# the three task train banks, Wang eps 1 % / 5 % per task for k in {0, 5, 10, 15, 25} and the full monitor.
# Output: runs/fast_beatit/{bank,real,test}/<task>/, runs/fast_beatit/volume/{volume.json,values.pt}.
#   sbatch configs/sweeps/fast_beatit/volume.sbatch
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 SANDBOX=$PWD
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
HYB=$(realpath $HOME/ssb-wt-hybrid); COOL=$HOME/coolmimic; SONIC=$HOME/gear_sonic_hf
export PYTHONPATH=$HYB
BT=runs/v061_study/beatit_thermal_s0/g1_recover_thermal70_v61_safetyshac1p/final_model.zip
GEN_REAL=runs/beatit_student_v061/trunc_grid/real/real_rollouts.pt
O=runs/fast_beatit
F='linesearch|solver iterations limit|warn_overflow|FutureWarning|warnings.warn'
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is)" \
  "sandbox $(git rev-parse --short HEAD) (+$(git status --short --untracked-files=no | wc -l) modified)"
[ -f $GEN_REAL ] || { echo "no $GEN_REAL"; exit 1; }

task() {  # name
  local t=$1 lib=data/g1_lafan1_clips_$1.pt
  local D=data/g1_sonic_push_gfall_sysid316v6_dv25_$1 D4=data/g1_sonic_push_gfall_sysid316v6_dv25_$1_seed4
  local tube=runs/sailr_df/calib/$1/calib/beatit_thermal/min_tube.json
  mkdir -p $O/bank/$t $O/real/$t $O/test/$t
  # 1. train bank
  if [ ! -f ${D4}_bank_net.pt ]; then
    { python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan_sysid_v6 --hip mjlab --library $lib \
        --sonic-dir $SONIC --episodes 8000 --seed 4 --dv-range 0 2.5 --bank-delays-s --out $D4.pt \
      && python examples/harvest_g1_sonic_trajectories.py --from-traj $D4.pt --bank-delays-s 0 0.1 0.2 0.3 \
        --bank-out ${D4}_bank.pt \
      && python examples/dedup_start_bank.py --bank ${D4}_bank.pt --W-from data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt
    } > $O/bank/$t/harvest.log 2>&1 || { echo "[$t] harvest FAILED"; tail -20 $O/bank/$t/harvest.log; return 1; }
    rm -f $D4.pt   # the raw trajectories (0.5 GB); the bank keeps what is needed
  fi
  echo "[$t] train bank ${D4}_bank_net.pt $(date +%T)"
  # 2. test-bank labels with dumps
  [ -f $O/test/$t/test.episodes.pt ] || python examples/label_start_states.py --policy $BT --task g1_recover_thermal70_v61 \
      --no-push --no-obs-noise --thermal-start-c 30 60 --bank ${D}_bank_net.pt --k 4 \
      --wm-tube wmmin:1:0.95 --tube-calib $tube --tube-tag beatit_thermal_s0 --dump-wm --out $O/test/$t/test.pt 2>&1 \
      | grep -v -E "$F" > $O/test/$t/test.log
  [ -f $O/test/$t/test.episodes.pt ] || { echo "[$t] test labels FAILED"; tail -20 $O/test/$t/test.log; return 1; }
  grep -E "^\[label\] (score|wrote)" $O/test/$t/test.log | cut -c1-200 | sed "s/^/[$t] /"
  # 3. real rollouts from the train bank
  [ -f $O/real/$t/real_rollouts.pt ] || python $COOL/scripts/real_margin_collect.py --rec-out $O/real/$t/real_rollouts.pt \
      --keep 0.25 --hmax 50 --policy $BT --task g1_recover_thermal70_v61 --no-push --no-obs-noise \
      --thermal-start-c 30 60 --bank ${D4}_bank_net.pt --k 1 --seed 11 --episode-s 4 --out $O/real/$t/label.pt 2>&1 \
      | grep -v -E "$F" > $O/real/$t/collect.log
  [ -f $O/real/$t/real_rollouts.pt ] || { echo "[$t] real rollouts FAILED"; tail -20 $O/real/$t/collect.log; return 1; }
  grep -E "^\[rec\] (critic|wrote)" $O/real/$t/collect.log | cut -c1-200 | sed "s/^/[$t] /"
  echo "[$t] done $(date +%T)"
}

rc=0
task dance & p1=$!
task fight & p2=$!
task bentknee & p3=$!
wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
[ $rc -eq 0 ] || { echo "a task FAILED"; exit 1; }
echo "== analysis $(date -Is)"
mkdir -p $O/volume
python examples/fast_beatit_volume.py --real $GEN_REAL $O/real/dance/real_rollouts.pt $O/real/fight/real_rollouts.pt \
  $O/real/bentknee/real_rollouts.pt --tests dance:$O/test/dance/test.episodes.pt fight:$O/test/fight/test.episodes.pt \
  bentknee:$O/test/bentknee/test.episodes.pt --generic-only --out $O/volume 2>&1 | grep -v -E "$F" | tee $O/volume/volume.log
[ -f $O/volume/volume.json ] || { echo "analysis FAILED"; exit 1; }
echo "done $(date -Is)"
