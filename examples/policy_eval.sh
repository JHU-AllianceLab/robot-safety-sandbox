#!/usr/bin/env bash
# Policy evaluation of a fallback: exact (gamma = 1, T = 3 s, safe set K) V^pi
# and Q^pi on SONIC's states, fitted networks, and the conformal thresholds.
#
#   # once: a dedicated calibration harvest (train clips, its own seed) and the states
#   examples/policy_eval.sh states
#   # per policy (SafetyPPO1P or SafetySHAC1P checkpoint, MLP or LSTM):
#   examples/policy_eval.sh label <checkpoint.zip> <tag>
#   examples/policy_eval.sh fit <tag>
#
# Env knobs: PY (python), OUT (default data/policy_eval), NUM_ENVS (4096),
# CALIB_EPISODES (50000), K_TRAIN (2 rollouts per train state for V),
# TASK (g1_recover_lafan_k: the margin and termination of the calibration
# harvest and the labels; g1_recover_lafan for g1_recover's own g_fall), TRAJ
# (data/g1_sonic_push: the trajectory files ${TRAJ}_{train,calib,test}.pt,
# harvested under TASK; data/g1_sonic_push_gfall for g_fall).
# g_fall, e.g.: TASK=g1_recover_lafan TRAJ=data/g1_sonic_push_gfall \
#   OUT=data/policy_eval_gfall examples/policy_eval.sh states
# Cost per policy (full train split, ~0.8M V states x 2 + ~2.4M Q states x 1,
# 150 steps each): ~0.6B env-steps, ~2.4 h on one RTX 5080.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-data/policy_eval}
NUM_ENVS=${NUM_ENVS:-4096}
CALIB_EPISODES=${CALIB_EPISODES:-50000}
K_TRAIN=${K_TRAIN:-2}
TASK=${TASK:-g1_recover_lafan_k}
TRAJ=${TRAJ:-data/g1_sonic_push}

case "${1:-}" in
  states)
    if [ ! -f "${TRAJ}_calib.pt" ]; then
      $PY examples/harvest_g1_sonic_trajectories.py --task "$TASK" --library data/g1_lafan1_clips_train.pt \
        --episodes "$CALIB_EPISODES" --seed 2 --out "${TRAJ}_calib.pt" --bank-delays-s
    fi
    $PY examples/build_policy_eval_states.py --train-traj "${TRAJ}_train.pt" \
      --calib-traj "${TRAJ}_calib.pt" --test-traj "${TRAJ}_test.pt" --out-dir "$OUT"
    ;;
  label)
    POL=$2; TAG=$3
    for sp in train calib test; do
      k=1; [ "$sp" = train ] && k=$K_TRAIN
      $PY examples/label_policy_values.py --policy "$POL" --states "$OUT/${sp}_v.pt" --mode v --task "$TASK" \
        --k "$k" --num-envs "$NUM_ENVS" --out "$OUT/${sp}_v_labels_${TAG}.pt"
      $PY examples/label_policy_values.py --policy "$POL" --states "$OUT/${sp}_q.pt" --mode q --task "$TASK" \
        --k 1 --num-envs "$NUM_ENVS" --out "$OUT/${sp}_q_labels_${TAG}.pt"
    done
    ;;
  fit)
    TAG=$2
    $PY examples/fit_policy_value.py --dir "$OUT" --mode v --tag "$TAG"
    $PY examples/fit_policy_value.py --dir "$OUT" --mode q --tag "$TAG"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
