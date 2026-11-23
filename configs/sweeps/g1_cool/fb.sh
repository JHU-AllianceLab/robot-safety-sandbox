#!/bin/bash
# COOLING POLICY, round 3 (the user, 2026-11-19): take over smoothly from the fallback; every
# motor's heat costs once it is above 50 C. In one job:
#   1. harvest smoke (64 envs, 3 s) + the fallback bank (examples/harvest_cool_fallback_states.py:
#      the thermal70 SHAC fallback after minor SONIC handovers, pushes < 1 m/s, a row every 1 s
#      from 0.5 to 15 s) -> data/g1_cool_fallback_bank.pt;
#   2. task checks (examples/check_g1_cool.py --task g1_cool_thermal70_fb);
#   3. PPO from scratch, 500M steps (capped at TRAIN_MIN minutes; then the last checkpoint)
#      -> runs/g1_cool/fb_s0;
#   4. evals of the new policy:
#      a. fallback 10 s -> new cooling policy, the same seed-0 starts as the first policy's
#         switch test (job 1968306): video + 512-env stats -> runs/g1_cool/video/fb_switch10s_seed0*;
#      b. 10-min all-motor curve from fallback states (ankles 65 C, other motors 30 C, no pushes):
#         new policy vs the first cooling policy vs the fallback -> runs/g1_cool/fb_s0/cool_curve_all.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h200,h100 --gres=gpu:1 --cpus-per-task=16 --time=04:00:00 --job-name=g1_cool_fb --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/fb.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
FB=runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
PILOT=runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip
BANK=data/g1_cool_fallback_bank.pt
OUT=runs/g1_cool/fb_s0
[[ -f data/g1_cool_handover_bank_dv1p0.pt ]] || python examples/build_cool_bank.py --max-dv 1.0
python examples/harvest_cool_fallback_states.py --fallback $FB --num-envs 64 --duration-s 3 \
  --out runs/g1_cool/fb_smoke/bank.pt --force
[[ -f $BANK ]] || python examples/harvest_cool_fallback_states.py --fallback $FB --num-envs 4096 --out $BANK
python examples/check_g1_cool.py --task g1_cool_thermal70_fb --num-envs 512 --steps 300 --device cuda:0
set +e
timeout ${TRAIN_MIN:-170}m python examples/train.py --config configs/g1_cool_thermal70_fb_ppo.yaml \
  --seed 0 --steps ${FB_STEPS:-500000000} --out $OUT --no-wandb
rc=$?
set -e
echo "[job] train exit $rc"
F=$OUT/g1_cool_thermal70_fb/final_model.zip
[[ -f $F ]] || F=$(ls -t $OUT/g1_cool_thermal70_fb/checkpoints/model_*_steps.zip | head -1)
echo "[job] eval $F"
python examples/cool_video.py --policy $F --fallback $FB --switch-s 10 --duration-s 30 \
  --panels 4 --num-envs 512 --seed 0 --task g1_cool_thermal70 \
  --label "Fallback (SHAC) for 10 s, then the NEW cooling policy (trained from fallback states, all motors above 50 C cost)" \
  --out runs/g1_cool/video/fb_switch10s_seed0.mp4
python examples/cool_curve.py --policy cooling_fb $F --policy cooling $PILOT --policy fallback $FB \
  --task g1_cool_thermal70_fb --num-envs 512 --duration-s 600 --out $OUT/cool_curve_all
python examples/plot_cool_curve.py $OUT/cool_curve_all/curve.pt
python examples/cool_motor_report.py $OUT/cool_curve_all/curve.pt
python examples/cool_motor_report.py $OUT/cool_curve_all/curve.pt --motors . --out $OUT/cool_curve_all/motors_all.png
echo "[job] g1_cool fb done"
