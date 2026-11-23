#!/bin/bash
# COOLING POLICY, round 2 (the user, 2026-11-18): the 500M pilot cooled well but fell from
# mocap starts and from pushes; train on g1_cool_thermal70_mix instead -- 30 % LAFAN train
# mocap spawns + 70 % handover rows kicked <= 1 m/s, pushes 0.19-0.97 m/s (small pushes
# only, by the user's choice; no bigger ones). Same PPO settings as the pilot. Then the
# pilot's eval (cool_fallback.sh: held-out mocap spawns, fresh / soaked hot ankles, one
# kick 0-4 m/s) on the final model, so the two rounds and the fallback compare directly.
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h100,h200,a100 --gres=gpu:1 --cpus-per-task=16 --time=04:00:00 --job-name=g1_cool_mix --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/mix.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ -f data/g1_cool_handover_bank_dv1p0.pt ]] || python examples/build_cool_bank.py --max-dv 1.0
[[ -f data/g1_cool_mocap_bank.pt ]] || python examples/build_cool_bank.py --mocap
python examples/check_g1_cool.py --task g1_cool_thermal70_mix --num-envs 512 --steps 300 --device cuda:0
OUT=runs/g1_cool/mix_s0
python examples/train.py --config configs/g1_cool_thermal70_mix_ppo.yaml --seed 0 \
  --steps ${MIX_STEPS:-300000000} --out $OUT --no-wandb
F=$OUT/g1_cool_thermal70_mix/final_model.zip
[[ -f $F ]] || F=$(ls -t $OUT/g1_cool_thermal70_mix/checkpoints/model_*_steps.zip | head -1)
echo "[job] eval $F"
POLICY=$F OUT_ROOT=$OUT/cool_eval_final TAG=cool_mix_s0_final bash configs/sweeps/shac_thermal70/cool_fallback.sh
echo "[job] g1_cool mix done"
