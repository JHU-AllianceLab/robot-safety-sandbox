#!/bin/bash
# A smoke test of +latch, the hand-over count and the filter videos, then wmlatch_jobs.txt.
#   sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=32 --mem=96G --time=01:30:00 --job-name=wmlatch --output=logs/slurm/%x_%j.out configs/sweeps/shac_vs_ppo_sonic/wmlatch.sh
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip
S=runs/shac_vs_ppo_sonic/wmlatch_smoke/shac_s0
if python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 64 --num-envs 32 --onset-s 0.02 0.02 --no-obs-noise --rollout-workers 8 --conditions sonic fallback wmmin:0.4:0.95 wmmin:0.4:0.95+latch rollout:1 --tube-calib runs/shac_vs_ppo_sonic/wmmin_calib/shac_s0/min_tube.json --video-episodes 1 40 --out $S \
   && ls $S/videos/reel.mp4; then
  echo "[job] smoke ok"
  examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/wmlatch_jobs.txt --keep-going
else
  echo "[job] smoke FAILED"
fi
echo "[job] all done"
