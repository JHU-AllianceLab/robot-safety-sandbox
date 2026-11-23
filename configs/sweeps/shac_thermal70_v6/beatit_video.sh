#!/bin/bash
#SBATCH --job-name=beatit_v6_video
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,a100,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# The v0.6 hybrid Beat-It (seed 1, the one under CoolMimic's SAILR run hyb1) recovering on its own for 20 s:
# record_handover_video.py on its training task g1_recover_lafan_thermal70_v6, one robot per episode:
# a held-out LAFAN clip (1 s kinematic lead-in + 0.1 s PD replay), a 0.2 s push of J N*s from one of
# the 8 suite directions, and 0.1 s after the push onset Beat-It takes over for 20 s (or until it falls).
# Start temperatures are the task's own draw (30-70 C); an overheat ends the episode like a fall.
# Output runs/shac_thermal70_v6/videos/beatit_s1_20s/ (ep*.mp4, reel.mp4, summary.txt).
#   sbatch configs/sweeps/shac_thermal70_v6/beatit_video.sh   [SEED=1 EPISODES=6]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
python -c "import safety_sb3; print('[video] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q ssb-wt-hybrid
SEED=${SEED:-1}
P=runs/shac_thermal70_v6/hybrid_s$SEED/g1_recover_thermal70_v6_safetyshac1p/final_model.zip
OUT=runs/shac_thermal70_v6/videos/beatit_s${SEED}_20s
echo "[video] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date -Is) policy $P"
python examples/record_handover_video.py --load $P --task g1_recover_lafan_thermal70_v6 \
  --episodes ${EPISODES:-6} --impulses 15 30 45 --window-s 20 --hold-s 1.5 --seed 0 --out-dir $OUT 2>&1 \
  | grep -v -E "linesearch|solver iterations limit|warn_overflow"
cat $OUT/summary.txt 2>/dev/null
ls -la $OUT
echo "[video] done $(date -Is)"
