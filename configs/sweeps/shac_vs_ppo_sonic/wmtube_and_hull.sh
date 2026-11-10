#!/bin/bash
# One agentic_qos job: a smoke test of the calibrated world-model tube, then
# wmtube_jobs.txt (calibrate + evaluate, 3 seeds), then rollout_hull_jobs.txt.
#   sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=4 --mem=23G --time=04:00:00 --job-name=wmtube_hull --output=logs/slurm/%x_%j.out configs/sweeps/shac_vs_ppo_sonic/wmtube_and_hull.sh
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab && export OMP_NUM_THREADS=4
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip
S=runs/shac_vs_ppo_sonic/wmtube_smoke/shac_s0
if python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 32 --num-envs 32 --onset-s 0.02 0.02 --seed 1 --conditions sonic wmcalib:16 --out $S/calib \
   && python examples/calibrate_wm_tube.py --calib $S/calib/push.pt --tag shac_s0 \
   && python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 32 --num-envs 32 --onset-s 0.02 0.02 --conditions sonic wmtube:1:16:0.95 --tube-calib $S/calib/tube.json --out $S/eval; then
  echo "[job] tube smoke ok"
  examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/wmtube_jobs.txt --keep-going
else
  echo "[job] tube smoke FAILED"
fi
examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/rollout_hull_jobs.txt --keep-going
echo "[job] all done"
