#!/bin/bash
# One agentic_qos job: a smoke test of the worst-margin conformal tube, then wmmin_jobs.txt.
#   sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=4 --mem=23G --time=02:00:00 --job-name=wmmin --output=logs/slurm/%x_%j.out configs/sweeps/shac_vs_ppo_sonic/wmmin.sh
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab && export OMP_NUM_THREADS=4
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip
S=runs/shac_vs_ppo_sonic/wmmin_smoke/shac_s0
if python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 32 --num-envs 32 --onset-s 0.02 0.02 --no-obs-noise --seed 1 --conditions sonic wmcalib:clean --out $S/calib \
   && python examples/calibrate_wm_min_tube.py --calib $S/calib/push.pt --tag shac_s0 \
   && python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 32 --num-envs 32 --onset-s 0.02 0.02 --no-obs-noise --conditions sonic wmmin:1:0.9 --tube-calib $S/calib/min_tube.json --out $S/eval; then
  echo "[job] smoke ok"
  examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/wmmin_jobs.txt --keep-going
else
  echo "[job] smoke FAILED"
fi
echo "[job] all done"
