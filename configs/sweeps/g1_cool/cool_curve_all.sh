#!/bin/bash
# Every motor's temperature under the first cooling policy vs the thermal70 SHAC fallback (the user,
# 2026-11-19: "can you also check the temperature of the arm?"): cool_curve.py's 10-min unpushed
# run from all four ankle windings at 65 C (fresh / soaked ankle housings), now recording every
# motor; then the ankle plot and examples/cool_motor_report.py (arms; and the whole body).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,l40s,h100,a100,h200 --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 --job-name=cool_curve_all --output=logs/slurm/%x_%j.out configs/sweeps/g1_cool/cool_curve_all.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export PYTHONFAULTHANDLER=1
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
POL=(--policy cooling runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip
     --policy fallback runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip)
O=runs/g1_cool/cool_curve_all
python examples/cool_curve.py "${POL[@]}" --num-envs 16 --duration-s 65 --extrapolate-s 120 --out $O/smoke
python examples/cool_motor_report.py $O/smoke/curve.pt
python examples/cool_curve.py "${POL[@]}" --num-envs 512 --duration-s 600 --out $O
python examples/plot_cool_curve.py $O/curve.pt
python examples/cool_motor_report.py $O/curve.pt
python examples/cool_motor_report.py $O/curve.pt --motors . --out $O/motors_all.png
echo "[job] cool_curve_all done"
