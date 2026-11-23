#!/bin/bash
# Does a jerky SONIC -> cooling handover add heat? (the user's question, 2026-11-18)
# examples/cool_handover_heat.py on the given cooling policy (default: the 500M pilot).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p l40s,a100,h100,h200,rtx6000 --gres=gpu:1 -c 8 --mem=64G \
#     -t 00:45:00 -J g1_cool_hoheat -o logs/slurm/%x_%j.out configs/sweeps/g1_cool/handover_heat.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
POLICY=${POLICY:-runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip}
TAG=${TAG:-pilot_s0}
python examples/cool_handover_heat.py --policy "$POLICY" --num-envs ${NUM_ENVS:-4096} \
  --out runs/g1_cool/handover_heat/$TAG.json
