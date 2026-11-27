#!/bin/bash
# Q^pi of the two heat-blind v0.6.1-study fallbacks (SafetyPPO safetyppo_noheight_s0, lambda-Reach
# thermal65/lambdareach_s0) for a SAILR filter "execute a iff Q(s, a) >= delta, else hand over".
# The port of the main checkout's configs/sweeps/sonic_baselines/q_eval.sbatch (2026-11-21, g1_recover)
# to their own training setting, g1_recover_noheight_v6 (sysID v0.6, critic 113-D): rollouts
# "task-agnostic first action, then the fallback" from the v6 SONIC hand-over bank
# (examples/recover_q_collect.py), then ONE backup of each arm's trained critic (q_fallback_fit.py
# --estimator critic1), delta = each critic's Wang eps = 1 % V threshold. A smoke pass first (both
# arms). Outputs runs/v061_study/q_eval/.
#   sbatch configs/sweeps/v061_study/q_eval.sbatch
#SBATCH --job-name=v061_q_eval
#SBATCH -A hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH -p rtx6000,l40s,h100,h200,b200
#SBATCH --exclude=gb210
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --output=logs/slurm/v061_q_eval_%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 MJLAB_WARP_QUIET=1
export PYTHONPATH=$HOME/safety-stable-baselines-baselines
export G1_SYSID_V061_DIR=${G1_SYSID_V061_DIR:-$HOME/G1-robot-model-with-SysID-v0.6.1}
O=runs/v061_study/q_eval; mkdir -p $O
declare -A POL=( [safetyppo]=runs/v061_study/safetyppo_noheight_s0/g1_recover_noheight_v6/final_model.zip
                 [lambdareach]=runs/thermal65/lambdareach_s0/g1_recover_noheight_v6/final_model.zip )
declare -A DELTA=( [safetyppo]=0.8334 [lambdareach]=0.8345 )
ARMS="safetyppo lambdareach"
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date -Is)"
echo "[q-eval] git $(git rev-parse --short HEAD) $(git status --short examples/recover_q_collect.py examples/q_fallback_fit.py | tr '\n' ' ')"

# load every Q file and re-score it on its data (the consumer's path: load_qnet + the stored recipe)
check_q() {  # qfile datafile
  python - "$1" "$2" <<'EOF'
import sys, torch
from robot_safety_sandbox.filters.q_net import load_qnet
qf, df = sys.argv[1], sys.argv[2]
net, meta = load_qnet(qf, "cuda:0")
for k in ("task", "policy", "policy_sha256", "obs_dim", "sanitize", "proposal_bound", "estimator", "delta"):
  assert meta.get(k) is not None, f"{qf}: no {k}"
D = torch.load(df, map_location="cpu", weights_only=False)
s, a = D["s"][:2048, 0].float().cuda(), D["a"][:2048, 0].float().cuda()
c0, c1 = meta["sanitize"]["clamp_cols"][0]
assert s.shape[1] == meta["obs_dim"] and float(s[:, c0:c1].abs().max()) <= meta["sanitize"]["clamp"] + 1e-3
assert float(a.abs().max()) <= meta["proposal_bound"] + 1e-2
q = net(s, a)
print(f"[q-eval] {qf}: task {meta['task']} obs_dim {meta['obs_dim']} clamp cols {c0}:{c1} "
      f"+-{meta['sanitize']['clamp']} pb {meta['proposal_bound']} delta {meta['delta']} "
      f"estimator {meta['estimator']} | Q(s0,a0) mean {float(q.mean()):+.3f} vs V(s0) "
      f"{float(D['v'][:2048, 0].float().mean()):+.3f}", flush=True)
print(f"[q-eval]   recipe: {meta['q_input_recipe']}", flush=True)
EOF
}

# --- smoke: both zips load, the layout checks pass, a short fit, the file loads ------------------
for ARM in $ARMS; do
  python examples/recover_q_collect.py --policy ${POL[$ARM]} --num-envs 256 --rollouts 512 \
    --out $O/smoke_data_$ARM.pt
  python examples/q_fallback_fit.py --data $O/smoke_data_$ARM.pt --estimator critic1 --steps 400 \
    --target-every 100 --batch 4096 --delta ${DELTA[$ARM]} --out $O/smoke_q_critic1_$ARM.pt
  check_q $O/smoke_q_critic1_$ARM.pt $O/smoke_data_$ARM.pt
done
echo "[q-eval] smoke passed $(date -Is)"

# --- full -------------------------------------------------------------------------------------
for ARM in $ARMS; do
  [ -f $O/data_$ARM.pt ] || python examples/recover_q_collect.py --policy ${POL[$ARM]} --out $O/data_$ARM.pt
done
for ARM in $ARMS; do
  python examples/q_fallback_fit.py --data $O/data_$ARM.pt --estimator critic1 --delta ${DELTA[$ARM]} \
    --out $O/q_critic1_$ARM.pt 2>&1 | tee $O/q_critic1_$ARM.log
  check_q $O/q_critic1_$ARM.pt $O/data_$ARM.pt
done
echo "[q-eval] done $(date -Is)"
