#!/bin/bash
#SBATCH --job-name=t65_lambda_calib
#SBATCH --account=hhu49_alliance1
#SBATCH --comment=accept_cost
#SBATCH --partition=rtx6000,l40s,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 65 C comparison, arm 3: the lambda-Reach filter's threshold delta. label_start_states.py rolls the
# deterministic fallback out once (K=1, 3 s) from ROWS start states of the v6 SONIC handover bank
# (the training world, task g1_recover_noheight_v6; each handover row brings its own push) and records
# V(s0); delta = the smallest V such that the rows with V(s0) >= delta fail <= 5 % of the time (the
# empirical row-level calibration of the earlier baselines). The v6 bank has no test split: these
# are training rows (optimistic). -> $RUN/delta.json
#   RUN=runs/thermal65/lambdareach_s0 sbatch configs/sweeps/thermal65/lambda_calib.sh   [ROWS=32768 EPS=0.05]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/safety-stable-baselines-baselines${PYTHONPATH:+:$PYTHONPATH}
RUN=${RUN:-runs/thermal65/lambdareach_s0}
P=$RUN/g1_recover_noheight_v6/final_model.zip
[ -f $P ] || { echo "missing $P"; exit 1; }
OUT=$RUN/calib
mkdir -p $OUT
echo "[calib] $(hostname) $(date -Is) policy $P rows ${ROWS:-32768}"
python examples/label_start_states.py --policy $P --task g1_recover_noheight_v6 \
  --bank data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt --k 1 --no-push --episode-s 3.0 \
  --max-rows ${ROWS:-32768} --num-envs 4096 --out $OUT/labeled.pt 2>&1 \
  | grep -v -E "linesearch|solver iterations limit|warn_overflow" | tail -40
python - "$OUT/labeled.pt" "$RUN/delta.json" "${EPS:-0.05}" <<'PY'
import json, sys, torch
src, dst, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
b = torch.load(src, map_location="cpu", weights_only=False)
d = b if isinstance(b, dict) else vars(b)
n, v, ps = d["n_label"].float(), d["value0"].float(), d["p_surv"].float()
m = n > 0
v, fail = v[m], 1.0 - ps[m]
order = torch.argsort(v, descending=True)
vs, fs = v[order], fail[order]
cum = torch.cumsum(fs, 0) / torch.arange(1, len(fs) + 1)
ok = torch.nonzero(cum <= eps).flatten()
# the largest prefix (lowest threshold) whose failure fraction is still <= eps
k = int(ok.max()) if ok.numel() else -1
delta = float(vs[k]) if k >= 0 else float("inf")
rec = {"delta": delta, "eps": eps, "rows": int(m.sum()), "accepted": int(k + 1),
       "accepted_frac": (k + 1) / int(m.sum()), "fail_overall": float(fail.mean()),
       "fail_accepted": float(cum[k]) if k >= 0 else None,
       "V_quantiles": [float(x) for x in torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))],
       "source": src}
json.dump(rec, open(dst, "w"), indent=1)
print("[calib]", json.dumps(rec))
PY
echo "[calib] done $(date -Is)"
