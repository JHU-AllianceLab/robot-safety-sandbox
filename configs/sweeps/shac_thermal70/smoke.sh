#!/bin/bash
# The 70 C-margin SHAC, step 0: a smoke test on the existing identified-G1 bank
# (configs/sweeps/shac_thermal/data_jobs.txt) -- a short SHAC run on g1_recover_thermal70
# (171-D world-model state, 127-D actor obs, 70 C in the margin and the termination, no
# barrier), the CPU rollout's margin and winding temperatures against the live env's on
# g1_recover_lafan_thermal70 (validate_cpu_rollout.py), and the report pipeline
# (wmreport_jobs.txt: calibration, tube fit, every filter, wm_filter_report.py) on 64
# episodes, with checks that the 70 C limit is the one in force. The training array
# (jobs.txt) and the report can wait on it with --dependency=afterok. Conda env mjlab316.
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100,a100,h200 --gres=gpu:1 --cpus-per-task=32 --mem=96G --time=02:00:00 --job-name=thermal70_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70/smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

BANK=data/g1_sonic_push_gfall_sysid316_train_bank_net.pt
[[ -f $BANK ]] || { echo "[smoke] FAILED: no $BANK (run configs/sweeps/shac_thermal/data_jobs.txt)"; exit 1; }

S=runs/shac_thermal70/smoke
rm -rf $S && mkdir -p $S

# 1. a short SHAC run on the 70 C task
python examples/train.py --config configs/g1_recover_sonic_thermal70_safetyshac1p.yaml --seed 0 \
  --out $S/train --num-envs 1024 --steps 3000000 --wm-warmup-iters 50 --actor-delay-iters 10 \
  --lr-decay-iters 1200 --checkpoint-every 1000000000 --video-interval 1000000000 --no-wandb \
  2>&1 | tee $S/train.log
grep -q "state = the 'critic' group (171 dims); actor obs 127 dims" $S/train.log \
  || { echo "[smoke] FAILED: the world-model state is not the 171-D critic group / actor 127-D"; exit 1; }
grep -q "barrier None; overheat termination True at 70 C; margin thermal term {'limit_c': 70.0, 'scale_c': 10.0}" $S/train.log \
  || { echo "[smoke] FAILED: the margin / termination are not at 70 C (or a barrier is on)"; exit 1; }
grep -q "'start_level_c': (30.0, 70.0)" $S/train.log \
  || { echo "[smoke] FAILED: the start levels are not U(30, 70)"; exit 1; }
P=$S/train/g1_recover_thermal70_safetyshac1p/final_model.zip
[[ -f $P ]] || { echo "[smoke] FAILED: no $P"; exit 1; }

# 2. the CPU rollout reproduces the env: the margin (its 70 C term binds near the start
#    temperatures' top, so a CPU limit of 85 C would show here) and winding temperatures
python examples/validate_cpu_rollout.py --policy $P --task g1_recover_lafan_thermal70 --num-envs 64 \
  --rounds 2 --horizon-s 1.0 --out $S/validate.json
python - <<PY
import json
r = json.load(open("$S/validate.json"))
g, ro, tw = r["one_step"]["g"], r["rollout"], r["one_step"]["Tw"]
print("[smoke] one-step |g_cpu - g_gpu| max", g["max"], "p99", g["p99"], "| |Tw_cpu - Tw_gpu| max",
      tw["max"], "| rollout verdict agreement", ro["agreement"], "gmin diff median", ro["gmin_abs_diff_median"])
assert g["p99"] < 0.02, "the CPU margin departs from the env's"
assert tw["p99"] < 0.01, "the CPU heat model departs from the env's"
PY

# 3. the report pipeline on 64 episodes, every filter of the report
F="--onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise"
C="--sonic-dir $HOME/gear_sonic_hf --task g1_recover_lafan_thermal70 --hip mjlab --suites push --num-envs 32"
python examples/eval_sonic_qcbf.py --policy $P --tag smoke $C --episodes 64 $F --seed 1 \
  --conditions sonic wmcalib:clean --out $S/wmreport/calib/smoke
python examples/calibrate_wm_min_tube.py --calib $S/wmreport/calib/smoke/push.pt --tag smoke
python examples/eval_sonic_qcbf.py --policy $P --tag smoke $C --episodes 64 $F --rollout-workers 30 \
  --conditions sonic fallback rollout:1 wmmin:0.4:0.95 wmmin:1:0.95 wmmin:3:0.95 wmmin:1:0.95+dwell1 \
  wmmin:1:0.95+dwell2 wmmin:1:0.95+latch --tube-calib $S/wmreport/calib/smoke/min_tube.json \
  --out $S/wmreport/eval/smoke
python examples/wm_filter_report.py --root $S/wmreport --tags smoke
python - <<PY
import torch
d = torch.load("$S/wmreport/calib/smoke/push.pt", map_location="cpu", weights_only=False)
r = d["conditions"]["wmcalib:clean"]
print("[smoke] calib recorded", int(torch.isfinite(r["calib_real"][:, 0]).sum()), "of", r["calib_real"].shape[0])
assert torch.isfinite(r["calib_real"][:, 0]).any() and torch.isfinite(r["calib_pred"][:, 0, 0]).any()
# the 70 C termination: an episode overheats exactly when its hottest winding reached 70 C
e = torch.load("$S/wmreport/eval/smoke/push.pt", map_location="cpu", weights_only=False)["conditions"]
for nm in ("sonic", "fallback", "rollout:1", "wmmin:1:0.95"):
  c = e[nm]
  oh, pk = c["overheated"].bool(), c["tw_peak_c"]
  done = torch.isfinite(pk)
  print(f"[smoke] {nm}: survived {c['survived'].float().mean():.3f}, overheated "
        f"{oh.float().mean():.3f}, hottest winding at spawn {c['tw_start_max_c'].mean():.1f} C, "
        f"peak {pk[done].mean():.1f} C (max of the episodes that did not overheat "
        f"{pk[done & ~oh].max() if (done & ~oh).any() else float('nan'):.2f} C)")
  assert float(c["tw_start_max_c"].max()) <= 70.0, "a start temperature above the limit"
  # an overheated episode's peak is reported as the termination's limit: 70, not 85
  assert bool(((pk[oh] >= 70.0) & (pk[oh] < 71.0)).all()), f"{nm}: overheated peaks {pk[oh]}"
  assert bool((pk[done & ~oh] < 70.0).all()), f"{nm}: a winding reached 70 C without ending the episode"
  assert "cbf_clip_frac" not in c, f"{nm}: a barrier is on"
rise = float((e["sonic"]["tw_peak_c"] - e["sonic"]["tw_start_max_c"]).nanmean())
assert rise > 0.05, f"SONIC's windings did not heat ({rise:.3f} C): is the actuator force read?"
same = all(torch.equal(e[nm]["tw_start_max_c"], e["sonic"]["tw_start_max_c"]) for nm in e)
print("[smoke] start temperatures identical across conditions:", same)
PY
echo "[job] smoke ok"
