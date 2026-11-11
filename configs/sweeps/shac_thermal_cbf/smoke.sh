#!/bin/bash
# The temperature-barrier SHAC, step 0: a smoke test on the existing identified-G1 bank
# (configs/sweeps/shac_thermal/data_jobs.txt) -- a short SHAC run on g1_recover_thermal_cbf
# (171-D world-model state, 127-D actor obs, the 70 C barrier on the joint targets), the CPU
# rollout's margin, winding temperatures and barrier against the live env's on
# g1_recover_lafan_thermal_cbf (validate_cpu_rollout.py), and the report pipeline
# (wmreport_jobs.txt: calibration, tube fit, every filter, wm_filter_report.py) on 64
# episodes, with checks that the barrier holds. The training array (jobs.txt) can wait on it
# with --dependency=afterok. Conda env mjlab316 (MuJoCo 3.16).
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100,a100,h200 --gres=gpu:1 --cpus-per-task=32 --mem=96G --time=02:00:00 --job-name=thermal_cbf_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal_cbf/smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

BANK=data/g1_sonic_push_gfall_sysid316_train_bank_net.pt
[[ -f $BANK ]] || { echo "[smoke] FAILED: no $BANK (run configs/sweeps/shac_thermal/data_jobs.txt)"; exit 1; }

S=runs/shac_thermal_cbf/smoke
rm -rf $S && mkdir -p $S

# 1. a short SHAC run on the barrier task
python examples/train.py --config configs/g1_recover_sonic_thermal_cbf_safetyshac1p.yaml --seed 0 \
  --out $S/train --num-envs 1024 --steps 3000000 --wm-warmup-iters 50 --actor-delay-iters 10 \
  --lr-decay-iters 1200 --checkpoint-every 1000000000 --video-interval 1000000000 --no-wandb \
  2>&1 | tee $S/train.log
grep -q "state = the 'critic' group (171 dims); actor obs 127 dims" $S/train.log \
  || { echo "[smoke] FAILED: the world-model state is not the 171-D critic group / actor 127-D"; exit 1; }
grep -q "barrier {'limit_c': 70.0, 'gamma': 1.0}; overheat termination False" $S/train.log \
  || { echo "[smoke] FAILED: the barrier / termination are not the design's"; exit 1; }
P=$S/train/g1_recover_thermal_cbf_safetyshac1p/final_model.zip
[[ -f $P ]] || { echo "[smoke] FAILED: no $P"; exit 1; }

# 2. the CPU rollout reproduces the env: margin, winding temperatures, and (through qpos
#    after a step) the barrier's clip
python examples/validate_cpu_rollout.py --policy $P --task g1_recover_lafan_thermal_cbf --num-envs 64 \
  --rounds 2 --horizon-s 1.0 --out $S/validate.json
python - <<PY
import json
r = json.load(open("$S/validate.json"))
g, ro, tw, qp = r["one_step"]["g"], r["rollout"], r["one_step"]["Tw"], r["one_step"]["qpos"]
print("[smoke] one-step |g_cpu - g_gpu| p99", g["p99"], "| |Tw_cpu - Tw_gpu| max", tw["max"],
      "| qpos p99", qp["p99"], "| rollout verdict agreement", ro["agreement"])
assert g["p99"] < 0.02, "the CPU margin departs from the env's"
assert tw["p99"] < 0.01, "the CPU heat model departs from the env's"
PY

# 3. the report pipeline on 64 episodes, every filter of the report
F="--onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise"
C="--sonic-dir $HOME/gear_sonic_hf --task g1_recover_lafan_thermal_cbf --hip mjlab --suites push --num-envs 32"
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
# the barrier acts on every controller and holds the windings at 70 C
e = torch.load("$S/wmreport/eval/smoke/push.pt", map_location="cpu", weights_only=False)["conditions"]
for nm in ("sonic", "fallback", "rollout:1", "wmmin:1:0.95"):
  c = e[nm]
  print(f"[smoke] {nm}: survived {c['survived'].float().mean():.3f}, barrier clipped "
        f"{c['cbf_clip_frac'].mean():.4f} of motor-steps, hottest winding at spawn "
        f"{c['tw_start_max_c'].mean():.1f} C, peak {c['tw_peak_c'].mean():.1f} C (max "
        f"{c['tw_peak_c'].max():.2f} C), over 70 C in {int(c['cbf_over_limit'].sum())} episodes")
  assert float(c["tw_start_max_c"].max()) <= 70.0, "a start temperature above the barrier"
  # speed heating can push a winding at the limit a little past it (u_bar < 0); a
  # broken barrier would let SONIC's windings run ~8 C above their start (to ~78 C)
  assert float(c["tw_peak_c"].max()) < 73.0, f"{nm}: the barrier let a winding past 73 C"
assert float(e["sonic"]["cbf_clip_frac"].mean()) > 0.0, "the barrier never clipped SONIC"
same = all(torch.equal(e[nm]["tw_start_max_c"], e["sonic"]["tw_start_max_c"]) for nm in e)
print("[smoke] start temperatures identical across conditions:", same)
PY
echo "[job] smoke ok"
