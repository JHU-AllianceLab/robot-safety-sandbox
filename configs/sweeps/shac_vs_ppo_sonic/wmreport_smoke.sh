#!/bin/bash
# Smoke test of the report run (wmreport_jobs.txt): category exclusion, the margin-state
# recording of wmcalib:clean, the tube fit on it, latch 2 s / forever and the 2x2 filter video.
#   sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=16 --mem=48G --time=00:30:00 --job-name=wmreport_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_vs_ppo_sonic/wmreport_smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
P=runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip
S=runs/shac_vs_ppo_sonic/wmreport_smoke/shac_s0
F="--onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise"
rm -rf $S
python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 64 --num-envs 32 $F --seed 1 --conditions sonic wmcalib:clean --out $S/calib
python examples/calibrate_wm_min_tube.py --calib $S/calib/push.pt --tag shac_s0
python - <<PY
import torch
d = torch.load("$S/calib/push.pt", map_location="cpu", weights_only=False)
r = d["conditions"]["wmcalib:clean"]
cats = d["meta"]["categories"]
bad = [cats.index(c) for c in ("fallAndGetUp", "sprint")]
assert not torch.isin(d["episodes"]["category"], torch.tensor(bad)).any(), "excluded category drawn"
assert float(d["episodes"]["dv"].max()) <= 4.0, "kick above 4 m/s"
ps, rs = r["calib_pred_states"], r["calib_real_states"]
print("[smoke] states", tuple(ps.shape), tuple(rs.shape), r["calib_state_names"])
print("[smoke] step-1 height pred/real", ps[:4, 0, 0, 0].tolist(), rs[:4, 0, 0].tolist())
print("[smoke] step-1 tilt pred/real", ps[:4, 0, 0, 1].tolist(), rs[:4, 0, 1].tolist())
ok = torch.isfinite(rs[:, 0, 0])
assert ok.any() and torch.isfinite(ps[ok, 0, 0, 0]).all(), "states not recorded"
print("[smoke] step-1 |height err| median", float((ps[ok, 0, 0, 0] - rs[ok, 0, 0]).abs().median()))
PY
python examples/eval_sonic_qcbf.py --policy $P --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf --hip mjlab --suites push --episodes 64 --num-envs 32 $F --rollout-workers 8 --conditions sonic rollout:1 wmmin:1:0.95+dwell2 wmmin:1:0.95+latch --tube-calib $S/calib/min_tube.json --video-episodes 1 40 --video-conditions sonic rollout:1 wmmin:1:0.95+dwell2 wmmin:1:0.95+latch --video-size 640 480 --video-cols 2 --out $S/eval
ls -la $S/eval/videos/
echo "[job] smoke ok"
