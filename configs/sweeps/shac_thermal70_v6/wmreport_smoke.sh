#!/bin/bash
# Smoke of the calibration pipeline (wmreport_jobs.txt) on the HYBRID Beat-It seed 0, small sizes: the SONIC +
# wmcalib:clean rollouts run with the natively hybrid world model (--wm-thermal nn), the min tube fits, and the
# Beat-It monitor (wmmin:1:0.95) runs with that tube. Output runs/shac_thermal70_v6/wmreport_smoke (throwaway).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p l40s,rtx6000,h100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=01:00:00 --job-name=wmreport_v6smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70_v6/wmreport_smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import safety_sb3; print('[smoke] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q ssb-wt-hybrid
P=runs/shac_thermal70_v6/hybrid_s0/g1_recover_thermal70_v6_safetyshac1p/final_model.zip
S=runs/shac_thermal70_v6/wmreport_smoke
rm -rf $S
COM="--policy $P --tag v6hyb_s0 --sonic-dir $HOME/gear_sonic_hf --task g1_recover_lafan_thermal70_v6 --hip mjlab --suites push --onset-s 0.02 0.02 --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise"
F="linesearch|solver iterations limit|warn_overflow"
date -Is; echo "== 1. calibration rollouts"
python examples/eval_sonic_qcbf.py $COM --episodes 128 --num-envs 128 --seed 1 --conditions sonic wmcalib:clean \
  --out $S/calib/v6hyb_s0 2>&1 | grep -v -E "$F" | tail -25
date -Is; echo "== 2. min tube"
python examples/calibrate_wm_min_tube.py --calib $S/calib/v6hyb_s0/push.pt --tag v6hyb_s0 2>&1 | tail -25
date -Is; echo "== 3. Beat-It monitor with the tube"
python examples/eval_sonic_qcbf.py $COM --episodes 64 --num-envs 64 --rollout-workers 8 --conditions sonic wmmin:1:0.95 \
  --tube-calib $S/calib/v6hyb_s0/min_tube.json --out $S/eval/v6hyb_s0 2>&1 | grep -v -E "$F" | tail -30
[[ -f $S/eval/v6hyb_s0/push.json ]] || { echo "[smoke] FAILED: no eval record"; exit 1; }
date -Is; echo "[smoke] PASSED"
