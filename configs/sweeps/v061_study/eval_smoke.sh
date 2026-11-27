#!/bin/bash
# Smoke of the v0.6.1 study's evaluation worlds with models that exist today (throwaway, runs/v061_study/eval_smoke):
#   a. lambda-Reach v6 (heat-blind) in g1_recover_noheight_v61_heat, soaked U(60, 65): overheat flag, peak winding;
#   b. lambda-Reach v6 in g1_recover_noheight_v6 (temperature-agnostic world);
#   c. Beat-It v6 s1 + its v6 tube in g1_recover_thermal70_v61, soaked U(60, 65) (v0.6.1 pin, soaked draw);
#   d. the SONIC-driven twin g1_recover_lafan_thermal70_v61 builds (calibration rollouts, 128 episodes).
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/robot-safety-sandbox-mjc316}"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 G1_SYSID_V061_DIR=$HOME/G1-robot-model-with-SysID-v0.6.1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
O=runs/v061_study/eval_smoke; rm -rf $O; mkdir -p $O
BANK=data/g1_sonic_push_gfall_sysid316v6_dv35_test_bank_net.pt
LR=$(ls runs/thermal65/lambdareach_s0/*/final_model.zip | head -1)
BI=runs/shac_thermal70_v6/hybrid_s1/g1_recover_thermal70_v6_safetyshac1p/final_model.zip
C="--no-push --no-obs-noise --bank $BANK --k 1 --max-rows 1000 --num-envs 1024"
PYTHONPATH=$HOME/safety-stable-baselines-baselines python examples/label_start_states.py --policy $LR \
  --task g1_recover_noheight_v61_heat $C --thermal-start-c 60 65 --out $O/a.pt > $O/a.log 2>&1; echo "a: $?"
PYTHONPATH=$HOME/safety-stable-baselines-baselines python examples/label_start_states.py --policy $LR \
  --task g1_recover_noheight_v6 $C --out $O/b.pt > $O/b.log 2>&1; echo "b: $?"
PYTHONPATH=$HOME/ssb-wt-hybrid python examples/label_start_states.py --policy $BI --task g1_recover_thermal70_v61 $C \
  --wm-tube wmmin:1:0.95 --tube-calib runs/shac_thermal70_v6/wmreport/calib/v6hyb_s1/min_tube.json --tube-tag v6hyb_s1 \
  --thermal-start-c 60 65 --out $O/c.pt > $O/c.log 2>&1; echo "c: $?"
PYTHONPATH=$HOME/ssb-wt-hybrid python examples/eval_sonic_qcbf.py --policy $BI --tag v6hyb_s1 --sonic-dir $HOME/gear_sonic_hf \
  --task g1_recover_lafan_thermal70_v61 --hip mjlab --suites push --episodes 128 --num-envs 128 --onset-s 0.02 0.02 \
  --dv-range 0 4 --dv-bins 0 1 2 3 4 --exclude-categories fallAndGetUp sprint --no-obs-noise --seed 1 \
  --conditions sonic wmcalib:clean --out $O/d > $O/d.log 2>&1; echo "d: $?"
for x in a b c d; do echo "== $x"; grep -E "\[label\] done|Traceback|Error|SystemExit|sysid|qcbf\] push" $O/$x.log | tail -4 | cut -c1-250; done
python - <<'PY'
import torch
for x in "abc":
  try:
    e = torch.load(f"runs/v061_study/eval_smoke/{x}.episodes.pt", weights_only=False)
  except Exception as err:
    print(x, "no episodes", err); continue
  s = e.get("score0", e["value0"])
  msg = f"{x}: n {e['row'].numel()} fail {e['fell'].float().mean():.3f}"
  if "overheat" in e:
    msg += f" overheat {e['overheat'].float().mean():.3f} tw_peak mean {e['tw_peak'].mean():.1f} min {e['tw_peak'].min():.1f}"
  print(msg, "thermal_start", e["label"].get("thermal_start_c"))
PY
