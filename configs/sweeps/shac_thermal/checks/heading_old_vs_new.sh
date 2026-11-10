#!/bin/bash
# The SONIC reference-vs-library orientation check on the old stack (original checkout,
# env mjlab) and the new one (this worktree, env mjlab316), CPU, same library file.
source $HOME/miniforge3/etc/profile.d/conda.sh
cat > /dev/shm/hcheck_$$.py <<'PY'
import os, sys, torch
root = sys.argv[1]
sys.path.insert(0, root)
from robot_safety_sandbox.envs.g1_recover import sonic
from robot_safety_sandbox.envs.g1_recover.states import load_bank, to_heading_frame
lib = load_bank(os.path.join(root, "data/g1_lafan1_clips_train.pt"), "cpu")
m = lib["meta"]
csv = os.path.join(root, "data/lafan1_g1")
orig = sonic.SonicReference.build.__func__
try:
  ref = sonic.SonicReference.build(csv, m["clips"], "cpu", library=lib)
  print(f"[{sys.argv[2]}] build passed", flush=True)
except ValueError as e:
  print(f"[{sys.argv[2]}] build raised: {e}", flush=True)
  ref = sonic.SonicReference.build(csv, m["clips"], "cpu", library=None) if sys.argv[2] == "old" else None
if ref is not None:
  rows = ref.rows(lib["clip"], lib["frame"])
  z = torch.zeros(rows.numel(), 3)
  qh = to_heading_frame(ref.root_quat[rows], z, z)[0]
  err = 1.0 - (qh * lib["root_quat_h"]).sum(1).abs()
  print(f"[{sys.argv[2]}] torch {torch.__version__}: max err {float(err.max()):.3e} at row {int(err.argmax())}; "
        f"row 95889 err {float(err[95889]):.3e}; root_quat[row 95889] {ref.root_quat[rows[95889]].tolist()}", flush=True)
PY
conda activate mjlab && python /dev/shm/hcheck_$$.py /weka/home/jhu/yzeng52/robot-safety-sandbox old 2>&1 | grep "^\[old\]"
conda activate mjlab316 && python /dev/shm/hcheck_$$.py /home/jhu/yzeng52/robot-safety-sandbox-mjc316 new 2>&1 | grep "^\[new\]"
rm -f /dev/shm/hcheck_$$.py
