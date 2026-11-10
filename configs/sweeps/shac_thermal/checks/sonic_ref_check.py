"""SonicReference on the MuJoCo 3.16 stack reproduces the clip libraries built on mjlab 1.2."""
import os
import sys

sys.path.insert(0, os.getcwd())
from robot_safety_sandbox.envs.g1_recover.sonic import SonicReference  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import load_bank  # noqa: E402

import torch  # noqa: E402

for dev, path in [(d, p) for d in ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
                  for p in ("data/g1_lafan1_clips_train.pt", "data/g1_lafan1_clips_test.pt")]:
  lib = load_bank(path, "cpu")
  m = lib["meta"]
  csv = m["csv_dir"] if os.path.isdir(m["csv_dir"]) else "data/lafan1_g1"
  ref = SonicReference.build(csv, m["clips"], dev, library=lib)
  print(f"[ref] {dev} {path}: grid {m.get('time_grid', 'mjlab-1.2 (no key)')}, {len(m['clips'])} clips, "
        f"{int(ref.length.sum())} frames == library {int(sum(m['clip_frames']))}: OK", flush=True)
