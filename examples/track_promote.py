"""Promote a pilot run's checkpoint at the budget B to a final run (AMENDMENTS C12, B15).

    python examples/track_promote.py --arm t70_wm_d1 --seed 2 --budget 600000000

The final budget B comes from the pilot (docs/experiments/track-squat-filtered.md).
When B <= the pilot's length, the pilot seed need not be retrained: every schedule
of the recipe is independent of the total step count (KL-adaptive lr, constant
gamma, fixed std clamp), so the pilot's checkpoint at B IS a run of B steps. This
copies the largest checkpoint <= B (and its normalizer and discriminator, saved on
the same grid) to runs/track_squat/final/<arm>/s<seed>/ as final_model.zip /
tensornormalize.pt / amp_disc.pt, with the run's config and meta, and records
where it came from in promoted.json. Refuses a checkpoint more than one
checkpoint interval below B.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import track_common as tc  # noqa: E402


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--arm", required=True, choices=sorted(tc.ARMS))
  p.add_argument("--seed", type=int, default=tc.PILOT_SEED)
  p.add_argument("--budget", type=int, required=True)
  p.add_argument("--from-stage", default="pilot")
  p.add_argument("--to-stage", default="final")
  p.add_argument("--max-gap", type=int, default=25_000_000,
                 help="refuse when the best checkpoint is further than this below the budget")
  a = p.parse_args(argv)
  src = tc.run_dir(a.from_stage, a.arm, a.seed)
  dst = tc.run_dir(a.to_stage, a.arm, a.seed)
  ck = os.path.join(src, "checkpoints")
  steps = sorted(int(m.group(1)) for f in os.listdir(ck)
                 if (m := re.fullmatch(r"model_(\d+)_steps\.zip", f)))
  ok = [n for n in steps if n <= a.budget]
  if not ok or a.budget - ok[-1] > a.max_gap:
    raise SystemExit(f"[promote] no checkpoint within {a.max_gap} below {a.budget} in {ck} "
                     f"(have {steps[-3:]})")
  n = ok[-1]
  files = {f"model_{n}_steps.zip": "final_model.zip", f"tensornorm_{n}.pt": "tensornormalize.pt",
           f"amp_disc_{n}.pt": "amp_disc.pt"}
  missing = [f for f in files if not os.path.isfile(os.path.join(ck, f))]
  if missing:
    raise SystemExit(f"[promote] {ck} lacks {missing}")
  os.makedirs(dst, exist_ok=True)
  for f in ("config.yaml", "track_meta.json", "g1_sysid.json"):
    if os.path.isfile(os.path.join(src, f)):
      shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
  with open(os.path.join(dst, "promoted.json"), "w") as f:
    json.dump({"from": os.path.abspath(src), "checkpoint_steps": n, "budget": a.budget}, f,
              indent=1)
  # final_model.zip last: run_jobs.sh reads it as "this job is done"
  for s, d in sorted(files.items(), key=lambda kv: kv[1] == "final_model.zip"):
    shutil.copy2(os.path.join(ck, s), os.path.join(dst, d))
  print(f"[promote] {src} @ {n} steps -> {dst}")


if __name__ == "__main__":
  main()
