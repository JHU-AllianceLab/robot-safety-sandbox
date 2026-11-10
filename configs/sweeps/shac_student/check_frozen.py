"""Check a student checkpoint acts exactly like its teacher, then make its actor obs stats exact.

    python configs/sweeps/shac_student/check_frozen.py TEACHER.zip STUDENT.zip [--state-dim 101]

A student run (configs/g1_recover_sonic_noheight_student_wm.yaml) only fits a
world model: its actor tensors must equal the teacher's bit for bit. Its actor
obs normalizer is frozen from the start of learn(), but the env reset inside
SB3's _setup_learn runs before the freeze and folds one batch into the stats
(weight num_envs / 1.5e9 ~ 3e-6 at the teacher's count; the variance of a dim
whose reset batch is far from its running stats can move ~1e-4). The check: the
count grew by at most one reset batch (--max-fold samples), the mean moved
< 1e-3 of the obs's std and the variance < 1e-3 of itself; the teacher's
obs_mean / obs_var / count are then
written into the student's tensornormalize.pt, so the student checkpoint runs
the teacher's policy exactly in every eval. Its critic stats are left alone
(they are the student state's, and no eval reads them). Exits 1 on a mismatch.
"""

import argparse
import os
import sys

import torch
from stable_baselines3.common.save_util import load_from_zip_file

p = argparse.ArgumentParser()
p.add_argument("teacher")
p.add_argument("student")
p.add_argument("--state-dim", type=int, default=None, help="the student world model's state dim")
p.add_argument("--max-fold", type=int, default=4096,
               help="samples the pre-freeze reset may fold into the stats (the run's num_envs)")
p.add_argument("--tol", type=float, default=1e-3, help="max move of the mean (in std) and the variance")
a = p.parse_args()
_, pt, _ = load_from_zip_file(a.teacher, device="cpu")
_, ps, _ = load_from_zip_file(a.student, device="cpu")
actor = [k for k in pt["policy"] if k.startswith(("mlp_extractor.policy_net", "action_net"))]
bad = [k for k in actor if k not in ps["policy"] or not torch.equal(pt["policy"][k], ps["policy"][k])]
print(f"[frozen] actor tensors: {len(actor) - len(bad)}/{len(actor)} identical", flush=True)
wm = ps["world_model"]
S = int(wm["s_mean"].numel()) if "s_mean" in wm else None
print(f"[frozen] student world model state: {S}-D (teacher's: "
      f"{int(pt['world_model']['s_mean'].numel()) if 's_mean' in pt['world_model'] else None}-D)")
nt_path = os.path.join(os.path.dirname(a.teacher), "tensornormalize.pt")
ns_path = os.path.join(os.path.dirname(a.student), "tensornormalize.pt")
nt = torch.load(nt_path, map_location="cpu", weights_only=True)
ns = torch.load(ns_path, map_location="cpu", weights_only=True)
# the mean's move in units of the obs's std (what normalization feels; |mean| can be ~0),
# the variance's and the count's relative to themselves
std = nt["obs_var"].clamp_min(1e-12).sqrt()
rel = {"obs_mean/std": float(((ns["obs_mean"] - nt["obs_mean"]).abs() / std).max()),
       "obs_var": float(((ns["obs_var"] - nt["obs_var"]).abs() / nt["obs_var"].clamp_min(1e-12)).max()),
       "count": float(((ns["count"] - nt["count"]).abs() / nt["count"]).max())}
fold = float((ns["count"] - nt["count"]).max())
print(f"[frozen] actor obs stats, max move: {rel}; count +{fold:.0f} samples (max {a.max_fold})")
ok = (not bad and 0 <= fold <= a.max_fold and max(rel["obs_mean/std"], rel["obs_var"]) < a.tol
      and (a.state_dim is None or S == a.state_dim))
if not ok:
  print(f"[frozen] FAILED: actor mismatches {bad[:3]}, stats {rel}, count +{fold:.0f}, "
        f"state {S} vs {a.state_dim}")
  sys.exit(1)
for k in ("obs_mean", "obs_var", "count"):
  ns[k] = nt[k].clone()
torch.save(ns, ns_path)
print(f"[frozen] ok; wrote the teacher's actor obs stats into {ns_path}")
