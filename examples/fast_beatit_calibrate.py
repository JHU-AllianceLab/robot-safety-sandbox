"""Calibrate the FAST Beat-It monitor (k imagined steps + V_(50-k)) on a task's labelled bank: Wang et al. Prop. 1
thresholds on sigma_k, next to the full monitor's, written into the task's thresholds.json (the user, 2026-11-26).

Input: the task bank labelled by examples/label_start_states.py WITH --dump-wm (configs/sweeps/sailr_df/calib.sh does
this), the value functions (runs/fast_beatit/volume_distill/values.pt or the Hugging Face release's
beatit_values.pt). Output: thresholds.json gains

    "beatit_fast": {"values": <path>, "variant": "distill", "by_k": {"10": {"delta", "volume", "soundness"}, ...},
                    "full": {"delta", "volume", "soundness"}}

SAILR training reads it with FAST_K=<k> (coolmimic slurm/sailr_df_train.sbatch).

  cd $SANDBOX && python examples/fast_beatit_calibrate.py --labels runs/sailr_df/calib/<task>/label/beatit_thermal_hot30_60.episodes.pt \
      --values runs/fast_beatit/volume_distill/values.pt --thresholds runs/sailr_df/calib/<task>/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_beatit_volume import wang  # noqa: E402
from truncated_monitor_volume import MLP, Fallback, load_eps  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--labels", required=True, help="label_start_states --dump-wm output (*.episodes.pt)")
  p.add_argument("--values", required=True)
  p.add_argument("--variant", default="", help="the values file's variant (default: its only one)")
  p.add_argument("--ks", type=int, nargs="+", default=[10, 15, 25])
  p.add_argument("--thresholds", default="", help="the task's thresholds.json to update (else print only)")
  p.add_argument("--key", default="beatit_fast", help="where in thresholds.json: beatit_fast (teacher, push-bank V), "
                 "beatit_fast_tuned (teacher, task-tuned V), beatit_student / beatit_student_tuned (the no-foot student)")
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  a = p.parse_args()
  dev = a.device
  te = load_eps(a.labels)
  dump = te["settings"]["dump"]
  fb = Fallback(dump["policy"], dump, dev)
  blob = torch.load(a.values, map_location=dev, weights_only=False)
  H = int(blob.get("H", 50))
  if not a.variant:
    vs = [k for k, v in blob.items() if isinstance(v, dict) and k != "layout"]
    if len(vs) != 1:
      raise SystemExit(f"{a.values} holds variants {vs}: pass --variant")
    a.variant = vs[0]
  if fb.H < H:
    raise SystemExit(f"the labels' rollout horizon {fb.H} < {H}: label with --wm-tube wmmin:1:0.95 or longer")
  Vs = {}
  for k in a.ks:
    sd = blob[a.variant][str(H - k)]
    Vs[k] = MLP(sd["mean"].numel(), blob["hidden"], blob["layers"], sd["mean"], sd["std"]).to(dev)
    Vs[k].load_state_dict(sd)
    Vs[k].eval()
  sc = {"full": []} | {k: [] for k in a.ks}
  with torch.no_grad():
    for j in range(0, len(te["row"]), 4096):
      s, c, aa = te["wm_S"][j:j + 4096].to(dev), te["wm_C"][j:j + 4096].to(dev).float(), te["wm_a"][j:j + 4096].to(dev)
      out, at = fb.rollout(s, c, aa, steps=H, keep=set(a.ks))
      pre = out.cummin(1).values
      sc["full"].append(pre[:, H - 1].cpu())
      for k in a.ks:
        sc[k].append(torch.minimum(pre[:, k - 1], Vs[k](torch.cat([at[k][0], c], 1))).cpu())
  fell, row = te["fell"].bool(), te["row"]
  res = {"values": os.path.abspath(a.values), "variant": a.variant, "labels": os.path.abspath(a.labels),
         "rows": int(row.unique().numel()), "by_k": {}}
  for n, x in sc.items():
    w = wang(torch.cat(x), fell, row)["wang_0.01"]
    d = {"delta": w["delta"], "volume": w["volume"], "soundness": w["soundness"]}
    if n == "full":
      res["full"] = d
    else:
      res["by_k"][str(n)] = d
    print(f"[fast-calib] {'full' if n == 'full' else f'k {n}'}: Wang eps 1 % threshold {w['delta']}, volume "
          f"{w['volume']:.3f}, soundness {w['soundness']}", flush=True)
  if a.thresholds:
    t = json.load(open(a.thresholds)) if os.path.isfile(a.thresholds) else {}
    res["policy"] = dump["policy"]
    t[a.key] = res
    json.dump(t, open(a.thresholds, "w"), indent=1)
    print(f"[fast-calib] wrote {a.key} into {a.thresholds}", flush=True)


if __name__ == "__main__":
  main()
