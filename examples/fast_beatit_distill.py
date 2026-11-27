"""Fast Beat-It monitor with a terminal value DISTILLED from the full monitor's own imagination, on tracker states
too (the user, 2026-11-26).

The real-rollout V (examples/fast_beatit_volume.py) matches the full monitor's calibrated volume on the task banks,
but on the SAILR trackers' own states it rejects 2-3x as often as the full monitor (its training states were
pushed hand-over states only). Here V_(50-k) is fitted to what the full monitor would still imagine after step k:

    target(x, a) = min_{k <= t < 50} m_t   along the full monitor's rollout from (x, a)
    input        = (x^_k, ctx)             the world model's state after k imagined steps

on two kinds of starts: tracker states with the tracker's proposals (coolmimic scripts/fast_monitor_onpolicy.py
--record-out: the finished trackers and earlier checkpoints, deploy mode, pushes) and the real-rollout states of the
train banks with the fallback's own action as the proposal. sigma_k = min(min_{t<k} m_t, V_(50-k)) then equals the
full monitor's score wherever V is exact. Thresholds: Wang eps 1 % per task on the task calibration banks (as
fast_beatit_volume.py; those rows are in neither training set).

  cd $SANDBOX && python examples/fast_beatit_distill.py --tracker DIR/*.pt --real A.pt B.pt --tests dance:T.pt ... --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_beatit_volume import fit, wang  # noqa: E402
from truncated_monitor_volume import MLP, Fallback, load_eps  # noqa: E402

FOOT = ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces")


@torch.no_grad()
def targets(fb, S, C, A, ks, H, dev, chunk=8192):
  X = {k: [] for k in ks}
  Y = {k: [] for k in ks}
  for j in range(0, len(S), chunk):
    s, c, a = S[j:j + chunk].to(dev), C[j:j + chunk].to(dev), A[j:j + chunk].to(dev)
    out, at = fb.rollout(s, c, a, steps=H, keep=set(ks))
    for k in ks:
      X[k].append(torch.cat([at[k][0], c], 1).cpu())
      Y[k].append(out[:, k:].amin(1).cpu())
  return {k: torch.cat(X[k]) for k in ks}, {k: torch.cat(Y[k]) for k in ks}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--tracker", nargs="*", default=[], help="fast_monitor_onpolicy.py --record-out shards (none: push states only)")
  p.add_argument("--real", nargs="*", default=[], help="real_rollouts.pt files (their states, fallback proposals)")
  p.add_argument("--real-per-file", type=int, default=400000)
  p.add_argument("--push-dumps", nargs="*", default=[], help="push states from label_start_states --dump-wm episode "
                 "files (wm_S, wm_C) of a SONIC push bank NOT used for calibration (e.g. your task's seed-4 bank); "
                 "already in the model's layout; up to --real-per-file each, fallback proposals")
  p.add_argument("--tests", nargs="+", required=True, help="task:labels.episodes.pt (--dump-wm); the first one's dump "
                 "settings (policy, fallback box, actor columns) define the model")
  p.add_argument("--nofoot", action="store_true", help="drop the foot terms from recorded / real states (the no-foot "
                 "student's 188-D layout; --layout-from names them)")
  p.add_argument("--layout-from", default="", help="a real_rollouts.pt whose names / dims give the 200-D layout")
  p.add_argument("--init-values", default="", help="fine-tune: start from this values.pt (its --init-variant)")
  p.add_argument("--init-variant", default="pushbank")
  p.add_argument("--variant-name", default="distill", help="the key the fitted value functions are saved under")
  p.add_argument("--out", required=True)
  p.add_argument("--H", type=int, default=50)
  p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 25])
  p.add_argument("--hidden", type=int, default=512)
  p.add_argument("--layers", type=int, default=3)
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--batch", type=int, default=8192)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--near-band", type=float, default=0.3)
  p.add_argument("--w-near", type=float, default=4.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  torch.manual_seed(a.seed)
  dev, H, ks = a.device, a.H, sorted(a.ks)
  os.makedirs(a.out, exist_ok=True)
  first = load_eps(a.tests[0].split(":", 1)[1])
  dump = first["settings"]["dump"]
  del first
  fb = Fallback(dump["policy"], dump, dev)
  keep = None
  if a.nofoot:  # the 200-D layout's columns without the foot terms
    lay = torch.load(a.layout_from or a.real[0], map_location="cpu", weights_only=False)
    keep, o = [], 0
    for nm, dm in zip(lay["names"], lay["dims"]):
      if nm not in FOOT:
        keep += list(range(o, o + dm))
      o += dm
    keep = torch.tensor(keep)
    del lay
  cut = (lambda x: x[:, keep]) if keep is not None else (lambda x: x)  # noqa: E731
  # starts
  tS, tC, tA, src = [], [], [], {}
  for f in a.tracker:
    d = torch.load(f, map_location="cpu", weights_only=False)
    tS.append(cut(d["S"].float())); tC.append(d["C"].float()); tA.append(d["a"].float())
    src[os.path.basename(f)] = int(len(d["S"]))
  D = len(cut(torch.zeros(1, 200))[0]) if keep is not None else None
  tS = torch.cat(tS) if tS else torch.zeros(0, D or fb.wm.state_dim)
  tC = torch.cat(tC) if tC else torch.zeros(0, fb.wm.context_dim)
  tA = torch.cat(tA) if tA else torch.zeros(0, fb.wm.action_dim)
  print(f"[distill] tracker states {len(tS)} from {src}", flush=True)
  rS, rC = [], []
  for f in a.real:
    d = torch.load(f, map_location="cpu", weights_only=False)
    i = torch.randperm(len(d["S"]))[:a.real_per_file]
    rS.append(cut(d["S"][i].float())); rC.append(d["C"][i].float())
    del d
  for f in a.push_dumps:
    d = load_eps(f)
    i = torch.randperm(len(d["wm_S"]))[:a.real_per_file]
    rS.append(d["wm_S"][i].float()); rC.append(d["wm_C"][i].float())
    del d
  rS = torch.cat(rS) if rS else torch.zeros(0, tS.shape[1])
  rC = torch.cat(rC) if rC else torch.zeros(0, tC.shape[1])
  with torch.no_grad():
    rA = torch.cat([fb.pi(torch.nan_to_num(rS[j:j + 16384].to(dev))).cpu() for j in range(0, len(rS), 16384)]) \
      if len(rS) else torch.zeros(0, tA.shape[1])
  print(f"[distill] real-rollout states {len(rS)} (fallback proposals)", flush=True)
  S, C, A = torch.cat([tS, rS]), torch.cat([tC, rC]), torch.cat([tA, rA])
  kind = torch.cat([torch.zeros(len(tS), dtype=torch.int8), torch.ones(len(rS), dtype=torch.int8)])
  print(f"[distill] starts: {len(S)} ({int((kind == 0).sum())} tracker, {int((kind == 1).sum())} push), state "
        f"{S.shape[1]}-D", flush=True)
  del tS, tC, tA, rS, rC, rA
  X, Y = targets(fb, S, C, A, ks, H, dev)
  Vs, fits = {}, {}
  init = torch.load(a.init_values, map_location=dev, weights_only=False) if a.init_values else None
  for k in ks:
    ok = torch.isfinite(Y[k]) & torch.isfinite(X[k]).all(1)
    V0 = None
    if init is not None:  # fine-tune: the pretrained V_(H-k), its own input normalizer kept
      sd = init[a.init_variant][str(H - k)]
      V0 = MLP(sd["mean"].numel(), init["hidden"], init["layers"], sd["mean"], sd["std"]).to(dev)
      V0.load_state_dict(sd)
    Vs[H - k], fits[str(H - k)] = fit(X[k][ok], Y[k][ok], a, dev, f"{a.variant_name} V_{H - k} (after {k} imagined steps)",
                                      init=V0)
    if bool((kind == 0).any()):
      fits[str(H - k)]["target_neg_frac_tracker"] = float((Y[k][kind == 0] < 0).float().mean())
  del X, Y, S, C, A
  res = {"H": H, "ks": ks, "tracker": src, "real": a.real, "value_fit": {a.variant_name: fits}, "tasks": {},
         "variant": a.variant_name, "init": a.init_values or None, "nofoot": a.nofoot,
         "procedure": __doc__}
  for spec in a.tests:
    task, path = spec.split(":", 1)
    te = load_eps(path)
    sc = {"full": []} | {f"{a.variant_name}_k{k}": [] for k in ks}
    with torch.no_grad():
      for j in range(0, len(te["row"]), 4096):
        s, c, aa = te["wm_S"][j:j + 4096].to(dev), te["wm_C"][j:j + 4096].to(dev).float(), te["wm_a"][j:j + 4096].to(dev)
        out, at = fb.rollout(s, c, aa, steps=H, keep=set(ks))
        pre = out.cummin(1).values
        sc["full"].append(pre[:, H - 1].cpu())
        for k in ks:
          v = Vs[H - k](torch.cat([at[k][0], c], 1))
          sc[f"{a.variant_name}_k{k}"].append(torch.minimum(pre[:, k - 1], v).cpu())
    fell, row = te["fell"].bool(), te["row"]
    sc = {n: torch.cat(x) for n, x in sc.items()}
    tr = {"test": path, "rows": int(row.unique().numel()), "fallback_saves": 1 - float(fell.float().mean()), "scores": {}}
    df = None
    for n, s in sc.items():
      w = wang(s, fell, row)
      if n == "full":
        df = w["wang_0.01"]["delta"]
        in_full = s >= df
      d1 = w["wang_0.01"]["delta"]
      ins = s >= (d1 if d1 is not None else float("inf"))
      w["overlap_vs_full_eps1"] = {"full_keeps_fast_rejects": float((in_full & ~ins).float().mean()),
                                   "fast_keeps_full_rejects": float((ins & ~in_full).float().mean())}
      tr["scores"][n] = w
      print(f"[distill] {task} {n:12s} Wang1 vol {w['wang_0.01']['volume']:.3f} sound {w['wang_0.01']['soundness']} "
            f"delta {d1} | AUC {w['auc']:.3f} | full-keeps-fast-rejects {w['overlap_vs_full_eps1']['full_keeps_fast_rejects']:.4f}",
            flush=True)
    res["tasks"][task] = tr
  json.dump(res, open(os.path.join(a.out, "volume.json"), "w"), indent=1)
  torch.save({a.variant_name: {str(h): m.state_dict() for h, m in Vs.items()}, "hidden": a.hidden, "layers": a.layers,
              "H": H, "nofoot": a.nofoot, "policy": dump["policy"]}, os.path.join(a.out, "values.pt"))
  print(f"[distill] wrote {a.out}/volume.json, values.pt", flush=True)


if __name__ == "__main__":
  main()
