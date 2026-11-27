"""Fast Beat-It monitor (k imagined steps + a terminal value from REAL rollouts) vs the full 50-step monitor,
calibrated per SAILR task (the user, 2026-11-26).

    sigma_k(x) = min( min_{j<k} m_j , V_{50-k}( x^_k , ctx ) )        k = 50: the full monitor (no V)

m_j: the heat-aware v0.6.1 Beat-It world model's imagined margins under its fallback (as the training monitor);
x^_k: the predicted state after k steps (k = 0: the real state). V_h(x, ctx) ~ min of the env's REAL margin over
the next h steps of the fallback, one MLP per h, fitted on pooled real rollouts (coolmimic
scripts/real_margin_collect.py) from several banks: the generic push train bank and each task's own TRAIN bank (a
SONIC harvest of the task clips with another seed than the task's calibration bank). One V set serves every task;
only the Wang threshold is per task.

Per task, on that task's calibration bank (the dv U(0, 2.5) SONIC bank of its clips, K rollouts per row, hot30_60,
labels = the fallback's real 3 s outcome; label_start_states --dump-wm): Wang et al. Prop. 1 at eps 1 % / 5 %,
beta 1e-10, one rollout per row (5 draws): threshold delta_k on sigma_k, volume, soundness; AUC; and how the
fast monitor's set overlaps the full monitor's (rows the full monitor keeps but the fast one rejects = the extra
conservatism, and the reverse).

  cd $SANDBOX && python examples/fast_beatit_volume.py --real A.pt B.pt ... --tests dance:D.episodes.pt ... --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncated_monitor_volume import MLP, Fallback, auc, load_eps  # noqa: E402
from v061_report import EPSS, one_per_row, strict  # noqa: E402


def wang(sig, fell, row):
  sig = torch.nan_to_num(sig.double(), nan=-1e9)
  fell = fell.double()
  out = {"auc": auc(sig, fell == 0)}
  for eps in EPSS:
    ds = [strict(sig[i], fell[i], eps) for i in (one_per_row(row, d) for d in range(5))]
    out[f"wang_{eps}"] = ds[0]
    out[f"wang_{eps}_range"] = [min(x["volume"] for x in ds), max(x["volume"] for x in ds)]
  return out


def fit(X, Y, a, dev, tag, init=None):
  """V on (X, Y); ``init``: a pretrained MLP to fine-tune (its normalizer is kept)."""
  if init is not None:
    V = init
  else:
    sub = X[torch.randperm(len(X))[:500000]]
    V = MLP(X.shape[1], a.hidden, a.layers, sub.mean(0).to(dev), sub.std(0).clamp_min(1e-3).to(dev)).to(dev)
  Xg, Yg = X.to(dev), Y.to(dev)
  w = 1.0 + a.w_near * (Yg.abs() < a.near_band).float()
  perm = torch.randperm(len(Yg), device=dev)
  nv = max(1, len(Yg) // 50)
  vi, ti = perm[:nv], perm[nv:]
  spe = math.ceil(len(ti) / a.batch)
  opt = torch.optim.AdamW(V.parameters(), lr=a.lr, weight_decay=1e-4)
  sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(a.epochs * spe, 200), pct_start=0.05)
  tic = time.time()
  for _ in range(a.epochs):
    V.train()
    pp = ti[torch.randperm(len(ti), device=dev)]
    for j in range(spe):
      b = pp[j * a.batch:(j + 1) * a.batch]
      loss = (w[b] * F.smooth_l1_loss(V(Xg[b]), Yg[b], beta=0.02, reduction="none")).mean()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      sch.step()
  V.eval()
  with torch.no_grad():
    pv = V(Xg[vi])
    e = (pv - Yg[vi]).abs()
  st = {"samples": int(len(Yg)), "val_abs_err_mean": float(e.mean()),
        "val_abs_err_p99": float(torch.quantile(e[:200000], 0.99)),
        "val_sign_agree": float(((pv >= 0) == (Yg[vi] >= 0)).float().mean()), "seconds": time.time() - tic}
  print(f"[fast] {tag}: {st['samples']} samples, val |V - R| mean {st['val_abs_err_mean']:.4f} p99 "
        f"{st['val_abs_err_p99']:.4f}, sign agreement {st['val_sign_agree']:.4f} ({st['seconds']:.0f} s)", flush=True)
  del Xg, Yg, w
  return V, st


def load_real(paths):
  S, C, R, src, layout = [], [], [], [], None
  for i, p in enumerate(paths):
    re = torch.load(p, map_location="cpu", weights_only=False)
    lay = list(zip(re["names"], re["dims"]))
    if layout is None:
      layout = lay
    elif lay != layout:
      raise SystemExit(f"{p}: critic layout {lay} != {layout}")
    S.append(re["S"].float())
    C.append(re["C"].float())
    R.append(re["R"].float())
    src.append(torch.full((len(re["S"]),), i, dtype=torch.int16))
    print(f"[fast] real {p}: {len(re['S'])} samples from {re['episodes']} episodes", flush=True)
    del re
  return torch.cat(S), torch.cat(C), torch.cat(R), torch.cat(src), layout


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--real", nargs="+", required=True, help="real_rollouts.pt files; the FIRST is the generic bank")
  p.add_argument("--tests", nargs="+", required=True, help="task:test.episodes.pt (label_start_states --dump-wm)")
  p.add_argument("--out", required=True)
  p.add_argument("--H", type=int, default=50)
  p.add_argument("--ks", type=int, nargs="+", default=[0, 5, 10, 15, 25])
  p.add_argument("--generic-only", action="store_true", help="also fit V on the first --real file alone (compare)")
  p.add_argument("--chunk", type=int, default=4096)
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
  dev, H = a.device, a.H
  os.makedirs(a.out, exist_ok=True)
  S, C, R, src, layout = load_real(a.real)
  X = torch.cat([S, C], 1)
  ks = sorted({k for k in a.ks if 0 <= k < H})
  hs = sorted({H - k for k in ks})
  variants = {"pooled": torch.ones(len(X), dtype=torch.bool)}
  if a.generic_only:
    variants["generic"] = src == 0
  Vs, fits = {v: {} for v in variants}, {v: {} for v in variants}
  for vname, m in variants.items():
    for h in hs:
      y = R[:, h - 1]
      ok = m & torch.isfinite(y) & torch.isfinite(X).all(1)
      Vs[vname][h], fits[vname][str(h)] = fit(X[ok], y[ok], a, dev, f"{vname} V_{h} ({h * 0.02:g} s)")
  del S, C, R, X
  res = {"H": H, "ks": ks, "real": a.real, "layout": layout, "value_fit": fits, "tasks": {}, "procedure": __doc__}
  for spec in a.tests:
    task, path = spec.split(":", 1)
    te = load_eps(path)
    dump = te["settings"]["dump"]
    fb = Fallback(dump["policy"], dump, dev)
    assert fb.H >= H, f"{path}: dump H {fb.H} < {H}"
    if te["wm_S"].shape[1] + te["wm_C"].shape[1] != Vs["pooled"][hs[0]].mean.numel():
      raise SystemExit(f"{path}: state {te['wm_S'].shape[1]} + ctx {te['wm_C'].shape[1]} != V input")
    sc = {"full": []} | {f"{v}_k{k}": [] for v in variants for k in ks}
    with torch.no_grad():
      for j in range(0, len(te["row"]), a.chunk):
        S_ = te["wm_S"][j:j + a.chunk].to(dev)
        C_ = te["wm_C"][j:j + a.chunk].to(dev).float()
        A_ = te["wm_a"][j:j + a.chunk].to(dev)
        out, at = fb.rollout(S_, C_, A_, steps=H, keep=set(ks))
        pre = out.cummin(1).values
        sc["full"].append(pre[:, H - 1].cpu())
        for v in variants:
          for k in ks:
            val = Vs[v][H - k](torch.cat([at[k][0], C_], 1))
            mk = pre[:, k - 1] if k > 0 else torch.full_like(val, float("inf"))
            sc[f"{v}_k{k}"].append(torch.minimum(mk, val).cpu())
    fell, row = te["fell"].bool(), te["row"]
    sc = {n: torch.cat(x) for n, x in sc.items()}
    tr = {"test": path, "rows": int(row.unique().numel()), "episodes": int(len(row)),
          "fallback_saves": 1 - float(fell.float().mean()), "scores": {}}
    # the full monitor's own score vs the labeller's (parity of the offline rollout)
    if "score0" in te:  # the labeller's score = its 1 s min margin - the tube eps
      lab = te["score0"].float() + float(dump.get("eps", 0.0) or 0.0)
      d = (lab - sc["full"]).abs()
      tr["parity_vs_labeller"] = {"p50": float(d.median()), "p99": float(torch.quantile(d, 0.99)), "max": float(d.max())}
    full_w = wang(sc["full"], fell, row)
    df = full_w["wang_0.01"]["delta"]
    in_full = sc["full"] >= (df if df is not None else float("inf"))
    for n, s in sc.items():
      w = wang(s, fell, row)
      d1 = w["wang_0.01"]["delta"]
      ins = s >= (d1 if d1 is not None else float("inf"))
      w["overlap_vs_full_eps1"] = {
        "full_keeps_fast_rejects": float((in_full & ~ins).float().mean()),
        "fast_keeps_full_rejects": float((ins & ~in_full).float().mean()),
        "both_keep": float((ins & in_full).float().mean())}
      tr["scores"][n] = w
    res["tasks"][task] = tr
    print(f"[fast] {task}: {tr['rows']} rows / {tr['episodes']} episodes, fallback saves {tr['fallback_saves']:.3f}"
          + (f", parity vs labeller p99 {tr['parity_vs_labeller']['p99']:.4f}" if "parity_vs_labeller" in tr else ""),
          flush=True)
    for n, w in tr["scores"].items():
      w1, w5, o = w["wang_0.01"], w["wang_0.05"], w["overlap_vs_full_eps1"]
      print(f"[fast]   {n:12s} Wang1 vol {w1['volume']:.3f} {w['wang_0.01_range']} sound {w1['soundness']} delta "
            f"{w1['delta']} | Wang5 {w5['volume']:.3f} | AUC {w['auc']:.3f} | full-keeps-fast-rejects "
            f"{o['full_keeps_fast_rejects']:.4f}, fast-keeps-full-rejects {o['fast_keeps_full_rejects']:.4f}", flush=True)
  json.dump(res, open(os.path.join(a.out, "volume.json"), "w"), indent=1)
  torch.save({v: {str(h): m.state_dict() for h, m in Vs[v].items()} for v in variants}
             | {"hidden": a.hidden, "layers": a.layers, "layout": layout, "H": H},
             os.path.join(a.out, "values.pt"))
  print(f"[fast] wrote {a.out}/volume.json, values.pt", flush=True)


if __name__ == "__main__":
  main()
