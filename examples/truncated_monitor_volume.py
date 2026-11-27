"""Calibrated safe-set volume of a TRUNCATED Beat-It monitor (the user, 2026-11-26).

Beat-It's score of a hand-over state is sigma = min_{t<H} m_t - eps over H = 50 imagined steps of its
fallback (m_t = g_t, or min(g_t, -0.1) once the predicted survival drops below 0.5; eps = the
world-model tube). Truncate after k steps and bound the rest with a terminal value:

    sigma_k = min( min_{t<k} m_t , V(s_k, ctx, alive_k, (H - k) / H) ) - eps

V regresses R_k = min_{t>=k} m_t (one MLP for every k = 0..H-1, from the imagined rollouts of the
SONIC push TRAINING bank's hand-over states). k = 0 is V alone (a value-function monitor); k = H the
full monitor. Also: no terminal value (sigma_k = min over the first k steps - eps; optimistic) and
Beat-It's own SHAC critic (AVOID backup, gamma 0.99, the value SHAC bootstraps its horizon with) as V.

Each score is evaluated like the v0.6.1 study's baselines (examples/v061_report.py) on the held-out
dv 3.5 m/s TEST bank, temperature-agnostic world, the SAME episodes (labels: the fallback's 3 s
rollout, fail = a fall): fallback saves; uncalibrated set {sigma >= 0} volume / soundness; Wang et al.
Prop. 1 calibrated sets at eps 1 % / 5 % (beta 1e-10, one rollout per row, 5 draws): volume /
soundness; AUC. Plus the time per call (k world-model steps + V) on the GPU (batch 4096 and 1) and on
one CPU thread (batch 1, the robot case).

Inputs: two episodes files of examples/label_start_states.py --wm-tube ... --dump-wm (the train bank
and the test bank, same model and tube).

  PYTHONPATH=~/ssb-wt-hybrid python examples/truncated_monitor_volume.py \
      --train <train>.episodes.pt --test <test>.episodes.pt --out runs/v061_study/eval/truncated
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v061_report import EPSS, one_per_row, strict  # noqa: E402

KS = (0, 1, 2, 3, 5, 10, 15, 20, 25, 35, 50)


class MLP(nn.Module):
  def __init__(self, d_in, hidden, layers, mean, std):
    super().__init__()
    self.register_buffer("mean", mean)
    self.register_buffer("std", std)
    mods, d = [], d_in
    for _ in range(layers):
      mods += [nn.Linear(d, hidden), nn.SiLU()]
      d = hidden
    self.net = nn.Sequential(*mods, nn.Linear(d, 1))

  def forward(self, x):
    return self.net(torch.clamp((torch.nan_to_num(x) - self.mean) / self.std, -10, 10)).squeeze(-1)


class Fallback:
  """Beat-It's world model + deterministic fallback exactly as label_start_states.WMTube.margins."""

  def __init__(self, policy: str, dump: dict, device: str):
    from robot_safety_sandbox.eval.policies import load_twin

    self.model, self.norm = load_twin(policy, device, quiet=True)
    self.wm = self.model.world_model
    self.dev = device
    self.low = torch.tensor(dump["low"], device=device)
    self.high = torch.tensor(dump["high"], device=device)
    self.sl = slice(*dump["last_action_slice"])
    self.idx = torch.tensor(dump["actor_idx"], device=device)
    self.z_clip, self.H, self.eps = dump["z_clip"], int(dump["H"]), float(dump["eps"])
    tn = torch.load(os.path.join(os.path.dirname(policy), "tensornormalize.pt"), map_location=device,
                    weights_only=False)
    self.cm, self.cs = tn["critic_obs_mean"].float(), torch.sqrt(tn["critic_obs_var"].float() + 1e-8)

  def pi(self, dec):
    o = dec.index_select(-1, self.idx)
    o[:, self.sl] = torch.maximum(torch.minimum(o[:, self.sl], self.high), self.low)
    return torch.clamp(self.model.policy._predict(self.norm(o), deterministic=True), self.low, self.high)

  def critic(self, s):
    return self.model._value(torch.clamp((s - self.cm) / self.cs, -10.0, 10.0))

  @torch.no_grad()
  def rollout(self, S, C, a, steps=None, keep=()):
    """margins [n, steps] and, for t in keep, the state BEFORE step t (t = 0: S) and survival."""
    wm, H = self.wm, self.H if steps is None else steps
    cn = wm.encode_context(C) if C is not None else None
    z = wm.encode(torch.nan_to_num(S))
    n = S.shape[0]
    alive = torch.ones(n, device=S.device)
    out = torch.empty(n, H, device=S.device)
    dec, act = torch.nan_to_num(S), a
    at = {}
    for t in range(H):
      if t in keep:
        at[t] = (dec, alive.clone())
      st = wm.step(z, act, cn, z_clip=self.z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      out[:, t] = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      if t + 1 < H:
        dec = wm.decode(z)
        act = self.pi(dec)
    if H in keep:
      at[H] = (wm.decode(z), alive.clone())
    return out, at


def v_in(dec, C, alive, k, H):
  la = torch.log(alive.clamp_min(1e-6)).unsqueeze(-1)
  parts = [dec] + ([C] if C is not None else []) + [la, torch.full_like(la, (H - k) / H)]
  return torch.cat(parts, -1)


def load_eps(path):
  e = torch.load(path, map_location="cpu", weights_only=False)
  if "wm_S" not in e:
    raise SystemExit(f"{path}: no wm_S (label with --dump-wm)")
  return e


def auc(score, ok):
  s = score.double().numpy()
  y = ok.numpy().astype(bool)
  o = np.argsort(s, kind="mergesort")
  r = np.empty(len(s))
  r[o] = np.arange(1, len(s) + 1)
  n1, n0 = y.sum(), (~y).sum()
  return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def volume_stats(sig: torch.Tensor, fell: torch.Tensor, row: torch.Tensor) -> dict:
  sig = torch.nan_to_num(sig.double(), nan=-1e9)
  fell = fell.double()
  in0 = sig >= 0
  out = {"fallback_saves": float(1 - fell.mean()), "omega0_volume": float(in0.double().mean()),
         "omega0_soundness": float(1 - fell[in0].mean()) if in0.any() else None,
         "auc": auc(sig, fell == 0)}
  for eps in EPSS:
    ds = []
    for d in range(5):
      i = one_per_row(row, d)
      ds.append(strict(sig[i], fell[i], eps))
    out[f"wang_{eps}"] = ds[0]
    out[f"wang_{eps}_range"] = [min(x["volume"] for x in ds), max(x["volume"] for x in ds)]
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--train", required=True)
  p.add_argument("--test", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--hidden", type=int, default=512)
  p.add_argument("--layers", type=int, default=3)
  p.add_argument("--epochs", type=int, default=15)
  p.add_argument("--batch", type=int, default=8192)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--near-band", type=float, default=0.3)
  p.add_argument("--w-near", type=float, default=4.0)
  p.add_argument("--baselines", default="runs/v061_study/eval/report.json")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  torch.manual_seed(a.seed)
  dev = a.device
  os.makedirs(a.out, exist_ok=True)
  tr, te = load_eps(a.train), load_eps(a.test)
  dump = te["settings"]["dump"]
  assert dump["policy"] == tr["settings"]["dump"]["policy"], "train / test scored with different models"
  fb = Fallback(dump["policy"], dump, dev)
  H, eps = fb.H, fb.eps
  thr = eps                                   # sigma >= 0  <=>  min margin >= eps
  print(f"[trunc] {dump['policy']}: H {H}, tube eps {eps:.4f}; train {len(tr['row'])} episodes, test "
        f"{len(te['row'])} episodes ({int(te['row'].unique().numel())} rows)", flush=True)

  # ------------------------------------------------ parity with the labeller's score0 / value0
  chk = {}
  with torch.no_grad():
    i = torch.arange(min(8192, len(te["row"])))
    S, C, A0 = te["wm_S"][i].to(dev), (te["wm_C"][i].to(dev) if "wm_C" in te else None), te["wm_a"][i].to(dev)
    out, _ = fb.rollout(S, C, A0)
    chk["score0_max_abs_diff"] = float((out.amin(1) - eps - te["score0"][i].to(dev)).abs().max())
    chk["value0_max_abs_diff"] = float((fb.critic(S) - te["value0"][i].to(dev)).abs().max())
  print(f"[trunc] parity on 8192 test episodes: |score0 - labeller| max {chk['score0_max_abs_diff']:.2e}, "
        f"|critic - value0| max {chk['value0_max_abs_diff']:.2e}", flush=True)

  # ------------------------------------------------ terminal-value data: every k of every train start
  Xs, Ys = [], []
  with torch.no_grad():
    for j in range(0, len(tr["row"]), 8192):
      S = tr["wm_S"][j:j + 8192].to(dev)
      C = tr["wm_C"][j:j + 8192].to(dev) if "wm_C" in tr else None
      out, at = fb.rollout(S, C, tr["wm_a"][j:j + 8192].to(dev), keep=set(range(H)))
      suf = out.flip(1).cummin(1).values.flip(1)          # suf[:, k] = min_{t >= k} m_t
      for k in range(H):
        dec, al = at[k]
        Xs.append(v_in(dec, C, al, k, H).cpu())
        Ys.append(suf[:, k].cpu())
  X, Y = torch.cat(Xs), torch.cat(Ys)
  del Xs, Ys
  ok = torch.isfinite(X).all(1) & torch.isfinite(Y)
  X, Y = X[ok], Y[ok]
  sub = X[torch.randperm(len(X))[:500000]]
  m_, s_ = sub.mean(0), sub.std(0).clamp_min(1e-3)
  V = MLP(X.shape[1], a.hidden, a.layers, m_.to(dev), s_.to(dev)).to(dev)
  Xg, Yg = X.to(dev), Y.to(dev)
  w = 1.0 + a.w_near * ((Yg - thr).abs() < a.near_band).float()
  perm = torch.randperm(len(Yg), device=dev)
  nv = len(Yg) // 50
  vi, ti = perm[:nv], perm[nv:]
  spe = math.ceil(len(ti) / a.batch)
  opt = torch.optim.AdamW(V.parameters(), lr=a.lr, weight_decay=1e-4)
  sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.05)
  tic = time.time()
  print(f"[trunc] terminal value: {len(Yg)} samples ({X.shape[1]} inputs), {sum(q.numel() for q in V.parameters()) / 1e6:.2f} M params",
        flush=True)
  for ep in range(a.epochs):
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
      e = (V(Xg[vi]) - Yg[vi]).abs()
    print(f"[trunc] epoch {ep + 1}/{a.epochs} val |V - R_k| mean {float(e.mean()):.4f} p99 "
          f"{float(torch.quantile(e[:200000], 0.99)):.4f} ({time.time() - tic:.0f} s)", flush=True)
  del Xg, Yg, w, X, Y

  # ------------------------------------------------ scores of the test episodes
  sc = {m: {k: [] for k in KS} for m in ("learned", "none", "critic")}
  with torch.no_grad():
    for j in range(0, len(te["row"]), 8192):
      S = te["wm_S"][j:j + 8192].to(dev)
      C = te["wm_C"][j:j + 8192].to(dev) if "wm_C" in te else None
      out, at = fb.rollout(S, C, te["wm_a"][j:j + 8192].to(dev), keep=set(k for k in KS if k < H))
      pre = out.cummin(1).values
      n = S.shape[0]
      for k in KS:
        if k == H:
          full = out.amin(1)
          for m in sc:
            sc[m][k].append((full - eps).cpu())
          continue
        mk = pre[:, k - 1] if k > 0 else torch.full((n,), float("inf"), device=dev)
        dec, al = at[k]
        sc["none"][k].append((mk - eps).clamp(max=1e6).cpu())
        sc["learned"][k].append((torch.minimum(mk, V(v_in(dec, C, al, k, H))) - eps).cpu())
        sc["critic"][k].append((torch.minimum(mk, fb.critic(dec)) - eps).cpu())
  fell, row = te["fell"].bool(), te["row"]
  res = {"policy": dump["policy"], "H": H, "tube_eps": eps, "parity": chk, "train": a.train, "test": a.test,
         "episodes": int(len(row)), "rows": int(row.unique().numel()), "scores": {}}
  for m in sc:
    res["scores"][m] = {}
    for k in KS:
      s = torch.cat(sc[m][k])
      st = volume_stats(s, fell, row)
      st["abs_err_vs_full"] = {q: float(np.percentile((s - torch.cat(sc[m][H])).abs().numpy(), p))
                               for q, p in (("p50", 50), ("p90", 90), ("p99", 99))}
      res["scores"][m][str(k)] = st
  print(f"[trunc] test bank: fallback saves {res['scores']['learned'][str(H)]['fallback_saves']:.4f}", flush=True)
  for m in sc:
    for k in KS:
      st = res["scores"][m][str(k)]
      w1, w5 = st["wang_0.01"], st["wang_0.05"]
      print(f"[trunc] {m:7s} k {k:2d}: AUC {st['auc']:.3f} | Omega0 {st['omega0_volume']:.3f} / sound "
            f"{st['omega0_soundness'] if st['omega0_soundness'] is None else round(st['omega0_soundness'], 4)} | "
            f"Wang1 {w1['volume']:.3f} [{st['wang_0.01_range'][0]:.3f}, {st['wang_0.01_range'][1]:.3f}] / "
            f"{w1['soundness']} | Wang5 {w5['volume']:.3f} / {w5['soundness']} | |sigma - full| p50 "
            f"{st['abs_err_vs_full']['p50']:.3f} p99 {st['abs_err_vs_full']['p99']:.3f}", flush=True)

  # ------------------------------------------------ time per call
  tm = {}
  Sb = te["wm_S"][:4096].to(dev)
  Cb = te["wm_C"][:4096].to(dev) if "wm_C" in te else None
  Ab = te["wm_a"][:4096].to(dev)

  def call(fbk, Vm, S, C, A, k):
    if k == 0:
      return Vm(v_in(S, C, torch.ones(S.shape[0], device=S.device), 0, H))
    out, at = fbk.rollout(S, C, A, steps=k, keep={k} if k < H else ())
    m = out.amin(1)
    if k < H:
      dec, al = at[k]
      m = torch.minimum(m, Vm(v_in(dec, C, al, k, H)))
    return m

  def timeit(f, reps, cuda):
    for _ in range(3):
      f()
    if cuda:
      torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
      f()
    if cuda:
      torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / reps

  with torch.no_grad():
    for nb in (4096, 1):
      for k in KS:
        tm[f"gpu_b{nb}_k{k}"] = timeit(lambda: call(fb, V, Sb[:nb], None if Cb is None else Cb[:nb], Ab[:nb], k),
                                       20 if nb > 1 else 50, True)
    torch.set_num_threads(1)
    fbc = Fallback(dump["policy"], dump, "cpu")
    Vc = MLP(V.mean.numel(), a.hidden, a.layers, V.mean.cpu(), V.std.cpu())
    Vc.load_state_dict({q: v.cpu() for q, v in V.state_dict().items()})
    for k in KS:
      tm[f"cpu1_b1_k{k}"] = timeit(lambda: call(fbc, Vc, Sb[:1].cpu(), None if Cb is None else Cb[:1].cpu(),
                                                Ab[:1].cpu(), k), 20, False)
  res["time_ms"] = tm
  res["gpu"] = torch.cuda.get_device_name()
  print("[trunc] time per call, ms (" + res["gpu"] + " / 1 CPU thread): " +
        " | ".join(f"k {k}: gpu4096 {tm[f'gpu_b4096_k{k}']:.2f}, gpu1 {tm[f'gpu_b1_k{k}']:.2f}, cpu1 {tm[f'cpu1_b1_k{k}']:.2f}"
                   for k in KS), flush=True)
  if a.baselines and os.path.isfile(a.baselines):
    bl = json.load(open(a.baselines))["label"].get("nt", {})
    res["baselines_nt"] = {m: {q: v.get(q) for q in ("fallback_saves", "omega0_volume", "omega0_soundness",
                                                      "wang_0.01", "wang_0.05")} for m, v in bl.items()}
  torch.save({"state_dict": V.state_dict(), "d_in": int(V.mean.numel()), "hidden": a.hidden, "layers": a.layers,
              "H": H, "eps": eps, "policy": dump["policy"]}, os.path.join(a.out, "terminal_value.pt"))
  json.dump(res, open(os.path.join(a.out, "report.json"), "w"), indent=1)
  print(f"[trunc] wrote {a.out}/report.json, terminal_value.pt", flush=True)


if __name__ == "__main__":
  main()
