"""Fit V^pi or Q^pi to rollout labels and calibrate the safety threshold eps.

    python examples/fit_policy_value.py --dir data/policy_eval --mode v --tag <policy tag>
    python examples/fit_policy_value.py --dir data/policy_eval --mode q --tag <policy tag>

Reads ``<dir>/{train,calib,test}_<mode>.pt`` (``build_policy_eval_states.py``)
and their labels ``..._labels_<tag>.pt`` (``label_policy_values.py``). Fits an
MLP (``policy_eval.ValueNet``; input = the 68-D phi of the state, the last
applied action and, for Q, the action) by MSE on every train rollout's label,
then sets eps on the calibration split two ways (``policy_eval``):

* false-safe control: the smallest eps whose accepted set (V_hat - eps > 0)
  has a Clopper-Pearson upper bound on its unsafe fraction <= alpha at
  confidence 1 - beta, Bonferroni over the eps grid, alpha the first of the
  ladder 0.1 / 0.2 / 0.5 / 1 / 2 / 5 % that some eps meets;
* split conformal on the residuals V_hat - V, at each ``--residual-alphas``.

Calibration and test use one rollout of one state per SONIC episode
(exchangeable). The report applies every eps to the test split (held-out
clips: a shift the guarantee does not cover, so it shows how far it carries)
and breaks the test results down by time from the kick, kick size and
category. Output: ``<dir>/<mode>_<tag>.pt`` (state dict, eps, report) and the
report as ``.json``.

``--with-pi`` (``--mode q``) also fits on the pi-centred rows
(``label_policy_values.py --q-center pi``: ``<split>_q_pi_labels_<tag>.pt``,
whose played ``q_action`` replaces the states file's): every train row of
both, ONE of the two per calib state (a seeded coin flip, so calibration keeps
one sample per state), both on test, reported per centre too. It writes
``q_<variant>_<tag>.pt`` (``--variant``, default ``exact_pi``) and copies
``v_<tag>.pt`` to ``v_<variant>_<tag>.pt``, so the pair loads as one variant
(``policy_value_filter.PolicyValues``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

from robot_safety_sandbox.envs.g1_recover.policy_eval import (  # noqa: E402
  ALPHA_LADDER, ValueNet, evaluate_threshold, false_safe_threshold, features,
  residual_threshold)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--dir", default=os.path.join(_ZOO, "data", "policy_eval"))
  p.add_argument("--mode", choices=("v", "q"), default="v")
  p.add_argument("--tag", required=True, help="the labels' policy tag (..._labels_<tag>.pt)")
  p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
  p.add_argument("--steps", type=int, default=20000)
  p.add_argument("--batch", type=int, default=4096)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--beta", type=float, default=1e-3)
  p.add_argument("--residual-alphas", type=float, nargs="+", default=[0.05, 0.01])
  p.add_argument("--with-pi", action="store_true",
                 help="--mode q: also fit on the pi-centred rows (module docstring)")
  p.add_argument("--variant", default=None,
                 help="output <mode>_<variant>_<tag>.pt (default: none; exact_pi with --with-pi)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  if a.with_pi and a.mode != "q":
    p.error("--with-pi needs --mode q")
  if a.variant is None and a.with_pi:
    a.variant = "exact_pi"
  return a


def _load(d: str, split: str, mode: str, tag: str, center: str = "row"):
  """``(rows, x, v)`` of one split. ``center="pi"``: the pi-centred labels
  (``label_policy_values.py --q-center pi``) and their played actions."""
  rows = torch.load(os.path.join(d, f"{split}_{mode}.pt"), map_location="cpu", weights_only=False)
  pi = "_pi" if center == "pi" else ""
  lab = torch.load(os.path.join(d, f"{split}_{mode}{pi}_labels_{tag}.pt"), map_location="cpu",
                   weights_only=False)
  if center == "pi":
    if mode != "q" or "q_action" not in lab:
      raise SystemExit(f"{split}_{mode}{pi}_labels_{tag}.pt has no played q_action")
    rows = dict(rows, q_action=lab["q_action"].float())
  x = features(rows, rows["q_action"] if mode == "q" else None)
  return rows, x, lab["v"]


def _cat_rows(a: dict, b: dict) -> dict:
  """The columns the report reads (:func:`_breakdown`), stacked."""
  return {k: torch.cat([a[k], b[k]]) for k in ("t_rel", "dv", "center")}


def load_splits(d: str, mode: str, tag: str, with_pi: bool, seed: int = 0,
                splits=("train", "calib", "test")):
  """``splits`` of :func:`_load`, SONIC-centred plus, ``with_pi``, pi-centred
  (module docstring): ``(rows, x, v)`` each, ``rows["center"]`` 0 / 1."""
  out = {}
  coin = torch.Generator().manual_seed(seed)
  for split in splits:
    r0, x0, v0 = _load(d, split, mode, tag)
    r0 = dict(r0, center=torch.zeros(x0.shape[0], dtype=torch.long))
    if not with_pi:
      out[split] = (r0, x0, v0)
      continue
    r1, x1, v1 = _load(d, split, mode, tag, "pi")
    r1 = dict(r1, center=torch.ones(x1.shape[0], dtype=torch.long))
    if split == "calib":
      # one sample per state: keep the SONIC- or the pi-centred row
      pick = torch.rand(x0.shape[0], generator=coin) < 0.5
      v0 = v0.clone()
      v1 = v1.clone()
      v0[pick] = float("nan")
      v1[~pick] = float("nan")
    if v0.shape[1] != v1.shape[1]:
      k = min(v0.shape[1], v1.shape[1])
      v0, v1 = v0[:, :k], v1[:, :k]
    out[split] = (_cat_rows(r0, r1), torch.cat([x0, x1]), torch.cat([v0, v1]))
    del r0, r1, x0, x1
  return out


def _breakdown(rows, score, label, eps) -> dict:
  out = {}
  t = rows["t_rel"]
  dv = rows["dv"]
  cats = None
  for name, m in {"before the kick": t < 0, "0-0.3 s after": (t >= 0) & (t < 0.3),
                  "0.3-1 s after": (t >= 0.3) & (t < 1.0), ">= 1 s after": t >= 1.0}.items():
    if int(m.sum()) >= 20:
      out[f"t: {name}"] = evaluate_threshold(score[m], label[m], eps)
  for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.01)]:
    m = (dv >= lo) & (dv < hi)
    if int(m.sum()) >= 20:
      out[f"dv {lo}-{hi}"] = evaluate_threshold(score[m], label[m], eps)
  return out


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  t0 = time.time()
  sp = load_splits(a.dir, a.mode, a.tag, a.with_pi, a.seed)
  _, x_tr, v_tr = sp["train"]
  _, x_ca, v_ca = sp["calib"]
  te_rows, x_te, v_te = sp["test"]
  del sp
  # every train rollout is a sample; calib / test take each row's FIRST rollout
  k = v_tr.shape[1]
  ok = ~torch.isnan(v_tr)
  xs = x_tr.repeat_interleave(k, 0)[ok.reshape(-1)]
  ys = v_tr.reshape(-1)[ok.reshape(-1)]
  c_ok, t_ok = ~torch.isnan(v_ca[:, 0]), ~torch.isnan(v_te[:, 0])
  x_ca, y_ca = x_ca[c_ok], v_ca[c_ok, 0]
  x_te, y_te = x_te[t_ok], v_te[t_ok, 0]
  te_rows = {kk: (vv[t_ok] if torch.is_tensor(vv) and vv.dim() > 0 and vv.shape[0] == t_ok.numel() else vv)
             for kk, vv in te_rows.items()}
  print(f"[fit] {a.mode}: train {ys.numel()} rollouts ({x_tr.shape[0]} states), calib "
        f"{y_ca.numel()}, test {y_te.numel()}; safe fraction train {float((ys >= 0).float().mean()):.3f}")

  net = ValueNet(xs.shape[1], tuple(a.hidden)).to(dev)
  net.set_normalizer(xs.to(dev))
  opt = torch.optim.Adam(net.parameters(), lr=a.lr)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
  xs_d, ys_d = xs.to(dev), ys.to(dev)
  for step in range(a.steps):
    idx = torch.randint(0, xs_d.shape[0], (a.batch,), device=dev)
    loss = torch.nn.functional.mse_loss(net(xs_d[idx]), ys_d[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 2000 == 0 or step == a.steps - 1:
      with torch.no_grad():
        vl = torch.nn.functional.mse_loss(net(x_ca.to(dev)), y_ca.to(dev))
      print(f"[fit] step {step}: train mse {loss.item():.4f}  calib mse {vl.item():.4f}", flush=True)
  net.eval()
  with torch.no_grad():
    s_ca = net(x_ca.to(dev)).cpu()
    s_te = net(x_te.to(dev)).cpu()

  report = {"mode": a.mode, "tag": a.tag, "n_train_rollouts": int(ys.numel()),
            "n_calib": int(y_ca.numel()), "n_test": int(y_te.numel()),
            "calib_safe_frac": float((y_ca >= 0).float().mean()),
            "test_safe_frac": float((y_te >= 0).float().mean()),
            "test_mse": float(((s_te - y_te) ** 2).mean()), "thresholds": {}}
  if a.with_pi:
    report["with_pi"] = True
    report["test_by_center"] = {}
    for c, name in ((0, "sonic"), (1, "pi")):
      m = te_rows["center"] == c
      report["test_by_center"][name] = {
        "n": int(m.sum()), "mse": float(((s_te[m] - y_te[m]) ** 2).mean()),
        "bias": float((s_te[m] - y_te[m]).mean()), "safe_frac": float((y_te[m] >= 0).float().mean())}
      r = report["test_by_center"][name]
      print(f"[fit] test, {name}-centred actions: {r['n']} rows, mse {r['mse']:.4f}, bias "
            f"{r['bias']:+.4f}, safe {r['safe_frac']:.3f}")
  fs = false_safe_threshold(s_ca, y_ca < 0.0, beta=a.beta, alphas=ALPHA_LADDER)
  report["thresholds"]["false_safe"] = {"calibration": fs}
  for alpha in a.residual_alphas:
    report["thresholds"][f"residual_{alpha:g}"] = {"calibration": residual_threshold(s_ca, y_ca, alpha)}
  for name, th in report["thresholds"].items():
    eps = th["calibration"]["eps"]
    th["test"] = evaluate_threshold(s_te, y_te, eps)
    th["test_breakdown"] = _breakdown(te_rows, s_te, y_te, eps) if math.isfinite(eps) else {}
    c, t = th["calibration"], th["test"]
    print(f"[fit] {name}: eps {eps:+.4f}" + (f" (alpha {c.get('alpha')}, CP bound {c.get('bound', float('nan')):.4f})"
                                              if name == "false_safe" else "")
          + f"  | test: accept {t['accept_rate']:.3f}, false-safe {t['false_safe_rate']:.4f} "
            f"({t['false_safe']}/{t['accepted']}), recall {t['recall']:.3f}, coverage {t['coverage']:.4f}")
  stem = f"{a.mode}_{a.tag}" if a.variant is None else f"{a.mode}_{a.variant}_{a.tag}"
  out = os.path.join(a.dir, f"{stem}.pt")
  torch.save({"state_dict": net.state_dict(), "in_dim": xs.shape[1], "hidden": list(a.hidden),
              "eps": {n: th["calibration"]["eps"] for n, th in report["thresholds"].items()},
              "report": report}, out)
  with open(os.path.splitext(out)[0] + ".json", "w") as f:
    json.dump(report, f, indent=1)
  print(f"[fit] wrote {out} ({time.time() - t0:.0f}s)")
  if a.with_pi:
    # V is unchanged: the same fit under the new variant's name
    src = os.path.join(a.dir, f"v_{a.tag}.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["copied_from"] = os.path.abspath(src)
    dst = os.path.join(a.dir, f"v_{a.variant}_{a.tag}.pt")
    torch.save(ck, dst)
    print(f"[fit] copied {src} -> {dst}")


if __name__ == "__main__":
  main()
