"""Bootstrapped (TD) V^pi and Q^pi of a fixed policy, next to the rollout-label fit.

    python examples/fit_policy_value_td.py --dir data/policy_eval_gfall --tag <policy tag>

``fit_policy_value.py`` regresses V_hat on exact rollout labels, V^pi_T(s) =
min_{0 <= t <= T} g(s_t) with T = 3 s: one label per start state, nothing
bootstrapped. This script fits the same networks by temporal differences on
EVERY state of the recorded rollouts (``record_policy_transitions.py``):

    V(s_t)   <- (1 - gamma) g(s_t) + gamma min(g(s_t), V'(s_{t+1}))
                V'(s_{t+1}) = g(s_L) when s_{t+1} = s_L is the fall, else the
                TARGET network's V(s_{t+1}) -- at the 3 s truncation s_T too
    Q(s, a)  <- (1 - gamma) g(s) + gamma min(g(s), V'(s'))        s' = step(s, a)

with the TD V as V' for Q (one recorded step per train_q row). gamma = 1 is
the rollout labels' min backup; gamma < 1 is safety_sb3's (backups.py).
Bootstrapping at the truncation makes the fixed point the policy's
INFINITE-horizon safety value (V >= 0: never leaves g >= 0), not the 3 s one,
and the network's errors feed its own targets.

At gamma = 1 that fixed point is not unique: on a rollout that never falls,
any V at or below the smallest g along it satisfies V = min(g, V'), and
nothing pulls V up to g, so TD from a network that starts near 0 stays there
(and the min selects its negative errors). ``--init g`` starts from the upper
bound V_0 = g instead (``--init-steps`` of regression on g(s)), from which
the iteration V_{k+1} = min(g, V_k') descends: in a table V_k is the k-step
value and the limit the LARGEST fixed point, the infinite-horizon value.
gamma < 1's (1 - gamma) g anchor makes the fixed point unique.

Inputs in --dir: td_train_v_traj_<tag>.pt, td_train_q_step_<tag>.pt (the
recordings), train_q.pt, {calib,test}_{v,q}.pt with their _labels_<tag>.pt (the
exact 3 s labels: the calibration and the ground truth here), {v,q}_<tag>.pt
(the rollout-label fits, for the comparison) and, optional,
td_calib_v_h6_<tag>.pt (6 s rollouts of calib states, g per step: the 3 s and
the 6 s outcome of the SAME rollout).

``--q-pi`` also fits Q on td_train_q_pi_step_<tag>.pt (``record_policy_transitions.py
--q-center pi``: pi's own first action plus the row's noise) and calibrates /
tests it on the SONIC- and pi-centred exact labels (``fit_policy_value.py
--with-pi``). ``--v-from <variant>`` takes V from v_<variant>_<tag>.pt instead
of fitting it (its gamma must match), so a Q-only refit keeps V bit for bit.

Outputs: {v,q}_<variant>_<tag>.pt / .json -- the state dict, eps and a report in
fit_policy_value.py's format, the thresholds calibrated on the same calib
labels -- and compare_<variant>_<tag>.json: both fits side by side (error and bias
against the 3 s labels, AUC, the false-safe threshold on test) and, with the
6 s rollouts, how each fit scores the states that are safe for 3 s but not 6.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
sys.path.insert(0, os.path.join(_ZOO, "examples"))

from fit_policy_value import _breakdown, load_splits  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_eval import (  # noqa: E402
  ALPHA_LADDER, ValueNet, evaluate_threshold, false_safe_threshold, features,
  residual_threshold)

CLAMP = 3.0          # g's clamp (margins.compose); the targets stay in it


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--dir", default=os.path.join(_ZOO, "data", "policy_eval"))
  p.add_argument("--tag", required=True, help="the policy tag (..._<tag>.pt)")
  p.add_argument("--gamma", type=float, default=1.0)
  p.add_argument("--init", choices=("default", "g"), default="default",
                 help="V's start: the default init, or regressed on g(s) (the upper bound)")
  p.add_argument("--init-steps", type=int, default=5000)
  p.add_argument("--variant", default="td",
                 help="name of this fit: {v,q}_<variant>_<tag>.pt, compare_<variant>_<tag>.json")
  p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
  p.add_argument("--steps", type=int, default=100000, help="TD steps for V")
  p.add_argument("--q-steps", type=int, default=20000, help="regression steps for Q")
  p.add_argument("--batch", type=int, default=4096)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--q-lr", type=float, default=1e-3)
  p.add_argument("--tau", type=float, default=0.005, help="target network Polyak rate")
  p.add_argument("--beta", type=float, default=1e-3)
  p.add_argument("--residual-alphas", type=float, nargs="+", default=[0.05, 0.01])
  p.add_argument("--q-pi", action="store_true",
                 help="also fit Q on the pi-centred recording (module docstring)")
  p.add_argument("--v-from", default=None,
                 help="take V from v_<this variant>_<tag>.pt instead of fitting it")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args(argv)


def backup(g0: torch.Tensor, nxt: torch.Tensor, gamma: float) -> torch.Tensor:
  return ((1.0 - gamma) * g0 + gamma * torch.minimum(g0, nxt)).clamp(-CLAMP, CLAMP)


def _predict(net, x: torch.Tensor, dev, chunk: int = 1 << 18) -> torch.Tensor:
  with torch.no_grad():
    return torch.cat([net(x[i:i + chunk].to(dev).float()).cpu() for i in range(0, x.shape[0], chunk)])


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float:
  """P(score of a positive > score of a negative), ties counted half."""
  from scipy.stats import rankdata
  pos = positive.bool().cpu()
  n1, n0 = int(pos.sum()), int((~pos).sum())
  if n1 == 0 or n0 == 0:
    return float("nan")
  rk = torch.as_tensor(rankdata(score.double().cpu().numpy()))
  return float((rk[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_v_td(a, rec: dict, x_ca, y_ca, dev) -> tuple[ValueNet, dict]:
  X = rec["x"].to(dev)                                   # [R, T + 1, F] float16
  G = rec["g"].to(dev)
  ns, fa = rec["n_states"].to(dev), rec["fell_at"].to(dev)
  T1 = G.shape[1]
  t = torch.arange(T1, device=dev)
  # transition s_t -> s_{t+1}: bootstrapped when s_{t+1} has an x (including
  # s_T, the truncation), terminal when s_{t+1} is the fall state
  boot = (t[None, :] + 1) <= (ns[:, None] - 1)
  term = (fa[:, None] > 0) & ((t[None, :] + 1) == fa[:, None])
  tr_r, tr_t = (boot | term).nonzero(as_tuple=True)
  tr_term = term[tr_r, tr_t]
  has_x = t[None, :] < ns[:, None]
  sr, st = has_x.nonzero(as_tuple=True)
  pick = torch.randperm(sr.numel(), device=dev)[:1_000_000]
  net = ValueNet(X.shape[2], tuple(a.hidden)).to(dev)
  net.set_normalizer(X[sr[pick], st[pick]].float())
  if a.init == "g":
    # V_0 = g: regress on every state's own margin, then descend from there
    gs = G[sr, st]
    opt0 = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(a.init_steps):
      i = torch.randint(0, sr.numel(), (a.batch,), device=dev)
      loss = torch.nn.functional.mse_loss(net(X[sr[i], st[i]].float()), gs[i])
      opt0.zero_grad()
      loss.backward()
      opt0.step()
    print(f"[td] V init on g(s): {a.init_steps} steps, mse {loss.item():.4f}", flush=True)
  tgt = copy.deepcopy(net)
  for p in tgt.parameters():
    p.requires_grad_(False)
  opt = torch.optim.Adam(net.parameters(), lr=a.lr)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
  M = tr_r.numel()
  print(f"[td] V: {rec['g'].shape[0]} rollouts, {sr.numel()} states, {M} transitions "
        f"({int(tr_term.sum())} into a fall, {int((boot & (t[None, :] + 1 == T1 - 1)).sum())} "
        f"bootstrapped at the truncation); gamma {a.gamma}, init {a.init}, {a.steps} steps, tau {a.tau}")
  log = []
  for step in range(a.steps):
    i = torch.randint(0, M, (a.batch,), device=dev)
    r, tt, tm = tr_r[i], tr_t[i], tr_term[i]
    with torch.no_grad():
      v_next = tgt(X[r, tt + 1].float())
      y = backup(G[r, tt], torch.where(tm, G[r, tt + 1], v_next), a.gamma)
    loss = torch.nn.functional.mse_loss(net(X[r, tt].float()), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    with torch.no_grad():
      for p, q in zip(net.parameters(), tgt.parameters()):
        q.lerp_(p, a.tau)
    if step % 5000 == 0 or step == a.steps - 1:
      s = _predict(net, x_ca, dev)
      row = {"step": step, "td_loss": loss.item(), "calib_mse_vs_3s": float(((s - y_ca) ** 2).mean()),
             "calib_bias_vs_3s": float((s - y_ca).mean())}
      log.append(row)
      print(f"[td] step {step}: td loss {row['td_loss']:.4f}  calib mse vs 3 s labels "
            f"{row['calib_mse_vs_3s']:.4f}  bias {row['calib_bias_vs_3s']:+.4f}", flush=True)
  net.eval()
  info = {"rollouts": int(rec["g"].shape[0]), "states": int(sr.numel()), "transitions": int(M),
          "into_fall": int(tr_term.sum()), "init": a.init, "steps": a.steps, "tau": a.tau, "log": log}
  return net, info


def _q_targets(a, rec: dict, raw_q: dict, v_net, dev):
  """Q's inputs and targets of one recording: its rows' actions (the played
  ones of a pi-centred recording), and how many rows fell after the action."""
  rows = rec["rows"]
  qa = raw_q["q_action"]
  if "q_action" in rec:                        # --q-center pi: the played actions
    qa = qa.clone()
    qa[rows] = rec["q_action"].to(qa.dtype)
  x = features(raw_q, qa)[rows]
  G, ns, fa = rec["g"], rec["n_states"], rec["fell_at"]
  term = fa == 1
  ok = term | (ns >= 2)
  v1 = _predict(v_net, rec["x"][:, 1], dev)
  y = backup(G[:, 0], torch.where(term, G[:, 1], v1), a.gamma)
  return x[ok], y[ok], int(term.sum())


def fit_q_td(a, recs: list, raw_q: dict, v_net, dev) -> tuple[ValueNet, dict]:
  parts = [_q_targets(a, rec, raw_q, v_net, dev) for rec in recs]
  xs = torch.cat([p[0] for p in parts]).to(dev)
  ys = torch.cat([p[1] for p in parts]).to(dev)
  n_term = sum(p[2] for p in parts)
  print(f"[td] Q: {xs.shape[0]} (s, a) rows ({' + '.join(str(p[0].shape[0]) for p in parts)}), "
        f"{n_term} fall after the action; target min(g(s), V_td(s'))")
  net = ValueNet(xs.shape[1], tuple(a.hidden)).to(dev)
  net.set_normalizer(xs)
  opt = torch.optim.Adam(net.parameters(), lr=a.q_lr)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.q_steps)
  for step in range(a.q_steps):
    i = torch.randint(0, xs.shape[0], (a.batch,), device=dev)
    loss = torch.nn.functional.mse_loss(net(xs[i]), ys[i])
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 5000 == 0 or step == a.q_steps - 1:
      print(f"[td] Q step {step}: mse to the bootstrapped targets {loss.item():.4f}", flush=True)
  net.eval()
  return net, {"rows": int(xs.shape[0]), "rows_per_recording": [int(p[0].shape[0]) for p in parts],
               "fall_after_action": int(n_term)}


def report(a, mode: str, net, splits, dev, extra: dict) -> dict:
  """fit_policy_value.py's report for ``net``, on the same calib / test labels."""
  (x_ca, y_ca), (x_te, y_te, te_rows) = splits
  s_ca, s_te = _predict(net, x_ca, dev), _predict(net, x_te, dev)
  rep = {"mode": mode, "tag": a.tag, "method": a.variant, "gamma": a.gamma, **extra,
         "n_calib": int(y_ca.numel()), "n_test": int(y_te.numel()),
         "calib_safe_frac": float((y_ca >= 0).float().mean()),
         "test_safe_frac": float((y_te >= 0).float().mean()),
         "test_mse": float(((s_te - y_te) ** 2).mean()), "thresholds": {}}
  rep["thresholds"]["false_safe"] = {"calibration": false_safe_threshold(
    s_ca, y_ca < 0.0, beta=a.beta, alphas=ALPHA_LADDER)}
  for alpha in a.residual_alphas:
    rep["thresholds"][f"residual_{alpha:g}"] = {"calibration": residual_threshold(s_ca, y_ca, alpha)}
  for name, th in rep["thresholds"].items():
    eps = th["calibration"]["eps"]
    th["test"] = evaluate_threshold(s_te, y_te, eps)
    th["test_breakdown"] = _breakdown(te_rows, s_te, y_te, eps) if math.isfinite(eps) else {}
    c, t = th["calibration"], th["test"]
    print(f"[{a.variant}] {mode} {name}: eps {eps:+.4f}" + (f" (alpha {c.get('alpha')})" if name == "false_safe" else "")
          + f"  | test: accept {t['accept_rate']:.3f}, false-safe {t['false_safe_rate']:.4f} "
            f"({t['false_safe']}/{t['accepted']}), recall {t['recall']:.3f}")
  return rep


def _splits(d: str, mode: str, tag: str, with_pi: bool = False):
  sp = load_splits(d, mode, tag, with_pi, splits=("calib", "test"))
  _, x_ca, v_ca = sp["calib"]
  te_rows, x_te, v_te = sp["test"]
  c_ok, t_ok = ~torch.isnan(v_ca[:, 0]), ~torch.isnan(v_te[:, 0])
  te_rows = {k: (v[t_ok] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == t_ok.numel() else v)
             for k, v in te_rows.items()}
  return (x_ca[c_ok], v_ca[c_ok, 0]), (x_te[t_ok], v_te[t_ok, 0], te_rows)


def _load_net(path: str, dev) -> tuple[ValueNet, dict]:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  net = ValueNet(ck["in_dim"], tuple(ck["hidden"]))
  net.load_state_dict(ck["state_dict"])
  return net.to(dev).eval(), ck


def _side(net, splits, rep_fs: dict, dev) -> dict:
  (x_ca, y_ca), (x_te, y_te, _) = splits
  s_ca, s_te = _predict(net, x_ca, dev), _predict(net, x_te, dev)
  return {"calib_mse_vs_3s": float(((s_ca - y_ca) ** 2).mean()),
          "test_mse_vs_3s": float(((s_te - y_te) ** 2).mean()),
          "test_bias_vs_3s": float((s_te - y_te).mean()),
          "test_auc_safe_3s": _auc(s_te, y_te >= 0),
          "test_pred_safe_frac": float((s_te >= 0).float().mean()),
          "test_label_safe_frac": float((y_te >= 0).float().mean()),
          "false_safe": {"eps": rep_fs["calibration"]["eps"], "alpha": rep_fs["calibration"].get("alpha"),
                         **{k: rep_fs["test"][k] for k in ("accept_rate", "false_safe_rate", "recall")}}}


def horizon_check(d: str, tag: str, nets: dict, dev) -> dict | None:
  """The 3 s and 6 s outcome of the same calib rollouts, and each fit's V on them."""
  path = os.path.join(d, f"td_calib_v_h6_{tag}.pt")
  if not os.path.exists(path):
    return None
  rec = torch.load(path, map_location="cpu", weights_only=False)
  dt, G = float(rec["meta"]["dt"]), rec["g"]
  T3 = int(round(3.0 / dt))
  done = rec["n_states"] > 0
  Gi = torch.where(torch.isnan(G), torch.full_like(G, float("inf")), G)
  h3, h6 = Gi[:, :T3 + 1].amin(1), Gi.amin(1)
  rows = torch.load(os.path.join(d, "calib_v.pt"), map_location="cpu", weights_only=False)
  x = features(rows)[rec["rows"]][done]
  h3, h6 = h3[done], h6[done]
  late = (h3 >= 0) & (h6 < 0)
  out = {"rollouts": int(done.sum()), "safe_3s": float((h3 >= 0).float().mean()),
         "safe_6s": float((h6 >= 0).float().mean()), "safe_3s_unsafe_6s": float(late.float().mean()),
         "fits": {}}
  for name, net in nets.items():
    s = _predict(net, x, dev)
    out["fits"][name] = {"mse_vs_3s": float(((s - h3) ** 2).mean()), "mse_vs_6s": float(((s - h6) ** 2).mean()),
                         "auc_safe_3s": _auc(s, h3 >= 0), "auc_safe_6s": _auc(s, h6 >= 0),
                         "mean_v_safe_both": float(s[(h6 >= 0)].mean()) if bool((h6 >= 0).any()) else None,
                         "mean_v_safe_3s_unsafe_6s": float(s[late].mean()) if bool(late.any()) else None,
                         "pred_unsafe_on_safe_3s_unsafe_6s": float((s[late] < 0).float().mean()) if bool(late.any()) else None}
  return out


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev, d = a.device, a.dir
  t0 = time.time()
  sp_v, sp_q = _splits(d, "v", a.tag), _splits(d, "q", a.tag, a.q_pi)

  if a.v_from:
    path = os.path.join(d, f"v_{a.v_from}_{a.tag}.pt")
    v_td, ck = _load_net(path, dev)
    if abs(float(ck.get("gamma", 1.0)) - a.gamma) > 1e-12:
      raise SystemExit(f"{path} was fit with gamma {ck.get('gamma')}, not {a.gamma}")
    info_v = {"from": os.path.abspath(path)}
    print(f"[td] V from {path}")
  else:
    rec_v = torch.load(os.path.join(d, f"td_train_v_traj_{a.tag}.pt"), map_location="cpu", weights_only=False)
    v_td, info_v = fit_v_td(a, rec_v, sp_v[0][0], sp_v[0][1], dev)
    del rec_v
  names = [f"td_train_q_step_{a.tag}.pt"] + ([f"td_train_q_pi_step_{a.tag}.pt"] if a.q_pi else [])
  recs = [torch.load(os.path.join(d, nm), map_location="cpu", weights_only=False) for nm in names]
  raw_q = torch.load(os.path.join(d, "train_q.pt"), map_location="cpu", weights_only=False)
  q_td, info_q = fit_q_td(a, recs, raw_q, v_td, dev)
  del recs, raw_q

  comp = {"tag": a.tag, "variant": a.variant, "gamma": a.gamma, "init": a.init,
          "td": {"v": info_v, "q": info_q}, "v": {}, "q": {}}
  nets_v = {a.variant: v_td}
  for mode, net, sp, info in (("v", v_td, sp_v, info_v), ("q", q_td, sp_q, info_q)):
    rep = report(a, mode, net, sp, dev, {k: v for k, v in info.items() if k != "log"})
    out = os.path.join(d, f"{mode}_{a.variant}_{a.tag}.pt")
    torch.save({"state_dict": net.state_dict(), "in_dim": int(net.mean.numel()), "hidden": list(a.hidden),
                "gamma": a.gamma, "init": a.init, "eps": {n: th["calibration"]["eps"] for n, th in rep["thresholds"].items()},
                "report": rep}, out)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
      json.dump(rep, f, indent=1)
    comp[mode][a.variant] = _side(net, sp, rep["thresholds"]["false_safe"], dev)
    # the rollout-label fits on the same splits (with --q-pi: SONIC- and pi-centred)
    others = {"rollout_labels": f"{mode}_{a.tag}.pt"}
    if mode == "q" and a.q_pi:
      others["rollout_labels_pi"] = f"q_exact_pi_{a.tag}.pt"
    for name, fn in others.items():
      mc_path = os.path.join(d, fn)
      if os.path.exists(mc_path):
        mc, ck = _load_net(mc_path, dev)
        comp[mode][name] = _side(mc, sp, ck["report"]["thresholds"]["false_safe"], dev)
        if mode == "v" and name == "rollout_labels":
          nets_v["rollout_labels"] = mc
  h = horizon_check(d, a.tag, nets_v, dev)
  if h is not None:
    comp["horizon_check"] = h
  out = os.path.join(d, f"compare_{a.variant}_{a.tag}.json")
  with open(out, "w") as f:
    json.dump(comp, f, indent=1)
  for mode in ("v", "q"):
    for name, s in comp[mode].items():
      fs = s["false_safe"]
      print(f"[compare] {mode} {name:14s}: test mse vs 3 s {s['test_mse_vs_3s']:.4f}, bias "
            f"{s['test_bias_vs_3s']:+.4f}, AUC {s['test_auc_safe_3s']:.3f} | false-safe eps "
            f"{fs['eps']:+.4f}: accept {fs['accept_rate']:.3f}, false-safe {fs['false_safe_rate']:.4f}, "
            f"recall {fs['recall']:.3f}")
  if h is not None:
    print(f"[compare] 6 s rollouts of {h['rollouts']} calib states: safe for 3 s {h['safe_3s']:.3f}, "
          f"for 6 s {h['safe_6s']:.3f}, safe 3 s but not 6 s {h['safe_3s_unsafe_6s']:.4f}")
    for name, s in h["fits"].items():
      print(f"[compare]   {name:14s}: mse vs 3 s {s['mse_vs_3s']:.4f}, vs 6 s {s['mse_vs_6s']:.4f}; "
            f"AUC 3 s {s['auc_safe_3s']:.3f}, 6 s {s['auc_safe_6s']:.3f}; V < 0 on the late falls "
            f"{s['pred_unsafe_on_safe_3s_unsafe_6s']}")
  print(f"[compare] -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
  main()
