"""Policy evaluation of a fallback's Q^pi with the backup its critic was trained with.

    PYTHONPATH=$HOME/safety-stable-baselines-baselines \
    python examples/q_fallback_fit.py --data runs/v061_study/q_eval/data_safetyppo.pt \
        --estimator critic1 --delta 0.8334 --out runs/v061_study/q_eval/q_critic1_safetyppo.pt
    ... --data .../data_lambdareach.pt --estimator critic1 --delta 0.8345 --out .../q_critic1_lambdareach.pt
    ... --estimator td1 / lambda_reach (each arm's own backup)

mjc316 (sysID v0.6) port of the main checkout's script: the fit and its validation are unchanged;
additive only -- the Q file also carries what a consumer needs to rebuild the input (task, policy +
sha256, obs_dim, the critic-obs sanitize: nan -> 0 and the clamped last-action columns, the
proposal bound, the estimator, ``--delta``), a Monte Carlo check over all first actions, and with
``--delta`` (default off) a threshold report: on the rollouts' bank start states s_0 the fraction
with Q(s_0, pi(s_0)) >= delta vs V(s_0) >= delta (``pi0`` of the data), and per first-action kind
the fraction of proposals Q(s_0, a_0) >= delta accepts and the fall rates of the accepted / rejected.

Q^pi(s, a) of the fallback pi on rollouts "a first, then pi" (``recover_q_collect.py``: the
fallback's own g1_recover training setting; or ``walk_sailr_q_collect.py``): step k of a rollout
s_k, a_k, g_k = margin of s_{k+1}, done_k; both baselines' AVOID operator at gamma 0.99
(safety_sb3.backups; no time-out bootstrap, a rollout's end is terminal):

  td1           SafetyPPO: the one-step avoid target  y_k = (1-gamma) g_k + gamma min(g_k, V_{k+1}),
                y_k = g_k where done_k  (backups.avoid_target; SafetyPPO's GAE is built on it)
  lambda_reach  lambda-Reach: backups.lambda_reach_return(g, done, V_next, lam 0.99, delta 0.99,
                n_max 200, "expected", "renormalize") -- its training estimator

with V_{k+1} = Q_target(s_{k+1}, a_{k+1}), a_{k+1} the fallback's own action (stored), so the
bootstrap is Q^pi(s', pi(s')). The target net is refreshed every ``--target-every`` steps (FQE
rounds), the targets recomputed over the whole data set each round. And

  critic1       ONE backup of the fallback's own trained critic: y_k = (1-gamma) g_k + gamma
                min(g_k, V_critic(s_{k+1})), y_k = g_k where done_k (the stored ``v``). Fixed
                targets; Q then lives on the critic's scale, so a delta calibrated for the
                critic's V means "the action keeps the robot in {V >= delta}".

Q net: MLP on cat(s, a) (the sanitized critic group of the filter's input -- 113-D on
g1_recover_noheight_v6, 114-D on the old g1_recover -- and the executed
29-D action), inputs standardized by the train split's mean / std, ``--hidden`` ELU layers.
``--use-ctx``: s also carries the rollout's 33-D domain-randomization context (the data's
``ctx``; the same ``sim_ctx`` Beat-It's world model reads): input cat(critic group, ctx, a).

Validation on held-out rollouts (``--val-frac``):
  * consistency with the fallback's critic: Q(s_k, pi(s_k)) vs V(s_k), k >= 1 (and k = 0 where
    the first action was pi's own);
  * Monte Carlo: Q(s_0, a_0) vs the rollout's realized avoid value G_0 (the avoid recursion along
    the stored margins, terminal at its end), per first-action kind;
  * fall separation: AUC of Q(s_0, a_0) vs of V(s_0) for "the rollout fell", per first-action
    kind (and per Bare checkpoint on walk data).
Output: ``--out`` (state dict, input stats, settings, report) and the report as ``.json``.
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
if _ZOO not in sys.path:
  sys.path.insert(0, _ZOO)

from robot_safety_sandbox.filters.q_net import QNet  # noqa: E402

GAMMA = 0.99


def auc(score: torch.Tensor, pos: torch.Tensor) -> float:
  """P(score of a negative > score of a positive) + ties/2: high score = safe, pos = fell."""
  score, pos = score.double().cpu(), pos.bool().cpu()
  n1, n0 = int(pos.sum()), int((~pos).sum())
  if n1 == 0 or n0 == 0:
    return math.nan
  r = torch.empty_like(score)
  o = torch.argsort(score)
  r[o] = torch.arange(1, len(score) + 1, dtype=torch.double)
  # average ranks over ties
  uniq, inv, cnt = torch.unique(score, return_inverse=True, return_counts=True)
  if int(cnt.max()) > 1:
    s = torch.zeros(len(uniq), dtype=torch.double).index_add_(0, inv, r)
    r = (s / cnt.double())[inv]
  return float((r[~pos].sum() - n0 * (n0 + 1) / 2) / (n0 * n1))


def avoid_mc(g: torch.Tensor, L: torch.Tensor, gamma: float) -> torch.Tensor:
  """[R] realized avoid value at k = 0: G_{L-1} = g_{L-1}, G_k = (1-gamma) g_k + gamma min(g_k, G_{k+1})."""
  R, H = g.shape
  G = torch.zeros(R, device=g.device)
  for k in range(H - 1, -1, -1):
    gk = g[:, k]
    cont = (1 - gamma) * gk + gamma * torch.minimum(gk, G)
    G = torch.where(k == L - 1, gk, torch.where(k < L - 1, cont, G))
  return G


def stats(x: torch.Tensor, y: torch.Tensor) -> dict:
  x, y = x.double(), y.double()
  if x.numel() < 2:
    return {"n": int(x.numel())}
  r = torch.corrcoef(torch.stack([x, y]))[0, 1]
  return {"n": int(x.numel()), "mae": float((x - y).abs().mean()), "bias": float((x - y).mean()),
          "pearson": float(r), "mean_q": float(x.mean()), "mean_ref": float(y.mean())}


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--data", required=True)
  p.add_argument("--estimator", choices=("td1", "lambda_reach", "critic1"), required=True)
  p.add_argument("--gamma", type=float, default=GAMMA)
  p.add_argument("--reach-lambda", type=float, default=0.99)
  p.add_argument("--reach-delta", type=float, default=0.99)
  p.add_argument("--reach-n-max", type=int, default=200)
  p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
  p.add_argument("--steps", type=int, default=60000)
  p.add_argument("--batch", type=int, default=65536)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--target-every", type=int, default=250)
  p.add_argument("--val-frac", type=float, default=0.1)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--use-ctx", action="store_true",
                 help="Q also reads the 33-D domain-randomization context (data from --store-ctx)")
  p.add_argument("--delta", type=float, default=None,
                 help="the consumer's threshold (execute a iff Q(s, a) >= delta): stored in the Q "
                      "file and scored on the data (default: none)")
  p.add_argument("--out", required=True)
  a = p.parse_args(argv)
  dev = "cuda:0"
  torch.manual_seed(a.seed)
  from safety_sb3 import backups

  D = torch.load(a.data, map_location="cpu", weights_only=False)
  meta = D["meta"]
  S = D["s"].to(dev)                       # [R, H, s] fp16
  A = D["a"].to(dev)
  G = D["g"].float().to(dev)
  DONE = D["done"].bool().to(dev)
  V = D["v"].float().to(dev)
  L = D["len"].long().to(dev)
  R, H, s_dim = S.shape
  a_dim = A.shape[-1]
  obs_dim, ctx_dim = s_dim, 0
  if a.use_ctx:
    if "ctx" not in D:
      raise SystemExit(f"--use-ctx: {a.data} has no ctx (collect with --store-ctx)")
    C = D["ctx"].to(dev).half()
    ctx_dim = int(C.shape[1])
    S = torch.cat([S, C.unsqueeze(1).expand(R, H, ctx_dim)], dim=-1)
    s_dim = obs_dim + ctx_dim
  ar = torch.arange(H, device=dev)
  valid = ar.unsqueeze(0) < L.unsqueeze(1)                     # [R, H]
  if not bool(DONE[torch.arange(R, device=dev), L - 1].all()):
    raise ValueError("every window's last valid step must be done (fall or time-out)")
  perm = torch.randperm(R, generator=torch.Generator().manual_seed(a.seed)).to(dev)
  n_val = int(round(a.val_frac * R))
  is_val = torch.zeros(R, dtype=torch.bool, device=dev)
  is_val[perm[:n_val]] = True
  tr_mask = valid & ~is_val.unsqueeze(1)
  tr_idx = tr_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
  Sf, Af = S.reshape(R * H, s_dim), A.reshape(R * H, a_dim)

  q = QNet(s_dim, a_dim, a.hidden).to(dev)
  with torch.no_grad():
    x = torch.cat([Sf[tr_idx].float(), Af[tr_idx].float()], dim=-1)
    q.mu.copy_(x.mean(0))
    q.sd.copy_(x.std(0).clamp_min(1e-3))
    del x
  qt = QNet(s_dim, a_dim, a.hidden).to(dev)
  qt.load_state_dict(q.state_dict())
  opt = torch.optim.Adam(q.parameters(), lr=a.lr)

  @torch.no_grad()
  def q_all(net) -> torch.Tensor:
    out = torch.empty(R * H, device=dev)
    for i in range(0, R * H, 1 << 20):
      out[i:i + (1 << 20)] = net(Sf[i:i + (1 << 20)], Af[i:i + (1 << 20)])
    return out.reshape(R, H)

  @torch.no_grad()
  def targets() -> torch.Tensor:
    if a.estimator == "critic1":                 # fixed: one backup of the trained critic
      vn = torch.zeros_like(V)
      vn[:, :-1] = V[:, 1:]                      # V_critic(s_{k+1})
      return backups.avoid_target(G, vn, (~DONE).float(), a.gamma).reshape(-1)
    qb = q_all(qt)
    vn = torch.zeros_like(qb)
    vn[:, :-1] = qb[:, 1:]                       # Q(s_{k+1}, pi(s_{k+1}))
    if a.estimator == "td1":
      y = backups.avoid_target(G, vn, (~DONE).float(), a.gamma)
    else:
      y = backups.lambda_reach_return(G.T.contiguous(), DONE.T.contiguous(), vn.T.contiguous(),
                                      lam=a.reach_lambda, delta=a.reach_delta,
                                      n_max=a.reach_n_max, target="expected",
                                      tail="renormalize").T
    return y.reshape(-1)

  vm = valid & is_val.unsqueeze(1)
  t0 = time.time()
  y = targets()
  log = []
  for step in range(1, a.steps + 1):
    b = tr_idx[torch.randint(0, tr_idx.numel(), (a.batch,), device=dev)]
    loss = torch.mean((q(Sf[b], Af[b]) - y[b]) ** 2)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % a.target_every == 0:
      qt.load_state_dict(q.state_dict())
      if a.estimator != "critic1":
        y = targets()
      if step % (a.target_every * 20) == 0 or step == a.steps:
        with torch.no_grad():
          qa = q_all(q)
          res = (qa - y.reshape(R, H))[vm]
          rec = {"step": step, "train_loss": float(loss), "val_bellman_rmse": float(res.pow(2).mean().sqrt()),
                 "val_q_mean": float(qa[vm].mean()), "s": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"[q-fit] {a.estimator} step {step}: loss {rec['train_loss']:.5f} val Bellman RMSE "
              f"{rec['val_bellman_rmse']:.4f} mean Q {rec['val_q_mean']:+.3f} ({rec['s']:.0f} s)",
              flush=True)

  # --- validation ------------------------------------------------------------------
  with torch.no_grad():
    qa = q_all(q)
  fk = D["first_kind"].to(dev)
  fell = D["fell"].bool().to(dev)
  names = list(meta.get("first_kinds") or ["learner", "fallback"])
  pi_kind = names.index("pi") if "pi" in names else names.index("fallback")
  mc = avoid_mc(G, L, a.gamma)
  k1 = vm.clone()
  k1[:, 0] = False
  pk = is_val & (fk == pi_kind)
  rep = {"rows": R, "val_rows": n_val, "H": H, "estimator": a.estimator, "gamma": a.gamma,
         "first_kinds": names, "log": log,
         "critic_consistency_k_ge_1": stats(qa[k1], V[k1]),
         "critic_consistency_k0_pi_first": stats(qa[:, 0][pk], V[:, 0][pk]),
         "critic_vs_mc_k0_pi_first": stats(V[:, 0][pk], mc[pk]),
         "per_kind": {}}
  for i_, nm in enumerate(names):
    m = is_val & (fk == i_)
    if not bool(m.any()):
      continue
    rep["per_kind"][nm] = {"n": int(m.sum()), "fell": float(fell[m].float().mean()),
                           "mc": stats(qa[:, 0][m], mc[m]),
                           "auc_fell": {"Q(s,a)": auc(qa[:, 0][m], fell[m]),
                                        "V(s)": auc(V[:, 0][m], fell[m])},
                           "mean_Q": float(qa[:, 0][m].mean()),
                           "mean_V": float(V[:, 0][m].mean())}
  nonpi = is_val & (fk != pi_kind)
  rep["auc_fell_all_first_actions"] = {"Q(s,a)": auc(qa[:, 0][nonpi | pk], fell[nonpi | pk]),
                                       "V(s)": auc(V[:, 0][nonpi | pk], fell[nonpi | pk])}
  # (port, additive) Monte Carlo over every first action of the held-out rollouts
  rep["mc_all_first_actions"] = {"Q(s0,a0)": stats(qa[:, 0][is_val], mc[is_val]),
                                 "V(s0)": stats(V[:, 0][is_val], mc[is_val])}
  # (port, additive) the consumer's threshold on the bank start states s_0
  if a.delta is not None:
    dl = float(a.delta)
    q_pi0 = None
    if "pi0" in D:
      with torch.no_grad():
        q_pi0 = q(S[:, 0], D["pi0"].to(dev)).float()             # Q(s_0, pi(s_0)), every rollout
    v0, q0 = V[:, 0], qa[:, 0]

    def frac_block(m):
      out_ = {"n": int(m.sum()), "fell": float(fell[m].float().mean()),
              "frac_V_ge_delta": float((v0[m] >= dl).float().mean())}
      if q_pi0 is not None:
        qa_, va_ = q_pi0[m] >= dl, v0[m] >= dl
        f_ = fell[m]
        out_.update({"frac_Qpi_ge_delta": float(qa_.float().mean()),
                     "agree": float((qa_ == va_).float().mean()),
                     "Q_only": float((qa_ & ~va_).float().mean()),
                     "V_only": float((va_ & ~qa_).float().mean()),
                     "fell_given_Qpi_ge_delta": float(f_[qa_].float().mean()) if bool(qa_.any()) else math.nan,
                     "fell_given_V_ge_delta": float(f_[va_].float().mean()) if bool(va_.any()) else math.nan,
                     "Q(s0,pi(s0)) vs V(s0)": stats(q_pi0[m], v0[m])})
      return out_

    allr = torch.ones(R, dtype=torch.bool, device=dev)
    dr = {"delta": dl, "start_states_val": frac_block(is_val), "start_states_all": frac_block(allr),
          "per_kind_accept": {}}
    for i_, nm in enumerate(names):
      m = is_val & (fk == i_)
      if not bool(m.any()):
        continue
      acc = q0[m] >= dl
      dr["per_kind_accept"][nm] = {
        "n": int(m.sum()), "accept_frac": float(acc.float().mean()),
        "fell_accepted": float(fell[m][acc].float().mean()) if bool(acc.any()) else math.nan,
        "fell_rejected": float(fell[m][~acc].float().mean()) if bool((~acc).any()) else math.nan}
    rep["delta_report"] = dr
  if "ckpt" in D:
    ck = D["ckpt"].to(dev)
    per = {}
    for it in sorted(set(ck.tolist())):
      m = is_val & (fk != pi_kind) & (ck == it)
      per[int(it)] = {"n": int(m.sum()), "fell": float(fell[m].float().mean()),
                      "auc_Q": auc(qa[:, 0][m], fell[m]), "auc_V": auc(V[:, 0][m], fell[m])}
    rep["per_ckpt"] = per
  c1, c0 = rep["critic_consistency_k_ge_1"], rep["critic_consistency_k0_pi_first"]
  print(f"[q-fit] {a.estimator}: Q(s,pi(s)) vs critic V: k>=1 MAE {c1.get('mae', math.nan):.4f} "
        f"bias {c1.get('bias', math.nan):+.4f} r {c1.get('pearson', math.nan):.4f} | k=0 MAE "
        f"{c0.get('mae', math.nan):.4f} r {c0.get('pearson', math.nan):.4f}", flush=True)
  for nm, r in rep["per_kind"].items():
    print(f"[q-fit]   first {nm:10s} n {r['n']:6d} fell {r['fell']:.3f} | vs MC: MAE "
          f"{r['mc'].get('mae', math.nan):.3f} r {r['mc'].get('pearson', math.nan):.3f} | AUC(fell) "
          f"Q {r['auc_fell']['Q(s,a)']:.3f} V {r['auc_fell']['V(s)']:.3f} | mean Q "
          f"{r['mean_Q']:+.3f} V {r['mean_V']:+.3f}", flush=True)
  print(f"[q-fit]   all first actions: AUC(fell) Q {rep['auc_fell_all_first_actions']['Q(s,a)']:.3f}"
        f" V {rep['auc_fell_all_first_actions']['V(s)']:.3f}", flush=True)
  mca = rep["mc_all_first_actions"]
  print(f"[q-fit]   all first actions vs MC: Q(s0,a0) MAE {mca['Q(s0,a0)'].get('mae', math.nan):.3f} "
        f"r {mca['Q(s0,a0)'].get('pearson', math.nan):.3f} | V(s0) MAE "
        f"{mca['V(s0)'].get('mae', math.nan):.3f} r {mca['V(s0)'].get('pearson', math.nan):.3f}",
        flush=True)
  if "delta_report" in rep:
    dr = rep["delta_report"]
    for nm_ in ("start_states_val", "start_states_all"):
      b_ = dr[nm_]
      print(f"[q-fit]   delta {dr['delta']:.4f} on {nm_} (n {b_['n']}): V>=d {b_['frac_V_ge_delta']:.3f}"
            + (f" | Q(s,pi(s))>=d {b_['frac_Qpi_ge_delta']:.3f} agree {b_['agree']:.3f} (Q only "
               f"{b_['Q_only']:.3f}, V only {b_['V_only']:.3f}) | fell | Q>=d {b_['fell_given_Qpi_ge_delta']:.3f}"
               f" V>=d {b_['fell_given_V_ge_delta']:.3f}" if "frac_Qpi_ge_delta" in b_ else ""),
            flush=True)
    for nm_, b_ in dr["per_kind_accept"].items():
      print(f"[q-fit]   delta accept first {nm_:10s}: Q(s0,a0)>=d {b_['accept_frac']:.3f} | fell "
            f"accepted {b_['fell_accepted']:.3f} rejected {b_['fell_rejected']:.3f}", flush=True)
  san = meta.get("sanitize") or {"group": "critic", "dim": obs_dim,
                                 "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                                 "clamp_cols": [[69, 98]], "clamp": meta.get("fallback_bound", 3.0)}
  pb_ = float(meta.get("proposal_bound", 6.5))
  cols = ", ".join(f"{c0_}:{c1_}" for c0_, c1_ in san["clamp_cols"])
  recipe = (f"s = env {meta.get('task')} '{san['group']}' obs group ({san['dim']}-D, noise off), "
            f"nan/+inf/-inf -> 0, columns {cols} (last action) clamped to +-{san['clamp']:g}"
            + (f", then cat the {ctx_dim}-D sim_context" if ctx_dim else "")
            + f"; a = the {a_dim}-D proposal in env action units clamped to +-{pb_:g}; "
            f"Q = net(s, a) (the net standardizes cat(s, a) itself)")
  out = {"state_dict": q.state_dict(), "s_dim": s_dim, "a_dim": a_dim, "hidden": list(a.hidden),
         "obs_dim": obs_dim, "ctx_dim": ctx_dim,
         "inputs": ["fb_critic sanitized as the filter's critic_obs (%d)" % obs_dim]
                   + (["sim_ctx (%d)" % ctx_dim] if ctx_dim else []) + ["action (%d)" % a_dim],
         "estimator": a.estimator, "gamma": a.gamma,
         "reach": {"lambda": a.reach_lambda, "delta": a.reach_delta, "n_max": a.reach_n_max},
         # (port, additive) what a consumer needs to rebuild the input and threshold it
         "task": meta.get("task"), "policy": meta.get("policy"),
         "policy_sha256": meta.get("policy_sha256"), "bank": meta.get("bank"),
         "sanitize": san, "proposal_bound": pb_,
         "fallback_bound": float(meta.get("fallback_bound", 3.0)),
         "obs_layout": meta.get("obs_layout"), "delta": a.delta,
         "delta_source": ("the fallback critic V's threshold of a Wang eps = 1% calibration, "
                          "borrowed for Q" if a.delta is not None else None),
         "q_input_recipe": recipe,
         "settings": vars(a), "data": os.path.abspath(a.data), "data_meta": meta, "report": rep}
  torch.save(out, a.out)
  with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
    json.dump({k: v for k, v in out.items() if k != "state_dict"}, f, indent=1, default=str)
  print(f"[q-fit] -> {a.out}", flush=True)


if __name__ == "__main__":
  main()
