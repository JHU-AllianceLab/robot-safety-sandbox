"""Score SHAC's model-based Q (its world model + critic) against exact 3 s labels.

    python examples/score_world_model_q.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip --tag shac_s0

A SafetySHAC1P checkpoint carries a world model f (one step of the RAW
privileged critic obs given the action and the sim context, with heads for the
margin of the next state and the probability that the step terminates) and a
critic V (the gamma 0.99 avoid value of the normalized critic obs). Their
one-step backup is a Q, the critic's own target with H = 1:

    Q_f(s, a) = (1 - p) * ((1 - gamma) g + gamma min(g, V(f(s, a)))) + p * g_term

(``safety_sb3.backups.lambda_return``, lam = 1, soft termination; g / g_term
the margin of s' given the step continues / terminates). f was fit on SHAC's
own BOX-CLAMPED actions plus N(0, 0.1^2) noise; SONIC's actions mostly leave
the +-3 box, so Q_f there is an extrapolation. ``model`` feeds the action as
played (unclamped, as the labels played it), ``model_box`` clamps it first.

On the held-out states of --states (policy_eval's test_q.pt: SONIC push
trajectory states on the held-out clips) each row has exact 3 s labels at two
first actions: SONIC's action plus the row's noise (``test_q_labels_<tag>.pt``)
and pi's own action plus the same noise (``test_q_pi_labels_<tag>.pt``, which
holds the played action). Each row's state is restored in the env (the
labeller's sequential spawn), and the script scores:

1. SEPARATION: AUROC of the label's safe (v >= 0) and Spearman rho with v,
   per action centre, for Q_f and for the fitted Q nets (exact, exact_pi,
   td_g099, td_g099_pi; a missing one is skipped);
2. PAIRS: on each state, which of its two labelled actions is safer -- what a
   Q-CBF correction relies on -- over the pairs whose labels disagree on
   safety, and over those whose labels differ by more than --pair-gap (single
   rollouts under domain randomization: the labels are noisy, so 1.0 is out of
   reach);
3. f's ONE-STEP ERROR: the restored state is stepped once under u_SONIC and,
   restored again, once under pi(s); f's predicted next state is compared with
   the real one, as the error of the predicted CHANGE relative to the real
   change (dims f predicts; 1.0 = no better than "nothing moves");
4. V: SHAC's critic and the fitted V nets against the state's exact V label
   (``test_v_labels_<tag>.pt``).

Writes ``<out>.json`` (the numbers) and ``<out>.pt`` (per-row scores).
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
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from safety_sb3 import backups  # noqa: E402

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_eval import features  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_value_filter import (  # noqa: E402
  load_value_net)
from robot_safety_sandbox.envs.g1_recover.spawn import RecoverSpawn  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import find_obs_stats, load_twin  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402

#: The fitted Q / V nets compared with Q_f, by name -> file stem in --value-dir.
FITTED_Q = {"fit exact": "q_{tag}", "fit exact_pi": "q_exact_pi_{tag}",
            "fit td_g099": "q_td_g099_{tag}", "fit td_g099_pi": "q_td_g099_pi_{tag}"}
FITTED_V = {"fit exact": "v_{tag}", "fit td_g099": "v_td_g099_{tag}"}


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="a SafetySHAC1P final_model.zip")
  p.add_argument("--tag", required=True, help="its label / fit tag, e.g. shac_s0")
  p.add_argument("--value-dir", default=os.path.join(_ZOO, "data", "policy_eval_gfall"))
  p.add_argument("--states", default=None, help="default <value-dir>/test_q.pt")
  p.add_argument("--task", default="g1_recover_lafan", help="the labels' task")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--pair-gap", type=float, default=0.25,
                 help="pairs whose labels differ by more than this also count")
  p.add_argument("--out", default=None, help="default <value-dir>/model_q_<tag>")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  a.states = a.states or os.path.join(a.value_dir, "test_q.pt")
  a.out = a.out or os.path.join(a.value_dir, f"model_q_{a.tag}")
  return a


# --- statistics ---------------------------------------------------------------------------

def auroc(score: torch.Tensor, pos: torch.Tensor) -> float:
  """P(score of a random positive > that of a random negative), ties half."""
  s, y = score.double(), pos.bool()
  n1, n0 = int(y.sum()), int((~y).sum())
  if n1 == 0 or n0 == 0:
    return float("nan")
  order = torch.argsort(s)
  ranks = torch.empty_like(s)
  ranks[order] = torch.arange(1, s.numel() + 1, dtype=s.dtype)
  # average ranks over ties
  uniq, inv = torch.unique(s, return_inverse=True)
  if uniq.numel() < s.numel():
    sums = torch.zeros_like(uniq).index_add_(0, inv, ranks)
    cnt = torch.zeros_like(uniq).index_add_(0, inv, torch.ones_like(ranks))
    ranks = (sums / cnt)[inv]
  return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
  def rank(v):
    r = torch.empty_like(v, dtype=torch.float64)
    r[torch.argsort(v)] = torch.arange(v.numel(), dtype=torch.float64)
    return r
  rx, ry = rank(x.double()), rank(y.double())
  rx, ry = rx - rx.mean(), ry - ry.mean()
  return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def pair_accuracy(s_a, s_b, v_a, v_b, mask) -> tuple[float, int]:
  """How often sign(s_b - s_a) matches sign(v_b - v_a) over ``mask``."""
  m = mask & torch.isfinite(s_a) & torch.isfinite(s_b)
  n = int(m.sum())
  if n == 0:
    return float("nan"), 0
  ok = torch.sign(s_b[m] - s_a[m]) == torch.sign(v_b[m] - v_a[m])
  return float(ok.float().mean()), n


# --- the model-based Q ------------------------------------------------------------------

class ModelQ:
  """SHAC's one-step backup through its world model, on raw critic obs."""

  def __init__(self, model, policy_zip: str, device):
    self.model, self.wm = model, model.world_model
    if not bool(self.wm.fitted):
      raise SystemExit("[model-q] the checkpoint's world model was never fitted")
    pt = find_obs_stats(policy_zip, "tensornorm", "tensornormalize.pt")
    st = torch.load(pt, map_location=device, weights_only=True)
    if "critic_obs_mean" not in st:
      raise SystemExit(f"[model-q] {pt} has no critic normalizer (not a privileged-critic run)")
    self.c_mean, self.c_var = st["critic_obs_mean"].to(device), st["critic_obs_var"].to(device)
    self.gamma = float(model.gamma)
    self.z_clip = getattr(model, "wm_z_clip", None)
    self.low = torch.as_tensor(model.action_space.low, dtype=torch.float32, device=device)
    self.high = torch.as_tensor(model.action_space.high, dtype=torch.float32, device=device)

  def value(self, s_raw: torch.Tensor) -> torch.Tensor:
    x = torch.clamp((s_raw - self.c_mean) / torch.sqrt(self.c_var + 1e-8), -10.0, 10.0)
    return self.model.policy.predict_values(x).squeeze(-1)

  @torch.no_grad()
  def step(self, s_raw, a, ctx):
    """f's prediction: (raw next state, Q_f, p_terminate)."""
    wm = self.wm
    cn = wm.encode_context(ctx)
    st = wm.step(wm.encode(s_raw), a, cn, z_clip=self.z_clip)
    s1 = wm.decode(st.z_next)
    v1 = self.value(s1)
    nd = 1.0 - torch.sigmoid(st.done_logit)
    q = backups.lambda_return(backups.AVOID, st.g[None], v1[None], nd[None], self.gamma,
                              lam=1.0, g_term=st.g_term[None])[0]
    return s1, q, 1.0 - nd


# --- the run ------------------------------------------------------------------------------

def _spawn_term(mj) -> RecoverSpawn:
  for cfgs in getattr(mj.event_manager, "_mode_term_cfgs", {}).values():
    for c in cfgs:
      if isinstance(getattr(c, "func", None), RecoverSpawn):
        return c.func
  raise SystemExit("[model-q] the env has no RecoverSpawn reset term")


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  t0 = time.time()
  vd, tag = a.value_dir, a.tag
  stem = os.path.splitext(os.path.basename(a.states))[0]           # test_q
  raw = torch.load(a.states, map_location="cpu", weights_only=False)
  rows = load_bank(a.states, "cpu")
  R = int(rows["root_z"].numel())
  lab_s = torch.load(os.path.join(vd, f"{stem}_labels_{tag}.pt"), map_location="cpu",
                     weights_only=False)
  lab_p = torch.load(os.path.join(vd, f"{stem}_pi_labels_{tag}.pt"), map_location="cpu",
                     weights_only=False)
  v_stem = stem[:-2] + "_v" if stem.endswith("_q") else None
  lab_v = (torch.load(os.path.join(vd, f"{v_stem}_labels_{tag}.pt"), map_location="cpu",
                      weights_only=False) if v_stem else None)
  y_s = lab_s["v"].float().reshape(R, -1).nanmean(1)
  y_p = lab_p["v"].float().reshape(R, -1).nanmean(1)
  u_sn = raw["q_action"].float()                                    # SONIC + noise (played)
  u_pn = lab_p["q_action"].float()                                  # pi + the same noise (played)
  u_s = raw["sonic_action"].float()
  noise = u_sn - u_s
  y_v = None
  if lab_v is not None:
    y_v = lab_v["v"].float()
    y_v = (y_v.nanmean(1) if y_v.dim() == 2 else y_v)[raw["v_row"].long()]

  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  if not hasattr(model, "world_model"):
    raise SystemExit(f"[model-q] {a.policy} ({type(model).__name__}) has no world model")
  mq = ModelQ(model, a.policy, dev)

  overrides = dict(dataset=a.states, sampler="sequential", sequential_repeats=1,
                   episode_length_s=3.0, rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_tensor(a.task, a.num_envs, dev, cfg_overrides=overrides, random_push=push_kw,
                    critic_obs_key="critic")
  mj, n = env.mj, env.num_envs
  if env.critic_obs() is None or int(env.critic_obs().shape[1]) != int(mq.wm.state_dim):
    raise SystemExit(f"[model-q] the env's critic group does not match the world model's "
                     f"{mq.wm.state_dim}-D state")
  c_test = env.sim_context()
  if (0 if c_test is None else int(c_test.shape[1])) != int(mq.wm.context_dim):
    raise SystemExit(f"[model-q] sim context {None if c_test is None else c_test.shape[1]}-D, "
                     f"the world model wants {mq.wm.context_dim}")
  spawn = _spawn_term(mj)
  layout = env._obs_layout(env.critic_obs_key)          # {term: (start, size, cfg)}
  margin_fn = spec(a.task).margin_fn
  low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
  high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
  dyn = mq.wm.dynamic_mask

  def restore(w0: int, w1: int) -> torch.Tensor:
    spawn._cursor = int(w0)
    mj._zoo_spawn_cursor = int(w0)
    env.reset()
    r = mj._zoo_spawn_row.clone()
    return torch.where((r >= w0) & (r < w1), r, torch.full_like(r, -1))

  F = lambda: torch.full((R,), math.nan)  # noqa: E731
  out = {k: F() for k in ("q_model_sn", "q_model_pn", "q_box_sn", "q_box_pn", "q_model_s",
                          "q_model_pi", "v_critic", "g0", "p_term_s", "p_term_pi",
                          "f_err_s", "f_err_pi", "f_abs_s", "f_abs_pi", "pi_gap")}
  out["done_s"] = torch.zeros(R, dtype=torch.bool)
  out["done_pi"] = torch.zeros(R, dtype=torch.bool)
  for w0 in range(0, R, n):
    w1 = min(R, w0 + n)
    torch.manual_seed(a.seed * 1_000_003 + w0)
    row = restore(w0, w1)
    take = row >= 0
    rid = row[take].cpu()
    s0 = env.critic_obs().clone()
    ctx = env.sim_context()
    g0 = margin_fn(mj)[0].clone()
    obs = torch.nan_to_num(env.obs_groups()[env.obs_key].float(), nan=0.0, posinf=0.0,
                           neginf=0.0)
    with torch.no_grad():
      pi = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    rc = row.clamp_min(0).cpu()
    act = {k: v[rc].to(dev) for k, v in (("sn", u_sn), ("pn", u_pn), ("s", u_s))}
    act["pi"] = pi
    q = {}
    for k, u in act.items():
      s1, q[k], p = mq.step(s0, u, ctx)
      if k in ("s", "pi"):
        out[f"p_term_{k}"][rid] = p[take].cpu()
    for k in ("sn", "pn"):
      out[f"q_model_{k}"][rid] = q[k][take].cpu()
      _, qb, _ = mq.step(s0, act[k].clamp(mq.low, mq.high), ctx)
      out[f"q_box_{k}"][rid] = qb[take].cpu()
    out["q_model_s"][rid] = q["s"][take].cpu()
    out["q_model_pi"][rid] = q["pi"][take].cpu()
    with torch.no_grad():
      out["v_critic"][rid] = mq.value(s0)[take].cpu()
    out["g0"][rid] = g0[take].cpu()
    # pi now vs the labeller's pi (played - noise): the restore reproduces its state?
    out["pi_gap"][rid] = (pi[take].cpu() - (u_pn[rid] - noise[rid])).norm(dim=1)
    # f's one-step error, under u_SONIC and (restored again) under pi(s); each is
    # predicted from the critic obs the env actually stepped from
    for k in ("s", "pi"):
      start = s0
      if k == "pi":
        if not bool((restore(w0, w1) == row).all()):
          raise SystemExit("[model-q] the second restore spawned other rows")
        start = env.critic_obs().clone()
        diff = (start - s0)[take].abs().amax(0)
        if float(diff.max()) > 1e-3:
          names = [f"{t}[{d - st}] {float(diff[d]):.2g}" for t, (st, sz, _c) in layout.items()
                   for d in range(st, st + sz) if float(diff[d]) > 1e-3]
          print(f"[model-q] note: restoring the same rows again moves {len(names)} critic dims: "
                + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""), flush=True)
      _o, _g, dones, _t, _l = env.step_tensor(act[k])
      s_true = env.critic_obs().clone()
      s_pred, _, _ = mq.step(start, act[k], ctx)
      z0, zt, zp = mq.wm.encode(start), mq.wm.encode(s_true), mq.wm.encode(s_pred)
      d_true = (zt - z0)[:, dyn]
      e = (zp - zt)[:, dyn]
      rel = e.norm(dim=1) / d_true.norm(dim=1).clamp_min(1e-6)
      absr = e.pow(2).mean(1).sqrt()
      ok = take & ~dones
      out[f"f_err_{k}"][row[ok].cpu()] = rel[ok].cpu()
      out[f"f_abs_{k}"][row[ok].cpu()] = absr[ok].cpu()
      out[f"done_{k}"][row[take & dones].cpu()] = True
    print(f"[model-q] rows {w0}-{w1 - 1} done ({time.time() - t0:.0f}s)", flush=True)
  if torch.isnan(out["q_model_sn"]).any():
    raise SystemExit(f"[model-q] {int(torch.isnan(out['q_model_sn']).sum())} rows never restored")

  # --- the fitted nets on the same rows (CPU)
  fit_q, fit_v = {}, {}
  with torch.no_grad():
    for name, st in FITTED_Q.items():
      path = os.path.join(vd, st.format(tag=tag) + ".pt")
      if not os.path.exists(path):
        print(f"[model-q] skip {name}: no {path}")
        continue
      net = load_value_net(path)
      fit_q[name] = {k: net(features(rows, u)).float() for k, u in (("sn", u_sn), ("pn", u_pn))}
    for name, st in FITTED_V.items():
      path = os.path.join(vd, st.format(tag=tag) + ".pt")
      if os.path.exists(path):
        fit_v[name] = load_value_net(path)(features(rows)).float()

  # --- the numbers
  scorers = {"model (SHAC f + critic)": {"sn": out["q_model_sn"], "pn": out["q_model_pn"]},
             "model, action box-clamped": {"sn": out["q_box_sn"], "pn": out["q_box_pn"]}}
  scorers.update(fit_q)
  dist = (u_pn - u_sn).norm(dim=1)
  out_box = ((u_s.abs() > 3.0).any(1)).float().mean()
  decisive = (y_s >= 0) != (y_p >= 0)
  gapm = (y_p - y_s).abs() > a.pair_gap
  rep = {"tag": tag, "policy": os.path.abspath(a.policy), "states": os.path.abspath(a.states),
         "rows": R, "gamma": mq.gamma, "pair_gap": a.pair_gap,
         "safe_frac": {"sonic_centred": float((y_s >= 0).float().mean()),
                       "pi_centred": float((y_p >= 0).float().mean())},
         "action_gap_median": float(dist.median()),
         "sonic_action_outside_box": float(out_box),
         "pi_restore_gap_median": float(out["pi_gap"].median()),
         "decisive_pairs": int(decisive.sum()), "gap_pairs": int(gapm.sum()),
         "separation": {}, "pairs": {}, "one_step": {}, "value": {}}
  print(f"\n[model-q] {tag}: {R} held-out states; safe {rep['safe_frac']}; SONIC's action leaves the "
        f"+-3 box on {float(out_box):.2f}; ||pi-centred - SONIC-centred|| median "
        f"{float(dist.median()):.2f}; pi(s) re-derived within {float(out['pi_gap'].median()):.3f} "
        f"(median) of the labeller's")
  print(f"  {'scorer':28s} {'AUROC sonic':>11s} {'AUROC pi':>9s} {'rho sonic':>9s} {'rho pi':>7s} "
        f"{'pairs: decisive':>16s} {'pairs: |dv|>gap':>16s}")
  for name, sc in scorers.items():
    r = {"auroc_sonic": auroc(sc["sn"], y_s >= 0), "auroc_pi": auroc(sc["pn"], y_p >= 0),
         "rho_sonic": spearman(sc["sn"], y_s), "rho_pi": spearman(sc["pn"], y_p)}
    acc_d, n_d = pair_accuracy(sc["sn"], sc["pn"], y_s, y_p, decisive)
    acc_g, n_g = pair_accuracy(sc["sn"], sc["pn"], y_s, y_p, gapm)
    rep["separation"][name] = r
    rep["pairs"][name] = {"decisive": acc_d, "n_decisive": n_d, "gap": acc_g, "n_gap": n_g}
    print(f"  {name:28s} {r['auroc_sonic']:11.3f} {r['auroc_pi']:9.3f} {r['rho_sonic']:9.3f} "
          f"{r['rho_pi']:7.3f} {acc_d:10.3f} ({n_d:4d}) {acc_g:10.3f} ({n_g:4d})")
  pmean = float((y_p > y_s).float().mean())
  print(f"  (the label says the pi-centred action is safer on {pmean:.2f} of all pairs; "
        f"'always pick pi' scores {float((y_p[decisive] > y_s[decisive]).float().mean()):.3f} "
        f"on the decisive ones)")
  rep["pairs"]["always pi"] = {"decisive": float((y_p[decisive] > y_s[decisive]).float().mean())}
  for k, lbl in (("s", "u_SONIC"), ("pi", "pi(s)")):
    rel, ab = out[f"f_err_{k}"], out[f"f_abs_{k}"]
    m = torch.isfinite(rel)
    rep["one_step"][lbl] = {"rel_err_median": float(rel[m].median()),
                            "rel_err_p90": float(rel[m].quantile(0.9)),
                            "abs_rms_median": float(ab[m].median()),
                            "terminated": float(out[f"done_{k}"].float().mean()),
                            "p_term_mean": float(out[f"p_term_{k}"].mean())}
    o = rep["one_step"][lbl]
    print(f"  f one step under {lbl:8s}: error of the predicted change / real change median "
          f"{o['rel_err_median']:.3f} (p90 {o['rel_err_p90']:.3f}); per-dim RMS {o['abs_rms_median']:.3f} "
          f"(normalized); terminated {o['terminated']:.3f} vs predicted {o['p_term_mean']:.3f}")
  if y_v is not None:
    vs = {"SHAC critic": out["v_critic"]}
    vs.update(fit_v)
    for name, v in vs.items():
      rep["value"][name] = {"auroc": auroc(v, y_v >= 0), "rho": spearman(v, y_v)}
      print(f"  V {name:16s}: AUROC {rep['value'][name]['auroc']:.3f}, rho "
            f"{rep['value'][name]['rho']:.3f} against the state's exact V label")
  rep["wall_s"] = time.time() - t0
  rep["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
  torch.save({"meta": rep, "rows": out, "fit_q": {k: {c: t.cpu() for c, t in v.items()}
                                                  for k, v in fit_q.items()},
              "labels": {"sonic_centred": y_s, "pi_centred": y_p, "v": y_v}}, a.out + ".pt")
  with open(a.out + ".json", "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[model-q] wrote {a.out}.json / .pt ({rep['wall_s']:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
