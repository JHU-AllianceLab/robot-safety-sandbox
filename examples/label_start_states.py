"""Label every start state of a bank with K rollouts of a recovery policy.

    python examples/label_start_states.py \
        --policy runs/g1_recover_push_fast_privcrit/g1_recover_push_fast/final_model.zip \
        --bank data/g1_recover_states_lafan1_train.pt \
        --out data/g1_recover_states_lafan1_train_labeled.pt --k 8

A row's LABEL is how often the policy survives an episode started from it
under the push distribution the training env uses: ``p_surv`` = survived / K.
It is what the frontier samplers weight with (``p (1 - p)``, spawn.py
``sampler="weighted"``) and what the E2 question is asked against: does the
policy's own value V(s) at the start rank the rows like the rollouts do? If
it does (AUC >= 0.9), the value can stand in for rollout labels and the
value-weighted arm (S5) needs no labeling pass at all.

THE ROLLOUTS. One env, ``--task`` (default ``g1_recover_lafan``), spawning
with the SEQUENTIAL sampler (``envs/g1_recover/spawn.py``): a cursor walks
every ``start_ok`` row in bank order, each ``--k`` times in a row, so a pass
over the bank issues each row exactly K episodes and no env idles. The push
is the training push: one horizontal random push per episode (``max_pushes``
1), its onset 0.1-0.4 s after the spawn, 0.2 s long, at ``--force-max x s``
newtons with ``s ~ U(--force-min-frac, 1)`` redrawn for every episode (the
bridge's per-env ``force_scale``); ``--no-push`` turns it off. A
``KIND_HANDOVER`` row brings its own pending push and gets no random one (the
spawn's ``consume`` flag; nothing to do here). The policy acts
deterministically. An episode ends on a FALL (a termination) or survives:
back at rest (the ``recovered`` truncation, ``rest_terminate``) or the
``--episode-s`` cap.

ATTRIBUTION. mjlab resets an env INSIDE the step its episode ends in, so the
spawn's ``_zoo_spawn_row`` already names the NEXT episode when the step
returns. The labeler keeps its own copy per env, taken when each episode
starts, and books an ending episode's outcome on the OLD copy. The outcomes
of a row are exactly its K FIRST-PASS episodes: the spawn hands out entries
in env-id order, so the cursor before a step says which pass each new episode
belongs to, and episodes of a later pass (the cursor cycles) are ignored.
Keeping instead "the first K to end" would bias a small bank toward falls: once
the pass wraps while a row's earlier episodes still run, its falls end first
(a survivor needs the rest dwell or the cap). The run stops once every target
row has K. The env is NOT
reset again after it is built: its construction reset already issued the
first wave of sequential entries (and handed out their pending pushes), and a
second reset would discard that wave until the cursor came round again.

THE VALUE. ``value0`` of an episode is V(its first observation), on the input
the critic was trained on (``labels.PolicyValue``: the privileged critic group
for an asymmetric critic). A row's ``value0`` is the mean over its K episodes
(the start differs only by yaw, planar offset and the per-episode
randomization), and ``p_value = sigmoid(a value0 + b)`` with (a, b) fit by
``labels.fit_logistic`` on the labeled rows (targets ``p_surv``, weighted by
``n_label``), so value-weighted sampling works on a calibrated probability.

OUTPUT. ``--out``: the input bank, every field ``load_bank`` returns (all
``OPTIONAL_FIELDS``, plus ``part`` for a multi-file input), with ``p_surv``,
``n_label``, ``value0``, ``ttf_mean`` (mean time to fall of the failed
episodes, s) and ``p_value`` written on the labeled rows (others keep their
input values), and ``meta["label"]`` (settings + report). The report is also
printed and written to ``<out stem>.report.json``: overall survival, the
episode AUC of V (P(V of a surviving episode > V of a falling one), ties
half), the row Spearman of value0 vs p_surv, survival and AUC per category,
the shares of rows with p_surv in [0, 0.1], (0.1, 0.9) and [0.9, 1], survival
by push force, and the E2 decision line. Also the value read as a certificate
(``cert_frac``, ``cert_precision`` = P(survive | V(s0) >= 0), ``cert_recall`` =
P(V(s0) >= 0 | survive)) and the policy's action statistics over every
rollout step (``action_rate`` ||a_t - a_{t-1}||^2, ``action_mag`` ||a_t||^2,
``action_sat_frac``), for comparing differently regularized learners.
"""

from __future__ import annotations

import argparse
import json
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

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.labels import (  # noqa: E402
  PolicyValue, auc, fit_logistic, row_names, spearman)
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  KIND_HANDOVER, load_bank, save_bank)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402

#: Names for rows without a category of their own (walker / synthetic rows),
#: by ``kind``.
KIND_NAMES = ("clean", "pushed", "synthetic", "mocap", "handover")
#: The E2 threshold: an episode AUC of V at or above it makes the
#: value-weighted sampling arm (S5) viable without rollout labels.
AUC_VIABLE = 0.9


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True,
                 help="the labeling policy's safety_sb3 checkpoint zip")
  p.add_argument("--bank", nargs="+",
                 default=[os.path.join(_ZOO, "data", "g1_recover_states_lafan1_train.pt")],
                 help="the bank(s) to label (several are concatenated in order)")
  p.add_argument("--out", default=None,
                 help="default: <first bank stem>_labeled.pt")
  p.add_argument("--k", type=int, default=8, help="rollouts per row")
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--force-max", type=float, default=200.0,
                 help="push magnitude at scale 1, N")
  p.add_argument("--force-min-frac", type=float, default=0.2,
                 help="per-episode scale drawn in [this, 1]")
  p.add_argument("--no-push", action="store_true", help="no random push")
  p.add_argument("--episode-s", type=float, default=3.0, help="episode cap, s")
  p.add_argument("--max-rows", type=int, default=None,
                 help="label only the first M start_ok rows (smoke tests)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args(argv)
  if args.k < 1:
    p.error("--k must be >= 1")
  if not 0.0 <= args.force_min_frac <= 1.0:
    p.error("--force-min-frac must be in [0, 1]")
  if args.out is None:
    args.out = os.path.splitext(args.bank[0])[0] + "_labeled.pt"
  return args


def report_path(out: str) -> str:
  return os.path.splitext(out)[0] + ".report.json"


def _rank_in_group(r: torch.Tensor) -> torch.Tensor:
  """For each entry of ``r``, how many EARLIER entries hold the same value."""
  n = r.numel()
  if n == 0:
    return r
  order = torch.argsort(r, stable=True)
  rs = r[order]
  ar = torch.arange(n, device=r.device)
  new = torch.ones(n, dtype=torch.bool, device=r.device)
  new[1:] = rs[1:] != rs[:-1]
  first = torch.cummax(torch.where(new, ar, torch.zeros_like(ar)), 0).values
  out = torch.empty_like(ar)
  out[order] = ar - first
  return out


def _construction_obs(env) -> torch.Tensor:
  """The raw actor obs of the episodes the env's construction reset started
  (sanitized as ``step_tensor`` does)."""
  obs = env.obs_groups()[env.obs_key].float()
  return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def rollouts(env, model, norm, value, target: torch.Tensor, kinds: torch.Tensor,
             args) -> dict[str, torch.Tensor]:
  """Run until every ``target`` row has ``args.k`` ended episodes; return the
  per-episode records (row, value0, fell, ttf, force)."""
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  if not hasattr(mj, "_zoo_spawn_row"):
    raise SystemExit(f"task {args.task!r}: its spawn publishes no _zoo_spawn_row "
                     f"(a g1_recover task is needed)")
  push = getattr(env, "_push", None)
  if push is None and not args.no_push:
    print(f"[label] WARNING: task {args.task!r} has no push process; labeling "
          f"without pushes")
  target = target.to(dev)
  kinds = kinds.to(dev)
  R = int(target.numel())
  k = int(args.k)
  counts = torch.zeros(R, dtype=torch.long, device=dev)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  fmax = 0.0 if args.no_push else float(args.force_max)
  env.force_max = fmax
  env.force_scale = torch.ones(n, device=dev)

  def draw(mask):
    m = int(mask.sum())
    if m:
      env.force_scale[mask] = (args.force_min_frac + (1.0 - args.force_min_frac)
                               * torch.rand(m, device=dev))

  draw(torch.ones(n, dtype=torch.bool, device=dev))
  obs = _construction_obs(env)
  cur_row = mj._zoo_spawn_row.clone()
  # which sequential entry each running episode took (the last reset issued
  # the n entries before the cursor, in env-id order): only first-pass ones count
  seq_len = int(mj._zoo_spawn_seq_len)
  ar = torch.arange(n, device=dev)
  first = (int(mj._zoo_spawn_cursor) - n + ar) < seq_len
  crit = env.critic_obs() if value.asymmetric else None
  v0 = value(obs, crit)
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  # What the deterministic policy's actions look like (a smoothness read-out to
  # compare policies trained with different regularizers): the action rate
  # ||a_t - a_{t-1}||^2 (the first step of an episode has no own predecessor
  # and is left out), the magnitude ||a_t||^2 (summed over action dims), and
  # the fraction of action dims at the box. Every step of every env counts.
  prev_act = torch.zeros(n, int(low.numel()), device=dev)
  a_stats = torch.zeros(6, device=dev)  # rate sum, n, mag sum, n, sat sum, n
  ep: dict[str, list] = {"row": [], "value0": [], "fell": [], "ttf": [], "force": []}
  n_target = int(target.sum())
  remaining = n_target
  stall_limit = 3 * int(mj.max_episode_length) + 10
  since_accept, it, t0, episodes = 0, 0, time.time(), 0
  while remaining > 0:
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    has_prev = (elapsed > 0).float()
    a_stats[0] += ((act - prev_act).pow(2).sum(-1) * has_prev).sum()
    a_stats[1] += has_prev.sum()
    a_stats[2] += act.pow(2).sum(-1).sum()
    a_stats[3] += n
    a_stats[4] += ((act >= high - 1e-4) | (act <= low + 1e-4)).float().mean(-1).sum()
    a_stats[5] += n
    prev_act = act
    cursor = int(mj._zoo_spawn_cursor)
    obs, _g, dones, timeouts, _l = env.step_tensor(act)
    reset_policy_memory(model.policy, dones)       # a recurrent policy restarts
    elapsed += 1
    it += 1
    since_accept += 1
    if bool(dones.any()):
      d = dones.nonzero(as_tuple=False).squeeze(-1)
      rows = cur_row[d]
      fell = ~timeouts[d]
      valid = (rows >= 0) & first[d]
      safe = rows.clamp_min(0)
      valid &= target[safe]
      # first k ENDED episodes per row; two envs ending on the same row in
      # this step are ranked so the cap is exact
      rank = _rank_in_group(torch.where(valid, safe, torch.full_like(safe, -1)))
      acc = valid & (counts[safe] + rank < k)
      if bool(acc.any()):
        ra = rows[acc]
        counts.index_add_(0, ra, torch.ones_like(ra))
        fa = fell[acc]
        ep["row"].append(ra)
        ep["value0"].append(v0[d][acc])
        ep["fell"].append(fa)
        ep["ttf"].append(torch.where(fa, elapsed[d][acc].float() * dt,
                                     torch.full_like(fa, float("nan"), dtype=torch.float)))
        force = env.force_scale[d][acc] * fmax
        ep["force"].append(torch.where(kinds[ra] == KIND_HANDOVER,
                                       torch.full_like(force, float("nan")), force))
        episodes += int(acc.sum())
        since_accept = 0
        remaining = int((target & (counts < k)).sum())
      # the episodes that just started (mjlab already reset these envs); they
      # took entries cursor, cursor + 1, ... in env-id order (d is ascending)
      cur_row[d] = mj._zoo_spawn_row[d]
      first[d] = (cursor + ar[:d.numel()]) < seq_len
      crit = env.critic_obs() if value.asymmetric else None
      v0[d] = value(obs[d], crit[d] if crit is not None else None)
      elapsed[d] = 0
      draw(dones)
    if since_accept > stall_limit:
      raise SystemExit(
        f"[label] no labeled episode in {since_accept} steps with {remaining} "
        f"rows still short of {k}: is the spawn sampler sequential and the bank "
        f"the one the env spawns from?")
    if it % 250 == 0:
      el = max(time.time() - t0, 1e-6)
      print(f"[label] step {it}  rows done {n_target - remaining}/{n_target}  "
            f"episodes {episodes}/{n_target * k}  ({episodes / el:.0f} ep/s, "
            f"{it * n / el:.0f} env-steps/s)", flush=True)
  el = max(time.time() - t0, 1e-6)
  print(f"[label] done: {episodes} episodes in {it} steps, {el:.0f}s "
        f"({episodes / el:.0f} ep/s, {it * n / el:.0f} env-steps/s)")
  out = {key: torch.cat(v) for key, v in ep.items()}
  out["action_stats"] = a_stats.cpu()
  return out


def write_labels(bank: dict, ep: dict, target: torch.Tensor) -> tuple[float, float]:
  """Fold the episodes into per-row columns of ``bank`` (in place, CPU) and
  fill ``p_value``; returns the logistic fit (a, b)."""
  R = int(bank["kind"].numel())
  row = ep["row"].cpu()
  fell = ep["fell"].cpu().float()
  one = torch.ones_like(fell)
  cnt = torch.zeros(R).index_add_(0, row, one)
  surv = torch.zeros(R).index_add_(0, row, 1.0 - fell)
  vsum = torch.zeros(R).index_add_(0, row, ep["value0"].cpu().float())
  nfell = torch.zeros(R).index_add_(0, row, fell)
  ttf = torch.nan_to_num(ep["ttf"].cpu().float(), nan=0.0)
  tsum = torch.zeros(R).index_add_(0, row, ttf * fell)
  lab = target.cpu() & (cnt > 0)
  nan = torch.full((R,), float("nan"))
  bank["n_label"] = torch.where(lab, cnt.long(), bank["n_label"].cpu())
  bank["p_surv"] = torch.where(lab, surv / cnt.clamp_min(1.0), bank["p_surv"].cpu())
  bank["value0"] = torch.where(lab, vsum / cnt.clamp_min(1.0), bank["value0"].cpu())
  ttf_mean = torch.where(nfell > 0, tsum / nfell.clamp_min(1.0), nan)
  bank["ttf_mean"] = torch.where(lab, ttf_mean, bank["ttf_mean"].cpu())
  a, b = fit_logistic(bank["value0"][lab], bank["p_surv"][lab], weight=cnt[lab])
  if a == a:
    p_value = torch.sigmoid(a * bank["value0"] + b)
    bank["p_value"] = torch.where(lab, p_value, bank["p_value"].cpu())
  return a, b


def report(bank: dict, ep: dict, target: torch.Tensor, ab: tuple, args) -> dict:
  row = ep["row"].cpu()
  fell = ep["fell"].cpu()
  v0 = ep["value0"].cpu()
  lab = target.cpu() & (bank["n_label"] > 0)
  p = bank["p_surv"][lab]
  out: dict = {"rows_labeled": int(lab.sum()), "episodes": int(row.numel()),
               "k": int(args.k), "logistic_a": ab[0], "logistic_b": ab[1]}
  out["survival"] = 1.0 - float(fell.float().mean()) if row.numel() else float("nan")
  out["auc_value_episodes"] = auc(v0, ~fell)
  out["spearman_value_psurv_rows"] = spearman(bank["value0"][lab], p)
  out["frac_p_le_0.1"] = float((p <= 0.1).float().mean()) if p.numel() else float("nan")
  out["frac_p_mid"] = float(((p > 0.1) & (p < 0.9)).float().mean()) if p.numel() else float("nan")
  out["frac_p_ge_0.9"] = float((p >= 0.9).float().mean()) if p.numel() else float("nan")
  # The value as a CERTIFICATE ("safe iff V >= 0"), per episode, at its start:
  # how much it certifies, and how often a certified start then fell.
  cert, surv = v0 >= 0.0, ~fell
  n_cert, n_surv = int(cert.sum()), int(surv.sum())
  out["cert_frac"] = float(cert.float().mean()) if row.numel() else float("nan")
  out["cert_precision"] = (float((cert & surv).sum()) / n_cert if n_cert
                           else float("nan"))
  out["cert_recall"] = (float((cert & surv).sum()) / n_surv if n_surv
                        else float("nan"))
  st = ep.get("action_stats")
  if st is not None:
    out["action_rate"] = float(st[0] / st[1].clamp_min(1.0))
    out["action_mag"] = float(st[2] / st[3].clamp_min(1.0))
    out["action_sat_frac"] = float(st[4] / st[5].clamp_min(1.0))
  print(f"[label] {out['rows_labeled']} rows x {args.k} = {out['episodes']} episodes; "
        f"survival {out['survival']:.3f}")
  print(f"[label] value vs outcome: episode AUC {out['auc_value_episodes']:.3f} "
        f"(P(V_survived > V_fell)), row Spearman(value0, p_surv) "
        f"{out['spearman_value_psurv_rows']:.3f}; p_value = sigmoid({ab[0]:.3f} V "
        f"+ {ab[1]:.3f})")
  print(f"[label] rows by p_surv: [0, 0.1] {out['frac_p_le_0.1']:.3f}, (0.1, 0.9) "
        f"{out['frac_p_mid']:.3f}, [0.9, 1] {out['frac_p_ge_0.9']:.3f}")
  print(f"[label] certificate V(s0) >= 0: certifies {out['cert_frac']:.3f} of starts, "
        f"precision P(survive | V>=0) {out['cert_precision']:.3f}, recall "
        f"P(V>=0 | survive) {out['cert_recall']:.3f}")
  if "action_rate" in out:
    print(f"[label] actions: rate {out['action_rate']:.4f}, magnitude "
          f"{out['action_mag']:.3f} (sum over dims), at the box "
          f"{out['action_sat_frac']:.4f} of dims")
  # per category (the row's own file resolves the name)
  names = row_names(bank, fallback=KIND_NAMES)
  uniq = sorted(set(names))
  code = {c: i for i, c in enumerate(uniq)}
  code_row = torch.tensor([code[x] for x in names], dtype=torch.long)
  code_ep = code_row[row]
  per = {}
  print("[label] per category:")
  for c in sorted(set(code_row[lab].tolist())):
    em = code_ep == c
    rm = lab & (code_row == c)
    c = uniq[c]
    s = 1.0 - float(fell[em].float().mean()) if bool(em.any()) else float("nan")
    a_c = auc(v0[em], ~fell[em])
    per[c] = {"rows": int(rm.sum()), "episodes": int(em.sum()), "survival": s,
              "auc_value": a_c, "p_mean": float(bank["p_surv"][rm].mean())}
    print(f"    {c:15s} {int(rm.sum()):7d} rows  survival {s:.3f}  AUC {a_c:.3f}")
  out["per_category"] = per
  # by push force (random pushes only; handover rows carry their own)
  force = ep["force"].cpu()
  fm = torch.isfinite(force)
  if not args.no_push and bool(fm.any()):
    lo, hi = args.force_min_frac * args.force_max, args.force_max
    edges = torch.linspace(lo, hi, 5)
    byf = {}
    print("[label] survival by push force:")
    for a_, b_ in zip(edges[:-1].tolist(), edges[1:].tolist()):
      m = fm & (force >= a_) & (force <= b_)
      if bool(m.any()):
        s = 1.0 - float(fell[m].float().mean())
        byf[f"{a_:.0f}_{b_:.0f}"] = s
        print(f"    {a_:5.0f}-{b_:5.0f} N: {int(m.sum()):7d} episodes, survival {s:.3f}")
    out["survival_by_force"] = byf
  ttf = ep["ttf"].cpu()[fell]
  if ttf.numel():
    q = torch.quantile(ttf, torch.tensor([0.1, 0.5, 0.9])).tolist()
    out["time_to_fall_q10_q50_q90"] = q
    print(f"[label] time to fall: q10 {q[0]:.2f} s, q50 {q[1]:.2f} s, q90 {q[2]:.2f} s")
  a_ep = out["auc_value_episodes"]
  if a_ep != a_ep:
    out["e2_decision"] = ("AUC undefined (every labeled episode survived, or every "
                          "one fell): no E2 decision from this pass")
  elif a_ep >= AUC_VIABLE:
    out["e2_decision"] = (f"AUC {a_ep:.3f} >= {AUC_VIABLE}: the value ranks starts like "
                          f"the rollouts -> the value-weighted arm S5 is viable")
  else:
    out["e2_decision"] = (f"AUC {a_ep:.3f} < {AUC_VIABLE}: the value does not rank "
                          f"starts well enough -> weight with rollout labels (S3/S4)")
  print(f"[label] E2: {out['e2_decision']}")
  return out


def main(argv=None):
  args = _parse(argv)
  torch.manual_seed(args.seed)
  dev = args.device
  paths = list(args.bank)
  bank = load_bank(paths[0] if len(paths) == 1 else paths, "cpu")
  R = int(bank["kind"].numel())
  target = bank["start_ok"].clone()
  if args.max_rows is not None:
    idx = target.nonzero(as_tuple=False).squeeze(-1)[:int(args.max_rows)]
    target = torch.zeros(R, dtype=torch.bool)
    target[idx] = True
  n_target = int(target.sum())
  if n_target == 0:
    raise SystemExit("no start_ok row to label")

  model, norm = load_twin(args.policy, dev)
  model.policy.set_training_mode(False)
  value = PolicyValue(model, args.policy, dev)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0 if args.no_push else 1)
  overrides = dict(dataset=paths[0] if len(paths) == 1 else paths,
                   sampler="sequential", sequential_repeats=int(args.k),
                   episode_length_s=float(args.episode_s), rest_terminate=True,
                   rest_dwell_s=0.4, rest_arm_s=0.8)
  env = make_tensor(args.task, args.num_envs, dev, cfg_overrides=overrides,
                    random_push=push_kw, critic_obs_key=value.critic_obs_key)
  seq_len = getattr(env.mj, "_zoo_spawn_seq_len", None)
  print(f"[label] {args.policy} ({'asymmetric' if value.asymmetric else 'symmetric'} "
        f"critic) on {args.task} x {args.num_envs}: {n_target} rows x {args.k}, "
        f"push {'off' if args.no_push else f'{args.force_min_frac * args.force_max:.0f}-{args.force_max:.0f} N'}, "
        f"episode cap {args.episode_s} s; sequential pass {seq_len} entries")
  ep = rollouts(env, model, norm, value, target, bank["kind"], args)
  ab = write_labels(bank, ep, target)
  stats = report(bank, ep, target, ab, args)
  label_meta = {
    "source": "label_start_states.py", "policy": os.path.abspath(args.policy),
    "task": args.task, "k": args.k, "force_max": args.force_max,
    "force_min_frac": args.force_min_frac, "no_push": bool(args.no_push),
    "episode_s": args.episode_s, "max_rows": args.max_rows, "seed": args.seed,
    "banks": [os.path.abspath(p) for p in paths],
    "critic": "asymmetric" if value.asymmetric else "symmetric",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "report": stats,
  }
  bank["meta"] = dict(bank.get("meta", {}) or {}, label=label_meta)
  save_bank(bank, args.out)
  with open(report_path(args.out), "w") as f:
    json.dump(stats, f, indent=2)
  print(f"[label] wrote {args.out} and {report_path(args.out)}")
  env.close()


if __name__ == "__main__":
  main()
