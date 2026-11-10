"""Cross-seed tables of the Q-CBF evaluation: SHAC vs PPO, exact vs td_g099.

    python examples/summarize_sonic_qcbf.py      # runs/shac_vs_ppo_sonic/qcbf_eval -> .../summary.{md,json}
    python examples/summarize_sonic_qcbf.py --root runs/shac_vs_ppo_sonic/qcbf_eval \
        --group SHAC shac_s0 shac_s1 shac_s2 --group PPO ppo_s0 ppo_s1 ppo_s2

Each run directory under ``--root`` holds one fallback policy's
``eval_sonic_qcbf.py`` outputs, ``push.json`` and ``badref.json``, and is
named by its tag. A GROUP is one learner and its runs are its seeds. By
default a run belongs to the group of its tag's prefix before ``_s<seed>``
(shac -> SHAC, ppo -> PPO). Per condition and group the tables give:

* SR: the mean +- std over the seeds, and pooled over them (k / n with a
  Wilson 95 % CI, see POOLING);
* against SONIC on the same episodes: rescued / broken summed over the seeds,
  a p-value (POOLING) and the mean SR difference;
* the filter: the intervention rate before / after each episode's split (the
  push, or SONIC's first sight of a corrupted frame), false alarms (episodes
  with an intervention during normal tracking), the mean deviation
  ||u - u_SONIC|| per engaged step, the best-effort share of the engaged
  steps, how often the trust region bound and how often pi's own action
  failed the constraint (means over the seeds);
* the learners against EACH OTHER, per condition: the groups ran the same
  episodes, so SHAC vs PPO is a paired comparison too (:func:`contrast`: the
  SR difference of the seed means and the sign test over episodes). Under
  ``sonic`` both groups ran the same unfiltered controller, so its row is the
  noise floor.

POOLING. The seeds of a group are NOT independent samples of episodes: run
with one ``--seed`` (the job files do), every policy runs the SAME episode
list, so an easy episode counts once per seed with nearly the same outcome,
and SONIC's runs are the same episodes over again. Summing k and n and
testing the summed discordant pairs would make the CIs ~sqrt(seeds) too
narrow and the p-values too small. So, per group and condition:

  clustered     the seeds share their episodes and each run's push.pt /
                badref.pt holds the per-episode outcomes: SR's CI is the
                Wilson interval at the episode-clustered effective sample
                size p (1 - p) / Var(mean over episodes of the seed-mean
                outcome), between E (seeds identical) and seeds x E
                (independent); the p-value is the exact two-sided SIGN test
                over episodes of the seed-mean (filter - SONIC) outcome
                (rescued / broken by episode), which is McNemar for one seed;
  conservative  shared episodes but no per-episode outcomes (a .pt missing,
                badref.partial.json): the CI at n = one seed's episodes, and
                the largest per-seed McNemar p (all seeds must agree);
  independent   the runs used different seeds (meta ``seed``) or their
                episode lists differ: summed k / n and the McNemar p of the
                summed pairs, as for independent samples.

Each pooled entry says which (``method``).

Tables (``<out>.md``, the same numbers in ``<out>.json``): push SR, the
paired group contrast, push intervention, exact vs td_g099 per kappa / solver
and for the switch, push SR by |dv| bin, bad-reference SR per scenario (one
table per group) with the pooled and per-family rows, its rescued / broken,
its intervention rates, and its paired group contrast per scenario. SONIC's
own SR is the ``sonic`` row. At one
``--seed`` every policy runs the same episodes, so its runs differ by
simulator noise only.

A run without a file (still going) is skipped with a notice. Without
``badref.json``, ``badref.partial.json`` (the scenarios done so far) is used
and flagged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import torch

_EX = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_EX)
sys.path.insert(0, _EX)

import qcbf_eval_lib as ql  # noqa: E402

DEFAULT_ROOT = os.path.join(_ZOO, "runs", "shac_vs_ppo_sonic", "qcbf_eval")
#: Display names of the tag prefixes.
GROUP_NAMES = {"shac": "SHAC", "ppo": "PPO"}
#: Intervention read-outs averaged over the seeds: key -> label.
INTERVENTION_KEYS = {
  "rate_pre": "int. rate pre", "rate_post": "int. rate post",
  "false_alarm_episodes": "false-alarm eps", "dev_per_engaged_post": "dev/engaged post",
  "best_effort_per_engaged_post": "best effort/engaged post",
  "trust_region_per_engaged_post": "trust region/engaged post",
  "fallback_infeasible_per_engaged_post": "pi infeasible/engaged post",
}
_EXTRA_KEYS = ("dev_per_step_post", "best_effort_rate_post", "gd_iters_per_engaged_post",
               "intervened_post_episodes", "rate_post_sonic_survived", "rate_post_sonic_fell",
               "jerr_pre", "jerr_post", "jerr_bad_post", "q_safe_mean_engaged_post",
               "seg_gap_mean_post")
#: Episode flags a pooled SR can exclude.
_FLAGS = ("ref_out_k", "bad_out_k")


# --- loading ----------------------------------------------------------------------------

def _read(path: str):
  with open(path) as f:
    return json.load(f)


def discover(root: str) -> dict:
  """``{tag: run directory}`` of the directories under ``root`` that hold an
  evaluation file."""
  out = {}
  if not os.path.isdir(root):
    return out
  for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f))
                                for f in ("push.json", "badref.json", "badref.partial.json")):
      out[name] = d
  return out


def default_group(tag: str) -> str:
  m = re.match(r"^(.+?)_s\d+$", tag)
  prefix = m.group(1) if m else tag
  return GROUP_NAMES.get(prefix, prefix.upper())


def load_run(d: str) -> dict:
  """A run's push / badref summaries (None when absent) and its per-episode
  ``outcomes`` (:func:`load_outcomes`)."""
  run = {"push": None, "badref": None, "badref_partial": False}
  p = os.path.join(d, "push.json")
  if os.path.exists(p):
    run["push"] = _read(p)
  b, bp = os.path.join(d, "badref.json"), os.path.join(d, "badref.partial.json")
  if os.path.exists(b):
    run["badref"] = _read(b)
  elif os.path.exists(bp):
    run["badref"], run["badref_partial"] = _read(bp), True
  run["outcomes"] = load_outcomes(d)
  return run


def episode_view(eps: dict, conditions: dict) -> dict:
  """What the pooled statistics need of one episode list: the episodes'
  identity (``key``: library row and onset), ``dv``, the K flags, and every
  condition's per-episode ``survived``."""
  out = {"key": torch.stack([eps["row"].long(), eps["on"].long()], 1),
         "dv": eps["dv"].float(),
         "survived": {c: r["survived"].bool() for c, r in conditions.items()}}
  out.update({f: eps[f].bool() for f in _FLAGS if f in eps})
  return out


def load_outcomes(d: str) -> dict:
  """``{"push": view or None, "badref": {scenario: view}}`` from the run's
  push.pt / badref.pt (:func:`episode_view`); missing or unreadable files
  leave their entries empty, and the pooling falls back (module docstring)."""
  out = {"push": None, "badref": {}}
  for suite in ("push", "badref"):
    path = os.path.join(d, f"{suite}.pt")
    if not os.path.exists(path):
      continue
    try:
      blob = torch.load(path, map_location="cpu", weights_only=False)
      if suite == "push":
        out["push"] = episode_view(blob["episodes"], blob["conditions"])
      else:
        out["badref"] = {s: episode_view(v["episodes"], v["conditions"])
                         for s, v in blob["scenarios"].items()}
    except Exception as e:  # noqa: BLE001 - a broken file must not stop the tables
      print(f"[summary] {path}: unreadable ({type(e).__name__}: {e}); pooling without it")
  return out


def concat_views(views: list):
  """Views of several episode lists end to end (None if any is missing)."""
  if not views or any(v is None for v in views):
    return None
  conds = set(views[0]["survived"]).intersection(*[set(v["survived"]) for v in views[1:]])
  out = {"key": torch.cat([v["key"] for v in views]), "dv": torch.cat([v["dv"] for v in views]),
         "survived": {c: torch.cat([v["survived"][c] for v in views]) for c in conds}}
  out.update({f: torch.cat([v[f] for v in views]) for f in _FLAGS if all(f in v for v in views)})
  return out


# --- aggregation ------------------------------------------------------------------------

def _num(x) -> float:
  return float(x) if isinstance(x, (int, float)) and x is not None else math.nan


def mean_std(values) -> tuple[float, float, int]:
  """Mean and sample std of the finite values (std NaN below two)."""
  v = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
  if not v:
    return (math.nan, math.nan, 0)
  m = sum(v) / len(v)
  s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else math.nan
  return (m, s, len(v))


def wilson_p(p: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
  """:func:`qcbf_eval_lib.wilson_ci` at a proportion ``p`` and a (possibly
  fractional, effective) sample size ``n``."""
  if not (n > 0 and math.isfinite(p)):
    return (math.nan, math.nan)
  z2 = z * z
  den = 1.0 + z2 / n
  centre = (p + z2 / (2 * n)) / den
  half = z * math.sqrt(max(p * (1.0 - p), 0.0) / n + z2 / (4 * n * n)) / den
  return (max(0.0, centre - half), min(1.0, centre + half))


def pooled(parts, n_eff=None, method: str = "independent") -> dict:
  """``{n, k, sr, ci}`` of survival dicts summed; the CI at ``n_eff`` (the
  summed n by default, i.e. independent parts)."""
  k = sum(int(p["k"]) for p in parts)
  n = sum(int(p["n"]) for p in parts)
  sr = k / n if n else math.nan
  ci = list(ql.wilson_ci(k, n)) if n_eff is None else list(wilson_p(sr, n_eff))
  return {"n": n, "k": k, "sr": sr, "ci": ci, "method": method}


def clustered(Y: torch.Tensor, mask=None) -> dict:
  """``Y`` ``[S, E]`` bool: S runs of the SAME E episodes. SR = the mean, its
  Wilson CI at the effective sample size p (1 - p) / Var(SR), with Var(SR)
  the variance over episodes of the per-episode mean over runs, / E. That
  is E when the runs agree on every episode and S E when they are
  independent (clipped to that range)."""
  if mask is not None:
    Y = Y[:, mask.bool()]
  S, E = int(Y.shape[0]), int(Y.shape[1])
  n, k = S * E, int(Y.sum())
  if E == 0:
    return {"n": 0, "k": 0, "sr": math.nan, "ci": [math.nan, math.nan], "episodes": 0,
            "n_eff": math.nan, "method": "clustered"}
  y = Y.double().mean(0)
  p = float(y.mean())
  var = float(y.var(unbiased=False)) / E
  n_eff = float(n) if var <= 0 else min(float(n), max(float(E), p * (1.0 - p) / var))
  return {"n": n, "k": k, "sr": p, "ci": list(wilson_p(p, n_eff)), "episodes": E,
          "n_eff": n_eff, "method": "clustered"}


def sign_test(Y: torch.Tensor, R: torch.Tensor) -> dict:
  """Filter ``Y`` vs SONIC ``R`` (``[S, E]`` bool, the same episodes): the
  exact two-sided sign test over episodes of the seed-mean difference
  (McNemar's test for S = 1)."""
  d = (Y.double() - R.double()).mean(0)
  b, c = int((d > 0).sum()), int((d < 0).sum())
  return {"episodes_better": b, "episodes_worse": c, "p": ql.mcnemar_exact_p(b, c),
          "test": "sign test over episodes of the seed-mean (filter - SONIC) outcome"}


def _stack(views: list, cond: str):
  if any(v is None or cond not in v["survived"] for v in views):
    return None
  return torch.stack([v["survived"][cond] for v in views])


def aggregate(sums: list, views=None, cond=None, shared: bool = True) -> dict:
  """One condition's per-seed summaries (``summarize_condition`` dicts; None
  where a seed lacks it) -> the cross-seed read-out (module docstring).
  ``views`` (aligned with ``sums``, :func:`episode_view`) carry the
  per-episode outcomes of condition ``cond``; ``shared``: the seeds were
  run with one ``--seed`` (POOLING)."""
  pairs = [(s, v) for s, v in zip(sums, views or [None] * len(sums)) if s]
  if not pairs:
    return {"seeds": 0}
  sums, views = [s for s, _ in pairs], [v for _, v in pairs]
  srs = [s["survival"]["sr"] for s in sums]
  m, sd, _ = mean_std(srs)
  out = {"seeds": len(sums), "sr_mean": m, "sr_std": sd, "sr_runs": srs}
  Y = None if cond is None else _stack(views, cond)
  if Y is not None and not all(torch.equal(views[0]["key"], v["key"]) for v in views[1:]):
    shared, Y = False, None          # the outcomes say the episode lists differ
  method = "independent" if not shared else ("clustered" if Y is not None else "conservative")
  flags = [f for f in _FLAGS if all(f"survival_excl_{f}" in s for s in sums)]
  for f, key in [(None, "survival")] + [(f, f"survival_excl_{f}") for f in flags]:
    name = "pooled" if f is None else f"pooled_excl_{f}"
    if method == "clustered" and (f is None or f in views[0]):
      out[name] = clustered(Y, None if f is None else ~views[0][f])
    else:
      # conservative: the CI of one seed's episodes (the seeds share them)
      out[name] = pooled([s[key] for s in sums], None if method == "independent"
                         else max(int(s[key]["n"]) for s in sums),
                         "independent" if method == "independent" else "conservative")
  if all(s.get("paired") for s in sums):
    b = sum(int(s["paired"]["rescued"]) for s in sums)
    c = sum(int(s["paired"]["broken"]) for s in sums)
    vs = {"rescued": b, "broken": c,
          "delta_sr_mean": mean_std([s["paired"]["delta_sr"] for s in sums])[0],
          "p_seeds": [s["paired"]["mcnemar_p"] for s in sums]}
    R = None if Y is None else _stack(views, ql.REFERENCE)
    if method == "clustered" and R is not None:
      vs.update(sign_test(Y, R))
    elif method == "independent":
      vs.update(p=ql.mcnemar_exact_p(b, c),
                test="exact McNemar on the summed discordant pairs (independent episodes)")
    else:
      vs.update(p=max(vs["p_seeds"]),
                test="largest per-seed exact McNemar p (per-episode outcomes unavailable)")
    out["vs_sonic"] = vs
  iv = {}
  for key in list(INTERVENTION_KEYS) + list(_EXTRA_KEYS):
    vals = [s.get("intervention", {}).get(key) for s in sums]
    iv[key] = mean_std(vals)[0]
  out["intervention"] = iv
  if all("by_dv" in s for s in sums):
    bins = list(sums[0]["by_dv"])
    if method == "clustered":
      idx = ql.bin_index(views[0]["dv"], [float(x) for x in _edges(bins)])
      out["by_dv"] = {b: clustered(Y, idx == i) for i, b in enumerate(bins)}
    else:
      out["by_dv"] = {b: pooled([s["by_dv"][b] for s in sums],
                                None if method == "independent"
                                else max(int(s["by_dv"][b]["n"]) for s in sums), method)
                      for b in bins}
  return out


def contrast(views_a: list, views_b: list, cond: str):
  """Group a against group b under ``cond``, on the SAME episodes: ``delta_sr``
  = SR(a) - SR(b) of the seed means, and the exact two-sided sign test over
  episodes of (a's seed-mean outcome - b's), which is McNemar's test for one
  seed each. None unless every run of both groups has its per-episode
  outcomes (:func:`episode_view`) of one and the same episode list."""
  if not views_a or not views_b:
    return None
  Ya, Yb = _stack(views_a, cond), _stack(views_b, cond)
  if Ya is None or Yb is None:
    return None
  key = views_a[0]["key"]
  if not all(torch.equal(key, v["key"]) for v in list(views_a) + list(views_b)):
    return None
  d = Ya.double().mean(0) - Yb.double().mean(0)
  b, c = int((d > 0).sum()), int((d < 0).sum())
  return {"seeds": [int(Ya.shape[0]), int(Yb.shape[0])], "episodes": int(d.numel()),
          "delta_sr": float(d.mean()), "episodes_better": b, "episodes_worse": c,
          "p": ql.mcnemar_exact_p(b, c),
          "test": "sign test over episodes of the seed-mean outcome difference (a - b)"}


def _edges(labels: list) -> list:
  """The bin edges behind ``qcbf_eval_lib.bin_labels``' "a-b" labels."""
  pts = [lab.split("-") for lab in labels]
  return [float(pts[0][0])] + [float(p[1]) for p in pts]


def _conditions(summaries: list) -> list:
  """Condition names in first-seen order."""
  out = []
  for s in summaries:
    for name in (s or {}):
      if name not in out:
        out.append(name)
  return out


def _shared(runs: dict, tags: list, suite: str) -> bool:
  """The runs ``tags`` were made with one ``--seed`` (their meta), so their
  episode lists coincide (POOLING); a run without a seed in its meta counts
  as sharing."""
  seeds = {((runs[t].get(suite) or {}).get("meta") or {}).get("seed") for t in tags
           if runs[t].get(suite)}
  seeds.discard(None)
  return len(seeds) <= 1


def build(runs: dict, groups: dict) -> dict:
  """The cross-seed tables (module docstring) of ``runs`` ({tag: load_run}),
  grouped by ``groups`` ({group: [tags]})."""
  res = {"groups": groups, "push": {}, "badref": {}, "badref_pooled": {},
         "badref_by_family": {}, "partial_badref": sorted(t for t, r in runs.items()
                                                            if r["badref_partial"])}
  oc = {t: r.get("outcomes") or {"push": None, "badref": {}} for t, r in runs.items()}
  push = {t: (r["push"] or {}).get("conditions") for t, r in runs.items()}
  for cond in _conditions(list(push.values())):
    res["push"][cond] = {g: aggregate([(push[t] or {}).get(cond) for t in tags],
                                      [oc[t]["push"] for t in tags], cond,
                                      _shared(runs, tags, "push"))
                         for g, tags in groups.items()}
  bad = {t: r["badref"] for t, r in runs.items() if r["badref"]}
  shared_b = {g: _shared(runs, tags, "badref") for g, tags in groups.items()}
  scen = []
  for b in bad.values():
    for s in b.get("scenarios", {}):
      if s not in scen:
        scen.append(s)
  for s in scen:
    per = {t: b["scenarios"].get(s, {}).get("conditions") for t, b in bad.items()}
    first = next(b["scenarios"][s] for b in bad.values() if s in b["scenarios"])
    res["badref"][s] = {"family": first.get("family"), "after": first.get("after"),
                        "conditions": {c: {g: aggregate([(per.get(t) or {}).get(c) for t in tags],
                                                        [oc[t]["badref"].get(s) for t in tags],
                                                        c, shared_b[g])
                                           for g, tags in groups.items()}
                                       for c in _conditions(list(per.values()))}}
  full = {t: b for t, b in bad.items() if "pooled" in b}

  def views_of(t, fam=None):
    # a complete run's scenarios end to end, in its own (the driver's pooled) order
    names = [s for s, v in full[t]["scenarios"].items() if fam is None or v.get("family") == fam]
    return concat_views([oc[t]["badref"].get(s) for s in names])

  for c in _conditions([b["pooled"] for b in full.values()]):
    res["badref_pooled"][c] = {g: aggregate([full[t]["pooled"].get(c) for t in tags if t in full],
                                            [views_of(t) for t in tags if t in full], c,
                                            shared_b[g])
                               for g, tags in groups.items()}
  fams = []
  for b in full.values():
    for f in b.get("by_family", {}):
      if f not in fams:
        fams.append(f)
  for f in fams:
    res["badref_by_family"][f] = {
      c: {g: aggregate([full[t]["by_family"].get(f, {}).get(c) for t in tags if t in full],
                       [views_of(t, f) for t in tags if t in full], c, shared_b[g])
          for g, tags in groups.items()}
      for c in _conditions([b.get("by_family", {}).get(f) for b in full.values()])}
  res["exact_vs_td"] = exact_vs_td(res["push"])
  # the groups against each other, paired on the shared episodes (contrast)
  gl = list(groups)
  res["push_contrast"], res["badref_contrast"] = {}, {}
  for i, ga in enumerate(gl):
    for gb in gl[i + 1:]:
      pair = f"{ga} vs {gb}"
      va, vb = ([oc[t]["push"] for t in groups[g]] for g in (ga, gb))
      rows = {c: contrast(va, vb, c) for c in res["push"]}
      res["push_contrast"][pair] = {c: r for c, r in rows.items() if r}
      per = {}
      for s, v in res["badref"].items():
        va, vb = ([oc[t]["badref"].get(s) for t in groups[g]] for g in (ga, gb))
        rows = {c: contrast(va, vb, c) for c in v["conditions"]}
        per[s] = {c: r for c, r in rows.items() if r}
      va, vb = ([views_of(t) if t in full else None for t in groups[g]] for g in (ga, gb))
      rows = {c: contrast(va, vb, c) for c in res["badref_pooled"]}
      per["all scenarios"] = {c: r for c, r in rows.items() if r}
      res["badref_contrast"][pair] = {s: r for s, r in per.items() if r}
  return res


def exact_vs_td(push: dict) -> list:
  """Rows pairing each qcbf (kappa, solver) and the switch across the two
  fits: ``{"key", "conditions": {variant: name}, groups: {g: {variant: agg}}}``."""
  keyed: dict = {}
  for name in push:
    try:
      c = ql.parse_condition(name)
    except ValueError:
      continue
    if c.kind == "qcbf":
      key = f"qcbf kappa {ql.kappa_str(c.kappa)}" + ("" if c.solver == ql.DEFAULT_SOLVER
                                                       else f" {c.solver}")
    elif c.kind == "lrsf":
      key = "lrsf (switch)"
    else:
      continue
    keyed.setdefault(key, {})[c.variant] = name
  rows = []
  for key, by_var in keyed.items():
    if len(by_var) < 2:
      continue
    groups = {}
    for v, name in by_var.items():
      for g, agg in push[name].items():
        groups.setdefault(g, {})[v] = agg
    rows.append({"key": key, "conditions": by_var, "groups": groups})
  return rows


# --- markdown ------------------------------------------------------------------------------

def _f(x, fmt=".3f") -> str:
  x = _num(x)
  return "-" if not math.isfinite(x) else format(x, fmt)


def _sr(agg: dict) -> str:
  if not agg or not agg.get("seeds"):
    return "-"
  sd = agg["sr_std"]
  return (f"{agg['sr_mean']:.3f}" + (f" ± {sd:.3f}" if math.isfinite(sd) else "")
          + f" ({agg['seeds']})")


def _ci(p: dict) -> str:
  if not p or not p.get("n"):
    return "-"
  return f"{p['sr']:.3f} [{_f(p['ci'][0])}, {_f(p['ci'][1])}]"


def _vs(agg: dict) -> str:
  v = (agg or {}).get("vs_sonic")
  if not v:
    return "-"
  return f"+{v['rescued']} / -{v['broken']} (p {_f(v.get('p'), '.2g')})"


def _iv(agg: dict) -> str:
  """Intervention rate pre / post the split, and the deviation per engaged step."""
  iv = (agg or {}).get("intervention")
  if not iv:
    return "-"
  return (f"{_f(iv.get('rate_pre'))} / {_f(iv.get('rate_post'))}, "
          f"{_f(iv.get('dev_per_engaged_post'), '.2f')}")


def _delta(r: dict) -> str:
  if not r:
    return "-"
  return (f"{_f(r['delta_sr'], '+.3f')} (+{r['episodes_better']} / -{r['episodes_worse']}, "
          f"p {_f(r['p'], '.2g')})")


def _table(head: list, rows: list) -> list:
  out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
  out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
  return out + [""]


def to_markdown(res: dict) -> str:
  groups = list(res["groups"])
  L = ["# Q-CBF on SONIC: SHAC vs PPO fallbacks", ""]
  L.append("Groups: " + "; ".join(f"{g} = {', '.join(t)}" for g, t in res["groups"].items()))
  L.append("SR = survival rate, mean ± std over the seeds (number of seeds); pooled = all "
           "seeds' episodes, Wilson 95 % CI. vs SONIC = episodes rescued / broken against "
           "unfiltered SONIC on the same episodes, summed over the seeds. The seeds run the "
           "SAME episodes (one --seed), so the pooled CI uses the episode-clustered effective "
           "sample size and p is a sign test over episodes of the seed-mean outcome difference "
           "(see each entry's `method` / `test` in the .json).")
  L.append("")
  if res["push"]:
    L += ["## Push suite (held-out LAFAN clips, velocity kick)", ""]
    head = ["condition"] + [h for g in groups for h in (f"{g} SR", f"{g} pooled", f"{g} vs SONIC")]
    rows = [[c] + [x for g in groups for x in (_sr(a[g]), _ci(a[g].get("pooled")), _vs(a[g]))]
            for c, a in res["push"].items()]
    L += _table(head, rows)
    for pair, by_c in res.get("push_contrast", {}).items():
      if not by_c:
        continue
      L += [f"### {pair} on the same episodes (paired)", "",
            "delta SR = the first group's SR - the second's (seed means); episodes where the "
            "first did better / worse; p = sign test over episodes. Under `sonic` both "
            "ran unfiltered SONIC: that row is the simulator-noise floor.", ""]
      L += _table(["condition", "delta SR (+better / -worse, p)"],
                  [[c, _delta(r)] for c, r in by_c.items()])
    L += ["### Intervention (means over the seeds)", ""]
    head = ["condition"] + [f"{g} {lab}" for g in groups for lab in INTERVENTION_KEYS.values()]
    rows = [[c] + [_f(a[g].get("intervention", {}).get(k), ".3f" if "dev" not in k else ".2f")
                   for g in groups for k in INTERVENTION_KEYS]
            for c, a in res["push"].items() if c != ql.REFERENCE]
    L += _table(head, rows)
  if res["exact_vs_td"]:
    L += ["### Exact (3 s rollout labels) vs td_g099 (TD, gamma 0.99) values", ""]
    variants = sorted({v for r in res["exact_vs_td"] for v in r["conditions"]})
    head = ["filter"] + [f"{g} SR {v}" for g in groups for v in variants] + \
           [f"{g} int. rate post {v}" for g in groups for v in variants]
    rows = []
    for r in res["exact_vs_td"]:
      rows.append([r["key"]] + [_sr(r["groups"].get(g, {}).get(v)) for g in groups for v in variants]
                  + [_f((r["groups"].get(g, {}).get(v) or {}).get("intervention", {}).get("rate_post"))
                     for g in groups for v in variants])
    L += _table(head, rows)
  dv_rows = [(c, g, a[g]["by_dv"]) for c, a in res["push"].items() for g in groups
             if a[g].get("by_dv")]
  if dv_rows:
    bins = list(dv_rows[0][2])
    L += ["### Push SR by |dv| (m/s), pooled over the seeds", ""]
    L += _table(["condition", "group"] + bins,
                [[c, g] + [_ci(d[b]) for b in bins] for c, g, d in dv_rows])
  if res["badref"]:
    L += ["## Bad-reference suite (ill-posed references for SONIC)", ""]
    if res["partial_badref"]:
      L += [f"Partial (badref.partial.json): {', '.join(res['partial_badref'])}", ""]
    conds = _conditions([s["conditions"] for s in res["badref"].values()])
    for g in groups:
      L += [f"### {g}: SR per scenario", ""]
      rows = [[s, v["family"]] + [_sr(v["conditions"].get(c, {}).get(g)) for c in conds]
              for s, v in res["badref"].items()]
      if res["badref_pooled"]:
        rows.append(["**all scenarios**", ""]
                    + [_sr(res["badref_pooled"].get(c, {}).get(g)) for c in conds])
      for fam, by_c in res["badref_by_family"].items():
        rows.append([f"family {fam}", ""] + [_sr(by_c.get(c, {}).get(g)) for c in conds])
      L += _table(["scenario", "family"] + conds, rows)
      L += [f"### {g}: rescued / broken vs SONIC per scenario", ""]
      rows = [[s] + [_vs(v["conditions"].get(c, {}).get(g)) for c in conds if c != ql.REFERENCE]
              for s, v in res["badref"].items()]
      if res["badref_pooled"]:
        rows.append(["**all scenarios**"] + [_vs(res["badref_pooled"].get(c, {}).get(g))
                                             for c in conds if c != ql.REFERENCE])
      L += _table(["scenario"] + [c for c in conds if c != ql.REFERENCE], rows)
      L += [f"### {g}: intervention rate pre / post, deviation per engaged step", ""]
      rows = [[s] + [_iv(v["conditions"].get(c, {}).get(g)) for c in conds if c != ql.REFERENCE]
              for s, v in res["badref"].items()]
      if res["badref_pooled"]:
        rows.append(["**all scenarios**"] + [_iv(res["badref_pooled"].get(c, {}).get(g))
                                             for c in conds if c != ql.REFERENCE])
      L += _table(["scenario"] + [c for c in conds if c != ql.REFERENCE], rows)
    for pair, by_s in res.get("badref_contrast", {}).items():
      if not by_s:
        continue
      L += [f"### {pair} per scenario (paired, delta SR as in the push suite)", ""]
      L += _table(["scenario"] + conds,
                  [[s if s != "all scenarios" else "**all scenarios**"]
                   + [_delta(r.get(c)) for c in conds] for s, r in by_s.items()])
  return "\n".join(L)


# --- main ---------------------------------------------------------------------------------

def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--root", default=DEFAULT_ROOT, help="the run directories' parent")
  p.add_argument("--group", nargs="+", action="append", metavar=("NAME", "TAG"),
                 help="a group and its run tags (repeatable; default: by tag prefix)")
  p.add_argument("--out", default=None, help="output stem (default <root>/summary)")
  return p.parse_args(argv)


def main(argv=None) -> dict:
  a = _parse(argv)
  found = discover(a.root)
  if a.group:
    groups = {g[0]: list(g[1:]) for g in a.group}
  else:
    groups = {}
    for tag in found:
      groups.setdefault(default_group(tag), []).append(tag)
  runs = {}
  for g, tags in groups.items():
    keep = []
    for t in tags:
      if t not in found:
        print(f"[summary] {g}: no evaluation under {os.path.join(a.root, t)}; skipped")
        continue
      runs[t] = load_run(found[t])
      keep.append(t)
    groups[g] = keep
  groups = {g: t for g, t in groups.items() if t}
  if not runs:
    raise SystemExit(f"[summary] no evaluation found under {a.root}")
  res = build(runs, groups)
  stem = a.out or os.path.join(a.root, "summary")
  os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
  with open(stem + ".json", "w") as f:
    json.dump(ql.clean_json(res), f, indent=1)
  md = to_markdown(res)
  with open(stem + ".md", "w") as f:
    f.write(md + "\n")
  print(md)
  print(f"[summary] {len(runs)} runs in {len(groups)} groups -> {stem}.md / .json")
  return res


if __name__ == "__main__":
  main()
