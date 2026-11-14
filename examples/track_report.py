"""Aggregate the squat-tracking evals into one report: per-seed points, arm summaries,
the pre-registered contrasts, paired tests, and learning curves.

    python examples/track_report.py --stage pilot
    python examples/track_report.py --stage final

Reads runs/track_squat/eval/<stage>/<arm>_s<seed>/<world>/{summary.json,records.pt}
(examples/eval_track.py; the SONIC anchor under eval/<stage>/sonic/<world>/), the
audits runs/track_squat/audit/<world>/audit.json (the oracle engagement floor of
each tube, C1), and the runs' TensorBoard events. Writes
runs/track_squat/report/<stage>/{report.json, report.log, curves_*.png}.

The unit of inference is the TRAINING RUN (task seed i, paired with fallback seed
i). Pre-registered (AMENDMENTS C9), on three primary endpoints -- any-failure per
episode, tracked-frame fraction TF, and dTw_peak (the hottest motor's rise) -- each
on the Full suite in the policy's own world:

  P1 learning effect     W_wm_dX policy minus W_bare policy, BOTH deployed with the
                         same dX filter (fallback seed = the task seed), per seed i
  P2 deployment effect   each policy deployed with its world's dX filter minus the
                         same policy deployed bare
  P3 world contrast      t70 arm minus the cbf arm of the same filter config, each in
                         its own world (descriptive: the worlds differ in dynamics,
                         termination, margin, fallback and eps)

A contrast reports the per-seed differences, their mean, and a bootstrap interval
over seeds (all resamples when there are few seeds), and per seed a McNemar exact
test of any-failure on the paired starts. With 3 seeds, differences inside the
largest seed spread are treated as null (C10). Every cbf summary is given with and
without fallback seed 0 (its tube eps 0.589 is the outlier, C4).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import track_common as tc  # noqa: E402

PRIMARY = ("any_fail", "tf", "dtw_peak")
SECONDARY = ("survival", "overheat_incidence", "fail_contact", "fail_tilt", "completion",
             "err_body_alive", "heat_dose_per_T", "tau2_per_step", "tw_peak", "engaged_frac",
             "handovers_per_ep")
CURVE_TAGS = ("env/track/fail_per_1k", "env/track/fail_overheat_per_1k", "env/track/err_body",
              "env/track/err_q",
              "env/track/tracked_frac", "env/track/tw_max_mean", "env/track/completion",
              "env/safety/engaged_frac", "amp/ep_task_return", "amp/r_style_mean",
              "rollout/ep_rew_mean")


# --- statistics (pure) ----------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
  """Two-sided exact McNemar p-value from the discordant counts b, c."""
  n = b + c
  if n == 0:
    return 1.0
  k = min(b, c)
  tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
  return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(xs, level: float = 0.95, n_random: int = 20000, seed: int = 0):
  """Percentile interval of the mean over resamples of ``xs`` (every resample when
  len(xs)**len(xs) <= 50000, else ``n_random`` random ones)."""
  xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
  n = len(xs)
  if n == 0:
    return (math.nan, math.nan)
  if n ** n <= 50000:
    means = sorted(sum(c) / n for c in itertools.product(xs, repeat=n))
  else:
    import random
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(n_random))
  lo = means[int(math.floor((1 - level) / 2 * (len(means) - 1)))]
  hi = means[int(math.ceil((1 + level) / 2 * (len(means) - 1)))]
  return (lo, hi)


def mean_std(xs):
  xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
  if not xs:
    return (math.nan, math.nan, 0)
  m = sum(xs) / len(xs)
  sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
  return (m, sd, len(xs))


# --- loading ------------------------------------------------------------------------------

def load_evals(stage: str) -> dict:
  """{(arm, seed, world): {"summary": ..., "records_path": ...}} plus ("sonic", None, world)."""
  base = os.path.join(tc.RUNS, "eval", stage)
  out = {}
  if not os.path.isdir(base):
    return out
  for d in sorted(os.listdir(base)):
    for world in tc.WORLDS:
      p = os.path.join(base, d, world, "summary.json")
      if not os.path.isfile(p):
        continue
      if d == "sonic":
        key = ("sonic", None, world)
      else:
        arm, _, s = d.rpartition("_s")
        if arm not in tc.ARMS or not s.isdigit():
          continue
        key = (arm, int(s), world)
      with open(p) as f:
        out[key] = {"summary": json.load(f),
                    "records_path": os.path.join(base, d, world, "records.pt")}
  return out


def native_condition(arm: str, seed: int, dwell=None) -> str:
  """The condition key a policy is deployed under: its own filter at its paired
  fallback seed, or another dwell's filter (``dwell``), or bare."""
  world, kind, d = tc.ARMS[arm]
  if dwell is None:
    if kind is None:
      return "bare"
    dwell = d
  if dwell == "bare":
    return "bare"
  return f"wm_d{int(dwell)}@fb{seed}"


def _cond(evals, arm, seed, world, cond):
  e = evals.get((arm, seed, world))
  if e is None:
    return None
  return e["summary"]["conditions"].get(cond)


def _records(evals, arm, seed, world, cond):
  import torch
  e = evals.get((arm, seed, world))
  if e is None or not os.path.isfile(e["records_path"]):
    return None
  r = torch.load(e["records_path"], map_location="cpu", weights_only=False)["records"]
  return r.get(cond)


def paired_mcnemar(ra, rb, key: str = "any_fail"):
  if ra is None or rb is None:
    return None
  a, b = ra[key].bool(), rb[key].bool()
  if a.shape != b.shape:
    return None
  n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
  return {"a_only": n10, "b_only": n01, "p": mcnemar_exact(n10, n01)}


# --- the report ------------------------------------------------------------------------------

def contrasts(evals, seeds) -> dict:
  out = {}

  def emit(name, pairs):
    """pairs: [(seed, (arm_a, world_a, cond_a), (arm_b, world_b, cond_b))]"""
    rows, diffs = [], {k: [] for k in PRIMARY + SECONDARY}
    for s, A, B in pairs:
      sa, sb = _cond(evals, A[0], s, A[1], A[2]), _cond(evals, B[0], s, B[1], B[2])
      if sa is None or sb is None:
        continue
      row = {"seed": s, "a": A, "b": B}
      for k in PRIMARY + SECONDARY:
        va, vb = sa.get(k), sb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
          row[k] = va - vb
          diffs[k].append(va - vb)
      row["mcnemar_any_fail"] = paired_mcnemar(_records(evals, A[0], s, A[1], A[2]),
                                               _records(evals, B[0], s, B[1], B[2]))
      rows.append(row)
    if not rows:
      return
    summ = {}
    for k, v in diffs.items():
      if v:
        m, sd, n = mean_std(v)
        summ[k] = {"mean": m, "sd": sd, "n": n, "ci": bootstrap_mean_ci(v),
                   "range": [min(v), max(v)]}
    out[name] = {"per_seed": rows, "summary": summ}

  for w in tc.WORLDS:
    sh = tc.WORLD_SHORT[w]
    for dw in (0, 1):
      cond = lambda s, dw=dw: f"wm_d{dw}@fb{s}"  # noqa: E731
      emit(f"P1_{sh}_d{dw}", [(s, (f"{sh}_wm_d{dw}", w, cond(s)), (f"{sh}_bare", w, cond(s)))
                              for s in seeds])
    for arm in (a for a in tc.ARMS if tc.arm_world(a) == w):
      for dw in (0, 1):
        emit(f"P2_{arm}_d{dw}", [(s, (arm, w, f"wm_d{dw}@fb{s}"), (arm, w, "bare"))
                                 for s in seeds])
  for cfgname in ("bare", "wm_d0", "wm_d1"):
    emit(f"P3_{cfgname}", [(s, (f"t70_{cfgname}", "thermal70", native_condition(f"t70_{cfgname}", s)),
                            (f"cbf_{cfgname}", "thermal_cbf", native_condition(f"cbf_{cfgname}", s)))
                           for s in seeds])
  return out


def arm_summaries(evals, seeds) -> dict:
  out = {}
  for arm in tc.ARMS:
    w = tc.arm_world(arm)
    for label, ss in (("all", seeds), ("no_fb0", [s for s in seeds if s != 0])):
      if label == "no_fb0" and not arm.startswith("cbf"):
        continue
      pts = {}
      for s in ss:
        c = _cond(evals, arm, s, w, native_condition(arm, s))
        if c is not None:
          pts[s] = c
      if not pts:
        continue
      row = {"seeds": sorted(pts)}
      for k in PRIMARY + SECONDARY:
        vals = [p.get(k) for p in pts.values() if isinstance(p.get(k), (int, float))]
        if vals:
          m, sd, n = mean_std(vals)
          row[k] = {"mean": m, "sd": sd, "points": {s: pts[s].get(k) for s in pts}}
      out[f"{arm}/{label}"] = row
  return out


def oracle_floor() -> dict:
  out = {}
  for w in tc.WORLDS:
    p = os.path.join(tc.RUNS, "audit", w, "audit.json")
    if os.path.isfile(p):
      with open(p) as f:
        tubes = (json.load(f).get("oracle") or {}).get("tubes") or {}
      out[w] = {t: v.get("floor_frac") for t, v in tubes.items()}
  return out


def engagement_excess(evals, floor) -> dict:
  out = {}
  for (arm, seed, world), e in evals.items():
    for cname, c in e["summary"]["conditions"].items():
      cd = c.get("condition", {})
      if cd.get("kind") != "wm" or "engaged_frac" not in c:
        continue
      fb = cd.get("fb_seed")
      tag = tc.tube_tag(world, fb) if fb is not None else None
      fl = (floor.get(world) or {}).get(tag)
      out[f"{arm}_s{seed}/{world}/{cname}"] = {
        "engaged_frac": c["engaged_frac"], "floor": fl,
        "excess": None if fl is None else c["engaged_frac"] - fl}
  return out


def hash_check(evals) -> dict:
  """Every condition of every policy in a world should start from the same states."""
  out = {}
  for w in tc.WORLDS:
    seen = {}
    for (arm, seed, world), e in evals.items():
      if world != w:
        continue
      es = e["summary"]["meta"].get("eval_seed")
      suite = e["summary"]["meta"].get("suite")
      for cname, c in e["summary"]["conditions"].items():
        h = c.get("hashes") or {}
        seen.setdefault((es, suite), {}).setdefault(json.dumps(h, sort_keys=True), []).append(
          f"{arm}_s{seed}/{cname}")
    out[w] = {f"seed{k[0]}_{k[1]}": {"distinct": len(v),
                                     "groups": {h[:60]: len(n) for h, n in v.items()}}
              for k, v in seen.items()}
  return out


def curves(stage: str, outdir: str) -> dict:
  """Training curves from each run's TensorBoard events -> one PNG per tag."""
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except Exception as e:                       # noqa: BLE001
    return {"error": f"{type(e).__name__}: {e}"}
  series = {}
  for arm in tc.ARMS:
    for s in tc.SEEDS:
      d = tc.run_dir(stage, arm, s)
      if not os.path.isdir(d):
        continue
      for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub)
        if not (sub.startswith("track_") and os.path.isdir(p)):
          continue
        acc = EventAccumulator(p, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in CURVE_TAGS:
          if tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, {})[f"{arm}_s{s}"] = [(ev.step, ev.value)
                                                        for ev in acc.Scalars(tag)]
  written = []
  for tag, runs in series.items():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, pts in sorted(runs.items()):
      ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, lw=1)
    ax.set_title(tag)
    ax.set_xlabel("env steps")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    path = os.path.join(outdir, f"curves_{tag.replace('/', '_')}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)
  return {"tags": sorted(series), "png": written, "budget": budget_rule(series)}


def repeat_floor(stage: str) -> dict:
  """The nondeterminism floor: the same policy and condition evaluated twice
  (eval_track --repeat 2) -- differences of the primary endpoints and the
  discordant episodes, the noise any contrast must clear."""
  import torch
  base = os.path.join(tc.RUNS, "eval", stage, "repeat")
  out = {}
  for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
    p = os.path.join(base, d, "summary.json")
    if not os.path.isfile(p):
      continue
    conds = json.load(open(p))["conditions"]
    recs = torch.load(os.path.join(base, d, "records.pt"), map_location="cpu",
                      weights_only=False)["records"]
    for k in conds:
      if not k.endswith("#r0") or k[:-1] + "1" not in conds:
        continue
      k1 = k[:-1] + "1"
      out[f"{d}/{k[:-3]}"] = {
        "diff": {m: conds[k][m] - conds[k1][m] for m in PRIMARY + ("survival",)},
        "mcnemar_any_fail": paired_mcnemar(recs.get(k), recs.get(k1))}
  return out


def budget_rule(series: dict, lo: float = 4e8, hi: float = 1.5e9, ema: float = 0.05) -> dict:
  """The final budget B from the pilot's joint-tracking error curves (plan section 4):
  per run, t95 = the first step at which the EMA of err_q is within 5% of its final
  value; B = clamp(roundup_100M(1.25 max t95), lo, hi); B = hi when any run still
  improved err_q by more than 5% over its last 20% of steps."""
  t95s, still = {}, []
  for name, pts in (series.get("env/track/err_q") or {}).items():
    if len(pts) < 5:
      continue
    e, sm = None, []
    for _, v in pts:
      e = v if e is None else (1 - ema) * e + ema * v
      sm.append(e)
    steps = [p[0] for p in pts]
    final = sm[-1]
    t95s[name] = next(st for st, v in zip(steps, sm) if v <= 1.05 * final)
    i80 = next(i for i, st in enumerate(steps) if st >= 0.8 * steps[-1])
    if sm[i80] > 0 and (sm[i80] - final) / sm[i80] > 0.05:
      still.append(name)
  if not t95s:
    return {"error": "no env/track/err_q curves"}
  B = hi if still else min(hi, max(lo, math.ceil(1.25 * max(t95s.values()) / 1e8) * 1e8))
  return {"B": int(B), "t95": t95s, "still_improving": still}


def _fmt(x, nd=3):
  return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--stage", default="pilot")
  p.add_argument("--no-curves", action="store_true")
  a = p.parse_args(argv)
  evals = load_evals(a.stage)
  out = tc.report_dir(a.stage)
  os.makedirs(out, exist_ok=True)
  seeds = sorted({k[1] for k in evals if k[1] is not None})
  floor = oracle_floor()
  rep = {"stage": a.stage, "seeds": seeds, "n_evals": len(evals),
         "arms": arm_summaries(evals, seeds), "contrasts": contrasts(evals, seeds),
         "oracle_floor": floor, "engagement_excess": engagement_excess(evals, floor),
         "hashes": hash_check(evals), "repeat_floor": repeat_floor(a.stage),
         "sonic": {w: e["summary"]["conditions"] for (k, _, w), e in evals.items() if k == "sonic"},
         "points": {f"{k[0]}_s{k[1]}/{k[2]}": {c: {m: v.get(m) for m in PRIMARY + SECONDARY}
                                              for c, v in e["summary"]["conditions"].items()}
                    for k, e in evals.items() if k[0] != "sonic"}}
  if not a.no_curves:
    rep["curves"] = curves(a.stage, out)
  with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(rep, f, indent=1, default=repr)
  lines = [f"# squat tracking report, stage {a.stage}: {len(evals)} evals, seeds {seeds}", ""]
  lines.append("## arms (native: own world, own filter at the paired fallback seed; Full suite)")
  lines.append(f"{'arm':22s} " + " ".join(f"{k:>16s}" for k in PRIMARY + ("survival",
                                                                         "overheat_incidence",
                                                                         "engaged_frac")))
  for name, row in rep["arms"].items():
    cells = []
    for k in PRIMARY + ("survival", "overheat_incidence", "engaged_frac"):
      v = row.get(k)
      cells.append(f"{_fmt(v['mean'])}+-{_fmt(v['sd'], 2):>5s}" if v else f"{'-':>16s}")
    lines.append(f"{name:22s} " + " ".join(f"{c:>16s}" for c in cells))
  lines += ["", "## pre-registered contrasts (a - b; per-seed mean, bootstrap CI over seeds)"]
  for name, c in rep["contrasts"].items():
    parts = []
    for k in PRIMARY:
      s = c["summary"].get(k)
      if s:
        parts.append(f"{k} {_fmt(s['mean'])} [{_fmt(s['ci'][0])}, {_fmt(s['ci'][1])}]")
    mc = [r.get("mcnemar_any_fail") for r in c["per_seed"]]
    ps = ", ".join(f"s{r['seed']} p={_fmt(m['p'])}" for r, m in zip(c["per_seed"], mc) if m)
    lines.append(f"{name:18s} " + "; ".join(parts) + (f"  | McNemar {ps}" if ps else ""))
  lines += ["", "## start-state pairing (distinct hash groups per world; 1 = all paired)"]
  for w, v in rep["hashes"].items():
    lines.append(f"{w}: " + ", ".join(f"{k}: {g['distinct']}" for k, g in v.items()))
  if rep["repeat_floor"]:
    lines += ["", "## nondeterminism floor (same policy and condition, evaluated twice)"]
    for k, v in rep["repeat_floor"].items():
      m = v["mcnemar_any_fail"] or {}
      lines.append(f"{k}: " + ", ".join(f"d{n} {_fmt(x)}" for n, x in v["diff"].items())
                   + f"; discordant any-fail {m.get('a_only')}/{m.get('b_only')}")
  if rep["sonic"]:
    lines += ["", "## SONIC anchor"]
    for w, conds in rep["sonic"].items():
      for cn, c in conds.items():
        lines.append(f"{w:12s} {cn:14s} any_fail {_fmt(c.get('any_fail'))} TF {_fmt(c.get('tf'))} "
                     f"dTw {_fmt(c.get('dtw_peak'), 2)} overheat {_fmt(c.get('overheat_incidence'))}")
  bud = (rep.get("curves") or {}).get("budget")
  if bud:
    lines += ["", f"## budget rule (pilot err_q curves): {bud}"]
  with open(os.path.join(out, "report.log"), "w") as f:
    f.write("\n".join(lines) + "\n")
  print("\n".join(lines))
  print(f"[track-report] wrote {out}/report.json, report.log")


if __name__ == "__main__":
  main()
