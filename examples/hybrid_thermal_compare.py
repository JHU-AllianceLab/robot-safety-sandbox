"""Step 0 of the hybrid thermal world model: the learned (nn) vs hybrid WM filters on thermal70.

    python examples/hybrid_thermal_compare.py \
        --nn-root runs/shac_thermal70/wmreport --root runs/shac_thermal70/hybrid_step0 \
        --out runs/shac_thermal70/hybrid_step0/compare.json

Reads, per SHAC seed (tags thermal70_s0..2):
  * the nn filters: ``<nn-root>/eval/<tag>/push.pt`` (the original WM-filter report);
  * the hybrid filters: ``<root>/eval/<mode>/<tag>/push.pt`` (mode hybrid, hybrid-temps);
  * the calibration tables ``<root>/calib/<mode>/<tag>/min_tube.json`` (mode nn too:
    the hottest-winding prediction error of each model) and the heat model's
    one-step check (``thermal_diag`` in the hybrid calibration run's push.pt).
Every condition's episodes are the same list in every run (asserted). Pooled over
seeds: survival, overheated, fell for other causes, intervention rates; and per
condition, hybrid vs nn paired on the same episodes (rescued / broken, McNemar).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcbf_eval_lib as ql  # noqa: E402

TAGS = ("thermal70_s0", "thermal70_s1", "thermal70_s2")
MODES = ("hybrid", "hybrid-temps")
FILTERS = ("wmmin:0.4:0.95", "wmmin:1:0.95", "wmmin:3:0.95", "wmmin:1:0.95+dwell1",
           "wmmin:1:0.95+dwell2", "wmmin:1:0.95+latch")


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--nn-root", default="runs/shac_thermal70/wmreport")
  p.add_argument("--root", default="runs/shac_thermal70/hybrid_step0")
  p.add_argument("--tags", nargs="+", default=list(TAGS))
  p.add_argument("--modes", nargs="+", default=list(MODES))
  p.add_argument("--out", default=None)
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(a.root, "compare.json")
  return a


def _load(path):
  return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None


def _pct(k, n):
  return 100.0 * k / n if n else math.nan


def _row(recs: list, summ: list) -> dict:
  surv = torch.cat([r["survived"].bool() for r in recs])
  over = torch.cat([r.get("overheated", torch.zeros_like(r["survived"])).bool() for r in recs])
  n = int(surv.numel())
  iv = [s["intervention"] for s in summ]
  w = [s["episodes"] for s in summ]
  def wmean(key):
    vals = [(x.get(key), k) for x, k in zip(iv, w) if x.get(key) is not None
            and not (isinstance(x.get(key), float) and math.isnan(x.get(key)))]
    return sum(v * k for v, k in vals) / sum(k for _, k in vals) if vals else None
  peak = torch.cat([r["tw_peak_c"].double() for r in recs]) if "tw_peak_c" in recs[0] else None
  return {"episodes": n, "survival_pct": _pct(int(surv.sum()), n),
          "overheated_pct": _pct(int(over.sum()), n),
          "fell_other_pct": _pct(int((~surv & ~over).sum()), n),
          "intervention_pre": wmean("rate_pre"), "intervention_post": wmean("rate_post"),
          "false_alarm_episodes": wmean("false_alarm_episodes"),
          "peak_winding_mean_C": float(peak.mean()) if peak is not None else None,
          "per_seed_survival_pct": [_pct(int(r["survived"].sum()), int(r["survived"].numel()))
                                    for r in recs]}


def main(argv=None):
  a = _parse(argv)
  rep = {"nn_root": os.path.abspath(a.nn_root), "root": os.path.abspath(a.root), "tags": a.tags,
         "conditions": {}, "paired_vs_nn": {}, "hottest_winding_error": {},
         "thermal_diag": {}, "coverage": {}, "missing": []}
  nn = {t: _load(os.path.join(a.nn_root, "eval", t, "push.pt")) for t in a.tags}
  hy = {m: {t: _load(os.path.join(a.root, "eval", m, t, "push.pt")) for t in a.tags}
        for m in a.modes}
  for t in a.tags:
    if nn[t] is None:
      rep["missing"].append(f"nn eval {t}")
    for m in a.modes:
      if hy[m][t] is None:
        rep["missing"].append(f"{m} eval {t}")
  # the episode lists must be the same run to run
  for m in a.modes:
    for t in a.tags:
      if nn[t] is not None and hy[m][t] is not None:
        for key in ("spawn_clip", "spawn_frame", "on", "dv"):
          if key in nn[t]["episodes"] and not torch.equal(nn[t]["episodes"][key],
                                                          hy[m][t]["episodes"][key]):
            raise SystemExit(f"[compare] {m} {t}: episode column {key} differs from the nn run")
  tags_ok = {m: [t for t in a.tags if nn[t] is not None and hy[m][t] is not None]
             for m in a.modes}

  def block(runs, tags, cond):
    recs = [runs[t]["conditions"][cond] for t in tags if cond in runs[t]["conditions"]]
    summ = [runs[t]["summary"][cond] for t in tags if cond in runs[t]["conditions"]]
    return _row(recs, summ) if len(recs) == len(tags) and recs else None

  for m in a.modes:
    tags = tags_ok[m]
    if not tags:
      continue
    for cond in ("sonic",) + FILTERS:
      r_nn, r_h = block(nn, tags, cond), block(hy[m], tags, cond)
      rep["conditions"].setdefault(cond, {})
      rep["conditions"][cond]["nn"] = r_nn
      rep["conditions"][cond][m] = r_h
      if r_nn is None or r_h is None:
        continue      # sonic too: the same controller twice, i.e. the rerun noise floor
      s_nn = torch.cat([nn[t]["conditions"][cond]["survived"].bool() for t in tags])
      s_h = torch.cat([hy[m][t]["conditions"][cond]["survived"].bool() for t in tags])
      p = ql.paired(s_nn, s_h)
      rep["paired_vs_nn"].setdefault(cond, {})[m] = {
        "tags": tags, "rescued": p["rescued"], "broken": p["broken"],
        "mcnemar_p": p["mcnemar_p"], "delta_survival_pct": _pct(int(s_h.sum()) - int(s_nn.sum()),
                                                               int(s_nn.numel()))}
  for m in ("nn",) + tuple(a.modes):
    for t in a.tags:
      j = os.path.join(a.root, "calib", m, t, "min_tube.json")
      if os.path.exists(j):
        with open(j) as f:
          tube = json.load(f)
        rep["hottest_winding_error"].setdefault(m, {})[t] = tube.get("hottest_winding_error")
        rep["coverage"].setdefault(m, {})[t] = {
          h: {"eps_0.95": row["levels"]["0.95"]["eps_one_sided"],
              "coverage_0.95": row["levels"]["0.95"]["one_sided_coverage"],
              "fall_recall_0.95": row["levels"]["0.95"]["fall_recall"],
              "false_alarm_0.95": row["levels"]["0.95"]["false_alarm"]}
          for h, row in tube["horizons"].items()}
      else:
        rep["missing"].append(f"{m} calib table {t}")
  for m in a.modes:
    for t in a.tags:
      d = _load(os.path.join(a.root, "calib", m, t, "push.pt"))
      if d is not None:
        td = d["meta"]["runs"].get("wmcalib:clean", {}).get("thermal_diag")
        if td is not None:
          rep["thermal_diag"].setdefault(m, {})[t] = {k: v for k, v in td.items()
                                                      if k != "per_motor"}
          rep["thermal_diag"][m][t]["per_motor"] = td["per_motor"]

  with open(a.out, "w") as f:
    json.dump(ql.clean_json(rep), f, indent=1)
  # --- the table
  print(f"[compare] seeds {a.tags}; missing: {rep['missing'] or 'none'}")
  hdr = f"{'condition':24s} {'model':13s} {'surv%':>6s} {'over%':>6s} {'fell%':>6s} {'int pre':>7s} {'int post':>8s}  vs nn"
  print(hdr)
  for cond, rows in rep["conditions"].items():
    for m in ("nn",) + tuple(a.modes):
      r = rows.get(m)
      if r is None:
        continue
      pv = rep["paired_vs_nn"].get(cond, {}).get(m)
      pv_s = (f"+{pv['rescued']}/-{pv['broken']} (p {pv['mcnemar_p']:.2g})" if pv else "")
      ip, io = r["intervention_pre"], r["intervention_post"]
      print(f"{cond:24s} {m:13s} {r['survival_pct']:6.1f} {r['overheated_pct']:6.1f} "
            f"{r['fell_other_pct']:6.1f} {ip if ip is None else round(ip, 3)!s:>7s} "
            f"{io if io is None else round(io, 3)!s:>8s}  {pv_s}")
  print(f"[compare] wrote {a.out}")


if __name__ == "__main__":
  main()
