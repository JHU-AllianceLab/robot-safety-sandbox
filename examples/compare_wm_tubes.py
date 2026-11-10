"""Compare the conformal worst-margin tubes of several world models of the same policies.

    python examples/compare_wm_tubes.py \
        --arm teacher runs/shac_noheight/wmreport/calib/noheight_s{0,1,2} \
        --arm control runs/shac_student/calib/full_s{0,1,2} \
        --arm student runs/shac_student/calib/nofoot_s{0,1,2} \
        --out runs/shac_student/tube_compare.json

Each DIR is a calibration run of ``eval_sonic_qcbf.py --conditions sonic
wmcalib:clean`` (``push.pt``) plus its ``calibrate_wm_min_tube.py`` fit
(``min_tube.json``). The k-th DIR of every arm should be the SAME policy on the
SAME episodes (same --seed / --episodes / bank). That pins the start states,
pushes and domain randomization, but NOT the simulator's float determinism:
runs of the same --seed differ (even SONIC alone survives 1243-1252 / 2048),
so the post-push trajectories and real margins differ between runs and the
arms are compared over the same episode distribution, not paired (checked
per seed and reported as ``real_g_identical``; expect False). The yardstick
for a difference is a rerun of one arm (configs/sweeps/shac_student/
teacher_rerun_jobs.txt).

Per arm, the mean over its seeds (and each seed) of:
  * the tube width the filter uses, ``min_eps`` (one-sided, refit on all
    episodes) at c = 0.9 / 0.95 / 0.99 and H = 0.1 / 0.4 / 1 / 3 s, and the
    two-sided half-width;
  * the split-half check at each c: one-sided coverage on the test half, the
    coverage among the episodes that really fell within H, fall recall and
    false alarms of "G^ - eps < 0";
  * the score quantiles r = G^ - G of the calibration half (before any eps);
  * the world model's accuracy (``wm_filter_report.accuracy``): |g^ - g| at
    1 / 10 steps, 0.4 / 1 / 3 s, and the median |r_H|;
  * the score r_H by DECISION DELAY (decision step - push onset, U{0 .. 1 s}
    in the calibration), pooled over seeds: its 0.9 / 0.95 quantile per delay
    bin at H = 0.4 / 1 s -- how soon after a kick the model's prediction is
    good (a state without base_lin_vel cannot see the kick's velocity jump at
    once).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_filter_report import _load, accuracy, calib_arrays  # noqa: E402

COVERAGES = ("0.9", "0.95", "0.99")
DELAY_BINS_S = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01))
H_STEPS = {"0.1 s": "5", "0.4 s": "20", "1 s": "50", "3 s": "150"}


def _mean(xs):
  xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
  return sum(xs) / len(xs) if xs else None


def by_delay(cals: list[dict]) -> dict | None:
  """Quantiles of r_H = min_{j<H} g^ - min_{j<H} g per decision-delay bin, pooled."""
  if not all("decision" in c for c in cals):
    return None
  dt = cals[0]["dt"]
  pred = torch.cat([c["pred_g"] for c in cals])
  real = torch.cat([c["real_g"] for c in cals])
  dec = torch.cat([c["decision"] for c in cals]).double() * dt
  out = {}
  for h, steps in (("0.4 s", 20), ("1 s", 50)):
    # a fall ends the real record (NaN): the real min is then the fall's margin, as in the tube fit
    r = torch.nan_to_num(pred[:, :steps], nan=math.inf).amin(1) \
        - torch.nan_to_num(real[:, :steps], nan=math.inf).amin(1)
    out[h] = {}
    for lo, hi in DELAY_BINS_S:
      m = (dec >= lo) & (dec < hi) & torch.isfinite(r)
      x = r[m]
      out[h][f"{lo:g}-{min(hi, 1.0):g} s"] = {
        "n": int(m.sum()),
        "q0.9": float(x.quantile(0.9)) if x.numel() else None,
        "q0.95": float(x.quantile(0.95)) if x.numel() else None,
        "abs_median": float(x.abs().median()) if x.numel() else None}
  return out


def arm_summary(dirs: list[str]) -> dict:
  seeds, cals = [], []
  for d in dirs:
    tube = json.load(open(os.path.join(d, "min_tube.json")))
    cal = calib_arrays(_load(os.path.join(d, "push.pt")))
    cals.append(cal)
    hz = {name: tube["horizons"][name.split()[0]] for name in H_STEPS}
    seeds.append({
      "dir": d, "episodes": tube["episodes"],
      "min_eps": {c: {h: tube["min_eps"][c][s] for h, s in H_STEPS.items()} for c in COVERAGES},
      "two_sided_eps": {c: {h: tube["two_sided_eps"][c][s] for h, s in H_STEPS.items()}
                        for c in COVERAGES},
      "test": {c: {h: {"coverage": hz[h]["levels"][c]["one_sided_coverage"],
                       "coverage_fell": hz[h]["levels"][c]["one_sided_coverage_by_group"].get(
                         "real fell within H"),
                       "fall_recall": hz[h]["levels"][c]["fall_recall"],
                       "false_alarm": hz[h]["levels"][c]["false_alarm"]}
                   for h in H_STEPS} for c in COVERAGES},
      "score_quantiles": {h: hz[h]["score_quantiles_calib"] for h in H_STEPS},
      "real_fell": {h: hz[h]["real_fell"] for h in H_STEPS},
    })
  acc = accuracy(cals)
  mean = {
    "min_eps": {c: {h: _mean([s["min_eps"][c][h] for s in seeds]) for h in H_STEPS}
                for c in COVERAGES},
    "two_sided_eps": {c: {h: _mean([s["two_sided_eps"][c][h] for s in seeds]) for h in H_STEPS}
                      for c in COVERAGES},
    "test": {c: {h: {k: _mean([s["test"][c][h][k] for s in seeds])
                     for k in ("coverage", "coverage_fell", "fall_recall", "false_alarm")}
                 for h in H_STEPS} for c in COVERAGES},
    "score_quantiles": {h: {q: _mean([s["score_quantiles"][h][q] for s in seeds])
                            for q in seeds[0]["score_quantiles"][h]} for h in H_STEPS},
  }
  accm = {name: {"g_mean": m["g"]["mean"], "g_p90": m["g"]["p90"], "up_share": m["up_share"],
                 "abs_r_median": acc["safe_value"]["marks"][name]["abs_median"],
                 "abs_r_p90": acc["safe_value"]["marks"][name]["abs_p90"],
                 "signed_r_median": acc["safe_value"]["marks"][name]["signed_p50"]}
          for name, m in acc["marks"].items()}
  return {"seeds": seeds, "mean": mean, "accuracy": accm, "by_delay": by_delay(cals),
          "_cals": cals}


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "DIR"),
                 help="an arm: its name, then one calibration dir per seed")
  p.add_argument("--out", default=None, help="json (default: print only)")
  a = p.parse_args(argv)
  arms = {}
  for spec in a.arm:
    if len(spec) < 2:
      raise SystemExit(f"--arm {spec}: a name and at least one dir")
    arms[spec[0]] = arm_summary(spec[1:])
  names = list(arms)
  n_seeds = {len(v["seeds"]) for v in arms.values()}
  # paired design: the same policy on the same episodes -> identical real margins
  paired = {}
  if len(n_seeds) == 1:
    for k in range(n_seeds.pop()):
      ref = arms[names[0]]["_cals"][k]["real_g"]
      paired[k] = {n: bool(arms[n]["_cals"][k]["real_g"].shape == ref.shape and torch.equal(
        torch.nan_to_num(arms[n]["_cals"][k]["real_g"], nan=-99.0),
        torch.nan_to_num(ref, nan=-99.0))) for n in names[1:]}
  for v in arms.values():
    v.pop("_cals")

  w = max(10, max(len(n) for n in names) + 2)
  print("[tubes] seeds:", {n: [os.path.basename(s["dir"].rstrip("/")) for s in v["seeds"]]
                           for n, v in arms.items()})
  print("[tubes] real margins identical to", names[0], "per seed (same policy, same episodes):",
        paired or "n/a (different seed counts)")
  print("\nTube width eps_H the filter uses (one-sided, min over H; mean over seeds):")
  print(f"  {'c':>5s} {'H':>6s} " + "".join(f"{n:>{w}s}" for n in names))
  for c in COVERAGES:
    for h in H_STEPS:
      print(f"  {c:>5s} {h:>6s} " + "".join(f"{arms[n]['mean']['min_eps'][c][h]:>{w}.3f}"
                                            for n in names))
  print("\n  per seed, c=0.95, 1 s: " + "; ".join(
    f"{n} " + " ".join(f"{s['min_eps']['0.95']['1 s']:.3f}" for s in arms[n]["seeds"])
    for n in names))
  print("\nTwo-sided half-width at c=0.95:")
  for h in H_STEPS:
    print(f"  {'':>5s} {h:>6s} " + "".join(f"{arms[n]['mean']['two_sided_eps']['0.95'][h]:>{w}.3f}"
                                          for n in names))
  print("\nSplit-half test at c=0.95 (coverage | coverage among real falls | fall recall | "
        "false alarm):")
  for h in H_STEPS:
    print(f"  {h:>6s} " + "  ".join(
      f"{n}: " + " ".join(f"{arms[n]['mean']['test']['0.95'][h][k]:.3f}"
                          for k in ("coverage", "coverage_fell", "fall_recall", "false_alarm"))
      for n in names))
  print("\nScore r = G^ - G on the calibration half, quantiles 0.5 / 0.9 / 0.95 / 0.99:")
  for h in H_STEPS:
    print(f"  {h:>6s} " + "  ".join(
      f"{n}: " + " ".join(f"{v:+.3f}" for v in arms[n]["mean"]["score_quantiles"][h].values())
      for n in names))
  print("\nWorld-model accuracy (pooled over seeds): |g^ - g| mean / p90 while up; median |r_H|; "
        "median signed r_H:")
  for mark in arms[names[0]]["accuracy"]:
    print(f"  {mark:>8s} " + "  ".join(
      f"{n}: {arms[n]['accuracy'][mark]['g_mean']:.3f}/{arms[n]['accuracy'][mark]['g_p90']:.3f} "
      f"|r| {arms[n]['accuracy'][mark]['abs_r_median']:.3f} r {arms[n]['accuracy'][mark]['signed_r_median']:+.3f}"
      for n in names))
  if all(arms[n]["by_delay"] for n in names):
    print("\nScore r_H by decision delay after the push (pooled over seeds): q0.95 (|r| median)")
    for h in ("0.4 s", "1 s"):
      for b in arms[names[0]]["by_delay"][h]:
        print(f"  {h:>6s} {b:>9s} " + "".join(
          f"{arms[n]['by_delay'][h][b]['q0.95']:>{w - 8}.3f} ({arms[n]['by_delay'][h][b]['abs_median']:.3f})"
          for n in names) + f"   n={arms[names[0]]['by_delay'][h][b]['n']}")
  if a.out:
    with open(a.out, "w") as f:
      json.dump({"arms": arms, "real_g_identical": paired}, f, indent=1)
    print(f"\n[tubes] wrote {a.out}")


if __name__ == "__main__":
  main()
