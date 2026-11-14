"""Conformal tube on the WORST margin: eps_H so the real min margin over H stays above the predicted one minus eps_H.

    python examples/calibrate_wm_min_tube.py \
        --calib runs/shac_vs_ppo_sonic/wmmin_calib/shac_s0/push.pt --tag shac_s0

Reads the ``wmcalib:clean`` records of ``eval_sonic_qcbf.py`` (kicked held-out
episodes, SONIC up to a random decision step k, then pi; one NOISE-FREE
world-model rollout of "u_SONIC at k, then pi" and the real margins of the
same, 3 s each). For a horizon H (0.1 / 0.4 / 1 / 3 s) and each episode:

  predicted worst margin  G^ = min over j <= H of the rollout's predicted g_j
  real worst margin       G  = min over j <= H of the real g_j (a fall holds its value)
  score                   r  = G^ - G      (> 0: reality was worse than predicted)

Split conformal prediction: on the calibration half (even rows), eps_H(c) is
the ceil((n + 1) c)-th smallest score, so for a new episode drawn like these,
P(G >= G^ - eps_H) >= c (one-sided: the filter's bound) -- and with |r| the
two-sided interval [G^ - eps, G^ + eps], P(|G - G^| <= eps) >= c.

On the test half (odd rows): the coverage of both, the two-sided width 2 eps,
the coverage split by whether the real robot fell within H, by kick size and
by the predicted worst margin, and the filter-level check of "G^ - eps < 0":
recall of the real falls within H (G < 0) and the false-alarm rate. The
tables the filter reads (``min_eps``: {c: {H steps: eps}}) are refit on all
episodes.

Writes ``--out`` (json; default ``min_tube.json`` next to --calib).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

COVERAGES = (0.9, 0.95, 0.99)
HORIZONS_S = (0.1, 0.4, 1.0, 3.0)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--calib", required=True, help="push.pt of a wmcalib:clean run")
  p.add_argument("--tag", required=True)
  p.add_argument("--condition", default="wmcalib:clean")
  p.add_argument("--out", default=None)
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(os.path.dirname(a.calib), "min_tube.json")
  return a


def conformal_quantile(x: torch.Tensor, c: float) -> float:
  n = int(x.numel())
  k = min(n, int(math.ceil((n + 1) * c))) - 1
  return float(torch.sort(x).values[k])


def forward_fill(g: torch.Tensor) -> torch.Tensor:
  out = g.clone()
  for j in range(1, out.shape[1]):
    out[:, j] = torch.where(torch.isnan(out[:, j]), out[:, j - 1], out[:, j])
  return out


def main(argv=None):
  a = _parse(argv)
  d = torch.load(a.calib, map_location="cpu", weights_only=False)
  rec = d["conditions"][a.condition]
  dt = float(d["meta"]["dt"])
  pred = rec["calib_pred"].double()[:, 0]                 # [E, H] (one rollout)
  real = rec["calib_real"].double()
  dv = d["episodes"]["dv"].double()
  ok = torch.isfinite(pred[:, 0]) & torch.isfinite(real[:, 0])
  pred, real, dv = pred[ok], forward_fill(real[ok]), dv[ok]
  n_all, Hmax = int(pred.shape[0]), int(pred.shape[1])
  idx = torch.arange(n_all)
  A, B = idx % 2 == 0, idx % 2 == 1
  # --wm-thermal of the calibration run (the eval refuses a table of the other mode)
  wm_thermal = ((d["meta"].get("wmrollout") or {}).get("thermal") or {}).get("mode", "nn")
  rep = {"tag": a.tag, "calib": os.path.abspath(a.calib), "condition": a.condition, "dt": dt,
         "wm_thermal": wm_thermal,
         "episodes": n_all, "calibration_half": int(A.sum()), "test_half": int(B.sum()),
         "min_eps": {f"{c:g}": {} for c in COVERAGES},
         "two_sided_eps": {f"{c:g}": {} for c in COVERAGES}, "horizons": {}}
  if "calib_pred_tw" in rec:
    # thermal tasks: the predicted hottest winding vs the real one (live steps only)
    tp = rec["calib_pred_tw"].double()[:, 0][ok]
    tr = rec["calib_real_tw"].double()[ok]
    temp = {}
    for hs in HORIZONS_S:
      H = int(round(hs / dt))
      if H > Hmax:
        continue
      e = tp[:, H - 1] - tr[:, H - 1]
      e = e[torch.isfinite(e)]
      hot = torch.isfinite(tr[:, H - 1]) & (tr[:, H - 1] >= 60.0)
      eh = (tp[:, H - 1] - tr[:, H - 1])[hot]
      temp[f"{hs:g}"] = {
        "steps": H, "n": int(e.numel()),
        "err_C_mean": float(e.mean()) if e.numel() else None,
        "abs_err_C_median": float(e.abs().median()) if e.numel() else None,
        "abs_err_C_p95": float(torch.quantile(e.abs(), 0.95)) if e.numel() else None,
        "n_real_at_least_60C": int(hot.sum()),
        "err_C_mean_real_at_least_60C": float(eh.mean()) if eh.numel() else None,
        "abs_err_C_p95_real_at_least_60C": (float(torch.quantile(eh.abs(), 0.95))
                                            if eh.numel() else None)}
    rep["hottest_winding_error"] = temp
    print(f"[min-tube] {a.tag} ({wm_thermal}): predicted - real hottest winding, C")
    for h, row in temp.items():
      print(f"  H={h:>3s}s: n {row['n']}, mean {row['err_C_mean']}, |.| median "
            f"{row['abs_err_C_median']}, p95 {row['abs_err_C_p95']}; real >= 60 C: "
            f"n {row['n_real_at_least_60C']}, mean {row['err_C_mean_real_at_least_60C']}")

  for hs in HORIZONS_S:
    H = int(round(hs / dt))
    if H > Hmax:
      continue
    Gp, Gr = pred[:, :H].amin(1), real[:, :H].amin(1)
    r = Gp - Gr
    fell = Gr < 0
    row = {"steps": H, "real_fell": float(fell[B].double().mean()),
           "score_quantiles_calib": {q: float(torch.quantile(r[A], q)) for q in (0.5, 0.9, 0.95, 0.99)},
           "uncalibrated": {}, "levels": {}}
    flag0 = Gp < 0
    row["uncalibrated"] = {
      "one_sided_coverage": float((Gr[B] >= Gp[B]).double().mean()),
      "fall_recall": float(flag0[B & fell].double().mean()) if (B & fell).any() else None,
      "false_alarm": float(flag0[B & ~fell].double().mean()) if (B & ~fell).any() else None}
    for c in COVERAGES:
      eps = conformal_quantile(r[A], c)
      eps2 = conformal_quantile(r[A].abs(), c)
      one = Gr >= Gp - eps
      two = (Gr - Gp).abs() <= eps2
      flag = Gp - eps < 0
      groups = {"real fell within H": B & fell, "real did not fall": B & ~fell,
                "kick 0-2.5 m/s": B & (dv < 2.5), "kick 2.5-5 m/s": B & (dv >= 2.5),
                "predicted worst < 0.5": B & (Gp < 0.5), "predicted worst >= 0.5": B & (Gp >= 0.5)}
      row["levels"][f"{c:g}"] = {
        "eps_one_sided": eps, "eps_two_sided": eps2, "two_sided_width": 2.0 * eps2,
        "one_sided_coverage": float(one[B].double().mean()),
        "two_sided_coverage": float(two[B].double().mean()),
        "one_sided_coverage_by_group": {k: (float(one[m].double().mean()) if m.any() else None)
                                        for k, m in groups.items()},
        "group_sizes": {k: int(m.sum()) for k, m in groups.items()},
        "fall_recall": float(flag[B & fell].double().mean()) if (B & fell).any() else None,
        "false_alarm": float(flag[B & ~fell].double().mean()) if (B & ~fell).any() else None}
      rep["min_eps"][f"{c:g}"][str(H)] = conformal_quantile(r, c)       # refit on all episodes
      rep["two_sided_eps"][f"{c:g}"][str(H)] = conformal_quantile(r.abs(), c)
    rep["horizons"][f"{hs:g}"] = row

  print(f"[min-tube] {a.tag}: {n_all} calibration episodes (one noise-free rollout each)")
  for h, row in rep["horizons"].items():
    u = row["uncalibrated"]
    print(f"  H={h:>3s}s ({row['steps']} steps): real falls {row['real_fell']:.3f}; score median "
          f"{row['score_quantiles_calib'][0.5]:+.3f}; eps=0: coverage {u['one_sided_coverage']:.3f}, "
          f"recall {u['fall_recall']}, false alarm {u['false_alarm']}")
    for c, lv in row["levels"].items():
      g = lv["one_sided_coverage_by_group"]
      print(f"    c={c:<4s} eps {lv['eps_one_sided']:+.3f} -> coverage {lv['one_sided_coverage']:.3f} "
            f"(two-sided width {lv['two_sided_width']:.3f}: {lv['two_sided_coverage']:.3f}); fell "
            f"{g['real fell within H']}, not {g['real did not fall']}; recall {lv['fall_recall']}, "
            f"false alarm {lv['false_alarm']}")
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[min-tube] wrote {a.out}")


if __name__ == "__main__":
  main()
