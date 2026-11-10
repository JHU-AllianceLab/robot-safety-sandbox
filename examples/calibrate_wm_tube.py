"""Calibrate the world model's margin tube: eps_j so the real margin stays above the R rollouts' lowest prediction minus eps_j.

    python examples/calibrate_wm_tube.py \
        --calib runs/shac_vs_ppo_sonic/wmtube_calib/shac_s0/push.pt --tag shac_s0

Reads the ``wmcalib:<R>`` records of ``eval_sonic_qcbf.py`` (kicked held-out
episodes, sensor noise on, SONIC up to a random decision step k, then pi):
per episode the R noisy world-model rollouts' predicted margins of "u_SONIC at
k, then pi", ``[R, H]``, and the REAL margins of the same, ``[H]``. After a
real fall the real margin holds its terminal (anchored) value; a predicted fall
holds by construction (``Evaluator.wm_margins``).

Per step j, with m_j / M_j the lowest / highest of the R predictions:

  lower residual  e-_j = m_j - g_j      (> 0: the real margin is below every rollout's)
  upper residual  e+_j = g_j - M_j

ONE-SIDED tube (the filter's): eps_j(c) is the conformal c-quantile of e-_j
(rank ceil((n + 1) c) of n), so P(g_j >= m_j - eps_j) >= c. TWO-SIDED tube:
[m_j - eps-_j, M_j + eps+_j], each side at (1 + c) / 2, so P(inside) >= c.
Calibrated on half the episodes (even rows), evaluated on the other half (odd):
coverage per step and over whole horizons (every j <= H), the tube's width
and VOLUME (sum over j <= H of width x dt, in margin x seconds) per horizon,
and the filter-level check: does "min over j <= H of (m_j - eps_j) < 0" flag
the real falls within H (recall), and how often does it flag an episode that
did not fall (false alarm). The tables the filter reads (``lower_eps``) are
refit on all episodes.

Writes ``--out`` (json; default next to --calib, ``tube.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

COVERAGES = (0.9, 0.95, 0.99)
HORIZONS_S = (0.4, 1.0, 3.0)
MARKS = (1, 5, 10, 20, 35, 50, 75, 100, 150)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--calib", required=True, help="push.pt of a wmcalib:<R> run")
  p.add_argument("--tag", required=True)
  p.add_argument("--condition", default=None, help="default: the run's wmcalib:<R> condition")
  p.add_argument("--out", default=None)
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(os.path.dirname(a.calib), "tube.json")
  return a


def conformal_quantile(x: torch.Tensor, c: float) -> torch.Tensor:
  """Per column of ``x`` [n, H]: the ceil((n + 1) c)-th smallest (the last if past n)."""
  n = int(x.shape[0])
  k = min(n, int(math.ceil((n + 1) * c))) - 1
  return torch.sort(x, dim=0).values[k]


def forward_fill(g: torch.Tensor) -> torch.Tensor:
  """NaN after a real fall -> the last recorded value, per row."""
  out = g.clone()
  for j in range(1, out.shape[1]):
    out[:, j] = torch.where(torch.isnan(out[:, j]), out[:, j - 1], out[:, j])
  return out


def main(argv=None):
  a = _parse(argv)
  d = torch.load(a.calib, map_location="cpu", weights_only=False)
  conds = d["conditions"]
  name = a.condition or next(k for k in conds if k.startswith("wmcalib:"))
  rec = conds[name]
  pred, real = rec["calib_pred"].double(), rec["calib_real"].double()
  dt = float(d["meta"]["dt"])
  ok = torch.isfinite(pred[:, 0, 0]) & torch.isfinite(real[:, 0])
  pred, real = pred[ok], forward_fill(real[ok])
  n_all, R, H = int(pred.shape[0]), int(pred.shape[1]), int(pred.shape[2])
  m, M = pred.amin(1), pred.amax(1)                        # [n, H]
  e_lo, e_hi = m - real, real - M
  idx = torch.arange(n_all)
  A, B = idx % 2 == 0, idx % 2 == 1
  horizons = [int(round(h / dt)) for h in HORIZONS_S if round(h / dt) <= H]
  marks = [j for j in MARKS if j <= H]

  rep = {"tag": a.tag, "calib": os.path.abspath(a.calib), "condition": name, "rollouts": R,
         "steps": H, "dt": dt, "episodes": n_all, "calibration_half": int(A.sum()),
         "test_half": int(B.sum()),
         "real_fell_within": {f"{h * dt:g}": float((real[:, :h].amin(1) < 0).double().mean())
                              for h in horizons},
         "lower_eps": {}, "two_sided": {}, "test": {}}

  def evaluate(eps_lo: torch.Tensor, eps_lo2: torch.Tensor, eps_hi2: torch.Tensor, rows) -> dict:
    mm, MM, gg = m[rows], M[rows], real[rows]
    one = gg >= mm - eps_lo                                  # [n, H]
    two = (gg >= mm - eps_lo2) & (gg <= MM + eps_hi2)
    width = (MM - mm + eps_lo2 + eps_hi2).clamp_min(0.0)
    out = {"per_step": {}, "per_horizon": {}}
    for j in marks:
      out["per_step"][str(j)] = {
        "eps_lower": float(eps_lo[j - 1]), "one_sided": float(one[:, j - 1].double().mean()),
        "two_sided": float(two[:, j - 1].double().mean()),
        "width_median": float(width[:, j - 1].median()),
        "hull_width_median": float((MM - mm)[:, j - 1].median())}
    for h in horizons:
      fell = gg[:, :h].amin(1) < 0
      flag = (mm[:, :h] - eps_lo[:h]).amin(1) < 0
      flag0 = mm[:, :h].amin(1) < 0
      out["per_horizon"][f"{h * dt:g}"] = {
        "steps": h,
        "one_sided_every_step": float(one[:, :h].all(1).double().mean()),
        "two_sided_every_step": float(two[:, :h].all(1).double().mean()),
        "volume_median": float((width[:, :h].sum(1) * dt).median()),
        "hull_volume_median": float(((MM - mm)[:, :h].sum(1) * dt).median()),
        "real_fell": float(fell.double().mean()),
        "fall_recall": float(flag[fell].double().mean()) if fell.any() else None,
        "false_alarm": float(flag[~fell].double().mean()) if (~fell).any() else None,
        "fall_recall_uncalibrated": float(flag0[fell].double().mean()) if fell.any() else None,
        "false_alarm_uncalibrated": float(flag0[~fell].double().mean()) if (~fell).any() else None}
    return out

  # the uncalibrated hull (eps = 0) on the test half, for reference
  z = torch.zeros(H, dtype=torch.float64)
  rep["test"]["uncalibrated"] = evaluate(z, z, z, B)
  for c in COVERAGES:
    eps_lo = conformal_quantile(e_lo[A], c)
    c2 = (1.0 + c) / 2.0
    eps_lo2, eps_hi2 = conformal_quantile(e_lo[A], c2), conformal_quantile(e_hi[A], c2)
    rep["test"][f"{c:g}"] = evaluate(eps_lo, eps_lo2, eps_hi2, B)
    # the filter's tables: refit on every episode
    rep["lower_eps"][f"{c:g}"] = conformal_quantile(e_lo, c).tolist()
    rep["two_sided"][f"{c:g}"] = {"lower": conformal_quantile(e_lo, c2).tolist(),
                                  "upper": conformal_quantile(e_hi, c2).tolist()}

  print(f"[tube] {a.tag}: {n_all} calibration episodes ({R} rollouts, {H} steps); real falls "
        f"within " + ", ".join(f"{k} s {v:.3f}" for k, v in rep["real_fell_within"].items()))
  for key in ["uncalibrated"] + [f"{c:g}" for c in COVERAGES]:
    t = rep["test"][key]
    print(f"  coverage {key} (test half)")
    for j, row in t["per_step"].items():
      print(f"    j={j:>3s} eps {row['eps_lower']:+.3f}  P(real >= low - eps) {row['one_sided']:.3f}  "
            f"P(in two-sided) {row['two_sided']:.3f}  width {row['width_median']:.3f} "
            f"(hull {row['hull_width_median']:.3f})")
    for h, row in t["per_horizon"].items():
      print(f"    H={h:>3s}s every step: one-sided {row['one_sided_every_step']:.3f}, two-sided "
            f"{row['two_sided_every_step']:.3f}; volume {row['volume_median']:.3f} (hull "
            f"{row['hull_volume_median']:.3f}) margin*s; falls {row['real_fell']:.3f}: recall "
            f"{row['fall_recall']}, false alarm {row['false_alarm']}")
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[tube] wrote {a.out}")


if __name__ == "__main__":
  main()
