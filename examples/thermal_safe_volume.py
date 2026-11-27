"""Safe-set volume of the thermal Beat-It vs the start temperature (the user, 2026-11-23).

Reads the per-episode files of ``examples/label_start_states.py --thermal-start-c T0`` (one per
T0; Beat-It v6 hybrid s1 in its own world g1_recover_thermal70_v6: failure = a fall or a winding
at 70 C within the 3 s episode) and writes ``<dir>/{report.json, report.md, volume_vs_T0.png}``.

Per T0, Wang et al. (2025) Prop. 1 as in ``runs/walk_sailr/wang_calib/wang_calib.py`` (the
temperature-agnostic reference): eps 1e-2, beta 1e-10, ONE random rollout per bank row (draw 0
pre-registered, the others reported), the largest superlevel set S_delta = {sigma >= delta} with
binom.cdf(falls in S, |S|, eps) <= beta. sigma = score0 = min_{j<50} ghat_j - eps_tube (1 s,
the fallback's own first action, natively hybrid world model), volume = |S| / rows.
Also the volume of the SAILR rule (min ghat >= 0.488, i.e. sigma >= 0.488 - eps_tube) and
whether that set passes the same test, and the share of start states the fallback saves.

  python examples/thermal_safe_volume.py --dir runs/thermal_volume/<job>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import binom

EPS, BETA = 1e-2, 1e-10
REF = {"label": "old non-thermal Beat-It on the old robot (shac_s0; another model)", "volume": 0.8135975915704967,
       "delta": 0.48823344707489014}


def strict(sig: torch.Tensor, fail: torch.Tensor, eps: float, beta: float) -> dict:
  """wang_calib.py's procedure; an empty set (no level passes) is volume 0."""
  o = torch.argsort(sig, descending=True)
  s, f = sig[o], fail[o]
  k = torch.cumsum(f, 0).long()
  n = torch.arange(1, len(s) + 1)
  last = torch.ones_like(s, dtype=torch.bool)
  last[:-1] = s[1:] != s[:-1]                       # ties: whole level sets
  ok = torch.tensor(binom.cdf(k.numpy(), n.numpy(), eps) <= beta) & last
  if not ok.any():
    return {"delta": None, "n_in_set": 0, "falls_in_set": 0, "fail_in_set": None, "volume": 0.0,
            "n_rows": len(s), "binom_cdf": None}
  i = int(ok.nonzero().max())
  return {"delta": float(s[i]), "n_in_set": i + 1, "falls_in_set": int(k[i]),
          "fail_in_set": float(k[i]) / (i + 1), "volume": (i + 1) / len(s), "n_rows": len(s),
          "binom_cdf": float(binom.cdf(int(k[i]), i + 1, eps))}


def one_per_row(row: torch.Tensor, draw: int) -> torch.Tensor:
  g = torch.Generator().manual_seed(draw)
  key = torch.rand(row.numel(), generator=g)
  o = torch.argsort(row.double() * 2 + key.double())
  first = torch.ones_like(row, dtype=torch.bool)
  first[1:] = row[o][1:] != row[o][:-1]
  return o[first]


def level(path: str, sailr: float) -> dict:
  e = torch.load(path, map_location="cpu", weights_only=False)
  t0 = e["label"].get("thermal_start_c")
  eps_tube = float(e["settings"]["eps"])
  row, sig = e["row"], e["score0"].double()
  fell = e["fell"].double()
  oh = e["overheat"].double() if "overheat" in e else torch.zeros_like(fell)
  peak = e["tw_peak"].double() if "tw_peak" in e else torch.full_like(fell, float("nan"))
  k = int(e["label"]["k"])
  draws = []
  for d in range(5):                                # wang_calib.py: draws 0-4
    idx = one_per_row(row, d)
    r = strict(sig[idx], fell[idx], EPS, BETA)
    r["draw"] = d
    draws.append(r)
  idx = one_per_row(row, 0)
  s0, f0 = sig[idx], fell[idx]
  m = s0 >= sailr - eps_tube
  nm, km = int(m.sum()), int(f0[m].sum())
  return {
    "T0": t0, "file": os.path.abspath(path), "rows": int(idx.numel()), "episodes": int(row.numel()),
    "k": k, "bank": e["label"]["banks"], "eps_tube": eps_tube, "wang": draws[0], "wang_draws": draws,
    "sailr_rule": {"min_ghat_ge": sailr, "volume": nm / idx.numel(), "n_in_set": nm,
                   "falls_in_set": km, "fail_in_set": km / nm if nm else None,
                   "binom_cdf": float(binom.cdf(km, nm, EPS)) if nm else None,
                   "passes_wang_test": bool(nm and binom.cdf(km, nm, EPS) <= BETA)},
    # all rollouts
    "fallback_saves": float(1 - fell.mean()), "fail_fall": float((fell * (1 - oh)).mean()),
    "fail_overheat": float(oh.mean()), "tw_peak_mean": float(np.nanmean(peak.numpy())),
    "tw_peak_p95": float(np.nanpercentile(peak.numpy(), 95)),
    "min_ghat_median": float((sig + eps_tube).median()),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--dir", required=True)
  p.add_argument("--sailr-threshold", type=float, default=0.488)
  a = p.parse_args()
  files = sorted(glob.glob(os.path.join(a.dir, "T*.episodes.pt")))
  if not files:
    raise SystemExit(f"no T*.episodes.pt in {a.dir}")
  res = sorted((level(f, a.sailr_threshold) for f in files), key=lambda r: r["T0"])
  out = {"eps": EPS, "beta": BETA, "reference": REF, "sailr_threshold": a.sailr_threshold,
         "levels": res, "procedure": __doc__}
  json.dump(out, open(os.path.join(a.dir, "report.json"), "w"), indent=1)

  def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"

  md = ["# Thermal Beat-It: safe-set volume vs start temperature", "",
        "Beat-It v6 hybrid s1 in its own world (failure = a fall or a winding at 70 C within 3 s). "
        "Every motor's winding, housing and structure start at T0. Start states: "
        f"{res[0]['rows']:,} rows of {', '.join(os.path.basename(b) for b in res[0]['bank'])} (no push). Wang et al.: eps 1e-2, beta 1e-10, one "
        f"random rollout per row (draw 0). Reference: {REF['label']} volume {100 * REF['volume']:.1f} %.",
        "", "| T0 | Wang volume | delta* (min ghat) | fail in set | volume at min ghat >= "
        f"{a.sailr_threshold} | its fail rate | passes test | fallback saves | fails: fall / overheat "
        "| peak winding mean / p95 |", "|---|" + "---|" * 9]
  for r in res:
    w, s = r["wang"], r["sailr_rule"]
    d = "none" if w["delta"] is None else f"{w['delta'] + r['eps_tube']:.3f}"
    md.append(f"| {r['T0']:g} C | {pct(w['volume'])} | {d} | {pct(w['fail_in_set'])} | "
              f"{pct(s['volume'])} | {pct(s['fail_in_set'])} | {'yes' if s['passes_wang_test'] else 'no'} | "
              f"{pct(r['fallback_saves'])} | {pct(r['fail_fall'])} / {pct(r['fail_overheat'])} | "
              f"{r['tw_peak_mean']:.1f} / {r['tw_peak_p95']:.1f} C |")
  md += ["", "Other Wang draws (volume): " + "; ".join(
    f"{r['T0']:g} C " + ", ".join(pct(d["volume"]) for d in r["wang_draws"]) for r in res)]
  open(os.path.join(a.dir, "report.md"), "w").write("\n".join(md) + "\n")
  print("\n".join(md))

  import matplotlib  # noqa: PLC0415
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: PLC0415

  T = np.array([r["T0"] for r in res], dtype=float)
  ser = [("Wang-calibrated safe set", [r["wang"]["volume"] for r in res], "#2a6fdb", "o"),
         (f"Beat-It's set at SAILR threshold {a.sailr_threshold}", [r["sailr_rule"]["volume"] for r in res],
          "#e07b28", "s"),
         ("starts the fallback actually saves", [r["fallback_saves"] for r in res], "#1b9e77", "^")]
  fig, ax = plt.subplots(figsize=(7.2, 4.2))
  for lab, y, c, mk in ser:
    ax.plot(T, 100 * np.array(y), color=c, lw=2, marker=mk, ms=6, label=lab)
  ax.axhline(100 * REF["volume"], color="#7a7a7a", lw=1.2, ls="--")
  # the reference label on whichever side of its line the left-hand data leaves free
  left = [100 * v for _, y, _, _ in ser for t, v in zip(T, y) if t <= T.min() + 20]
  above = max(left, default=0) <= 100 * REF["volume"]
  ax.text(T.min(), 100 * REF["volume"] + (1.5 if above else -1.5), f"old non-thermal Beat-It, old robot: {100 * REF['volume']:.1f} % (another model)",
          fontsize=8, color="#555555", va="bottom" if above else "top")
  ax.set_xlabel("start temperature of every motor, T0 (C)")
  ax.set_ylabel("share of start states (%)")
  ax.set_ylim(0, 105)
  ax.set_title("Thermal Beat-It: safe-set volume vs start temperature (fail = fall or 70 C)",
               fontsize=10, loc="left")
  ax.grid(alpha=0.25, lw=0.6)
  for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
  ax.legend(fontsize=8, frameon=False, loc="lower left")
  fig.tight_layout()
  fig.savefig(os.path.join(a.dir, "volume_vs_T0.png"), dpi=150)
  print(f"wrote {a.dir}/report.md, report.json, volume_vs_T0.png")


if __name__ == "__main__":
  main()
