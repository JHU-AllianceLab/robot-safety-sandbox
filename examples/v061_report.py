"""Report of the v0.6.1 filter study (configs/sweeps/v061_study/eval.sh; the user, 2026-11-23).

Reads ``<root>/label/<model>_<cond>.episodes.pt`` and ``<root>/sweep/<model>_T<T0>.episodes.pt``
(examples/label_start_states.py) and writes ``<root>/{report.md, report.json, sweep.png}``.

Per model and condition, with sigma = the filter's own score of a start state (Beat-It: score0 =
min over 1 s of the world model's margin - its tube eps; SafetyPPO / lambda-Reach / RL-DH: value0 =
V(x0). RL-DH's hyperplane scores (score0 = best-action margin, score0_ref, score0_exec) do not rank start
states, episode AUC 0.50 / 0.65 / 0.50 against 0.89 for its critic V (eval 2047327), so its set is the
critic's) and failure = the rollout's termination within 3 s (a fall;
in the hot conditions also a winding at 70 C):
  * fallback saves       share of rollouts that do not fail (all start states)
  * uncalibrated set     Omega_0 = {sigma >= 0}: volume (share of rollouts in it) and soundness
                         (share of its rollouts that do not fail)
  * Wang calibrated set  Wang et al. Prop. 1 (as runs/walk_sailr/wang_calib/wang_calib.py): one random
                         rollout per bank row (draw 0; draws 1-4 give the range), the largest superlevel
                         set {sigma >= delta} with binom.cdf(falls in it, |set|, eps) <= beta = 1e-10, at
                         eps = 1 % and 5 %: volume (share of rows) and soundness (1 - its failure rate)

  python examples/v061_report.py --root runs/v061_study/eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from scipy.stats import binom

BETA = 1e-10
EPSS = (0.01, 0.05)
NAMES = {"beatit_thermal": "Beat-It (heat-aware)", "beatit_noheight": "Beat-It (heat-blind)",
         "safetyppo": "SafetyPPO", "lambdareach": "λ-Reach", "rldh": "RL-DH"}
CONDS = {"nt": "Temperature-agnostic world (fail = fall)",
         "hot30_60": "Soaked start T0 ~ U(30, 60) C (fail = fall or 70 C)",
         "hot60_65": "Soaked start T0 ~ U(60, 65) C (fail = fall or 70 C)"}


def strict(sig: torch.Tensor, fail: torch.Tensor, eps: float) -> dict:
  o = torch.argsort(sig, descending=True)
  s, f = sig[o], fail[o]
  k = torch.cumsum(f, 0).long()
  n = torch.arange(1, len(s) + 1)
  last = torch.ones_like(s, dtype=torch.bool)
  last[:-1] = s[1:] != s[:-1]
  ok = torch.tensor(binom.cdf(k.numpy(), n.numpy(), eps) <= BETA) & last
  if not ok.any():
    return {"delta": None, "volume": 0.0, "soundness": None, "n_in_set": 0}
  i = int(ok.nonzero().max())
  return {"delta": float(s[i]), "volume": (i + 1) / len(s), "soundness": 1 - float(k[i]) / (i + 1),
          "n_in_set": i + 1}


def one_per_row(row: torch.Tensor, draw: int) -> torch.Tensor:
  g = torch.Generator().manual_seed(draw)
  key = torch.rand(row.numel(), generator=g)
  o = torch.argsort(row.double() * 2 + key.double())
  first = torch.ones_like(row, dtype=torch.bool)
  first[1:] = row[o][1:] != row[o][:-1]
  return o[first]


def stats(path: str, model: str) -> dict:
  e = torch.load(path, map_location="cpu", weights_only=False)
  fld = "score0" if model.startswith("beatit") else "value0"
  sig, fell, row = e[fld].double(), e["fell"].double(), e["row"]
  oh = e["overheat"].double() if "overheat" in e else torch.zeros_like(fell)
  in0 = sig >= 0
  out = {"file": path, "score": fld, "episodes": int(row.numel()), "rows": int(row.unique().numel()),
         "fallback_saves": float(1 - fell.mean()), "fail_overheat": float(oh.mean()),
         "omega0_volume": float(in0.double().mean()),
         "omega0_soundness": float(1 - fell[in0].mean()) if in0.any() else None}
  for eps in EPSS:
    ds = []
    for d in range(5):
      idx = one_per_row(row, d)
      ds.append(strict(sig[idx], fell[idx], eps))
    out[f"wang_{eps:g}"] = ds[0]
    out[f"wang_{eps:g}_range"] = [min(x["volume"] for x in ds), max(x["volume"] for x in ds)]
  return out


def pct(x, nd=1):
  return "n/a" if x is None else f"{100 * x:.{nd}f} %"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--root", required=True)
  a = p.parse_args()
  res = {"label": {}, "sweep": {}}
  for f in sorted(glob.glob(os.path.join(a.root, "label", "*.episodes.pt"))):
    m, c = re.match(r"(.+)_(nt|hot30_60|hot60_65)\.episodes\.pt$", os.path.basename(f)).groups()
    res["label"].setdefault(c, {})[m] = stats(f, m)
  for f in sorted(glob.glob(os.path.join(a.root, "sweep", "*.episodes.pt"))):
    m, t = re.match(r"(.+)_T(\d+)\.episodes\.pt$", os.path.basename(f)).groups()
    res["sweep"].setdefault(m, {})[int(t)] = stats(f, m)
  json.dump(res, open(os.path.join(a.root, "report.json"), "w"), indent=1)

  md = ["# v0.6.1 filter study: soundness and calibrated safe-set volume", "",
        "Held-out SONIC hand-over bank on the v0.6 body (kicks |dv| up to 3.5 m/s), K rollouts per start "
        "state of each filter's own fallback, 3 s, no push. Uncalibrated set = {score >= 0}; Wang "
        "calibrated sets at eps 1 % and 5 % (beta 1e-10, one random rollout per state; range over 5 draws). "
        "Soundness = share of the set's rollouts that do not fail.", ""]
  for c, title in CONDS.items():
    if c not in res["label"]:
      continue
    md += [f"## {title}", "",
           "| filter | fallback saves | {score >= 0}: volume | soundness | Wang eps 5 %: volume | soundness "
           "| Wang eps 1 %: volume | soundness |", "|---|" + "---|" * 7]
    for m in NAMES:
      s = res["label"][c].get(m)
      if s is None:
        continue
      w5, w1 = s["wang_0.05"], s["wang_0.01"]
      r5, r1 = s["wang_0.05_range"], s["wang_0.01_range"]
      md.append(f"| {NAMES[m]} | {pct(s['fallback_saves'])} | {pct(s['omega0_volume'])} | "
                f"{pct(s['omega0_soundness'])} | {pct(w5['volume'])} ({100 * r5[0]:.0f}-{100 * r5[1]:.0f}) | "
                f"{pct(w5['soundness'], 2)} | {pct(w1['volume'])} ({100 * r1[0]:.0f}-{100 * r1[1]:.0f}) | "
                f"{pct(w1['soundness'], 2)} |")
    md.append("")
  if res["sweep"]:
    md += ["## Temperature sweep: heat-aware vs heat-blind Beat-It (soaked T0, fail = fall or 70 C)", "",
           "| T0 | " + " | ".join(f"{NAMES[m]}: saves / Wang 1 % / {{score>=0}} soundness"
                                  for m in ("beatit_thermal", "beatit_noheight") if m in res["sweep"]) + " |",
           "|---|" + "---|" * len(res["sweep"])]
    temps = sorted({t for v in res["sweep"].values() for t in v})
    for t in temps:
      cells = []
      for m in ("beatit_thermal", "beatit_noheight"):
        if m not in res["sweep"]:
          continue
        s = res["sweep"][m].get(t)
        cells.append("n/a" if s is None else f"{pct(s['fallback_saves'])} / {pct(s['wang_0.01']['volume'])} / "
                     f"{pct(s['omega0_soundness'])}")
      md.append(f"| {t} C | " + " | ".join(cells) + " |")
  open(os.path.join(a.root, "report.md"), "w").write("\n".join(md) + "\n")
  print("\n".join(md))

  if not res["sweep"]:
    return
  import matplotlib  # noqa: PLC0415
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: PLC0415

  col = {"beatit_thermal": "#2a6fdb", "beatit_noheight": "#e07b28"}
  fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
  for m, sw in res["sweep"].items():
    t = np.array(sorted(sw))
    axs[0].plot(t, [100 * sw[x]["wang_0.01"]["volume"] for x in t], color=col.get(m, "#555"), lw=2,
                marker="o", ms=5, label=f"{NAMES[m]}: Wang set (eps 1 %)")
    axs[0].plot(t, [100 * sw[x]["fallback_saves"] for x in t], color=col.get(m, "#555"), lw=1.2, ls="--",
                label=f"{NAMES[m]}: fallback saves")
    axs[1].plot(t, [100 * (sw[x]["omega0_soundness"] or np.nan) for x in t], color=col.get(m, "#555"), lw=2,
                marker="o", ms=5, label=NAMES[m])
  axs[0].set_ylabel("share of start states (%)")
  axs[0].set_title("Calibrated safe-set volume", fontsize=10, loc="left")
  axs[1].set_ylabel("soundness of {score >= 0} (%)")
  axs[1].set_title("Soundness of the uncalibrated set", fontsize=10, loc="left")
  for ax in axs:
    ax.set_xlabel("soaked start temperature T0 (C)")
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
      ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
  axs[0].set_ylim(0, 105)
  fig.tight_layout()
  fig.savefig(os.path.join(a.root, "sweep.png"), dpi=150)
  print(f"wrote {a.root}/report.md, report.json, sweep.png")


if __name__ == "__main__":
  main()
