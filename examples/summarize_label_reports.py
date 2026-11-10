"""Compare policies on ONE start-state bank, from label_start_states.py outputs.

    python examples/summarize_label_reports.py \
        --group privcrit runs/shac_vs_ppo/eval/privcrit.pt \
        --group SHAC runs/shac_vs_ppo/eval/shac_s0.pt runs/shac_vs_ppo/eval/shac_s1.pt \
        --group PPO runs/shac_vs_ppo/eval/ppo_s*.pt \
        --out runs/shac_vs_ppo/summary_table.json

Each file is one policy's labeled copy of the SAME bank (label_start_states.py
``--out``), with its ``.report.json`` beside it; a GROUP is one learner, its
files its seeds. Per group the table shows the mean +- std over the group's
files of:

* from the report (episode level): survival, the AUC of V(s0) against the
  outcome, the certificate read-out (share of starts with V(s0) >= 0, its
  precision P(survive | V >= 0) and recall P(V >= 0 | survive)), the median
  time to fall, and the policy's action rate / magnitude / saturation;
* from the labeled bank (row level, rows weighted by their episode count):
  survival by push severity (quartiles of [0.2, 1] x force_max) and by the
  handover delay (push onset -> snapshot), for banks that carry those columns.

A missing file (an unfinished seed, an unmatched glob) is skipped with a
notice, so the table can be refreshed while runs are still going. Files of one
table must label the same bank: a mismatch is reported, not silently mixed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

from robot_safety_sandbox.envs.g1_recover.states import load_bank  # noqa: E402

#: report keys shown, in order: key -> (label, format)
REPORT_KEYS = {
  "survival": ("survival", ".3f"),
  "auc_value_episodes": ("AUC of V(s0) vs outcome", ".3f"),
  "cert_frac": ("certified V(s0)>=0", ".3f"),
  "cert_precision": ("cert precision", ".3f"),
  "cert_recall": ("cert recall", ".3f"),
  "ttf_q50": ("time to fall q50 (s)", ".2f"),
  "action_rate": ("action rate", ".4f"),
  "action_mag": ("action magnitude", ".2f"),
  "action_sat_frac": ("action at box", ".4f"),
}
SEVERITY_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0 + 1e-6)


def _row_breakdown(bank: dict) -> dict[str, float]:
  """Survival by severity quartile and by handover delay (row level)."""
  out: dict[str, float] = {}
  w = bank["n_label"].float()
  lab = w > 0
  surv = torch.nan_to_num(bank["p_surv"].float()) * w

  def rate(mask):
    m = mask & lab
    return float(surv[m].sum() / w[m].sum()) if bool(m.any()) else float("nan")

  sev = bank.get("push_severity")
  if sev is not None and bool(torch.isfinite(sev[lab]).any()):
    for a, b in zip(SEVERITY_EDGES[:-1], SEVERITY_EDGES[1:]):
      out[f"severity {a:.1f}-{min(b, 1.0):.1f}"] = rate((sev >= a) & (sev < b))
  delay = bank.get("handover_delay_s")
  if delay is not None and bool(torch.isfinite(delay[lab]).any()):
    d = torch.round(delay * 100) / 100
    for v in sorted(set(d[lab & torch.isfinite(d)].tolist())):
      out[f"delay {v:.2f} s"] = rate(d == v)
  return out


def _load(path: str):
  rep_path = os.path.splitext(path)[0] + ".report.json"
  if not (os.path.isfile(path) and os.path.isfile(rep_path)):
    return None
  with open(rep_path) as f:
    rep = json.load(f)
  q = rep.get("time_to_fall_q10_q50_q90")
  rep["ttf_q50"] = q[1] if q else float("nan")
  bank = load_bank(path, "cpu")
  label = (bank.get("meta") or {}).get("label", {})
  return rep, _row_breakdown(bank), tuple(label.get("banks", ()))


def _stat(vals: list[float]) -> str:
  vals = [v for v in vals if v == v]
  if not vals:
    return "-"
  m = sum(vals) / len(vals)
  if len(vals) == 1:
    return f"{m:.3f}"
  sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
  return f"{m:.3f} +- {sd:.3f}"


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--group", nargs="+", action="append", required=True,
                 metavar=("NAME", "FILE"),
                 help="a learner name and its labeled bank files (one per seed)")
  p.add_argument("--out", default=None, help="also write the table as JSON")
  args = p.parse_args(argv)

  groups: dict[str, list] = {}
  banks_seen: dict[tuple, list[str]] = {}
  for g in args.group:
    name, files = g[0], g[1:]
    rows = []
    for f in files:
      r = _load(f)
      if r is None:
        print(f"[summary] skip {f}: no labeled bank / report (yet)")
        continue
      rows.append(r)
      banks_seen.setdefault(r[2], []).append(f)
    groups[name] = rows
  if len(banks_seen) > 1:
    print("[summary] WARNING: the files label DIFFERENT banks:")
    for b, fs in banks_seen.items():
      print(f"    {list(b)}: {fs}")

  table: dict[str, dict[str, str]] = {}
  for key, (label, _fmt) in REPORT_KEYS.items():
    table[label] = {n: _stat([float(r[0].get(key, float("nan"))) for r in rows])
                    for n, rows in groups.items()}
  extra = []
  for rows in groups.values():
    for r in rows:
      extra += [k for k in r[1] if k not in extra]
  for k in extra:
    table[f"survival, {k}"] = {n: _stat([r[1].get(k, float("nan")) for r in rows])
                               for n, rows in groups.items()}

  names = list(groups)
  w0 = max(len(k) for k in table) + 2
  w = max(18, *(len(n) + 6 for n in names))
  print(f"{'':{w0}}" + "".join(f"{n} (n={len(groups[n])})".rjust(w) for n in names))
  for k, row in table.items():
    print(f"{k:{w0}}" + "".join(row[n].rjust(w) for n in names))
  if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
      json.dump({"groups": {n: [len(groups[n])] for n in names}, "table": table},
                f, indent=2)
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
  main()
