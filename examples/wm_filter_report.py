"""Every number of the WM-filter report, from the report run (configs/sweeps/shac_vs_ppo_sonic/wmreport_jobs.txt).

    python examples/wm_filter_report.py --root runs/shac_vs_ppo_sonic/wmreport \
        --out runs/shac_vs_ppo_sonic/wmreport/report.json [--pick-videos 5]

Reads, per SHAC seed (shac_s0..2), ``<root>/calib/<tag>/push.pt`` (the
``wmcalib:clean`` calibration episodes), ``<root>/calib/<tag>/min_tube.json``
(calibrate_wm_min_tube.py) and ``<root>/eval/<tag>/push.pt`` (the filters).
Writes one json with:

``accuracy``  the world model's one noise-free rollout of "u_SONIC at the
    decision step, then SHAC on the predicted state" against the real same,
    from the calibration episodes. Per step j after the decision (1 .. 150, one
    step = 0.02 s) and per quantity -- the margin g, the pelvis height (m) and
    the tilt from vertical (deg), the margin's height and tilt inputs -- the
    mean / median / 90th percentile of |predicted - real| over the episodes
    whose real robot is still up at step j (``curves``, and ``marks`` at 1, 10,
    20 = 0.4 s, 50 = 1 s, 150 = 3 s). And the SAFE-VALUE difference: for each
    horizon H the worst margin over the horizon, r_H = min_{j <= H} g^_j -
    min_{j <= H} g_j (a real fall holds its last value), its quantiles and
    those of |r_H|, over every calibration episode.
``conformal`` per seed and level c the tube offset eps_H(c) the filters use
    (refit on every calibration episode), the split-half check (eps on the even
    rows, coverage P(real worst >= predicted worst - eps) on the odd ones,
    overall and among the episodes that really fell within H), and the 1 s
    score distribution (histogram of r_50 pooled over the seeds).
``survival``  per filter and kick bin (0-1, 1-2, 2-3, 3-4 m/s), pooled over the
    seeds: the survival rate (Wilson 95 % interval), the share of post-kick
    control steps SHAC acted on, the hand-overs per episode, the episode count;
    and each seed's overall survival rate.
``videos`` (with --pick-videos N) candidate episodes of the seed-0 evaluation
    for the filter videos: SONIC falls and both of the two best WM tube
    filters survive, and SONIC falls with a kick in 0-1 / 1-2 / 2-3 m/s.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

TAGS = ("shac_s0", "shac_s1", "shac_s2")
MARKS = {"1 step": 1, "10 steps": 10, "0.4 s": 20, "1 s": 50, "3 s": 150}
QUANTITIES = ("g", "pelvis_height_m", "tilt_deg")
TUBE_H = {"0.4 s": "20", "1 s": "50", "3 s": "150"}
LEVELS = ("0.9", "0.95", "0.99")
KICK_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0)
#: the report's filters, in table order
FILTERS = [("SONIC alone", "sonic"), ("SHAC alone", "fallback"),
           ("exact-sim filter 1 s", "rollout:1"),
           ("WM tube 0.4 s/95%", "wmmin:0.4:0.95"), ("WM tube 1 s/95%", "wmmin:1:0.95"),
           ("WM tube 3 s/95%", "wmmin:3:0.95"),
           ("WM tube 1 s/95%, latch 1 s", "wmmin:1:0.95+dwell1"),
           ("WM tube 1 s/95%, latch 2 s", "wmmin:1:0.95+dwell2"),
           ("WM tube 1 s/95%, latch forever", "wmmin:1:0.95+latch")]
WM_FILTERS = [n for _, n in FILTERS if n.startswith("wmmin")]


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--root", default="runs/shac_vs_ppo_sonic/wmreport")
  p.add_argument("--out", default=None, help="default <root>/report.json")
  p.add_argument("--tags", nargs="+", default=list(TAGS))
  p.add_argument("--pick-videos", type=int, default=0, metavar="N",
                 help="also list N candidate episodes per video category (seed-0 evaluation)")
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(a.root, "report.json")
  return a


def _load(path):
  return torch.load(path, map_location="cpu", weights_only=False)


def forward_fill(g: torch.Tensor) -> torch.Tensor:
  out = g.clone()
  for j in range(1, out.shape[1]):
    out[:, j] = torch.where(torch.isnan(out[:, j]), out[:, j - 1], out[:, j])
  return out


def wilson(k: int, n: int, z: float = 1.96):
  if n == 0:
    return (math.nan, math.nan)
  p = k / n
  den = 1 + z * z / n
  mid = (p + z * z / (2 * n)) / den
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (mid - half, mid + half)


def _q(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x, q)) if x.numel() else math.nan


# --- accuracy and the conformal scores, from the calibration runs --------------------------

def calib_arrays(d: dict) -> dict:
  rec = d["conditions"]["wmcalib:clean"]
  pred = rec["calib_pred"].double()[:, 0]                 # [E, H]
  real = rec["calib_real"].double()                       # [E, H], NaN after a fall
  ok = torch.isfinite(pred[:, 0]) & torch.isfinite(real[:, 0])
  out = {"pred_g": pred[ok], "real_g": real[ok], "dt": float(d["meta"]["dt"]),
         "dv": d["episodes"]["dv"].double()[ok], "category": d["episodes"]["category"][ok],
         "n_recorded": int(ok.sum()), "n_episodes": int(ok.numel())}
  if "calib_decision" in rec:                              # decision - onset, steps
    out["decision"] = rec["calib_decision"].long()[ok]
  if "calib_pred_states" in rec:
    out["pred_st"] = rec["calib_pred_states"].double()[:, 0][ok]    # [E, H, 2]
    out["real_st"] = rec["calib_real_states"].double()[ok]          # [E, H, 2]
  return out


def accuracy(cal: list[dict]) -> dict:
  pred_g = torch.cat([c["pred_g"] for c in cal])
  real_g = torch.cat([c["real_g"] for c in cal])
  have_st = all("pred_st" in c for c in cal)
  H = int(pred_g.shape[1])
  # the real robot is up at step j: its state was recorded (NaN from the fall step on)
  if have_st:
    pred_st = torch.cat([c["pred_st"] for c in cal])
    real_st = torch.cat([c["real_st"] for c in cal])
    up = torch.isfinite(real_st[..., 0])
  else:
    up = torch.isfinite(real_g)
  err = {"g": (pred_g - real_g).abs()}
  if have_st:
    err["pelvis_height_m"] = (pred_st[..., 0] - real_st[..., 0]).abs()
    err["tilt_deg"] = (pred_st[..., 1] - real_st[..., 1]).abs()
  curves = {k: {"mean": [], "median": [], "p90": [], "n": []} for k in err}
  for j in range(H):
    m = up[:, j]
    for k, e in err.items():
      x = e[m, j]
      x = x[torch.isfinite(x)]
      curves[k]["mean"].append(float(x.mean()) if x.numel() else math.nan)
      curves[k]["median"].append(_q(x, 0.5))
      curves[k]["p90"].append(_q(x, 0.9))
      curves[k]["n"].append(int(x.numel()))
  marks = {name: {k: {s: curves[k][s][j - 1] for s in ("mean", "median", "p90", "n")}
                  for k in err} for name, j in MARKS.items() if j <= H}
  for name, j in MARKS.items():
    if j <= H:
      marks[name]["up_share"] = float(up[:, j - 1].double().mean())
  if have_st:
    # the model's margin head against the margin of its own decoded state (height and tilt
    # terms; the contact term is not in the state) -- and the same for the real robot
    c50 = math.cos(math.radians(50.0))

    def state_margin(st):
      ht = (st[..., 0] - 0.45) / 0.25
      tt = (torch.cos(torch.deg2rad(st[..., 1])) - c50) / (1.0 - c50)
      return torch.minimum(ht, tt).clamp(-3.0, 3.0)

    gs_pred, gs_real = state_margin(pred_st), state_margin(real_st)
    for name, j in MARKS.items():
      if j <= H:
        m = up[:, j - 1] & torch.isfinite(gs_real[:, j - 1])
        marks[name]["g_head_vs_own_state"] = float((pred_g[m, j - 1] - gs_pred[m, j - 1]).abs().mean())
        marks[name]["state_margin_err"] = float((gs_pred[m, j - 1] - gs_real[m, j - 1]).abs().mean())
        marks[name]["real_g_vs_real_state"] = float((real_g[m, j - 1] - gs_real[m, j - 1]).abs().mean())
    g_head_gap = {}
    for j in list(range(1, 11)) + [20, 50]:
      if j <= H:
        m = up[:, j - 1] & torch.isfinite(gs_real[:, j - 1])
        g_head_gap[str(j)] = float((pred_g[m, j - 1] - gs_pred[m, j - 1]).abs().mean())
  # the safe value: the worst margin over the horizon
  realf = forward_fill(real_g)
  gp, gr = torch.cummin(pred_g, 1).values, torch.cummin(realf, 1).values
  r = gp - gr                                             # [E, H], > 0: reality was worse
  sv = {"signed": {q: [] for q in ("p05", "p25", "p50", "p75", "p95")},
        "abs": {"mean": [], "median": [], "p90": []}}
  for j in range(H):
    x = r[:, j]
    for q, v in (("p05", 0.05), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p95", 0.95)):
      sv["signed"][q].append(_q(x, v))
    ax = x.abs()
    sv["abs"]["mean"].append(float(ax.mean()))
    sv["abs"]["median"].append(_q(ax, 0.5))
    sv["abs"]["p90"].append(_q(ax, 0.9))
  sv_marks = {name: {"abs_mean": sv["abs"]["mean"][j - 1], "abs_median": sv["abs"]["median"][j - 1],
                     "abs_p90": sv["abs"]["p90"][j - 1], "signed_p50": sv["signed"]["p50"][j - 1],
                     "signed_p05": sv["signed"]["p05"][j - 1], "signed_p95": sv["signed"]["p95"][j - 1],
                     "real_fell": float((gr[:, j - 1] < 0).double().mean())}
              for name, j in MARKS.items() if j <= H}
  return {"episodes": int(pred_g.shape[0]), "steps": H, "dt": cal[0]["dt"], "states": have_st,
          "curves": curves, "marks": marks, "safe_value": {"curves": sv, "marks": sv_marks},
          "g_head_gap": g_head_gap if have_st else None}


def conformal(cal: list[dict], tubes: list[dict], tags) -> dict:
  per_seed = {}
  for tag, tube in zip(tags, tubes):
    row = {"episodes": tube["episodes"], "calibration_half": tube["calibration_half"],
           "test_half": tube["test_half"], "levels": {}}
    for hname, hs in TUBE_H.items():
      hkey = {"0.4 s": "0.4", "1 s": "1", "3 s": "3"}[hname]
      hrow = tube["horizons"].get(hkey)
      if hrow is None:
        continue
      for c in LEVELS:
        lv = hrow["levels"][c]
        row["levels"].setdefault(hname, {})[c] = {
          "eps_filter": tube["min_eps"][c][hs],                    # refit on every episode
          "eps_split": lv["eps_one_sided"],
          "coverage_test": lv["one_sided_coverage"],
          "coverage_test_fell": lv["one_sided_coverage_by_group"]["real fell within H"],
          "coverage_test_not_fell": lv["one_sided_coverage_by_group"]["real did not fall"],
          "fall_recall": lv["fall_recall"], "false_alarm": lv["false_alarm"]}
      row.setdefault("real_fell", {})[hname] = hrow["real_fell"]
    per_seed[tag] = row
  pess = {}
  for hname, hs in TUBE_H.items():
    vals = []
    for c in cal:
      H = int(hs)
      realf = forward_fill(c["real_g"])
      r = c["pred_g"][:, :H].amin(1) - realf[:, :H].amin(1)
      vals.append(float((r <= 0).double().mean()))
    pess[hname] = {"min": min(vals), "max": max(vals)}
  mean = {}
  for hname in TUBE_H:
    for c in LEVELS:
      vals = [per_seed[t]["levels"][hname][c] for t in tags if hname in per_seed[t]["levels"]]
      if vals:
        # a coverage is None for a seed whose test half has no episode in its group
        # (e.g. none that really fell within H): the mean is over the seeds that have one
        mean.setdefault(hname, {})[c] = {
          k: (sum(x) / len(x) if (x := [v[k] for v in vals if v[k] is not None]) else None)
          for k in ("eps_filter", "coverage_test", "coverage_test_fell", "coverage_test_not_fell")}
        mean[hname][c]["eps_filter_min"] = min(v["eps_filter"] for v in vals)
        mean[hname][c]["eps_filter_max"] = max(v["eps_filter"] for v in vals)
  # the 1 s score distribution, pooled
  scores = []
  for c in cal:
    realf = forward_fill(c["real_g"])
    scores.append(c["pred_g"][:, :50].amin(1) - realf[:, :50].amin(1))
  s = torch.cat(scores)
  lo, hi = -1.0, 3.0
  bins = 80
  sc = s.clamp(lo, hi)
  hist = torch.histc(sc, bins=bins, min=lo, max=hi)
  return {"per_seed": per_seed, "mean": mean, "pessimistic_share": pess,
          "score_1s": {"n": int(s.numel()), "edges": [lo + (hi - lo) * i / bins for i in range(bins + 1)],
                       "counts": [int(x) for x in hist.tolist()],
                       "below_lo": int((s < lo).sum()), "above_hi": int((s > hi).sum()),
                       "quantiles": {q: _q(s, float(q)) for q in ("0.5", "0.9", "0.95", "0.99")},
                       "min": float(s.min()), "max": float(s.max())}}


# --- survival and hand-overs, from the evaluation runs ------------------------------------

def survival(evals: list[dict], tags) -> dict:
  bins = [f"{KICK_EDGES[i]:g}-{KICK_EDGES[i + 1]:g}" for i in range(len(KICK_EDGES) - 1)]
  out = {"bins": bins, "filters": {}}
  dv = torch.cat([d["episodes"]["dv"].double() for d in evals])
  edges = torch.tensor(KICK_EDGES[1:-1], dtype=torch.float64)
  bi = torch.bucketize(dv, edges, right=True)
  for label, name in FILTERS:
    if not all(name in d["conditions"] for d in evals):
      continue
    cat = lambda f: torch.cat([f(d["conditions"][name]) for d in evals])   # noqa: E731
    sv = cat(lambda r: r["survived"].bool())
    eng = cat(lambda r: r["tel_engaged"][:, 1].double())
    st = cat(lambda r: r["tel_steps"][:, 1].double())
    sw = cat(lambda r: r["tel_switches"][:, 1].double())
    row = {"label": label, "bins": {}, "per_seed": {}}
    # the seeds run the SAME episode list (row i is one episode in every run): an
    # episode's runs are one cluster, so the interval uses the per-episode mean over seeds
    S = len(evals)
    E0 = int(evals[0]["episodes"]["dv"].numel())
    same = all(bool((d["episodes"]["row"] == evals[0]["episodes"]["row"]).all()) for d in evals)
    sv_ep = sv.double().view(S, E0).mean(0) if same else None
    bi_ep = bi.view(S, E0)[0] if same else None
    for i, b in enumerate(bins + ["all"]):
      m = torch.ones_like(sv) if b == "all" else bi == i
      n, k = int(m.sum()), int(sv[m].sum())
      lo, hi = wilson(k, n)
      if same:
        me = torch.ones_like(bi_ep, dtype=torch.bool) if b == "all" else bi_ep == i
        x = sv_ep[me]
        nd = int(me.sum())
        p = float(x.mean()) if nd else math.nan
        se = float(x.std(unbiased=True)) / math.sqrt(nd) if nd > 1 else 0.0
        # Wilson at the effective sample size of the clustered mean (nd .. S * nd)
        n_eff = min(S * nd, max(nd, p * (1 - p) / se ** 2)) if se > 0 else nd
        lo, hi = wilson(p * n_eff, n_eff)
      row["bins"][b] = {"n": n, "n_distinct": int(me.sum()) if same else n,
                        "sr": k / n if n else math.nan, "sr_lo": lo, "sr_hi": hi,
                        "ci": "Wilson at the clustered effective n" if same else "Wilson",
                        "shac_share": float(eng[m].sum() / st[m].sum()) if n else math.nan,
                        "handovers": float(sw[m].mean()) if n else math.nan}
    for tag, d in zip(tags, evals):
      row["per_seed"][tag] = float(d["conditions"][name]["survived"].double().mean())
    out["filters"][name] = row
  return out


def pick_videos(ev0: dict, ranked_wm: list[str], n: int) -> dict:
  c = ev0["conditions"]
  e = ev0["episodes"]
  dv = e["dv"].double()
  cats = ev0["meta"]["categories"]
  son = c["sonic"]["survived"].bool()
  best = ranked_wm[:2]
  both = c[best[0]]["survived"].bool() & c[best[1]]["survived"].bool()
  out = {"best_wm": best, "categories": {}}
  used = set()

  def take(mask, key):
    idx = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    # spread over clip categories: round-robin by category
    by = {}
    for i in idx:
      by.setdefault(int(e["category"][i]), []).append(i)
    pick = []
    while len(pick) < n and any(by.values()):
      for k in sorted(by):
        if by[k] and len(pick) < n:
          i = by[k].pop(0)
          if i not in used:
            pick.append(i)
    used.update(pick)
    out["categories"][key] = [{"episode": i, "dv": float(dv[i]), "category": cats[int(e["category"][i])],
                               "outcomes": {nm: bool(c[nm]["survived"][i])
                                            for nm in ["sonic", "rollout:1"] + best}}
                              for i in pick]
    out["categories"][key + "_available"] = len(idx)

  take(~son & both, "sonic_fails_wm_survives")
  for lo, hi in ((0, 1), (1, 2), (2, 3)):
    take(~son & (dv >= lo) & (dv < hi), f"sonic_fails_{lo}-{hi}")
  return out


def main(argv=None):
  a = _parse(argv)
  cal_d = [_load(os.path.join(a.root, "calib", t, "push.pt")) for t in a.tags]
  tubes = [json.load(open(os.path.join(a.root, "calib", t, "min_tube.json"))) for t in a.tags]
  evals = [_load(os.path.join(a.root, "eval", t, "push.pt")) for t in a.tags]
  cal = [calib_arrays(d) for d in cal_d]
  meta = {"tags": list(a.tags),
          "calibration": {t: {"episodes": c["n_episodes"], "recorded": c["n_recorded"],
                              "dv_max": float(c["dv"].max()),
                              "categories": sorted({d["meta"]["categories"][int(i)] for i in c["category"].unique()})}
                          for t, c, d in zip(a.tags, cal, cal_d)},
          "evaluation": {t: {"episodes": int(d["episodes"]["dv"].numel()),
                             "dv_max": float(d["episodes"]["dv"].max()),
                             "exclude_categories": d["meta"].get("exclude_categories"),
                             "obs_noise": d["meta"].get("obs_noise")}
                         for t, d in zip(a.tags, evals)}}
  rep = {"meta": meta, "accuracy": accuracy(cal), "conformal": conformal(cal, tubes, a.tags),
         "survival": survival(evals, a.tags)}
  ranked = sorted([n for n in WM_FILTERS if n in rep["survival"]["filters"]],
                  key=lambda n: -rep["survival"]["filters"][n]["bins"]["all"]["sr"])
  rep["survival"]["wm_ranked"] = ranked
  if a.pick_videos:
    rep["videos"] = pick_videos(evals[0], ranked, a.pick_videos)
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  acc = rep["accuracy"]
  print(f"[report] {acc['episodes']} calibration episodes, states recorded: {acc['states']}")
  for name, m in acc["marks"].items():
    print(f"  {name:>8s}: up {m['up_share']:.2f}; " + "; ".join(
      f"{k} mean {m[k]['mean']:.3f} p90 {m[k]['p90']:.3f}" for k in QUANTITIES if k in m)
      + f"; |r| median {acc['safe_value']['marks'][name]['abs_median']:.3f}")
  for hname, row in rep["conformal"]["mean"].items():
    print(f"  tube {hname}: " + "; ".join(f"c={c} eps {v['eps_filter']:+.3f} cov {v['coverage_test']:.3f}"
                                         for c, v in row.items()))
  sv = rep["survival"]
  for name, row in sv["filters"].items():
    print(f"  {row['label']:32s} " + " ".join(f"{100 * row['bins'][b]['sr']:5.1f}" for b in sv["bins"] + ["all"])
          + "  | SHAC " + " ".join(f"{100 * row['bins'][b]['shac_share']:3.0f}" for b in sv["bins"] + ["all"]))
  if a.pick_videos:
    print(f"  best WM tube filters: {rep['videos']['best_wm']}")
    for k, v in rep["videos"]["categories"].items():
      print(f"  {k}: {v if isinstance(v, int) else [x['episode'] for x in v]}")
  print(f"[report] wrote {a.out}")


if __name__ == "__main__":
  main()
