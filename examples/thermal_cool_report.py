"""Does a controller cool hot motors? Reads ``push.pt`` files written by
``eval_sonic_qcbf.py --thermal-trace-s`` (usually with ``--thermal-start-group``
for hot ankles) and reports, per condition (pooled over the files):

* outcomes: survived / overheated / fell for other reasons, before / after the kick;
* the hottest ankle winding over time (and the hottest ankle housing), median /
  p10 / p90 over the episodes still running;
* cooling: the drop from the spawn to the episode's end, the share of survivors
  that ended cooler / below 50 C, the time to shed 5 / 10 C, the rate before the
  kick and the slope over the last 10 s;
* the heat a kick costs: the hottest ankle winding just before the kick vs its
  peak in the 3 s after it, by push size;
* each ankle motor's RMS torque over the survivors' last 10 s against the torque
  at which its winding stops cooling (heat in = conduction to the housing, at
  the end temperatures, standing still) and the continuous 70 C torque (every
  node at steady state), from the sysID heat model with the simulator's heat
  scale.

CPU only, small: python examples/thermal_cool_report.py RUN/push.pt [...] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

GROUP = "ankle"
TIMES_S = (0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 60)


def _q(x: torch.Tensor, q: float) -> float | None:
  x = x[~torch.isnan(x)]
  return float(torch.quantile(x.double(), q)) if x.numel() else None


def _stats(x: torch.Tensor) -> dict:
  return {"n": int((~torch.isnan(x)).sum()), "p10": _q(x, 0.1), "median": _q(x, 0.5),
          "p90": _q(x, 0.9)}


def load(paths: list[str]) -> dict:
  out: dict = {}
  keys = ("tw_trace_m_c", "th_trace_m_c", "tau2_trace_m", "overheated", "survived", "fell_pre",
          "time_to_fall")
  for p in paths:
    d = torch.load(p, map_location="cpu", weights_only=False)
    dt = float(d["meta"]["dt"])
    for cname, rec in d["conditions"].items():
      if "tw_trace_m_c" not in rec:
        raise SystemExit(f"{p}: {cname} has no temperature trace (run with --thermal-trace-s)")
      run = d["meta"]["runs"][cname]
      meta = (run["thermal_motor_names"], run["thermal_trace_s"], run["overheat_limit_c"], dt)
      cur = out.setdefault(cname, {"parts": [], "meta": meta, "files": []})
      if cur["meta"] != meta:
        raise SystemExit(f"{p}: {cname} disagrees with the earlier files (motors / trace / limit)")
      part = {k: rec[k] for k in keys}
      part["on"] = d["episodes"]["on"]
      part["dv"] = d["episodes"]["dv"]
      cur["parts"].append(part)
      cur["files"].append(p)
  for c in out.values():
    T = max(p["tw_trace_m_c"].shape[1] for p in c["parts"])

    def pad(x):
      if x.dim() < 2 or x.shape[1] == T:
        return x
      fill = torch.full((x.shape[0], T - x.shape[1], *x.shape[2:]), math.nan, dtype=x.dtype)
      return torch.cat([x, fill], 1)

    c["rec"] = {k: torch.cat([pad(p[k]) if k.endswith(("_c", "_m")) else p[k]
                              for p in c["parts"]]) for k in c["parts"][0]}
    del c["parts"]
  return out


def heat_limits(names: list[str], idx: list[int], limit_c: float):
  """Per ankle motor: (k70, Rwh, cont70) -- heat per tau^2 at the limit standing
  still (W / (N m)^2), winding -> housing resistance, and the continuous torque
  that holds the winding at the limit with every node at steady state. None
  when the sysID model cannot be loaded."""
  try:
    from robot_safety_sandbox.envs.assets.g1_sysid import motor_thermal_cls
    m = motor_thermal_cls().from_json(sim_heat_scale=True)
  except Exception as e:  # noqa: BLE001 -- the report works without it
    print(f"[cool] no heat model ({type(e).__name__}: {e}); torque limits skipped")
    return None
  if list(m.names) != list(names):
    raise SystemExit("the heat model's motor order differs from the trace's")
  out = {}
  for i in idx:
    stall = 1.0 + float(m.stall_s[i])
    a = float(m.a[i])
    rsa = float(m.Rsa[i]) if m.Cs is not None else 0.0
    k70 = a * (1.0 + m.alpha * (limit_c - 25.0)) * stall
    room = limit_c - m.Ta - float(m.P_idle[i]) * (float(m.Rha[i]) + rsa)
    cont = math.sqrt(max(room, 0.0) / (k70 * (float(m.Rwh[i]) + float(m.Rha[i]) + rsa)))
    out[i] = {"a": a, "alpha": m.alpha, "stall": stall, "Rwh": float(m.Rwh[i]), "cont70": cont}
  return out


def report(c: dict, dv_bins: list[float], push_s: float, late_s: float) -> dict:
  names, trace_s, lim, dt = c["meta"]
  lim = 70.0 if lim is None else float(lim)
  r = c["rec"]
  tw, th, tau2 = r["tw_trace_m_c"].double(), r["th_trace_m_c"].double(), r["tau2_trace_m"].double()
  E, T, K = tw.shape
  idx = [i for i, nm in enumerate(names) if GROUP in nm]
  rest = [i for i in range(K) if i not in idx]
  hot = tw[:, :, idx].amax(2)                  # [E, T]  NaN once the episode ended
  hot_h = th[:, :, idx].amax(2)
  alive = ~torch.isnan(hot)
  last = alive.sum(1) - 1                      # the last sample of each episode
  ar = torch.arange(E)
  surv, over, pre = r["survived"].bool(), r["overheated"].bool(), r["fell_pre"].bool()
  fell = ~surv & ~over
  stride = int(round(trace_s / dt))
  on = r["on"].long()
  j_pre = (on // stride).clamp_max(T - 1)      # the last sample before the kick
  start, end = hot[:, 0], hot[ar, last]

  curve = []
  for t in TIMES_S:
    j = int(round(t / trace_s))
    if j >= T:
      break
    curve.append({"t_s": t, "winding": _stats(hot[:, j]), "housing": _stats(hot_h[:, j])})

  def first_below(drop):
    ok = hot <= (start - drop).unsqueeze(1)
    j = torch.where(ok, torch.arange(T).expand(E, T), torch.full((E, T), T)).amin(1)
    return torch.where(j < T, j.double() * trace_s, torch.full((E,), math.nan, dtype=torch.double))

  t5, t10 = first_below(5.0), first_below(10.0)
  pre_rate = torch.where(j_pre > 0, (hot[ar, j_pre] - start) / (j_pre.double() * trace_s),
                         torch.full((E,), math.nan, dtype=torch.double))
  nl = max(1, int(round(late_s / trace_s)))
  j_lo = (last - nl).clamp_min(0)
  late_slope = torch.where(last - j_lo > 0, (end - hot[ar, j_lo]) / ((last - j_lo).double() * trace_s),
                           torch.full((E,), math.nan, dtype=torch.double))

  # the kick's heat: peak over (j_pre, j_pre + push_s] minus the pre-kick sample
  npk = max(1, int(round(push_s / trace_s)))
  win = j_pre.unsqueeze(1) + torch.arange(1, npk + 1).unsqueeze(0)
  win_ok = win < T
  vals = hot[ar.unsqueeze(1), win.clamp_max(T - 1)]
  vals = torch.where(win_ok & ~torch.isnan(vals), vals, torch.full_like(vals, -math.inf))
  peak = vals.amax(1)
  kick_dT = torch.where(torch.isfinite(peak), peak - hot[ar, j_pre],
                        torch.full((E,), math.nan, dtype=torch.double))
  dv = r["dv"].double()
  by_dv = []
  for lo, hi in zip(dv_bins[:-1], dv_bins[1:]):
    m = (dv >= lo) & (dv < hi) if hi != dv_bins[-1] else (dv >= lo) & (dv <= hi)
    n = int(m.sum())
    by_dv.append({"dv": [lo, hi], "episodes": n,
                  "survived": int((m & surv).sum()), "overheated": int((m & over).sum()),
                  "fell_other": int((m & fell).sum()),
                  "kick_heat_c": _stats(kick_dT[m]),
                  "kick_heat_c_max": float(kick_dT[m].nan_to_num(-math.inf).max()) if n else None})

  # the ankle motors' late RMS torque vs the torque that stops the winding's cooling
  lims = heat_limits(list(names), idx, lim)
  motors = {}
  sv = surv.nonzero(as_tuple=False).squeeze(-1)
  for i in idx:
    tt = tau2[:, :, i]
    js = torch.arange(T).unsqueeze(0)
    m_late = (js > j_lo.unsqueeze(1)) & (js <= last.unsqueeze(1)) & ~torch.isnan(tt)
    rms = torch.sqrt((tt.nan_to_num(0.0) * m_late).sum(1) / m_late.sum(1).clamp_min(1))
    rms = torch.where(m_late.any(1), rms, torch.full_like(rms, math.nan))
    tw_e, th_e = tw[ar, last, i], th[ar, last, i]
    mot = {"start_c": _stats(tw[:, 0, i]), "end_c_survivors": _stats(tw_e[sv]),
           "housing_end_c_survivors": _stats(th_e[sv]),
           "late_rms_torque_nm_survivors": _stats(rms[sv])}
    if lims is not None:
      L = lims[i]
      k_now = L["a"] * (1.0 + L["alpha"] * (tw_e - 25.0)) * L["stall"]
      tau_cool = torch.sqrt(((tw_e - th_e).clamp_min(0.0)) / (L["Rwh"] * k_now))
      mot.update(cooling_torque_nm_at_end_survivors=_stats(tau_cool[sv]),
                 still_cooling_share_survivors=(float((rms[sv] < tau_cool[sv]).double().mean())
                                                if sv.numel() else None),
                 continuous_70c_torque_nm=L["cont70"],
                 above_continuous_share_survivors=(float((rms[sv] > L["cont70"]).double().mean())
                                                   if sv.numel() else None))
    motors[names[i]] = mot

  rest_peak = tw[:, :, rest].nan_to_num(-math.inf).amax(2).amax(1)
  return {
    "files": c["files"], "episodes": E, "trace_s": trace_s, "limit_c": lim,
    "group": GROUP, "group_motors": [names[i] for i in idx],
    "survived": int(surv.sum()), "overheated": int(over.sum()),
    "overheated_pre_kick": int((over & pre).sum()), "fell_other": int(fell.sum()),
    "fell_other_pre_kick": int((fell & pre).sum()),
    "hot_start_c": _stats(start), "hot_end_c_survivors": _stats(end[surv]),
    "hot_drop_c_survivors": _stats((start - end)[surv]),
    "ended_cooler_share_survivors": float((end[surv] < start[surv]).double().mean()) if surv.any() else None,
    "ended_below_50c_share_survivors": float((end[surv] < 50.0).double().mean()) if surv.any() else None,
    "time_to_shed_5c_s": _stats(t5), "shed_5c_share": float((~torch.isnan(t5)).double().mean()),
    "time_to_shed_10c_s": _stats(t10), "shed_10c_share": float((~torch.isnan(t10)).double().mean()),
    "pre_kick_rate_c_per_s": _stats(pre_rate),
    "late_slope_c_per_s_survivors": _stats(late_slope[surv]),
    "other_motors_peak_c": _stats(rest_peak),
    "curve": curve, "kick_by_dv": by_dv, "motors": motors}


def _f(x, nd=1):
  return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("files", nargs="+", help="push.pt of eval_sonic_qcbf.py --thermal-trace-s")
  p.add_argument("--out", required=True, help="JSON report")
  p.add_argument("--dv-bins", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5])
  p.add_argument("--kick-window-s", type=float, default=3.0)
  p.add_argument("--late-s", type=float, default=10.0)
  a = p.parse_args()
  res = {cn: report(c, a.dv_bins, a.kick_window_s, a.late_s) for cn, c in load(a.files).items()}
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
  for cn, s in res.items():
    E = s["episodes"]
    print(f"[cool] {cn}: {E} episodes; survived {s['survived']}, overheated {s['overheated']} "
          f"({s['overheated_pre_kick']} before the kick), fell (other) {s['fell_other']} "
          f"({s['fell_other_pre_kick']} before the kick); other motors' peak median "
          f"{_f(s['other_motors_peak_c']['median'])} C, p90 {_f(s['other_motors_peak_c']['p90'])} C")
    hs, he, hd = s["hot_start_c"], s["hot_end_c_survivors"], s["hot_drop_c_survivors"]
    print(f"[cool]   hottest ankle winding: spawn median {_f(hs['median'])} C -> survivors' end "
          f"{_f(he['median'])} C (drop median {_f(hd['median'])}, p10 {_f(hd['p10'])}, p90 "
          f"{_f(hd['p90'])} C); ended cooler {_f(100 * (s['ended_cooler_share_survivors'] or 0))}%, "
          f"below 50 C {_f(100 * (s['ended_below_50c_share_survivors'] or 0))}%")
    print(f"[cool]   shed 5 C: {_f(100 * s['shed_5c_share'])}% (median {_f(s['time_to_shed_5c_s']['median'])} s); "
          f"10 C: {_f(100 * s['shed_10c_share'])}% (median {_f(s['time_to_shed_10c_s']['median'])} s); "
          f"rate before the kick median {_f(s['pre_kick_rate_c_per_s']['median'], 2)} C/s; "
          f"last-{a.late_s:g}-s slope median {_f(s['late_slope_c_per_s_survivors']['median'], 3)} C/s "
          f"(p10 {_f(s['late_slope_c_per_s_survivors']['p10'], 3)}, p90 "
          f"{_f(s['late_slope_c_per_s_survivors']['p90'], 3)})")
    print(f"[cool]   {'t s':>5s} {'alive':>5s} {'winding p10/med/p90 C':>22s} {'housing p10/med/p90 C':>22s}")
    for row in s["curve"]:
      w, h = row["winding"], row["housing"]
      print(f"[cool]   {row['t_s']:5g} {w['n']:5d} "
            f"{_f(w['p10']):>6s}/{_f(w['median'])}/{_f(w['p90']):<6s}      "
            f"{_f(h['p10']):>6s}/{_f(h['median'])}/{_f(h['p90'])}")
    print(f"[cool]   kick heat (hottest ankle winding, peak in {a.kick_window_s:g} s after - before):")
    for b in s["kick_by_dv"]:
      k = b["kick_heat_c"]
      print(f"[cool]     dv {b['dv'][0]:g}-{b['dv'][1]:g} m/s: {b['episodes']:4d} eps, survived "
            f"{b['survived']}, overheated {b['overheated']}, fell {b['fell_other']}; dT median "
            f"{_f(k['median'])} p90 {_f(k['p90'])} max {_f(b['kick_heat_c_max'])} C")
    print(f"[cool]   {'motor':18s} {'start med':>9s} {'end med':>7s} {'Th end':>6s} "
          f"{'late rms Nm med/p90':>19s} {'cool-at Nm':>10s} {'cooling':>7s} {'cont70':>6s} {'>cont':>5s}")
    for nm, m in s["motors"].items():
      rm = m["late_rms_torque_nm_survivors"]
      cool = m.get("cooling_torque_nm_at_end_survivors", {}).get("median")
      sc = m.get("still_cooling_share_survivors")
      ac = m.get("above_continuous_share_survivors")
      print(f"[cool]   {nm:18s} {_f(m['start_c']['median']):>9s} "
            f"{_f(m['end_c_survivors']['median']):>7s} {_f(m['housing_end_c_survivors']['median']):>6s} "
            f"{_f(rm['median'], 2):>9s}/{_f(rm['p90'], 2):<9s} {_f(cool, 2):>10s} "
            f"{_f(None if sc is None else 100 * sc, 0):>6s}% {_f(m.get('continuous_70c_torque_nm'), 2):>6s} "
            f"{_f(None if ac is None else 100 * ac, 0):>4s}%")


if __name__ == "__main__":
  main()
