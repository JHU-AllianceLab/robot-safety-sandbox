"""Which motor trips the overheat termination: reads ``push.pt`` files written by
``eval_sonic_qcbf.py --thermal-per-motor`` on a thermal task with an overheat
termination (``*_thermal70``, ``*_thermal``) and reports, per condition (pooled
over the files):

* trips: in the episodes the overheat termination ended, the motors at or over
  the limit at that step (``tw_end_m_c``); the "primary" motor is the one
  furthest over it;
* where each tripping motor started (winding at the spawn) and whether it was
  the hottest motor at the spawn; before / after the kick (``fell_pre``);
* every motor's rise over the episode (peak - spawn), all episodes.

CPU only, small: python examples/thermal_trip_report.py RUN/push.pt [...] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

GROUPS = (("ankle", ("ankle",)), ("knee", ("knee",)), ("hip", ("hip",)), ("waist", ("waist",)),
          ("arm", ("shoulder", "elbow", "wrist")))


def group_of(name: str) -> str:
  for g, keys in GROUPS:
    if any(k in name for k in keys):
      return g
  return "other"


def _q(x: torch.Tensor, q: float) -> float | None:
  return float(torch.quantile(x.double(), q)) if x.numel() else None


def load(paths: list[str]) -> dict:
  """{condition: {"rec": pooled records, "names": motor names, "limit": C, "files": n}}."""
  out: dict = {}
  for p in paths:
    d = torch.load(p, map_location="cpu", weights_only=False)
    for cname, rec in d["conditions"].items():
      if "tw_end_m_c" not in rec:
        raise SystemExit(f"{p}: condition {cname} has no per-motor windings "
                         f"(run eval_sonic_qcbf.py with --thermal-per-motor)")
      run = d["meta"]["runs"][cname]
      names, limit = run["thermal_motor_names"], run["overheat_limit_c"]
      if limit is None:
        raise SystemExit(f"{p}: {cname} ran on a task without an overheat termination")
      cur = out.setdefault(cname, {"parts": [], "names": names, "limit": limit, "files": []})
      if cur["names"] != names or cur["limit"] != limit:
        raise SystemExit(f"{p}: {cname} disagrees with the earlier files on the motors / limit")
      cur["parts"].append({k: rec[k] for k in ("tw_start_m_c", "tw_peak_m_c", "tw_end_m_c",
                                               "overheated", "survived", "fell_pre",
                                               "time_to_fall")})
      cur["files"].append(p)
  for c in out.values():
    c["rec"] = {k: torch.cat([p[k] for p in c["parts"]]) for k in c["parts"][0]}
    del c["parts"]
  return out


def report(c: dict) -> dict:
  r, names, lim = c["rec"], c["names"], float(c["limit"])
  t0, pk, te = r["tw_start_m_c"].double(), r["tw_peak_m_c"].double(), r["tw_end_m_c"].double()
  over = r["overheated"].bool()
  pre = r["fell_pre"].bool()
  E, K = t0.shape
  if bool(torch.isnan(te).any()):
    raise SystemExit(f"{int(torch.isnan(te).any(1).sum())} episodes have no end windings")
  trip = (te >= lim) & over.unsqueeze(1)                       # [E, K]
  n_over = int(over.sum())
  no_trip = int((over & ~trip.any(1)).sum())
  primary = torch.where(trip, te, torch.full_like(te, -1e9)).argmax(1)  # [E] (valid where over)
  hottest0 = t0.argmax(1)
  rise = pk - t0
  # s from the kick onset to the step the termination fired (onset ~ the spawn when
  # --onset-s is one step)
  ttf = r["time_to_fall"].double()
  motors = {}
  for i, nm in enumerate(names):
    m_trip = trip[:, i]
    m_prim = over & (primary == i)
    st = t0[m_prim, i]
    tt = ttf[m_prim]
    motors[nm] = {
      "group": group_of(nm),
      "trips": int(m_trip.sum()),
      "primary_trips": int(m_prim.sum()),
      "primary_share": int(m_prim.sum()) / n_over if n_over else None,
      "primary_trips_pre_kick": int((m_prim & pre).sum()),
      "primary_start_c": {"min": float(st.min()) if st.numel() else None,
                          "median": _q(st, 0.5), "max": float(st.max()) if st.numel() else None},
      "primary_time_to_trip_s": {"min": float(tt.min()) if tt.numel() else None,
                                 "median": _q(tt, 0.5),
                                 "max": float(tt.max()) if tt.numel() else None},
      "primary_was_hottest_at_spawn": int((m_prim & (hottest0 == i)).sum()),
      "hottest_at_spawn_episodes": int((hottest0 == i).sum()),
      "rise_c_all": {"mean": float(rise[:, i].mean()), "p90": _q(rise[:, i], 0.9),
                     "max": float(rise[:, i].max())},
      "peak_c_all": {"median": _q(pk[:, i], 0.5), "p90": _q(pk[:, i], 0.9),
                     "max": float(pk[:, i].max())},
    }
  groups = {}
  for g, _ in GROUPS:
    idx = [i for i, nm in enumerate(names) if group_of(nm) == g]
    gm = over & torch.isin(primary, torch.tensor(idx))
    groups[g] = {"primary_trips": int(gm.sum()),
                 "primary_share": int(gm.sum()) / n_over if n_over else None,
                 "motors": len(idx)}
  return {"episodes": E, "files": c["files"], "limit_c": lim, "overheated": n_over,
          "overheated_pre_kick": int((over & pre).sum()),
          "survived": int(r["survived"].bool().sum()),
          "fell_other": int((~r["survived"].bool() & ~over).sum()),
          "time_to_trip_s": {"median": _q(ttf[over], 0.5), "p10": _q(ttf[over], 0.1),
                             "p90": _q(ttf[over], 0.9)},
          "overheated_without_a_motor_over_the_limit": no_trip,
          "multi_motor_trips": int((trip.sum(1) > 1).sum()),
          "primary_was_hottest_at_spawn": int((over & (primary == hottest0)).sum()),
          "groups": groups, "motors": motors}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("files", nargs="+", help="push.pt of eval_sonic_qcbf.py --thermal-per-motor")
  p.add_argument("--out", required=True, help="JSON report")
  a = p.parse_args()
  res = {c: report(v) for c, v in load(a.files).items()}
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
  for cname, s in res.items():
    n = s["overheated"]
    print(f"[trip] {cname}: {s['episodes']} episodes, overheated {n} ({s['overheated_pre_kick']} "
          f"before the kick), survived {s['survived']}, fell (other) {s['fell_other']}; "
          f"time to trip median {s['time_to_trip_s']['median']} s; overheated with no motor over "
          f"{s['limit_c']:g} C: {s['overheated_without_a_motor_over_the_limit']} (must be 0); "
          f"2+ motors over at once: {s['multi_motor_trips']}; the tripping motor was the "
          f"hottest at the spawn in {s['primary_was_hottest_at_spawn']}/{n}")
    print("[trip]   by group: " + ", ".join(
      f"{g} {v['primary_trips']} ({100 * (v['primary_share'] or 0):.1f}%)"
      for g, v in s["groups"].items()))
    rows = sorted(s["motors"].items(), key=lambda kv: -kv[1]["primary_trips"])
    print(f"[trip]   {'motor':22s} {'trips':>5s} {'share':>6s} {'start C min/med/max':>20s} "
          f"{'trip s med':>10s} {'hottest@spawn':>13s} {'rise mean/p90 C':>16s} {'peak med/p90/max C':>19s}")
    for nm, m in rows:
      st = m["primary_start_c"]
      sts = (f"{st['min']:.1f}/{st['median']:.1f}/{st['max']:.1f}" if st["median"] is not None
             else "-")
      tts = m["primary_time_to_trip_s"]["median"]
      pkc = m["peak_c_all"]
      print(f"[trip]   {nm:22s} {m['primary_trips']:5d} "
            f"{100 * (m['primary_share'] or 0):5.1f}% {sts:>20s} "
            f"{(f'{tts:.1f}' if tts is not None else '-'):>10s} "
            f"{m['primary_was_hottest_at_spawn']:>13d} "
            f"{m['rise_c_all']['mean']:7.2f}/{m['rise_c_all']['p90']:.2f} "
            f"{pkc['median']:6.1f}/{pkc['p90']:.1f}/{pkc['max']:.1f}")


if __name__ == "__main__":
  main()
