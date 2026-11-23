"""Every motor's winding temperature under each policy of a cool_curve.py run
(the user, 2026-11-19: "for the cooling policy, can you also check the
temperature of the arm?").

For the motors matching ``--motors`` (default: the arms), per policy and
housing scenario, over the envs still standing: the median winding C at a few
times, the p90 at the end, the late heat input (mean Q of the last 60 s, W)
and the steady state that heat would reach if held indefinitely (the heat
model's resistances: Ts = Ta + (Q + P_idle) Rsa, Th = Ts + (Q + P_idle) Rha,
Tw = Th + Q Rwh; the housings' and structures' time constants of up to hours
mean a 10 min run is far from it), the same at zero torque ("idle-held": the
drive's own idle heat), and the hours that held heat takes to bring the
winding from the rest temperature to 70 C. Also the hottest motor of the whole body at the end, and a plot
(``motors.png`` beside curve.pt): the hottest matching winding over time, and
each matching joint (left / right, the hotter) at the end and at steady state.

    python examples/cool_motor_report.py runs/g1_cool/cool_curve_all/curve.pt
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"cooling": "#2a78d6", "fallback": "#eb6834", "cooling_fb": "#1baf7a"}
LABELS = {"cooling": "Cooling (first run)", "fallback": "Fallback (SHAC)", "cooling_fb": "Cooling (new: from fallback, temp ramp)"}
LIMIT_C = 70.0


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("curve")
  p.add_argument("--motors", default="shoulder|elbow|wrist", help="regex on the motor names")
  p.add_argument("--scenario", default=None, help="the plot's scenario (default: the first)")
  p.add_argument("--out", default=None)
  a = p.parse_args()
  d = torch.load(a.curve, map_location="cpu", weights_only=False)
  meta = d["meta"]
  names = list(meta["motor_names"])
  pols = list(meta["policies"])
  scen = meta["scenarios"]
  sid = d["scenario_of_env"]
  T = int(meta["duration_s"])
  sel = [i for i, nm in enumerate(names) if re.search(a.motors, nm)]
  if not sel:
    raise SystemExit(f"no motor matches {a.motors!r}: {names}")

  from robot_safety_sandbox.envs.assets.g1_sysid import motor_thermal_cls
  th = motor_thermal_cls().from_json(like=torch.zeros(1), sim_heat_scale=True)
  if list(th.names) != names:
    raise SystemExit("the heat model's motor order differs from the run's")
  P = {k: getattr(th, k).reshape(-1).double().numpy()
       for k in ("Cw", "Rwh", "Ch", "Rha", "Cs", "Rsa", "P_idle")}
  Ta = float(th.Ta)

  def steady(q):
    """Winding C under a held heat input q (W per motor, all motors)."""
    tot = q + P["P_idle"]
    return Ta + tot * (P["Rsa"] + P["Rha"]) + q * P["Rwh"]

  def hours_to(q, limit=LIMIT_C, t_max_h=8.0, every_s=10.0):
    """Hours until each winding reaches ``limit`` C under the held heat q,
    from every node at the run's rest temperature (the arms' start); NaN:
    not within t_max_h. Exact solution of the linear 3-node model
    (T(t) = T_ss + V exp(L t) V^-1 (T0 - T_ss)) on a ``every_s`` grid. The
    copper heat's rise with temperature is left out (~+10 % at 70 C vs
    40 C), so the times are a little long."""
    t = np.arange(0.0, t_max_h * 3600.0 + every_s, every_s)
    hit = np.full(len(q), np.nan)
    for i in range(len(q)):
      Cw, Rwh, Ch, Rha, Cs, Rsa = (P[k][i] for k in ("Cw", "Rwh", "Ch", "Rha", "Cs", "Rsa"))
      A = np.array([[-1 / (Rwh * Cw), 1 / (Rwh * Cw), 0.0],
                    [1 / (Rwh * Ch), -(1 / Rwh + 1 / Rha) / Ch, 1 / (Rha * Ch)],
                    [0.0, 1 / (Rha * Cs), -(1 / Rha + 1 / Rsa) / Cs]])
      b = np.array([q[i] / Cw, P["P_idle"][i] / Ch, Ta / (Rsa * Cs)])
      T_ss = -np.linalg.solve(A, b)
      lam, V = np.linalg.eig(A)
      c = np.linalg.solve(V, np.full(3, float(meta["rest_c"])) - T_ss)
      Tw = T_ss[0] + (V[0] * c * np.exp(np.outer(t, lam))).sum(1).real
      k = np.flatnonzero(Tw >= limit)
      if k.size:
        hit[i] = t[k[0]] / 3600.0
    return hit

  idle_ss = steady(np.zeros(len(names)))

  marks = [t for t in (10, 60, 120, 300, 600) if t <= T]
  res = {}
  for j, (sn, _c) in enumerate(scen):
    m = (sid == j)
    for name in pols:
      Tw = d[name]["Tw_all"][m].double().numpy()            # [n, T+1, M]
      Q = d[name]["heat_all"][m].double().numpy()
      ok = ~np.isnan(Tw[:, -1, 0])                          # standing at the end
      med = np.nanmedian(Tw[ok], axis=0)                    # [T+1, M]
      p90 = np.nanpercentile(Tw[ok][:, -1], 90, axis=0)
      late = np.nanmean(Q[ok][:, -60:], axis=(0, 1))        # [M]
      res[(sn, name)] = dict(Tw=Tw[ok], med=med, p90=p90, late=late, ss=steady(late),
                             t70=hours_to(np.nan_to_num(late)), n=int(ok.sum()),
                             n_all=int(m.sum()))
  # --- the table
  for sn, _c in scen:
    print(f"== {sn}: winding C, median of the envs standing at {T} s "
          f"(" + ", ".join(f"{LABELS.get(n_, n_)} {res[(sn, n_)]['n']}/{res[(sn, n_)]['n_all']}"
                         for n_ in pols) + ")")
    for n_ in pols:
      r = res[(sn, n_)]
      print(f"  {LABELS.get(n_, n_)}:  {'motor':24s}" + "".join(f"{t:>6d}s" for t in marks)
            + "   p90 end  heat W  if held  idle-held  to 70 C")
      for i in sel:
        print(f"  {'':{len(LABELS.get(n_, n_)) + 1}s}  {names[i]:24s}"
              + "".join(f"{r['med'][t, i]:7.1f}" for t in marks)
              + f"  {r['p90'][i]:8.1f}  {r['late'][i]:6.2f}  {r['ss'][i]:6.1f}"
              + f"  {idle_ss[i]:9.1f}  "
              + ("   > 8 h" if np.isnan(r['t70'][i]) else f"{r['t70'][i]:6.2f} h"))
    for n_ in pols:
      r = res[(sn, n_)]
      k = int(np.argmax(r["med"][-1]))
      ks = int(np.argmax(r["ss"]))
      hot_sel = r["Tw"][:, :, sel].max(axis=(1, 2))
      print(f"  {LABELS.get(n_, n_)}: hottest motor at {T} s {names[k]} {r['med'][-1, k]:.1f} C; "
            f"hottest at steady state {names[ks]} {r['ss'][ks]:.1f} C; matching motors' peak "
            f"over the run (any env) median {np.median(hot_sel):.1f}, max {hot_sel.max():.1f} C")
  # --- the plot
  sn = a.scenario or scen[0][0]
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
  plt.rcParams.update({"font.size": 11})
  t = np.arange(T + 1) / 60.0
  for ax in (ax0, ax1):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
      ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
      ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK2)
  for n_ in pols:
    r = res[(sn, n_)]
    hot = r["Tw"][:, :, sel].max(axis=2)                     # [n, T+1]
    c = COLORS.get(n_, INK2)
    ax0.fill_between(t, np.percentile(hot, 10, axis=0), np.percentile(hot, 90, axis=0),
                     color=c, alpha=0.13, lw=0)
    ax0.plot(t, np.median(hot, axis=0), color=c, lw=2, label=LABELS.get(n_, n_))
    ax0.annotate(f"{np.median(hot[:, -1]):.1f} C", xy=(t[-1], np.median(hot[:, -1])),
                 xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
  ax0.axhline(LIMIT_C, color=INK2, lw=1.2, ls=(0, (4, 3)))
  ax0.text(T / 60.0, LIMIT_C + 0.6, "70 C limit", color=INK2, fontsize=10, ha="right")
  ax0.set_ylim(25, 72)
  ax0.set_xlim(0, T / 60.0)
  ax0.set_xlabel("time (min)", color=INK2)
  ax0.set_ylabel("hottest arm winding (C)", color=INK2)
  ax0.set_title("Hottest arm motor over time", loc="left", fontsize=11, color=INK)
  ax0.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
  # per joint (the hotter side): at the end (filled) and at steady state (hollow)
  joints = sorted({re.sub(r"^(left|right)_", "", names[i]) for i in sel},
                  key=lambda s_: [re.sub(r"^(left|right)_", "", names[i]) for i in sel].index(s_))
  y = np.arange(len(joints))
  off = {n_: (k - (len(pols) - 1) / 2) * 0.22 for k, n_ in enumerate(pols)}
  x_hi = LIMIT_C + 2
  for n_ in pols:
    r = res[(sn, n_)]
    c = COLORS.get(n_, INK2)
    end, ss = [], []
    for jn in joints:
      idx = [i for i in sel if re.sub(r"^(left|right)_", "", names[i]) == jn]
      end.append(max(r["med"][-1, i] for i in idx))
      ss.append(max(r["ss"][i] for i in idx))
    ax1.scatter(end, y + off[n_], s=64, color=c, zorder=3)
    ax1.scatter(ss, y + off[n_], s=64, facecolors=SURFACE, edgecolors=c, linewidths=2, zorder=3)
    for e_, s_, yy in zip(end, ss, y + off[n_]):
      ax1.plot([e_, s_], [yy, yy], color=c, lw=1, alpha=0.5, zorder=2)
      if s_ >= LIMIT_C - 10:
        ax1.annotate(f"{s_:.0f} C", xy=(s_, yy), xytext=(7, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
    x_hi = max(x_hi, max(ss) + 8)
  ax1.axvline(LIMIT_C, color=INK2, lw=1.2, ls=(0, (4, 3)))
  ax1.set_yticks(y)
  ax1.set_yticklabels([j.replace("_joint", "").replace("_", " ") for j in joints], color=INK2)
  ax1.invert_yaxis()
  ax1.set_xlim(25, x_hi)
  ax1.set_xlabel("winding (C), hotter side", color=INK2)
  ax1.set_title(f"Per joint: at {T // 60} min (filled), if held (hollow)", loc="left",
                fontsize=11, color=INK)
  fig.suptitle(f"Arm motors, robot standing alone with no pushes (arms start at "
               f"{meta['rest_c']:.0f} C); median of robots still standing", fontsize=10,
               color=INK2, y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  out = a.out or os.path.join(os.path.dirname(a.curve), "motors.png")
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  print(f"[report] wrote {out}")


if __name__ == "__main__":
  main()
