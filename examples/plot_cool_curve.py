"""Plot examples/cool_curve.py's output: the hottest ankle winding over time, per
policy (median of the envs still standing, p10-p90 band), against the heat
model's zero-torque limit, for each housing scenario; the first minute and the
whole run side by side. Writes cool_curve.png next to curve.pt and prints a
table of the medians.

    python examples/plot_cool_curve.py runs/g1_cool/cool_curve/curve.pt
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

#: reference palette (dataviz skill, light mode): categorical slots 1-2, neutrals
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"cooling": "#2a78d6", "fallback": "#eb6834", "cooling_fb": "#1baf7a"}
LIMIT = "#8a8984"
LABELS = {"cooling": "Cooling (first run)", "fallback": "Fallback (SHAC)", "cooling_fb": "Cooling (new: from fallback, temp ramp)"}


def passive_curve(meta, th_c: float, seconds: int):
  """The heat model's zero-torque (drive on) hottest-ankle winding, 1 s grid, from
  the run's start temperatures (winding ankle_c, housing th_c, structure)."""
  from robot_safety_sandbox.envs.assets.g1_sysid import motor_thermal_cls
  th = motor_thermal_cls().from_json(like=torch.zeros(1), sim_heat_scale=True)
  names = list(th.names)
  ank = [names.index(n) for n in meta["ankle_motors"]]
  k = len(names)
  Tw = torch.full((1, k), float(meta["rest_c"]))
  Th, Ts = Tw.clone(), Tw.clone()
  Tw[:, ank] = meta["ankle_c"]
  Th[:, ank] = min(th_c, meta["ankle_c"])
  Ts[:, ank] = min(meta["structure_c"], th_c)
  T, zero, out = (Tw, Th, Ts), torch.zeros(1, k), [float(Tw[0, ank].max())]
  for _ in range(int(seconds)):
    for _ in range(50):
      T = th.step(T, zero, zero, 0.02)
    out.append(float(T[0][0, ank].max()))
  return torch.tensor(out)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("curve")
  p.add_argument("--zoom-s", type=float, default=60.0)
  p.add_argument("--out", default=None)
  a = p.parse_args()
  d = torch.load(a.curve, map_location="cpu", weights_only=False)
  meta = d["meta"]
  scen = meta["scenarios"]
  sid = d["scenario_of_env"]
  pols = [k for k in meta["policies"]]
  T = int(meta["duration_s"])
  t = torch.arange(T + 1).float()

  plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                       "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
  fig, axes = plt.subplots(len(scen), 2, figsize=(9.5, 3.6 * len(scen)), facecolor=SURFACE,
                           gridspec_kw={"width_ratios": [1, 1.6]})
  rows = []
  for j, (sn, th_c) in enumerate(scen):
    ref_hot = passive_curve(meta, th_c, T)                # 1 s grid, 0..T
    ref_t = torch.arange(T + 1).float()
    keep = ref_t <= T + 1e-6
    stats = {}
    for name in pols:
      r = d[name]
      m = sid == j
      hot = r["Tw"][m].amax(-1)                           # [n, T+1] hottest ankle
      stand = ~torch.isnan(hot)
      med = hot.nanquantile(0.5, dim=0)
      lo, hi = hot.nanquantile(0.1, dim=0), hot.nanquantile(0.9, dim=0)
      stats[name] = (med, lo, hi, stand.float().mean(0))
    for c, xmax in enumerate((a.zoom_s, float(T))):
      ax = axes[j, c] if len(scen) > 1 else axes[c]
      ax.set_facecolor(SURFACE)
      ax.grid(True, color=GRID, linewidth=0.8)
      ax.set_axisbelow(True)
      for s in ("top", "right"):
        ax.spines[s].set_visible(False)
      sc = 1.0 if c == 0 else 60.0
      ax.plot(ref_t[keep] / sc, ref_hot[keep], color=LIMIT, lw=2, ls=(0, (4, 3)),
              label="Physics limit (zero torque)")
      for name in pols:
        med, lo, hi, _f = stats[name]
        ax.fill_between(t / sc, lo, hi, color=COLORS.get(name, INK2), alpha=0.13, lw=0)
        ax.plot(t / sc, med, color=COLORS.get(name, INK2), lw=2, label=LABELS.get(name, name))
      ax.set_xlim(0, xmax / sc)
      ax.set_ylim(25, 70)
      ax.set_xlabel("time (s)" if c == 0 else "time (min)")
      if c == 0:
        ax.set_ylabel("hottest ankle winding (C)")
      ttl = (f"{sn.capitalize()} (housing {th_c:.0f} C): first {a.zoom_s:.0f} s" if c == 0
             else f"{sn.capitalize()}: full {T // 60} min")
      ax.set_title(ttl, loc="left", fontsize=11, color=INK)
      if c == 1:
        # direct labels at the right end
        ends = [(float(stats[n][0][~torch.isnan(stats[n][0])][-1]), n,
                 COLORS.get(n, INK2)) for n in pols]
        ends.append((float(ref_hot[keep][-1]), "limit", LIMIT))
        ax.set_xlim(0, xmax / sc)
        ends.sort()
        last = -1e9
        for y, lab, col in ends:
          y_txt = max(y, last + 2.6)
          ax.annotate(f"{lab} {y:.1f} C", xy=(T / 60, y), xytext=(1.02, y_txt),
                      textcoords=("axes fraction", "data"), color=INK, fontsize=10,
                      va="center", annotation_clip=False)
          last = y_txt
    for k in (10, 30, 60, 120, 300, 600):
      if k <= T:
        rows.append((sn, k, *(float(stats[n][0][k]) for n in pols),
                     *(float(stats[n][3][k]) for n in pols), float(ref_hot[k])))
  h, l = (axes[0, 1] if len(scen) > 1 else axes[1]).get_legend_handles_labels()
  fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
  fig.suptitle(f"All four ankle windings start at {meta['ankle_c']:.0f} C; robot stands alone, "
               f"no pushes (median, p10-p90 band of robots still standing)",
               y=0.955, fontsize=10, color=INK2)
  fig.subplots_adjust(left=0.08, right=0.85, top=0.86, bottom=0.08, hspace=0.42, wspace=0.18)
  out = a.out or os.path.join(os.path.dirname(a.curve), "cool_curve.png")
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  print(f"[plot] wrote {out}")
  hdr = ("scenario", "t s", *(f"{n} C" for n in pols), *(f"{n} standing" for n in pols), "limit C")
  print("  ".join(f"{x:>16s}" for x in hdr))
  for r in rows:
    print("  ".join(f"{x:>16}" if isinstance(x, str) else
                    (f"{x:>16d}" if isinstance(x, int) else f"{x:>16.2f}") for x in r))


if __name__ == "__main__":
  main()
