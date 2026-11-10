"""The push suite: the fixed-push evaluation every LAFAN-recovery experiment reads.

    python examples/eval_push_suite.py --calibrate --policy privcrit=runs/.../final_model.zip
    python examples/eval_push_suite.py --frozen    --policy S3_s0=runs/.../final_model.zip

This module holds the PURE half (no simulator, CPU-tested in
``tests/test_push_suite.py``): the grid of push cells, their assignment to
episodes, the per-episode record schema, the survival curves and F-levels, and
the frozen-grid proposal. ``examples/eval_push_suite.py`` is the simulator half
that fills the records. The experiment plan this serves is
``docs/experiments/g1-lafan-recovery.md``.

WHY A SUITE. A recovery policy trained with a per-env force curriculum reports
its own frontier in its own units (``game/force_scale_mean`` x ``force_max``,
direction-blind, on the training start mix), which moves whenever the start
mix, the push timing or the ceiling changes. Comparing arms needs ONE protocol
that does not: the same start states, the same pushes, the same seed, for
every checkpoint. E1 runs it on a wide CALIBRATION grid to measure the push
range (where survival falls from 90 % to 10 %); the grid is then FROZEN to
five levels spanning that range and every later experiment evaluates on it.

SETTINGS (where an episode starts and who is in control when the push lands):

    std       the nominal stance at rest, pushed ``onset_s`` after the spawn
              (E-std: the guardrail; a better LAFAN policy must not get worse
              here)
    lafan     a held-out LAFAN frame (the test split of the mocap bank),
              pushed ``onset_s`` after the spawn (E-lafan: novel poses and
              momentum)
    handover  a held-out clip REPLAYED open-loop from a library frame
              (``envs/g1_recover/playback.py``); the push lands U(0.02, 0.1) s
              into the replay and the fallback takes over ``delay`` s after
              the push ONSET, with the push still acting if it has not ended
              (E-handover: the state a filter actually hands over in; the
              primary metric)

and two opt-in settings for read-offs of single experiments:

    walker    the walker bank (the training mixture's other 30 %), every
              source sampled uniformly over its rows, pushed like std (E3's
              forgetting check)
    pre       a held-out bank of COLLECTED handover states (``KIND_HANDOVER``
              rows, e.g. replayed test clips + push), no scripted push: the
              row's own pending push, if any, plays out (E4's E-pre). Its
              ``impulse`` column is the impulse that PRODUCED the row (binned
              to 4 N s) and ``delay`` the row's onset -> handover time

CELLS. A cell is one push: impulse J (N s) x duration T (s) x direction (one
of ``n_dirs`` evenly spaced HEADING-frame angles of the force, 0 = along the
heading = a push from BEHIND, +90 deg = toward the robot's left = a push from
its RIGHT) x handover delay (s; 0 outside the handover setting). The force is
``J / (steps dt)`` with ``steps = round(T / dt)``, so the delivered impulse is
exactly J whatever the control rate. A push is described by impulse because a
0.2 s shove changes the base velocity by about ``dv = J / m`` before the feet
can respond (m = 33.34 kg: 260 N x 0.2 s ~ 1.56 m/s, 100 N x 0.2 s ~ 0.6 m/s);
whether impulse ALONE predicts survival is one of E1's questions (the
per-duration curves overlap or they do not). Episodes are assigned to cells
round-robin in the order they are issued (:func:`assign_cells`), so any prefix
of the issue order is balanced to within one episode per cell and every
policy sees the same cell sequence.

RECORDS. One row per finished episode, columnar (``dict`` of equal-length
columns: a 1-D tensor / array per numeric field, a list of ``str`` per
:data:`STR_FIELDS` field). :data:`RECORD_FIELDS` documents each one. The
stance columns are measured straight from the robot data (pelvis band, tilt,
per-axis body twist, ``targets.make_l_stance`` UNGATED) and only once the
push has ENDED and the fallback is in control, for the two tolerances in
:data:`TOLERANCES`. Reaching that set says nothing about staying in it:
reach-avoid certifies the first reach only, so ``stay_frac`` and
``fell_after_reach`` measure what happens after it rather than assume it.

SUMMARY (:func:`summarize`). Per setting x policy: survival per impulse level
(pooled, and per direction / duration / delay / category), the F90 / F50 /
F10 levels of those curves (:func:`f_level`), the prefix-fall rate of the
handover replays, behaviour statistics (median peak speed and distance,
reach / stay / fall-after-reach per tolerance), and the AUC of the policy's
own value at the takeover state against survival (the monitor-quality reading
of E4). Handover episodes whose REPLAY fell before the takeover are not the
fallback's failures: they are counted in ``prefix_fall_rate`` and left out of
every survival number.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

#: The G1's total mass (mjlab's model), kg: impulse <-> base velocity change.
G1_MASS_KG = 33.34
#: The two stance targets the suite scores, name -> (lin_tol m/s, ang_tol
#: rad/s) per body axis: ``tight`` is g1_stabilize's ``l_stance`` (and the
#: g1_recover_lafan_ra target), ``loose`` the g1_recover_lafan_ra_loose one.
TOLERANCES = {"tight": (0.25, 0.30), "loose": (0.5, 0.6)}
SETTINGS = ("std", "lafan", "handover")
#: Opt-in settings (see the module docstring), never run by default.
EXTRA_SETTINGS = ("walker", "pre")
ALL_SETTINGS = SETTINGS + EXTRA_SETTINGS
#: ``pre``: the producing impulse is binned to this width, N s.
PRE_IMPULSE_BIN = 4.0
#: E1's calibration grid: 0..60 N s in steps of 4, three durations.
CALIBRATION_IMPULSES = tuple(float(j) for j in range(0, 61, 4))
CALIBRATION_DURATIONS = (0.1, 0.2, 0.3)
HANDOVER_DELAYS = (0.0, 0.1, 0.2)
N_DIRECTIONS = 8
#: The training push's duration (the g1_recover_lafan* random_push duration_s):
#: the frozen grid keeps it alone when the duration curves overlap, and the
#: suggested training ceiling converts an impulse to a force with it.
TRAIN_DURATION_S = 0.2
#: E1's decision rule for the training ceiling: ~1.3 x the E-lafan F10.
CEILING_FACTOR = 1.3
#: Labels of the 8 directions by the side the push comes FROM (the force's
#: heading-frame angle 0, 45, ..., 315 deg); 4 directions use random_push.SIDES.
DIR_NAMES_8 = ("behind", "rear-right", "right", "front-right", "front",
               "front-left", "left", "rear-left")
DIR_NAMES_4 = ("behind", "right", "front", "left")

#: Per-episode record columns (see the module docstring). ``<tol>`` is each
#: :data:`TOLERANCES` key.
RECORD_FIELDS = {
  "policy": "the --policy NAME the episode ran",
  "setting": "std / lafan / handover (/ walker / pre)",
  "cell": "index into the run's cell list",
  "impulse": "push impulse, N s",
  "duration": "push duration asked for, s",
  "force": "push force applied, N (impulse / (steps dt))",
  "dir_idx": "direction index, 0 = from behind, counterclockwise",
  "n_dirs": "number of directions in the grid",
  "delay": "handover: takeover delay after the push onset, s (0 elsewhere)",
  "onset_s": "push onset after the spawn, s",
  "handover_s": "fallback takeover after the spawn, s (0 outside handover)",
  "row": "spawn bank row (env._zoo_spawn_row; -1 = nominal stance)",
  "clip": "clip name of the spawn row ('' when none)",
  "category": "clip category of the spawn row ('' when none)",
  "survived": "no failure termination within eval_s of the takeover",
  "prefix_fell": "handover: the replay fell BEFORE the takeover",
  "fall_t": "s from the takeover to the fall (NaN if none)",
  "peak_speed": "peak planar base speed from the push onset to the end, m/s",
  "dist": "planar base distance takeover -> end of the record, m",
  "dist_spawn": "planar base distance spawn -> end of the record, m",
  "value0": "the policy's own V at the takeover state (NaN if not read)",
  "t_reach_<tol>": "s from the later of push end and takeover to the first "
                   "step inside the stance set (NaN if never)",
  "stay_frac_<tol>": "share of the steps after the first reach spent inside",
  "fell_after_reach_<tol>": "reached, then fell inside the window",
}
#: Columns holding one ``str`` per episode (everything else is numeric).
STR_FIELDS = ("policy", "setting", "clip", "category")


# --- units ---------------------------------------------------------------------

def impulse_to_dv(impulse, mass: float = G1_MASS_KG):
  """Base velocity change of a free body of ``mass`` kg, m/s (``J / m``)."""
  return impulse / mass


def dv_to_impulse(dv, mass: float = G1_MASS_KG):
  """The impulse that changes a ``mass`` kg body's velocity by ``dv``, N s."""
  return dv * mass


def push_steps(duration: float, dt: float) -> int:
  """Control steps a push of ``duration`` s lasts (at least one)."""
  return max(1, int(round(float(duration) / float(dt))))


def push_force(impulse: float, duration: float, dt: Optional[float] = None) -> float:
  """The force, N, that delivers ``impulse`` over ``duration``. With ``dt`` the
  duration is first rounded to whole control steps (:func:`push_steps`), so
  the impulse actually delivered is exactly ``impulse``."""
  t = float(duration) if dt is None else push_steps(duration, dt) * float(dt)
  return float(impulse) / t


def direction_label(idx: int, n_dirs: int) -> str:
  """The side a push of direction ``idx`` (of ``n_dirs``) comes FROM
  (``any`` for a one-direction grid: the ``pre`` setting's own pushes)."""
  if n_dirs == 1:
    return "any"
  if n_dirs == 8:
    return DIR_NAMES_8[int(idx)]
  if n_dirs == 4:
    return DIR_NAMES_4[int(idx)]
  return f"{360.0 * int(idx) / int(n_dirs):g}deg"


# --- cells and their assignment -------------------------------------------------

def build_cells(impulses: Iterable[float], durations: Iterable[float],
                n_dirs: int, delays: Iterable[float] = (0.0,)) -> list[dict]:
  """Every (impulse, duration, direction, delay) combination, as a list of
  dicts with ``cell`` (its index), ``impulse``, ``duration``, ``dir_idx``,
  ``rel`` (the force's heading-frame angle, rad) and ``delay``.

  Impulse 0 is kept for every duration and direction: it costs one level of
  the grid and leaves every per-direction / per-duration curve its baseline.
  """
  imp = [float(j) for j in impulses]
  dur = [float(t) for t in durations]
  dly = [float(d) for d in delays]
  if int(n_dirs) < 1:
    raise ValueError(f"n_dirs must be >= 1, got {n_dirs}")
  if not imp or not dur or not dly:
    raise ValueError("build_cells needs at least one impulse, duration and delay")
  if min(imp) < 0.0 or min(dur) <= 0.0 or min(dly) < 0.0:
    raise ValueError(f"impulses must be >= 0, durations > 0, delays >= 0; got "
                     f"{imp}, {dur}, {dly}")
  cells = []
  for j, t, d, delay in itertools.product(imp, dur, range(int(n_dirs)), dly):
    cells.append(dict(cell=len(cells), impulse=j, duration=t, dir_idx=d,
                      rel=2.0 * math.pi * d / int(n_dirs), delay=delay))
  return cells


def cell_table(cells: Sequence[dict], device="cpu") -> dict[str, torch.Tensor]:
  """The cells as columns (``[n_cells]`` tensors), for indexing per env."""
  out = {}
  for k in ("impulse", "duration", "rel", "delay"):
    out[k] = torch.tensor([float(c[k]) for c in cells], device=device)
  out["dir_idx"] = torch.tensor([int(c["dir_idx"]) for c in cells],
                                dtype=torch.long, device=device)
  return out


def assign_cells(n_cells: int, start: int, count: int, device="cpu") -> torch.Tensor:
  """Cells of the episodes issued ``start .. start + count - 1``: round-robin
  over the issue order, ``(start + i) mod n_cells``. Any prefix of the order
  holds every cell ``floor(k / n_cells)`` or ``ceil(k / n_cells)`` times."""
  if int(n_cells) < 1:
    raise ValueError(f"n_cells must be >= 1, got {n_cells}")
  return (int(start) + torch.arange(int(count), device=device)) % int(n_cells)


# --- survival curves and F-levels -------------------------------------------------

def f_level(levels, survival, q: float = 0.5) -> tuple[float, str]:
  """The impulse at which survival FIRST falls to ``q``, by linear interpolation
  between the two grid levels around the crossing.

  Returns ``(value, flag)``. ``flag`` is ``"ok"`` for a crossing inside the
  grid; ``"<min"`` when survival is already below ``q`` at the smallest level
  and ``">max"`` when it never falls to ``q`` -- both with value NaN, since the
  level lies outside what was measured; ``"empty"`` without data. Levels are
  sorted first and non-finite entries dropped. Taking the FIRST crossing is the
  same as reading the running minimum of the curve, so a noisy upturn above a
  crossing does not move it.
  """
  x = np.asarray(levels, dtype=float).reshape(-1)
  s = np.asarray(survival, dtype=float).reshape(-1)
  keep = np.isfinite(x) & np.isfinite(s)
  x, s = x[keep], s[keep]
  if x.size == 0:
    return float("nan"), "empty"
  order = np.argsort(x, kind="stable")
  x, s = x[order], s[order]
  below = np.nonzero(s <= q)[0]
  if below.size == 0:
    return float("nan"), ">max"
  i = int(below[0])
  if i == 0:
    return (float(x[0]), "ok") if s[0] == q else (float("nan"), "<min")
  x0, x1, s0, s1 = x[i - 1], x[i], s[i - 1], s[i]
  return float(x0 + (s0 - q) / (s0 - s1) * (x1 - x0)), "ok"


def f50(levels, survival) -> tuple[float, str]:
  """:func:`f_level` at 50 % survival."""
  return f_level(levels, survival, 0.5)


def _curve(level: np.ndarray, ok: np.ndarray) -> dict:
  """Survival per unique level: aligned lists ``levels``, ``n``, ``fell``,
  ``surv``."""
  lv = np.unique(level[np.isfinite(level)])
  return {"levels": [float(v) for v in lv],
          "n": [int((level == v).sum()) for v in lv],
          "fell": [int((~ok[level == v]).sum()) for v in lv],
          "surv": [float(ok[level == v].mean()) for v in lv]}


# --- records ---------------------------------------------------------------------

def concat_records(parts: Sequence[dict]) -> dict:
  """Concatenate columnar record dicts (tensors are concatenated on the CPU,
  string lists extended). Every part must carry the same columns."""
  parts = [p for p in parts if p]
  if not parts:
    return {}
  keys = list(parts[0])
  for p in parts[1:]:
    if set(p) != set(keys):
      raise ValueError(f"record columns differ: {sorted(set(p) ^ set(keys))}")
  out = {}
  for k in keys:
    if k in STR_FIELDS or isinstance(parts[0][k], (list, tuple)):
      out[k] = [str(v) for p in parts for v in p[k]]
    else:
      out[k] = torch.cat([torch.as_tensor(p[k]).reshape(-1).cpu() for p in parts])
  return out


def _columns(records: dict) -> dict[str, np.ndarray]:
  cols = {}
  n = None
  for k, v in records.items():
    if k in STR_FIELDS or isinstance(v, (list, tuple)):
      a = np.asarray([str(x) for x in v], dtype=object)
    else:
      a = torch.as_tensor(v).reshape(-1).detach().cpu().double().numpy()
    if n is not None and a.shape[0] != n:
      raise ValueError(f"record column {k!r} has {a.shape[0]} rows, not {n}")
    n = a.shape[0]
    cols[k] = a
  return cols


def _ordered_unique(a: np.ndarray) -> list:
  seen = {}
  for v in a.tolist():
    seen.setdefault(v, None)
  return list(seen)


def _nanmedian(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(np.median(a)) if a.size else float("nan")


def _nanmean(a: np.ndarray) -> float:
  a = a[np.isfinite(a)]
  return float(a.mean()) if a.size else float("nan")


def _behaviour(cols: dict, m: np.ndarray, reach_s: float,
               tolerances: dict) -> dict:
  """Flat behaviour statistics of the episodes in ``m``."""
  out = {"n": int(m.sum()),
         "peak_speed_median": _nanmedian(cols["peak_speed"][m]),
         "dist_median": _nanmedian(cols["dist"][m])}
  for name in tolerances:
    if f"t_reach_{name}" not in cols:
      continue
    tr = cols[f"t_reach_{name}"][m]
    reached = np.isfinite(tr)
    k = max(int(m.sum()), 1)
    out[f"{name}_reach_rate"] = float(reached.sum()) / k
    out[f"{name}_reach_rate_within"] = float((reached & (tr <= reach_s + 1e-9)).sum()) / k
    out[f"{name}_t_reach_median"] = _nanmedian(tr[reached])
    out[f"{name}_stay_frac_mean"] = _nanmean(cols[f"stay_frac_{name}"][m][reached])
    far = cols[f"fell_after_reach_{name}"][m][reached]
    out[f"{name}_fell_after_reach"] = float(far.mean()) if far.size else float("nan")
  return out


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
  from robot_safety_sandbox.envs.g1_recover.labels import auc
  return auc(torch.as_tensor(score), torch.as_tensor(positive.astype(bool)))


def _group(cols: dict, m: np.ndarray, reach_s: float, tolerances: dict) -> dict:
  prefix = cols["prefix_fell"] > 0.5
  valid = m & ~prefix
  ok = cols["survived"] > 0.5
  imp = cols["impulse"]
  curve = _curve(imp[valid], ok[valid])
  res = {
    "n": int(m.sum()), "n_valid": int(valid.sum()),
    "survival": float(ok[valid].mean()) if valid.any() else float("nan"),
    "mean_surv_levels": (float(np.mean(curve["surv"])) if curve["surv"]
                         else float("nan")),
    "prefix_fall_rate": float(prefix[m].mean()) if m.any() else float("nan"),
    "curve": curve,
    "f": {f"f{int(round(q * 100))}": list(f_level(curve["levels"], curve["surv"], q))
          for q in (0.9, 0.5, 0.1)},
  }
  n_dirs = int(np.nanmax(cols["n_dirs"][m])) if "n_dirs" in cols else (
    int(np.nanmax(cols["dir_idx"][m])) + 1)
  splits = (("by_dir", cols["dir_idx"], lambda v: direction_label(int(v), n_dirs), True),
            ("by_duration", cols["duration"], lambda v: f"{v:g}", False),
            ("by_delay", cols["delay"], lambda v: f"{v:g}", False),
            ("by_category", cols.get("category"), str, False))
  for key, col, label, always in splits:
    if col is None:
      continue
    vals = sorted(v for v in _ordered_unique(col[valid]) if v not in ("", "-"))
    if len(vals) < (1 if always else 2):
      continue
    res[key] = {}
    for v in vals:
      g = valid & (col == v)
      c = _curve(imp[g], ok[g])
      res[key][label(v)] = {"n": int(g.sum()), "survival": float(ok[g].mean()),
                            "curve": c, "f50": list(f50(c["levels"], c["surv"]))}
  res["behaviour"] = _behaviour(cols, valid, reach_s, tolerances)
  by_imp = {"levels": curve["levels"]}
  for v in curve["levels"]:
    for k, x in _behaviour(cols, valid & (imp == v), reach_s, tolerances).items():
      by_imp.setdefault(k, []).append(x)
  res["behaviour_by_impulse"] = by_imp
  if "value0" in cols and np.isfinite(cols["value0"][valid]).any():
    v0 = cols["value0"]
    res["value_auc"] = _auc(v0[valid], ok[valid])
    res["value_auc_by_impulse"] = [_auc(v0[valid & (imp == v)], ok[valid & (imp == v)])
                                   for v in curve["levels"]]
  return res


def summarize(records: dict, reach_s: float = 3.0,
              tolerances: Optional[dict] = None) -> dict:
  """The nested summary of a record set: ``{"meta": ..., "settings": {setting:
  {policy: group}}}`` with settings and policies in order of appearance.

  A group carries ``n`` / ``n_valid`` (episodes; valid = not a handover replay
  that fell before the takeover), ``survival`` (pooled over the valid ones),
  ``mean_surv_levels`` (the mean of the per-level survival: the primary
  metric's average over the grid, insensitive to how many episodes each level
  happened to get), ``prefix_fall_rate``, ``curve`` (survival per impulse),
  ``f`` (``f90`` / ``f50`` / ``f10`` as ``[value, flag]``), ``by_dir`` /
  ``by_duration`` / ``by_delay`` / ``by_category`` (``{label: {n, survival,
  curve, f50}}``, a split with a single value left out except directions),
  ``behaviour`` and ``behaviour_by_impulse`` (see :func:`_behaviour`;
  ``<tol>_reach_rate_within`` counts reaches within ``reach_s`` of the count
  start), and, when the records carry ``value0``, ``value_auc`` (pooled) and
  ``value_auc_by_impulse``.
  """
  tolerances = TOLERANCES if tolerances is None else tolerances
  cols = _columns(records)
  out = {"meta": {"n_records": int(cols["survived"].shape[0]),
                  "reach_s": float(reach_s),
                  "tolerances": {k: list(v) for k, v in tolerances.items()},
                  "settings": _ordered_unique(cols["setting"]),
                  "policies": _ordered_unique(cols["policy"])},
         "settings": {}}
  for s in out["meta"]["settings"]:
    for p in out["meta"]["policies"]:
      m = (cols["setting"] == s) & (cols["policy"] == p)
      if m.any():
        out["settings"].setdefault(s, {})[p] = _group(cols, m, reach_s, tolerances)
  return out


# --- E1: freezing the grid ------------------------------------------------------

def propose_frozen_grid(summary: dict, n_levels: int = 5, setting: str = "lafan",
                        policy: Optional[str] = None, round_to: float = 0.5,
                        overlap_tol: float = 0.1) -> dict:
  """E1's frozen evaluation grid from a calibration summary.

  ``impulses``: ``n_levels`` levels evenly spanning F90..F10 of the POOLED
  curve of ``setting`` (E-lafan) for ``policy`` (default: the first policy of
  the run, i.e. the first ``--policy``), rounded to ``round_to`` N s. A level
  outside the calibration grid (flag ``<min`` / ``>max``) is replaced by the
  grid's end and named in ``warnings``: widen the calibration grid then.

  ``durations``: if the per-duration F50s agree within ``overlap_tol``
  (relative spread), impulse alone describes the push and the grid keeps the
  training duration (:data:`TRAIN_DURATION_S`) alone; otherwise every
  calibration duration stays. ``durations_overlap`` says which (None when
  fewer than two durations had a finite F50).

  ``force_max_suggested``: the training ceiling of E1's decision rule,
  :data:`CEILING_FACTOR` x the F10 at the training duration (pooled when that
  duration was not calibrated), as a force at that duration. It is a starting
  point: raise it if ``game/force_scale_mean`` sits near 1 in training.
  """
  groups = summary["settings"].get(setting)
  if not groups:
    raise ValueError(f"no {setting!r} results in this summary; the frozen grid "
                     f"is read off the {setting} setting")
  if policy is None:
    policy = next(p for p in summary["meta"]["policies"] if p in groups)
  g = groups[policy]
  levels = g["curve"]["levels"]
  warnings = []
  (f90, fl90), (f10, fl10) = g["f"]["f90"], g["f"]["f10"]
  lo, hi = f90, f10
  if not math.isfinite(lo):
    lo = levels[0] if fl90 == "<min" else levels[-1]
    warnings.append(f"F90 is {fl90} on the calibration grid; using {lo:g} N s")
  if not math.isfinite(hi):
    hi = levels[-1] if fl10 == ">max" else levels[0]
    warnings.append(f"F10 is {fl10} on the calibration grid; using {hi:g} N s")
  if hi < lo:
    lo, hi = hi, lo
  raw = np.linspace(lo, hi, int(n_levels)) if n_levels > 1 else np.array([lo])
  impulses = sorted({round(float(v) / round_to) * round_to for v in raw})
  f50_by_dur = {k: v["f50"] for k, v in g.get("by_duration", {}).items()}
  finite = [v[0] for v in f50_by_dur.values() if math.isfinite(v[0])]
  cal_durs = sorted({float(k) for k in f50_by_dur}) or [TRAIN_DURATION_S]
  overlap = None
  if len(finite) >= 2:
    spread = (max(finite) - min(finite)) / max(float(np.mean(finite)), 1e-9)
    overlap = bool(spread <= overlap_tol)
  if overlap:
    durations = ([TRAIN_DURATION_S] if any(abs(d - TRAIN_DURATION_S) < 1e-9
                                           for d in cal_durs)
                 else [cal_durs[len(cal_durs) // 2]])
  else:
    durations = cal_durs
  ref = g.get("by_duration", {}).get(f"{TRAIN_DURATION_S:g}")
  f10_train = (f_level(ref["curve"]["levels"], ref["curve"]["surv"], 0.1)
               if ref is not None else (f10, fl10))
  f10_ceiling = f10_train[0] if math.isfinite(f10_train[0]) else hi
  return {
    "impulses": impulses, "durations": durations,
    "source": {"setting": setting, "policy": policy, "f90": [f90, fl90],
               "f10": [f10, fl10]},
    "f50_by_duration": f50_by_dur, "durations_overlap": overlap,
    "force_max_suggested": float(CEILING_FACTOR * f10_ceiling / TRAIN_DURATION_S),
    "warnings": warnings,
  }


# --- comparing arms across seeds -------------------------------------------------

def arm_of(name: str) -> tuple[str, Optional[int]]:
  """``"S3_s0"`` -> ``("S3", 0)``; a name without a ``_s<seed>`` suffix is its
  own arm with seed None."""
  m = re.match(r"^(.*)_s(\d+)$", str(name))
  return (m.group(1), int(m.group(2))) if m else (str(name), None)


def _metric(group: dict, metric: str) -> float:
  if metric == "f50":
    return float(group["f"]["f50"][0])
  if metric.startswith("f50_"):                 # f50_<direction label>
    d = group.get("by_dir", {}).get(metric[4:])
    return float(d["f50"][0]) if d else float("nan")
  return float(group[metric])


def compare_arms(summary: dict, setting: str,
                 metrics: Sequence[str] = ("mean_surv_levels", "f50", "f50_behind")
                 ) -> dict:
  """Per ARM (policies grouped by :func:`arm_of`): for each metric the
  per-seed ``values``, their ``mean`` and the within-arm seed ``gap``
  (max - min; 0 with one seed). ``f50_<side>`` reads a per-direction F50."""
  groups = summary["settings"].get(setting, {})
  arms: dict = {}
  for p in summary["meta"]["policies"]:
    if p not in groups:
      continue
    arm, seed = arm_of(p)
    a = arms.setdefault(arm, {"seeds": [], "policies": []})
    a["seeds"].append(seed)
    a["policies"].append(p)
    for k in metrics:
      a.setdefault(k, {"values": []})["values"].append(_metric(groups[p], k))
  for a in arms.values():
    for k in metrics:
      v = np.asarray(a[k]["values"], dtype=float)
      fin = v[np.isfinite(v)]
      a[k]["mean"] = float(fin.mean()) if fin.size else float("nan")
      a[k]["gap"] = float(fin.max() - fin.min()) if fin.size else float("nan")
  return arms


def significant(a: dict, b: dict) -> bool:
  """The plan's decision rule: two arms' metric (``{"mean", "gap"}`` from
  :func:`compare_arms`) differ only if the difference of the means exceeds
  the LARGER within-arm seed gap. Otherwise the simpler arm wins."""
  d = abs(float(a["mean"]) - float(b["mean"]))
  gap = max(float(a["gap"]), float(b["gap"]))
  return bool(math.isfinite(d) and d > gap)
