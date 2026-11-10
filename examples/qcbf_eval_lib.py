"""Pure helpers of the Q-CBF evaluation on SONIC (``eval_sonic_qcbf.py``).

The driver needs a simulator, SONIC, a fallback policy and its fitted values,
and none of that runs on a login node. Everything that decides what the
numbers MEAN lives here instead, in torch and the standard library only, so
``tests/test_eval_sonic_qcbf.py`` checks it on fabricated per-episode records
in seconds:

* the CONDITION grammar and the :class:`Controller` that turns a condition,
  its intervention and its monitor into this step's action plus a telemetry
  record. The interventions are passed in, so a test can pass fakes:

      sonic                          SONIC alone, unfiltered
      fallback                       the fallback policy pi alone, from the spawn
      qcbf:<variant>:<kappa>[:<solver>]
                                     SONIC through QCBFDescentIntervention on
                                     pi's fitted Q / V of fit ``variant``
                                     (``exact``, ``td_g099``), threshold
                                     kappa V(x), ``solver`` gd+segment (default)
                                     | gd | segment
      lrsf:<variant>                 the last-resort switch: SONIC while
                                     Q(x, u_SONIC) > 0, else pi
                                     (LeastRestrictiveIntervention, eps 0)
      rollout:<seconds>[:<R>]        the rollout filter: SONIC while a
                                     simulated rollout (u_SONIC for one step,
                                     then pi, <seconds> in all) keeps g >= 0,
                                     else pi (``cpu_rollout.py``; the driver
                                     passes the rollout's min g as the verdict).
                                     With R: R rollouts, pi reading each
                                     through the sensor-noise law, and the
                                     worst one decides
      wmrollout:<seconds>[:<R>]      the same filter with the rollout imagined
                                     in pi's own learned world model (a SHAC
                                     checkpoint) instead of simulated
      wmtube:<seconds>:<R>:<c>       the world-model filter on a CALIBRATED
                                     tube: SONIC while min over the R noisy
                                     rollouts of the predicted margin, minus
                                     eps_j (per step, coverage c, from the
                                     driver's --tube-calib), stays >= 0
      wmcalib:<R> | wmcalib:clean    calibration episodes for it: SONIC up to
                                     a random decision step, where R noisy
                                     world-model rollouts (clean: one
                                     noise-free rollout) of "u_SONIC, then pi"
                                     are recorded, and pi from the next step,
                                     whose real margins are recorded
      wmmin:<seconds>:<c>            the world-model filter on a conformal
                                     bound of the WORST margin: SONIC while
                                     min over j <= H of the one noise-free
                                     rollout's predicted margin, minus eps_H(c)
                                     (--tube-calib, calibrate_wm_min_tube.py),
                                     stays >= 0

* the seeded EPISODE LISTS. Spawn rows are drawn uniformly among the rows
  with room for the whole episode, then onsets and pushes. There is one list
  per suite (and per bad-reference scenario), drawn once and shared by every
  condition;
* :class:`EpisodeBook`, per-env accumulators split at each episode's
  ``split`` step (PRE: normal tracking, before the push lands or before SONIC
  can see a corrupted frame; POST: from then on) and copied into per-episode
  records when the episode ends. :func:`step_outcomes` is the harvester's
  fall / survive rule;
* the STATISTICS: Wilson intervals, the exact McNemar test on the episodes a
  condition rescued or broke relative to unfiltered SONIC (the same episodes,
  so the pairs are what carries the comparison), survival by |dv| bin and by
  category, and intervention rates and deviations per phase. These are the
  suite summaries the driver writes to ``push.json`` / ``badref.json``, and
  ``summarize_sonic_qcbf.py`` reads them back.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Callable, Optional, Sequence

import torch

#: Telemetry phases: before each episode's split step, and from it on.
PHASES = ("pre", "post")
#: Per-step filter status (QCBFDescentIntervention.STATUS). The switch and the
#: fallback-only condition book the fallback's action as ``best_effort``.
STATUS = ("pass", "gd", "segment", "best_effort")
PASS, GD, SEGMENT, BEST_EFFORT = range(4)
#: |dv| bins of the survival breakdown, m/s; the last bin is closed.
DV_BINS = (0.0, 1.5, 2.5, 3.5, 5.0)
KINDS = ("sonic", "fallback", "qcbf", "lrsf", "rollout", "wmrollout", "wmtube", "wmcalib", "wmmin")
#: The kinds that switch on a rollout verdict (the driver computes it).
ROLLOUT_KINDS = ("rollout", "wmrollout")
VERDICT_KINDS = ROLLOUT_KINDS + ("wmtube", "wmcalib", "wmmin")
#: ``<condition>+latch``: once the verdict engages pi, pi keeps the episode.
LATCH_SUFFIX = "+latch"
#: ``<condition>+dwell<seconds>``: each hand-over to pi holds for at least that long.
DWELL_SUFFIX = "+dwell"
#: QCBFDescentIntervention.SOLVERS; the first is its default.
SOLVERS = ("gd+segment", "gd", "segment")
DEFAULT_SOLVER = SOLVERS[0]
#: The condition the paired statistics compare against.
REFERENCE = "sonic"

#: Default condition lists (the spec's, plus pi alone on the push suite).
PUSH_CONDITIONS = (
  "sonic", "fallback", "qcbf:exact:0.7", "qcbf:td_g099:0.7", "lrsf:exact", "lrsf:td_g099",
  "qcbf:exact:0.0", "qcbf:exact:0.5", "qcbf:exact:0.9",
  "qcbf:td_g099:0.0", "qcbf:td_g099:0.5", "qcbf:td_g099:0.9",
  "qcbf:exact:0.7:gd", "qcbf:exact:0.7:segment")
BADREF_CONDITIONS = ("sonic", "qcbf:exact:0.7", "qcbf:td_g099:0.7", "lrsf:exact", "lrsf:td_g099")
#: ``--quick``: every controller kind and both fits, on few episodes.
QUICK_PUSH_CONDITIONS = ("sonic", "qcbf:exact:0.7", "lrsf:td_g099", "fallback")
QUICK_BADREF_CONDITIONS = ("sonic", "qcbf:exact:0.7", "qcbf:td_g099:0.7")

_NAN = float("nan")


# --- conditions ------------------------------------------------------------------

def kappa_str(k: float) -> str:
  s = f"{k:.3f}".rstrip("0")
  return s + "0" if s.endswith(".") else s


@dataclass(frozen=True)
class Condition:
  """One controller configuration (module docstring). ``name`` is canonical:
  ``qcbf:exact:0.70`` and ``qcbf:exact:0.7:gd+segment`` both read
  ``qcbf:exact:0.7``."""
  name: str
  kind: str
  variant: Optional[str] = None
  kappa: Optional[float] = None
  solver: Optional[str] = None
  horizon_s: Optional[float] = None
  #: rollout kinds: noisy rollouts per check (None: one noise-free rollout)
  n_rollouts: Optional[int] = None
  #: wmtube: the tube's coverage level c (eps_j calibrated at it)
  coverage: Optional[float] = None
  #: verdict kinds: after the first engaged step pi acts for the rest of the
  #: episode (the driver overrides the verdict; ``<condition>+latch``)
  latch: bool = False
  #: verdict kinds: a minimum dwell on pi, s: from each hand-over to pi, pi acts
  #: for at least this long before the verdict may hand back (``+dwell<s>``)
  dwell_s: Optional[float] = None

  @property
  def needs_values(self) -> bool:
    return self.kind in ("qcbf", "lrsf")

  @property
  def needs_fallback(self) -> bool:
    return self.kind != "sonic"


def parse_condition(text: str) -> Condition:
  """:class:`Condition` of ``text``; ValueError on anything else."""
  usage = ("one of sonic, fallback, qcbf:<variant>:<kappa>[:<solver>] "
           f"(solver {' | '.join(SOLVERS)}), lrsf:<variant>, rollout:<seconds>[:<R>], "
           "wmrollout:<seconds>[:<R>], wmtube:<seconds>:<R>:<c>, wmcalib:<R> | wmcalib:clean, "
           "wmmin:<seconds>:<c>; any verdict kind but wmcalib takes one +latch or "
           "+dwell<seconds> suffix")
  text = str(text).strip()
  head, sep, tail = text.rpartition("+")
  if sep and ("+" + tail == LATCH_SUFFIX or ("+" + tail).startswith(DWELL_SUFFIX)):
    c = parse_condition(head)
    if c.kind not in VERDICT_KINDS or c.kind == "wmcalib" or c.latch or c.dwell_s is not None:
      raise ValueError(f"condition {text!r}: {LATCH_SUFFIX} / {DWELL_SUFFIX}<s> need one verdict "
                       f"kind ({', '.join(k for k in VERDICT_KINDS if k != 'wmcalib')})")
    if "+" + tail == LATCH_SUFFIX:
      return replace(c, name=c.name + LATCH_SUFFIX, latch=True)
    try:
      d = float(tail[len(DWELL_SUFFIX) - 1:])
    except ValueError:
      raise ValueError(f"condition {text!r}: {DWELL_SUFFIX}<seconds> ({usage})") from None
    if not (math.isfinite(d) and d > 0.0):
      raise ValueError(f"condition {text!r}: the dwell must be > 0 s")
    return replace(c, name=f"{c.name}{DWELL_SUFFIX}{d:g}", dwell_s=d)
  parts = text.split(":")
  kind = parts[0]
  if kind in ("sonic", "fallback"):
    if len(parts) != 1:
      raise ValueError(f"condition {text!r}: {kind} takes no arguments ({usage})")
    return Condition(kind, kind)
  if kind == "lrsf":
    if len(parts) != 2 or not parts[1]:
      raise ValueError(f"condition {text!r}: lrsf:<variant> ({usage})")
    return Condition(f"lrsf:{parts[1]}", "lrsf", variant=parts[1])
  if kind == "wmcalib":
    if len(parts) == 2 and parts[1] == "clean":
      return Condition("wmcalib:clean", "wmcalib")
    if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) < 1:
      raise ValueError(f"condition {text!r}: wmcalib:<R> | wmcalib:clean ({usage})")
    return Condition(f"wmcalib:{int(parts[1])}", "wmcalib", n_rollouts=int(parts[1]))
  if kind == "wmmin":
    try:
      if len(parts) != 3:
        raise IndexError
      h, c = float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
      raise ValueError(f"condition {text!r}: wmmin:<seconds>:<c> ({usage})") from None
    if not (math.isfinite(h) and h > 0.0 and 0.0 < c < 1.0):
      raise ValueError(f"condition {text!r}: horizon > 0, 0 < c < 1")
    return Condition(f"wmmin:{h:g}:{c:g}", "wmmin", horizon_s=h, coverage=c)
  if kind == "wmtube":
    try:
      h, r, c = float(parts[1]), int(parts[2]), float(parts[3])
      if len(parts) != 4:
        raise IndexError
    except (IndexError, ValueError):
      raise ValueError(f"condition {text!r}: wmtube:<seconds>:<R>:<c> ({usage})") from None
    if not (math.isfinite(h) and h > 0.0 and r >= 1 and 0.0 < c < 1.0):
      raise ValueError(f"condition {text!r}: horizon > 0, R >= 1, 0 < c < 1")
    return Condition(f"wmtube:{h:g}:{r}:{c:g}", "wmtube", horizon_s=h, n_rollouts=r, coverage=c)
  if kind in ROLLOUT_KINDS:
    if len(parts) not in (2, 3):
      raise ValueError(f"condition {text!r}: {kind}:<seconds>[:<R>] ({usage})")
    try:
      h = float(parts[1])
    except ValueError:
      raise ValueError(f"condition {text!r}: horizon {parts[1]!r} is not a number") from None
    if not (math.isfinite(h) and h > 0.0):
      raise ValueError(f"condition {text!r}: the horizon must be > 0 s")
    r = None
    if len(parts) == 3:
      try:
        r = int(parts[2])
      except ValueError:
        raise ValueError(f"condition {text!r}: rollouts {parts[2]!r} is not an integer") from None
      if r < 1:
        raise ValueError(f"condition {text!r}: at least one rollout")
    return Condition(f"{kind}:{h:g}" + ("" if r is None else f":{r}"), kind, horizon_s=h,
                     n_rollouts=r)
  if kind == "qcbf":
    if len(parts) not in (3, 4) or not parts[1]:
      raise ValueError(f"condition {text!r}: qcbf:<variant>:<kappa>[:<solver>] ({usage})")
    try:
      kappa = float(parts[2])
    except ValueError:
      raise ValueError(f"condition {text!r}: kappa {parts[2]!r} is not a number") from None
    if not (math.isfinite(kappa) and 0.0 <= kappa <= 1.0):
      # kappa > 1 asks Q(x, u) to beat V(x) = Q(x, pi(x)): even the fallback's
      # own action would be infeasible
      raise ValueError(f"condition {text!r}: kappa must lie in [0, 1]")
    solver = parts[3] if len(parts) == 4 else DEFAULT_SOLVER
    if solver not in SOLVERS:
      raise ValueError(f"condition {text!r}: solver {solver!r} ({usage})")
    name = f"qcbf:{parts[1]}:{kappa_str(kappa)}" + ("" if solver == DEFAULT_SOLVER else f":{solver}")
    return Condition(name, "qcbf", parts[1], kappa, solver)
  raise ValueError(f"unknown condition {text!r}: {usage}")


def parse_conditions(texts: Sequence[str]) -> list[Condition]:
  """Parse a list, refusing duplicates (after canonicalization)."""
  out, seen = [], set()
  for t in texts:
    c = parse_condition(t)
    if c.name in seen:
      raise ValueError(f"condition {c.name!r} listed twice")
    seen.add(c.name)
    out.append(c)
  if not out:
    raise ValueError("no conditions")
  return out


def reference_first(conds: Sequence[Condition]) -> list[Condition]:
  """``conds`` with :data:`REFERENCE` moved to the front, the rest in order.
  Every other condition's paired statistics need SONIC's outcomes on the
  same episodes, and the driver summarizes each condition when it finishes."""
  return sorted(conds, key=lambda c: c.name != REFERENCE)


# --- the controller ------------------------------------------------------------------

class Controller:
  """This step's action for the tracked envs under one :class:`Condition`.

  ``intervention`` / ``monitor`` are the filter modules (a
  ``QCBFDescentIntervention`` or ``LeastRestrictiveIntervention``, and a
  ``PolicyValueMonitor``), required for qcbf / lrsf and unused otherwise.
  ``__call__(a_nom, a_safe, x, verdict=None)`` takes SONIC's env action, the
  fallback's (``None`` under ``sonic``), the value features ``x`` of the same
  envs and, under ``rollout``, the rollout's min g (``verdict``; it is also
  what the record's ``v`` holds there).
  It returns the applied action and the telemetry record, all ``(N,)``:
  ``engaged`` (the action is not SONIC's), ``status`` (:data:`STATUS`),
  ``deviation`` = ||u - u_SONIC||, ``v`` = V(x), ``q`` = Q(x, u),
  ``q_nom`` = Q(x, u_SONIC) (NaN without values), ``iters`` (descent
  iterations), ``q_safe`` = Q(x, pi(x)) (NaN where not evaluated),
  ``fb_infeasible`` (an engaged step where pi's own action fails the
  constraint: Q(x, pi(x)) < tau for qcbf, <= eps for the switch),
  ``tr_bound`` (the solver's trust region bound) and ``seg_gap`` (on a step
  the descent won: the segment candidate's distance minus the descent's; NaN
  otherwise, or when the segment found nothing)."""

  def __init__(self, cond: Condition, intervention=None, monitor=None):
    if cond.kind not in KINDS:
      raise ValueError(f"unknown controller kind {cond.kind!r}")
    if cond.needs_values and (intervention is None or monitor is None):
      raise ValueError(f"{cond.name}: needs an intervention and a monitor")
    self.cond, self.itv, self.mon = cond, intervention, monitor

  @property
  def needs_values(self) -> bool:
    return self.cond.needs_values

  @property
  def needs_fallback(self) -> bool:
    return self.cond.needs_fallback

  def __call__(self, a_nom: torch.Tensor, a_safe: Optional[torch.Tensor],
               x: Optional[torch.Tensor], verdict: Optional[torch.Tensor] = None):
    n, dev = int(a_nom.shape[0]), a_nom.device
    nan = torch.full((n,), _NAN, device=dev)
    zero_l = torch.zeros(n, dtype=torch.long, device=dev)
    no = torch.zeros(n, dtype=torch.bool, device=dev)
    none = dict(q_safe=nan, fb_infeasible=no, tr_bound=no, seg_gap=nan)
    kind = self.cond.kind
    if kind == "sonic":
      return a_nom, dict(engaged=no, status=zero_l, deviation=torch.zeros(n, device=dev),
                         v=nan, q=nan, q_nom=nan, iters=zero_l, **none)
    if a_safe is None or a_safe.shape != a_nom.shape:
      raise ValueError(f"{self.cond.name}: fallback action "
                       f"{None if a_safe is None else tuple(a_safe.shape)} does not match "
                       f"SONIC's {tuple(a_nom.shape)}")
    if kind == "fallback":
      return a_safe, dict(engaged=torch.ones(n, dtype=torch.bool, device=dev),
                          status=torch.full_like(zero_l, BEST_EFFORT),
                          deviation=(a_safe - a_nom).norm(dim=-1), v=nan, q=nan, q_nom=nan,
                          iters=zero_l, **none)
    if kind in VERDICT_KINDS:
      # the rollout's min g is the verdict: below 0, u_SONIC is not certified
      if verdict is None or tuple(verdict.shape) != (n,):
        raise ValueError(f"{self.cond.name}: rollout verdict "
                         f"{None if verdict is None else tuple(verdict.shape)} for {n} envs")
      eng = verdict < 0.0
      action = torch.where(eng.unsqueeze(-1), a_safe, a_nom)
      return action, dict(engaged=eng, status=torch.where(eng, BEST_EFFORT, PASS).long(),
                          deviation=(action - a_nom).norm(dim=-1), v=verdict.float(), q=nan,
                          q_nom=nan, iters=zero_l, **none)
    if x is None or int(x.shape[0]) != n:
      raise ValueError(f"{self.cond.name}: value features {None if x is None else tuple(x.shape)} "
                       f"for {n} envs")
    fallback = lambda **_ctx: a_safe  # noqa: E731 - the interventions call fallback(**ctx)
    action, info = self.itv(a_nom, fallback, self.mon, x=x)
    if kind == "qcbf":
      last = self.itv.last
      eng, st = info.engaged.bool(), last["status"].long()
      dev_u = last["deviation"].float()
      q_safe = last.get("q_safe", nan).float()
      d_seg = last.get("dist_seg", nan).float()
      won = (st == GD) & torch.isfinite(d_seg)
      return action, dict(engaged=eng, status=st, deviation=dev_u, v=last["v"].float(),
                          q=last["q"].float(), q_nom=last["q_nom"].float(),
                          iters=last["gd_iters_used"].long(), q_safe=q_safe,
                          fb_infeasible=eng & (q_safe < last.get("tau", nan).float()),
                          tr_bound=last.get("tr_bound", no).bool() & eng,
                          seg_gap=torch.where(won, d_seg - dev_u, nan))
    # lrsf: the switch's value is Q(x, u_SONIC); V and Q(x, pi(x)) for the record
    eng = info.engaged.bool()
    with torch.no_grad():
      v = self.mon.v_fn(x=x).float()
      q_safe = self.mon(a_safe, x=x).float()
    q_nom = info.value.float()
    eps = float(getattr(self.itv, "eps", 0.0))
    return action, dict(engaged=eng, status=torch.where(eng, BEST_EFFORT, 0).long(),
                        deviation=(action - a_nom).norm(dim=-1), v=v,
                        q=torch.where(eng, q_safe, q_nom), q_nom=q_nom, iters=zero_l,
                        q_safe=q_safe, fb_infeasible=eng & (q_safe <= eps), tr_bound=no,
                        seg_gap=nan)


# --- episode lists -----------------------------------------------------------------------

def _uniform(gen: torch.Generator, n: int, lo: float, hi: float) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(n, generator=gen, device=gen.device)


def draw_rows(pool: torch.Tensor, room: torch.Tensor, need: torch.Tensor,
              u: torch.Tensor) -> torch.Tensor:
  """For each episode e, the row of ``pool`` (bank row ids) at quantile
  ``u[e]`` among the rows whose ``room`` (frames left in the clip) is at
  least ``need[e]``: uniform over the eligible rows. ValueError when an
  episode has none."""
  order = torch.argsort(room, stable=True)
  r_sorted, p_sorted = room[order], pool[order]
  first = torch.searchsorted(r_sorted, need.to(r_sorted.dtype))
  count = int(r_sorted.numel()) - first
  if bool((count <= 0).any()):
    raise ValueError(f"no spawn row has room for {int(need[count <= 0].max())} frames "
                     f"(the longest clip leaves {int(r_sorted.max())})")
  j = first + torch.minimum((u * count).long(), count - 1)
  return p_sorted[j]


def draw_episodes(pool: torch.Tensor, room: torch.Tensor, episodes: int,
                  gen: torch.Generator, dt: float, onset_s: Sequence[float], window: int,
                  look_ahead: int, dv_range: Sequence[float],
                  fits: Optional[Callable] = None, max_rounds: int = 64) -> dict:
  """A seeded episode list (``[E]`` tensors on the generator's device).

  Per episode: ``on`` (the onset, control steps, U(onset_s) / dt rounded,
  >= 1), the push ``dv`` ~ U(dv_range) m/s at ``rel`` ~ U(-pi, pi) in the
  heading frame (``dv_h`` (E, 3), the harvester's kick, z = 0), then the spawn
  ``row``: uniform among the ``pool`` rows with ``room >= on + window +
  look_ahead`` frames, so the tracked frames and SONIC's look-ahead window stay
  in the clip. ``fits(rows, on) -> bool[E]`` (a bad-reference scenario's
  :meth:`BadReferenceFactory.fits`) rejects rows. Rejected rows are redrawn
  for those episodes only, at most ``max_rounds`` times, and then a
  ValueError. Draws come in a fixed order: onsets, |dv|, angles, rows. The
  push therefore does not depend on how many rows were rejected."""
  E = int(episodes)
  on = torch.round(_uniform(gen, E, float(onset_s[0]), float(onset_s[1])) / dt).long().clamp_min(1)
  mag = _uniform(gen, E, float(dv_range[0]), float(dv_range[1]))
  rel = _uniform(gen, E, -math.pi, math.pi)
  need = on + int(window) + int(look_ahead)
  pool, room = pool.to(on.device), room.to(on.device)
  rows = draw_rows(pool, room, need, torch.rand(E, generator=gen, device=gen.device))
  rounds = 0
  if fits is not None:
    bad = ~fits(rows, on)
    while bool(bad.any()):
      rounds += 1
      if rounds > max_rounds:
        raise ValueError(f"{int(bad.sum())} episodes found no spawn row that fits in "
                         f"{max_rounds} redraws")
      redo = draw_rows(pool, room, need, torch.rand(E, generator=gen, device=gen.device))
      rows = torch.where(bad, redo, rows)
      bad = ~fits(rows, on)
  dv_h = torch.stack([mag * torch.cos(rel), mag * torch.sin(rel), torch.zeros_like(mag)], 1)
  return {"row": rows, "on": on, "dv": mag, "rel": rel, "dv_h": dv_h, "redraw_rounds": rounds}


# --- per-episode telemetry ---------------------------------------------------------------

#: Per-phase sums (and the max of the deviation) of :class:`EpisodeBook`.
#: ``q_safe_sum``: Q(x, pi(x)) over the engaged steps; ``fb_infeasible`` /
#: ``tr_bound``: step counts (Controller); ``seg_gap_sum`` / ``seg_gap_n``: the
#: segment's excess distance over the descent's, on the steps the descent won.
TEL_KEYS = ("steps", "engaged", "dev_sum", "dev_max", "v_sum", "q_sum", "q_nom_sum",
            "iters_sum", "jerr_sum", "jerr_bad_sum", "q_safe_sum", "fb_infeasible",
            "tr_bound", "seg_gap_sum", "seg_gap_n", "switches")


def step_outcomes(track: torch.Tensor, dones: torch.Tensor, timeouts: torch.Tensor,
                  k: torch.Tensor, on: torch.Tensor, window: int) -> dict:
  """The harvester's rule after a step (``k`` = steps since the spawn,
  counted after it): ``fell`` = a tracked env the env ended on a termination
  (``after = k - on <= 0``: before the push landed, which lands after the step
  that reaches the onset and only on an env still standing); ``up`` = a
  tracked env still standing ``window`` steps after the onset (survived);
  ``cut`` = a tracked env the env TRUNCATED (its own time cap). The episode
  length is set so that ``cut`` never happens, and the driver treats it as an
  error."""
  after = k - on
  return {"fell": track & dones & ~timeouts, "cut": track & dones & timeouts,
          "up": track & ~dones & (after >= int(window)), "after": after}


class EpisodeBook:
  """Per-env telemetry accumulators and the per-episode records they end in.

  ``start(mask)`` zeroes the envs that begin an episode. ``step(ids, phase,
  tel, ...)`` adds one control step of the envs ``ids`` (unique) to the
  ``phase`` (0 pre, 1 post) sums: steps, engaged steps, the deviation's sum
  and max, the status counts, the sums of V, Q(x, u), Q(x, u_SONIC), descent
  iterations and the joint tracking error (against the true reference, and
  for the bad-reference suite against the corrupted one). ``finish(mask, ep,
  survived, k, on)`` books the outcome of the envs in ``mask`` into their
  episodes ``ep``: survived, fell before the push (``k - on <= 0``),
  ``time_to_fall`` (s from the onset, negative before it, NaN when it
  survived), steps run, the env slot, the first engaged step (s from the
  onset), and the env's sums. ``records()`` returns them on the CPU, ``[E]``
  and ``[E, 2]`` (``tel_status`` ``[E, 2, 4]``)."""

  def __init__(self, num_envs: int, episodes: int, dt: float, device="cpu"):
    n, E, self.dt = int(num_envs), int(episodes), float(dt)
    f32 = dict(dtype=torch.float32, device=device)
    self.acc = {k: torch.zeros(n, 2, **f32) for k in TEL_KEYS}
    self.acc_status = torch.zeros(n, 2, len(STATUS), dtype=torch.long, device=device)
    self.first = torch.full((n,), _NAN, **f32)
    self.rec = {k: torch.zeros(E, 2, **f32) for k in TEL_KEYS}
    self.rec_status = torch.zeros(E, 2, len(STATUS), dtype=torch.long, device=device)
    self.survived = torch.zeros(E, dtype=torch.bool, device=device)
    self.fell_pre = torch.zeros(E, dtype=torch.bool, device=device)
    self.ttf = torch.full((E,), _NAN, **f32)
    self.steps_run = torch.zeros(E, dtype=torch.long, device=device)
    self.env_id = torch.full((E,), -1, dtype=torch.long, device=device)
    self.first_engaged = torch.full((E,), _NAN, **f32)
    self.done = torch.zeros(E, dtype=torch.bool, device=device)

  def start(self, mask: torch.Tensor) -> None:
    for t in self.acc.values():
      t[mask] = 0.0
    self.acc_status[mask] = 0
    self.first[mask] = _NAN

  def step(self, ids: torch.Tensor, phase: torch.Tensor, tel: dict,
           jerr: Optional[torch.Tensor] = None, jerr_bad: Optional[torch.Tensor] = None,
           rel_steps: Optional[torch.Tensor] = None) -> None:
    n = int(ids.numel())
    if n == 0:
      return
    nan = torch.full((n,), _NAN, device=ids.device)

    def add(key, val):
      self.acc[key][ids, phase] += val.float()

    eng = tel["engaged"].bool()
    dev = tel["deviation"].float()
    add("steps", torch.ones(n, device=ids.device))
    add("engaged", eng.float())
    add("dev_sum", dev)
    self.acc["dev_max"][ids, phase] = torch.maximum(self.acc["dev_max"][ids, phase], dev)
    add("v_sum", tel["v"])
    add("q_sum", tel["q"])
    add("q_nom_sum", tel["q_nom"])
    add("iters_sum", tel["iters"])
    add("jerr_sum", nan if jerr is None else jerr)
    add("jerr_bad_sum", nan if jerr_bad is None else jerr_bad)
    zero = torch.zeros(n, device=ids.device)
    q_safe = tel.get("q_safe")
    add("q_safe_sum", nan if q_safe is None else torch.where(eng, q_safe.float(), zero))
    for key in ("fb_infeasible", "tr_bound"):
      add(key, zero if tel.get(key) is None else tel[key].float())
    gap = tel.get("seg_gap")
    ok = torch.zeros(n, dtype=torch.bool, device=ids.device) if gap is None else torch.isfinite(gap)
    add("seg_gap_sum", zero if gap is None else torch.where(ok, gap.float(), zero))
    add("seg_gap_n", ok.float())
    # a hand-over either way (SONIC <-> the filter's action); the driver sets it
    add("switches", zero if tel.get("switch") is None else tel["switch"].float())
    self.acc_status[ids, phase, tel["status"].long()] += 1
    if rel_steps is not None:
      fe = self.first[ids]
      self.first[ids] = torch.where(eng & torch.isnan(fe), rel_steps.float() * self.dt, fe)

  def finish(self, mask: torch.Tensor, ep: torch.Tensor, survived: bool,
             k: torch.Tensor, on: torch.Tensor) -> None:
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    e = ep[ids]
    if bool((e < 0).any()) or bool(self.done[e].any()):
      raise RuntimeError("an idle env, or an episode already booked, finished an episode")
    after = k[ids] - on[ids]
    self.survived[e] = bool(survived)
    self.fell_pre[e] = (not survived) & (after <= 0)
    self.ttf[e] = (torch.full(after.shape, _NAN, device=after.device) if survived
                   else after.float() * self.dt)
    self.steps_run[e] = k[ids]
    self.env_id[e] = ids
    self.first_engaged[e] = self.first[ids]
    for key in TEL_KEYS:
      self.rec[key][e] = self.acc[key][ids]
    self.rec_status[e] = self.acc_status[ids]
    self.done[e] = True

  def records(self) -> dict:
    if not bool(self.done.all()):
      raise RuntimeError(f"{int((~self.done).sum())} episodes never finished")
    out = {"survived": self.survived, "fell_pre": self.fell_pre, "time_to_fall": self.ttf,
           "steps": self.steps_run, "env_id": self.env_id, "first_engaged_s": self.first_engaged,
           "tel_status": self.rec_status}
    out.update({"tel_" + k: v for k, v in self.rec.items()})
    return {k: v.detach().cpu() for k, v in out.items()}


# --- statistics ----------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
  """Wilson score interval of k successes in n trials (95% by default);
  (NaN, NaN) for n = 0."""
  k, n = int(k), int(n)
  if n <= 0:
    return (_NAN, _NAN)
  p = k / n
  z2 = z * z
  den = 1.0 + z2 / n
  centre = (p + z2 / (2 * n)) / den
  half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / den
  return (max(0.0, centre - half), min(1.0, centre + half))


def binom_half_cdf(m: int, n: int) -> float:
  """P(X <= m) for X ~ Binomial(n, 1/2), in log space (n into the 1e5s)."""
  m, n = int(m), int(n)
  if m < 0:
    return 0.0
  if m >= n:
    return 1.0
  base = math.lgamma(n + 1) - n * math.log(2.0)
  logs = [base - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(m + 1)]
  top = max(logs)
  return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def mcnemar_exact_p(b: int, c: int) -> float:
  """Two-sided exact McNemar p of b vs c discordant pairs: under H0 each
  discordant pair goes either way with probability 1/2, so
  p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2); 1 when b + c = 0."""
  b, c = int(b), int(c)
  if b + c == 0:
    return 1.0
  return min(1.0, 2.0 * binom_half_cdf(min(b, c), b + c))


def survival(survived: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
  """``{n, k, sr, ci}`` of the episodes in ``mask`` (all by default)."""
  s = survived.bool()
  if mask is not None:
    s = s[mask.bool()]
  n, k = int(s.numel()), int(s.sum())
  return {"n": n, "k": k, "sr": (k / n) if n else _NAN, "ci": list(wilson_ci(k, n))}


def paired(ref: torch.Tensor, survived: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
  """The same episodes under the reference condition (``ref``, SONIC) and
  this one: ``rescued`` (the reference fell, this survived), ``broken`` (the
  reverse), the concordant counts, the SR difference and the exact McNemar p
  on rescued vs broken."""
  r, s = ref.bool(), survived.bool()
  if r.shape != s.shape:
    raise ValueError(f"paired: {tuple(r.shape)} reference outcomes vs {tuple(s.shape)}")
  if mask is not None:
    r, s = r[mask.bool()], s[mask.bool()]
  n = int(s.numel())
  b, c = int((~r & s).sum()), int((r & ~s).sum())
  return {"n": n, "rescued": b, "broken": c, "both_survived": int((r & s).sum()),
          "both_fell": int((~r & ~s).sum()),
          "delta_sr": ((b - c) / n) if n else _NAN, "mcnemar_p": mcnemar_exact_p(b, c)}


def bin_labels(edges: Sequence[float]) -> list[str]:
  return [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]


def bin_index(values: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
  """The bin of each value, ``[e_i, e_i+1)`` with the last bin closed; -1
  outside ``[e_0, e_-1]``."""
  e = torch.as_tensor(list(edges), dtype=torch.float64)
  v = values.double().cpu()
  idx = torch.bucketize(v, e, right=True) - 1
  idx = torch.where(v == e[-1], torch.full_like(idx, len(edges) - 2), idx)
  bad = (idx < 0) | (idx > len(edges) - 2) | torch.isnan(v)
  return torch.where(bad, torch.full_like(idx, -1), idx)


def by_bins(values: torch.Tensor, survived: torch.Tensor, edges: Sequence[float],
            ref: Optional[torch.Tensor] = None) -> dict:
  """:func:`survival` (and :func:`paired` against ``ref``) per bin of ``values``."""
  idx = bin_index(values, edges)
  out = {}
  for i, label in enumerate(bin_labels(edges)):
    m = idx == i
    out[label] = survival(survived, m)
    if ref is not None:
      out[label]["paired"] = paired(ref, survived, m)
  return out


def by_category(category: torch.Tensor, survived: torch.Tensor, names: Sequence[str],
                ref: Optional[torch.Tensor] = None) -> dict:
  """:func:`survival` per category id present (``names[id]``; -1 is "-")."""
  out = {}
  cat = category.long().cpu()
  for c in sorted(set(cat.tolist())):
    name = names[c] if 0 <= c < len(names) else "-"
    m = cat == c
    out[name] = survival(survived, m)
    if ref is not None:
      out[name]["paired"] = paired(ref, survived, m)
  return out


def _ratio(num, den) -> float:
  num, den = float(num), float(den)
  return num / den if den > 0 else _NAN


def intervention_stats(rec: dict, ref: Optional[torch.Tensor] = None) -> dict:
  """Filter telemetry of one condition, pooled over its episodes, per phase
  (``_pre`` / ``_post`` keys): ``rate`` = engaged steps / steps,
  ``dev_per_step`` and ``dev_per_engaged`` = mean ||u - u_SONIC|| per step and
  per engaged step, ``dev_max`` (the largest) and ``dev_max_mean`` (the mean
  over the episodes that intervened of their largest), ``status`` (the
  fraction of steps of each :data:`STATUS`), ``best_effort_rate`` (per step)
  and ``best_effort_per_engaged``, ``gd_iters_per_engaged``, the means of V,
  Q(x, u) and Q(x, u_SONIC) per step, and ``jerr`` / ``jerr_bad`` = the mean
  |q - q_ref| per step and joint, rad; and, from the solver,
  ``q_safe_mean_engaged`` (Q(x, pi(x)) per engaged step),
  ``fallback_infeasible_per_engaged`` (the share of engaged steps where pi's
  own action fails the constraint, i.e. no certified action exists at all
  unless Q extrapolates), ``trust_region_per_engaged`` (where the trust
  region bound) and ``seg_gap_mean`` (how much closer the descent got than
  the paper's segment search, on the steps it won). Episode level:
  ``false_alarm_episodes`` = the share of episodes with an intervention
  before their split (the filter acting on normal tracking),
  ``intervened_post_episodes``, ``intervened_episodes``, the median
  ``first_engaged_s`` (from the onset). Against ``ref`` (SONIC's outcomes),
  ``rate_post_sonic_survived`` / ``_sonic_fell``: how much of the
  intervention lands where SONIC would have fallen."""
  st = rec["tel_steps"].double()
  en = rec["tel_engaged"].double()
  dev = rec["tel_dev_sum"].double()
  dmax = rec["tel_dev_max"].double()
  status = rec["tel_status"].double()
  out: dict = {}
  for p, ph in enumerate(PHASES):
    S, G = st[:, p].sum(), en[:, p].sum()
    has = en[:, p] > 0
    out[f"steps_{ph}"] = int(S)
    out[f"rate_{ph}"] = _ratio(G, S)
    out[f"dev_per_step_{ph}"] = _ratio(dev[:, p].sum(), S)
    out[f"dev_per_engaged_{ph}"] = _ratio(dev[:, p].sum(), G)
    out[f"dev_max_{ph}"] = float(dmax[:, p].max()) if dmax.shape[0] else _NAN
    out[f"dev_max_mean_{ph}"] = float(dmax[has, p].mean()) if bool(has.any()) else _NAN
    out[f"status_{ph}"] = {name: _ratio(status[:, p, i].sum(), S) for i, name in enumerate(STATUS)}
    out[f"best_effort_rate_{ph}"] = _ratio(status[:, p, BEST_EFFORT].sum(), S)
    out[f"best_effort_per_engaged_{ph}"] = _ratio(status[:, p, BEST_EFFORT].sum(), G)
    out[f"gd_iters_per_engaged_{ph}"] = _ratio(rec["tel_iters_sum"][:, p].double().sum(), G)
    for key, name in (("v_sum", "v_mean"), ("q_sum", "q_mean"), ("q_nom_sum", "q_nom_mean"),
                      ("jerr_sum", "jerr"), ("jerr_bad_sum", "jerr_bad")):
      out[f"{name}_{ph}"] = _ratio(rec["tel_" + key][:, p].double().sum(), S)
    if "tel_q_safe_sum" in rec:             # records written before these keys lack them
      tot = lambda key: rec["tel_" + key][:, p].double().sum()  # noqa: E731
      out[f"q_safe_mean_engaged_{ph}"] = _ratio(tot("q_safe_sum"), G)
      out[f"fallback_infeasible_per_engaged_{ph}"] = _ratio(tot("fb_infeasible"), G)
      out[f"trust_region_per_engaged_{ph}"] = _ratio(tot("tr_bound"), G)
      out[f"seg_gap_mean_{ph}"] = _ratio(tot("seg_gap_sum"), tot("seg_gap_n"))
    if "tel_switches" in rec:
      sw = rec["tel_switches"][:, p].double()
      out[f"switches_per_episode_{ph}"] = float(sw.mean()) if sw.numel() else _NAN
  pre, post = st[:, 0] > 0, st[:, 1] > 0
  out["false_alarm_episodes"] = (float((en[pre, 0] > 0).double().mean())
                                 if bool(pre.any()) else _NAN)
  out["intervened_post_episodes"] = (float((en[post, 1] > 0).double().mean())
                                     if bool(post.any()) else _NAN)
  out["intervened_episodes"] = (float((en.sum(1) > 0).double().mean())
                                if en.shape[0] else _NAN)
  fe = rec["first_engaged_s"].double()
  fe = fe[torch.isfinite(fe)]
  out["first_engaged_s_median"] = float(fe.median()) if fe.numel() else _NAN
  if ref is not None:
    r = ref.bool()
    for name, m in (("sonic_survived", r), ("sonic_fell", ~r)):
      out[f"rate_post_{name}"] = _ratio(en[m, 1].sum(), st[m, 1].sum())
  return out


def summarize_condition(rec: dict, eps: dict, cond: Optional[Condition] = None,
                        ref: Optional[torch.Tensor] = None, categories: Sequence[str] = (),
                        dv_edges: Optional[Sequence[float]] = DV_BINS) -> dict:
  """Everything ``push.json`` / ``badref.json`` report for one condition.
  ``eps`` holds the episode list's columns (``category``, ``ref_out_k``, and
  ``dv`` when ``dv_edges`` is given, ``bad_out_k`` on the bad-reference
  suite). ``ref`` is SONIC's ``survived`` on the same episodes (None for
  SONIC itself)."""
  surv = rec["survived"].bool()
  E = int(surv.numel())
  for key in ("category", "ref_out_k"):
    if int(eps[key].numel()) != E:
      raise ValueError(f"episode column {key!r} has {eps[key].numel()} rows for {E} episodes")
  ttf = rec["time_to_fall"].double()
  post_fall = ~surv & ~rec["fell_pre"].bool()
  ok = ~eps["ref_out_k"].bool()
  out = {
    "condition": asdict(cond) if cond is not None else None,
    "episodes": E,
    "survival": survival(surv),
    "survival_excl_ref_out_k": survival(surv, ok),
    "fell_pre": _ratio(int(rec["fell_pre"].sum()), E),
    "time_to_fall_mean": float(ttf[post_fall].mean()) if bool(post_fall.any()) else _NAN,
    "time_to_fall_median": float(ttf[post_fall].median()) if bool(post_fall.any()) else _NAN,
    "by_category": by_category(eps["category"], surv, categories, ref),
    "intervention": intervention_stats(rec, ref),
  }
  if "bad_out_k" in eps:
    out["survival_excl_bad_out_k"] = survival(surv, ~eps["bad_out_k"].bool())
  if dv_edges is not None:
    out["by_dv"] = by_bins(eps["dv"], surv, dv_edges, ref)
  if ref is not None:
    out["paired"] = paired(ref, surv)
    out["paired_excl_ref_out_k"] = paired(ref, surv, ok)
  return out


def summarize_suite(records: dict, eps: dict, conditions: Sequence[Condition],
                    categories: Sequence[str] = (),
                    dv_edges: Optional[Sequence[float]] = DV_BINS) -> dict:
  """:func:`summarize_condition` of every condition, paired against
  :data:`REFERENCE` when the suite ran it."""
  ref = records[REFERENCE]["survived"] if REFERENCE in records else None
  return {c.name: summarize_condition(records[c.name], eps, c,
                                      None if c.name == REFERENCE else ref,
                                      categories, dv_edges)
          for c in conditions if c.name in records}


def concat_records(parts: Sequence[dict]) -> dict:
  """Records (or episode columns) of several episode lists, end to end."""
  keys = [k for k in parts[0] if torch.is_tensor(parts[0][k]) and parts[0][k].dim() > 0]
  return {k: torch.cat([p[k] for p in parts]) for k in keys}


def headline(s: dict) -> str:
  """One line of a condition summary, for the log."""
  sv, iv = s["survival"], s["intervention"]
  line = (f"SR {sv['sr']:.3f} [{sv['ci'][0]:.3f}, {sv['ci'][1]:.3f}] "
          f"(excl. ref out of K {s['survival_excl_ref_out_k']['sr']:.3f}); "
          f"intervention pre {iv['rate_pre']:.3f} / post {iv['rate_post']:.3f}, "
          f"false alarms {iv['false_alarm_episodes']:.3f}, "
          f"dev/engaged step {iv['dev_per_engaged_post']:.2f}")
  if s.get("paired"):
    p = s["paired"]
    line += f"; vs sonic +{p['rescued']} / -{p['broken']} (McNemar p {p['mcnemar_p']:.2g})"
  return line


def clean_json(x):
  """JSON-safe copy: tensors to numbers / lists, NaN and inf to None,
  dataclasses to dicts, tuples to lists."""
  if is_dataclass(x) and not isinstance(x, type):
    return clean_json(asdict(x))
  if torch.is_tensor(x):
    return clean_json(x.item() if x.dim() == 0 else x.tolist())
  if isinstance(x, dict):
    return {str(k): clean_json(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [clean_json(v) for v in x]
  if isinstance(x, float):
    return x if math.isfinite(x) else None
  return x
