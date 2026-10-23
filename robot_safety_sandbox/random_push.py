"""Scripted random pushes on a body: the single-player stand-in for a wrench adversary.

A wrench adversary (``dstb_mode="wrench"``, ``--adversary``) picks the force
DIRECTION every step and the bridge applies it, sustained, at ``force_max x
force_scale`` (the global ramp x the per-env survival scale). This is the same
channel driven by a random process instead of a learned player, so a
SINGLE-player recover policy can be trained against pushes with the same two
curricula (ForceRampCallback / PerEnvForceScaleCallback) and no second actor:

* an onset every ``interval_s`` (uniform per push, per env), the first one
  ``first_delay_s`` after the reset so pushes are not synchronized across envs;
* each push lasts ``duration_s`` with ONE direction, drawn uniformly in the
  horizontal plane (``horizontal=True``, a shove) or on the sphere;
* optionally a share ``rear_frac`` of the pushes comes FROM BEHIND the robot:
  the force points along its heading at the onset, within
  ``rear_half_width_deg`` of it, so it drives the robot forward (the direction
  a push-trained G1 handles worst); the rest stay uniform;
* the magnitude is ``force_max x force_scale[env]`` for the whole push, read at
  every step so the ramp and the per-env scale act exactly as on the adversary;
* a push is an IMPULSE, not a sustained load: 200 N for 0.2 s on a 33 kg G1 is
  ~1.2 m/s of base velocity, the range the harvested ``pushed`` states came
  from (``examples/harvest_g1_recover_states.py --max-push-v``).

The bridge calls :meth:`forces` once per control step and writes the result as
an external wrench on ``adversary_body``; it calls :meth:`reset` for the envs
that ended, so a new episode starts quiet and a push never carries over a reset.
``age`` (control steps since the last onset) is kept per env for diagnostics.

DIRECTIONS RELATIVE TO THE ROBOT. When the bridge passes the robot's heading
to :meth:`forces`, each push also records ``rel``: the angle of its force in
the heading frame at the onset (0 = along the heading = a push from behind,
+pi/2 = toward the robot's left = a push from its right). :meth:`record` folds
the episodes that end into a survival rate per side the LAST push came from
(``push/surv_from_<side>``), so a run shows whether the weak side improves even
while the direction-blind per-env force scale moves.

ONE PUSH PER EPISODE. ``max_pushes=k`` caps the random onsets of an episode at
``k`` (``None`` = unlimited, the process above; ``0`` = no random push). With
``k = 1`` an episode is one event -- a start state, its push, the outcome --
which is what a reach-avoid recovery target needs: "back in the stance set
after the push" only means something once no further push can come.

SCRIPTED PUSHES. :meth:`schedule` replaces an env's random push with ONE push
of a given ABSOLUTE magnitude (newtons, ignoring ``force_max`` and
``force_scale``), direction and duration, after a given delay. Timing is exact
and counted in :meth:`forces` calls (= control steps): with ``onset=k`` the
first ``k`` calls are quiet and the push acts on the next ``steps`` calls, so
``onset=0`` puts it on the very next call. The bridge uses this to resume a
push that was still acting when a start state was captured (a handover row's
pending push resumes on the FIRST step of the new episode); an evaluation uses
it for fixed-force probes. :meth:`exhaust` stops the random process for the
rest of an episode (a handover row's push already happened). Both are cleared
by :meth:`reset`, so a new episode always starts as the random process.

QUIET. :meth:`quiet` says when the robot is free of pushes for good: nothing
acting, nothing scripted, and no random push left to come this episode (the
cap is reached or :meth:`exhaust` was called). With ``max_pushes=None`` and no
exhaust the random process never runs out, so there ``quiet`` only means
"between pushes". A target margin gated on it (``envs/g1_recover/targets.py``)
cannot be met before the episode's push has landed and ended.

With no :meth:`schedule` / :meth:`exhaust` call and ``max_pushes=None`` the
process draws exactly the random numbers it drew before these existed and
returns bit-identical forces.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

#: The side a push comes from, by 90-degree sectors of ``rel``.
SIDES = ("behind", "right", "front", "left")


class RandomPush:
  """Per-env random push process (see the module docstring).

  :param num_envs: envs.
  :param dt: control step, s.
  :param device: torch device.
  :param interval_s: (lo, hi) seconds between consecutive onsets, uniform.
  :param duration_s: how long each push lasts, s (at least one step).
  :param first_delay_s: (lo, hi) seconds from a reset to the first onset.
  :param horizontal: ``True`` -> directions uniform in the x-y plane (world
      frame); ``False`` -> uniform on the sphere.
  :param rear_frac: share of the pushes drawn from behind the robot (force
      within ``rear_half_width_deg`` of its heading); 0 = all uniform. Needs
      ``horizontal`` and the heading in :meth:`forces`.
  :param rear_half_width_deg: half-width of that rear sector, degrees.
  :param ema: per-episode weight of :meth:`record`'s survival averages.
  :param max_pushes: at most this many RANDOM onsets per episode (a scripted
      push counts only when scheduled with ``count=True``); ``None`` =
      unlimited (the default, the process as it always was), ``0`` = none.

  Per-env state a script may read (do not write it): ``t_left`` (steps the
  current push still acts after the last :meth:`forces` call), ``n_pushes``
  (onsets counted toward ``max_pushes`` this episode), ``cur_force_w`` (the
  force applied at the last :meth:`forces` call, ``[n, 3]``, world frame, N),
  ``scripted`` (the push in progress, see :meth:`active`, is a scheduled one),
  plus :meth:`active` and :meth:`quiet`.
  """

  def __init__(self, num_envs: int, dt: float, device,
               interval_s: Sequence[float] = (2.0, 4.0), duration_s: float = 0.2,
               first_delay_s: Sequence[float] = (0.5, 2.0), horizontal: bool = True,
               rear_frac: float = 0.0, rear_half_width_deg: float = 60.0,
               ema: float = 0.002, max_pushes: Optional[int] = None):
    self.n = int(num_envs)
    self.dt = float(dt)
    self.device = device
    if self.dt <= 0.0:
      raise ValueError(f"dt must be > 0, got {dt}")
    lo, hi = (float(x) for x in interval_s)
    if not 0.0 < lo <= hi:
      raise ValueError(f"interval_s must be 0 < lo <= hi, got {interval_s}")
    flo, fhi = (float(x) for x in first_delay_s)
    if not 0.0 <= flo <= fhi:
      raise ValueError(f"first_delay_s must be 0 <= lo <= hi, got {first_delay_s}")
    if duration_s <= 0.0:
      raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if not 0.0 <= rear_frac <= 1.0:
      raise ValueError(f"rear_frac must be in [0, 1], got {rear_frac}")
    if not 0.0 < rear_half_width_deg <= 180.0:
      raise ValueError(f"rear_half_width_deg must be in (0, 180], got "
                       f"{rear_half_width_deg}")
    if rear_frac > 0.0 and not horizontal:
      raise ValueError("rear_frac draws directions in the heading plane and "
                       "needs horizontal=True")
    if max_pushes is not None and int(max_pushes) < 0:
      raise ValueError(f"max_pushes must be None or >= 0, got {max_pushes}")
    self.max_pushes = None if max_pushes is None else int(max_pushes)
    self._interval = (self._steps(lo), self._steps(hi))
    self._first = (self._steps(flo), self._steps(fhi))
    self.duration = max(1, self._steps(duration_s))
    self.horizontal = bool(horizontal)
    self.rear_frac = float(rear_frac)
    self.rear_half_width = math.radians(float(rear_half_width_deg))
    self.ema = float(ema)
    long = dict(dtype=torch.long, device=device)
    self.t_next = torch.zeros(self.n, **long)   # steps until the next onset
    self.t_left = torch.zeros(self.n, **long)   # steps left in the current push
    self.age = torch.zeros(self.n, **long)      # steps since the last onset
    self.dir = torch.zeros(self.n, 3, device=device)
    # heading-frame angle of the last push's force (NaN: none yet / no heading)
    self.rel = torch.full((self.n,), float("nan"), device=device)
    self._surv = torch.full((len(SIDES),), float("nan"), device=device)
    boolean = dict(dtype=torch.bool, device=device)
    # onsets counted toward max_pushes this episode, and exhaust()'s stop flag
    self.n_pushes = torch.zeros(self.n, **long)
    self.blocked = torch.zeros(self.n, **boolean)
    # what the last forces() call applied (a push acting on that call is
    # still "active" until the next call, see active())
    self.cur_force_w = torch.zeros(self.n, 3, device=device)
    self._applied = torch.zeros(self.n, **boolean)
    # the ARMED scripted push of each env (schedule()): it starts once _s_wait
    # calls have passed, then acts _s_steps calls at _s_force newtons along
    # _s_dir (world) or at the heading-frame angle _s_rel (_s_use_rel)
    self._s_armed = torch.zeros(self.n, **boolean)
    self._s_wait = torch.zeros(self.n, **long)
    self._s_steps = torch.zeros(self.n, **long)
    self._s_force = torch.zeros(self.n, device=device)
    self._s_dir = torch.zeros(self.n, 3, device=device)
    self._s_rel = torch.full((self.n,), float("nan"), device=device)
    self._s_use_rel = torch.zeros(self.n, **boolean)
    self._s_count = torch.zeros(self.n, **boolean)
    # the push in progress (active()) is a scripted one (absolute magnitude
    # _s_force); cleared at the next forces() call once it has run out
    self.scripted = torch.zeros(self.n, **boolean)
    # Python-side flags: until schedule() / exhaust() is first called, forces()
    # skips their bookkeeping entirely (no extra sync, no extra random draw).
    self._any_scripted = False
    self._any_blocked = False
    self.reset()

  def _steps(self, seconds: float) -> int:
    return int(round(seconds / self.dt))

  def _draw(self, lo_hi, k: int) -> torch.Tensor:
    lo, hi = lo_hi
    return torch.randint(lo, hi + 1, (k,), device=self.device)

  def _directions(self, k: int, heading: Optional[torch.Tensor]
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``k`` unit directions (world frame) and their heading-frame angles
    (NaN without a heading)."""
    nan = torch.full((k,), float("nan"), device=self.device)
    if not self.horizontal:
      v = torch.randn(k, 3, device=self.device)
      return v / v.norm(dim=1, keepdim=True).clamp_min(1e-6), nan
    rel = (torch.rand(k, device=self.device) * 2.0 - 1.0) * torch.pi
    if self.rear_frac > 0.0:
      if heading is None:
        raise ValueError("rear_frac > 0 needs the robot's heading in forces()")
      rear = torch.rand(k, device=self.device) < self.rear_frac
      cone = (torch.rand(k, device=self.device) * 2.0 - 1.0) * self.rear_half_width
      rel = torch.where(rear, cone, rel)
    # a uniform angle is uniform in either frame, so without a heading it
    # stands for the world angle and the relative one stays unknown
    ang = rel + heading if heading is not None else rel
    dirs = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)],
                       dim=1)
    return dirs, (rel if heading is not None else nan)

  def reset(self, mask: Optional[torch.Tensor] = None) -> None:
    """Restart the envs in ``mask`` (None = all): quiet, first onset in
    ``first_delay_s``. An active push on an env that ended is dropped, and so
    are its armed scripted push, its push count and its :meth:`exhaust`."""
    if mask is None:
      mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
    m = mask.bool()
    k = int(m.sum())
    if k == 0:
      return
    self.t_next[m] = self._draw(self._first, k)
    self.t_left[m] = 0
    self.age[m] = 0
    self.dir[m] = 0.0
    self.rel[m] = float("nan")
    self.n_pushes[m] = 0
    self.blocked[m] = False
    # out of place: these two ARE the tensors forces() returned to its caller
    self.cur_force_w = torch.where(m.unsqueeze(1),
                                   torch.zeros_like(self.cur_force_w),
                                   self.cur_force_w)
    self._applied = self._applied & ~m
    self._s_armed[m] = False
    self.scripted[m] = False

  def forces(self, force_max: float, force_scale: Optional[torch.Tensor] = None,
             heading: Optional[torch.Tensor] = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """This step's push force per env, ``[n, 3]`` (zeros where no push is
    active), and the active mask ``[n]``; then advance one control step.

    A scripted push (:meth:`schedule`) whose delay has run out starts here
    and acts at its own absolute magnitude; while one is armed, or once the
    episode's ``max_pushes`` / :meth:`exhaust` stop is reached, no random onset
    happens.

    :param force_max: the global magnitude (ForceRampCallback writes it).
    :param force_scale: per-env multiplier ``[n]`` or None (= 1).
    :param heading: the robot's world-frame yaw ``[n]``, rad, or None; read
        only at an onset (required when ``rear_frac > 0``, and for a scripted
        push given by its heading-frame angle).
    """
    if self._any_scripted:
      self.scripted &= self.t_left > 0      # a scripted push that ran out last call
      self._scripted_onsets(heading)
    onset = (self.t_next <= 0) & (self.t_left <= 0)
    if self._any_scripted:
      onset &= ~self._s_armed               # the scripted push owns the env
    if self.max_pushes is not None or self._any_blocked:
      onset &= self._may_push()
    k = int(onset.sum())
    if k:
      h = heading.reshape(-1).to(self.device)[onset] if heading is not None else None
      self.dir[onset], self.rel[onset] = self._directions(k, h)
      self.t_left[onset] = self.duration
      # the next onset is measured from this one, and never inside this push
      self.t_next[onset] = self.duration + self._draw(self._interval, k)
      self.age[onset] = 0
      self.n_pushes += onset.long()
    active = self.t_left > 0
    mag = torch.full((self.n,), float(force_max), device=self.device)
    if force_scale is not None:
      mag = mag * force_scale.reshape(-1).to(self.device)
    if self._any_scripted:
      mag = torch.where(self.scripted, self._s_force, mag)
    f = self.dir * (mag * active.float()).unsqueeze(1)
    self.cur_force_w = f
    self._applied = active
    self.t_next -= 1
    self.t_left = (self.t_left - 1).clamp_min(0)
    self.age += 1
    return f, active

  def _may_push(self) -> torch.Tensor:
    """Envs that may still get a RANDOM onset this episode."""
    ok = ~self.blocked
    if self.max_pushes is not None:
      ok &= self.n_pushes < self.max_pushes
    return ok

  def _scripted_onsets(self, heading: Optional[torch.Tensor]) -> None:
    """Start the armed scripted pushes whose delay has run out; count down the
    others. Runs at the top of :meth:`forces`, so an onset-0 push acts on the
    call right after :meth:`schedule`."""
    start = self._s_armed & (self._s_wait <= 0)
    self._s_wait = torch.where(self._s_armed & ~start, self._s_wait - 1,
                               self._s_wait)
    k = int(start.sum())
    if k == 0:
      return
    h = heading.reshape(-1).to(self.device) if heading is not None else None
    by_rel = start & self._s_use_rel
    dirs = self._s_dir
    if bool(by_rel.any()):
      if h is None:
        raise ValueError("a scripted push given by its heading-frame angle "
                         "(schedule(rel=...)) needs the robot's heading in "
                         "forces()")
      ang = self._s_rel + h
      d_rel = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)],
                          dim=1)
      dirs = torch.where(by_rel.unsqueeze(1), d_rel, dirs)
    self.dir = torch.where(start.unsqueeze(1), dirs, self.dir)
    # record()'s side label: the given angle, or the world direction's angle in
    # the heading frame (NaN without a heading, or for a vertical push)
    rel = self._s_rel
    if h is not None:
      world = torch.atan2(self._s_dir[:, 1], self._s_dir[:, 0]) - h
      world = torch.remainder(world + torch.pi, 2.0 * torch.pi) - torch.pi
      flat = self._s_dir[:, :2].norm(dim=1) > 1e-6
      world = torch.where(flat, world, torch.full_like(world, float("nan")))
      rel = torch.where(self._s_use_rel, self._s_rel, world)
    self.rel = torch.where(start, rel, self.rel)
    self.t_left = torch.where(start, self._s_steps, self.t_left)
    # random pushes (if any are left) resume one interval after this push's
    # onset, as after a random one
    self.t_next[start] = self._s_steps[start] + self._draw(self._interval, k)
    self.age = torch.where(start, torch.zeros_like(self.age), self.age)
    self.n_pushes += (start & self._s_count).long()
    self.scripted |= start
    self._s_armed &= ~start

  def _per_env(self, x, m: torch.Tensor, k: int, dtype) -> torch.Tensor:
    """``x`` (a scalar, ``[n]`` = one value per env, or ``[k]`` = one per env
    in ``m``) as the ``[k]`` values of the envs in ``m``."""
    t = torch.as_tensor(x, device=self.device).to(dtype)
    if t.dim() == 0:
      return t.expand(k).clone()
    t = t.reshape(-1)
    if t.shape[0] == self.n:
      return t[m]
    if t.shape[0] == k:
      return t
    raise ValueError(f"expected a scalar, {self.n} or {k} values, got "
                     f"{tuple(t.shape)}")

  def schedule(self, mask: torch.Tensor, force_n, steps, onset=None,
               dir_w: Optional[torch.Tensor] = None, rel=None,
               count: bool = False) -> None:
    """Arm ONE scripted push on the envs in ``mask``, replacing their random
    push: an active one is cut off now, and no random onset happens until the
    scripted push has started. Afterwards the random process resumes (one
    interval after the scripted onset), still subject to ``max_pushes``.

    TIMING, in :meth:`forces` calls (= control steps): the first
    ``onset[i]`` calls are quiet, then the push acts on exactly ``steps[i]``
    calls; ``onset=None`` / ``0`` -> it acts on the very next call.

    :param mask: ``[n]`` bool, the envs to arm.
    :param force_n: ABSOLUTE magnitude, N (``force_max`` / ``force_scale`` are
        not applied): a scalar, ``[n]`` or ``[mask.sum()]``.
    :param steps: duration, control steps (>= 1), same shapes.
    :param onset: delay, control steps (>= 0), same shapes, or None (= 0).
    :param dir_w: world-frame directions, ``[3]``, ``[n, 3]`` or
        ``[mask.sum(), 3]`` (normalized here). Exactly one of ``dir_w`` /
        ``rel``.
    :param rel: heading-frame angles, rad (0 = along the heading = a push
        from behind, +pi/2 = toward the robot's left), resolved against the
        heading :meth:`forces` receives at the onset (which must then get one).
    :param count: the push counts toward ``max_pushes`` (scalar or per env).
    """
    if (dir_w is None) == (rel is None):
      raise ValueError("schedule() takes exactly one of dir_w / rel")
    m = mask.reshape(-1).bool().to(self.device)
    k = int(m.sum())
    if k == 0:
      return
    f = self._per_env(force_n, m, k, torch.float32)
    st = self._per_env(steps, m, k, torch.long)
    on = self._per_env(0 if onset is None else onset, m, k, torch.long)
    if bool((st < 1).any()) or bool((on < 0).any()) or bool((f < 0).any()):
      raise ValueError("schedule() needs steps >= 1, onset >= 0, force_n >= 0")
    # the push these envs had (random or scripted) stops here
    self.t_left[m] = 0
    self.scripted[m] = False
    self._s_armed[m] = True
    self._s_force[m] = f
    self._s_steps[m] = st
    self._s_wait[m] = on
    self._s_count[m] = self._per_env(count, m, k, torch.bool)
    if dir_w is not None:
      d = torch.as_tensor(dir_w, device=self.device).float()
      d = d.reshape(1, 3).expand(k, 3) if d.dim() == 1 else d.reshape(-1, 3)
      if d.shape[0] == self.n:
        d = d[m]
      elif d.shape[0] != k:
        raise ValueError(f"dir_w: expected [3], [{self.n}, 3] or [{k}, 3], got "
                         f"{tuple(torch.as_tensor(dir_w).shape)}")
      self._s_dir[m] = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
      self._s_rel[m] = float("nan")
      self._s_use_rel[m] = False
    else:
      r = self._per_env(rel, m, k, torch.float32)
      self._s_rel[m] = torch.remainder(r + torch.pi, 2.0 * torch.pi) - torch.pi
      self._s_dir[m] = 0.0
      self._s_use_rel[m] = True
    self._any_scripted = True

  def exhaust(self, mask: torch.Tensor) -> None:
    """No more RANDOM pushes for the envs in ``mask`` this episode (cleared by
    :meth:`reset`), whatever ``max_pushes`` is. A push already acting and an
    armed scripted push are left alone, so the bridge may call this and
    :meth:`schedule` in either order."""
    self.blocked |= mask.reshape(-1).bool().to(self.device)
    self._any_blocked = True

  def active(self) -> torch.Tensor:
    """``[n]`` bool: a push is in progress -- it acted at the last
    :meth:`forces` call or still has steps left (``t_left > 0``: it acts again
    at the next call). Read between a :meth:`forces` call and the physics step
    it feeds, the push applied in that step counts as active."""
    return self._applied | (self.t_left > 0)

  def quiet(self) -> torch.Tensor:
    """``[n]`` bool: free of pushes for good -- none in progress
    (:meth:`active`), none scripted and armed, and no random onset left this
    episode (``n_pushes >= max_pushes``, or :meth:`exhaust` was called).

    With ``max_pushes=None`` the random process never runs out, so there
    ``quiet`` is True only BETWEEN pushes (nothing active, nothing armed)."""
    q = ~self.active() & ~self._s_armed
    if self.max_pushes is not None:
      q &= self.blocked | (self.n_pushes >= self.max_pushes)
    return q

  def record(self, ended: torch.Tensor, fell: torch.Tensor) -> dict[str, float]:
    """Fold the episodes in ``ended`` into the survival rate per side their
    LAST push came from (``fell`` = ended on a failure); call it before
    :meth:`reset`. Episodes with no push, or none with a known heading, are
    left out. Returns ``push/surv_from_<side>`` for the sides seen so far."""
    m = ended.bool() & ~torch.isnan(self.rel)
    # sector 0 = rel in [-45, 45) deg (from behind), then counterclockwise
    side = (torch.remainder(self.rel[m] + torch.pi / 4, 2.0 * torch.pi)
            // (torch.pi / 2)).long().clamp_(0, len(SIDES) - 1)
    one = torch.nn.functional.one_hot(side, len(SIDES)).float()
    count = one.sum(0)
    rate = (one * (~fell.bool()[m]).float()[:, None]).sum(0) / count.clamp_min(1.0)
    keep = (1.0 - self.ema) ** count
    new = torch.where(torch.isnan(self._surv), rate,
                      self._surv * keep + rate * (1.0 - keep))
    self._surv = torch.where(count > 0, new, self._surv)
    return {f"push/surv_from_{s}": v for s, v in zip(SIDES, self._surv.tolist())
            if v == v}                                    # drop NaN
