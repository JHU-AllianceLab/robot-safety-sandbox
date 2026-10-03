"""Reach-avoid-stay as a switch between two safety filters.

Reach-avoid certifies getting into a target set safely; it says nothing about
remaining there. Reach-avoid-stay is composed here out of two filters and a
per-env mode:

  REACH  the reach-avoid filter (its monitor V_RA, its fallback pi_RA), whose
         target is the stay filter's certified set {V_stay > 0}.
  STAY   the avoid filter on the STAY constraint min(g, l_goal) (monitor
         V_stay, fallback pi_stay): upright AND inside the goal set.

The mode is latched, with hysteresis on the stay certificate itself:

  REACH -> STAY   when V_stay(x) > eps_in     the stay filter can hold from here
  STAY -> REACH   when V_stay(x) <= eps_out   it no longer can (a disturbance
                                              the stay policy did not absorb)

``eps_out <= eps_in``. Every episode starts in REACH. Switching on V_stay rather
than on the goal box is the point: a state inside the box that is already
leaving it is not one the stay policy can hold, and the reach twin was trained
to arrive where V_stay > 0, not merely where l_goal >= 0.

Each sub-filter is an ordinary :class:`~.core.SafetyFilter` and is evaluated
every step, so both keep their own per-env state current; only the active
mode's action and telemetry are returned. The two twins normalize observations
differently, so the caller hands both the RAW observation (``s_obs``) and each
twin's callables apply that twin's own normalizer.
"""

from __future__ import annotations

import torch

from .core import SafetyFilter
from .telemetry import EngagementLog, FilterInfo


class ReachAvoidStayFilter:
  """Two filters, one per phase, and a latched per-env mode.

  Duck-types :class:`~.core.SafetyFilter` for the evaluation loop:
  ``__call__(a_nom, **ctx) -> (action, FilterInfo)``, ``reset(done)``,
  ``telemetry``.

  :param reach: the reach-avoid filter (REACH mode).
  :param stay: the avoid filter on the stay constraint (STAY mode).
  :param stay_value_fn: ``(**ctx) -> (N,)`` V_stay, the switching certificate.
  :param eps_in: enter STAY when V_stay > eps_in.
  :param eps_out: leave STAY when V_stay <= eps_out (default: eps_in, i.e. no
      hysteresis).
  """

  def __init__(self, reach: SafetyFilter, stay: SafetyFilter, stay_value_fn,
               *, eps_in: float = 0.0, eps_out: float | None = None):
    if reach.num_envs != stay.num_envs or reach.device != stay.device:
      raise ValueError("reach and stay filters disagree on num_envs/device")
    self.reach, self.stay, self.stay_value_fn = reach, stay, stay_value_fn
    self.eps_in = float(eps_in)
    self.eps_out = self.eps_in if eps_out is None else float(eps_out)
    if self.eps_out > self.eps_in:
      raise ValueError(f"eps_out ({self.eps_out}) must be <= eps_in "
                       f"({self.eps_in}): leaving STAY must not be easier "
                       f"than entering it")
    self.num_envs, self.device = reach.num_envs, reach.device
    #: per-env mode, True = STAY
    self.mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #: this step's transitions (for metrics)
    self.entered = torch.zeros_like(self.mode)
    self.left = torch.zeros_like(self.mode)
    #: this step's V_stay
    self.stay_value = torch.zeros(self.num_envs, device=self.device)
    self.telemetry = EngagementLog(self.num_envs, self.device)

  def peek(self, **ctx) -> torch.Tensor:
    """The mode this step's call WILL use, without committing it."""
    return self._next_mode(self.stay_value_fn(**ctx))

  def _next_mode(self, v_stay: torch.Tensor) -> torch.Tensor:
    return torch.where(self.mode, v_stay > self.eps_out, v_stay > self.eps_in)

  def __call__(self, a_nom: torch.Tensor, **ctx) -> tuple[torch.Tensor, FilterInfo]:
    v_stay = self.stay_value_fn(**ctx)
    new = self._next_mode(v_stay)
    self.entered, self.left = new & ~self.mode, self.mode & ~new
    self.mode, self.stay_value = new, v_stay
    a_r, i_r = self.reach(a_nom, **ctx)
    a_s, i_s = self.stay(a_nom, **ctx)
    m = self.mode
    action = torch.where(m.unsqueeze(-1), a_s, a_r)
    caution = (torch.where(m, i_s.caution, i_r.caution)
               if i_r.caution is not None and i_s.caution is not None else None)
    info = FilterInfo(engaged=torch.where(m, i_s.engaged, i_r.engaged),
                      value=torch.where(m, i_s.value, i_r.value),
                      caution=caution)
    self.telemetry.update(info)
    return action, info

  def reset(self, done: torch.Tensor) -> None:
    """New episodes start in REACH. Call EVERY step."""
    self.mode &= ~done
    self.reach.reset(done)
    self.stay.reset(done)
