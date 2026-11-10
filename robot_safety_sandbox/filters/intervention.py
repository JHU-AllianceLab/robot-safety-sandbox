"""Interventions phi(x, u) : how to modify the nominal when it isn't certified.

Third of the three modules a safety filter composes (Hsu, Hu & Fisac, "The
Safety Filter: A Unified View", Def. 2). The intervention is the ONLY module
with authority over the returned action; it consults the monitor and the
fallback itself, which is why ``SafetyFilter`` carries no branching logic.

Two families:

  LeastRestrictiveIntervention   hard binary switch — pass the nominal through
                                 or hand over to the fallback entirely.
  OptIntervention                minimal modification: solve for the closest
                                 action to the nominal that the monitor still
                                 certifies. ``QCBFIntervention`` is the
                                 projected-gradient solver for the class-K
                                 barrier constraint Q(x, u) >= kappa * V(x).

Plus one NON-CANONICAL member, kept separate on purpose:

  HeuristicSmoothingIntervention  the switch above wrapped in field heuristics
                                  (median smoothing, engagement latch, release
                                  hysteresis, rest-speed gate). NOT a category
                                  from the literature and NOT Def-2 valid
                                  instant-by-instant. Never name it after a
                                  published filter.

Margin convention throughout: safe iff >= 0.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from .telemetry import FilterInfo


class Intervention(ABC):
  """phi(x, u_nom) -> (N, A), returned alongside the step's FilterInfo."""

  def __init__(self, num_envs: int, device: str, action_dim: int | None = None):
    self.num_envs, self.device = num_envs, device
    self.action_dim = action_dim

  @abstractmethod
  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Return (filtered action (N, A), FilterInfo)."""

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Default: none held."""


class LeastRestrictiveIntervention(Intervention):
  """The switching filter of the literature: ONE memoryless comparison.

      phi(x, u) = u          if  Delta(x, u) > eps
                  pi^<(x)    otherwise

  That is the whole rule. No history, no latch, no hysteresis, no gate on the
  robot's speed — the decision at step t depends only on the monitor's value at
  step t, so the filter holds authority for exactly as long as the monitor says
  it must and returns it the instant it does not. "Least restrictive" is that
  minimality: among all filters that keep the monitor certified, the switch
  overrides on the smallest possible set of states.

  This is the form Def. 2 validity is stated for (Hsu, Hu & Fisac, Def. 2:
  Delta(x, pi^<(x)) >= 0 => Delta(x, phi(x, u)) >= 0 for all u), and with an
  exact monitor it satisfies it BY CONSTRUCTION: whenever the fallback itself is
  certified, either the nominal already scores above eps and is safe to pass, or
  the output IS the fallback. Nothing here can be talked into an uncertified
  action by an adversarial nominal, because nothing here has state for an
  adversary to drive.

  Composed with a ValueMonitor this is the Safety Value Filter; with a
  CriticMonitor, the Safety Critic Filter (the Gameplay-Filters baseline); with
  an AdversarialRolloutMonitor, the Gameplay Filter — the same object in all
  three.

  For the deployed variant that trades pointwise strictness for chatter
  robustness, see :class:`HeuristicSmoothingIntervention` — which is a
  DIFFERENT object with a deliberately unofficial name, not a configuration of
  this one.

  :param eps: engagement threshold on the margin; hand over when
      ``Delta <= eps``. The canonical setting is 0.0 (the boundary of the
      monitor's own safe set); a positive eps buys conservatism, a negative one
      spends it.
  """

  def __init__(self, num_envs: int, device: str, eps: float = 0.0,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.eps = float(eps)

  def __call__(self, a_nom: torch.Tensor, fallback, monitor, *,
               speed: torch.Tensor | None = None,
               fresh: torch.Tensor | None = None,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """``speed`` and ``fresh`` are accepted so this is a drop-in for the
    heuristic variant (the runner supplies both every step) and then DISCARDED:
    a memoryless rule has no episode boundary to honour and no release
    condition to gate. They are not forwarded to the monitor or the fallback,
    which would otherwise see kwargs that belong to the switch."""
    del speed, fresh
    value = monitor(a_nom, fallback=fallback, **ctx)
    engaged = value <= self.eps
    a_safe = fallback(**ctx)
    action = torch.where(engaged.unsqueeze(-1), a_safe, a_nom)
    return action, FilterInfo(engaged=engaged, value=value)


class HeuristicSmoothingIntervention(Intervention):
  """Latched eps-switch on the monitored margin, with a caution band.

  ⚠ NOT A LITERATURE CATEGORY. This is :class:`LeastRestrictiveIntervention`
  plus four field heuristics, and it is deliberately named after what it does
  rather than after any published filter, because it is NOT the least-
  restrictive switch and must never be reported as one. The four additions —
  median smoothing, an engagement latch, release hysteresis, and a rest-speed
  release gate — all EXPAND the override set beyond the minimal one, and they
  make the rule stateful, so Def. 2 validity does not hold instant-by-instant
  (see ``tests/test_filter_validity.py``: the canonical switch passes on
  streams where this does not). They were tuned empirically on the gap/crawl
  gauntlets, where the nominal is a blind walker approaching a hazard.

  The protocol proven out by those gauntlets (the library form of the original
  eval_filter.py BatchValueFilter):

    engage   when median-smoothed Delta <= eps  (or raw Delta clearly below)
    release  when Delta > eps + hysteresis AND the robot is near rest
    caution  while eps < Delta <= caution and not engaged: the caller should
             slow the nominal down (e.g. zero its velocity command) so a later
             handover happens from a braking stance instead of mid-trot, and a
             released nominal doesn't sprint straight back into re-engagement
             chatter.

  Field rules baked in (FUNCTIONAL state, not telemetry — they change what the
  robot does, so they live here rather than in EngagementLog):
  - The margin history is a 5-step median (single-step dips at contact events
    otherwise cause spurious engagements).
  - Fresh episodes reseed the history and drop the latch (the reset contract).
  - Release requires NEAR REST, not just a recovered margin: releasing at speed
    hands the nominal a state it never visits in training.

  Composing this with a ValueMonitor does NOT give you the Safety Value Filter,
  and with a CriticMonitor it is NOT the Gameplay-Filters baseline: those names
  denote the canonical switch. A composition built on this intervention is a
  smoothed variant of the named filter and has to be reported as one — the
  difference between the two is precisely the safety margin the smoothing
  spends.

  :param eps: engagement threshold on the margin.
  :param caution: upper edge of the caution band (>= eps).
  :param hysteresis: release requires margin > eps + hysteresis.
  :param rest_speed: release additionally requires speed < rest_speed.
  :param median_window: length of the median-smoothing history.
  :param dip_margin: raw margin below eps - dip_margin engages immediately,
      bypassing the median (a deep dip is not smoothing noise).
  """

  def __init__(self, num_envs: int, device: str, eps: float = 0.0,
               caution: float = 0.45, hysteresis: float = 0.15,
               rest_speed: float = 0.4, median_window: int = 5,
               dip_margin: float = 0.15, action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.eps, self.caution = eps, max(caution, eps)
    self.hys, self.rest = hysteresis, rest_speed
    self.window, self.dip = median_window, dip_margin
    self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.v_hist: torch.Tensor | None = None

  def __call__(self, a_nom: torch.Tensor, fallback, monitor, *,
               speed: torch.Tensor, fresh: torch.Tensor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """``speed``: (N,) planar base speed; ``fresh``: (N,) bool, True on the
    first step of an episode. Extra kwargs are forwarded to the monitor and
    the fallback."""
    v_raw = monitor(a_nom, fallback=fallback, **ctx)
    if self.v_hist is None:
      self.v_hist = v_raw.unsqueeze(0).repeat(self.window, 1)
    self.v_hist = torch.cat([self.v_hist[1:], v_raw.unsqueeze(0)], dim=0)
    if bool(fresh.any()):
      self.v_hist[:, fresh] = v_raw[fresh].unsqueeze(0)
      self.engaged &= ~fresh
    v_med = self.v_hist.median(dim=0).values

    engage = (v_med <= self.eps) | (v_raw <= self.eps - self.dip)
    release = (v_med > self.eps + self.hys) & (speed < self.rest)
    self.engaged = (self.engaged | engage) & ~release
    caution = (v_med <= self.caution) & ~self.engaged

    a_safe = fallback(**ctx)
    action = torch.where(self.engaged.unsqueeze(-1), a_safe, a_nom)
    return action, FilterInfo(engaged=self.engaged, value=v_raw,
                              caution=caution)

  def reset(self, done: torch.Tensor) -> None:
    self.engaged &= ~done


class OptIntervention(Intervention):
  """Minimal modification: argmin_u ||u_nom - u||^2 s.t. the monitor certifies.

  Subclasses supply the constraint and the solver; they share the contract that
  an env whose nominal already satisfies the constraint passes through
  untouched, and that an env for which no feasible action is found falls back
  to pi^<(x) (best effort).
  """

  @abstractmethod
  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Return (projected action, FilterInfo)."""


class QCBFIntervention(OptIntervention):
  """Q-CBF / R-CBF projected-gradient projection through a learned Q(x, u).

  Faithful (batched) port of safe_adaptation_dev's projected-gradient R-CBF
  (script/eval_safety_filter.py::rcbf_projected_gradient, derivation in
  research_notes/rcbf_gradient_method.md). The optimization per env:

    u*(x) = argmin_u ||u_task - u||^2   s.t.   Q(x, u) >= kappa * V(x)

  with Q the learned (robust) state-action safety value (safe iff >= 0) and
  V(x) = Q(x, u_safe(x)) the safety value under the fallback's own action —
  i.e. the barrier is CLASS-K (preserve a kappa-fraction of the current safety
  level), not a fixed floor. Algorithm, per the note:

    1. tau <- kappa * V(x);  if Q(x, u_task) >= tau: return u_task
    2. if V(x) < tau (kappa > 1 with V > 0, or extreme states): return u_safe
       (best-effort — even the fallback action cannot meet the threshold)
    3. normalized gradient ascent u <- Proj_U(u + lr * dQ/du / ||dQ/du||) until
       Q(x, u) >= tau (gradients via autograd through the critic — and, for a
       robust Q(x, u, pi_dstb(x, u)), through the disturbance actor: pass a
       q_fn that composes them and the total derivative comes for free)
    4. backtracking binary search on the segment [u_task, u_feas] for the
       closest-to-u_task action that is still feasible
    5. never feasible within n_iter: return the best candidate seen,
       initialized to u_safe (stateless best-effort, matching the reference)

  Differences from the reference (deliberate, both mechanical):
  - batched over N envs (per-env masks replace the scalar early returns);
  - Q convention is the zoo's "safe iff >= 0" (the reference's critics are
    trained the same way in the two-player game, so no sign flip is actually
    involved).

  The legacy 1-D variant (line search on the blend (1-a) u_task + a u_safe) is
  dominated by this method (see the note's comparison) and is not ported;
  LeastRestrictiveIntervention covers the a in {0, 1} switching case.

  Practical notes:
  - Requires a CriticMonitor: the raw ``monitor.q_fn`` must be differentiable
    w.r.t. the action; one autograd call per ascent step, batched over envs.
  - On-policy twins (``*PPO*``) learn V(s), not Q(s, a): use this intervention
    with an off-policy twin's critic (any of {Safety,ReachAvoid}SAC{1P,2P}) or a
    distilled Q head.
  - A learned Q under optimization pressure is a certificate under attack (the
    Goodhart lesson): prefer an ensemble-LCB q_fn and validate against witness
    rollouts before trusting the numbers.

  :param kappa: barrier coefficient in [0, 1]; the constraint is
      Q(x, u) >= kappa * V(x).
  :param lr: normalized-gradient-ascent step size in action units.
  :param n_iter: max ascent steps.
  :param n_backtrack: binary-search refinements toward u_task.
  :param action_low/high: action-box bounds for the projection clip.
  """

  def __init__(self, num_envs: int, device: str, kappa: float = 0.8,
               lr: float = 0.05, n_iter: int = 10, n_backtrack: int = 10,
               action_low: float = -1.0, action_high: float = 1.0,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.kappa, self.lr = kappa, lr
    self.n_iter, self.n_backtrack = n_iter, n_backtrack
    self.lo, self.hi = action_low, action_high

  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    q_fn = getattr(monitor, "q_fn", None)
    if q_fn is None:
      raise TypeError(
        "QCBFIntervention needs a monitor exposing a differentiable q_fn "
        f"(a CriticMonitor); got {type(monitor).__name__}.")

    a_safe = fallback(**ctx)
    with torch.no_grad():
      v_hat = q_fn(action=a_safe, **ctx)               # V(x) = Q(x, u_safe)
      tau = self.kappa * v_hat
      q_task = q_fn(action=a_nom, **ctx)

    pass_through = q_task >= tau                        # step 1
    best_effort = v_hat < tau                           # step 2
    need = ~pass_through & ~best_effort

    # step 3: normalized gradient ascent from u_task until feasible
    u = a_nom.clone()
    u_feas = a_safe.clone()                             # default if never found
    found = torch.zeros_like(need)
    if bool(need.any()):
      for _ in range(self.n_iter):
        u_g = u.detach().requires_grad_(True)
        q = q_fn(action=u_g, **ctx)
        newly = need & ~found & (q.detach() >= tau)
        u_feas = torch.where(newly.unsqueeze(-1), u.detach(), u_feas)
        found |= newly
        active = need & ~found
        if not bool(active.any()):
          break
        grad = torch.autograd.grad(q.sum(), u_g)[0]
        gnorm = grad.norm(dim=-1, keepdim=True)
        step = self.lr * grad / gnorm.clamp_min(1e-8)
        stalled = (gnorm.squeeze(-1) < 1e-8) & active
        active = active & ~stalled
        u = torch.where(active.unsqueeze(-1),
                        (u.detach() + step).clamp(self.lo, self.hi),
                        u.detach())
      # final feasibility check for the last iterate
      with torch.no_grad():
        q = q_fn(action=u, **ctx)
      newly = need & ~found & (q >= tau)
      u_feas = torch.where(newly.unsqueeze(-1), u, u_feas)
      found |= newly

      # step 4: backtrack — binary search [u_task, u_feas] per found env
      if bool(found.any()):
        u_lo, u_hi = a_nom.clone(), u_feas.clone()
        for _ in range(self.n_backtrack):
          u_mid = 0.5 * (u_lo + u_hi)
          with torch.no_grad():
            q_mid = q_fn(action=u_mid, **ctx)
          ok = (q_mid >= tau).unsqueeze(-1)
          u_hi = torch.where(ok, u_mid, u_hi)
          u_lo = torch.where(ok, u_lo, u_mid)
        u_feas = torch.where(found.unsqueeze(-1), u_hi, u_feas)

    # compose: pass-through | projected | best-effort u_safe (step 5 folds
    # never-feasible envs into u_safe via u_feas's initialization)
    action = torch.where(pass_through.unsqueeze(-1), a_nom,
                         torch.where(best_effort.unsqueeze(-1), a_safe,
                                     u_feas))
    with torch.no_grad():
      q_final = q_fn(action=action, **ctx)

    intervened = ~pass_through          # stateless (no latch)
    dev = torch.norm(action - a_nom, dim=-1)
    return action, FilterInfo(engaged=intervened, value=q_final,
                              intervention=dev)


class QCBFDescentIntervention(OptIntervention):
  """Q-CBF minimal modification, solved by gradient descent on an augmented
  Lagrangian, with the line search of the paper as a safety net.

  The Q-CBF filter of Oh, Lidard et al., "Safety with Agency: Human-Centered
  Safety Filter" (arXiv:2504.11717, Def. 2 / Prop. 1): with a nominal
  controller u_nom (e.g. a motion tracker) and a fallback pi^< whose own
  safety values V^pi(x), Q^pi(x, u) were fitted,

      u*(x) = argmin_u ||u - u_nom||^2   s.t.   Q(x, u) >= tau(x),
      tau(x) = kappa * V(x) + margin

  — keep at least a kappa-fraction of the safety the fallback has from here,
  and otherwise stay as close to the nominal as possible. Sibling of
  :class:`QCBFIntervention`, which ASCENDS Q until feasible and then
  backtracks toward u_nom: that finds a feasible point, not the closest one
  (the ascent path follows dQ/du, not the projection's geometry). This class
  solves the projection itself. Per env, batched with masks:

    1. pass-through   Q(x, u_nom) >= tau: u_nom untouched, ONE Q forward (plus
                      V(x)), no gradients — the common case.
    2. descent        "gd": gradient descent on the augmented Lagrangian of
                      f(u) = ||u - u_nom||^2 with c(u) = (tau - Q(x, u)) / s,
                          L = f + (max(0, lam + rho c)^2 - lam^2) / (2 rho),
                      from u_nom, dual ascent lam <- max(0, lam + rho c) every
                      ``dual_every`` steps, rho *= ``rho_growth`` where the
                      violation did not shrink 4x since the last update, and a
                      per-env stop at a KKT point. The BEST FEASIBLE iterate
                      (closest to u_nom with Q >= tau) is kept, then polished
                      by bisection on the ray from u_nom through it.
    3. segment        "segment": the paper's Algorithm 2 — the closest
                      feasible point on (1 - a) u_nom + a u_safe, a in [0, 1]
                      (coarse grid, then bisection).
    4. best effort    nothing feasible: u_safe (``best_effort="fallback"``, the
                      literature's choice) or the visited candidate with the
                      highest Q (``"argmax"``; u_safe or one inside the trust
                      region).

  ``solver="gd+segment"`` (default) runs both 2 and 3 and returns the feasible
  candidate closest to u_nom; "gd" / "segment" alone are the ablations.

  Why these choices. The defaults were tuned on synthetic Q's (halfspace and
  ball in 29-D, closed form; random ELU MLPs) and on the fitted G1 Q^pi / V^pi
  nets of data/policy_eval_gfall (SONIC's next action on held-out test rows
  with Q(x, u_SONIC) < 0.7 V(x); ppo_s0 exact and shac_s0 td_g099), against a
  multi-start damped-SQP reference. With the defaults the descent was feasible
  on 100 % of those envs and its deviation was within 0.05 % (median) / 0.2 %
  (p90) of the reference's; ~2 % of the fitted-Q envs converge to a different,
  worse KKT point (up to 1.2-1.7x) — a local method's lot, and the single-
  start SQP from u_nom does the same.

  - The constraint is NORMALIZED by s = ||dQ/du|| at u_nom, so c is a
    first-order distance to the boundary in ACTION units, the multiplier is
    ~2x the deviation, and rho / the step size are dimensionless: the same
    defaults serve a Q whose slope is 0.05 or 5 per action unit.
  - Plain gradient steps eta = lr / (2 + rho ||dc/du||^2) (the inverse
    curvature of L along the constraint normal for a locally linear Q), not
    Adam: Adam's per-coordinate, sign-like steps distort the EUCLIDEAN geometry
    the objective is stated in; on the fitted Q's it was feasible on 50-89 % of
    the envs, with median deviations 6-14 % above the reference. Nor the
    rank-1 Gauss-Newton preconditioner (the exact AL Hessian for a linear Q):
    it ignores Q's curvature, overshoots, and was feasible on 74-95 %.
  - rho grows only on insufficient progress: growing it on every infeasible
    dual update (the naive rule) shrinks the tangential step ~1 / rho and left
    the p90 deviation 3-6 % above the reference instead of 0.2 %.
  - An augmented Lagrangian, not a pure penalty: the multiplier makes the
    constraint exact at a bounded rho, where a penalty needs rho -> inf and the
    step size with it. Its iterates approach the boundary from OUTSIDE, which
    is why the best feasible iterate is tracked and the ray polish exists.
  - The segment search costs no gradients, and it is the only candidate that
    exists when dQ/du vanishes (flat, saturated regions of a fitted Q) or GD
    stalls in a local optimum; it is also what the paper ships.

  Contract details:

  - ``monitor.q_fn(action=u, **ctx) -> (N,)`` must be differentiable in u;
    ``monitor.v_fn(**ctx) -> (N,)`` is OPTIONAL — absent, V(x) = Q(x, u_safe)
    (the sibling's convention).
  - The caller may hold ``torch.no_grad()``: the descent runs under
    ``torch.enable_grad()`` and differentiates w.r.t. the ACTION only
    (``torch.autograd.grad``), so nothing accumulates in the Q net's ``.grad``
    and every returned tensor is detached. ``torch.inference_mode()`` cannot
    be escaped that way and is refused when a descent is needed.
  - The fallback is called EVERY step (a recurrent fallback must see every
    observation), but Q(x, u_safe) only on steps where some env needs help,
    unless V comes from it; ``last["q_safe"]`` is NaN where not evaluated.
  - No action box by default (a tracker's targets legitimately exceed a
    policy's box); ``action_low/high`` clamp every MODIFIED action (descent
    iterates are projected), never a passed-through nominal.
  - TRUST REGION (``max_dev``, off by default): a fitted Q is only data
    where it saw actions, and its slope in u is small (on the G1 fits
    ||dQ/du|| ~ 0.07-0.13 per action unit at u_SONIC), so an unbounded
    projection happily "certifies" actions tens of units from u_nom by
    extrapolation, and the best effort never fires. With ``max_dev`` the
    descent's iterates are projected onto the ball ||u - u_nom|| <= max_dev,
    and a GD or segment candidate outside it does not count as feasible —
    such an env goes to the best effort. ``last["tr_bound"]`` marks the envs
    where the limit bound (an iterate was projected or a candidate refused).
  - Cost on a step where some env needs help: at most ``n_iter`` batched Q
    forward+backward passes over all N envs (envs that converged or passed are
    masked, not removed — the monitor may hold the state; the loop ends when
    every env has converged) plus at most ``n_polish + n_grid + n_bisect + 4``
    forward passes (2 more with an action box). Never a loop over envs.

  Telemetry: ``FilterInfo(engaged = not passed through, value = Q(x, u*),
  intervention = ||u* - u_nom||)`` plus ``self.last``, a dict of per-env
  tensors for the latest call:

      status         long: 0 pass, 1 gd, 2 segment, 3 best effort (``STATUS``)
      v, tau         V(x) and the threshold kappa * V(x) + margin
      q_nom          Q(x, u_nom)
      q_safe         Q(x, u_safe) (NaN where not evaluated)
      q              Q(x, u*), the returned action's value
      deviation      ||u* - u_nom||
      gd_iters_used  long: descent iterations the env was active for
      dist_gd        distance of the GD candidate (inf: none feasible / not
                     run)
      dist_seg       distance of the segment candidate (inf likewise)
      tr_bound       bool: the trust region bound (False without ``max_dev``)

  :param kappa: class-K barrier coefficient (the paper's gamma).
  :param margin: added to the threshold, tau = kappa * V + margin.
  :param solver: "gd+segment" | "gd" | "segment".
  :param n_iter: max descent iterations (envs stop early at a KKT point).
  :param lr: step multiplier on 1 / (2 + rho ||dc/du||^2); 1 is the
      linear-model step, smaller is safer on a strongly curved Q.
  :param rho0: initial penalty (dimensionless: c is normalized).
  :param rho_growth: penalty factor at a dual update that found the
      violation not shrunk 4x since the previous one.
  :param rho_max: penalty cap.
  :param dual_every: descent steps between dual updates.
  :param tol: KKT tolerance of the early stop — feasible, 1 - cos(u - u_nom,
      dQ/du) < tol and the first-order slack < tol * ||u - u_nom||; 0 runs
      all ``n_iter`` steps.
  :param n_polish: bisection steps on the ray through the GD candidate.
  :param n_grid: coarse grid points on the segment (a = 1/n_grid, ..., 1).
  :param n_bisect: bisection steps after the grid.
  :param best_effort: "fallback" | "argmax".
  :param action_low/high: optional box (floats or tensors broadcastable to
      (N, A)) for the modified actions.
  :param max_dev: optional trust-region radius on ||u - u_nom|| (None or inf:
      no limit).
  """

  PASS, GD, SEGMENT, BEST_EFFORT = 0, 1, 2, 3
  STATUS = ("pass", "gd", "segment", "best_effort")
  SOLVERS = ("gd+segment", "gd", "segment")
  BEST_EFFORT_RULES = ("fallback", "argmax")

  def __init__(self, num_envs: int, device: str, kappa: float = 0.7,
               margin: float = 0.0, solver: str = "gd+segment",
               n_iter: int = 60, lr: float = 1.0, rho0: float = 2.0,
               rho_growth: float = 2.0, rho_max: float = 1e4,
               dual_every: int = 3, tol: float = 1e-3, n_polish: int = 10,
               n_grid: int = 8, n_bisect: int = 8,
               best_effort: str = "fallback", action_low=None,
               action_high=None, action_dim: int | None = None,
               max_dev: float | None = None):
    super().__init__(num_envs, device, action_dim)
    if solver not in self.SOLVERS:
      raise ValueError(f"solver must be one of {self.SOLVERS}, got {solver!r}")
    if best_effort not in self.BEST_EFFORT_RULES:
      raise ValueError(f"best_effort must be one of {self.BEST_EFFORT_RULES}, "
                       f"got {best_effort!r}")
    if n_iter < 1 or dual_every < 1 or n_grid < 1:
      raise ValueError("n_iter, dual_every and n_grid must be >= 1")
    if not (lr > 0 and rho0 > 0 and rho_growth >= 1 and rho_max >= rho0):
      raise ValueError("need lr > 0, rho0 > 0, rho_growth >= 1, "
                       "rho_max >= rho0")
    if max_dev is not None and not float(max_dev) > 0:
      raise ValueError(f"max_dev must be > 0 (None / inf: no limit), got {max_dev}")
    self.max_dev = (None if max_dev is None or math.isinf(float(max_dev))
                    else float(max_dev))
    self.kappa, self.margin, self.solver = float(kappa), float(margin), solver
    self.n_iter, self.lr = int(n_iter), float(lr)
    self.rho0, self.rho_growth = float(rho0), float(rho_growth)
    self.rho_max, self.dual_every = float(rho_max), int(dual_every)
    self.tol, self.n_polish = float(tol), int(n_polish)
    self.n_grid, self.n_bisect = int(n_grid), int(n_bisect)
    self.best_effort = best_effort
    self.lo, self.hi = action_low, action_high
    self.last: dict[str, torch.Tensor] = {}

  def settings(self) -> dict:
    """The solver's hyperparameters, for run metadata."""
    out = {k: getattr(self, k) for k in (
      "kappa", "margin", "solver", "n_iter", "lr", "rho0", "rho_growth",
      "rho_max", "dual_every", "tol", "n_polish", "n_grid", "n_bisect",
      "best_effort", "max_dev")}
    out["action_box"] = self.lo is not None or self.hi is not None
    return out

  # -- helpers -----------------------------------------------------------------

  def _ball(self, u: torch.Tensor, a_nom: torch.Tensor) -> torch.Tensor:
    """``u`` projected onto the trust region around ``a_nom`` (untouched,
    bit for bit, inside it)."""
    if self.max_dev is None:
      return u
    w = u - a_nom
    r = w.norm(dim=-1, keepdim=True)
    out = r > self.max_dev
    return torch.where(out, a_nom + w * (self.max_dev / r.clamp_min(1e-12)), u)

  def _inside(self, d: torch.Tensor) -> torch.Tensor:
    """``d = ||u - u_nom||`` within the trust region (float slack for the
    ball's own boundary points)."""
    if self.max_dev is None:
      return torch.ones_like(d, dtype=torch.bool)
    return d <= self.max_dev * (1.0 + 1e-4)

  def _box(self, u: torch.Tensor) -> torch.Tensor:
    if self.lo is None and self.hi is None:
      return u
    lo = None if self.lo is None else torch.as_tensor(self.lo, dtype=u.dtype,
                                                      device=u.device)
    hi = None if self.hi is None else torch.as_tensor(self.hi, dtype=u.dtype,
                                                      device=u.device)
    return torch.clamp(u, lo, hi)

  @staticmethod
  def _q(q_fn, u: torch.Tensor, ctx: dict) -> torch.Tensor:
    """Q(x, u) without a graph, checked to be (N,)."""
    with torch.no_grad():
      q = q_fn(action=u, **ctx)
    if q.shape != u.shape[:1]:
      raise ValueError(f"q_fn must return (N,) = {tuple(u.shape[:1])}, "
                       f"got {tuple(q.shape)}")
    return q.detach().to(u.dtype)

  @staticmethod
  def _q_grad(q_fn, u: torch.Tensor,
              ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Q(x, u) and dQ/du — w.r.t. the action ONLY, so the net's parameters
    never see a .grad, whatever the caller's grad mode."""
    with torch.enable_grad():
      ug = u.detach().requires_grad_(True)
      q = q_fn(action=ug, **ctx)
      if q.shape != u.shape[:1]:
        raise ValueError(f"q_fn must return (N,) = {tuple(u.shape[:1])}, "
                         f"got {tuple(q.shape)}")
      if not q.requires_grad:
        raise TypeError(
          "QCBFDescentIntervention: q_fn(action=u) is not differentiable in u "
          "(it returned a tensor without a graph — is it wrapped in no_grad?)")
      (g,) = torch.autograd.grad(q.sum(), ug, allow_unused=True)
    if g is None:                                   # Q ignores the action
      g = torch.zeros_like(u)
    return q.detach().to(u.dtype), g.detach().to(u.dtype)

  @staticmethod
  def _keep_max(top_u, top_q, u, q, mask):
    """Best-effort bookkeeping: the highest-Q action visited so far."""
    up = mask & (q > top_q)
    return torch.where(up.unsqueeze(-1), u, top_u), torch.where(up, q, top_q)

  # -- the two solvers ---------------------------------------------------------

  def _descend(self, q_fn, ctx, a_nom, u0, tau, need, top):
    """AL gradient descent. Returns (u, dist, q) of the closest feasible
    iterate inside the trust region (dist inf where none), the per-env
    iteration count, the updated best-effort bookkeeping ``top`` and where
    the trust region bound."""
    n = a_nom.shape[0]
    inf = torch.full((n,), float("inf"), dtype=a_nom.dtype, device=a_nom.device)
    best_u, best_d, best_q = u0.clone(), inf.clone(), -inf
    lam = torch.zeros_like(inf)
    c_prev = torch.full_like(inf, float("inf"))
    rho = torch.full_like(inf, self.rho0)
    iters = torch.zeros(n, dtype=torch.long, device=a_nom.device)
    hit = torch.zeros_like(need)
    active = need.clone()
    scale = None
    u = u0.clone()

    def record(u, q, best_u, best_d, best_q):
      d = (u - a_nom).norm(dim=-1)
      better = need & (q >= tau) & (d < best_d) & self._inside(d)
      return (torch.where(better.unsqueeze(-1), u, best_u),
              torch.where(better, d, best_d), torch.where(better, q, best_q))

    for k in range(self.n_iter):
      q, g = self._q_grad(q_fn, u, ctx)
      top = self._keep_max(*top, u, q, need)
      best_u, best_d, best_q = record(u, q, best_u, best_d, best_q)
      gnorm = g.norm(dim=-1)
      if scale is None:                    # s = ||dQ/du|| at the start point
        scale = gnorm.clamp_min(1e-6)
      w = u - a_nom
      d = w.norm(dim=-1)
      if self.tol > 0:                     # KKT: feasible, aligned, active
        cos = (w * g).sum(-1) / (d * gnorm).clamp_min(1e-12)
        slack = (q - tau) / gnorm.clamp_min(1e-12)
        kkt = (q >= tau) & (cos > 1.0 - self.tol) & (slack < self.tol * d)
        active = active & ~kkt
      active = active & (gnorm > 0)        # no gradient: descent cannot move
      if not bool(active.any()):
        break
      iters += active.long()
      c = (tau - q) / scale
      gc = -g / scale.unsqueeze(-1)
      mult = (lam + rho * c).clamp_min(0.0)
      grad_l = 2.0 * w + mult.unsqueeze(-1) * gc
      eta = self.lr / (2.0 + rho * gc.pow(2).sum(-1))
      step = u - eta.unsqueeze(-1) * grad_l
      if self.max_dev is not None:
        hit = hit | (active & ((step - a_nom).norm(dim=-1) > self.max_dev))
      u_new = self._box(self._ball(step, a_nom))
      u = torch.where(active.unsqueeze(-1), u_new, u)
      if (k + 1) % self.dual_every == 0:
        lam = torch.where(active, mult, lam)
        # the textbook rule: raise rho only where the violation did not
        # shrink 4x since the last dual update — growing it on every
        # infeasible update freezes the TANGENTIAL progress (the step is
        # ~1 / rho along the boundary) long before the multiplier converges
        grow = active & (c > 0) & (c > 0.25 * c_prev)
        c_prev = torch.where(active, c.clamp_min(0.0), c_prev)
        rho = torch.where(grow, (rho * self.rho_growth).clamp_max(self.rho_max),
                          rho)
    q = self._q(q_fn, u, ctx)                       # the last iterate
    top = self._keep_max(*top, u, q, need)
    best_u, best_d, best_q = record(u, q, best_u, best_d, best_q)

    if self.n_polish > 0:
      # the descent lands NEAR the boundary, feasible or just outside; the
      # closest feasible point along its own direction is one bisection away.
      # Bracket t in [0, 1] on the ray through the best feasible iterate, else
      # t in [1, 2] through the last one (checked feasible at 2 first). Only
      # the upper end has to be feasible for the bisection to return a
      # feasible point.
      found = torch.isfinite(best_d)
      base = torch.where(found.unsqueeze(-1), best_u, u)
      lo = torch.where(found, torch.zeros_like(inf), torch.ones_like(inf))
      hi = lo + 1.0
      ray = lambda t: self._box(self._ball(a_nom + t.unsqueeze(-1) * (base - a_nom),
                                           a_nom))
      q_hi = torch.where(found, best_q, -inf)
      if bool((need & ~found).any()):
        u2 = ray(hi)
        q2 = self._q(q_fn, u2, ctx)
        top = self._keep_max(*top, u2, q2, need & ~found)
        q_hi = torch.where(found, q_hi, q2)
      ok_hi = need & (q_hi >= tau)
      if bool(ok_hi.any()):
        for _ in range(self.n_polish):
          mid = 0.5 * (lo + hi)
          qm = self._q(q_fn, ray(mid), ctx)
          ok = ok_hi & (qm >= tau)
          hi = torch.where(ok, mid, hi)
          q_hi = torch.where(ok, qm, q_hi)
          lo = torch.where(ok_hi & ~ok, mid, lo)
        best_u, best_d, best_q = record(ray(hi), q_hi, best_u, best_d, best_q)
    return best_u, best_d, best_q, iters, top, hit

  def _segment(self, q_fn, ctx, a_nom, s0, s1, tau, need, top):
    """The paper's line search on (1 - a) s0 + a s1: the smallest feasible a
    on a coarse grid, refined by bisection against the grid point before it.
    Returns (u, dist, q) (dist inf where no grid point is feasible, or the
    point lies outside the trust region), the updated best-effort
    bookkeeping, and where the trust region refused the point."""
    n = a_nom.shape[0]
    inf = torch.full((n,), float("inf"), dtype=a_nom.dtype, device=a_nom.device)
    found = torch.zeros_like(need)
    lo, hi, q_hi = torch.zeros_like(inf), torch.ones_like(inf), -inf
    point = lambda a: s0 + a.unsqueeze(-1) * (s1 - s0)
    for i in range(1, self.n_grid + 1):
      a = torch.full_like(inf, i / self.n_grid)
      q = self._q(q_fn, point(a), ctx)
      # "argmax" never picks an extrapolated point outside the trust region
      top = self._keep_max(*top, point(a), q,
                           need & self._inside((point(a) - a_nom).norm(dim=-1)))
      newly = need & ~found & (q >= tau)
      hi = torch.where(newly, a, hi)
      lo = torch.where(newly, a - 1.0 / self.n_grid, lo)
      q_hi = torch.where(newly, q, q_hi)
      found = found | newly
      if not bool((need & ~found).any()):
        break
    if bool(found.any()):
      for _ in range(self.n_bisect):
        mid = 0.5 * (lo + hi)
        qm = self._q(q_fn, point(mid), ctx)
        ok = found & (qm >= tau)
        hi = torch.where(ok, mid, hi)
        q_hi = torch.where(ok, qm, q_hi)
        lo = torch.where(found & ~ok, mid, lo)
    u = point(hi)
    d = torch.where(found, (u - a_nom).norm(dim=-1), inf)
    refused = found & ~self._inside(d)       # certified by Q's extrapolation only
    return u, torch.where(refused, inf, d), q_hi, top, refused

  # -- the intervention --------------------------------------------------------

  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    q_fn = getattr(monitor, "q_fn", None)
    if q_fn is None:
      raise TypeError(
        "QCBFDescentIntervention needs a monitor exposing a differentiable "
        f"q_fn (a CriticMonitor or alike); got {type(monitor).__name__}.")
    v_fn = getattr(monitor, "v_fn", None)
    a_nom = a_nom.detach()
    n = a_nom.shape[0]
    a_safe = fallback(**ctx).detach().to(a_nom.dtype)
    if a_safe.shape != a_nom.shape:
      raise ValueError(f"fallback action {tuple(a_safe.shape)} does not match "
                       f"the nominal {tuple(a_nom.shape)}")

    q_nom = self._q(q_fn, a_nom, ctx)
    q_safe = None
    if v_fn is None:
      q_safe = self._q(q_fn, a_safe, ctx)
      v = q_safe
    else:
      with torch.no_grad():
        v = v_fn(**ctx)
      if v.shape != (n,):
        raise ValueError(f"v_fn must return (N,) = ({n},), "
                         f"got {tuple(v.shape)}")
      v = v.detach().to(a_nom.dtype)
    tau = self.kappa * v + self.margin
    passed = q_nom >= tau
    need = ~passed

    inf = torch.full((n,), float("inf"), dtype=a_nom.dtype, device=a_nom.device)
    status = torch.full((n,), self.PASS, dtype=torch.long, device=a_nom.device)
    iters = torch.zeros(n, dtype=torch.long, device=a_nom.device)
    action, value = a_nom.clone(), q_nom.clone()
    d_gd, d_seg = inf.clone(), inf.clone()
    tr_bound = torch.zeros_like(need)

    if bool(need.any()):
      if torch.is_inference_mode_enabled():
        raise RuntimeError(
          "QCBFDescentIntervention needs autograd for the descent; call it "
          "under torch.no_grad() if you must, not torch.inference_mode().")
      if q_safe is None:
        q_safe = self._q(q_fn, a_safe, ctx)
      s0, s1 = self._box(a_nom), self._box(a_safe)
      q_s0 = q_nom if s0 is a_nom else self._q(q_fn, s0, ctx)
      q_s1 = q_safe if s1 is a_safe else self._q(q_fn, s1, ctx)
      # best-effort bookkeeping starts from the two endpoints (u_safe wins
      # ties and NaNs, so "argmax" never does worse than "fallback" by Q)
      top = self._keep_max(s1.clone(), q_s1.nan_to_num(nan=-float("inf")),
                           s0, q_s0, need)

      u_gd, q_gd = s0, inf
      if self.solver in ("gd+segment", "gd"):
        u_gd, d_gd, q_gd, iters, top, hit = self._descend(q_fn, ctx, a_nom, s0,
                                                          tau, need, top)
        tr_bound = tr_bound | hit
      u_seg, q_seg = s1, inf
      if self.solver in ("gd+segment", "segment"):
        u_seg, d_seg, q_seg, top, hit = self._segment(q_fn, ctx, a_nom, s0, s1,
                                                      tau, need, top)
        tr_bound = tr_bound | hit

      use_gd = need & torch.isfinite(d_gd) & (d_gd <= d_seg)
      use_seg = need & ~use_gd & torch.isfinite(d_seg)
      fail = need & ~use_gd & ~use_seg
      if self.best_effort == "fallback":
        u_be, q_be = s1, q_s1
      else:
        u_be, q_be = top
      for mask, u_c, q_c, code in ((use_gd, u_gd, q_gd, self.GD),
                                   (use_seg, u_seg, q_seg, self.SEGMENT),
                                   (fail, u_be, q_be, self.BEST_EFFORT)):
        action = torch.where(mask.unsqueeze(-1), u_c, action)
        value = torch.where(mask, q_c, value)
        status = torch.where(mask, torch.full_like(status, code), status)

    deviation = (action - a_nom).norm(dim=-1)
    self.last = {
      "status": status, "v": v, "tau": tau, "q_nom": q_nom,
      "q_safe": (q_safe if q_safe is not None
                 else torch.full_like(inf, float("nan"))),
      "q": value, "deviation": deviation, "gd_iters_used": iters,
      "dist_gd": d_gd, "dist_seg": d_seg, "tr_bound": tr_bound & need,
    }
    return action, FilterInfo(engaged=need, value=value,
                              intervention=deviation)
