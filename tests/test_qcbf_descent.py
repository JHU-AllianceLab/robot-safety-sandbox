"""Unit tests for QCBFDescentIntervention: the Q-CBF projection by gradient
descent on an augmented Lagrangian, with the segment line search as a net.

    u* = argmin_u ||u - u_nom||^2   s.t.   Q(x, u) >= kappa * V(x) + margin

Toy differentiable Q's with KNOWN answers stand in for the fitted G1 nets:
a linear halfspace (closed-form projection), a quadratic ball (closed-form
projection onto the ball), an MLP in 2-D (brute force on a grid) and an MLP
in 29-D, the G1 action dimension (a long multi-start SQP reference and the
KKT conditions). Everything is seeded and CPU-only, and sized for a login node.
"""

from __future__ import annotations

import math
import os
import sys
import types

import pytest
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.filters import (  # noqa: E402
  CriticMonitor, FilterInfo, PolicyFallback, QCBFDescentIntervention,
  SafetyFilter)

DEV = "cpu"
A = 29                                 # the G1 action dimension
Q_DESCENT = QCBFDescentIntervention


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
  """The toy problems are tiny, so intra-op threads only cost — and on a
  CPU-quota'd login node (one core, 58 visible CPUs) torch's default of one
  OpenMP thread per visible CPU turned this file from seconds into a timeout.
  Restored afterwards so no other test module is affected."""
  prev = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(prev)


# --- toy problems ------------------------------------------------------------

def _monitor(q_fn, v_fn=None):
  """The duck-typed monitor the intervention reads: q_fn (+ optional v_fn)."""
  mon = types.SimpleNamespace(q_fn=q_fn)
  if v_fn is not None:
    mon.v_fn = v_fn
  return mon


def _const(t):
  return lambda **_: t


def _linear(n, gen, slope=(0.05, 5.0), dist=(0.05, 3.0)):
  """Q(u) = w.u + b with ||w|| spread over two decades (the normalization
  must make one set of defaults work for every slope), and tau placed so the
  true minimal deviation is ``dist``. Returns q_fn, a_nom, tau, u*."""
  w = torch.nn.functional.normalize(torch.randn(n, A, generator=gen), dim=-1)
  w = w * torch.empty(n, 1).uniform_(*slope, generator=gen)
  b = torch.randn(n, generator=gen)
  a = torch.randn(n, A, generator=gen) * 1.5
  d = torch.empty(n).uniform_(*dist, generator=gen)
  q_a = (w * a).sum(-1) + b
  tau = q_a + d * w.norm(dim=-1)
  u_star = a + ((tau - q_a) / w.pow(2).sum(-1)).unsqueeze(-1) * w
  return (lambda action, **_: (w * action).sum(-1) + b), a, tau, u_star


def _ball(n, gen):
  """Q(u) = r^2 - ||u - c||^2 >= 0: the feasible set is a ball, the answer
  the radial projection of u_nom onto it."""
  c = torch.randn(n, A, generator=gen)
  r = torch.empty(n).uniform_(0.5, 2.5, generator=gen)
  out = torch.empty(n).uniform_(0.05, 3.0, generator=gen)
  a = c + torch.nn.functional.normalize(torch.randn(n, A, generator=gen),
                                        dim=-1) * (r + out).unsqueeze(-1)
  u_star = c + torch.nn.functional.normalize(a - c, dim=-1) * r.unsqueeze(-1)
  q_fn = lambda action, **_: r.pow(2) - (action - c).pow(2).sum(-1)
  return q_fn, a, torch.zeros(n), u_star


class _QNet(nn.Module):
  """ValueNet-shaped toy critic: standardized cat(x, u) -> ELU MLP -> (N,).
  ``gain`` > 1 makes it visibly nonlinear over a few action units."""

  def __init__(self, dx, da, hidden, gain, gen):
    super().__init__()
    layers, d = [], dx + da
    for h in hidden:
      lin = nn.Linear(d, h)
      with torch.no_grad():
        lin.weight.copy_(torch.randn(h, d, generator=gen) * gain / math.sqrt(d))
        lin.bias.copy_(torch.randn(h, generator=gen) * 0.5)
      layers += [lin, nn.ELU()]
      d = h
    out = nn.Linear(d, 1)
    with torch.no_grad():
      out.weight.copy_(torch.randn(1, d, generator=gen) / math.sqrt(d))
      out.bias.zero_()
    self.net = nn.Sequential(*layers, out)
    self.register_buffer("mean", torch.randn(dx + da, generator=gen) * 0.1)
    self.register_buffer("std", torch.empty(dx + da).uniform_(0.5, 2.0,
                                                              generator=gen))

  def forward(self, x, u):
    return self.net((torch.cat([x, u], -1) - self.mean) / self.std).squeeze(-1)


def _mlp_problem(n, gen, da=A, dx=16, hidden=(128, 128), gain=2.0,
                 dist=(0.1, 2.5)):
  """An MLP critic and thresholds a first-order ``dist`` above Q(u_nom)."""
  net = _QNet(dx, da, hidden, gain, gen)
  for p in net.parameters():
    p.requires_grad_(False)
  x = torch.randn(n, dx, generator=gen)
  a = torch.randn(n, da, generator=gen)
  ug = a.clone().requires_grad_(True)
  q_a = net(x, ug)
  g = torch.autograd.grad(q_a.sum(), ug)[0]
  d = torch.empty(n).uniform_(*dist, generator=gen)
  tau = q_a.detach() + d * g.norm(dim=-1)
  return net, x, a, tau


def _sqp_reference(q_fn, a, tau, gen, n_iter=200, starts=3):
  """Independent reference solver for an MLP: damped SQP (linearize Q, project
  u_nom onto the linearized halfspace) from u_nom and from random starts
  around it, each polished by bisection on its ray; the closest feasible
  result wins (inf where none is feasible)."""
  q0, g0 = _value_grad(q_fn, a)
  d_lin = ((tau - q0) / g0.norm(dim=-1)).clamp_min(0).unsqueeze(-1)
  best = torch.full((a.shape[0],), float("inf"))
  for s in range(starts):
    kick = torch.randn(a.shape, generator=gen) * d_lin / math.sqrt(a.shape[1])
    u = a + (0 if s == 0 else kick)
    for _ in range(n_iter):
      q, g = _value_grad(q_fn, u)
      step = (tau - q - (g * (a - u)).sum(-1)).clamp_min(0)
      u = u + 0.5 * (a + (step / g.pow(2).sum(-1)).unsqueeze(-1) * g - u)
    lo, hi = torch.full_like(best, 0.9), torch.full_like(best, 2.0)
    ray = lambda t: a + t.unsqueeze(-1) * (u - a)
    with torch.no_grad():
      ok_hi = q_fn(ray(hi)) >= tau
      for _ in range(30):
        mid = 0.5 * (lo + hi)
        ok = q_fn(ray(mid)) >= tau
        hi, lo = torch.where(ok, mid, hi), torch.where(ok, lo, mid)
      feas = ok_hi & (q_fn(ray(hi)) >= tau)
    d = (ray(hi) - a).norm(dim=-1)
    best = torch.where(feas & (d < best), d, best)
  return best


def _value_grad(q_fn, u):
  with torch.enable_grad():
    ug = u.detach().requires_grad_(True)
    q = q_fn(ug)
    return q.detach(), torch.autograd.grad(q.sum(), ug)[0]


def _solve(q_fn, a, tau, a_safe=None, v_fn=None, **kw):
  """Run the intervention with V := tau (kappa = 1, margin 0) unless told
  otherwise; returns (action, info, last)."""
  n = a.shape[0]
  itv = Q_DESCENT(n, DEV, **{"kappa": 1.0, **kw})
  mon = _monitor(q_fn, _const(tau) if v_fn is None else v_fn)
  a_safe = a.clone() if a_safe is None else a_safe
  u, info = itv(a, _const(a_safe), mon)
  return u, info, itv.last


# --- contract ----------------------------------------------------------------

def test_exported_and_a_sibling_of_the_projected_gradient_qcbf():
  from robot_safety_sandbox.filters import OptIntervention, QCBFIntervention
  assert issubclass(Q_DESCENT, OptIntervention)
  assert Q_DESCENT is not QCBFIntervention


def test_pass_through_is_exact_and_costs_one_forward():
  gen = torch.Generator().manual_seed(0)
  q_fn, a, tau, _ = _linear(64, gen)
  tau = tau.clone()
  tau[::2] = -1e3                                  # half the envs are safe
  calls = []

  def counting_q(action, **ctx):
    calls.append(action.requires_grad)
    return q_fn(action)

  u, info, last = _solve(counting_q, a, tau)
  ok = ~info.engaged
  assert torch.equal(ok, tau < -100)
  assert torch.equal(u[ok], a[ok])
  assert torch.all(last["status"][ok] == Q_DESCENT.PASS)
  assert torch.all(last["gd_iters_used"][ok] == 0)
  assert torch.all(info.intervention[ok] == 0)
  assert torch.all(last["status"][~ok] != Q_DESCENT.PASS)

  # every env safe: ONE Q evaluation (V comes from v_fn), no gradient
  calls.clear()
  u, info, last = _solve(counting_q, a, torch.full((64,), -1e3))
  assert calls == [False]
  assert torch.equal(u, a) and not info.engaged.any()
  assert torch.isnan(last["q_safe"]).all()         # not evaluated


def test_telemetry_shapes_and_meaning():
  gen = torch.Generator().manual_seed(1)
  q_fn, a, tau, _ = _linear(32, gen)
  u, info, last = _solve(q_fn, a, tau, margin=0.0)
  assert isinstance(info, FilterInfo)
  for key in ("status", "v", "tau", "q_nom", "q_safe", "q", "deviation",
              "gd_iters_used", "dist_gd", "dist_seg", "tr_bound"):
    assert last[key].shape == (32,), key
    assert not last[key].requires_grad, key
  assert last["status"].dtype == torch.long
  assert last["gd_iters_used"].dtype == torch.long
  assert torch.allclose(info.value, q_fn(u))
  assert torch.allclose(last["q"], info.value)
  assert torch.allclose(info.intervention, (u - a).norm(dim=-1))
  assert torch.allclose(last["q_nom"], q_fn(a))
  assert set(last["status"].tolist()) <= set(range(len(Q_DESCENT.STATUS)))


def test_threshold_is_kappa_v_plus_margin_and_v_defaults_to_q_of_fallback():
  gen = torch.Generator().manual_seed(2)
  q_fn, a, _, _ = _linear(16, gen)
  a_safe = a + torch.randn(a.shape, generator=gen)
  itv = Q_DESCENT(16, DEV, kappa=0.7, margin=0.1)
  itv(a, _const(a_safe), _monitor(q_fn))           # no v_fn
  assert torch.allclose(itv.last["v"], q_fn(a_safe))
  assert torch.allclose(itv.last["q_safe"], q_fn(a_safe))
  assert torch.allclose(itv.last["tau"], 0.7 * q_fn(a_safe) + 0.1)


def test_contract_errors():
  gen = torch.Generator().manual_seed(3)
  q_fn, a, tau, _ = _linear(8, gen)
  itv = Q_DESCENT(8, DEV)
  with pytest.raises(TypeError, match="q_fn"):
    itv(a, _const(a), types.SimpleNamespace(value_fn=None))
  with pytest.raises(ValueError, match=r"\(N,\)"):
    itv(a, _const(a), _monitor(lambda action, **_: q_fn(action)[:, None]))
  with pytest.raises(ValueError, match="fallback"):
    itv(a, _const(a[:, :3]), _monitor(q_fn, _const(tau)))

  def frozen_q(action, **_):
    with torch.no_grad():
      return q_fn(action)
  with pytest.raises(TypeError, match="differentiable"):
    itv(a, _const(a), _monitor(frozen_q, _const(tau)))
  with torch.inference_mode():
    with pytest.raises(RuntimeError, match="inference_mode"):
      itv(a.clone(), _const(a.clone()), _monitor(q_fn, _const(tau.clone())))
  for bad in (dict(solver="newton"), dict(best_effort="zero"), dict(n_iter=0)):
    with pytest.raises(ValueError):
      Q_DESCENT(8, DEV, **bad)


# --- optimality against known answers ----------------------------------------

def test_linear_halfspace_matches_the_closed_form_projection():
  gen = torch.Generator().manual_seed(10)
  q_fn, a, tau, u_star = _linear(256, gen)
  u, info, last = _solve(q_fn, a, tau, solver="gd")
  d_star = (u_star - a).norm(dim=-1)
  assert torch.all(q_fn(u) >= tau), "output must be feasible"
  assert torch.all(last["status"] == Q_DESCENT.GD)
  rel = info.intervention / d_star - 1.0
  assert rel.min() > -1e-4                         # cannot beat the optimum
  assert rel.max() < 2e-3, f"worst excess deviation {rel.max():.2e}"
  assert ((u - u_star).norm(dim=-1) / d_star).max() < 0.05


def test_quadratic_ball_matches_the_closed_form_projection():
  gen = torch.Generator().manual_seed(11)
  q_fn, a, tau, u_star = _ball(256, gen)
  u, info, last = _solve(q_fn, a, tau, solver="gd")
  d_star = (u_star - a).norm(dim=-1)
  assert torch.all(q_fn(u) >= tau)
  assert torch.all(last["status"] == Q_DESCENT.GD)
  rel = info.intervention / d_star - 1.0
  assert rel.min() > -1e-4 and rel.max() < 2e-3, (rel.min(), rel.max())


def test_mlp_in_two_dims_matches_brute_force():
  """In 2-D the GLOBAL optimum is a grid search away."""
  gen = torch.Generator().manual_seed(12)
  n, h, half = 6, 0.02, 5.0
  net, x, a, tau = _mlp_problem(n, gen, da=2, dx=4, hidden=(16, 16),
                                gain=2.5, dist=(0.2, 1.2))
  q_fn = lambda action, **_: net(x, action)
  u, info, last = _solve(q_fn, a, tau, solver="gd")
  assert torch.all(q_fn(u) >= tau)
  ticks = torch.arange(-half, half + h / 2, h)
  grid = torch.stack(torch.meshgrid(ticks, ticks, indexing="ij"), -1)
  grid = grid.view(-1, 2)
  d_brute = torch.full((n,), float("inf"))
  with torch.no_grad():
    for i in range(n):
      cand = a[i] + grid
      feas = net(x[i].expand(len(cand), -1), cand) >= tau[i]
      if feas.any():
        d_brute[i] = (cand[feas] - a[i]).norm(dim=-1).min()
  assert torch.all(d_brute < half), "the grid must contain the optimum"
  # the grid's own resolution is h * sqrt(2) / 2
  assert torch.all(info.intervention >= d_brute - h)
  assert torch.all(info.intervention <= d_brute * 1.02 + h), (
    info.intervention, d_brute)


def test_mlp_in_29_dims_is_feasible_near_optimal_and_kkt():
  gen = torch.Generator().manual_seed(13)
  n = 64
  net, x, a, tau = _mlp_problem(n, gen)
  q_fn = lambda action, **_: net(x, action)
  u, info, last = _solve(q_fn, a, tau, solver="gd")
  assert torch.all(last["status"] == Q_DESCENT.GD), "descent must succeed"
  q, g = _value_grad(lambda v: net(x, v), u)
  assert torch.all(q >= tau)
  d_ref = _sqp_reference(lambda v: net(x, v), a, tau, gen)
  assert torch.isfinite(d_ref).all()
  ratio = info.intervention / d_ref                 # < 1: GD beat the reference
  assert ratio.median() < 1.005 and ratio.max() < 1.05, ratio
  # KKT of the projection: u - u_nom parallel to dQ/du, constraint active
  w = u - a
  cos = (w * g).sum(-1) / (w.norm(dim=-1) * g.norm(dim=-1))
  assert cos.min() > 0.99
  slack = (q - tau) / g.norm(dim=-1)                # first-order, action units
  assert (slack / w.norm(dim=-1)).max() < 0.01


# --- safety net, best effort, modes ------------------------------------------

def _plateau(n, gen):
  """Q flat (zero gradient) around u_nom, rising toward u_safe: descent
  cannot move, the segment search still finds the boundary."""
  a = torch.randn(n, A, generator=gen)
  e = torch.nn.functional.normalize(torch.randn(n, A, generator=gen), dim=-1)
  a_safe = a + 3.0 * e
  q_fn = lambda action, **_: (((action - a) * e).sum(-1) - 1.0).clamp_min(0.0)
  return q_fn, a, a_safe, torch.full((n,), 0.5)    # boundary at 1.5 along e


def test_segment_rescues_a_stalled_descent():
  gen = torch.Generator().manual_seed(20)
  q_fn, a, a_safe, tau = _plateau(16, gen)
  u, info, last = _solve(q_fn, a, tau, a_safe=a_safe)              # gd+segment
  assert torch.all(last["status"] == Q_DESCENT.SEGMENT)
  assert torch.all(q_fn(u) >= tau)
  assert torch.allclose(info.intervention, torch.full((16,), 1.5), atol=0.01)
  _, _, last_gd = _solve(q_fn, a, tau, a_safe=a_safe, solver="gd")
  assert torch.all(last_gd["status"] == Q_DESCENT.BEST_EFFORT)


def test_segment_mode_finds_the_closest_feasible_point_on_the_segment():
  gen = torch.Generator().manual_seed(21)
  q_fn, a, tau, u_star = _linear(64, gen)
  normal = torch.nn.functional.normalize(u_star - a, dim=-1)
  noise = torch.randn(a.shape, generator=gen)
  noise = noise - (noise * normal).sum(-1, keepdim=True) * normal
  a_safe = a + 3.0 * (u_star - a) + 0.3 * noise    # feasible, off the normal
  q_a, q_s = q_fn(a), q_fn(a_safe)
  assert torch.all(q_s > tau)
  alpha = (tau - q_a) / (q_s - q_a)                # Q is linear on the segment
  u_ref = a + alpha.unsqueeze(-1) * (a_safe - a)
  u, info, last = _solve(q_fn, a, tau, a_safe=a_safe, solver="segment")
  assert torch.all(last["status"] == Q_DESCENT.SEGMENT)
  assert torch.all(last["gd_iters_used"] == 0)
  assert torch.all(q_fn(u) >= tau)
  seg_len = (a_safe - a).norm(dim=-1)
  assert ((u - u_ref).norm(dim=-1) / seg_len).max() < 2.0 ** -10


def test_default_mode_returns_the_closer_of_the_two_candidates():
  gen = torch.Generator().manual_seed(22)
  q_fn, a, tau, u_star = _linear(64, gen)
  a_safe = u_star + 0.5 * torch.randn(a.shape, generator=gen)
  a_safe = torch.where((q_fn(a_safe) >= tau).unsqueeze(-1), a_safe,
                       2 * u_star - a)
  u, info, last = _solve(q_fn, a, tau, a_safe=a_safe)
  best = torch.minimum(last["dist_gd"], last["dist_seg"])
  assert torch.allclose(info.intervention, best)
  want = torch.where(last["dist_gd"] <= last["dist_seg"], Q_DESCENT.GD,
                     Q_DESCENT.SEGMENT)
  assert torch.equal(last["status"], want)
  # the projection is never beaten by a point of the segment
  assert torch.all(last["dist_gd"] <= last["dist_seg"] * (1 + 1e-3))


@pytest.mark.parametrize("rule", ["fallback", "argmax"])
def test_best_effort_when_nothing_is_feasible(rule):
  gen = torch.Generator().manual_seed(30)
  n = 16
  c = torch.randn(n, A, generator=gen)
  q_fn = lambda action, **_: -1.0 - (action - c).pow(2).sum(-1)   # Q <= -1
  a = c + torch.randn(n, A, generator=gen)
  a_safe = c + 2 * torch.randn(n, A, generator=gen)
  u, info, last = _solve(q_fn, a, torch.zeros(n), a_safe=a_safe,
                         best_effort=rule)
  assert torch.all(last["status"] == Q_DESCENT.BEST_EFFORT)
  assert info.engaged.all()
  assert torch.isinf(last["dist_gd"]).all()
  assert torch.isinf(last["dist_seg"]).all()
  if rule == "fallback":
    assert torch.equal(u, a_safe)
  else:
    # the highest-Q visited candidate: at least as good as both endpoints
    assert torch.all(q_fn(u) >= torch.maximum(q_fn(a), q_fn(a_safe)))
    assert torch.allclose(info.value, q_fn(u))


def test_batched_envs_are_independent():
  """A mixed batch (pass / gd / segment / best effort) gives every env the
  same answer it gets alone: masks, never cross-env coupling."""
  gen = torch.Generator().manual_seed(40)
  n = 12
  # one elementwise family, Q_i(u) = clamp(w_i . u + b_i, floor_i, cap_i):
  # rows 0-3 linear (descent), 4-6 safe nominal (pass), 7-9 flat at u_nom
  # (only the segment finds the boundary), 10-11 capped below tau (nothing
  # is feasible)
  w = torch.nn.functional.normalize(torch.randn(n, A, generator=gen), dim=-1)
  a = torch.randn(n, A, generator=gen)
  b = -(w * a).sum(-1) - 1.0                       # Q(u_nom) = -1 (unclamped)
  floor = torch.full((n,), -1e9)
  cap = torch.full((n,), 1e9)
  floor[7:10] = 0.0
  cap[10:] = 0.5
  tau = torch.full((n,), 0.5)
  tau[4:7] = -1e3
  tau[10:] = 1.0
  a_safe = a + 3.0 * w + 0.2 * torch.randn(n, A, generator=gen)
  q_rows = lambda action, rows: torch.minimum(
    torch.maximum((w[rows] * action).sum(-1) + b[rows], floor[rows]), cap[rows])
  mon = _monitor(q_rows, lambda rows: tau[rows])
  fb = lambda rows: a_safe[rows]
  itv = Q_DESCENT(n, DEV, kappa=1.0)
  rows = torch.arange(n)
  u_all, _ = itv(a, fb, mon, rows=rows)
  status = itv.last["status"]
  assert status.tolist() == [1] * 4 + [0] * 3 + [2] * 3 + [3] * 2
  for i in range(n):
    one = Q_DESCENT(1, DEV, kappa=1.0)
    u_i, _ = one(a[i:i + 1], fb, mon, rows=rows[i:i + 1])
    assert one.last["status"].item() == status[i].item(), i
    assert torch.allclose(u_i[0], u_all[i], atol=1e-6), i


def test_action_box_clamps_modified_actions_only():
  gen = torch.Generator().manual_seed(50)
  n = 32
  w = torch.nn.functional.normalize(torch.randn(n, A, generator=gen), dim=-1)
  q_fn = lambda action, **_: (w * action).sum(-1)
  a = 0.5 * torch.randn(n, A, generator=gen).clamp(-1, 1)
  tau = q_fn(a) + 1.0                              # reachable inside [-1, 1]
  tau[:4] = -1e3
  a[:4] = 3.0                                # a safe nominal outside the box
  u, info, last = _solve(q_fn, a, tau, action_low=-1.0, action_high=1.0)
  assert torch.equal(u[:4], a[:4]), "a passed-through nominal is not clamped"
  assert u[4:].abs().max() <= 1.0 + 1e-6
  assert torch.all(q_fn(u)[4:] >= tau[4:])
  assert torch.all(last["status"][4:] == Q_DESCENT.GD)


@pytest.mark.parametrize("solver", ["gd+segment", "gd", "segment"])
def test_trust_region_keeps_the_answer_near_the_nominal_or_hands_over(solver):
  """A Q unbounded away from u_nom (a halfspace) "certifies" any far action.
  With max_dev the answer lies within max_dev of u_nom, or is the fallback's
  action: no extrapolated certificate is accepted."""
  gen = torch.Generator().manual_seed(55)
  n, R = 64, 2.0
  q_fn, a, tau, u_star = _linear(n, gen, dist=(0.3, 6.0))
  d_star = (u_star - a).norm(dim=-1)
  w = torch.nn.functional.normalize(u_star - a, dim=-1)
  a_safe = a + w * (d_star + 1.0).unsqueeze(-1)       # feasible, beyond u*
  near, far = d_star < 0.8 * R, d_star > R           # the region holds u* / cannot
  assert bool(near.any()) and bool(far.any())
  u, info, last = _solve(q_fn, a, tau, a_safe=a_safe, solver=solver, max_dev=R)
  dev = info.intervention
  assert torch.all((dev <= R * (1 + 1e-4)) | torch.all(u == a_safe, dim=-1))
  assert torch.all(last["status"][far] == Q_DESCENT.BEST_EFFORT)
  assert torch.equal(u[far], a_safe[far])
  assert bool(last["tr_bound"][far].all())
  # where the region holds the projection, the answer is the projection
  assert torch.all(last["status"][near] != Q_DESCENT.BEST_EFFORT)
  assert torch.all(q_fn(u)[near] >= tau[near])
  assert float((dev[near] / d_star[near] - 1).abs().max()) < 1e-2
  if solver == "gd":
    assert torch.all(last["status"][near] == Q_DESCENT.GD)
  # no limit: the far envs get their far projection back
  u2, info2, last2 = _solve(q_fn, a, tau, a_safe=a_safe, solver=solver, max_dev=float("inf"))
  assert bool((info2.intervention[far] > R).all()) and not bool(last2["tr_bound"].any())
  assert torch.all(last2["status"][far] != Q_DESCENT.BEST_EFFORT)


def test_trust_region_contract_and_settings():
  with pytest.raises(ValueError, match="max_dev"):
    Q_DESCENT(4, DEV, max_dev=0.0)
  assert Q_DESCENT(4, DEV, max_dev=float("inf")).max_dev is None
  s = Q_DESCENT(4, DEV, kappa=0.5, max_dev=3.0, action_low=-1.0, action_high=1.0).settings()
  assert s["kappa"] == 0.5 and s["max_dev"] == 3.0 and s["action_box"] is True
  assert {"lr", "rho0", "rho_growth", "rho_max", "dual_every", "tol", "n_polish", "n_grid",
          "n_bisect", "n_iter", "margin", "solver", "best_effort"} <= set(s)
  # inside the region every step is the unlimited solver's, bit for bit
  gen = torch.Generator().manual_seed(56)
  q_fn, a, tau, u_star = _linear(32, gen, dist=(0.1, 0.5))
  u1, _, l1 = _solve(q_fn, a, tau, max_dev=50.0)
  u0, _, l0 = _solve(q_fn, a, tau)
  assert torch.equal(u1, u0) and torch.equal(l1["status"], l0["status"])
  assert not bool(l1["tr_bound"].any())
  # "argmax" best effort: the highest-Q visited point INSIDE the region (or
  # u_safe), not the segment's midpoint 4 away where this Q peaks
  n = 16
  a = torch.randn(n, A, generator=gen)
  c = a + 4.0 * torch.nn.functional.normalize(torch.randn(n, A, generator=gen), dim=-1)
  q_fn = lambda action, **_: -(action - c).pow(2).sum(-1)     # <= 0 < tau: infeasible
  u, info, last = _solve(q_fn, a, torch.ones(n), a_safe=a + 2.0 * (c - a), max_dev=2.0,
                         best_effort="argmax")
  assert torch.all(last["status"] == Q_DESCENT.BEST_EFFORT)
  assert float(info.intervention.max()) <= 2.0 * (1 + 1e-4)
  assert float(q_fn(u).min()) > -4.0 - 1e-2                    # the ball's best point


# --- autograd hygiene and cost -----------------------------------------------

def test_no_gradient_leaks_under_a_no_grad_caller():
  gen = torch.Generator().manual_seed(60)
  net, x, a, tau = _mlp_problem(32, gen, hidden=(32, 32))
  for p in net.parameters():
    p.requires_grad_(True)                         # a live, trainable net
  policy = nn.Linear(A, A)                         # a nominal with a graph
  a_nom = policy(a)
  assert a_nom.requires_grad
  q_fn = lambda action, **_: net(x, action)
  itv = Q_DESCENT(32, DEV, kappa=1.0)
  with torch.no_grad():
    tau = net(x, a_nom) + 0.5
    u, info = itv(a_nom, _const(a.clone()), _monitor(q_fn, _const(tau)))
    assert not torch.is_grad_enabled(), "the caller's grad mode is restored"
  assert itv.last["gd_iters_used"].sum() > 0, "the descent must have run"
  assert all(p.grad is None for p in net.parameters())
  assert all(p.grad is None for p in policy.parameters())
  assert not u.requires_grad
  for t in (info.engaged, info.value, info.intervention, *itv.last.values()):
    assert not t.requires_grad


def test_cost_is_batched_and_independent_of_the_batch_size():
  """4096 envs: every Q call sees the whole batch, gradients are taken at most
  n_iter times, and the number of calls does not grow with N."""
  for n in (64, 4096):
    gen = torch.Generator().manual_seed(70)
    net, x, a, tau = _mlp_problem(n, gen, dx=8, hidden=(32,), gain=1.5)
    shapes, grads = [], [0]

    def q_fn(action, **_):
      shapes.append(tuple(action.shape))
      grads[0] += int(action.requires_grad)
      return net(x, action)

    itv = Q_DESCENT(n, DEV, kappa=1.0)
    a_safe = a + torch.randn(a.shape, generator=gen)
    with torch.no_grad():
      itv(a, _const(a_safe), _monitor(q_fn, _const(tau)))
    assert set(shapes) == {(n, A)}
    assert grads[0] <= itv.n_iter
    assert len(shapes) <= (itv.n_iter + itv.n_polish + itv.n_grid
                           + itv.n_bisect + 4)
    assert itv.last["gd_iters_used"].max() > 0, "the descent must have run"


def test_composes_with_safety_filter_and_a_critic_monitor():
  gen = torch.Generator().manual_seed(80)
  net, x, a, tau = _mlp_problem(16, gen, hidden=(32, 32))
  q_fn = lambda action, s_obs, **_: net(s_obs, action)
  a_safe = a + torch.randn(a.shape, generator=gen)
  filt = SafetyFilter(PolicyFallback(16, DEV, lambda s_obs, **_: a_safe, A),
                      CriticMonitor(16, DEV, q_fn, A),
                      Q_DESCENT(16, DEV, kappa=0.7, action_dim=A))
  u, info = filt(a, s_obs=x)
  assert u.shape == (16, A) and isinstance(info, FilterInfo)
  last = filt.intervention.last
  assert torch.allclose(last["v"], net(x, a_safe).detach())   # V = Q(x, u_safe)
  ok = last["status"] != Q_DESCENT.BEST_EFFORT
  assert torch.all(net(x, u)[ok] >= last["tau"][ok] - 1e-6)
  filt.reset(torch.ones(16, dtype=torch.bool))     # stateless: a no-op
