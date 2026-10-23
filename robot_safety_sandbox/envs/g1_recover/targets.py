"""Reach targets for the reach-avoid recovery tasks (g1_recover_lafan_ra*).

:func:`make_l_stance` builds g1_stabilize's stance margin ``l_stance`` (pelvis
in its height band, roughly upright, every body-frame velocity component
small; each term divided by its own tolerance, min over the terms) with the
velocity tolerances as parameters, optionally GATED on the push process.

WHAT THE REACH-AVOID VALUE SAYS. ``V = min(g, max(l, gamma V'))`` certifies
that the robot can get INTO the target set without falling first. It says
nothing about what happens after the first reach: it does not promise that
the robot stays in the set, or that it stays up afterwards. The recovery tasks
end an episode (a truncation, ``envs/g1_recover/rest.py``) once ``l >= 0`` has
held for a dwell time with the same margin; that dwell only decides WHEN the
episode ends and adds nothing to what ``V`` certifies. Whether the robot then
stays is for the evaluation to measure (a staying check), not for the
objective to promise.

WHY THE GATE. A LAFAN start state can be slow enough to sit in the target
already, before the push it is about to receive has landed; ungated, the
reach would be credited at step 0 and the push would carry no gradient. With
``gate_on_push`` the margin reads ``-1`` (one full tolerance outside) until the
push process is QUIET -- no push acting, none armed, none left to come this
episode (``random_push.RandomPush.quiet``, published by the bridge as
``env._zoo_push_quiet`` after the step's push is applied and before physics,
so the push acting in a step keeps that step's state out of the target). A
missing attribute (a task without a push process) reads as quiet. Under a push
process without ``max_pushes`` the random pushes never run out and ``quiet``
only means "between pushes"; the recovery tasks therefore cap it at one.
"""

from __future__ import annotations

from typing import Callable

import torch

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
  ANG_TOL, LIN_TOL, PELVIS_BAND, PELVIS_MID, STANCE_COS)

#: The value the gated margin reads while a push is in progress or still to come.
GATED_L = -1.0


def make_l_stance(lin_tol: float = LIN_TOL, ang_tol: float = ANG_TOL,
                  gate_on_push: bool = True) -> Callable:
  """``l_fn(env) -> [num_envs]``: g1_stabilize's ``l_stance`` with per-axis
  body linear velocity <= ``lin_tol`` (m/s) and per-axis body angular velocity
  <= ``ang_tol`` (rad/s); with ``gate_on_push`` it reads :data:`GATED_L` where
  ``env._zoo_push_quiet`` is False. Unclamped (``margins.compose`` clamps).

  ``make_l_stance(LIN_TOL, ANG_TOL, gate_on_push=False)`` is ``l_stance``
  value for value. The returned function carries ``lin_tol``, ``ang_tol`` and
  ``gate_on_push`` attributes so a rest termination can be checked against the
  task's target (they must agree: the episode ends on dwelling in THIS set).
  """
  lin_tol, ang_tol = float(lin_tol), float(ang_tol)
  if lin_tol <= 0.0 or ang_tol <= 0.0:
    raise ValueError(f"make_l_stance needs positive tolerances, got "
                     f"lin_tol={lin_tol}, ang_tol={ang_tol}")
  gate = bool(gate_on_push)

  def l_fn(env) -> torch.Tensor:
    d = env.scene["robot"].data
    base_z = d.root_link_pos_w[:, 2]
    up = -d.projected_gravity_b[:, 2]
    v_b, w_b = d.root_link_lin_vel_b, d.root_link_ang_vel_b
    l = torch.stack([
      (PELVIS_BAND - (base_z - PELVIS_MID).abs()) / PELVIS_BAND,
      (up - STANCE_COS) / (1.0 - STANCE_COS),
      (lin_tol - v_b[:, 0].abs()) / lin_tol,
      (lin_tol - v_b[:, 1].abs()) / lin_tol,
      (lin_tol - v_b[:, 2].abs()) / lin_tol,
      (ang_tol - w_b[:, 0].abs()) / ang_tol,
      (ang_tol - w_b[:, 1].abs()) / ang_tol,
      (ang_tol - w_b[:, 2].abs()) / ang_tol,
    ], dim=0).amin(dim=0)
    if gate:
      quiet = getattr(env, "_zoo_push_quiet", None)
      if quiet is not None:
        l = torch.where(quiet.reshape(-1).to(l.device), l,
                        torch.full_like(l, GATED_L))
    return l

  l_fn.lin_tol, l_fn.ang_tol, l_fn.gate_on_push = lin_tol, ang_tol, gate
  l_fn.__name__ = l_fn.__qualname__ = (
    f"l_stance_lin{lin_tol:g}_ang{ang_tol:g}{'_gated' if gate else ''}")
  return l_fn
