"""The ViaLasso safe set K and its avoid margin, on the 29-DoF mjlab G1.

The G1 push-recovery pipeline of ``robot_states_reachability_checker``
(``tools/g1/g1c/failure.py``, ``tools/g1/safeppo/env.py``) defines "has not
fallen" as a set K and anchors its avoid margin to it. A state is in K iff all
of these hold:

    pelvis height        z >= 0.45 m                     (LOW_BASE)
    tilt                 arccos(R22) <= 1.0 rad          (TILT; R the pelvis
                                                          rotation, 57 deg)
    non-foot contact     no collision geom but the soles  (NONFOOT_FLOOR)
                         touches the ground
    joint limits         every joint within its MJCF      (JOINT_LIMIT)
                         range widened by 0.1 rad

and the margin (``post_transition_margin``) is

    g = clip(min((z - 0.45) / 0.1, (1.0 - tilt) / 1.0, m_q / 0.1), -3, 3)
    m_q = min_j min(q_j - (lo_j - 0.1), (hi_j + 0.1) - q_j)
    g = -1 on non-foot ground contact
    g = min(g, -0.1) outside K,  max(g, 0) inside K

so g < 0 exactly when the state is outside K, and a failure reads at most
-0.1 whatever the continuous terms say. Here the joint term runs over all 29
joints (the original robot drives only its 12 leg joints).

Two differences from the original, both from mjlab's step structure:

* the original also checks height, tilt and joint limits on every physics
  substep (an excursion inside a 20 ms tick counts as a failure). Here K is
  read once per control step, at the state the termination and reward
  managers grade (``ManagerBasedRlEnv.step``: one physics substep behind
  ``qpos`` for the derived pose, the same lag every termination term has);
* non-foot contact is the ``nonfoot_ground_touch`` sensor's ``found`` field
  (any contact, no force threshold), the sensor ``envs/velocity/g1.py`` builds
  over every ``*_collision`` geom except the soles: the original's rule.

:func:`k_failure` is the termination term (it replaces ``fell_over`` and
``illegal_contact`` in the ``*_k`` tasks, so an episode ends exactly when the
state leaves K) and :func:`g_k` the margin.
"""

from __future__ import annotations

import torch

#: K and margin constants (``configs/nominal.yaml`` failure block and
#: ``safeppo/env.py``).
H_MIN = 0.45
H_SCALE = 0.1
THETA_MAX = 1.0
JOINT_OVERSHOOT = 0.1
JOINT_SCALE = 0.1
FAILURE_ANCHOR = -0.1
CLAMP = 3.0
NONFOOT_SENSOR = "nonfoot_ground_touch"


def tilt(quat_w: torch.Tensor) -> torch.Tensor:
  """arccos(R22): the angle between the pelvis z axis and world z (w-first)."""
  q = quat_w / torch.linalg.vector_norm(quat_w, dim=-1, keepdim=True)
  r22 = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
  return torch.arccos(r22.clamp(-1.0, 1.0))


def nonfoot_contact(env) -> torch.Tensor:
  """Any non-foot collision geom touching the ground now, ``(num_envs,)``."""
  found = env.scene[NONFOOT_SENSOR].data.found
  while found.dim() > 1:
    found = found.amax(dim=-1)
  return found > 0


def terms(env) -> dict[str, torch.Tensor]:
  """The raw quantities K and the margin read: ``z``, ``tilt``, ``joint``
  (m_q, rad: the worst joint's distance inside its widened range, negative
  beyond it) and ``nonfoot`` (bool)."""
  d = env.scene["robot"].data
  q = d.joint_pos
  lim = d.joint_pos_limits
  lo = lim[..., 0] - JOINT_OVERSHOOT
  hi = lim[..., 1] + JOINT_OVERSHOOT
  return {"z": d.root_link_pos_w[:, 2], "tilt": tilt(d.root_link_quat_w),
          "joint": torch.minimum(q - lo, hi - q).amin(dim=-1),
          "nonfoot": nonfoot_contact(env)}


def outside_k(t: dict[str, torch.Tensor]) -> torch.Tensor:
  return ((t["z"] < H_MIN) | (t["tilt"] > THETA_MAX) | (t["joint"] < 0.0)
          | t["nonfoot"])


def k_failure(env) -> torch.Tensor:
  """Termination term: the state is outside K."""
  return outside_k(terms(env))


def g_k(env) -> torch.Tensor:
  """The avoid margin anchored to K (module docstring); g < 0 iff outside K."""
  t = terms(env)
  g = torch.minimum(torch.minimum((t["z"] - H_MIN) / H_SCALE,
                                  (THETA_MAX - t["tilt"]) / THETA_MAX),
                    t["joint"] / JOINT_SCALE).clamp(-CLAMP, CLAMP)
  g = torch.where(t["nonfoot"], torch.full_like(g, -1.0), g)
  out = outside_k(t)
  return torch.where(out, torch.minimum(g, torch.full_like(g, FAILURE_ANCHOR)),
                     g.clamp_min(0.0))


def use_k_terminations(cfg) -> None:
  """End episodes on leaving K: replace the velocity cfg's ``fell_over`` (70
  deg) and ``illegal_contact`` (10 N) terms with :func:`k_failure`. Mutates
  ``cfg`` (a fresh cfg from its builder); ``time_out`` and any rest
  truncation stay."""
  from mjlab.managers.termination_manager import TerminationTermCfg
  cfg.terminations.pop("fell_over", None)
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations["k_failure"] = TerminationTermCfg(func=k_failure)
