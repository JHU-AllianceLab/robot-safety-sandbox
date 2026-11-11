"""The unified G1 fallback setup (Zizhe + Yuanhong, agreed 2026-11-09 / 11-10).

One environment both pipelines train and evaluate on, so their numbers are
comparable. It is ``g1_recover_thermal_cbf`` (the identified G1 with its heat
model and the 70 C temperature barrier below every controller) with three
changes, all here:

* MARGIN. ``g = clamp(min(70 deg - tilt, d_joint / 0.2), -3, 3)``, tilt in rad
  (the angle between the pelvis z axis and world z), d_joint the worst joint's
  distance inside its MJCF range widened by 0.1 rad (the S7-SONIC joint term,
  ``k_set``'s widening), negative beyond it. No pelvis-height term, no non-foot
  contact term, no temperature term (the barrier owns temperature). A failure
  step reads at most -0.1 (the reward hook's anchor, ``base.TERMINAL_MARGIN``).
  Upright and mid-range the margin reads min(1.22, ...).
* FAILURE. Only ``fell_over``, tilt > 70 deg, checked once per control step
  (20 ms) as every mjlab termination is. The 10 N ``illegal_contact``
  termination is dropped; a joint past its widened range makes g negative but
  does not end the episode (as the pelvis height did in g1_recover).
* CRITIC. 98 noise-free actor terms + the pelvis linear velocity in the
  HEADING (yaw-only) frame = 101-D, plus the barrier tasks' winding and housing
  temperatures (29 + 29) = 159-D. No foot terms, no pelvis height, no
  body-frame velocimeter. The actor is the thermal tasks' 98 + 29 winding
  temperatures = 127-D.

Unchanged from the thermal_cbf tasks (and so from g1_recover): the identified
G1 (``envs/assets/g1_sysid.py``: 7520-14 on hip pitch, dcmotor actuators,
measured ranges, sole capsules condim 3 and every other geom condim 1,
self-collision on), MuJoCo / Warp 3.16, 5 ms x 4 = 50 Hz, solver 10 / 20 / CCD
50, domain randomization foot friction U(0.3, 1.2) + torso COM + encoder bias
(drawn once per env), the 29-D action clamped to +-3 with
``target = default + scale * a - bias``, uniform actor sensor noise.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg

#: failure: pelvis tilt beyond this (rad) -- velocity cfg's ``fell_over``
TILT_MAX = math.radians(70.0)
#: the joint term: each joint's MJCF range widened by this much (rad) ...
JOINT_OVERSHOOT = 0.1
#: ... and the distance inside it divided by this (rad)
JOINT_SCALE = 0.2
#: the critic's velocity term (replaces the body-frame ``base_lin_vel``)
CRITIC_LIN_VEL_TERM = "base_lin_vel_yaw"


def pelvis_tilt(env) -> torch.Tensor:
  """Angle between the pelvis z axis and world z, rad, ``(num_envs,)``: what
  ``fell_over`` (mjlab ``bad_orientation``) compares with 70 deg."""
  up = -env.scene["robot"].data.projected_gravity_b[:, 2]
  return torch.arccos(up.clamp(-1.0, 1.0))


def joint_clearance(env) -> torch.Tensor:
  """The worst joint's distance inside its range widened by
  ``JOINT_OVERSHOOT``, rad, ``(num_envs,)``; negative beyond it."""
  d = env.scene["robot"].data
  q, lim = d.joint_pos, d.joint_pos_limits
  lo = lim[..., 0] - JOINT_OVERSHOOT
  hi = lim[..., 1] + JOINT_OVERSHOOT
  return torch.minimum(q - lo, hi - q).amin(dim=-1)


def g_tilt_joint(env) -> torch.Tensor:
  """AVOID margin: ``min(70 deg - tilt, d_joint / 0.2)``. Unclamped;
  ``compose`` clamps to +-3, the reward hook anchors failures at -0.1."""
  return torch.minimum(TILT_MAX - pelvis_tilt(env), joint_clearance(env) / JOINT_SCALE)


#: what a CPU rollout must reproduce (cpu_rollout.build_spec reads it)
g_tilt_joint.fall_height_term = False
g_tilt_joint.tilt_joint = {"tilt_max": TILT_MAX, "joint_overshoot": JOINT_OVERSHOOT,
                           "joint_scale": JOINT_SCALE}


def base_lin_vel_yaw(env) -> torch.Tensor:
  """Pelvis linear velocity in the heading (yaw-only) frame, m/s, ``(n, 3)``."""
  from robot_safety_sandbox.envs.g1_recover.push_bench import lin_vel_heading
  return lin_vel_heading(env)


def g1_recover_unified_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`~robot_safety_sandbox.envs.g1_recover.env_cfg.g1_recover_thermal_cbf_env_cfg`
  with the unified critic and terminations (module docstring). Same kwargs
  (``env_overrides``), but the critic-layout switches are fixed."""
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_thermal_cbf_env_cfg
  for k in ("critic_pelvis_height", "critic_foot_terms", "critic_base_lin_vel"):
    if k in kwargs:
      raise ValueError(f"the *_unified tasks fix the critic layout ({k})")
  cfg = g1_recover_thermal_cbf_env_cfg(play=play, critic_foot_terms=False, **kwargs)
  # the body-frame velocimeter -> the heading-frame velocity, in the same slot
  critic = cfg.observations["critic"]
  if "base_lin_vel" not in critic.terms:
    raise ValueError(f"the critic group has no base_lin_vel ({sorted(critic.terms)})")
  critic.terms = {(CRITIC_LIN_VEL_TERM if k == "base_lin_vel" else k):
                  (ObservationTermCfg(func=base_lin_vel_yaw) if k == "base_lin_vel" else v)
                  for k, v in critic.terms.items()}
  # failure = tilt > 70 deg only
  cfg.terminations.pop("illegal_contact", None)
  if "fell_over" not in cfg.terminations:
    raise ValueError(f"no fell_over termination ({sorted(cfg.terminations)})")
  lim = cfg.terminations["fell_over"].params["limit_angle"]
  if not math.isclose(lim, TILT_MAX):
    raise ValueError(f"fell_over at {math.degrees(lim):g} deg, the margin at 70")
  return cfg
