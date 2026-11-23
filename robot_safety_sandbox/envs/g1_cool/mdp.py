"""Reward terms of the g1_cool tasks: stand after a handover and cool the motors.

The heat terms read the env's :class:`~robot_safety_sandbox.envs.g1_recover.
thermal.MotorThermalState` (the sysID heat model the env already integrates every
physics substep) and use its own parameters, so the reward and the temperatures
the episode ends on agree by construction.

THE COOLING SIGNAL. A winding's temperature moves as

    Cw dTw/dt = Q(tau, w, Tw) - (Tw - Th) / Rwh,
    Q = a tau^2 (1 + alpha (Tw - 25)) stall(|w|) + b2 w^2 + b1 |w|

The conduction term does not depend on the action, so the fastest a policy can
cool a winding is to make its own heat input Q as small as possible: the reward
scores ``Q / Cw`` (C/s of heating the motor's own losses cause), weighted per
motor. Rewarding -dTw/dt instead adds an action-free term whose sum over an
episode telescopes to start minus end temperature -- the same optimum with a
weak, noisy per-step signal.
"""

from __future__ import annotations

import torch

from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state

#: motor weights of the heat terms: the user's priority (2026-11-18) -- the ankle
#: and waist linkage motors (the small 5020s that trip 70 C) count fully, the rest
#: a little ("cooling other joints would be nice but not super important")
HOT_WEIGHTS = (("ankle", 1.0), ("waist_roll", 1.0), ("waist_pitch", 1.0))
OTHER_WEIGHT = 0.1


def _motor_weights(env, groups, default: float) -> torch.Tensor:
  """``(29,)`` per-motor weights (first matching substring wins), cached on the env."""
  st = thermal_state(env)
  key = ("_zoo_cool_w", tuple(groups), float(default))
  cache = env.__dict__.setdefault("_zoo_cool_cache", {})
  w = cache.get(key)
  if w is None:
    vals = []
    for nm in st.names:
      vals.append(next((float(v) for s, v in groups if s in nm), float(default)))
    w = torch.tensor(vals, device=st.Tw.device)
    cache[key] = w
  return w


def motor_heat_w(st) -> torch.Tensor:
  """Each winding's heat input from its own losses NOW, W, ``(num_envs, 29)``:
  the heat model's ``Q`` at the last physics substep's torque and speed."""
  tau, w = st.motor_inputs()
  th = st.th
  wa = w.abs()
  return (th.a * tau * tau * (1.0 + th.alpha * (st.Tw - 25.0)) * th.stall(wa)
          + th.b2 * w * w + th.b1 * wa)


def heat_rate_c_per_s(env, groups=HOT_WEIGHTS, default: float = OTHER_WEIGHT) -> torch.Tensor:
  """Weighted winding heating rate from the motors' own losses,
  ``sum_i w_i Q_i / Cw_i``, C/s, ``(num_envs,)``."""
  st = thermal_state(env)
  return (motor_heat_w(st) / st.th.Cw * _motor_weights(env, groups, default)).sum(1)


def cool(env, sigma_c_per_s: float = 1.0, groups=HOT_WEIGHTS,
         default: float = OTHER_WEIGHT) -> torch.Tensor:
  """``1 / (1 + H / sigma)``, H = :func:`heat_rate_c_per_s`: 1 with no heat
  input, -> 0 as the weighted motors heat. Bounded and positive, so a hot robot
  is never paid to end its episode. A rational, not exponential, kernel: right
  after a handover H is ~10-15 C/s (the walker's torques), where exp(-H) is
  flat (and underflows) and would give the policy no signal."""
  return 1.0 / (1.0 + heat_rate_c_per_s(env, groups, default) / float(sigma_c_per_s))


def ramp_weights(Tw: torch.Tensor, lo_c: float = 50.0, hi_c: float = 65.0,
                 base: float = 0.05) -> torch.Tensor:
  """Temperature-dependent motor weights, ``(num_envs, 29)``: ``base`` below
  ``lo_c``, then ``base + ((Tw - lo_c) / (hi_c - lo_c))^2`` -- 1 + base at
  ``hi_c``, still rising to the 70 C termination (1.78 + base). The same for
  every motor: heat into a cool motor is (almost) free, heat into a hot one
  costs more the hotter it is (the user, 2026-11-19: "for all the motor, if its
  within some heat range (perhaps 50-65) than it receives penalty, and if it is
  below 50 then we think that's fine")."""
  x = ((Tw - float(lo_c)) / (float(hi_c) - float(lo_c))).clamp_min(0.0)
  return float(base) + x * x


def heat_rate_ramp_c_per_s(env, lo_c: float = 50.0, hi_c: float = 65.0,
                           base: float = 0.05) -> torch.Tensor:
  """``sum_i ramp(Tw_i) Q_i / Cw_i``, C/s, ``(num_envs,)``: the heat rate of
  :func:`heat_rate_c_per_s` with :func:`ramp_weights` (the winding's own
  temperature) in place of the fixed per-motor weights. The weight is a
  function of the state only, so the action still enters through Q alone: the
  action-controllable part of a cost on the time spent above ``lo_c``, scored
  every step (a cost on the level itself paid a hot robot to fall, see
  env_cfg.REWARDS)."""
  st = thermal_state(env)
  w = ramp_weights(st.Tw, lo_c, hi_c, base)
  return (motor_heat_w(st) / st.th.Cw * w).sum(1)


def cool_ramp(env, sigma_c_per_s: float = 1.0, lo_c: float = 50.0, hi_c: float = 65.0,
              base: float = 0.05) -> torch.Tensor:
  """:func:`cool` on :func:`heat_rate_ramp_c_per_s`: ``1 / (1 + H / sigma)``,
  in (0, 1], so the reward of a hot robot stays positive."""
  return 1.0 / (1.0 + heat_rate_ramp_c_per_s(env, lo_c, hi_c, base) / float(sigma_c_per_s))


def winding_excess(env, ref_c: float = 45.0, scale_c: float = 10.0, groups=HOT_WEIGHTS,
                   default: float = OTHER_WEIGHT) -> torch.Tensor:
  """``sum_i w_i relu(Tw_i - ref) / scale`` (a cost): pressure to keep cooling
  while a weighted winding is above ``ref_c``."""
  st = thermal_state(env)
  ex = (st.Tw - float(ref_c)).clamp_min(0.0) / float(scale_c)
  return (ex * _motor_weights(env, groups, default)).sum(1)


def base_lin_vel_xy_l2(env) -> torch.Tensor:
  """``||v_xy||^2`` of the pelvis in its own frame (a cost: stand still)."""
  v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  return (v * v).sum(1)


def base_ang_vel_l2(env) -> torch.Tensor:
  """``||omega||^2`` of the pelvis (a cost)."""
  w = env.scene["robot"].data.root_link_ang_vel_b
  return (w * w).sum(1)


def pelvis_below(env, min_m: float = 0.62) -> torch.Tensor:
  """``relu(min_m - pelvis height)^2`` (a cost): the robot may settle lower than
  the 0.76 m stance to unload its ankles, but not crouch; kneeling or sitting
  already ends the episode (non-foot ground contact)."""
  z = env.scene["robot"].data.root_link_pos_w[:, 2]
  return (float(min_m) - z).clamp_min(0.0).square()


def handover_action_rate(env, window_s: float = 0.2) -> torch.Tensor:
  """``||a_t - a_{t-1}||^2`` over the first ``window_s`` of the episode (a cost).
  At the first step ``a_{t-1}`` is the walker's last applied action (the spawn
  seeds it from the handover row), so this is the jump AT the handover and its
  aftermath, on top of the episode-long ``action_rate_l2``."""
  am = env.action_manager
  rate = (am.action - am.prev_action).square().sum(1)
  steps = max(1, int(round(float(window_s) / float(env.step_dt))))
  return torch.where(env.episode_length_buf <= steps, rate, torch.zeros_like(rate))
