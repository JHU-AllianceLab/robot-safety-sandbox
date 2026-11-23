"""g1_cool cfg: after a handover from SONIC, stand and cool the hot motors.

The user's cooling mode (2026-11-18): a policy that takes over from the task
policy when the motors run hot, keeps the robot standing through minor pushes and
the handover itself, and brings the ankle and waist windings down (the other
motors matter less). Built on the 70 C margin world of the thermal SHAC fallback
(``g1_recover_thermal70_env_cfg``), so the robot, heat model, observation groups
(actor 127-D with the 29 winding temperatures, critic 171-D), action term, domain
randomization and fall / overheat terminations are exactly the fallback's and
the existing eval tools run it unchanged. What changes:

* SPAWN: SONIC handover rows of the sysID bank whose kick was minor (the
  derived bank ``data/g1_cool_handover_bank_dv1p5.pt``, start_ok = kick dv <=
  1.5 m/s; ``examples/build_cool_bank.py``), uniform; the row's last action seeds
  the action history, so the first step's action rate is the handover jump.
* TEMPERATURES: the ankle and waist roll / pitch windings start hot (U(45, 68)
  C, housing 0-30 C below: right after a hot burst up to long soaked
  operation); the other motors at the level + spread draw U(30, 50) C.
* EPISODE: ``episode_length_s`` = 20 s, no rest termination; a fall or a winding
  at 70 C ends it as a failure (the fallback's terminations).
* PUSHES (task registration, not here): minor scripted pushes every 3-6 s.
* REWARDS: the velocity stack is replaced by :data:`REWARDS` (mdp.py): the
  cooling term (1 / (1 + weighted heat-input rate)), staying alive / not
  failing, upright, still, not crouched, smooth (episode-long action rate + the
  handover window), joint limits (the winding-excess cost is defined but off).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import rewards as mj_rewards
from mjlab.managers.reward_manager import RewardTermCfg

from robot_safety_sandbox.envs.g1_cool import mdp
from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_thermal70_env_cfg

__all__ = ["COOL_BANK", "COOL_MIX_BANKS", "COOL_MIX_WEIGHTS", "EPISODE_LENGTH_S", "FB_BANK",
           "FB_HOT_EXTRA", "FB_SOURCE_BANK", "REWARDS", "REWARDS_FB",
           "g1_cool_thermal70_env_cfg", "g1_cool_thermal70_fb_env_cfg",
           "g1_cool_thermal70_mix_env_cfg"]

#: cooling episode length, s (the user's choice)
EPISODE_LENGTH_S = 20.0
#: the derived handover bank (examples/build_cool_bank.py)
COOL_BANK = "data/g1_cool_handover_bank_dv1p5.pt"
#: g1_cool_thermal70_mix (the user, 2026-11-18: "small push (<1 m/s), about 30 %"
#: mocap): handover rows kicked <= 1 m/s + LAFAN train mocap frames (no
#: fallAndGetUp / sprint), both from examples/build_cool_bank.py
COOL_MIX_BANKS = ("data/g1_cool_handover_bank_dv1p0.pt", "data/g1_cool_mocap_bank.pt")
COOL_MIX_WEIGHTS = {"handover": 0.7, "mocap": 0.3, "nominal": 0.0, "clean": 0.0,
                    "pushed": 0.0, "fell": 0.0, "synthetic": 0.0}
#: the hot motor groups at the spawn: (substr, Tw lo, Tw hi, housing gap lo, hi), C
HOT_GROUPS = (("ankle", 45.0, 68.0, 0.0, 30.0),
              ("waist_roll", 45.0, 68.0, 0.0, 30.0),
              ("waist_pitch", 45.0, 68.0, 0.0, 30.0))

#: ``name: (func, weight, params)``. Per control step (50 Hz). The cooling term
#: is the largest positive reward; alive keeps survival worth more than any heat.
#: winding_excess is OFF (weight 0): a cost on the temperature LEVEL, which the
#: policy can only lower over tens of seconds, made the per-step reward of a hot
#: robot negative (~-1.4 in the first smoke), i.e. paid it to end the episode;
#: minimizing the heat input (cool) already cools as fast as the physics allows.
REWARDS = {
  "cool": (mdp.cool, 2.0, dict(sigma_c_per_s=1.0)),
  "winding_excess": (mdp.winding_excess, 0.0, dict(ref_c=45.0, scale_c=10.0)),
  "alive": (mj_rewards.is_alive, 1.0, {}),
  "terminated": (mj_rewards.is_terminated, -20.0, {}),
  "upright": (mj_rewards.flat_orientation_l2, -2.0, {}),
  "lin_vel_xy": (mdp.base_lin_vel_xy_l2, -0.5, {}),
  "ang_vel": (mdp.base_ang_vel_l2, -0.05, {}),
  "pelvis_low": (mdp.pelvis_below, -20.0, dict(min_m=0.62)),
  "action_rate": (mj_rewards.action_rate_l2, -0.01, {}),
  "handover_rate": (mdp.handover_action_rate, -0.05, dict(window_s=0.2)),
  "joint_limits": (mj_rewards.joint_pos_limits, -1.0, {}),
}


#: g1_cool_thermal70_fb (the user, 2026-11-19): spawn on states of the thermal70
#: SHAC FALLBACK running after a minor-kick SONIC handover (0.5-15 s in, pushes
#: < 1 m/s; examples/harvest_cool_fallback_states.py), so the cooling policy learns
#: to take over from the fallback; the harvest's own spawns are FB_SOURCE_BANK's
FB_BANK = "data/g1_cool_fallback_bank.pt"
FB_SOURCE_BANK = COOL_MIX_BANKS[0]
#: 1-4 RANDOM other motors (arms, legs, ...) start hot too: U(45, 68) C, housing
#: 0-30 C below (ThermalCfg.hot_extra), so the temperature-ramp cost is seen on
#: every motor within a 20 s episode (a shoulder took 10 min to reach 49 C)
FB_HOT_EXTRA = (1, 4, 45.0, 68.0, 0.0, 30.0)
#: REWARDS with (the user, 2026-11-19) the cooling term on EVERY motor's heat,
#: weighted by its own temperature (mdp.cool_ramp: 0.05 below 50 C, ramping
#: quadratically to 1.05 at 65 C and 1.83 at 70 C) instead of the fixed ankle /
#: waist 1, rest 0.1; and a stronger, longer handover term (-0.1 over the first
#: 0.5 s vs -0.05 over 0.2 s), the seeded action being the FALLBACK's last one:
#: spreading a jump over the window divides its cost by the steps it takes
REWARDS_FB = dict(
  REWARDS,
  cool=(mdp.cool_ramp, 2.0, dict(sigma_c_per_s=1.0, lo_c=50.0, hi_c=65.0, base=0.05)),
  handover_rate=(mdp.handover_action_rate, -0.1, dict(window_s=0.5)),
)


def g1_cool_thermal70_env_cfg(play: bool = False,
                              episode_length_s: float = EPISODE_LENGTH_S,
                              hot_groups=HOT_GROUPS,
                              reward_weights: dict | None = None,
                              rewards: dict | None = None,
                              **kwargs) -> ManagerBasedRlEnvCfg:
  """The cooling task (module docstring). ``rewards`` replaces the stack
  (default :data:`REWARDS`), ``reward_weights`` overrides its weights by name
  (``0`` drops a term); the other kwargs go to
  :func:`g1_recover_thermal70_env_cfg` (``dataset``, ``thermal_*``, ...)."""
  stack = REWARDS if rewards is None else rewards
  if kwargs.pop("rest_terminate", False):
    raise ValueError("a cooling episode has no rest termination")
  kwargs.setdefault("dataset", COOL_BANK)
  kwargs.setdefault("weights", {"handover": 1.0, "nominal": 0.0, "clean": 0.0,
                                "pushed": 0.0, "fell": 0.0, "synthetic": 0.0,
                                "mocap": 0.0})
  kwargs.setdefault("sampler", "uniform")
  kwargs.setdefault("thermal_start_level_c", (30.0, 50.0))
  kwargs.setdefault("thermal_housing_gap_c", (0.0, 10.0))
  cfg = g1_recover_thermal70_env_cfg(
    play=play, episode_length_s=float(episode_length_s), rest_terminate=False,
    thermal_hot_groups=tuple(tuple(g) for g in hot_groups), **kwargs)
  if play:
    cfg.episode_length_s = float(episode_length_s)
  # the minor pushes keep coming after a handover spawn (the recovery tasks stop
  # them: there the row's push is the episode's one push)
  cfg.events["recover_spawn"].params["consume_handover_push"] = False
  weights = {k: w for k, (_f, w, _p) in stack.items()}
  for k, w in (reward_weights or {}).items():
    if k not in weights:
      raise ValueError(f"reward_weights: no term {k!r} (terms: {sorted(weights)})")
    weights[k] = float(w)
  cfg.rewards = {}
  for name, (func, _w, params) in stack.items():
    if weights[name] != 0.0:
      cfg.rewards[name] = RewardTermCfg(func=func, weight=weights[name], params=dict(params))
  return cfg


def g1_cool_thermal70_mix_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_cool_thermal70_env_cfg` spawning 70 % on handover rows kicked
  <= 1 m/s and 30 % on LAFAN train mocap frames (:data:`COOL_MIX_BANKS`, each
  pool uniform); the 0.2-1 m/s pushes are the task registration's."""
  kwargs.setdefault("dataset", list(COOL_MIX_BANKS))
  kwargs.setdefault("weights", dict(COOL_MIX_WEIGHTS))
  return g1_cool_thermal70_env_cfg(play=play, **kwargs)


def g1_cool_thermal70_fb_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_cool_thermal70_env_cfg` taking over from the FALLBACK
  (:data:`FB_BANK`, uniform), with :data:`REWARDS_FB` (every motor's heat
  weighted by its own temperature, the stronger handover term) and
  :data:`FB_HOT_EXTRA` random hot motors on top of the hot ankles / waist; the
  0.2-1 m/s pushes are the task registration's. ``dataset=FB_SOURCE_BANK``
  (with ``weights={"handover": 1.0}``) spawns at the SONIC handovers instead:
  the harvest of FB_BANK."""
  kwargs.setdefault("dataset", FB_BANK)
  kwargs.setdefault("rewards", REWARDS_FB)
  kwargs.setdefault("thermal_hot_extra", FB_HOT_EXTRA)
  return g1_cool_thermal70_env_cfg(play=play, **kwargs)
