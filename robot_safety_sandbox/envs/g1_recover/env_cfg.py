"""g1_recover cfg: g1_stabilize with the stock spawn replaced by the bank spawn.

Everything that defines the robot and the world is g1_stabilize's: flat
ground, the twist command pinned to zero (so the ``phase`` observation reads
zero and the command the fallback sees is the one the filter holds while it is
engaged), the same domain randomization and the same terminations. What
changes:

* ``reset_base`` / ``reset_robot_joints`` are replaced by one
  :class:`~robot_safety_sandbox.envs.g1_recover.spawn.RecoverSpawn` event;
* a :class:`~robot_safety_sandbox.envs.g1_recover.spawn.RecoverCurriculum`
  term moves the per-source difficulty levels and logs survival;
* episodes are ``episode_length_s`` long (default 8 s, not 20). Recovery takes
  1-2 s, after which a 20 s episode is mostly quiet standing, and with no
  timeout bootstrapping only the last ~1 s before a timeout carries an
  optimistic target; shorter episodes also mean more starts per sample.
* optionally (``rest_terminate=True``) the episode ends as a TRUNCATION once
  the robot has been back in the stance set for ``rest_dwell_s`` (after
  ``rest_arm_s``): :class:`~robot_safety_sandbox.envs.g1_recover.rest.RestReached`.
  Every episode is then one event, critical state -> fall or rest.

THE REST TARGET. By default the rest set is g1_stabilize's ``stance_margins``.
Setting any of ``rest_lin_tol`` / ``rest_ang_tol`` / ``rest_gate_on_push``
swaps in ``compose(g_fall, targets.make_l_stance(lin_tol, ang_tol,
gate_on_push))`` (None tolerances: g1_stabilize's 0.25 m/s / 0.30 rad/s), so
a reach-avoid task's episode ends after dwelling in ITS OWN target set -- the
truncation then agrees with the l the learner reaches for, and the gate keeps
a start that already sits in the set from ending before its push has landed.
The episode end stays a truncation, never the "reach-avoid" end criterion's
success termination, which the curricula and safety counters would read as a
fall.

THE SAMPLER. ``sampler`` / ``uniform_mix`` / ``weight_column`` /
``online_alpha`` / ``sequential_repeats`` choose how mocap / handover rows are
drawn (or, ``"sequential"``, walk the whole bank for labeling); ``dataset``
may be a list of banks. See ``spawn.py``; the defaults are today's spawn.
"""

from __future__ import annotations

import os

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from robot_safety_sandbox.envs.g1_recover.rest import RestReached
from robot_safety_sandbox.envs.g1_recover.spawn import (
  SAMPLERS, RecoverCurriculum, RecoverSpawn)
from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
  ANG_TOL, CLAMP, LIN_TOL, g1_stabilize_env_cfg, g_fall)
from robot_safety_sandbox.margins import compose

#: Recovery episode length, s.
EPISODE_LENGTH_S = 8.0


def pelvis_height(env) -> torch.Tensor:
  """Pelvis (root link) height above the flat ground, m, ``(num_envs, 1)``:
  the quantity the height term of the fall margin g reads. No actor observation
  carries it (forward kinematics recovers it only while a foot is planted)."""
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def rest_target_margins(lin_tol: float | None = None,
                        ang_tol: float | None = None,
                        gate_on_push: bool = False):
  """``(g, l)`` of a rest target: g_fall and ``targets.make_l_stance`` at these
  tolerances (None -> g1_stabilize's), clamped like ``stance_margins``."""
  from robot_safety_sandbox.envs.g1_recover.targets import make_l_stance
  return compose(g_fall, make_l_stance(
    lin_tol=LIN_TOL if lin_tol is None else float(lin_tol),
    ang_tol=ANG_TOL if ang_tol is None else float(ang_tol),
    gate_on_push=bool(gate_on_push)), clamp=CLAMP)


#: the privileged foot terms of the critic group (all read the contact sensor or
#: the world-frame foot sites); ``critic_foot_terms=False`` drops them
CRITIC_FOOT_TERMS = ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces")


def g1_recover_env_cfg(play: bool = False,
                       dataset: str | list[str] | None = None,
                       weights: dict | None = None, levels: int = 5,
                       band_width: float = 0.4, init_level: int = 0,
                       episode_length_s: float = EPISODE_LENGTH_S,
                       ema: float = 0.002, rest_terminate: bool = False,
                       rest_dwell_s: float = 0.4,
                       rest_arm_s: float = 0.8, sampler: str = "ladder",
                       uniform_mix: float = 0.15,
                       weight_column: str = "p_surv",
                       online_alpha: float = 0.1,
                       sequential_repeats: int = 1,
                       rest_lin_tol: float | None = None,
                       rest_ang_tol: float | None = None,
                       rest_gate_on_push: bool = False,
                       critic_pelvis_height: bool = True,
                       critic_foot_terms: bool = True,
                       critic_base_lin_vel: bool = True) -> ManagerBasedRlEnvCfg:
  """All keyword arguments can be set from a run config's ``env_overrides``.
  The rest-target kwargs act only with ``rest_terminate=True``.
  ``critic_pelvis_height=False`` leaves the pelvis height out of the critic
  group (the world-model state of a SHAC run); the ``*_noheight`` tasks set it
  together with their height-free margin. ``critic_foot_terms=False`` also
  drops the foot terms (``CRITIC_FOOT_TERMS``: foot height, air time, contact
  flags and contact forces), leaving the actor terms + base_lin_vel -- the
  ``*_noheight_nofoot`` tasks, whose critic group is a deployable world-model
  state. ``critic_base_lin_vel=False`` drops base_lin_vel too, leaving exactly
  the actor terms, noise-free (the ``*_noheight_nofoot_nolinvel`` tasks)."""
  if sampler not in SAMPLERS:
    raise ValueError(f"g1_recover sampler {sampler!r}: one of {SAMPLERS}")
  cfg = g1_stabilize_env_cfg(play=play)
  if not play:
    cfg.episode_length_s = float(episode_length_s)
  if "critic" in cfg.observations and critic_pelvis_height:
    # Privileged critic group only (the actor is untouched): with base_lin_vel
    # (already in the group) and projected_gravity, the critic then sees every
    # term of g it can bind on but the transient non-foot contact. Inert unless
    # a run asks for the critic group (train_on_policy --privileged-critic).
    cfg.observations["critic"].terms["pelvis_height"] = ObservationTermCfg(
      func=pelvis_height)
  if "critic" in cfg.observations and not critic_foot_terms:
    terms = cfg.observations["critic"].terms
    missing = [k for k in CRITIC_FOOT_TERMS if k not in terms]
    if missing:
      raise ValueError(f"the critic group has no {missing} to drop ({sorted(terms)})")
    for k in CRITIC_FOOT_TERMS:
      del terms[k]
  if "critic" in cfg.observations and not critic_base_lin_vel:
    terms = cfg.observations["critic"].terms
    if "base_lin_vel" not in terms:
      raise ValueError(f"the critic group has no base_lin_vel to drop ({sorted(terms)})")
    del terms["base_lin_vel"]
  if rest_terminate:
    # A truncation (time_out=True), see rest.py: survival for the curricula,
    # terminal at its own margin for the avoid backup.
    params = dict(dwell_s=float(rest_dwell_s), arm_s=float(rest_arm_s))
    if (rest_lin_tol is not None or rest_ang_tol is not None
        or rest_gate_on_push):
      params["margin_fn"] = rest_target_margins(rest_lin_tol, rest_ang_tol,
                                                rest_gate_on_push)
    cfg.terminations["recovered"] = TerminationTermCfg(
      func=RestReached, time_out=True, params=params)
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  common = {"levels": int(levels), "init_level": int(init_level),
            "sampler": str(sampler)}
  if dataset is not None and not isinstance(dataset, (str, os.PathLike)):
    dataset = [os.fspath(x) for x in dataset]   # a YAML list / a tuple
  cfg.events["recover_spawn"] = EventTermCfg(
    func=RecoverSpawn, mode="reset",
    params=dict(common, dataset=dataset, weights=weights,
                band_width=float(band_width), uniform_mix=float(uniform_mix),
                weight_column=str(weight_column),
                sequential_repeats=int(sequential_repeats)))
  cfg.curriculum["recover"] = CurriculumTermCfg(
    func=RecoverCurriculum, params=dict(common, ema=float(ema),
                                        online_alpha=float(online_alpha)))
  return cfg


def g1_recover_noheight_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_env_cfg` with no pelvis height in the critic group (the
  ``*_noheight`` tasks, whose margin is ``g_fall_noheight``); same kwargs."""
  if kwargs.pop("critic_pelvis_height", False):
    raise ValueError("the *_noheight tasks have no pelvis height in the critic group")
  return g1_recover_env_cfg(play=play, critic_pelvis_height=False, **kwargs)


def g1_recover_noheight_nofoot_env_cfg(play: bool = False,
                                       **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_noheight_env_cfg` with no foot terms in the critic group
  either (the ``*_noheight_nofoot`` tasks): the critic group, i.e. a SHAC
  world-model state, is the actor terms + base_lin_vel (101-D); same kwargs."""
  for k in ("critic_pelvis_height", "critic_foot_terms"):
    if kwargs.pop(k, False):
      raise ValueError(f"the *_noheight_nofoot tasks fix {k}=False")
  return g1_recover_env_cfg(play=play, critic_pelvis_height=False,
                            critic_foot_terms=False, **kwargs)


def g1_recover_noheight_nofoot_nolinvel_env_cfg(play: bool = False,
                                                **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_noheight_nofoot_env_cfg` with no base_lin_vel in the
  critic group either (the ``*_noheight_nofoot_nolinvel`` tasks): the critic
  group, i.e. a SHAC world-model state, is exactly the actor terms, noise-free
  (98-D); same kwargs."""
  for k in ("critic_pelvis_height", "critic_foot_terms", "critic_base_lin_vel"):
    if kwargs.pop(k, False):
      raise ValueError(f"the *_noheight_nofoot_nolinvel tasks fix {k}=False")
  return g1_recover_env_cfg(play=play, critic_pelvis_height=False,
                            critic_foot_terms=False, critic_base_lin_vel=False, **kwargs)


def use_sysid_robot(cfg: ManagerBasedRlEnvCfg, v_bus: float | None = None) -> None:
  """Swap the robot for the lab's identified G1 (``envs/assets/g1_sysid.py``:
  measured waist ranges, refitted collision capsules, MuJoCo ``<dcmotor>``
  actuators at battery voltage ``v_bus``). Names, gains, peak torques and
  armatures are mjlab's G1's, so sensors, events, the action scale and the
  observation layout carry over unchanged."""
  from robot_safety_sandbox.envs.assets.g1_sysid import (DEFAULT_V_BUS,
                                                         get_g1_sysid_robot_cfg)
  cfg.scene.entities["robot"] = get_g1_sysid_robot_cfg(
    DEFAULT_V_BUS if v_bus is None else float(v_bus))


def g1_recover_sysid_env_cfg(play: bool = False, v_bus: float | None = None,
                             **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_env_cfg` on the lab's identified G1 (:func:`use_sysid_robot`),
  no motor temperatures (the ``*_sysid`` tasks: harvesting start states on
  the robot the ``*_thermal`` tasks train on); same kwargs, plus ``v_bus``."""
  cfg = g1_recover_env_cfg(play=play, **kwargs)
  use_sysid_robot(cfg, v_bus)
  return cfg


def g1_recover_thermal_env_cfg(play: bool = False, v_bus: float | None = None,
                               thermal_start_level_c=(30.0, 80.0),
                               thermal_start_floor_c: float = 30.0,
                               thermal_housing_gap_c=(0.0, 20.0),
                               thermal_sim_heat_scale: bool = True,
                               thermal_overheat_termination: bool = True,
                               thermal_barrier_c: float | None = None,
                               thermal_barrier_gamma: float | None = None,
                               **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_noheight_env_cfg` on the lab's identified G1 with its
  motor heat model in the loop (the ``*_thermal`` tasks, ``thermal.py``):

  * actor obs + ``motor_temp`` (29 winding temperatures, +-0.5 C noise): 127-D;
  * critic group (the world-model state) + ``motor_temp`` and
    ``motor_temp_housing`` (noise-free): 113 + 58 = 171-D;
  * termination ``motor_overheat`` (a winding at 85 C: the robot goes limp),
    unless ``thermal_overheat_termination=False``;
  * metrics term ``motor_temp_max``: integrates the heat model every physics
    substep, draws the start temperatures at reset, logs the episode's peak;
  * with ``thermal_barrier_c`` (the ``*_thermal_cbf`` tasks): the temperature
    barrier on the joint targets (``thermal.ThermalBarrierAction``, limit
    ``thermal_barrier_c``, rate ``thermal_barrier_gamma``) and its metrics
    ``cbf_clip_frac``, ``cbf_negative_bound_frac``, ``cbf_overshoot_c``.

  The margin is the task registration's. The other ``thermal_*`` kwargs set
  :class:`~robot_safety_sandbox.envs.g1_recover.thermal.ThermalCfg`."""
  import dataclasses

  from mjlab.managers.metrics_manager import MetricsTermCfg
  from mjlab.utils.noise import UniformNoiseCfg

  from robot_safety_sandbox.envs.g1_recover.thermal import (
    CBF_GAMMA, LIMIT_C, ThermalBarrierActionCfg, ThermalCfg, ThermalIntegrator,
    cbf_clip_frac, cbf_negative_bound_frac, cbf_overshoot_c, motor_housing_temp,
    motor_overheat, motor_winding_temp)
  if kwargs.pop("critic_pelvis_height", False):
    raise ValueError("the *_thermal tasks fix critic_pelvis_height=False")
  cfg = g1_recover_env_cfg(play=play, critic_pelvis_height=False, **kwargs)
  use_sysid_robot(cfg, v_bus)
  tc = ThermalCfg(start_level_c=tuple(float(x) for x in thermal_start_level_c),
                  start_floor_c=float(thermal_start_floor_c),
                  housing_gap_c=tuple(float(x) for x in thermal_housing_gap_c),
                  sim_heat_scale=bool(thermal_sim_heat_scale))
  p = {"thermal_cfg": tc}
  cfg.observations["actor"].terms["motor_temp"] = ObservationTermCfg(
    func=motor_winding_temp, params=dict(p), noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5))
  cfg.observations["critic"].terms["motor_temp"] = ObservationTermCfg(
    func=motor_winding_temp, params=dict(p))
  cfg.observations["critic"].terms["motor_temp_housing"] = ObservationTermCfg(
    func=motor_housing_temp, params=dict(p))
  if thermal_overheat_termination:
    cfg.terminations["motor_overheat"] = TerminationTermCfg(
      func=motor_overheat, params=dict(p, limit_c=LIMIT_C))
  cfg.metrics["motor_temp_max"] = MetricsTermCfg(
    func=ThermalIntegrator, params=dict(p), per_substep=True, reduce="max")
  if thermal_barrier_c is not None:
    old = cfg.actions["joint_pos"]
    keep = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    gamma = CBF_GAMMA if thermal_barrier_gamma is None else float(thermal_barrier_gamma)
    cfg.actions["joint_pos"] = ThermalBarrierActionCfg(
      **keep, limit_c=float(thermal_barrier_c), gamma=gamma, thermal_cfg=tc)
    lim = {"limit_c": float(thermal_barrier_c)}
    cfg.metrics["cbf_clip_frac"] = MetricsTermCfg(func=cbf_clip_frac)
    cfg.metrics["cbf_negative_bound_frac"] = MetricsTermCfg(func=cbf_negative_bound_frac)
    cfg.metrics["cbf_overshoot_c"] = MetricsTermCfg(func=cbf_overshoot_c, params=lim,
                                                    reduce="max")
  elif thermal_barrier_gamma is not None:
    raise ValueError("thermal_barrier_gamma without thermal_barrier_c")
  return cfg


def g1_recover_thermal_cbf_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_thermal_env_cfg` with the temperature barrier as a layer
  below the policy (the ``*_thermal_cbf`` tasks; the user's design,
  2026-11-10): every motor's torque clipped so its winding stays below
  ``CBF_LIMIT_C`` = 70 C (rate ``CBF_GAMMA`` = 1/s); start levels U(30, 70), so
  every episode starts inside the barrier's safe set; no overheat termination
  (the margin, ``g_fall_noheight``, is the registration's: temperature is
  not in it). Observations as the ``*_thermal`` tasks (actor 127-D, critic
  171-D). Same kwargs."""
  from robot_safety_sandbox.envs.g1_recover.thermal import CBF_GAMMA, CBF_LIMIT_C
  kwargs.setdefault("thermal_start_level_c", (30.0, CBF_LIMIT_C))
  kwargs.setdefault("thermal_barrier_c", CBF_LIMIT_C)
  kwargs.setdefault("thermal_barrier_gamma", CBF_GAMMA)
  if kwargs.pop("thermal_overheat_termination", False):
    raise ValueError("the *_thermal_cbf tasks have no overheat termination")
  return g1_recover_thermal_env_cfg(play=play, thermal_overheat_termination=False, **kwargs)


def g1_recover_k_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
  """:func:`g1_recover_env_cfg` whose episodes end when the state leaves the
  ViaLasso safe set K (``k_set.use_k_terminations``); same kwargs."""
  from robot_safety_sandbox.envs.g1_recover.k_set import use_k_terminations
  cfg = g1_recover_env_cfg(play=play, **kwargs)
  use_k_terminations(cfg)
  return cfg
