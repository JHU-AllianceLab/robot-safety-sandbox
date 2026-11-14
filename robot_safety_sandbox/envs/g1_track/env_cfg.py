"""g1_track cfg: a LAFAN squat-tracking task policy in its fallback's own world.

The task policy trains inside a world-model tube filter whose fallback is a
frozen g1_recover_thermal70 / g1_recover_thermal_cbf SHAC policy, so the env is
THAT task's env with a task added on top, never a modified copy:

* UNCHANGED (the fallback and its world model read them; ``tests/
  test_g1_track_cfg.py`` checks term by term): the ``actor`` (127-D) and
  ``critic`` (171-D) groups, the pinned ``twist`` command, the startup domain
  randomization (the 33-D context), the ``joint_pos`` action term (the barrier in
  the cbf world), the robot, the thermal model and its start draw, the sim, and
  the fall / overheat terminations.
* REMOVED: the bank spawn and its curriculum (the reference command spawns),
  and the velocity task's whole reward stack -- among it ``action_rate_l2`` on
  the EXECUTED action, which would charge every hand-over to the policy, and
  the -200 termination penalty.
* ADDED: the ``ref`` command (command.py: reference, mixed init, spawn), the
  tracking rewards (mjlab's BeyondMimic terms at their sigmas and weights, plus
  a joint-position term over all but the wrists' pitch / yaw, and the
  self-collision cost), ``reference_finished`` (a time-out at the segment's
  end), three observation groups -- ``task_policy`` (334-D, deployable, the
  sensor noise of the actor group), ``task_critic`` (540-D, privileged) and
  ``amp`` (68-D) -- and the ``track_stats`` step event (the ``env._zoo_track``
  bus, mdp.py).

The episode is ``episode_length_s`` = 14 s, longer than the 13.02 s segment, so
only ``reference_finished`` (or a failure) ends one; mjlab's ``time_out`` stays as
a backstop. There is no tracking-error termination: the failure counters count
every non-time-out termination, and the dwell filter deliberately leaves the
reference for a second at a time.
"""

from __future__ import annotations

import copy
import dataclasses
import functools

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp as track_mdp

from robot_safety_sandbox.envs.g1_recover.env_cfg import (
  g1_recover_thermal70_env_cfg, g1_recover_thermal_cbf_env_cfg, pelvis_height)
from robot_safety_sandbox.envs.g1_recover.thermal import LIMIT70_C
from robot_safety_sandbox.envs.g1_track import mdp
from robot_safety_sandbox.envs.g1_track.command import TASK_ACTION_BOUND, TrackRefCommandCfg
from robot_safety_sandbox.envs.g1_track.reference import GROUND_SNAP_M, SegmentSpec
from robot_safety_sandbox.envs.velocity import mdp as vel_mdp

__all__ = ["TASK_ACTION_BOUND", "g1_track_env_cfg", "g1_track_squat_thermal70_env_cfg",
           "g1_track_squat_thermal_cbf_env_cfg"]

#: episode length, s: longer than the default 13.02 s segment
EPISODE_LENGTH_S = 14.0
#: future reference frames the task policy sees, control steps ahead
FUTURE_FRAMES = (0, 2, 5, 10, 25)
#: tracking reward terms, ``name: (func, std, weight)``; the first five are
#: mjlab's BeyondMimic terms at its values (no global anchor position term)
TRACK_REWARDS = {
  "track_anchor_ori": (track_mdp.motion_global_anchor_orientation_error_exp, 0.4, 0.5),
  "track_body_pos": (track_mdp.motion_relative_body_position_error_exp, 0.3, 1.0),
  "track_body_ori": (track_mdp.motion_relative_body_orientation_error_exp, 0.4, 1.0),
  "track_body_lin_vel": (track_mdp.motion_global_body_linear_velocity_error_exp, 1.0, 1.0),
  "track_body_ang_vel": (track_mdp.motion_global_body_angular_velocity_error_exp, 3.14, 1.0),
  "track_joint_pos": (mdp.ref_joint_pos_exp, 0.5, 0.5),
}
SELF_COLLISION_WEIGHT = -1.0


def g1_track_env_cfg(base_builder, play: bool = False, *,
                     clip: str = "fightAndSports1_subject1", t0: float = 166.0,
                     t1: float = 179.0, p_start: float = 0.5,
                     rsi_min_left_s: float = 1.0, spawn_noise: bool = True,
                     future: tuple = FUTURE_FRAMES,
                     episode_length_s: float = EPISODE_LENGTH_S,
                     ground_snap_m: float = GROUND_SNAP_M,
                     **base_kw) -> ManagerBasedRlEnvCfg:
  """The squat-tracking task on ``base_builder``'s world (module docstring).
  ``base_kw`` goes to the base builder and may only be its ``thermal_*`` /
  ``v_bus`` options: the bank-spawn options would set nothing, and
  ``rest_terminate`` would add a termination the task must not have."""
  bad = sorted(k for k in base_kw if not (k.startswith("thermal_") or k == "v_bus"))
  if bad:
    raise TypeError(f"g1_track passes only thermal_* / v_bus to its world, not {bad}")
  cfg = base_builder(play=play, **base_kw)
  cfg.events.pop("recover_spawn")
  cfg.curriculum.pop("recover")
  if "recovered" in cfg.terminations:
    raise ValueError("the g1_track world must not end episodes at rest")
  seg = SegmentSpec(clip=str(clip), t0=float(t0), t1=float(t1),
                    ground_snap_m=float(ground_snap_m))
  if not play:
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    if (seg.T - 1) * dt >= float(episode_length_s):
      raise ValueError(f"episode_length_s={episode_length_s} does not cover the "
                       f"{(seg.T - 1) * dt:.2f} s segment")
    cfg.episode_length_s = float(episode_length_s)
  ref = {"command_name": "ref"}

  cfg.commands["ref"] = TrackRefCommandCfg(
    resampling_time_range=(1.0e9, 1.0e9), segment=seg, p_start=float(p_start),
    rsi_min_left_s=float(rsi_min_left_s), spawn_noise=bool(spawn_noise) and not play)

  actor, critic = cfg.observations["actor"], cfg.observations["critic"].terms
  a = actor.terms
  pol = {k: copy.deepcopy(a[k]) for k in ("base_ang_vel", "projected_gravity", "joint_pos",
                                          "joint_vel", "actions", "motor_temp")}
  pol["joint_pos"].params = {"biased": True}      # what a deployed robot's encoders read
  pol["ref_phase"] = ObservationTermCfg(func=mdp.ref_phase, params=dict(ref))
  pol["ref_future"] = ObservationTermCfg(func=mdp.ref_future,
                                         params=dict(ref, offsets=tuple(future)))
  crit = {k: dataclasses.replace(copy.deepcopy(v), noise=None) for k, v in pol.items()}
  crit.update(
    base_lin_vel=dataclasses.replace(copy.deepcopy(critic["base_lin_vel"]), noise=None),
    root_height=ObservationTermCfg(func=pelvis_height),
    anchor_pos_err=ObservationTermCfg(func=mdp.anchor_pos_err_h, params=dict(ref)),
    body_pos=ObservationTermCfg(func=track_mdp.robot_body_pos_b, params=dict(ref)),
    body_ori=ObservationTermCfg(func=track_mdp.robot_body_ori_b, params=dict(ref)),
    body_pos_err=ObservationTermCfg(func=mdp.rel_body_pos_err_h, params=dict(ref)),
    motor_temp_housing=copy.deepcopy(critic["motor_temp_housing"]),
    foot_contact=copy.deepcopy(critic["foot_contact"]))
  cfg.observations["task_policy"] = ObservationGroupCfg(
    terms=pol, concatenate_terms=True, enable_corruption=actor.enable_corruption,
    history_length=actor.history_length)
  cfg.observations["task_critic"] = ObservationGroupCfg(
    terms=crit, concatenate_terms=True, enable_corruption=False,
    history_length=actor.history_length)
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={"amp": ObservationTermCfg(func=mdp.AmpObs)}, concatenate_terms=True,
    enable_corruption=False, history_length=actor.history_length)

  cfg.rewards = {}
  for name, (func, std, weight) in TRACK_REWARDS.items():
    cfg.rewards[name] = RewardTermCfg(func=func, weight=weight, params=dict(ref, std=std))
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=vel_mdp.self_collision_cost, weight=SELF_COLLISION_WEIGHT,
    params={"sensor_name": "self_collision", "force_threshold": 10.0})

  cfg.terminations["reference_finished"] = TerminationTermCfg(
    func=mdp.reference_finished, time_out=True, params=dict(ref))
  cfg.events["track_stats"] = EventTermCfg(
    func=mdp.TrackEpisodeStats, mode="step",
    params=dict(ref, overheat_c=LIMIT70_C))
  return cfg


#: the margin world (overheat at 70 C ends the episode) and the barrier world
g1_track_squat_thermal70_env_cfg = functools.partial(g1_track_env_cfg,
                                                     g1_recover_thermal70_env_cfg)
g1_track_squat_thermal_cbf_env_cfg = functools.partial(g1_track_env_cfg,
                                                       g1_recover_thermal_cbf_env_cfg)
