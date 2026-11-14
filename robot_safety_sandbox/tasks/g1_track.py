"""Unitree G1 squat tracking (mode=CUMULATIVE pi_task) inside a WM-tube filter.

The task policy tracks a held-out LAFAN segment (fightAndSports1_subject1,
166-179 s: stand, four deep squat bounces, walk-off; ``envs/g1_track``) on dense
tracking rewards, trained bare or inside a world-model tube filter whose
fallback is a frozen thermal SHAC policy. Each task is the world of that
fallback with the task added, and carries the fallback's OWN margin as
``dense_margins`` (the reward stays the tracking stack; the filter and the
failure counters read the margin):

``g1_track_squat_thermal70``   the margin world: margin
                               ``g_fall_noheight_thermal70`` (tilt, non-foot
                               contact, (70 C - hottest winding) / 10 C), a
                               winding at 70 C ends the episode.
``g1_track_squat_thermal_cbf`` the barrier world: the 70 C torque-clipping
                               barrier under every controller, margin
                               ``g_fall_noheight``, no overheat termination.

The margins are exactly those of g1_recover_thermal70 / g1_recover_thermal_cbf.
The policy reads ``task_policy`` and its value net ``task_critic``; the
fallback's ``actor`` / ``critic`` groups are untouched. The task policy acts in
+-6.5 (the user's box: the frozen action term cannot reach the squat's knees in
+-3); the fallback keeps +-3.
"""

from __future__ import annotations

from ..margins import compose
from ..registry import CUMULATIVE, TaskSpec, register
from .g1_stabilize import G1_BRIDGE_KWARGS, G1_CTRL_DIM


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_recover.thermal import g_fall_noheight_thermal70
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall_noheight
  from robot_safety_sandbox.envs.g1_track.env_cfg import (
    TASK_ACTION_BOUND, g1_track_squat_thermal70_env_cfg,
    g1_track_squat_thermal_cbf_env_cfg)
  kwargs = dict(G1_BRIDGE_KWARGS, action_bound=TASK_ACTION_BOUND, dense_margins=True,
                obs_key="task_policy", critic_obs_key="task_critic")
  for world, builder, g, what in (
      ("thermal70", g1_track_squat_thermal70_env_cfg, g_fall_noheight_thermal70,
       "no fall and every winding below 70 C (70 C ends the episode)"),
      ("thermal_cbf", g1_track_squat_thermal_cbf_env_cfg, g_fall_noheight,
       "no fall, a 70 C torque-clipping barrier under every controller")):
    register(TaskSpec(
      task_id=f"g1_track_squat_{world}", cfg_builder=builder,
      margin_fn=compose(g), mode=CUMULATIVE, ctrl_dim=G1_CTRL_DIM,
      kwargs=dict(kwargs),
      description=f"Track the fightAndSports1 squat segment on the identified G1 "
                  f"(dense tracking reward, task_policy / task_critic groups); "
                  f"margin of the fallback's world: {what}."))
