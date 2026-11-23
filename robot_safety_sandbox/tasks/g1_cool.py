"""Unitree G1 COOLING policy (mode=CUMULATIVE): stand after a handover from SONIC,
ride out minor pushes, and cool the hot ankle and waist motors.

``g1_cool_thermal70`` the 70 C margin world of the thermal SHAC fallback
                      (``envs/g1_cool``): spawn at minor-kick SONIC handover
                      rows with hot ankle / waist windings, 20 s episodes, the
                      cooling reward stack, minor scripted pushes (100 N for
                      0.2 s, ~0.6 m/s, every 3-6 s from 2-5 s on). A fall or a
                      winding at 70 C ends the episode (the failure counters
                      read the fallback's margin, ``g_fall_noheight_thermal70``).
``g1_cool_thermal70_fb``
                      (the user, 2026-11-19) take over from the FALLBACK:
                      spawns on the thermal70 SHAC fallback's states after a
                      minor SONIC handover, every motor's heat weighted by its
                      own temperature (free below 50 C, ramping to 65 C), 1-4
                      random extra hot motors, the handover-jump cost -0.1
                      over 0.5 s, pushes of 0.19-0.97 m/s as _mix.
``g1_cool_thermal70_mix``
                      the same with 30 % of the spawns on LAFAN train mocap
                      frames (70 % on handover rows kicked <= 1 m/s) and
                      pushes of 0.19-0.97 m/s (U(0.2, 1) x 160 N for 0.2 s):
                      the mocap starts the 500M pilot fell from at the eval,
                      small pushes only (the user, 2026-11-18).

Trained by stock PPO (``examples/train_on_policy.py``'s cumulative branch, the
numpy bridge): the scripted pushes act inside the episode, which the SHAC world
model refuses. The policy reads the fallback's 127-D ``actor`` group.
"""

from __future__ import annotations

from ..margins import compose
from ..registry import CUMULATIVE, TaskSpec, register
from .g1_stabilize import G1_BRIDGE_KWARGS, G1_CTRL_DIM

#: the minor pushes (random_push.RandomPush kwargs; magnitude = force_max)
COOL_RANDOM_PUSH = dict(interval_s=(3.0, 6.0), duration_s=0.2, first_delay_s=(2.0, 5.0),
                        horizontal=True)
COOL_PUSH_FORCE_N = 100.0
#: g1_cool_thermal70_mix: every push U(0.2, 1) x 160 N for 0.2 s = 0.19-0.97 m/s on
#: the 33 kg G1 (the user: "small push (<1 m/s)")
COOL_MIX_RANDOM_PUSH = dict(COOL_RANDOM_PUSH, force_frac=(0.2, 1.0))
COOL_MIX_PUSH_FORCE_N = 160.0


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_cool.env_cfg import (
    g1_cool_thermal70_env_cfg, g1_cool_thermal70_fb_env_cfg, g1_cool_thermal70_mix_env_cfg)
  from robot_safety_sandbox.envs.g1_recover.thermal import g_fall_noheight_thermal70
  register(TaskSpec(
    task_id="g1_cool_thermal70", cfg_builder=g1_cool_thermal70_env_cfg,
    margin_fn=compose(g_fall_noheight_thermal70), mode=CUMULATIVE,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dense_margins=True, dstb_mode="wrench",
                random_push=dict(COOL_RANDOM_PUSH), force_max=COOL_PUSH_FORCE_N),
    description="Cooling mode on the identified G1: from a minor-kick SONIC handover "
                "with hot ankle / waist windings, stand through minor pushes and "
                "cool them (dense reward; fall or a winding at 70 C ends the episode)."))
  register(TaskSpec(
    task_id="g1_cool_thermal70_mix", cfg_builder=g1_cool_thermal70_mix_env_cfg,
    margin_fn=compose(g_fall_noheight_thermal70), mode=CUMULATIVE,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dense_margins=True, dstb_mode="wrench",
                random_push=dict(COOL_MIX_RANDOM_PUSH), force_max=COOL_MIX_PUSH_FORCE_N),
    description="g1_cool_thermal70 with 30 % LAFAN mocap spawns (70 % handover rows "
                "kicked <= 1 m/s) and pushes of 0.2-1 m/s (U(0.2, 1) x 160 N, 0.2 s)."))
  register(TaskSpec(
    task_id="g1_cool_thermal70_fb", cfg_builder=g1_cool_thermal70_fb_env_cfg,
    margin_fn=compose(g_fall_noheight_thermal70), mode=CUMULATIVE,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dense_margins=True, dstb_mode="wrench",
                random_push=dict(COOL_MIX_RANDOM_PUSH), force_max=COOL_MIX_PUSH_FORCE_N),
    description="Cooling mode taking over from the thermal70 SHAC fallback (its states "
                "0.5-15 s after a minor SONIC handover), every motor's heat weighted by "
                "its temperature (free below 50 C), 1-4 random extra hot motors, a "
                "stronger handover-jump cost, pushes of 0.2-1 m/s."))
