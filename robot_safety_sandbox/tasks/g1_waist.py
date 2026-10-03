"""Unitree G1 reach-avoid-stay to a waist (roll, pitch) + body-velocity goal.

Two single-player tasks, one per phase of the switched controller
(``filters/switch.py``); the margins and the reasoning are in
``envs/g1_waist/env_cfg.py``.

``g1_waist_stay_arm_random``   AVOID on h = min(g, soft_floor(l_box)): stay
                               upright AND inside the goal set, a fall ranked
                               below any box violation. Trained first.
``g1_waist_reach_arm_random``  REACH-AVOID with g and l = min(l_box, V_stay),
                               V_stay from the frozen stay twin: get safely to
                               where the stay policy can take over.

Both keep the actor on the legs and waist (action dims 0..14) with the arms
(15..28) swinging at random -- g1_stabilize_arm_random's generator, split and
magnitude curriculum -- but at HALF its registered magnitude: ``dstb_gain`` 1.0
(waypoints up to +-0.44 rad from the default pose at the shoulders and elbows,
against 2.0 / +-0.88 rad there). The goal's 0.1 m/s velocity tolerance is far
tighter than the stand's, and the arms' momentum is what the pelvis has to
absorb to meet it. Same 29-wide head and the same observation layout in both
tasks, so the reach task warm-starts WHOLE from a trained stay run.
"""

from __future__ import annotations

from ..registry import AVOID, REACH_AVOID, TaskSpec, register
from .g1_arm_random import G1_ARM_DIM, G1_ARM_RANDOM_BRIDGE_KWARGS
from .g1_stabilize import G1_CTRL_DIM

#: The random arms' magnitude (end of its curriculum) on the waist tasks.
G1_WAIST_ARM_GAIN = 1.0
G1_WAIST_BRIDGE_KWARGS = dict(G1_ARM_RANDOM_BRIDGE_KWARGS,
                              dstb_gain=G1_WAIST_ARM_GAIN)


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_waist.env_cfg import (
    g1_waist_reach_env_cfg, g1_waist_stay_env_cfg, reach_margins, stay_margins)
  register(TaskSpec(
    task_id="g1_waist_stay_arm_random", cfg_builder=g1_waist_stay_env_cfg,
    margin_fn=stay_margins, mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_WAIST_BRIDGE_KWARGS),
    description="Hold a waist roll/pitch + zero-velocity goal without falling "
                "while the arms swing at random (avoid on min(g, l)); the stay "
                "half of g1 reach-avoid-stay."))
  register(TaskSpec(
    task_id="g1_waist_reach_arm_random", cfg_builder=g1_waist_reach_env_cfg,
    margin_fn=reach_margins, mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_WAIST_BRIDGE_KWARGS),
    description="Reach, without falling, a state the trained stay policy "
                "certifies for a waist roll/pitch + zero-velocity goal, arms "
                "swinging at random; the reach half of g1 reach-avoid-stay."))
