"""Unitree G1 fall recovery: avoid-only, spawned from harvested walker states.

The fallback a G1 safety filter hands control to. It trains on the states it
will actually take over from — the walker mid-gait, the walker just after a
push, the walker a moment before it falls — plus synthetic tilted / moving
stances and the plain stance (``envs/g1_recover/spawn.py``; the bank comes from
``examples/harvest_g1_recover_states.py``). The objective is AVOID only,
``V = min(g, gamma V')`` on g1_stabilize's fall margin: V > 0 reads "will not
fall", and nothing asks the robot to stop moving (another policy handles
bringing it to rest).

``g1_recover``            the actor owns all 29 joints and may use its arms to
                          balance.
``g1_recover_arm_random`` the actor owns legs + waist; the arms swing at random
                          (g1_stabilize_arm_random's generator, magnitude
                          curriculum and all), starting each episode from the
                          arm command in the restored state.
``g1_recover_arm_random_wrench``
                          g1_recover_arm_random plus a LEARNED wrench on
                          ``torso_link`` under ``--adversary`` (SafetyPPO2P):
                          the arms stay the scripted joint slice and the
                          adversary picks a 3-dim force direction, applied at
                          force_max x the per-env force_scale (g1_stabilize's
                          wrench channel). Without ``--adversary`` it is
                          g1_recover_arm_random.

All keep the 29-wide head and g1_stabilize's observations, so they warm-start
from any g1_stabilize* checkpoint, and the wrench task WHOLE from a trained
g1_recover_arm_random one.
"""

from __future__ import annotations

from ..margins import avoid_only
from ..registry import AVOID, TaskSpec, register
from .g1_arm_random import G1_ARM_DIM, G1_ARM_RANDOM_BRIDGE_KWARGS
from .g1_stabilize import G1_BRIDGE_KWARGS, G1_CTRL_DIM


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import stance_margins
  register(TaskSpec(
    task_id="g1_recover", cfg_builder=g1_recover_env_cfg,
    margin_fn=avoid_only(stance_margins), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="Recover from harvested walker / pushed / synthetic states "
                "without falling, using all 29 joints (avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_arm_random", cfg_builder=g1_recover_env_cfg,
    margin_fn=avoid_only(stance_margins), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS),
    description="g1_recover with legs + waist only while the arms swing at "
                "random from the restored arm pose (avoid-only)."))
  # The random arms keep their joint slice (dstb_joint_slice, dstb_gain,
  # joint_noise); dstb_mode / dstb_dim / adversary_body describe the LEARNED
  # player, a 3-dim torso wrench as on g1_stabilize.
  register(TaskSpec(
    task_id="g1_recover_arm_random_wrench", cfg_builder=g1_recover_env_cfg,
    margin_fn=avoid_only(stance_margins), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=True,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS, dstb_mode="wrench",
                adversary_body=G1_BRIDGE_KWARGS["adversary_body"]),
    description="g1_recover_arm_random plus an adversarial torso wrench "
                "(--adversary), ramped by a force curriculum (avoid-only)."))
