"""Unitree G1 flat-ground stabilization / locomotion vs an ARM adversary.

The robustification step past ``g1_stabilize``, whose adversary was a 3-dim
worst-case wrench on ``torso_link``. Here the adversary is an ARM POLICY: it
commands the G1's 14 arm joints while the control player commands the 12 leg and
3 waist joints, and it is penalized for driving those arms into the robot's own
body. See ``envs/g1_arm_adversary/env_cfg.py`` for the mechanism and for why the
self-collision penalty cannot live in ``g``.

What differs from the ``g1_stabilize`` registration, and why:

``dstb_dim=14``      the arm joints, not a 3-dim force.
``dstb_mode="joint"`` the bridge OVERWRITES action dims 15..28 with the
                     adversary's command instead of applying an external wrench.
``ctrl_dim=29``      unchanged, and deliberately so: the actor keeps its full
                     29-wide head (its 14 arm outputs are discarded) so a trained
                     ``g1_stabilize`` / ``g1_stabilize_adv`` checkpoint warm-starts
                     WHOLE — head and log_std included — which a 15-wide actor
                     could not. That transfer is the point of the task.
``dstb_gain=1.0``    the adversary's joint authority, ramped from 0.1 by
                     ``callbacks.DstbGainRampCallback``. 1.0 = the same per-joint
                     authority the actor has, since mjlab's per-joint
                     ``G1_ACTION_SCALE`` is applied to both players alike.
``adversary_body``   absent: a joint-mode adversary applies no external wrench,
                     so ``force_max`` and the force ramp mean nothing here and
                     the trainer skips them.

Both tasks are mode="reach-avoid" and ``supports_adversary=True`` -> with
``--adversary`` they resolve to ReachAvoidPPO2P / ReachAvoidSAC2P. Run WITHOUT
``--adversary`` and the arms simply sit at the default pose: that 1P run is the
honest baseline this task is measured against, and its margins are bit-identical
to ``g1_stabilize``'s.
"""

from __future__ import annotations

from ..registry import REACH_AVOID, TaskSpec, register

#: Bridge kwargs shared by every G1 arm-adversary task. ``ctrl_gain`` and
#: ``action_bound`` carry over from ``tasks/g1_stabilize.py`` — see that module
#: for why the G1 wants 1.0 and 3.0 respectively (T006).
G1_ARM_BRIDGE_KWARGS = {
  "ctrl_gain": 1.0,
  "action_bound": 3.0,
  "dstb_mode": "joint",
  "dstb_joint_slice": (15, 29),
  "dstb_gain": 1.0,
}
#: Actuated joints (the floating base is not one) = the control action width.
G1_CTRL_DIM = 29
#: Arm joints = the disturbance action width.
G1_ARM_DIM = 14


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_arm_adversary.env_cfg import (
    arm_locomote_margins,
    arm_stance_margins,
    g1_arm_adv_locomote_env_cfg,
    g1_arm_adv_stabilize_env_cfg,
  )
  register(TaskSpec(
    task_id="g1_stabilize_arm_adv", cfg_builder=g1_arm_adv_stabilize_env_cfg,
    margin_fn=arm_stance_margins, mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=True,
    kwargs=dict(G1_ARM_BRIDGE_KWARGS),
    description="Flat ground, zero command: return to a stable stand while an "
                "adversary drives the arms (and pays for hitting the body)."))
  register(TaskSpec(
    task_id="g1_locomote_arm_adv", cfg_builder=g1_arm_adv_locomote_env_cfg,
    margin_fn=arm_locomote_margins, mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=True,
    kwargs=dict(G1_ARM_BRIDGE_KWARGS),
    description="Flat ground, constant forward command: keep walking while an "
                "adversary drives the arms (and pays for hitting the body)."))
