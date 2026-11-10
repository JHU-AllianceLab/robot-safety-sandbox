"""Unitree G1 flat-ground stabilization / locomotion vs an adversarial force.

The humanoid twin of ``go2_stabilize`` / ``go2_locomote``: same shape of problem
(flat ground, stock spawns, no curricula, no staged pipeline, task-local
margins), 29 actuated joints instead of 12. Both tasks are mode="reach-avoid"
and support ``--adversary`` -> the two-player game (ReachAvoidPPO2P /
ReachAvoidSAC2P).

G1 specifics vs the Go2 defaults:

``ctrl_dim=29``     the G1's actuated joints (12 leg, 3 waist, 14 arm).
``ctrl_gain=1.0``   the mjlab action term already carries a PER-JOINT scale
                    (``G1_ACTION_SCALE`` = 0.25 * effort / stiffness, e.g. 0.35
                    rad/unit at a knee, 0.55 at a hip pitch), so the bridge does
                    not rescale on top of it the way it does for the Go2's flat
                    0.25. The mjlab scale IS the joint authority.
``action_bound=3.0`` the mjlab JointPositionAction has no clip of its own, so the
                    bridge's bound is the only limit on the per-joint delta. The
                    historical 1.0 caps a knee at 0.35 rad — a stride cap that
                    makes a humanoid shuffle (T006). 3.0 leaves the stride to the
                    policy; the sampled action still lives near +-1.
``adversary_body``  ``torso_link`` — a push lands on the upper body, as it does
                    for Digit. Re-register with ``pelvis`` for a pure base wrench
                    (the Go2's convention).
"""

from __future__ import annotations

from ..margins import compose
from ..registry import REACH_AVOID, TaskSpec, register

#: Bridge kwargs shared by every G1 task — see the module docstring.
G1_BRIDGE_KWARGS = {
  "ctrl_gain": 1.0,
  "action_bound": 3.0,
  "adversary_body": "torso_link",
}
#: Actuated joints (the floating base is not one).
G1_CTRL_DIM = 29


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
    g_fall,
    g1_locomote_env_cfg,
    g1_stabilize_env_cfg,
    l_stance,
    l_track,
  )
  # REACH-AVOID.  avoid margin: g_fall (don't fall)
  #               reach margin: l_stance (the stance band)
  register(TaskSpec(
    task_id="g1_stabilize", cfg_builder=g1_stabilize_env_cfg,
    margin_fn=compose(g_fall, l_stance), mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, supports_adversary=True,
    kwargs=dict(G1_BRIDGE_KWARGS),
    description="Flat ground, zero command: return to a stable stand despite "
                "an adversarial torso force (the humanoid go2_stabilize)."))
  # REACH-AVOID.  avoid margin: g_fall (don't fall)
  #               reach margin: l_track (walk at the command)
  register(TaskSpec(
    task_id="g1_locomote", cfg_builder=g1_locomote_env_cfg,
    margin_fn=compose(g_fall, l_track), mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, supports_adversary=True,
    kwargs=dict(G1_BRIDGE_KWARGS),
    description="Flat ground, constant forward command: keep walking at it "
                "despite an adversarial torso force."))
