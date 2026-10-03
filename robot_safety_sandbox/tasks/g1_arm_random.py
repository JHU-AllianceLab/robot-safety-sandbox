"""Unitree G1 flat-ground stabilization / locomotion while the ARMS MOVE AT RANDOM.

The single-player stepping stone between ``g1_stabilize`` (arms parked at the
default pose) and ``g1_stabilize_arm_adv`` (a learned adversary owns the arms):
the same 29-joint split, but action dims 15..28 are driven by a smooth random
process (``joint_noise.SmoothJointNoise``: minimum-jerk glides between random
waypoints) instead of a second player. The actor keeps the legs and waist, and
learns to hold its stance — or keep walking — while its own arms swing.

What differs from the ``g1_stabilize_arm_adv`` registration, and why:

``joint_noise``        the bridge generates the arm commands itself; the kwargs
                       are the generator's (segment durations, in seconds).
``supports_adversary`` False: the arm slice already has an owner, and the bridge
                       refuses a joint_noise env with adversary=True.
``dstb_gain``          the random arms' MAGNITUDE — the half-width of the
                       waypoint range, in the actor's own action units (mjlab's
                       per-joint ``G1_ACTION_SCALE`` then applies, 0.44 rad/unit
                       at the shoulders, elbows and wrist roll, 0.075 at wrist
                       pitch/yaw). This registered value is the curriculum's
                       END point; the trainer ramps up to it with the joint
                       adversary's two callbacks (``DstbGainRampCallback`` from
                       ``--dstb-gain-init``, then ``PerEnvForceScaleCallback``
                       on ``dstb_scale``). An env built OUTSIDE the trainer —
                       eval, play, the training video — runs at this full value.
``margin_fn``          ``g1_stabilize``'s, unchanged, so V here certifies the
                       same "can hold / return to the stance" and a checkpoint
                       warm-starts from (or into) either neighbor WHOLE: same
                       29-wide head, same obs, with ``last_action`` carrying the
                       arm command about to be applied, as it does for the
                       adversary.

No self-collision PENALTY, unlike the adversary: a random process is not
optimizing anything, so there is nothing that could learn to exploit the
contact solver, and no second learner to charge for it. Self-collision is
instead kept rare by the waypoint RANGE — see :data:`G1_ARM_WAYPOINT_LO`.

``g1_stabilize_arm_random_cumulative`` is the same stand under the CUMULATIVE
backup (reward + gamma V', stock SB3 PPO) instead of reach-avoid: identical
bridge, random arms and magnitude curriculum, but trained on the env's dense
reward — with its joint-level terms narrowed to the legs and waist, see
``envs/g1_arm_random/env_cfg.py``. It is registered with ``dense_margins`` and
g1_stabilize's margins, so its run logs the same ``safety/*`` failure and
margin scalars as the reach-avoid twin, for a like-for-like comparison.
"""

from __future__ import annotations

from ..margins import avoid_only
from ..registry import AVOID, CUMULATIVE, REACH_AVOID, TaskSpec, register

#: The arm joints in action order, per side (dims 15..21 left, 22..28 right;
#: ``tests/test_g1_arm_random.py`` pins this against the compiled model).
G1_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")

#: Waypoint range per arm joint (``SmoothJointNoise`` lo/hi, normalized units):
#: the full [-1, 1] everywhere EXCEPT the shoulder roll, which may only swing
#: OUTWARD of the default pose (left roll >= 0, right roll <= 0 — the right arm's
#: roll axis is mirrored). Measured with the arm-adversary cfg's arm-vs-body
#: contact sensor (zero actor, upright steps, magnitude 3): with the roll on the
#: torso side of 0 rad the arm is on the torso in 25-80% of steps, against
#: <= 4% anywhere outward of +0.1 rad (the default is +0.2). No other joint
#: moved that rate at all. A symmetric range therefore put the
#: arms into the body in 16% of upright steps at magnitude 2; one-sided roll
#: removes the cause without shrinking any other joint's motion.
G1_ARM_WAYPOINT_LO = tuple(
  (0.0 if side == "left" else -1.0) if j == "shoulder_roll" else -1.0
  for side in ("left", "right") for j in G1_ARM_JOINTS)
G1_ARM_WAYPOINT_HI = tuple(
  (1.0 if side == "left" else 0.0) if j == "shoulder_roll" else 1.0
  for side in ("left", "right") for j in G1_ARM_JOINTS)

#: Bridge kwargs shared by every G1 random-arm task. ``ctrl_gain`` and
#: ``action_bound`` carry over from ``tasks/g1_stabilize.py`` (T006), the slice
#: from ``tasks/g1_arm_adversary.py``. ``dstb_gain`` 2.0 = waypoints up to
#: +-0.88 rad from the default pose at the shoulders and elbows; the arm
#: adversary's reach is +-3.0 (its action is clipped at ``action_bound``).
G1_ARM_RANDOM_BRIDGE_KWARGS = {
  "ctrl_gain": 1.0,
  "action_bound": 3.0,
  "dstb_mode": "joint",
  "dstb_joint_slice": (15, 29),
  "dstb_gain": 2.0,
  "joint_noise": {"seg_min_s": 0.5, "seg_max_s": 2.0,
                  "lo": G1_ARM_WAYPOINT_LO, "hi": G1_ARM_WAYPOINT_HI},
}
#: Actuated joints (the floating base is not one) = the control action width.
G1_CTRL_DIM = 29
#: Arm joints = the width of the randomly driven slice.
G1_ARM_DIM = 14


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_arm_random.env_cfg import (
    g1_arm_random_dense_env_cfg,
  )
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
    g1_locomote_env_cfg,
    g1_stabilize_env_cfg,
    locomote_margins,
    stance_margins,
  )
  register(TaskSpec(
    task_id="g1_stabilize_arm_random", cfg_builder=g1_stabilize_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS),
    description="Flat ground, zero command: hold a stable stand while the arms "
                "swing smoothly at random, magnitude ramped by a curriculum."))
  # Avoid-only twin: same g, the stance target stripped (avoid_only), so the
  # backup is V = min(g, gamma V') and nothing asks the robot to stop moving --
  # only not to fall. Same obs and head, so it warm-starts from either twin.
  register(TaskSpec(
    task_id="g1_stabilize_arm_random_avoid", cfg_builder=g1_stabilize_env_cfg,
    margin_fn=avoid_only(stance_margins), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS),
    description="g1_stabilize_arm_random avoid-only: don't fall while the arms "
                "swing at random; no stance (reach) target."))
  register(TaskSpec(
    task_id="g1_locomote_arm_random", cfg_builder=g1_locomote_env_cfg,
    margin_fn=locomote_margins, mode=REACH_AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS),
    description="Flat ground, constant forward command: keep walking while the "
                "arms swing smoothly at random, magnitude ramped by a curriculum."))
  register(TaskSpec(
    task_id="g1_stabilize_arm_random_cumulative",
    cfg_builder=g1_arm_random_dense_env_cfg,
    margin_fn=stance_margins, mode=CUMULATIVE,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS, dense_margins=True),
    description="g1_stabilize_arm_random on the dense reward (cumulative "
                "backup, stock SB3 PPO), joint terms narrowed to legs + waist."))
