"""The dense-reward (mode="cumulative") cfg for the G1 random-arm stand.

The reach-avoid twin ``g1_stabilize_arm_random`` needs no cfg of its own: its
learner reads only the margins (g, l), and the random arm motion lives in the
bridge (``joint_noise.SmoothJointNoise``), so ``g1_stabilize_env_cfg`` serves it
unchanged. A CUMULATIVE learner instead trains on the env's dense reward stack,
and that stack was written for a robot that owns all 29 joints.

Five of its terms score every joint, or every action dim, and so would score the
ARMS — which the actor does not command here and cannot influence:

  stand_still       w=-1    sum of squared deviation from the default pose. At
                            the full random magnitude the arms alone put ~-3.5
                            per step on it, against a +1 ceiling on each tracking
                            term: the return would be mostly arm noise.
  pose              w=+1    exp(-mean(err^2 / std^2)) with a 0.05 rad standing
                            std. Swinging arms pin it at ~0 whatever the legs do,
                            deleting the main posture signal.
  joint_acc_l2      w=-2.5e-7   arm accelerations from the swing.
  joint_pos_limits  w=-10       arm soft-limit excursions.
  action_rate_l2    w=-0.05     reads the SUBMITTED action, whose arm dims are
                                the random command.

Each is narrowed to the actor's own 15 joints / action dims (legs + waist).
Uncontrollable terms do not bias the optimum — the actor cannot move them — but
they add variance to every return and advantage, and at these sizes they would
bury the terms it CAN move.

Left whole-body on purpose: ``angular_momentum`` (w=-0.02) and ``body_ang_vel``
(w=-0.05, torso) — countering the arms' momentum with the waist and legs is the
balance behavior this task is after — and ``self_collisions``, which the arms
trip in ~0.5% of steps with the one-sided shoulder roll (tasks/g1_arm_random.py).
The gait terms (foot_gait, foot_clearance, foot_slip, soft_landing) are gated
off by the zero command and are untouched.
"""

from __future__ import annotations

import re

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g1_stabilize_env_cfg

#: The joints the ACTOR owns — action dims 0..14, legs and waist — as regexes
#: over the G1's joint names. ``tests/test_g1_arm_random.py`` pins that they
#: resolve to exactly the non-arm slice of the compiled model.
ACTOR_JOINTS = (r".*_hip_.*_joint", r".*_knee_joint", r".*_ankle_.*_joint",
                r"waist_.*_joint")
#: The actor's action width: the dims before the random arm slice.
ACTOR_ACTION_DIM = 15
#: Matches a posture-std KEY that names an arm joint (so it can be dropped once
#: ``pose`` no longer covers the arms — mjlab rejects an unmatched key).
_ARM_KEY = re.compile(r".*(shoulder|elbow|wrist).*")


def actor_action_rate_l2(env, action_dim: int = ACTOR_ACTION_DIM) -> torch.Tensor:
  """``mdp.action_rate_l2`` over the actor's own action dims only. The arm dims
  of the submitted action carry the random command, not the actor's output."""
  am = env.action_manager
  return torch.sum(torch.square(
    am.action[:, :action_dim] - am.prev_action[:, :action_dim]), dim=1)


def _actor_joints() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=ACTOR_JOINTS)


def g1_arm_random_dense_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``g1_stabilize_env_cfg`` (flat ground, zero command) with the joint-level
  reward terms narrowed to the joints the actor commands."""
  cfg = g1_stabilize_env_cfg(play=play)
  r = cfg.rewards
  pose = r["pose"].params
  pose["asset_cfg"] = _actor_joints()
  for k in ("std_standing", "std_walking", "std_running"):
    pose[k] = {pat: std for pat, std in pose[k].items()
               if not _ARM_KEY.fullmatch(pat)}
  for name in ("stand_still", "joint_acc_l2", "joint_pos_limits"):
    r[name].params["asset_cfg"] = _actor_joints()
  r["action_rate_l2"].func = actor_action_rate_l2
  return cfg
