"""The waist-posture goal: one (roll, pitch) target per episode.

A command term, so the goal is part of the observation (``generated_commands``)
the way the ``twist`` velocity command already is. The body-frame velocity half
of the goal IS that twist command (pinned to zero for now), so the full goal the
policy sees is ``(waist_roll*, waist_pitch*)`` here plus ``(vx*, vy*, wz*)``.

Sampling: uniform over the box ``[-scale * max_angle, +scale * max_angle]^2``.
``scale`` is the magnitude curriculum: 1.0 (the full box) by default, which is
what an env built OUTSIDE the trainer runs at -- eval, play, the training video
-- and ramped from ``--target-scale-init`` by
:class:`~robot_safety_sandbox.callbacks.TargetScaleRampCallback` during
training, the way the random arms' magnitude is ramped.

One goal per episode: ``resampling_time_range`` defaults to 1e9 s, so the only
draw is the one at reset.

Draw order. mjlab applies reset EVENTS before it resets the command manager, so
a spawn that must place the robot relative to this episode's goal (the stay
task's ``reset_waist_near_target``) calls :meth:`WaistTargetCommand.sample`
itself; the command's own reset then keeps that draw instead of replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

#: The goal joints, in command order.
WAIST_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
#: Half-width of the goal box at scale 1, rad. The G1's waist roll and pitch
#: range is +-0.52 rad; 0.4 keeps the goal (and the stay tolerance around it)
#: off the joint stops.
MAX_ANGLE = 0.4


@dataclass(kw_only=True)
class WaistTargetCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  joint_names: tuple[str, ...] = WAIST_JOINTS
  max_angle: float = MAX_ANGLE
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env) -> "WaistTargetCommand":
    return WaistTargetCommand(self, env)


class WaistTargetCommand(CommandTerm):
  """Per-env waist (roll, pitch) goal, absolute joint angles in rad.

  The G1's default waist pose is 0 on both joints, so these are also the
  targets relative to the default pose (what ``joint_pos_rel`` reads).
  """

  cfg: WaistTargetCommandCfg

  def __init__(self, cfg: WaistTargetCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    ids, _ = self.robot.find_joints(cfg.joint_names, preserve_order=True)
    self.joint_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
    self.target = torch.zeros(self.num_envs, len(ids), device=self.device)
    #: Magnitude curriculum, a fraction of ``max_angle``.
    self.scale = 1.0
    #: Read by the trainer: attach a TargetScaleRampCallback to this term.
    self.ramped = True
    # Envs whose goal a reset event already drew for the episode being started.
    self._drawn = torch.zeros(self.num_envs, dtype=torch.bool,
                              device=self.device)
    self.metrics["error_waist"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target

  def sample(self, env_ids: torch.Tensor) -> None:
    """Draw this episode's goal for ``env_ids`` now, ahead of the command
    manager's reset (which will then keep it)."""
    half = float(self.scale) * float(self.cfg.max_angle)
    u = torch.rand(len(env_ids), self.target.shape[1], device=self.device)
    self.target[env_ids] = (2.0 * u - 1.0) * half
    self._drawn[env_ids] = True

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    fresh = env_ids[~self._drawn[env_ids]]
    if len(fresh) > 0:
      self.sample(fresh)
    self._drawn[env_ids] = False

  def _update_metrics(self) -> None:
    # Episode mean of the larger of the two |q - q*| (mjlab's convention for a
    # tracking error: accumulated per step, normalized by the longest episode).
    q = self.robot.data.joint_pos[:, self.joint_ids]
    err = (q - self.target).abs().amax(dim=1)
    self.metrics["error_waist"] += err / float(self._env.max_episode_length)

  def _update_command(self) -> None:
    pass
