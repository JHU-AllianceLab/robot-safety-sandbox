"""End a recovery episode once the robot is back at rest.

A recovery task's episodes are mostly quiet standing: the recovery takes 1-2 s
and the rest of an 8 s episode carries no decision and (under the avoid
backup) no advantage. :class:`RestReached` is a TERMINATION term that ends the
episode once the stance margin ``l`` (upright, pelvis in its height band, base
twist small -- ``g1_stabilize.env_cfg.stance_margins``) has been non-negative
for ``dwell_s`` without a break, and only after ``arm_s`` of episode time.

Register it with ``time_out=True``: it is a TRUNCATION, not a failure. The
bridge then reports it as a timeout, so the survival curricula
(``RecoverCurriculum``, ``PerEnvForceScaleCallback``) promote on it, the
safety counters and the gamma ladder do not count it as a fall, and the avoid
backup values the step at its own margin ``g`` -- which is sound because a
state that has sat in the stance set for ``dwell_s`` is one from which
standing still keeps the robot safe. The dwell keeps a one-step graze through
the set from ending the episode; ``arm_s`` keeps the term quiet until the
task's first disturbance has had time to land (a nominal spawn is IN the
stance set from step 0).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from mjlab.managers.manager_base import ManagerTermBase


class RestReached(ManagerTermBase):
  """Termination (``time_out=True``): ``l >= 0`` for ``dwell_s`` after ``arm_s``.

  Params (``TerminationTermCfg.params``): ``dwell_s`` (default 0.4),
  ``arm_s`` (default 0.0), ``margin_fn`` (``env -> (g, l)``; default
  ``stance_margins``; tests inject their own).
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    dt = float(env.step_dt)
    self.dwell = max(1, int(round(float(p.get("dwell_s", 0.4)) / dt)))
    self.arm = int(round(float(p.get("arm_s", 0.0)) / dt))
    fn: Optional[Callable] = p.get("margin_fn")
    if fn is None:
      from robot_safety_sandbox.envs.g1_stabilize.env_cfg import stance_margins
      fn = stance_margins
    self.margin_fn = fn
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids=None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self.count.zero_()
    else:
      self.count[env_ids] = 0

  def __call__(self, env, **_params) -> torch.Tensor:
    _g, l = self.margin_fn(env)
    inside = l.reshape(-1) >= 0.0
    self.count = torch.where(inside, self.count + 1, torch.zeros_like(self.count))
    armed = env.episode_length_buf >= self.arm
    return armed & (self.count >= self.dwell)
