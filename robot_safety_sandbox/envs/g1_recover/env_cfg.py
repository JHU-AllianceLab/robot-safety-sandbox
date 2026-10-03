"""g1_recover cfg: g1_stabilize with the stock spawn replaced by the bank spawn.

Everything that defines the robot and the world is g1_stabilize's: flat
ground, the twist command pinned to zero (so the ``phase`` observation reads
zero and the command the fallback sees is the one the filter holds while it is
engaged), the same domain randomization and the same terminations. What
changes:

* ``reset_base`` / ``reset_robot_joints`` are replaced by one
  :class:`~robot_safety_sandbox.envs.g1_recover.spawn.RecoverSpawn` event;
* a :class:`~robot_safety_sandbox.envs.g1_recover.spawn.RecoverCurriculum`
  term moves the per-source difficulty levels and logs survival;
* episodes are ``episode_length_s`` long (default 8 s, not 20). Recovery takes
  1-2 s, after which a 20 s episode is mostly quiet standing, and with no
  timeout bootstrapping only the last ~1 s before a timeout carries an
  optimistic target; shorter episodes also mean more starts per sample.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg

from robot_safety_sandbox.envs.g1_recover.spawn import (
  RecoverCurriculum, RecoverSpawn)
from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g1_stabilize_env_cfg

#: Recovery episode length, s.
EPISODE_LENGTH_S = 8.0


def g1_recover_env_cfg(play: bool = False, dataset: str | None = None,
                       weights: dict | None = None, levels: int = 5,
                       band_width: float = 0.4, init_level: int = 0,
                       episode_length_s: float = EPISODE_LENGTH_S,
                       ema: float = 0.002) -> ManagerBasedRlEnvCfg:
  """All keyword arguments can be set from a run config's ``env_overrides``."""
  cfg = g1_stabilize_env_cfg(play=play)
  if not play:
    cfg.episode_length_s = float(episode_length_s)
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  common = {"levels": int(levels), "init_level": int(init_level)}
  cfg.events["recover_spawn"] = EventTermCfg(
    func=RecoverSpawn, mode="reset",
    params=dict(common, dataset=dataset, weights=weights,
                band_width=float(band_width)))
  cfg.curriculum["recover"] = CurriculumTermCfg(
    func=RecoverCurriculum, params=dict(common, ema=float(ema)))
  return cfg
