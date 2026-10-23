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
* optionally (``rest_terminate=True``) the episode ends as a TRUNCATION once
  the robot has been back in the stance set for ``rest_dwell_s`` (after
  ``rest_arm_s``): :class:`~robot_safety_sandbox.envs.g1_recover.rest.RestReached`.
  Every episode is then one event, critical state -> fall or rest.

THE REST TARGET. By default the rest set is g1_stabilize's ``stance_margins``.
Setting any of ``rest_lin_tol`` / ``rest_ang_tol`` / ``rest_gate_on_push``
swaps in ``compose(g_fall, targets.make_l_stance(lin_tol, ang_tol,
gate_on_push))`` (None tolerances: g1_stabilize's 0.25 m/s / 0.30 rad/s), so
a reach-avoid task's episode ends after dwelling in ITS OWN target set -- the
truncation then agrees with the l the learner reaches for, and the gate keeps
a start that already sits in the set from ending before its push has landed.
The episode end stays a truncation, never the "reach-avoid" end criterion's
success termination, which the curricula and safety counters would read as a
fall.

THE SAMPLER. ``sampler`` / ``uniform_mix`` / ``weight_column`` /
``online_alpha`` / ``sequential_repeats`` choose how mocap / handover rows are
drawn (or, ``"sequential"``, walk the whole bank for labeling); ``dataset``
may be a list of banks. See ``spawn.py``; the defaults are today's spawn.
"""

from __future__ import annotations

import os

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from robot_safety_sandbox.envs.g1_recover.rest import RestReached
from robot_safety_sandbox.envs.g1_recover.spawn import (
  SAMPLERS, RecoverCurriculum, RecoverSpawn)
from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
  ANG_TOL, CLAMP, LIN_TOL, g1_stabilize_env_cfg, g_fall)
from robot_safety_sandbox.margins import compose

#: Recovery episode length, s.
EPISODE_LENGTH_S = 8.0


def pelvis_height(env) -> torch.Tensor:
  """Pelvis (root link) height above the flat ground, m, ``(num_envs, 1)``:
  the quantity the height term of the fall margin g reads. No actor observation
  carries it (forward kinematics recovers it only while a foot is planted)."""
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def rest_target_margins(lin_tol: float | None = None,
                        ang_tol: float | None = None,
                        gate_on_push: bool = False):
  """``(g, l)`` of a rest target: g_fall and ``targets.make_l_stance`` at these
  tolerances (None -> g1_stabilize's), clamped like ``stance_margins``."""
  from robot_safety_sandbox.envs.g1_recover.targets import make_l_stance
  return compose(g_fall, make_l_stance(
    lin_tol=LIN_TOL if lin_tol is None else float(lin_tol),
    ang_tol=ANG_TOL if ang_tol is None else float(ang_tol),
    gate_on_push=bool(gate_on_push)), clamp=CLAMP)


def g1_recover_env_cfg(play: bool = False,
                       dataset: str | list[str] | None = None,
                       weights: dict | None = None, levels: int = 5,
                       band_width: float = 0.4, init_level: int = 0,
                       episode_length_s: float = EPISODE_LENGTH_S,
                       ema: float = 0.002, rest_terminate: bool = False,
                       rest_dwell_s: float = 0.4,
                       rest_arm_s: float = 0.8, sampler: str = "ladder",
                       uniform_mix: float = 0.15,
                       weight_column: str = "p_surv",
                       online_alpha: float = 0.1,
                       sequential_repeats: int = 1,
                       rest_lin_tol: float | None = None,
                       rest_ang_tol: float | None = None,
                       rest_gate_on_push: bool = False) -> ManagerBasedRlEnvCfg:
  """All keyword arguments can be set from a run config's ``env_overrides``.
  The rest-target kwargs act only with ``rest_terminate=True``."""
  if sampler not in SAMPLERS:
    raise ValueError(f"g1_recover sampler {sampler!r}: one of {SAMPLERS}")
  cfg = g1_stabilize_env_cfg(play=play)
  if not play:
    cfg.episode_length_s = float(episode_length_s)
  if "critic" in cfg.observations:
    # Privileged critic group only (the actor is untouched): with base_lin_vel
    # (already in the group) and projected_gravity, the critic then sees every
    # term of g it can bind on but the transient non-foot contact. Inert unless
    # a run asks for the critic group (train_on_policy --privileged-critic).
    cfg.observations["critic"].terms["pelvis_height"] = ObservationTermCfg(
      func=pelvis_height)
  if rest_terminate:
    # A truncation (time_out=True), see rest.py: survival for the curricula,
    # terminal at its own margin for the avoid backup.
    params = dict(dwell_s=float(rest_dwell_s), arm_s=float(rest_arm_s))
    if (rest_lin_tol is not None or rest_ang_tol is not None
        or rest_gate_on_push):
      params["margin_fn"] = rest_target_margins(rest_lin_tol, rest_ang_tol,
                                                rest_gate_on_push)
    cfg.terminations["recovered"] = TerminationTermCfg(
      func=RestReached, time_out=True, params=params)
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  common = {"levels": int(levels), "init_level": int(init_level),
            "sampler": str(sampler)}
  if dataset is not None and not isinstance(dataset, (str, os.PathLike)):
    dataset = [os.fspath(x) for x in dataset]   # a YAML list / a tuple
  cfg.events["recover_spawn"] = EventTermCfg(
    func=RecoverSpawn, mode="reset",
    params=dict(common, dataset=dataset, weights=weights,
                band_width=float(band_width), uniform_mix=float(uniform_mix),
                weight_column=str(weight_column),
                sequential_repeats=int(sequential_repeats)))
  cfg.curriculum["recover"] = CurriculumTermCfg(
    func=RecoverCurriculum, params=dict(common, ema=float(ema),
                                        online_alpha=float(online_alpha)))
  return cfg
