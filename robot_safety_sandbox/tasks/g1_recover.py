"""Unitree G1 fall recovery: avoid-only, spawned from harvested walker states.

The fallback a G1 safety filter hands control to. It trains on the states it
will actually take over from — the walker mid-gait, the walker just after a
push, the walker a moment before it falls — plus synthetic tilted / moving
stances and the plain stance (``envs/g1_recover/spawn.py``; the bank comes from
``examples/harvest_g1_recover_states.py``). The objective is AVOID only,
``V = min(g, gamma V')`` on g1_stabilize's fall margin: V > 0 reads "will not
fall", and nothing asks the robot to stop moving (another policy handles
bringing it to rest). The exceptions are the two reach-avoid LAFAN twins
below, whose target is the stance set after the push.

``g1_recover``            the actor owns all 29 joints and may use its arms to
                          balance.
``g1_recover_wrench``     g1_recover plus a LEARNED wrench on ``torso_link``
                          under ``--adversary`` (SafetyPPO2P): the adversary
                          picks a 3-dim force direction, applied at force_max x
                          the per-env force_scale (g1_stabilize's wrench
                          channel). Without ``--adversary`` it is g1_recover.
``g1_recover_push``       g1_recover plus SCRIPTED random pushes on ``torso_link``
                          (``random_push.RandomPush``: random horizontal direction,
                          an onset every 2-4 s, 0.2 s long, at force_max x the
                          per-env force_scale, ramped by the force curriculum),
                          single-player. The bridge reports a per-episode regime
                          (spawn source x push bucket) for a multi-gamma learner.
``g1_recover_push_fast``  g1_recover_push with the push 0.1-0.4 s after the
                          spawn; with ``env_overrides`` ``rest_terminate`` (end
                          at rest, ``envs/g1_recover/rest.py``) and a 3 s cap
                          every episode is one event, so a fixed gamma ~0.99
                          covers it. ``examples/harvest_g1_frontier_states.py``
                          builds a start bank from a trained push policy's own
                          frontier for it.
``g1_recover_push_rear``  g1_recover_push_fast with half the pushes drawn from
                          BEHIND the robot (force within 60 deg of its heading,
                          driving it forward), the side a push_fast policy
                          withstands least; the other half stay uniform.
``g1_recover_lafan``      the LAFAN recovery reference (AVOID): push_fast's push
                          timing with at most ONE random push per episode
                          (``max_pushes=1``), so every episode is one event --
                          start state, its push, fall or not. Meant for a bank
                          with retargeted-mocap / handover rows (``env_overrides``
                          ``dataset`` / ``weights`` / ``sampler``); a handover
                          row's pending push resumes on the first step and
                          replaces the random one (bridge ``_apply_pending_push``).
``g1_recover_lafan_ra``   g1_recover_lafan as REACH-AVOID: the target is the
                          stance set with |v| <= 0.25 m/s and |w| <= 0.30 rad/s
                          per body axis (g1_stabilize's l_stance), GATED on the
                          push (unreachable until the episode's push has landed
                          and ended, ``envs/g1_recover/targets.py``). The
                          episode keeps ``end_criterion="failure"`` and ends as
                          a truncation after dwelling in that same gated target
                          (``rest_terminate`` with the same tolerances), never
                          on a success termination, which the survival
                          curricula and the safety counters would count as a
                          fall. The value certifies reaching the target safely;
                          it says nothing about staying there.
``g1_recover_lafan_ra_loose``
                          g1_recover_lafan_ra with the looser target
                          |v| <= 0.5 m/s, |w| <= 0.6 rad/s.
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
from any g1_stabilize* checkpoint, and each wrench task WHOLE from a trained
checkpoint of the task it extends (g1_recover / g1_recover_arm_random).
"""

from __future__ import annotations

import functools

from ..margins import compose
from ..registry import AVOID, REACH_AVOID, TaskSpec, register
from .g1_arm_random import G1_ARM_DIM, G1_ARM_RANDOM_BRIDGE_KWARGS
from .g1_stabilize import G1_BRIDGE_KWARGS, G1_CTRL_DIM

#: The LAFAN recovery tasks' push: push_fast's timing (the push lands 0.1-0.4 s
#: after the spawn, 0.2 s long) and at most one random push per episode.
LAFAN_RANDOM_PUSH = dict(interval_s=(1.0, 2.0), duration_s=0.2,
                         first_delay_s=(0.1, 0.4), horizontal=True, max_pushes=1)
#: Reach targets of the reach-avoid LAFAN tasks: (lin_tol m/s, ang_tol rad/s)
#: per body axis. The strict one is g1_stabilize's stance tolerance.
LAFAN_RA_TOLS = {"g1_recover_lafan_ra": (0.25, 0.30),
                 "g1_recover_lafan_ra_loose": (0.5, 0.6)}


def register_all() -> None:
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  from robot_safety_sandbox.envs.g1_recover.targets import make_l_stance
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall
  # Every g1_recover* task but the g1_recover_lafan_ra* twins (below) is
  # AVOID ONLY:
  #   avoid margin: g_fall (don't fall: pelvis height, pelvis tilt, non-foot
  #                 ground contact -- g1_stabilize's)
  #   reach margin: none (compose(g_fall) declares no target set; nothing asks
  #                 the robot to come back to a stance)
  register(TaskSpec(
    task_id="g1_recover", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="Recover from harvested walker / pushed / synthetic states "
                "without falling, using all 29 joints (avoid-only)."))
  # g1_recover's env, margins and head; the learned player is g1_stabilize's
  # 3-dim torso wrench.
  register(TaskSpec(
    task_id="g1_recover_wrench", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=True,
    kwargs=dict(G1_BRIDGE_KWARGS, dstb_mode="wrench"),
    description="g1_recover plus an adversarial torso wrench (--adversary), "
                "ramped by a force curriculum (avoid-only)."))
  # g1_recover's env, margins and head; the disturbance is a SCRIPTED random
  # push on torso_link (random_push.RandomPush) at force_max x force_scale, so
  # a single player trains against the wrench adversary's two curricula. The
  # regime a multi-gamma learner scores each episode with is spawn source x
  # the per-env force scale in quarters (bridge regime()).
  register(TaskSpec(
    task_id="g1_recover_push", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dstb_mode="wrench",
                random_push=dict(interval_s=(2.0, 4.0), duration_s=0.2,
                                 first_delay_s=(0.5, 2.0), horizontal=True),
                regime_force_buckets=4),
    description="g1_recover plus scripted random pushes on torso_link (random "
                "horizontal direction, every 2-4 s for 0.2 s, at force_max x the "
                "per-env force scale, ramped by the force curriculum); regimes "
                "= spawn source x push bucket (avoid-only, single-player)."))
  # The SHORT-HORIZON variant: the push lands right after the spawn (0.1-0.4 s)
  # and, with env_overrides rest_terminate / episode_length_s (see
  # configs/g1_recover_push_fast_safetyppo1p.yaml), the episode is one event:
  # critical state -> fall or back at rest, 1-3 s. Every sample then sits
  # within a short horizon of its outcome, so a fixed gamma ~0.99 is right.
  register(TaskSpec(
    task_id="g1_recover_push_fast", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dstb_mode="wrench",
                random_push=dict(interval_s=(1.0, 2.0), duration_s=0.2,
                                 first_delay_s=(0.1, 0.4), horizontal=True),
                regime_force_buckets=4),
    description="g1_recover_push with the push 0.1-0.4 s after the spawn, for "
                "short episodes that end at rest (avoid-only, single-player)."))
  # push_fast with the push directions REAR-BIASED: the push_fast_cont policy
  # survives ~120 N from directly behind vs ~245 N on the chest, and the
  # direction-blind per-env force scale settles where rear pushes are mostly
  # lost. Half the pushes come from within 60 deg of straight behind (2/3 of
  # all pushes land in that 120 deg sector), so the scale settles nearer the
  # rear frontier and those pushes carry gradient.
  register(TaskSpec(
    task_id="g1_recover_push_rear", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=dict(G1_BRIDGE_KWARGS, dstb_mode="wrench",
                random_push=dict(interval_s=(1.0, 2.0), duration_s=0.2,
                                 first_delay_s=(0.1, 0.4), horizontal=True,
                                 rear_frac=0.5, rear_half_width_deg=60.0),
                regime_force_buckets=4),
    description="g1_recover_push_fast with half the pushes from behind the "
                "robot (within 60 deg of its heading), the rest uniform "
                "(avoid-only, single-player)."))
  # The LAFAN recovery reference: push_fast's push, ONE per episode. The start
  # bank and its sampler come from the run config (env_overrides), so the
  # experiment arms differ only in config, never in the task.
  def lafan_kwargs() -> dict:     # a fresh copy per task, push dict included
    return dict(G1_BRIDGE_KWARGS, dstb_mode="wrench",
                random_push=dict(LAFAN_RANDOM_PUSH), regime_force_buckets=4)
  register(TaskSpec(
    task_id="g1_recover_lafan", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="Recover from mocap / handover start states and one push "
                "0.1-0.4 s after the spawn without falling (avoid-only, "
                "single-player)."))
  # REACH-AVOID twins. avoid margin: g_fall; reach margin: the stance set at
  # the arm's tolerances, gated on the push (targets.make_l_stance). The rest
  # truncation dwells in the SAME gated set (rest_lin_tol / rest_ang_tol /
  # rest_gate_on_push, which env_overrides can still override), and
  # end_criterion stays "failure": a reach-avoid success termination is a
  # non-time_out termination, which the force-scale curriculum,
  # RecoverCurriculum and the safety counters would all count as a fall.
  for task_id, (lin_tol, ang_tol) in LAFAN_RA_TOLS.items():
    register(TaskSpec(
      task_id=task_id,
      cfg_builder=functools.partial(
        g1_recover_env_cfg, rest_terminate=True, rest_lin_tol=lin_tol,
        rest_ang_tol=ang_tol, rest_gate_on_push=True),
      margin_fn=compose(g_fall, make_l_stance(lin_tol, ang_tol,
                                              gate_on_push=True)),
      mode=REACH_AVOID, end_criterion="failure",
      ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
      kwargs=lafan_kwargs(),
      description=f"Reach the stance set (|v| <= {lin_tol:g} m/s, |w| <= "
                  f"{ang_tol:g} rad/s per axis) after one push without falling "
                  f"first, from mocap / handover start states; episodes end "
                  f"after dwelling in it (reach-avoid, single-player)."))
  register(TaskSpec(
    task_id="g1_recover_arm_random", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=G1_ARM_DIM, supports_adversary=False,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS),
    description="g1_recover with legs + waist only while the arms swing at "
                "random from the restored arm pose (avoid-only)."))
  # The random arms keep their joint slice (dstb_joint_slice, dstb_gain,
  # joint_noise); dstb_mode / dstb_dim / adversary_body describe the LEARNED
  # player, a 3-dim torso wrench as on g1_stabilize.
  register(TaskSpec(
    task_id="g1_recover_arm_random_wrench", cfg_builder=g1_recover_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=True,
    kwargs=dict(G1_ARM_RANDOM_BRIDGE_KWARGS, dstb_mode="wrench",
                adversary_body=G1_BRIDGE_KWARGS["adversary_body"]),
    description="g1_recover_arm_random plus an adversarial torso wrench "
                "(--adversary), ramped by a force curriculum (avoid-only)."))
