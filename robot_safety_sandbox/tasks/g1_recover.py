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
``g1_recover_lafan_k``    g1_recover_lafan on the ViaLasso pipeline's safe set
                          K (``envs/g1_recover/k_set.py``): its margin g_k
                          (pelvis height, tilt, all 29 joint limits, non-foot
                          contact; g < 0 iff outside K) and its termination
                          (the episode ends when the state leaves K, instead
                          of at 70 deg tilt / a 10 N non-foot contact).
``g1_recover_noheight``, ``g1_recover_lafan_noheight``
                          g1_recover / g1_recover_lafan with NO pelvis height:
                          margin g_fall_noheight (tilt, non-foot contact) and a
                          critic group (a SHAC run's world-model state) without
                          the pelvis_height term. Terminations unchanged (they
                          never read the height). Its 113-D critic group still
                          has foot_height (world-frame foot sites).
``g1_recover_noheight_nofoot``, ``g1_recover_lafan_noheight_nofoot``
                          the *_noheight tasks with no foot terms (height, air
                          time, contact flags / forces) in the critic group
                          either: actor terms + base_lin_vel, 101-D -- the
                          state of a deployable (student) world model.
``g1_recover_noheight_nofoot_nolinvel``, ``g1_recover_lafan_noheight_nofoot_nolinvel``
                          ... and no base_lin_vel either: the critic group is
                          exactly the actor terms (98-D, noise-free).
``g1_recover_thermal``, ``g1_recover_lafan_thermal``
                          the *_noheight tasks on the lab's identified G1
                          (``envs/assets/g1_sysid.py``: dcmotor actuators,
                          measured waist ranges, refitted collision capsules)
                          with its motor heat model in the loop
                          (``envs/g1_recover/thermal.py``): margin min(tilt,
                          non-foot contact, (85 C - hottest winding) / 10 C),
                          a winding at 85 C ends the episode, start temperatures
                          drawn per reset; actor + 29 winding temperatures,
                          critic group + winding and housing temperatures.
``g1_recover_thermal_cbf``, ``g1_recover_lafan_thermal_cbf``
                          the *_thermal tasks with thermal safety moved BELOW the
                          policy: a 70 C temperature barrier clips every motor's
                          torque (whatever drives the robot), temperature is out
                          of the margin (g_fall_noheight) and the terminations,
                          start levels U(30, 70).
``g1_recover_unified``    THE UNIFIED SETUP (Zizhe + Yuanhong, 2026-11-10;
                          ``envs/g1_recover/unified.py``): g1_recover_thermal_cbf
                          with margin min(70 deg - tilt, joint clearance / 0.2)
                          (ranges + 0.1 rad), failure = tilt > 70 deg only (no
                          illegal_contact), critic = actor terms + heading-frame
                          pelvis velocity (101-D) + winding / housing temps.
                          The default for new G1 fallback runs
                          (configs/g1_recover_unified_safety{ppo,shac}1p.yaml).
``g1_recover_lafan_sysid`` g1_recover_lafan on the identified G1, no
                          temperatures: SONIC's start states on that robot.
``g1_recover_bench``     the SONIC push benchmark's world (``envs/g1_recover/
                          push_bench.py``): K as margin (joint term over 0.2
                          rad) and termination, 2 ms x 10 physics, SONIC's
                          hip-pitch motor, foot friction 1, no domain
                          randomization, no sensor noise, the 70-D exact actor
                          observation of his connector, no in-episode push
                          (start states carry the push; SHAC-compatible).
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
  from robot_safety_sandbox.envs.g1_recover.env_cfg import (
    g1_recover_env_cfg, g1_recover_k_env_cfg, g1_recover_noheight_env_cfg,
    g1_recover_noheight_nofoot_env_cfg, g1_recover_noheight_nofoot_nolinvel_env_cfg,
    g1_recover_sysid_env_cfg, g1_recover_thermal_cbf_env_cfg, g1_recover_thermal_env_cfg)
  from robot_safety_sandbox.envs.g1_recover.k_set import CLAMP as K_CLAMP, g_k
  from robot_safety_sandbox.envs.g1_recover.thermal import g_fall_noheight_thermal
  from robot_safety_sandbox.envs.g1_recover.unified import (g1_recover_unified_env_cfg,
                                                            g_tilt_joint)
  from robot_safety_sandbox.envs.g1_recover.targets import make_l_stance
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall, g_fall_noheight
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
  # The NO-PELVIS-HEIGHT ablation: g1_recover / g1_recover_lafan with the pelvis
  # height out of the margin (g_fall_noheight: tilt and non-foot contact) and
  # out of the critic group, i.e. the world-model state of a SHAC run (the actor
  # never had it). Terminations, spawn and push unchanged.
  register(TaskSpec(
    task_id="g1_recover_noheight", cfg_builder=g1_recover_noheight_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_recover with no pelvis height in the margin or the critic "
                "group (avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_lafan_noheight", cfg_builder=g1_recover_noheight_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan with no pelvis height in the margin or the "
                "critic group (avoid-only, single-player)."))
  # ... and no foot terms in the critic group either: a deployable world-model
  # state (actor terms + base_lin_vel, 101-D). The student world models of the
  # no-height SHAC policies (configs/sweeps/shac_student/) train / are scored
  # on these; margin, terminations, spawn and push are the *_noheight tasks'.
  register(TaskSpec(
    task_id="g1_recover_noheight_nofoot", cfg_builder=g1_recover_noheight_nofoot_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_recover_noheight with no foot terms in the critic group "
                "(actor terms + base_lin_vel; avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_lafan_noheight_nofoot",
    cfg_builder=g1_recover_noheight_nofoot_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan_noheight with no foot terms in the critic "
                "group (avoid-only, single-player)."))
  # ... and no base_lin_vel: the world-model state is the actor obs itself
  # (noise-free), 98-D.
  register(TaskSpec(
    task_id="g1_recover_noheight_nofoot_nolinvel",
    cfg_builder=g1_recover_noheight_nofoot_nolinvel_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_recover_noheight_nofoot with no base_lin_vel in the critic "
                "group either (the actor terms, noise-free; avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_lafan_noheight_nofoot_nolinvel",
    cfg_builder=g1_recover_noheight_nofoot_nolinvel_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan_noheight_nofoot with no base_lin_vel in the "
                "critic group either (avoid-only, single-player)."))
  # MOTOR TEMPERATURES on the lab's identified G1 (envs/assets/g1_sysid.py,
  # envs/g1_recover/thermal.py): the *_noheight tasks plus a winding limit.
  # margin g_fall_noheight_thermal = min(tilt, non-foot contact,
  # (85 C - hottest winding) / 10 C); a winding at 85 C ends the episode (the
  # G1 controller goes to damping). Actor obs + 29 winding temps (127-D);
  # critic group / world-model state + winding and housing temps (171-D).
  register(TaskSpec(
    task_id="g1_recover_thermal", cfg_builder=g1_recover_thermal_env_cfg,
    margin_fn=compose(g_fall_noheight_thermal), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_recover_noheight on the identified G1 with its motor heat model: "
                "no fall and every winding below 85 C (avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_lafan_thermal", cfg_builder=g1_recover_thermal_env_cfg,
    margin_fn=compose(g_fall_noheight_thermal), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan_noheight on the identified G1 with its motor heat "
                "model: no fall and every winding below 85 C (avoid-only, single-player)."))
  # THE TEMPERATURE BARRIER below the policy (thermal.ThermalBarrierAction; the
  # user's decoupled design, 2026-11-10): every motor's torque clipped so its
  # winding stays below 70 C, for whatever drives the robot. Temperature leaves
  # the margin (g_fall_noheight, as the *_noheight tasks) and the terminations;
  # start levels U(30, 70). Observations as the *_thermal tasks.
  register(TaskSpec(
    task_id="g1_recover_thermal_cbf", cfg_builder=g1_recover_thermal_cbf_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_recover_noheight on the identified G1 with its motor heat model and "
                "a 70 C temperature barrier clipping every motor's torque (avoid-only)."))
  register(TaskSpec(
    task_id="g1_recover_lafan_thermal_cbf", cfg_builder=g1_recover_thermal_cbf_env_cfg,
    margin_fn=compose(g_fall_noheight), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan_noheight on the identified G1 with its motor heat model "
                "and a 70 C temperature barrier (avoid-only, single-player)."))
  # THE UNIFIED SETUP (envs/g1_recover/unified.py; agreed 2026-11-10): the
  # thermal_cbf world with margin g_tilt_joint = min(70 deg - tilt, joint
  # clearance / 0.2), failure = tilt > 70 deg only, critic = actor terms +
  # heading-frame pelvis velocity + temperatures. No in-episode push (the push
  # lives in the start states), so both SafetyPPO1P and SafetySHAC1P train on it.
  register(TaskSpec(
    task_id="g1_recover_unified", cfg_builder=g1_recover_unified_env_cfg,
    margin_fn=compose(g_tilt_joint), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="The unified G1 fallback setup: the identified G1 with its heat model and "
                "a 70 C barrier; margin min(70 deg - tilt, joint clearance / 0.2), failure "
                "at 70 deg tilt only; critic 101-D + temperatures (avoid-only)."))
  # The identified G1 without temperatures: g1_recover_lafan's world (margin
  # g_fall, its terminations) on the robot the *_thermal tasks use, for
  # harvesting SONIC's start states on it.
  register(TaskSpec(
    task_id="g1_recover_lafan_sysid", cfg_builder=g1_recover_sysid_env_cfg,
    margin_fn=compose(g_fall), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan on the identified G1 (dcmotor actuators, measured "
                "ranges and collision geometry), no temperatures (avoid-only)."))
  # The same task on the ViaLasso safe set K: margin g_k (already clipped and
  # anchored to K, so compose's clamp is a no-op) and the episode ends on
  # leaving K (k_set.use_k_terminations).
  register(TaskSpec(
    task_id="g1_recover_lafan_k", cfg_builder=g1_recover_k_env_cfg,
    margin_fn=compose(g_k, clamp=K_CLAMP), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, dstb_dim=3, supports_adversary=False,
    kwargs=lafan_kwargs(),
    description="g1_recover_lafan with the ViaLasso safe set K as margin and "
                "termination (pelvis >= 0.45 m, tilt <= 1 rad, joints within "
                "range + 0.1 rad, no non-foot contact; avoid-only)."))
  # The SONIC push benchmark's world (envs/g1_recover/push_bench.py). No
  # random_push: the push lives in the start states, so SHAC's world model
  # sees no unexplained force.
  from robot_safety_sandbox.envs.g1_recover.push_bench import bench_env_cfg, g_bench
  register(TaskSpec(
    task_id="g1_recover_bench", cfg_builder=bench_env_cfg,
    margin_fn=compose(g_bench, clamp=K_CLAMP), mode=AVOID,
    ctrl_dim=G1_CTRL_DIM, kwargs=dict(G1_BRIDGE_KWARGS),
    description="Recover from post-push start states without leaving K, in the "
                "SONIC push benchmark's world (2 ms x 10 physics, SONIC hip, foot "
                "friction 1, no randomization, 70-D exact observation; avoid-only)."))
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
