"""The squat-tracking task cfgs against their fallback worlds -- simulator-free.

The task policy trains inside a filter whose fallback (a frozen
g1_recover_thermal70 / g1_recover_thermal_cbf SHAC policy) and world model read
the env's ``actor`` / ``critic`` groups, its startup randomization (the 33-D
context) and its action term. What is pinned, per world:

  * those, the robot, the sim, the thermal metrics, the pinned twist command and
    the fall / overheat terminations are EQUAL to the fallback task's cfg, term
    by term;
  * the bank spawn, its curriculum and any rest truncation are gone; the reward
    is exactly the tracking stack (the velocity stack is cleared);
  * the reference command, the episode length (> the segment), the three new
    observation groups and the stats event;
  * the registrations: cumulative, the fallback's own margin, dense margins, the
    task groups as obs / critic keys, the +-6.5 action box.

Imports the sandbox (minutes on a login node), builds no env.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox import CUMULATIVE, algo_name, spec  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.env_cfg import (  # noqa: E402
  g1_recover_thermal70_env_cfg, g1_recover_thermal_cbf_env_cfg)
from robot_safety_sandbox.envs.g1_track import mdp  # noqa: E402
from robot_safety_sandbox.envs.g1_track.command import TrackRefCommandCfg  # noqa: E402
from robot_safety_sandbox.envs.g1_track.env_cfg import (  # noqa: E402
  EPISODE_LENGTH_S, TRACK_REWARDS, g1_track_squat_thermal70_env_cfg,
  g1_track_squat_thermal_cbf_env_cfg)

WORLDS = [
  ("g1_track_squat_thermal70", "g1_recover_thermal70", g1_track_squat_thermal70_env_cfg,
   g1_recover_thermal70_env_cfg),
  ("g1_track_squat_thermal_cbf", "g1_recover_thermal_cbf", g1_track_squat_thermal_cbf_env_cfg,
   g1_recover_thermal_cbf_env_cfg),
]
IDS = [w[0] for w in WORLDS]
POLICY_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                "motor_temp", "ref_phase", "ref_future"]
CRITIC_EXTRA = ["base_lin_vel", "root_height", "anchor_pos_err", "body_pos", "body_ori",
                "body_pos_err", "motor_temp_housing", "foot_contact"]


def _sysid_or_skip():
  from robot_safety_sandbox.envs.assets import g1_sysid
  if not g1_sysid.SYSID_DIR.is_dir():
    pytest.skip(f"no G1 sysID repo at {g1_sysid.SYSID_DIR}")


@pytest.fixture(scope="module", params=WORLDS, ids=IDS)
def pair(request):
  _sysid_or_skip()
  task, base_task, track, base = request.param
  return task, base_task, track(), base()


def test_fallback_world_is_unchanged(pair):
  _, _, t, b = pair
  for g in ("actor", "critic"):
    tg, bg = t.observations[g], b.observations[g]
    assert list(tg.terms) == list(bg.terms), g
    for name in tg.terms:
      assert tg.terms[name] == bg.terms[name], f"{g}/{name}"
    assert (tg.concatenate_terms, tg.enable_corruption, tg.history_length) == (
      bg.concatenate_terms, bg.enable_corruption, bg.history_length)

  def startup(cfg):
    return {k: v for k, v in cfg.events.items() if v.mode == "startup"}
  assert startup(t) == startup(b) and set(startup(t)) == {"foot_friction", "encoder_bias",
                                                          "base_com"}
  assert t.actions == b.actions
  assert t.sim == b.sim and t.decimation == b.decimation
  assert t.scene.entities == b.scene.entities
  assert t.scene.sensors == b.scene.sensors
  assert t.metrics == b.metrics
  assert t.commands["twist"] == b.commands["twist"]
  for k in b.terminations:
    assert t.terminations[k] == b.terminations[k], k


def test_spawn_curriculum_and_velocity_rewards_are_gone(pair):
  task, _, t, b = pair
  assert "recover_spawn" in b.events and "recover" in b.curriculum
  assert "recover_spawn" not in t.events and "recover" not in t.curriculum
  assert not t.curriculum
  assert set(t.events) == set(b.events) - {"recover_spawn"} | {"track_stats"}
  assert set(t.rewards) == set(TRACK_REWARDS) | {"self_collisions"}
  for name, (func, std, weight) in TRACK_REWARDS.items():
    r = t.rewards[name]
    assert (r.func, r.weight, r.params) == (func, weight, {"command_name": "ref", "std": std})
  assert t.rewards["self_collisions"].weight == -1.0
  # the w_style rule (sum of positive non-zoo weights) gives 5.0
  assert sum(max(r.weight, 0.0) for r in t.rewards.values()) == pytest.approx(5.0)
  thermal = {"motor_overheat"} if task.endswith("thermal70") else set()
  assert set(t.terminations) == {"time_out", "fell_over", "illegal_contact",
                                 "reference_finished"} | thermal
  assert "recovered" not in t.terminations
  rf = t.terminations["reference_finished"]
  assert rf.func is mdp.reference_finished and rf.time_out


def test_reference_command_and_episode_length(pair):
  _, _, t, _ = pair
  c = t.commands["ref"]
  assert isinstance(c, TrackRefCommandCfg)
  assert c.resampling_time_range == (1.0e9, 1.0e9)
  seg = c.segment
  assert (seg.clip, seg.t0, seg.t1, seg.T) == ("fightAndSports1_subject1", 166.0, 179.0, 651)
  assert (c.p_start, c.rsi_min_left_s, c.spawn_noise) == (0.5, 1.0, True)
  assert c.seed_action_bound == 6.5 and c.joint_margin == 0.01
  dt = t.sim.mujoco.timestep * t.decimation
  assert dt == pytest.approx(0.02)
  assert t.episode_length_s == EPISODE_LENGTH_S == 14.0
  assert math.ceil(t.episode_length_s / dt) > seg.T - 1
  ev = t.events["track_stats"]
  assert ev.mode == "step" and ev.func is mdp.TrackEpisodeStats
  assert ev.params == {"command_name": "ref", "overheat_c": 70.0}


def test_task_groups(pair):
  _, _, t, _ = pair
  a = t.observations["actor"].terms
  pol = t.observations["task_policy"]
  assert list(pol.terms) == POLICY_TERMS
  assert pol.enable_corruption and pol.concatenate_terms and pol.history_length == 1
  for k in ("base_ang_vel", "projected_gravity", "joint_vel", "motor_temp", "actions"):
    assert pol.terms[k] == a[k], k
  jp = pol.terms["joint_pos"]
  assert jp.func is a["joint_pos"].func and jp.noise == a["joint_pos"].noise
  assert jp.params == {"biased": True}
  assert pol.terms["ref_future"].params == {"command_name": "ref",
                                            "offsets": (0, 2, 5, 10, 25)}
  crit = t.observations["task_critic"]
  assert list(crit.terms) == POLICY_TERMS + CRITIC_EXTRA
  assert not crit.enable_corruption
  assert all(v.noise is None for v in crit.terms.values())
  assert crit.terms["motor_temp_housing"] == t.observations["critic"].terms["motor_temp_housing"]
  amp = t.observations["amp"]
  assert list(amp.terms) == ["amp"] and amp.terms["amp"].func is mdp.AmpObs
  assert not amp.enable_corruption


def test_registrations():
  _sysid_or_skip()
  for task, base_task, _, _ in WORLDS:
    s, p = spec(task), spec(base_task)
    assert s.mode == CUMULATIVE and algo_name(task, family="on_policy") == "PPO"
    assert s.margin_fn.avoid_margin is p.margin_fn.avoid_margin
    assert not s.margin_fn.has_target
    assert s.ctrl_dim == 29
    assert s.kwargs == {"ctrl_gain": 1.0, "action_bound": 6.5, "adversary_body": "torso_link",
                        "dense_margins": True, "obs_key": "task_policy",
                        "critic_obs_key": "task_critic"}


def test_builder_options():
  _sysid_or_skip()
  with pytest.raises(TypeError):
    g1_track_squat_thermal70_env_cfg(dataset="bank.pt")
  with pytest.raises(TypeError):
    g1_track_squat_thermal_cbf_env_cfg(rest_terminate=True)
  c = g1_track_squat_thermal70_env_cfg(thermal_start_level_c=(30.0, 60.0), t0=170.0, t1=176.0,
                                       p_start=1.0)
  assert c.observations["actor"].terms["motor_temp"].params["thermal_cfg"].start_level_c == (
    30.0, 60.0)
  assert c.commands["ref"].segment.frames == (8500, 8800) and c.commands["ref"].p_start == 1.0
  with pytest.raises(ValueError):
    g1_track_squat_thermal70_env_cfg(t0=100.0, t1=179.0)      # 79 s > the 14 s episode
  play = g1_track_squat_thermal_cbf_env_cfg(play=True)
  assert not play.commands["ref"].spawn_noise
