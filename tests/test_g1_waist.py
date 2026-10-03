"""Tests for G1 reach-avoid-stay to a waist + velocity goal (envs/g1_waist,
tasks/g1_waist.py, filters/switch.py).

CPU, simulator-free (stubs for the mjlab env):
  * the goal margin l_box -- normalized per term, zero at each tolerance, and
    blind to the velocity components that are not part of the goal;
  * the stay margins are avoid-only on min(g, soft_floor(l_box)), a failure
    termination anchored at -CLAMP; the reach margins take
    min(l_box, V_stay) and refuse to run without a frozen stay value;
  * the waist goal command -- sampling range = scale * max_angle, and a goal a
    reset event drew early survives the command reset;
  * the reach-avoid-stay switch -- latch, hysteresis, reset, which phase acts;
  * its metric, the goal-magnitude ramp, and the two registrations.

GPU (skipped without CUDA): the real envs -- the noise-free observation rebuild
matches the observation manager, the stay spawn lands on the goal with the legs in a
random stance (both feet flat on the floor), the actor
observation layout, and the reach margin read through a real frozen twin.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import CLAMP  # noqa: E402
from robot_safety_sandbox.envs.g1_waist.command import (  # noqa: E402
  MAX_ANGLE, WaistTargetCommand, WaistTargetCommandCfg)
from robot_safety_sandbox.envs.g1_waist.env_cfg import (  # noqa: E402
  L_FLOOR, Q_TOL, V_TOL, reach_margins, soft_floor, stay_margins,
  target_margin)

# --- stubs ---------------------------------------------------------------------

_WAIST_IDS = (13, 14)


class _Robot:
  def __init__(self, n, waist=(0.0, 0.0), lin=(0.0, 0.0, 0.0),
               ang=(0.0, 0.0, 0.0), pelvis_z=0.76):
    q = torch.zeros(n, 29)
    q[:, list(_WAIST_IDS)] = torch.tensor(waist)
    self.data = SimpleNamespace(
      joint_pos=q,
      root_link_pos_w=torch.tensor([[0.0, 0.0, pelvis_z]]).expand(n, 3),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3),
      root_link_lin_vel_b=torch.tensor([list(lin)]).expand(n, 3),
      root_link_ang_vel_b=torch.tensor([list(ang)]).expand(n, 3))

  def find_joints(self, names, preserve_order=False):
    assert tuple(names) == ("waist_roll_joint", "waist_pitch_joint")
    return list(_WAIST_IDS), list(names)


class _GoalTerm:
  def __init__(self, n, goal):
    self.joint_ids = torch.tensor(_WAIST_IDS)
    self.command = torch.tensor([list(goal)]).expand(n, 2)


class _Commands:
  def __init__(self, n, goal, twist):
    self._goal = _GoalTerm(n, goal)
    self._twist = torch.tensor([list(twist)]).expand(n, 3)

  def get_term(self, name):
    assert name == "waist_target"
    return self._goal

  def get_command(self, name):
    assert name == "twist"
    return self._twist


class _Env:
  """The slice of the mjlab env the g1_waist margins read (no contact sensor:
  the fall margin then uses height and tilt only)."""

  def __init__(self, *, goal=(0.2, -0.1), waist=None, lin=(0.0, 0.0, 0.0),
               ang=(0.0, 0.0, 0.0), twist=(0.0, 0.0, 0.0), pelvis_z=0.76,
               n=1):
    waist = goal if waist is None else waist
    self.num_envs, self.device = n, "cpu"
    self.scene = {"robot": _Robot(n, waist, lin, ang, pelvis_z)}
    self.command_manager = _Commands(n, goal, twist)


# --- the goal margin -----------------------------------------------------------

def test_goal_margin_is_one_on_the_goal_and_zero_at_each_tolerance():
  assert float(target_margin(_Env())) == pytest.approx(1.0)
  g = (0.2, -0.1)
  for waist in ((0.2 + Q_TOL, -0.1), (0.2, -0.1 - Q_TOL)):
    assert float(target_margin(_Env(goal=g, waist=waist))) == pytest.approx(
      0.0, abs=1e-5)
  for kw in (dict(lin=(V_TOL, 0.0, 0.0)), dict(lin=(0.0, -V_TOL, 0.0)),
             dict(ang=(0.0, 0.0, V_TOL))):
    assert float(target_margin(_Env(**kw))) == pytest.approx(0.0, abs=1e-5)
  # past a tolerance: negative, in units of that tolerance
  assert float(target_margin(_Env(waist=(0.2 + 2 * Q_TOL, -0.1)))) == \
    pytest.approx(-1.0)


def test_goal_margin_ignores_the_velocity_components_outside_the_goal():
  # vz, wx, wy are not part of the goal (the twist command cannot set them)
  e = _Env(lin=(0.0, 0.0, 5.0), ang=(5.0, 5.0, 0.0))
  assert float(target_margin(e)) == pytest.approx(1.0)


def test_goal_margin_measures_velocity_against_the_twist_command():
  e = _Env(lin=(0.5, 0.0, 0.0), twist=(0.5, 0.0, 0.0))
  assert float(target_margin(e)) == pytest.approx(1.0)


# --- stay and reach margins ----------------------------------------------------

def test_stay_is_avoid_only_on_safe_and_in_goal():
  assert getattr(stay_margins, "has_target", True) is False
  h, l = stay_margins(_Env())
  assert float(l) == 0.0 and float(h) > 0.0
  # upright but out of the goal set -> the stay constraint is violated
  h, _ = stay_margins(_Env(waist=(0.2 + 2 * Q_TOL, -0.1)))
  assert float(h) < 0.0
  # on the goal but fallen (pelvis under 0.45 m) -> violated too
  h, _ = stay_margins(_Env(pelvis_z=0.3))
  assert float(h) < 0.0
  # a box violation, however large, stays above the soft floor ...
  h, _ = stay_margins(_Env(waist=(0.2 + 50 * Q_TOL, -0.1)))
  assert -L_FLOOR <= float(h) < -0.99 * L_FLOOR
  # ... while a fall still reaches the clamp
  h, _ = stay_margins(_Env(pelvis_z=-1.0))
  assert float(h) == -CLAMP


def test_soft_floor_is_identity_inside_and_saturates_outside():
  l = torch.tensor([2.0, 1.0, 0.3, 0.0])
  assert torch.equal(soft_floor(l), l)
  # matches l to first order just past the edge, bends toward -L_FLOOR beyond
  assert float(soft_floor(torch.tensor([-1e-3]))) == pytest.approx(-1e-3, rel=1e-5)
  assert float(soft_floor(torch.tensor([-L_FLOOR]))) == pytest.approx(
    L_FLOOR * math.tanh(-1.0))
  # (float32 tanh reaches exactly -L_FLOOR by l = -3: bounded, not strictly)
  far = soft_floor(torch.tensor([-1.0, -3.0, -100.0]))
  assert bool((far >= -L_FLOOR).all()) and bool((far[1:] < -0.99 * L_FLOOR).all())
  assert bool((far[:-1] >= far[1:]).all())                # still monotone


def test_stay_failure_termination_is_anchored_at_the_clamp():
  # the hook values a failure termination at the margin_fn's terminal_margin
  # (-CLAMP for the stay margins), not the zoo-wide TERMINAL_MARGIN
  from robot_safety_sandbox.base import TERMINAL_MARGIN, safety_margin_hook
  assert stay_margins.terminal_margin == -CLAMP

  def run(margin_fn, terminated):
    e = _Env(n=2, pelvis_z=0.76)   # upright, on the goal: h > 0
    e.common_step_counter, e.extras = 1, {}
    e.termination_manager = SimpleNamespace(
      terminated=torch.tensor(terminated), active_terms=[])
    return safety_margin_hook(e, margin_fn=margin_fn)

  h = run(stay_margins, [True, False])
  assert float(h[0]) == -CLAMP and float(h[1]) > 0.0
  # a margin_fn without the attribute keeps the zoo-wide anchor
  def plain(env):
    return stay_margins(env)
  h = run(plain, [True, False])
  assert float(h[0]) == pytest.approx(TERMINAL_MARGIN) and float(h[1]) > 0.0


class _StayValue:
  def __init__(self, v):
    self.v = torch.tensor([float(v)])

  def value(self, env):
    return self.v


def test_reach_target_is_the_stay_certificate_inside_the_goal_box():
  e = _Env()                                   # l_box = 1 on the goal
  e._zoo_stay_value = _StayValue(0.4)
  g, l = reach_margins(e)
  assert float(l) == pytest.approx(0.4) and float(g) > 0.0
  # a V_stay that overestimates off the goal cannot declare arrival there
  e = _Env(waist=(0.2 + 2 * Q_TOL, -0.1))       # l_box = -1
  e._zoo_stay_value = _StayValue(2.5)
  assert float(reach_margins(e)[1]) == pytest.approx(-1.0)


def test_reach_margin_without_a_stay_value_fails_loudly():
  with pytest.raises(RuntimeError, match="stay value"):
    reach_margins(_Env())


# --- the goal command ----------------------------------------------------------

class _CmdEnv:
  def __init__(self, n):
    self.num_envs, self.device = n, "cpu"
    self.scene = {"robot": _Robot(n)}
    self.max_episode_length = 1000


def _command(n=4000):
  return WaistTargetCommand(WaistTargetCommandCfg(), _CmdEnv(n))


def test_goal_range_is_scale_times_max_angle():
  term = _command()
  ids = torch.arange(4000)
  for scale in (1.0, 0.25, 0.0):
    term.scale = scale
    term.sample(ids)
    half = scale * MAX_ANGLE
    assert float(term.command.abs().max()) <= half + 1e-6
    if scale > 0:     # and it fills the box
      assert float(term.command.abs().max()) > 0.95 * half
  assert term.ramped is True and term.cfg.resampling_time_range[0] >= 1e8


def test_a_goal_drawn_by_the_spawn_survives_the_command_reset():
  term = _command(8)
  early = torch.tensor([1, 5])
  term.sample(early)
  kept = term.command[early].clone()
  term.reset(torch.arange(8))              # the command manager's reset
  assert torch.equal(term.command[early], kept)
  # ...and only that once: the next reset draws fresh goals for them too
  term.reset(torch.arange(8))
  assert not torch.equal(term.command[early], kept)


# --- the reach-avoid-stay switch -------------------------------------------------

def _switch(v_stay, eps_in=0.1, eps_out=0.0):
  """Reach fallback returns 1, stay fallback returns 2, the nominal 0; both
  monitors certify everything, so each phase passes the nominal -- except the
  reach monitor here, which never certifies, so REACH always acts with 1."""
  from robot_safety_sandbox.filters import (
    ReachAvoidStayFilter, safety_value_filter)
  n = v_stay.shape[0]
  reach = safety_value_filter(n, "cpu", lambda **_: torch.full((n,), -1.0),
                              lambda **_: torch.ones(n, 1))
  stay = safety_value_filter(n, "cpu", lambda **_: torch.full((n,), 1.0),
                             lambda **_: torch.full((n, 1), 2.0))
  box = {"v": v_stay}
  f = ReachAvoidStayFilter(reach, stay, lambda **_: box["v"],
                           eps_in=eps_in, eps_out=eps_out)
  return f, box


def test_switch_latches_with_hysteresis_and_resets_to_reach():
  f, box = _switch(torch.tensor([0.05, 0.2]))
  a0 = torch.zeros(2, 1)
  a, info = f(a0)
  assert f.mode.tolist() == [False, True]          # 0.05 <= eps_in < 0.2
  assert a[:, 0].tolist() == [1.0, 0.0]            # REACH fallback | STAY nominal
  assert f.entered.tolist() == [False, True]
  box["v"] = torch.tensor([0.05, 0.05])            # inside the hysteresis band
  f(a0)
  assert f.mode.tolist() == [False, True]          # STAY holds above eps_out
  box["v"] = torch.tensor([0.05, -0.01])
  f(a0)
  assert f.mode.tolist() == [False, False] and f.left.tolist() == [False, True]
  box["v"] = torch.tensor([0.5, 0.5])
  assert f.peek().tolist() == [True, True]         # peek does not commit
  assert f.mode.tolist() == [False, False]
  f(a0)
  f.reset(torch.tensor([True, False]))
  assert f.mode.tolist() == [False, True]


def test_switch_refuses_an_exit_threshold_above_the_entry_one():
  with pytest.raises(ValueError, match="eps_out"):
    _switch(torch.zeros(1), eps_in=0.0, eps_out=0.1)


# --- the metric ------------------------------------------------------------------

def test_ras_stats_count_reach_time_hold_and_leaves():
  from robot_safety_sandbox.eval.metrics import ReachAvoidStayStats, StepRecord
  f, box = _switch(torch.tensor([0.0, 0.0]))
  held = {"h": torch.tensor([1.0, -1.0])}
  m = ReachAvoidStayStats(f, 2, "cpu", dt=0.02,
                          constraint_fn=lambda mj: held["h"])
  env = SimpleNamespace(mj=None)
  a0 = torch.zeros(2, 1)

  def step(v, done=(False, False)):
    box["v"] = torch.tensor(v)
    f(a0)
    d = torch.tensor(done)
    m.update(StepRecord(t=0, env=env, a_nom=a0, a_filt=a0, info=None,
                        out=SimpleNamespace(done=d)))
    f.reset(d)

  step([0.0, 0.0])            # t=0: both REACH
  step([0.5, 0.0])            # t=1: env 0 enters STAY
  step([0.5, 0.5])            # t=2: env 1 enters STAY
  step([-1.0, 0.5], done=(False, True))   # env 0 hands back; env 1 ends
  r = m.result()
  assert r["ras_reached_rate"] == 1.0              # 1 ended + 1 censored, both reached
  assert r["ras_time_to_stay_s"] == pytest.approx(1.5 * 0.02)
  assert r["ras_leaves_per_reach"] == pytest.approx(0.5)
  assert r["ras_stay_frac"] == pytest.approx(4 / 8)
  # STAY steps not on an ending env: t1 env0 (h>=0), t2 env0 (h>=0), t2 env1
  # (h<0); t3 env1 is STAY but ends that step, so it is left out
  assert r["ras_hold_rate"] == pytest.approx(2 / 3)


# --- the goal-magnitude ramp -----------------------------------------------------

def test_target_scale_ramp_is_linear_to_the_full_range():
  from robot_safety_sandbox.callbacks import TargetScaleRampCallback
  term = SimpleNamespace(scale=1.0)
  mj = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: term))
  cb = TargetScaleRampCallback("waist_target", ramp_steps=100, scale_start=0.2)
  cb.model = SimpleNamespace(env=SimpleNamespace(mj=mj),
                             logger=SimpleNamespace(record=lambda *a, **k: None))
  cb._on_training_start()
  assert term.scale == pytest.approx(0.2)
  for t, want in ((50, 0.6), (100, 1.0), (500, 1.0)):
    cb.num_timesteps = t
    cb._on_step()
    assert term.scale == pytest.approx(want)


# --- registration ----------------------------------------------------------------

def test_waist_tasks_register_as_stay_avoid_and_reach_reach_avoid():
  from robot_safety_sandbox import algo_name, spec
  from robot_safety_sandbox.tasks.g1_arm_random import G1_ARM_RANDOM_BRIDGE_KWARGS
  stay, reach = spec("g1_waist_stay_arm_random"), spec("g1_waist_reach_arm_random")
  assert algo_name("g1_waist_stay_arm_random") == "SafetyPPO1P"
  assert algo_name("g1_waist_reach_arm_random") == "ReachAvoidPPO1P"
  for s in (stay, reach):
    assert s.ctrl_dim == 29 and s.dstb_dim == 14 and not s.supports_adversary
    assert s.kwargs["dstb_gain"] == 1.0                     # half the stands'
    assert s.kwargs["dstb_joint_slice"] == (15, 29)
    assert s.kwargs["joint_noise"] == G1_ARM_RANDOM_BRIDGE_KWARGS["joint_noise"]
    assert s.end_criterion == "failure"


# --- GPU: the real envs ----------------------------------------------------------

_CUDA = torch.cuda.is_available()
_gpu = pytest.mark.skipif(not _CUDA, reason="needs CUDA (mjlab/mujoco_warp)")


def _bridge(task, n, **cfg_kw):
  """The task's bridge with actor observation noise OFF, so the observation
  manager's output and the noise-free rebuild can be compared exactly."""
  from robot_safety_sandbox import spec
  from robot_safety_sandbox.base import MjlabTensorSafetyEnv
  s = spec(task)

  def cfg_builder(play=False, **kw):
    cfg = s.cfg_builder(play=play, **{**cfg_kw, **kw})
    cfg.observations["actor"].enable_corruption = False
    return cfg
  return MjlabTensorSafetyEnv(n, "cuda:0", cfg_builder=cfg_builder,
                              margin_fn=s.margin_fn, ctrl_dim=s.ctrl_dim,
                              dstb_dim=s.dstb_dim, **s.kwargs)


@_gpu
def test_obs_rebuild_matches_the_observation_manager_and_the_layout():
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_waist.stay_value import current_obs
  base = make_tensor("g1_stabilize_arm_random", 4, "cuda:0")
  base_dim = int(base.observation_space.shape[0])
  base.close()
  env = _bridge("g1_waist_stay_arm_random", 32)
  try:
    assert env._track_reach is False       # avoid task: no reach counters
    obs = env.reset()
    assert obs.shape[1] == base_dim + 5          # + base_lin_vel (3) + goal (2)
    goal = env.mj.command_manager.get_command("waist_target")
    assert torch.equal(obs[:, -2:], goal)        # the goal is the last 2 columns
    assert torch.equal(current_obs(env.mj), obs)
    a = 0.3 * torch.randn(32, 29, device="cuda:0")
    for _ in range(40):                          # through falls and auto-resets
      obs, *_ = env.step_tensor(a)
      assert torch.equal(current_obs(env.mj), obs)
  finally:
    env.close()


@_gpu
def test_stay_spawn_puts_the_waist_on_this_episodes_goal():
  from robot_safety_sandbox import make_tensor
  env = make_tensor("g1_waist_stay_arm_random", 64, "cuda:0",
                    cfg_overrides={"spawn_waist_noise": 0.05})
  try:
    env.reset()
    mj = env.mj
    term = mj.command_manager.get_term("waist_target")
    q = mj.scene["robot"].data.joint_pos[:, term.joint_ids]
    assert float((q - term.command).abs().max()) <= 0.05 + 1e-5
    assert float(term.command.abs().max()) > 0.1   # full range outside training
  finally:
    env.close()


def _sole_bottoms(mj):
  """(N, 14) lowest point of each G1 sole capsule, after a reset's forward."""
  from mjlab.utils.lab_api.math import quat_apply
  from robot_safety_sandbox.envs.g1_waist.stance import SOLE_GEOMS
  robot = mj.scene["robot"]
  local = torch.as_tensor(robot.find_geoms(SOLE_GEOMS, preserve_order=True)[0],
                          device=mj.device)
  size = torch.as_tensor(
    mj.sim.mj_model.geom_size[robot.indexing.geom_ids[local].cpu().numpy()],
    dtype=torch.float32, device=mj.device)
  pos, quat = robot.data.geom_pos_w[:, local], robot.data.geom_quat_w[:, local]
  axis_z = quat_apply(quat.reshape(-1, 4), torch.tensor(
    [0.0, 0.0, 1.0], device=mj.device).expand(quat.shape[0] * 14, 3))[:, 2]
  return (pos[..., 2] - (size[:, 1] * axis_z.reshape(-1, 14)).abs()
          - size[:, 0])


@_gpu
def test_stay_spawn_stance_is_varied_with_both_feet_flat_on_the_floor():
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_waist.stance import CLEARANCE, LEG_JOINTS
  env = make_tensor("g1_waist_stay_arm_random", 256, "cuda:0",
                    cfg_overrides={"spawn_stance_scale": 1.0})
  try:
    env.reset()
    mj = env.mj
    d = mj.scene["robot"].data
    b = _sole_bottoms(mj)
    # every sole capsule of both feet at the spawn clearance: flat, touching
    assert float((b - CLEARANCE).abs().max()) < 1.5e-3
    ids, _ = mj.scene["robot"].find_joints(LEG_JOINTS, preserve_order=True)
    q, lim = d.joint_pos[:, ids], d.soft_joint_pos_limits[:, ids]
    assert bool(((q >= lim[..., 0] - 1e-4) & (q <= lim[..., 1] + 1e-4)).all())
    assert float(q[:, LEG_JOINTS.index("left_knee_joint")].std()) > 0.1
    assert float(d.root_link_pos_w[:, 2].std()) > 0.02
  finally:
    env.close()


@_gpu
def test_stay_spawn_stance_scale_zero_is_the_stock_pose():
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_waist.stance import LEG_JOINTS
  env = make_tensor("g1_waist_stay_arm_random", 16, "cuda:0",
                    cfg_overrides={"spawn_stance_scale": 0.0})
  try:
    assert "reset_leg_stance" not in env.mj.cfg.events
    env.reset()
    d = env.mj.scene["robot"].data
    ids, _ = env.mj.scene["robot"].find_joints(LEG_JOINTS, preserve_order=True)
    assert torch.allclose(d.joint_pos[:, ids], d.default_joint_pos[:, ids])
  finally:
    env.close()


@_gpu
def test_reach_margin_reads_the_frozen_stay_twin():
  pytest.importorskip("safety_sb3")
  from safety_sb3 import SafetyPPO1P
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_waist.env_cfg import target_margin
  from robot_safety_sandbox.envs.g1_waist.stay_value import current_obs
  with tempfile.TemporaryDirectory() as tmp:
    stay_env = make_tensor("g1_waist_stay_arm_random", 4, "cuda:0")
    dim = int(stay_env.observation_space.shape[0])
    SafetyPPO1P("MlpPolicy", stay_env, n_steps=8, batch_size=32,
                device="cuda:0").save(os.path.join(tmp, "final_model.zip"))
    stay_env.close()
    torch.save({"obs_mean": torch.zeros(dim), "obs_var": torch.ones(dim),
                "count": torch.tensor(1.0)},
               os.path.join(tmp, "tensornormalize.pt"))
    # Record what the margin itself read: mjlab computes margins BEFORE its
    # end-of-step sim.forward(), so derived quantities there lag the post-step
    # state by one physics substep (by design, identically for every task) and
    # a recompute after the step would not match.
    seen = {}

    def margin_fn(env):
      from robot_safety_sandbox.envs.g1_waist.stay_value import stay_value
      seen["l_box"], seen["v"] = target_margin(env), stay_value(env)
      return reach_margins(env)

    env = _bridge("g1_waist_reach_arm_random", 16, stay_run=tmp)
    env.margin_fn = margin_fn
    env.mj.reward_manager.get_term_cfg("zoo_safety_hook").params[
      "margin_fn"] = margin_fn
    try:
      assert env._track_reach is True      # reach-avoid task: reach counters on
      env.reset()
      a = 0.2 * torch.randn(16, 29, device="cuda:0")
      *_, l = env.step_tensor(a)
      want = torch.minimum(seen["l_box"], seen["v"]).clamp(-CLAMP, CLAMP)
      assert torch.allclose(l, want)
      # the env's frozen value == the eval harness's own load of the same twin
      from robot_safety_sandbox.eval.policies import load_twin, safety_modules
      model, norm = load_twin(os.path.join(tmp, "final_model.zip"), "cuda:0",
                              quiet=True)
      value_fn = safety_modules(model, 16, "cuda:0")["value_fn"]
      obs = current_obs(env.mj)
      assert obs.shape[1] == dim
      assert torch.allclose(env.mj._zoo_stay_value.value(env.mj),
                            value_fn(s_obs=norm(obs)))
    finally:
      env.close()
