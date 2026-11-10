"""Unit tests for the motor-temperature G1 tasks (envs/g1_recover/thermal.py,
envs/assets/g1_sysid.py) -- simulator-free. What they pin:

  * the sysID repo is the pinned v0.4.0 model (hashes) and its heat model loads;
  * the start-temperature draw (the user's spec): per robot a level
    T0 ~ U(30, 80), windings in [30, T0], housing 0-20 C under the winding and
    >= 30, structure = housing;
  * the state maps mjlab's actuator / joint order to Unitree motor order by
    name, and one step equals g1_sysid.MotorThermal's on motor-ordered inputs;
  * the margin (85 C - hottest winding) / 10 C, min-composed with
    g_fall_noheight, and the overheat termination at 85 C;
  * the task registrations and their observation layout (actor + winding
    temps, critic + winding + housing temps), the dcmotor robot, the guards;
  * the CPU rollout's heat step equals the env's.
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover import thermal as T  # noqa: E402


def _sysid_or_skip():
  from robot_safety_sandbox.envs.assets import g1_sysid
  if not g1_sysid.SYSID_DIR.is_dir():
    pytest.skip(f"no G1 sysID repo at {g1_sysid.SYSID_DIR}")
  return g1_sysid


# --- a stub env: a robot with mjlab-ordered actuators and joints ----------------

class _Robot:
  def __init__(self, names_motor_order, n_envs, gen):
    # mjlab orders actuators by motor type and joints by tree: use two other orders
    self.actuator_names = tuple(n + "_joint" for n in sorted(names_motor_order))
    self.joint_names = tuple(n + "_joint" for n in reversed(names_motor_order))
    k = len(names_motor_order)
    self.data = SimpleNamespace(
      actuator_force=20.0 * torch.randn(n_envs, k, generator=gen, dtype=torch.float64).float(),
      joint_vel=2.0 * torch.randn(n_envs, k, generator=gen, dtype=torch.float64).float())


class _Env:
  def __init__(self, n_envs=8, seed=0):
    g1_sysid = _sysid_or_skip()
    names = g1_sysid.motor_thermal_cls().from_json().names
    self.gen = torch.Generator().manual_seed(seed)
    self.scene = {"robot": _Robot(names, n_envs, self.gen)}
    self.device = "cpu"
    self.num_envs = n_envs
    self.physics_dt = 0.005


def test_sysid_repo_is_the_pinned_model():
  g1_sysid = _sysid_or_skip()
  prov = g1_sysid.provenance()
  assert prov["version"] == g1_sysid.PINNED_VERSION == "0.4.0"
  th = g1_sysid.motor_thermal_cls().from_json(sim_heat_scale=True)
  assert len(th.names) == 29 and th.names[0] == "left_hip_pitch"
  p = g1_sysid.sysid_params()
  assert set(p.MOTOR_OF_JOINT) == {n + "_joint" for n in th.names}


def test_start_draw_follows_the_spec():
  torch.manual_seed(0)
  st = T.thermal_state(_Env(n_envs=4), T.ThermalCfg())
  Tw, Th, Ts = st.draw(20000)
  hot = Tw.amax(1)
  assert float(Tw.min()) >= 30.0 and float(hot.max()) <= 80.0
  assert bool((Th <= Tw).all()) and float(Th.min()) >= 30.0
  assert float((Tw - Th).max()) <= 20.0 + 1e-5
  assert torch.equal(Ts, Th)
  # the hottest winding is spread evenly over 30-80 (the level), not piled at 80
  q = torch.quantile(hot, torch.tensor([0.1, 0.5, 0.9]))
  assert 30.0 < float(q[0]) < 40.0 and 50.0 < float(q[1]) < 60.0 and 70.0 < float(q[2]) < 80.0
  # ... while most motors sit well below the hottest one
  assert float((hot - Tw.median(1).values).mean()) > 8.0
  # one level per robot: a cold robot has every motor cold
  cold = hot < 35.0
  assert cold.any() and float(Tw[cold].max()) < 35.0


def test_reset_draws_only_the_given_envs():
  env = _Env(n_envs=6)
  st = T.thermal_state(env, T.ThermalCfg())
  before = [x.clone() for x in st.T]
  st.reset(torch.tensor([1, 4]))
  for b, x in zip(before, st.T):
    keep = torch.tensor([0, 2, 3, 5])
    assert torch.equal(b[keep], x[keep])
    assert not torch.equal(b[torch.tensor([1, 4])], x[torch.tensor([1, 4])])


def test_one_cfg_per_env():
  env = _Env()
  T.thermal_state(env, T.ThermalCfg())
  assert T.thermal_state(env, T.ThermalCfg()) is T.thermal_state(env)
  with pytest.raises(ValueError):
    T.thermal_state(env, T.ThermalCfg(start_level_c=(30.0, 70.0)))
  with pytest.raises(RuntimeError):
    T.thermal_state(SimpleNamespace())


def test_step_maps_by_name_and_matches_the_sysid_model():
  g1_sysid = _sysid_or_skip()
  env = _Env(n_envs=5)
  st = T.thermal_state(env, T.ThermalCfg())
  T0 = tuple(x.clone().double() for x in st.T)
  st.step(0.005)
  robot = env.scene["robot"]
  # the same inputs, rearranged into motor order by name, through the sysID model
  ref = g1_sysid.motor_thermal_cls().from_json(like=torch.zeros(1, dtype=torch.float64),
                                              sim_heat_scale=True)
  a_idx = [robot.actuator_names.index(n + "_joint") for n in ref.names]
  j_idx = [robot.joint_names.index(n + "_joint") for n in ref.names]
  tau = ref.motor_torque(robot.data.actuator_force[:, a_idx].double())
  w = ref.motor_speed(robot.data.joint_vel[:, j_idx].double())
  want = ref.step(T0, tau, w, 0.005)
  for got, exp in zip(st.T, want):
    assert torch.allclose(got.double(), exp, atol=1e-4)
  # a loaded, still motor heats; the winding moves, the structure barely
  assert float((st.Tw - T0[0].float()).abs().max()) > 0.0


def test_integrator_term_steps_at_physics_dt_and_reports_the_hottest():
  env = _Env(n_envs=3)
  tc = T.ThermalCfg()
  term = T.ThermalIntegrator(SimpleNamespace(params={"thermal_cfg": tc}), env)
  st = T.thermal_state(env)
  w0 = st.Tw.clone()
  out = term(env, thermal_cfg=tc)
  assert out.shape == (3,) and torch.equal(out, st.Tw.amax(1))
  assert not torch.equal(w0, st.Tw)
  term.reset(torch.tensor([0]))
  assert float(st.Tw[0].max()) <= 80.0


def _env_with_temps(hottest, tilt_cos=1.0):
  env = _Env(n_envs=len(hottest))
  st = T.thermal_state(env, T.ThermalCfg())
  Tw = torch.full((len(hottest), 29), 40.0)
  Tw[:, 7] = torch.tensor(hottest, dtype=torch.float32)
  st.set(None, Tw, Tw - 5.0, Tw - 5.0)
  grav = torch.tensor([[0.0, 0.0, -float(tilt_cos)]] * len(hottest))
  env.scene["robot"].data.projected_gravity_b = grav
  return env


def test_thermal_margin_and_termination():
  env = _env_with_temps([60.0, 75.0, 84.9, 85.0, 90.0])
  g = T.g_thermal(env)
  assert torch.allclose(g, torch.tensor([2.5, 1.0, 0.01, 0.0, -0.5]), atol=1e-4)
  assert T.motor_overheat(env).tolist() == [False, False, False, True, True]
  # upright and no contact sensor: the fall terms read 1, so the thermal term binds
  gm = T.g_fall_noheight_thermal(env)
  assert torch.allclose(gm, torch.minimum(torch.ones(5), g), atol=1e-5)
  # a cold robot tilted past 50 deg still fails on the tilt
  tilted = _env_with_temps([40.0], tilt_cos=math.cos(math.radians(60.0)))
  assert float(T.g_fall_noheight_thermal(tilted)) < 0.0
  obs_w = T.motor_winding_temp(env)
  obs_w[:] = 0.0                                       # a copy, not the state
  assert float(T.thermal_state(env).Tw.max()) == 90.0


def test_thermal_tasks_register_their_margin_robot_and_terms():
  _sysid_or_skip()
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from mjlab.actuator import BuiltinDcMotorActuatorCfg
  from robot_safety_sandbox.registry import AVOID, spec
  for tid, parent in (("g1_recover_thermal", "g1_recover_noheight"),
                      ("g1_recover_lafan_thermal", "g1_recover_lafan_noheight")):
    s, p = spec(tid), spec(parent)
    assert s.mode == AVOID and s.margin_fn.avoid_margin is T.g_fall_noheight_thermal, tid
    assert s.margin_fn.reach_margin is None
    assert (s.ctrl_dim, s.dstb_dim, s.kwargs) == (p.ctrl_dim, p.dstb_dim, p.kwargs), tid
  cfg = spec("g1_recover_thermal").cfg_builder()
  base = spec("g1_recover_noheight").cfg_builder()
  actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
  assert list(actor) == list(base.observations["actor"].terms) + ["motor_temp"]
  assert list(critic) == list(base.observations["critic"].terms) + ["motor_temp",
                                                                    "motor_temp_housing"]
  assert (actor["motor_temp"].noise.n_min, actor["motor_temp"].noise.n_max) == (-0.5, 0.5)
  assert critic["motor_temp"].noise is None and critic["motor_temp_housing"].noise is None
  assert actor["motor_temp"] is not critic["motor_temp"]
  assert set(cfg.terminations) == set(base.terminations) | {"motor_overheat"}
  assert cfg.terminations["motor_overheat"].params["limit_c"] == T.LIMIT_C == 85.0
  m = cfg.metrics["motor_temp_max"]
  assert m.per_substep and m.reduce == "max" and m.func is T.ThermalIntegrator
  acts = cfg.scene.entities["robot"].articulation.actuators
  assert acts and all(isinstance(a, BuiltinDcMotorActuatorCfg) for a in acts)
  tcs = {t.params["thermal_cfg"] for t in (actor["motor_temp"], critic["motor_temp"],
                                           critic["motor_temp_housing"])}
  assert tcs == {T.ThermalCfg()}
  # env_overrides reach the draw; the height stays out of the critic group
  c2 = spec("g1_recover_thermal").cfg_builder(thermal_start_level_c=[30, 60])
  assert c2.metrics["motor_temp_max"].params["thermal_cfg"].start_level_c == (30.0, 60.0)
  with pytest.raises(ValueError):
    spec("g1_recover_thermal").cfg_builder(critic_pelvis_height=True)
  # the harvest twin: the identified robot, g1_recover_lafan's margin, no temperatures
  sy = spec("g1_recover_lafan_sysid")
  assert sy.margin_fn.avoid_margin is spec("g1_recover_lafan").margin_fn.avoid_margin
  sc = sy.cfg_builder()
  assert "motor_temp" not in sc.observations["actor"].terms
  assert "motor_overheat" not in sc.terminations
  assert all(isinstance(a, BuiltinDcMotorActuatorCfg)
             for a in sc.scene.entities["robot"].articulation.actuators)


def test_sonic_hip_swap_refuses_the_identified_robot():
  _sysid_or_skip()
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.envs.g1_recover.sonic import use_sonic_hip_pitch
  from robot_safety_sandbox.registry import spec
  with pytest.raises(ValueError, match="dcmotor"):
    use_sonic_hip_pitch(spec("g1_recover_lafan_thermal").cfg_builder())


def test_cpu_rollout_heat_step_matches_the_env():
  """cpu_rollout.Roller._heat (one substep, numpy, from MjData's
  actuator_force / qvel through the spec's motor -> actuator / dof maps)
  equals MotorThermalState.step on the same forces and speeds."""
  g1_sysid = _sysid_or_skip()
  sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))
  import cpu_rollout as cr
  import dataclasses
  env = _Env(n_envs=1)
  st = T.thermal_state(env, T.ThermalCfg())
  robot = env.scene["robot"]
  T0 = tuple(x[0].double().numpy().copy() for x in st.T)
  # MjData: actuators / dofs in the robot's (mjlab) order, behind a 6-dof free joint
  d = SimpleNamespace(actuator_force=robot.data.actuator_force[0].double().numpy(),
                      qvel=np.concatenate([np.zeros(6), robot.data.joint_vel[0].double().numpy()]))
  th_np = g1_sysid.motor_thermal_cls().from_json(sim_heat_scale=True)
  r = object.__new__(cr.Roller)
  r.th = g1_sysid.motor_thermal_cls()(**{f.name: getattr(th_np, f.name)
                                         for f in dataclasses.fields(th_np)})
  r.s = SimpleNamespace(thermal=dict(act=st.act_perm.numpy(), vadr=6 + st.jnt_perm.numpy(),
                                     dt=0.005))
  r.temps = {(0, 0): T0}
  r._heat(d, (0, 0))
  st.step(0.005)
  for got, exp in zip(r.temps[(0, 0)], st.T):
    assert np.allclose(got, exp[0].double().numpy(), atol=1e-4)
