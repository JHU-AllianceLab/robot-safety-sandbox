"""Unit tests for the unified G1 fallback setup (envs/g1_recover/unified.py,
task g1_recover_unified) -- simulator-free. What they pin, item by item from
the agreement (2026-11-10):

  * the margin min(70 deg - tilt, joint clearance / 0.2), the clearance taken
    inside each range widened by 0.1 rad; compose clamps it to +-3;
  * failure = tilt > 70 deg only (fell_over), no illegal_contact;
  * the critic: the 98 actor terms + the heading-frame pelvis velocity in the
    body-frame velocimeter's slot, + winding / housing temperatures; no foot
    terms, no pelvis height; the actor is thermal_cbf's (98 + 29 temperatures);
  * the identified G1 (dcmotor, 7520-14 on hip pitch), the barrier, 5 ms x 4,
    solver 10 / 20 / CCD 50, the three DR terms, the +-3 action bound;
  * the CPU rollout's tilt_joint margin equals the env's.
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

from robot_safety_sandbox.envs.g1_recover import unified as U  # noqa: E402

ACTOR_98 = ["base_ang_vel", "projected_gravity", "command", "phase", "joint_pos",
            "joint_vel", "actions"]


def _sysid_or_skip():
  from robot_safety_sandbox.envs.assets import g1_sysid
  if not g1_sysid.SYSID_DIR.is_dir():
    pytest.skip(f"no G1 sysID repo at {g1_sysid.SYSID_DIR}")
  return g1_sysid


def _quat_tilted(angle: torch.Tensor) -> torch.Tensor:
  """w-first quaternions tilted by ``angle`` about the world x axis."""
  h = angle / 2
  return torch.stack([h.cos(), h.sin(), torch.zeros_like(h), torch.zeros_like(h)], -1)


def _stub_env(tilt: torch.Tensor, q: torch.Tensor, lim: torch.Tensor):
  # projected gravity of a pelvis tilted by `tilt` about x: (0, -sin, -cos)
  pg = torch.stack([torch.zeros_like(tilt), -tilt.sin(), -tilt.cos()], -1)
  data = SimpleNamespace(projected_gravity_b=pg, joint_pos=q,
                         joint_pos_limits=lim.expand(q.shape[0], -1, -1))
  return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_margin_is_tilt_and_widened_joint_clearance():
  lim = torch.tensor([[-1.0, 1.0], [-0.5, 2.0], [0.0, 0.3]])
  q = torch.tensor([[0.0, 0.0, 0.15],       # mid-range: joint term 0.25 / 0.2
                    [0.0, 0.0, 0.15],
                    [1.1, 0.0, 0.15],       # at the widened bound: joint term 0
                    [0.0, -0.7, 0.15],      # 0.1 past it: joint term -0.5
                    [0.0, 0.0, 0.15]])
  tilt = torch.tensor([0.0, math.radians(60.0), 0.0, 0.0, math.radians(80.0)])
  g = U.g_tilt_joint(_stub_env(tilt, q, lim))
  want = torch.tensor([min(U.TILT_MAX, 0.25 / 0.2), min(U.TILT_MAX - math.radians(60.0), 1.25),
                       0.0, -0.5, U.TILT_MAX - math.radians(80.0)])
  assert torch.allclose(g, want, atol=1e-6), (g, want)
  assert abs(U.TILT_MAX - math.radians(70.0)) < 1e-12
  assert (U.JOINT_OVERSHOOT, U.JOINT_SCALE) == (0.1, 0.2)
  # upright and mid-range the tilt term binds at 70 deg = 1.22 rad
  assert abs(float(g[0]) - math.radians(70.0)) < 1e-6


def test_compose_clamps_and_declares_no_target():
  from robot_safety_sandbox.margins import compose
  lim = torch.tensor([[-10.0, 10.0]])
  q = torch.tensor([[-12.0]])              # 1.9 rad past the widened range: -9.5
  m = compose(U.g_tilt_joint)
  g, _l = m(_stub_env(torch.tensor([0.0]), q, lim))
  assert float(g[0]) == pytest.approx(-3.0)
  assert m.avoid_margin is U.g_tilt_joint and m.reach_margin is None


def test_unified_task_registers_the_agreed_env():
  _sysid_or_skip()
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from mjlab.actuator import BuiltinDcMotorActuatorCfg
  from robot_safety_sandbox.envs.g1_recover import thermal as T
  from robot_safety_sandbox.registry import AVOID, spec
  s, p = spec("g1_recover_unified"), spec("g1_recover_thermal_cbf")
  assert s.mode == AVOID and s.margin_fn.avoid_margin is U.g_tilt_joint
  assert s.margin_fn.reach_margin is None
  assert (s.ctrl_dim, s.dstb_dim, s.kwargs) == (p.ctrl_dim, p.dstb_dim, p.kwargs)
  assert s.kwargs["action_bound"] == 3.0
  cfg, base = s.cfg_builder(), p.cfg_builder()
  actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
  # actor: thermal_cbf's, 98 + 29 winding temperatures
  assert list(actor) == list(base.observations["actor"].terms) == ACTOR_98 + ["motor_temp"]
  # critic: the 98 noise-free + heading-frame velocity (101) + temperatures
  assert list(critic) == ACTOR_98 + [U.CRITIC_LIN_VEL_TERM, "motor_temp", "motor_temp_housing"]
  assert critic[U.CRITIC_LIN_VEL_TERM].func is U.base_lin_vel_yaw
  assert not cfg.observations["critic"].enable_corruption
  assert cfg.observations["actor"].enable_corruption
  # failure: tilt > 70 deg only
  assert "illegal_contact" in base.terminations
  assert set(cfg.terminations) == set(base.terminations) - {"illegal_contact"}
  assert cfg.terminations["fell_over"].params["limit_angle"] == pytest.approx(U.TILT_MAX)
  assert not [k for k, t in cfg.terminations.items()
              if not t.time_out and k != "fell_over"]
  # robot, barrier, physics, DR
  acts = cfg.scene.entities["robot"].articulation.actuators
  assert acts and all(isinstance(a, BuiltinDcMotorActuatorCfg) for a in acts)
  hip = [a for a in acts if any("hip_pitch" in n for n in a.target_names_expr)]
  assert hip and all(a.effort_limit == pytest.approx(88.0) for a in hip)
  assert isinstance(cfg.actions["joint_pos"], T.ThermalBarrierActionCfg)
  assert cfg.actions["joint_pos"].limit_c == 70.0
  mj = cfg.sim.mujoco
  assert (mj.timestep, cfg.decimation) == (0.005, 4)
  assert (mj.iterations, mj.ls_iterations, mj.ccd_iterations) == (10, 20, 50)
  for k in ("foot_friction", "base_com", "encoder_bias"):
    assert cfg.events[k].mode == "startup", k
  assert cfg.events["foot_friction"].params["ranges"] == (0.3, 1.2)
  # env_overrides pass through; the critic layout is fixed
  c2 = s.cfg_builder(episode_length_s=4.0)
  assert c2.episode_length_s == 4.0
  for k in ("critic_pelvis_height", "critic_foot_terms", "critic_base_lin_vel"):
    with pytest.raises(ValueError):
      s.cfg_builder(**{k: True})


def test_cpu_rollout_tilt_joint_margin_matches_the_env():
  sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))
  import cpu_rollout as cr
  gen = torch.Generator().manual_seed(0)
  n, k = 64, 29
  lo = -1.0 - torch.rand(k, generator=gen, dtype=torch.float64)
  hi = 1.0 + torch.rand(k, generator=gen, dtype=torch.float64)
  lim = torch.stack([lo, hi], -1)
  q = 1.3 * (2 * torch.rand(n, k, generator=gen, dtype=torch.float64) - 1) * hi
  tilt = torch.rand(n, generator=gen, dtype=torch.float64) * math.radians(100.0)
  g_env = U.g_tilt_joint(_stub_env(tilt, q, lim)).clamp(-3.0, 3.0)
  g_env = torch.where(tilt > U.TILT_MAX, g_env.clamp_max(-0.1), g_env)   # the hook's anchor
  spec = SimpleNamespace(root_qadr=0, joint_qadr=np.arange(7, 7 + k), joint_range=lim.numpy(),
                         joint_overshoot=0.1, joint_scale=0.2, tilt_max=U.TILT_MAX, clamp=3.0)
  roller = SimpleNamespace(s=spec)
  quat = _quat_tilted(tilt)
  for i in range(n):
    qpos = np.concatenate([[0.0, 0.0, 0.7], quat[i].numpy(), q[i].numpy()])
    g_cpu = cr.Roller.margin_tilt_joint(roller, SimpleNamespace(qpos=qpos))
    assert g_cpu == pytest.approx(float(g_env[i]), abs=1e-9), i
