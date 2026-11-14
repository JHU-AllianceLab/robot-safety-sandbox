"""CPU checks of the AMP learner (robot_safety_sandbox/amp/).

The modules load through a stub package whose __path__ is the amp directory, so
neither the sandbox package nor mjlab is imported (the PPO test imports
safety_sb3, which takes a minute on a loaded login node).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types

import numpy as np
import pytest
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = "_amp_under_test"
if _PKG not in sys.modules:
  _pkg = types.ModuleType(_PKG)
  _pkg.__path__ = [os.path.join(_ZOO, "robot_safety_sandbox", "amp")]
  sys.modules[_PKG] = _pkg
feats = importlib.import_module(f"{_PKG}.features")
disc_mod = importlib.import_module(f"{_PKG}.discriminator")
env_mod = importlib.import_module(f"{_PKG}.env")
F = feats.FEATURE_DIM


def _rand_quat(n, g):
  q = torch.randn(n, 4, generator=g, dtype=torch.float64)
  return q / q.norm(dim=-1, keepdim=True)


def _quat_mat(q):
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
    torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
    torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def _quat_mul(a, b):
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


# --- features ------------------------------------------------------------------

def test_layout():
  assert F == 68 and feats.FEATURE_VERSION == "g1-amp-v1"
  cover = sorted(i for sl in feats.SLICES.values() for i in range(F)[sl])
  assert cover == list(range(F))


def test_quat_apply_inverse_is_the_transposed_rotation():
  g = torch.Generator().manual_seed(0)
  q, v = _rand_quat(64, g), torch.randn(64, 3, generator=g, dtype=torch.float64)
  want = (_quat_mat(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
  assert torch.allclose(feats.quat_apply_inverse(q, v), want, atol=1e-12)


def test_library_rows_equal_the_sim_features_at_any_yaw():
  """A yaw-free library row (root_quat_h, velocities in the heading frame) and the
  same state placed at a random yaw (world quaternion, body-frame velocities, as
  the env term reads them) give the same phi."""
  g = torch.Generator().manual_seed(1)
  n = 32
  q_h = _rand_quat(n, g)
  q_h[:, 3] = 0.0                                     # (w, x, y, 0): no yaw component
  q_h = q_h / q_h.norm(dim=-1, keepdim=True)
  yaw = torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi
  q_yaw = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
  q_w = _quat_mul(q_yaw, q_h)
  v_h, w_h = (torch.randn(n, 3, generator=g, dtype=torch.float64) for _ in range(2))
  lib = {"root_quat_h": q_h, "lin_vel_h": v_h, "ang_vel_h": w_h,
         "joint_pos": torch.randn(n, 29, generator=g, dtype=torch.float64),
         "joint_vel": torch.randn(n, 29, generator=g, dtype=torch.float64)}
  root_h = torch.rand(n, generator=g, dtype=torch.float64)
  phi_lib, _ = feats.features_from_library(lib, root_h)
  R_yaw = _quat_mat(q_yaw)
  v_w = (R_yaw @ v_h.unsqueeze(-1)).squeeze(-1)
  w_w = (R_yaw @ w_h.unsqueeze(-1)).squeeze(-1)
  phi_sim = feats.amp_features(root_h, q_w, feats.quat_apply_inverse(q_w, v_w),
                               feats.quat_apply_inverse(q_w, w_w), lib["joint_pos"],
                               lib["joint_vel"])
  assert torch.allclose(phi_lib.double(), phi_sim, atol=1e-5)


def test_library_clamp_counts_rows():
  lib = {"root_quat_h": torch.tensor([[1.0, 0, 0, 0]] * 3), "lin_vel_h": torch.zeros(3, 3),
         "ang_vel_h": torch.zeros(3, 3), "joint_pos": torch.zeros(3, 29),
         "joint_vel": torch.zeros(3, 29)}
  lib["joint_pos"][1, 14] = 0.6
  rng = torch.tensor([[-1.0, 1.0]] * 29)
  rng[14] = torch.tensor([-0.63, 0.50])
  phi, n = feats.features_from_library(lib, torch.zeros(3), rng)
  assert n == 1 and phi[1, feats.SLICES["qpos"]][14] == pytest.approx(0.5)


# --- discriminator ---------------------------------------------------------------

def test_style_reward():
  r = disc_mod.style_reward(torch.tensor([1.0, -1.0, 3.0, 0.0]))
  assert r.tolist() == [1.0, 0.0, 0.0, 0.75]


def test_gradient_penalty_of_a_linear_discriminator():
  """With no hidden layer D(x^) = w.x^ + b, so the penalty is |w|^2 exactly."""
  torch.manual_seed(0)
  D = disc_mod.AMPDiscriminator(torch.zeros(F), torch.ones(F) * 2.0, hidden=())
  torch.nn.init.normal_(D.head.weight)
  e, p = torch.randn(16, 2 * F) * 0.1, torch.randn(16, 2 * F) * 0.1
  _, st = disc_mod.disc_loss(D, e, p)
  assert float(st["gp"]) == pytest.approx(float(D.head.weight.detach().square().sum()), rel=1e-5)


def test_discriminator_learns_toy_data_and_normalizer_round_trips():
  torch.manual_seed(0)
  D = disc_mod.AMPDiscriminator(torch.zeros(F), torch.ones(F), hidden=(64, 32))
  opt = torch.optim.Adam(D.parameters(), lr=1e-3)
  first = None
  for _ in range(200):
    e, p = torch.randn(256, 2 * F) + 1.0, torch.randn(256, 2 * F) - 1.0
    loss, st = disc_mod.disc_loss(D, e, p)
    opt.zero_grad()
    loss.backward()
    opt.step()
    first = first if first is not None else float(st["d_loss"])
  assert float(st["d_acc"]) > 0.95 and float(st["d_loss"]) < first
  D2 = disc_mod.AMPDiscriminator(torch.ones(F), torch.ones(F) * 3, hidden=(64, 32))
  D2.load_state_dict(D.state_dict())
  x = torch.randn(8, 2 * F)
  assert torch.equal(D(x), D2(x)) and torch.equal(D2.mean, torch.zeros(2 * F))


def test_replay_ring_wraps_and_keeps_the_newest():
  g = torch.Generator().manual_seed(0)
  R = disc_mod.PolicyReplay(5, 1, "cpu", g)
  R.insert(torch.arange(3.0)[:, None])
  R.insert(torch.arange(3.0, 7.0)[:, None])
  assert len(R) == 5 and sorted(R.buf[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0]
  R.insert(torch.arange(10.0, 20.0)[:, None])
  assert sorted(R.buf[:, 0].tolist()) == [15.0, 16.0, 17.0, 18.0, 19.0]


# --- AMPTaskEnv ------------------------------------------------------------------

class _ToyBase:
  """Base env: amp group = a counter per env; episode length 3; reward 1."""

  def __init__(self, n=4):
    self.num_envs, self.device = n, "cpu"
    self.t = torch.zeros(n)
    self.k = 0

  def obs_groups(self):
    return {"amp": self.t[:, None].expand(-1, F).clone()}

  def reset(self):
    self.t.zero_()
    return torch.zeros(self.num_envs, 2)

  def step_tensor(self, a):
    self.k += 1
    self.t += 1
    done = self.t >= 3
    self.t = torch.where(done, torch.zeros_like(self.t), self.t)
    return torch.zeros(self.num_envs, 2), torch.ones(self.num_envs), done, done, None


class _ToyFilter:
  """Filter layer: engaged pattern scripted per step."""

  def __init__(self, env, pattern):
    self.env, self.pattern, self.i = env, pattern, 0
    self.last_info = {"engaged": torch.zeros(env.num_envs, dtype=torch.bool)}

  def __getattr__(self, name):
    return getattr(self.__dict__["env"], name)

  def step_tensor(self, a):
    out = self.env.step_tensor(a)
    self.last_info = {"engaged": torch.tensor(self.pattern[self.i])}
    self.i += 1
    return out


def test_reward_mix_masking_and_free_pairs():
  base = _ToyBase(2)
  # env 0 engaged at step 4 only; env 1 never. Episodes: steps 1, 2 valid, 3 done
  filt = _ToyFilter(base, [[False, False], [False, False], [False, False], [True, False],
                           [False, False], [False, False]])
  env = env_mod.AMPTaskEnv(filt, w_style=2.0, step_dt=0.02)

  class _ConstD(torch.nn.Module):
    def forward(self, x):
      return torch.zeros(x.shape[0])            # r_s = 0.75 everywhere

  env.attach(_ConstD())
  env.reset()
  env.begin_rollout()
  rewards, dones = [], []
  for _ in range(6):
    _, r, d, _, _ = env.step_tensor(torch.zeros(2, 1))
    rewards.append(r)
    dones.append(d)
  # valid steps pay 1 + 2 * 0.02 * 0.75; the done step (t = 3) pays the task reward only
  for r, d in zip(rewards, dones):
    want = torch.where(d, torch.ones(2), torch.full((2,), 1.0 + 0.03))
    assert torch.allclose(r, want)
  P = env.rollout_pairs()
  # env 1: steps 1, 2, 4, 5 valid (3, 6 done) -> 4 free pairs. env 0: steps 1, 2 free;
  # step 4 engaged; step 5's pair starts in the state the fallback reached -> 2 free
  assert P.shape == (6, 2 * F)
  # the first state of every pair after a reset is the reset state (counter 0)
  assert set(P[:, 0].tolist()) <= {0.0, 1.0}
  st = env.rollout_stats()
  assert st["r_style_mean"] == pytest.approx(0.75) and st["ep_task_return"] == pytest.approx(3.0)
  assert st["engaged_frac"] == pytest.approx(1 / 12)


# --- AMPPPO on a toy tensor env (imports safety_sb3) ------------------------------

def _toy_tensor_env(n=8):
  from gymnasium import spaces
  from safety_sb3 import TensorVecEnv

  class Toy(TensorVecEnv):
    critic_observation_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)

    def __init__(self):
      super().__init__(n, spaces.Box(-np.inf, np.inf, (4,), np.float32),
                       spaces.Box(-6.5, 6.5, (3,), np.float32), "cpu")
      self.x = torch.zeros(n, F)
      self.t = torch.zeros(n)

    def obs_groups(self):
      return {"amp": self.x.clone()}

    def critic_obs(self):
      return torch.cat([self.x[:, :4], self.t[:, None]], 1)

    def reset(self):
      self.x.zero_()
      self.t.zero_()
      return self.x[:, :4].clone()

    def step_tensor(self, a):
      self.x = 0.9 * self.x + 0.1 * a.repeat(1, F // 3 + 1)[:, :F]
      self.t += 1
      d = self.t >= 5
      self.t = torch.where(d, torch.zeros_like(self.t), self.t)
      self.x = torch.where(d[:, None], torch.zeros_like(self.x), self.x)
      return (self.x[:, :4].clone(), -a.square().mean(1), d, d,
              torch.zeros(n))

  return Toy()


def test_amp_ppo_learns_saves_and_loads_without_d(tmp_path):
  ppo_mod = importlib.import_module(f"{_PKG}.ppo")
  from safety_sb3 import CumulativePPO1P
  g = torch.Generator().manual_seed(0)
  phi = torch.randn(200, F, generator=g)
  path = tmp_path / "expert.pt"
  torch.save({"phi": phi, "pair_idx": torch.arange(199), "mean": phi.mean(0),
              "std": phi.std(0).clamp_min(0.05), "meta": {"version": feats.FEATURE_VERSION}},
             path)
  env = env_mod.AMPTaskEnv(_toy_tensor_env(), w_style=1.0)
  model = ppo_mod.AMPPPO("MlpPolicy", env, amp=dict(expert_path=str(path), batch=64, d_steps=3,
                                                    replay_size=1000, hidden=(32, 16)),
                         n_steps=8, batch_size=32, n_epochs=2, gamma_anneal=False,
                         normalize_obs=True, device="cpu", seed=0,
                         policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])))
  w0 = [p.detach().clone() for p in model._amp_disc.parameters()]
  model.learn(total_timesteps=8 * 8 * 3)
  assert any(not torch.equal(a, b) for a, b in zip(w0, model._amp_disc.parameters()))
  logged = model.logger.name_to_value
  assert len(model._amp_replay) > 0
  assert model._asymmetric
  model.save(tmp_path / "m.zip")
  plain = CumulativePPO1P.load(tmp_path / "m.zip", device="cpu")
  assert not hasattr(plain, "_amp_disc")
  model.save_amp(str(tmp_path / "amp_disc.pt"))
  before = model._amp_disc(torch.randn(4, 2 * F, generator=g))
  model2 = ppo_mod.AMPPPO("MlpPolicy", env_mod.AMPTaskEnv(_toy_tensor_env(), w_style=1.0),
                          amp=dict(expert_path=str(path), hidden=(32, 16), replay_size=1000),
                          n_steps=8, batch_size=32, gamma_anneal=False, normalize_obs=True,
                          device="cpu", seed=1,
                          policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])))
  model2.load_amp(str(tmp_path / "amp_disc.pt"))
  g2 = torch.Generator().manual_seed(0)
  torch.randn(200, F, generator=g2)
  assert torch.allclose(model2._amp_disc(torch.randn(4, 2 * F, generator=g2)), before)
  del logged
