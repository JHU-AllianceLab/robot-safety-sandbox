"""The greedy eta-net (envs/g1_recover/eta_net.py) and its phi / W.

Simulator-free:

  * GREEDY -- on random W-scaled points the chunked GPU / CPU net equals the
    one-row-at-a-time greedy net exactly, for chunk sizes that split clusters;
  * NET -- kept rows are pairwise >= eta apart and every dropped row lies
    within eta of a kept one;
  * PHI -- 1 + 3 + 29 + 3 + 3 + 29 = 68 dims, x / y / yaw invariant, gravity
    a unit vector, the body-frame angular velocity;
  * W -- per-group pooled std times the group weights.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover.eta_net import (  # noqa: E402
  GROUP_WEIGHTS, compute_W, eta_net, group_slices, phi_of_bank)


def _naive(X: np.ndarray, d_min: float, order) -> list:
  kept = []
  for i in order:
    if all(np.sum((X[i] - X[k]) ** 2) >= d_min ** 2 for k in kept):
      kept.append(int(i))
  return kept


@pytest.mark.parametrize("chunk", [7, 64, 1000])
def test_greedy_equals_sequential(chunk):
  g = np.random.default_rng(0)
  # clustered points: many conflicts inside a chunk
  centres = g.normal(size=(40, 5))
  X = centres[g.integers(0, 40, 600)] + 0.05 * g.normal(size=(600, 5))
  order = g.permutation(len(X))
  want = _naive(X, 0.1, order)
  got = eta_net(X, 0.1, order=order, chunk=chunk, block=50, device="cpu")
  assert got.tolist() == want


def test_net_properties():
  g = np.random.default_rng(1)
  X = g.normal(size=(800, 6)) * 0.2
  kept = eta_net(X, 0.15, order=g.permutation(800), chunk=128, device="cpu")
  K = X[kept]
  d = np.sqrt(((K[:, None] - K[None]) ** 2).sum(-1))
  np.fill_diagonal(d, np.inf)
  assert d.min() >= 0.15
  dropped = np.setdiff1d(np.arange(800), kept)
  dd = np.sqrt(((X[dropped][:, None] - K[None]) ** 2).sum(-1)).min(1)
  assert (dd < 0.15).all()


def _bank(n=50, j=29, seed=0):
  g = torch.Generator().manual_seed(seed)
  q = torch.randn(n, 4, generator=g)
  q[:, 3] = 0.0                                    # no yaw-axis component: heading-frame quats
  q = q / q.norm(dim=1, keepdim=True)
  return {"root_z": 0.7 + 0.05 * torch.randn(n, generator=g), "root_quat_h": q,
          "lin_vel_h": torch.randn(n, 3, generator=g), "ang_vel_h": torch.randn(n, 3, generator=g),
          "joint_pos": torch.randn(n, j, generator=g), "joint_vel": torch.randn(n, j, generator=g)}


def test_phi_layout():
  b = _bank()
  phi = phi_of_bank(b)
  sl = group_slices(29)
  assert phi.shape == (50, 68) and sl["joint_vel"].stop == 68
  grav = phi[:, sl["gravity"]]
  assert torch.allclose(grav.norm(dim=1), torch.ones(50, dtype=torch.float64), atol=1e-12)
  # identity orientation: gravity (0, 0, -1), body angular velocity = heading-frame one
  b["root_quat_h"] = torch.tensor([[1.0, 0, 0, 0]]).repeat(50, 1)
  phi = phi_of_bank(b)
  assert torch.allclose(phi[:, sl["gravity"]], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).expand(50, 3))
  assert torch.allclose(phi[:, sl["ang_vel"]], b["ang_vel_h"].double())
  assert torch.allclose(phi[:, sl["joint_pos"]], b["joint_pos"].double())


def test_W_pooled_per_group():
  phi = phi_of_bank(_bank(n=400, seed=2))
  W = compute_W(phi)
  sl = group_slices(29)
  std = phi.std(0, unbiased=False)
  for gname, s in sl.items():
    pooled = math.sqrt(float((std[s] ** 2).mean()))
    assert torch.allclose(W[s], torch.full_like(W[s], GROUP_WEIGHTS[gname] / pooled))
