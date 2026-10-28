"""Greedy eta-net thinning of a start-state bank (the ViaLasso pool's nodes).

The G1 pipeline of ``robot_states_reachability_checker`` thins its state pool
into NODES: in the metric ``d(x, y) = ||W (phi(x) - phi(y))||_2`` it keeps a
state iff no state kept before it is closer than eta (``tools/g1/g1c/knn.py``
``eta_net``, eta = 0.1), visiting the states in a random order. Kept states
are pairwise >= eta apart and every dropped state lies within eta of a kept
one. This module does the same to a bank in the schema of
``envs/g1_recover/states.py``.

PHI (``tools/g1/g1c/phi.py``, here with 29 joints instead of 12; x / y / yaw
invariant, which the heading-frame bank already is):

    height       pelvis z                                   1
    gravity      projected gravity R^T (0, 0, -1)           3
    joint_pos    joint angles (entity order)               29
    lin_vel      root linear velocity, heading frame        3
    ang_vel      root angular velocity, BODY frame          3
    joint_vel    joint velocities                          29   (68 in all)

W (``compute_W``, mode "pooled"): every component of a group gets the group's
weight (:data:`GROUP_WEIGHTS`, the original's) over the group's pooled std,
the RMS of its component stds over the bank. W is computed on the bank being
thinned unless one is passed in (a test bank takes its train bank's W, so the
two nets are in the same units).

:func:`eta_net` is the original's exact greedy algorithm: chunks of rows are
tested against every kept row (float64 GEMM, borderline pairs recomputed
directly), then thinned among themselves by the lexicographic-first maximal
independent set, which equals visiting them one by one.
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

GROUPS = ("height", "gravity", "joint_pos", "lin_vel", "ang_vel", "joint_vel")
#: ``configs/nominal.yaml`` metric.group_weights
GROUP_WEIGHTS = {"height": 1.0, "gravity": 1.0, "joint_pos": 1.0,
                 "lin_vel": 0.3, "ang_vel": 0.3, "joint_vel": 0.3}
#: ``configs/nominal.yaml`` metric.eta
ETA = 0.1


def group_slices(n_joints: int) -> dict[str, slice]:
  j = int(n_joints)
  edges = np.cumsum([0, 1, 3, j, 3, 3, j])
  return {g: slice(int(a), int(b)) for g, a, b in zip(GROUPS, edges[:-1], edges[1:])}


def phi_of_bank(bank: dict, device="cpu") -> torch.Tensor:
  """phi of every bank row, ``[N, 6 + 2 * joints]`` float64."""
  f = lambda k: bank[k].to(device=device, dtype=torch.float64)  # noqa: E731
  q = f("root_quat_h")
  down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64, device=device).expand(q.shape[0], 3)
  return torch.cat([f("root_z")[:, None], quat_apply_inverse(q, down), f("joint_pos"),
                    f("lin_vel_h"), quat_apply_inverse(q, f("ang_vel_h")),
                    f("joint_vel")], dim=1)


def compute_W(phi: torch.Tensor, std_floor: float = 1e-6) -> torch.Tensor:
  """Pooled-std W over ``phi`` (module docstring); refuses a group or
  component without spread, as the original does."""
  sl = group_slices((phi.shape[1] - 10) // 2)     # 1 + 3 + j + 3 + 3 + j
  std = phi.double().std(0, unbiased=False)
  if bool((~(std >= std_floor)).any()):
    raise ValueError(f"compute_W: phi components {torch.nonzero(~(std >= std_floor)).flatten().tolist()} "
                     f"have std < {std_floor}")
  W = torch.zeros_like(std)
  for g in GROUPS:
    s = float(torch.sqrt((std[sl[g]] ** 2).mean()))
    W[sl[g]] = GROUP_WEIGHTS[g] / s
  return W


def _conflicts(Q: torch.Tensor, Y: torch.Tensor, r2: float, sq_q: torch.Tensor,
               sq_y: torch.Tensor) -> torch.Tensor:
  """``(|Q|, |Y|)`` bool: ``||q - y||^2 < r2``, exact in float64. GEMM score on
  mean-centred rows; entries within the GEMM error band of r2 are recomputed
  by direct differences."""
  d2 = torch.addmm(sq_q[:, None] + sq_y[None, :], Q, Y.T, alpha=-2.0)
  band = 1e-12 * float(sq_q.max() + sq_y.max() + 1.0)
  out = d2 < r2 - band
  amb = (d2 - r2).abs() <= band
  if bool(amb.any()):
    i, j = amb.nonzero(as_tuple=True)
    out[i, j] = ((Q[i] - Y[j]) ** 2).sum(1) < r2
  return out


@torch.inference_mode()
def eta_net(X, d_min: float, *, order=None, chunk: int = 4096, block: int = 65536,
            device: str = "cuda") -> np.ndarray:
  """Greedy d_min-net of the rows of X (W-scaled phi): visit rows in ``order``
  (default index order) and keep a row iff no row kept so far is closer than
  d_min. Returns the kept row ids in visiting order."""
  Xg = torch.as_tensor(X).to(device=device, dtype=torch.float64)
  Xc = Xg - Xg.mean(0)
  sq = (Xc * Xc).sum(1)
  order = (torch.arange(len(Xc), device=device) if order is None
           else torch.as_tensor(order, device=device).long())
  r2 = float(d_min) ** 2
  kept = torch.empty(len(Xc), dtype=torch.long, device=device)
  nk = 0
  for s in range(0, len(order), chunk):
    ids = order[s:s + chunk]
    Q, sq_q = Xc[ids], sq[ids]
    alive = torch.ones(len(ids), dtype=torch.bool, device=device)
    for b in range(0, nk, block):
      kb = kept[b:min(nk, b + block)]
      alive &= ~_conflicts(Q, Xc[kb], r2, sq_q, sq[kb]).any(1)
    ids, Q, sq_q = ids[alive], Q[alive], sq_q[alive]
    if len(ids) == 0:
      continue
    C = _conflicts(Q, Q, r2, sq_q, sq_q)
    C.fill_diagonal_(False)
    earlier = torch.tril(C, -1)                    # earlier[i, j]: j < i conflicts with i
    acc = torch.zeros(len(ids), dtype=torch.bool, device=device)
    rej = torch.zeros_like(acc)
    while True:
      und = ~(acc | rej)
      if not bool(und.any()):
        break
      acc |= und & ~(earlier & (und | acc)[None, :]).any(1)   # every earlier neighbour rejected -> keep
      rej |= ~acc & (C & acc[None, :]).any(1)                   # a kept neighbour -> drop
    kept[nk:nk + int(acc.sum())] = ids[acc]
    nk += int(acc.sum())
  return kept[:nk].cpu().numpy()
