"""Policy evaluation of a G1 fallback: its exact (gamma = 1) safety value, a
learned estimate of it, and a conformal threshold on that estimate.

For a FIXED policy pi (a trained fallback, MLP or LSTM, SafetyPPO or SHAC) and
the ViaLasso safe set K (``k_set``), the undiscounted finite-horizon safety
value and Q function are

    V^pi(s)    = min_{0 <= t <= T} g_k(s_t),             s_0 = s, s_{t+1} ~ pi
    Q^pi(s, a) = min(g_k(s), V^pi_{T-1}(s')),           s' = step(s, a)

so ``V^pi(s) >= 0`` iff pi keeps the robot in K for T seconds from s
(``g_k`` is >= 0 exactly inside K). With gamma = 1 these are not contractions,
so they are labeled by rollouts (Monte Carlo, exact for each rollout's
randomization) and a network is fit to the labels; the decision ``V_hat(s) -
eps > 0`` then gets its eps from a calibration split:

* :func:`false_safe_threshold` -- the smallest eps on a fixed grid such that,
  among the calibration states it accepts, the fraction actually unsafe has a
  Clopper-Pearson upper bound <= alpha at confidence 1 - beta (Bonferroni over
  the grid, so picking the eps is covered too), trying a pre-registered ladder
  of alphas in order (the friend's Gamma calibration);
* :func:`residual_threshold` -- split conformal on the residuals ``V_hat - V``:
  eps = their ceil((n + 1)(1 - alpha))-th smallest value, so that
  ``P(V^pi(s) >= V_hat(s) - eps) >= 1 - alpha`` over the calibration
  distribution.

Both need EXCHANGEABLE calibration samples: one state (and one rollout) per
SONIC episode, which ``examples/build_policy_eval_states.py`` draws.

The states come from SONIC's harvested trajectories (the monitor's query
distribution): :func:`trajectory_rows` turns ``(episode, step)`` pairs into
start-state rows (``states.py`` schema) whose last action is SONIC's target
into that state, and gives SONIC's next action for Q.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from robot_safety_sandbox.envs.g1_recover.eta_net import phi_of_bank
from robot_safety_sandbox.envs.g1_recover.states import (
  KIND_HANDOVER, speed_of, to_heading_frame)

#: The friend's pre-registered relaxation ladder of the false-safe rate.
ALPHA_LADDER = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


# --- states ------------------------------------------------------------------------

def trajectory_rows(res: dict, e: torch.Tensor, j: torch.Tensor) -> dict:
  """Start-state rows of trajectory file ``res`` at episodes ``e``, steps ``j``
  (``j >= 1``: the action INTO step j is the target recorded at j - 1). The
  last action is clamped to the bridge's box like every bank row; SONIC's NEXT
  action (``sonic_action``, the target applied FROM the state, in env action
  units) is NOT clamped -- it is what SONIC actually applies."""
  meta = res["meta"]
  if bool((j < 1).any()):
    raise ValueError("trajectory_rows: steps must be >= 1")
  scale = torch.as_tensor(meta["env_action_scale"])
  offset = torch.as_tensor(meta["env_action_offset"])
  bound = float(meta["action_bound"])
  dt, pre = float(meta["dt"]), int(meta["onset_index"])
  q = res["traj/root_quat"][e, j]
  quat_h, lin_h, ang_h = to_heading_frame(q, res["traj/lin_vel"][e, j], res["traj/ang_vel"][e, j])
  m = int(e.numel())
  frame = res["traj/ref_frame"][e, j]
  return {
    "root_z": res["traj/root_pos"][e, j, 2], "root_quat_h": quat_h,
    "lin_vel_h": lin_h, "ang_vel_h": ang_h,
    "joint_pos": res["traj/joint_pos"][e, j], "joint_vel": res["traj/joint_vel"][e, j],
    "action": ((res["traj/target"][e, j - 1] - offset) / scale).clamp(-bound, bound),
    "command": torch.zeros(m, 3), "phase_step": torch.zeros(m, dtype=torch.long),
    "kind": torch.full((m,), KIND_HANDOVER, dtype=torch.long),
    "fell": res["ep/fell"][e], "time_to_fall": res["ep/time_to_fall"][e] - (j - pre).float() * dt,
    "push_severity": res["ep/dv"][e] / (float(meta["dv_range"][1]) or 1.0),
    "difficulty": torch.zeros(m), "speed": speed_of(lin_h), "env_id": res["ep/env_id"][e],
    "clip": res["ep/clip"][e], "category": res["ep/category"][e], "frame": frame,
    "t": frame.float() / 50.0, "push_impulse": res["ep/impulse"][e],
    "push_rel": res["ep/rel"][e], "push_onset_s": res["ep/onset_s"][e],
    "push_duration_s": torch.zeros(m), "src_row": res["ep/row"][e],
    "traj_index": e.clone(),
    # policy-evaluation columns (not bank schema; read straight from the file)
    "step": j.clone(), "t_rel": (j - pre).float() * dt, "dv": res["ep/dv"][e],
    "sonic_action": (res["traj/target"][e, j] - offset) / scale,
  }


def features(rows: dict, q_action: torch.Tensor | None = None) -> torch.Tensor:
  """The value's input: the 68-D phi of the state (x / y / yaw invariant,
  ``eta_net.phi_of_bank``), the last applied action (part of what the policy
  observes, so part of its state), and for Q the action ``q_action``."""
  x = [phi_of_bank(rows).float(), rows["action"].float()]
  if q_action is not None:
    x.append(q_action.float())
  return torch.cat(x, dim=1)


# --- the value network ---------------------------------------------------------------

class ValueNet(nn.Module):
  """MLP on standardized features (the mean / std ride in the module)."""

  def __init__(self, in_dim: int, hidden=(256, 256, 256)):
    super().__init__()
    layers, d = [], in_dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ELU()]
      d = h
    layers.append(nn.Linear(d, 1))
    self.net = nn.Sequential(*layers)
    self.register_buffer("mean", torch.zeros(in_dim))
    self.register_buffer("std", torch.ones(in_dim))

  def set_normalizer(self, x: torch.Tensor) -> None:
    self.mean.copy_(x.mean(0))
    self.std.copy_(x.std(0).clamp_min(1e-6))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net((x - self.mean) / self.std).squeeze(-1)


# --- conformal thresholds --------------------------------------------------------------

def clopper_pearson_upper(k: torch.Tensor, n: torch.Tensor, beta: float) -> torch.Tensor:
  """One-sided (1 - beta) upper bound on a binomial rate with k events in n
  trials (1 where n = 0)."""
  from scipy.stats import beta as beta_dist
  k = torch.as_tensor(k, dtype=torch.float64)
  n = torch.as_tensor(n, dtype=torch.float64)
  ub = torch.as_tensor(beta_dist.ppf(1.0 - beta, (k + 1).numpy(), (n - k).clamp_min(1e-12).numpy()),
                       dtype=torch.float64)
  ub = torch.where(k >= n, torch.ones_like(ub), ub)
  return torch.where(n <= 0, torch.ones_like(ub), ub)


def false_safe_threshold(score: torch.Tensor, unsafe: torch.Tensor, beta: float = 1e-3,
                         alphas=ALPHA_LADDER, grid: int = 200) -> dict:
  """Smallest eps (accept ``score > eps``) whose false-safe rate among the
  accepted calibration samples has a Clopper-Pearson upper bound <= alpha at
  confidence 1 - beta, Bonferroni-corrected over ``grid`` candidate eps (the
  score quantiles, fixed before looking at the labels); the first alpha of the
  ladder that some eps meets. Returns eps, alpha, the bound, accept rate and
  recall (accepted among the truly safe), or eps = +inf if none does."""
  score = score.double().cpu()
  unsafe = unsafe.bool().cpu()
  cand = torch.unique(torch.quantile(score, torch.linspace(0.0, 1.0, grid + 1, dtype=torch.float64)))
  acc = score[None, :] > cand[:, None]
  n = acc.sum(1)
  k = (acc & unsafe[None, :]).sum(1)
  ub = clopper_pearson_upper(k, n, beta / len(cand))
  safe_n = int((~unsafe).sum())
  for a in alphas:
    ok = ub <= a
    if bool(ok.any()):
      i = int(torch.nonzero(ok)[0])                  # cand ascending: the smallest eps
      return {"eps": float(cand[i]), "alpha": a, "beta": beta, "bound": float(ub[i]),
              "n_cal": int(score.numel()), "accepted": int(n[i]), "false_safe": int(k[i]),
              "accept_rate": float(n[i]) / score.numel(),
              "recall": float((acc[i] & ~unsafe).sum()) / max(safe_n, 1), "grid": int(len(cand))}
  return {"eps": float("inf"), "alpha": None, "beta": beta, "n_cal": int(score.numel()),
          "accepted": 0, "recall": 0.0, "grid": int(len(cand))}


def residual_threshold(pred: torch.Tensor, label: torch.Tensor, alpha: float) -> dict:
  """Split conformal on ``r = pred - label``: eps = the ceil((n+1)(1-alpha))-th
  smallest r, so P(label >= pred - eps) >= 1 - alpha for a new exchangeable
  sample (+inf when n is too small for alpha)."""
  r = (pred.double() - label.double()).cpu().sort().values
  n = int(r.numel())
  kq = math.ceil((n + 1) * (1.0 - alpha))
  eps = float("inf") if kq > n else float(r[kq - 1])
  return {"eps": eps, "alpha": alpha, "n_cal": n}


def evaluate_threshold(score: torch.Tensor, label: torch.Tensor, eps: float) -> dict:
  """On a test split: the accepted states (score > eps), the false-safe rate
  among them, recall of the truly safe (label >= 0), and the residual
  coverage P(label >= score - eps)."""
  score, label = score.double().cpu(), label.double().cpu()
  unsafe = label < 0.0
  acc = score > eps
  na, nk = int(acc.sum()), int((acc & unsafe).sum())
  return {"n": int(score.numel()), "accept_rate": na / max(score.numel(), 1),
          "false_safe_rate": nk / max(na, 1), "false_safe": nk, "accepted": na,
          "recall": int((acc & ~unsafe).sum()) / max(int((~unsafe).sum()), 1),
          "coverage": float((label >= score - eps).double().mean()) if math.isfinite(eps) else 1.0}
