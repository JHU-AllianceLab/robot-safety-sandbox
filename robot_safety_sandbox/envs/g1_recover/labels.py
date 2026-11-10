"""Start-state labels: rank statistics, a logistic fit, the frontier weight and
the value reader the labeling / harvesting scripts share.

A start state is LABELED by rolling the fallback out from it K times under a
push distribution (``examples/label_start_states.py``): the share of rollouts
that do not fall is ``p_surv``. Two questions are then asked of the labels,
and the helpers here answer them without a simulator:

* does the policy's own value V(s) rank states like the rollouts do? ``auc``
  (Mann-Whitney, ties counted half, the probability that a random surviving
  episode has the higher value) and ``spearman`` (rank correlation of the
  per-row mean value with ``p_surv``) -- the E2 question of whether V can
  stand in for rollout labels when weighting starts;
* where is the frontier? ``frontier_weight(p) = p (1 - p)`` peaks at
  p = 0.5 (a start the policy loses half the time carries the most signal)
  and vanishes on starts that are always or never survived. ``fit_logistic``
  maps a value to a survival probability, ``p_value = sigmoid(a V + b)``, so a
  value-weighted sampler uses the same p(1 - p) on a calibrated scale.

THE VALUE OF A STATE (:class:`PolicyValue`). An on-policy twin's critic is
``policy.predict_values``, but WHAT it reads depends on the policy:

* symmetric critic (stock ``ActorCriticPolicy``): the actor observation,
  normalized with the checkpoint's ``obs_mean`` / ``obs_var``;
* asymmetric (privileged) critic (``safety_sb3.policies_asym.
  AsymmetricActorCriticPolicy``, ``train_on_policy --privileged-critic``): the
  CRITIC observation group (base linear velocity, pelvis height, ...: 114
  dims on g1_recover vs the actor's 98), normalized with the SECOND set of
  statistics in the same ``tensornormalize.pt`` (``critic_obs_mean`` /
  ``critic_obs_var``). Feeding it the normalized actor obs is a shape error at
  best; :func:`robot_safety_sandbox.eval.policies.load_twin` returns only the
  actor normalizer, so every value read in the scripts goes through here.
  The env must then be built with ``critic_obs_key="critic"`` so the bridge
  exposes ``env.critic_obs()``.

Everything except :class:`PolicyValue` is pure torch and CPU-tested
(``tests/test_lafan_labels.py``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


# --- rank statistics ---------------------------------------------------------

def rankdata(x: torch.Tensor) -> torch.Tensor:
  """1-based ranks of ``x`` (1-D), ties sharing their AVERAGE rank (float64),
  as ``scipy.stats.rankdata(method="average")``."""
  x = torch.as_tensor(x).reshape(-1).double()
  n = x.numel()
  if n == 0:
    return x
  order = torch.argsort(x, stable=True)
  xs = x[order]
  new = torch.ones(n, dtype=torch.bool, device=x.device)
  new[1:] = xs[1:] != xs[:-1]
  gid = torch.cumsum(new.long(), 0) - 1               # tie group of each sorted entry
  counts = torch.bincount(gid).double()
  starts = torch.cumsum(counts, 0) - counts             # 0-based first index per group
  avg = starts + (counts + 1.0) / 2.0                   # mean of the 1-based ranks
  ranks = torch.empty(n, dtype=torch.float64, device=x.device)
  ranks[order] = avg[gid]
  return ranks


def auc(score: torch.Tensor, positive: torch.Tensor) -> float:
  """Area under the ROC curve of ``score`` for the ``positive`` class.

  The Mann-Whitney statistic: the probability that a random positive scores
  higher than a random negative, a TIE counting one half. Entries with a
  non-finite score are left out. NaN when either class is empty.
  """
  s = torch.as_tensor(score).reshape(-1).double()
  pos = torch.as_tensor(positive).reshape(-1).bool().to(s.device)
  keep = torch.isfinite(s)
  s, pos = s[keep], pos[keep]
  n_pos = int(pos.sum())
  n_neg = int(pos.numel()) - n_pos
  if n_pos == 0 or n_neg == 0:
    return float("nan")
  r = rankdata(s)
  u = float(r[pos].sum()) - n_pos * (n_pos + 1) / 2.0
  return u / (n_pos * n_neg)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
  """Spearman rank correlation (Pearson on average ranks). Pairs with a
  non-finite member are left out; NaN with fewer than two pairs or a constant
  side."""
  x = torch.as_tensor(a).reshape(-1).double()
  y = torch.as_tensor(b).reshape(-1).double().to(x.device)
  keep = torch.isfinite(x) & torch.isfinite(y)
  x, y = x[keep], y[keep]
  if x.numel() < 2:
    return float("nan")
  rx, ry = rankdata(x), rankdata(y)
  rx, ry = rx - rx.mean(), ry - ry.mean()
  den = float(torch.sqrt((rx * rx).sum() * (ry * ry).sum()))
  if den == 0.0:
    return float("nan")
  return float((rx * ry).sum()) / den


# --- calibration ---------------------------------------------------------------

def fit_logistic(x: torch.Tensor, y: torch.Tensor,
                 weight: Optional[torch.Tensor] = None, iters: int = 50,
                 ridge: float = 1e-6) -> tuple[float, float]:
  """``(a, b)`` maximizing the likelihood of ``y ~ sigmoid(a x + b)``.

  ``y`` may be 0/1 labels or FRACTIONS in [0, 1] (a row's ``p_surv``: the
  cross-entropy is the same Bernoulli likelihood with a soft target), with an
  optional per-entry ``weight`` (e.g. the number of rollouts behind each
  fraction). Newton's method on standardized ``x`` with step halving, so it
  converges in a handful of iterations; ``ridge`` (on the standardized slope)
  keeps perfectly separable data finite instead of diverging. Non-finite
  pairs are left out. A constant ``x``, or targets that are all 0 or all 1
  (no finite maximum), return slope 0 and the logit of the mean (clamped to
  [1e-6, 1 - 1e-6]).
  """
  x = torch.as_tensor(x).reshape(-1).double()
  y = torch.as_tensor(y).reshape(-1).double().to(x.device)
  w = (torch.ones_like(x) if weight is None
       else torch.as_tensor(weight).reshape(-1).double().to(x.device))
  keep = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
  x, y, w = x[keep], y[keep].clamp(0.0, 1.0), w[keep]
  if x.numel() == 0:
    return float("nan"), float("nan")
  ybar = float((w * y).sum() / w.sum())
  degenerate = ybar <= 0.0 or ybar >= 1.0      # one outcome only: no MLE exists
  ybar = min(max(ybar, 1e-6), 1.0 - 1e-6)
  b0 = float(torch.logit(torch.tensor(ybar, dtype=torch.float64)))
  mu, sd = float((w * x).sum() / w.sum()), float(x.std(unbiased=False))
  if degenerate or not sd > 1e-12:
    return 0.0, b0
  z = (x - mu) / sd
  theta = torch.tensor([0.0, b0], dtype=torch.float64, device=x.device)
  feats = torch.stack([z, torch.ones_like(z)], dim=1)   # (n, 2): slope, bias
  reg = torch.tensor([ridge, 0.0], dtype=torch.float64, device=x.device)

  def loss(t):
    lin = feats @ t
    # softplus(lin) - y lin = -[y log s + (1 - y) log(1 - s)], stable form
    return float((w * (torch.nn.functional.softplus(lin) - y * lin)).sum()
                 + 0.5 * (reg * t * t).sum())

  cur = loss(theta)
  for _ in range(int(iters)):
    p = torch.sigmoid(feats @ theta)
    grad = feats.T @ (w * (p - y)) + reg * theta
    hess = (feats.T * (w * p * (1.0 - p))) @ feats + torch.diag(reg + 1e-12)
    step = torch.linalg.solve(hess, grad)
    t = 1.0
    while t > 1e-6:
      cand = theta - t * step
      new = loss(cand)
      if new <= cur + 1e-12:
        break
      t *= 0.5
    if t <= 1e-6:
      break
    theta, prev, cur = cand, cur, new
    if float(step.abs().max()) * t < 1e-10 or abs(prev - cur) < 1e-14:
      break
  slope, bias = float(theta[0]), float(theta[1])
  return slope / sd, bias - slope * mu / sd


def frontier_weight(p: torch.Tensor) -> torch.Tensor:
  """``p (1 - p)``: 0.25 at p = 0.5, 0 at p in {0, 1} (the frontier weight
  the weighted spawn sampler draws starts with)."""
  p = torch.as_tensor(p)
  return p * (1.0 - p)


# --- reading a bank's names --------------------------------------------------

def row_names(bank: dict, column: str = "category",
              names_key: str = "categories",
              fallback: Optional[Sequence[str]] = None) -> list[str]:
  """Per-row name of an index column (``category`` -> ``meta["categories"]``,
  ``clip`` -> ``meta["clips"]``).

  The index is into the meta of the FILE the row came from: a single-file bank
  reads ``meta[names_key]``; a bank from several files (``load_bank`` with a
  list: ``meta = {"parts": [meta_0, ...], "paths": [...]}``) resolves each row
  through ``meta["parts"][bank["part"][row]]``. Rows with index -1 (no such
  column in their file: the walker rows) are named by ``fallback[kind]`` when
  given, else ``"-"``.
  """
  meta = bank.get("meta", {}) or {}
  idx = bank[column].reshape(-1).long().cpu().tolist()
  n = len(idx)
  if "parts" in meta:
    parts = bank.get("part")
    part = (parts.reshape(-1).long().cpu().tolist() if parts is not None
            else [0] * n)
    tables = [m.get(names_key, []) or [] for m in meta["parts"]]
  else:
    part = [0] * n
    tables = [meta.get(names_key, []) or []]
  kind = bank["kind"].reshape(-1).long().cpu().tolist() if "kind" in bank else None
  out = []
  for i in range(n):
    tab = tables[part[i]] if 0 <= part[i] < len(tables) else []
    if 0 <= idx[i] < len(tab):
      out.append(str(tab[idx[i]]))
    elif fallback is not None and kind is not None and 0 <= kind[i] < len(fallback):
      out.append(str(fallback[kind[i]]))
    else:
      out.append("-")
  return out


# --- the value of a state ------------------------------------------------------

def is_asymmetric(policy) -> bool:
  """True when ``policy``'s value net reads the privileged CRITIC obs."""
  try:
    from safety_sb3.policies_asym import AsymmetricActorCriticPolicy
    if isinstance(policy, AsymmetricActorCriticPolicy):
      return True
  except ImportError:
    pass
  return getattr(policy, "critic_observation_space", None) is not None


class PolicyValue:
  """``V(s)`` of a loaded on-policy twin, on the observation its critic reads.

  :param model: a safety_sb3 on-policy learner (``load_twin``'s model).
  :param policy_path: its checkpoint zip (the obs statistics are found next to
      it, as ``load_twin`` finds them).
  :param device: torch device.

  ``asymmetric`` says whether the env must expose the critic group
  (``make_tensor(..., critic_obs_key="critic")``). Call with the RAW actor obs
  and the RAW ``env.critic_obs()``; the matching normalizer (clip +-10, the
  training convention) is applied here.
  """

  def __init__(self, model, policy_path: str, device):
    from robot_safety_sandbox.eval.policies import find_obs_stats
    self.policy = model.policy
    if not hasattr(self.policy, "predict_values"):
      raise SystemExit(f"{policy_path}: no value net (predict_values); an "
                       f"on-policy twin is needed to read V(s)")
    self.asymmetric = is_asymmetric(self.policy)
    pt = find_obs_stats(policy_path, "tensornorm", "tensornormalize.pt")
    if not pt:
      raise SystemExit(f"obs-norm stats not found next to {policy_path}")
    st = torch.load(pt, map_location=device, weights_only=True)
    if self.asymmetric:
      if "critic_obs_mean" not in st:
        raise SystemExit(
          f"{pt}: the policy's critic is asymmetric but the file carries no "
          f"critic_obs_mean / critic_obs_var")
      self.mean, self.var = st["critic_obs_mean"].to(device), st["critic_obs_var"].to(device)
    else:
      self.mean, self.var = st["obs_mean"].to(device), st["obs_var"].to(device)
    self.critic_obs_key = "critic" if self.asymmetric else None

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)

  def __call__(self, obs: torch.Tensor,
               critic_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``V`` per row of ``obs`` (raw actor obs) / ``critic_obs`` (raw critic
    obs, required for an asymmetric critic), float ``[n]``."""
    x = critic_obs if self.asymmetric else obs
    if x is None:
      raise ValueError("this policy's critic reads the privileged critic obs; "
                       "build the env with critic_obs_key='critic' and pass "
                       "env.critic_obs()")
    if x.shape[-1] != self.mean.shape[0]:
      raise ValueError(f"value net expects {self.mean.shape[0]}-dim obs, got "
                       f"{x.shape[-1]} ({'critic' if self.asymmetric else 'actor'} group)")
    with torch.no_grad():
      v = self.policy.predict_values(self.normalize(x.float()))
    return v.reshape(x.shape[0], -1)[:, 0].float()
