"""The AMP discriminator, its loss, the style reward, and the two pair sources.

Peng et al. 2021 ("AMP: Adversarial Motion Priors"): D scores a transition
(phi_t, phi_t+1) with least-squares GAN targets, +1 for expert pairs and -1 for
policy pairs, plus a gradient penalty on the expert manifold; the policy is paid
r_s = max(0, 1 - 0.25 (d - 1)^2) for looking expert.

The input normalizer is FROZEN at the expert statistics (a buffer, saved in the
state_dict): a running normalizer that also saw policy pairs would move the
meaning of the gradient-penalty and logit terms under D, and couple D to the
policy's own drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: normalized inputs are clamped to this many standard deviations
X_CLIP = 10.0


class AMPDiscriminator(nn.Module):
  """MLP on the normalized pair [phi_t, phi_t+1] -> a scalar score.

  :param feat_mean, feat_std: [F] expert per-feature statistics; the pair input
      reuses them for both halves.
  """

  def __init__(self, feat_mean: torch.Tensor, feat_std: torch.Tensor,
               hidden: tuple = (1024, 512)):
    super().__init__()
    f = int(feat_mean.numel())
    self.register_buffer("mean", torch.cat([feat_mean, feat_mean]).float().clone())
    self.register_buffer("std", torch.cat([feat_std, feat_std]).float().clone())
    layers, d = [], 2 * f
    for h in hidden:
      layers += [nn.Linear(d, int(h)), nn.ReLU()]
      d = int(h)
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(d, 1)
    # start near d = 0 (r_s = 0.75 everywhere), so the first rollouts' style reward
    # is flat instead of a random function of the pose
    nn.init.normal_(self.head.weight, std=0.01)
    nn.init.zeros_(self.head.bias)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / self.std).clamp(-X_CLIP, X_CLIP)

  def score_normalized(self, xn: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(xn)).squeeze(-1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Raw pairs [..., 2F] -> d [...]."""
    return self.score_normalized(self.normalize(x))

  def hidden_weights(self):
    return [m.weight for m in self.trunk if isinstance(m, nn.Linear)]


def style_reward(d: torch.Tensor) -> torch.Tensor:
  """AMP's least-squares style reward: 1 at d = +1 (expert-like), 0 at d <= -1."""
  return (1.0 - 0.25 * (d - 1.0).square()).clamp_min(0.0)


def disc_loss(disc: AMPDiscriminator, expert: torch.Tensor, policy: torch.Tensor, *,
              gp_coef: float = 5.0, logit_reg: float = 0.05,
              weight_decay: float = 1e-4) -> tuple[torch.Tensor, dict]:
  """LSGAN loss + gradient penalty on the expert inputs (w.r.t. the NORMALIZED
  input, so its weight does not depend on feature units) + an L2 penalty on the
  output layer (logit regularization) + weight decay on the hidden layers.
  ``gp_coef`` 5 = Peng's w_gp / 2 with w_gp = 10; logit_reg / weight_decay as the
  IsaacGymEnvs AMP config. Returns (loss, detached stats)."""
  xe = disc.normalize(expert).detach().requires_grad_(True)
  de = disc.score_normalized(xe)
  dp = disc(policy)
  l_e = (de - 1.0).square().mean()
  l_p = (dp + 1.0).square().mean()
  grad, = torch.autograd.grad(de.sum(), xe, create_graph=True)
  gp = grad.square().sum(-1).mean()
  lreg = disc.head.weight.square().sum()
  wd = sum(w.square().sum() for w in disc.hidden_weights())
  loss = l_e + l_p + gp_coef * gp + logit_reg * lreg + weight_decay * wd
  stats = {"d_loss": loss.detach(), "d_loss_expert": l_e.detach(),
           "d_loss_policy": l_p.detach(), "gp": gp.detach(), "logit_reg": lreg.detach(),
           "d_expert": de.detach().mean(), "d_policy": dp.detach().mean(),
           "d_acc": 0.5 * ((de.detach() > 0).float().mean() + (dp.detach() < 0).float().mean())}
  return loss, stats


class ExpertPairs:
  """The expert transitions: features ``phi`` [R, F] and pair start rows
  ``pair_idx`` [P] (pair i = (phi[pair_idx[i]], phi[pair_idx[i] + 1]))."""

  def __init__(self, phi: torch.Tensor, pair_idx: torch.Tensor, device,
               generator: torch.Generator):
    self.phi = phi.float().to(device)
    self.idx = pair_idx.long().to(device)
    self.gen = generator

  def __len__(self) -> int:
    return int(self.idx.numel())

  def sample(self, n: int) -> torch.Tensor:
    j = torch.randint(len(self), (n,), device=self.idx.device, generator=self.gen)
    i = self.idx[j]
    return torch.cat([self.phi[i], self.phi[i + 1]], dim=-1)


class PolicyReplay:
  """GPU ring buffer of past policy pairs: D trained on the current rollout alone
  forgets the policy's earlier modes and the policy can cycle back into them."""

  def __init__(self, capacity: int, dim: int, device, generator: torch.Generator):
    self.buf = torch.zeros(int(capacity), int(dim), device=device)
    self.capacity, self.n, self.ptr = int(capacity), 0, 0
    self.gen = generator

  def __len__(self) -> int:
    return self.n

  def insert(self, x: torch.Tensor) -> None:
    m = int(x.shape[0])
    if m == 0:
      return
    if m >= self.capacity:
      x, m = x[-self.capacity:], self.capacity
    end = self.ptr + m
    if end <= self.capacity:
      self.buf[self.ptr:end] = x
    else:
      k = self.capacity - self.ptr
      self.buf[self.ptr:] = x[:k]
      self.buf[:m - k] = x[k:]
    self.ptr = end % self.capacity
    self.n = min(self.capacity, self.n + m)

  def sample(self, n: int) -> torch.Tensor:
    j = torch.randint(self.n, (n,), device=self.buf.device, generator=self.gen)
    return self.buf[j]
