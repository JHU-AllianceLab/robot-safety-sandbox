"""The Q^pi network of a fallback (``examples/q_fallback_fit.py``) and its loader.

``QNet(s, a)``: an MLP on ``cat(s, a)`` standardized by the fit's train-split mean / std
(``mu`` / ``sd`` buffers), ELU hidden layers, one output. ``s`` is the fallback's critic group
sanitized as the walk filter's ``critic_obs`` (nan -> 0, last action 69:98 clamped to +-3), with
the 33-D domain-randomization context appended when the fit used ``--use-ctx`` (``ctx_dim``);
``a`` the executed action (env action units, the proposal box +-6.5).
"""

from __future__ import annotations

import torch
from torch import nn


class QNet(nn.Module):
  def __init__(self, s_dim: int, a_dim: int, hidden=(512, 256, 128)):
    super().__init__()
    layers, d = [], s_dim + a_dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ELU()]
      d = h
    layers.append(nn.Linear(d, 1))
    self.net = nn.Sequential(*layers)
    self.register_buffer("mu", torch.zeros(s_dim + a_dim))
    self.register_buffer("sd", torch.ones(s_dim + a_dim))
    self.s_dim, self.a_dim = s_dim, a_dim

  def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    x = torch.cat([s.float(), a.float()], dim=-1)
    return self.net((x - self.mu) / self.sd).squeeze(-1)


def load_qnet(path: str, device) -> tuple[QNet, dict]:
  """``(net in eval mode, the file's metadata without the weights)``."""
  ck = torch.load(path, map_location=device, weights_only=False)
  net = QNet(int(ck["s_dim"]), int(ck["a_dim"]), tuple(ck["hidden"])).to(device)
  net.load_state_dict(ck["state_dict"])
  net.eval()
  for p in net.parameters():
    p.requires_grad_(False)
  meta = {k: v for k, v in ck.items() if k != "state_dict"}
  meta.setdefault("obs_dim", int(ck["s_dim"]) - int(ck.get("ctx_dim", 0) or 0))
  meta.setdefault("ctx_dim", 0)
  return net, meta
