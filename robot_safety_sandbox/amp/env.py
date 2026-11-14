"""AMPTaskEnv: the style reward enters the task reward before the PPO collector sees it.

Wrapper order (examples/track_common.build_stack): the base env, then
WMTubeFilteredEnv (or its BARE pass-through), then TrackStatsEnv, then THIS,
then TensorVecNormalize, which the learner adds. Sitting under the normalizer
means the mixed reward goes straight into the buffer and GAE, and the collector
stays safety_sb3's own.

Per step, with phi read from the base env's ``amp`` observation group:

    pair  = [phi_t, phi_t+1]                    (the EXECUTED transition)
    valid = ~done                               (phi_t+1 of a done row is a reset state)
    r     = r_task + w_style * step_dt * r_s(D(pair)) * valid

``w_style`` is in reward-weight units (mjlab scales every reward term by dt, so the
style term is scaled the same way).

The style reward is paid on every valid executed transition, the fallback's
included (the user's "reward from executed transitions"). The discriminator's
POLICY samples are another matter (AMENDMENTS A4): only "free" pairs enter them,
pairs whose two states were both reached by the task policy's own actions --
the action into s_t and the action s_t -> s_t+1 both proposals, not fallback
substitutions. Otherwise an arm that engages more would train a different
discriminator. Engagement is read from the filter wrapper below (its
``last_info["engaged"]``); a chain without one (no filter layer) counts nothing
as engaged.
"""

from __future__ import annotations

import torch

from .discriminator import style_reward


def _find_layer(env, attr: str):
  """The first layer of a wrapper chain that OWNS ``attr`` (in its __dict__),
  walking each layer's own ``venv`` / ``env`` (delegating __getattr__ would hop
  over layers)."""
  e, seen = env, set()
  while e is not None and id(e) not in seen:
    d = getattr(e, "__dict__", {})
    if attr in d:
      return e
    seen.add(id(e))
    e = d.get("venv", d.get("env"))
  return None


def w_style_from_env(env, frac: float = 0.4) -> float:
  """``frac`` x the sum of the positive reward weights of the base env's task terms
  (names not starting with ``zoo_``: the margin probe has weight 1 and pays 0).
  frac 0.4 makes the style term at most ~30% of the reward at its maximum, a
  naturalness regularizer under a task-heavy objective."""
  mj = env.mj
  rm = mj.reward_manager
  w = sum(max(float(rm.get_term_cfg(n).weight), 0.0) for n in rm.active_terms
          if not n.startswith("zoo_"))
  return float(frac) * w


class AMPTaskEnv:
  """See the module docstring. Delegates everything it does not define."""

  def __init__(self, venv, *, w_style: float, step_dt: float = 0.02, amp_group: str = "amp"):
    self.venv = venv
    self.w_style, self.step_dt, self.amp_group = float(w_style), float(step_dt), amp_group
    self.num_envs, self.device = int(venv.num_envs), str(venv.device)
    self.disc = None
    self._phi = None
    self._filter = _find_layer(venv, "last_info")
    n, dev = self.num_envs, self.device
    self._prev_eng = torch.zeros(n, dtype=torch.bool, device=dev)
    self._ep_task = torch.zeros(n, device=dev)
    self._recording = False
    self._pairs: list = []
    self._free: list = []
    self._acc: dict = {}

  def __getattr__(self, name):
    try:
      venv = self.__dict__["venv"]
    except KeyError:
      raise AttributeError(name) from None
    return getattr(venv, name)

  def attach(self, disc) -> None:
    """The discriminator that scores pairs (None: no style reward)."""
    self.disc = disc

  def _amp(self) -> torch.Tensor:
    return self.venv.obs_groups()[self.amp_group].float()

  def _engaged(self) -> torch.Tensor:
    if self._filter is None:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return self._filter.last_info["engaged"].bool()

  # --- the env API -------------------------------------------------------------

  def reset(self):
    obs = self.venv.reset()
    self._phi = self._amp()
    self._prev_eng.zero_()
    self._ep_task.zero_()
    return obs

  def step_tensor(self, actions: torch.Tensor):
    obs, r_task, dones, timeouts, l_x = self.venv.step_tensor(actions)
    phi1 = self._amp()
    d = dones.bool()
    valid = ~d
    eng = self._engaged()
    free = valid & ~eng & ~self._prev_eng
    pair = torch.cat([self._phi, phi1], dim=-1)
    if self.disc is not None:
      with torch.no_grad():
        dsc = self.disc(pair)
      rs = style_reward(dsc) * valid.float()
    else:
      dsc = torch.zeros_like(r_task)
      rs = torch.zeros_like(r_task)
    r = r_task + (self.w_style * self.step_dt) * rs
    self._ep_task += r_task
    if self._recording:
      self._pairs.append(pair)
      self._free.append(free)
      a = self._acc
      vf, ef, ff = valid.float(), (valid & eng).float(), free.float()
      for k, v in (("r_task", r_task.sum()), ("r_style", rs.sum()), ("valid", vf.sum()),
                   ("rs_free", (rs * ff).sum()), ("free", ff.sum()),
                   ("rs_eng", (rs * ef).sum()), ("eng", ef.sum()), ("d", (dsc * vf).sum()),
                   ("n", torch.tensor(float(self.num_envs), device=self.device))):
        a[k] = a.get(k, 0.0) + v
      if bool(d.any()):
        a["ep_ret"] = a.get("ep_ret", 0.0) + (self._ep_task * d.float()).sum()
        a["ep_n"] = a.get("ep_n", 0.0) + d.float().sum()
    self._ep_task = torch.where(d, torch.zeros_like(self._ep_task), self._ep_task)
    # the action into the NEXT pair's first state; none after a reset
    self._prev_eng = eng & valid
    self._phi = phi1
    return obs, r, dones, timeouts, l_x

  # --- what AMPPPO reads once per rollout -------------------------------------

  def begin_rollout(self) -> None:
    self._recording = True
    self._pairs, self._free, self._acc = [], [], {}

  def rollout_pairs(self) -> torch.Tensor:
    """The rollout's free pairs, [M, 2F] (one boolean gather: one host sync)."""
    self._recording = False
    if not self._pairs:
      return torch.zeros(0, 0, device=self.device)
    p, f = torch.cat(self._pairs), torch.cat(self._free)
    self._pairs, self._free = [], []
    return p[f]

  def rollout_stats(self) -> dict[str, float]:
    """Means over the rollout; ``r_style_gap`` = free minus engaged style reward,
    the implicit per-step price of an engagement (in r_s units)."""
    a = {k: float(v) for k, v in self._acc.items()}
    if not a:
      return {}

    def div(x, y):
      return x / y if y > 0 else float("nan")

    out = {
      "r_task_mean": div(a["r_task"], a["n"]),
      "r_style_mean": div(a["r_style"], a["valid"]),
      "r_style_free": div(a["rs_free"], a["free"]),
      "r_style_engaged": div(a["rs_eng"], a["eng"]),
      "d_policy_steps": div(a["d"], a["valid"]),
      "valid_frac": div(a["valid"], a["n"]),
      "free_frac": div(a["free"], a["n"]),
      "engaged_frac": div(a["eng"], a["n"]),
      "w_style_eff": div(self.w_style * self.step_dt * a["r_style"], abs(a["r_task"])),
    }
    out["r_style_gap"] = out["r_style_free"] - out["r_style_engaged"]
    if a.get("ep_n", 0.0) > 0:
      out["ep_task_return"] = a["ep_ret"] / a["ep_n"]
    return out
