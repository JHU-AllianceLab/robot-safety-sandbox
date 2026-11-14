"""AMPPPO: safety_sb3's tensor-path CumulativePPO1P plus the AMP discriminator.

Nothing in the PPO update changes. The style reward is already inside the reward
the collector stores (AMPTaskEnv, under the normalizer); this class only owns the
discriminator D:

* ``_setup_model`` builds D (frozen expert normalizer), its own Adam, the expert
  pairs and the policy-pair replay, and hands D to the AMPTaskEnv in the chain;
* ``collect_rollouts`` brackets the rollout so the env records the free pairs,
  and logs ``amp/*``;
* ``train`` runs the D steps on (expert, policy) batches, THEN the PPO update,
  then files a share of this rollout's pairs into the replay.

D is NOT part of the policy zip (``_excluded_save_params``): the eval tools load
``final_model.zip`` with a plain ``CumulativePPO1P.load``, and SB3 refuses unknown
state_dict keys. D lives beside it, ``amp_disc.pt`` / ``checkpoints/amp_disc_*.pt``
(:class:`AMPCheckpointCallback`, on the model checkpoints' own grid).
"""

from __future__ import annotations

import hashlib
import os

import torch
from safety_sb3 import CumulativePPO1P
from stable_baselines3.common.callbacks import BaseCallback

from .discriminator import AMPDiscriminator, ExpertPairs, PolicyReplay, disc_loss
from .env import AMPTaskEnv
from .features import FEATURE_DIM, FEATURE_VERSION, SLICES

#: the AMP recipe (spec_ppo-amp.md (b)); a run's ``amp:`` config overrides keys
AMP_DEFAULTS = dict(
  expert_path="data/g1_amp_expert_lafan_train_v1.pt",
  hidden=(1024, 512),
  lr=1e-4,
  d_steps=20,            # per PPO update (= 5 epochs x 4 minibatches)
  batch=4096,            # expert pairs and policy pairs per D step, each
  replay_size=1_000_000,
  replay_frac=0.5,       # share of a D step's policy pairs drawn from the replay
  insert_frac=0.25,      # share of a rollout's free pairs filed into the replay
  gp_coef=5.0,
  logit_reg=0.05,
  weight_decay=1e-4,
  grad_clip=1.0,
  seed_offset=7919,      # D's sampling generator seed = seed + this
)


def _find_amp_env(env) -> AMPTaskEnv | None:
  e, seen = env, set()
  while e is not None and id(e) not in seen:
    if isinstance(e, AMPTaskEnv):
      return e
    seen.add(id(e))
    d = getattr(e, "__dict__", {})
    e = d.get("venv", d.get("env"))
  return None


def _sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


class AMPPPO(CumulativePPO1P):
  """CumulativePPO1P with an AMP discriminator. ``amp``: overrides of
  :data:`AMP_DEFAULTS`. Every other keyword is the PPO recipe's."""

  def __init__(self, policy, env, *, amp: dict | None = None, **kwargs):
    unknown = set(amp or {}) - set(AMP_DEFAULTS)
    if unknown:
      raise ValueError(f"unknown amp keys {sorted(unknown)}; known {sorted(AMP_DEFAULTS)}")
    self.amp_cfg = {**AMP_DEFAULTS, **(amp or {})}
    super().__init__(policy, env, **kwargs)

  # --- setup / persistence ----------------------------------------------------

  def _setup_model(self) -> None:
    super()._setup_model()
    if self.env is not None:       # a bare load() has no env and no D
      self._amp_setup()

  def _amp_setup(self) -> None:
    cfg, dev = self.amp_cfg, self.device
    self._amp_env = _find_amp_env(self.env)
    if self._amp_env is None:
      raise ValueError("AMPPPO needs an AMPTaskEnv in its env chain (amp.env.AMPTaskEnv)")
    path = str(cfg["expert_path"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    meta = data.get("meta", {})
    if meta.get("version") != FEATURE_VERSION:
      raise ValueError(f"{path}: feature version {meta.get('version')!r}, this code computes "
                       f"{FEATURE_VERSION!r}; rebuild it with examples/build_amp_expert.py")
    amp_dim = int(self._amp_env.venv.obs_groups()[self._amp_env.amp_group].shape[-1])
    if amp_dim != FEATURE_DIM or int(data["phi"].shape[1]) != FEATURE_DIM:
      raise ValueError(f"AMP feature width: env {amp_dim}, expert {tuple(data['phi'].shape)}, "
                       f"code {FEATURE_DIM}")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(self.seed or 0) + int(cfg["seed_offset"]))
    self._amp_expert = ExpertPairs(data["phi"], data["pair_idx"], dev, gen)
    self._amp_disc = AMPDiscriminator(data["mean"], data["std"], tuple(cfg["hidden"])).to(dev)
    self._amp_opt = torch.optim.Adam(self._amp_disc.parameters(), lr=float(cfg["lr"]),
                                     betas=(0.9, 0.999))
    self._amp_replay = PolicyReplay(int(cfg["replay_size"]), 2 * FEATURE_DIM, dev, gen)
    self._amp_pairs = None
    self._amp_meta = {"version": FEATURE_VERSION, "expert_path": os.path.abspath(path),
                      "expert_sha256": _sha256(path), "expert_pairs": len(self._amp_expert),
                      "expert_meta": {k: v for k, v in meta.items() if k != "feature_names"}}
    self._amp_env.attach(self._amp_disc)
    print(f"[amp] D {2 * FEATURE_DIM}->{list(cfg['hidden'])}->1 on {len(self._amp_expert)} "
          f"expert pairs ({path}); w_style {self._amp_env.w_style}")

  def _excluded_save_params(self) -> list[str]:
    return super()._excluded_save_params() + [
      "_amp_env", "_amp_expert", "_amp_disc", "_amp_opt", "_amp_replay", "_amp_pairs"]

  def save_amp(self, path: str) -> None:
    """D weights + its frozen normalizer, its optimizer, the recipe and the expert
    provenance (the replay and the expert set are not saved)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"cfg": dict(self.amp_cfg), "disc": self._amp_disc.state_dict(),
                "opt": self._amp_opt.state_dict(), "meta": self._amp_meta,
                "num_timesteps": int(self.num_timesteps)}, path)

  def load_amp(self, path: str) -> None:
    st = torch.load(path, map_location=self.device, weights_only=False)
    self._amp_disc.load_state_dict(st["disc"])
    self._amp_opt.load_state_dict(st["opt"])

  # --- the loop ---------------------------------------------------------------

  def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
    self._amp_env.begin_rollout()
    ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
    self._amp_pairs = self._amp_env.rollout_pairs()
    for k, v in self._amp_env.rollout_stats().items():
      self.logger.record(f"amp/{k}", v)
    P = self._amp_pairs
    self.logger.record("amp/free_pairs", int(P.shape[0]))
    if P.shape[0]:
      # where the policy's own motion sits, in expert standard deviations, per
      # feature group: a height or velocity shortcut shows up here first
      xn = self._amp_disc.normalize(P)[:, :FEATURE_DIM]
      for name, sl in SLICES.items():
        self.logger.record(f"amp/gap_{name}", float(xn[:, sl].mean(0).abs().mean()))
    return ok

  def train(self) -> None:
    self._amp_update()
    super().train()

  def _amp_update(self) -> None:
    cfg, P, R = self.amp_cfg, self._amp_pairs, self._amp_replay
    n_cur = 0 if P is None else int(P.shape[0])
    if n_cur == 0 and len(R) == 0:
      self.logger.record("amp/d_updates", 0)
      return
    B = int(cfg["batch"])
    n_rep = int(round(B * float(cfg["replay_frac"]))) if len(R) else 0
    if n_cur == 0:
      n_rep = B
    n_now = B - n_rep
    gen = self._amp_expert.gen
    sums: dict = {}
    self._amp_disc.train()
    for _ in range(int(cfg["d_steps"])):
      parts = []
      if n_now:
        parts.append(P[torch.randint(n_cur, (n_now,), device=P.device, generator=gen)])
      if n_rep:
        parts.append(R.sample(n_rep))
      loss, st = disc_loss(self._amp_disc, self._amp_expert.sample(B), torch.cat(parts),
                           gp_coef=float(cfg["gp_coef"]), logit_reg=float(cfg["logit_reg"]),
                           weight_decay=float(cfg["weight_decay"]))
      self._amp_opt.zero_grad(set_to_none=True)
      loss.backward()
      st["d_grad_norm"] = torch.nn.utils.clip_grad_norm_(self._amp_disc.parameters(),
                                                         float(cfg["grad_clip"]))
      self._amp_opt.step()
      for k, v in st.items():
        sums[k] = sums.get(k, 0.0) + v.detach()
    self._amp_disc.eval()
    n = float(cfg["d_steps"])
    for k, v in sums.items():
      self.logger.record(f"amp/{k}", float(v) / n)
    self.logger.record("amp/d_updates", int(cfg["d_steps"]))
    if n_cur:
      k = int(n_cur * float(cfg["insert_frac"]))
      if k:
        R.insert(P[torch.randperm(n_cur, device=P.device, generator=gen)[:k]])
    self.logger.record("amp/replay_fill", len(R) / R.capacity)


class AMPCheckpointCallback(BaseCallback):
  """Save D as ``<save_path>/amp_disc_<num_timesteps>.pt`` on the SAME call grid
  as SB3's CheckpointCallback (``save_freq`` in calls), so every
  ``model_<n>_steps.zip`` has its discriminator under the same ``<n>``."""

  def __init__(self, save_path: str, save_freq: int):
    super().__init__()
    self.save_path, self.save_freq = save_path, int(save_freq)

  def _on_step(self) -> bool:
    if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
      self.model.save_amp(os.path.join(self.save_path, f"amp_disc_{self.num_timesteps}.pt"))
    return True
