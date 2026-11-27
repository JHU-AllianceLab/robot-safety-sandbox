"""RL-DH on g1_recover: a learned half-space filter of SONIC.

RL-DH (Lavanakul et al., arXiv 2402.05279; safety_sb3.hyperplane) learns a
state-dependent half-space ``{u | a(x)^T u >= b(x)}`` of safe controls with
PPO, and deploys it as the min-norm projection of a reference controller's
action onto that half-space. ResSafe (arXiv 2609.15988) trains it on a G1 with
the reference policy in the loop. Here the reference is SONIC, the tracker
that produced g1_recover's start states: every episode SONIC resumes the
spawn row's LAFAN clip at the row's frame (``clip`` / ``frame`` bank columns)
and the learned half-space filters its action.

The environment is the run's own g1_recover env (start bank, domain
randomization, margin ``g``, terminations), untouched. The training reward is
[DH] Eq. 7 on it (``reward="paper"``): +1 per step whose next state is safe
(``g >= 0``), ``fail_reward`` (-1) per unsafe one, so only the sign of ``g``
is used. This module only
supplies the ``reference_fn`` of :class:`safety_sb3.hyperplane.
HyperplaneFilterEnv` (sampler ``"reference"``) and the control box ``U``:

* :class:`SonicReferenceFn` keeps one :class:`~.sonic.SonicTracker` in step
  with the env. The filter env calls it once per state, after each reset or
  step: it writes the APPLIED (filtered) action into SONIC's action history
  (``--sonic-history applied``, as ``examples/eval_sonic_qcbf.py``), advances
  the clip cursor, restarts SONIC on every env that just respawned (mjlab's
  ``episode_length_buf == 0``) from its new row, and returns SONIC's action.
* ``U`` defaults to :func:`~.policy_value_filter.sonic_action_box`, the
  targets SONIC's raw action can command (about +-20 env units). SONIC's
  actions leave the bridge's +-3 box on most steps, so clipping to it would
  change the reference; ``box="action"`` uses the bridge box instead.

HIP PITCH. SONIC drives the env's robot as the task built it: the hip pitch
is the robot's own actuator (``--hip mjlab`` in the harvester /
``examples/eval_sonic_qcbf.py``), never SONIC's 7520-22 swapped in. On the
identified G1 (``envs/assets/g1_sysid.py``, the ``*_v6`` tasks) that is the
only choice (``sonic.use_sonic_hip_pitch`` refuses its dcmotor actuators), and
the v6 banks were harvested with it (``configs/sweeps/shac_thermal70_v6/
data_jobs.txt``), so the reference is the harvest's controller. A bank whose
meta records another hip (the main checkout's ``g1_recover`` banks, harvested
with ``--hip sonic``) is refused unless ``allow_hip_mismatch=True``: SONIC
would then run on another actuator than the one that produced its rows.

CAVEAT. SONIC restarts cold at the handover: its 10-tick histories are
filled with the spawn state and its last action is zero, whereas in the
harvest it had been tracking the clip all along. A property of the reference,
identical for every filter trained or evaluated here.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from robot_safety_sandbox.envs.g1_recover.policy_value_filter import (
  record_applied_action, sonic_action_box)
from robot_safety_sandbox.envs.g1_recover.sonic import (
  DEFAULT_SONIC_DIR, SonicPolicy, SonicReference, SonicTracker)
from robot_safety_sandbox.envs.g1_recover.states import load_bank

_ZOO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.abspath(__file__)))))

SONIC_HISTORIES = ("applied", "nominal")
BOXES = ("sonic", "action")


def spawn_bank(mj) -> dict:
  """The bank g1_recover's ``recover_spawn`` term draws from (row indices in
  ``mj._zoo_spawn_row`` index it)."""
  term = None
  try:
    term = mj.event_manager.get_term_cfg("recover_spawn").func
  except Exception:  # noqa: BLE001 -- fall back to the module cache below
    term = None
  bank = getattr(term, "bank", None)
  if bank is None:
    from robot_safety_sandbox.envs.g1_recover import spawn as _spawn
    banks = list(_spawn._BANK_CACHE.values())
    if len(banks) != 1:
      raise ValueError("cannot find the spawn bank: no recover_spawn term with a bank "
                       f"and {len(banks)} cached banks")
    bank = banks[0]
  # (load_bank fills an absent clip / frame column with -1, so these keys are
  # always there; SonicReferenceFn checks each respawned row's values)
  for k in ("clip", "frame", "meta"):
    if k not in bank:
      raise ValueError(f"the spawn bank has no {k!r}: a SONIC reference needs "
                       "harvested rows that name their clip and frame")
  return bank


class SonicReferenceFn:
  """``reference_fn(venv, obs) -> u_ref`` for ``HyperplaneFilterEnv``: SONIC
  resuming each spawn row's clip. Set :attr:`filter_env` to the filter env once
  it exists (``history="applied"`` reads its ``applied_control``)."""

  #: the hip-pitch actuator SONIC drives here: always the env robot's own
  #: (module docstring HIP PITCH)
  hip = "mjlab"

  def __init__(self, venv, sonic_dir: Optional[str] = None,
               history: str = "applied", library: Optional[str] = None,
               allow_hip_mismatch: bool = False):
    if history not in SONIC_HISTORIES:
      raise ValueError(f"history must be one of {SONIC_HISTORIES}, got {history!r}")
    self.venv, self.mj = venv, venv.mj
    dev = self.dev = venv.device
    bank = spawn_bank(self.mj)
    meta = bank["meta"]
    # the harvester records its --hip in the bank meta (dedup keeps it)
    self.bank_hip = meta.get("hip")
    if self.bank_hip is not None and self.bank_hip != self.hip:
      msg = (f"the spawn bank was harvested with SONIC on the {self.bank_hip!r} hip "
             f"pitch, but here SONIC drives the env robot's own ({self.hip!r}): not "
             "the controller that produced the rows")
      if not allow_hip_mismatch:
        raise ValueError(msg + " (allow_hip_mismatch=True to accept)")
      print(f"[rl-dh] WARNING: {msg}", flush=True)
    lib_path = library or meta.get("library")
    if not lib_path or not os.path.isfile(lib_path):
      local = os.path.join(_ZOO, "data", os.path.basename(str(lib_path or "")))
      if not os.path.isfile(local):
        raise FileNotFoundError(f"the bank's clip library {lib_path} is not here "
                                f"(nor {local}); pass library=")
      lib_path = local
    lib = load_bank(lib_path, "cpu")
    lm = lib["meta"]
    if list(lm["clips"]) != list(meta["clips"]):
      raise ValueError(f"the bank's clips differ from the library's ({lib_path}): "
                       "its clip ids would index the wrong clips")
    csv_dir = lm["csv_dir"]
    if not os.path.isdir(csv_dir):
      csv_dir = os.path.join(_ZOO, "data", "lafan1_g1")
    ref = SonicReference.build(csv_dir, lm["clips"], dev, library=lib)
    policy = SonicPolicy.from_onnx(sonic_dir or DEFAULT_SONIC_DIR).to(dev).eval()
    self.tracker = SonicTracker(venv, policy, ref)
    self.clip = bank["clip"].to(dev).long()
    self.frame = bank["frame"].to(dev).long()
    self.n_clips = len(lm["clips"])
    self.history = history
    self.library = lib_path
    self.filter_env = None
    self._started = False
    self.n_restarts = 0
    self.n_nonfinite = 0

  def box(self, kind: str = "sonic") -> tuple[torch.Tensor, torch.Tensor]:
    """``U`` as ``(low, high)``, ``(29,)``: SONIC's reachable targets, or the
    bridge's action box."""
    if kind not in BOXES:
      raise ValueError(f"box must be one of {BOXES}, got {kind!r}")
    if kind == "sonic":
      return sonic_action_box(self.tracker)
    sp = self.venv.action_space
    return (torch.as_tensor(sp.low, device=self.dev, dtype=torch.float32),
            torch.as_tensor(sp.high, device=self.dev, dtype=torch.float32))

  @torch.no_grad()
  def __call__(self, venv, obs) -> torch.Tensor:
    tr = self.tracker
    if self._started:
      applied = getattr(self.filter_env, "applied_control", None)
      if self.history == "applied" and applied is not None:
        record_applied_action(tr, applied)
      tr.advance()
    fresh = self.mj.episode_length_buf.reshape(-1) == 0
    if not self._started:
      fresh = torch.ones_like(fresh)
    row = getattr(self.mj, "_zoo_spawn_row", None)
    if row is None:
      raise ValueError("the env publishes no _zoo_spawn_row (not a g1_recover spawn)")
    if bool(fresh.any()):
      bad = fresh & (row < 0)
      if bool(bad.any()):
        raise ValueError(f"{int(bad.sum())} envs respawned from a non-bank row "
                         "(nominal / synthetic): SONIC has no clip to resume there")
      r = row.clamp_min(0)
      clip, frame = self.clip[r], self.frame[r]
      # load_bank fills a missing clip / frame column with -1, and a negative
      # clip would index the LAST clip: refuse a row that names no clip / frame
      bad = fresh & ((clip < 0) | (clip >= self.n_clips) | (frame < 0))
      if bool(bad.any()):
        raise ValueError(f"{int(bad.sum())} envs respawned from bank rows without a "
                         f"clip in [0, {self.n_clips}) and a frame >= 0 (e.g. row "
                         f"{int(r[bad][0])}: clip {int(clip[bad][0])}, frame "
                         f"{int(frame[bad][0])}): SONIC has no clip to resume there")
      tr.start(fresh, clip, frame)
      self.n_restarts += int(fresh.sum())
    self._started = True
    u = tr.act()
    # A NaN physics state reaches SONIC unguarded (it reads robot.data) and
    # would poison the obs normalizer through the appended u_ref; sanitize the
    # way the bridge sanitizes its own obs. The env's SONIC state recovers at
    # its respawn (start refills the histories).
    bad = ~torch.isfinite(u).all(dim=-1)
    if bool(bad.any()):
      self.n_nonfinite += int(bad.sum())
      u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    return u


def _filter_env_class():
  """``HyperplaneFilterEnv`` whose privileged critic group also carries
  ``u_ref`` (``critic_reference=True``), so the value net keeps seeing
  everything the actor sees, as the fallback runs' critic group (a superset of
  the actor group) does; plus a ``dh/ref_nonfinite`` count in ``metrics()``
  and :meth:`prime`, ``reset()`` without the inner reset."""
  import gymnasium as gym
  import numpy as np
  from safety_sb3.hyperplane import HyperplaneFilterEnv

  class SonicHyperplaneFilterEnv(HyperplaneFilterEnv):
    def __init__(self, venv, *, critic_reference: bool = True, **kw):
      super().__init__(venv, **kw)
      cs = self.critic_observation_space
      self.critic_reference = bool(critic_reference and self.append_reference
                                   and cs is not None)
      if self.critic_reference:
        lo = np.concatenate([np.broadcast_to(cs.low, cs.shape).ravel(),
                             np.full(self.m, -np.inf, np.float32)]).astype(np.float32)
        hi = np.concatenate([np.broadcast_to(cs.high, cs.shape).ravel(),
                             np.full(self.m, np.inf, np.float32)]).astype(np.float32)
        self.critic_observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    @torch.no_grad()
    def prime(self, obs: torch.Tensor) -> torch.Tensor:
      """The first observation of the episodes the inner env is ALREADY in,
      without resetting it: ``reset()`` minus ``venv.reset()``. ``obs`` is the
      inner env's current raw actor obs. The reference runs once on it (the
      first call of :class:`SonicReferenceFn` starts SONIC on EVERY env from
      its current ``_zoo_spawn_row``), and ``u_ref`` is appended as after a
      reset. Call it once, before the first ``step_tensor``, in place of
      ``reset()``: for a caller whose inner env's construction reset already
      issued the episodes it needs (``examples/label_start_states.py``: the
      sequential spawn's first wave, which a second reset would discard)."""
      if self._u_ref is not None:
        raise RuntimeError("prime() starts the reference; the env already has one "
                           "(a reset, step or prime ran before)")
      self._refresh_reference(obs)
      return self._obs(obs)

    def critic_obs(self):
      c = self.venv.critic_obs()
      if not self.critic_reference or c is None:
        return c
      return torch.cat([c.reshape(self.num_envs, -1).float(), self._u_ref], dim=-1)

    def metrics(self):
      out = super().metrics()
      ref = self.reference_fn
      out["dh/ref_nonfinite"] = float(getattr(ref, "n_nonfinite", 0))
      return out

  return SonicHyperplaneFilterEnv


def make_sonic_dh_env(venv, *, sonic_dir: Optional[str] = None,
                      history: str = "applied", box: str = "sonic",
                      explore_prob: float = 0.05, append_reference: bool = True,
                      critic_reference: bool = True,
                      reward: str = "paper", bonus: float = 0.0,
                      fail_reward: float = -1.0, seed: Optional[int] = None,
                      allow_hip_mismatch: bool = False):
  """``(filter env, SonicReferenceFn)`` around the g1_recover tensor env
  ``venv``: a ``HyperplaneFilterEnv`` with sampler ``"reference"`` (SONIC),
  ``U`` from ``box``. ``reward="paper"`` is [DH] Eq. 7 on the env's margin:
  ``1 + bonus`` per step whose next state has ``g >= 0``, ``fail_reward``
  otherwise (train with CumulativePPO1P, ``bootstrap_on_timeout=True``);
  ``"margin"`` passes ``g`` through (SafetyPPO1P; ours, not the paper's).
  ``critic_reference`` appends ``u_ref`` to the privileged critic group too
  (with ``append_reference`` and a critic group only). ``allow_hip_mismatch``:
  accept a bank harvested on another hip pitch than the env robot's own
  (module docstring HIP PITCH)."""
  ref = SonicReferenceFn(venv, sonic_dir=sonic_dir, history=history,
                         allow_hip_mismatch=allow_hip_mismatch)
  low, high = (x.detach().cpu().numpy() for x in ref.box(box))
  env = _filter_env_class()(
    venv, critic_reference=critic_reference, sampler="reference",
    explore_prob=explore_prob, reward=reward, bonus=bonus, fail_reward=fail_reward,
    low=low, high=high, reference_fn=ref, append_reference=append_reference,
    seed=seed)
  ref.filter_env = env
  return env, ref
