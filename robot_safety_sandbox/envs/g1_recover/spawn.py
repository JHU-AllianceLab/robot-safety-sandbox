"""The g1_recover spawn (a mixture over start-state sources) and its curriculum.

SOURCES. Every episode starts from one of five sources, drawn per env with
fixed mixture weights (renormalized over the sources the bank actually has):

    nominal    the default knees-bent stance at rest (g1_stabilize's spawn)
    clean      walker rows, undisturbed mid-gait        sorted by speed
    pushed     walker rows after a push, walker SURVIVED sorted by push size
    fell       walker rows the walker then FELL from     sorted by time-to-fall,
                                                         longest (easiest) first
    synthetic  stance + random tilt / twist / offsets    sorted by scale

CURRICULUM. Each env holds one difficulty level per source, 0..L. A level
selects a window of that source's sorted rows (level 0 = the easiest sixth,
level L = the hardest 40%); an episode that ends on the TIMEOUT promotes the
level of the source it spawned from, one that ends on a FALL demotes it. So
an env that keeps falling on fast pushed states backs off to slower ones,
while unrecoverable rows stay in the mix without swamping the batch.

LAST ACTION. A harvested row carries the action the walker last applied. mjlab
zeroes the action buffer AFTER the reset events run, so the spawn writes the
state from ``__call__`` and seeds ``last_action`` from ``reset()``, which the
event manager calls later in the same reset. The first observation then looks
like a real handover. The same per-env action is published as
``env._zoo_reset_action`` for the bridge, whose random-arm generator glides
from it instead of snapping the arms to the default pose (base.py).

LOGGING. :class:`RecoverCurriculum` returns its statistics from ``compute``,
which mjlab logs as ``Curriculum/recover/<key>`` and the bridge forwards:
``surv_<source>`` (survival rate, exponential average over ~1/ema episodes),
``lvl_<source>`` (mean level), ``top_<source>`` (fraction of envs at the top
level) and ``surv_v<lo>_<hi>`` (walker rows by spawn speed, m/s). Keys stay
within SB3's 36-character console width, which truncates longer ones.
"""

from __future__ import annotations

import os

import torch

from mjlab.managers.manager_base import ManagerTermBase

from robot_safety_sandbox.envs.g1_recover.states import (
  KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC, load_bank, sample_yaw_xy,
  write_states)

SOURCES = ("nominal", "clean", "pushed", "fell", "synthetic")
SRC_NOMINAL, SRC_CLEAN, SRC_PUSHED, SRC_FELL, SRC_SYNTHETIC = range(5)
#: The plan's mixture: 15% stance, 35% clean walker, 35% pushed walker (fell
#: rows oversampled: 20 of the 35), 15% synthetic.
DEFAULT_WEIGHTS = {"nominal": 0.15, "clean": 0.35, "pushed": 0.15,
                   "fell": 0.20, "synthetic": 0.15}
#: Walker rows are logged by spawn speed in these bands (m/s).
SPEED_BANDS = (0.0, 0.5, 1.0, 1.5, float("inf"))
_ROW_FIELDS = ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos",
               "joint_vel", "action")
_BANK_CACHE: dict = {}


def default_dataset() -> str:
  return os.environ.get(
    "SAFE_ZOO_G1_RECOVER_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "data", "g1_recover_states.pt"))


def band(level: torch.Tensor, levels: int, width: float):
  """(lo, hi) quantiles of a source's sorted rows for each ``level``."""
  hi = (level.float() + 1.0) / (levels + 1.0)
  return (hi - width).clamp_min(0.0), hi


def source_rows(bank: dict) -> dict[str, torch.Tensor]:
  """Row indices of each bank source, sorted easy -> hard."""
  kind, fell = bank["kind"], bank["fell"]
  walker = (kind == KIND_CLEAN) | (kind == KIND_PUSHED)

  def sorted_by(mask, key):
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    return idx[torch.argsort(key[idx], stable=True)]

  return {
    "clean": sorted_by((kind == KIND_CLEAN) & ~fell, bank["speed"]),
    "pushed": sorted_by((kind == KIND_PUSHED) & ~fell, bank["push_severity"]),
    "fell": sorted_by(walker & fell, -bank["time_to_fall"]),
    "synthetic": sorted_by(kind == KIND_SYNTHETIC, bank["difficulty"]),
  }


class RecoverState:
  """Per-env bookkeeping shared by the spawn and the curriculum."""

  def __init__(self, num_envs: int, device, levels: int, init_level: int):
    self.levels = int(levels)
    self.level = torch.full((num_envs, len(SOURCES)), int(init_level),
                            dtype=torch.long, device=device)
    self.source = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.speed = torch.zeros(num_envs, device=device)
    self.spawned = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.reset_action: torch.Tensor | None = None


def get_state(env, levels: int = 5, init_level: int = 0) -> RecoverState:
  st = getattr(env, "_recover_state", None)
  if st is None:
    st = RecoverState(env.num_envs, env.device, levels, init_level)
    env._recover_state = st
  return st


def _ids(env, env_ids) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device)
  return env_ids


class RecoverSpawn(ManagerTermBase):
  """Reset event: draw a source and a row per env, place it, seed the action.

  Params (``EventTermCfg.params``): ``dataset`` (bank path; None -> the
  default / ``$SAFE_ZOO_G1_RECOVER_DATASET``), ``weights`` (source -> weight),
  ``levels``, ``band_width``, ``init_level``, ``xy_half`` (planar spawn
  offset, m).
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.levels = int(p.get("levels", 5))
    self.width = float(p.get("band_width", 0.4))
    self.xy_half = float(p.get("xy_half", 0.5))
    self.state = get_state(env, self.levels, int(p.get("init_level", 0)))
    path = p.get("dataset") or default_dataset()
    self.bank, self.rows = None, {}
    if os.path.exists(path):
      key = (os.path.abspath(path), str(env.device))
      if key not in _BANK_CACHE:
        _BANK_CACHE[key] = load_bank(path, env.device)
      self.bank = _BANK_CACHE[key]
      self.rows = source_rows(self.bank)
    else:
      print(f"[g1_recover] WARNING: no start-state bank at {path}; every "
            f"episode spawns in the nominal stance. Harvest one with "
            f"examples/harvest_g1_recover_states.py.")
    weights = dict(DEFAULT_WEIGHTS, **(p.get("weights") or {}))
    unknown = set(weights) - set(SOURCES)
    if unknown:
      raise ValueError(f"g1_recover weights: unknown sources {sorted(unknown)}")
    w = torch.tensor([float(weights[s]) if (s == "nominal" or
                                            self.rows.get(s, torch.empty(0))
                                            .numel() > 0) else 0.0
                      for s in SOURCES], device=env.device)
    if float(w.sum()) <= 0.0:
      w[SRC_NOMINAL] = 1.0
    self.weights = w / w.sum()
    print("[g1_recover] spawn mixture: " + ", ".join(
      f"{s} {float(self.weights[i]):.2f} ({self.rows[s].numel() if s in self.rows else '-'} rows)"
      for i, s in enumerate(SOURCES)))
    robot = env.scene["robot"]
    n = env.num_envs
    self._nominal = {
      "root_z": robot.data.default_root_state[:, 2].clone(),
      "root_quat_h": torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
      .expand(n, 4).clone(),
      "lin_vel_h": torch.zeros(n, 3, device=env.device),
      "ang_vel_h": torch.zeros(n, 3, device=env.device),
      "joint_pos": robot.data.default_joint_pos.clone(),
      "joint_vel": torch.zeros_like(robot.data.default_joint_pos),
    }

  def _reset_action(self, env) -> torch.Tensor:
    st = self.state
    if st.reset_action is None:
      st.reset_action = torch.zeros_like(env.action_manager.action)
      env._zoo_reset_action = st.reset_action
    return st.reset_action

  def __call__(self, env, env_ids, **_params) -> None:
    ids = _ids(env, env_ids)
    n = ids.numel()
    if n == 0:
      return
    dev = env.device
    st = self.state
    src = torch.multinomial(self.weights, n, replacement=True)
    rows = {k: v[ids].clone() for k, v in self._nominal.items()}
    action = torch.zeros(n, self._reset_action(env).shape[1], device=dev)
    for s_name, s_idx in (("clean", SRC_CLEAN), ("pushed", SRC_PUSHED),
                          ("fell", SRC_FELL), ("synthetic", SRC_SYNTHETIC)):
      m = src == s_idx
      if not bool(m.any()):
        continue
      pool = self.rows[s_name]
      lo, hi = band(st.level[ids[m], s_idx], self.levels, self.width)
      pos = lo + torch.rand(lo.shape, device=dev) * (hi - lo)
      pick = pool[(pos * pool.numel()).long().clamp_(0, pool.numel() - 1)]
      for k in rows:
        rows[k][m] = self.bank[k][pick]
      action[m] = self.bank["action"][pick]
    yaw, xy = sample_yaw_xy(n, dev, self.xy_half)
    write_states(env, ids, rows, yaw, xy)
    self._reset_action(env)[ids] = action
    st.source[ids] = src
    st.speed[ids] = torch.linalg.norm(rows["lin_vel_h"][:, :2], dim=1)
    st.spawned[ids] = True

  def reset(self, env_ids=None) -> None:
    """Runs after mjlab's action-manager reset: seed ``last_action``."""
    env = self._env
    ids = _ids(env, env_ids)
    env.action_manager._action[ids] = self._reset_action(env)[ids]


class RecoverCurriculum(ManagerTermBase):
  """Per-env, per-source levels (promote on timeout, demote on fall) + logs.

  mjlab calls it at the START of a reset, before the spawn samples, so the
  outcome it reads (``termination_manager.terminated``) is the episode that is
  ending and the level it moves is the one the next spawn will use.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.state = get_state(env, int(p.get("levels", 5)),
                           int(p.get("init_level", 0)))
    self.ema = float(p.get("ema", 0.002))
    self._surv = torch.full((len(SOURCES),), float("nan"), device=env.device)
    self._band_surv = torch.full((len(SPEED_BANDS) - 1,), float("nan"),
                                 device=env.device)
    self._bands = torch.tensor(SPEED_BANDS, device=env.device)
    self._log: dict = {}

  def _blend(self, avg: torch.Tensor, rate: torch.Tensor,
             count: torch.Tensor) -> torch.Tensor:
    keep = (1.0 - self.ema) ** count
    new = torch.where(torch.isnan(avg), rate, avg * keep + rate * (1.0 - keep))
    return torch.where(count > 0, new, avg)

  def __call__(self, env, env_ids, **_params) -> dict:
    st = self.state
    ids = _ids(env, env_ids)
    ids = ids[st.spawned[ids]]
    if ids.numel() == 0:
      return self._log
    ok = ~env.termination_manager.terminated[ids]
    src = st.source[ids]
    has_level = src != SRC_NOMINAL
    step = torch.where(ok, 1, -1) * has_level.long()
    st.level[ids, src] = (st.level[ids, src] + step).clamp_(0, st.levels)

    ns = len(SOURCES)
    one = torch.nn.functional.one_hot(src, ns).float()
    count = one.sum(0)
    rate = (one * ok.float()[:, None]).sum(0) / count.clamp_min(1.0)
    self._surv = self._blend(self._surv, rate, count)

    walker = (src == SRC_CLEAN) | (src == SRC_PUSHED) | (src == SRC_FELL)
    b = (torch.bucketize(st.speed[ids], self._bands, right=True) - 1).clamp_(
      0, len(SPEED_BANDS) - 2)
    ob = torch.nn.functional.one_hot(b, len(SPEED_BANDS) - 1).float()
    ob = ob * walker.float()[:, None]
    bcount = ob.sum(0)
    brate = (ob * ok.float()[:, None]).sum(0) / bcount.clamp_min(1.0)
    self._band_surv = self._blend(self._band_surv, brate, bcount)

    lvl = st.level.float()
    vals = torch.cat([self._surv, lvl[:, 1:].mean(0),
                      (st.level[:, 1:] == st.levels).float().mean(0),
                      self._band_surv]).tolist()      # one device sync
    keys = ([f"surv_{s}" for s in SOURCES]
            + [f"lvl_{s}" for s in SOURCES[1:]]
            + [f"top_{s}" for s in SOURCES[1:]]
            + [f"surv_v{lo:g}_{hi:g}" for lo, hi in
               zip(SPEED_BANDS[:-1], SPEED_BANDS[1:])])
    self._log = {k: v for k, v in zip(keys, vals) if v == v}   # drop NaN
    return self._log
