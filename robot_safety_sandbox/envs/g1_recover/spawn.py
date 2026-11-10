"""The g1_recover spawn (a mixture over start-state sources) and its curriculum.

SOURCES. Every episode starts from one of seven sources, drawn per env with
fixed mixture weights (renormalized over the sources the bank actually has):

    nominal    the default knees-bent stance at rest (g1_stabilize's spawn)
    clean      walker rows, undisturbed mid-gait        sorted by speed
    pushed     walker rows after a push, walker SURVIVED sorted by push size
    fell       walker rows the walker then FELL from     sorted by time-to-fall,
                                                         longest (easiest) first
    synthetic  stance + random tilt / twist / offsets    sorted by scale
    mocap      kinematic retargeted-mocap frames         sorted by speed
    handover   states captured at a handover to the      sorted by the impulse
               fallback (``KIND_HANDOVER``), possibly    of the push behind
               with a push still acting                  them (NaN -> 0)

A row whose ``start_ok`` is False (``states.OPTIONAL_FIELDS``; True wherever a
bank does not say) belongs to NO source, so no sampler below ever spawns it.
``mocap`` and ``handover`` default to weight 0, which leaves every mixture
written before they existed unchanged -- and means a mocap bank spawns only
the nominal stance until ``weights`` gives ``mocap`` a share (the spawn warns
when a loaded bank leaves the mixture all-nominal). ``dataset`` may be a list
of banks (``states.load_bank``): a walker bank plus a mocap bank feed their
sources side by side.

CURRICULUM. Each env holds one difficulty level per source, 0..L. A level
selects a window of that source's sorted rows (level 0 = the easiest sixth,
level L = the hardest 40%); an episode that ends on the TIMEOUT promotes the
level of the source it spawned from, one that ends on a FALL demotes it. So
an env that keeps falling on fast pushed states backs off to slower ones,
while unrecoverable rows stay in the mix without swamping the batch.

SAMPLERS (``sampler``): how the ROW is chosen once a source is drawn.

    ladder      (default) every source, mocap and handover included, draws
                from its level window as above.
    uniform     clean / pushed / fell / synthetic keep the ladder; mocap and
    weighted    handover rows are drawn by a per-row probability within their
    online      pool instead -- uniform: equal; weighted: w = p (1 - p) with
                p = bank[weight_column] (NaN -> 0.5); online: the same weight on
                a LIVE per-row estimate ``RecoverState.p_row`` initialized from
                that column. p (1 - p) is the FRONTIER weight: a row the policy
                always survives, or never does, carries no gradient toward
                the boundary, a coin-flip row carries the most. The probability
                of row i is (1 - uniform_mix) w_i / sum(w) + uniform_mix / |pool|
                (uniform if sum(w) = 0): the floor keeps every row in play
                however wrong its label is.
    sequential  ignores the mixture and the levels: a cursor walks EVERY
                spawnable row of the bank in bank order, each one
                ``sequential_repeats`` times in a row, cycling forever. Envs
                that reset in the same call take consecutive entries; no env is
                ever given the nominal stance. For labelers (survival per row
                from rollouts), not for training.

ONLINE p. :class:`RecoverCurriculum` runs at the start of a reset, before the
spawn redraws, and moves p_row of the row each ending mocap / handover episode
started from toward that episode's outcome, ``p <- p + online_alpha (ok - p)``
(clamped to [0, 1]; an env pair that ended on the same row in one call both
count). The weights then follow the policy as it improves: the frontier moves
with it instead of staying where the labeling policy left it.

LAST ACTION. A harvested row carries the action the walker last applied. mjlab
zeroes the action buffer AFTER the reset events run, so the spawn writes the
state from ``__call__`` and seeds ``last_action`` from ``reset()``, which the
event manager calls later in the same reset. The first observation then looks
like a real handover. The same per-env action is published as
``env._zoo_reset_action`` for the bridge, whose random-arm generator glides
from it instead of snapping the arms to the default pose (base.py).

PUBLISHED ON THE MJLAB ENV. Set at construction (zeros / -1), then updated IN
PLACE -- the same tensor objects -- for the envs of every reset:

    _zoo_spawn_source     long[n]   each env's source (index into SOURCES)
    _zoo_spawn_row        long[n]   the global bank row of each env's episode
                                    (-1 = the nominal stance)
    _zoo_pending_push     dict      the push a stored row still carries, which
                                    the bridge (base.py) turns into a scripted
                                    push right after the reset:
        force_w   float[n, 3]  N, WORLD frame: the row's ``push_force_h``
                               rotated by the spawn's fresh yaw (the rotation
                               the twist gets, so the push keeps its angle to
                               the body)
        steps     long[n]      control steps it still acts (0 = none; also 0
                               when the stored force is zero)
        consume   bool[n]      True on every KIND_HANDOVER row: the push that
                               made it already happened, so no random push may
                               follow in that episode
    _zoo_n_spawn_sources  int       5 while neither mocap nor handover can be
                                    drawn (an existing run's regime table is
                                    unchanged), else len(SOURCES)
    _zoo_spawn_seq_len    int       (sequential only) spawnable rows x repeats:
                                    the entries of one full pass
    _zoo_spawn_cursor     int       (sequential only) entries issued so far;
                                    grows monotonically, never wrapped

mjlab auto-resets INSIDE ``env.step``, so after a step these already name the
NEW episode of an env that ended in it: a consumer pairing outcomes with rows
copies ``_zoo_spawn_row`` BEFORE the step. Every reset issues sequential
entries, the full resets at construction included, so pair outcomes with those
copied rows; the cursor only says when a full pass has been issued.

LOGGING. :class:`RecoverCurriculum` returns its statistics from ``compute``,
which mjlab logs as ``Curriculum/recover/<key>`` and the bridge forwards:
``surv_<source>`` (survival rate, exponential average over ~1/ema episodes;
mocap / handover included, from their first ended episode on),
``lvl_<source>`` (mean level), ``top_<source>`` (fraction of envs at the top
level) -- for mocap / handover only when they draw by the ladder with a
weight > 0 --, ``surv_v<lo>_<hi>`` (walker rows by spawn speed, m/s) and, under
the weighted / online samplers, ``p_mean_<pool>`` (mean per-row p over the
pool) and ``front_<pool>`` (fraction of the pool with 0.1 < p < 0.9) for each
non-empty mocap / handover pool. Keys stay within SB3's 36-character console
width (which truncates longer ones), except ``p_mean_handover``, shown as
``p_mean_hand...`` there (no collision; TensorBoard / W&B keep it whole).
"""

from __future__ import annotations

import os

import torch

from mjlab.managers.manager_base import ManagerTermBase

from robot_safety_sandbox.envs.g1_recover.states import (
  KIND_CLEAN, KIND_HANDOVER, KIND_MOCAP, KIND_PUSHED, KIND_SYNTHETIC,
  OPTIONAL_FIELDS, heading_to_world, load_bank, sample_yaw_xy, write_states)

SOURCES = ("nominal", "clean", "pushed", "fell", "synthetic", "mocap",
           "handover")
(SRC_NOMINAL, SRC_CLEAN, SRC_PUSHED, SRC_FELL, SRC_SYNTHETIC, SRC_MOCAP,
 SRC_HANDOVER) = range(7)
#: The sources before mocap / handover existed. While neither can be drawn the
#: spawn draws over these five alone, normalized as before -- the same tensor
#: and the same multinomial call, so an existing config's random stream is
#: untouched.
N_BASE_SOURCES = 5
#: The plan's mixture: 15% stance, 35% clean walker, 35% pushed walker (fell
#: rows oversampled: 20 of the 35), 15% synthetic; mocap / handover opt in.
DEFAULT_WEIGHTS = {"nominal": 0.15, "clean": 0.35, "pushed": 0.15,
                   "fell": 0.20, "synthetic": 0.15, "mocap": 0.0,
                   "handover": 0.0}
SAMPLERS = ("ladder", "uniform", "weighted", "online", "sequential")
#: The pools the uniform / weighted / online samplers draw by row.
ROW_SOURCES = ("mocap", "handover")
#: The ``front_<pool>`` band: rows the policy neither always nor never survives.
FRONTIER = (0.1, 0.9)
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
  """Row indices of each bank source, sorted easy -> hard. A row with
  ``start_ok`` False (where the bank has that column) is in none of them."""
  kind, fell = bank["kind"], bank["fell"]
  ok = bank.get("start_ok")
  ok = torch.ones_like(fell, dtype=torch.bool) if ok is None else ok.bool()
  walker = (kind == KIND_CLEAN) | (kind == KIND_PUSHED)
  impulse = bank.get("push_impulse")
  if impulse is None:
    impulse = torch.zeros_like(bank["speed"])
  impulse = torch.where(torch.isnan(impulse), torch.zeros_like(impulse), impulse)

  def sorted_by(mask, key):
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    return idx[torch.argsort(key[idx], stable=True)]

  return {
    "clean": sorted_by((kind == KIND_CLEAN) & ~fell & ok, bank["speed"]),
    "pushed": sorted_by((kind == KIND_PUSHED) & ~fell & ok,
                        bank["push_severity"]),
    "fell": sorted_by(walker & fell & ok, -bank["time_to_fall"]),
    "synthetic": sorted_by((kind == KIND_SYNTHETIC) & ok, bank["difficulty"]),
    "mocap": sorted_by((kind == KIND_MOCAP) & ok, bank["speed"]),
    "handover": sorted_by((kind == KIND_HANDOVER) & ok, impulse),
  }


def frontier_probs(p: torch.Tensor, uniform_mix: float) -> torch.Tensor:
  """Draw probabilities of a pool's rows from their survival estimates ``p``:
  ``(1 - uniform_mix) w / sum(w) + uniform_mix / n`` with the frontier weight
  ``w = p (1 - p)`` (NaN p -> 0.5, p clamped to [0, 1]); uniform when
  ``sum(w) = 0``. float64, sums to 1, and no host sync -- the online sampler
  recomputes it at every draw."""
  p = p.double()
  p = torch.where(torch.isnan(p), torch.full_like(p, 0.5), p).clamp(0.0, 1.0)
  w = p * (1.0 - p)
  n = w.numel()
  s = w.sum()
  mixed = (1.0 - uniform_mix) * w / s.clamp_min(1e-300) + uniform_mix / n
  return torch.where(s > 0.0, mixed, torch.full_like(w, 1.0 / n))


class RecoverState:
  """Per-env bookkeeping shared by the spawn and the curriculum, plus the
  spawn's choices the curriculum's update and logs read."""

  def __init__(self, num_envs: int, device, levels: int, init_level: int):
    self.levels = int(levels)
    self.level = torch.full((num_envs, len(SOURCES)), int(init_level),
                            dtype=torch.long, device=device)
    self.source = torch.zeros(num_envs, dtype=torch.long, device=device)
    # global bank row of each env's current episode (-1 = nominal stance)
    self.row = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self.speed = torch.zeros(num_envs, device=device)
    self.spawned = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.reset_action: torch.Tensor | None = None
    # Set by the spawn when it is built:
    self.p_row: torch.Tensor | None = None     # online: live p per bank row
    self.p_weight: torch.Tensor | None = None  # the p weighted / online reads
    self.pools: dict[str, torch.Tensor] = {}   # row-drawn pools: name -> rows
    self.weights: torch.Tensor | None = None   # the normalized mixture
    self.sampler = "ladder"


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


def _is_path(x) -> bool:
  return isinstance(x, (str, os.PathLike))


class RecoverSpawn(ManagerTermBase):
  """Reset event: draw a source and a row per env, place it, seed the action,
  publish the row and its pending push.

  Params (``EventTermCfg.params``): ``dataset`` (a bank path, or a list of
  paths concatenated in order; None -> the default /
  ``$SAFE_ZOO_G1_RECOVER_DATASET``), ``weights`` (source -> weight),
  ``levels``, ``band_width``, ``init_level``, ``xy_half`` (planar spawn
  offset, m), ``sampler`` (one of :data:`SAMPLERS`), ``uniform_mix``,
  ``weight_column``, ``sequential_repeats`` (module docstring).
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.levels = int(p.get("levels", 5))
    self.width = float(p.get("band_width", 0.4))
    self.xy_half = float(p.get("xy_half", 0.5))
    self.sampler = str(p.get("sampler") or "ladder")
    if self.sampler not in SAMPLERS:
      raise ValueError(f"g1_recover sampler {self.sampler!r}: one of {SAMPLERS}")
    self.uniform_mix = float(p.get("uniform_mix", 0.15))
    if not 0.0 <= self.uniform_mix <= 1.0:
      raise ValueError(f"g1_recover uniform_mix must be in [0, 1], got "
                       f"{self.uniform_mix}")
    self.weight_column = str(p.get("weight_column") or "p_surv")
    self.repeats = int(p.get("sequential_repeats", 1))
    if self.repeats < 1:
      raise ValueError(f"g1_recover sequential_repeats must be >= 1, got "
                       f"{self.repeats}")
    self.state = st = get_state(env, self.levels, int(p.get("init_level", 0)))
    dev = env.device
    dataset = p.get("dataset") or default_dataset()
    paths = [dataset] if _is_path(dataset) else list(dataset)
    missing = [x for x in paths if not os.path.exists(x)]
    self.bank, self.rows = None, {}
    if not missing:
      key = (tuple(os.path.abspath(x) for x in paths), str(dev))
      if key not in _BANK_CACHE:
        _BANK_CACHE[key] = load_bank(dataset if _is_path(dataset) else paths,
                                     dev)
      self.bank = _BANK_CACHE[key]
      self.rows = source_rows(self.bank)
    elif not _is_path(dataset):
      # A list is an explicit choice: never train on part of it silently.
      raise FileNotFoundError(f"g1_recover dataset: no bank at {missing}")
    else:
      print(f"[g1_recover] WARNING: no start-state bank at {dataset}; every "
            f"episode spawns in the nominal stance. Harvest one with "
            f"examples/harvest_g1_recover_states.py.")
    weights = dict(DEFAULT_WEIGHTS, **(p.get("weights") or {}))
    unknown = set(weights) - set(SOURCES)
    if unknown:
      raise ValueError(f"g1_recover weights: unknown sources {sorted(unknown)}")
    w = torch.tensor([float(weights[s]) if (s == "nominal" or
                                            self.rows.get(s, torch.empty(0))
                                            .numel() > 0) else 0.0
                      for s in SOURCES], device=dev)
    if float(w.sum()) <= 0.0:
      w[SRC_NOMINAL] = 1.0
    k = (len(SOURCES) if float(w[N_BASE_SOURCES:].sum()) > 0.0
         else N_BASE_SOURCES)
    self._draw_w = w[:k] / w[:k].sum()      # what multinomial draws over
    self.weights = torch.zeros_like(w)
    self.weights[:k] = self._draw_w
    # The bank sources that can be drawn, in the original loop order.
    self._draw_sources = [(s, i) for i, s in enumerate(SOURCES)
                          if i != SRC_NOMINAL and float(self.weights[i]) > 0.0]

    # The source of every bank row (-1: in no source -- start_ok False or an
    # unknown kind), for the sequential walk and its logged source.
    self._row_src = None
    if self.bank is not None:
      self._row_src = torch.full((int(self.bank["kind"].shape[0]),), -1,
                                 dtype=torch.long, device=dev)
      for i, s in enumerate(SOURCES):
        if s in self.rows:
          self._row_src[self.rows[s]] = i
    self._row_pools: dict[str, torch.Tensor] = {}
    self._cdf: dict[str, torch.Tensor] = {}
    if self.sampler == "sequential":
      self._init_sequential(env)
      emits_new = bool((self._row_src[self._seq_rows] >= N_BASE_SOURCES).any())
    else:
      if self.sampler != "ladder" and self.bank is not None:
        self._init_row_pools()
      emits_new = k > N_BASE_SOURCES
    st.weights, st.sampler = self.weights, self.sampler

    print("[g1_recover] spawn mixture: " + ", ".join(
      f"{s} {float(self.weights[i]):.2f} ({self.rows[s].numel() if s in self.rows else '-'} rows)"
      for i, s in enumerate(SOURCES)))
    if self.sampler == "sequential":
      print(f"[g1_recover] sequential sampler: {self._seq_rows.numel()} "
            f"spawnable rows x {self.repeats} = {self._seq_len} entries per "
            f"pass (mixture and levels ignored)")
    elif self.sampler != "ladder":
      print(f"[g1_recover] sampler {self.sampler}: mocap / handover rows by "
            f"row probability (uniform_mix {self.uniform_mix:g}"
            + ("" if self.sampler == "uniform"
               else f", weight p(1-p) of {self.weight_column!r}") + ")")
    if (self.bank is not None and self.sampler != "sequential"
        and float(self.weights[SRC_NOMINAL]) >= 1.0
        and any(r.numel() > 0 for r in self.rows.values())):
      print(f"[g1_recover] WARNING: the bank at {dataset} has spawnable rows "
            f"(" + ", ".join(f"{s} {r.numel()}" for s, r in self.rows.items()
                             if r.numel() > 0)
            + ") but every source holding them has weight 0, so every episode "
            "spawns in the nominal stance. Give them a weight (e.g. "
            "weights: {mocap: ...} for a mocap bank).")

    # Publish the per-env spawn source for the bridge's regime() (a multi-gamma
    # learner scores each episode with the critic head of its regime). The
    # tensor is updated in place at every reset, so it names the episode each
    # env is in; the source count sizes the regime table before the first reset.
    env._zoo_spawn_source = st.source
    env._zoo_n_spawn_sources = len(SOURCES) if emits_new else N_BASE_SOURCES
    env._zoo_spawn_row = st.row
    n = env.num_envs
    self._pending = {
      "force_w": torch.zeros(n, 3, device=dev),
      "steps": torch.zeros(n, dtype=torch.long, device=dev),
      "consume": torch.zeros(n, dtype=torch.bool, device=dev),
    }
    env._zoo_pending_push = self._pending
    # Whether any row can carry a pending push / consume the random push; a
    # bank without such rows skips the per-reset rotation altogether.
    self._carry = self.bank is not None and bool(
      ((self.bank["kind"] == KIND_HANDOVER)
       | (self.bank["push_left_steps"] > 0)).any())
    robot = env.scene["robot"]
    self._nominal = {
      "root_z": robot.data.default_root_state[:, 2].clone(),
      "root_quat_h": torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
      .expand(n, 4).clone(),
      "lin_vel_h": torch.zeros(n, 3, device=dev),
      "ang_vel_h": torch.zeros(n, 3, device=dev),
      "joint_pos": robot.data.default_joint_pos.clone(),
      "joint_vel": torch.zeros_like(robot.data.default_joint_pos),
    }

  def _init_sequential(self, env) -> None:
    if self.bank is None:
      raise ValueError("g1_recover sampler 'sequential' needs a start-state "
                       "bank (dataset)")
    self._seq_rows = (self._row_src >= 0).nonzero(as_tuple=False).squeeze(-1)
    if self._seq_rows.numel() == 0:
      raise ValueError("g1_recover sampler 'sequential': the bank has no "
                       "spawnable row (every row start_ok False?)")
    self._seq_len = int(self._seq_rows.numel()) * self.repeats
    self._cursor = 0
    env._zoo_spawn_seq_len = self._seq_len
    env._zoo_spawn_cursor = self._cursor

  def _init_row_pools(self) -> None:
    """uniform / weighted / online: the mocap / handover pools, their p and,
    for the fixed weights, their cumulative probabilities (built once)."""
    st = self.state
    self._row_pools = {s: self.rows[s] for s in ROW_SOURCES
                       if self.rows[s].numel() > 0}
    st.pools = dict(self._row_pools)
    if self.sampler == "uniform":
      return
    col = self.bank.get(self.weight_column)
    if col is None:
      if self.weight_column not in OPTIONAL_FIELDS:
        raise ValueError(f"g1_recover weight_column {self.weight_column!r} is "
                         f"not a bank column")
      col = torch.full_like(self.bank["speed"], float("nan"))
    if col.dim() != 1:
      raise ValueError(f"g1_recover weight_column {self.weight_column!r} must "
                       f"be one value per row, has shape {tuple(col.shape)}")
    col = col.float()
    pw = torch.where(torch.isnan(col), torch.full_like(col, 0.5),
                     col).clamp(0.0, 1.0)
    for s, pool in self._row_pools.items():
      if bool(torch.isnan(col[pool]).all()):
        print(f"[g1_recover] WARNING: {self.weight_column!r} is NaN on every "
              f"{s} row (unlabeled bank): the {self.sampler} sampler starts "
              f"{'' if self.sampler == 'online' else 'and stays '}uniform on it")
    if self.sampler == "online":
      st.p_row = pw.clone()
      st.p_weight = st.p_row
    else:
      st.p_weight = pw
      self._cdf = {s: torch.cumsum(frontier_probs(pw[pool], self.uniform_mix), 0)
                   for s, pool in self._row_pools.items()}

  def _reset_action(self, env) -> torch.Tensor:
    st = self.state
    if st.reset_action is None:
      st.reset_action = torch.zeros_like(env.action_manager.action)
      env._zoo_reset_action = st.reset_action
    return st.reset_action

  def _ladder_pick(self, s_idx: int, pool: torch.Tensor,
                   env_ids: torch.Tensor, dev) -> torch.Tensor:
    lo, hi = band(self.state.level[env_ids, s_idx], self.levels, self.width)
    pos = lo + torch.rand(lo.shape, device=dev) * (hi - lo)
    return pool[(pos * pool.numel()).long().clamp_(0, pool.numel() - 1)]

  def _row_pick(self, name: str, k: int) -> torch.Tensor:
    pool = self._row_pools[name]
    n = pool.numel()
    if self.sampler == "uniform":
      j = (torch.rand(k, device=pool.device) * n).long().clamp_(max=n - 1)
      return pool[j]
    cdf = self._cdf.get(name)
    if cdf is None:                       # online: from the live p, every draw
      cdf = torch.cumsum(frontier_probs(self.state.p_row[pool],
                                        self.uniform_mix), 0)
    u = torch.rand(k, dtype=cdf.dtype, device=cdf.device) * cdf[-1]
    return pool[torch.searchsorted(cdf, u, right=True).clamp_(max=n - 1)]

  def _publish_pending(self, ids: torch.Tensor, row: torch.Tensor,
                       yaw: torch.Tensor) -> None:
    pp = self._pending
    if not self._carry:
      pp["force_w"][ids] = 0.0
      pp["steps"][ids] = 0
      pp["consume"][ids] = False
      return
    has = row >= 0
    r = row.clamp_min(0)
    f_w = heading_to_world(self.bank["push_force_h"][r].float(), yaw)
    f_w = f_w * has[:, None]
    steps = self.bank["push_left_steps"][r].clamp_min(0) * has
    steps = torch.where(torch.linalg.norm(f_w, dim=1) > 0.0, steps,
                        torch.zeros_like(steps))
    pp["force_w"][ids] = f_w
    pp["steps"][ids] = steps
    pp["consume"][ids] = has & (self.bank["kind"][r] == KIND_HANDOVER)

  def __call__(self, env, env_ids, **_params) -> None:
    ids = _ids(env, env_ids)
    n = ids.numel()
    if n == 0:
      return
    dev = env.device
    st = self.state
    rows = {k: v[ids].clone() for k, v in self._nominal.items()}
    action = torch.zeros(n, self._reset_action(env).shape[1], device=dev)
    row = torch.full((n,), -1, dtype=torch.long, device=dev)
    if self.sampler == "sequential":
      entry = (torch.arange(n, device=dev) + self._cursor) % self._seq_len
      pick = self._seq_rows[torch.div(entry, self.repeats,
                                      rounding_mode="floor")]
      self._cursor += n
      env._zoo_spawn_cursor = self._cursor
      src = self._row_src[pick]
      rows = {k: self.bank[k][pick] for k in rows}
      action = self.bank["action"][pick]
      row = pick
    else:
      src = torch.multinomial(self._draw_w, n, replacement=True)
      for s_name, s_idx in self._draw_sources:
        m = src == s_idx
        if not bool(m.any()):
          continue
        if s_name in self._row_pools:
          pick = self._row_pick(s_name, int(m.sum()))
        else:
          pick = self._ladder_pick(s_idx, self.rows[s_name], ids[m], dev)
        for k in rows:
          rows[k][m] = self.bank[k][pick]
        action[m] = self.bank["action"][pick]
        row[m] = pick
    yaw, xy = sample_yaw_xy(n, dev, self.xy_half)
    write_states(env, ids, rows, yaw, xy)
    self._reset_action(env)[ids] = action
    st.source[ids] = src
    st.row[ids] = row
    st.speed[ids] = torch.linalg.norm(rows["lin_vel_h"][:, :2], dim=1)
    st.spawned[ids] = True
    self._publish_pending(ids, row, yaw)

  def reset(self, env_ids=None) -> None:
    """Runs after mjlab's action-manager reset: seed ``last_action``."""
    env = self._env
    ids = _ids(env, env_ids)
    env.action_manager._action[ids] = self._reset_action(env)[ids]


class RecoverCurriculum(ManagerTermBase):
  """Per-env, per-source levels (promote on timeout, demote on fall), the
  online sampler's per-row p, and the logs.

  mjlab calls it at the START of a reset, before the spawn samples, so the
  outcome it reads (``termination_manager.terminated``) is the episode that is
  ending, the level it moves is the one the next spawn will use, and
  ``RecoverState.row`` still names the row that episode started from.

  Params (``CurriculumTermCfg.params``): ``levels``, ``init_level``, ``ema``,
  ``sampler`` (p_row moves only under ``"online"``), ``online_alpha``.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.state = get_state(env, int(p.get("levels", 5)),
                           int(p.get("init_level", 0)))
    self.ema = float(p.get("ema", 0.002))
    self.sampler = str(p.get("sampler") or "ladder")
    self.alpha = float(p.get("online_alpha", 0.1))
    if not 0.0 <= self.alpha <= 1.0:
      raise ValueError(f"g1_recover online_alpha must be in [0, 1], got "
                       f"{self.alpha}")
    self._surv = torch.full((len(SOURCES),), float("nan"), device=env.device)
    self._band_surv = torch.full((len(SPEED_BANDS) - 1,), float("nan"),
                                 device=env.device)
    self._bands = torch.tensor(SPEED_BANDS, device=env.device)
    self._extra_lvl: list[int] | None = None
    self._log: dict = {}

  def _blend(self, avg: torch.Tensor, rate: torch.Tensor,
             count: torch.Tensor) -> torch.Tensor:
    keep = (1.0 - self.ema) ** count
    new = torch.where(torch.isnan(avg), rate, avg * keep + rate * (1.0 - keep))
    return torch.where(count > 0, new, avg)

  def _extra_level_sources(self) -> list[int]:
    """mocap / handover get lvl_ / top_ logs only where their level means
    something: drawn by the ladder, with a weight > 0. Resolved once the spawn
    has published its mixture."""
    if self._extra_lvl is None:
      st = self.state
      if st.weights is None:
        return []
      self._extra_lvl = ([] if st.sampler != "ladder" else
                         [i for i in (SRC_MOCAP, SRC_HANDOVER)
                          if float(st.weights[i]) > 0.0])
    return self._extra_lvl

  def _update_p(self, src: torch.Tensor, row: torch.Tensor,
                ok: torch.Tensor) -> None:
    """``p <- p + alpha (ok - p)`` on the rows the ending mocap / handover
    episodes started from; envs that shared a row in this call all count."""
    p_row = self.state.p_row
    m = ((src == SRC_MOCAP) | (src == SRC_HANDOVER)) & (row >= 0)
    r = row.clamp_min(0)
    delta = self.alpha * (ok.float() - p_row[r]) * m.float()
    p_row.index_add_(0, r, delta)
    p_row[r] = p_row[r].clamp_(0.0, 1.0)

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
    if self.sampler == "online" and st.p_row is not None:
      # Only episodes that actually ENDED carry an outcome. A full mj.reset()
      # (the learner's first env.reset() right after construction, a resumed
      # learn(), a script's env.reset()) also lands here for every env with
      # the termination buffers all False, which would book each running
      # episode as a survival. (The level step above has always done the same
      # at a full reset; it is left as it was.)
      ended = env.termination_manager.dones[ids]
      self._update_p(src[ended], st.row[ids][ended], ok[ended])

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
    base = slice(1, N_BASE_SOURCES)
    parts = [self._surv, lvl[:, base].mean(0),
             (st.level[:, base] == st.levels).float().mean(0), self._band_surv]
    keys = ([f"surv_{s}" for s in SOURCES]
            + [f"lvl_{s}" for s in SOURCES[base]]
            + [f"top_{s}" for s in SOURCES[base]]
            + [f"surv_v{lo:g}_{hi:g}" for lo, hi in
               zip(SPEED_BANDS[:-1], SPEED_BANDS[1:])])
    extra = self._extra_level_sources()
    if extra:
      parts += [lvl[:, extra].mean(0),
                (st.level[:, extra] == st.levels).float().mean(0)]
      keys += ([f"lvl_{SOURCES[i]}" for i in extra]
               + [f"top_{SOURCES[i]}" for i in extra])
    if st.p_weight is not None:
      lo_f, hi_f = FRONTIER
      for name, pool in st.pools.items():
        pv = st.p_weight[pool]
        parts += [pv.mean().reshape(1),
                  ((pv > lo_f) & (pv < hi_f)).float().mean().reshape(1)]
        keys += [f"p_mean_{name}", f"front_{name}"]
    vals = torch.cat(parts).tolist()      # one device sync
    self._log = {k: v for k, v in zip(keys, vals) if v == v}   # drop NaN
    return self._log
