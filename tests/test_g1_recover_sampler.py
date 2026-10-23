"""The g1_recover bank and spawn samplers (envs/g1_recover/states.py, spawn.py).

Simulator-free (CPU, a fake mjlab env that records what the spawn writes):

  * the BANK -- one file loads as before (its own meta) with every optional
    field filled; several files concatenate in order with a ``part`` column and
    per-file meta that still resolves clip names; save_bank round-trips;
  * the SOURCES -- mocap rows sorted by speed, handover rows by push impulse
    (NaN -> 0), and ``start_ok`` False keeps a row out of EVERY source;
  * the DEFAULT -- the ladder with mocap / handover at weight 0 reproduces the
    original five-source spawn draw for draw (same random stream);
  * the SAMPLERS -- uniform draws a pool's rows equally; weighted follows the
    mixed frontier formula; online moves the per-row p toward each ending
    episode's outcome before the redraw and draws from the live p; sequential
    walks every spawnable row in bank order with repeats and a cursor;
  * the PENDING PUSH -- a handover row's heading-frame push is rotated by the
    spawn's fresh yaw exactly like its twist, and every handover row consumes
    the episode's random push;
  * the LOGS -- the new keys appear only where they mean something and none
    collides in SB3's console;
  * the CFG -- the sampler kwargs reach the spawn and the curriculum, and the
    rest-target kwargs swap the rest termination's margin.

A GPU test (skipped without CUDA, 8 envs) builds the real env from two banks.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover import spawn as spawn_mod  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.spawn import (  # noqa: E402
  DEFAULT_WEIGHTS, N_BASE_SOURCES, SOURCES, SRC_CLEAN, SRC_FELL, SRC_HANDOVER,
  SRC_MOCAP, SRC_NOMINAL, SRC_PUSHED, SRC_SYNTHETIC, RecoverCurriculum,
  RecoverSpawn, band, frontier_probs, get_state, source_rows)
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_CLEAN, KIND_HANDOVER, KIND_MOCAP, KIND_PUSHED, KIND_SYNTHETIC,
  OPTIONAL_FIELDS, from_heading_frame, heading_to_world, load_bank,
  sample_yaw_xy, save_bank)

NJ = 29
_TMP = tempfile.TemporaryDirectory()
_COUNT = [0]


@pytest.fixture(autouse=True)
def _fresh_cache():
  spawn_mod._BANK_CACHE.clear()
  yield
  spawn_mod._BANK_CACHE.clear()


# --- fakes ----------------------------------------------------------------------

class _Robot:
  """Records the state the spawn writes (write_states' three calls)."""

  def __init__(self, n):
    root = torch.zeros(n, 13)
    root[:, 2], root[:, 3] = 0.76, 1.0
    self.data = SimpleNamespace(default_root_state=root,
                                default_joint_pos=torch.zeros(n, NJ))
    self.pose = torch.zeros(n, 7)
    self.vel = torch.zeros(n, 6)
    self.q = torch.zeros(n, NJ)

  def write_root_link_pose_to_sim(self, pose, env_ids):
    self.pose[env_ids] = pose

  def write_root_link_velocity_to_sim(self, vel, env_ids):
    self.vel[env_ids] = vel

  def write_joint_state_to_sim(self, q, qd, env_ids):
    self.q[env_ids] = q


class _Scene(dict):
  def __init__(self, robot, n):
    super().__init__(robot=robot)
    self.env_origins = torch.zeros(n, 3)


def _env(n):
  return SimpleNamespace(
    num_envs=n, device="cpu", scene=_Scene(_Robot(n), n),
    action_manager=SimpleNamespace(action=torch.zeros(n, NJ),
                                   _action=torch.zeros(n, NJ)),
    termination_manager=SimpleNamespace(
      terminated=torch.zeros(n, dtype=torch.bool),
      dones=torch.zeros(n, dtype=torch.bool)))


def _bank(kinds, **cols) -> dict:
  """A schema-complete bank; ``root_z`` encodes the row (0.5 + 0.001 i)."""
  kinds = torch.as_tensor(kinds, dtype=torch.long)
  n = kinds.numel()
  i = torch.arange(n, dtype=torch.float32)
  bank = {
    "root_z": 0.5 + 0.001 * i,
    "root_quat_h": torch.tensor([1.0, 0, 0, 0]).expand(n, 4).clone(),
    "lin_vel_h": torch.zeros(n, 3), "ang_vel_h": torch.zeros(n, 3),
    "joint_pos": torch.zeros(n, NJ), "joint_vel": torch.zeros(n, NJ),
    "action": (0.01 * i)[:, None].expand(n, NJ).clone(),
    "command": torch.zeros(n, 3),
    "phase_step": torch.zeros(n, dtype=torch.long), "kind": kinds,
    "fell": torch.zeros(n, dtype=torch.bool),
    "time_to_fall": torch.full((n,), math.nan),
    "push_severity": torch.zeros(n), "difficulty": torch.zeros(n),
    "speed": torch.zeros(n), "env_id": torch.zeros(n, dtype=torch.long),
  }
  bank.update(cols)
  return bank


def _path(bank) -> str:
  _COUNT[0] += 1
  path = os.path.join(_TMP.name, f"bank{_COUNT[0]}.pt")
  save_bank(bank, path)
  return path


def _build(bank_or_path, n, curriculum=None, **params):
  env = _env(n)
  path = bank_or_path if isinstance(bank_or_path, (str, list)) else _path(bank_or_path)
  sp = RecoverSpawn(SimpleNamespace(params=dict(dataset=path, **params)), env)
  cur = None
  if curriculum is not None:
    cur = RecoverCurriculum(SimpleNamespace(params=curriculum), env)
  return env, sp, cur


def _only(**w):
  """A mixture with every source at 0 except the named ones."""
  return {s: float(w.get(s, 0.0)) for s in SOURCES}


# --- bank -----------------------------------------------------------------------

def test_one_file_loads_as_before_with_every_optional_field_filled():
  bank = _bank([KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC])
  bank["junk"] = torch.ones(3)                       # unknown column: dropped
  bank["meta"] = {"walker": "w.zip", "seed": 3}
  b = load_bank(_path(bank), "cpu")
  assert b["meta"] == {"walker": "w.zip", "seed": 3}     # exactly the file's
  assert "junk" not in b
  assert set(FIELDS) | set(OPTIONAL_FIELDS) | {"part", "meta"} == set(b)
  for name, (dtype, shape, fill) in OPTIONAL_FIELDS.items():
    col = b[name]
    assert col.dtype == dtype and tuple(col.shape) == (3, *shape), name
    if isinstance(fill, float) and math.isnan(fill):
      assert bool(torch.isnan(col).all()), name
    else:
      assert bool((col == fill).all()), name
  assert b["start_ok"].all() and b["push_force_h"].shape == (3, 3)
  assert b["part"].tolist() == [0, 0, 0]
  assert torch.equal(b["root_z"], bank["root_z"])


def test_several_files_concatenate_in_order_and_keep_per_file_meta():
  walker = _bank([KIND_CLEAN] * 3)
  walker["meta"] = {"walker": "w.zip"}
  mocap = _bank([KIND_MOCAP] * 2, clip=torch.tensor([1, 0]),
                category=torch.tensor([0, 0]), t=torch.tensor([0.1, 0.2]),
                airborne=torch.tensor([True, False]))
  mocap["meta"] = {"clips": ["walk1_subject1", "dance2_subject4"],
                   "categories": ["walk"]}
  pa, pb = _path(walker), _path(mocap)
  b = load_bank([pa, pb], "cpu")
  assert b["kind"].tolist() == [KIND_CLEAN] * 3 + [KIND_MOCAP] * 2
  assert b["part"].tolist() == [0, 0, 0, 1, 1]
  assert b["clip"].tolist() == [-1, -1, -1, 1, 0]        # not remapped
  assert b["airborne"].tolist() == [False, False, False, True, False]
  assert torch.isnan(b["t"][:3]).all() and torch.allclose(b["t"][3:], mocap["t"])
  assert b["meta"] == {"parts": [walker["meta"], mocap["meta"]],
                       "paths": [pa, pb]}
  name = b["meta"]["parts"][int(b["part"][3])]["clips"][int(b["clip"][3])]
  assert name == "dance2_subject4"
  # a list of one path is still the multi-file form
  one = load_bank((pb,), "cpu")
  assert one["meta"] == {"parts": [mocap["meta"]], "paths": [pb]}
  # a merged bank written back keeps resolving its names as a single path
  out = os.path.join(_TMP.name, "deep", "er", "merged.pt")
  save_bank(b, out)
  again = load_bank(out, "cpu")
  assert again["part"].tolist() == [0, 0, 0, 1, 1]
  assert again["meta"] == b["meta"]
  with pytest.raises(ValueError):
    load_bank([], "cpu")


def test_a_wrong_shaped_optional_field_is_refused():
  bad = _bank([KIND_HANDOVER] * 2, push_force_h=torch.zeros(2))   # wants (2, 3)
  with pytest.raises(ValueError, match="push_force_h"):
    load_bank(_path(bad), "cpu")


def test_save_bank_moves_tensors_to_the_cpu_and_creates_dirs():
  bank = _bank([KIND_CLEAN])
  bank["meta"] = {"k": 1}
  out = os.path.join(_TMP.name, "a", "b", "c.pt")
  save_bank(bank, out)
  raw = torch.load(out, weights_only=False)
  assert raw["meta"] == {"k": 1} and raw["root_z"].device.type == "cpu"


def test_heading_to_world_is_the_twist_rotation():
  torch.manual_seed(0)
  v, yaw = torch.randn(16, 3), (torch.rand(16) * 2 - 1) * math.pi
  q = torch.tensor([1.0, 0, 0, 0]).expand(16, 4)
  _q, v_w, _w = from_heading_frame(q, v, torch.zeros(16, 3), yaw)
  assert torch.allclose(heading_to_world(v, yaw), v_w)


# --- sources --------------------------------------------------------------------

def test_mocap_and_handover_sources_sorted_and_start_ok_respected():
  kinds = [KIND_CLEAN, KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC,
           KIND_MOCAP, KIND_MOCAP, KIND_MOCAP,
           KIND_HANDOVER, KIND_HANDOVER, KIND_HANDOVER, KIND_HANDOVER]
  b = _bank(kinds,
            speed=torch.tensor([0.5, 0.1, 0, 0, 2.0, 0.3, 1.0, 0, 0, 0, 0]),
            push_impulse=torch.tensor([math.nan] * 7 + [30.0, math.nan, 10.0, 5.0]),
            start_ok=torch.tensor([True, False, True, True, True, True, False,
                                   True, True, True, False]))
  rows = source_rows(b)
  assert rows["clean"].tolist() == [0]                    # row 1 not spawnable
  assert rows["pushed"].tolist() == [2] and rows["synthetic"].tolist() == [3]
  assert rows["mocap"].tolist() == [5, 4]                 # by speed, 6 out
  assert rows["handover"].tolist() == [8, 9, 7]           # NaN impulse = 0; 10 out
  # a bank without the optional columns (the walker tests' fake) still works
  assert source_rows({k: v for k, v in b.items()
                      if k not in ("start_ok", "push_impulse")})["clean"].tolist() == [1, 0]


# --- the default is the original spawn ------------------------------------------

def _walker_bank_with_mocap():
  g = torch.Generator().manual_seed(1)
  kinds = ([KIND_CLEAN] * 12 + [KIND_PUSHED] * 12 + [KIND_SYNTHETIC] * 8
           + [KIND_MOCAP] * 10)
  n = len(kinds)
  fell = torch.zeros(n, dtype=torch.bool)
  fell[[2, 5, 13, 17, 20]] = True
  return _bank(kinds, fell=fell, speed=torch.rand(n, generator=g),
               push_severity=torch.rand(n, generator=g),
               difficulty=torch.rand(n, generator=g),
               time_to_fall=torch.where(fell, torch.rand(n, generator=g),
                                        torch.full((n,), math.nan)))


def _original_draw(bank, weights5, level, ids, levels=5, width=0.4, xy_half=0.5):
  """The five-source spawn as it was, draw for draw."""
  rows = source_rows(bank)
  n = ids.numel()
  src = torch.multinomial(weights5, n, replacement=True)
  pick = torch.full((n,), -1, dtype=torch.long)
  for s_name, s_idx in (("clean", SRC_CLEAN), ("pushed", SRC_PUSHED),
                        ("fell", SRC_FELL), ("synthetic", SRC_SYNTHETIC)):
    m = src == s_idx
    if not bool(m.any()):
      continue
    pool = rows[s_name]
    lo, hi = band(level[ids[m], s_idx], levels, width)
    pos = lo + torch.rand(lo.shape) * (hi - lo)
    pick[m] = pool[(pos * pool.numel()).long().clamp_(0, pool.numel() - 1)]
  _yaw, xy = sample_yaw_xy(n, "cpu", xy_half)
  return src, pick, xy


def test_default_ladder_reproduces_the_original_random_stream():
  bank = _walker_bank_with_mocap()
  env, sp, _ = _build(bank, 64)
  rows = source_rows(bank)
  w5 = torch.tensor([float(DEFAULT_WEIGHTS[s]) if (s == "nominal" or
                                                   rows[s].numel() > 0) else 0.0
                     for s in SOURCES[:N_BASE_SOURCES]])
  w5 = w5 / w5.sum()
  assert torch.equal(sp._draw_w, w5)                # the same tensor, bitwise
  st = get_state(env)
  st.level[:] = torch.randint(0, 6, st.level.shape, generator=torch.Generator().manual_seed(2))
  robot = env.scene["robot"]
  for seed, ids in ((0, torch.arange(64)), (1, torch.tensor([3, 9, 40])),
                    (2, torch.arange(0, 64, 2))):
    torch.manual_seed(seed)
    sp(env, ids)
    torch.manual_seed(seed)
    src, pick, xy = _original_draw(bank, w5, st.level, ids)
    assert torch.equal(st.source[ids], src)
    assert torch.equal(st.row[ids], pick)
    assert torch.equal(robot.pose[ids, :2], xy)
    nominal = pick < 0
    assert torch.allclose(robot.pose[ids[~nominal], 2], bank["root_z"][pick[~nominal]])
  assert not bool((st.source == SRC_MOCAP).any())   # weight 0: never drawn
  assert env._zoo_n_spawn_sources == N_BASE_SOURCES  # regime tables unchanged
  assert env._zoo_spawn_row is st.row
  pp = env._zoo_pending_push
  assert not bool(pp["consume"].any()) and int(pp["steps"].abs().sum()) == 0
  assert not hasattr(env, "_zoo_spawn_seq_len")


def test_a_mocap_bank_with_default_weights_spawns_nominal_and_warns(capsys):
  env, sp, _ = _build(_bank([KIND_MOCAP] * 4), 8)
  sp(env, torch.arange(8))
  assert bool((get_state(env).source == SRC_NOMINAL).all())
  assert "WARNING" in capsys.readouterr().out
  with pytest.raises(FileNotFoundError):
    _build([os.path.join(_TMP.name, "nope.pt")], 4)


# --- samplers -------------------------------------------------------------------

def test_uniform_sampler_draws_every_pool_row_equally():
  kinds = [KIND_CLEAN] * 2 + [KIND_MOCAP] * 4
  bank = _bank(kinds, speed=torch.tensor([0.0, 0.0, 0.4, 0.1, 0.3, 0.2]))
  env, sp, _ = _build(bank, 500, weights=_only(mocap=1.0), sampler="uniform")
  st = get_state(env)
  # level 0 would confine the ladder to the easiest row alone (index 0 of 4)
  st.level[:] = 0
  counts = torch.zeros(len(kinds))
  for _ in range(40):
    sp(env, torch.arange(500))
    assert bool((st.source == SRC_MOCAP).all())
    counts += torch.bincount(st.row, minlength=len(kinds)).float()
  freq = counts[2:] / counts.sum()
  assert counts[:2].sum() == 0
  assert torch.allclose(freq, torch.full((4,), 0.25), atol=0.02)


def _expected_probs(p, mix):
  p = [0.5 if math.isnan(x) else x for x in p]
  w = [x * (1 - x) for x in p]
  s = sum(w)
  if s == 0:
    return torch.full((len(p),), 1.0 / len(p), dtype=torch.float64)
  return torch.tensor([(1 - mix) * wi / s + mix / len(p) for wi in w],
                      dtype=torch.float64)


def test_frontier_probs_formula():
  p = [0.5, 0.1, 0.9, 0.0, 1.0, math.nan, 0.3]
  got = frontier_probs(torch.tensor(p), 0.2)
  assert got.dtype == torch.float64 and abs(float(got.sum()) - 1.0) < 1e-12
  assert torch.allclose(got, _expected_probs(p, 0.2))
  # no frontier at all -> uniform, whatever the mix
  assert torch.allclose(frontier_probs(torch.tensor([0.0, 1.0, 1.0]), 0.0),
                        torch.full((3,), 1 / 3, dtype=torch.float64))
  # p outside [0, 1] is clamped, so w never goes negative
  assert bool((frontier_probs(torch.tensor([1.5, -0.2, 0.5]), 0.0) >= 0).all())


def test_weighted_sampler_follows_the_mixed_frontier_formula():
  p = [0.5, 0.1, 0.9, 0.0, 1.0, math.nan, 0.3]
  kinds = [KIND_CLEAN] + [KIND_MOCAP] * len(p)
  bank = _bank(kinds, p_surv=torch.tensor([0.5] + p))
  n, calls = 1000, 60
  env, sp, _ = _build(bank, n, weights=_only(mocap=1.0), sampler="weighted",
                      uniform_mix=0.2)
  st = get_state(env)
  counts = torch.zeros(len(kinds))
  for _ in range(calls):
    sp(env, torch.arange(n))
    counts += torch.bincount(st.row, minlength=len(kinds)).float()
  pool = source_rows(bank)["mocap"]                 # speed ties: bank order
  freq = (counts[pool] / counts.sum()).double()
  want = _expected_probs([p[int(r) - 1] for r in pool], 0.2)
  assert torch.allclose(freq, want, atol=0.01)
  assert float(freq.min()) > 0.0                    # the floor keeps p=0/1 rows
  assert st.p_row is None                           # fixed weights
  # no floor: rows the label calls certain are never drawn
  env, sp, _ = _build(bank, n, weights=_only(mocap=1.0), sampler="weighted",
                      uniform_mix=0.0)
  for _ in range(10):
    sp(env, torch.arange(n))
    rows = get_state(env).row
    assert not bool(((rows == 4) | (rows == 5)).any())  # p = 0.0 and 1.0


def test_weighted_sampler_rejects_an_unknown_weight_column():
  with pytest.raises(ValueError, match="weight_column"):
    _build(_bank([KIND_MOCAP] * 3), 4, weights=_only(mocap=1.0),
           sampler="weighted", weight_column="p_typo")
  with pytest.raises(ValueError, match="sampler"):
    _build(_bank([KIND_MOCAP] * 3), 4, sampler="random")


def test_online_sampler_moves_p_toward_outcomes_before_the_redraw():
  kinds = [KIND_CLEAN] * 2 + [KIND_MOCAP] * 4 + [KIND_HANDOVER] * 2
  p0 = torch.tensor([math.nan, math.nan, 0.5, 0.5, math.nan, 0.2, 0.9, 0.4])
  bank = _bank(kinds, p_surv=p0)
  n, alpha = 48, 0.5
  env, sp, cur = _build(bank, n, curriculum=dict(sampler="online",
                                                 online_alpha=alpha),
                        weights=_only(clean=0.2, mocap=0.4, handover=0.4),
                        sampler="online")
  st = get_state(env)
  init = torch.where(torch.isnan(p0), torch.full_like(p0, 0.5), p0)
  assert torch.equal(st.p_row, init) and st.p_weight is st.p_row
  torch.manual_seed(0)
  sp(env, torch.arange(n))
  ended_rows, ended_src = st.row.clone(), st.source.clone()
  # a FULL reset before any step (the learner's first env.reset()): the
  # termination buffers are all False, nothing ended, p must not move
  cur(env, torch.arange(n))
  assert torch.equal(st.p_row, init)
  term = torch.rand(n) < 0.4
  env.termination_manager.terminated = term
  env.termination_manager.dones = torch.ones(n, dtype=torch.bool)
  # one mjlab reset: the curriculum first (ending rows), then the spawn redraws
  cur(env, torch.arange(n))
  log = cur._log
  sp(env, torch.arange(n))
  want = init.clone()
  delta = torch.zeros_like(init)
  for e in range(n):
    if int(ended_src[e]) in (SRC_MOCAP, SRC_HANDOVER):
      r = int(ended_rows[e])
      delta[r] += alpha * (float(~term[e]) - init[r])
  want = (want + delta).clamp(0.0, 1.0)
  assert torch.allclose(st.p_row, want, atol=1e-6)
  assert torch.equal(st.p_row[:2], init[:2])        # walker rows never move
  assert not torch.equal(st.p_row, init)
  pm = st.p_row[source_rows(bank)["mocap"]]
  assert log["p_mean_mocap"] == pytest.approx(float(pm.mean()), abs=1e-6)
  assert log["front_mocap"] == pytest.approx(
    float(((pm > 0.1) & (pm < 0.9)).float().mean()), abs=1e-6)
  assert "p_mean_handover" in log and "front_handover" in log
  # the draw reads the LIVE p: rows driven to certainty drop out (no floor)
  env, sp, cur = _build(bank, n, curriculum=dict(sampler="online"),
                        weights=_only(mocap=1.0), sampler="online",
                        uniform_mix=0.0)
  st = get_state(env)
  st.p_row[2], st.p_row[3] = 1.0, 0.0
  for _ in range(10):
    sp(env, torch.arange(n))
    assert not bool(((st.row == 2) | (st.row == 3)).any())
  # a successful row climbs toward 1 under repeated outcomes
  st.row[:] = 4
  st.source[:] = SRC_MOCAP
  env.termination_manager.terminated = torch.zeros(n, dtype=torch.bool)
  env.termination_manager.dones = torch.ones(n, dtype=torch.bool)
  cur(env, torch.arange(1))
  assert float(st.p_row[4]) == pytest.approx(0.5 + 0.1 * 0.5)


def test_sequential_sampler_walks_every_spawnable_row_with_repeats():
  kinds = [KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC, KIND_MOCAP, KIND_MOCAP,
           KIND_HANDOVER, KIND_CLEAN]
  bank = _bank(kinds, fell=torch.tensor([False, True, False, False, False,
                                         False, False]),
               start_ok=torch.tensor([True, True, True, True, False, True,
                                      False]))
  env, sp, _ = _build(bank, 4, weights=_only(nominal=1.0),   # ignored
                      sampler="sequential", sequential_repeats=2)
  st = get_state(env)
  assert env._zoo_spawn_seq_len == 10 and env._zoo_spawn_cursor == 0
  assert env._zoo_n_spawn_sources == len(SOURCES)
  issued = []
  for ids in (torch.arange(4), torch.tensor([2]), torch.tensor([0, 3]),
              torch.arange(4)):
    sp(env, ids)
    issued += st.row[ids].tolist()
    assert env._zoo_spawn_cursor == len(issued)
  # spawnable rows [0, 1, 2, 3, 5], each twice, cycling; never nominal
  assert issued == [0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 0]
  assert st.row.tolist() == [3, 5, 5, 0]
  src = {0: SRC_CLEAN, 1: SRC_FELL, 2: SRC_SYNTHETIC, 3: SRC_MOCAP,
         5: SRC_HANDOVER}
  assert st.source.tolist() == [src[r] for r in st.row.tolist()]
  robot = env.scene["robot"]
  assert torch.allclose(robot.pose[:, 2], bank["root_z"][st.row])
  assert torch.allclose(env._recover_state.reset_action,
                        bank["action"][st.row])
  with pytest.raises(ValueError):
    _build(_bank([KIND_CLEAN] * 2, start_ok=torch.zeros(2, dtype=torch.bool)),
           2, sampler="sequential")


# --- pending push ---------------------------------------------------------------

def test_pending_push_rotates_with_the_spawn_yaw_and_handover_consumes():
  force_h = torch.tensor([[0.0, 0, 0]] * 3 + [[100.0, 0, 0], [0, -60.0, 0],
                                              [0.0, 0, 0]])
  kinds = [KIND_MOCAP] * 3 + [KIND_HANDOVER] * 3
  bank = _bank(kinds, push_force_h=force_h,
               push_left_steps=torch.tensor([0, 0, 0, 5, 3, 4]),
               push_impulse=torch.tensor([math.nan] * 3 + [20.0, 12.0, 0.0]),
               lin_vel_h=force_h / 100.0)            # rotated the same way
  n = 256
  # uniform: every handover row is drawn (the ladder's level 0 would keep to
  # the lowest-impulse one)
  env, sp, _ = _build(bank, n, weights=_only(nominal=0.3, mocap=0.3,
                                             handover=0.4), sampler="uniform")
  pp = env._zoo_pending_push
  tensors = (pp["force_w"], pp["steps"], pp["consume"], env._zoo_spawn_row)
  torch.manual_seed(0)
  sp(env, torch.arange(n))
  assert tensors == (pp["force_w"], pp["steps"], pp["consume"],
                     env._zoo_spawn_row)             # updated in place
  st = get_state(env)
  row = st.row
  robot = env.scene["robot"]
  ho = row >= 3
  assert bool(ho.any()) and bool((row < 0).any()) and bool(((row >= 0) & ~ho).any())
  # world force = the stored heading-frame force at the env's fresh yaw
  assert torch.allclose(pp["force_w"][ho], 100.0 * robot.vel[ho, :3], atol=1e-4)
  assert torch.allclose(pp["force_w"][ho].norm(dim=1),
                        force_h[row[ho]].norm(dim=1), atol=1e-4)
  r100 = row == 3
  assert float(pp["force_w"][r100, 0].std()) > 1.0   # a fresh yaw per env
  assert pp["steps"][ho].tolist() == [{3: 5, 4: 3, 5: 0}[int(r)] for r in row[ho]]
  assert bool(pp["consume"][ho].all())               # incl. the zero-force row
  assert not bool(pp["consume"][~ho].any())
  assert float(pp["force_w"][~ho].abs().sum()) == 0.0
  assert int(pp["steps"][~ho].abs().sum()) == 0
  assert env._zoo_n_spawn_sources == len(SOURCES)


# --- logs -----------------------------------------------------------------------

def _console_ok(log):
  """SB3's console shows '   Curriculum/recover/<key>' cut to 36 chars and
  raises on two keys that cut to the same string."""
  shown = [(f"   Curriculum/recover/{k}"[:33] + "...")
           if len(f"   Curriculum/recover/{k}") > 36
           else f"   Curriculum/recover/{k}" for k in log]
  return len(set(shown)) == len(shown)


def test_new_logs_appear_only_where_they_mean_something():
  kinds = [KIND_CLEAN] * 4 + [KIND_MOCAP] * 4 + [KIND_HANDOVER] * 3
  bank = _bank(kinds, p_surv=torch.tensor([math.nan] * 4 + [0.5, 0.05, 0.95, 0.3]
                                          + [0.5, 0.5, 0.0]))
  old = ({f"surv_{s}" for s in SOURCES[:N_BASE_SOURCES]}
         | {f"lvl_{s}" for s in SOURCES[1:N_BASE_SOURCES]}
         | {f"top_{s}" for s in SOURCES[1:N_BASE_SOURCES]})
  torch.manual_seed(0)
  # ladder, mocap weighted in, handover not
  env, sp, cur = _build(bank, 64, curriculum={},
                        weights=_only(nominal=0.2, clean=0.4, mocap=0.4))
  sp(env, torch.arange(64))
  log = cur(env, torch.arange(64))
  new = set(log) - old - {k for k in log if k.startswith("surv_v")}
  assert {"surv_mocap", "lvl_mocap", "top_mocap"} <= new
  assert not ({"lvl_handover", "top_handover", "surv_handover", "p_mean_mocap"}
              & set(log))
  # weighted: no level logs for row-drawn pools, p logs for both pools
  env, sp, cur = _build(bank, 64, curriculum=dict(sampler="weighted"),
                        weights=_only(clean=0.2, mocap=0.4, handover=0.4),
                        sampler="weighted")
  sp(env, torch.arange(64))
  log = cur(env, torch.arange(64))
  assert {"p_mean_mocap", "front_mocap", "p_mean_handover", "front_handover",
          "surv_mocap", "surv_handover"} <= set(log)
  assert "lvl_mocap" not in log and "lvl_handover" not in log
  assert log["front_mocap"] == pytest.approx(0.5)   # 0.5 and 0.3 of 4
  assert log["p_mean_handover"] == pytest.approx(1.0 / 3.0)
  new = set(log) - old - {k for k in log if k.startswith("surv_v")}
  assert all(len(k) <= 17 for k in new)
  # every key but p_mean_handover fits SB3's console; none collides
  assert all(3 + len(f"Curriculum/recover/{k}") <= 36
             for k in log if k != "p_mean_handover")
  assert _console_ok(log)


# --- cfg ------------------------------------------------------------------------

def test_cfg_forwards_the_sampler_kwargs():
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  cfg = g1_recover_env_cfg(dataset=("a.pt", "b.pt"), sampler="online",
                           uniform_mix=0.3, weight_column="p_value",
                           online_alpha=0.2, sequential_repeats=3)
  sp = cfg.events["recover_spawn"].params
  assert sp["dataset"] == ["a.pt", "b.pt"] and sp["sampler"] == "online"
  assert sp["uniform_mix"] == 0.3 and sp["weight_column"] == "p_value"
  assert sp["sequential_repeats"] == 3
  cu = cfg.curriculum["recover"].params
  assert cu["sampler"] == "online" and cu["online_alpha"] == 0.2
  d = g1_recover_env_cfg().events["recover_spawn"].params
  assert d["sampler"] == "ladder" and d["uniform_mix"] == 0.15
  assert d["weight_column"] == "p_surv" and d["sequential_repeats"] == 1
  assert d["dataset"] is None
  with pytest.raises(ValueError):
    g1_recover_env_cfg(sampler="nope")


def _standing(vx, quiet=True):
  data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.76]]),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    root_link_lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]),
    root_link_ang_vel_b=torch.zeros(1, 3))
  env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
  env._zoo_push_quiet = torch.tensor([quiet])
  return env


def test_cfg_rest_target_kwargs_swap_the_rest_margin():
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  plain = g1_recover_env_cfg(rest_terminate=True).terminations["recovered"]
  assert "margin_fn" not in plain.params              # today's stance_margins
  loose = g1_recover_env_cfg(rest_terminate=True, rest_lin_tol=0.5,
                             rest_ang_tol=0.6).terminations["recovered"]
  fn = loose.params["margin_fn"]
  assert fn.has_target and loose.time_out is True
  assert float(fn(_standing(0.4))[1]) > 0.0           # 0.4 m/s inside 0.5
  gated = g1_recover_env_cfg(rest_terminate=True, rest_gate_on_push=True)
  fn = gated.terminations["recovered"].params["margin_fn"]
  assert float(fn(_standing(0.4))[1]) < 0.0           # default 0.25 m/s
  assert float(fn(_standing(0.0))[1]) > 0.0
  assert float(fn(_standing(0.0, quiet=False))[1]) < 0.0   # push not done
  # the kwargs do nothing without the rest termination
  assert "recovered" not in g1_recover_env_cfg(rest_lin_tol=0.5).terminations


# --- GPU: the real env ----------------------------------------------------------

_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA, reason="needs CUDA (mjlab/mujoco_warp)")
def test_real_env_spawns_from_two_banks_and_publishes_rows_and_pushes():
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_recover.states import snapshot
  dev = "cuda:0"
  with tempfile.TemporaryDirectory() as tmp:
    probe = make_tensor("g1_recover", 2, dev,
                        cfg_overrides={"dataset": os.path.join(tmp, "none.pt")})
    probe.reset()
    default_q = probe.mj.scene["robot"].data.default_joint_pos[0].cpu()
    probe.close()
    walker = _bank([KIND_CLEAN] * 4, joint_pos=default_q.expand(4, NJ).clone(),
                   root_z=torch.full((4,), 0.76))
    force_h = torch.tensor([[80.0, 0.0, 0.0]]).expand(4, 3).clone()
    other = _bank([KIND_MOCAP] * 2 + [KIND_HANDOVER] * 2,
                  joint_pos=default_q.expand(4, NJ).clone(),
                  root_z=torch.full((4,), 0.76),
                  lin_vel_h=torch.tensor([[0.5, 0.0, 0.0]]).expand(4, 3).clone(),
                  push_force_h=force_h * torch.tensor([0, 0, 1, 1.0])[:, None],
                  push_left_steps=torch.tensor([0, 0, 6, 6]),
                  p_surv=torch.tensor([0.5, 0.5, 0.5, 0.5]))
    pa, pb = os.path.join(tmp, "w.pt"), os.path.join(tmp, "o.pt")
    save_bank(walker, pa)
    save_bank(other, pb)
    env = make_tensor("g1_recover", 8, dev, cfg_overrides={
      "dataset": [pa, pb], "sampler": "weighted",
      "weights": {"nominal": 0, "clean": 0, "pushed": 0, "fell": 0,
                  "synthetic": 0, "mocap": 0.5, "handover": 0.5}})
    env.reset()
    mj = env.mj
    row = mj._zoo_spawn_row
    assert bool((row >= 4).all())                    # only the second file's rows
    src = mj._recover_state.source
    assert bool(((src == SRC_MOCAP) == (row < 6)).all())
    pp = mj._zoo_pending_push
    ho = row >= 6
    assert bool((pp["consume"] == ho).all())
    assert bool((pp["steps"][ho] == 6).all()) and int(pp["steps"][~ho].sum()) == 0
    # the push keeps its heading-frame angle: along the restored twist
    s = snapshot(mj)
    lin_w = mj.scene["robot"].data.root_link_lin_vel_w
    cos = torch.nn.functional.cosine_similarity(pp["force_w"][ho], lin_w[ho], dim=1)
    assert bool((cos > 0.99).all())
    assert torch.allclose(s["lin_vel_h"][:, 0], torch.full((8,), 0.5, device=dev),
                          atol=1e-3)
    for _ in range(5):
      env.step_tensor(torch.zeros(8, env.action_space.shape[0], device=dev))
    assert mj._zoo_spawn_row is row                  # still the same tensor
    env.close()
