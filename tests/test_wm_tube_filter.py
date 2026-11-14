"""CPU checks of the train-time world-model tube filter (filters/wm_tube.py and
filtered_env.WMTubeFilteredEnv), without mjlab:

  * CONDITIONS -- the grammar and canonical names equal the eval's
    (qcbf_eval_lib.parse_condition), refusals included;
  * DWELL -- DwellSwitchIntervention against a literal transcription of
    eval_sonic_qcbf.py:917-933 on random verdict / done streams (64 envs x 5k
    steps, dwell 0 / 3 / 50 steps, both NaN policies): engaged, hand-over,
    held, switch, raw verdict and the action all equal; the strict switch; the
    hold length; reset;
  * MONITOR on the real thermal70_s0 checkpoint: verdicts torch.equal to a
    transcription of E:557-603 / 619-627 (full batch and the compacted ids),
    compact vs full batch, the clamps (proposal box, fallback box, last-action
    columns), ``_actor_mean`` == ``_predict(deterministic=True)``;
  * LOAD -- tag / coverage refusals; python random, numpy, torch and the cudnn
    flags unchanged by ``load_wm_tube`` (SB3's load reseeds them all);
  * WRAPPER on a fake env: executed action, BARE pass-through, filter-state
    critic columns, hand-over holds, the coverage probe's arithmetic, metrics,
    and no RNG drawn by a filtered step.

The package is loaded under a private name through stub packages: importing
``robot_safety_sandbox`` itself pulls in mjlab / warp (minutes on a login node).
The real-checkpoint tests skip when runs/shac_thermal70 is absent. Run with
OMP_NUM_THREADS=1 on a login node (~1 min, most of it torch's import and the
checkpoint load).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
import random
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_ZOO, "robot_safety_sandbox")
sys.path.insert(0, os.path.join(_ZOO, "examples"))


def _stub(name: str, path: str) -> None:
  if name not in sys.modules:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


# a private package name: the real robot_safety_sandbox stays importable in the
# same session, and its eval/__init__ (which imports the bridge) never runs here
_stub("_rss_wmt", _PKG)
_stub("_rss_wmt.eval", os.path.join(_PKG, "eval"))
wt = importlib.import_module("_rss_wmt.filters.wm_tube")
fe = importlib.import_module("_rss_wmt.filtered_env")
F = importlib.import_module("_rss_wmt.filters")
import qcbf_eval_lib as ql  # noqa: E402

RUN = os.path.join(_ZOO, "runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p")
TUBE = os.path.join(_ZOO, "runs/shac_thermal70/wmreport/calib/thermal70_s0/min_tube.json")
TAG = "thermal70_s0"
HAVE_RUN = os.path.isfile(os.path.join(RUN, "final_model.zip")) and os.path.isfile(TUBE)
needs_run = pytest.mark.skipif(not HAVE_RUN, reason="runs/shac_thermal70 s0 not on this machine")


# --- conditions ----------------------------------------------------------------

@pytest.mark.parametrize("text", ["wmmin:1:0.95", "wmmin:1.0:0.950", "wmmin:1:0.95+dwell1",
                                  "wmmin:0.4:0.9+dwell1.0", "wmmin:3:0.99+dwell0.5",
                                  "wmmin:1:0.95+latch", "wmmin:0.4:0.950+latch"])
def test_condition_names_are_the_evals(text):
  ours, theirs = wt.parse_wm_condition(text), ql.parse_condition(text)
  assert ours.name == theirs.name
  assert (ours.horizon_s, ours.coverage, ours.dwell_s, ours.latch) == (
    theirs.horizon_s, theirs.coverage, theirs.dwell_s, theirs.latch)
  assert ours.base == ours.name.split("+")[0]


@pytest.mark.parametrize("text", ["wmmin:1", "wmmin:0:0.95", "wmmin:1:1.0", "wmmin:1:0.95+latchx",
                                  "wmmin:1:0.95+latch+dwell1", "wmmin:1:0.95+dwell1+latch",
                                  "wmmin:1:0.95+dwell0", "wmmin:1:0.95+dwellx",
                                  "wmtube:1:4:0.95", "rollout:1", "wmmin:nan:0.95"])
def test_condition_refusals(text):
  with pytest.raises(ValueError):
    wt.parse_wm_condition(text)


# --- the dwell state machine ---------------------------------------------------

class ScriptedMonitor(wt.Monitor):
  """Returns this step's scripted verdict where active (NaN elsewhere, as the
  WM monitor does for an env it did not roll out); records ``active``."""

  def __init__(self, n):
    super().__init__(n, "cpu", 2)
    self.v, self.active = None, None

  def __call__(self, a, fallback=None, *, active=None, **_):
    self.active = active
    return self.v if active is None else torch.where(active, self.v, math.nan)


def _filter(n, D, nan_engages):
  mon = ScriptedMonitor(n)
  itv = wt.DwellSwitchIntervention(n, "cpu", dwell_steps=D, nan_engages=nan_engages,
                                   action_dim=2)
  a_safe = torch.full((n, 2), 2.0)
  fb = F.PolicyFallback(n, "cpu", lambda **_: a_safe, 2)
  return F.SafetyFilter(fb, mon, itv), mon, itv, a_safe


@pytest.mark.parametrize("D", [0, 3, 50])
@pytest.mark.parametrize("nan_engages", [False, True])
def test_dwell_matches_eval_transcription(D, nan_engages):
  n, T = 64, 5000
  g = torch.Generator().manual_seed(1000 + D + 7 * nan_engages)
  # mostly passing verdicts, failing ones in short bursts, some NaN, some episode ends
  v_all = 0.3 + 0.5 * torch.randn(T, n, generator=g)
  v_all = torch.where(torch.rand(T, n, generator=g) < 0.01, math.nan, v_all)
  done_all = torch.rand(T, n, generator=g) < 0.01
  filt, mon, itv, a_safe = _filter(n, D, nan_engages)
  a_nom = torch.zeros(n, 2)
  # E:917-933, every env tracked; an episode end starts E's fresh per-wave state
  k = torch.zeros(n, dtype=torch.long)
  hold_until = torch.zeros(n, dtype=torch.long)
  prev_eng = torch.zeros(n, dtype=torch.bool)
  for t in range(T):
    raw = v_all[t]
    mon.v = raw
    act, info = filt(a_nom)
    verdict = raw
    if D:
      verdict = torch.where(k < hold_until, torch.full_like(verdict, -1.0), verdict)
    held = k < hold_until
    eng_now = verdict < 0.0                         # qcbf_eval_lib.Controller, :331
    if nan_engages:
      eng_now = eng_now | torch.isnan(verdict)      # the one deliberate deviation
    start = eng_now & ~prev_eng
    if D:
      hold_until = torch.where(start, k + D, hold_until)
    switch = eng_now != prev_eng
    prev_eng = eng_now
    k += 1
    last = itv.last
    assert torch.equal(info.engaged, eng_now), t
    assert torch.equal(last["held"], held), t
    assert torch.equal(last["new_handover"], start), t
    assert torch.equal(last["switch"], switch), t
    assert torch.equal(act, torch.where(eng_now.unsqueeze(-1), a_safe, a_nom)), t
    # held envs are not rolled out: their raw verdict is NaN, the rest the eval's
    assert torch.equal(torch.isnan(last["verdict"]), torch.isnan(raw) | held), t
    fin = ~torch.isnan(last["verdict"])
    assert torch.equal(last["verdict"][fin], raw[fin]), t
    if D:
      assert torch.equal(mon.active, ~held), t
    else:
      assert mon.active is None
    d = done_all[t]
    filt.reset(d)
    k[d], hold_until[d], prev_eng[d] = 0, 0, False


def test_strict_switch_and_nan_policy():
  for nan_engages in (False, True):
    filt, mon, itv, _ = _filter(4, 0, nan_engages)
    mon.v = torch.tensor([0.0, -1e-12, 1e-12, math.nan])
    _, info = filt(torch.zeros(4, 2))
    # verdict 0 passes (the canonical switch would engage at <= 0)
    assert info.engaged.tolist() == [False, True, False, nan_engages]


def test_hold_length_and_reset():
  D = 5
  filt, mon, itv, _ = _filter(2, D, True)
  pattern, left = [], []
  for t in range(12):
    mon.v = torch.tensor([-1.0 if t == 2 else 1.0, 1.0])
    _, info = filt(torch.zeros(2, 2))
    pattern.append(bool(info.engaged[0]))
    left.append(int(itv.hold_left()[0]))
  # the hand-over at step 2 and D - 1 held steps, then the verdict decides again
  assert pattern == [False, False] + [True] * D + [False] * 5
  assert left == [0, 0, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]
  mon.v = torch.tensor([-1.0, -1.0])
  filt(torch.zeros(2, 2))
  assert itv.hold_until.tolist() == [17, 17] and itv.prev_eng.all()
  filt.reset(torch.tensor([True, False]))
  assert itv.k.tolist() == [0, 13] and itv.hold_until.tolist() == [0, 17]
  assert itv.prev_eng.tolist() == [False, True]


# --- the latch -----------------------------------------------------------------

def _latch_filter(n, nan_engages):
  mon = ScriptedMonitor(n)
  itv = wt.DwellSwitchIntervention(n, "cpu", nan_engages=nan_engages, action_dim=2, latch=True)
  a_safe = torch.full((n, 2), 2.0)
  fb = F.PolicyFallback(n, "cpu", lambda **_: a_safe, 2)
  return F.SafetyFilter(fb, mon, itv), mon, itv, a_safe


@pytest.mark.parametrize("nan_engages", [False, True])
def test_latch_matches_eval_transcription(nan_engages):
  """E:977-985 (+latch): ``fired`` overrides the verdict with -1; set by every
  engagement, cleared at an episode end; latched envs are not rolled out."""
  n, T = 64, 3000
  g = torch.Generator().manual_seed(77 + 7 * nan_engages)
  v_all = 0.4 + 0.5 * torch.randn(T, n, generator=g)
  v_all = torch.where(torch.rand(T, n, generator=g) < 0.005, math.nan, v_all)
  done_all = torch.rand(T, n, generator=g) < 0.02
  filt, mon, itv, a_safe = _latch_filter(n, nan_engages)
  a_nom = torch.zeros(n, 2)
  fired = torch.zeros(n, dtype=torch.bool)
  prev_eng = torch.zeros(n, dtype=torch.bool)
  for t in range(T):
    raw = v_all[t]
    mon.v = raw
    act, info = filt(a_nom)
    held = fired.clone()
    verdict = torch.where(fired, torch.full_like(raw, -1.0), raw)
    eng_now = verdict < 0.0
    if nan_engages:
      eng_now = eng_now | torch.isnan(verdict)
    fired |= eng_now
    start = eng_now & ~prev_eng
    switch = eng_now != prev_eng
    prev_eng = eng_now
    last = itv.last
    assert torch.equal(info.engaged, eng_now), t
    assert torch.equal(last["held"], held), t
    assert torch.equal(mon.active, ~held), t
    assert torch.equal(last["new_handover"], start), t
    assert torch.equal(last["switch"], switch), t
    assert torch.equal(act, torch.where(eng_now.unsqueeze(-1), a_safe, a_nom)), t
    assert int(itv.hold_left().abs().sum()) == 0
    d = done_all[t]
    filt.reset(d)
    fired[d], prev_eng[d] = False, False
  # some envs latched and some episodes ran clean (the stream is not degenerate)
  assert 0 < int(fired.sum()) < n


def test_latch_refuses_a_dwell():
  with pytest.raises(ValueError, match="latch"):
    wt.DwellSwitchIntervention(2, "cpu", dwell_steps=3, latch=True, action_dim=2)


# --- precision / RNG / helpers -------------------------------------------------

def test_matmul_precision_restores_either_api():
  m = torch.backends.cuda.matmul
  try:
    m.allow_tf32 = True                    # a trainer-wide TF32 through the legacy flag
    with wt.matmul_precision("fp32"):
      assert m.fp32_precision == "ieee"
    assert m.allow_tf32 is True and torch.get_float32_matmul_precision() == "high"
    m.allow_tf32 = False
    with wt.matmul_precision("fp32"):      # already IEEE: nothing written, legacy getters work
      assert m.fp32_precision != "tf32" and m.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    with wt.matmul_precision("tf32"):
      assert m.fp32_precision == "tf32"
    assert m.allow_tf32 is False and torch.get_float32_matmul_precision() == "highest"
  finally:
    m.fp32_precision = "none"
  with pytest.raises(ValueError):
    with wt.matmul_precision("fp16"):
      pass


def _rng_state():
  return (random.getstate(), np.random.get_state()[1].copy(), torch.random.get_rng_state().clone(),
          torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)


def _same_rng(a, b):
  return (a[0] == b[0] and np.array_equal(a[1], b[1]) and torch.equal(a[2], b[2])
          and a[3:] == b[3:])


def test_preserve_rng_puts_everything_back():
  before = _rng_state()
  with wt.preserve_rng():                  # what SB3's set_random_seed does
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, True
    assert not _same_rng(before, _rng_state())
  assert _same_rng(before, _rng_state())


def test_unwrap_env():
  class Deleg:
    def __init__(self, env):
      self.env = env

    def __getattr__(self, name):
      return getattr(self.__dict__["env"], name)

  class Norm:                              # TensorVecNormalize: .venv, no __getattr__
    def __init__(self, venv):
      self.venv = venv

  base = SimpleNamespace(mj=1)
  inner = Deleg(base)
  chain = Norm(Deleg(inner))
  assert fe.unwrap_env(chain) is base
  assert fe.unwrap_env(chain, Norm) is chain
  assert fe.unwrap_env(chain.venv.env, Deleg) is inner
  assert fe.unwrap_env(chain, dict) is None


def test_jsonable_is_exact_and_address_free():
  @dataclasses.dataclass
  class Cfg:
    a: tuple = (1, 2)
    f: object = None

  x = {"t": torch.tensor([0.1, 2.5], dtype=torch.float32), "c": Cfg(f=len), "s": slice(None),
       "inf": float("inf"), "np": np.float32(0.25), "obj": object()}
  j = wt._jsonable(x)
  assert j["t"] == [float(np.float32(0.1)), 2.5]
  assert j["c"]["a"] == [1, 2] and j["c"]["f"] == "builtins.len"
  assert j["inf"] == "inf" and j["np"] == 0.25 and "0x" not in str(j)


def test_last_action_cols():
  idx = list(range(98)) + list(range(113, 142))
  assert wt._last_action_cols(idx, list(range(69, 98)), 29) == slice(69, 98)
  with pytest.raises(ValueError):
    wt._last_action_cols(idx, list(range(70, 99)), 29)    # 98 is not in the actor obs
  with pytest.raises(ValueError):
    wt._last_action_cols(idx, None, 29)


# --- the real checkpoint -------------------------------------------------------

@pytest.fixture(scope="module")
def loaded():
  if not HAVE_RUN:
    pytest.skip("runs/shac_thermal70 s0 not on this machine")
  torch.manual_seed(1234)
  np.random.seed(1234)
  random.seed(1234)
  before = _rng_state()
  bundle = wt.load_wm_tube(RUN, TUBE, tag=TAG, cond="wmmin:1:0.95", device="cpu")
  return bundle, _same_rng(before, _rng_state())


@needs_run
def test_load_leaves_rng_and_reads_the_tube(loaded):
  bundle, rng_same = loaded
  assert rng_same
  assert bundle.horizon == 50 and bundle.la == slice(69, 98) and bundle.actor_dim == 127
  assert bundle.eps == pytest.approx(0.06616708636283875)
  assert int(bundle.wm.state_dim) == 171 and int(bundle.wm.context_dim) == 33
  assert bundle.z_clip == 20.0 and bundle.meta["tag"] == TAG
  assert bundle.meta["sysid"]["overheat_c"] == 70.0
  assert not bundle.model.policy.training


@needs_run
def test_load_refusals():
  with pytest.raises(ValueError, match="calibrated for"):
    wt.load_wm_tube(RUN, TUBE, tag="thermal70_s1", device="cpu")
  with pytest.raises(ValueError, match="frozen calibrations"):
    wt.load_wm_tube(RUN, TUBE, tag=TAG, cond="wmmin:1:0.99", device="cpu")
  with pytest.raises(ValueError, match="frozen calibrations"):
    wt.load_wm_tube(RUN, TUBE, tag=TAG, cond="wmmin:3:0.95", device="cpu")
  with pytest.raises(ValueError, match="'nn' world model"):
    wt.load_wm_tube(RUN, TUBE, tag=TAG, device="cpu", wm_thermal="hybrid")


HYB_TUBE = os.path.join(_ZOO, "runs/shac_thermal70/hybrid_step0/calib/hybrid/thermal70_s0/"
                              "min_tube.json")


@needs_run
@pytest.mark.skipif(not os.path.isfile(HYB_TUBE), reason="hybrid step-0 calibration absent")
def test_release_round_trip(tmp_path):
  """export_safety_filters.export_run -> load_wm_tube on the copy through its
  manifest (no calibration run reachable by path): the same eps; the hybrid
  tube carries its heat-model record; a tampered tube, a wrong world model and
  a monitor without (or with a stray) heat model are refused."""
  ex = importlib.import_module("export_safety_filters")
  pt = importlib.import_module("_rss_wmt.filters.pretrained")
  man = ex.export_run("thermal70", 0, str(tmp_path), {"commit": "test", "dirty": False})
  assert sorted(man["tubes"]) == ["tubes/joint.json", "tubes/separate.json"]
  p = pt.variant_paths(str(tmp_path), "joint", 0)
  assert p["tag"] == TAG and p["wm_thermal"] == "nn"
  ref = wt.load_wm_tube(RUN, TUBE, tag=TAG, device="cpu")
  b = wt.load_wm_tube(p["run_dir"], p["tube_json"], tag=TAG, device="cpu")
  assert b.eps == ref.eps and b.meta["manifest"] and b.thermal is None
  assert b.meta["zip_sha256"] == ref.meta["zip_sha256"]
  with open(os.path.join(p["run_dir"], "g1_sysid.json")) as f:
    assert "/" not in __import__("json").load(f)["sysid_dir"]
  ps = pt.variant_paths(str(tmp_path), "separate", 0)
  for cond in ("wmmin:1:0.95", "wmmin:0.4:0.95+latch"):
    h = wt.load_wm_tube(ps["run_dir"], ps["tube_json"], tag=TAG, cond=cond, device="cpu",
                        wm_thermal="hybrid")
    assert h.wm_thermal == "hybrid" and h.thermal["cold_c"] == 50.0
    assert h.thermal["limit_c"] == 70.0 and h.thermal["scale_c"] == 10.0
  with pytest.raises(ValueError, match="hybrid"):
    wt.WMTubeMonitor(4, "cpu", h, proposal_bound=6.5, action_dim=29)
  with pytest.raises(ValueError, match="'nn'"):
    wt.WMTubeMonitor(4, "cpu", b, proposal_bound=6.5, action_dim=29,
                     hybrid=SimpleNamespace(mode="hybrid"))
  with pytest.raises(ValueError, match="'hybrid' world model"):
    wt.load_wm_tube(ps["run_dir"], ps["tube_json"], tag=TAG, device="cpu")
  with open(p["tube_json"], "a") as f:
    f.write(" ")
  with pytest.raises(ValueError, match="sha256"):
    wt.load_wm_tube(p["run_dir"], p["tube_json"], tag=TAG, device="cpu")
  with pytest.raises(ValueError, match="variant"):
    pt.variant_paths(str(tmp_path), "fused", 0)


def _states(bundle, n, seed, la_box=6.5):
  """Critic states and contexts from the world model's own normalizer, the
  last-action dims anywhere in the task box (as a task policy leaves them)."""
  g = torch.Generator().manual_seed(seed)
  wm = bundle.wm
  S = wm.s_mean + 0.5 * wm.s_std * torch.randn(n, int(wm.state_dim), generator=g)
  last = bundle.actor_idx[bundle.la]
  S[:, last] = la_box * (2 * torch.rand(n, last.numel(), generator=g) - 1)
  S[:, 6:11] = 0.0                       # command / phase, pinned to 0 in these tasks
  C = wm.c_mean + 0.5 * wm.c_std * torch.randn(n, int(wm.context_dim), generator=g)
  return S, C


def _eval_margins(bundle, S, C, ids, a_first, H):
  """E:557-603 (wm_margins, one noise-free rollout) and E:665-677 (_pi_obs) as
  the eval runs them: policy._predict, the env box +-3 as low / high."""
  m, wm, norm = bundle.model, bundle.wm, bundle.norm
  low, high = -3.0 * torch.ones(29), 3.0 * torch.ones(29)
  s = torch.nan_to_num(S.float())[ids]
  c = C[ids]
  cn, z = wm.encode_context(c), wm.encode(s)
  alive = torch.ones(int(s.shape[0]))
  out = torch.empty(int(s.shape[0]), H)
  act = a_first
  with torch.no_grad():
    for t in range(H):
      st = wm.step(z, act, cn, z_clip=m.wm_z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      out[:, t] = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      if t + 1 < H:
        o = wm.decode(z).index_select(-1, m._actor_idx_t)
        seg = o[:, 69:98]
        o2 = o.clone()
        o2[:, 69:98] = torch.maximum(torch.minimum(seg, high), low)
        act = torch.clamp(m.policy._predict(norm(o2), deterministic=True), low, high)
  return out


def _monitor(bundle, n, **kw):
  return wt.WMTubeMonitor(n, "cpu", bundle, proposal_bound=6.5, action_dim=29, **kw)


@needs_run
def test_monitor_is_the_evals_min_verdict(loaded):
  bundle, _ = loaded
  n = 24
  S, C = _states(bundle, n, 0)
  a = 6.5 * (2 * torch.rand(n, 29, generator=torch.Generator().manual_seed(1)) - 1)
  mon = _monitor(bundle, n)
  ids = torch.arange(n)
  ref = _eval_margins(bundle, S, C, ids, a, 50)
  v = mon(a, wm_state=S, sim_ctx=C)
  assert torch.equal(v, ref.amin(dim=1) - bundle.eps)          # E:619-627
  assert torch.equal(mon.last["pred_min"], ref.amin(dim=1))
  assert torch.equal(mon.margins(None, a, wm_state=S, sim_ctx=C), ref)
  # the compacted batch is the eval's on the same ids
  act = torch.zeros(n, dtype=torch.bool)
  act[::3] = True
  sub = act.nonzero().squeeze(-1)
  v_sub = mon(a, wm_state=S, sim_ctx=C, active=act)
  ref_sub = _eval_margins(bundle, S, C, sub, a[sub], 50)
  assert torch.equal(v_sub[sub], ref_sub.amin(dim=1) - bundle.eps)
  assert torch.isnan(v_sub[~act]).all() and torch.equal(mon.last["computed"], act)
  # ... and the full batch's to rounding
  assert torch.allclose(v_sub[sub], v[sub], atol=1e-5, rtol=0)


@needs_run
def test_compact_and_full_batch_agree(loaded):
  bundle, _ = loaded
  n = 16
  S, C = _states(bundle, n, 3)
  a = 3.0 * (2 * torch.rand(n, 29, generator=torch.Generator().manual_seed(4)) - 1)
  act = torch.rand(n, generator=torch.Generator().manual_seed(5)) < 0.5
  vc = _monitor(bundle, n, mode="compact")(a, wm_state=S, sim_ctx=C, active=act)
  mf = _monitor(bundle, n, mode="full")
  vf = mf(a, wm_state=S, sim_ctx=C, active=act)
  assert torch.equal(torch.isnan(vc), torch.isnan(vf)) and torch.equal(torch.isnan(vc), ~act)
  assert torch.allclose(vc[act], vf[act], atol=1e-5, rtol=0)
  assert not mf.last["done_clamped"][~act].any()


@needs_run
def test_clamps_and_policy_view(loaded):
  bundle, _ = loaded
  m = bundle.model
  n = 8
  S, C = _states(bundle, n, 6)
  A = S[:, bundle.actor_idx]
  # pi == SB3's deterministic prediction, bit for bit
  x = bundle.norm(bundle.pi_obs(A))
  with torch.no_grad():
    assert torch.equal(m._actor_mean(x), m.policy._predict(x, deterministic=True))
  # the fallback's last-action input and its action stay in +-3
  A_hi = A.clone()
  A_hi[:, bundle.la] = 6.5
  A_3 = A.clone()
  A_3[:, bundle.la] = 3.0
  assert torch.equal(bundle.pi_obs(A_hi)[:, bundle.la], torch.full((n, 29), 3.0))
  assert torch.equal(bundle.pi_obs(A_hi)[:, :69], A_hi[:, :69])
  with torch.no_grad():
    assert torch.equal(bundle.pi(A_hi), bundle.pi(A_3))
    wild = bundle.pi(1e3 * torch.randn(n, 127, generator=torch.Generator().manual_seed(7)))
  assert wild.abs().max() <= 3.0 and (wild.abs() == 3.0).any()
  # the first imagined action is the proposal clamped to the proposal box
  mon = _monitor(bundle, n)
  big, box = torch.full((n, 29), 50.0), torch.full((n, 29), 6.5)
  assert torch.equal(mon(big, wm_state=S, sim_ctx=C), mon(box, wm_state=S, sim_ctx=C))
  assert torch.equal(mon.margins(None, big, 5, wm_state=S, sim_ctx=C),
                     mon.margins(None, box, 5, wm_state=S, sim_ctx=C))


# --- the wrapper on a fake env -------------------------------------------------

class FakeEnv:
  """The bridge surface WMTubeFilteredEnv reads: cached groups, sim context,
  dense margins, a critic group; random states and episode ends from its own
  generator (so any draw from torch's global RNG is the wrapper's)."""

  def __init__(self, bundle, n, seed=0, p_done=0.05):
    from gymnasium import spaces
    self.num_envs, self.device = n, "cpu"
    self.mj = SimpleNamespace(step_dt=0.02)
    self.dense_margins, self.dense_reward = True, True
    self.action_space = spaces.Box(-6.5, 6.5, shape=(29,), dtype=np.float32)
    self.critic_observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
    self.b, self.n, self.p_done = bundle, n, p_done
    self.gen = torch.Generator().manual_seed(seed)
    self._C = _states(bundle, n, seed + 1)[1]
    self.applied = []
    self._draw()

  def _draw(self):
    S, _ = _states(self.b, self.n, int(torch.randint(0, 2**30, (1,), generator=self.gen)))
    self._og = {"critic": S, "actor": S[:, self.b.actor_idx],
                "task_critic": torch.randn(self.n, 10, generator=self.gen)}

  def obs_groups(self):
    return self._og

  def sim_context(self):
    return self._C

  def critic_obs(self):
    return self._og["task_critic"]

  def reset(self):
    self._draw()
    return torch.zeros(self.n, 5)

  def step_tensor(self, a):
    self.applied.append(a.clone())
    d = torch.rand(self.n, generator=self.gen) < self.p_done
    tout = d & (torch.rand(self.n, generator=self.gen) < 0.5)
    self._g = torch.randn(self.n, generator=self.gen)
    self._draw()
    return torch.zeros(self.n, 5), torch.zeros(self.n), d, tout, torch.zeros(self.n)

  def safety_margins(self):
    return self._g, torch.zeros_like(self._g)

  def metrics(self):
    return {"safety/failures_total": 0.0}

  def close(self):
    pass


def _short(bundle, H=5):
  """The same checkpoint with a 5-step tube: the wrapper's logic, cheaply."""
  return dataclasses.replace(bundle, horizon=H)


@needs_run
def test_bare_is_a_pass_through(loaded):
  env = FakeEnv(loaded[0], 6)
  w = fe.WMTubeFilteredEnv(env, None)
  assert w.critic_observation_space.shape == (12,)
  w.reset()
  a = 9.0 * torch.ones(6, 29)              # not even clamped: the bare env's own action
  w.step_tensor(a)
  assert w.executed_action is a and env.applied[-1].equal(a)
  co = w.critic_obs()
  assert co.shape == (6, 12) and torch.equal(co[:, :10], env.critic_obs())
  assert not co[:, 10:].any() and not w.last_engaged.any()
  assert torch.isnan(w.last_info["verdict"]).all() and w.finished is None
  assert w.metrics() == {"safety/failures_total": 0.0}


@needs_run
@pytest.mark.parametrize("D", [0, 5])
def test_filtered_wrapper(loaded, D):
  bundle = _short(loaded[0])
  n, T = 16, 60
  env = FakeEnv(bundle, n, seed=D)
  filt, _ = wt.wm_tube_filter(env, dwell_s=D * 0.02, proposal_bound=6.5, bundle=bundle)
  # put eps at the median predicted margin, so about half of the steps engage
  S0, C0 = env.obs_groups()["critic"], env.sim_context()
  filt.monitor.eps = float(filt.monitor.margins(None, torch.zeros(n, 29), wm_state=S0,
                                                sim_ctx=C0).amin(1).median())
  w = fe.WMTubeFilteredEnv(env, filt)
  w.reset()
  gen = torch.Generator().manual_seed(9)
  rng0 = torch.random.get_rng_state().clone()
  eng_hist, new_hist, done_hist = [], [], []
  for t in range(T):
    A = env.obs_groups()["actor"]
    a = 8.0 * (2 * torch.rand(n, 29, generator=gen) - 1)
    with torch.no_grad():
      a_safe = bundle.pi(A)
    w.step_tensor(a)
    eng = w.last_engaged
    a_p = a.clamp(-6.5, 6.5)
    ex = w.executed_action
    assert torch.equal(ex[eng], a_safe[eng]) and torch.equal(ex[~eng], a_p[~eng])
    assert torch.equal(env.applied[-1], ex)
    d = w.finished["done"]
    fo = w.filter_obs()
    assert not fo[d].any()
    assert torch.equal(fo[:, 0], (eng & ~d).float())
    assert torch.equal(w.critic_obs()[:, 10:], fo)
    if D:
      assert torch.equal(torch.isnan(w.last_info["verdict"]), w.last_info["held"])
    eng_hist.append(eng.clone())
    new_hist.append(w.last_info["new_handover"].clone())
    done_hist.append(d.clone())
  # a noise-free filter draws no random numbers
  assert torch.equal(rng0, torch.random.get_rng_state())
  E, N_, Dn = torch.stack(eng_hist), torch.stack(new_hist), torch.stack(done_hist)
  assert E.any() and (~E).any()
  # each hand-over holds for min(D, steps until the episode ends) steps
  for t, e in N_.nonzero().tolist():
    for j in range(t, min(T, t + max(D, 1))):
      assert E[j, e], (t, e, j)
      if Dn[j, e]:
        break
  m = w.metrics()
  assert m["safety/failures_total"] == 0.0
  assert m["safety/engaged_frac"] == pytest.approx(float(E.float().mean()))
  assert m["filter/handovers_per_1k"] == pytest.approx(1000 * float(N_.float().mean()))
  for k in ("filter/hold_frac", "filter/rollout_frac", "filter/verdict_p05",
            "filter/raw_trigger_frac", "filter/episodes", "filter/ms_per_step",
            "filter/fallback_sat_frac"):
    assert k in m, k
  assert "filter/cmd_phase_absmax" not in m              # the fake env has no obs layout
  assert ("filter/cov_n" in m) == (D >= bundle.horizon)
  if D:
    assert m["filter/hold_share"] > 0
  assert "filter/engaged_frac" not in w.metrics()        # drained: an empty window


@needs_run
def test_coverage_probe_arithmetic(loaded):
  bundle = _short(loaded[0], H=4)
  n = 4
  env = FakeEnv(bundle, n)
  filt, _ = wt.wm_tube_filter(env, dwell_s=0.08, proposal_bound=6.5, bundle=bundle)
  w = fe.WMTubeFilteredEnv(env, filt)
  assert w._probe and w._eps == bundle.eps
  w._eps = 0.1
  B = lambda *x: torch.tensor(x, dtype=torch.bool)  # noqa: E731
  Fl = lambda *x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
  no, nan4 = B(0, 0, 0, 0), Fl(*[math.nan] * 4)
  # step 0: envs 0-2 pass a rolled-out verdict (pred_min 0.5); env 3's was not rolled out
  w._probe_step(no, Fl(0.5, 0.5, 0.5, math.nan), Fl(0.6, 0.6, 0.6, 0.6), no, no, no)
  # step 1: hand-overs everywhere; env 3 has no prediction to score
  w._probe_step(B(1, 1, 1, 1), nan4, Fl(0.45, 0.2, 0.3, 0.1), no, no, no)
  assert w._cov["on"].tolist() == [True, True, True, False]
  # step 2 (3 of H = 4 real steps): env 1 falls (its terminal g closes the window),
  # env 2 times out (censored)
  w._probe_step(no, nan4, Fl(0.55, -0.5, 0.4, 0.0), B(0, 1, 1, 0), B(0, 1, 0, 0), B(0, 0, 1, 0))
  assert w._cov["on"].tolist() == [True, False, False, False]
  # step 3: env 0 completes its window; env 3 (no probe) times out
  w._probe_step(no, Fl(0.7, 0.7, 0.7, 0.7), Fl(0.5, 0.1, 0.1, 0.1), B(0, 0, 0, 1), no,
                B(0, 0, 0, 1))
  acc = {k: float(v) for k, v in w._acc.items()}
  # env 0: min real g 0.45 over 0.6, 0.45, 0.55, 0.5 -> r = 0.05 <= eps;
  # env 1: min real g -0.5 -> r = 1.0, and a real violation
  assert acc["cov_n"] == 2 and acc["cov_hit"] == 1 and acc["cov_miss"] == 1
  assert acc["cov_fell"] == 1 and acc["cov_cens"] == 1
  assert not w._cov["on"].any()
  scores = torch.cat(list(w._scores))
  assert sorted(scores[torch.isfinite(scores)].tolist()) == pytest.approx([0.05, 1.0])
  # the next candidate is this step's prediction, cleared where the episode ended
  assert w._cov["cand_pred"].tolist()[:3] == pytest.approx([0.7] * 3)
  assert math.isnan(w._cov["cand_pred"][3])


def test_exports():
  for name in ("WMTubeMonitor", "DwellSwitchIntervention", "wm_tube_filter", "load_wm_tube",
               "check_binding", "fallback_layout", "thermal_world_info", "parse_wm_condition"):
    assert getattr(F, name) is getattr(wt, name)
  assert "WMTubeFilteredEnv" in dir(fe) and "FilteredTensorEnv" in dir(fe)
