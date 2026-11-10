"""The Q-CBF filter's runtime values (envs/g1_recover/policy_value_filter.py).

CPU and simulator-free. The value tests load the REAL fits in
data/policy_eval_gfall and are skipped without them. Only the test splits
(test_{v,q}.pt, a few MB) and their labels are read, never the train files.

  * CHECKPOINTS -- PolicyValues loads {v,q}_<tag>.pt (exact) and
    {v,q}_td_g099_<tag>.pt, checks the input widths, exposes eps and gamma and
    freezes the nets; a missing variant fails loudly;
  * FEATURES -- state_features of the held-out rows (load_bank) is
    policy_eval.features (the fits' input) bit for bit; runtime_features of a
    fake live env holding those states at random headings and planar offsets,
    with the applied action past the box, matches it to float precision
    (snapshot's heading frame, the clamp);
  * VALUES -- the module's V and Q on those rows equal ValueNet(features(rows,
    q_action)) of a separately loaded net exactly, and reproduce each fit's
    reported test MSE against its 3 s labels;
  * MONITOR -- q_fn is differentiable in the action only (the nets get no
    grad), takes x from the ctx or set_state, ignores other ctx and batches
    candidate actions; v_fn; the CriticMonitor call is no-grad; the existing
    interventions run on it with their own call convention;
  * SONIC MAP -- on a real SonicTracker over a fake env (random action-term
    order, scales, offsets, defaults): env_action_to_sonic_raw inverts act()'s
    map both ways, clamps at +-ACTION_CLIP and agrees with the offline
    rebuild's (target - default) / scale; record_applied_action puts the
    applied action into the next history entry, and a pass-through keeps
    SONIC's raw bit for bit.
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox import filters  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_eval import ValueNet, features  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_value_filter import (  # noqa: E402
  ACTION_DIM, DEFAULT_VALUE_DIR, Q_DIM, STATE_DIM, PolicyValueMonitor, PolicyValues,
  env_action_to_sonic_raw, load_value_checkpoint, load_value_net, record_applied_action,
  runtime_features, sonic_action_box, state_features, value_path)
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  ACTION_CLIP, HISTORY_KEYS, SONIC_JOINTS, SonicReference, SonicTracker, _csv_joints,
  sonic_action_scale)
from robot_safety_sandbox.envs.g1_recover.states import from_heading_frame, load_bank  # noqa: E402

#: The G1 tasks' bridge action box (tasks/g1_stabilize.py action_bound).
BOUND = 3.0
TAGS = ("ppo_s0", "shac_s0")
VARIANTS = ("exact", "td_g099")


def _have(*names) -> bool:
  return all(os.path.exists(os.path.join(DEFAULT_VALUE_DIR, n)) for n in names)


def _have_fit(tag: str, variant: str) -> bool:
  return all(os.path.exists(value_path(DEFAULT_VALUE_DIR, k, tag, variant)) for k in "vq")


needs_fits = pytest.mark.skipif(
  not (_have("test_q.pt", "test_v.pt") and _have_fit("ppo_s0", "exact")
       and _have_fit("ppo_s0", "td_g099")),
  reason=f"no policy-value fits in {DEFAULT_VALUE_DIR}")


@pytest.fixture(scope="module")
def split():
  """The test splits as the fits read them (torch.load) and as a bank
  (load_bank, which keeps only the schema's columns)."""
  if not _have("test_q.pt", "test_v.pt"):
    pytest.skip(f"no test splits in {DEFAULT_VALUE_DIR}")
  out = {}
  for mode in "vq":
    path = os.path.join(DEFAULT_VALUE_DIR, f"test_{mode}.pt")
    out[mode] = (torch.load(path, map_location="cpu", weights_only=False), load_bank(path, "cpu"))
  return out


def _values(tag: str, variant: str) -> PolicyValues:
  if not _have_fit(tag, variant):
    pytest.skip(f"no {variant} fit of {tag}")
  return PolicyValues(tag, variant, DEFAULT_VALUE_DIR, "cpu")


def _own_net(path: str) -> tuple[ValueNet, dict]:
  """A fit's net loaded the way fit_policy_value_td.py's _load_net does."""
  ck = torch.load(path, map_location="cpu", weights_only=False)
  net = ValueNet(ck["in_dim"], tuple(ck["hidden"]))
  net.load_state_dict(ck["state_dict"])
  return net.eval(), ck


def _subset(rows: dict, idx: torch.Tensor) -> dict:
  n = int(rows["root_z"].numel())
  return {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == n else v)
          for k, v in rows.items()}


# --- checkpoints -------------------------------------------------------------------

def test_value_paths():
  assert value_path("d", "v", "ppo_s0") == os.path.join("d", "v_ppo_s0.pt")
  assert value_path("d", "q", "shac_s2", "td_g099") == os.path.join("d", "q_td_g099_shac_s2.pt")
  with pytest.raises(ValueError):
    value_path("d", "x", "ppo_s0")


@needs_fits
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("tag", TAGS)
def test_checkpoints(tag, variant):
  pv = _values(tag, variant)
  assert int(pv.v_net.mean.numel()) == STATE_DIM == 97
  assert int(pv.q_net.mean.numel()) == Q_DIM == 126
  for kind in "vq":
    assert {"false_safe", "residual_0.05"} <= set(pv.eps[kind])
    assert all(isinstance(e, float) for e in pv.eps[kind].values())
  assert pv.gamma == (1.0 if variant == "exact" else 0.99)
  for net in (pv.v_net, pv.q_net):
    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())
  net, info = load_value_checkpoint(value_path(DEFAULT_VALUE_DIR, "q", tag, variant))
  assert "state_dict" not in info and info["in_dim"] == Q_DIM
  assert isinstance(load_value_net(value_path(DEFAULT_VALUE_DIR, "v", tag, variant)), ValueNet)


@needs_fits
def test_missing_fit_fails_loudly(tmp_path):
  with pytest.raises(FileNotFoundError):
    PolicyValues("ppo_s0", "td_g042")
  # a V file in Q's place is refused on its width
  ck = torch.load(value_path(DEFAULT_VALUE_DIR, "v", "ppo_s0"), map_location="cpu", weights_only=False)
  for k in "vq":
    torch.save(ck, tmp_path / f"{k}_fake.pt")
  with pytest.raises(ValueError, match="Q's input"):
    PolicyValues("fake", "exact", str(tmp_path))


# --- the state input ---------------------------------------------------------------

@needs_fits
def test_state_features_are_the_fits_inputs(split):
  for mode in "vq":
    raw, bank = split[mode]
    want = features(raw)
    x = state_features(bank, -BOUND, BOUND)
    assert x.dtype == torch.float32 and tuple(x.shape) == (want.shape[0], STATE_DIM)
    assert torch.equal(x, want)
    # the env's action space bounds (numpy, per joint) clamp the same way
    lo, hi = np.full(ACTION_DIM, -BOUND, np.float32), np.full(ACTION_DIM, BOUND, np.float32)
    assert torch.equal(state_features(bank, lo, hi), want)
  raw, bank = split["q"]
  xq = torch.cat([state_features(bank, -BOUND, BOUND), raw["q_action"]], 1)
  assert torch.equal(xq, features(raw, raw["q_action"]))


def _live_env(rows: dict, idx: torch.Tensor, seed: int = 0):
  """A stand-in for ``env.mj`` holding bank rows ``idx`` at random headings and
  planar offsets (what ``states.snapshot`` reads), the applied action pushed
  past the box wherever the row's clamped action sits on it."""
  g = torch.Generator().manual_seed(seed)
  m = int(idx.numel())
  yaw = (torch.rand(m, generator=g) * 2.0 - 1.0) * math.pi
  quat_w, lin_w, ang_w = from_heading_frame(
    rows["root_quat_h"][idx], rows["lin_vel_h"][idx], rows["ang_vel_h"][idx], yaw)
  pos = torch.cat([torch.randn(m, 2, generator=g) * 3.0, rows["root_z"][idx, None]], 1)
  act = rows["action"][idx].clone()
  on_box = act.abs() == BOUND
  act[on_box] *= 1.7
  data = SimpleNamespace(root_link_pos_w=pos, root_link_quat_w=quat_w, root_link_lin_vel_w=lin_w,
                         root_link_ang_vel_w=ang_w, joint_pos=rows["joint_pos"][idx].clone(),
                         joint_vel=rows["joint_vel"][idx].clone())
  mj = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                       action_manager=SimpleNamespace(action=act))
  return mj, int(on_box.sum())


@needs_fits
def test_runtime_features_on_a_live_state(split):
  raw, bank = split["q"]
  idx = torch.randperm(int(bank["root_z"].numel()), generator=torch.Generator().manual_seed(1))[:1024]
  mj, n_out = _live_env(bank, idx)
  assert n_out > 100                                  # the clamp is exercised
  lo, hi = np.full(ACTION_DIM, -BOUND, np.float32), np.full(ACTION_DIM, BOUND, np.float32)
  x = runtime_features(mj, lo, hi, "cpu")
  want = features(_subset(raw, idx))
  assert torch.equal(x[:, -ACTION_DIM:], want[:, -ACTION_DIM:])   # the clamp, exactly
  torch.testing.assert_close(x, want, rtol=0.0, atol=2e-5)        # the heading round trip
  pv = _values("ppo_s0", "exact")
  u = raw["q_action"][idx]
  with torch.no_grad():
    torch.testing.assert_close(pv.q(x, u), pv.q(want, u), rtol=0.0, atol=1e-3)
    torch.testing.assert_close(pv.v(x), pv.v(want), rtol=0.0, atol=1e-3)


# --- the values --------------------------------------------------------------------------

@needs_fits
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("tag", TAGS)
def test_values_reproduce_the_fits(split, tag, variant):
  pv = _values(tag, variant)
  for mode in "vq":
    raw, bank = split[mode]
    net, ck = _own_net(value_path(DEFAULT_VALUE_DIR, mode, tag, variant))
    x = state_features(bank, -BOUND, BOUND)
    with torch.no_grad():
      if mode == "q":
        want = net(features(raw, raw["q_action"]))
        got = pv.q(x, raw["q_action"])
      else:
        want = net(features(raw))
        got = pv.v(x)
    torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
    # the fit's own report: MSE against the 3 s labels on the rows with a label
    lab = torch.load(os.path.join(DEFAULT_VALUE_DIR, f"test_{mode}_labels_{tag}.pt"),
                     map_location="cpu", weights_only=False)["v"][:, 0]
    ok = ~torch.isnan(lab)
    mse = float(((got[ok] - lab[ok]) ** 2).mean())
    rep = float(ck["report"]["test_mse"])
    assert abs(mse - rep) <= 1e-4 * rep, (tag, variant, mode, mse, rep)


# --- the monitor --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def batch(split):
  if not _have_fit("ppo_s0", "exact"):
    pytest.skip("no exact fit of ppo_s0")
  raw, bank = split["q"]
  idx = torch.arange(256)
  x = state_features(_subset(bank, idx), -BOUND, BOUND)
  return x, raw["q_action"][idx].clone(), raw["sonic_action"][idx].clone(), raw["action"][idx].clone()


@needs_fits
def test_monitor_gradient_and_ctx(batch):
  x, u0, _, _ = batch
  pv = _values("ppo_s0", "exact")
  mon = pv.monitor(num_envs=x.shape[0])
  assert isinstance(mon, filters.CriticMonitor) and mon.action_dim == ACTION_DIM
  u = u0.clone().requires_grad_(True)
  with torch.no_grad():                                   # the driver's context
    with torch.enable_grad():                             # the solver's
      q = mon.q_fn(action=u, x=x, obs=torch.zeros(x.shape[0], 7))
      (grad,) = torch.autograd.grad(q.sum(), u)
  assert q.shape == (x.shape[0],) and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
  assert all(p.grad is None for p in pv.q_net.parameters())
  with torch.no_grad():
    torch.testing.assert_close(q.detach(), pv.q(x, u0), rtol=0.0, atol=0.0)
    v = mon.v_fn(x=x, obs=None)
    torch.testing.assert_close(v, pv.v(x), rtol=0.0, atol=0.0)
  # the CriticMonitor call (what the least-restrictive switch reads) is no-grad
  qm = mon(u, x=x)
  assert not qm.requires_grad
  torch.testing.assert_close(qm, q.detach(), rtol=0.0, atol=0.0)
  # set_state instead of the ctx
  mon.set_state(x)
  with torch.no_grad():
    torch.testing.assert_close(mon.q_fn(action=u0), q.detach(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(mon.v_fn(), v, rtol=0.0, atol=0.0)
  # offsets (e.g. a calibrated eps)
  off = PolicyValueMonitor(pv, q_offset=0.5, v_offset=0.25)
  with torch.no_grad():
    torch.testing.assert_close(off.q_fn(action=u0, x=x), q.detach() - 0.5)
    torch.testing.assert_close(off.v_fn(x=x), v - 0.25)


@needs_fits
def test_monitor_candidates_and_errors(batch):
  x, u0, a_nom, a_last = batch
  pv = _values("ppo_s0", "exact")
  mon = pv.monitor()
  U = torch.stack([u0, a_nom, a_last])                    # (3, N, 29) candidates
  with torch.no_grad():
    qb = mon.q_fn(action=U, x=x)
    assert qb.shape == (3, x.shape[0])
    for k in range(3):
      torch.testing.assert_close(qb[k], mon.q_fn(action=U[k], x=x), rtol=0.0, atol=1e-5)
  # an inference-mode x still lets the solver differentiate
  with torch.inference_mode():
    xi = x.clone()
  u = u0.clone().requires_grad_(True)
  (g,) = torch.autograd.grad(mon.q_fn(action=u, x=xi).sum(), u)
  assert torch.isfinite(g).all()
  with pytest.raises(RuntimeError, match="no state"):
    pv.monitor().q_fn(action=u0)
  with pytest.raises(ValueError):
    mon.q_fn(action=u0, x=x[:, :90])
  with pytest.raises(ValueError):
    mon.q_fn(action=u0[:10], x=x)
  with pytest.raises(ValueError):
    pv.monitor(num_envs=5).v_fn(x=x)


@needs_fits
def test_interventions_run_on_the_monitor(batch):
  """The existing interventions' call convention: q_fn(action=..., **ctx) and
  fallback(**ctx) with the same ctx."""
  x, _, a_nom, a_last = batch
  n = x.shape[0]
  pv = _values("ppo_s0", "exact")
  mon = pv.monitor(num_envs=n)
  a_safe = a_last.clamp(-BOUND, BOUND)

  def fallback(**ctx):
    assert "x" in ctx
    return a_safe

  with torch.no_grad():
    q_nom, q_safe, v = pv.q(x, a_nom), pv.q(x, a_safe), pv.v(x)

  # the paper's last-resort switch: SONIC iff Q(x, u_SONIC) > 0
  lrsf = filters.LeastRestrictiveIntervention(n, "cpu", eps=0.0)
  act, info = lrsf(a_nom, fallback, mon, x=x)
  assert torch.equal(info.engaged, q_nom <= 0.0)
  assert torch.equal(act, torch.where(info.engaged.unsqueeze(1), a_safe, a_nom))

  # the projected-gradient sibling (V taken as Q(x, u_safe))
  qcbf = filters.QCBFIntervention(n, "cpu", kappa=0.7, action_low=-25.0, action_high=25.0)
  act, info = qcbf(a_nom, fallback, mon, x=x)
  passed = q_nom >= 0.7 * q_safe
  assert torch.equal(act[passed], a_nom[passed]) and torch.equal(info.engaged, ~passed)
  assert torch.isfinite(info.value).all()

  # the gradient-descent solver, once it exists: it reads v_fn
  cls = getattr(filters.intervention, "QCBFDescentIntervention", None)
  if cls is None:
    return
  solver = cls(n, "cpu", kappa=0.7)
  act, info = solver(a_nom, fallback, mon, x=x)
  tau = 0.7 * v + float(getattr(solver, "margin", 0.0))
  passed = q_nom >= tau
  assert torch.equal(act[passed], a_nom[passed])
  with torch.no_grad():
    q_out = pv.q(x, act)
  ok = (q_out >= tau - 1e-4) | (act == a_safe).all(1)
  assert bool(ok[~passed].all())


# --- the applied action in SONIC's history -----------------------------------------------

class _StubPolicy:
  """SONIC's networks replaced: a zero token and a preset decoder output."""

  def __init__(self, n: int):
    self.out = torch.zeros(n, ACTION_DIM)

  def encode(self, pos, vel, anchor):
    return torch.zeros(pos.shape[0], 64)

  def decode(self, obs):
    assert obs.shape[1] == 994
    return self.out.clone()


def _unit(q: torch.Tensor) -> torch.Tensor:
  q = q / q.norm(dim=-1, keepdim=True)
  return torch.where(q[..., :1] < 0, -q, q)


def _tracker(n: int = 6, seed: int = 0):
  """A real SonicTracker on a fake env: mjlab's entity joint order, a random
  action-term order, random per-joint env scales / offsets and defaults."""
  g = torch.Generator().manual_seed(seed)
  names = list(_csv_joints())
  target_ids = torch.randperm(ACTION_DIM, generator=g)
  default = torch.rand(ACTION_DIM, generator=g) - 0.5
  data = SimpleNamespace(
    default_joint_pos=default.expand(n, ACTION_DIM).clone(),
    root_link_quat_w=_unit(torch.randn(n, 4, generator=g)),
    root_link_ang_vel_w=torch.randn(n, 3, generator=g),
    joint_pos_biased=torch.randn(n, ACTION_DIM, generator=g),
    joint_vel=torch.randn(n, ACTION_DIM, generator=g))
  term = SimpleNamespace(
    target_ids=target_ids.tolist(), scale=0.2 + 0.6 * torch.rand(ACTION_DIM, generator=g),
    offset=(torch.rand(ACTION_DIM, generator=g) - 0.5).expand(n, ACTION_DIM).clone())
  mj = SimpleNamespace(scene={"robot": SimpleNamespace(joint_names=names, data=data)},
                       action_manager=SimpleNamespace(get_term=lambda name: term))
  frames = 80
  ref = SonicReference(torch.randn(frames, ACTION_DIM, generator=g),
                       torch.randn(frames, ACTION_DIM, generator=g),
                       _unit(torch.randn(frames, 4, generator=g)), [frames], ["clip"])
  pol = _StubPolicy(n)
  tr = SonicTracker(SimpleNamespace(mj=mj, device="cpu", num_envs=n), pol, ref)
  zero = torch.zeros(n, dtype=torch.long)
  tr.start(torch.ones(n, dtype=torch.bool), zero, zero)
  return tr, pol, names, target_ids, default


def test_sonic_raw_inverts_act():
  tr, pol, names, target_ids, default = _tracker()
  n = tr.n
  g = torch.Generator().manual_seed(3)
  pol.out = 12.0 * torch.randn(n, ACTION_DIM, generator=g)  # some past the +-20 clip
  a_env = tr.act()
  assert float(tr.raw.abs().max()) == ACTION_CLIP
  raw = env_action_to_sonic_raw(tr, a_env)
  torch.testing.assert_close(raw, tr.raw, rtol=1e-5, atol=1e-4)
  # the offline rebuild's map (sonic.rebuild_inputs), derived from the names alone
  tnames = [names[i] for i in target_ids.tolist()]
  t_to_s = [tnames.index(j) for j in SONIC_JOINTS]
  to_s = [names.index(j) for j in SONIC_JOINTS]
  want = (tr.target[:, t_to_s] - default[to_s]) / torch.as_tensor(sonic_action_scale(SONIC_JOINTS))
  torch.testing.assert_close(want, tr.raw, rtol=1e-5, atol=1e-4)
  # the other way: an env action -> raw -> act()'s map gives it back, on the
  # joints the clip leaves alone
  a = 1.5 * torch.randn(n, ACTION_DIM, generator=g)
  r = env_action_to_sonic_raw(tr, a)
  inside = r.abs() < ACTION_CLIP
  assert float(inside.float().mean()) > 0.5
  pol.out = r
  torch.testing.assert_close(tr.act()[inside], a[inside], rtol=1e-5, atol=1e-4)
  # far outside: clamped like SONIC's own output
  big = env_action_to_sonic_raw(tr, 1e3 * torch.sign(torch.randn(n, ACTION_DIM, generator=g)))
  assert float(big.abs().max()) == ACTION_CLIP
  with pytest.raises(ValueError):
    env_action_to_sonic_raw(tr, a[:, :10])


def test_sonic_action_box():
  """The env actions SONIC's raw +-ACTION_CLIP can command: SONIC's own
  output lies inside, the box's corners map to +-ACTION_CLIP, and an action
  inside it round-trips without the clamp."""
  tr, pol, _, _, _ = _tracker(n=8, seed=5)
  lo, hi = sonic_action_box(tr)
  assert lo.shape == (ACTION_DIM,) and bool((lo < hi).all())
  g = torch.Generator().manual_seed(6)
  pol.out = 30.0 * torch.randn(tr.n, ACTION_DIM, generator=g)  # many past the clip
  a_nom = tr.act()
  assert bool(((a_nom >= lo - 1e-4) & (a_nom <= hi + 1e-4)).all())
  for end in (lo, hi):
    r = env_action_to_sonic_raw(tr, end.expand(tr.n, ACTION_DIM))
    torch.testing.assert_close(r.abs(), torch.full_like(r, ACTION_CLIP), rtol=1e-5, atol=1e-4)
  u = lo + (hi - lo) * torch.rand(tr.n, ACTION_DIM, generator=g)
  r = env_action_to_sonic_raw(tr, u)
  assert float(r.abs().max()) < ACTION_CLIP
  pol.out = r
  torch.testing.assert_close(tr.act(), u, rtol=1e-5, atol=1e-4)


def test_record_applied_action():
  tr, pol, _, _, _ = _tracker(n=6, seed=1)
  pol.out = 5.0 * torch.randn(tr.n, ACTION_DIM, generator=torch.Generator().manual_seed(4))
  a_nom = tr.act()
  raw_nom = tr.raw.clone()
  a = a_nom.clone()
  a[1] += 0.5
  a[3] -= 0.7
  mask = torch.ones(tr.n, dtype=torch.bool)
  mask[3] = False                                        # env 3 keeps SONIC's own
  record_applied_action(tr, a, mask)
  keep = [0, 2, 3, 4, 5]
  assert torch.equal(tr.last_action[keep], raw_nom[keep])   # pass-throughs bit for bit
  torch.testing.assert_close(tr.last_action[1], env_action_to_sonic_raw(tr, a)[1])
  assert not torch.allclose(tr.last_action[1], raw_nom[1])
  assert torch.equal(tr.raw, raw_nom)                    # the nominal is kept
  tr.advance()
  k = [name for name, _ in HISTORY_KEYS].index("action")
  assert torch.equal(tr.hist[k][:, -1], tr.last_action)
  # no mask: every modified env carries its applied action
  a_nom = tr.act()
  b = a_nom.clone()
  b[0] *= 0.5
  record_applied_action(tr, b)
  torch.testing.assert_close(tr.last_action[0], env_action_to_sonic_raw(tr, b)[0])
  assert torch.equal(tr.last_action[1:], tr.raw[1:])
