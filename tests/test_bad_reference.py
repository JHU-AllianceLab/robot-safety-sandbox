"""The bad-reference suite's corruptions (envs/g1_recover/bad_reference.py).

CPU and simulator-free, on a synthetic base reference: four smooth random
clips in SONIC_JOINTS order, inside the G1's soft joint limits, with a
drifting heading and a small tilt.

  * CONTRACT (every scenario) -- E clips of L frames; frames before the onset
    are the base clip's frames frame0 + f bit for bit (frame 0 = the spawn),
    and the onset is the first frame that changes; frames outside the
    corrupted span are the time-aligned original; corrupted joints lie in
    the limits; unit quaternions; joint_vel is SONIC's forward_difference of
    each clip; the same seed gives the same reference and the global RNG is
    untouched;
  * TRACKER -- SONIC's encoder window (reference_window) of a corrupted
    episode equals the base's until the corruption enters its look-ahead, the
    tracker's delta heading agrees, and episode_length leaves room for the
    whole window at the last step;
  * FAMILIES -- each changes what it claims: the hf_osc spectrum peaks at its
    frequency (legs and waist only); jitter is white with its sigma; speedup
    / reverse read the clip at k x / backwards (k x / reversed velocities);
    skip, freeze_jump and teleport jump (teleport into another clip, the
    heading continuous); lean tilts the root by its angle about the heading's
    axis; pose reaches its (clamped) joint vector and root tilt; mirror is
    an involution matching the reflected clip; yaw turns the root by its
    angle;
  * fits / errors -- the room check, and loud failures on bad inputs.

sonic.py imports mjlab at the top, and mjlab imports mujoco_warp: minutes on
a login node. These tests need only sonic.py's pure-torch half, so unless
the real module is already imported (a full-suite run) or TEST_REAL_SONIC=1,
they compile exactly those definitions out of sonic.py's source (with this
module's w-first quaternion product standing in for mjlab's in
reference_window), and they load bad_reference.py by path, without the
robot_safety_sandbox package (whose __init__ imports mjlab).

On a loaded login node run them with ``OMP_NUM_THREADS=4``: torch's default
of one thread per core makes these small tensor ops crawl (~1 min vs over
10).
"""

from __future__ import annotations

import __future__
import ast
import importlib.util
import math
import os
import sys
import types
from typing import Optional, Sequence

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

_PKG = os.path.join(_ROOT, "robot_safety_sandbox", "envs", "g1_recover")
_SONIC_NAME = "robot_safety_sandbox.envs.g1_recover.sonic"
#: sonic.py's definitions these tests use (none of them touches mjlab but
#: reference_window's quat_mul).
_PURE = ("SONIC_JOINTS", "NUM_FUTURE", "FUTURE_STEP", "REF_LEAD", "SonicReference",
         "forward_difference", "reference_window")


def _load_by_path(path: str, name: str):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod          # dataclasses look their module up
  spec.loader.exec_module(mod)
  return mod


def _sonic_pure(quat_mul):
  """sonic.py's :data:`_PURE` definitions, compiled from its source."""
  path = os.path.join(_PKG, "sonic.py")
  with open(path) as fh:
    tree = ast.parse(fh.read(), path)
  keep, got = [], set()
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _PURE:
      keep.append(node)
      got.add(node.name)
    elif isinstance(node, ast.Assign):
      ids = {t.id for t in node.targets if isinstance(t, ast.Name)} & set(_PURE)
      if ids:
        keep.append(node)
        got |= ids
  if got != set(_PURE):
    raise RuntimeError(f"sonic.py no longer defines {sorted(set(_PURE) - got)}")
  mod = types.ModuleType("_sonic_pure")
  mod.__dict__.update(torch=torch, os=os, Optional=Optional, Sequence=Sequence,
                      quat_mul=quat_mul)
  code = compile(ast.Module(body=keep, type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True)
  exec(code, mod.__dict__)
  return mod


if _SONIC_NAME in sys.modules or os.environ.get("TEST_REAL_SONIC") == "1":
  from robot_safety_sandbox.envs.g1_recover import bad_reference as br  # noqa: E402
  from robot_safety_sandbox.envs.g1_recover import sonic  # noqa: E402
else:
  br = _load_by_path(os.path.join(_PKG, "bad_reference.py"), "_bad_reference_under_test")
  sonic = _sonic_pure(br.quat_mul)
  br._sonic_module = sonic

SONIC_JOINTS = sonic.SONIC_JOINTS
J = len(SONIC_JOINTS)
IDX = {n: i for i, n in enumerate(SONIC_JOINTS)}
FPS = 50.0
LENGTHS = (900, 1100, 1300, 1000)
E = 48
WINDOW = 150                      # 3 s tracked after the onset
ONSET = (50, 100)                 # frames
DUR = 75                          # the default 1.5 s

#: The G1 MJCF joint ranges, rad (left side; the right side's roll ranges are
#: their mirror images).
_G1_RANGE = {
  "hip_pitch": (-2.5307, 2.8798), "hip_roll": (-0.5236, 2.9671),
  "hip_yaw": (-2.7576, 2.7576), "knee": (-0.087267, 2.8798),
  "ankle_pitch": (-0.87267, 0.5236), "ankle_roll": (-0.2618, 0.2618),
  "waist_yaw": (-2.618, 2.618), "waist_roll": (-0.52, 0.52), "waist_pitch": (-0.52, 0.52),
  "shoulder_pitch": (-3.0892, 2.6704), "shoulder_roll": (-1.5882, 2.2515),
  "shoulder_yaw": (-2.618, 2.618), "elbow": (-1.0472, 2.0944),
  "wrist_roll": (-1.97222, 1.97222), "wrist_pitch": (-1.61443, 1.61443),
  "wrist_yaw": (-1.61443, 1.61443)}


def _soft_limits(factor: float = 0.9):
  """mjlab's soft limits (``soft_joint_pos_limit_factor`` 0.9) of the G1
  ranges, SONIC order: ``(low, high, mid, half)``."""
  lo, hi = [], []
  for n in SONIC_JOINTS:
    side = n.split("_")[0]
    key = n[:-len("_joint")]
    if side in ("left", "right"):
      key = key[len(side) + 1:]
    a, b = _G1_RANGE[key]
    if side == "right" and key.endswith("roll"):
      a, b = -b, -a
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    lo.append(mid - factor * half)
    hi.append(mid + factor * half)
  lo, hi = torch.tensor(lo), torch.tensor(hi)
  return lo, hi, 0.5 * (lo + hi), 0.5 * (hi - lo)


LOW, HIGH, MID, HALF = _soft_limits()


def _axis_quat(axis, angle):
  return br.quat_axis_angle(torch.tensor(axis).expand(angle.shape + (3,)), angle)


def _base(seed: int = 0, lengths=LENGTHS):
  """Smooth random clips: each joint mid + 0.3 half x a sum of three
  0.2-1.5 Hz sinusoids (inside the soft limits), the root a drifting
  heading with a small pitch and roll."""
  g = torch.Generator().manual_seed(seed)
  pos, quat = [], []
  for n in lengths:
    t = torch.arange(n).float() / FPS
    f = 0.2 + 1.3 * torch.rand(3, J, generator=g)
    ph = 2 * math.pi * torch.rand(3, J, generator=g)
    a = torch.rand(3, J, generator=g) + 0.2
    s = (a * torch.sin(2 * math.pi * f * t[:, None, None] + ph)).sum(1) / a.sum(0)
    pos.append(MID + 0.3 * HALF * s)
    c = 2 * math.pi * torch.rand(4, generator=g)
    yaw = c[0] + 0.4 * t + 0.3 * torch.sin(0.7 * t + c[1])
    pitch = 0.08 * torch.sin(1.1 * t + c[2])
    roll = 0.05 * torch.sin(1.3 * t + c[3])
    q = br.quat_mul(br.z_quat(yaw), br.quat_mul(_axis_quat((0.0, 1.0, 0.0), pitch),
                                                _axis_quat((1.0, 0.0, 0.0), roll)))
    quat.append(q)
  pos = torch.cat(pos).float()
  vel = torch.cat([sonic.forward_difference(p, FPS) for p in pos.split(list(lengths))])
  return sonic.SonicReference(pos, vel, torch.cat(quat).float(), list(lengths),
                              [f"clip{i}" for i in range(len(lengths))])


@pytest.fixture(scope="module")
def base():
  return _base()


@pytest.fixture(scope="module")
def factory(base):
  return br.BadReferenceFactory(base, LOW, HIGH)


def _episodes(seed: int = 1, n: int = E):
  """Spawns with room for every scenario: reverse reads back to frame
  onset - 1 - 75, speedup_5x up to 4 x 75 frames past frame0 + L."""
  g = torch.Generator().manual_seed(seed)
  clip = torch.randint(0, len(LENGTHS), (n,), generator=g)
  onset = torch.randint(ONSET[0], ONSET[1] + 1, (n,), generator=g)
  L = br.episode_length(onset, WINDOW)
  hi = torch.tensor(LENGTHS)[clip] - L - 4 * DUR - 2
  frame0 = 160 + (torch.rand(n, generator=g) * (hi - 160).float()).long()
  return clip, frame0, onset, L


def _gen(seed: int = 7):
  return torch.Generator().manual_seed(seed)


def _views(ref, Lc: int):
  n = ref.length.numel()
  return (ref.joint_pos.view(n, Lc, J), ref.joint_vel.view(n, Lc, J),
          ref.root_quat.view(n, Lc, 4))


def _original(base, clip, frame0, Lc: int):
  f = frame0[:, None] + torch.arange(Lc)
  rows = base.rows(clip[:, None].expand_as(f).reshape(-1), f.reshape(-1))
  n = clip.numel()
  return base.joint_pos[rows].view(n, Lc, J), base.root_quat[rows].view(n, Lc, 4)


def _at(x, idx):
  return x[torch.arange(x.shape[0]), idx]


def _wrap(a):
  return torch.atan2(torch.sin(a), torch.cos(a))


def _rotvec(q):
  q = torch.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  n = v.norm(dim=-1, keepdim=True)
  ang = 2 * torch.atan2(n, q[..., :1])
  return v / n.clamp_min(1e-12) * ang


# --- the registry ---------------------------------------------------------------

def test_registry_covers_every_family():
  names = set(br.SCENARIOS)
  for prefix in ("hf_osc_", "jitter_", "speedup_", "lean_fwd_", "lean_back_", "lean_roll_",
                 "skip_", "yaw_"):
    assert sum(n.startswith(prefix) for n in names) >= 2, prefix
  for n in ("teleport", "reverse", "freeze_jump", "mirror", "pose_knee_up", "pose_bend_fwd",
            "pose_split", "pose_arch_back"):
    assert n in names, n
  for n, sc in br.SCENARIOS.items():
    assert sc.family in br.FAMILIES and sc.after in br.AFTER, n
    assert len(sc.doc) > 60, f"{n}: say what makes it ill-posed"
  assert set(br.QUICK_SCENARIOS) <= names
  for k, p in br.POSES.items():
    assert p.doc and set(p.joints) <= set(SONIC_JOINTS), k
  assert set(br.NEUTRAL) <= set(SONIC_JOINTS)


# --- the contract, every scenario -------------------------------------------------

@pytest.mark.parametrize("name", list(br.SCENARIOS))
def test_contract(name, base, factory):
  clip, frame0, onset, L = _episodes()
  assert bool(factory.fits(clip, frame0, onset, L, name).all())
  state = torch.random.get_rng_state()
  ref = factory.build(clip, frame0, onset, L, name, _gen())
  assert torch.equal(torch.random.get_rng_state(), state), "drew from the global RNG"
  info = ref.corruption
  assert ref.joint_pos.shape == (E * L, J) and ref.root_quat.shape == (E * L, 4)
  assert torch.equal(ref.length, torch.full((E,), L)) and len(ref.names) == E
  assert torch.equal(ref.start, torch.arange(E) * L)
  pos, vel, quat = _views(ref, L)
  pos0, quat0 = _original(base, clip, frame0, L)
  f = torch.arange(L)
  pre = f[None] < onset[:, None]
  # before the onset: the base clip from frame0, bit for bit (frame 0 = spawn)
  assert torch.equal(pos[pre], pos0[pre]) and torch.equal(quat[pre], quat0[pre])
  assert torch.equal(pos[:, 0], base.joint_pos[base.start[clip] + frame0])
  assert torch.equal(quat[:, 0], base.root_quat[base.start[clip] + frame0])
  # the onset is the first frame that changes
  d_pos = (_at(pos, onset) - _at(pos0, onset)).abs().amax(-1)
  d_q = 1.0 - (_at(quat, onset) * _at(quat0, onset)).sum(-1).abs()
  changed = (d_pos > 1e-6) | (d_q > 1e-7)
  assert float(changed.float().mean()) >= 0.9, f"{name}: onset frame unchanged"
  # the corrupted span and the untouched rest
  m = info.corrupted
  rel = f[None] - onset[:, None]
  assert torch.equal(m, (rel >= 0) & (f[None] < info.end[:, None]))
  if br.SCENARIOS[name].after == br.RETURN:
    assert bool((info.end < L).all())
  else:
    assert bool((info.end == L).all())
  assert torch.equal(pos[~m], pos0[~m]) and torch.equal(quat[~m], quat0[~m])
  assert torch.equal(base.joint_pos[info.true_rows], pos0)
  # kinematically valid
  assert bool((pos >= LOW - 1e-6).all() and (pos <= HIGH + 1e-6).all())
  assert float((quat.norm(dim=-1) - 1.0).abs().max()) < 1e-5
  assert torch.allclose(info.tilt_max, br.tilt_of(quat).amax(1))
  # velocities: SONIC's forward differences of each clip
  for e in range(E):
    assert torch.equal(vel[e], sonic.forward_difference(pos[e], FPS))
  # common random numbers
  ref2 = factory.build(clip, frame0, onset, L, name, _gen())
  assert torch.equal(ref2.joint_pos, ref.joint_pos) and torch.equal(ref2.root_quat, ref.root_quat)


@pytest.mark.parametrize("name", [n for n in br.SCENARIOS
                                  if n.split("_")[0] in ("hf", "jitter", "teleport", "pose",
                                                         "yaw") or n == "lean_roll_30"])
def test_random_scenarios_follow_the_seed(name, factory):
  clip, frame0, onset, L = _episodes()
  a = factory.build(clip, frame0, onset, L, name, _gen(1))
  b = factory.build(clip, frame0, onset, L, name, _gen(2))
  if name in ("yaw_180", "pose_bend_fwd", "pose_split", "pose_arch_back"):
    assert torch.equal(a.joint_pos, b.joint_pos)       # nothing random
  else:
    assert not (torch.equal(a.joint_pos, b.joint_pos) and torch.equal(a.root_quat, b.root_quat))
  if name not in ("yaw_180", "pose_bend_fwd", "pose_split", "pose_arch_back"):
    with pytest.raises(ValueError, match="torch.Generator"):
      factory.build(clip, frame0, onset, L, name)


def test_jitter_stream_does_not_depend_on_the_onset(base, factory):
  clip, frame0, onset, L = _episodes()
  a = factory.build(clip, frame0, onset, L, "jitter_0.1", _gen())
  b = factory.build(clip, frame0, onset + 3, L, "jitter_0.1", _gen())
  pos0, _ = _original(base, clip, frame0, L)
  both = a.corruption.corrupted & b.corruption.corrupted
  da, db = _views(a, L)[0] - pos0, _views(b, L)[0] - pos0
  inside = (pos0 + da > LOW + 1e-3) & (pos0 + da < HIGH - 1e-3)   # unclamped
  assert torch.allclose(da[both & inside.all(-1)], db[both & inside.all(-1)], atol=1e-6)


# --- SONIC's view -----------------------------------------------------------------

def test_tracker_window_until_the_corruption_is_visible(base, factory):
  clip, frame0, onset, L = _episodes()
  ref = factory.build(clip, frame0, onset, L, "teleport", _gen())
  la = br.look_ahead()
  assert la == sonic.REF_LEAD + (sonic.NUM_FUTURE - 1) * sonic.FUTURE_STEP == 46
  ep = torch.arange(E)
  # the tracker's delta heading: fixed at start from frame 0's yaw
  q_robot = br.z_quat(torch.rand(E, generator=_gen(3)) * 6.0)
  dq_bad = br.quat_mul(br.yaw_quat(q_robot), br.quat_conj(br.yaw_quat(ref.root_quat[ref.rows(ep, torch.zeros_like(ep))])))
  dq_base = br.quat_mul(br.yaw_quat(q_robot), br.quat_conj(br.yaw_quat(base.root_quat[base.rows(clip, frame0)])))
  assert torch.equal(dq_bad, dq_base)
  for t in range(0, int(onset.min()) - la):
    pb, vb, qb = sonic.reference_window(ref, ep, torch.full((E,), t + sonic.REF_LEAD), dq_bad)
    p0, v0, q0 = sonic.reference_window(base, clip, frame0 + t + sonic.REF_LEAD, dq_base)
    assert torch.equal(pb, p0) and torch.equal(qb, q0)
    if t + la < int(onset.min()) - 1:
      assert torch.allclose(vb, v0, atol=1e-3)
  # once the onset is inside the look-ahead, SONIC sees the jump
  t = onset - la
  pb, _, _ = sonic.reference_window(ref, ep, t + sonic.REF_LEAD, dq_bad)
  p0, _, _ = sonic.reference_window(base, clip, frame0 + t + sonic.REF_LEAD, dq_base)
  assert bool(((pb[:, -1] - p0[:, -1]).abs().amax(-1) > 1e-6).float().mean() > 0.9)
  # episode_length: the last tracked step's window ends on the clip's last frame
  last = onset + WINDOW - 1
  offs = torch.arange(sonic.NUM_FUTURE) * sonic.FUTURE_STEP
  read = last[:, None] + sonic.REF_LEAD + offs
  assert int(read.max()) == L - 1 and bool((read < L).all())


# --- the families -----------------------------------------------------------------

def _build(factory, name, n=E, dur=DUR, seed=7):
  clip, frame0, onset, L = _episodes(n=n)
  ref = factory.build(clip, frame0, onset, L, name, _gen(seed), duration=dur)
  return ref, clip, frame0, onset, L


@pytest.mark.parametrize("name,freq,amp", [("hf_osc_4hz_0.3", 4.0, 0.3),
                                           ("hf_osc_4hz_0.6", 4.0, 0.6),
                                           ("hf_osc_8hz_0.3", 8.0, 0.3)])
def test_hf_osc_peaks_at_its_frequency(name, freq, amp, base, factory):
  D = 100                                        # 0.5 Hz bins
  ref, clip, frame0, onset, L = _build(factory, name, dur=D)
  pos, _, _ = _views(ref, L)
  pos0, _ = _original(base, clip, frame0, L)
  delta = pos - pos0
  legs = [i for i, n in enumerate(SONIC_JOINTS) if br._is_leg_or_waist(n)]
  arms = [i for i in range(J) if i not in legs]
  assert len(legs) == 15 and torch.equal(delta[..., arms], torch.zeros_like(delta[..., arms]))
  win = onset[:, None] + torch.arange(D)
  seg = delta[torch.arange(E)[:, None], win]     # [E, D, J]
  for j in (IDX["left_hip_pitch_joint"], IDX["right_knee_joint"], IDX["waist_yaw_joint"]):
    spec = torch.fft.rfft(seg[..., j], dim=1).abs()
    peak = spec[:, 1:].argmax(1) + 1
    assert bool((peak.float() * FPS / D == freq).all()), SONIC_JOINTS[j]
  hp = seg[..., IDX["left_hip_pitch_joint"]]      # wide range: never clamped
  assert float(hp.abs().max()) <= amp + 1e-5
  assert float(hp.abs().amax(1).min()) > 0.9 * amp


def test_jitter_is_white_with_its_sigma(base, factory):
  for name, sigma in (("jitter_0.1", 0.1), ("jitter_0.3", 0.3)):
    ref, clip, frame0, onset, L = _build(factory, name)
    pos, _, _ = _views(ref, L)
    pos0, _ = _original(base, clip, frame0, L)
    wide = [IDX[n] for n in ("left_hip_pitch_joint", "right_hip_yaw_joint", "waist_yaw_joint",
                             "left_shoulder_pitch_joint", "right_shoulder_yaw_joint")]
    win = onset[:, None] + torch.arange(DUR)
    d = (pos - pos0)[torch.arange(E)[:, None], win][..., wide]
    assert abs(float(d.std()) / sigma - 1.0) < 0.05
    r1 = (d[:, 1:] * d[:, :-1]).mean() / d.pow(2).mean()
    assert abs(float(r1)) < 0.05


@pytest.mark.parametrize("name,rate", [("speedup_3x", 3), ("speedup_5x", 5), ("reverse", -1)])
def test_rate_families_read_the_clip_at_their_rate(name, rate, base, factory):
  ref, clip, frame0, onset, L = _build(factory, name)
  pos, vel, quat = _views(ref, L)
  f = torch.arange(L)[None]
  rel = f - onset[:, None]
  tau = torch.where(rel < 0, frame0[:, None] + f,
                    torch.where(rel < DUR, frame0[:, None] + onset[:, None] - 1 + rate * (rel + 1),
                                frame0[:, None] + f + (rate - 1) * DUR))
  rows = base.rows(clip[:, None].expand_as(tau).reshape(-1), tau.reshape(-1)).view(E, L)
  assert torch.equal(pos, base.joint_pos[rows])
  assert torch.allclose(quat, base.root_quat[rows], atol=1e-6)    # corrupted frames: renormalized
  assert torch.equal(ref.corruption.src_frame, tau)
  # velocities inside the window: rate x the clip's around the source frame
  inside = (rel >= 0) & (rel < DUR - 1)
  v_src = base.joint_vel[torch.where(torch.tensor(rate < 0), rows - 1, rows)]
  ratio = (vel[inside] * v_src[inside]).sum() / v_src[inside].pow(2).sum()
  assert abs(float(ratio) - rate) < 0.1 * abs(rate)


def test_skip_freeze_teleport_jump(base, factory):
  # skip: the clip jumps S ahead at the onset and plays on
  for name, S in (("skip_0.5s", 25), ("skip_1.0s", 50)):
    ref, clip, frame0, onset, L = _build(factory, name)
    pos, _, _ = _views(ref, L)
    assert torch.equal(_at(pos, onset), base.joint_pos[base.rows(clip, frame0 + onset + S)])
    step = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1)
    assert float((_at(step, onset - 1) > 5 * step.median()).float().mean()) > 0.9
  # freeze_jump: the last clean frame held 0.5 s, then the time-aligned frame
  ref, clip, frame0, onset, L = _build(factory, "freeze_jump")
  pos, vel, _ = _views(ref, L)
  held = base.joint_pos[base.rows(clip, frame0 + onset - 1)]
  for k in range(25):
    assert torch.equal(_at(pos, onset + k), held)
  for k in range(24):
    assert torch.equal(_at(vel, onset + k), torch.zeros(E, J))
  assert torch.equal(_at(pos, onset + 25), base.joint_pos[base.rows(clip, frame0 + onset + 25)])
  assert bool((ref.corruption.end == onset + 25).all())
  # teleport: another clip from the onset, heading continuous, pose jumps
  ref, clip, frame0, onset, L = _build(factory, "teleport")
  pos, _, quat = _views(ref, L)
  info = ref.corruption
  other, g0 = info.draws["other_clip"], info.draws["other_frame"]
  assert bool((other != clip).all()) and len(set(other.tolist())) > 1
  rel = torch.arange(L)[None] - onset[:, None]
  after = rel >= 0
  assert bool((info.src_clip[after] != clip[:, None].expand_as(after)[after]).all())
  tgt = base.rows(other[:, None].expand_as(rel).reshape(-1),
                  (g0[:, None] + rel).reshape(-1)).view(E, L)
  assert torch.equal(pos[after], base.joint_pos[tgt][after])
  assert bool((g0[:, None] + rel[:, -1:] < torch.tensor(LENGTHS)[other][:, None]).all())
  _, quat0 = _original(base, clip, frame0, L)
  assert float(_wrap(br.yaw_of(_at(quat, onset)) - br.yaw_of(_at(quat0, onset))).abs().max()) < 1e-5
  qh = lambda q: br.quat_mul(br.quat_conj(br.yaw_quat(q)), q)  # noqa: E731
  dot = (qh(quat[after]) * qh(base.root_quat[tgt][after])).sum(-1).abs()
  assert float(dot.min()) > 1 - 1e-6
  step = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1)
  assert float((_at(step, onset - 1) > 5 * step.median()).float().mean()) > 0.9


def test_teleport_lands_only_in_the_target_clips(base):
  f = br.BadReferenceFactory(base, LOW, HIGH, teleport_clips=[2, 0])
  assert f.teleport_clips.tolist() == [0, 2]
  ref, clip, frame0, onset, L = _build(f, "teleport")
  other = ref.corruption.draws["other_clip"]
  assert bool((other != clip).all()) and set(other.tolist()) <= {0, 2}
  # clip 0 can only go to 2 and clip 2 only to 0; the others go to both
  assert bool((other[clip == 0] == 2).all()) and bool((other[clip == 2] == 0).all())
  rest = (clip == 1) | (clip == 3)
  assert int(rest.sum()) > 8 and set(other[rest].tolist()) == {0, 2}
  on = torch.tensor([60])
  with pytest.raises(ValueError, match="other than"):
    br.BadReferenceFactory(base, LOW, HIGH, teleport_clips=[1]).build(
      torch.tensor([1]), torch.tensor([200]), on, br.episode_length(on, WINDOW), "teleport",
      _gen())
  with pytest.raises(ValueError, match="teleport_clips"):
    br.BadReferenceFactory(base, LOW, HIGH, teleport_clips=[len(LENGTHS)])


def test_unedited_joints_keep_the_base_value_past_the_limits(base):
  """A third of the LAFAN frames have a joint past the 0.9 soft limits.
  Limits tighter than the synthetic clips stand in for that: a joint the
  family does not edit stays the base's bit for bit, and an edited one stays
  inside the limits widened to its source frame's value."""
  lo, hi = MID - 0.1 * HALF, MID + 0.1 * HALF
  f = br.BadReferenceFactory(base, lo, hi)
  arms = torch.tensor([not br._is_leg_or_waist(n) for n in SONIC_JOINTS])
  for name in ("lean_fwd_30", "yaw_90", "skip_0.5s", "speedup_3x", "reverse", "teleport",
               "freeze_jump", "hf_osc_4hz_0.6", "jitter_0.1", "pose_split", "mirror"):
    ref, clip, frame0, onset, L = _build(f, name)
    pos, _, _ = _views(ref, L)
    info = ref.corruption
    rows = base.rows(info.src_clip.reshape(-1), info.src_frame.reshape(-1)).view(E, L)
    src = base.joint_pos[rows]
    m = info.corrupted
    assert bool(((src < lo) | (src > hi))[m].any()), name          # the test bites
    fam = br.SCENARIOS[name].family
    if fam in ("lean", "yaw", "skip", "speedup", "reverse", "teleport", "freeze"):
      assert torch.equal(pos[m], src[m]), name
    elif fam == "hf_osc":
      assert torch.equal(pos[m][:, arms], src[m][:, arms]), name
      assert not torch.equal(pos[m][:, ~arms], src[m][:, ~arms]), name
    lo_w, hi_w = torch.minimum(lo, src), torch.maximum(hi, src)
    assert bool(((pos >= lo_w - 1e-6) & (pos <= hi_w + 1e-6))[m].all()), name


@pytest.mark.parametrize("pose", list(br.POSES))
def test_pose_soles_flat_and_arms_straight(pose):
  """Pose conventions: the sole pitch is root + hip + knee + ankle pitch (flat
  at 0; the lifted foot of a sided pose excepted), and elbow 0 is a right
  angle, so a straight arm needs ~ +1.5."""
  p = br.POSES[pose]
  v = _pose_vector(p).clamp(LOW, HIGH)
  for side in ("left", "right"):
    assert float(v[IDX[f"{side}_elbow_joint"]]) >= 1.4, side
    if p.sided and side == "left":
      continue
    pitch = p.root_pitch + sum(float(v[IDX[f"{side}_{j}_joint"]])
                               for j in ("hip_pitch", "knee", "ankle_pitch"))
    assert abs(pitch) < 1e-6, side


@pytest.mark.parametrize("name", [n for n in br.SCENARIOS if n.startswith("lean_")])
def test_lean_tilts_the_root_by_its_angle(name, base, factory):
  sc = br.SCENARIOS[name]
  ref, clip, frame0, onset, L = _build(factory, name)
  _, _, quat = _views(ref, L)
  _, quat0 = _original(base, clip, frame0, L)
  rv = _rotvec(br.quat_mul(quat, br.quat_conj(quat0)))       # world-frame rotation
  psi = br.yaw_of(quat0)
  if sc.params["axis"] == "pitch":
    ax = torch.stack([-torch.sin(psi), torch.cos(psi), torch.zeros_like(psi)], -1)
  else:
    ax = torch.stack([torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)], -1)
  ang = (rv * ax).sum(-1)                                     # signed, about the heading's axis
  rel = torch.arange(L)[None] - onset[:, None]
  theta = math.radians(sc.params["deg"])
  held = rel >= 14                                            # the 0.3 s ramp is done
  sgn = ref.corruption.draws["side"]
  assert torch.allclose(ang[held], (theta * sgn[:, None]).expand_as(ang)[held], atol=1e-4)
  assert float(rv[held].norm(dim=-1).sub(theta).abs().max()) < 1e-4    # all of it about that axis
  ramp = ang[(rel >= 0) & (rel < 15)].view(E, 15) * sgn[:, None]
  assert bool((ramp[:, 1:] >= ramp[:, :-1] - 1e-5).all()) and bool((ramp[:, 0] > 0).all())
  if sc.params["side"] == 0:
    assert set(sgn.tolist()) == {-1.0, 1.0}
  else:
    assert bool((sgn == sc.params["side"]).all())


def _pose_vector(pose: br.Pose) -> torch.Tensor:
  v = torch.zeros(J)
  for n, a in {**br.NEUTRAL, **pose.joints}.items():
    v[IDX[n]] = a
  return v


@pytest.mark.parametrize("pose", list(br.POSES))
def test_pose_reaches_its_target(pose, base, factory):
  p = br.POSES[pose]
  ref, clip, frame0, onset, L = _build(factory, f"pose_{pose}")
  pos, _, quat = _views(ref, L)
  _, quat0 = _original(base, clip, frame0, L)
  rel = torch.arange(L)[None] - onset[:, None]
  held = rel >= 14
  left = _pose_vector(p)
  right = br.mirror_joints(left, SONIC_JOINTS)
  side = ref.corruption.draws["right_side"]
  want = torch.where(side[:, None], right, left).clamp(LOW, HIGH)
  assert torch.allclose(pos[held], want[:, None].expand_as(pos)[held], atol=1e-6)
  if p.sided:
    assert set(side.tolist()) == {True, False}
    lk = pos[held][:, IDX["left_knee_joint"]]
    assert bool(((lk > 2.0) == ~side[:, None].expand_as(held)[held]).all())
  # the root: the clip's heading, the pose's tilt
  roll = torch.where(side, -p.root_roll, p.root_roll)
  tilt = br.quat_mul(_axis_quat((0.0, 1.0, 0.0), torch.full((E,), p.root_pitch)),
                     _axis_quat((1.0, 0.0, 0.0), roll))
  qh = br.quat_mul(br.quat_conj(br.yaw_quat(quat0)), quat)
  dot = (qh[held] * tilt[:, None].expand_as(qh)[held]).sum(-1).abs()
  assert float(dot.min()) > 1 - 1e-6


def test_mirror_is_an_involution_and_reflects_the_clip(base, factory):
  perm, sign = br.mirror_map(SONIC_JOINTS)
  assert torch.equal(perm[perm], torch.arange(J)) and torch.equal(sign[perm], sign)
  x = torch.randn(5, J, generator=_gen())
  assert torch.equal(br.mirror_joints(br.mirror_joints(x, SONIC_JOINTS), SONIC_JOINTS), x)
  m = br.mirror_joints(x, SONIC_JOINTS)
  for a, b, s in (("left_hip_roll_joint", "right_hip_roll_joint", -1),
                  ("left_hip_pitch_joint", "right_hip_pitch_joint", 1),
                  ("left_knee_joint", "right_knee_joint", 1),
                  ("left_ankle_roll_joint", "right_ankle_roll_joint", -1),
                  ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1),
                  ("left_elbow_joint", "right_elbow_joint", 1),
                  ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1),
                  ("waist_yaw_joint", "waist_yaw_joint", -1),
                  ("waist_roll_joint", "waist_roll_joint", -1),
                  ("waist_pitch_joint", "waist_pitch_joint", 1)):
    assert torch.equal(m[:, IDX[a]], s * x[:, IDX[b]]), a
  # the limits are mirror images, so mirroring a valid pose needs no clamp
  ml, mh = br.mirror_joints(LOW, SONIC_JOINTS), br.mirror_joints(HIGH, SONIC_JOINTS)
  assert torch.allclose(torch.minimum(ml, mh), LOW, atol=1e-6)
  assert torch.allclose(torch.maximum(ml, mh), HIGH, atol=1e-6)
  ref, clip, frame0, onset, L = _build(factory, "mirror")
  pos, _, quat = _views(ref, L)
  pos0, quat0 = _original(base, clip, frame0, L)
  after = (torch.arange(L)[None] >= onset[:, None])
  assert torch.equal(pos[after], br.mirror_joints(pos0, SONIC_JOINTS)[after])
  yaw0 = br.yaw_of(_at(quat0, onset))[:, None]
  back = br.mirror_quat(quat, yaw0)
  assert float((1 - (back[after] * quat0[after]).sum(-1).abs()).max()) < 1e-6
  assert float(_wrap(br.yaw_of(_at(quat, onset)) - yaw0[:, 0]).abs().max()) < 1e-5
  assert torch.allclose(br.tilt_of(quat), br.tilt_of(quat0), atol=1e-4)


@pytest.mark.parametrize("name,deg", [("yaw_90", 90.0), ("yaw_180", 180.0)])
def test_yaw_turns_the_root(name, deg, base, factory):
  ref, clip, frame0, onset, L = _build(factory, name)
  _, _, quat = _views(ref, L)
  _, quat0 = _original(base, clip, frame0, L)
  after = (torch.arange(L)[None] >= onset[:, None])
  d = _wrap(br.yaw_of(quat) - br.yaw_of(quat0))[after]
  assert float((d.abs() - math.radians(deg)).abs().max()) < 1e-4
  if deg == 90.0:
    assert bool((d > 0).any() and (d < 0).any())
  qh = lambda q: br.quat_mul(br.quat_conj(br.yaw_quat(q)), q)  # noqa: E731
  assert float((1 - (qh(quat[after]) * qh(quat0[after])).sum(-1).abs()).max()) < 1e-6


# --- fits and the failures ---------------------------------------------------------

def test_fits_checks_the_room(base, factory):
  n = torch.tensor(LENGTHS)
  clip = torch.tensor([0, 0, 1, 1])
  onset = torch.tensor([60, 60, 60, 60])
  L = br.episode_length(onset, WINDOW)
  frame0 = torch.tensor([200, int(n[0]) - L - 10, 0, 200])
  ok = factory.fits(clip, frame0, onset, L, "speedup_5x")
  assert ok.tolist() == [True, False, True, True]
  assert factory.fits(clip, frame0, onset, L, "reverse").tolist() == [True, True, False, True]
  assert bool(factory.fits(clip, frame0, onset, L, "teleport").all())
  assert bool(factory.fits(clip, frame0, onset, L, "jitter_0.1").all())
  # what fits promises: the built reference reads inside every clip
  ref = factory.build(clip[ok], frame0[ok], onset[ok], L, "speedup_5x")
  sf = ref.corruption.src_frame
  assert bool(((sf >= 0) & (sf < n[clip[ok]][:, None])).all())


def test_failures_are_loud(base, factory):
  clip, frame0, onset, L = _episodes(n=4)
  with pytest.raises(ValueError, match="unknown bad-reference scenario"):
    factory.build(clip, frame0, onset, L, "nope", _gen())
  with pytest.raises(ValueError, match="onset"):
    factory.build(clip, frame0, torch.zeros_like(onset), L, "skip_0.5s")
  with pytest.raises(ValueError, match="onset"):
    factory.build(clip, frame0, torch.full_like(onset, L), L, "skip_0.5s")
  with pytest.raises(ValueError, match="frame0"):
    factory.build(clip, frame0 + 5000, onset, L, "skip_0.5s")
  with pytest.raises(ValueError, match="episodes"):
    factory.build(clip, frame0[:2], onset, L, "skip_0.5s")
  with pytest.raises(ValueError, match="limits"):
    br.BadReferenceFactory(base, LOW[:10], HIGH)
  one = _base(lengths=(1200,))
  with pytest.raises(ValueError, match="target clip other than"):
    br.BadReferenceFactory(one, LOW, HIGH).build(
      torch.zeros(2, dtype=torch.long), torch.tensor([10, 20]), torch.tensor([60, 60]),
      L, "teleport", _gen())
  # a custom scenario object runs through the same checks
  sc = br.Scenario("skip", br.CONTINUE, "a custom skip for an ablation", dict(skip_s=0.2))
  ref = factory.build(clip, frame0, onset, L, sc)
  assert ref.corruption.scenario == "custom_skip"
  with pytest.raises(ValueError, match="after"):
    factory.build(clip, frame0, onset, L, br.Scenario("skip", "later", "x", dict(skip_s=0.2)))
