"""Scripted random pushes (random_push.RandomPush), the bridge's push channel
and its per-episode regime(), and the g1_recover_push registration.

Simulator-free (CPU):

  * the PROCESS -- quiet until the first onset, then one direction for exactly
    ``duration`` steps at ``force_max x force_scale`` (horizontal, unit), quiet
    again, the next onset inside the interval; ``reset`` drops an active push;
  * the BRIDGE -- ``_mj_step`` writes the push as the wrench on adversary_body
    (zeros where quiet) and restarts the envs that ended; random_push with an
    adversary, or off the wrench channel, is refused before the sim is built;
  * the REGIME -- spawn source x force bucket, None when the task has neither;
  * the REGISTRATION -- avoid-only, single-player, random_push on, 4 buckets;
  * ONE PUSH PER EPISODE -- ``max_pushes`` caps the random onsets, an
    unreachable cap draws exactly the unlimited process, and the tensors
    ``forces`` returns are never mutated afterwards;
  * SCRIPTED PUSHES -- ``schedule``: onset 0 acts on the next call for exactly
    ``steps`` calls, onset k waits k calls, the magnitude is absolute, ``rel``
    resolves against the heading at the onset, ``count`` feeds ``max_pushes``,
    the random process resumes one interval after the scripted onset;
    ``exhaust`` stops random onsets until the reset; ``quiet``'s truth table;
  * the BRIDGE'S PENDING PUSH -- a restarted env's pending push resumes on the
    first step of its episode at its exact force and consumes the random one;
    ``mj._zoo_push_quiet`` is published before physics and after the resets.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.random_push import RandomPush  # noqa: E402

DT = 0.02


def _quiet(p, steps, fmax=100.0):
  for _ in range(steps):
    f, a = p.forces(fmax, None)
    assert not bool(a.any()) and float(f.abs().sum()) == 0.0


def test_push_is_quiet_then_one_direction_at_force_max_times_scale():
  torch.manual_seed(0)
  p = RandomPush(4, DT, "cpu", interval_s=(1.0, 2.0), duration_s=0.2,
                 first_delay_s=(0.5, 0.5))
  _quiet(p, 25)                                  # 0.5 s of first delay
  scale = torch.tensor([1.0, 0.5, 0.25, 0.0])
  f0, a = p.forces(100.0, scale)
  assert bool(a.all())
  assert torch.allclose(f0.norm(dim=1), 100.0 * scale)
  assert torch.allclose(f0[:, 2], torch.zeros(4))          # horizontal
  d0 = p.dir.clone()
  assert torch.allclose(d0.norm(dim=1), torch.ones(4))
  for _ in range(9):                                        # 0.2 s = 10 steps
    f, a = p.forces(100.0, scale)
    assert bool(a.all()) and torch.allclose(f, f0)
  f, a = p.forces(100.0, scale)
  assert not bool(a.any()) and float(f.abs().sum()) == 0.0
  # the next onset falls inside [1.0, 2.0] s after the previous one
  steps_to_next = int(p.t_next[0]) + 1
  assert 50 - 10 <= steps_to_next <= 100 - 10 + 1
  assert int(p.age[0]) == 11


def test_sphere_directions_and_validation():
  p = RandomPush(8, DT, "cpu", horizontal=False, first_delay_s=(0.0, 0.0))
  f, a = p.forces(10.0)
  assert bool(a.all()) and torch.allclose(f.norm(dim=1), torch.full((8,), 10.0))
  assert float(f[:, 2].abs().sum()) > 0.0
  for bad in (dict(interval_s=(2.0, 1.0)), dict(duration_s=0.0),
              dict(first_delay_s=(-1.0, 1.0)), dict(dt=0.0)):
    kw = dict(num_envs=2, dt=DT, device="cpu")
    kw.update(bad)
    with pytest.raises(ValueError):
      RandomPush(**kw)


def test_reset_drops_an_active_push_and_redraws_the_first_delay():
  torch.manual_seed(1)
  p = RandomPush(3, DT, "cpu", first_delay_s=(0.0, 0.0), duration_s=0.4)
  f, a = p.forces(50.0)
  assert bool(a.all())
  p.reset(torch.tensor([True, False, False]))
  assert int(p.t_left[0]) == 0 and float(p.dir[0].abs().sum()) == 0.0
  assert bool((p.t_left[1:] > 0).all())
  f, a = p.forces(50.0)
  assert bool(a[0]) and bool(a[1:].all())       # env 0: first delay 0 -> a new push at once
  assert not torch.allclose(f[0], torch.zeros(3))


def _rel_of(p, heading):
  ang = torch.atan2(p.dir[:, 1], p.dir[:, 0]) - heading
  return torch.remainder(ang + torch.pi, 2 * torch.pi) - torch.pi


@pytest.mark.parametrize("rear_frac", [0.0, 0.5, 1.0])
def test_rear_frac_draws_that_share_from_behind_the_heading(rear_frac):
  torch.manual_seed(3)
  n = 40000
  p = RandomPush(n, DT, "cpu", first_delay_s=(0.0, 0.0), rear_frac=rear_frac,
                 rear_half_width_deg=60.0)
  heading = (torch.rand(n) * 2 - 1) * torch.pi
  f, a = p.forces(100.0, None, heading)
  assert bool(a.all()) and torch.allclose(f.norm(dim=1), torch.full((n,), 100.0))
  rel = _rel_of(p, heading)
  gap = torch.remainder(rel - p.rel + torch.pi, 2 * torch.pi) - torch.pi
  assert float(gap.abs().max()) < 1e-4
  # within 60 deg of the heading: rear_frac + (1 - rear_frac) / 3
  cone = float((rel.abs() <= torch.pi / 3).float().mean())
  assert abs(cone - (rear_frac + (1.0 - rear_frac) / 3.0)) < 0.01
  # chest pushes (force against the heading, |rel| > 135 deg) keep the uniform share
  chest = float((rel.abs() > 3 * torch.pi / 4).float().mean())
  assert abs(chest - (1.0 - rear_frac) / 4.0) < 0.01
  # rear draws are symmetric: as many from back-left as from back-right
  assert abs(float((rel > 0).float().mean()) - 0.5) < 0.01


def test_rear_frac_validation_and_needs_a_heading():
  for bad in (dict(rear_frac=-0.1), dict(rear_frac=1.5),
              dict(rear_half_width_deg=0.0), dict(rear_half_width_deg=200.0),
              dict(rear_frac=0.5, horizontal=False)):
    with pytest.raises(ValueError):
      RandomPush(2, DT, "cpu", **bad)
  p = RandomPush(2, DT, "cpu", first_delay_s=(0.0, 0.0), rear_frac=0.5)
  with pytest.raises(ValueError, match="heading"):
    p.forces(10.0)
  # uniform pushes still run without a heading; their side is just unknown
  p = RandomPush(2, DT, "cpu", first_delay_s=(0.0, 0.0))
  p.forces(10.0)
  assert bool(torch.isnan(p.rel).all())
  assert p.record(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)) == {}


def test_record_averages_survival_per_side_and_skips_unpushed_episodes():
  p = RandomPush(6, DT, "cpu", ema=0.5)
  deg = torch.tensor([0.0, 30.0, 90.0, 180.0, -90.0, float("nan")])
  p.rel = torch.deg2rad(deg)
  ended = torch.tensor([True, True, True, True, False, True])
  fell = torch.tensor([True, False, False, True, True, True])
  log = p.record(ended, fell)
  # env 4 is still running, env 5 never had a push
  assert log == {"push/surv_from_behind": 0.5, "push/surv_from_right": 1.0,
                 "push/surv_from_front": 0.0}
  # the next episodes blend in at (1 - ema) ** count
  p.rel = torch.deg2rad(torch.tensor([10.0, 20.0, 0.0, 0.0, -100.0, 0.0]))
  ended = torch.tensor([True, True, False, False, True, False])
  fell = torch.tensor([False, False, False, False, True, False])
  log = p.record(ended, fell)
  assert log["push/surv_from_behind"] == pytest.approx(0.5 * 0.25 + 1.0 * 0.75)
  assert log["push/surv_from_left"] == 0.0 and log["push/surv_from_front"] == 0.0
  p.reset()
  assert bool(torch.isnan(p.rel).all())


# --- the bridge ------------------------------------------------------------------

class _Robot:
  def __init__(self, n=3):
    self.forces = []
    self.data = SimpleNamespace(heading_w=torch.zeros(n))

  def write_external_wrench_to_sim(self, forces, torques, body_ids, env_ids):
    self.forces.append(forces.clone())
    self.body_ids = list(body_ids)


class _Mj:
  def __init__(self, n):
    self.n, self.ctrl, self.num_envs, self.device = n, None, n, "cpu"
    self.end_next = torch.zeros(n, dtype=torch.bool)
    self.step_dt = DT

  def step(self, ctrl):
    self.ctrl = ctrl.clone()
    term, self.end_next = self.end_next, torch.zeros(self.n, dtype=torch.bool)
    z = torch.zeros(self.n)
    return ({"actor": torch.zeros(self.n, 3)}, z, term,
            torch.zeros(self.n, dtype=torch.bool), {"zoo_g": z, "zoo_l": z})


def _core(n=3, **push_kw):
  from robot_safety_sandbox.base import _MjlabCore
  core = _MjlabCore.__new__(_MjlabCore)
  core.ctrl_dim, core.dstb_dim, core.ctrl_gain = 29, 3, 1.0
  core.adversary, core.dstb_mode, core.dstb_gain = False, "wrench", 0.25
  core._dstb_slice, core.dstb_scale, core._joint_noise = None, None, None
  core.hybrid_skill, core.dense_reward, core.dense_margins = None, False, False
  core.obs_key, core._log, core._failures_total = "actor", {}, 0
  core._zero_counts()
  core.mj = _Mj(n)
  core._robot, core._adv_body_ids = _Robot(n), [7]
  core._all_ids, core._zero_wrench = torch.arange(n), torch.zeros(n, 1, 3)
  core.force_max = 120.0
  core._push = RandomPush(n, DT, "cpu", first_delay_s=(0.0, 0.0),
                          duration_s=0.2, **push_kw)
  return core


def test_bridge_writes_the_push_wrench_and_restarts_ended_envs():
  torch.manual_seed(2)
  core = _core()
  core.force_scale = torch.tensor([1.0, 0.5, 0.3])
  core._mj_step(torch.zeros(3, 29))
  f = core._robot.forces[-1].reshape(3, 3)
  assert torch.allclose(f.norm(dim=1), 120.0 * core.force_scale)
  assert core._robot.body_ids == [7]
  assert core._log["push/force_max_N"] == 120.0 and core._log["push/active_frac"] == 1.0
  # env 1 ends this step: its push is dropped and (first delay 0) a NEW one
  # starts next step with a fresh direction; the others keep theirs.
  d_before = core._push.dir.clone()
  core.mj.end_next[1] = True
  core._mj_step(torch.zeros(3, 29))
  assert int(core._push.t_left[1]) == 0
  core._mj_step(torch.zeros(3, 29))
  assert torch.allclose(core._push.dir[0], d_before[0])
  assert torch.allclose(core._push.dir[2], d_before[2])
  assert not torch.allclose(core._push.dir[1], d_before[1])
  # after the 10-step push the wrench written is zero
  for _ in range(12):
    core._mj_step(torch.zeros(3, 29))
  assert float(core._robot.forces[-1][0].abs().sum()) == 0.0


def test_bridge_logs_survival_by_the_side_the_push_came_from():
  torch.manual_seed(4)
  core = _core()
  core._robot.data.heading_w = torch.tensor([0.0, 1.0, -2.0])
  core._mj_step(torch.zeros(3, 29))              # every env's push lands
  rel = core._push.rel.clone()
  d = core._push.dir
  world = torch.atan2(d[:, 1], d[:, 0])
  assert torch.allclose(torch.remainder(world - core._robot.data.heading_w - rel
                                        + torch.pi, 2 * torch.pi) - torch.pi,
                        torch.zeros(3), atol=1e-5)
  core.mj.end_next[1] = True                     # env 1 falls
  core._mj_step(torch.zeros(3, 29))
  side = ("behind", "right", "front", "left")[
    int(torch.remainder(rel[1] + torch.pi / 4, 2 * torch.pi) // (torch.pi / 2))]
  assert core._log[f"push/surv_from_{side}"] == 0.0
  assert sum(k.startswith("push/surv_from_") for k in core._log) == 1


def test_random_push_is_refused_with_an_adversary_or_off_the_wrench():
  from robot_safety_sandbox.base import _MjlabCore
  for kw in (dict(adversary=True, dstb_mode="wrench", dstb_dim=3),
             dict(adversary=False, dstb_mode="joint", dstb_dim=14,
                  dstb_joint_slice=(15, 29))):
    core = _MjlabCore.__new__(_MjlabCore)
    with pytest.raises(ValueError, match="random_push"):
      core._init_core(2, "cpu", None, None, ctrl_dim=29,
                      ctrl_gain=1.0, force_max=15.0, adversary_body="torso_link",
                      render_mode=None, random_push={}, **kw)


# --- the regime ------------------------------------------------------------------

def test_regime_composes_spawn_source_and_force_bucket():
  from robot_safety_sandbox.base import _MjlabCore
  core = _MjlabCore.__new__(_MjlabCore)
  core.mj = SimpleNamespace(num_envs=5, device="cpu",
                            _zoo_spawn_source=torch.tensor([0, 1, 2, 3, 4]),
                            _zoo_n_spawn_sources=5)
  core._regime_buckets = 4
  assert core.n_regimes == 20
  assert core.regime().tolist() == [0, 4, 8, 12, 16]            # no scale yet: bucket 0
  core.force_scale = torch.tensor([0.0, 0.3, 0.5, 0.74, 1.0])
  assert core.regime().tolist() == [0, 5, 10, 14, 19]
  # the spawn's tensor is read live: a reset that moves a source moves its regime
  core.mj._zoo_spawn_source[0] = 2
  assert int(core.regime()[0]) == 8
  # no buckets -> the source alone; neither -> None
  core._regime_buckets = 0
  assert core.n_regimes == 5 and core.regime().tolist() == [2, 1, 2, 3, 4]
  core.mj = SimpleNamespace(num_envs=5, device="cpu")
  assert core.n_regimes == 1 and core.regime() is None


# --- the registration --------------------------------------------------------------

def test_recover_push_task_registers_single_player_avoid():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  s = spec("g1_recover_push")
  assert s.mode == AVOID and not s.supports_adversary and s.ctrl_dim == 29
  assert s.kwargs["random_push"] is not None and s.kwargs["dstb_mode"] == "wrench"
  assert s.kwargs["regime_force_buckets"] == 4
  assert s.kwargs["adversary_body"] == "torso_link"
  assert algo_name("g1_recover_push", adversary=False, family="on_policy") == "SafetyPPO1P"
  with pytest.raises(ValueError):
    algo_name("g1_recover_push", adversary=True, family="on_policy")


def test_recover_push_rear_is_push_fast_with_rear_biased_directions():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  fast, rear = spec("g1_recover_push_fast"), spec("g1_recover_push_rear")
  assert rear.mode == AVOID and not rear.supports_adversary
  assert rear.cfg_builder is fast.cfg_builder and rear.margin_fn is not None
  rp = dict(rear.kwargs["random_push"])
  assert rp.pop("rear_frac") == 0.5 and rp.pop("rear_half_width_deg") == 60.0
  assert rp == fast.kwargs["random_push"]
  assert {k: v for k, v in rear.kwargs.items() if k != "random_push"} == \
         {k: v for k, v in fast.kwargs.items() if k != "random_push"}
  assert algo_name("g1_recover_push_rear", adversary=False, family="on_policy") == "SafetyPPO1P"
  RandomPush(4, DT, "cpu", **rear.kwargs["random_push"])     # the kwargs construct


# --- one push per episode -----------------------------------------------------------

def _onsets(p, calls, fmax=100.0, heading=None):
  """Per env, the call indices (0-based) at which a push STARTS acting."""
  prev = torch.zeros(p.n, dtype=torch.bool)
  out = [[] for _ in range(p.n)]
  for c in range(calls):
    _f, a = p.forces(fmax, None, heading)
    for i in (a & ~prev).nonzero().flatten().tolist():
      out[i].append(c)
    prev = a.clone()
  return out


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_max_pushes_caps_the_random_onsets_of_an_episode(cap):
  torch.manual_seed(5)
  kw = dict(interval_s=(0.1, 0.1), duration_s=0.04, first_delay_s=(0.0, 0.0))
  p = RandomPush(3, DT, "cpu", max_pushes=cap, **kw)
  assert [len(o) for o in _onsets(p, 60)] == [cap] * 3
  assert p.n_pushes.tolist() == [cap] * 3
  p.reset(torch.tensor([True, False, False]))      # a new episode may push again
  assert [len(o) for o in _onsets(p, 60)] == [cap, 0, 0]
  # unlimited: an onset every 7 calls (the 2-call push + the 0.1 s interval)
  p = RandomPush(3, DT, "cpu", **kw)
  assert _onsets(p, 60) == [list(range(0, 60, 7))] * 3
  with pytest.raises(ValueError, match="max_pushes"):
    RandomPush(3, DT, "cpu", max_pushes=-1)


def test_an_unreachable_cap_draws_exactly_the_unlimited_process():
  def run(**kw):
    torch.manual_seed(11)
    p = RandomPush(16, DT, "cpu", interval_s=(0.1, 0.3), first_delay_s=(0.0, 0.2),
                   rear_frac=0.5, **kw)
    out = []
    for t in range(200):
      heading = torch.linspace(-3.0, 3.0, 16) + 0.01 * t
      f, a = p.forces(80.0, torch.linspace(0.2, 1.0, 16), heading)
      out.append((f.clone(), a.clone()))
      p.reset(torch.arange(16) == t % 16)
    return out
  for (f0, a0), (f1, a1) in zip(run(), run(max_pushes=10 ** 6)):
    assert torch.equal(f0, f1) and torch.equal(a0, a1)


def test_forces_outputs_are_not_mutated_by_a_later_reset():
  p = RandomPush(3, DT, "cpu", first_delay_s=(0.0, 0.0))
  f, a = p.forces(50.0)
  f0, a0 = f.clone(), a.clone()
  assert torch.equal(p.cur_force_w, f) and bool(p.active().all())
  p.reset()
  assert torch.equal(f, f0) and torch.equal(a, a0)
  assert float(p.cur_force_w.abs().sum()) == 0.0 and not bool(p.active().any())


# --- scripted pushes ------------------------------------------------------------------

def _quiet_process(n, **kw):
  """No random onset within the test horizons (first delay 10 s)."""
  return RandomPush(n, DT, "cpu", first_delay_s=(10.0, 10.0), **kw)


def test_scripted_push_onset_zero_acts_on_the_next_call_for_exactly_steps_calls():
  p = _quiet_process(4)
  mask = torch.tensor([True, False, True, False])
  p.schedule(mask, force_n=torch.tensor([70.0, 0.0, 30.0, 0.0]), steps=3, onset=0,
             dir_w=torch.tensor([[2.0, 0.0, 0.0]] * 4))    # normalized
  assert torch.equal(p.quiet(), ~mask)       # armed on 0, 2; 1, 3 between pushes
  scale = torch.full((4,), 0.5)
  want = torch.tensor([[70.0, 0, 0], [0, 0, 0], [30.0, 0, 0], [0, 0, 0]])
  for _ in range(3):
    f, a = p.forces(100.0, scale)                           # force_max x scale ignored
    assert torch.equal(a, mask) and torch.allclose(f, want)
    assert torch.equal(p.cur_force_w, f) and bool(p.scripted[0])
    assert torch.equal(p.active(), mask)
  assert p.t_left.tolist() == [0, 0, 0, 0]
  assert torch.equal(p.active(), mask)          # the push acted in the last call
  assert torch.equal(p.scripted, mask)          # ... and it was the scripted one
  f, a = p.forces(100.0, scale)
  assert not bool(a.any()) and float(f.abs().sum()) == 0.0
  assert not bool(p.active().any()) and not bool(p.scripted.any())
  assert torch.equal(p.cur_force_w, torch.zeros(4, 3))
  # not counted toward a cap by default, and the age restarts at the onset
  assert p.n_pushes.tolist() == [0, 0, 0, 0] and int(p.age[0]) == 4


def test_scripted_push_onset_k_waits_k_calls():
  p = _quiet_process(3)
  p.schedule(torch.ones(3, dtype=torch.bool), force_n=10.0,
             steps=torch.tensor([2, 1, 3]), onset=torch.tensor([0, 2, 5]),
             dir_w=torch.tensor([0.0, 1.0, 0.0]))
  on = _onsets(p, 12)
  assert on == [[0], [2], [5]]
  # and each lasted exactly its steps: replay to read the active mask per call
  p = _quiet_process(3)
  p.schedule(torch.ones(3, dtype=torch.bool), force_n=10.0,
             steps=torch.tensor([2, 1, 3]), onset=torch.tensor([0, 2, 5]),
             dir_w=torch.tensor([0.0, 1.0, 0.0]))
  acts = torch.stack([p.forces(1.0)[1] for _ in range(10)])
  assert acts.sum(0).tolist() == [2, 1, 3]
  assert acts[:, 2].nonzero().flatten().tolist() == [5, 6, 7]


def test_scripted_rel_resolves_against_the_heading_at_the_onset():
  p = _quiet_process(3)
  rel = torch.tensor([0.0, torch.pi / 2, 3 * torch.pi])     # 3 pi wraps to pi
  p.schedule(torch.ones(3, dtype=torch.bool), force_n=10.0, steps=1, onset=1, rel=rel)
  p.forces(1.0, None, torch.tensor([9.0, 9.0, 9.0]))        # not the onset: ignored
  h = torch.tensor([0.0, 1.0, -2.0])
  f, a = p.forces(1.0, None, h)
  ang = torch.tensor([0.0, torch.pi / 2, torch.pi]) + h
  want = 10.0 * torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros(3)], dim=1)
  assert bool(a.all()) and torch.allclose(f, want, atol=1e-5)
  assert torch.allclose(torch.cos(p.rel), torch.cos(rel), atol=1e-6)
  # a rel push needs the heading at its onset
  p = _quiet_process(2)
  p.schedule(torch.ones(2, dtype=torch.bool), force_n=10.0, steps=1, rel=0.0)
  with pytest.raises(ValueError, match="heading"):
    p.forces(1.0)


def test_scripted_world_push_is_labelled_by_the_side_it_comes_from():
  p = _quiet_process(2)
  p.schedule(torch.ones(2, dtype=torch.bool), force_n=50.0, steps=2,
             dir_w=torch.tensor([1.0, 0.0, 0.0]))
  p.forces(1.0, None, torch.tensor([0.0, torch.pi]))       # heading x / -x
  assert torch.allclose(p.rel.abs(), torch.tensor([0.0, torch.pi]), atol=1e-6)
  log = p.record(torch.ones(2, dtype=torch.bool), torch.tensor([False, True]))
  assert log == {"push/surv_from_behind": 1.0, "push/surv_from_front": 0.0}
  # without a heading the side stays unknown
  p = _quiet_process(1)
  p.schedule(torch.ones(1, dtype=torch.bool), force_n=5.0, steps=1,
             dir_w=torch.tensor([1.0, 0.0, 0.0]))
  p.forces(1.0)
  assert bool(torch.isnan(p.rel).all())


def test_scripted_push_cuts_the_random_one_and_random_pushes_resume_after():
  torch.manual_seed(0)
  # random: an onset every 10 calls (0.2 s), each 10 calls long, from call 0
  p = RandomPush(2, DT, "cpu", interval_s=(0.2, 0.2), duration_s=0.2,
                 first_delay_s=(0.0, 0.0))
  f, a = p.forces(100.0)
  assert bool(a.all())
  p.schedule(torch.tensor([True, False]), force_n=50.0, steps=2, onset=3,
             dir_w=torch.tensor([0.0, 1.0, 0.0]))
  acts, forces = [], []
  for _ in range(20):
    f, a = p.forces(100.0)
    acts.append(bool(a[0]))
    forces.append(f[0].clone())
  # 3 quiet calls (the random push was cut, no random onset while armed),
  # 2 scripted calls at exactly 50 N along +y, then quiet until one interval
  # (10 calls) + the scripted duration (2) after the scripted onset
  assert acts[:5] == [False, False, False, True, True]
  assert torch.allclose(forces[3], torch.tensor([0.0, 50.0, 0.0]))
  assert acts[5:15] == [False] * 10 and acts[15] is True
  assert torch.allclose(forces[15].norm(), torch.tensor(100.0))
  assert int(p.n_pushes[0]) == 2                  # two random onsets, scripted not counted


@pytest.mark.parametrize("count", [False, True])
def test_count_makes_a_scripted_push_use_up_the_cap(count):
  p = RandomPush(1, DT, "cpu", interval_s=(0.1, 0.1), duration_s=0.04,
                 first_delay_s=(0.2, 0.2), max_pushes=1)
  p.schedule(torch.ones(1, dtype=torch.bool), force_n=20.0, steps=2, count=count,
             dir_w=torch.tensor([1.0, 0.0, 0.0]))
  on = _onsets(p, 40)[0]
  # the scripted push at call 0; uncounted, the one random push follows one
  # interval (5) + its duration (2) after it, at call 7
  assert on == ([0] if count else [0, 7])
  assert int(p.n_pushes[0]) == 1
  assert bool(p.quiet()[0])


def test_exhaust_stops_random_onsets_until_the_reset():
  torch.manual_seed(3)
  kw = dict(interval_s=(0.1, 0.1), duration_s=0.04, first_delay_s=(0.1, 0.1))
  p = RandomPush(2, DT, "cpu", **kw)
  p.exhaust(torch.tensor([True, False]))
  on = _onsets(p, 40)
  assert on[0] == [] and len(on[1]) > 3
  p.reset(torch.tensor([True, False]))
  assert _onsets(p, 10)[0] == [5]                 # the first delay again
  # a push already acting is left alone; an armed scripted push too
  p = RandomPush(2, DT, "cpu", duration_s=0.1, first_delay_s=(0.0, 0.0))
  p.forces(1.0)
  p.exhaust(torch.ones(2, dtype=torch.bool))
  p.schedule(torch.tensor([False, True]), force_n=3.0, steps=1, onset=10,
             dir_w=torch.tensor([1.0, 0.0, 0.0]))
  acts = torch.stack([p.forces(1.0)[1] for _ in range(12)])
  assert acts[:4, 0].all() and not acts[4:, 0].any()       # 5-step push ran out
  assert acts[:, 1].nonzero().flatten().tolist() == [10]   # cut, then scripted


def test_quiet_truth_table():
  kw = dict(interval_s=(0.1, 0.1), duration_s=0.04, first_delay_s=(0.04, 0.04))

  def trace(p, calls=6):
    out = [bool(p.quiet()[0])]
    for _ in range(calls):
      p.forces(1.0)
      out.append(bool(p.quiet()[0]))
    return out
  # pushes at calls 2-3, the next at 7: unlimited -> "between pushes"
  assert trace(RandomPush(1, DT, "cpu", **kw)) == [
    True, True, True, False, False, True, True]
  # no random push at all
  assert trace(RandomPush(1, DT, "cpu", max_pushes=0, **kw)) == [True] * 7
  # one push: quiet only once it is over (the push of the last call counts)
  assert trace(RandomPush(1, DT, "cpu", max_pushes=1, **kw)) == [
    False, False, False, False, False, True, True]
  # an armed scripted push is never quiet, even with no random push left
  p = RandomPush(1, DT, "cpu", max_pushes=0, **kw)
  p.schedule(torch.ones(1, dtype=torch.bool), force_n=1.0, steps=1, onset=2,
             dir_w=torch.tensor([1.0, 0.0, 0.0]))
  assert trace(p) == [False, False, False, False, True, True, True]
  # exhaust before the one push: quiet at once
  p = RandomPush(1, DT, "cpu", max_pushes=1, **kw)
  p.exhaust(torch.ones(1, dtype=torch.bool))
  assert trace(p) == [True] * 7
  p.reset()
  assert not bool(p.quiet()[0])


def test_schedule_validation_and_empty_mask():
  p = _quiet_process(2)
  m = torch.ones(2, dtype=torch.bool)
  for bad in (dict(), dict(dir_w=torch.ones(3), rel=0.0)):
    with pytest.raises(ValueError, match="exactly one"):
      p.schedule(m, force_n=1.0, steps=1, **bad)
  for bad in (dict(steps=0), dict(onset=-1), dict(force_n=-1.0)):
    kw = dict(force_n=1.0, steps=1, dir_w=torch.ones(3))
    kw.update(bad)
    with pytest.raises(ValueError):
      p.schedule(m, **kw)
  with pytest.raises(ValueError):
    p.schedule(m, force_n=torch.ones(5), steps=1, dir_w=torch.ones(3))
  p.schedule(torch.zeros(2, dtype=torch.bool), force_n=1.0, steps=1, dir_w=torch.ones(3))
  assert bool(p.quiet().all()) and not p._any_scripted
  # per-selected-env values ([mask.sum()]) are accepted as well as [n]
  p.schedule(torch.tensor([False, True]), force_n=torch.tensor([7.0]), steps=1,
             dir_w=torch.tensor([[0.0, 0.0, 1.0]]))
  f, _a = p.forces(1.0)
  assert torch.allclose(f, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]]))


# --- the bridge's pending push ---------------------------------------------------------

class _SpawnMj(_Mj):
  """A fake mjlab env whose 'spawn' hands the envs that end a pending push,
  and that records the quiet flag the bridge published before each step."""

  def __init__(self, n, force_w, steps, consume):
    super().__init__(n)
    self._zoo_pending_push = {
      "force_w": torch.zeros(n, 3), "steps": torch.zeros(n, dtype=torch.long),
      "consume": torch.zeros(n, dtype=torch.bool)}
    self._row = (force_w, steps, consume)
    self.quiet_at_step = []

  def step(self, ctrl):
    self.quiet_at_step.append(self._zoo_push_quiet.clone())
    ended = self.end_next.clone()
    out = super().step(ctrl)
    pend, (fw, st, co) = self._zoo_pending_push, self._row
    pend["force_w"][ended] = fw[ended]        # in place, as the spawn does
    pend["steps"][ended] = st[ended]
    pend["consume"][ended] = co[ended]
    return out


def test_bridge_resumes_a_pending_push_on_the_first_step_and_consumes_the_random_one():
  torch.manual_seed(6)
  core = _core(max_pushes=1)
  core._push = RandomPush(3, DT, "cpu", first_delay_s=(0.1, 0.1), duration_s=0.04,
                          max_pushes=1)      # random onset at the 6th call, 2 calls
  core.force_scale = torch.tensor([1.0, 0.5, 0.5])
  fw = torch.tensor([[0.0, 0.0, 0.0], [0.0, 40.0, 0.0], [30.0, 0.0, 0.0]])
  core.mj = _SpawnMj(3, fw, steps=torch.tensor([0, 3, 0]),
                     consume=torch.tensor([False, True, False]))
  core._mj_step(torch.zeros(3, 29))          # step 1: envs 1 and 2 end
  core.mj._zoo_push_quiet = None
  core.mj.end_next[1:] = True
  core._mj_step(torch.zeros(3, 29))
  q = core.mj._zoo_push_quiet                # published after the resets
  assert q.tolist() == [False, False, False]
  assert bool(core._push._s_armed[1]) and bool(core._push.blocked[1])
  assert not bool(core._push._s_armed[2]) and not bool(core._push.blocked[2])
  wrench = []
  for _ in range(12):
    core._mj_step(torch.zeros(3, 29))
    wrench.append(core._robot.forces[-1].reshape(3, 3).clone())
  # env 1: the pending push, exactly, on the first 3 steps of its episode;
  # then nothing (its random push was consumed)
  for k in range(3):
    assert torch.allclose(wrench[k][1], fw[1])
  assert all(float(w[1].abs().sum()) == 0.0 for w in wrench[3:])
  # env 2: no pending push -> its random push (5-step first delay, 2 steps) at
  # force_max x scale; env 0 kept its first episode's timing (onset at call 6)
  on2 = [k for k, w in enumerate(wrench) if float(w[2].norm()) > 0]
  assert on2 == [5, 6] and torch.allclose(wrench[5][2].norm(), torch.tensor(60.0))
  on0 = [k for k, w in enumerate(wrench) if float(w[0].norm()) > 0]
  assert on0 == [3, 4]
  # the quiet flag inside each step: env 1 not quiet while its push acts (the
  # push applied in that step counts), quiet from the step after it
  qs = torch.stack(core.mj.quiet_at_step[2:])
  assert qs[:, 1].tolist() == [False] * 3 + [True] * 9
  assert qs[:, 2].tolist() == [False] * 7 + [True] * 5


def test_bridge_publishes_quiet_without_a_pending_push():
  core = _core()
  core._mj_step(torch.zeros(3, 29))
  q = core.mj._zoo_push_quiet
  assert q.dtype == torch.bool and q.shape == (3,)
  assert not bool(q.any())                       # unlimited pushes, one acting


def test_initial_reset_paths_apply_the_pending_push():
  from robot_safety_sandbox.base import MjlabNumpySafetyEnv, MjlabTensorSafetyEnv
  for cls in (MjlabTensorSafetyEnv, MjlabNumpySafetyEnv):
    env = cls.__new__(cls)
    env.mj = SimpleNamespace(
      reset=lambda: ({"actor": torch.zeros(2, 3)}, {}),
      _zoo_pending_push={"force_w": torch.tensor([[0.0, 0.0, 0.0], [0.0, -25.0, 0.0]]),
                         "steps": torch.tensor([0, 4]),
                         "consume": torch.tensor([False, True])})
    env.obs_key, env._joint_noise = "actor", None
    env._hyb_latch = torch.zeros(2, dtype=torch.bool)
    env._push = RandomPush(2, DT, "cpu", first_delay_s=(0.5, 0.5), max_pushes=1)
    env.reset()
    assert env.mj._zoo_push_quiet.tolist() == [False, False]
    f, a = env._push.forces(999.0)
    assert a.tolist() == [False, True]
    assert torch.allclose(f[1], torch.tensor([0.0, -25.0, 0.0]))
    assert env._push.blocked.tolist() == [False, True]
