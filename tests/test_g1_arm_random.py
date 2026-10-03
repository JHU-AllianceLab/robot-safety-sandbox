"""Unit tests for the G1 RANDOM-ARM tasks (tasks/g1_arm_random.py) and the
generator behind them (joint_noise.SmoothJointNoise).

Simulator-free: the bridge is exercised through its real ``_mj_step`` on a stub
core whose ``mj`` records the ctrl it is handed. What they pin:

  * the SIGNAL — starts at the default pose, stays inside its per-joint range,
    is smooth across waypoints (bounded first and second differences), and
    restarts only the envs it is told to;
  * the SPLIT — the arm slice and the one-sided shoulder roll sit on the joints
    their names say, read off the compiled mjlab asset;
  * the BRIDGE — the random command overwrites exactly the arm slice, scaled by
    dstb_gain x dstb_scale (the same authority the arm adversary has), the
    actor's own arm outputs are discarded, and an env that just ended restarts
    its arms from the default pose;
  * the REGISTRATION — single-player, g1_stabilize's margins and head, and a
    bridge that refuses to be given an adversary on top.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.joint_noise import SmoothJointNoise, _min_jerk  # noqa: E402

DT = 0.02   # the G1 cfgs' control step (decimation 4 x 5 ms)


def _run(noise, steps):
  return torch.stack([noise.step() for _ in range(steps)])   # (T, N, D)


# --- the signal ---------------------------------------------------------------

def test_min_jerk_profile_is_rest_to_rest():
  u = torch.tensor([0.0, 1.0])
  assert _min_jerk(u).tolist() == pytest.approx([0.0, 1.0])
  h = 1e-4   # zero slope at both ends, so waypoints are passed at rest
  for a in (0.0, 1.0 - h):
    slope = (_min_jerk(torch.tensor(a + h)) - _min_jerk(torch.tensor(a))) / h
    assert abs(float(slope)) < 1e-2


def test_every_episode_opens_at_the_default_pose():
  torch.manual_seed(0)
  noise = SmoothJointNoise(64, 14, DT, "cpu")
  assert torch.equal(noise.step(), torch.zeros(64, 14))


def test_commands_stay_inside_each_joints_range():
  torch.manual_seed(0)
  lo = torch.full((14,), -1.0); hi = torch.full((14,), 1.0)
  lo[1], hi[8] = 0.0, 0.0             # the G1's one-sided shoulder rolls
  noise = SmoothJointNoise(128, 14, DT, "cpu", lo=lo.tolist(), hi=hi.tolist())
  out = _run(noise, 1500)
  assert bool((out >= lo - 1e-6).all()) and bool((out <= hi + 1e-6).all())
  # ...and it actually uses them rather than idling near the start pose.
  span = out.amax(dim=(0, 1)) - out.amin(dim=(0, 1))
  assert bool((span > 0.9 * (hi - lo)).all()), span


def test_commands_are_smooth_across_waypoints():
  """Bounded by the min-jerk peaks: |v| <= 1.875 D/T, |a| <= 5.77 D/T^2, with
  D the widest waypoint jump and T the shortest segment. A step, or a restart
  that did not start from the waypoint just reached, would blow both."""
  torch.manual_seed(0)
  seg_min = 0.5
  noise = SmoothJointNoise(256, 14, DT, "cpu", seg_min_s=seg_min, seg_max_s=2.0)
  out = _run(noise, 1000)             # ~20 s: many segment boundaries per joint
  d1 = (out[1:] - out[:-1]).abs().max()
  d2 = (out[2:] - 2 * out[1:-1] + out[:-2]).abs().max()
  jump = 2.0
  assert float(d1) <= 1.875 * jump * DT / seg_min * 1.05
  assert float(d2) <= 5.78 * jump * (DT / seg_min) ** 2 * 1.05


def test_reset_restarts_only_the_masked_envs():
  torch.manual_seed(0)
  noise = SmoothJointNoise(4, 14, DT, "cpu")
  _run(noise, 40)
  state = lambda: [x.clone() for x in (noise._p0, noise._p1, noise._t,
                                       noise._T)]
  before = state()
  noise.reset(torch.tensor([True, False, True, False]))
  after = state()
  for b, a in zip(before, after):       # untouched envs keep their segment
    assert torch.equal(a[[1, 3]], b[[1, 3]])
  assert torch.equal(after[0][[0, 2]], torch.zeros(2, 14))   # back at default
  assert torch.equal(after[2][[0, 2]], torch.zeros(2, 14))   # clock restarted
  assert torch.equal(noise.step()[[0, 2]], torch.zeros(2, 14))


@pytest.mark.parametrize("kw, msg", [
  (dict(seg_min_s=0.0), "seg_min_s"),
  (dict(seg_min_s=2.0, seg_max_s=1.0), "seg_min_s"),
  (dict(lo=0.5, hi=-0.5), "lo <= hi"),
  (dict(hi=1.5), "lo <= hi"),
])
def test_bad_arguments_fail_loud(kw, msg):
  with pytest.raises(ValueError, match=msg):
    SmoothJointNoise(2, 14, DT, "cpu", **kw)


# --- the joint split ----------------------------------------------------------

def test_one_sided_rolls_sit_on_the_shoulder_roll_joints():
  """Pinned against the COMPILED G1: the waypoint ranges are indexed by slice
  position, so an asset reshuffle must fail here, not silently free the roll
  that keeps the arms out of the torso."""
  mujoco = pytest.importorskip("mujoco")
  g1 = pytest.importorskip("mjlab.asset_zoo.robots.unitree_g1.g1_constants")
  from robot_safety_sandbox.tasks.g1_arm_random import (
    G1_ARM_RANDOM_BRIDGE_KWARGS as KW, G1_ARM_WAYPOINT_HI, G1_ARM_WAYPOINT_LO)
  m = g1.get_spec().compile()
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
           for i in range(1, m.njnt)]
  lo_i, hi_i = KW["dstb_joint_slice"]
  arm = names[lo_i:hi_i]
  assert len(arm) == len(G1_ARM_WAYPOINT_LO) == len(G1_ARM_WAYPOINT_HI) == 14
  for n, lo, hi in zip(arm, G1_ARM_WAYPOINT_LO, G1_ARM_WAYPOINT_HI):
    if n == "left_shoulder_roll_joint":
      assert (lo, hi) == (0.0, 1.0)      # outward = positive on the left
    elif n == "right_shoulder_roll_joint":
      assert (lo, hi) == (-1.0, 0.0)     # mirrored axis on the right
    else:
      assert (lo, hi) == (-1.0, 1.0), n


# --- the bridge ---------------------------------------------------------------

class _Mj:
  """Stands in for the mjlab env: records the ctrl, ends the envs it is told."""

  def __init__(self, n):
    self.n, self.ctrl, self.end_next = n, None, torch.zeros(n, dtype=torch.bool)

  def step(self, ctrl):
    self.ctrl = ctrl.clone()
    term, self.end_next = self.end_next, torch.zeros(self.n, dtype=torch.bool)
    z = torch.zeros(self.n)
    return ({"actor": torch.zeros(self.n, 3)}, z, term,
            torch.zeros(self.n, dtype=torch.bool), {"zoo_g": z, "zoo_l": z})


def _core(n=3, gain=2.0):
  from robot_safety_sandbox.base import _MjlabCore
  from robot_safety_sandbox.tasks.g1_arm_random import (
    G1_ARM_RANDOM_BRIDGE_KWARGS as KW)
  core = _MjlabCore.__new__(_MjlabCore)
  core.ctrl_dim, core.dstb_dim, core.ctrl_gain = 29, 14, 1.0
  core.adversary, core.dstb_mode, core.dstb_gain = False, "joint", gain
  core._dstb_slice, core.dstb_scale = tuple(KW["dstb_joint_slice"]), None
  core._joint_noise = SmoothJointNoise(n, 14, DT, "cpu", **KW["joint_noise"])
  core.hybrid_skill, core.dense_reward, core.dense_margins = None, False, False
  core.obs_key, core._log, core._failures_total = "actor", {}, 0
  core._zero_counts()
  core.mj = _Mj(n)
  return core


def _twin(core):
  """An identically-seeded copy of the core's generator, to predict it."""
  twin = SmoothJointNoise.__new__(SmoothJointNoise)
  twin.__dict__.update({k: (v.clone() if torch.is_tensor(v) else v)
                        for k, v in core._joint_noise.__dict__.items()})
  return twin


def test_bridge_overwrites_only_the_arm_slice_with_the_scaled_noise():
  torch.manual_seed(0)
  core = _core(gain=2.0)
  core.dstb_scale = torch.tensor([1.0, 0.5, 0.3])
  for _ in range(30):                  # get the arms off the default pose
    core._mj_step(torch.zeros(3, 29))
  twin = _twin(core)
  actor = torch.randn(3, 29)
  core._mj_step(actor)
  want = 2.0 * twin.step() * core.dstb_scale.reshape(-1, 1)
  assert torch.equal(core.mj.ctrl[:, :15], actor[:, :15])   # legs + waist: actor
  assert torch.allclose(core.mj.ctrl[:, 15:], want)         # arms: the noise
  assert not torch.allclose(core.mj.ctrl[:, 15:], actor[:, 15:])


def test_bridge_restarts_the_arms_of_an_env_that_just_ended():
  torch.manual_seed(0)
  core = _core()
  for _ in range(30):
    core._mj_step(torch.zeros(3, 29))
  core.mj.end_next = torch.tensor([True, False, False])
  core._mj_step(torch.zeros(3, 29))    # env0's episode ends on this step
  core._mj_step(torch.zeros(3, 29))    # ...so its next command is the default
  assert torch.equal(core.mj.ctrl[0, 15:], torch.zeros(14))
  assert bool(core.mj.ctrl[1:, 15:].abs().sum() > 0)


@pytest.mark.parametrize("over, msg", [
  (dict(adversary=True), "same joint slice"),
  (dict(dstb_mode="wrench", dstb_joint_slice=None), "needs dstb_joint_slice"),
  (dict(dstb_mode="action", dstb_dim=29), "runs alone"),
])
def test_bridge_refuses_joint_noise_it_cannot_honor(over, msg):
  """Validated before any simulator is built, so this needs no GPU."""
  from robot_safety_sandbox.base import _MjlabCore
  kw = dict(ctrl_dim=29, dstb_dim=14, ctrl_gain=1.0, force_max=0.0,
            adversary=False, adversary_body="torso_link", render_mode=None,
            dstb_mode="joint", dstb_joint_slice=(15, 29), joint_noise={})
  kw.update(over)
  core = _MjlabCore.__new__(_MjlabCore)
  with pytest.raises(ValueError, match=msg):
    core._init_core(2, "cpu", None, None, **kw)


def test_bridge_refuses_a_joint_slice_nothing_owns():
  from robot_safety_sandbox.base import _MjlabCore
  core = _MjlabCore.__new__(_MjlabCore)
  with pytest.raises(ValueError, match="only meaningful"):
    core._init_core(2, "cpu", None, None, ctrl_dim=29, dstb_dim=3,
                    ctrl_gain=1.0, force_max=0.0, adversary=True,
                    adversary_body="torso_link", render_mode=None,
                    dstb_mode="wrench", dstb_joint_slice=(15, 29))


# --- the registration ---------------------------------------------------------

def test_random_arm_tasks_register_single_player_on_g1_stabilize_margins():
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
    g1_locomote_env_cfg, g1_stabilize_env_cfg, locomote_margins,
    stance_margins)
  from robot_safety_sandbox.registry import algo_name, spec
  for tid, cfg, margins, twin in (
      ("g1_stabilize_arm_random", g1_stabilize_env_cfg, stance_margins,
       "g1_stabilize_arm_adv"),
      ("g1_locomote_arm_random", g1_locomote_env_cfg, locomote_margins,
       "g1_locomote_arm_adv")):
    s, adv = spec(tid), spec(twin)
    assert s.cfg_builder is cfg and s.margin_fn is margins
    assert algo_name(tid) == "ReachAvoidPPO1P"
    assert not s.supports_adversary
    with pytest.raises(ValueError, match="does not define an adversary"):
      algo_name(tid, adversary=True)
    # same head and arm split as the adversary twin -> checkpoints transfer
    assert (s.ctrl_dim, s.dstb_dim) == (adv.ctrl_dim, adv.dstb_dim)
    assert s.kwargs["dstb_joint_slice"] == adv.kwargs["dstb_joint_slice"]
    assert s.kwargs["dstb_mode"] == "joint" and "joint_noise" in s.kwargs
