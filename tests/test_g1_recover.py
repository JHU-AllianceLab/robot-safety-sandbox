"""Tests for the G1 fall-recovery tasks (tasks/g1_recover.py, envs/g1_recover).

Simulator-free (CPU) tests pin:

  * the FRAMES — heading-frame storage round-trips, and a stored quaternion
    carries no yaw;
  * the BANK — rows split into the spawn's sources (walker-fell rows of either
    kind go to ``fell``) and each source is sorted easy -> hard;
  * the CURRICULUM — levels select the documented quantile windows, a timeout
    promotes and a fall demotes (clamped), the nominal source has no level,
    envs that never spawned are ignored, and the logs carry the rates;
  * the HOLD — with a restored arm command published, the random-arm slice
    starts exactly at that command, follows ONE minimum-jerk glide to the
    generator's first waypoint, and is pure generator output after it;
  * the REGISTRATION — avoid-only, single-player, g1_stabilize's head.

GPU tests (skipped without CUDA) build the real env and check that a restored
row reads back through ``snapshot`` and that the spawn seeds ``last_action``.
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

from robot_safety_sandbox.envs.g1_recover.spawn import (  # noqa: E402
  SOURCES, SRC_CLEAN, SRC_FELL, SRC_NOMINAL, SRC_PUSHED, SRC_SYNTHETIC,
  RecoverCurriculum, band, get_state, source_rows)
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC, from_heading_frame,
  to_heading_frame)
from robot_safety_sandbox.joint_noise import SmoothJointNoise, _min_jerk  # noqa: E402
from mjlab.utils.lab_api.math import yaw_quat  # noqa: E402

DT = 0.02


def _yaw_of(q):
  qy = yaw_quat(q)
  return 2.0 * torch.atan2(qy[:, 3], qy[:, 0])


# --- frames -------------------------------------------------------------------

def test_heading_frame_round_trips_and_strips_yaw():
  torch.manual_seed(0)
  q = torch.nn.functional.normalize(torch.randn(64, 4), dim=1)
  v, w = torch.randn(64, 3), torch.randn(64, 3)
  qh, vh, wh = to_heading_frame(q, v, w)
  assert torch.allclose(_yaw_of(qh), torch.zeros(64), atol=1e-5)
  q2, v2, w2 = from_heading_frame(qh, vh, wh, _yaw_of(q))
  same = torch.minimum((q2 - q).abs().amax(1), (q2 + q).abs().amax(1))
  assert float(same.max()) < 1e-5                       # up to quaternion sign
  assert torch.allclose(v2, v, atol=1e-5) and torch.allclose(w2, w, atol=1e-5)
  # the heading-frame speed is yaw-independent
  assert torch.allclose(vh[:, :2].norm(dim=1), v[:, :2].norm(dim=1), atol=1e-5)


# --- bank ---------------------------------------------------------------------

def _bank():
  kind = torch.tensor([KIND_CLEAN, KIND_CLEAN, KIND_CLEAN, KIND_PUSHED,
                       KIND_PUSHED, KIND_PUSHED, KIND_SYNTHETIC, KIND_SYNTHETIC])
  fell = torch.tensor([False, False, True, False, True, False, False, False])
  return {
    "kind": kind, "fell": fell,
    "speed": torch.tensor([1.2, 0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "push_severity": torch.tensor([0.0, 0.0, 0.0, 0.8, 0.5, 0.2, 0.0, 0.0]),
    "time_to_fall": torch.tensor([math.nan, math.nan, 0.4, math.nan, 1.5,
                                  math.nan, math.nan, math.nan]),
    "difficulty": torch.tensor([0, 0, 0, 0, 0, 0, 0.9, 0.1]),
  }


def test_bank_splits_into_sources_sorted_easy_to_hard():
  rows = source_rows(_bank())
  assert rows["clean"].tolist() == [1, 0]           # by speed
  assert rows["pushed"].tolist() == [5, 3]          # by push size, fell rows out
  assert rows["fell"].tolist() == [4, 2]            # longest time-to-fall first
  assert rows["synthetic"].tolist() == [7, 6]       # by perturbation scale


def test_levels_select_the_documented_windows():
  lo, hi = band(torch.arange(6), 5, 0.4)
  assert torch.allclose(hi, torch.arange(1, 7) / 6.0)
  assert float(lo[0]) == 0.0 and float(lo[1]) == 0.0
  assert torch.allclose(lo[5:], torch.tensor([0.6]))
  assert bool((hi - lo <= 0.4 + 1e-6).all())


# --- curriculum ---------------------------------------------------------------

def _fake_env(n, terminated):
  return SimpleNamespace(num_envs=n, device="cpu",
                         termination_manager=SimpleNamespace(
                           terminated=terminated))


def test_curriculum_promotes_on_timeout_and_demotes_on_fall():
  env = _fake_env(5, torch.tensor([False, True, False, True, False]))
  cur = RecoverCurriculum(SimpleNamespace(params={"levels": 5}), env)
  st = get_state(env)
  st.spawned[:] = True
  st.source[:] = torch.tensor([SRC_CLEAN, SRC_FELL, SRC_NOMINAL, SRC_PUSHED,
                               SRC_SYNTHETIC])
  st.level[:] = 2
  st.level[0, SRC_CLEAN] = 5                         # already at the top
  st.level[1, SRC_FELL] = 0                          # already at the bottom
  log = cur(env, torch.arange(5))
  assert int(st.level[0, SRC_CLEAN]) == 5            # timeout, clamped
  assert int(st.level[1, SRC_FELL]) == 0             # fall, clamped
  assert int(st.level[2, SRC_NOMINAL]) == 2          # nominal has no level
  assert int(st.level[3, SRC_PUSHED]) == 1           # fall -> down
  assert int(st.level[4, SRC_SYNTHETIC]) == 3        # timeout -> up
  assert int(st.level[0, SRC_PUSHED]) == 2           # other sources untouched
  assert log["surv_clean"] == 1.0 and log["surv_fell"] == 0.0
  assert log["surv_pushed"] == 0.0 and log["surv_nominal"] == 1.0
  # the four walker / synthetic sources always; mocap / handover only when the
  # spawn draws them by the ladder with a weight > 0 (none here)
  assert {f"lvl_{s}" for s in SOURCES[1:5]} <= set(log)
  assert "lvl_mocap" not in log and "lvl_handover" not in log
  # SB3 truncates console keys past 36 chars ("   " + key); none may collide
  assert all(3 + len(f"Curriculum/recover/{k}") <= 36 for k in log)


def test_curriculum_ignores_envs_that_never_spawned():
  env = _fake_env(3, torch.tensor([True, True, True]))
  cur = RecoverCurriculum(SimpleNamespace(params={}), env)
  st = get_state(env)
  st.source[:] = SRC_CLEAN
  before = st.level.clone()
  assert cur(env, torch.arange(3)) == {}
  assert torch.equal(st.level, before)


# --- hold ---------------------------------------------------------------------

class _Mj:
  def __init__(self, n, reset_action):
    self.n, self.ctrl = n, None
    self.end_next = torch.zeros(n, dtype=torch.bool)
    self._zoo_reset_action = reset_action
    self.step_dt = DT

  def step(self, ctrl):
    self.ctrl = ctrl.clone()
    term, self.end_next = self.end_next, torch.zeros(self.n, dtype=torch.bool)
    z = torch.zeros(self.n)
    return ({"actor": torch.zeros(self.n, 3)}, z, term,
            torch.zeros(self.n, dtype=torch.bool), {"zoo_g": z, "zoo_l": z})


def _core(reset_action, n=3, gain=2.0):
  from robot_safety_sandbox.base import _MjlabCore
  from robot_safety_sandbox.tasks.g1_arm_random import (
    G1_ARM_RANDOM_BRIDGE_KWARGS as KW)
  core = _MjlabCore.__new__(_MjlabCore)
  core.ctrl_dim, core.dstb_dim, core.ctrl_gain = 29, 14, 1.0
  core.adversary, core.dstb_mode, core.dstb_gain = False, "joint", gain
  core._dstb_slice, core.dstb_scale = tuple(KW["dstb_joint_slice"]), None
  core._joint_noise = SmoothJointNoise(n, 14, DT, "cpu", **KW["joint_noise"])
  core._hold = torch.zeros(n, 14)
  core._hold_t = torch.zeros(n, 14)
  core._hold_T = torch.ones(n, 14)
  core._hold_on = False
  core.hybrid_skill, core.dense_reward, core.dense_margins = None, False, False
  core.obs_key, core._log, core._failures_total = "actor", {}, 0
  core._zero_counts()
  core.mj = _Mj(n, reset_action)
  core._reset_joint_noise()
  return core


def test_arms_start_at_the_restored_command_and_glide_once():
  torch.manual_seed(0)
  a0 = torch.zeros(3, 29)
  a0[:, 15:] = torch.linspace(-2.5, 2.5, 14)
  core = _core(a0)
  w1 = core._joint_noise._p1.clone()                  # first waypoints
  T = core._joint_noise._T.clone()                    # first-segment lengths
  steps = int(math.ceil(float(T.max()) / DT)) + 5
  for k in range(steps):
    core._mj_step(torch.zeros(3, 29))
    arms = core.mj.ctrl[:, 15:]
    u = (k * DT / T).clamp(0.0, 1.0)
    inside = u < 1.0
    want = a0[:, 15:] + (2.0 * w1 - a0[:, 15:]) * _min_jerk(u)
    assert torch.allclose(arms[inside], want[inside], atol=1e-5), k
    if k == 0:
      assert torch.allclose(arms, a0[:, 15:])         # no snap at the spawn
  assert float(core._hold_glide().abs().max()) == 0.0  # hold is gone after T


def test_an_env_that_ends_restarts_its_arms_from_its_new_restored_command():
  torch.manual_seed(0)
  core = _core(torch.zeros(3, 29))
  for _ in range(20):
    core._mj_step(torch.zeros(3, 29))
  new = torch.zeros(3, 29)
  new[:, 15:] = 1.7
  core.mj._zoo_reset_action = new                     # what the spawn wrote
  core.mj.end_next = torch.tensor([True, False, False])
  core._mj_step(torch.zeros(3, 29))                   # env0 ends here
  core._mj_step(torch.zeros(3, 29))                   # its first command
  assert torch.allclose(core.mj.ctrl[0, 15:], torch.full((14,), 1.7))
  assert not torch.allclose(core.mj.ctrl[1, 15:], torch.full((14,), 1.7))


def test_without_a_published_action_the_hold_stays_off():
  from robot_safety_sandbox.base import _MjlabCore
  core = _core(torch.zeros(3, 29))
  core._hold_on = False
  del core.mj._zoo_reset_action
  _MjlabCore._reset_joint_noise(core)
  assert core._hold_on is False


class _Robot:
  """Records the external wrench the bridge writes."""

  def write_external_wrench_to_sim(self, forces, torques, body_ids, env_ids):
    self.forces, self.body_ids = forces.clone(), list(body_ids)


def test_a_wrench_adversary_pushes_while_the_noise_keeps_the_arms():
  """g1_recover_arm_random_wrench's step: actions = [29 ctrl | 3 wrench]. The
  arms are the noise's (the actor's arm outputs are discarded as before), the
  wrench is the adversary's direction at force_max x force_scale, on the body."""
  torch.manual_seed(0)
  core = _core(torch.zeros(3, 29))
  core.adversary, core.dstb_mode, core.dstb_dim = True, "wrench", 3
  core.force_max, core.force_scale = 15.0, torch.tensor([1.0, 0.5, 0.3])
  core._robot, core._adv_body_ids = _Robot(), [7]
  core._all_ids, core._zero_wrench = torch.arange(3), torch.zeros(3, 1, 3)
  for _ in range(60):                  # get the arms off the default pose; the
    core._mj_step(torch.zeros(3, 32))  # zero restored command holds nothing
  twin = SmoothJointNoise.__new__(SmoothJointNoise)
  twin.__dict__.update({k: (v.clone() if torch.is_tensor(v) else v)
                        for k, v in core._joint_noise.__dict__.items()})
  actions = torch.randn(3, 32)
  core._mj_step(actions)
  assert torch.equal(core.mj.ctrl[:, :15], actions[:, :15])     # actor
  assert torch.allclose(core.mj.ctrl[:, 15:], 2.0 * twin.step())  # noise
  unit = actions[:, 29:] / actions[:, 29:].norm(dim=1, keepdim=True)
  want = unit * 15.0 * core.force_scale.reshape(-1, 1)
  assert torch.allclose(core._robot.forces.reshape(3, 3), want)
  assert core._robot.body_ids == [7]


def test_bridge_accepts_joint_noise_under_a_wrench_adversary():
  """Validated before any simulator is built (the mjlab build then fails on the
  None cfg_builder, which is past every check)."""
  from robot_safety_sandbox.base import _MjlabCore
  core = _MjlabCore.__new__(_MjlabCore)
  with pytest.raises(TypeError):
    core._init_core(2, "cpu", None, None, ctrl_dim=29, dstb_dim=3,
                    ctrl_gain=1.0, force_max=15.0, adversary=True,
                    adversary_body="torso_link", render_mode=None,
                    dstb_mode="wrench", dstb_joint_slice=(15, 29),
                    joint_noise={})
  assert core._dstb_slice == (15, 29) and core.dstb_dim == 3


# --- registration -------------------------------------------------------------

def test_recover_tasks_register_avoid_only_single_player():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  full, arm = spec("g1_recover"), spec("g1_recover_arm_random")
  for s in (full, arm):
    assert s.mode == AVOID and s.ctrl_dim == 29 and not s.supports_adversary
    assert getattr(s.margin_fn, "has_target", True) is False
    assert algo_name(s.task_id) == "SafetyPPO1P"
    assert s.cfg_builder is full.cfg_builder
  assert "joint_noise" not in full.kwargs
  assert arm.kwargs["dstb_joint_slice"] == (15, 29) and arm.dstb_dim == 14


def test_recover_wrench_task_is_the_random_arm_task_plus_a_torso_wrench():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  arm, w = spec("g1_recover_arm_random"), spec("g1_recover_arm_random_wrench")
  assert w.mode == AVOID and w.supports_adversary
  assert algo_name(w.task_id) == "SafetyPPO1P"
  assert algo_name(w.task_id, adversary=True) == "SafetyPPO2P"
  # same env, margins, head and random arms -> a g1_recover_arm_random
  # checkpoint warm-starts whole
  assert w.cfg_builder is arm.cfg_builder and w.ctrl_dim == arm.ctrl_dim
  assert getattr(w.margin_fn, "has_target", True) is False
  for k in ("joint_noise", "dstb_joint_slice", "dstb_gain", "ctrl_gain",
            "action_bound"):
    assert w.kwargs[k] == arm.kwargs[k], k
  # the learned player is g1_stabilize's wrench
  assert w.dstb_dim == 3 and w.kwargs["dstb_mode"] == "wrench"
  assert w.kwargs["adversary_body"] == spec("g1_stabilize").kwargs["adversary_body"]


def test_recover_wrench_task_is_g1_recover_plus_a_torso_wrench():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  full, w = spec("g1_recover"), spec("g1_recover_wrench")
  assert w.mode == AVOID and w.supports_adversary
  assert algo_name(w.task_id) == "SafetyPPO1P"
  assert algo_name(w.task_id, adversary=True) == "SafetyPPO2P"
  # same env, margins and head -> a g1_recover checkpoint warm-starts whole
  assert w.cfg_builder is full.cfg_builder and w.ctrl_dim == full.ctrl_dim
  assert getattr(w.margin_fn, "has_target", True) is False
  for k in ("ctrl_gain", "action_bound"):
    assert w.kwargs[k] == full.kwargs[k], k
  assert "joint_noise" not in w.kwargs and "dstb_joint_slice" not in w.kwargs
  assert w.dstb_dim == 3 and w.kwargs["dstb_mode"] == "wrench"
  assert w.kwargs["adversary_body"] == spec("g1_stabilize").kwargs["adversary_body"]


# --- GPU: the real env ----------------------------------------------------------

_CUDA = torch.cuda.is_available()


def _tiny_bank(default_q: torch.Tensor, path: str) -> dict:
  """Four rows per kind: standing pose, 1 m/s forward, a recognizable action."""
  n, nj = 12, default_q.numel()
  kind = torch.tensor([KIND_CLEAN] * 4 + [KIND_PUSHED] * 4
                      + [KIND_SYNTHETIC] * 4)
  bank = {
    "root_z": torch.full((n,), 0.76),
    "root_quat_h": torch.tensor([1.0, 0, 0, 0]).expand(n, 4).clone(),
    "lin_vel_h": torch.tensor([1.0, 0, 0]).expand(n, 3).clone(),
    "ang_vel_h": torch.zeros(n, 3),
    "joint_pos": default_q.expand(n, nj).clone(),
    "joint_vel": torch.zeros(n, nj),
    "action": (0.01 * torch.arange(nj, dtype=torch.float)).expand(n, nj).clone(),
    "command": torch.zeros(n, 3), "phase_step": torch.zeros(n, dtype=torch.long),
    "kind": kind, "fell": torch.zeros(n, dtype=torch.bool),
    "time_to_fall": torch.full((n,), math.nan),
    "push_severity": torch.linspace(0, 1, n), "difficulty": torch.linspace(0, 1, n),
    "speed": torch.ones(n), "env_id": torch.zeros(n, dtype=torch.long),
  }
  assert set(FIELDS) <= set(bank)
  torch.save(bank, path)
  return bank


@pytest.mark.skipif(not _CUDA, reason="needs CUDA (mjlab/mujoco_warp)")
def test_spawn_places_bank_rows_and_seeds_last_action():
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_recover.states import snapshot
  with tempfile.TemporaryDirectory() as tmp:
    probe = make_tensor("g1_recover", 4, "cuda:0",
                        cfg_overrides={"dataset": os.path.join(tmp, "none.pt")})
    probe.reset()
    default_q = probe.mj.scene["robot"].data.default_joint_pos[0].cpu()
    probe.close()
    path = os.path.join(tmp, "bank.pt")
    bank = _tiny_bank(default_q, path)
    env = make_tensor("g1_recover", 32, "cuda:0", cfg_overrides={
      "dataset": path, "weights": {"nominal": 0, "clean": 1, "pushed": 0,
                                   "fell": 0, "synthetic": 0}})
    env.reset()
    mj = env.mj
    assert bool((mj._recover_state.source == SRC_CLEAN).all())
    want = bank["action"][0].to("cuda:0")
    assert torch.allclose(mj.action_manager.action, want.expand(32, -1))
    s = snapshot(mj)                                   # read back the restore
    assert torch.allclose(s["lin_vel_h"], torch.tensor(
      [1.0, 0.0, 0.0], device="cuda:0").expand(32, 3), atol=1e-4)
    assert torch.allclose(s["root_z"], torch.full((32,), 0.76, device="cuda:0"),
                          atol=1e-4)
    assert torch.allclose(s["joint_pos"], default_q.to("cuda:0").expand(32, -1),
                          atol=1e-5)
    env.close()
