"""Unit tests for the G1 ARM-adversary task (envs/g1_arm_adversary/).

Simulator-free: a stub stands in for the mjlab env, so these run on CPU and pin
the CONTRACT rather than any physics. What they pin:

  * the JOINT SPLIT — that action dims 15..28 really are the 14 arm joints, read
    off the compiled mjlab asset, so an asset reshuffle fails here instead of
    silently handing the adversary somebody's knee;
  * the ZERO-SUM ESCAPE HATCH — the self-collision penalty appears only while the
    DISTURBANCE player is collecting, never in ``g``, and never at all under
    ``end_criterion="reach-avoid"``;
  * that with no arm self-contact the margins are bit-identical to
    ``g1_stabilize``'s, which is what makes a 1P run of this task an honest
    baseline and a warm-start from ``g1_stabilize`` meaningful;
  * the bridge's joint-partition validation (``dstb_mode="joint"``).

The margin STUBS reuse tests/test_g1_margins.py's ``_Env``, because the arm task's
g and l ARE ``g1_stabilize``'s — only the disturbance-phase shaping is new.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (  # noqa: E402
  CLAMP, locomote_margins, stance_margins)
from robot_safety_sandbox.envs.g1_arm_adversary.env_cfg import (  # noqa: E402
  ARM_CONTACT_THRESHOLD, ARM_DIM, ARM_GEOMS, ARM_SELFCOL_SENSOR, ARM_SLICE,
  N_ACTUATED, arm_locomote_margins, arm_stance_margins)
from test_g1_margins import _Env, _Sensor  # noqa: E402


def _env(*, contact_arm_n=0.0, dstb_phase=False, end_criterion="failure",
         with_arm_sensor=True, **kw):
  """test_g1_margins._Env plus the arm self-collision sensor and the two flags
  the bridge stamps on the mjlab env."""
  env = _Env(**kw)
  if with_arm_sensor:
    # [B, N_primaries, H, 3] — the shape an 8-geom, history_length=4 sensor emits.
    hist = torch.zeros(1, 8, 4, 3)
    hist[0, 3, 1, 0] = float(contact_arm_n)
    inst = torch.zeros(1, 8, 3)
    inst[0, 3, 0] = float(contact_arm_n)
    env.scene[ARM_SELFCOL_SENSOR] = _Sensor(force_history=hist, force=inst)
  env._zoo_dstb_phase = dstb_phase
  env._zoo_end_criterion = end_criterion
  return env


# --- the joint split ----------------------------------------------------------

def test_arm_slice_really_is_the_arms():
  """Pinned against the COMPILED mjlab G1, not against a comment: the whole task
  rests on dims 15..28 being arm joints and 0..14 being legs and waist."""
  mujoco = pytest.importorskip("mujoco")
  g1 = pytest.importorskip(
    "mjlab.asset_zoo.robots.unitree_g1.g1_constants")
  m = g1.get_spec().compile()
  # joint 0 is the floating base; the actuated joints follow in action order.
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
           for i in range(1, m.njnt)]
  assert len(names) == N_ACTUATED
  ctrl_names, arm_names = names[:ARM_SLICE.start], names[ARM_SLICE]
  assert len(arm_names) == ARM_DIM
  assert all(("shoulder" in n or "elbow" in n or "wrist" in n)
             for n in arm_names), arm_names
  assert all(("hip" in n or "knee" in n or "ankle" in n or "waist" in n)
             for n in ctrl_names), ctrl_names


def test_arm_geoms_exist_on_the_asset():
  mujoco = pytest.importorskip("mujoco")
  g1 = pytest.importorskip(
    "mjlab.asset_zoo.robots.unitree_g1.g1_constants")
  m = g1.get_spec().compile()
  present = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(m.ngeom)}
  assert set(ARM_GEOMS) <= present, set(ARM_GEOMS) - present


# --- the disturbance-phase penalty --------------------------------------------

def test_control_phase_margins_are_g1_stabilize_exactly():
  """A 1P run, and every control-player rollout of a 2P run, must see the
  unmodified task — even with the arms hard against the torso."""
  for n in (0.0, 40.0, 5000.0):
    env = _env(contact_arm_n=n, dstb_phase=False)
    g_ref, l_ref = stance_margins(env)
    g, l = arm_stance_margins(env)
    assert torch.equal(g, g_ref)
    assert torch.equal(l, l_ref)


def test_no_arm_contact_is_identical_in_both_phases():
  """frac = 0 has to be an exact no-op, not a near-one: this is what keeps the
  two players' problems the same game when the adversary behaves."""
  ctrl = _env(contact_arm_n=0.0, dstb_phase=False)
  dstb = _env(contact_arm_n=0.0, dstb_phase=True)
  assert torch.equal(arm_stance_margins(ctrl)[1], arm_stance_margins(dstb)[1])
  assert torch.equal(arm_locomote_margins(ctrl)[1], arm_locomote_margins(dstb)[1])


def test_self_collision_raises_l_only_for_the_disturbance():
  """The whole point: the MINIMIZING player sees a value spike it must avoid,
  and the maximizing player is paid nothing for it."""
  ctrl = _env(contact_arm_n=40.0, dstb_phase=False)
  dstb = _env(contact_arm_n=40.0, dstb_phase=True)
  l_ctrl = float(arm_stance_margins(ctrl)[1])
  l_dstb = float(arm_stance_margins(dstb)[1])
  assert l_dstb > l_ctrl
  assert l_dstb == pytest.approx(CLAMP)      # saturated past the threshold


def test_penalty_is_graded_below_the_threshold():
  """A lerp toward the ceiling, not a step: a disturbance that is winning (l
  deeply negative) still gets a gradient out of a grazing contact."""
  half = ARM_CONTACT_THRESHOLD / 2.0
  l0 = float(arm_stance_margins(_env(contact_arm_n=0.0, dstb_phase=True))[1])
  lh = float(arm_stance_margins(_env(contact_arm_n=half, dstb_phase=True))[1])
  lf = float(arm_stance_margins(_env(contact_arm_n=ARM_CONTACT_THRESHOLD,
                                     dstb_phase=True))[1])
  assert l0 < lh < lf
  assert lh == pytest.approx(l0 + 0.5 * (CLAMP - l0))


def test_penalty_is_monotone_and_stays_clamped():
  prev = -float("inf")
  for n in (0.0, 2.0, 5.0, 9.9, 10.0, 100.0, 5000.0):
    l = float(arm_stance_margins(_env(contact_arm_n=n, dstb_phase=True))[1])
    assert l >= prev - 1e-9
    assert -CLAMP <= l <= CLAMP
    prev = l


def test_g_is_never_shaped():
  """g stays the shared physical safety margin: the failure counters, the
  league's _ever_gneg bookkeeping and the certificate all read it."""
  for phase in (False, True):
    env = _env(contact_arm_n=5000.0, dstb_phase=phase)
    assert torch.equal(arm_stance_margins(env)[0], stance_margins(env)[0])
    assert torch.equal(arm_locomote_margins(env)[0], locomote_margins(env)[0])


def test_reach_avoid_end_criterion_disables_the_shaping():
  """Under end_criterion="reach-avoid" a shaped l would END the episode on the
  adversary's self-collision, which is nonsense — so it is skipped outright."""
  env = _env(contact_arm_n=5000.0, dstb_phase=True, end_criterion="reach-avoid")
  ref = _env(contact_arm_n=5000.0, dstb_phase=False)
  assert torch.equal(arm_stance_margins(env)[1], arm_stance_margins(ref)[1])


def test_margins_survive_a_cfg_without_the_arm_sensor():
  """A variant that drops the sensor stays trainable — the penalty simply never
  fires — rather than raising inside the reward hook."""
  env = _env(dstb_phase=True, with_arm_sensor=False)
  g, l = arm_stance_margins(env)
  assert torch.isfinite(g).all() and torch.isfinite(l).all()
  assert torch.equal(l, stance_margins(env)[1])


def test_locomote_variant_shapes_the_same_way():
  ctrl = _env(contact_arm_n=40.0, dstb_phase=False, cmd=(1.0, 0.0))
  dstb = _env(contact_arm_n=40.0, dstb_phase=True, cmd=(1.0, 0.0))
  assert float(arm_locomote_margins(dstb)[1]) > float(arm_locomote_margins(ctrl)[1])
  assert float(arm_locomote_margins(dstb)[1]) == pytest.approx(CLAMP)


def test_shaping_is_off_by_default_when_nothing_stamped_the_flag():
  """Any env built without DstbPhaseCallback — an eval env, a 1P run, a shadow
  rollout sim — must get the unshaped task, from the getattr default alone."""
  env = _Env()
  env.scene[ARM_SELFCOL_SENSOR] = _Sensor(
    force_history=torch.full((1, 8, 4, 3), 500.0))
  assert torch.equal(arm_stance_margins(env)[1], stance_margins(env)[1])


# --- the task registration + the bridge's joint mode --------------------------

def test_tasks_register_with_a_consistent_split():
  from robot_safety_sandbox.registry import algo_name, spec
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  for tid in ("g1_stabilize_arm_adv", "g1_locomote_arm_adv"):
    s = spec(tid)
    lo, hi = s.kwargs["dstb_joint_slice"]
    assert s.ctrl_dim == N_ACTUATED
    assert s.dstb_dim == ARM_DIM == hi - lo
    assert (lo, hi) == (ARM_SLICE.start, ARM_SLICE.stop)
    assert s.kwargs["dstb_mode"] == "joint"
    assert s.supports_adversary
    # The actor keeps g1_stabilize's full-width head, which is what lets a
    # trained g1_stabilize checkpoint warm-start whole.
    assert s.ctrl_dim == spec("g1_stabilize").ctrl_dim
    assert algo_name(tid, adversary=True) == "ReachAvoidPPO2P"


@pytest.mark.parametrize("bad, msg", [
  (None, "needs dstb_joint_slice"),
  ((15, 40), "not a sub-range"),
  ((15, 20), "must match the slice"),
])
def test_joint_mode_validates_its_slice(bad, msg):
  """Fail loud at construction rather than driving the wrong joints for a run."""
  from robot_safety_sandbox.base import _MjlabCore

  core = _MjlabCore.__new__(_MjlabCore)
  core.ctrl_dim, core.dstb_dim = 29, 14
  with pytest.raises(ValueError, match=msg):
    # Reproduces _init_core's validation block without building a simulator.
    _validate_slice(core, bad)


def _validate_slice(core, dstb_joint_slice):
  """The slice-validation half of _MjlabCore._init_core, called directly so the
  test needs no GPU. Kept in lockstep with base.py by
  test_joint_mode_slice_matches_base_implementation below."""
  if dstb_joint_slice is None:
    raise ValueError(
      "dstb_mode='joint' needs dstb_joint_slice=(start, stop): which action "
      "dims the adversary owns (e.g. (15, 29) for the G1's 14 arm joints).")
  lo, hi = (int(x) for x in dstb_joint_slice)
  if not 0 <= lo < hi <= core.ctrl_dim:
    raise ValueError(
      f"dstb_joint_slice=({lo}, {hi}) is not a sub-range of the env's "
      f"{core.ctrl_dim} action dims.")
  if hi - lo != core.dstb_dim:
    raise ValueError(
      f"dstb_joint_slice=({lo}, {hi}) spans {hi - lo} joints but dstb_dim="
      f"{core.dstb_dim}; the adversary's action width must match the slice "
      f"it drives.")


def test_joint_mode_slice_matches_base_implementation():
  """The stub above duplicates base.py's validation; if base.py's messages move,
  this fails rather than letting the stub drift into testing itself."""
  import inspect
  from robot_safety_sandbox.base import _MjlabCore
  src = inspect.getsource(_MjlabCore._init_core)
  for frag in ("needs dstb_joint_slice", "not a sub-range",
               "must match the slice"):
    assert frag in src, frag


# --- the per-env authority scale (PerEnvForceScaleCallback, name="dstb_scale") --

class _Logger:
  def __init__(self):
    self.rec = {}

  def record(self, k, v):
    self.rec[k] = v


def _scale_cb(name, n=4, **kw):
  from robot_safety_sandbox.callbacks import PerEnvForceScaleCallback
  bridge = type("B", (), {"num_envs": n, "device": "cpu"})()
  cb = PerEnvForceScaleCallback(name=name, **kw)
  # BaseCallback.logger is a read-only property forwarding to model.logger.
  cb.model = type("M", (), {"env": bridge, "logger": _Logger()})()
  cb._on_training_start()
  return cb, bridge


def _episode_end(cb, dones, timeouts):
  cb.locals = {"dones": torch.tensor(dones), "timeouts": torch.tensor(timeouts)}
  cb._on_step()


def test_per_env_scale_writes_the_joint_attribute_not_force_scale():
  """The arm adversary's knob is dstb_scale; writing force_scale instead would
  be read by nothing in joint mode — a curriculum that silently does nothing."""
  cb, bridge = _scale_cb("dstb_scale", init=0.5)
  assert torch.allclose(bridge.dstb_scale, torch.full((4,), 0.5))
  assert not hasattr(bridge, "force_scale")
  # the bridge holds the SAME tensor the callback updates in place
  _episode_end(cb, [True, False, False, False], [True, False, False, False])
  assert float(bridge.dstb_scale[0]) == pytest.approx(0.55)


def test_per_env_scale_weakens_where_the_robot_falls_and_strengthens_where_it_survives():
  cb, bridge = _scale_cb("dstb_scale", init=0.5)
  # env0 survives (timeout), env1 falls, env2/env3 still running
  _episode_end(cb, [True, True, False, False], [True, False, False, False])
  assert bridge.dstb_scale.tolist() == pytest.approx([0.55, 0.45, 0.5, 0.5])


def test_per_env_scale_is_clamped_and_reports_the_floor():
  cb, bridge = _scale_cb("dstb_scale", init=0.35, lo=0.3, hi=1.0)
  for _ in range(5):   # env0 keeps falling
    _episode_end(cb, [True, False, False, False], [False, False, False, False])
  assert float(bridge.dstb_scale[0]) == pytest.approx(0.3)
  assert cb.logger.rec["game/dstb_scale_at_floor"] == pytest.approx(0.25)
  assert "game/dstb_scale_mean" in cb.logger.rec
  for _ in range(40):  # env1 keeps surviving
    _episode_end(cb, [False, True, False, False], [False, True, False, False])
  assert float(bridge.dstb_scale[1]) == pytest.approx(1.0)


def test_wrench_default_is_unchanged():
  """The wrench tasks' behavior and log key must not move: default name still
  drives force_scale and logs game/force_scale_mean."""
  cb, bridge = _scale_cb("force_scale")
  from robot_safety_sandbox.callbacks import PerEnvForceScaleCallback
  assert PerEnvForceScaleCallback().name == "force_scale"
  _episode_end(cb, [True, False, False, False], [False, False, False, False])
  assert hasattr(bridge, "force_scale") and not hasattr(bridge, "dstb_scale")
  assert "game/force_scale_mean" in cb.logger.rec
