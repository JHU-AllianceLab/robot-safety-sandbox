"""Unit tests for the G1 reach-avoid margins (envs/g1_stabilize/env_cfg.py).

Simulator-free: a stub stands in for the mjlab env, so these run on CPU and pin
the margin CONTRACT rather than any physics. What they pin:

  * the sign convention — g < 0 exactly on the failure set the env terminates on
    (pelvis too low, torso past the tilt limit, OR any non-foot ground contact);
  * that both margins are clamped to +-CLAMP, which is what keeps a several-
    hundred-newton body slam from handing the value regression a -150 reward;
  * that the contact term degrades gracefully when the sensor is absent, and
    reads the instantaneous force under the shadow-sim flag;
  * the l scales: 1.0 at perfect command tracking, positive in the spawn stance.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (  # noqa: E402
  CLAMP, LOCO_VEL_TOL, locomote_margins, stance_margins)


def _cos(deg: float) -> float:
  return math.cos(math.radians(deg))


class _Data:
  def __init__(self, pos, grav, lin, ang):
    self.root_link_pos_w = pos
    self.projected_gravity_b = grav
    self.root_link_lin_vel_b = lin
    self.root_link_ang_vel_b = ang


class _Sensor:
  def __init__(self, force_history=None, force=None):
    self.data = type("D", (), {"force_history": force_history, "force": force})()


class _Commands:
  def __init__(self, cmd):
    self._cmd = cmd

  def get_command(self, name):
    assert name == "twist"
    return self._cmd


class _Env:
  """The slice of the mjlab env the margins actually read."""

  def __init__(self, *, pelvis_z=0.76, tilt_cos=1.0, lin_vel=(0.0, 0.0, 0.0),
               ang_vel=(0.0, 0.0, 0.0), contact_n=0.0, cmd=(0.0, 0.0),
               with_sensor=True, instantaneous=False):
    n = 1
    pos = torch.tensor([[0.0, 0.0, pelvis_z]])
    # projected_gravity_b[:, 2] = -cos(tilt); the margins read -that as "up".
    grav = torch.tensor([[0.0, 0.0, -float(tilt_cos)]])
    self.scene = {
      "robot": type("R", (), {"data": _Data(pos, grav,
                                            torch.tensor([list(lin_vel)]),
                                            torch.tensor([list(ang_vel)]))})(),
    }
    if with_sensor:
      # [B, N_slots, H, 3] — the shape a history_length=4 contact sensor emits.
      hist = torch.zeros(n, 2, 4, 3)
      hist[0, 1, 2, 0] = float(contact_n)   # one substep, one slot, x-axis
      inst = torch.zeros(n, 2, 3)
      inst[0, 1, 0] = float(contact_n)
      self.scene["nonfoot_ground_touch"] = _Sensor(
        force_history=None if instantaneous else hist, force=inst)
    self._zoo_instantaneous_contact = instantaneous
    self.command_manager = _Commands(torch.tensor([list(cmd) + [0.0]]))


def _g(env):
  return float(stance_margins(env)[0])


def _l_stance(env):
  return float(stance_margins(env)[1])


# --- g: the failure set -------------------------------------------------------

def test_spawn_stance_is_safe_and_in_the_target_set():
  env = _Env()                       # the knees-bent keyframe, at rest
  g, l = stance_margins(env)
  assert float(g) > 0.0
  assert float(l) > 0.0              # as go2_stabilize does, G1 spawns in target


def test_low_pelvis_is_a_failure():
  assert _g(_Env(pelvis_z=0.30)) < 0.0
  assert _g(_Env(pelvis_z=0.50)) > 0.0


def test_tilt_past_the_limit_is_a_failure():
  # 50 deg is the limit; cos(60 deg) is past it, cos(30 deg) is not.
  assert _g(_Env(tilt_cos=_cos(60.0))) < 0.0
  assert _g(_Env(tilt_cos=_cos(30.0))) > 0.0


def test_nonfoot_ground_contact_is_a_failure_even_while_upright_and_high():
  """The term that catches a kneel or a hand plant: a G1's pelvis stays high in
  poses that are already a fall, so height and tilt alone do not decide it."""
  upright_and_high = dict(pelvis_z=0.76, tilt_cos=1.0)
  assert _g(_Env(contact_n=0.0, **upright_and_high)) > 0.0
  assert _g(_Env(contact_n=40.0, **upright_and_high)) < 0.0


def test_margins_are_clamped():
  hard_slam = _Env(contact_n=5000.0)
  assert _g(hard_slam) == -CLAMP
  assert _l_stance(_Env(lin_vel=(50.0, 0.0, 0.0))) == -CLAMP
  g, l = locomote_margins(_Env(lin_vel=(-99.0, 0.0, 0.0), cmd=(1.0, 0.0)))
  assert float(l) == -CLAMP


# --- the contact term's fallbacks ---------------------------------------------

def test_margins_survive_a_cfg_without_the_contact_sensor():
  """A variant that drops nonfoot_ground_touch stays trainable on the other two
  terms rather than raising inside the reward hook."""
  env = _Env(with_sensor=False)
  g, l = stance_margins(env)
  assert float(g) > 0.0 and torch.isfinite(g).all()
  assert _g(_Env(with_sensor=False, pelvis_z=0.30)) < 0.0


def test_instantaneous_flag_reads_the_live_force():
  """Rollout shadow sims that cannot round-trip the force history set
  _zoo_instantaneous_contact; a half-filled window would under-report the peak."""
  assert _g(_Env(contact_n=40.0, instantaneous=True)) < 0.0
  assert _g(_Env(contact_n=0.0, instantaneous=True)) > 0.0


# --- l: the two target sets ---------------------------------------------------

def test_locomote_target_is_one_at_perfect_tracking():
  env = _Env(lin_vel=(1.0, 0.0, 0.0), cmd=(1.0, 0.0))
  assert abs(float(locomote_margins(env)[1]) - 1.0) < 1e-6


def test_locomote_target_is_zero_at_the_tolerance_edge():
  env = _Env(lin_vel=(1.0 - LOCO_VEL_TOL, 0.0, 0.0), cmd=(1.0, 0.0))
  assert abs(float(locomote_margins(env)[1])) < 1e-6


def test_locomote_target_is_negative_when_standing_under_a_walk_command():
  env = _Env(lin_vel=(0.0, 0.0, 0.0), cmd=(1.0, 0.0))
  assert float(locomote_margins(env)[1]) < 0.0


def test_stance_target_rejects_a_moving_robot():
  assert _l_stance(_Env(lin_vel=(1.0, 0.0, 0.0))) < 0.0
  assert _l_stance(_Env(ang_vel=(0.0, 0.0, 2.0))) < 0.0
  assert _l_stance(_Env(tilt_cos=_cos(40.0))) < 0.0   # past the 25 deg band
  assert _l_stance(_Env(pelvis_z=0.50)) < 0.0             # under the band floor


# --- the declarations -----------------------------------------------------------

def test_g1_tasks_declare_their_avoid_and_reach_margins():
  """Every G1 stabilize / recover task spells its margins out: reach-avoid tasks
  declare avoid g_fall plus a reach margin, avoid-only tasks g_fall alone."""
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall, l_stance, l_track
  from robot_safety_sandbox.registry import AVOID, REACH_AVOID, spec
  expect = {
    "g1_stabilize": (REACH_AVOID, l_stance), "g1_locomote": (REACH_AVOID, l_track),
    "g1_stabilize_arm_random": (REACH_AVOID, l_stance),
    "g1_locomote_arm_random": (REACH_AVOID, l_track),
    "g1_stabilize_arm_random_avoid": (AVOID, None),
    "g1_recover": (AVOID, None), "g1_recover_wrench": (AVOID, None),
    "g1_recover_push": (AVOID, None), "g1_recover_push_fast": (AVOID, None),
    "g1_recover_arm_random": (AVOID, None), "g1_recover_arm_random_wrench": (AVOID, None),
  }
  for tid, (mode, reach) in expect.items():
    s = spec(tid)
    assert s.mode == mode, tid
    assert s.margin_fn.avoid_margin is g_fall, tid
    assert s.margin_fn.reach_margin is reach, tid


def test_composed_pairs_match_the_separate_margins():
  """stance_margins / locomote_margins are compose(g_fall, l_*): the clamped
  pair, value for value."""
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall, l_stance, l_track
  for env in (_Env(pelvis_z=0.30, tilt_cos=_cos(60.0), lin_vel=(2.0, 0.1, 0.0),
                  ang_vel=(0.0, 0.5, 0.0), contact_n=300.0),
              _Env(pelvis_z=0.76, tilt_cos=_cos(10.0), lin_vel=(0.1, 0.0, 0.0))):
    g, l = stance_margins(env)
    assert torch.allclose(g, g_fall(env).clamp(-CLAMP, CLAMP))
    assert torch.allclose(l, l_stance(env).clamp(-CLAMP, CLAMP))
  lenv = _Env(lin_vel=(0.6, 0.0, 0.0), cmd=(1.0, 0.0))
  g2, l2 = locomote_margins(lenv)
  assert torch.allclose(g2, g_fall(lenv).clamp(-CLAMP, CLAMP))
  assert torch.allclose(l2, l_track(lenv).clamp(-CLAMP, CLAMP))


# --- the no-pelvis-height ablation (tasks g1_recover*_noheight) ----------------

def test_noheight_margin_ignores_the_pelvis_height():
  """g_fall_noheight is g_fall without its height term: a low but upright,
  contact-free pelvis stays safe; tilt and non-foot contact still fail; where
  the height term does not bind the two margins agree."""
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall, g_fall_noheight
  low = _Env(pelvis_z=0.30)
  assert float(g_fall(low)) < 0.0 < float(g_fall_noheight(low))
  assert float(g_fall_noheight(_Env(tilt_cos=_cos(60.0)))) < 0.0
  assert float(g_fall_noheight(_Env(contact_n=50.0))) < 0.0
  for env in (_Env(tilt_cos=_cos(30.0)), _Env(contact_n=5.0), _Env(pelvis_z=0.9)):
    assert torch.allclose(g_fall(env), g_fall_noheight(env))
  assert float(g_fall_noheight(_Env(pelvis_z=-5.0))) == float(g_fall_noheight(_Env()))
  # the hook cpu_rollout.build_spec reads to drop the CPU margin's height term
  assert g_fall.fall_height_term is True and g_fall_noheight.fall_height_term is False


def test_noheight_tasks_register_the_ablated_margin_and_critic():
  """g1_recover_noheight / g1_recover_lafan_noheight: avoid-only on
  g_fall_noheight, otherwise their parents' registration; their cfg has no
  pelvis_height in the critic group (and refuses one), the parent's still has."""
  import pytest
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall_noheight
  from robot_safety_sandbox.registry import AVOID, spec
  for tid, parent in (("g1_recover_noheight", "g1_recover"),
                      ("g1_recover_lafan_noheight", "g1_recover_lafan")):
    s, p = spec(tid), spec(parent)
    assert s.mode == AVOID and s.margin_fn.avoid_margin is g_fall_noheight, tid
    assert s.margin_fn.reach_margin is None, tid
    assert (s.ctrl_dim, s.dstb_dim, s.kwargs) == (p.ctrl_dim, p.dstb_dim, p.kwargs), tid
  ablated = spec("g1_recover_noheight").cfg_builder()
  full = spec("g1_recover").cfg_builder()
  a_terms = list(ablated.observations["critic"].terms)
  f_terms = list(full.observations["critic"].terms)
  assert "pelvis_height" in f_terms and "pelvis_height" not in a_terms
  assert [t for t in f_terms if t != "pelvis_height"] == a_terms
  assert list(ablated.observations["actor"].terms) == list(full.observations["actor"].terms)
  assert set(ablated.terminations) == set(full.terminations)
  with pytest.raises(ValueError):
    spec("g1_recover_noheight").cfg_builder(critic_pelvis_height=True)


def test_nofoot_tasks_keep_only_the_actor_terms_and_base_lin_vel():
  """g1_recover_noheight_nofoot / g1_recover_lafan_noheight_nofoot: the
  *_noheight registration (margin, mode, kwargs), a critic group of the actor
  terms + base_lin_vel (the student world model's state), and the actor obs and
  terminations of the parent; the builder refuses the privileged terms back."""
  import pytest
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from robot_safety_sandbox.envs.g1_recover.env_cfg import CRITIC_FOOT_TERMS
  from robot_safety_sandbox.registry import spec
  for tid, parent in (("g1_recover_noheight_nofoot", "g1_recover_noheight"),
                      ("g1_recover_lafan_noheight_nofoot", "g1_recover_lafan_noheight")):
    s, p = spec(tid), spec(parent)
    assert s.mode == p.mode and s.margin_fn.avoid_margin is p.margin_fn.avoid_margin, tid
    assert (s.ctrl_dim, s.dstb_dim, s.kwargs) == (p.ctrl_dim, p.dstb_dim, p.kwargs), tid
  student = spec("g1_recover_noheight_nofoot").cfg_builder()
  teacher = spec("g1_recover_noheight").cfg_builder()
  actor = list(student.observations["actor"].terms)
  assert actor == list(teacher.observations["actor"].terms)
  assert list(student.observations["critic"].terms) == actor + ["base_lin_vel"]
  assert [t for t in teacher.observations["critic"].terms if t not in CRITIC_FOOT_TERMS] \
      == actor + ["base_lin_vel"]
  assert set(student.terminations) == set(teacher.terminations)
  for k in ("critic_pelvis_height", "critic_foot_terms"):
    with pytest.raises(ValueError):
      spec("g1_recover_noheight_nofoot").cfg_builder(**{k: True})


def test_nolinvel_tasks_keep_exactly_the_actor_terms():
  """g1_recover_noheight_nofoot_nolinvel / its lafan twin: the *_noheight
  registration, a critic group of exactly the actor terms (the 98-D student
  world model's state, which world_model_spec maps one to one), the parent's
  actor obs and terminations; the builder refuses every privileged term back."""
  import pytest
  import robot_safety_sandbox  # noqa: F401  (registers on import)
  from robot_safety_sandbox.registry import spec
  for tid, parent in (("g1_recover_noheight_nofoot_nolinvel", "g1_recover_noheight"),
                      ("g1_recover_lafan_noheight_nofoot_nolinvel", "g1_recover_lafan_noheight")):
    s, p = spec(tid), spec(parent)
    assert s.mode == p.mode and s.margin_fn.avoid_margin is p.margin_fn.avoid_margin, tid
    assert (s.ctrl_dim, s.dstb_dim, s.kwargs) == (p.ctrl_dim, p.dstb_dim, p.kwargs), tid
  student = spec("g1_recover_noheight_nofoot_nolinvel").cfg_builder()
  teacher = spec("g1_recover_noheight").cfg_builder()
  actor = list(student.observations["actor"].terms)
  assert actor == list(teacher.observations["actor"].terms)
  assert list(student.observations["critic"].terms) == actor
  # noise-free: the critic group never corrupts (mjlab drops its terms' noise)
  assert not student.observations["critic"].enable_corruption
  assert set(student.terminations) == set(teacher.terminations)
  for k in ("critic_pelvis_height", "critic_foot_terms", "critic_base_lin_vel"):
    with pytest.raises(ValueError):
      spec("g1_recover_noheight_nofoot_nolinvel").cfg_builder(**{k: True})


def test_cpu_rollout_margin_drops_the_height_term_at_z_min_minus_inf():
  """The CPU rollout's g_fall (cpu_rollout.Roller.margin) with the spec's
  z_min = -inf, what build_spec writes for a g_fall_noheight task, equals
  g_fall_noheight; with z_min = Z_MIN it equals g_fall (a low pelvis fails)."""
  import numpy as np
  from types import SimpleNamespace
  sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples"))
  import cpu_rollout as cr
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
    CONTACT_FORCE_THRESHOLD, TILT_COS, Z_MIN, Z_SCALE, g_fall, g_fall_noheight)

  def cpu_g(z_min, pelvis_z, tilt_deg, force):
    r = object.__new__(cr.Roller)
    r.s = SimpleNamespace(root_qadr=0, z_min=z_min, z_scale=Z_SCALE, tilt_cos=TILT_COS,
                          force_threshold=CONTACT_FORCE_THRESHOLD, clamp=CLAMP)
    r.tilt_norm = 1.0 - TILT_COS
    h = math.radians(tilt_deg) / 2.0                  # pitch about y, w-first quaternion
    d = SimpleNamespace(qpos=np.array([0.0, 0.0, pelvis_z, math.cos(h), 0.0, math.sin(h), 0.0]))
    return r.margin(d, force)

  for z, tilt, f in ((0.30, 0.0, 0.0), (0.60, 20.0, 0.0), (0.76, 55.0, 0.0), (0.50, 0.0, 30.0)):
    env = _Env(pelvis_z=z, tilt_cos=_cos(tilt), contact_n=f)
    want_no = float(g_fall_noheight(env).clamp(-CLAMP, CLAMP))
    want_full = float(g_fall(env).clamp(-CLAMP, CLAMP))
    assert abs(cpu_g(-math.inf, z, tilt, f) - want_no) < 1e-5, (z, tilt, f)
    assert abs(cpu_g(Z_MIN, z, tilt, f) - want_full) < 1e-5, (z, tilt, f)
