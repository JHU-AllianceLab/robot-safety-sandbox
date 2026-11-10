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
