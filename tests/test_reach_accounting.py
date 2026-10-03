"""The bridge's always-on reach accounting (base._MjlabCore._count_reach):
safety/reach_rate and safety/time_to_reach_s.

Simulator-free: a bare core fed scripted (terminated, truncated, g, l) steps.
What it pins:
  * a reach counts only if g stayed >= 0 up to it -- a g < 0 the env did not
    terminate on still disqualifies a later reach;
  * a fall AFTER the reach does not undo it (reach-avoid asks nothing after);
  * episodes are counted when they END, so a running episode waits for a later
    window, and an ended env starts its next episode with a clean slate;
  * time to reach is steps x step_dt, averaged over the episodes that reached;
  * tasks without a real target emit no reach keys at all.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.base import TERMINAL_MARGIN, _MjlabCore  # noqa: E402

DT = 0.02


def _core(track=True):
  core = _MjlabCore.__new__(_MjlabCore)
  core._failures_total, core._track_reach, core._reach_ep = 0, track, None
  core._zero_counts()
  core.mj = SimpleNamespace(step_dt=DT)
  return core


def _step(core, g, l, term=(), trunc=()):
  n = len(g)
  t = torch.zeros(n, dtype=torch.bool)
  tr = torch.zeros(n, dtype=torch.bool)
  t[list(term)] = True
  tr[list(trunc)] = True
  core._count_step(t, tr, torch.tensor(g, dtype=torch.float),
                   torch.tensor(l, dtype=torch.float))


F = TERMINAL_MARGIN          # the anchored g of a failure termination


def test_reach_rate_and_time_follow_the_reach_avoid_outcome():
  core = _core()
  # env 0: reaches at step 2, times out at step 4                -> reached, 2
  # env 1: g < 0 at step 1 (not terminated), l >= 0 at step 2    -> NOT reached
  # env 2: reaches at step 1, falls at step 3                    -> reached, 1
  # env 3: reaches at step 1, still running at the drain         -> not counted
  _step(core, g=[1, -0.2, 1, 1], l=[-1, -1, 0.5, 0.5])
  _step(core, g=[1, 1, 1, 1], l=[0.3, 0.3, -1, 0.2])
  _step(core, g=[1, 1, F, 1], l=[-1, -1, -1, 0.2], term=[2])
  _step(core, g=[1, 1, 1, 1], l=[-1, -1, -1, 0.2], trunc=[0, 1])
  out = core._drain_counts()
  assert out["safety/reach_rate"] == pytest.approx(2 / 3)
  assert out["safety/time_to_reach_s"] == pytest.approx((2 + 1) / 2 * DT)
  assert out["safety/episodes"] == 3.0

  # next window: env 3 ends, reached back at its step 1; env 0 restarted at
  # step 5 and reaches at its first step, then times out -> both count
  _step(core, g=[1, 1, 1, 1], l=[0.4, -1, -1, 0.2], trunc=[0, 3])
  out = core._drain_counts()
  assert out["safety/reach_rate"] == pytest.approx(1.0)
  assert out["safety/time_to_reach_s"] == pytest.approx((1 + 1) / 2 * DT)


def test_a_fall_is_never_a_reach_even_inside_the_target():
  core = _core()
  _step(core, g=[F], l=[0.9], term=[0])
  out = core._drain_counts()
  assert out["safety/reach_rate"] == 0.0
  assert "safety/time_to_reach_s" not in out


def test_no_reach_keys_without_a_real_target_or_before_any_episode_ends():
  core = _core(track=False)
  _step(core, g=[1], l=[0.0], trunc=[0])
  assert not any(k.startswith("safety/reach") or "time_to_reach" in k
                 for k in core._drain_counts())
  core = _core()
  _step(core, g=[1], l=[0.5])                # reached, but still running
  assert "safety/reach_rate" not in core._drain_counts()
