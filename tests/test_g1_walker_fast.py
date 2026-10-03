"""Unit tests for the fast G1 walker's command box (envs/velocity/g1.py).

Simulator-free: the cfg builders only assemble dataclasses, so these run on CPU
and pin the COMMAND contract of ``g1_walker_flat_fast``:

  * vx in (-0.5, 1.5) and vy in (-0.5, 0.5) from the first reset — i.e. the
    ``command_vel`` curriculum is gone, so no stage can rewrite the box later
    (and the first reset, at common_step_counter == 0, samples it too);
  * yaw rate is untouched: (-1, 1) in training, the play cfg's (-0.7, 0.7);
  * nothing else moves: the fast cfg equals the stock flat cfg outside the
    twist ranges and the curriculum.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

pytest.importorskip("mjlab")

from robot_safety_sandbox.envs.velocity.g1 import (  # noqa: E402
  unitree_g1_flat_env_cfg, unitree_g1_flat_fast_env_cfg)


def _addr_free(s: str) -> str:
  return re.sub(r" at 0x[0-9a-f]+", "", s)


@pytest.mark.parametrize("play,yaw", [(False, (-1.0, 1.0)), (True, (-0.7, 0.7))])
def test_fast_box_is_fixed_from_the_first_reset(play, yaw):
  cfg = unitree_g1_flat_fast_env_cfg(play=play)
  r = cfg.commands["twist"].ranges
  assert r.lin_vel_x == (-0.5, 1.5)
  assert r.lin_vel_y == (-0.5, 0.5)
  assert r.ang_vel_z == yaw
  assert "command_vel" not in cfg.curriculum


def test_fast_cfg_changes_only_the_command_box():
  fast = unitree_g1_flat_fast_env_cfg(play=False)
  stock = unitree_g1_flat_env_cfg(play=False)
  assert "command_vel" in stock.curriculum
  stock.curriculum.pop("command_vel")
  stock.commands["twist"].ranges.lin_vel_x = (-0.5, 1.5)
  stock.commands["twist"].ranges.lin_vel_y = (-0.5, 0.5)
  # repr, not ==: the robot's spec_fn is a fresh lambda per build, so compare
  # the printed cfg with its memory addresses masked.
  assert _addr_free(repr(fast)) == _addr_free(repr(stock))


def test_fast_walker_registers_as_a_cumulative_g1_task():
  from robot_safety_sandbox.registry import CUMULATIVE, spec

  fast, stock = spec("g1_walker_flat_fast"), spec("g1_walker_flat")
  assert fast.mode == CUMULATIVE
  assert fast.cfg_builder is unitree_g1_flat_fast_env_cfg
  assert (fast.ctrl_dim, fast.kwargs) == (stock.ctrl_dim, stock.kwargs)
