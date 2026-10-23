"""The open-loop clip replay (envs/g1_recover/playback.py PlaybackDriver).

Simulator-free (CPU), on a synthetic two-clip library whose action column
encodes its own row index, so every replayed target names the frame it came
from:

  * START -- env i replays rows[i]+1 .. rows[i]+steps[i]; the policy's action
    passes through everywhere else; steps = 0 arms nothing;
  * BOUNDS -- a replay that would run past its clip (next_ok < steps), a row
    out of range or negative steps is refused before any state changes;
  * ADVANCE -- one frame per env step, ``finished`` exactly once on the step
    the last frame was sent, then the policy again; ``cursor`` names the last
    frame sent;
  * STOP / RESTART -- stop hands back to the policy mid-replay without a
    ``finished``; a new start replaces a running replay.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover.playback import PlaybackDriver  # noqa: E402

A = 3   # action width


def _library(lengths=(6, 4)):
  """Clips laid out contiguously; action[row] = row in every column."""
  n = sum(lengths)
  frame, next_ok = [], []
  for L in lengths:
    frame += list(range(L))
    next_ok += [L - 1 - f for f in range(L)]
  return {"action": torch.arange(n, dtype=torch.float32)[:, None].expand(n, A).clone(),
          "next_ok": torch.tensor(next_ok), "frame": torch.tensor(frame)}


def _pol(n):
  return torch.full((n, A), -1.0)


def test_start_replays_the_following_frames_and_passes_the_policy_elsewhere():
  d = PlaybackDriver(_library(), 3, "cpu")
  mask = torch.tensor([True, True, False])
  d.start(mask, torch.tensor([0, 6, 0]), torch.tensor([3, 2, 5]))
  assert d.active.tolist() == [True, True, False]
  sent = []
  for _ in range(3):
    a = d.actions(_pol(3))
    sent.append(a[:, 0].tolist())
    d.advance()
  # env 0: rows 1, 2, 3; env 1 (second clip, row 6): rows 7, 8, then policy
  assert [s[0] for s in sent] == [1.0, 2.0, 3.0]
  assert [s[1] for s in sent] == [7.0, 8.0, -1.0]
  assert [s[2] for s in sent] == [-1.0, -1.0, -1.0]     # never armed
  assert not bool(d.active.any())


def test_advance_reports_finished_once_and_cursor_tracks_the_last_frame():
  d = PlaybackDriver(_library(), 2, "cpu")
  d.start(torch.tensor([True, True]), torch.tensor([1, 6]), torch.tensor([2, 1]))
  assert d.cursor.tolist() == [1, 6]
  assert d.target_rows().tolist() == [2, 7]
  d.actions()
  f1 = d.advance()
  assert f1.tolist() == [False, True]
  assert d.cursor.tolist() == [2, 7]
  d.actions()
  f2 = d.advance()
  assert f2.tolist() == [True, False]
  assert d.cursor.tolist() == [3, 7]
  f3 = d.advance()
  assert not bool(f3.any())
  assert d.target_rows().tolist() == [-1, -1]


def test_zero_steps_arms_nothing_and_none_policy_means_zeros():
  d = PlaybackDriver(_library(), 2, "cpu")
  d.start(torch.tensor([True, True]), 0, torch.tensor([0, 2]))
  assert d.active.tolist() == [False, True]
  a = d.actions(None)
  assert torch.equal(a[0], torch.zeros(A))
  assert torch.equal(a[1], torch.full((A,), 1.0))


def test_bounds_are_checked_against_the_clip():
  lib = _library((6, 4))
  d = PlaybackDriver(lib, 2, "cpu")
  m = torch.tensor([True, False])
  # row 3 of the first clip has 2 frames after it
  d.start(m, torch.tensor([3, 0]), torch.tensor([2, 99]))   # env 1 not masked: fine
  with pytest.raises(ValueError, match="leave its clip"):
    d.start(m, torch.tensor([3, 0]), torch.tensor([3, 0]))
  with pytest.raises(ValueError, match="leave its clip"):
    d.start(m, torch.tensor([9, 0]), torch.tensor([1, 0]))   # last frame of clip 2
  with pytest.raises(IndexError):
    d.start(m, torch.tensor([10, 0]), torch.tensor([1, 0]))
  with pytest.raises(IndexError):
    d.start(m, torch.tensor([-1, 0]), torch.tensor([1, 0]))
  with pytest.raises(ValueError, match=">= 0"):
    d.start(m, torch.tensor([0, 0]), torch.tensor([-1, 0]))
  # the refused calls left the first replay untouched
  assert d.row.tolist() == [3, -1] and d.steps.tolist() == [2, 0]
  with pytest.raises(ValueError, match="next_ok"):
    PlaybackDriver({"action": lib["action"]}, 2, "cpu")


def test_stop_and_restart():
  d = PlaybackDriver(_library(), 2, "cpu")
  d.start(torch.tensor([True, True]), torch.tensor([0, 0]), torch.tensor([4, 4]))
  d.actions()
  d.advance()
  d.stop(torch.tensor([True, False]))
  assert d.active.tolist() == [False, True]
  a = d.actions(_pol(2))
  assert a[0, 0] == -1.0 and a[1, 0] == 2.0
  f = d.advance()
  assert not bool(f[0])                        # a stop is not a finish
  # restart env 1 mid-replay on the second clip
  d.start(torch.tensor([False, True]), torch.tensor([0, 6]), torch.tensor([0, 3]))
  assert d.k.tolist()[1] == 0 and d.active.tolist() == [False, True]
  assert d.actions(_pol(2))[1, 0] == 7.0
