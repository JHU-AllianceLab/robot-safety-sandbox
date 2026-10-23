"""Open-loop replay of a mocap clip as the robot's PD targets.

The P-pre-nominal start states (``examples/harvest_g1_handover_states.py``)
are the states a safety filter would actually hand over in: a NOMINAL
controller is running a motion, a push lands, and some time later the filter
switches to the fallback. With no tracking controller in the repo, the
nominal here is the clip itself: an env spawned on library frame ``r`` (the
spawn restores the frame's pose, twist and last action) is driven by the PD
targets of frames ``r+1, r+2, ...``, one per control step (the library is
stored at the 50 fps control rate, ``examples/build_g1_recover_mocap_states.py
--library``). The bank's ``action`` column is exactly that target, ``(q -
offset) / scale`` clamped to the bridge's action bound, so the replay is a
kinematic reference under the env's own PD gains and dynamics: it reproduces
slow motions closely and DRIFTS on fast ones (nothing closes the loop on the
base), which is the point -- it is the handover a real tracker would also be
in trouble at. A replay that falls on its own before the handover is counted
and discarded by the harvester, not labeled.

:class:`PlaybackDriver` is per-env bookkeeping only (pure torch, no simulator;
``tests/test_playback.py``): :meth:`start` arms a replay of ``steps[i]``
frames after ``rows[i]``, :meth:`actions` substitutes the replayed target for
the policy's action on the envs that are replaying, and :meth:`advance` moves
every replay one frame on after the env step. A replay never leaves its clip:
``start`` asserts ``next_ok[rows] >= steps`` (``next_ok`` = frames after the
row within its clip, a library column).
"""

from __future__ import annotations

from typing import Optional, Union

import torch


class PlaybackDriver:
  """Per-env open-loop replay of consecutive bank frames.

  :param bank: a clip library (``load_bank``): needs ``action`` (the PD target
      of each frame) and ``next_ok`` (frames after the row within its clip).
  :param num_envs: envs.
  :param device: torch device the bank columns are moved to.

  Per env: ``row`` (the frame the replay started on, -1 = none), ``k``
  (frames replayed so far), ``steps`` (frames to replay), ``active``.
  Call order each control step::

      act = driver.actions(policy_act)     # replayed targets where active
      obs, ... = env.step_tensor(act)
      finished = driver.advance()          # True where the replay just ended
  """

  def __init__(self, bank: dict, num_envs: int, device):
    if "next_ok" not in bank:
      raise ValueError("PlaybackDriver needs a clip library with a next_ok "
                       "column (build_g1_recover_mocap_states.py --library)")
    self.device = device
    self.n = int(num_envs)
    self.action = bank["action"].to(device)
    self.next_ok = bank["next_ok"].to(device).long()
    self.rows_total = int(self.action.shape[0])
    long = dict(dtype=torch.long, device=device)
    self.row = torch.full((self.n,), -1, **long)
    self.k = torch.zeros(self.n, **long)
    self.steps = torch.zeros(self.n, **long)
    self.active = torch.zeros(self.n, dtype=torch.bool, device=device)

  def _per_env(self, x: Union[int, torch.Tensor]) -> torch.Tensor:
    t = torch.as_tensor(x, device=self.device).long()
    return t.expand(self.n) if t.dim() == 0 else t.reshape(-1)

  def start(self, mask: torch.Tensor, rows: Union[int, torch.Tensor],
            steps: Union[int, torch.Tensor]) -> None:
    """Env ``i`` in ``mask`` replays frames ``rows[i]+1 .. rows[i]+steps[i]``
    (``rows`` / ``steps`` are per-env ``[num_envs]`` tensors or scalars).
    ``steps[i] = 0`` arms nothing. Replaces any replay the env was running."""
    m = mask.reshape(-1).bool().to(self.device)
    r = self._per_env(rows)
    s = self._per_env(steps)
    if bool(m.any()):
      rm, sm = r[m], s[m]
      if bool((rm < 0).any()) or bool((rm >= self.rows_total).any()):
        raise IndexError(f"playback rows out of range [0, {self.rows_total})")
      if bool((sm < 0).any()):
        raise ValueError("playback steps must be >= 0")
      short = self.next_ok[rm] < sm
      if bool(short.any()):
        i = int(short.nonzero()[0])
        raise ValueError(
          f"playback would leave its clip: row {int(rm[i])} has "
          f"{int(self.next_ok[rm[i]])} frames after it, {int(sm[i])} asked")
    self.row = torch.where(m, r, self.row)
    self.steps = torch.where(m, s, self.steps)
    self.k = torch.where(m, torch.zeros_like(self.k), self.k)
    self.active = torch.where(m, s > 0, self.active)

  def stop(self, mask: torch.Tensor) -> None:
    """End the replay of the envs in ``mask`` now (``k`` keeps its value)."""
    self.active &= ~mask.reshape(-1).bool().to(self.device)

  @property
  def cursor(self) -> torch.Tensor:
    """The bank row of the last frame each env was sent (``row + k``; the
    start row before the first step), -1 where no replay was started."""
    return torch.where(self.row >= 0, self.row + self.k, self.row)

  def target_rows(self) -> torch.Tensor:
    """The bank row whose PD target each ACTIVE env receives this step
    (``row + 1 + k``), -1 elsewhere."""
    return torch.where(self.active, self.row + 1 + self.k,
                       torch.full_like(self.row, -1))

  def actions(self, policy_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The action tensor to send: active envs get ``action[row + 1 + k]``,
    the others ``policy_actions`` (zeros when None)."""
    idx = (self.row + 1 + self.k).clamp(0, self.rows_total - 1)
    replay = self.action[idx]
    if policy_actions is None:
      policy_actions = torch.zeros_like(replay)
    return torch.where(self.active.unsqueeze(1), replay,
                       policy_actions.to(replay.dtype))

  def advance(self) -> torch.Tensor:
    """One env step has run: move every active replay one frame on. Returns
    the envs whose replay ENDED with this step (``k`` reached ``steps``)."""
    was = self.active
    self.k = torch.where(was, self.k + 1, self.k)
    self.active = was & (self.k < self.steps)
    return was & ~self.active
