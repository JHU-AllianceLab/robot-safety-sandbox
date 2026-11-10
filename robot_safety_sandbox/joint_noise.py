"""Smooth random joint commands — a SCRIPTED joint-partition disturbance.

The random-policy twin of a learned joint-mode adversary (``dstb_mode="joint"``,
e.g. ``g1_stabilize_arm_adv``): instead of a second player choosing the
commands on a slice of the robot's own actuators, :class:`SmoothJointNoise`
draws them at random, and the bridge writes them over that slice exactly the
way it writes an adversary's (see ``base._MjlabCore._write_joint_slice``). The
run stays single-player; only the source of the slice's commands changes.

THE SIGNAL. Each joint of each env follows its own chain of MINIMUM-JERK
segments between random waypoints:

    waypoint     w_k ~ U(lo, hi)                   (normalized action units)
    duration     T_k ~ U(seg_min_s, seg_max_s)     (seconds, per segment)
    command      w_k + (w_{k+1} - w_k) * s(t / T_k),
                 s(u) = 10u^3 - 15u^4 + 6u^5

``s`` has zero first AND second derivative at both ends, so the command is C2:
no step, no velocity kink, no acceleration jump at a waypoint — the joint glides
from rest to rest the way a reaching arm does, rather than jittering the way an
Ornstein-Uhlenbeck process does at a 50 Hz control rate. Joints and envs are
independent, so the arms move uncoordinated and the segment boundaries never
line up.

Every episode starts from ``0`` = the default pose (mjlab's JointPositionAction
target is ``default + scale * action``), which is where the reset event puts
the arms, so an episode never opens with a jump.

``lo`` / ``hi`` bound the waypoints PER JOINT, inside ``[-1, 1]`` (default: the
whole of it). A one-sided range keeps a joint on one side of the default pose —
the G1 task uses it to stop the shoulders rolling the arms into the torso (see
``tasks/g1_arm_random.py``).

MAGNITUDE is not this module's business: the output lives in ``[-1, 1]`` and the
bridge multiplies it by ``dstb_gain`` (and the per-env ``dstb_scale``), which is
what ``DstbGainRampCallback`` / ``PerEnvForceScaleCallback`` ramp. Because the
segment durations do not scale with it, raising the magnitude raises the arm
SPEED in proportion — the curriculum hardens both at once.
"""

from __future__ import annotations

import torch


def _min_jerk(u: torch.Tensor) -> torch.Tensor:
  """The quintic minimum-jerk profile on ``u`` in [0, 1]: s(0)=0, s(1)=1, with
  s' and s'' zero at both ends."""
  return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


class SmoothJointNoise:
  """Batched smooth random commands, ``(num_envs, dim)`` in ``[-1, 1]``.

  Call :meth:`step` once per control step (it returns this step's command and
  advances the clock by ``dt``), and :meth:`reset` on the envs whose episode
  just ended. Sync-free: resets are masks, not index lists, so nothing here
  waits on the GPU.
  """

  def __init__(self, num_envs: int, dim: int, dt: float, device, *,
               seg_min_s: float = 0.5, seg_max_s: float = 2.0,
               lo=-1.0, hi=1.0):
    if not 0.0 < seg_min_s <= seg_max_s:
      raise ValueError(
        f"SmoothJointNoise needs 0 < seg_min_s <= seg_max_s, got "
        f"({seg_min_s}, {seg_max_s}).")
    if not dt > 0.0:
      raise ValueError(f"SmoothJointNoise needs dt > 0, got {dt}.")
    self.num_envs, self.dim = int(num_envs), int(dim)
    self.dt = float(dt)
    self.device = device
    self.seg_min_s, self.seg_max_s = float(seg_min_s), float(seg_max_s)
    # Per-joint waypoint bounds: a scalar applies to every joint.
    self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device).expand(
      self.dim).clone()
    self.hi = torch.as_tensor(hi, dtype=torch.float32, device=device).expand(
      self.dim).clone()
    if not bool(((-1.0 <= self.lo) & (self.lo <= self.hi)
                 & (self.hi <= 1.0)).all()):
      raise ValueError(
        f"SmoothJointNoise needs -1 <= lo <= hi <= 1 per joint, got "
        f"lo={self.lo.tolist()}, hi={self.hi.tolist()}.")
    shape = (self.num_envs, self.dim)
    self._p0 = torch.zeros(shape, device=device)   # segment start waypoint
    self._p1 = torch.zeros(shape, device=device)   # segment end waypoint
    self._t = torch.zeros(shape, device=device)    # time into the segment
    self._T = torch.ones(shape, device=device)     # segment duration
    self.reset()

  def _waypoints(self) -> torch.Tensor:
    return self.lo + (self.hi - self.lo) * torch.rand(
      self.num_envs, self.dim, device=self.device)

  def _durations(self) -> torch.Tensor:
    return self.seg_min_s + (self.seg_max_s - self.seg_min_s) * torch.rand(
      self.num_envs, self.dim, device=self.device)

  def reset(self, mask: torch.Tensor | None = None) -> None:
    """Restart the envs in ``mask`` ((num_envs,) bool; None = all) from the
    default pose, heading for a fresh random waypoint."""
    if mask is None:
      self._p0.zero_()
      self._p1.copy_(self._waypoints())
      self._t.zero_()
      self._T.copy_(self._durations())
      return
    m = mask.reshape(-1, 1).bool()
    self._p0.masked_fill_(m, 0.0)
    self._p1.copy_(torch.where(m, self._waypoints(), self._p1))
    self._t.masked_fill_(m, 0.0)
    self._T.copy_(torch.where(m, self._durations(), self._T))

  def step(self) -> torch.Tensor:
    """This step's command, then advance by ``dt``. A joint whose segment
    finishes starts the next one from the waypoint it just reached, carrying
    the leftover time so the timing does not drift."""
    u = (self._t / self._T).clamp(0.0, 1.0)
    out = self._p0 + (self._p1 - self._p0) * _min_jerk(u)
    self._t += self.dt
    done = self._t >= self._T
    self._t.copy_(torch.where(done, self._t - self._T, self._t))
    self._p0.copy_(torch.where(done, self._p1, self._p0))
    self._p1.copy_(torch.where(done, self._waypoints(), self._p1))
    self._T.copy_(torch.where(done, self._durations(), self._T))
    return out
