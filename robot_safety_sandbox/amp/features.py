"""The AMP feature phi(s): one definition for the env term and the expert builder.

A discriminator separates expert from policy on ANY difference between the two
feature pipelines, so both sides call :func:`amp_features` on the same physical
quantities:

  root_h       pelvis height above the LOWER ankle-roll link, m (AMENDMENTS B12). A
               kinematic quantity of the pose: the library's grounded root_z and
               the sim's contact height can differ by a centimetre, a free tell.
  gravity_b    gravity direction in the root frame (3): yaw-free tilt
  lin_vel_b    root linear velocity in the root frame (3)
  ang_vel_b    root angular velocity in the root frame (3)
  joint_pos    absolute joint angles, mjlab entity order = Unitree CSV order (29)
  joint_vel    joint velocities (29)

68-D. Everything is heading-invariant, so a library frame stored yaw-free
(``root_quat_h``, velocities in the heading frame) and a sim state at any yaw give
the same phi. No key-body positions: they are a function of the joints on this
robot, and leaving them out keeps the builder CPU-only.

Pure torch on purpose: the env's observation term imports this lazily, the
tests and the builder load it by file path without mjlab.
"""

from __future__ import annotations

import torch

FEATURE_VERSION = "g1-amp-v1"
N_JOINTS = 29
FEATURE_DIM = 1 + 3 + 3 + 3 + 2 * N_JOINTS
#: name -> slice into phi, for the per-group gap diagnostics
SLICES = {
  "height": slice(0, 1),
  "grav": slice(1, 4),
  "linvel": slice(4, 7),
  "angvel": slice(7, 10),
  "qpos": slice(10, 10 + N_JOINTS),
  "qvel": slice(10 + N_JOINTS, FEATURE_DIM),
}


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate ``v`` by the inverse of the unit quaternion ``q`` (w, x, y, z): the
  world-frame vector expressed in the body frame. Same formula as mjlab's
  ``lab_api.math.quat_apply_inverse``."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v - w * t + torch.cross(xyz, t, dim=-1)


def amp_features(root_h: torch.Tensor, root_quat_wxyz: torch.Tensor,
                 lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor,
                 joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
  """phi = [root_h, gravity_b, lin_vel_b, ang_vel_b, joint_pos, joint_vel], [..., 68].

  ``root_quat_wxyz`` may carry any yaw (only the gravity direction is used)."""
  down = torch.zeros_like(lin_vel_b)
  down[..., 2] = -1.0
  grav = quat_apply_inverse(root_quat_wxyz, down)
  return torch.cat([root_h.unsqueeze(-1), grav, lin_vel_b, ang_vel_b, joint_pos, joint_vel],
                   dim=-1)


def features_from_library(lib: dict, root_h: torch.Tensor,
                          joint_range: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
  """phi of every row of a clip library (``envs/g1_recover/states.py`` layout).

  The library stores the root yaw-free (``root_quat_h``) with its velocities in
  that heading frame; rotating them by the inverse of ``root_quat_h`` gives the
  body-frame velocities exactly (yaw cancels). ``root_h`` comes from forward
  kinematics of the same rows (``examples/build_amp_expert.py``).
  ``joint_range`` [29, 2]: clamp the joints into the ROBOT's range -- the mocap
  exceeds the sysID waist pitch limit on some frames, and an unreachable angle
  would let the discriminator tell expert from policy for free.
  Returns (phi [R, 68], number of clamped rows)."""
  q = lib["root_quat_h"].float()
  qpos = lib["joint_pos"].float()
  clamped = 0
  if joint_range is not None:
    lo, hi = joint_range[:, 0].float(), joint_range[:, 1].float()
    out = (qpos < lo) | (qpos > hi)
    clamped = int(out.any(1).sum())
    qpos = torch.maximum(torch.minimum(qpos, hi), lo)
  return amp_features(root_h.float(), q, quat_apply_inverse(q, lib["lin_vel_h"].float()),
                      quat_apply_inverse(q, lib["ang_vel_h"].float()), qpos,
                      lib["joint_vel"].float()), clamped
