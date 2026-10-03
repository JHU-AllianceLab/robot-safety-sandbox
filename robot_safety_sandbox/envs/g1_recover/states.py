"""Start states for the G1 recovery task: dataset schema, frames, restore.

A START STATE is everything the next control step depends on: the floating
base pose and twist, the 29 joint positions and velocities, and the action the
previous controller last applied (mjlab's ``last_action`` observation, and the
PD targets the joints were tracking). ``examples/harvest_g1_recover_states.py``
writes a bank of them to one ``.pt`` file; the training spawn
(``envs/g1_recover/spawn.py``) and the harvester's own replay check restore
them through :func:`write_states`, so what is validated is what is trained on.

HEADING FRAME. The G1's observations are invariant to position on a flat plane
and to heading (body-frame twist, projected gravity, joint states), so a state
is stored with its yaw removed: the quaternion is premultiplied by the inverse
of its own yaw, and the world twist is rotated into that yaw frame. The spawn
draws a fresh yaw and planar offset, which turns every stored row into a
continuum of equivalent starts. Height is stored as-is (flat ground at z=0).

DATASET FIELDS (all tensors share the leading row dimension N):

    root_z (N,)            pelvis height, m
    root_quat_h (N, 4)     pelvis orientation, yaw removed (w, x, y, z)
    lin_vel_h (N, 3)       root-link linear velocity, heading frame, m/s
    ang_vel_h (N, 3)       root-link angular velocity, heading frame, rad/s
    joint_pos (N, 29)      absolute joint positions (mjlab entity joint order)
    joint_vel (N, 29)      joint velocities
    action (N, 29)         the last applied action (mjlab action order)
    command (N, 3)         the walker's twist command at the snapshot (label)
    phase_step (N,)        episode step mod the gait period (replay check only)
    kind (N,)              KIND_CLEAN / KIND_PUSHED / KIND_SYNTHETIC
    fell (N,)              the walker fell within the label window (walker rows)
    time_to_fall (N,)      s from snapshot to that fall; NaN if it survived
    push_severity (N,)     push size in [0, 1] (0 for clean rows)
    difficulty (N,)        the synthetic perturbation scale in [0, 1]
    speed (N,)             horizontal speed |lin_vel_h[:2]|
    env_id (N,)            the harvest env the row came from (replay check)

plus ``meta`` (a plain dict: source policy, task, ranges, counts).
"""

from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  yaw_quat,
)

#: Dataset ``kind`` codes.
KIND_CLEAN = 0       # walker mid-gait, undisturbed
KIND_PUSHED = 1      # walker 0-300 ms after a random push
KIND_SYNTHETIC = 2   # default stance + random tilt / twist / joint offsets

#: The fields every row carries (``meta`` rides alongside).
FIELDS = ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos",
          "joint_vel", "action", "command", "phase_step", "kind", "fell",
          "time_to_fall", "push_severity", "difficulty", "speed", "env_id")

#: The G1 sole collision capsules: 7 per foot (``left_foot1..7_collision``).
SOLE_GEOMS = tuple(f"{side}_foot{i}_collision"
                   for side in ("left", "right") for i in range(1, 8))


def to_heading_frame(quat_w: torch.Tensor, lin_vel_w: torch.Tensor,
                     ang_vel_w: torch.Tensor):
  """World pose/twist -> (yaw-free quaternion, heading-frame lin, ang)."""
  q_yaw = yaw_quat(quat_w)
  quat_h = quat_mul(quat_inv(q_yaw), quat_w)
  return (quat_h, quat_apply_inverse(q_yaw, lin_vel_w),
          quat_apply_inverse(q_yaw, ang_vel_w))


def from_heading_frame(quat_h: torch.Tensor, lin_vel_h: torch.Tensor,
                       ang_vel_h: torch.Tensor, yaw: torch.Tensor):
  """The inverse of :func:`to_heading_frame` at a chosen heading ``yaw``."""
  zero = torch.zeros_like(yaw)
  q_yaw = quat_from_euler_xyz(zero, zero, yaw)
  return (quat_mul(q_yaw, quat_h), quat_apply(q_yaw, lin_vel_h),
          quat_apply(q_yaw, ang_vel_h))


def snapshot(env) -> dict[str, torch.Tensor]:
  """The restorable state of EVERY env of a live mjlab env, heading frame.

  Reads the true joint positions (``joint_pos``, not the encoder-biased
  reading the observation sees) and the root LINK twist, the same quantities
  :func:`write_states` writes back.
  """
  d = env.scene["robot"].data
  quat_h, lin_h, ang_h = to_heading_frame(
    d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w)
  return {
    "root_z": d.root_link_pos_w[:, 2].clone(),
    "root_quat_h": quat_h,
    "lin_vel_h": lin_h,
    "ang_vel_h": ang_h,
    "joint_pos": d.joint_pos.clone(),
    "joint_vel": d.joint_vel.clone(),
    "action": env.action_manager.action.clone(),
  }


def write_states(env, env_ids: torch.Tensor, rows: dict[str, torch.Tensor],
                 yaw: torch.Tensor, xy: torch.Tensor) -> None:
  """Place ``rows`` (heading frame) into ``env_ids`` at heading ``yaw`` and
  planar offset ``xy`` from each env's origin. Writes state only — call from a
  reset event (mjlab runs ``sim.forward()`` after it) or follow with an
  explicit ``env.sim.forward()``. The action is NOT written here: mjlab zeroes
  the action buffer after the reset events run, so the spawn seeds it later,
  from its ``reset()`` hook (see ``spawn.RecoverSpawn``)."""
  robot = env.scene["robot"]
  quat_w, lin_w, ang_w = from_heading_frame(
    rows["root_quat_h"], rows["lin_vel_h"], rows["ang_vel_h"], yaw)
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, :2] += xy
  pos[:, 2] = rows["root_z"]
  robot.write_root_link_pose_to_sim(torch.cat([pos, quat_w], dim=-1),
                                    env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1),
                                        env_ids=env_ids)
  robot.write_joint_state_to_sim(rows["joint_pos"], rows["joint_vel"],
                                 env_ids=env_ids)


class SoleGeometry:
  """The 14 sole capsules of a G1 entity, for ground alignment.

  :meth:`min_z` is the lowest point of any sole, per env, after a
  ``sim.forward()``: a capsule's lowest point is always one of its two end
  spheres' bottoms, so it is ``min(endpoint z) - radius`` in any orientation.
  """

  def __init__(self, env):
    robot = env.scene["robot"]
    local, names = robot.find_geoms(SOLE_GEOMS, preserve_order=True)
    if len(local) != len(SOLE_GEOMS):
      raise RuntimeError(f"expected the 14 G1 sole capsules, found {names}")
    self._robot = robot
    self._local = torch.as_tensor(local, device=env.device)
    gid = robot.indexing.geom_ids[self._local].cpu().numpy()
    size = env.sim.mj_model.geom_size[gid]              # (14, 3): r, half_len
    self._radius = torch.as_tensor(size[:, 0], dtype=torch.float32,
                                   device=env.device)
    self._half = torch.as_tensor(size[:, 1], dtype=torch.float32,
                                 device=env.device)

  def min_z(self) -> torch.Tensor:
    d = self._robot.data
    pos = d.geom_pos_w[:, self._local]                  # (N, 14, 3)
    quat = d.geom_quat_w[:, self._local]                # (N, 14, 4)
    axis = quat_apply(quat.reshape(-1, 4),
                      torch.tensor([0.0, 0.0, 1.0], device=pos.device)
                      .expand(quat.shape[0] * quat.shape[1], 3))
    axis_z = axis[:, 2].reshape(quat.shape[0], quat.shape[1])
    end_lo = pos[..., 2] - (self._half * axis_z).abs()  # the lower endpoint
    return (end_lo - self._radius).amin(dim=1)


def speed_of(lin_vel_h: torch.Tensor) -> torch.Tensor:
  return torch.linalg.norm(lin_vel_h[:, :2], dim=1)


def tilt_deg(quat_h: torch.Tensor) -> torch.Tensor:
  """Angle between the pelvis up-axis and world vertical, degrees."""
  up = quat_apply(quat_h, torch.tensor([0.0, 0.0, 1.0], device=quat_h.device)
                  .expand(quat_h.shape[0], 3))
  return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))


def load_bank(path: str, device) -> dict:
  """Load a saved bank onto ``device``; ``meta`` stays a plain dict."""
  raw = torch.load(path, map_location=device, weights_only=False)
  missing = [f for f in FIELDS if f not in raw]
  if missing:
    raise ValueError(f"{path}: not a g1_recover state bank, missing {missing}")
  bank = {f: raw[f].to(device) for f in FIELDS}
  bank["meta"] = raw.get("meta", {})
  return bank


def sample_yaw_xy(n: int, device, xy_half: float = 0.5):
  yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
  xy = (torch.rand(n, 2, device=device) * 2.0 - 1.0) * xy_half
  return yaw, xy
