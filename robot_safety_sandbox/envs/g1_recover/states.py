"""Start states for the G1 recovery task: dataset schema, frames, restore.

A START STATE is everything the next control step depends on: the floating
base pose and twist, the 29 joint positions and velocities, and the action the
previous controller last applied (mjlab's ``last_action`` observation, and the
PD targets the joints were tracking). ``examples/harvest_g1_recover_states.py``
writes a bank of them to one ``.pt`` file; the training spawn
(``envs/g1_recover/spawn.py``) and the harvester's own replay check restore
them through :func:`write_states`, so what is validated is what is trained on.
Other producers (the mocap builder, the start-state labeler, the handover
harvesters) write the same schema through :func:`save_bank`, and the spawn can
train on several banks at once (:func:`load_bank` with a list of paths).

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
    kind (N,)              KIND_CLEAN / KIND_PUSHED / KIND_SYNTHETIC /
                           KIND_MOCAP / KIND_HANDOVER
    fell (N,)              the walker fell within the label window (walker rows)
    time_to_fall (N,)      s from snapshot to that fall; NaN if it survived
    push_severity (N,)     push size in [0, 1] (0 for clean rows)
    difficulty (N,)        the synthetic perturbation scale in [0, 1]
    speed (N,)             horizontal speed |lin_vel_h[:2]|
    env_id (N,)            the harvest env the row came from (replay check)

plus ``meta`` (a plain dict: source policy, task, ranges, counts).

OPTIONAL FIELDS (:data:`OPTIONAL_FIELDS`). Columns only some producers write:
the mocap builder's provenance, the start-state labeler's labels, a handover
harvester's pending push. :func:`load_bank` returns EVERY one of them, filled
with its documented default where a file lacks it, so a consumer never has to
ask which producer wrote a bank and banks of different producers concatenate:

    clip (N,)              index into the clip names of the row's file
                           (``meta["clips"]``; -1 = not a clip frame)
    category (N,)          index into ``meta["categories"]`` (-1)
    t (N,)                 s into the clip (NaN)
    airborne (N,)          no sole near the floor at that frame (False)
    frame (N,)             50 fps frame index within the clip (-1)
    next_ok (N,)           how many of the following rows continue the same
                           clip, contiguously at 50 fps: a replay of rows
                           i+1 .. i+k stays inside the clip iff next_ok >= k (0)
    start_ok (N,)          the spawn may start an episode here (True). False
                           keeps a row for replay / bookkeeping only; no
                           sampler ever spawns it
    p_surv (N,)            P(survive) of the labeling policy from this row,
                           from rollouts (NaN = unlabeled)
    n_label (N,)           rollouts behind p_surv (0)
    value0 (N,)            the labeling policy's critic value at the row (NaN)
    p_value (N,)           value0 mapped to a probability (NaN)
    ttf_mean (N,)          mean time-to-fall of the failed rollouts, s (NaN)
    push_force_h (N, 3)    a PENDING push still acting at the row, N, in the
                           row's HEADING frame (the twist's frame); the spawn
                           rotates it by its fresh yaw (0 = none)
    push_left_steps (N,)   control steps that push still acts (0 = none)
    push_impulse (N,)      N*s of the push that produced the row (NaN)
    handover_delay_s (N,)  time from the push onset to the handover, s, as the
                           harvester defined it (NaN)
    push_duration_s (N,)   that push's full duration, s (NaN)
    push_onset_s (N,)      its onset after the replay start, s (NaN)
    push_rel (N,)          its heading-frame angle, rad (0 = from behind; NaN)
    src_row (N,)           the library row the replay started from (-1)
    monitor_value (N,)     the monitor's value at a monitor-rule handover (NaN)
    value (N,)             the frontier harvester's policy value at the
                           snapshot (NaN)
    delay_steps (N,)       the frontier harvester's steps from the push end to
                           the snapshot (-1)

SEVERAL FILES. ``load_bank([a, b, ...])`` concatenates the files' rows in the
given order and adds ``part`` (the index of the file each row came from); its
``meta`` is ``{"parts": [meta_a, meta_b, ...], "paths": [...]}``. Indices that
point into a file's own meta (``clip``, ``category``) are NOT remapped: the
name of row i's clip is ``meta["parts"][part[i]]["clips"][clip[i]]``. A single
path (a plain ``str``) loads exactly as before: ``meta`` is that file's own
dict and ``part`` is zero (or the file's own ``part`` column, so a merged bank
written back with :func:`save_bank` keeps resolving its names).
"""

from __future__ import annotations

import math
import os
from typing import Sequence, Union

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
KIND_MOCAP = 3       # a kinematic retargeted-mocap frame (no controller made it)
KIND_HANDOVER = 4    # captured at a handover to the fallback (after a replayed
                     # clip + push, or a push under a recovery policy); may
                     # carry a PENDING push that is still acting

#: The fields every row carries (``meta`` rides alongside).
FIELDS = ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos",
          "joint_vel", "action", "command", "phase_step", "kind", "fell",
          "time_to_fall", "push_severity", "difficulty", "speed", "env_id")

_NAN = float("nan")
#: The optional fields (module docstring): name -> (dtype, trailing shape,
#: fill value for a file that lacks the column).
OPTIONAL_FIELDS: dict[str, tuple[torch.dtype, tuple[int, ...], object]] = {
  "clip": (torch.long, (), -1),
  "category": (torch.long, (), -1),
  "t": (torch.float32, (), _NAN),
  "airborne": (torch.bool, (), False),
  "frame": (torch.long, (), -1),
  "next_ok": (torch.long, (), 0),
  "start_ok": (torch.bool, (), True),
  "p_surv": (torch.float32, (), _NAN),
  "n_label": (torch.long, (), 0),
  "value0": (torch.float32, (), _NAN),
  "p_value": (torch.float32, (), _NAN),
  "ttf_mean": (torch.float32, (), _NAN),
  "push_force_h": (torch.float32, (3,), 0.0),
  "push_left_steps": (torch.long, (), 0),
  "push_impulse": (torch.float32, (), _NAN),
  "handover_delay_s": (torch.float32, (), _NAN),
  # the harvesters' analysis columns, kept so a labeled copy of a bank (and
  # every later reader) still has them
  "push_duration_s": (torch.float32, (), _NAN),
  "push_onset_s": (torch.float32, (), _NAN),
  "push_rel": (torch.float32, (), _NAN),
  "src_row": (torch.long, (), -1),
  "monitor_value": (torch.float32, (), _NAN),
  "value": (torch.float32, (), _NAN),
  "delay_steps": (torch.long, (), -1),
}

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


def heading_to_world(vec_h: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Heading-frame vectors ``(N, 3)`` -> world, at heading ``yaw``: the rotation
  :func:`from_heading_frame` applies to the twist (a stored pending push force
  is placed with it, so it keeps its angle to the restored body)."""
  zero = torch.zeros_like(yaw)
  return quat_apply(quat_from_euler_xyz(zero, zero, yaw), vec_h)


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


def optional_column(name: str, n: int, device) -> torch.Tensor:
  """The default (fill-value) column of optional field ``name`` for ``n`` rows."""
  dtype, shape, fill = OPTIONAL_FIELDS[name]
  return torch.full((int(n), *shape), fill, dtype=dtype, device=device)


def _load_one(path: str, device) -> dict:
  """One file: FIELDS as stored, every OPTIONAL field (filled where absent),
  the file's own ``part`` column if it has one, and its meta. Other columns are
  dropped, as they always were."""
  raw = torch.load(path, map_location=device, weights_only=False)
  missing = [f for f in FIELDS if f not in raw]
  if missing:
    raise ValueError(f"{path}: not a g1_recover state bank, missing {missing}")
  bank = {f: raw[f].to(device) for f in FIELDS}
  n = int(bank["kind"].shape[0])
  for name, (dtype, shape, _fill) in OPTIONAL_FIELDS.items():
    if name not in raw:
      bank[name] = optional_column(name, n, device)
      continue
    col = raw[name].to(device=device, dtype=dtype)
    if tuple(col.shape) != (n, *shape):
      raise ValueError(f"{path}: optional field {name!r} has shape "
                       f"{tuple(col.shape)}, expected {(n, *shape)}")
    bank[name] = col
  part = raw.get("part")
  if torch.is_tensor(part) and tuple(part.shape) == (n,):
    bank["part"] = part.to(device=device, dtype=torch.long)
  bank["meta"] = raw.get("meta", {})
  return bank


def load_bank(path_or_paths: Union[str, Sequence[str]], device) -> dict:
  """Load a saved bank (one path) or the concatenation of several (a list or
  tuple of paths) onto ``device``; ``meta`` stays a plain dict.

  Returns FIELDS, every :data:`OPTIONAL_FIELDS` key, ``part`` (long per row:
  the index of the source file) and ``meta`` -- that file's own meta for a
  single ``str`` path, ``{"parts": [...], "paths": [...]}`` for a list (even
  of one path), see the module docstring."""
  if isinstance(path_or_paths, (str, os.PathLike)):
    bank = _load_one(os.fspath(path_or_paths), device)
    if "part" not in bank:
      bank["part"] = torch.zeros(int(bank["kind"].shape[0]), dtype=torch.long,
                                 device=device)
    return bank
  paths = [os.fspath(p) for p in path_or_paths]
  if not paths:
    raise ValueError("load_bank: empty list of paths")
  parts = [_load_one(p, device) for p in paths]
  bank = {k: torch.cat([b[k] for b in parts])
          for k in FIELDS + tuple(OPTIONAL_FIELDS)}
  bank["part"] = torch.cat([
    torch.full((int(b["kind"].shape[0]),), i, dtype=torch.long, device=device)
    for i, b in enumerate(parts)])
  bank["meta"] = {"parts": [b["meta"] for b in parts], "paths": paths}
  return bank


def save_bank(bank: dict, path: str) -> None:
  """Write ``bank`` to ``path`` the way :func:`load_bank` reads it: every
  tensor detached to the CPU (a bank built on the GPU reloads anywhere), every
  other value (``meta``) as-is; parent directories are created."""
  out = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
         for k, v in bank.items()}
  parent = os.path.dirname(os.path.abspath(path))
  os.makedirs(parent, exist_ok=True)
  torch.save(out, path)


def sample_yaw_xy(n: int, device, xy_half: float = 0.5):
  yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
  xy = (torch.rand(n, 2, device=device) * 2.0 - 1.0) * xy_half
  return yaw, xy
