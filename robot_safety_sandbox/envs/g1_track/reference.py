"""The squat-tracking reference: one LAFAN segment on the identified G1, at 50 fps.

THE SEGMENT (:class:`SegmentSpec`). A clip of ``data/lafan1_g1`` and a CSV time
window. The default is the user's (2026-11-12): ``fightAndSports1_subject1``,
166.0-179.0 s -- a stand, four deep squat bounces on the feet, a walk-off.

THE TIME GRID. Frames are the clip library's and SONIC's: mjlab's MotionLoader
resamples the WHOLE 30 fps clip to 50 fps on the "mjlab-main" grid (t_k = 0.02 k
from CSV row 0; ``sonic._motion_loader``), and the segment is frames
``round(50 t0) .. round(50 t1)`` -- 8300 .. 8950, T = 651 for the default.

JOINTS are clipped into the robot's hard range shrunk by ``joint_margin`` (0.01
rad): the reference's waist pitch runs past the identified +0.503 rad stop on a
large share of the frames, and a target the robot cannot reach would hold the
joint against its stop.

FORWARD KINEMATICS on the identified G1 (``envs/assets/g1_sysid.get_spec``: its
refitted capsules decide what is closest to the floor), one ``mj_kinematics``
per frame on the CPU, then GROUNDING (:func:`ground_shift`) by the bank
builder's rule: a frame whose lowest collision point is below ``ground_snap_m``
(3 cm) stands on the floor and is shifted vertically so that point sits
``sole_clearance`` (2 mm, the builder's SOLE_CLEARANCE) above it -- the
retargeted CSV sinks and floats by about a centimetre. The default segment also
has three real hops (169.8, 172.3 and 175.0 s, feet up to 15 cm, 0.3-0.4 s):
their frames take the shift interpolated between the grounded frames around
them, so each jump keeps its arc and the shift has no step. ``ground_snap_m`` =
inf grounds every frame instead.

VELOCITIES by MotionLoader's conventions: central differences (``torch.gradient``)
of the grounded positions and the clipped joints, and the SO(3) central
difference of orientations, computed with one frame of context past each end of
the segment so its first and last frames get central differences too.

Nothing here imports mjlab or the sandbox at module level (tests load the file
by path); the defaults that need them (:func:`sysid_model`, the sonic module)
are imported where they are used.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

import numpy as np
import torch

#: The 14 bodies mjlab's G1 tracking task tracks (``tasks/tracking/config/g1/
#: env_cfgs.py``); the anchor is the torso.
TRACK_BODIES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
ANCHOR_BODY = "torso_link"
#: the sole capsules: ``left_foot1..7_collision``, ``right_foot1..7_collision``
_SOLE = re.compile(r"^(left|right)_foot\d+_collision$")
DEFAULT_CSV_DIR = os.path.join(
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "data", "lafan1_g1")
#: a frame whose lowest point is below this stands on the floor, m (the bank
#: builder's --snap default)
GROUND_SNAP_M = 0.03


@dataclass(frozen=True)
class SegmentSpec:
  """Which motion to track and how it is prepared (module docstring)."""

  clip: str = "fightAndSports1_subject1"
  t0: float = 166.0
  t1: float = 179.0
  csv_dir: str = DEFAULT_CSV_DIR
  input_fps: int = 30
  fps: int = 50
  grid: str = "mjlab-main"
  joint_margin: float = 0.01
  sole_clearance: float = 0.002
  ground_snap_m: float = GROUND_SNAP_M

  @property
  def frames(self) -> tuple[int, int]:
    """First and last frame on the 50 fps grid (both included)."""
    return int(round(self.t0 * self.fps)), int(round(self.t1 * self.fps))

  @property
  def T(self) -> int:
    f0, f1 = self.frames
    return f1 - f0 + 1


@dataclass
class TrackReference:
  """The prepared segment, ``T`` frames. Joints in ``joint_names`` order (the
  Unitree CSV order, which is mjlab's entity order); bodies in ``body_names``
  order; positions / orientations / velocities in the reference's own world
  frame (grounded), quaternions w-first."""

  seg: SegmentSpec
  joint_names: tuple
  body_names: tuple
  q: torch.Tensor            # [T, J] clipped joint positions, rad
  qd: torch.Tensor           # [T, J]
  root_pos: torch.Tensor     # [T, 3] pelvis
  root_quat: torch.Tensor    # [T, 4]
  root_lin: torch.Tensor     # [T, 3] world frame
  root_ang: torch.Tensor     # [T, 3] world frame
  body_pos: torch.Tensor     # [T, B, 3]
  body_quat: torch.Tensor    # [T, B, 4]
  body_lin: torch.Tensor     # [T, B, 3]
  body_ang: torch.Tensor     # [T, B, 3]
  dz: torch.Tensor           # [T] the grounding shift, m
  nonfoot_clear: torch.Tensor  # [T] lowest non-sole point minus lowest sole point, m
  joint_range: torch.Tensor  # [J, 2] the FK model's hard range
  meta: dict = field(default_factory=dict)

  @property
  def T(self) -> int:
    return int(self.q.shape[0])

  @property
  def anchor_index(self) -> int:
    return self.body_names.index(ANCHOR_BODY)

  def to(self, device) -> "TrackReference":
    moved = {k: v.to(device) for k, v in vars(self).items() if torch.is_tensor(v)}
    return replace(self, **moved)


# --- the model and its kinematics ----------------------------------------------

def sysid_model():
  """The identified G1 compiled on its own (``g1_sysid.get_spec``): no floor, the
  free joint first, names without an entity prefix."""
  from robot_safety_sandbox.envs.assets.g1_sysid import get_spec
  return get_spec().compile()


def collision_geoms(model) -> tuple[list[int], list[int]]:
  """``(sole ids, other collision geom ids)`` of a G1 model, by name."""
  import mujoco
  sole, other = [], []
  for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if name.endswith("_collision"):
      (sole if _SOLE.match(name) else other).append(g)
  if len(sole) != 14:
    raise ValueError(f"expected the G1's 14 sole capsules, found {len(sole)}")
  return sole, other


def geom_lowest_z(model, data, geom_ids) -> np.ndarray:
  """Lowest world z of each geom in ``geom_ids`` after a kinematics pass."""
  import mujoco
  ids = np.asarray(geom_ids, dtype=int)
  z = data.geom_xpos[ids, 2]
  rz = data.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, :]     # world z of each local axis
  size = model.geom_size[ids]
  typ = model.geom_type[ids]
  g = mujoco.mjtGeom
  low = z - model.geom_rbound[ids]
  cap = typ == g.mjGEOM_CAPSULE
  low[cap] = z[cap] - np.abs(rz[cap, 2]) * size[cap, 1] - size[cap, 0]
  sph = typ == g.mjGEOM_SPHERE
  low[sph] = z[sph] - size[sph, 0]
  box = typ == g.mjGEOM_BOX
  low[box] = z[box] - (np.abs(rz[box]) * size[box]).sum(1)
  cyl = typ == g.mjGEOM_CYLINDER
  low[cyl] = (z[cyl] - np.abs(rz[cyl, 2]) * size[cyl, 1]
              - size[cyl, 0] * np.sqrt(np.clip(1.0 - rz[cyl, 2] ** 2, 0.0, 1.0)))
  return low


def fk_frames(model, root_pos, root_quat, q, joint_names, body_names, geom_ids=()):
  """Forward kinematics of ``F`` frames (arrays ``[F, 3]``, ``[F, 4]`` w-first,
  ``[F, J]``): ``(body_pos [F, B, 3], body_quat [F, B, 4], geom_low [F, G])``,
  float64, one ``mj_kinematics`` per frame."""
  import mujoco
  obj = mujoco.mjtObj
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  if len(free) != 1:
    raise ValueError(f"expected one free joint, the model has {len(free)}")
  fa = int(model.jnt_qposadr[free[0]])
  jid = [mujoco.mj_name2id(model, obj.mjOBJ_JOINT, n) for n in joint_names]
  bid = [mujoco.mj_name2id(model, obj.mjOBJ_BODY, n) for n in body_names]
  if min(jid + bid) < 0:
    raise ValueError("joint or body names missing from the model")
  adr = model.jnt_qposadr[jid]
  root_pos, root_quat, q = (np.asarray(x, dtype=np.float64) for x in (root_pos, root_quat, q))
  n = root_pos.shape[0]
  d = mujoco.MjData(model)
  bp = np.empty((n, len(bid), 3))
  bq = np.empty((n, len(bid), 4))
  low = np.empty((n, len(geom_ids)))
  for i in range(n):
    d.qpos[fa:fa + 3] = root_pos[i]
    d.qpos[fa + 3:fa + 7] = root_quat[i]
    d.qpos[adr] = q[i]
    mujoco.mj_kinematics(model, d)
    bp[i] = d.xpos[bid]
    bq[i] = d.xquat[bid]
    if len(geom_ids):
      low[i] = geom_lowest_z(model, d, geom_ids)
  return bp, bq, low


def ground_shift(lowest: torch.Tensor, clearance: float, snap: float) -> torch.Tensor:
  """Per-frame vertical shift ``[F]`` for the frames' lowest points ``[F]``:
  frames below ``snap`` are moved so it sits at ``clearance``; the others
  (airborne) take the shift interpolated linearly in time between the grounded
  frames around them (constant past the ends), never into the floor."""
  on = lowest < snap
  if not bool(on.any()):
    return torch.zeros_like(lowest)
  idx = np.arange(lowest.shape[0], dtype=np.float64)
  on_np = on.numpy()
  dz = np.interp(idx, idx[on_np], (clearance - lowest[on]).numpy())
  return torch.maximum(torch.as_tensor(dz, dtype=lowest.dtype), clearance - lowest)


def so3_derivative(quat: torch.Tensor, dt: float) -> torch.Tensor:
  """World-frame angular velocity of a ``[F, ..., 4]`` orientation sequence:
  MotionLoader's central difference ``axis_angle(q[i+1] q[i-1]^-1) / 2 dt``, the
  first and last frame repeating their neighbour's."""
  from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul
  q_rel = quat_mul(quat[2:].contiguous(), quat_conjugate(quat[:-2].contiguous()))
  w = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([w[:1], w, w[-1:]], dim=0)


# --- building it -------------------------------------------------------------------

def build_reference(seg: SegmentSpec = SegmentSpec(), *, model=None,
                    body_names: tuple = TRACK_BODIES, device="cpu",
                    sonic=None) -> TrackReference:
  """Load, slice, clip, ground and differentiate ``seg`` (module docstring).
  ``model``: the compiled robot for FK and the joint ranges (default
  :func:`sysid_model`); ``sonic``: the module that provides ``_motion_loader`` and
  ``_csv_joints`` (default ``envs/g1_recover/sonic.py``)."""
  if sonic is None:
    from robot_safety_sandbox.envs.g1_recover import sonic
  model = sysid_model() if model is None else model
  import mujoco
  path = os.path.join(seg.csv_dir, f"{seg.clip}.csv")
  ml = sonic._motion_loader(path, int(seg.input_fps), int(seg.fps), seg.grid)
  if abs(float(ml.output_dt) - 1.0 / seg.fps) > 1e-12:
    raise ValueError(f"motion loader dt {ml.output_dt} is not 1/{seg.fps}")
  f0, f1 = seg.frames
  n = int(ml.output_frames)
  if not 0 <= f0 < f1 < n:
    raise ValueError(f"{seg.clip} {seg.t0}-{seg.t1} s = frames {f0}..{f1} is not "
                     f"inside the clip's {n} frames")
  # one frame of context past each end (when the clip has it), so the
  # segment's end frames get central differences
  lo, hi = max(f0 - 1, 0), min(f1 + 1, n - 1)
  sl = slice(f0 - lo, f0 - lo + seg.T)
  joint_names = tuple(sonic._csv_joints())
  jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in joint_names]
  if min(jid) < 0:
    raise ValueError("the FK model lacks some of the CSV's joints")
  rng = torch.as_tensor(model.jnt_range[jid], dtype=torch.float64)
  q_raw = ml.motion_dof_poss[lo:hi + 1].double()
  q_lo, q_hi = rng[:, 0] + seg.joint_margin, rng[:, 1] - seg.joint_margin
  q = torch.minimum(torch.maximum(q_raw, q_lo), q_hi)
  root_pos = ml.motion_base_poss[lo:hi + 1].double().clone()
  root_quat = ml.motion_base_rots[lo:hi + 1].double()
  sole, other = collision_geoms(model)
  bp, bq, low = fk_frames(model, root_pos.numpy(), root_quat.numpy(), q.numpy(),
                          joint_names, body_names, sole + other)
  low = torch.as_tensor(low)
  lowest = low.amin(1)
  dz = ground_shift(lowest, seg.sole_clearance, seg.ground_snap_m)
  root_pos[:, 2] += dz
  body_pos = torch.as_tensor(bp)
  body_pos[..., 2] += dz[:, None]
  body_quat = torch.as_tensor(bq)
  dt = 1.0 / seg.fps
  root_lin = torch.gradient(root_pos, spacing=dt, dim=0)[0]
  qd = torch.gradient(q, spacing=dt, dim=0)[0]
  body_lin = torch.gradient(body_pos, spacing=dt, dim=0)[0]
  root_ang = so3_derivative(root_quat, dt)
  body_ang = so3_derivative(body_quat, dt)
  nonfoot_clear = low[:, len(sole):].amin(1) - low[:, :len(sole)].amin(1)
  seg_q = q_raw[sl]
  meta = dict(clip=seg.clip, t0=seg.t0, t1=seg.t1, f0=f0, f1=f1, T=seg.T,
              grid=seg.grid, clip_frames=n,
              clipped_low={nm: int(c) for nm, c in zip(joint_names, (seg_q < q_lo).sum(0))
                           if c},
              clipped_high={nm: int(c) for nm, c in zip(joint_names, (seg_q > q_hi).sum(0))
                            if c},
              dz_range=(float(dz[sl].min()), float(dz[sl].max())),
              airborne_frames=int((lowest[sl] >= seg.ground_snap_m).sum()),
              nonfoot_clear_min=float(nonfoot_clear[sl].min()))
  ref = TrackReference(
    seg=seg, joint_names=joint_names, body_names=tuple(body_names),
    q=q[sl], qd=qd[sl], root_pos=root_pos[sl], root_quat=root_quat[sl],
    root_lin=root_lin[sl], root_ang=root_ang[sl], body_pos=body_pos[sl],
    body_quat=body_quat[sl], body_lin=body_lin[sl], body_ang=body_ang[sl],
    dz=dz[sl], nonfoot_clear=nonfoot_clear[sl], joint_range=rng, meta=meta)
  as32 = {k: v.float() for k, v in vars(ref).items() if torch.is_tensor(v)}
  return replace(ref, **as32).to(device)


# --- where an episode starts, and where the reference is placed --------------------

def sample_start_frames(n: int, T: int, p_start: float, min_left: int,
                        device) -> torch.Tensor:
  """The mixed init: frame 0 with probability ``p_start``, else a frame drawn
  uniformly from ``0 .. T-1-min_left`` (at least ``min_left`` steps remain).
  Torch's global RNG, like every other draw at a reset."""
  hi = int(T) - 1 - int(min_left)
  if hi < 0:
    raise ValueError(f"a {T}-frame reference cannot leave {min_left} steps")
  first = torch.rand(n, device=device) < float(p_start)
  rsi = torch.randint(0, hi + 1, (n,), device=device)
  return torch.where(first, torch.zeros_like(rsi), rsi)


def placement(root_pos_s: torch.Tensor, root_quat_s: torch.Tensor, yaw: torch.Tensor,
              xy: torch.Tensor, origins: torch.Tensor):
  """``(yaw_q [n, 4], offset [n, 3])`` that place the reference so its root at
  frame s (``root_pos_s`` / ``root_quat_s``, ``[n, ·]``) lands at heading ``yaw``
  and planar position ``origins + xy``, height unchanged:
  ``world = yaw_q * p + offset`` for positions, ``yaw_q * q`` for orientations.
  This is exactly where ``states.write_states`` puts the robot for the same yaw
  and xy, so a spawn at frame s starts ON the placed reference."""
  from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, yaw_quat
  qy = yaw_quat(root_quat_s.contiguous())
  ref_yaw = 2.0 * torch.atan2(qy[:, 3], qy[:, 0])
  zero = torch.zeros_like(yaw)
  yaw_q = quat_from_euler_xyz(zero, zero, yaw - ref_yaw)
  p = root_pos_s.clone()
  p[:, 2] = 0.0
  offset = torch.zeros_like(root_pos_s)
  offset[:, :2] = origins[:, :2] + xy
  return yaw_q, offset - quat_apply(yaw_q, p)
