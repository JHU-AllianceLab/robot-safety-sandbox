"""Random leg stances for the G1 stay spawn.

The stock spawn puts the legs in one pose, every episode: mjlab's knees-bent
keyframe (hip pitch -0.312, knee 0.669, ankle pitch -0.363). This module gives
the spawn a DISTRIBUTION of stances instead, each one a pose a standing robot
could actually be in:

  * both feet flat on the floor (every sole capsule within ``FLAT_TOL`` of z=0),
  * every leg joint inside its soft limits,
  * the whole-body CoM over the support polygon, ``COM_MARGIN`` inside its edge,
  * the two feet ``FOOT_GAP`` apart or more.

Random per-joint offsets would satisfy none of these (they lift one foot, bury
the other, tilt both soles), so a stance is sampled in STANCE space and the
leg joints are solved for it by inverse kinematics:

  pelvis height   nominal + ``height``         (m; negative = deeper squat)
  pelvis tilt     ``roll``, ``pitch``          (rad, relative to the floor)
  pelvis offset   ``shift`` in x and y         (m, over the feet)
  stance width    nominal + ``width``          (m, between the ankles)
  foot split      ``split``                    (m, one foot ahead of the other)
  toe angle       ``foot_yaw``, per foot       (rad, about the vertical)

each uniform over its range times ``scale`` (0 = the nominal stance only, the
stock behaviour). Each leg is a 6-DoF chain for a 6-DoF foot pose, so the IK is
exactly determined; a draw whose IK does not converge, or that fails a check
above, is rejected and redrawn.

The bank is built ONCE per process on the CPU (``mujoco``, damped least squares
from the nominal pose) and shared by every env that asks for the same ranges;
the reset event then only indexes it. Arms and waist stay at their defaults while
the bank is built: the waist goal is drawn per episode, after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, astuple

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_mul, yaw_quat

#: The 12 leg joints, in the order a bank row stores them.
LEG_JOINTS = tuple(f"{side}_{j}_joint" for side in ("left", "right")
                   for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                             "ankle_pitch", "ankle_roll"))
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
SOLE_GEOMS = tuple(f"{side}_foot{i}_collision"
                   for side in ("left", "right") for i in range(1, 8))
#: Acceptance checks (see the module docstring).
FLAT_TOL = 0.002        # m
COM_MARGIN = 0.02       # m
FOOT_GAP = 0.03         # m, closest sole points of the two feet
#: Height the spawn adds above the bank's touching pose, m.
CLEARANCE = 0.003


@dataclass(frozen=True)
class StanceRanges:
  """Uniform ranges (lo, hi) of the stance parameters, at ``scale`` 1."""
  height: tuple[float, float] = (-0.12, 0.02)
  roll: tuple[float, float] = (-0.15, 0.15)
  pitch: tuple[float, float] = (-0.20, 0.20)
  shift: tuple[float, float] = (-0.04, 0.04)
  width: tuple[float, float] = (-0.03, 0.12)
  split: tuple[float, float] = (-0.12, 0.12)
  foot_yaw: tuple[float, float] = (-0.25, 0.25)


class _Model:
  """The host MjModel's ids for the robot's floating base, legs and soles."""

  def __init__(self, m: mujoco.MjModel, prefix: str, default_q: dict,
               soft_lim: dict):
    def jid(n):
      return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
    self.m = m
    self.d = mujoco.MjData(m)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and m.jnt_bodyid[j] == pelvis]
    if len(free) != 1:
      raise RuntimeError("expected one free joint on the G1 pelvis")
    self.pelvis, self.free_q = pelvis, int(m.jnt_qposadr[free[0]])
    self.leg_q = np.array([m.jnt_qposadr[jid(n)] for n in LEG_JOINTS])
    self.leg_v = np.array([m.jnt_dofadr[jid(n)] for n in LEG_JOINTS])
    self.lim = np.array([soft_lim[n] for n in LEG_JOINTS])
    self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + b)
                 for b in FOOT_BODIES]
    self.sole = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, prefix + g)
                          for g in SOLE_GEOMS])
    if (self.sole < 0).any():
      raise RuntimeError("missing G1 sole capsules")
    self.rad = m.geom_size[self.sole, 0]
    self.half = m.geom_size[self.sole, 1]
    # Every robot joint at its default; the pelvis at the origin, level.
    q = self.d.qpos
    q[:] = m.qpos0
    for n, v in default_q.items():
      j = jid(n)
      if j >= 0 and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        q[m.jnt_qposadr[j]] = v
    self.q_default = q.copy()

  def set_base(self, pos, quat):
    self.d.qpos[self.free_q:self.free_q + 3] = pos
    self.d.qpos[self.free_q + 3:self.free_q + 7] = quat

  def fk(self):
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)

  def sole_ends(self):
    """(14, 2, 3) world endpoints of each sole capsule's axis."""
    pos = self.d.geom_xpos[self.sole]
    axis = self.d.geom_xmat[self.sole].reshape(-1, 3, 3)[:, :, 2]
    off = (self.half[:, None] * axis)[:, None, :]
    return np.concatenate([pos[:, None] - off, pos[:, None] + off], axis=1)

  def sole_bottoms(self):
    """(14,) lowest point of each sole capsule."""
    return self.sole_ends()[..., 2].min(axis=1) - self.rad

  def foot_pose(self, k):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, self.d.xmat[self.feet[k]])
    return self.d.xpos[self.feet[k]].copy(), quat


def _quat_rpy(roll, pitch, yaw):
  quat = np.empty(4)
  r = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
  p = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
  y = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
  mujoco.mju_mat2Quat(quat, (y @ p @ r).flatten())
  return quat


def _quat_mul(a, b):
  out = np.empty(4)
  mujoco.mju_mulQuat(out, a, b)
  return out


def _rot_err(target, current):
  """Rotation vector taking ``current`` to ``target`` (world frame)."""
  inv, dq, v = np.empty(4), np.empty(4), np.empty(3)
  mujoco.mju_negQuat(inv, current)
  mujoco.mju_mulQuat(dq, target, inv)
  mujoco.mju_quat2Vel(v, dq, 1.0)
  return v


def _solve_legs(md: _Model, targets, iters=60, damping=1e-3, tol=1e-5):
  """Damped least squares on the 12 leg joints for both foot poses. Returns
  the final error norm."""
  jp, jr = np.zeros((3, md.m.nv)), np.zeros((3, md.m.nv))
  for _ in range(iters):
    md.fk()
    err, rows = [], []
    for k in range(2):
      pos, quat = md.foot_pose(k)
      err += [targets[k][0] - pos, _rot_err(targets[k][1], quat)]
      mujoco.mj_jacBody(md.m, md.d, jp, jr, md.feet[k])
      rows += [jp[:, md.leg_v].copy(), jr[:, md.leg_v].copy()]
    e, J = np.concatenate(err), np.vstack(rows)
    if np.abs(e).max() < tol:
      return float(np.abs(e).max())
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(12), e)
    md.d.qpos[md.leg_q] += np.clip(dq, -0.2, 0.2)
  md.fk()
  return float(np.abs(e).max())


def _hull(pts):
  """Convex hull (counter-clockwise) of 2-D points, monotone chain."""
  pts = sorted(map(tuple, pts))

  def half(seq):
    out = []
    for p in seq:
      while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                               - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
        out.pop()
      out.append(p)
    return out
  lo, hi = half(pts), half(reversed(pts))
  return np.array(lo[:-1] + hi[:-1])


def _inside_margin(poly, p):
  """Signed distance of ``p`` inside the CCW polygon (positive = inside)."""
  a, b = poly, np.roll(poly, -1, axis=0)
  edge = b - a
  cross = edge[:, 0] * (p[1] - a[:, 1]) - edge[:, 1] * (p[0] - a[:, 0])
  return float((cross / np.linalg.norm(edge, axis=1)).min())


def build_bank(m: mujoco.MjModel, default_q: dict, soft_lim: dict, n: int,
               ranges: StanceRanges = StanceRanges(), scale: float = 1.0,
               seed: int = 0, prefix: str = "robot/", max_tries: int | None = None):
  """``n`` accepted stances: dict of numpy arrays ``leg_q`` (n, 12) in
  LEG_JOINTS order, ``root_z`` (n,), ``root_quat`` (n, 4) (w, x, y, z; roll and
  pitch only), and ``params`` (n, 8) (height, roll, pitch, shift x, shift y,
  width, split, mean foot yaw) for diagnostics; plus ``accept_rate``."""
  md = _Model(m, prefix, default_q, soft_lim)
  rng = np.random.default_rng(seed)
  # The nominal stance: default joints, pelvis level at the origin.
  md.d.qpos[:] = md.q_default
  md.set_base([0, 0, 0], [1, 0, 0, 0])
  md.fk()
  floor = md.sole_bottoms().min()               # the floor, in this frame
  nominal = [md.foot_pose(k) for k in range(2)]
  h0 = -floor                                   # nominal pelvis height
  ankle = [(p - [0, 0, floor], q) for p, q in nominal]

  def u(r):
    lo, hi = r
    return scale * rng.uniform(lo, hi)
  out = {"leg_q": [], "root_z": [], "root_quat": [], "params": []}
  tries, max_tries = 0, max_tries or 50 * n
  while len(out["root_z"]) < n:
    tries += 1
    if tries > max_tries:
      raise RuntimeError(f"stance bank: only {len(out['root_z'])}/{n} accepted "
                         f"in {max_tries} draws; the ranges are too wide")
    dz, roll, pitch = u(ranges.height), u(ranges.roll), u(ranges.pitch)
    sx, sy = u(ranges.shift), u(ranges.shift)
    dw, split = u(ranges.width), u(ranges.split)
    yaw = (u(ranges.foot_yaw), u(ranges.foot_yaw))
    # Feet on the floor around the origin; the pelvis placed over them.
    targets = []
    for k, sgn in ((0, 1.0), (1, -1.0)):
      p, q = ankle[k]
      pos = p + np.array([sgn * split / 2, sgn * dw / 2, 0.0])
      targets.append((pos, _quat_mul(_quat_rpy(0, 0, sgn * yaw[k]), q)))
    base_quat = _quat_rpy(roll, pitch, 0.0)
    md.d.qpos[:] = md.q_default
    md.set_base([sx, sy, h0 + dz], base_quat)
    if _solve_legs(md, targets) > 1e-4:
      continue
    q_leg = md.d.qpos[md.leg_q].copy()
    if ((q_leg < md.lim[:, 0]) | (q_leg > md.lim[:, 1])).any():
      continue
    bottoms = md.sole_bottoms()
    if bottoms.max() - bottoms.min() > FLAT_TOL:
      continue
    ends = md.sole_ends()
    left, right = ends[:7].reshape(-1, 3)[:, :2], ends[7:].reshape(-1, 3)[:, :2]
    gap = np.linalg.norm(left[:, None] - right[None], axis=-1).min() - 2 * md.rad.max()
    if gap < FOOT_GAP:
      continue
    com = md.d.subtree_com[md.pelvis][:2]
    if _inside_margin(_hull(np.vstack([left, right])), com) < COM_MARGIN:
      continue
    out["leg_q"].append(q_leg)
    out["root_z"].append(h0 + dz - bottoms.min())
    out["root_quat"].append(base_quat)
    out["params"].append([dz, roll, pitch, sx, sy, dw, split, (yaw[0] + yaw[1]) / 2])
  res = {k: np.array(v) for k, v in out.items()}
  res["accept_rate"] = n / tries
  return res


_BANK_CACHE: dict = {}


def stance_bank(env, scale: float, n: int, seed: int,
                ranges: StanceRanges = StanceRanges()) -> dict:
  """This process's bank for (scale, n, seed, ranges), as tensors on
  ``env.device``; built on first use."""
  key = (float(scale), int(n), int(seed), astuple(ranges))
  if key not in _BANK_CACHE:
    robot = env.scene["robot"]
    names = robot.joint_names
    default_q = dict(zip(names, robot.data.default_joint_pos[0].tolist()))
    lim = robot.data.soft_joint_pos_limits[0].tolist()
    soft = {nm: tuple(lim[i]) for i, nm in enumerate(names)}
    b = build_bank(env.sim.mj_model, default_q, soft, n, ranges, scale, seed)
    print(f"[g1_waist] stance bank: {n} stances at scale {scale:g} "
          f"(acceptance {b['accept_rate']:.0%})")
    t = {k: torch.as_tensor(v, dtype=torch.float32, device=env.device)
         for k, v in b.items() if k != "accept_rate"}
    t["leg_ids"] = torch.as_tensor(
      robot.find_joints(LEG_JOINTS, preserve_order=True)[0], device=env.device)
    _BANK_CACHE[key] = t
  return _BANK_CACHE[key]


def reset_leg_stance(env, env_ids, scale: float = 1.0, n_bank: int = 4096,
                     seed: int = 0) -> None:
  """Reset event (after ``reset_base`` and the stock joint reset): put each
  reset env's legs in a stance drawn from the bank, and its pelvis at that
  stance's height and tilt, keeping the planar position, heading and twist
  ``reset_base`` drew."""
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  bank = stance_bank(env, scale, n_bank, seed)
  pick = torch.randint(0, bank["root_z"].shape[0], (len(env_ids),),
                       device=env.device)
  robot = env.scene["robot"]
  robot.write_joint_position_to_sim(bank["leg_q"][pick],
                                    joint_ids=bank["leg_ids"], env_ids=env_ids)
  pose = robot.data.root_link_pose_w[env_ids].clone()
  pose[:, 3:7] = quat_mul(yaw_quat(pose[:, 3:7]), bank["root_quat"][pick])
  pose[:, 2] = (env.scene.env_origins[env_ids, 2] + bank["root_z"][pick]
                + CLEARANCE)
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
