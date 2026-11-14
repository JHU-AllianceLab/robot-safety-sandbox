"""CPU checks of the squat-tracking reference (envs/g1_track/reference.py).

What they pin: the segment's frames on the 50 fps "mjlab-main" grid (8300 ..
8950, T = 651), joints inside the identified robot's hard range shrunk by 0.01
rad, forward kinematics and grounding on the identified G1 (independent MuJoCo
passes), the finite-difference velocities, the on-feet facts the experiment
relies on, the mixed-init start draw and the world placement.

The modules are loaded by path: importing the package pulls in the whole
sandbox (minutes on a login node). mujoco, mjlab and the G1 sysID repo are
needed for the reference itself (about a minute: the clip is resampled whole).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel: str, name: str):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ZOO, rel))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod          # dataclasses look their module up
  spec.loader.exec_module(mod)
  return mod


R = _load("robot_safety_sandbox/envs/g1_track/reference.py", "_g1_track_reference_test")
DT = 0.02


def _mjlab_math():
  pytest.importorskip("mjlab")
  from mjlab.utils.lab_api import math as m
  return m


@pytest.fixture(scope="module")
def built():
  mujoco = pytest.importorskip("mujoco")
  pytest.importorskip("mjlab")
  sysid = _load("robot_safety_sandbox/envs/assets/g1_sysid.py", "_g1_track_sysid_test")
  if not sysid.SYSID_DIR.is_dir():
    pytest.skip(f"no G1 sysID repo at {sysid.SYSID_DIR}")
  seg = R.SegmentSpec()
  csv = os.path.join(seg.csv_dir, f"{seg.clip}.csv")
  if not os.path.isfile(csv):
    pytest.skip(f"no {csv}")
  sonic = _load("robot_safety_sandbox/envs/g1_recover/sonic.py", "_g1_track_sonic_test")
  model = sysid.get_spec().compile()
  ref = R.build_reference(seg, model=model, sonic=sonic)
  return SimpleNamespace(ref=ref, model=model, sonic=sonic, seg=seg, mujoco=mujoco,
                         csv=np.loadtxt(csv, delimiter=","))


def test_segment_frames():
  seg = R.SegmentSpec()
  assert (seg.clip, seg.t0, seg.t1) == ("fightAndSports1_subject1", 166.0, 179.0)
  assert seg.frames == (8300, 8950) and seg.T == 651
  assert seg.grid == "mjlab-main" and seg.fps == 50


def test_ground_shift():
  low = torch.tensor([0.010, -0.004, 0.001, 0.10, 0.15, 0.08, 0.005, 0.002], dtype=torch.float64)
  dz = R.ground_shift(low, 0.002, 0.03)
  on = low < 0.03
  assert torch.allclose((low + dz)[on], torch.full((int(on.sum()),), 0.002, dtype=torch.float64))
  # the hop keeps its arc: the shift is interpolated between the frames around it
  a, b = 0.002 - low[2], 0.002 - low[6]
  assert torch.allclose(dz[3:6], a + (b - a) * torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
  assert bool((low + dz >= 0.002 - 1e-12).all())
  assert torch.allclose(low + R.ground_shift(low, 0.002, math.inf),
                        torch.full_like(low, 0.002))
  assert torch.equal(R.ground_shift(low, 0.002, -1.0), torch.zeros_like(low))


def test_sample_start_frames():
  torch.manual_seed(0)
  T, left = 651, 50
  s = R.sample_start_frames(20000, T, 0.5, left, "cpu")
  assert int(s.min()) == 0 and int(s.max()) == T - 1 - left
  # half at frame 0, plus the uniform draw's own 1 / 601 share of it
  assert abs(float((s == 0).float().mean()) - (0.5 + 0.5 / (T - left))) < 0.015
  assert bool((R.sample_start_frames(100, T, 1.0, left, "cpu") == 0).all())
  with pytest.raises(ValueError):
    R.sample_start_frames(4, 10, 0.5, 10, "cpu")


def test_placement_lands_the_root_where_write_states_puts_the_robot():
  m = _mjlab_math()
  torch.manual_seed(1)
  n = 64
  root = torch.randn(n, 3)
  root[:, 2] = torch.rand(n) + 0.3
  quat = m.quat_from_euler_xyz(0.3 * torch.randn(n), 0.3 * torch.randn(n),
                               math.pi * (2 * torch.rand(n) - 1))
  yaw = math.pi * (2 * torch.rand(n) - 1)
  xy = torch.rand(n, 2) - 0.5
  origins = torch.randn(n, 3)
  yaw_q, off = R.placement(root, quat, yaw, xy, origins)
  placed = m.quat_apply(yaw_q, root) + off
  assert torch.allclose(placed[:, :2], origins[:, :2] + xy, atol=1e-5)
  assert torch.allclose(placed[:, 2], root[:, 2], atol=1e-6)
  # the placed root heads at yaw, exactly as states.from_heading_frame places it
  qw = m.quat_mul(yaw_q, quat)
  qy = m.yaw_quat(qw)
  heading = 2 * torch.atan2(qy[:, 3], qy[:, 0])
  wrapped = torch.remainder(heading - yaw + math.pi, 2 * math.pi)
  assert torch.allclose(wrapped, torch.full_like(yaw, math.pi), atol=1e-4)
  q_h = m.quat_mul(m.quat_inv(m.yaw_quat(quat)), quat)
  zero = torch.zeros_like(yaw)
  ws = m.quat_mul(m.quat_from_euler_xyz(zero, zero, yaw), q_h)
  assert torch.allclose(m.quat_error_magnitude(ws, qw), torch.zeros(n), atol=1e-4)
  # rigid: distances between placed points are kept
  p = root + torch.randn(n, 3)
  assert torch.allclose((m.quat_apply(yaw_q, p) + off - placed).norm(dim=1),
                        (p - root).norm(dim=1), atol=1e-5)


def test_slice_matches_the_csv(built):
  """Frames 8300 and 8950 are CSV rows 4980 and 5370 (t = 166.0, 179.0 s at 30 fps)."""
  ref, csv = built.ref, built.csv
  assert ref.T == 651 and ref.meta["f0"] == 8300 and ref.meta["f1"] == 8950
  assert ref.joint_names == tuple(built.sonic._csv_joints())
  lo, hi = ref.joint_range[:, 0] + 0.01, ref.joint_range[:, 1] - 0.01
  for i, row in ((0, 4980), (-1, 5370)):
    q_csv = torch.as_tensor(csv[row, 7:], dtype=torch.float32)
    assert torch.allclose(ref.q[i], torch.minimum(torch.maximum(q_csv, lo), hi), atol=2e-3)
    assert np.allclose(ref.root_pos[i, :2].numpy(), csv[row, :2], atol=2e-3)


def test_joints_inside_the_hard_range(built):
  ref = built.ref
  lo, hi = ref.joint_range[:, 0] + 0.01, ref.joint_range[:, 1] - 0.01
  assert bool((ref.q >= lo - 1e-6).all() and (ref.q <= hi + 1e-6).all())
  # the reference's waist pitch runs past the identified +0.503 rad stop
  assert ref.meta["clipped_high"].get("waist_pitch_joint", 0) > 100
  wp = ref.joint_names.index("waist_pitch_joint")
  assert float(ref.joint_range[wp, 1]) == pytest.approx(0.5029, abs=1e-4)


def test_fk_and_grounding_on_the_sysid_robot(built):
  """An independent MuJoCo pass over the stored root and joints reproduces the
  stored bodies, and grounded frames have their lowest collision point at 2 mm."""
  mujoco, model, ref = built.mujoco, built.model, built.ref
  d = mujoco.MjData(model)
  obj = mujoco.mjtObj
  adr = [model.jnt_qposadr[mujoco.mj_name2id(model, obj.mjOBJ_JOINT, n)] for n in ref.joint_names]
  bid = [mujoco.mj_name2id(model, obj.mjOBJ_BODY, n) for n in ref.body_names]
  col = [g for g in range(model.ngeom)
         if (mujoco.mj_id2name(model, obj.mjOBJ_GEOM, g) or "").endswith("_collision")]
  low = []
  for i in range(ref.T):
    d.qpos[:3] = ref.root_pos[i].double().numpy()
    d.qpos[3:7] = ref.root_quat[i].double().numpy()
    d.qpos[adr] = ref.q[i].double().numpy()
    mujoco.mj_kinematics(model, d)
    if i % 25 == 0:
      assert np.allclose(d.xpos[bid], ref.body_pos[i].numpy(), atol=1e-4)
      assert np.allclose(np.abs((d.xquat[bid] * ref.body_quat[i].numpy()).sum(1)), 1.0, atol=1e-5)
    # the lowest point of a capsule is an end sphere's bottom; of a sphere, its bottom
    zs = []
    for g in col:
      r, half = model.geom_size[g, 0], model.geom_size[g, 1]
      z, axis_z = d.geom_xpos[g, 2], d.geom_xmat[g].reshape(3, 3)[2, 2]
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        zs.append(min(z + half * axis_z, z - half * axis_z) - r)
      elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
        zs.append(z - r)
      else:
        pytest.fail(f"unexpected collision geom type {model.geom_type[g]}")
    low.append(min(zs))
  low = np.array(low)
  grounded = low - ref.dz.double().numpy() < R.GROUND_SNAP_M    # the CSV's own height
  assert np.allclose(low[grounded], 0.002, atol=1e-5)
  assert bool((low >= 0.002 - 1e-5).all())
  # the three hops of the default segment stay airborne
  assert ref.meta["airborne_frames"] == int((~grounded).sum()) >= 40
  assert float(low.max()) > 0.1
  # the shift follows the CSV's sole height on the stance frames (about +-1 cm
  # frame to frame) and has no hop-sized step (grounding every frame: 2.6 cm)
  assert float(ref.dz.diff().abs().max()) < 0.02


def test_velocities_are_central_differences(built):
  m = _mjlab_math()
  ref = built.ref
  inner = slice(1, -1)
  fd = (ref.body_pos[2:] - ref.body_pos[:-2]) / (2 * DT)
  assert torch.allclose(ref.body_lin[inner], fd, atol=1e-3)
  pelvis = ref.body_names.index("pelvis")
  assert torch.allclose(ref.body_pos[:, pelvis], ref.root_pos, atol=1e-6)
  assert torch.allclose(ref.body_lin[:, pelvis], ref.root_lin, atol=1e-3)
  assert torch.allclose(ref.body_ang[:, pelvis], ref.root_ang, atol=1e-3)
  assert torch.allclose(ref.qd[inner], (ref.q[2:] - ref.q[:-2]) / (2 * DT), atol=1e-3)
  # turning q[i-1] by omega_i over 2 dt lands on q[i+1] (world-frame rate)
  w = ref.body_ang[inner].reshape(-1, 3).double()
  ang = w.norm(dim=1) * 2 * DT
  turn = m.quat_from_angle_axis(ang, w / w.norm(dim=1, keepdim=True).clamp_min(1e-12))
  q_prev = ref.body_quat[:-2].reshape(-1, 4).double()
  q_next = ref.body_quat[2:].reshape(-1, 4).double()
  err = m.quat_error_magnitude(m.quat_mul(turn, q_prev), q_next)
  assert float(err.max()) < 2e-3


def test_the_segment_is_on_the_feet(built):
  """The facts the user's choice rests on (AMENDMENTS A1)."""
  ref = built.ref
  assert ref.meta["nonfoot_clear_min"] >= 0.035              # ankle linkage brace, 3.8 cm
  assert float(ref.root_pos[:, 2].min()) == pytest.approx(0.24, abs=0.01)
  knees = [ref.joint_names.index(k) for k in ("left_knee_joint", "right_knee_joint")]
  assert 2.75 < float(ref.q[:, knees].max()) <= 2.83
  up = 1 - 2 * (ref.root_quat[:, 1] ** 2 + ref.root_quat[:, 2] ** 2)
  tilt = torch.rad2deg(torch.acos(up.clamp(-1, 1)))
  assert 44.0 < float(tilt.max()) < 48.0
  speed = ref.root_lin[:, :2].norm(dim=1)
  assert float(speed[0]) == pytest.approx(0.55, abs=0.05)
  assert float(speed[-1]) == pytest.approx(1.22, abs=0.05)
  assert float(speed.max()) == pytest.approx(1.80, abs=0.05)
