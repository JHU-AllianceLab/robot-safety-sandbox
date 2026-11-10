"""CPU checks of the SONIC push benchmark rebuilt in mjlab (envs/g1_recover/push_bench.py).

The modules are loaded by path: importing the package pulls in mjlab / warp.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os

import numpy as np
import pytest
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel: str, name: str):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ZOO, rel))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


pb = _load("robot_safety_sandbox/envs/g1_recover/push_bench.py", "_pb_test")
k_set = _load("robot_safety_sandbox/envs/g1_recover/k_set.py", "_kset_test")


def his_safety_margin(z, tilt, joint_room, nonfoot, joint_scale=0.2):
  """tools/sonic_lafan/learning/recovery_env.safety_margin, on raw quantities
  (joint_room: the worst joint's distance inside its widened range)."""
  g = np.minimum(np.minimum((z - 0.45) / 0.1, 1 - tilt), joint_room / joint_scale).clip(-3, 3)
  g = np.where(nonfoot, -1.0, g)
  failed = (z < 0.45) | (tilt > 1.0) | (joint_room < 0) | nonfoot
  return np.where(failed, np.minimum(g, -0.1), np.maximum(g, 0.0))


def test_margin_is_his():
  rng = np.random.default_rng(0)
  n = 20000
  z = rng.uniform(0.2, 1.0, n)
  tilt = rng.uniform(0.0, 1.6, n)
  room = rng.uniform(-0.3, 0.5, n)
  nonfoot = rng.random(n) < 0.1
  t = {"z": torch.tensor(z), "tilt": torch.tensor(tilt), "joint": torch.tensor(room),
       "nonfoot": torch.tensor(nonfoot)}
  ours = k_set.margin_from_terms(t, joint_scale=pb.JOINT_SCALE).numpy()
  assert np.allclose(ours, his_safety_margin(z, tilt, room, nonfoot), atol=1e-12)
  # g < 0 exactly outside K, whatever the scale
  out = k_set.outside_k(t).numpy()
  assert ((ours < 0) == out).all()
  assert pb.g_bench.k_joint_scale == 0.2 and k_set.g_k.k_joint_scale == 0.1


def _robot_model():
  mujoco = pytest.importorskip("mujoco")
  spec = importlib.util.find_spec("mjlab")
  if spec is None:
    pytest.skip("mjlab is not installed")
  xml = os.path.join(os.path.dirname(spec.origin), "asset_zoo", "robots", "unitree_g1", "xmls",
                     "g1.xml")
  return mujoco, mujoco.MjModel.from_xml_path(xml)


def _dist_to_surface(mujoco, m, g, p_body):
  """|distance from the geom's axis segment (or centre) - radius| of a body-frame point."""
  rot = np.empty(9)
  mujoco.mju_quat2Mat(rot, m.geom_quat[g])
  local = rot.reshape(3, 3).T @ (p_body - m.geom_pos[g])
  r, h = m.geom_size[g, 0], m.geom_size[g, 1]
  if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
    return abs(np.linalg.norm(local) - r)
  zc = np.clip(local[2], -h, h)
  return abs(np.linalg.norm(local - np.array([0.0, 0.0, zc])) - r)


def test_surface_points_lie_on_the_collision_shapes():
  mujoco, m = _robot_model()
  rng = np.random.default_rng(1)
  for site in pb.PUSH_SITES:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, site)
    geoms = pb.collision_geoms(m, bid)
    assert geoms, site
    pts = pb.sample_surface_points(m, bid, 200, rng)
    for p in pts:
      # on the surface of at least one of the body's geoms
      assert min(_dist_to_surface(mujoco, m, g, p) for g in geoms) < 1e-9, site


def test_robot_mass_is_his():
  _mujoco, m = _robot_model()
  assert abs(float(m.body_mass.sum()) - 33.341142) < 1e-6


def test_push_force_is_his():
  f = pb.push_force(33.341142, torch.tensor([2.0]), torch.tensor([10]), 0.02,
                    torch.tensor([math.pi / 2]))
  assert torch.allclose(f, torch.tensor([[0.0, 333.41142, 0.0]]), atol=1e-4)


HIS_DEV = os.path.join(_ZOO, "data", "zz_test", "trials.csv")


@pytest.mark.skipif(not os.path.exists(HIS_DEV), reason="his dev-heldout table is not here")
def test_dev_table_is_his(tmp_path):
  mt = _load("examples/make_push_table.py", "_mpt_test")
  mt.main(["--preset", "dev", "--out", str(tmp_path / "dev")])
  his = list(csv.DictReader(open(HIS_DEV)))
  ours = list(csv.DictReader(open(tmp_path / "dev" / "trials.csv")))
  assert len(his) == len(ours) == 8640
  for h, o in zip(his, ours):
    for k in ("context_id", "motion", "phase_stratum", "start_frame", "onset_frame", "site",
              "direction"):
      assert h[k] == o[k], k
    assert float(h["dv_mps"]) == float(o["dv_mps"])
    assert abs(float(h["onset_yaw_rad"]) - float(o["onset_yaw_rad"])) < 1e-9
    assert np.allclose(json.loads(h["world_force_N"]), json.loads(o["world_force_N"]), atol=1e-6)
    assert o["direction_frame"] == "reference"


def test_train_table_has_his_contexts(tmp_path):
  if not os.path.isdir(os.path.join(_ZOO, "data", "lafan1_g1")):
    pytest.skip("no LAFAN CSVs")
  _robot_model()
  mt = _load("examples/make_push_table.py", "_mpt_test2")
  mt.main(["--preset", "train", "--out", str(tmp_path / "train")])
  c = json.load(open(tmp_path / "train" / "contract.json"))
  per = {m: sum(x["motion"] == m for x in c["contexts"]) for m in c["motions"]}
  # his README: walk / run 24 starts each, dance 19 (the 105 s cut), 16080 pushes
  assert per == {"walk1_subject1": 24, "run1_subject2": 24, "dance1_subject1": 19}
  assert c["trials"] == 16080
  last = max(x["start_frame"] for x in c["contexts"] if x["motion"] == "dance1_subject1")
  assert last + 100 + 510 + 45 <= 5250


def test_subset_library_cut():
  sub = _load("examples/subset_clip_library.py", "_sub_test")
  n = 300
  lib = {"clip": torch.tensor([0] * 100 + [1] * 200), "frame": torch.cat([torch.arange(100),
                                                                           torch.arange(200)]),
         "next_ok": torch.cat([99 - torch.arange(100), 199 - torch.arange(200)]),
         "start_ok": torch.ones(n, dtype=torch.bool), "kind": torch.full((n,), 3),
         "meta": {"clips": ["a", "b"]}}
  out = sub.subset(lib, ["b"], {"b": 150 / 50.0}, preview=45, fps=50.0)
  assert int(out["clip"].unique().numel()) == 1 and int(out["frame"].max()) == 150
  # no window plus preview passes the cut
  assert int((out["frame"] + out["next_ok"] + 45)[out["start_ok"]].max()) <= 150
  assert out["meta"]["clips"] == ["a", "b"]
