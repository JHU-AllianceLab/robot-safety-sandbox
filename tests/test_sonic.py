"""The torch port of SONIC (envs/g1_recover/sonic.py SonicPolicy).

Simulator-free (CPU). Skipped unless the GEAR-SONIC release is on disk
(``SONIC_DIR``, default ``~/Documents/gear_sonic_hf``) and ``onnx`` imports.

  * ENCODER -- the g1-mode token from (pos, vel, anchor) equals the ONNX
    encoder's output on the same values laid out in its 1762-d input (mode 0,
    the other modes' slots filled with noise they must ignore);
  * DECODER -- the action equals the ONNX decoder's on a random 994-d input;
  * SCALE -- SONIC's per-joint action scale matches the deploy's table;
  * VELOCITY -- reference joint velocities are SONIC's motion-lib forward
    differences (last frame = the second-to-last difference);
  * HIP -- the hip-pitch swap is idempotent and leaves mjlab's shared G1
    articulation (and so every other G1 cfg) alone.

ENCODER / DECODER need the release on disk; the rest do not.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SONIC_JOINTS, SonicPolicy, forward_difference,
  sonic_action_scale, use_sonic_hip_pitch)

_HAVE = all(os.path.exists(os.path.join(DEFAULT_SONIC_DIR, f))
            for f in ("model_encoder.onnx", "model_decoder.onnx"))
needs_release = pytest.mark.skipif(not _HAVE, reason=f"no SONIC release in {DEFAULT_SONIC_DIR}")


@pytest.fixture(scope="module")
def policy():
  if not _HAVE:
    pytest.skip(f"no SONIC release in {DEFAULT_SONIC_DIR}")
  pytest.importorskip("onnx")
  return SonicPolicy.from_onnx(DEFAULT_SONIC_DIR).eval()


def _run(path, x):
  import onnx
  from onnx.reference import ReferenceEvaluator
  sess = ReferenceEvaluator(onnx.load(path))
  return sess.run(None, {"obs_dict": x.astype(np.float32)})[0]


@needs_release
def test_encoder_matches_onnx(policy):
  g = np.random.default_rng(0)
  for trial in range(3):
    pos = g.uniform(-1.5, 1.5, (1, 10, 29)).astype(np.float32)
    vel = g.normal(0, 2, (1, 10, 29)).astype(np.float32)
    anchor = g.uniform(-1, 1, (1, 10, 6)).astype(np.float32)
    x = g.normal(0, 1, (1, 1762)).astype(np.float32)   # other modes: noise
    x[0, 0], x[0, 1:4] = 0.0, 0.0
    x[0, 4:294] = pos.reshape(-1)
    x[0, 294:584] = vel.reshape(-1)
    x[0, 601:661] = anchor.reshape(-1)
    want = _run(os.path.join(DEFAULT_SONIC_DIR, "model_encoder.onnx"), x)
    with torch.no_grad():
      got = policy.encode(torch.from_numpy(pos), torch.from_numpy(vel),
                          torch.from_numpy(anchor)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)


@needs_release
def test_decoder_matches_onnx(policy):
  g = np.random.default_rng(1)
  x = g.normal(0, 1, (1, 994)).astype(np.float32)
  want = _run(os.path.join(DEFAULT_SONIC_DIR, "model_decoder.onnx"), x)
  with torch.no_grad():
    got = policy.decode(torch.from_numpy(x)).numpy()
  np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_action_scale_matches_deploy():
  s = dict(zip(SONIC_JOINTS, sonic_action_scale(SONIC_JOINTS)))
  # gear_sonic_deploy policy_parameters.hpp g1_action_scale
  assert s["left_hip_pitch_joint"] == pytest.approx(0.25 * 139.0 / (0.025101925 * (20 * 3.1415926535) ** 2))
  assert s["left_hip_yaw_joint"] == pytest.approx(0.25 * 88.0 / (0.010177520 * (20 * 3.1415926535) ** 2))
  assert s["left_ankle_pitch_joint"] == pytest.approx(0.25 * 25.0 / (0.003609725 * (20 * 3.1415926535) ** 2))
  assert s["right_wrist_yaw_joint"] == pytest.approx(0.25 * 5.0 / (0.00425 * (20 * 3.1415926535) ** 2))
  assert s["waist_pitch_joint"] == s["left_ankle_roll_joint"]


def test_forward_difference_matches_motion_lib():
  # gear_sonic/utils/motion_lib/torch_humanoid_batch.py: dof_vel = diff / dt,
  # then the second-to-last difference repeated for the last frame
  p = torch.randn(9, 4).cumsum(0)
  d = (p[1:] - p[:-1]) * 50.0
  want = torch.cat([d, d[-2:-1]], 0)
  got = forward_difference(p, 50.0)
  assert got.shape == p.shape
  assert torch.allclose(got, want)


def test_hip_swap_is_idempotent_and_local():
  safety_sb3 = pytest.importorskip("safety_sb3")  # noqa: F841 (the registry imports it)
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as gc
  from robot_safety_sandbox.registry import spec
  build = spec("g1_recover_lafan").cfg_builder
  shared = [(tuple(a.target_names_expr), a.stiffness) for a in gc.G1_ARTICULATION.actuators]
  cfg = build(play=False)
  use_sonic_hip_pitch(cfg)
  use_sonic_hip_pitch(cfg)
  hip = [a for a in cfg.scene.entities["robot"].articulation.actuators
         if ".*_hip_pitch_joint" in a.target_names_expr]
  assert len(hip) == 1 and hip[0].target_names_expr == (".*_hip_pitch_joint",)
  assert hip[0].stiffness == pytest.approx(0.025101925 * (20 * 3.1415926535) ** 2)
  assert hip[0].effort_limit == 139.0
  assert [(tuple(a.target_names_expr), a.stiffness)
          for a in gc.G1_ARTICULATION.actuators] == shared
  fresh = [a for a in build(play=False).scene.entities["robot"].articulation.actuators
           if ".*_hip_pitch_joint" in a.target_names_expr]
  assert fresh[0].stiffness == pytest.approx(0.010177520 * (20 * 3.1415926535) ** 2, rel=1e-3)
