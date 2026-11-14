"""Build the AMP expert set: phi of every frame of the TRAIN clips, and its 50 fps pairs.

    python examples/build_amp_expert.py                       # the defaults below
    python examples/build_amp_expert.py --library data/g1_lafan1_clips_train.pt \
        --split data/lafan1_split.json --out data/g1_amp_expert_lafan_train_v1.pt

Only the train part of the LAFAN split may enter (the task's target segment is a
held-out clip): the builder refuses a library whose split meta is not "train" or
that holds any held-out clip.

phi is ``robot_safety_sandbox.amp.features`` -- the function the env's ``amp``
observation term calls -- on:
  * root_h by forward kinematics of the IDENTIFIED robot (envs/assets/g1_sysid
    ``get_spec``): pelvis z minus the lower ankle-roll link z, from the row's
    yaw-free root orientation and its joints, exactly what the env term measures;
  * the joints clamped into that robot's ranges (the mocap's waist pitch exceeds
    the measured limit on some frames; an angle the robot cannot reach would be a
    free tell for the discriminator).

A pair (i, i+1) needs ``next_ok[i] >= 1`` and the same clip at consecutive frames:
the library is 50 fps, the env step is 0.02 s, so a pair is one env transition.

Output (torch.save dict): phi [R, 68], pair_idx [P], mean / std [68] (std floored
at 0.05), meta (version, feature names, joint names and ranges, clamp count,
clips, library sha256).

Runs on the CPU (MuJoCo forward kinematics, ~1 min); loads the modules it needs by
file path, so it never imports the sandbox package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_FLOOR = 0.05
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _load(rel: str, name: str):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ZOO, rel))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def _sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def check_split(lib: dict, split: dict) -> list[str]:
  """The train clip names present in ``lib``; raises unless the library is the
  train part and holds no held-out clip."""
  meta = lib["meta"]
  part = (meta.get("split") or {}).get("part")
  if part != "train":
    raise SystemExit(f"[amp-expert] library split part is {part!r}, need 'train'")
  names = list(meta["clips"])
  present = sorted({names[int(c)] for c in torch.unique(lib["clip"]).tolist()})
  held = sorted(set(present) & set(split["test"]))
  if held:
    raise SystemExit(f"[amp-expert] held-out clips in the library: {held}")
  missing = sorted(set(split["train"]) - set(present))
  if missing:
    print(f"[amp-expert] WARNING: train clips absent from the library: {missing}")
  return present


def pair_index(lib: dict) -> torch.Tensor:
  """Rows i whose successor i+1 is the next 50 fps frame of the same clip."""
  clip, frame, nxt = lib["clip"], lib["frame"], lib["next_ok"]
  ok = (nxt[:-1] >= 1) & (clip[1:] == clip[:-1]) & (frame[1:] == frame[:-1] + 1)
  return torch.nonzero(ok).squeeze(1)


def root_heights(lib: dict, model, joint_names: list[str], lo, hi) -> torch.Tensor:
  """pelvis z - min ankle-roll link z of every row (forward kinematics)."""
  import mujoco
  d = mujoco.MjData(model)
  jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
  adr = np.array([model.jnt_qposadr[j] for j in jid])
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
  assert min([pelvis] + feet) >= 0
  quat = lib["root_quat_h"].double().numpy()
  qpos = np.clip(lib["joint_pos"].double().numpy(), lo, hi)
  out = np.empty(len(quat))
  for i in range(len(quat)):
    d.qpos[:3] = 0.0
    d.qpos[3:7] = quat[i]
    d.qpos[adr] = qpos[i]
    mujoco.mj_kinematics(model, d)
    out[i] = d.xpos[pelvis, 2] - min(d.xpos[feet[0], 2], d.xpos[feet[1], 2])
  return torch.as_tensor(out, dtype=torch.float32)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--library", default="data/g1_lafan1_clips_train.pt")
  ap.add_argument("--split", default="data/lafan1_split.json")
  ap.add_argument("--out", default="data/g1_amp_expert_lafan_train_v1.pt")
  args = ap.parse_args(argv)
  feats = _load("robot_safety_sandbox/amp/features.py", "_amp_features")
  sysid = _load("robot_safety_sandbox/envs/assets/g1_sysid.py", "_amp_g1_sysid")

  t0 = time.time()
  lib = torch.load(args.library, map_location="cpu", weights_only=False)
  split = json.load(open(args.split))
  present = check_split(lib, split)
  print(f"[amp-expert] {len(lib['root_z'])} rows, {len(present)} train clips "
        f"({time.time() - t0:.0f} s)")

  import mujoco
  model = sysid.get_spec().compile()
  # library joints are in mjlab entity order = the model's hinge order
  joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  assert len(joint_names) == feats.N_JOINTS, joint_names
  rng = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                  for n in joint_names])
  root_h = root_heights(lib, model, joint_names, rng[:, 0], rng[:, 1])
  print(f"[amp-expert] FK root_h: min {root_h.min():.3f} median {root_h.median():.3f} "
        f"max {root_h.max():.3f} m ({time.time() - t0:.0f} s)")

  phi, clamped = feats.features_from_library(lib, root_h, torch.as_tensor(rng))
  idx = pair_index(lib)
  mean = phi.mean(0)
  std = phi.std(0).clamp_min(STD_FLOOR)
  names = (["root_h"] + [f"grav_{a}" for a in "xyz"] + [f"lin_vel_b_{a}" for a in "xyz"]
           + [f"ang_vel_b_{a}" for a in "xyz"] + [f"q_{n}" for n in joint_names]
           + [f"qd_{n}" for n in joint_names])
  assert len(names) == feats.FEATURE_DIM and torch.isfinite(phi).all()
  meta = {"version": feats.FEATURE_VERSION, "feature_names": names, "joint_names": joint_names,
          "joint_range": rng.tolist(), "clamped_rows": clamped, "clips": present,
          "rows": int(phi.shape[0]), "pairs": int(idx.numel()), "std_floor": STD_FLOOR,
          "library": os.path.abspath(args.library), "library_sha256": _sha256(args.library),
          "split_json": os.path.abspath(args.split), "foot_bodies": list(FOOT_BODIES)}
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  torch.save({"phi": phi, "pair_idx": idx, "mean": mean, "std": std, "meta": meta}, args.out)
  print(f"[amp-expert] wrote {args.out}: {phi.shape[0]} rows, {idx.numel()} pairs, "
        f"{clamped} rows with a joint clamped into the robot's range ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
