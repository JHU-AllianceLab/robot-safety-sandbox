"""Build g1_recover start states from retargeted mocap (LAFAN1 on the G1).

    python examples/build_g1_recover_mocap_states.py \
        --csv-dir data/lafan1_g1 --out data/g1_recover_states_lafan1.pt
    python examples/build_g1_recover_mocap_states.py --library   # -> data/g1_lafan1_clips.pt

Input: Unitree's G1 retargeting of LAFAN1 (huggingface.co/datasets/
lvhaidong/LAFAN1_Retargeting_Dataset, the ``g1/*.csv`` files): 30 fps rows of
root xyz + quaternion (xyzw) + 29 joint angles in Unitree's joint order -- the
format mjlab's ``scripts/csv_to_npz.py`` reads, and its ``MotionLoader`` does
the 30 -> 50 fps resampling and finite-difference velocities here too.

KINEMATIC rows. Every --stride-th frame becomes a start state as it stands:
pose, root twist and joint velocities from the clip, heading frame, in the
bank schema of ``envs/g1_recover/states.py``. No controller produced these
states, so two things are set by convention rather than read off a rollout:

* GROUND. The retargeting is kinematic, so soles sit a few mm into or above
  the floor. A frame whose lowest sole point is below --snap m is moved so it
  sits SOLE_CLEARANCE above the floor (the synthetic rows' rule); a higher one
  is a flight phase (a jump, a sprint stride) and keeps its height.
* LAST ACTION. The PD target is the current pose: action = (q - offset) /
  scale in the action term's own joint order, clamped to the bridge's
  +-action_bound -- the nearest target the policy's action space can express.
  Mocap poses leave that box mostly at the wrists (scale 0.075 rad: +-0.23 rad
  of reach; the walker bank saturates them too), and at deep knee bends and
  raised arms on a few percent of rows.

FILTER. Frames that are already a fall under g1_stabilize's g are dropped:
pelvis below Z_MIN + --z-margin, tilt over --max-tilt, or any non-foot body
origin within --nonfoot-z of the floor (kneeling, crawling, hand plants, lying
-- what the illegal-contact term would fire on). fallAndGetUp clips keep their
upright stretches and the start of each fall.

Each row also carries ``clip`` (index into meta["clips"]), ``category`` (index
into meta["categories"]: the clip name's prefix, walk / run / dance / ...),
``t`` (s into the clip) and ``airborne``. Rows are ``kind = KIND_MOCAP``
(``envs/g1_recover/states.py``), which the spawn's ``mocap`` source draws.

CLIP LIBRARY (``--library``). The same rows at EVERY 50 fps frame (stride 1),
for the clip replay (``envs/g1_recover/playback.py``): a handover harvester
spawns on frame r and drives the robot with the PD targets (``action``) of
frames r+1, r+2, ... So no frame is dropped -- a replay must be able to run
THROUGH a frame the filter rejects as a start (a jump's landing, a fall's
first instant) -- and every clip's frames stay contiguous, in clip then time
order. Three more columns say what each row may be used for:

* ``frame``    the frame index within its clip (t = frame / 50);
* ``next_ok``  the frames that follow it within the same clip: a replay of
               rows r+1 .. r+k stays inside the clip iff next_ok[r] >= k;
* ``start_ok`` it passes the FILTER above AND at least --min-follow frames
               follow it (default 25 = 0.5 s, room for a push onset and a
               handover delay), so the spawn may start an episode there; the
               spawn never draws a row with start_ok False.

Without --library the stride bank is written exactly as before (no extra
columns, filtered frames dropped).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_MOCAP, SoleGeometry, save_bank, speed_of, tilt_deg,
  to_heading_frame, write_states)
from robot_safety_sandbox.envs.g1_stabilize.env_cfg import Z_MIN  # noqa: E402
from robot_safety_sandbox.eval import build_eval_env  # noqa: E402

SOLE_CLEARANCE = 0.002
#: Unitree's G1 CSV joint order (the README's), as mjlab's csv_to_npz uses it.
CSV_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint")


def _parse():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--csv-dir", default=os.path.join(_ZOO, "data", "lafan1_g1"))
  p.add_argument("--out", default=None,
                 help="default data/g1_recover_states_lafan1.pt, or "
                      "data/g1_lafan1_clips.pt with --library")
  p.add_argument("--library", action="store_true",
                 help="the clip LIBRARY: every 50 fps frame, none dropped, plus "
                      "frame / next_ok / start_ok (module docstring)")
  p.add_argument("--min-follow", type=int, default=25,
                 help="library: frames that must follow a row within its clip "
                      "for start_ok (25 = 0.5 s)")
  p.add_argument("--stride", type=int, default=None,
                 help="keep every n-th 50 fps frame (default 5 -> 10 Hz; the "
                      "library is always stride 1)")
  p.add_argument("--input-fps", type=float, default=30.0)
  p.add_argument("--snap", type=float, default=0.03,
                 help="lowest sole below this (m) -> put the robot on the floor")
  p.add_argument("--z-margin", type=float, default=0.05,
                 help="drop frames with the pelvis below Z_MIN + this (m)")
  p.add_argument("--max-tilt", type=float, default=45.0,
                 help="drop frames tilted more than this (deg); g fires at 50")
  p.add_argument("--nonfoot-z", type=float, default=0.15,
                 help="drop frames with a non-foot body origin below this (m)")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  if args.library and args.stride not in (None, 1):
    p.error("--library keeps every frame; --stride must be 1 (or unset)")
  if args.stride is None:
    args.stride = 1 if args.library else 5
  if args.out is None:
    args.out = os.path.join(_ZOO, "data", "g1_lafan1_clips.pt" if args.library
                            else "g1_recover_states_lafan1.pt")
  return args


def _clip_name(path: str) -> tuple[str, str]:
  name = os.path.splitext(os.path.basename(path))[0]
  m = re.match(r"([A-Za-z]+)\d*_subject\d+", name)
  return name, (m.group(1) if m else name)


def load_clips(args) -> tuple[dict, list[str], list[str], list[int]]:
  """Every clip, resampled to 50 fps and strided, world frame, CSV order;
  plus each clip's 50 fps frame count. ``frame`` is the 50 fps index of each
  kept row within its clip; rows stay in clip, then time order."""
  from mjlab.scripts.csv_to_npz import MotionLoader
  paths = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
  if not paths:
    raise SystemExit(f"no *.csv under {args.csv_dir}")
  names, cats, lengths = [], [], []
  cols: dict[str, list] = {k: [] for k in ("pos", "quat", "lin", "ang", "q",
                                           "qd", "clip", "category", "t",
                                           "frame")}
  for i, path in enumerate(paths):
    name, cat = _clip_name(path)
    if cat not in cats:
      cats.append(cat)
    names.append(name)
    ml = MotionLoader(path, int(args.input_fps), 50, "cpu")
    keep = torch.arange(0, ml.output_frames, args.stride)
    cols["pos"].append(ml.motion_base_poss[keep])
    cols["quat"].append(ml.motion_base_rots[keep])
    cols["lin"].append(ml.motion_base_lin_vels[keep])
    cols["ang"].append(ml.motion_base_ang_vels[keep])
    cols["q"].append(ml.motion_dof_poss[keep])
    cols["qd"].append(ml.motion_dof_vels[keep])
    cols["clip"].append(torch.full((keep.numel(),), i, dtype=torch.long))
    cols["category"].append(torch.full((keep.numel(),), cats.index(cat),
                                       dtype=torch.long))
    cols["t"].append(keep.float() / 50.0)
    cols["frame"].append(keep.long())
    lengths.append(int(ml.output_frames))
  return {k: torch.cat(v) for k, v in cols.items()}, names, cats, lengths


def main():
  args = _parse()
  t0 = time.time()
  raw, clip_names, categories, clip_frames = load_clips(args)
  M = raw["clip"].numel()
  print(f"[mocap] {len(clip_names)} clips, {M} frames after stride "
        f"{args.stride} ({M * args.stride / 50 / 3600:.2f} h of motion), "
        f"{time.time() - t0:.0f}s")

  env = build_eval_env("g1_recover", args.num_envs, args.device, play=True)
  mj, dev, N = env.mj, env.device, env.num_envs
  robot = mj.scene["robot"]
  csv_idx = torch.as_tensor(robot.find_joints(CSV_JOINTS, preserve_order=True)[0],
                            device=dev)
  term = mj.action_manager.get_term("joint_pos")
  act_ids = term.target_ids.to(dev)
  scale = term.scale if isinstance(term.scale, float) else term.scale[0]
  offset = term.offset if isinstance(term.offset, float) else term.offset[0]
  body_names = robot.body_names
  nonfoot = torch.as_tensor([i for i, b in enumerate(body_names)
                             if "ankle" not in b], device=dev)
  sole = SoleGeometry(mj)
  nj = robot.data.default_joint_pos.shape[1]

  quat_h, lin_h, ang_h = to_heading_frame(raw["quat"].to(dev), raw["lin"].to(dev),
                                          raw["ang"].to(dev))
  jpos = torch.zeros(M, nj, device=dev)
  jvel = torch.zeros(M, nj, device=dev)
  jpos[:, csv_idx] = raw["q"].to(dev)
  jvel[:, csv_idx] = raw["qd"].to(dev)
  rows = {"root_z": raw["pos"][:, 2].to(dev), "root_quat_h": quat_h,
          "lin_vel_h": lin_h, "ang_vel_h": ang_h, "joint_pos": jpos,
          "joint_vel": jvel}

  low_nonfoot = torch.zeros(M, device=dev)
  airborne = torch.zeros(M, dtype=torch.bool, device=dev)
  ids_all = torch.arange(N, device=dev)
  zero_yaw = torch.zeros(N, device=dev)
  zero_xy = torch.zeros(N, 2, device=dev)
  for b0 in range(0, M, N):
    k = min(N, M - b0)
    pad = torch.arange(N, device=dev) % k
    batch = {f: rows[f][b0:b0 + k][pad] for f in rows}
    write_states(mj, ids_all, batch, zero_yaw, zero_xy)
    mj.sim.forward()
    lowest = sole.min_z()[:k]
    snap = lowest < args.snap
    rows["root_z"][b0:b0 + k] += torch.where(snap, SOLE_CLEARANCE - lowest, 0.0)
    airborne[b0:b0 + k] = ~snap
    batch["root_z"] = rows["root_z"][b0:b0 + k][pad]
    write_states(mj, ids_all, batch, zero_yaw, zero_xy)
    mj.sim.forward()
    low_nonfoot[b0:b0 + k] = robot.data.body_link_pos_w[:k, nonfoot, 2].amin(1)

  tilt = tilt_deg(quat_h)
  passes = ((rows["root_z"] >= Z_MIN + args.z_margin) & (tilt <= args.max_tilt)
            & (low_nonfoot >= args.nonfoot_z))
  print(f"[mocap] {'passed the filter' if args.library else 'kept'} "
        f"{int(passes.sum())}/{M}: {'failed' if args.library else 'dropped'} "
        f"{int((rows['root_z'] < Z_MIN + args.z_margin).sum())} low pelvis, "
        f"{int((tilt > args.max_tilt).sum())} tilted, "
        f"{int((low_nonfoot < args.nonfoot_z).sum())} non-foot near the floor "
        f"(overlapping); {int((airborne & passes).sum())} kept rows airborne")
  # The stride bank drops the filtered frames; the library keeps every frame
  # (a replay runs through them) and marks the spawnable ones with start_ok.
  keep = torch.ones_like(passes) if args.library else passes

  bound = float(env.bridge.action_bound)
  action_raw = (jpos[:, act_ids] - offset) / scale
  action = action_raw.clamp(-bound, bound)
  clamped = (action_raw.abs() > bound)[passes]
  print(f"[mocap] last action clamped to +-{bound:g}: {100 * clamped.any(1).float().mean():.1f}% "
        f"of rows, {100 * clamped[:, :15].any(1).float().mean():.1f}% on legs + waist")
  k = int(keep.sum())
  bank = {
    "root_z": rows["root_z"][keep], "root_quat_h": quat_h[keep],
    "lin_vel_h": lin_h[keep], "ang_vel_h": ang_h[keep],
    "joint_pos": jpos[keep], "joint_vel": jvel[keep], "action": action[keep],
    "command": torch.zeros(k, 3, device=dev),
    "phase_step": torch.zeros(k, dtype=torch.long, device=dev),
    "kind": torch.full((k,), KIND_MOCAP, dtype=torch.long, device=dev),
    "fell": torch.zeros(k, dtype=torch.bool, device=dev),
    "time_to_fall": torch.full((k,), float("nan"), device=dev),
    "push_severity": torch.zeros(k, device=dev),
    "difficulty": torch.zeros(k, device=dev),
    "speed": speed_of(lin_h[keep]),
    "env_id": torch.full((k,), -1, dtype=torch.long, device=dev),
    "clip": raw["clip"].to(dev)[keep], "category": raw["category"].to(dev)[keep],
    "t": raw["t"].to(dev)[keep], "airborne": airborne[keep],
  }
  if args.library:
    # every frame of every clip, in clip then time order (load_clips), so the
    # frames after row r within its clip are rows r+1 .. r+next_ok[r]
    frame = raw["frame"].to(dev)
    length = torch.as_tensor(clip_frames, dtype=torch.long, device=dev)[bank["clip"]]
    bank["frame"] = frame
    bank["next_ok"] = length - 1 - frame
    bank["start_ok"] = passes & (bank["next_ok"] >= args.min_follow)
    print(f"[mocap] library: {int(bank['start_ok'].sum())}/{M} rows start_ok "
          f"(filter passed and >= {args.min_follow} frames follow in the clip)")
  assert all(f in bank for f in FIELDS)
  per_cat = {c: int((bank["category"] == i).sum()) for i, c in enumerate(categories)}
  bank = {f: v.cpu() for f, v in bank.items()}
  bank["meta"] = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "source": "LAFAN1, Unitree G1 retargeting (lvhaidong/LAFAN1_Retargeting_Dataset)",
    "csv_dir": os.path.abspath(args.csv_dir), "clips": clip_names,
    "categories": categories, "rows_per_category": per_cat,
    "stride": args.stride, "fps": 50, "snap": args.snap,
    "z_margin": args.z_margin, "max_tilt": args.max_tilt,
    "nonfoot_z": args.nonfoot_z, "frames_before_filter": M,
    "action": f"PD target = current pose, clamped to +-{bound:g}",
  }
  if args.library:
    bank["meta"].update(
      library=True, min_follow=args.min_follow, clip_frames=list(clip_frames),
      start_ok_rows=int(bank["start_ok"].sum()),
      start_ok_per_category={c: int(((bank["category"] == i) & bank["start_ok"]).sum())
                             for i, c in enumerate(categories)})
  save_bank(bank, args.out)
  print(f"[mocap] rows per category: {per_cat}")
  print(f"[mocap] wrote {k} rows -> {args.out} ({time.time() - t0:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
