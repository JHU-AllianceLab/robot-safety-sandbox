"""Push trial tables in the SONIC push benchmark's CSV schema, on this sandbox's G1.

    # his training-motion table, regenerated (directions from the robot's heading)
    python examples/make_push_table.py --preset train --out data/push_bench/train_v1
    # the WM-tube calibration table: the same generator, another seed
    python examples/make_push_table.py --preset calib --out data/push_bench/calib_v1
    # his dev-heldout protocol (his own file is data/zz_test/trials.csv)
    python examples/make_push_table.py --preset dev --out data/push_bench/dev_regen

His protocol (``tools/sonic_lafan/push_benchmark.py`` ``plan`` and
``tools/sonic_bench/heldout_table.py``): per motion, its usable frames are cut
into ``--per-motion`` equal strata and each stratum gets one seeded start frame
(``make_contexts``: ``SeedSequence([seed, motion index, stratum])``), leaving
room for ``--warm`` warm-up ticks, the scored ticks and ``--lookahead``. Every
context is crossed with the 12 push bodies (``push_bench.PUSH_SITES``), 4
directions and the ``--dvs``; each (context, body) gets one contact point,
drawn uniformly by area on the body's collision surface
(``SeedSequence([seed, context, body index, 123])``) and shared by its
directions and sizes. The force is ``m dv / (10 dt)`` (m = the robot's total
mass), 10 ticks long.

What is this sandbox's: the contact points lie on mjlab's G1 collision shapes
(capsules / spheres), and ``force_N`` uses its mass (33.341142 kg, the same as
his). Motion lengths and reference headings follow his 30 -> 50 Hz conversion
(``tools/sonic_lafan/motion.py``: linear interpolation and SLERP at t = 0.02 k)
of the CSVs in ``--csv-dir``, so with the same CSVs the start frames are his
(``tests/test_push_bench.py`` checks the dev table against his file).

``direction_frame`` (a column his files lack) says what the direction is
measured from at the onset: ``reference`` = the reference's root heading at the
onset frame (his dev table; ``world_force_N`` is then the force in the
reference's world), ``actual`` = the robot's actual heading after the warm-up
(his CPU training-motion table). ``examples/eval_push_bench.py`` reads it.

``--end-s NAME=SECONDS`` keeps a motion's contexts whose whole window, SONIC's
0.9 s preview included, ends by then (his 105 s cut of dance1_subject1):
contexts are drawn on the full clip first, then filtered, as he did.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time

import numpy as np

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _push_bench():
  """``envs/g1_recover/push_bench.py`` alone (the package import pulls in mjlab)."""
  path = os.path.join(_ZOO, "robot_safety_sandbox", "envs", "g1_recover", "push_bench.py")
  spec = importlib.util.spec_from_file_location("_push_bench", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


pb = _push_bench()

DT = 0.02
#: his three motions (training) and six dev-heldout motions
TRAIN_MOTIONS = ("walk1_subject1", "run1_subject2", "dance1_subject1")
DEV_MOTIONS = ("walk1_subject2", "run1_subject5", "dance1_subject2", "sprint1_subject2",
               "fight1_subject2", "jumps1_subject1")
PRESETS = {
  # his training-motion table (push_benchmark.plan): 24 contexts, 5 sizes, dance cut at 105 s
  "train": dict(motions=TRAIN_MOTIONS, per_motion=24, seed=2026102802, dvs=(0.5, 1.0, 1.5, 2.0, 2.5),
                after=510, lookahead=0, end_s={"dance1_subject1": 105.0}, direction_frame="actual"),
  # the WM-tube calibration table: the training generator with a fresh seed
  "calib": dict(motions=TRAIN_MOTIONS, per_motion=24, seed=2026110501, dvs=(0.5, 1.0, 1.5, 2.0, 2.5),
                after=510, lookahead=0, end_s={"dance1_subject1": 105.0}, direction_frame="actual"),
  # his dev-heldout table (heldout_table.py): 10 contexts, 3 sizes, + Gamma_T's 100-tick look-ahead
  "dev": dict(motions=DEV_MOTIONS, per_motion=10, seed=2026110101, dvs=(1.5, 2.0, 2.5),
              after=510, lookahead=100, end_s={}, direction_frame="reference"),
}


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--preset", choices=sorted(PRESETS), default=None)
  p.add_argument("--motions", nargs="+", default=None)
  p.add_argument("--per-motion", type=int, default=None)
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--dvs", type=float, nargs="+", default=None)
  p.add_argument("--warm", type=int, default=pb.WARM_TICKS)
  p.add_argument("--after", type=int, default=None, help="scored ticks after the onset")
  p.add_argument("--lookahead", type=int, default=None,
                 help="extra frames each context keeps past the scored ticks")
  p.add_argument("--end-s", nargs="*", default=None, metavar="NAME=SECONDS")
  p.add_argument("--preview", type=int, default=pb.PREVIEW_FRAMES,
                 help="SONIC's reference preview, frames (the --end-s filter)")
  p.add_argument("--direction-frame", choices=("reference", "actual"), default=None)
  p.add_argument("--csv-dir", default=os.path.join(_ZOO, "data", "lafan1_g1"))
  p.add_argument("--out", required=True)
  a = p.parse_args(argv)
  base = dict(PRESETS[a.preset]) if a.preset else {}
  for k in ("motions", "per_motion", "seed", "dvs", "after", "lookahead", "direction_frame"):
    if getattr(a, k) is not None:
      base[k] = getattr(a, k)
  if a.end_s is not None:
    base["end_s"] = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.end_s}
  need = ("motions", "per_motion", "seed", "dvs", "after", "lookahead", "direction_frame")
  miss = [k for k in need if k not in base]
  if miss:
    p.error(f"set --preset or {['--' + m.replace('_', '-') for m in miss]}")
  base.setdefault("end_s", {})
  for k, v in base.items():
    setattr(a, k, v)
  return a


# --- his motion conversion (tools/sonic_lafan/motion.py) ------------------------------------

def load_csv(csv_dir: str, name: str) -> np.ndarray:
  raw = np.loadtxt(os.path.join(csv_dir, name + ".csv"), delimiter=",")
  if raw.ndim != 2 or raw.shape[1] != 36:
    raise SystemExit(f"{name}.csv: expected rows of 36 columns")
  return raw


def frames_50hz(n_rows: int, source_fps: float = 30.0) -> int:
  """His 50 Hz frame count of an ``n_rows`` CSV (the clamped endpoint included)."""
  t_last = (n_rows - 1) / source_fps
  return int(math.floor(t_last / DT)) + 1


def reference_yaw(raw: np.ndarray, frames, source_fps: float = 30.0) -> np.ndarray:
  """The reference root's yaw at 50 Hz ``frames`` (SLERP of the CSV's xyzw)."""
  from scipy.spatial.transform import Rotation, Slerp
  t = np.arange(len(raw)) / source_fps
  times = np.minimum(np.asarray(frames, dtype=np.float64) * DT, t[-1])
  q = Slerp(t, Rotation.from_quat(raw[:, 3:7]))(times).as_quat()     # xyzw
  x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def make_contexts(lengths: dict, per_motion: int, seed: int, warm: int, horizon: int) -> list:
  """His ``make_contexts``: one seeded start per equal-duration stratum."""
  out = []
  for mi, (motion, n) in enumerate(lengths.items()):
    last = n - 1 - warm - horizon
    if last + 1 < per_motion:
      raise SystemExit(f"{motion}: too short for {per_motion} strata")
    edges = np.linspace(0, last + 1, per_motion + 1, dtype=int)
    for j in range(per_motion):
      rng = np.random.default_rng(np.random.SeedSequence([seed, mi, j]))
      start = int(rng.integers(edges[j], edges[j + 1]))
      out.append(dict(context_id=len(out), motion=motion, phase_stratum=j, start_frame=start,
                      onset_frame=start + warm, onset_reference_seconds=(start + warm) * DT))
  return out


def robot_model():
  """mjlab's G1 robot file, compiled on the CPU (no mjlab import)."""
  import mujoco
  spec = importlib.util.find_spec("mjlab")
  if spec is None or spec.origin is None:
    raise SystemExit("mjlab is not installed")
  xml = os.path.join(os.path.dirname(spec.origin), "asset_zoo", "robots", "unitree_g1", "xmls",
                     "g1.xml")
  with open(xml, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
  return mujoco.MjModel.from_xml_path(xml), xml, sha


def build_rows(a, model, contexts: list, yaw: dict) -> list:
  import mujoco
  mass = float(model.body_mass.sum())
  rows = []
  for c in contexts:
    th0 = float(yaw[c["context_id"]])
    for si, site in enumerate(pb.PUSH_SITES):
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site)
      rng = np.random.default_rng(np.random.SeedSequence([a.seed, c["context_id"], si, 123]))
      point = pb.sample_surface_points(model, bid, 1, rng)[0]
      for direction, angle in pb.DIRECTIONS.items():
        th = th0 + angle
        for dv in a.dvs:
          f = mass * dv / (pb.PUSH_TICKS * DT)
          force = f * np.array([math.cos(th), math.sin(th), 0.0])
          rows.append(dict(c, trial_id=len(rows), site=site, direction=direction, dv_mps=dv,
                           force_N=f, world_force_N=json.dumps(force.tolist()),
                           point_body=json.dumps(point.tolist()), collision_geom=-1,
                           onset_yaw_rad=th0, completed="NA",
                           direction_frame=a.direction_frame))
  return rows


def main(argv=None):
  a = _parse(argv)
  os.makedirs(a.out, exist_ok=False)
  raws = {m: load_csv(a.csv_dir, m) for m in a.motions}
  lengths = {m: frames_50hz(len(raws[m])) for m in a.motions}
  horizon = a.after + a.lookahead
  contexts = make_contexts(lengths, a.per_motion, a.seed, a.warm, horizon)
  dropped = {}
  if a.end_s:
    keep = []
    for c in contexts:
      end = a.end_s.get(c["motion"])
      last = c["start_frame"] + a.warm + horizon + a.preview
      if end is not None and last > int(round(end / DT)):
        dropped[c["motion"]] = dropped.get(c["motion"], 0) + 1
        continue
      keep.append(c)
    contexts = keep          # context ids keep the draw's numbering (his did too)
  yaw = {}
  for m in a.motions:
    cs = [c for c in contexts if c["motion"] == m]
    if cs:
      ys = reference_yaw(raws[m], [c["onset_frame"] for c in cs])
      yaw.update({c["context_id"]: float(y) for c, y in zip(cs, ys)})
  model, xml, xml_sha = robot_model()
  rows = build_rows(a, model, contexts, yaw)
  with open(os.path.join(a.out, "trials.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  csv_sha = {m: hashlib.sha256(open(os.path.join(a.csv_dir, m + ".csv"), "rb").read()).hexdigest()
             for m in a.motions}
  contract = dict(
    protocol=__doc__, preset=a.preset, seed=a.seed, motions=list(a.motions), per_motion=a.per_motion,
    sites=list(pb.PUSH_SITES), directions=list(pb.DIRECTIONS), dvs=list(a.dvs),
    warm_ticks=a.warm, push_ticks=pb.PUSH_TICKS, scored_ticks=a.after, lookahead=a.lookahead,
    end_s=a.end_s, preview_frames=a.preview, dropped_by_end=dropped,
    direction_frame=a.direction_frame, total_mass_kg=float(model.body_mass.sum()),
    robot_xml=xml, robot_xml_sha256=xml_sha, motion_lengths_50hz=lengths, csv_sha256=csv_sha,
    contexts=contexts, trials=len(rows), created=time.strftime("%Y-%m-%d %H:%M:%S"))
  with open(os.path.join(a.out, "contract.json"), "w") as f:
    json.dump(contract, f, indent=1)
  per = {m: sum(c["motion"] == m for c in contexts) for m in a.motions}
  print(f"[table] {os.path.join(a.out, 'trials.csv')}: {len(contexts)} contexts {per}, "
        f"{len(rows)} pushes ({a.direction_frame} frame); dropped by --end-s {dropped}")


if __name__ == "__main__":
  main()
