"""Write the release of the three G1 thermal safety filters (a folder to upload
as a private Hugging Face model repo).

    python examples/export_safety_filters.py --out runs/release/g1-thermal-safety-filters
    # then, once logged in (huggingface-cli login):
    huggingface-cli upload <owner>/g1-thermal-safety-filters \
        runs/release/g1-thermal-safety-filters . --repo-type model --private

Per world (``thermal70``: the 70 C margin world; ``thermal_cbf``: the barrier
world) and SHAC seed, the frozen fallback run (final_model.zip = the SHAC
policy, its world model and margin head; tensornormalize.pt; config.yaml;
g1_sysid.json) and its calibrated tubes:

  thermal70/s<i>/tubes/joint.json       runs/shac_thermal70/wmreport/calib     (learned WM)
  thermal70/s<i>/tubes/separate.json    runs/shac_thermal70/hybrid_step0/calib/hybrid
  thermal_cbf/s<i>/tubes/fall_only.json runs/shac_thermal_cbf/wmreport/calib

Each tube's calibration record (its push.json) must name THIS run's zip and be
noise-free with the box clamp, as filters/wm_tube.load_wm_tube requires; the
record is copied into the run's ``manifest.json`` with the sha256 of every
file, which is what load_wm_tube checks on a machine without the calibration
runs. Absolute paths are dropped (the sysID repo path, the tubes' calib path).
``layouts/<world>.json`` are the fallback_layout() references
(configs/track_squat/fallback_layout_<world>.json, from
``examples/smoke_wm_tube.py --dump-layout --world <world>``).

No torch, no mjlab: copies, hashes and JSON only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT = "g1-thermal-safety-filter/1"     # = filters/wm_tube.MANIFEST_FORMAT
RUN_FILES = ("final_model.zip", "tensornormalize.pt", "config.yaml", "g1_sysid.json")
WORLDS = {
  "thermal70": {
    "run": "runs/shac_thermal70/shac_s{s}/g1_recover_thermal70_safetyshac1p",
    "tag": "thermal70_s{s}",
    "tubes": {
      "joint": ("runs/shac_thermal70/wmreport/calib/thermal70_s{s}/min_tube.json", "nn"),
      "separate": ("runs/shac_thermal70/hybrid_step0/calib/hybrid/thermal70_s{s}/min_tube.json",
                   "hybrid")}},
  "thermal_cbf": {
    "run": "runs/shac_thermal_cbf/shac_s{s}/g1_recover_thermal_cbf_safetyshac1p",
    "tag": "thermalcbf_s{s}",
    "tubes": {
      "fall_only": ("runs/shac_thermal_cbf/wmreport/calib/thermalcbf_s{s}/min_tube.json", "nn")}},
}
LAYOUT = "configs/track_squat/fallback_layout_{world}.json"


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--out", default=os.path.join(_ZOO, "runs", "release",
                                               "g1-thermal-safety-filters"))
  p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  p.add_argument("--repo-id", default="<owner>/g1-thermal-safety-filters",
                 help="only written into the README's snippets")
  p.add_argument("--force", action="store_true", help="replace an existing --out")
  return p.parse_args(argv)


def sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _rel(path: str) -> str:
  p = os.path.realpath(path)
  z = os.path.realpath(_ZOO)
  return os.path.relpath(p, z) if p.startswith(z + os.sep) else os.path.basename(p)


def _git() -> dict:
  def run(*args):
    try:
      return subprocess.run(["git", "-C", _ZOO, *args], capture_output=True, text=True,
                            timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
      return None
  return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "HEAD"),
          "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _dump(obj, path: str) -> None:
  with open(path, "w") as f:
    json.dump(obj, f, indent=1)
    f.write("\n")


def export_tube(tube_src: str, wm_thermal: str, zip_src: str, tag: str, dst: str) -> dict:
  """Copy one tube (its absolute calib path made relative) after checking its
  calibration record; returns the manifest entry."""
  with open(tube_src) as f:
    tube = json.load(f)
  if tube.get("tag") != tag:
    raise SystemExit(f"{tube_src}: tag {tube.get('tag')!r}, expected {tag!r}")
  mode = str(tube.get("wm_thermal", "nn"))
  if mode != wm_thermal:
    raise SystemExit(f"{tube_src}: world model {mode!r}, expected {wm_thermal!r}")
  push_json = os.path.join(os.path.dirname(str(tube.get("calib", ""))), "push.json")
  with open(push_json) as f:
    pm = json.load(f).get("meta", {})
  if os.path.realpath(str(pm.get("policy"))) != os.path.realpath(zip_src):
    raise SystemExit(f"{tube_src}: calibrated {pm.get('policy')}, not {zip_src}")
  if pm.get("obs_noise") is not False or pm.get("pi_obs_clamp") != "box":
    raise SystemExit(f"{push_json}: obs_noise {pm.get('obs_noise')}, pi_obs_clamp "
                     f"{pm.get('pi_obs_clamp')}")
  thermal = (pm.get("wmrollout") or {}).get("thermal")
  rec_mode = "nn" if not thermal else thermal.get("mode", "nn")
  if rec_mode != wm_thermal:
    raise SystemExit(f"{push_json}: ran the {rec_mode!r} world model, the tube says {wm_thermal!r}")
  tube = dict(tube)
  tube["calib"] = _rel(str(tube.get("calib", "")))
  _dump(tube, dst)
  return {"sha256": sha256(dst), "wm_thermal": wm_thermal,
          "calibration": {"policy_sha256": sha256(zip_src), "tag": pm.get("tag"),
                          "task": pm.get("task"), "obs_noise": pm.get("obs_noise"),
                          "pi_obs_clamp": pm.get("pi_obs_clamp"), "created": pm.get("created"),
                          "thermal": thermal if wm_thermal != "nn" else None,
                          "calib_run": _rel(push_json), "episodes": tube.get("episodes"),
                          "min_eps": tube.get("min_eps")}}


def export_run(world: str, s: int, out: str, git: dict) -> dict:
  w = WORLDS[world]
  src = os.path.join(_ZOO, w["run"].format(s=s))
  tag = w["tag"].format(s=s)
  dst = os.path.join(out, world, f"s{s}")
  os.makedirs(os.path.join(dst, "tubes"), exist_ok=True)
  files = {}
  for name in RUN_FILES:
    p = os.path.join(src, name)
    if not os.path.isfile(p):
      raise SystemExit(f"missing {p}")
    q = os.path.join(dst, name)
    if name == "g1_sysid.json":
      with open(p) as f:
        rec = json.load(f)
      if "sysid_dir" in rec:
        rec["sysid_dir"] = os.path.basename(str(rec["sysid_dir"]).rstrip("/"))
      _dump(rec, q)
    else:
      shutil.copy2(p, q)
    files[name] = sha256(q)
  if files["final_model.zip"] != sha256(os.path.join(src, "final_model.zip")):
    raise SystemExit(f"{dst}: the copied zip differs from {src}")
  tubes = {}
  for name, (tpath, mode) in w["tubes"].items():
    rel = f"tubes/{name}.json"
    tubes[rel] = export_tube(os.path.join(_ZOO, tpath.format(s=s)), mode,
                             os.path.join(src, "final_model.zip"), tag, os.path.join(dst, rel))
  with open(os.path.join(src, "config.yaml")) as f:
    task = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("task:")), None)
  man = {"format": FORMAT, "tag": tag, "world": world, "seed": s, "task": task, "files": files,
         "tubes": tubes, "source": {"run": _rel(src), "code": git},
         "exported": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
  _dump(man, os.path.join(dst, "manifest.json"))
  print(f"[export] {world} s{s}: {tag}, tubes {sorted(tubes)}")
  return man


README = """---
license: other
tags:
- robotics
- safety-filter
- world-model
- unitree-g1
- mjlab
---

# G1 thermal safety filters

Three world-model safety filters for the Unitree G1 (sysID robot with motor
heat, MuJoCo Warp / mjlab), each with its frozen SHAC fallback policy and its
calibrated conformal bound. At every 50 Hz step a filter keeps the task
policy's action or hands control to the fallback: it imagines "the proposal for
one step, then the fallback" for H seconds in a learned world model and engages
when the worst predicted safety margin comes within the calibrated bound eps of
zero. Safety means no fall and no motor winding above 70 C.

| variant | world (env) | how the margin is predicted |
|---|---|---|
| `joint` | 70 C margin world: overheat at 70 C ends the episode | one network predicts motion and temperatures; its head learns min(fall, (70 - hottest winding) / 10) |
| `fall_only_cbf` | barrier world: a per-motor temperature CBF clips torque under every controller | the head learns the fall margin only; heat is the clip's job |
| `separate` | 70 C margin world | the network predicts motion; the sysID heat model computes the temperatures and the thermal margin exactly |

`joint` and `separate` share the `thermal70` checkpoints (`separate` attaches the
heat model at filter time; nothing was retrained). `fall_only_cbf` uses the
`thermal_cbf` checkpoints.

## Results (kicked SONIC tracking, 3 seeds x 1,024 kicks, same kicks per column)

Survival %, with the % of episodes that overheated in brackets.

| filter | joint | fall_only_cbf | separate |
|---|---|---|---|
| SONIC alone (no filter) | 47.6 (19.3) | 57.0 (0) | 47.6 (19.3) |
| `wmmin:0.4:0.95` | 60.2 (7.9) | 57.9 (0) | 64.1 (3.2) |
| `wmmin:1:0.95` | 63.0 (6.0) | 60.4 (0) | 65.5 (2.2) |
| `wmmin:1:0.95+dwell1` | 81.5 (4.1) | 82.6 (0) | 84.0 (1.2) |
| `wmmin:1:0.95+dwell2` | 84.2 (3.2) | 85.5 (0) | 86.2 (1.1) |
| `wmmin:1:0.95+latch` | 86.5 (2.7) | 88.2 (0) | 88.5 (0.9) |

The fallback alone survives 97.6% (thermal70 world, 0.4% overheat) and 96.4%
(barrier world). The barrier world's clip also acts under SONIC, so its column
is a different environment. The 3 s horizon is not released.

## Files

    layouts/thermal70.json, layouts/thermal_cbf.json   what the fallback reads of an env (check_binding reference)
    thermal70/s{{0,1,2}}/                                 final_model.zip, tensornormalize.pt, config.yaml,
                                                        g1_sysid.json, manifest.json, tubes/joint.json, tubes/separate.json
    thermal_cbf/s{{0,1,2}}/                               ..., tubes/fall_only.json

`manifest.json` holds the sha256 of every file and each tube's calibration
record; the loader refuses a file that does not match.

## Requirements

* mjlab at `dcdf6ce5`, MuJoCo / MuJoCo Warp 3.16.0, torch with CUDA
* `robot-safety-sandbox`, branch `{branch}` (export commit `{commit}`{dirty})
* `safety-stable-baselines` (`safety_sb3`), branch `feat/supermimic`
* `G1-robot-model-with-SysID` {sysid_version} at commit `{sysid_commit}` (the loader checks the sha256 of its thermal model files)

## Your env must provide

The filter reads the env exactly as the fallback was trained and calibrated:

* the fallback's `actor` (127-D) and `critic` (171-D) observation groups and
  the 33-D sim context (startup domain randomization), next to your policy's
  own groups;
* the sysID robot, its `MotorThermal` (winding / housing / structure) and the
  same start-temperature draw;
* the `joint_pos` action term of the fallback's world (29 joints, its scale and
  offset; in the barrier world, the temperature-barrier action term), 50 Hz
  control over 4 x 5 ms physics steps.

The simplest way is to build your task cfg on top of the fallback's world cfg
(`g1_recover_thermal70_env_cfg` or `g1_recover_thermal_cbf_env_cfg`) and add
your command, rewards, terminations and policy observation groups, as
`robot_safety_sandbox/envs/g1_track/env_cfg.py` does for a LAFAN tracking task.
`check_binding` (run by the loader) lists every mismatch.

## Use

```python
from robot_safety_sandbox.filters.pretrained import download, load_pretrained_filter
from robot_safety_sandbox.filtered_env import WMTubeFilteredEnv

root = download("{repo_id}")          # private repo: huggingface-cli login first
env.reset()
filt, bundle = load_pretrained_filter(
    env, root, "separate", seed=0,           # "joint" | "fall_only_cbf" | "separate"
    condition="wmmin:1:0.95+dwell1",         # see the results table
    proposal_bound=float(env.action_space.high.max()))
env = WMTubeFilteredEnv(env, filt)          # step it like the env; see executed_action, last_info
```

`env` is the sandbox's tensor env (`robot_safety_sandbox.make_tensor`). With an
on-policy learner (PPO) the filter is part of the environment: keep the
policy's own proposal and log-prob. `WMTubeFilteredEnv.executed_action` is what
the robot did; `metrics()` reports engagement, hand-overs and an empirical
coverage probe of the bound.

## Caveats

* The bound eps is calibrated on SONIC tracking LAFAN clips with kicks, from
  this start-state bank and start temperatures. Under another policy (a policy
  in training above all) the 95% guarantee does not carry over: watch the
  coverage probe, or recalibrate on your own rollouts
  (`examples/eval_sonic_qcbf.py` wmcalib + `examples/calibrate_wm_min_tube.py`).
* The world model's state is privileged (true joint state, foot contacts,
  temperatures): this is a simulation filter.
* `fall_only_cbf` only works in the barrier world: the clip is part of the env.
"""


def main(argv=None):
  a = _parse(argv)
  out = os.path.abspath(a.out)
  if os.path.exists(out):
    if not a.force:
      raise SystemExit(f"{out} exists (pass --force to replace it)")
    shutil.rmtree(out)
  os.makedirs(os.path.join(out, "layouts"))
  for world in WORLDS:
    src = os.path.join(_ZOO, LAYOUT.format(world=world))
    if not os.path.isfile(src):
      raise SystemExit(f"missing {src}: run examples/smoke_wm_tube.py --dump-layout --world {world}")
    shutil.copy2(src, os.path.join(out, "layouts", f"{world}.json"))
  git = _git()
  mans = [export_run(world, s, out, git) for world in WORLDS for s in a.seeds]
  sysid = {}
  with open(os.path.join(out, "thermal70", f"s{a.seeds[0]}", "g1_sysid.json")) as f:
    sysid = json.load(f)
  with open(os.path.join(out, "README.md"), "w") as f:
    f.write(README.format(branch=git.get("branch"), commit=(git.get("commit") or "?")[:10],
                          dirty=", with uncommitted changes" if git.get("dirty") else "",
                          sysid_version=sysid.get("version"), sysid_commit=sysid.get("commit"),
                          repo_id=a.repo_id))
  sums = []
  for root, _dirs, names in os.walk(out):
    for n in sorted(names):
      p = os.path.join(root, n)
      sums.append(f"{sha256(p)}  {os.path.relpath(p, out)}")
  with open(os.path.join(out, "SHA256SUMS"), "w") as f:
    f.write("\n".join(sorted(sums, key=lambda s: s.split("  ", 1)[1])) + "\n")
  size = sum(os.path.getsize(os.path.join(r, n)) for r, _d, ns in os.walk(out) for n in ns)
  print(f"[export] {len(mans)} runs -> {out} ({size / 1e6:.0f} MB); code {git}")
  if git.get("dirty"):
    print("[export] NOTE: the code has uncommitted changes; commit and push them so the "
          "README's branch has this loader", file=sys.stderr)


if __name__ == "__main__":
  main()
