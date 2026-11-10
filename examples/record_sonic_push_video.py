"""Videos of SONIC and the fallbacks under the push suite's kick, by kick size.

    python examples/record_sonic_push_video.py --sonic-dir $HOME/gear_sonic_hf --hip mjlab \
        --policy ppo_s0=runs/shac_vs_ppo_sonic/ppo_s0/g1_recover/final_model.zip \
        --policy shac_s0=runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip

Per |dv| bin (--bins, default 1.5-2.5 2.5-3.5 3.5-5 m/s), --per-bin episodes of
the push suite of ``eval_sonic_qcbf.py``: a held-out clip frame, the kick at
U(--onset-s) s with |dv| uniform in the bin at a uniform angle in the heading
frame, survival counted for --window-s after it (the env's fell_over 70 deg /
illegal_contact end an episode). Every episode runs once under SONIC alone
and once under each --policy alone (the suite's ``sonic`` and ``fallback``
conditions, through its ``Evaluator``: the same step order, kick and
termination rule), in ONE env, so each controller starts from the same spawn
with the same startup domain randomization.

Writes, in --out-dir:

    <bin>/ep<NN>_<category>_<dv>ms_<direction>.mp4   the controllers side by side
    reel_<bin>.mp4, reel.mp4                          the episodes end to end
    summary.txt, summary.json                         the outcome per episode and controller
    episodes_bank.pt                                  the spawn rows (the env's dataset)

A banner names the clip, the kick and the hip actuator. Each panel shows its
controller from the spawn until it survives the window or falls, then holds its
last frame under the outcome; the kick flashes for 0.3 s. Needs a GL context:
export MUJOCO_GL=egl (``slurm_jobs.sbatch`` does).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_EX = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_EX)
sys.path.insert(0, _ZOO)
sys.path.insert(0, _EX)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import qcbf_eval_lib as ql  # noqa: E402
from eval_sonic_qcbf import (  # noqa: E402
  DT, Evaluator, _seed, _spawn_term, _steps, episode_columns, write_episode_bank)
from harvest_g1_sonic_trajectories import make_env  # noqa: E402

from robot_safety_sandbox.envs.g1_recover import bad_reference as br  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SonicPolicy, SonicReference, SonicTracker)
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import KIND_MOCAP, load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402

#: The kick's direction in the robot's heading frame, by 45 deg sector from
#: straight ahead (``rel`` = 0: a shove from behind) counter-clockwise.
DIRS_8 = ("forward", "forward-left", "left", "back-left", "back", "back-right", "right",
          "forward-right")
#: The hip-pitch actuator of each --hip (``sonic.use_sonic_hip_pitch``).
HIP_DESC = {"sonic": "hip pitch 7520-22 (SONIC's: Kp 99.1, 139 N m)",
            "mjlab": "hip pitch 7520-14 (sandbox: Kp 40.2, 88 N m)"}
GREEN, RED, ORANGE = (40, 170, 80), (210, 55, 55), (235, 140, 30)
KICK_FLASH_S = 0.3
BANNER_H = 64


def parse_bin(text: str) -> tuple[float, float]:
  """``"lo-hi"`` m/s -> (lo, hi), 0 <= lo < hi."""
  try:
    lo, hi = (float(x) for x in str(text).split("-"))
  except ValueError:
    raise ValueError(f"bin {text!r}: expected lo-hi in m/s, e.g. 1.5-2.5") from None
  if not 0.0 <= lo < hi:
    raise ValueError(f"bin {text!r}: needs 0 <= lo < hi")
  return lo, hi


def direction_name(rel: float) -> str:
  """The 45 deg sector of a kick at angle ``rel`` (heading frame, rad)."""
  return DIRS_8[int(round(float(rel) / (math.pi / 4))) % 8]


def learner_name(tag: str) -> str:
  t = tag.lower()
  if t.startswith("ppo"):
    return f"SafetyPPO alone ({tag})"
  if t.startswith("shac"):
    return f"SHAC alone ({tag})"
  return f"{tag} alone"


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", action="append", required=True, metavar="TAG=PATH",
                 help="a fallback, its tag and final_model.zip (repeat for more)")
  p.add_argument("--sonic-dir", default=DEFAULT_SONIC_DIR)
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"),
                 help="the held-out clip library")
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--hip", choices=("sonic", "mjlab"), default="mjlab",
                 help="hip-pitch actuator (default the sandbox G1's, the fallbacks' own)")
  p.add_argument("--bins", nargs="+", default=["1.5-2.5", "2.5-3.5", "3.5-5"],
                 help="|dv| bins, m/s")
  p.add_argument("--per-bin", type=int, default=4, help="episodes per bin")
  p.add_argument("--onset-s", type=float, nargs=2, default=[0.5, 1.5])
  p.add_argument("--window-s", type=float, default=4.0, help="survival window after the kick")
  p.add_argument("--width", type=int, default=640, help="panel width, px")
  p.add_argument("--height", type=int, default=480, help="panel height, px")
  p.add_argument("--fps", type=int, default=50, help="50 = real time")
  p.add_argument("--hold-s", type=float, default=1.5, help="end card, s")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out-dir", default=os.path.join(_ZOO, "runs", "shac_vs_ppo_sonic", "videos",
                                                   "push_bins"))
  a = p.parse_args(argv)
  a.policies = []
  for s in a.policy:
    tag, sep, path = s.partition("=")
    if not sep or not tag or not path:
      p.error(f"--policy {s!r}: expected TAG=PATH")
    a.policies.append((tag, path))
  try:
    a.bin_ranges = [parse_bin(b) for b in a.bins]
  except ValueError as e:
    p.error(str(e))
  if a.per_bin < 1 or min(a.width, a.height, a.fps) < 1:
    p.error("--per-bin, --width, --height and --fps must be >= 1")
  if not 0.0 < a.onset_s[0] <= a.onset_s[1] or a.window_s <= 0.0:
    p.error("--onset-s needs 0 < lo <= hi and --window-s > 0")
  # what eval_sonic_qcbf.Evaluator reads from its args
  a.num_envs, a.pi_obs_clamp, a.sonic_history = 1, "box", "applied"
  return a


# --- drawing ------------------------------------------------------------------------------

def _font(size: int):
  from PIL import ImageFont
  try:
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
  except Exception:
    return ImageFont.load_default(size=size)


class Painter:
  """Panels and the banner of the side-by-side video."""

  def __init__(self):
    self.big, self.mid, self.small = _font(30), _font(22), _font(19)

  def panel(self, frame: np.ndarray, label: str, i: int, on: int, kick: str,
            outcome: tuple[str, tuple] | None) -> np.ndarray:
    """Frame ``i`` (control steps since the spawn) of one controller."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(frame))
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    d.rectangle((0, 0, w, 40), fill=(0, 0, 0, 150))
    d.text((10, 8), label, font=self.mid, fill=(255, 255, 255))
    rel = (i - on) * DT
    clock = f"kick in {-rel:.2f} s" if i < on else f"kick +{rel:.2f} s"
    d.text((w - 10, 10), clock, font=self.small, fill=(255, 255, 255), anchor="ra")
    if on <= i < on + int(round(KICK_FLASH_S / DT)) and outcome is None:
      box = d.textbbox((w // 2, h - 40), kick, font=self.mid, anchor="mm")
      d.rectangle((box[0] - 14, box[1] - 8, box[2] + 14, box[3] + 8), fill=ORANGE + (230,))
      d.text((w // 2, h - 40), kick, font=self.mid, fill=(255, 255, 255), anchor="mm")
    if outcome is not None:
      text, col = outcome
      box = d.textbbox((w // 2, h // 2), text, font=self.big, anchor="mm")
      d.rectangle((box[0] - 18, box[1] - 12, box[2] + 18, box[3] + 12), fill=col + (225,))
      d.text((w // 2, h // 2), text, font=self.big, fill=(255, 255, 255), anchor="mm")
    return np.asarray(im)

  def banner(self, width: int, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, BANNER_H), (25, 25, 30))
    d = ImageDraw.Draw(im)
    for j, ln in enumerate(lines[:2]):
      d.text((12, 6 + 28 * j), ln, font=self.small, fill=(235, 235, 235))
    return np.asarray(im)


def outcome_of(survived: bool, ttf: float, window_s: float) -> tuple[str, tuple]:
  if survived:
    return f"SURVIVED {window_s:g} s after the kick", GREEN
  if not ttf > 0.0:
    return "FELL before the kick", RED
  return f"FELL {ttf:.2f} s after the kick", RED


def compose(painter: Painter, runs: list[dict], on: int, kick: str, banner: np.ndarray,
            t: int) -> np.ndarray:
  """Video frame ``t``: the banner over one panel per run. A run is
  ``{"label", "frames", "outcome"}``; past its last frame it holds that frame
  under its outcome."""
  tiles = []
  for r in runs:
    last = len(r["frames"]) - 1
    i = min(t, last)
    tiles.append(painter.panel(r["frames"][i], r["label"], i, on, kick,
                               r["outcome"] if t >= last else None))
  return np.concatenate([banner, np.concatenate(tiles, axis=1)], axis=0)


# --- recording ----------------------------------------------------------------------------

def video_cfg(cfg, width: int, height: int) -> None:
  """The camera's frame size, and no debug drawings: the velocity command's
  arrows (neither SONIC nor the fallbacks follow it) and sensor rays."""
  cfg.viewer.width, cfg.viewer.height = int(width), int(height)
  terms = list((getattr(cfg, "commands", None) or {}).values())
  sensors = getattr(cfg.scene, "sensors", None) or ()
  terms += list(sensors.values() if isinstance(sensors, dict) else sensors)
  for t in terms:
    if hasattr(t, "debug_vis"):
      t.debug_vis = False


def main(argv=None):
  import imageio
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  t_start = time.time()
  os.makedirs(a.out_dir, exist_ok=True)
  lib = load_bank(a.library, "cpu")
  lm = lib["meta"]
  cats, clips = list(lm.get("categories", [])), list(lm.get("clips", []))
  csv_dir = lm["csv_dir"]
  if not os.path.isdir(csv_dir):
    csv_dir = os.path.join(_ZOO, "data", "lafan1_g1")
  base = SonicReference.build(csv_dir, lm["clips"], dev, library=lib)
  la = br.look_ahead()
  win = _steps(a.window_s)

  # --- one episode list per bin, end to end in the env's dataset
  pool = (lib["start_ok"] & (lib["kind"] == KIND_MOCAP)).nonzero(as_tuple=False).squeeze(-1)
  room = lib["next_ok"][pool]
  lists = {}
  for name, rng in zip(a.bins, a.bin_ranges):
    gen = torch.Generator().manual_seed(_seed(a.seed, f"video/{name}"))
    lists[name] = ql.draw_episodes(pool, room, a.per_bin, gen, DT, a.onset_s, win, la, rng)
  bank_path = os.path.join(a.out_dir, "episodes_bank.pt")
  firsts = write_episode_bank(lib, lists, bank_path)
  total = sum(int(v["row"].numel()) for v in lists.values())

  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  overrides = dict(dataset=bank_path, sampler="sequential", sequential_repeats=1,
                   weights=weights, episode_length_s=float(a.onset_s[1] + a.window_s + 1.0),
                   rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_env(a, overrides, push_kw, render_mode="rgb_array",
                 cfg_fn=lambda cfg: video_cfg(cfg, a.width, a.height))
  mj = env.mj
  if abs(float(mj.step_dt) - DT) > 1e-9:
    raise SystemExit(f"[video] control step {mj.step_dt} s: SONIC's references need {DT} s")
  spawn = _spawn_term(mj)
  seq = getattr(spawn, "_seq_rows", None)
  if (spawn.sampler != "sequential" or seq is None or int(seq.numel()) != total
          or not bool((seq.cpu() == torch.arange(total)).all())):
    raise SystemExit("[video] the spawn does not walk the episode bank row by row")
  mass = float(mj.sim.mj_model.body_mass.sum())
  tracker = SonicTracker(env, SonicPolicy.from_onnx(a.sonic_dir).to(dev).eval(), base)
  # (label, evaluator, condition): SONIC alone, then each fallback alone
  ctls = []
  for j, (tag, path) in enumerate(a.policies):
    model, norm = load_twin(path, dev)
    model.policy.set_training_mode(False)
    ev = Evaluator(env, tracker, base, model, norm, {}, spawn, a)
    if j == 0:      # SONIC's run still steps pi's memory resets: any model will do
      ctls.append(("SONIC", ev, ql.parse_condition("sonic")))
    ctls.append((learner_name(tag), ev, ql.parse_condition("fallback")))
  F = int(base.joint_pos.shape[0])
  no_k = torch.zeros(F, dtype=torch.bool, device=dev)       # K flags: not needed here
  no_clear = torch.full((F,), math.inf, device=dev)
  print(f"[video] {total} episodes in {len(lists)} bins x {len(ctls)} controllers; "
        f"{HIP_DESC[a.hip]}; robot mass {mass:.1f} kg; setup {time.time() - t_start:.0f}s",
        flush=True)

  painter = Painter()
  hold = int(round(a.hold_s * a.fps))
  summary, reel_parts = [], {}
  for name in a.bins:
    eps = episode_columns(lists[name], win, lib, base, no_k, no_clear, dev)
    sub = os.path.join(a.out_dir, name)
    os.makedirs(sub, exist_ok=True)
    reel_parts[name] = []
    for e in range(int(eps["on"].numel())):
      one = {k: v[e:e + 1] for k, v in eps.items()}
      on = int(one["on"][0])
      dv, rel = float(one["dv"][0]), float(one["rel"][0])
      cat = cats[int(one["category"][0])] if cats else str(int(one["category"][0]))
      clip = clips[int(one["clip"][0])] if clips else str(int(one["clip"][0]))
      way = direction_name(rel)
      kick = f"KICK {dv:.1f} m/s {way}"
      runs = []
      for label, ev, cond in ctls:
        frames = []

        def on_step(ep, k, dones, frames=frames):
          if int(ep[0]) >= 0 and (dones is None or not bool(dones[0])):
            frames.append(np.asarray(env.render()))

        rec = ev.run(f"video/{name}/ep{e}", ev.controller(cond), firsts[name] + e, one, base,
                     win, push=True, on_step=on_step)
        survived, ttf = bool(rec["survived"][0]), float(rec["time_to_fall"][0])
        runs.append({"label": label, "frames": frames,
                     "outcome": outcome_of(survived, ttf, a.window_s),
                     "survived": survived, "ttf_s": None if survived else ttf})
      lines = [f"{clip}  frame {int(one['frame0'][0])}  ({cat})      |      {HIP_DESC[a.hip]}",
               f"kick: |dv| = {dv:.2f} m/s ({mass * dv:.0f} N s) toward the robot's {way}, "
               f"{on * DT:.2f} s after the spawn; survival counted {a.window_s:g} s after it"]
      width = a.width * len(runs)
      banner = painter.banner(width, lines)
      ok = "".join("S" if r["survived"] else "F" for r in runs)
      fname = f"ep{e:02d}_{cat}_{dv:.1f}ms_{way}_{ok}.mp4"
      path = os.path.join(sub, fname)
      writer = imageio.get_writer(path, fps=a.fps, codec="libx264", quality=6,
                                  macro_block_size=1)
      try:
        T = max(len(r["frames"]) for r in runs)
        for t in range(T + hold):
          writer.append_data(compose(painter, runs, on, kick, banner, t))
      finally:
        writer.close()
      reel_parts[name].append(path)
      row = {"bin": name, "file": os.path.relpath(path, a.out_dir), "clip": clip,
             "frame": int(one["frame0"][0]), "category": cat, "dv": dv,
             "impulse_ns": mass * dv, "direction": way, "onset_s": on * DT,
             "outcomes": {r["label"]: {"survived": r["survived"], "ttf_s": r["ttf_s"]}
                          for r in runs}}
      summary.append(row)
      print(f"[video] {row['file']}: " + "; ".join(
        f"{r['label']} {r['outcome'][0]}" for r in runs), flush=True)

  def reel(paths, out):
    writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=6, macro_block_size=1)
    try:
      for f in paths:
        for frame in imageio.get_reader(f):
          writer.append_data(frame)
    finally:
      writer.close()

  for name, paths in reel_parts.items():
    reel(paths, os.path.join(a.out_dir, f"reel_{name}.mp4"))
  reel([p for name in a.bins for p in reel_parts[name]], os.path.join(a.out_dir, "reel.mp4"))
  meta = {"source": "record_sonic_push_video.py", "hip": a.hip, "hip_desc": HIP_DESC[a.hip],
          "library": os.path.abspath(a.library), "policies": dict(a.policies),
          "bins": a.bins, "per_bin": a.per_bin, "onset_s": list(a.onset_s),
          "window_s": a.window_s, "seed": a.seed, "robot_mass_kg": mass,
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
    json.dump({"meta": meta, "episodes": summary}, f, indent=1)
  with open(os.path.join(a.out_dir, "summary.txt"), "w") as f:
    f.write(f"{HIP_DESC[a.hip]}; kick 0.5-1.5 s after the spawn, survival {a.window_s:g} s "
            f"after it; robot mass {mass:.1f} kg\n\n")
    for r in summary:
      f.write(f"{r['file']}: {r['clip']} frame {r['frame']} ({r['category']}), kick "
              f"{r['dv']:.2f} m/s ({r['impulse_ns']:.0f} N s) {r['direction']} at "
              f"{r['onset_s']:.2f} s\n")
      for label, o in r["outcomes"].items():
        f.write(f"    {label:28s} {'survived' if o['survived'] else 'fell %.2f s after' % o['ttf_s']}\n")
  print(f"[video] wrote {len(summary)} episodes, reels and summary.txt to {a.out_dir} "
        f"({time.time() - t_start:.0f}s)", flush=True)
  env.close()


if __name__ == "__main__":
  main()
