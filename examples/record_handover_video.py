"""Record the deployment story of a fallback, one robot: a clip, a push, a handover.

    python examples/record_handover_video.py \
        --load runs/lafan_recovery/E6/C-push_s0/g1_recover_lafan/final_model.zip \
        --out-dir runs/lafan_recovery/E6/videos_handover/story

Each episode is one E-handover start (``harvest_g1_handover_states.py``) with
a visible lead-in, in one env of ``g1_recover_lafan`` spawning on the clip
LIBRARY (default: the held-out test clips):

1. LEAD-IN. The ``--lead-s`` of the clip BEFORE the spawn frame r are shown
   kinematically: each frame's pose is written into the sim and rendered, no
   physics (``states.write_states``), the planar path integrated back from
   the spawn pose with each frame's heading-frame twist.
2. NOMINAL. From frame r the clip is replayed open loop as PD targets
   (``envs/g1_recover/playback.py``) for ``--onset-s``, as in the harvest.
3. PUSH. A scripted horizontal push of ``J`` N*s over ``--duration-s`` lands,
   from one of the 8 suite directions (drawn); ``J = 0`` is no push at all.
4. HANDOVER. ``--delay-s`` after the push onset the replay stops and the
   fallback (``--load``, deterministic) takes over -- mid-push when the delay
   is shorter than the push -- for ``--window-s``, or until it falls.

Why the lead-in is kinematic: the open-loop replay drifts (nothing closes the
loop on the base) and a second of it falls on its own, even on walking clips,
which is why the harvest and the suite keep it to tenths of a second. A
replay that still falls before the handover is discarded and redrawn
(counted in the summary). Episodes cycle through ``--categories`` and
``--impulses``. With ``--paired`` (and one ``--room-delay-s`` for all), episode
N starts from the same clip frame with the same push in every run, whatever
the policy or ``--delay-s``.

Writes one mp4 per episode (``ep<NN>_<category>_<J>Ns_<direction>_<outcome>``),
a ``reel.mp4`` of all of them, and ``summary.txt``. A banner names the phase,
the clip and the push; a timeline at the bottom marks the clip (light blue:
kinematic, blue: PD replay), the push (orange) and the fallback (green); the
last frame is held with the outcome. Needs a GL context: export MUJOCO_GL=egl.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
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

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.playback import PlaybackDriver  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.push_suite import DIR_NAMES_8  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  load_bank, snapshot, write_states)
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402

LIGHT_BLUE, BLUE = (120, 170, 240), (50, 100, 220)
ORANGE, GREEN, RED = (240, 150, 30), (40, 190, 90), (225, 60, 60)
ROW_KEYS = ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos", "joint_vel")


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--load", required=True, help="the fallback, a safety_sb3 checkpoint .zip")
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"),
                 help="the clip library (build_g1_recover_mocap_states.py --library)")
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--out-dir", default=None,
                 help="default: <run>/videos/handover_story next to --load")
  p.add_argument("--episodes", type=int, default=8)
  p.add_argument("--categories", nargs="+", default=None,
                 help="clip categories to cycle through (default: all in the library)")
  p.add_argument("--impulses", type=float, nargs="+", default=[15.0, 30.0, 45.0],
                 help="push impulses J to cycle through, N*s (0 = no push: the "
                      "handover comes --onset-s + --delay-s into the replay)")
  p.add_argument("--lead-s", type=float, default=1.0,
                 help="kinematic playback of the clip before the spawn frame, s")
  p.add_argument("--onset-s", type=float, default=0.1,
                 help="PD replay before the push onset, s (the harvest draws 0.02-0.1)")
  p.add_argument("--duration-s", type=float, default=0.2, help="push duration T, s")
  p.add_argument("--delay-s", type=float, default=0.1,
                 help="push onset -> handover, s (the suite's delays: 0, 0.1, 0.2)")
  p.add_argument("--window-s", type=float, default=4.0, help="fallback time shown, s")
  p.add_argument("--hold-s", type=float, default=1.0, help="outcome card, s")
  p.add_argument("--pause-s", type=float, default=0.0,
                 help="freeze the video on the handover step for this long, s (0 = no pause)")
  p.add_argument("--fps", type=int, default=50, help="50 = real time at 50 Hz control")
  p.add_argument("--max-attempts", type=int, default=30,
                 help="replays per episode that may fall before the handover")
  p.add_argument("--paired", action="store_true",
                 help="draw each episode's start and push direction from (--seed, episode) "
                      "alone, so runs of other policies or delays show the same episodes")
  p.add_argument("--room-delay-s", type=float, default=None,
                 help="spawn only frames with room for this delay (default --delay-s); give "
                      "every run of a --paired comparison the largest delay")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args(argv)
  if min(args.onset_s, args.duration_s) <= 0.0 or min(args.lead_s, args.delay_s) < 0.0:
    p.error("--onset-s and --duration-s must be > 0, --lead-s and --delay-s >= 0")
  if args.room_delay_s is not None and args.room_delay_s < args.delay_s:
    p.error("--room-delay-s must be >= --delay-s")
  return args


def _font(size: int):
  from PIL import ImageFont
  try:
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
  except Exception:
    return ImageFont.load_default(size=size)


class Overlay:
  """Banner, timeline and outcome card drawn onto the rendered frames. ``g``
  counts frames from the start of the lead-in: ``g <= L`` is the kinematic
  lead-in, ``k = g - L`` the control steps since the spawn."""

  def __init__(self, dt: float, L: int, on: int, D: int, P: int, W: int):
    self.dt, self.L, self.on, self.D, self.P, self.W = dt, L, on, D, P, W
    self.total = L + P + W
    self.big, self.mid, self.small = _font(34), _font(24), _font(20)

  def phase(self, g: int):
    """(tag, colour) of frame ``g``."""
    k = g - self.L
    if k <= 0:
      return "NOMINAL: LAFAN clip", LIGHT_BLUE
    pushing = self.on < k <= self.on + self.D
    if k <= self.P:
      return ("NOMINAL + PUSH", ORANGE) if pushing else ("NOMINAL: LAFAN replay", BLUE)
    return ("FALLBACK + push still acting", GREEN) if pushing else ("FALLBACK", GREEN)

  def draw(self, frame: np.ndarray, g: int, lines: list[str],
           outcome: tuple[str, tuple] | None = None, flag: str | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    tag, col = self.phase(g)
    box_w = max([d.textlength(tag, font=self.big) + 30]
                + [d.textlength(ln, font=self.small) for ln in lines])
    d.rectangle((16, 16, 40 + box_w, 16 + 50 + 30 * len(lines)), fill=(0, 0, 0, 150))
    d.rectangle((16, 16, 30, 66), fill=col)
    d.text((42, 20), tag, font=self.big, fill=(255, 255, 255))
    for i, ln in enumerate(lines):
      d.text((28, 72 + 30 * i), ln, font=self.small, fill=(235, 235, 235))
    k = g - self.L
    clock = f"t = {k * self.dt:+5.2f} s"
    if k > self.P:
      clock += f"   fallback +{(k - self.P) * self.dt:4.2f} s"
    d.text((w - 16, 20), clock, font=self.mid, fill=(255, 255, 255), anchor="ra")
    # timeline: clip | replay | push | fallback, with a cursor
    x0, x1, y = 40, w - 40, h - 34

    def x(step):
      return x0 + (x1 - x0) * min(step, self.total) / self.total

    L, P = self.L, self.P
    d.rectangle((x0, y, x(L), y + 12), fill=LIGHT_BLUE)
    d.rectangle((x(L), y, x(L + P), y + 12), fill=BLUE)
    d.rectangle((x(L + P), y, x1, y + 12), fill=GREEN)
    if self.D > 0:
      d.rectangle((x(L + self.on), y - 4, x(L + self.on + self.D), y + 16), fill=ORANGE)
    d.line((x(g), y - 10, x(g), y + 22), fill=(255, 255, 255), width=4)
    d.text((x0, y - 28), "nominal (clip)", font=self.small, fill=(255, 255, 255))
    d.text((x(L + P) + 6, y - 28), "handover -> fallback", font=self.small,
           fill=(255, 255, 255))
    if flag is not None:     # above the timeline, clear of the robot in the middle
      box = d.textbbox((w // 2, y - 64), flag, font=self.big, anchor="mm")
      d.rectangle((box[0] - 20, box[1] - 10, box[2] + 20, box[3] + 10), fill=GREEN + (220,))
      d.text((w // 2, y - 64), flag, font=self.big, fill=(255, 255, 255), anchor="mm")
    if outcome is not None:
      text, oc = outcome
      box = d.textbbox((w // 2, h // 2), text, font=self.big, anchor="mm")
      d.rectangle((box[0] - 24, box[1] - 16, box[2] + 24, box[3] + 16), fill=oc + (220,))
      d.text((w // 2, h // 2), text, font=self.big, fill=(255, 255, 255), anchor="mm")
    return np.asarray(im)


def _yaw(quat_w: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_w.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lead_in_path(lib: dict, row: int, L: int, yaw_r: float, xy_r: torch.Tensor,
                 dt: float):
  """Heading and planar offset of rows ``row - L .. row - 1``, integrated back
  from the spawn pose of ``row`` with each frame's heading-frame twist
  (``x_{q+1} = x_q + R(yaw_q) v_q dt``, ``yaw_{q+1} = yaw_q + w_z,q dt``)."""
  yaws = torch.zeros(L, device=xy_r.device)
  xys = torch.zeros(L, 2, device=xy_r.device)
  yaw, xy = float(yaw_r), xy_r.clone()
  for i in range(L - 1, -1, -1):
    q = row - L + i
    yaw -= float(lib["ang_vel_h"][q, 2]) * dt
    v = lib["lin_vel_h"][q, :2]
    c, s = math.cos(yaw), math.sin(yaw)
    xy = xy - torch.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]]) * dt
    yaws[i], xys[i] = yaw, xy
  return yaws, xys


def main(argv=None):
  import imageio
  args = _parse(argv)
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  dev = args.device
  lib = load_bank(args.library, "cpu")
  if not bool((lib["next_ok"] > 0).any()):
    raise SystemExit(f"{args.library} is not a clip library (no next_ok)")
  clips = list(lib["meta"].get("clips", []))
  cats = list(lib["meta"].get("categories", []))
  want = args.categories or cats
  unknown = set(want) - set(cats)
  if unknown:
    raise SystemExit(f"unknown categories {sorted(unknown)}; the library has {cats}")

  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  overrides = dict(dataset=args.library, sampler="uniform", weights=weights,
                   episode_length_s=float(args.onset_s + args.delay_s + args.window_s + 1.0),
                   rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_tensor(args.task, 1, dev, render_mode="rgb_array", cfg_overrides=overrides,
                    random_push=push_kw)
  mj, push = env.mj, env._push
  robot = mj.scene["robot"]
  dt = float(mj.step_dt)
  L = int(round(args.lead_s / dt))
  on = max(1, int(round(args.onset_s / dt)))
  D = max(1, int(round(args.duration_s / dt)))
  P = on + int(round(args.delay_s / dt))
  room = on + int(round((args.delay_s if args.room_delay_s is None else args.room_delay_s) / dt))
  W = int(round(args.window_s / dt))
  hold = int(round(args.hold_s * args.fps))
  pause = int(round(args.pause_s * args.fps))
  ov = Overlay(dt, L, on, D, P, W)
  driver = PlaybackDriver(lib, 1, dev)
  dlib = {k: lib[k].to(dev) for k in ROW_KEYS + ("next_ok", "category", "clip", "frame")}
  # rows with L frames of the same clip right before them
  lead_ok = torch.zeros_like(dlib["next_ok"], dtype=torch.bool)
  if L > 0:
    lead_ok[L:] = ((dlib["clip"][L:] == dlib["clip"][:-L])
                   & (dlib["frame"][L:] - dlib["frame"][:-L] == L))
  else:
    lead_ok[:] = True
  model, norm = load_twin(args.load, dev)
  model.policy.set_training_mode(False)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  one = torch.ones(1, dtype=torch.bool, device=dev)
  ids = torch.zeros(1, dtype=torch.long, device=dev)

  out_dir = args.out_dir or os.path.join(
    os.path.dirname(os.path.abspath(args.load)), "videos", "handover_story")
  os.makedirs(out_dir, exist_ok=True)

  def act(obs):
    with torch.no_grad():
      pol = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    return driver.actions(pol)

  def end_episode(obs):
    """Truncate through the env's own time_out on the next step: a new spawn."""
    buf = mj.episode_length_buf
    buf[:] = torch.clamp_min(buf, int(mj.max_episode_length) - 1)
    obs, *_ = env.step_tensor(act(obs))
    driver.stop(one)
    return obs

  def lead_in(writer, row: int, lines: list[str]) -> None:
    """Render the L clip frames before ``row``, then put the spawn back."""
    snap = snapshot(mj)
    yaw_r = _yaw(robot.data.root_link_quat_w)[0]
    xy_r = robot.data.root_link_pos_w[0, :2] - mj.scene.env_origins[0, :2]
    dz = snap["root_z"][0] - dlib["root_z"][row]     # any ground alignment of the spawn
    yaws, xys = lead_in_path(dlib, row, L, float(yaw_r), xy_r, dt)
    for i in range(L):
      q = row - L + i
      rows = {k: dlib[k][q:q + 1].clone() for k in ROW_KEYS}
      rows["root_z"] += dz
      write_states(mj, ids, rows, yaws[i:i + 1], xys[i:i + 1])
      mj.sim.forward()
      writer.append_data(ov.draw(np.asarray(env.render()), i + 1, lines))
    write_states(mj, ids, {k: snap[k] for k in ROW_KEYS}, yaw_r.reshape(1),
                 xy_r.reshape(1, 2))
    mj.sim.forward()

  obs = env.reset()
  kept, summary = [], []
  replay_falls = {c: 0 for c in want}
  for ep in range(args.episodes):
    cat_name = want[ep % len(want)]
    c_idx = cats.index(cat_name)
    J = float(args.impulses[ep % len(args.impulses)])
    F = J / (D * dt)
    if args.paired:
      # whatever the earlier episodes did (policy, delay, how long they lasted), this
      # one's spawns and push direction follow from (seed, ep): reseed, then respawn
      torch.manual_seed(args.seed * 100003 + ep)
      rng = np.random.default_rng((args.seed, ep))
      obs = end_episode(obs)
    attempts, skips, path, outcome = 0, 0, None, None
    while outcome is None and attempts < args.max_attempts:
      row = int(mj._zoo_spawn_row[0])
      if (row < 0 or int(dlib["category"][row]) != c_idx or int(dlib["next_ok"][row]) < room
          or not bool(lead_ok[row])):
        skips += 1
        if skips > 100000:
          raise SystemExit(f"no {cat_name} frame with the lead-in and {room} frames "
                           f"after it was spawned")
        obs = end_episode(obs)
        continue
      attempts += 1
      d_idx = int(rng.integers(8))
      rel = 2.0 * math.pi * d_idx / 8
      name_c = clips[int(dlib["clip"][row])] if clips else str(int(dlib["clip"][row]))
      if args.paired:    # the frame shows that paired runs started from the same one
        name_c += f" frame {int(dlib['frame'][row])}"
      ov.D = D if J > 0 else 0
      if J > 0:
        lines = [f"clip: {name_c} ({cat_name})",
                 f"push: {J:g} N*s from {DIR_NAMES_8[d_idx]} ({F:.0f} N x {D * dt:.2f} s)",
                 f"at t = {on * dt:.2f} s, handover {args.delay_s:g} s after onset"]
      else:
        lines = [f"clip: {name_c} ({cat_name})", "no push",
                 f"handover at t = {P * dt:.2f} s"]
      path = os.path.join(out_dir, f".ep{ep:02d}_attempt.mp4")
      writer = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=6,
                                  macro_block_size=1)
      k, last, g_last = 0, None, 0
      try:
        lead_in(writer, row, lines)
        driver.start(one, torch.tensor([row], device=dev), torch.tensor([P], device=dev))
        if J > 0:
          push.schedule(one, force_n=torch.tensor([F], device=dev),
                        steps=torch.tensor([D], device=dev),
                        onset=torch.tensor([on], device=dev),
                        rel=torch.tensor([rel], device=dev))
        while True:
          obs, _g, dones, timeouts, _l = env.step_tensor(act(obs))
          driver.advance()
          k += 1
          if bool(dones[0]):                      # the env has already reset
            driver.stop(one)
            if k <= P:
              replay_falls[cat_name] += 1         # the replay fell, not the fallback
            elif bool(timeouts[0]):
              outcome = ("SURVIVED (episode cap)", GREEN)
            else:
              outcome = (f"FELL {(k - P) * dt:.2f} s after the handover", RED)
            break
          last, g_last = np.asarray(env.render()), L + k
          writer.append_data(ov.draw(last, g_last, lines))
          if k == P and pause:                    # the last replay step: the fallback acts next
            card = ov.draw(last, g_last, lines, flag="HANDOVER: the fallback takes over")
            for _ in range(pause):
              writer.append_data(card)
          if k - P >= W:
            outcome = (f"SURVIVED {args.window_s:g} s after the handover", GREEN)
            obs = end_episode(obs)
            break
        if outcome is not None and last is not None:
          card = ov.draw(last, g_last, lines, outcome)
          for _ in range(hold):
            writer.append_data(card)
      finally:
        writer.close()
      if outcome is None:
        os.remove(path)
    if outcome is None:
      print(f"[story] ep {ep}: {cat_name}: every replay fell before the handover "
            f"({args.max_attempts} attempts); skipped", flush=True)
      continue
    tag = "survived" if outcome[1] == GREEN else "fell"
    push_tag = f"{J:g}Ns_{DIR_NAMES_8[d_idx]}" if J > 0 else "nopush"
    name = f"ep{ep:02d}_{cat_name}_{push_tag}_{tag}.mp4"
    final = os.path.join(out_dir, name)
    os.replace(path, final)
    kept.append(final)
    summary.append(f"{name}: {lines[0]}; {lines[1]}; {outcome[0]}")
    print(f"[story] {summary[-1]}", flush=True)

  if kept:
    reel = os.path.join(out_dir, "reel.mp4")
    writer = imageio.get_writer(reel, fps=args.fps, codec="libx264", quality=6,
                                macro_block_size=1)
    try:
      for f in kept:
        for frame in imageio.get_reader(f):
          writer.append_data(frame)
    finally:
      writer.close()
    print(f"[story] wrote {reel} ({len(kept)} episodes)")
  with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
    fh.write(f"policy {args.load}\nlibrary {args.library}\nlead-in {args.lead_s} s "
             f"(kinematic), replay {args.onset_s} s, push {args.duration_s} s, handover "
             f"{args.delay_s} s after onset, window {args.window_s} s\n\n"
             + "\n".join(summary) + "\n\n"
             f"replays that fell before the handover (discarded): {replay_falls}\n")
  env.close()


if __name__ == "__main__":
  main()
