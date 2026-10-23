"""Record an mp4 of a trained recovery policy under random pushes.

    python examples/record_push_video.py \
        --load runs/g1_recover_push_ladder/g1_recover_push/final_model.zip \
        --force 200 --seconds 12 --out runs/g1_recover_push_ladder/g1_recover_push/videos/push_200N.mp4

Renders a small herd of envs of a random_push task (default ``g1_recover_push``:
bank spawn, a 0.2 s push every 2-4 s) with the policy's deterministic action,
at an explicit push magnitude: ``--force`` newtons, times ``--scale`` (the
per-env survival scale a training run would have; 1.0 = the full force). The
trainers' in-run clips do not know the curriculum's magnitude unless the
recorder mirrors it, so use this to see a checkpoint at the push level you
mean. The obs normalizer is found next to the checkpoint (tensornormalize.pt /
tensornorm_<steps>.pt). Needs a GL context for offscreen rendering: export
MUJOCO_GL=egl (see envs/velocity/g1.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch as th

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
from robot_safety_sandbox.callbacks import _record_clip  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402


def _parse_overrides(items):
  """``KEY=VAL`` pairs -> dict, values typed as YAML (3 -> int, 0.4 -> float)."""
  import yaml
  out = {}
  for item in items:
    if "=" not in item:
      raise SystemExit(f"--env-override expects KEY=VAL, got {item!r}")
    k, v = item.split("=", 1)
    out[k.strip()] = yaml.safe_load(v)
  return out


def _parse():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--load", required=True, help="a safety_sb3 checkpoint .zip")
  p.add_argument("--task", default="g1_recover_push",
                 help="a random_push task (g1_recover_push: 8 s episodes, a push "
                      "every 2-4 s; g1_recover_push_fast: the push right after the spawn)")
  p.add_argument("--force", type=float, default=200.0, help="push magnitude at scale 1, N")
  p.add_argument("--scale", type=float, default=1.0,
                 help="per-env multiplier on --force (a run's game/force_scale_mean)")
  p.add_argument("--seconds", type=float, default=12.0, help="seconds of robot motion")
  p.add_argument("--fps", type=int, default=50, help="50 = real time at 50 Hz control")
  p.add_argument("--num-envs", type=int, default=4, help="herd size drawn in one scene")
  p.add_argument("--env-override", action="append", default=[], metavar="KEY=VAL",
                 help="env cfg_builder param, e.g. init_level=3 or episode_length_s=6")
  p.add_argument("--out", default=None,
                 help="mp4 path (default: <run>/videos/push_<force>N.mp4 next to --load)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args()


def main():
  args = _parse()
  th.manual_seed(args.seed)
  dev = args.device
  overrides = _parse_overrides(args.env_override) if args.env_override else None
  env = make_tensor(args.task, args.num_envs, dev, render_mode="rgb_array",
                    cfg_overrides=overrides)
  if getattr(env, "_push", None) is None:
    raise SystemExit(f"task {args.task!r} has no random push; use a random_push task")
  env.force_max = float(args.force)
  env.force_scale = th.full((args.num_envs,), float(args.scale), device=dev)
  try:
    env.mj.cfg.viewer.max_extra_envs = max(0, args.num_envs - 1)   # draw the herd
  except Exception:
    pass
  model, norm = load_twin(args.load, dev)
  model.policy.set_training_mode(False)
  bound = float(env.action_space.high[0])
  n_frames = max(1, int(round(args.seconds / float(env.mj.step_dt))))

  def predict(obs, b):
    with th.no_grad():
      act = model.policy._predict(norm(obs), deterministic=True)
    obs, _g, _d, _t, _l = env.step_tensor(th.clamp(act, -b, b))
    return obs

  out = args.out
  if out is None:
    out = os.path.join(os.path.dirname(os.path.abspath(args.load)), "videos",
                       f"push_{args.force * args.scale:.0f}N.mp4")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  print(f"[video] {args.task} x {args.num_envs}, push {args.force * args.scale:.0f} N "
        f"({args.force:g} x {args.scale:g}), {args.seconds:g} s @ {args.fps} fps -> {out}")
  _record_clip(env, predict, n_frames, args.fps, out, bound)
  print(f"[video] wrote {out} ({n_frames} frames)")
  env.close()


if __name__ == "__main__":
  main()
