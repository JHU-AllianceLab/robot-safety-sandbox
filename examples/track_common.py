"""Shared by the squat-tracking trainer, eval and report: the arms, the paths, the stack.

Six arms (AMENDMENTS A3): two worlds x {bare, world-model tube filter with no
dwell, with a 1 s dwell}. A world is the frozen SHAC fallback's world with the
task added (tasks/g1_track.py):

  thermal70    no barrier; (70 C - hottest winding) / 10 C in the margin; 70 C
               ends the episode. Its BARE arm is the user's "no safety filter".
  thermal_cbf  the 70 C torque-clipping barrier under every controller; margin
               fall-only; no overheat termination.

Task seed i is paired with fallback / tube seed i (the user's choice); evals may
cross-pair (``fb_seed``).

The stack (``build_stack``), innermost first:

  make_tensor(task)      the task env (``task_policy`` / ``task_critic`` / ``amp``
                         groups beside the fallback's untouched ``actor`` / ``critic``)
  WMTubeFilteredEnv      the filter, or its BARE pass-through (same interface; the
                         critic gets the 2 filter-state columns in every arm)
  TrackStatsEnv          track/* metrics from the task's episode bus
  AMPTaskEnv             the style reward (training; eval builds without it)

and the learner adds TensorVecNormalize on top. The fallback bundle is loaded
BEFORE any learner exists (AMENDMENTS B10; ``load_wm_tube`` also preserves every
RNG), and ``check_binding`` holds the env to the fallback's frozen layout.
"""

from __future__ import annotations

import json
import os

ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join("runs", "track_squat")
#: the task policy's action box (AMENDMENTS A2); the fallback keeps +-3
PROPOSAL_BOUND = 6.5

WORLDS = {
  "thermal70": dict(
    task="g1_track_squat_thermal70", fallback_task="g1_recover_thermal70",
    run="runs/shac_thermal70/shac_s{s}/g1_recover_thermal70_safetyshac1p",
    tube="runs/shac_thermal70/wmreport/calib/thermal70_s{s}/min_tube.json",
    tag="thermal70_s{s}"),
  "thermal_cbf": dict(
    task="g1_track_squat_thermal_cbf", fallback_task="g1_recover_thermal_cbf",
    run="runs/shac_thermal_cbf/shac_s{s}/g1_recover_thermal_cbf_safetyshac1p",
    tube="runs/shac_thermal_cbf/wmreport/calib/thermalcbf_s{s}/min_tube.json",
    tag="thermalcbf_s{s}"),
}
WORLD_SHORT = {"thermal70": "t70", "thermal_cbf": "cbf"}

#: arm -> (world, filter kind or None, dwell seconds)
ARMS = {
  "t70_bare": ("thermal70", None, 0.0),
  "t70_wm_d0": ("thermal70", "wmmin", 0.0),
  "t70_wm_d1": ("thermal70", "wmmin", 1.0),
  "cbf_bare": ("thermal_cbf", None, 0.0),
  "cbf_wm_d0": ("thermal_cbf", "wmmin", 0.0),
  "cbf_wm_d1": ("thermal_cbf", "wmmin", 1.0),
}
SEEDS = (0, 1, 2)
PILOT_SEED = 2


def arm_world(arm: str) -> str:
  return ARMS[arm][0]


def arm_name(world: str, filt, dwell_s: float) -> str:
  """The arm with this world / filter / dwell (eval conditions are named alike)."""
  short = WORLD_SHORT[world]
  if filt is None:
    return f"{short}_bare"
  return f"{short}_wm_d{int(round(dwell_s))}"


def task_of(world: str) -> str:
  return WORLDS[world]["task"]


def world_of_task(task: str) -> str:
  for w, d in WORLDS.items():
    if d["task"] == task or d["fallback_task"] == task:
      return w
  raise ValueError(f"no world has task {task!r}")


def fallback_run(world: str, seed: int) -> str:
  return os.path.join(ZOO, WORLDS[world]["run"].format(s=int(seed)))


def tube_json(world: str, seed: int) -> str:
  """One min_tube.json per seed holds every coverage (0.95, 0.90, ...)."""
  return os.path.join(ZOO, WORLDS[world]["tube"].format(s=int(seed)))


def tube_tag(world: str, seed: int) -> str:
  return WORLDS[world]["tag"].format(s=int(seed))


def condition(conf: float = 0.95) -> str:
  return f"wmmin:1:{conf:g}"


def layout_json(world: str) -> str:
  return os.path.join(ZOO, "configs", "track_squat", f"fallback_layout_{world}.json")


def run_dir(stage: str, arm: str, seed: int) -> str:
  return os.path.join(RUNS, stage, arm, f"s{int(seed)}")


def eval_dir(stage: str, arm: str, seed: int, world: str) -> str:
  return os.path.join(RUNS, "eval", stage, f"{arm}_s{int(seed)}", world)


def report_dir(stage: str) -> str:
  return os.path.join(RUNS, "report", stage)


def load_bundle(world: str, fb_seed: int, *, conf: float = 0.95, device: str = "cuda:0"):
  from robot_safety_sandbox.filters.wm_tube import load_wm_tube
  return load_wm_tube(fallback_run(world, fb_seed), tube_json(world, fb_seed),
                      tag=tube_tag(world, fb_seed), cond=condition(conf), device=device)


def make_task_env(task: str, num_envs: int, device: str, *, env_overrides=None, cfg_fn=None):
  """``make_tensor(task)``, or the same env with ``cfg_fn(cfg)`` applied to its cfg
  last (eval: observation noise off) -- the pattern of
  harvest_g1_sonic_trajectories.make_env."""
  from robot_safety_sandbox.registry import CUMULATIVE, make_tensor, spec
  if cfg_fn is None:
    return make_tensor(task, int(num_envs), device, cfg_overrides=env_overrides or None)
  from robot_safety_sandbox.base import MjlabTensorSafetyEnv
  s = spec(task)

  def builder(play: bool = False, **kw):
    cfg = s.cfg_builder(play=play, **kw)
    cfg_fn(cfg)
    return cfg

  return MjlabTensorSafetyEnv(
    int(num_envs), device, cfg_builder=builder, margin_fn=s.margin_fn, ctrl_dim=s.ctrl_dim,
    dstb_dim=s.dstb_dim, adversary=False, end_criterion=s.end_criterion,
    cfg_overrides=env_overrides or None, dense_reward=s.mode == CUMULATIVE, **s.kwargs)


def build_stack(task: str, num_envs: int, device: str, *, filter_kind=None, dwell_s: float = 0.0,
                fb_seed: int = 0, conf: float = 0.95, env_overrides=None, bundle=None,
                amp_w_style=None, amp_w_style_frac: float = 0.4, stats: bool = True,
                coverage_probe: bool = True, shadow: bool = False, strict_layout: bool = True):
  """The wrapper stack of the module docstring. ``amp_w_style``: None = no AMP
  layer (eval), "auto" = ``amp_w_style_frac`` x the task's positive reward
  weights, or a number. Returns ``(env, info)``; ``info`` has the binding check,
  the fallback's provenance and the thermal world, for the run's meta."""
  from robot_safety_sandbox.filtered_env import WMTubeFilteredEnv
  from robot_safety_sandbox.filters.wm_tube import check_binding, thermal_world_info, wm_tube_filter

  world = world_of_task(task)
  env = make_task_env(task, num_envs, device, env_overrides=env_overrides)
  env.reset()
  info = {"task": task, "world": world, "num_envs": int(num_envs),
          "thermal_world": thermal_world_info(env), "filter": None}
  filt = None
  if filter_kind == "wmmin":
    if bundle is None:
      bundle = load_bundle(world, fb_seed, conf=conf, device=device)
    lay_path = layout_json(world)
    layout = None
    if os.path.isfile(lay_path):
      with open(lay_path) as f:
        layout = json.load(f)
    elif strict_layout:
      raise FileNotFoundError(f"{lay_path} is missing: run examples/smoke_wm_tube.py "
                              f"--dump-layout --world {world} first")
    info["binding"] = check_binding(bundle, env, layout_ref=layout)
    filt, bundle = wm_tube_filter(env, dwell_s=float(dwell_s), proposal_bound=PROPOSAL_BOUND,
                                  condition=condition(conf), bundle=bundle)
    info["filter"] = {"kind": "wmmin", "condition": condition(conf), "dwell_s": float(dwell_s),
                      "fb_seed": int(fb_seed), "fallback": bundle.meta}
  elif filter_kind is not None:
    raise ValueError(f"unknown filter kind {filter_kind!r}")
  out = WMTubeFilteredEnv(env, filt, coverage_probe=coverage_probe, shadow=shadow)
  if stats:
    from robot_safety_sandbox.track_stats import TrackStatsEnv
    out = TrackStatsEnv(out, step_dt=float(env.mj.step_dt))
  if amp_w_style is not None:
    from robot_safety_sandbox.amp.env import AMPTaskEnv, w_style_from_env
    w = (w_style_from_env(env, amp_w_style_frac) if amp_w_style == "auto"
         else float(amp_w_style))
    out = AMPTaskEnv(out, w_style=w, step_dt=float(env.mj.step_dt))
    info["w_style"] = w
  info["bundle"] = bundle
  return out, info
