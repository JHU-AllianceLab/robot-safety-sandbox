"""Evaluation entry point: pick an env, a filter, a task policy, an attack.

One script for every filter evaluation in this repo. The four axes are chosen
independently on the command line and the harness composes them; nothing about
any particular terrain lives here (see robot_safety_sandbox/eval/).

    --task / --preset / --env-override   WHICH WORLD
    --task-policy                        the pi_task being filtered
    --safety-policy / --filter           the certificate + which composition
    --dstb  / --dstb-scale               WHO IS ATTACKING, and how hard
    --stay-policy                        reach-avoid-STAY: a second (stay) twin
                                         to hand over to once it certifies

  # filter gauntlet on flat ground under a swept adversarial attack
  python examples/eval.py --task go2_locomote --adversary \
      --task-policy runs/go2_walker_flat/final_model.zip \
      --safety-policy runs/go2_stabilize_sac2p/final_model.zip \
      --filter gameplay --dstb policy --dstb-scale 0.5 --num-envs 256

  # the gap gauntlet (E021), now a preset rather than its own script
  python examples/eval.py --preset gap_gauntlet \
      --task-policy runs/go2_walker_flat/final_model.zip \
      --safety-policy runs/go2_gap_chain_ra/final_model.zip \
      --filter value --gap-width 0.35 --n-gaps 1 --num-envs 256 --steps 600

  # the unfiltered control arm
  python examples/eval.py ... --no-filter

  python examples/eval.py --task g1_waist_reach_arm_random --safety-only \
      --safety-policy runs/g1_waist_reach_arm_random/final_model.zip \
      --stay-policy runs/g1_waist_stay_arm_random/final_model.zip \
      --num-envs 256 --steps 1000

An attack STRENGTH sweep is the point of --dstb-scale: run the same command at
0.0 / 0.25 / ... / 1.0 and the result is a curve, not a point.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
# Dev fallback: safety_sb3 is a pip dependency in release; when running from a
# source checkout, look for the sibling repo or $SAFETY_SB3_PATH.
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from robot_safety_sandbox import list_tasks, spec  # noqa: E402
from robot_safety_sandbox.eval import (  # noqa: E402
  FILTERS, NominalPolicy, ReachAvoidStayStats, RolloutCfg, SwitchCfg,
  SwitchedTwinNominal, TwinNominal, TrajectoryRecorder, TwistCommandSurgery,
  VideoRecorder, ZeroNominal, build_eval_env, build_filter,
  build_reach_avoid_stay_filter, list_presets, load_nominal, load_twin,
  make_dstb, preset, protocol_metrics, run_eval, safety_modules)
from robot_safety_sandbox.eval.metrics import MetricSet  # noqa: E402


def _parse_overrides(pairs):
  """``KEY=VAL`` strings -> a cfg_builder kwargs dict (VAL parsed as YAML)."""
  import yaml
  out = {}
  for item in pairs or []:
    if "=" not in item:
      raise SystemExit(f"--env-override expects KEY=VAL, got {item!r}")
    k, v = item.split("=", 1)
    out[k.strip()] = yaml.safe_load(v)
  return out


def build_parser(pre_args):
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  # --- which world
  p.add_argument("--task", default=None, help=f"one of {list_tasks()}")
  p.add_argument("--preset", default=None, choices=list_presets(),
                 help="a named evaluation configuration (env surgery + the "
                      "task's own success metrics)")
  p.add_argument("--env-override", action="append", metavar="KEY=VAL",
                 default=None,
                 help="override a param of the task's cfg_builder (repeatable)")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--episode-s", type=float, default=None,
                 help="override the env's episode length (s)")
  p.add_argument("--cmd-vx", type=float, default=1.0,
                 help="forward velocity command given to the task policy")
  p.add_argument("--engaged-cmd-vx", type=float, default=1.0,
                 help="command held while the FALLBACK drives -- the value the "
                      "safety policy was trained under (feeding it 0 is OOD)")
  p.add_argument("--action-bound", type=float, default=1.0,
                 help="clamp the NOMINAL policy's action to +-this (the bridge action_bound it was "
                      "trained under). Default 1.0 = the historical box; a policy trained unclamped "
                      "must pass its wide bound so it is not silently re-clipped at eval (T006)")
  p.add_argument("--no-command-surgery", action="store_true",
                 help="do not drive the velocity command from the filter's "
                      "verdict (see eval/envs.py TwistCommandSurgery)")
  p.add_argument("--end-criterion", default=None,
                 choices=["failure", "reach-avoid", "timeout"])
  p.add_argument("--safety-obs-key", default=None,
                 help="override the safety policy's obs group (default: "
                      "auto-detected)")
  p.add_argument("--task-obs-key", default=None,
                 help="override the task policy's obs group (default: "
                      "auto-detected)")
  # --- the policies
  p.add_argument("--task-policy", default=None,
                 help="pi_task checkpoint (stock SB3 zip); omit for a zero "
                      "task policy (fallback-only run)")
  p.add_argument("--safety-only", action="store_true",
                 help="use the SAFETY policy's own control actor as the task "
                      "policy -- the disturbance-effect probe: evaluate the "
                      "deployable policy the two-player game produced against "
                      "its own adversary (pair with --no-filter and a --dstb "
                      "sweep)")
  p.add_argument("--safety-policy", default=None,
                 help="safety policy checkpoint (any of the eight MAP cells)")
  # --- the filter
  p.add_argument("--filter", default="value", choices=sorted(FILTERS),
                 help="which COMPOSITION to deploy; they differ in one module "
                      "each: " + " | ".join(f"{k}: {v}"
                                            for k, v in FILTERS.items()))
  p.add_argument("--stay-policy", default=None,
                 help="REACH-AVOID-STAY: the STAY twin (avoid on the stay "
                      "constraint, e.g. g1_waist_stay_arm_random). --safety-"
                      "policy is then the REACH-AVOID twin. Each phase is a "
                      "value filter and a latched per-env mode switches "
                      "between them on V_stay (filters/switch.py); needs "
                      "--filter value. Command surgery is off in this mode.")
  p.add_argument("--stay-eps-in", type=float, default=0.1,
                 help="--stay-policy: hand over to the stay phase when "
                      "V_stay > this")
  p.add_argument("--stay-eps-out", type=float, default=0.0,
                 help="--stay-policy: hand back to the reach phase when "
                      "V_stay <= this (<= --stay-eps-in)")
  p.add_argument("--stay-task", default=None,
                 help="--stay-policy: the task whose margin_fn is the stay "
                      "constraint, measured live for ras_hold_rate (default: "
                      "--task with '_reach' -> '_stay', when registered)")
  p.add_argument("--no-filter", action="store_true",
                 help="CONTROL arm: run the task policy unfiltered")
  p.add_argument("--eps", type=float, default=0.0,
                 help="switching threshold: hand over to the fallback when the "
                      "monitored margin <= eps")
  p.add_argument("--smoothing", action="store_true",
                 help="use HeuristicSmoothingIntervention (latch + 5-step "
                      "median + release hysteresis + rest-speed gate) instead "
                      "of the canonical LeastRestrictiveIntervention. NOT a "
                      "literature filter and not Def-2 valid step-by-step: "
                      "strictly more conservative, and results must be "
                      "reported as a smoothed VARIANT of the named filter. "
                      "Required to reproduce E051/E054 and every gauntlet "
                      "before 2026-07-25, which predate the canonical switch.")
  p.add_argument("--caution", type=float, default=0.45,
                 help="--smoothing only: upper edge of the caution band")
  p.add_argument("--hysteresis", type=float, default=0.15,
                 help="--smoothing only: release needs margin > eps + this")
  p.add_argument("--rest-speed", type=float, default=0.4,
                 help="--smoothing only: release also needs speed < this")
  p.add_argument("--kappa", type=float, default=0.8,
                 help="--filter qcbf: class-K coefficient, Q(x,u) >= kappa V(x)")
  p.add_argument("--horizon", type=int, default=20,
                 help="--filter rollout/gameplay: H sim steps per certification")
  p.add_argument("--rollouts", type=int, default=1,
                 help="--filter rollout/gameplay: parallel rollouts per env; "
                      "the verdict is their MIN (one failure condemns)")
  p.add_argument("--recertify-every", type=int, default=1,
                 help="--filter rollout/gameplay: re-certify every k steps")
  # NOTE: there is deliberately no --rollout-reach-avoid. Which reduction
  # certifies the rollout is DERIVED from the task's mode (see
  # eval/filters.py::reach_avoid_reduction); it is not the operator's to pick.
  p.add_argument("--contact-history", default="sync",
                 choices=("sync", "instantaneous", "ignore"))
  # --- the attack
  p.add_argument("--adversary", action="store_true",
                 help="give the env a live disturbance channel (the task must "
                      "declare supports_adversary)")
  p.add_argument("--dstb", default="none", choices=["none", "random", "policy"],
                 help="who plays the disturbance: nobody, uniform noise, or a "
                      "trained min-player (--adversary-policy, else "
                      "--safety-policy)")
  p.add_argument("--dstb-scale", type=float, default=1.0,
                 help="ATTACK STRENGTH, continuous. 1.0 = the task's own "
                      "training-time magnitude; sweep it for a curve")
  p.add_argument("--adversary-policy", default=None,
                 help="checkpoint supplying the disturbance actor, when it is "
                      "not the same safety policy as the certificate")
  # --- run
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--progress-every", type=int, default=0)
  p.add_argument("--out", default=None, help="write metrics JSON here")
  p.add_argument("--video", default=None,
                 help="write an mp4 here; the border tint is the fraction of "
                      "the herd the filter is driving. Needs an offscreen GL "
                      "context — use MUJOCO_GL=egl on a headless box "
                      "(otherwise mujoco raises 'gladLoadGL error')")
  p.add_argument("--video-fps", type=int, default=30)
  p.add_argument("--traj", default=None, metavar="DIR",
                 help="record trajectories into DIR: trajectories.pkl + "
                      "stats.json (the plot_trajectory_coverage.py contract) "
                      "and trajectories_full.npz (state, both raw actions, the "
                      "filter's decision, g/l, flags)")
  p.add_argument("--traj-envs", type=int, default=64,
                 help="--traj: record only the first N envs (the rest of the "
                      "metrics still use the full population)")
  p.add_argument("--traj-frame", default="spawn", choices=("spawn", "world"),
                 help="--traj: frame of trajectories.pkl. 'spawn' puts each "
                      "episode's start at the origin facing +x (what the "
                      "coverage plot's star assumes -- mjlab spreads envs over "
                      "the terrain grid and randomizes spawn yaw); 'world' "
                      "writes raw simulator coordinates. The npz keeps world "
                      "coordinates either way")
  p.add_argument("--tag", default=None, help="label copied into the summary")

  if pre_args.preset:
    ps = preset(pre_args.preset)
    if ps.add_args:
      ps.add_args(p)
    if ps.defaults:
      p.set_defaults(**ps.defaults)
    if ps.task:
      p.set_defaults(task=ps.task)
  return p


def main():
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--preset", default=None)
  pre_args, _ = pre.parse_known_args()
  args = build_parser(pre_args).parse_args()

  if not args.task:
    raise SystemExit("--task is required (or use a --preset that names one). "
                     f"Registered: {list_tasks()}")
  if not args.safety_policy:
    raise SystemExit("--safety-policy is required: the filter's certificate and "
                     "its fallback both come from a safety policy.")
  ps = preset(args.preset) if args.preset else None
  device = args.device

  # 1. THE ENVIRONMENT ---------------------------------------------------------
  overrides = _parse_overrides(args.env_override)
  if args.stay_policy:
    if args.filter != "value":
      raise SystemExit("--stay-policy composes two VALUE filters; use "
                       "--filter value (the default).")
    # a task whose target is built from the stay twin (g1_waist_reach_*) must
    # read the SAME twin the switch hands over to
    if "stay_run" in inspect.signature(spec(args.task).cfg_builder).parameters:
      overrides.setdefault("stay_run", args.stay_policy)
  env = build_eval_env(
    args.task, args.num_envs, device, adversary=args.adversary,
    env_overrides=overrides, end_criterion=args.end_criterion,
    cfg_transform=ps.cfg_transform(args) if (ps and ps.cfg_transform) else None,
    safety_obs_key=args.safety_obs_key, nominal_obs_key=args.task_obs_key,
    render_mode="rgb_array" if args.video else None)
  if args.episode_s is not None:
    env.mj.cfg.episode_length_s = args.episode_s
  print(f"[env] {args.task} n={env.num_envs} adversary={env.adversary} "
        f"obs: task='{env.nominal_obs_key}' safety='{env.safety_obs_key}'"
        + (f" preset={args.preset}" if ps else ""))

  # 2. THE SAFETY FILTER -------------------------------------------------------
  # The reach-avoid-stay path clamps the twins to the task's own action box
  # (the G1 tasks train at +-3); the single-twin path keeps its historical +-1.
  bound = float(env.bridge.action_bound) if args.stay_policy else 1.0
  safety_model, norm = load_twin(args.safety_policy, device)
  mods = safety_modules(safety_model, env.num_envs, device, bound=bound)
  mods["norm"] = norm
  switch = SwitchCfg(eps=args.eps, smoothing=args.smoothing,
                     caution=args.caution, hysteresis=args.hysteresis,
                     rest_speed=args.rest_speed)
  if args.stay_policy:
    stay_model, stay_norm = load_twin(args.stay_policy, device)
    stay_mods = safety_modules(stay_model, env.num_envs, device, bound=bound)
    stay_mods["norm"] = stay_norm
    bundle = build_reach_avoid_stay_filter(
      mods, stay_mods, env, switch=switch, eps_in=args.stay_eps_in,
      eps_out=args.stay_eps_out)
    # each phase normalizes with its own twin's statistics (see the builder)
    norm = lambda obs: obs.float()  # noqa: E731
  else:
    bundle = build_filter(
      args.filter, mods, env, switch=switch,
      rollout=RolloutCfg(horizon=args.horizon, rollouts=args.rollouts,
                         recertify_every=args.recertify_every,
                         contact_history=args.contact_history),
      kappa=args.kappa)
  print(f"[filter] {'reach-avoid-stay' if args.stay_policy else args.filter}: "
        + ("two value filters under a latched mode" if args.stay_policy
           else FILTERS[args.filter])
        + (" (NOT APPLIED: --no-filter control arm)" if args.no_filter else ""))

  # 3. THE TASK POLICY ---------------------------------------------------------
  if args.task_policy and args.safety_only:
    raise SystemExit("--task-policy and --safety-only are alternatives: the "
                     "task policy is either an external pi_task or the safety "
                     "policy's own control actor, not both.")
  if args.safety_only and args.stay_policy:
    print(f"[task-policy] the switched controller's own actors: "
          f"{mods['twin']} in REACH, {stay_mods['twin']} in STAY")
    task_policy = SwitchedTwinNominal(bundle.filt,
                                      bundle.extras["reach_fallback_fn"],
                                      bundle.extras["stay_fallback_fn"])
  elif args.safety_only:
    print(f"[task-policy] the safety policy's own control actor "
          f"({mods['twin']})")
    task_policy = TwinNominal(mods["fallback_fn"], norm)
  elif args.task_policy:
    # Clamp the nominal to the SAME action bound it was TRAINED under (--action-bound, passed by rsf
    # from interface.action.clip — the same source the compiler uses for the bridge). NOT a hardcoded
    # +-1: an unclamped-trained walker re-clipped to +-1 here would be a different policy at eval than
    # in training, silently corrupting every speed number (T006).
    task_policy = NominalPolicy(*load_nominal(args.task_policy, device),
                                device=device, bound=args.action_bound)
  else:
    print("[task-policy] none: zero action (fallback-only run)")
    task_policy = ZeroNominal(env.num_envs, env.ctrl_dim, device)

  # 4. THE ATTACK --------------------------------------------------------------
  dstb_fn = None
  if args.dstb != "none":
    if not env.adversary:
      raise SystemExit(f"--dstb {args.dstb} needs a live disturbance channel; "
                       "pass --adversary (the task must support one).")
    src = mods
    if args.adversary_policy:
      d_safety, d_norm = load_twin(args.adversary_policy, device)
      src = safety_modules(d_safety, env.num_envs, device) | {"norm": d_norm}
    dstb_fn = make_dstb(args.dstb, env, src)
    print(f"[dstb] {args.dstb} @ scale={args.dstb_scale} "
          f"(dim={env.dstb_dim}, mode={env.bridge.dstb_mode})")

  # 5. THE METRICS -------------------------------------------------------------
  metrics = protocol_metrics(env, bundle.filt)
  if args.stay_policy:
    stay_task = args.stay_task or args.task.replace("_reach", "_stay")
    constraint_fn = None
    if stay_task in list_tasks():
      stay_margin = spec(stay_task).margin_fn
      constraint_fn = lambda mj: stay_margin(mj)[0]  # noqa: E731
      print(f"[ras] stay constraint for ras_hold_rate: {stay_task}'s margin")
    else:
      print(f"[ras] no stay task {stay_task!r} registered: ras_hold_rate "
            f"not measured (pass --stay-task)")
    metrics = MetricSet(metrics, ReachAvoidStayStats(
      bundle.filt, env.num_envs, env.device, env.mj.step_dt, constraint_fn))
  if ps and ps.metrics:
    extra = ps.metrics(args, env)
    if extra is not None:
      metrics = MetricSet(metrics, extra)
  if args.traj:
    n_traj = min(args.traj_envs, env.num_envs)
    print(f"[traj] recording envs 0..{n_traj - 1} of {env.num_envs} "
          f"({args.traj_frame} frame) -> {args.traj}")
    metrics = MetricSet(metrics,
                        TrajectoryRecorder(env, args.traj,
                                           traj_envs=args.traj_envs,
                                           frame=args.traj_frame))

  surgery = None
  if args.stay_policy:
    # the twist command IS the goal's velocity half here; never drive it
    print("[command-surgery] off: --stay-policy (the command is part of the "
          "goal)")
  elif not args.no_command_surgery:
    surgery = TwistCommandSurgery(cmd_vx=args.cmd_vx,
                                  engaged_cmd_vx=args.engaged_cmd_vx).bind(env)
    if not surgery.available:
      surgery = None

  video = VideoRecorder(args.video, args.video_fps) if args.video else None
  summary = run_eval(
    env, task_policy, bundle.filt, metrics, steps=args.steps, norm=norm,
    dstb_fn=dstb_fn, dstb_scale=args.dstb_scale, no_filter=args.no_filter,
    command_surgery=surgery, video=video, seed=args.seed,
    progress_every=args.progress_every)
  if video is not None:
    video.save()

  summary.update(
    task=args.task, preset=args.preset, composition=args.filter,
    filter="off" if args.no_filter else os.path.basename(
      os.path.dirname(os.path.abspath(args.safety_policy))),
    safety_policy=mods["twin"], steps=args.steps, num_envs=args.num_envs,
    stay_policy=args.stay_policy,
    dstb=args.dstb, dstb_scale=args.dstb_scale, seed=args.seed, tag=args.tag)
  if bundle.shadow is not None:
    summary["certifications"] = bundle.shadow.seeds

  print("\n=== EVAL ===")
  for k, v in summary.items():
    print(f"  {k:28s} {v}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"[saved] {args.out}")
  bundle.close()
  env.close()


if __name__ == "__main__":
  main()
