"""Zoo trainer (model-based family): SHAC through a learned world model.

  python examples/train.py --family model_based --task g1_recover --steps 1000000000
  python examples/train.py --config configs/g1_recover_safetyshac1p.yaml \
      --out runs/g1_recover_shac

The learner is the MAP cell ``{Safety,ReachAvoid,Cumulative}SHAC1P`` from
safety_sb3 (registry.algo_name, family="model_based"): SuperMimic's recipe
(Liao et al., CoRL 2026) with the imagined return replaced by the task's own
operator. Each iteration collects a few vector steps with the deterministic
actor plus Gaussian noise, fits the world model on k-step replay windows, then
backpropagates the H-step imagined value -- the min-over-time safety value on an
avoid task -- through the frozen model into the actor, and refits the critic on
the same imagined rollout. See safety_sb3/shac.py.

Model state: the env's privileged "critic" obs group (``--privileged-critic``,
ON by default here: the actor obs alone is not Markov on G1 -- no base linear
velocity, no pelvis height). The bridge's ``world_model_spec()`` tells the
learner which critic dims form the actor obs, which hold the previous action,
and the actor obs's sensor noise; ``sim_context()`` hands it the realized
domain-randomization draw (foot friction, torso COM, encoder biases).

Refused here, not silently degraded: tasks with IN-EPISODE exogenous
disturbances (random pushes, scripted random joints, a learned adversary). A
deterministic world model cannot predict a push it is not told about, and
fitting through one teaches it wrong dynamics. Train on tasks whose randomness
lives in the start state and the sim context (g1_recover and its banks).

Warm start (``--load``): a SafetyPPO1P checkpoint with the same net shapes --
privileged critic included -- seeds the actor AND the critic (same parameter
names, same input), plus the obs normalizer; a SHAC checkpoint also restores
its world model. Use ``wm_warmup_iters`` / ``actor_delay_iters`` so the model
and critic catch up before the first actor step.

The run directory is ``<out>/<task>_<learner>`` (e.g.
``runs/g1_recover_safetyshac1p``) -- never the PPO trainer's ``<out>/<task>``,
whose checkpoint names it would otherwise overwrite one by one. Inside it the
layout is the on-policy trainer's: ``final_model.zip`` + ``tensornormalize.pt``,
``checkpoints/model_<N>_steps.zip`` + ``checkpoints/tensornorm_<N>.pt``. The
eval loaders recognize the zip by its ``world_model.pth``: eval.py,
eval_push_suite.py and play.py load it like a PPO twin. With the (default)
privileged critic, V(s) reads the critic obs group, so the VALUE filter is
unavailable in eval.py (named error); fallback/rollout filters work.
"""

from __future__ import annotations

import argparse
import os
import sys

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

import torch as th  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402

from _run_config import dump_config, merge_config  # noqa: E402
from robot_safety_sandbox import algo_name, list_tasks, make_tensor, spec  # noqa: E402
from robot_safety_sandbox.callbacks import (  # noqa: E402
  NormFreezeCallback, TensorNormSaveCallback, VideoWandbCallback)

import safety_sb3 as _safety_sb3  # noqa: E402

#: task kwargs that inject an in-episode exogenous disturbance the world model
#: cannot see (module docstring)
_EXOGENOUS = ("random_push", "joint_noise", "hybrid_skill")


def _ints(text: str) -> list[int]:
  return [int(x) for x in str(text).split(",") if x.strip()]


def _warm_start(model, path: str, device: str) -> None:
  """Seed the policy (actor + critic), the obs normalizer and, from a SHAC
  checkpoint, the world model. Shape-filtered; optimizer moments start fresh.
  Refuses a warm start that loaded no actor tensor at all."""
  from stable_baselines3.common.save_util import load_from_zip_file

  from robot_safety_sandbox.eval.policies import find_obs_stats
  _, params, _ = load_from_zip_file(path, device=device)
  actor = {n: p.detach().clone() for n, p in model.policy.named_parameters()
           if n.startswith(("mlp_extractor.policy_net", "action_net"))}
  src, tgt = params.get("policy", {}), model.policy.state_dict()
  ok = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
  model.policy.load_state_dict(ok, strict=False)
  skipped = [k for k in src if k not in ok]
  print(f"[warm-start] policy: {len(ok)}/{len(src)} tensors (skipped "
        f"{len(skipped)}: {skipped[:3]})")
  moved = [n for n, p in model.policy.named_parameters()
           if n in actor and not th.equal(p.detach(), actor[n])]
  if not moved:
    raise SystemExit(
      "[warm-start] NO actor tensor changed on load -- the checkpoint's actor "
      "keys / shapes do not match this policy (check --net against the source).")
  wm_src = params.get("world_model")
  if wm_src is not None:
    wm_tgt = model.world_model.state_dict()
    if all(k in wm_tgt and wm_tgt[k].shape == v.shape for k, v in wm_src.items()) \
        and set(wm_src) == set(wm_tgt):
      model.world_model.load_state_dict(wm_src)
      print("[warm-start] world model restored (coordinates frozen as saved)")
    else:
      print("[warm-start] world model in the checkpoint does not match this "
            "recipe's shapes; it starts fresh")
  tvn = model.env
  pt = find_obs_stats(path, "tensornorm", "tensornormalize.pt")
  if not pt:
    raise SystemExit(
      f"[warm-start] no tensornormalize.pt / tensornorm_<steps>.pt next to {path}: "
      "the loaded policy reads NORMALIZED obs and needs the stats it trained on.")
  st = th.load(pt, map_location=device, weights_only=True)
  tvn.obs_mean, tvn.obs_var, tvn.count = (
    st["obs_mean"].to(tvn.device), st["obs_var"].to(tvn.device),
    st["count"].to(tvn.device))
  msg = f"[warm-start] obs stats from {os.path.basename(pt)}"
  if getattr(tvn, "critic_observation_space", None) is not None:
    if "critic_obs_mean" in st and st["critic_obs_mean"].shape == tvn.critic_obs_mean.shape:
      tvn.critic_obs_mean, tvn.critic_obs_var, tvn.critic_count = (
        st["critic_obs_mean"].to(tvn.device), st["critic_obs_var"].to(tvn.device),
        st["critic_count"].to(tvn.device))
      msg += " (+ critic obs stats)"
    else:
      msg += " (NO matching critic obs stats: the critic's normalizer starts fresh)"
  print(msg)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--config", default=None,
                 help="YAML recipe of args (keys = flag dest names). Sets defaults; "
                      "explicit CLI flags override it. See configs/.")
  p.add_argument("--env-override", action="append", metavar="KEY=VAL", default=None,
                 help="override an env/task cfg_builder param (repeatable); also a "
                      "config `env_overrides:` dict. Forwarded to make_tensor.")
  p.add_argument("--task", required=True, help=f"one of {list_tasks()}")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--steps", type=int, default=1_000_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--checkpoint-every", type=int, default=25_000_000, metavar="N",
                 help="env-steps between checkpoints (policy + world model AND the "
                      "obs normalizer, on the same grid)")
  p.add_argument("--load", default=None,
                 help="warm-start .zip: a SafetyPPO1P (actor + critic) or SHAC "
                      "(+ world model) checkpoint with matching net shapes")
  p.add_argument("--norm-freeze-steps", type=int, default=5_000_000,
                 help="freeze obs-norm updates at the start of WARM-STARTED runs")
  p.add_argument("--terminal-type", choices=["all", "g"], default="all",
                 help="reach-avoid learners only: value a terminal step as "
                      "min(l,g) ('all') or g ('g')")
  p.add_argument("--end-criterion", choices=["failure", "reach-avoid", "timeout"],
                 default=None, help="WHEN the episode ends from (g,l); default = "
                                    "the task's TaskSpec value")
  p.add_argument("--privileged-critic", action=argparse.BooleanOptionalAction,
                 default=True,
                 help="use the env cfg's 'critic' obs group as the world-model state "
                      "(and the critic's input). ON by default: the G1 actor obs is "
                      "not Markov (no base linear velocity / pelvis height)")
  # --- objective --------------------------------------------------------------
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--gamma-anneal", action=argparse.BooleanOptionalAction,
                 default=False, help="anneal the discount (off: the paper's fixed 0.99)")
  p.add_argument("--gae-lambda", type=float, default=0.95,
                 help="lambda of the critic's nested lambda-return")
  p.add_argument("--softmin-tau", type=float, default=0.0,
                 help="soft-min temperature of the ACTOR objective, in margin units "
                      "(0 = the exact min; critic targets always use the exact one)")
  p.add_argument("--action-rate-coef", type=float, default=0.0,
                 help="weight of mean ||a_t - a_{t-1}||^2 over imagined actions on the "
                      "actor loss (value units, not PPO's)")
  p.add_argument("--action-mag-coef", type=float, default=0.0,
                 help="weight of mean ||a_t||^2 over imagined actions (value units)")
  p.add_argument("--action-bound-coef", type=float, default=1.0,
                 help="weight of the actor mean's squared excursion outside the "
                      "action box (the clamp passes no gradient back)")
  # --- networks / optimization -------------------------------------------------
  p.add_argument("--net", default="512,256,128",
                 help="hidden dims for the actor AND the critic (match a PPO "
                      "checkpoint to warm-start both)")
  p.add_argument("--lr", type=float, default=2e-4, help="actor LR (and the default "
                 "for the critic / world model)")
  p.add_argument("--critic-lr", type=float, default=None)
  p.add_argument("--wm-lr", type=float, default=None)
  p.add_argument("--lr-decay-iters", type=int, default=250_000,
                 help="cosine-decay every LR over this many updates, then hold (0=off)")
  p.add_argument("--lr-min-frac", type=float, default=0.03)
  p.add_argument("--critic-epochs", type=int, default=5)
  p.add_argument("--critic-minibatches", type=int, default=4)
  # --- imagination ---------------------------------------------------------------
  p.add_argument("--horizon", type=int, default=32, help="imagined steps H")
  p.add_argument("--imagine-batch", type=int, default=4096,
                 help="imagined rollouts per actor update")
  p.add_argument("--action-noise-std", type=float, default=0.1,
                 help="Gaussian collection noise on the actor's action (action units)")
  # --- world model / replay --------------------------------------------------------
  p.add_argument("--replay-size", type=int, default=1000, help="steps kept per env")
  p.add_argument("--learning-starts", type=int, default=64,
                 help="vector steps collected before the first update")
  p.add_argument("--utd-ratio", type=float, default=0.4,
                 help="updates per vector step (0.4 -> one per 2.5 vector steps)")
  p.add_argument("--wm-window", type=int, default=8, help="k of the k-step fit")
  p.add_argument("--wm-batch", type=int, default=2048, help="windows per WM step")
  p.add_argument("--wm-gradient-steps", type=int, default=10,
                 help="world-model gradient steps per iteration")
  p.add_argument("--wm-net", default="1024,1024,1024")
  p.add_argument("--wm-head-net", default="256,256")
  p.add_argument("--wm-warmup-iters", type=int, default=0,
                 help="iterations that fit only the world model")
  p.add_argument("--actor-delay-iters", type=int, default=0,
                 help="after the WM warmup, iterations that fit the critic but hold "
                      "the actor (a warm-started critic catching up)")
  p.add_argument("--actor-lr-warmup-iters", type=int, default=0,
                 help="ramp the actor LR linearly from 0 over its first N updates "
                      "(0 = off)")
  p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False,
                 help="TF32 matmuls (the paper's precision; faster, less exact)")
  p.add_argument("--log-interval", type=int, default=50, help="iterations per log dump")
  # --- telemetry -------------------------------------------------------------------
  p.add_argument("--video-interval", type=int, default=50_000_000)
  p.add_argument("--video-seconds", type=float, default=None)
  p.add_argument("--video-fps", type=int, default=50)
  p.add_argument("--video-wandb", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--video-herd", type=int, default=-1)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs"),
                 help="output root; ALWAYS keep runs under runs/ (git-ignored)")
  p.add_argument("--wandb-project", default="robot_safety_sandbox")
  p.add_argument("--no-wandb", action="store_true")
  args = merge_config(p)

  s = spec(args.task)
  exo = [k for k in _EXOGENOUS if s.kwargs.get(k) is not None]
  if exo:
    raise SystemExit(
      f"'{args.task}' injects in-episode exogenous disturbances ({exo}). SHAC's "
      "world model is deterministic and is not told about them, so it would fit "
      "wrong dynamics through every disturbed transition. Train on a task whose "
      "randomness lives in the start state / sim context (e.g. g1_recover).")
  algo = algo_name(args.task, family="model_based")
  Algo = getattr(_safety_sb3, algo, None)
  if Algo is None:
    raise SystemExit(f"this safety_sb3 does not export '{algo}'; update "
                     "safety-stable-baselines (feat: SHAC).")
  print(f"[algo] {args.task} -> {algo}")

  # Tagged with the learner, like the off-policy trainer: the PPO trainer
  # writes <out>/<task> with the same checkpoint names on the same grid.
  outdir = os.path.join(args.out, f"{args.task}_{algo.lower()}")
  os.makedirs(outdir, exist_ok=True)
  dump_config(outdir, args)

  pkw = {"critic_obs_key": "critic"} if args.privileged_critic else {}
  env = make_tensor(args.task, args.num_envs, args.device,
                    end_criterion=args.end_criterion,
                    cfg_overrides=args.env_overrides, **pkw)
  if args.privileged_critic:
    print(f"[world-model] state = the '{env.critic_obs_key}' group "
          f"({env.critic_observation_space.shape[0]} dims); actor obs "
          f"{env.observation_space.shape[0]} dims")
  ctx = env.sim_context()
  print(f"[world-model] sim context: {0 if ctx is None else ctx.shape[1]} dims "
        f"{[(f, len(i)) for _, f, i in getattr(env, '_ctx_cols', None) or []]}")

  akw = {}
  if algo.startswith("ReachAvoid"):
    akw["terminal_type"] = args.terminal_type
  net = _ints(args.net)
  model = Algo(
    "MlpPolicy", env, **akw,
    learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
    horizon=args.horizon, imagine_batch_size=args.imagine_batch,
    softmin_tau=args.softmin_tau, action_noise_std=args.action_noise_std,
    replay_size=args.replay_size, learning_starts=args.learning_starts,
    utd_ratio=args.utd_ratio, wm_window=args.wm_window,
    wm_batch_size=args.wm_batch, wm_gradient_steps=args.wm_gradient_steps,
    wm_net_arch=_ints(args.wm_net), wm_head_arch=_ints(args.wm_head_net),
    wm_learning_rate=args.wm_lr, wm_warmup_iters=args.wm_warmup_iters,
    actor_delay_iters=args.actor_delay_iters,
    actor_lr_warmup_iters=args.actor_lr_warmup_iters,
    critic_learning_rate=args.critic_lr, critic_epochs=args.critic_epochs,
    critic_minibatches=args.critic_minibatches,
    lr_decay_iters=args.lr_decay_iters, lr_min_frac=args.lr_min_frac,
    action_rate_coef=args.action_rate_coef, action_mag_coef=args.action_mag_coef,
    action_bound_coef=args.action_bound_coef,
    normalize_obs=True, gamma_anneal=args.gamma_anneal, tf32=args.tf32,
    policy_kwargs=dict(net_arch=dict(pi=net, vf=net)),
    seed=args.seed, verbose=1, device=args.device, tensorboard_log=outdir)
  S, A, C = model._state_dim, int(env.action_space.shape[0]), model._context_dim
  rows = args.replay_size * args.num_envs
  gib = rows * (S + A + 4) * 4 / 2**30
  print(f"[recipe] net={net} wm={args.wm_net} H={args.horizon} "
        f"imagine={args.imagine_batch} lr={args.lr} gamma={args.gamma} "
        f"softmin_tau={args.softmin_tau}; replay {args.replay_size} steps x "
        f"{args.num_envs} envs ~ {gib:.2f} GiB (+ {C}-dim context per env)")

  cbs = [
    CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.num_envs),
                       save_path=os.path.join(outdir, "checkpoints"),
                       name_prefix="model"),
    TensorNormSaveCallback(os.path.join(outdir, "checkpoints"),
                           save_freq_steps=args.checkpoint_every),
  ]
  if args.load:
    _warm_start(model, args.load, args.device)
    cbs.append(NormFreezeCallback(args.norm_freeze_steps))
  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=f"{args.task}_shac", config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
  _vtask = args.task + "_video"
  _vtask = _vtask if _vtask in list_tasks() else args.task
  cbs.append(VideoWandbCallback(
    lambda: make_tensor(_vtask, 8, args.device, render_mode="rgb_array",
                        end_criterion=args.end_criterion),
    interval=args.video_interval, video_seconds=args.video_seconds,
    fps=args.video_fps, outdir=outdir,
    to_wandb=args.video_wandb and not args.no_wandb,
    herd=None if args.video_herd < 0 else args.video_herd))

  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs),
              log_interval=args.log_interval)
  model.save(os.path.join(outdir, "final_model.zip"))
  model.env.save(os.path.join(outdir, "tensornormalize.pt"))
  print(f"[done] {outdir}")


if __name__ == "__main__":
  main()
