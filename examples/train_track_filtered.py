"""Train the squat-tracking task policy (PPO + AMP), bare or inside a WM-tube filter.

    python examples/train.py --config configs/g1_track_squat_ampppo.yaml --arm t70_wm_d1 \
        --seed 2 --stage pilot
    python examples/train_track_filtered.py --config configs/g1_track_squat_ampppo.yaml \
        --arm cbf_bare --seed 0 --stage final --out runs/track_squat/final/cbf_bare/s0

``--arm`` (examples/track_common.ARMS) picks the world and the filter; the task seed
picks the fallback / tube seed too unless ``--fb-seed`` says otherwise. The filter
is part of the env (the user's decision): PPO stores its own proposal and log-prob,
the reward and the style reward come from the executed transition, there is no
intervention penalty. The action-rate penalty sits on the PPO loss over the
proposals' means (``--action-rate-coef``): the task's reward has no action-rate
term, since one on the executed actions would charge every hand-over.

Writes into ``--out`` (default runs/track_squat/<stage>/<arm>/s<seed>):
final_model.zip (loads with a plain CumulativePPO1P.load), tensornormalize.pt,
amp_disc.pt, config.yaml, track_meta.json (arm, binding check, fallback and
expert provenance, w_style), g1_sysid.json (the thermal world, as the SHAC runs
record it), checkpoints/{model_<n>_steps.zip, tensornorm_<n>.pt, amp_disc_<n>.pt}
on one grid, and TensorBoard events under track_1/.

``--smoke`` adds end-of-run self-checks (metric keys present, files written,
widths) and exits non-zero if any fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _run_config import dump_config, merge_config  # noqa: E402

import track_common as tc  # noqa: E402

DEFAULTS = dict(num_envs=4096, steps=1_000_000_000, n_steps=24, lr=1e-3, desired_kl=0.01,
                ent_coef=0.005, vf_coef=1.0, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                net="512,256,128", init_std=0.5, std_floor=0.05, std_ceil=1.0,
                action_rate_coef=0.05, checkpoint_every=25_000_000)


def _parser():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--config", default=None)
  p.add_argument("--arm", required=True, choices=sorted(tc.ARMS))
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--fb-seed", type=int, default=None,
                 help="fallback / tube seed (default: the task seed, the user's pairing)")
  p.add_argument("--conf", type=float, default=0.95, help="tube coverage of the filter")
  p.add_argument("--stage", default="pilot", help="pilot / final / smoke (the run dir's level)")
  p.add_argument("--out", default=None, help="default runs/track_squat/<stage>/<arm>/s<seed>")
  p.add_argument("--device", default="cuda:0")
  for k, v in DEFAULTS.items():
    if k == "net":
      p.add_argument("--net", default=v, help="hidden sizes of pi and vf, comma-separated")
    else:
      p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
  p.add_argument("--w-style", default="auto",
                 help="style weight in reward-weight units, or 'auto' = --w-style-frac x the "
                      "task's positive reward weights")
  p.add_argument("--w-style-frac", type=float, default=0.4)
  p.add_argument("--amp-expert", default="data/g1_amp_expert_lafan_train_v1.pt")
  p.add_argument("--amp-d-steps", type=int, default=20)
  p.add_argument("--amp-batch", type=int, default=4096)
  p.add_argument("--amp-replay", type=int, default=1_000_000)
  p.add_argument("--amp-lr", type=float, default=1e-4)
  p.add_argument("--no-amp", action="store_true", help="tracking reward only (diagnostic)")
  p.add_argument("--no-coverage-probe", action="store_true")
  p.add_argument("--env-override", action="append", default=None, metavar="KEY=VAL",
                 help="forwarded to the task's cfg builder (repeatable)")
  p.add_argument("--wandb-project", default="track_squat")
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--smoke", action="store_true", help="end-of-run self-checks")
  return p


class _Extras:
  """Factory for the callback that writes tensornorm_<n>.pt and amp_disc_<n>.pt on
  the SAME call grid as SB3's CheckpointCallback, so every model_<n>_steps.zip has
  its normalizer and discriminator under the same n."""

  @staticmethod
  def make(save_path: str, save_freq: int, amp: bool):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
      def _on_step(self) -> bool:
        if save_freq > 0 and self.n_calls % save_freq == 0:
          n = self.num_timesteps
          os.makedirs(save_path, exist_ok=True)
          self.model.get_env().save(os.path.join(save_path, f"tensornorm_{n}.pt"))
          if amp:
            self.model.save_amp(os.path.join(save_path, f"amp_disc_{n}.pt"))
        return True

    return _CB()


class _KeyRecorder:
  """Collects the logger keys seen at each rollout end (env/* metrics), for --smoke."""

  @staticmethod
  def make(seen: set):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
      def _on_rollout_end(self) -> None:
        seen.update(self.model.logger.name_to_value.keys())

      def _on_step(self) -> bool:
        return True

    return _CB()


def _jsonable(x):
  return json.loads(json.dumps(x, default=lambda o: repr(o)))


def main(argv=None):
  if argv is not None:
    sys.argv = [sys.argv[0], *argv]
  args = merge_config(_parser())
  world, filt_kind, dwell = tc.ARMS[args.arm]
  task = tc.task_of(world)
  fb_seed = args.seed if args.fb_seed is None else args.fb_seed
  out = args.out or tc.run_dir(args.stage, args.arm, args.seed)
  os.makedirs(out, exist_ok=True)
  dump_config(out, args)
  print(f"[track] arm {args.arm}: task {task}, filter {filt_kind or 'none'}"
        + (f" dwell {dwell:g} s, fallback/tube seed {fb_seed}, c={args.conf:g}"
           if filt_kind else "") + f"; seed {args.seed}; -> {out}")

  import torch
  import torch.nn as nn
  from safety_sb3 import CumulativePPO1P  # noqa: F401  (imported so a missing install fails early)
  from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

  from robot_safety_sandbox.amp.ppo import AMPPPO
  from robot_safety_sandbox.callbacks import StdFloorCallback

  t0 = time.time()
  # the env and the frozen fallback first: nothing below may draw from an RNG the
  # fallback load could have moved (B10; load_wm_tube also restores every RNG)
  env, info = tc.build_stack(
    task, args.num_envs, args.device, filter_kind=filt_kind, dwell_s=dwell, fb_seed=fb_seed,
    conf=args.conf, env_overrides=args.env_overrides,
    amp_w_style=None if args.no_amp else args.w_style, amp_w_style_frac=args.w_style_frac,
    coverage_probe=not args.no_coverage_probe)
  info.pop("bundle", None)
  print(f"[track] stack built in {time.time() - t0:.0f} s; obs {env.observation_space.shape}, "
        f"critic {env.critic_observation_space.shape}, action {env.action_space.shape} "
        f"in +-{float(env.action_space.high.max()):g}")

  net = [int(x) for x in str(args.net).split(",") if x.strip()]
  kw = dict(
    n_steps=args.n_steps, batch_size=args.num_envs * args.n_steps // 4, n_epochs=5,
    gamma=args.gamma, gae_lambda=args.gae_lambda, learning_rate=args.lr,
    # schedules independent of the total step count (C12): a pilot checkpoint at the
    # budget B is then a valid final run; and PPO's GAE value does not want the anneal
    gamma_anneal=False, adaptive_lr=True, desired_kl=args.desired_kl,
    ent_coef=args.ent_coef, vf_coef=args.vf_coef, clip_range=args.clip_range,
    max_grad_norm=1.0, normalize_obs=True, bootstrap_on_timeout=False,
    action_rate_coef=args.action_rate_coef,
    policy_kwargs=dict(log_std_init=math.log(args.init_std), activation_fn=nn.ELU,
                       net_arch=dict(pi=net, vf=net)),
    seed=args.seed, verbose=1, device=args.device, tensorboard_log=out)
  amp = not args.no_amp
  if amp:
    model = AMPPPO("MlpPolicy", env, amp=dict(
      expert_path=args.amp_expert, d_steps=args.amp_d_steps, batch=args.amp_batch,
      replay_size=args.amp_replay, lr=args.amp_lr), **kw)
    info["amp"] = model._amp_meta
  else:
    model = CumulativePPO1P("MlpPolicy", env, **kw)
  info.update(arm=args.arm, seed=args.seed, fb_seed=fb_seed, stage=args.stage,
              proposal_bound=tc.PROPOSAL_BOUND)
  with open(os.path.join(out, "track_meta.json"), "w") as f:
    json.dump(_jsonable(info), f, indent=1)
  if info.get("thermal_world") is not None:
    with open(os.path.join(out, "g1_sysid.json"), "w") as f:
      json.dump(info["thermal_world"], f, indent=1)

  ck = os.path.join(out, "checkpoints")
  freq = max(1, args.checkpoint_every // args.num_envs)
  seen: set = set()
  cbs = [CheckpointCallback(save_freq=freq, save_path=ck, name_prefix="model"),
         _Extras.make(ck, freq, amp),
         StdFloorCallback(args.std_floor, args.std_ceil),
         _KeyRecorder.make(seen)]
  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=f"{args.stage}_{args.arm}_s{args.seed}",
               config=_jsonable(vars(args)), sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs), tb_log_name="track")

  model.save(os.path.join(out, "final_model.zip"))
  model.get_env().save(os.path.join(out, "tensornormalize.pt"))
  if amp:
    model.save_amp(os.path.join(out, "amp_disc.pt"))
  print(f"[track] done: {model.num_timesteps} steps in {(time.time() - t0) / 3600:.2f} h -> {out}")
  if args.smoke:
    sys.exit(_smoke_checks(model, env, out, seen, filt_kind, amp))


def _smoke_checks(model, env, out, seen, filt_kind, amp) -> int:
  """End-of-run checks for the smoke (configs/sweeps/track_squat/smoke.sh)."""
  from safety_sb3 import CumulativePPO1P
  fails = []

  def check(name, ok, detail=""):
    print(f"[smoke] {name} {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
      fails.append(name)

  need = ["env/track/steps", "env/track/fail_per_1k", "env/track/err_body",
          "env/track/tw_max_mean", "env/safety/failures_total"]
  if filt_kind:
    need += ["env/safety/engaged_frac", "env/filter/engaged_frac"]
  missing = [k for k in need if k not in seen]
  check("metrics", not missing, f"missing {missing}" if missing else f"{len(seen)} keys")
  if amp:
    last = model.logger.name_to_value
    check("amp-update", "amp/d_loss" in last and math.isfinite(float(last["amp/d_loss"])),
          f"d_loss {last.get('amp/d_loss')}")
  files = ["final_model.zip", "tensornormalize.pt", "config.yaml", "track_meta.json"]
  files += ["amp_disc.pt"] if amp else []
  absent = [f for f in files if not os.path.isfile(os.path.join(out, f))]
  check("files", not absent, f"absent {absent}" if absent else "")
  plain = CumulativePPO1P.load(os.path.join(out, "final_model.zip"), device="cpu")
  check("plain-load", not hasattr(plain, "_amp_disc"))
  og = env.obs_groups()
  dims = {k: int(og[k].shape[1]) for k in ("actor", "critic", "task_policy", "task_critic", "amp")}
  check("dims", dims["actor"] == 127 and dims["critic"] == 171 and dims["amp"] == 68
        and int(env.critic_observation_space.shape[0]) == dims["task_critic"] + 2, str(dims))
  return 1 if fails else 0


if __name__ == "__main__":
  main()
