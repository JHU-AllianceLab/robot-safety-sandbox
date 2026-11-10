"""Harvest a FRONTIER start-state bank from a trained push-recovery policy.

    python examples/harvest_g1_frontier_states.py \
        --policy runs/g1_recover_push_anneal/g1_recover_push/final_model.zip \
        --samples 100000 --out data/g1_frontier_states.pt

The walker bank (``harvest_g1_recover_states.py``) holds states of the WALKER
around its pushes. Once a recovery policy is trained, the states that still
carry signal are the ones around ITS pushes at the magnitudes it survives
about half the time -- its own safety frontier, which moves as it improves.
This script runs the policy in the push env (``g1_recover_push_fast``: the
recover bank spawn plus a random push 0.1-0.4 s later) with the per-env push
magnitude drawn uniformly in ``[--min-scale, 1] x --force-max`` per episode,
and for every push that ends with the robot still up:

1. waits ``k`` control steps (``k`` drawn from ``--delays``), snapshots the
   state (``states.snapshot``: pose, twist, joints, last action) together
   with the policy's own value estimate at that state;
2. keeps the policy in control for ``--window-s`` and labels the row: ``fell``
   with ``time_to_fall`` if the episode ended on a fall, survived if the robot
   came to rest (the ``recovered`` truncation, envs/g1_recover/rest.py) or the
   window ran out upright. Rows cut by the episode cap are dropped.

Rows are written with the walker bank's schema (``states.FIELDS``) as
``kind = KIND_PUSHED``, ``push_severity`` = the per-env scale (force =
severity x force_max), plus a ``value`` column (ignored by the spawn, kept for
analysis), so the training spawn consumes the file unchanged through
``env_overrides.dataset``: the ``pushed`` source (survived rows, sorted by
severity) and the ``fell`` source (fell rows, sorted by time-to-fall) with
their level curricula. Set ``weights`` to lean on those two.

``--kind handover`` writes the same rows as ``kind = KIND_HANDOVER`` instead
(the P-pre-self start states: post-push states of the recovery policy
itself), which the spawn's ``handover`` source draws and never follows with a
random push (the row's push already happened). They also carry
``push_impulse`` (force x the push duration, N*s), ``handover_delay_s`` (push
onset -> snapshot: the push duration plus the delay), ``value0`` (= the
``value`` column) and an empty pending push (``push_force_h`` 0,
``push_left_steps`` 0: the snapshot is taken after the push has ended).

THE VALUE at the snapshot is the policy's critic on the observation the
critic was trained on: the actor obs for a symmetric critic, the privileged
CRITIC obs group with its own normalizer for an asymmetric one
(``--privileged-critic`` runs; ``envs/g1_recover/labels.PolicyValue``, which
also builds the env with ``critic_obs_key="critic"`` then).

SPAWN. ``--dataset`` (one bank or several), ``--weights`` (a JSON source ->
weight mapping) and ``--sampler`` (the spawn's row sampler) pick what the
harvest env starts from, e.g. the LAFAN1 frames::

    --task g1_recover_lafan --dataset data/g1_recover_states_lafan1_train.pt \
    --weights '{"mocap": 1, "nominal": 0, "clean": 0, "pushed": 0, "fell": 0,
                "synthetic": 0}' --sampler uniform

The end-of-run table (survival by severity quartile and by delay) says where
the frontier sits for the harvested policy; re-harvest every ~100M steps so the
bank tracks the policy being trained.
"""

from __future__ import annotations

import argparse
import json
import os
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

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.labels import PolicyValue  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_HANDOVER, KIND_PUSHED, save_bank, snapshot, speed_of)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402

IDLE, DELAY, WINDOW = 0, 1, 2


def _parse():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True,
                 help="a safety_sb3 checkpoint of a push-recovery policy (its "
                      "tensornormalize.pt / tensornorm_<steps>.pt is found next to it)")
  p.add_argument("--task", default="g1_recover_push_fast",
                 help="a random_push task (the push lands shortly after the spawn)")
  p.add_argument("--dataset", nargs="+", default=None,
                 help="start-state bank(s) the harvest env spawns from (default: "
                      "the task's default walker bank)")
  p.add_argument("--weights", default=None,
                 help="JSON source -> weight mapping for the harvest env's spawn "
                      "(e.g. '{\"mocap\": 1, \"nominal\": 0}'; unnamed sources "
                      "keep their defaults)")
  p.add_argument("--sampler", default=None,
                 help="the spawn's row sampler (ladder / uniform / weighted / "
                      "online; default: the task's)")
  p.add_argument("--kind", choices=("pushed", "handover"), default="pushed",
                 help="write the rows as KIND_PUSHED (walker-bank sources) or "
                      "KIND_HANDOVER (the spawn's handover source)")
  p.add_argument("--out", default=os.path.join(_ZOO, "data", "g1_frontier_states.pt"))
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--samples", type=int, default=100000)
  p.add_argument("--force-max", type=float, default=300.0,
                 help="push magnitude at severity 1, N")
  p.add_argument("--min-scale", type=float, default=0.2,
                 help="smallest per-env severity drawn (x force-max)")
  p.add_argument("--delays", type=int, nargs="+", default=[0, 5, 10, 20],
                 help="control steps between the end of a push and the snapshot")
  p.add_argument("--window-s", type=float, default=2.0,
                 help="label window after the snapshot, s")
  p.add_argument("--episode-s", type=float, default=6.0,
                 help="episode cap in the harvest env (rows cut by it are dropped)")
  p.add_argument("--init-level", type=int, default=2,
                 help="the harvest env's spawn curriculum start level")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args()


def harvest(env, model, norm, value, args) -> dict[str, torch.Tensor]:
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  push = getattr(env, "_push", None)
  if push is None:
    raise SystemExit(f"task {args.task!r} has no random push; use a random_push task")
  env.force_max = float(args.force_max)
  window = max(1, round(args.window_s / dt))
  delays = torch.tensor([int(k) for k in args.delays], device=dev)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  term_mgr = mj.termination_manager
  has_rest = "recovered" in term_mgr.active_terms

  def draw_scale(mask):
    k = int(mask.sum())
    if k:
      env.force_scale[mask] = args.min_scale + (1.0 - args.min_scale) * torch.rand(k, device=dev)

  env.force_scale = torch.zeros(n, device=dev)
  draw_scale(torch.ones(n, dtype=torch.bool, device=dev))

  mode = torch.zeros(n, dtype=torch.long, device=dev)
  count = torch.zeros(n, dtype=torch.long, device=dev)
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  delay_of = torch.zeros(n, dtype=torch.long, device=dev)
  pend: dict[str, torch.Tensor] = {}
  pend_sev = torch.zeros(n, device=dev)
  pend_val = torch.zeros(n, device=dev)
  pend_delay = torch.zeros(n, dtype=torch.long, device=dev)
  kind = KIND_HANDOVER if args.kind == "handover" else KIND_PUSHED
  rows: dict[str, list] = {f: [] for f in FIELDS}
  rows["value"], rows["delay_steps"] = [], []
  ids_all = torch.arange(n, device=dev)
  got, dropped, t0, it = 0, 0, time.time(), 0

  def take(mask, obs):
    s = snapshot(mj)
    s["command"] = torch.zeros(n, 3, device=dev)
    s["phase_step"] = torch.zeros(n, dtype=torch.long, device=dev)
    for k, v in s.items():
      if k not in pend:
        pend[k] = torch.zeros_like(v)
      pend[k][mask] = v[mask]
    # the critic's own input (the privileged group for an asymmetric critic)
    crit = env.critic_obs() if value.asymmetric else None
    pend_val[mask] = value(obs[mask], crit[mask] if crit is not None else None)
    pend_sev[mask] = env.force_scale[mask]
    pend_delay[mask] = delay_of[mask]
    elapsed[mask] = 0

  def emit(mask, fell):
    nonlocal got
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
      return
    for k in ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos",
              "joint_vel", "action", "command", "phase_step"):
      rows[k].append(pend[k][idx])
    rows["kind"].append(torch.full((idx.numel(),), kind, dtype=torch.long, device=dev))
    rows["fell"].append(fell[idx])
    rows["time_to_fall"].append(torch.where(
      fell[idx], elapsed[idx].float() * dt,
      torch.full_like(elapsed[idx], float("nan"), dtype=torch.float)))
    rows["push_severity"].append(pend_sev[idx])
    rows["difficulty"].append(torch.zeros(idx.numel(), device=dev))
    rows["speed"].append(speed_of(pend["lin_vel_h"][idx]))
    rows["env_id"].append(ids_all[idx])
    rows["value"].append(pend_val[idx])
    rows["delay_steps"].append(pend_delay[idx])
    got += int(idx.numel())

  obs = env.reset()
  ones, zeros = (torch.ones(n, dtype=torch.bool, device=dev),
                 torch.zeros(n, dtype=torch.bool, device=dev))
  while got < args.samples:
    nobs = norm(obs)
    # 1. snapshots due now (k steps after a push ended)
    snap = (mode == DELAY) & (count <= 0)
    if bool(snap.any()):
      take(snap, obs)
      mode[snap] = WINDOW
      count[snap] = window
    # 2. the policy acts, the env steps (the bridge applies the push)
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(nobs, deterministic=True), low, high)
    obs, _g, dones, timeouts, _l = env.step_tensor(act)
    reset_policy_memory(model.policy, dones)       # a recurrent policy restarts
    terminated = dones & ~timeouts
    rested = term_mgr.get_term("recovered") if has_rest else zeros
    count -= 1
    elapsed += 1
    # 3. labels
    in_win = mode == WINDOW
    emit(in_win & terminated, ones)                       # fell inside the window
    emit(in_win & (timeouts & rested), zeros)             # came to rest
    emit(in_win & ~dones & (count <= 0), zeros)           # upright at the window's end
    cut = in_win & timeouts & ~rested                     # the cap: no label
    dropped += int(cut.sum())
    # 4. a push that just ended with the episode still alive starts a delay
    # "the push acted on this step and has none left". NOT age == duration:
    # age counts calls since the reset too, so that test also fires on the
    # duration-th QUIET step whenever the first onset comes later (68 % of
    # episodes with first_delay_s (0.1, 0.4)), snapshotting before the push
    # as if after it; active() also covers scripted pushes of any length.
    ended = push.active() & (push.t_left == 0) & ~dones
    arm = (mode == IDLE) & ended
    mode[in_win & ((count <= 0) | dones)] = IDLE
    mode[dones] = IDLE
    if bool(arm.any()):
      # Arm only as many as the bank still needs. A fall is labeled within the
      # window's first few tenths of a second, a survivor only at its end, so
      # stopping the moment `samples` rows exist would drop the captures in
      # flight -- nearly all survivors -- and bias the bank toward falls.
      # Instead labeled + in-flight rows never exceed `samples`, the loop runs
      # until exactly `samples` are labeled, and a capture cut by the cap frees
      # its slot for a new one.
      inflight = int(((mode == DELAY) | (mode == WINDOW)).sum())
      need = args.samples - got - inflight
      ids = arm.nonzero(as_tuple=False).squeeze(-1)
      arm = torch.zeros_like(arm)
      if need > 0:
        arm[ids[:need]] = True
    if bool(arm.any()):
      delay_of[arm] = delays[torch.randint(0, delays.numel(), (int(arm.sum()),), device=dev)]
      count[arm] = delay_of[arm]
      mode[arm] = DELAY
    draw_scale(dones)                                     # a new episode, a new severity
    it += 1
    if it % 250 == 0:
      rate = it / max(time.time() - t0, 1e-6)
      print(f"[harvest] step {it}  rows {got}/{args.samples}  dropped {dropped}  "
            f"({rate:.1f} steps/s)", flush=True)
  out = {k: torch.cat(v)[:args.samples] for k, v in rows.items()}
  if args.kind == "handover":
    # P-pre-self rows: the push is over at the snapshot (no pending push); its
    # impulse is the per-env force x the random push's duration
    m = out["kind"].numel()
    dur_s = float(push.duration) * dt
    out["push_impulse"] = out["push_severity"] * float(args.force_max) * dur_s
    out["handover_delay_s"] = (float(push.duration) + out["delay_steps"].float()) * dt
    out["value0"] = out["value"].clone()
    out["push_force_h"] = torch.zeros(m, 3, device=dev)
    out["push_left_steps"] = torch.zeros(m, dtype=torch.long, device=dev)
  return out


def _report(bank: dict, args) -> dict:
  sev, fell = bank["push_severity"], bank["fell"]
  stats = {}
  edges = torch.linspace(args.min_scale, 1.0, 5, device=sev.device)
  print("[frontier] survival by severity quartile (force = severity x "
        f"{args.force_max:g} N):")
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sev >= lo) & (sev <= hi)
    if int(m.sum()):
      s = 1.0 - float(fell[m].float().mean())
      stats[f"surv_sev_{float(lo):.2f}_{float(hi):.2f}"] = s
      print(f"    {float(lo)*args.force_max:5.0f}-{float(hi)*args.force_max:5.0f} N: "
            f"{int(m.sum()):6d} rows, survival {s:.2f}")
  print("[frontier] survival by delay after the push:")
  for k in sorted(set(bank["delay_steps"].tolist())):
    m = bank["delay_steps"] == k
    s = 1.0 - float(fell[m].float().mean())
    stats[f"surv_delay_{k}"] = s
    print(f"    {k:3d} steps: {int(m.sum()):6d} rows, survival {s:.2f}")
  ttf = bank["time_to_fall"][fell]
  if ttf.numel():
    q = torch.quantile(ttf, torch.tensor([0.1, 0.5, 0.9], device=ttf.device)).tolist()
    stats["time_to_fall_q10_q50_q90"] = q
    print(f"[frontier] time-to-fall of the fell rows: q10 {q[0]:.2f} s, "
          f"q50 {q[1]:.2f} s, q90 {q[2]:.2f} s")
  v = bank["value"]
  print(f"[frontier] policy value at the snapshot: survived mean {float(v[~fell].mean()):+.3f}, "
        f"fell mean {float(v[fell].mean()):+.3f}")
  return stats


def main():
  args = _parse()
  torch.manual_seed(args.seed)
  dev = args.device
  overrides = dict(episode_length_s=float(args.episode_s), rest_terminate=True,
                   rest_dwell_s=0.4, rest_arm_s=0.8, init_level=int(args.init_level))
  if args.dataset:
    overrides["dataset"] = args.dataset[0] if len(args.dataset) == 1 else list(args.dataset)
  if args.weights:
    overrides["weights"] = {k: float(v) for k, v in json.loads(args.weights).items()}
  if args.sampler:
    overrides["sampler"] = args.sampler
  # the policy first: an asymmetric critic needs the env's critic group
  model, norm = load_twin(args.policy, dev)
  model.policy.set_training_mode(False)
  value = PolicyValue(model, args.policy, dev)
  env = make_tensor(args.task, args.num_envs, dev, cfg_overrides=overrides,
                    critic_obs_key=value.critic_obs_key)
  print(f"[frontier] policy {args.policy} in {args.task} x {args.num_envs}; "
        f"push {args.min_scale * args.force_max:.0f}-{args.force_max:.0f} N, "
        f"delays {args.delays} steps, window {args.window_s} s")
  bank = harvest(env, model, norm, value, args)
  stats = _report(bank, args)
  bank["meta"] = {
    "source": "harvest_g1_frontier_states.py", "policy": os.path.abspath(args.policy),
    "task": args.task, "force_max": args.force_max, "min_scale": args.min_scale,
    "delays": list(args.delays), "window_s": args.window_s, "rows": int(bank["kind"].numel()),
    "fell_rows": int(bank["fell"].sum()), "stats": stats,
  }
  if args.kind != "pushed" or args.weights or args.sampler:
    bank["meta"].update(kind=args.kind, weights=args.weights, sampler=args.sampler,
                        dataset=args.dataset, value_critic=("asymmetric" if value.asymmetric
                                                            else "symmetric"))
  save_bank(bank, args.out)
  print(f"[frontier] wrote {bank['meta']['rows']} rows ({bank['meta']['fell_rows']} fell) "
        f"to {args.out}")


if __name__ == "__main__":
  main()
