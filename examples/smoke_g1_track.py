"""Smoke of the squat-tracking task env (plan step 3c), one world per run.

    python examples/smoke_g1_track.py --task g1_track_squat_thermal70 --fb-seed 2
    python examples/smoke_g1_track.py --task g1_track_squat_thermal_cbf --fb-seed 2

Each check prints ``[smoke] <name> OK|FAIL ...``; the exit code is 1 if any fails.

  dims       groups actor 127 / critic 171 / task_policy 334 / task_critic 540 /
             amp 68, sim_context 33, action box +-6.5, the task's obs keys
  spawn      noise-free spawns at the first frame of every phase bin sit on the
             placed reference bodies (< 1e-4 m); f = start + 1 after the reset;
             last_action seeded with the PD target of the spawned pose; with
             spawn noise every joint stays 0.01 rad inside its hard range
  amp        the amp group of those spawns equals amp.features.amp_features of
             the reference state (root height = pelvis z - lower ankle-roll z
             from the reference's FK, as the expert builder measures it), < 1e-3
  advance    f moves by one per step (a reset env to its start + 1), one resample
             per episode, reference_finished exactly T-1-s steps after a start at
             s, the 14 s backstop never fires
  bus        env._zoo_track every step; terminations counted from it equal the
             env's safety/failures_total; every observation group finite
  fallback   the frozen SHAC fallback alone (noise-free spawns, deterministic,
             last-action columns clamped to +-3 as the filter feeds it), from the
             first frame of each of the 10 phase bins: an episode survives when
             it does not terminate within --horizon-s (or reaches the segment's
             end first). Bin 0 must reach --min-survival; the other bins are
             reported (they start off the fallback's support).
  throughput env steps/s with the fallback acting.

Runs anywhere make_tensor does (``--device cpu`` with a few envs per bin works,
slowly).
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

DIMS = {"actor": 127, "critic": 171, "task_policy": 334, "task_critic": 540, "amp": 68}
CONTEXT_DIM = 33


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--task", default="g1_track_squat_thermal70",
                 choices=("g1_track_squat_thermal70", "g1_track_squat_thermal_cbf"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--envs-per-bin", type=int, default=64)
  p.add_argument("--fb-seed", type=int, default=2)
  p.add_argument("--fallback-run", default=None,
                 help="default runs/shac_<world>/shac_s<fb-seed>/g1_recover_<world>_safetyshac1p")
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--min-survival", type=float, default=0.9)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default=None, help="write the results here (json)")
  return p.parse_args(argv)


def main(argv=None) -> int:
  a = _parse(argv)
  import robot_safety_sandbox as rss
  from robot_safety_sandbox.envs.g1_track.command import TASK_ACTION_BOUND
  from mjlab.utils.lab_api.math import quat_apply_inverse

  from robot_safety_sandbox.amp.features import amp_features
  from robot_safety_sandbox.envs.g1_track.mdp import (
    AMP_FOOT_BODIES, CAUSES, COMPLETE, CUT, N_PHASE_BINS)
  from robot_safety_sandbox.eval.policies import load_twin

  world = a.task.removeprefix("g1_track_squat_")
  fb_dir = a.fallback_run or os.path.join(
    _ZOO, "runs", f"shac_{world}", f"shac_s{a.fb_seed}", f"g1_recover_{world}_safetyshac1p")
  torch.manual_seed(a.seed)
  n = a.envs_per_bin * N_PHASE_BINS
  env = rss.make_tensor(a.task, num_envs=n, device=a.device)
  mj, dev = env.mj, env.device
  cmd = mj.command_manager.get_term("ref")
  robot, T = cmd.robot, cmd.T
  results: dict = {"task": a.task, "T": T, "num_envs": n}
  fails = []

  def report(name, ok, msg):
    print(f"[smoke] {name} {'OK' if ok else 'FAIL'} {msg}", flush=True)
    results[name] = {"ok": bool(ok), "msg": msg}
    if not ok:
      fails.append(name)

  # --- dims -------------------------------------------------------------------
  dims = {k: int(v.shape[1]) for k, v in env.obs_groups().items()}
  ctx = env.sim_context()
  box = float(env.action_space.high[0])
  report("dims", dims == DIMS and ctx is not None and int(ctx.shape[1]) == CONTEXT_DIM
         and box == TASK_ACTION_BOUND and env.obs_key == "task_policy"
         and env.critic_obs_key == "task_critic",
         f"groups {dims}, context {None if ctx is None else int(ctx.shape[1])}, box +-{box:g}, "
         f"obs {env.obs_key} / {env.critic_obs_key}")

  # --- spawn ------------------------------------------------------------------
  bin_start = torch.arange(N_PHASE_BINS, device=dev) * T // N_PHASE_BINS
  starts = bin_start.repeat_interleave(a.envs_per_bin)
  cmd.spawn_noise = False
  cmd.start_override = starts
  env.reset()
  pos, _ = cmd.placed_bodies(cmd.start_f)
  err = float((cmd.robot_body_pos_w - pos).norm(dim=-1).max())
  term = mj.action_manager.get_term("joint_pos")
  seed = ((robot.data.joint_pos[:, term.target_ids] - term.offset) / term.scale).clamp(
    -TASK_ACTION_BOUND, TASK_ACTION_BOUND)
  seed_err = float((mj.action_manager.action - seed).abs().max())
  report("spawn", err < 1e-4 and bool((cmd.start_f == starts).all())
         and bool((cmd.f == starts + 1).all()) and seed_err < 1e-4,
         f"max body error {err:.2e} m over starts {bin_start.tolist()}, seeded action "
         f"error {seed_err:.1e}")
  # the amp group of a spawned reference state against amp_features of that
  # state from the reference's own FK (B12: the expert builder's side), so an
  # env/expert definition mismatch (root height, frames, joint order) shows here
  s = cmd.start_f
  feet = [cmd.ref.body_names.index(b) for b in AMP_FOOT_BODIES]
  rq = cmd.ref_root_quat[s]
  want = amp_features(cmd.ref_root_pos[s, 2] - cmd.ref_body_pos[s][:, feet, 2].amin(1), rq,
                      quat_apply_inverse(rq, cmd.ref_root_lin[s]),
                      quat_apply_inverse(rq, cmd.ref_root_ang[s]), cmd.q[s], cmd.qd[s])
  amp_err = float((env.obs_groups()["amp"] - want).abs().max())
  report("amp", amp_err < 1e-3, f"amp group vs amp_features of the reference state: max "
         f"|diff| {amp_err:.1e}")
  cmd.spawn_noise = True
  env.reset()
  lim = robot.data.joint_pos_limits
  q = robot.data.joint_pos
  inside = bool(((q >= lim[..., 0] + 0.01 - 1e-5) & (q <= lim[..., 1] - 0.01 + 1e-5)).all())
  pos, _ = cmd.placed_bodies(cmd.start_f)
  moved = float((cmd.robot_body_pos_w - pos).norm(dim=-1).mean())
  report("spawn_noise", inside and 0.0 < moved < 0.2,
         f"joints inside the shrunk range: {inside}; mean body offset {moved:.3f} m")

  # --- the fallback from every phase bin, with the advance / bus checks --------
  model, norm = load_twin(os.path.join(fb_dir, "final_model.zip"), dev)
  policy = model.policy
  policy.set_training_mode(False)
  la_start, la_size, _ = env._obs_layout("actor")["actions"]
  la = slice(la_start, la_start + la_size)

  def fallback_action():
    o = torch.nan_to_num(env.obs_groups()["actor"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    o = o.clone()
    o[:, la] = o[:, la].clamp(-3.0, 3.0)
    with torch.no_grad():
      return policy._predict(norm(o), deterministic=True).clamp(-3.0, 3.0)

  cmd.spawn_noise = False
  cmd.start_override = starts
  env.reset()
  env.metrics()
  fail0 = env._failures_total
  horizon = int(round(a.horizon_s / mj.step_dt))
  first = torch.zeros(n, dtype=torch.bool, device=dev)       # first episode over
  failed = torch.zeros(n, dtype=torch.bool, device=dev)
  overheat = torch.zeros(n, dtype=torch.bool, device=dev)
  bad = {"advance": 0, "resample": 0, "finish": 0, "cut": 0, "nan": 0, "frame": 0}
  bus_fail, finished = 0, 0
  t0 = time.time()
  for _ in range(horizon):
    prev_f = cmd.f.clone()
    obs, r, dones, timeouts, _ = env.step_tensor(fallback_action())
    bus = mj._zoo_track
    done, ep, st = bus["done"], bus["ep"], bus["step"]
    cause = ep["cause"].long()
    bad["frame"] += int((st["frame"] != prev_f).sum())
    bad["advance"] += int(((cmd.f != prev_f + 1) & ~done).sum())
    bad["advance"] += int(((cmd.f != cmd.start_f + 1) & done).sum())
    bad["resample"] += int((cmd.command_counter != 1).sum())
    complete = done & (cause == COMPLETE)
    bad["finish"] += int((ep["steps"][complete] != T - 1 - ep["start_f"][complete]).sum())
    bad["cut"] += int((done & (cause == CUT)).sum())
    term_now = done & (cause != COMPLETE) & (cause != CUT)
    bus_fail += int(term_now.sum())
    finished += int(complete.sum())
    bad["nan"] += sum(int((~torch.isfinite(v)).sum()) for v in env.obs_groups().values())
    new = done & ~first
    failed |= new & term_now
    overheat |= new & ep["overheat"]
    first |= done
    if not torch.equal(dones, done):
      bad["advance"] += 1
  if str(dev).startswith("cuda"):
    torch.cuda.synchronize()
  fps = n * horizon / (time.time() - t0)
  m = env.metrics()
  report("advance", bad["advance"] == 0 and bad["resample"] == 0 and bad["frame"] == 0
         and bad["finish"] == 0 and bad["cut"] == 0 and finished > 0,
         f"{finished} episodes reached the segment's end (the last bins must, for the "
         f"timing check to bite); violations {bad}")
  report("bus", bus_fail == m["safety/failures_total"] - fail0 and bad["nan"] == 0,
         f"terminations on the bus {bus_fail} vs safety/failures_total "
         f"{m['safety/failures_total'] - fail0:g}; non-finite obs {bad['nan']}")
  surv = (~failed).view(N_PHASE_BINS, a.envs_per_bin).float().mean(1)
  hot = overheat.view(N_PHASE_BINS, a.envs_per_bin).float().mean(1)
  results["fallback_survival"] = {int(s): float(v) for s, v in zip(bin_start, surv)}
  results["fallback_overheat"] = {int(s): float(v) for s, v in zip(bin_start, hot)}
  for b in range(N_PHASE_BINS):
    print(f"[smoke]   bin {b} (frame {int(bin_start[b])}): fallback survives "
          f"{float(surv[b]):.3f}, overheat events {float(hot[b]):.3f}")
  report("fallback", float(surv[0]) >= a.min_survival,
         f"{fb_dir}: bin 0 survives {float(surv[0]):.3f} over {a.horizon_s:g} s "
         f"(need {a.min_survival:g}); all bins {[round(float(v), 3) for v in surv]}")
  report("throughput", True, f"{fps:.0f} env-steps/s at {n} envs ({a.device})")
  results["fps"] = fps
  results["causes"] = list(CAUSES)
  if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
      json.dump(results, fh, indent=1)
  print(f"[smoke] {'PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
