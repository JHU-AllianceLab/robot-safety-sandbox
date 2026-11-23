"""Harvest the g1_cool_thermal70_fb spawn bank: states of the thermal70 SHAC
FALLBACK while it runs after a minor SONIC handover (the user, 2026-11-19: "it
should be able to transit smoothly from the fallback policy, so can you sample a
dataset from fallback policy").

Every env spawns at a SONIC handover row kicked <= 1 m/s
(``env_cfg.FB_SOURCE_BANK``) in the g1_cool_thermal70_fb world -- its start
temperatures (hot ankles / waist + 1-4 random hot motors) and its 0.2-1 m/s
pushes -- and the fallback acts for ``--duration-s``. From ``--first-s`` on, each
env is snapshotted every ``--every-s`` (a random phase per env), while it is
still in its first episode and no push is acting or armed. A row is the
restorable state (``states.snapshot``: base pose / twist in the heading frame,
joints, and the fallback's LAST APPLIED ACTION, which the spawn seeds the action
history with, so the cooling policy's first-step jump is scored against it).

Labels: ``fell`` / ``time_to_fall`` = the fallback fell (or overheated) within
``--fall-window-s`` after the snapshot; such rows are ``start_ok`` False (the
fallback itself was losing them), the rest are spawnable. ``kind`` =
KIND_HANDOVER (the spawn's row-drawn "handover" pool), ``t`` = seconds the
fallback had run, ``src_row`` / ``category`` / ``push_impulse`` = the SONIC row
it started from. Extra columns ``Tw`` / ``Th`` / ``Ts`` (the temperatures at the
snapshot; the spawn ignores them and redraws).

    python examples/harvest_cool_fallback_states.py --num-envs 4096 --device cuda:0

GPU (sim + policy). Never overwrites an existing output (--force to).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

FB = "runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip"
TASK = "g1_cool_thermal70_fb"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--fallback", default=FB)
  p.add_argument("--task", default=TASK)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--duration-s", type=float, default=15.0)
  p.add_argument("--first-s", type=float, default=0.5)
  p.add_argument("--every-s", type=float, default=1.0)
  p.add_argument("--fall-window-s", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None, help="default: the task's FB_BANK")
  p.add_argument("--force", action="store_true")
  a = p.parse_args()

  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from cool_curve import _last_action_slice, load_policy
  from robot_safety_sandbox.base import MjlabTensorSafetyEnv
  from robot_safety_sandbox.envs.g1_cool.env_cfg import FB_BANK, FB_SOURCE_BANK
  from robot_safety_sandbox.envs.g1_recover.spawn import RecoverSpawn
  from robot_safety_sandbox.envs.g1_recover.states import (
    FIELDS, KIND_HANDOVER, save_bank, snapshot, speed_of)
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  from robot_safety_sandbox.registry import CUMULATIVE, spec

  out = a.out or FB_BANK
  if os.path.exists(out) and not a.force:
    raise SystemExit(f"{out} exists (--force to overwrite)")
  if not os.path.exists(FB_SOURCE_BANK):
    raise SystemExit(f"no source bank {FB_SOURCE_BANK} (examples/build_cool_bank.py --max-dv 1.0)")
  dev = a.device
  s = spec(a.task)

  def builder(play: bool = False, **kw):
    return s.cfg_builder(play=play, dataset=FB_SOURCE_BANK, **kw)

  kw = dict(s.kwargs)
  kw.setdefault("dense_reward", s.mode == CUMULATIVE)
  n = int(a.num_envs)
  env = MjlabTensorSafetyEnv(
    n, dev, cfg_builder=builder, margin_fn=s.margin_fn, ctrl_dim=s.ctrl_dim,
    dstb_dim=s.dstb_dim, adversary=False, end_criterion=s.end_criterion,
    cfg_overrides={"episode_length_s": a.duration_s + 5.0}, **kw)
  mj = env.mj
  push = env._push
  dt = float(mj.step_dt)
  steps = int(round(a.duration_s / dt))
  first = int(round(a.first_s / dt))
  every = max(1, int(round(a.every_s / dt)))
  win = int(round(a.fall_window_s / dt))
  bound = float(env.action_bound)
  la = _last_action_slice(mj, env.obs_key)
  st = thermal_state(mj)
  model, norm, twin = load_policy(a.fallback, dev)
  spawn = next(c.func for cfgs in mj.event_manager._mode_term_cfgs.values() for c in cfgs
               if isinstance(getattr(c, "func", None), RecoverSpawn))
  src_bank = spawn.bank

  torch.manual_seed(a.seed)
  obs = env.reset()
  src_row = mj._zoo_spawn_row.clone()
  if bool((src_row < 0).any()):
    raise SystemExit("an env spawned off the source bank (nominal stance?)")
  phase = torch.randint(0, every, (n,), device=dev)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  end_k = torch.full((n,), -1, dtype=torch.long, device=dev)
  timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
  snaps: dict[str, list] = {}
  n_busy = 0
  tm = mj.termination_manager
  for k in range(steps + 1):
    if k >= first:
      due = alive & ((k - first - phase) % every == 0)
      busy = torch.zeros_like(due)
      if push is not None:
        busy = (push.t_left > 0) | push._s_armed
      n_busy += int((due & busy).sum())
      m = due & ~busy
      if bool(m.any()):
        sn = snapshot(mj)
        ids = torch.nonzero(m).squeeze(1)
        for key, v in sn.items():
          snaps.setdefault(key, []).append(v[ids].clone())
        snaps.setdefault("env_id", []).append(ids.clone())
        snaps.setdefault("k", []).append(torch.full_like(ids, k))
        for key, v in (("Tw", st.Tw), ("Th", st.Th), ("Ts", st.Ts)):
          snaps.setdefault(key, []).append(v[ids].clone())
    if k == steps:
      break
    o = obs
    if twin:
      o = obs.clone()
      o[:, la] = o[:, la].clamp(-bound, bound)
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(norm(o), deterministic=True), -bound, bound)
    obs, _r, dones, _to, _l = env.step_tensor(act)
    new = dones.bool() & alive
    if bool(new.any()):
      only_to = torch.ones_like(new)
      for t in tm.active_terms:
        v = tm.get_term(t).bool()
        if t != "time_out":
          only_to &= ~v
      end_k[new] = k + 1
      timed_out |= new & only_to
      alive &= ~new
    if (k + 1) % 250 == 0:
      print(f"[harvest] t={(k + 1) * dt:5.1f} s: {int(alive.sum())}/{n} still under the "
            f"fallback, {sum(int(x.numel()) for x in snaps.get('env_id', []))} rows", flush=True)

  cat = {key: torch.cat(v) for key, v in snaps.items()}
  N = int(cat["env_id"].numel())
  env_id, ks = cat["env_id"], cat["k"]
  ek = end_k[env_id]
  failed = (ek >= 0) & ~timed_out[env_id]
  fell = failed & (ek - ks <= win)
  ttf = torch.where(failed, (ek - ks).float() * dt, torch.full((N,), float("nan"), device=dev))
  sr = src_row[env_id]
  bank = {key: cat[key] for key in ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h",
                                    "joint_pos", "joint_vel", "action")}
  bank.update(
    command=torch.zeros(N, 3, device=dev),
    phase_step=torch.zeros(N, dtype=torch.long, device=dev),
    kind=torch.full((N,), KIND_HANDOVER, dtype=torch.long, device=dev),
    fell=fell,
    time_to_fall=ttf,
    push_severity=torch.zeros(N, device=dev),
    difficulty=torch.zeros(N, device=dev),
    speed=speed_of(cat["lin_vel_h"]),
    env_id=env_id,
    start_ok=~fell,
    t=ks.float() * dt,
    src_row=sr,
    category=src_bank["category"][sr].long(),
    push_impulse=src_bank["push_impulse"][sr].float(),
    Tw=cat["Tw"], Th=cat["Th"], Ts=cat["Ts"],
  )
  missing = [f for f in FIELDS if f not in bank]
  if missing:
    raise SystemExit(f"bank misses {missing}")
  src_meta = src_bank.get("meta") or {}
  bank["meta"] = {
    "g1_cool_fb": {"fallback": os.path.abspath(a.fallback), "task": a.task,
                   "source_bank": os.path.abspath(FB_SOURCE_BANK), "num_envs": n,
                   "duration_s": a.duration_s, "first_s": a.first_s, "every_s": a.every_s,
                   "fall_window_s": a.fall_window_s, "seed": a.seed,
                   "skipped_mid_push": n_busy},
    "categories": list(src_meta.get("categories", [])),
  }
  save_bank(bank, out)
  fail_env = int(((end_k >= 0) & ~timed_out).sum())
  sp = bank["speed"]
  print(f"[harvest] wrote {out}: {N} rows from {n} envs ({int((~fell).sum())} spawnable; "
        f"{int(fell.sum())} within {a.fall_window_s:g} s of a fallback failure; "
        f"{n_busy} snapshots skipped mid-push)", flush=True)
  print(f"[harvest] fallback failed in {fail_env}/{n} envs within {a.duration_s:g} s; row speed "
        f"median {float(sp.median()):.2f} m/s, p90 {float(sp.quantile(0.9)):.2f}; pelvis z median "
        f"{float(bank['root_z'].median()):.3f} m; hottest winding median "
        f"{float(cat['Tw'].amax(1).median()):.1f} C", flush=True)
  for lo, hi in ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 16.0)):
    mm = (bank["t"] >= lo) & (bank["t"] < hi)
    if bool(mm.any()):
      print(f"[harvest]   fallback {lo:4.1f}-{hi:4.1f} s in: {int(mm.sum())} rows, speed median "
            f"{float(sp[mm].median()):.2f} m/s, |action| RMS "
            f"{float(bank['action'][mm].pow(2).mean().sqrt()):.2f}", flush=True)


if __name__ == "__main__":
  main()
