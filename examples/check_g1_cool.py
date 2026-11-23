"""Smoke checks of the g1_cool_thermal70 task (GPU or CPU; exits non-zero on a failure):

* spawn: only minor-kick handover rows (kick dv <= the bank's max), the row's
  action seeds the action history (the first step's prev_action);
* start temperatures: ankle / waist roll+pitch windings in their hot range, the
  rest in the base draw's;
* every reward term finite each step; the cooling term in (0, 1]; the heat-rate
  term equals the heat model's own Q / Cw;
* scripted pushes fire; the episode is 20 s;
* g1_cool_thermal70_mix: ~30 % of the spawns on mocap rows, the rest on
  handover rows kicked <= 1 m/s; every push's force within U(0.2, 1) x 160 N.
* g1_cool_thermal70_fb: every spawn on a fallback-harvested row
  (examples/harvest_cool_fallback_states.py); 1-4 random other motors start
  hot; the cooling term is the temperature-ramp one (mdp.cool_ramp) and its
  heat rate equals sum ramp(Tw) Q / Cw; pushes as _mix.

    python examples/check_g1_cool.py --num-envs 64 --steps 300 --device cuda:0
    python examples/check_g1_cool.py --task g1_cool_thermal70_mix --num-envs 512
"""

from __future__ import annotations

import argparse
import math

import torch


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=300)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--task", default="g1_cool_thermal70")
  a = p.parse_args()
  fb = a.task.endswith("_fb")
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_cool import mdp
  from robot_safety_sandbox.envs.g1_cool.env_cfg import (EPISODE_LENGTH_S, FB_HOT_EXTRA,
                                                         HOT_GROUPS)
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state

  env = make_tensor(a.task, a.num_envs, a.device)
  mj = env.mj
  fails = []

  def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
      fails.append(msg)

  # The probe controller (the handover action held, + noise) falls within ~2 s,
  # before the task's first push (2-5 s): pull the push schedule in for the check
  # so it tests the push mechanics (the registered timing is printed).
  check(env._push is not None, f"the task has scripted pushes ({env._push and env._push._first} "
                               f"steps to the first, force {env.force_max} N)")
  env._push._first = (5, 10)
  env._push._interval = (20, 40)
  obs = env.reset()
  check(tuple(obs.shape) == (a.num_envs, 127), f"actor obs {tuple(obs.shape)} == (N, 127)")
  check(abs(float(mj.max_episode_length) * float(mj.step_dt) - EPISODE_LENGTH_S) < 0.05,
        f"episode {mj.max_episode_length} steps x {mj.step_dt} s ~ {EPISODE_LENGTH_S} s")
  st = thermal_state(mj)
  hot = torch.zeros(len(st.names), dtype=torch.bool)
  for sub, lo, hi, _g0, _g1 in HOT_GROUPS:
    idx = [i for i, nm in enumerate(st.names) if sub in nm]
    tw = st.Tw[:, idx]
    check(bool(((tw >= lo - 1e-4) & (tw <= hi + 1e-4)).all()),
          f"{sub} windings in [{lo}, {hi}] C (got {float(tw.min()):.1f}-{float(tw.max()):.1f})")
    hot[idx] = True
  cold = st.Tw[:, ~hot.to(st.Tw.device)]
  if fb:
    k_lo, k_hi, t_lo, t_hi = FB_HOT_EXTRA[:4]
    n_hot = (cold > 50.0 + 1e-4).sum(1)
    check(bool((cold <= t_hi + 1e-4).all() & (cold >= 30.0 - 1e-4).all())
          and int(n_hot.max()) <= int(k_hi),
          f"other windings in [30, {t_hi:g}] C, <= {int(k_hi)} above 50 C per robot (got "
          f"{float(cold.min()):.1f}-{float(cold.max()):.1f}, up to {int(n_hot.max())})")
    hit = torch.zeros(cold.shape[1])
    hit += (cold > 50.0).float().sum(0).cpu()
    print(f"[info] extra hot motors (> 50 C) per robot: mean {float(n_hot.float().mean()):.2f}; "
          f"motors hit: {int((hit > 0).sum())}/{cold.shape[1]}", flush=True)
    check(float(n_hot.float().mean()) > 0.5, "random extra hot motors are drawn")
  else:
    check(bool((cold <= 50.0 + 1e-4).all() & (cold >= 30.0 - 1e-4).all()),
          f"other windings in [30, 50] C (got {float(cold.min()):.1f}-{float(cold.max()):.1f})")
  check(bool((st.Th <= st.Tw + 1e-5).all() & (st.Ts <= st.Th + 1e-5).all()),
        "structure <= housing <= winding")
  # spawn rows: start_ok only (minor kicks / allowed mocap), the row's action
  # seeds the history; the mixture's shares
  from robot_safety_sandbox.envs.g1_recover.spawn import RecoverSpawn
  from robot_safety_sandbox.envs.g1_recover.states import KIND_HANDOVER, KIND_MOCAP
  spawn = next(c.func for cfgs in mj.event_manager._mode_term_cfgs.values() for c in cfgs
               if isinstance(getattr(c, "func", None), RecoverSpawn))
  bank = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in spawn.bank.items()}
  rows = mj._zoo_spawn_row.cpu()
  check(bool((rows >= 0).all()), "every env spawned on a bank row")
  r = rows.clamp_min(0)
  check(bool(bank["start_ok"][r].all()), "every spawn row is start_ok")
  kind = bank["kind"][r]
  hand, moc = kind == KIND_HANDOVER, kind == KIND_MOCAP
  dv = bank["push_impulse"][r].nan_to_num(0.0) / 33.0
  print(f"[info] spawns: {int(hand.sum())} handover (kick dv max "
        f"{float(dv[hand].max()) if hand.any() else 0.0:.2f} m/s), {int(moc.sum())} mocap "
        f"(categories {sorted(set(bank['category'][r][moc].tolist()))})", flush=True)
  if fb:
    check(bool(hand.all()), "every spawn is a (fallback) handover row")
    check("g1_cool_fb" in (bank["meta"] or {}), "the bank is the fallback harvest "
          "(meta g1_cool_fb)")
    print(f"[info] fallback rows: ran {float(bank['t'][r].min()):.1f}-"
          f"{float(bank['t'][r].max()):.1f} s before the snapshot; speed median "
          f"{float(bank['speed'][r].median()):.2f} m/s", flush=True)
  elif a.task.endswith("_mix"):
    share = float(moc.float().mean())
    check(bool((hand | moc).all()) and abs(share - 0.3) < 4.0 * (0.21 / a.num_envs) ** 0.5,
          f"mocap share {share:.3f} ~ 0.3, the rest handover")
    check(bool((dv[hand] <= 1.0 + 1e-4).all()), "handover rows kicked <= 1 m/s")
    cats = list(bank["meta"]["parts"][1]["categories"])
    bad = [cats.index(c) for c in ("fallAndGetUp", "sprint")]
    check(not bool(torch.isin(bank["category"][r][moc], torch.tensor(bad)).any()),
          "no fallAndGetUp / sprint mocap spawn")
  else:
    check(bool(hand.all()) and bool((dv <= 1.5 + 1e-4).all()),
          "every spawn is a handover row kicked <= 1.5 m/s")
  seeded = mj.action_manager.action.clone()
  check(torch.allclose(seeded.cpu(), bank["action"][r], atol=1e-5),
        "the reset action history holds the rows' last applied actions")

  push_steps = 0
  push_f = []
  for k in range(a.steps):
    act = (seeded + 0.05 * torch.randn_like(seeded)).clamp(-3, 3)
    obs, rew, dones, timeouts, _l = env.step_tensor(act)
    if k == 0:
      check(torch.allclose(mj.action_manager.prev_action, seeded, atol=1e-5),
            "first step's prev_action == the handover action (the jump is scored)")
    terms = {n: mj.reward_manager._step_reward[:, i] for i, n in
             enumerate(mj.reward_manager.active_terms)} \
      if hasattr(mj.reward_manager, "_step_reward") else {}
    if not torch.isfinite(rew).all():
      check(False, f"step {k}: non-finite reward")
      break
    for n, v in terms.items():
      if not torch.isfinite(v).all():
        check(False, f"step {k}: reward term {n} non-finite")
    c = mdp.cool_ramp(mj) if fb else mdp.cool(mj)
    if not bool(((c > 0) & (c <= 1)).all()):
      check(False, f"step {k}: cool term outside (0, 1]")
    q = mdp.motor_heat_w(st)
    if not bool((q >= -1e-6).all()):
      check(False, f"step {k}: negative heat input")
    if env._push is not None:
      act_m = env._push.active()
      push_steps += int(act_m.sum())
      rand_m = act_m & ~env._push.scripted
      if bool(rand_m.any()):
        push_f.append(env._push.cur_force_w[rand_m].norm(dim=1).cpu())
    if k == 0 or k == a.steps - 1:
      hr = mdp.heat_rate_ramp_c_per_s(mj) if fb else mdp.heat_rate_c_per_s(mj)
      print(f"[info] step {k}: reward mean {float(rew.mean()):.3f}; heat rate mean "
            f"{float(hr.mean()):.3f} C/s; cool {float(c.mean()):.3f}; "
            f"terms " + ", ".join(f"{n} {float(v.mean()):+.3f}" for n, v in terms.items()),
            flush=True)
  # heat-rate term == the model's Q / Cw, weighted
  if fb:
    x = ((st.Tw - 50.0) / 15.0).clamp_min(0.0)
    manual = (mdp.motor_heat_w(st) / st.th.Cw * (0.05 + x * x)).sum(1)
    check(torch.allclose(manual, mdp.heat_rate_ramp_c_per_s(mj)),
          "ramp heat-rate term == sum (0.05 + relu((Tw - 50) / 15)^2) Q / Cw")
    tc = mj.reward_manager.get_term_cfg("cool")
    hc = mj.reward_manager.get_term_cfg("handover_rate")
    check(tc.func is mdp.cool_ramp and hc.weight == -0.1 and hc.params["window_s"] == 0.5,
          f"reward: cool = cool_ramp ({tc.func.__name__}), handover_rate {hc.weight} over "
          f"{hc.params.get('window_s')} s")
  else:
    w = mdp._motor_weights(mj, mdp.HOT_WEIGHTS, mdp.OTHER_WEIGHT)
    manual = (mdp.motor_heat_w(st) / st.th.Cw * w).sum(1)
    check(torch.allclose(manual, mdp.heat_rate_c_per_s(mj)), "heat-rate term == sum w Q / Cw")
  check(push_steps > 0 and int(env._push.n_pushes.sum()) >= 0,
        f"scripted pushes acted ({push_steps} env-steps)")
  if push_f:
    f = torch.cat(push_f)
    ff = env._push.force_frac or (1.0, 1.0)
    lo, hi = ff[0] * env.force_max, ff[1] * env.force_max
    check(bool((f >= lo - 1e-3).all() & (f <= hi + 1e-3).all()),
          f"random push forces in [{lo:.0f}, {hi:.0f}] N (got {float(f.min()):.0f}-"
          f"{float(f.max()):.0f} N, {f.numel()} env-steps; dv up to "
          f"{float(f.max()) * 0.2 / 33.0:.2f} m/s)")
  print(f"[check] {'PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}", flush=True)
  raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
  main()
