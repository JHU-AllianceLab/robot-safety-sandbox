"""Does a jerky SONIC -> cooling-policy handover add heat? Same hot handover spawns, five handovers.

    python examples/cool_handover_heat.py \
        --policy runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip \
        --num-envs 4096 --out runs/g1_cool/handover_heat/pilot_s0.json

The g1_cool task spawns on SONIC handover rows with the row's last applied
action a_h seeding the action history, the ankle / waist roll+pitch windings
hot. Pushes are switched off. From the same spawns (seeded resets, checked) the
deterministic cooling policy pi takes over in one of these ways:

  policy     pi from the first step (as trained and deployed)
  ramp<T>    a_t = a_h + beta_t (pi(o_t) - a_h), beta_t = min(1, (t+1) / (T / dt)):
             a linear blend from SONIC's last action into pi over T s (pi reads
             the blended action as its last action, as it would on the robot)
  hold       a_h held for the whole horizon (reference: SONIC's last target frozen)

Per episode, while it has not fallen: the heat input of the hot motors from their
own losses, sum over ankle / waist roll+pitch of integral Q_i / Cw_i dt (C; the
winding rise without any cooling, the cooling reward's H integrated), the change
of the hottest of those windings, the applied action jump ||a_t - a_{t-1}||^2
(the step-0 one is the handover jump the reward scores), the peak hot-motor
torque, and falls. Within ``policy`` also the heat in the first 0.5 s by
quartile of the step-0 jump and their Spearman correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

WINDOWS_S = (0.1, 0.25, 0.5, 1.0, 2.0)
HOT_SUBSTR = ("ankle", "waist_roll", "waist_pitch")


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="cooling policy final_model.zip (+ vecnormalize.pkl)")
  p.add_argument("--task", default="g1_cool_thermal70")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--horizon-s", type=float, default=2.0)
  p.add_argument("--ramps", default="0.1,0.25,0.5", help="ramp lengths, s")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", required=True)
  return p.parse_args(argv)


def load_policy(zip_path: str, device: str):
  """The plain SB3 policy + its VecNormalize obs statistics (as eval_sonic_qcbf.load_pi)."""
  from robot_safety_sandbox.eval.policies import load_nominal
  model, vn = load_nominal(zip_path, device, quiet=True)
  if vn is None or not hasattr(vn, "obs_rms"):
    raise SystemExit(f"{zip_path}: no vecnormalize.pkl next to it")
  mean = torch.as_tensor(vn.obs_rms.mean, dtype=torch.float32, device=device)
  var = torch.as_tensor(vn.obs_rms.var, dtype=torch.float32, device=device)
  eps, clip = float(vn.epsilon), float(vn.clip_obs)
  model.policy.set_training_mode(False)

  def norm(obs):
    return torch.clamp((obs - mean) / torch.sqrt(var + eps), -clip, clip)

  return model, norm


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
  rx = torch.argsort(torch.argsort(x)).double()
  ry = torch.argsort(torch.argsort(y)).double()
  rx, ry = rx - rx.mean(), ry - ry.mean()
  return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


@torch.no_grad()
def run(env, model, norm, mode: str, ramp_s: float, horizon: int, seed: int, hot: torch.Tensor):
  from robot_safety_sandbox.envs.g1_cool import mdp
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  torch.manual_seed(seed)
  obs = env.reset()
  st = thermal_state(mj)
  rows = mj._zoo_spawn_row.clone()
  tw0 = st.Tw.clone()
  a_h = mj.action_manager.action.clone()          # SONIC's last applied action (seeded history)
  ramp_steps = max(1, int(round(ramp_s / dt))) if mode == "ramp" else 1
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  fell_at = torch.full((n,), -1, dtype=torch.long, device=dev)
  heat = torch.zeros(n, horizon, device=dev)       # hot-motor heat input this step, C
  hot_max = torch.zeros(n, horizon, device=dev)    # hottest hot winding after this step, C
  jump = torch.zeros(n, horizon, device=dev)       # ||a_t - a_{t-1}||^2 applied
  tau_pk = torch.zeros(n, device=dev)
  prev = a_h.clone()
  for t in range(horizon):
    if mode == "hold":
      act = a_h
    else:
      pi = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
      beta = min(1.0, (t + 1) / ramp_steps)
      act = a_h + beta * (pi - a_h)
    jump[:, t] = (act - prev).square().sum(1)
    prev = act
    obs, _r, dones, timeouts, _l = env.step_tensor(act)
    q = mdp.motor_heat_w(st) / st.th.Cw           # C/s per motor at the last substep
    heat[:, t] = torch.where(alive, (q[:, hot]).sum(1) * dt, torch.zeros_like(heat[:, t]))
    hot_max[:, t] = st.Tw[:, hot].max(1).values
    tau, _w = st.motor_inputs()
    tau_pk = torch.where(alive, torch.maximum(tau_pk, tau[:, hot].abs().max(1).values), tau_pk)
    fell = dones & ~timeouts & alive
    fell_at[fell] = t
    alive &= ~dones
  return {"rows": rows, "tw0": tw0, "a_h": a_h, "heat": heat, "hot_max": hot_max, "jump": jump,
          "tau_pk": tau_pk, "fell_at": fell_at, "alive": alive, "dt": dt}


def summarize(r: dict, hot_max0: torch.Tensor) -> dict:
  dt, H = r["dt"], int(r["heat"].shape[1])
  ok = r["alive"]                                    # survived the whole horizon
  m = lambda x: float(x[ok].mean())
  sem = lambda x: float(x[ok].std() / math.sqrt(max(1, int(ok.sum()))))
  out = {"episodes": int(ok.numel()), "fell_frac": float((r["fell_at"] >= 0).float().mean()),
         "jump0_mean": m(r["jump"][:, 0]), "jump0_median": float(r["jump"][ok, 0].median()),
         "jump_0p25s_mean": m(r["jump"][:, :max(1, round(0.25 / dt))].mean(1)),
         "jump_late_mean": m(r["jump"][:, -max(1, round(0.5 / dt)):].mean(1)),
         "tau_peak_hot_nm": m(r["tau_pk"]), "windows": {}}
  for w in WINDOWS_S:
    k = min(H, max(1, int(round(w / dt))))
    hin = r["heat"][:, :k].sum(1)
    d_hot = r["hot_max"][:, k - 1] - hot_max0
    out["windows"][f"{w:g}"] = {"heat_in_c": m(hin), "heat_in_sem": sem(hin),
                                "d_hottest_c": m(d_hot), "d_hottest_sem": sem(d_hot)}
  k1 = max(1, int(round(0.5 / dt)))
  out["late_heat_rate_c_per_s"] = m(r["heat"][:, -k1:].sum(1) / (k1 * dt))
  return out


def main(argv=None):
  a = _parse(argv)
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  env = make_tensor(a.task, a.num_envs, a.device)
  if getattr(env, "_push", None) is not None:      # no pushes: only the handover differs
    env._push._first = (10 ** 6, 10 ** 6)
    env._push._interval = (10 ** 6, 10 ** 6)
  env.force_max = 0.0
  model, norm = load_policy(a.policy, a.device)
  names = thermal_state(env.mj).names
  hot = torch.tensor([any(s in nm for s in HOT_SUBSTR) for nm in names], device=env.device)
  print(f"[heat] hot motors: {[nm for nm, h in zip(names, hot.tolist()) if h]}", flush=True)
  H = int(round(a.horizon_s / float(env.mj.step_dt)))
  conds = [("policy", "policy", 0.0)] + [(f"ramp{float(s):g}", "ramp", float(s))
                                         for s in a.ramps.split(",")] + [("hold", "hold", 0.0)]
  res, ref = {}, None
  for name, mode, ramp in conds:
    r = run(env, model, norm, mode, ramp, H, a.seed, hot)
    if ref is None:
      ref = r
    paired = bool(torch.equal(r["rows"], ref["rows"])) and bool(
      torch.allclose(r["tw0"], ref["tw0"])) and bool(torch.allclose(r["a_h"], ref["a_h"]))
    s = summarize(r, r["tw0"][:, hot].max(1).values)
    s["paired_with_policy"] = paired
    res[name] = s
    if name == "policy":
      ok = r["alive"]
      j0, h05 = r["jump"][ok, 0], r["heat"][ok, :int(round(0.5 / r["dt"]))].sum(1)
      qs = torch.quantile(j0, torch.tensor([0.25, 0.5, 0.75], device=j0.device))
      bins = torch.bucketize(j0, qs)
      s["by_jump0_quartile"] = [{"jump0_range": [float(j0[bins == b].min()), float(j0[bins == b].max())],
                                 "heat_0p5s_c": float(h05[bins == b].mean())} for b in range(4)]
      s["spearman_jump0_heat0p5s"] = spearman(j0, h05)
    w = s["windows"]
    print(f"[heat] {name:9s} paired={paired} fell {100 * s['fell_frac']:.1f}%  jump0 "
          f"{s['jump0_mean']:.1f} (median {s['jump0_median']:.1f}), late {s['jump_late_mean']:.2f}  "
          f"tau_pk {s['tau_peak_hot_nm']:.1f} Nm  heat-in C: "
          + "  ".join(f"{k}s {v['heat_in_c']:.3f}" for k, v in w.items())
          + f"  | hottest dT 0.5s {w['0.5']['d_hottest_c']:+.3f} 2s {w['2']['d_hottest_c']:+.3f}"
          f"  late rate {s['late_heat_rate_c_per_s']:.2f} C/s", flush=True)
  if "by_jump0_quartile" in res["policy"]:
    print("[heat] policy, heat in 0-0.5 s by step-0 jump quartile: " + ", ".join(
      f"[{q['jump0_range'][0]:.0f}-{q['jump0_range'][1]:.0f}] {q['heat_0p5s_c']:.3f} C"
      for q in res["policy"]["by_jump0_quartile"])
      + f"; Spearman {res['policy']['spearman_jump0_heat0p5s']:+.2f}", flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  with open(a.out, "w") as fh:
    json.dump({"policy": a.policy, "task": a.task, "num_envs": a.num_envs, "horizon_s": a.horizon_s,
               "seed": a.seed, "hot_motors": [nm for nm, h in zip(names, hot.tolist()) if h],
               "conditions": res}, fh, indent=1)
  print(f"[heat] wrote {a.out}", flush=True)


if __name__ == "__main__":
  main()
