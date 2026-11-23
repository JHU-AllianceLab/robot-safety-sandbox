"""Cooling curve: how fast does a policy, standing alone with no pushes, bring hot
ankles down? (the user, 2026-11-18: "if the ankle is at 65 degree, how fast it'll
cool to room temperature ... show me the cooling curve").

Every policy runs on the SAME starts (one seed: the same spawn rows and domain
randomization): a g1_cool handover spawn, every ankle winding (pitch and roll,
both legs) at ``--ankle-c``, the other motors at ``--rest-c`` throughout, the
ankle housings per scenario (``--housing name:C``, envs split evenly), the
structures at ``--structure-c``. No random push (``max_pushes = 0``); a handover
row's own pending kick, the one SONIC was still taking at the handover, still
finishes (as in deployment). The run lasts ``--duration-s``; an env that falls
or overheats stops being recorded at that step (its auto-reset would redraw
the temperatures).

Recorded every second for the 4 ankle motors: winding / housing / structure C,
the mean tau_m^2 and the mean winding heat input Q (W, the heat model's own
losses: copper + speed terms) of that second, plus who is still standing; and
for every motor (``Tw_all``, ``heat_all``, order ``meta["motor_names"]``) the
winding C and the mean Q of that second (examples/cool_motor_report.py). Also written,
from the same starts, two physics references (the heat model alone, no sim):

* ``passive``: zero torque, drive on (idle heat only), the fastest any policy
  could cool;
* per policy, ``<name>_hold``: the policy's own late heat input (mean of the
  last 60 s) held from the end of the run out to ``--extrapolate-s``, i.e.
  where its curve goes over the slow structure time constant (~3 h).

    python examples/cool_curve.py --out runs/g1_cool/cool_curve \\
      --policy cooling runs/g1_cool/pilot_s0/g1_cool_thermal70/final_model.zip \\
      --policy fallback runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p/final_model.zip
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch


def _last_action_slice(mj, key: str):
  om = mj.observation_manager
  names = om._group_obs_term_names[key]
  dims = om._group_obs_term_dim[key]
  cfgs = om._group_obs_term_cfgs[key]
  hits = [i for i, c in enumerate(cfgs)
          if getattr(getattr(c, "func", None), "__name__", "") == "last_action"]
  hits = hits or [i for i, nm in enumerate(names) if nm == "actions"]
  if len(hits) != 1:
    raise SystemExit(f"[curve] {len(hits)} last-action terms in {names}")
  sizes = [int(math.prod(d)) for d in dims]
  s = sum(sizes[:hits[0]])
  return slice(s, s + sizes[hits[0]])


def load_policy(zip_path: str, dev: str):
  """(model, obs normalizer, twin?): a SHAC twin (load_twin), else a PPO policy
  with its VecNormalize stats (load_nominal)."""
  from robot_safety_sandbox.eval.policies import load_nominal, load_twin
  try:
    model, norm = load_twin(zip_path, dev, quiet=True)
    return model, norm, True
  except SystemExit:
    model, vn = load_nominal(zip_path, dev, quiet=True)
    mean = torch.as_tensor(vn.obs_rms.mean, dtype=torch.float32, device=dev)
    var = torch.as_tensor(vn.obs_rms.var, dtype=torch.float32, device=dev)
    eps, clip = float(vn.epsilon), float(vn.clip_obs)
    return model, (lambda o: torch.clamp((o - mean) / torch.sqrt(var + eps), -clip, clip)), False


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--policy", nargs=2, action="append", metavar=("NAME", "ZIP"), required=True)
  p.add_argument("--task", default="g1_cool_thermal70")
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--duration-s", type=float, default=600.0)
  p.add_argument("--extrapolate-s", type=float, default=4 * 3600.0)
  p.add_argument("--ankle-c", type=float, default=65.0)
  p.add_argument("--housing", nargs="+", default=["fresh:35", "soaked:58"], metavar="NAME:C")
  p.add_argument("--structure-c", type=float, default=30.0)
  p.add_argument("--rest-c", type=float, default=30.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", required=True)
  a = p.parse_args()
  scen = [(s.split(":")[0], float(s.split(":")[1])) for s in a.housing]
  os.makedirs(a.out, exist_ok=True)

  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_cool.mdp import motor_heat_w
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state

  dev = a.device
  env = make_tensor(a.task, a.num_envs, dev,
                    cfg_overrides={"episode_length_s": a.duration_s + 30.0})
  mj = env.mj
  if env._push is not None:
    env._push.max_pushes = 0                     # no random push at all
  dt = float(mj.step_dt)
  per_s = int(round(1.0 / dt))
  T = int(a.duration_s)
  bound = float(env.action_bound)
  la = _last_action_slice(mj, env.obs_key)
  st = thermal_state(mj)
  names = list(st.names)
  ank = [i for i, nm in enumerate(names) if "ankle" in nm]
  ank_names = [names[i] for i in ank]
  th = st.th
  n = a.num_envs
  sid = torch.arange(n, device=dev) % len(scen)          # scenario of each env
  th_start = torch.tensor([c for _n, c in scen], device=dev)[sid]

  def start_temps():
    Tw = torch.full((n, len(names)), a.rest_c, device=dev)
    Th, Ts = Tw.clone(), Tw.clone()
    Tw[:, ank] = a.ankle_c
    Th[:, ank] = torch.minimum(th_start, torch.tensor(a.ankle_c, device=dev)).unsqueeze(1)
    Ts[:, ank] = torch.minimum(torch.tensor(a.structure_c, device=dev), Th[:, ank])
    return Tw, Th, Ts

  out = {"meta": {"task": a.task, "num_envs": n, "duration_s": a.duration_s, "dt": dt,
                  "ankle_c": a.ankle_c, "scenarios": scen, "structure_c": a.structure_c,
                  "rest_c": a.rest_c, "ambient_c": float(th.Ta), "seed": a.seed,
                  "ankle_motors": ank_names, "motor_names": names,
                  "policies": {k: v for k, v in a.policy}},
         "scenario_of_env": sid.cpu()}
  for name, zip_path in a.policy:
    model, norm, twin = load_policy(zip_path, dev)
    torch.manual_seed(a.seed)
    obs = env.reset()
    Tw0, Th0, Ts0 = start_temps()
    st.set(None, Tw0, Th0, Ts0)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fall_t = torch.full((n,), float("nan"), device=dev)
    rec = {k: torch.full((n, T + 1, len(ank)), float("nan"), device=dev)
           for k in ("Tw", "Th", "Ts", "tau2", "heat")}
    rec["Tw"][:, 0], rec["Th"][:, 0], rec["Ts"][:, 0] = Tw0[:, ank], Th0[:, ank], Ts0[:, ank]
    # every motor: winding C and mean heat input W per second
    rec["Tw_all"] = torch.full((n, T + 1, len(names)), float("nan"), device=dev)
    rec["heat_all"] = torch.full_like(rec["Tw_all"], float("nan"))
    rec["Tw_all"][:, 0] = Tw0
    acc = torch.zeros(n, len(ank), device=dev)
    acc_q = torch.zeros_like(acc)
    acc_all = torch.zeros(n, len(names), device=dev)
    for k in range(T * per_s):
      o = obs
      if twin:
        o = obs.clone()
        o[:, la] = o[:, la].clamp(-bound, bound)
      with torch.no_grad():
        act = torch.clamp(model.policy._predict(norm(o), deterministic=True), -bound, bound)
      obs, _r, dones, _to, _l = env.step_tensor(act)
      ended = dones.bool() & alive
      fall_t[ended] = (k + 1) * dt
      alive &= ~dones.bool()
      tau, _w = st.motor_inputs()
      acc += torch.where(alive.unsqueeze(1), tau[:, ank] ** 2, torch.zeros_like(acc))
      q_all = motor_heat_w(st)
      acc_q += torch.where(alive.unsqueeze(1), q_all[:, ank], torch.zeros_like(acc))
      acc_all += torch.where(alive.unsqueeze(1), q_all, torch.zeros_like(acc_all))
      if (k + 1) % per_s == 0:
        s = (k + 1) // per_s
        m = alive.unsqueeze(1)
        nan = torch.full_like(acc, float("nan"))
        rec["Tw"][:, s] = torch.where(m, st.Tw[:, ank], nan)
        rec["Th"][:, s] = torch.where(m, st.Th[:, ank], nan)
        rec["Ts"][:, s] = torch.where(m, st.Ts[:, ank], nan)
        rec["tau2"][:, s] = torch.where(m, acc / per_s, nan)
        rec["heat"][:, s] = torch.where(m, acc_q / per_s, nan)
        rec["Tw_all"][:, s] = torch.where(m, st.Tw, float("nan"))
        rec["heat_all"][:, s] = torch.where(m, acc_all / per_s, float("nan"))
        acc.zero_()
        acc_q.zero_()
        acc_all.zero_()
        if s % 60 == 0:
          hot = rec["Tw"][:, s].amax(1)
          print(f"[curve] {name} t={s:4d} s standing {int(alive.sum())}/{n}; hottest ankle "
                + ", ".join(f"{sn} {float(hot[(sid == j) & alive].median()):.1f} C"
                            for j, (sn, _c) in enumerate(scen) if bool(((sid == j) & alive).any())),
                flush=True)
    out[name] = {k: v.cpu() for k, v in rec.items()}
    out[name]["fall_t"] = fall_t.cpu()
    print(f"[curve] {name}: {int(alive.sum())}/{n} standing after {T} s", flush=True)

  # --- physics references (heat model only; per scenario, the median start)
  def integrate(T0, heat_w, seconds, t_from=0.0, every=10.0):
    """Ankle motors' (Tw, Th, Ts) under a constant winding heat input heat_w (W)
    per motor (a policy's late mean Q; 0 = passive), drive on (idle heat into the
    housing); returns the times and temperatures."""
    Tw, Th, Ts = (x.clone() for x in T0)
    tt, rows = [t_from], [torch.stack([Tw, Th, Ts])]
    h, t, nxt = 0.02, t_from, t_from + every
    Cw, Rwh, Ch, Rha, Cs, Rsa = (getattr(th, k).reshape(-1)[ank] for k in
                                 ("Cw", "Rwh", "Ch", "Rha", "Cs", "Rsa"))
    Pidle = th.P_idle.reshape(-1)[ank]
    steps = int(round(seconds / h))
    for _ in range(steps):
      qwh = (Tw - Th) / Rwh
      qhs = (Th - Ts) / Rha
      dTw = (heat_w - qwh) / Cw
      dTh = (qwh + Pidle - qhs) / Ch
      dTs = (qhs - (Ts - th.Ta) / Rsa) / Cs
      Tw, Th, Ts = Tw + h * dTw, Th + h * dTh, Ts + h * dTs
      t += h
      if t >= nxt - 1e-9:
        tt.append(t)
        rows.append(torch.stack([Tw, Th, Ts]))
        nxt += every
    return torch.tensor(tt), torch.stack(rows)        # [K], [K, 3, n_ank]

  refs = {}
  for j, (sn, _c) in enumerate(scen):
    e = int((sid == j).nonzero()[0])
    Tw0, Th0, Ts0 = start_temps()
    T0 = (Tw0[e, ank], Th0[e, ank], Ts0[e, ank])
    refs[f"passive/{sn}"] = integrate(T0, torch.zeros(len(ank), device=dev), a.extrapolate_s)
    for name, _z in a.policy:
      r = out[name]
      m = (sid.cpu() == j)
      last = r["Tw"][m, -1]
      ok = ~torch.isnan(last).any(1)
      if not bool(ok.any()):
        continue
      heat = r["heat"][m][ok][:, -60:].mean(dim=(0, 1)).to(dev)     # per ankle motor, W
      Tw_end = r["Tw"][m][ok][:, -1].median(0).values.to(dev)
      Th_end = r["Th"][m][ok][:, -1].median(0).values.to(dev)
      Ts_end = r["Ts"][m][ok][:, -1].median(0).values.to(dev)
      refs[f"{name}_hold/{sn}"] = integrate((Tw_end, Th_end, Ts_end), heat,
                                            a.extrapolate_s - a.duration_s, t_from=a.duration_s)
      out[name].setdefault("late_heat_w", {})[sn] = heat.cpu()
  out["refs"] = {k: (t.cpu(), v.cpu()) for k, (t, v) in refs.items()}
  torch.save(out, os.path.join(a.out, "curve.pt"))
  print(f"[curve] wrote {a.out}/curve.pt", flush=True)


if __name__ == "__main__":
  main()
