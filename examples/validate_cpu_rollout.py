"""How faithfully a CPU MuJoCo rollout (``cpu_rollout.py``) reproduces the live GPU env.

    python examples/validate_cpu_rollout.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip

On a G1 recover env built as the eval builds it (``--hip``, no interval push),
with the actor group's corruption OFF (so the live observation is comparable),
from the live envs' own states:

1. OBSERVATION: the actor group rebuilt from MjData vs the env's, per term;
2. ONE CONTROL STEP: qpos / qvel / g after the same action, CPU vs GPU;
3. ROLLOUTS: from ``--rounds`` x N spawn states, kicked (``--kick``, the
   eval's push: a horizontal pelvis velocity change), a perturbed first action
   (the fallback's plus N(0, ``--first-noise``^2), unclamped, as SONIC's leave
   the box) and then the fallback for ``--horizon-s``, stepped in the live
   env and on the CPU: the per-step drift, and whether the two agree on the
   verdict (min g >= 0) and on when g first goes negative;
4. the worker pool: the same rollouts through ``CpuRolloutPool`` (must equal
   the in-process ones) and its wall time per batch at 1, 2 and 3 s.

Writes ``--out`` (json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_HERE)
sys.path.insert(0, _ZOO)
sys.path.insert(0, _HERE)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import cpu_rollout as cr  # noqa: E402
from harvest_g1_sonic_trajectories import make_env  # noqa: E402

from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True)
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--hip", choices=("mjlab", "sonic"), default="mjlab")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--rounds", type=int, default=4)
  p.add_argument("--warm-steps", type=int, default=3,
                 help="fallback steps after each spawn before the snapshot")
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--first-noise", type=float, default=1.5)
  p.add_argument("--kick", type=float, nargs=2, default=[0.0, 5.0],
                 help="before each rollout the pelvis gets a horizontal velocity kick of "
                      "|dv| ~ U(lo, hi) m/s at a uniform angle (0 0: none), so some fall")
  p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
  p.add_argument("--out", default=os.path.join(_ZOO, "runs", "rollout_filter", "validate.json"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def _last_action_slice(env):
  om, key = env.mj.observation_manager, env.obs_key
  names, dims = om._group_obs_term_names[key], om._group_obs_term_dim[key]
  sizes = [int(np.prod(d)) for d in dims]
  if "actions" not in names:
    return None                     # nothing to clamp (e.g. g1_recover_bench's actor)
  i = names.index("actions")
  return slice(sum(sizes[:i]), sum(sizes[:i + 1]))


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  torch.set_num_threads(1)          # pi as the rollout workers run it
  dev = a.device
  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  work = os.path.join(os.path.dirname(a.out), "validate_work")
  os.makedirs(work, exist_ok=True)

  def no_corruption(cfg):
    cfg.observations["actor"].enable_corruption = False

  args = SimpleNamespace(task=a.task, hip=a.hip, num_envs=a.num_envs, device=dev)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_env(args, dict(episode_length_s=60.0, rest_terminate=False), push_kw,
                 cfg_fn=no_corruption)
  env.reset()
  mj = env.mj
  N = int(mj.num_envs)
  H = int(round(a.horizon_s / float(mj.step_dt)))
  low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
  high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
  sl = _last_action_slice(env)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)

  def pi(obs):
    pobs = obs.clone()
    if sl is not None:
      pobs[:, sl] = torch.maximum(torch.minimum(pobs[:, sl], high), low)
    return torch.clamp(model.policy._predict(norm(pobs), deterministic=True), low, high)

  m_cpu, n_cpu = load_twin(a.policy, "cpu")
  m_cpu.policy.set_training_mode(False)
  actor_path = os.path.join(work, "actor.pt")
  cr.export_actor(m_cpu.policy, n_cpu, int(env.obs_groups()[env.obs_key].shape[1]), sl,
                  env.action_space.low, env.action_space.high,
                  env.action_space.low, env.action_space.high, actor_path)
  spec = cr.build_spec(env, actor_path, work)
  print(f"[validate] {N} envs, H {H} steps; spec: obs {spec.obs_dim}-D "
        f"{[t[0] for t in spec.obs_terms]}, {len(spec.force_adr)} non-foot force sensors, "
        f"DR fields {sorted(spec.dr)}", flush=True)
  roller = cr.Roller(spec, list(range(N)))
  names = [t[0] for t in spec.obs_terms]

  rep = {"num_envs": N, "horizon": H, "hip": a.hip, "obs": {}, "one_step": {}, "rollout": {},
         "pool": {}}
  obs_err = {nm: [] for nm in names}
  step_err = {"qpos": [], "qvel": [], "g": []}
  roll = {"gpu_safe": [], "cpu_safe": [], "gpu_tfail": [], "cpu_tfail": [], "gpu_gmin": [],
          "cpu_gmin": [], "drift_root_z": [], "drift_joint": []}
  snaps = []
  gen = torch.Generator(device=dev).manual_seed(a.seed)
  with torch.no_grad():
    for r in range(a.rounds):
      env.reset()
      obs = env.obs_groups()[env.obs_key].float()
      act = pi(obs)
      for _ in range(a.warm_steps):
        obs, g, dones, _t, _l = env.step_tensor(act)
        act = pi(obs.float())
      ids = torch.arange(N, device=dev)
      qpos, qvel, ws = cr.live_state(env, ids)
      tm = cr.live_thermal(env, ids)          # None unless a *_thermal task

      def temps(i, tm=None):
        return None if tm is None else tm[i]
      last = mj.action_manager.action.detach().cpu().numpy().astype(np.float64)
      gpu_obs = obs.detach().cpu().numpy().astype(np.float64)
      # 1. the observation
      buf = np.zeros(spec.obs_dim)
      for i in range(N):
        roller.seed(i, qpos[i], qvel[i], ws[i], temps=temps(i, tm))
        roller.observe(i, last[i], buf)
        for nm, st, sz, _x in spec.obs_terms:
          obs_err[nm].append(float(np.abs(buf[st:st + sz] - gpu_obs[i, st:st + sz]).max()))
      # 2. one control step under the fallback's action
      a0 = act.detach().cpu().numpy().astype(np.float64)
      g_cpu = np.empty(N)
      for i in range(N):
        roller.seed(i, qpos[i], qvel[i], ws[i], temps=temps(i, tm))
        g_cpu[i] = roller.control_step(i, a0[i])
      q_cpu = np.stack([roller._d(i).qpos.copy() for i in range(N)])
      v_cpu = np.stack([roller._d(i).qvel.copy() for i in range(N)])
      obs1, g1, d1, _t, _l = env.step_tensor(act)
      q1, v1, _w = cr.live_state(env, ids)
      ok = ~d1.cpu().numpy()
      # the plane is infinite: x, y carry each env's tile origin on the GPU
      dq = np.abs(q_cpu - q1)[ok]
      dq[:, :2] = 0.0
      step_err["qpos"].extend(dq.max(1).tolist())
      step_err["qvel"].extend(np.abs(v_cpu - v1)[ok].max(1).tolist())
      step_err["g"].extend(np.abs(g_cpu - g1.cpu().numpy())[ok].tolist())
      if tm is not None:   # the winding temperatures after the step, C
        tw_cpu = np.stack([roller.temps[(i, 0)][0] for i in range(N)])
        tw_gpu = cr.live_thermal(env, ids)[:, 0]
        step_err.setdefault("Tw", []).extend(np.abs(tw_cpu - tw_gpu)[ok].max(1).tolist())
      # 3. a rollout: a kick, a perturbed first action, then the fallback
      if a.kick[1] > 0:
        robot = mj.scene["robot"]
        mag = a.kick[0] + (a.kick[1] - a.kick[0]) * torch.rand(N, device=dev, generator=gen)
        ang = 2 * torch.pi * torch.rand(N, device=dev, generator=gen)
        vel = robot.data.root_link_vel_w.clone()
        vel[:, 0] += mag * torch.cos(ang)
        vel[:, 1] += mag * torch.sin(ang)
        robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
        mj.sim.forward()
        obs1 = mj.observation_manager.compute_group(env.obs_key, update_history=False).float()
      qpos, qvel, ws = cr.live_state(env, ids)
      tm = cr.live_thermal(env, ids)
      first = pi(obs1.float()) + a.first_noise * torch.randn(N, act.shape[1], device=dev,
                                                             generator=gen)
      f_np = first.detach().cpu().numpy().astype(np.float64)
      snaps.append((qpos, qvel, ws, f_np, tm))
      gmin_c, tf_c, g_tr, q_tr = roller.rollout(list(range(N)), qpos, qvel, ws, f_np, H,
                                                trace=True, temps=tm)
      g_gpu = np.full((N, H), np.nan)
      q_gpu = np.full((N, H, qpos.shape[1]), np.nan)
      alive = np.ones(N, dtype=bool)
      cur = first
      for t in range(H):
        o, g, dn, _t, _l = env.step_tensor(cur)
        qn, _v, _w = cr.live_state(env, ids)
        gn, dn = g.cpu().numpy(), dn.cpu().numpy()
        g_gpu[alive, t] = gn[alive]
        q_gpu[alive, t] = qn[alive]
        alive &= ~dn
        cur = pi(o.float())
      gmin_g = np.nanmin(g_gpu, axis=1)
      neg = g_gpu < 0
      tf_g = np.where(neg.any(1), neg.argmax(1) + 1, H + 1)
      roll["gpu_safe"].extend((gmin_g >= 0).tolist())
      roll["cpu_safe"].extend((gmin_c >= 0).tolist())
      roll["gpu_tfail"].extend(tf_g.tolist())
      roll["cpu_tfail"].extend(tf_c.tolist())
      roll["gpu_gmin"].extend(gmin_g.tolist())
      roll["cpu_gmin"].extend(gmin_c.tolist())
      both = ~np.isnan(q_tr[:, :, 0]) & ~np.isnan(q_gpu[:, :, 0])
      dz = np.where(both, np.abs(q_tr[:, :, 2] - q_gpu[:, :, 2]), np.nan)
      dj = np.where(both, np.abs(q_tr[:, :, 7:] - q_gpu[:, :, 7:]).mean(2), np.nan)
      roll["drift_root_z"].append(dz)
      roll["drift_joint"].append(dj)
      print(f"[validate] round {r + 1}/{a.rounds}: verdict agreement "
            f"{float(np.mean((gmin_g >= 0) == (gmin_c >= 0))):.3f}, GPU safe "
            f"{float(np.mean(gmin_g >= 0)):.3f}, CPU safe {float(np.mean(gmin_c >= 0)):.3f}",
            flush=True)

  for nm in names:
    e = np.array(obs_err[nm])
    rep["obs"][nm] = {"max": float(e.max()), "p99": float(np.quantile(e, 0.99))}
  for k, e in step_err.items():
    e = np.array(e)
    rep["one_step"][k] = {"max": float(e.max()), "median": float(np.median(e)),
                          "p99": float(np.quantile(e, 0.99)), "n": int(e.size)}
  gs, cs = np.array(roll["gpu_safe"]), np.array(roll["cpu_safe"])
  tg, tc = np.array(roll["gpu_tfail"]), np.array(roll["cpu_tfail"])
  both_fail = (tg <= H) & (tc <= H)
  dz = np.concatenate(roll["drift_root_z"])
  dj = np.concatenate(roll["drift_joint"])
  marks = [t for t in (1, 5, 10, 25, 50, 100, H) if t <= H]
  rep["rollout"] = {
    "n": int(gs.size), "agreement": float(np.mean(gs == cs)),
    "gpu_safe": float(gs.mean()), "cpu_safe": float(cs.mean()),
    "gpu_safe_cpu_fail": int((gs & ~cs).sum()), "gpu_fail_cpu_safe": int((~gs & cs).sum()),
    "tfail_abs_diff_median_steps": float(np.median(np.abs(tg - tc)[both_fail]))
    if both_fail.any() else None,
    "gmin_abs_diff_median": float(np.median(np.abs(np.array(roll["gpu_gmin"])
                                                    - np.array(roll["cpu_gmin"])))),
    "drift_root_z_m": {str(t): float(np.nanmedian(dz[:, t - 1])) for t in marks},
    "drift_joint_rad": {str(t): float(np.nanmedian(dj[:, t - 1])) for t in marks}}

  print(json.dumps({k: rep[k] for k in ("obs", "one_step", "rollout")}, indent=1), flush=True)

  # 4. the pool: same answers, and its wall time
  pool = cr.CpuRolloutPool(spec, a.workers)
  try:
    qpos, qvel, ws, f_np, tm = snaps[-1]
    slots = np.arange(N)
    g_ref, t_ref = roller.rollout(list(range(N)), qpos, qvel, ws, f_np, H, temps=tm)
    g_p, t_p = pool.rollout(slots, qpos, qvel, ws, f_np, H, temps=tm)
    rep["pool"]["equal_to_in_process"] = bool(np.array_equal(g_ref, g_p)
                                              and np.array_equal(t_ref, t_p))
    # the workers run pi single-threaded: last-bit matmul differences grow over H
    rep["pool"]["gmin_abs_diff_max"] = float(np.abs(g_ref - g_p).max())
    rep["pool"]["verdict_agreement"] = float(np.mean((g_ref >= 0) == (g_p >= 0)))
    rep["pool"]["t_fail_equal"] = float(np.mean(t_ref == t_p))
    rep["pool"]["workers"] = pool.workers
    for hs in (1.0, 2.0, 3.0):
      h = int(round(hs / float(mj.step_dt)))
      ts = []
      for _ in range(3):
        t0 = time.perf_counter()
        pool.rollout(slots, qpos, qvel, ws, f_np, h, temps=tm)
        ts.append(time.perf_counter() - t0)
      rep["pool"][f"wall_s_{hs:g}s_x{N}"] = float(np.median(ts))
    t0 = time.perf_counter()
    roller.rollout([0], qpos[:1], qvel[:1], ws[:1], f_np[:1], H,
                   temps=None if tm is None else tm[:1])
    rep["pool"]["single_rollout_in_process_s"] = time.perf_counter() - t0
  finally:
    pool.close()

  print(json.dumps(rep, indent=1))
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[validate] wrote {a.out}")


if __name__ == "__main__":
  main()
