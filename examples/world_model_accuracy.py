"""How well SHAC's world model predicts the simulator under SHAC's own actions, by horizon.

    python examples/world_model_accuracy.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip --tag shac_s0

From --episodes kicked held-out states (a spawn on a start_ok row of the
held-out clip library, then the eval's kick: |dv| ~ U(--dv-range) m/s,
horizontal, at a uniform angle, before the first step), SHAC drives the REAL
env for --horizon-s (deterministic; sensor noise off unless --obs-noise), and
the world model is rolled from the same first state three ways:

  open loop          fed the actions SHAC played in the real env: the model's
                     own error compounding (what its k-step fit trains);
  closed loop        SHAC acts on the model's predicted state (its actor obs
                     rebuilt from it, as in SHAC's imagination), noise-free;
  closed loop noisy  the same with the env's sensor-noise law on SHAC's input
                     (one draw per step, as SHAC's imagination draws it).

Per horizon k, over the episodes the real env has not ended by step k: the
error of the predicted state s_k against the real one -- RMS in the model's
normalized coordinates (predicted dims), the relative error of the predicted
CHANGE from s_0 (1.0 = no better than "nothing moves"), and physical errors
(pelvis height, tilt, joint angles, joint velocities, base linear and angular
velocity) -- the error of the margin g_k, and the FALL VERDICT: does "g < 0
or P(terminated) > 0.5 within k steps" in the model match g < 0 within k
steps in the real env (over every episode; a real episode that ended counts
as a fall from its end step on).

Writes ``--out`` (json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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

from eval_sonic_qcbf import write_episode_bank  # noqa: E402

from robot_safety_sandbox.base import MjlabTensorSafetyEnv  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import use_sonic_hip_pitch  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES, RecoverSpawn  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import KIND_MOCAP, load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402

MODES = ("open loop", "closed loop", "closed loop noisy")
MARKS = (1, 2, 4, 8, 16, 32, 50, 75, 100, 150)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="a SafetySHAC1P final_model.zip")
  p.add_argument("--tag", required=True)
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"))
  p.add_argument("--hip", choices=("mjlab", "sonic"), default="mjlab")
  p.add_argument("--episodes", type=int, default=4096, help="one wave: the env count")
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--dv-range", type=float, nargs=2, default=[0.0, 5.0])
  p.add_argument("--obs-noise", action="store_true",
                 help="leave the env's sensor noise on for the REAL SHAC")
  p.add_argument("--out", default=None,
                 help="default runs/shac_vs_ppo_sonic/world_model_accuracy/<tag>.json")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(_ZOO, "runs", "shac_vs_ppo_sonic", "world_model_accuracy",
                                f"{a.tag}.json")
  return a


def _spawn_term(mj) -> RecoverSpawn:
  for cfgs in getattr(mj.event_manager, "_mode_term_cfgs", {}).values():
    for c in cfgs:
      if isinstance(getattr(c, "func", None), RecoverSpawn):
        return c.func
  raise SystemExit("[wm-acc] the env has no RecoverSpawn reset term")


def _tilt_deg(pg: torch.Tensor) -> torch.Tensor:
  """Angle from vertical (deg) of a projected-gravity vector [..., 3]."""
  c = (-pg[..., 2] / pg.norm(dim=-1).clamp_min(1e-9)).clamp(-1.0, 1.0)
  return torch.rad2deg(torch.acos(c))


@torch.no_grad()
def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  t0 = time.time()
  N = int(a.episodes)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  wm = getattr(model, "world_model", None)
  if wm is None or not bool(wm.fitted):
    raise SystemExit(f"[wm-acc] {a.policy} has no fitted world model")
  z_clip = getattr(model, "wm_z_clip", None)
  actor_idx = model._actor_idx_t.to(dev)
  n_lo, n_hi = model._noise_lo_t, model._noise_hi_t
  low = torch.as_tensor(model.action_space.low, dtype=torch.float32, device=dev)
  high = torch.as_tensor(model.action_space.high, dtype=torch.float32, device=dev)
  dyn = wm.dynamic_mask

  def pi(raw_actor_obs):
    return torch.clamp(model.policy._predict(norm(raw_actor_obs), deterministic=True), low, high)

  # --- the spawn rows: start_ok mocap rows of the held-out library, uniformly
  lib = load_bank(a.library, "cpu")
  pool = (lib["start_ok"] & (lib["kind"] == KIND_MOCAP)).nonzero(as_tuple=False).squeeze(-1)
  gen = torch.Generator().manual_seed(a.seed)
  rows = pool[torch.randint(0, pool.numel(), (N,), generator=gen)]
  out_dir = os.path.dirname(a.out)
  os.makedirs(out_dir, exist_ok=True)
  bank_path = os.path.join(out_dir, f"{a.tag}_spawn_bank.pt")
  write_episode_bank(lib, {"spawn": {"row": rows}}, bank_path)
  mag = a.dv_range[0] + (a.dv_range[1] - a.dv_range[0]) * torch.rand(N, generator=gen)
  ang = 2 * math.pi * torch.rand(N, generator=gen)

  s = spec(a.task)

  def builder(play: bool = False, **kw):
    cfg = s.cfg_builder(play=play, **kw)
    cfg.events.pop("push_robot", None)
    if a.hip == "sonic":
      use_sonic_hip_pitch(cfg)
    if not a.obs_noise:
      cfg.observations["actor"].enable_corruption = False
    return cfg

  weights = {k: 0.0 for k in SOURCES}
  weights["mocap"] = 1.0
  overrides = dict(dataset=bank_path, sampler="sequential", sequential_repeats=1, weights=weights,
                   episode_length_s=a.horizon_s + 2.0, rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = MjlabTensorSafetyEnv(
    N, dev, cfg_builder=builder, margin_fn=s.margin_fn, ctrl_dim=s.ctrl_dim,
    dstb_dim=s.dstb_dim, adversary=False, end_criterion=s.end_criterion,
    cfg_overrides=overrides, critic_obs_key="critic", **dict(s.kwargs, random_push=push_kw))
  mj = env.mj
  H = int(round(a.horizon_s / float(mj.step_dt)))
  spawn = _spawn_term(mj)
  spawn._cursor = 0
  mj._zoo_spawn_cursor = 0
  env.reset()
  layout = env._obs_layout(env.critic_obs_key)        # {term: (start, size, cfg)}
  ctx = env.sim_context()
  if int(wm.state_dim) != sum(sz for _, sz, _ in layout.values()):
    raise SystemExit("[wm-acc] the env's critic group does not match the world model's state")
  cn = wm.encode_context(ctx)

  # --- the kick, then the real rollout under SHAC
  robot = mj.scene["robot"]
  vel = robot.data.root_link_vel_w.clone()
  m_d, a_d = mag.to(dev), ang.to(dev)
  vel[:, 0] += m_d * torch.cos(a_d)
  vel[:, 1] += m_d * torch.sin(a_d)
  robot.write_root_link_velocity_to_sim(vel, env_ids=torch.arange(N, device=dev))
  mj.sim.forward()
  om = mj.observation_manager
  s0 = torch.nan_to_num(om.compute_group(env.critic_obs_key, update_history=False).float())
  obs = torch.nan_to_num(om.compute_group(env.obs_key, update_history=False).float())
  rebuilt_gap = float((obs - s0.index_select(-1, actor_idx)).abs().max())
  S = torch.empty(H + 1, N, s0.shape[1], device=dev)
  A = torch.empty(H, N, int(low.numel()), device=dev)
  G = torch.full((H, N), math.nan, device=dev)
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  ALIVE = torch.zeros(H, N, dtype=torch.bool, device=dev)    # step k's successor is valid
  ended = torch.full((N,), H + 1, dtype=torch.long, device=dev)
  S[0] = s0
  for t in range(H):
    act = pi(obs)
    A[t] = act
    obs, g, dones, _to, _l = env.step_tensor(act)
    G[t] = torch.where(alive, g, G[t])
    S[t + 1] = env.critic_obs().float()
    ALIVE[t] = alive & ~dones
    ended = torch.where(alive & dones, torch.full_like(ended, t + 1), ended)
    alive = alive & ~dones
  real_fell_any = (torch.nan_to_num(G, nan=1.0) < 0).any(0) | (ended <= H)

  # --- the model, three ways
  def roll(mode: str):
    z = wm.encode(s0)
    Zs, gs, ps = [], [], []
    for t in range(H):
      if mode == "open loop":
        act = A[t]
      else:
        o = wm.decode(z).index_select(-1, actor_idx)
        if mode == "closed loop noisy" and n_lo is not None:
          o = o + n_lo + (n_hi - n_lo) * torch.rand_like(o)
        act = pi(o)
      st = wm.step(z, act, cn, z_clip=z_clip)
      z = st.z_next
      Zs.append(z)
      gs.append(st.g)
      ps.append(torch.sigmoid(st.done_logit))
    return torch.stack(Zs), torch.stack(gs), torch.stack(ps)

  z0 = wm.encode(s0)
  Zr = wm.encode(S[1:])                                    # [H, N, D] real
  n_dyn = dyn.sum()

  def term(name):
    st, sz, _ = layout[name]
    return slice(st, st + sz)

  sl = {k: term(k) for k in ("pelvis_height", "projected_gravity", "joint_pos", "joint_vel",
                             "base_lin_vel", "base_ang_vel") if k in layout}
  rep = {"tag": a.tag, "policy": os.path.abspath(a.policy), "episodes": N, "horizon": H,
         "dv_range": list(a.dv_range), "real_obs_noise": bool(a.obs_noise), "hip": a.hip,
         "actor_obs_rebuilt_from_state_max_gap": rebuilt_gap,
         "real_fell_within_horizon": float(real_fell_any.float().mean()),
         "z_clip": z_clip, "modes": {}}
  marks = [k for k in MARKS if k <= H]
  for mode in MODES:
    Zp, gp, pp = roll(mode)
    Sp = wm.decode(Zp)
    # survival-weighted termination: P(terminated by step k)
    p_by = 1.0 - torch.cumprod(1.0 - pp, 0)
    gmin_p = torch.cummin(gp, 0).values
    gmin_r = torch.cummin(torch.nan_to_num(G, nan=-1.0), 0).values  # NaN = ended: a fall
    rows_out = {}
    for k in marks:
      i = k - 1
      m = ALIVE[i]          # the real s_k exists: the episode is still running
      n_ok = int(m.sum())
      dz = (Zp[i] - Zr[i]) * dyn
      rmse = (dz.pow(2).sum(-1) / n_dyn).sqrt()
      ch_r = ((Zr[i] - z0) * dyn).norm(dim=-1)
      rel = dz.norm(dim=-1) / ch_r.clamp_min(1e-6)
      row = {"n": n_ok,
             "z_rmse_median": float(rmse[m].median()) if n_ok else None,
             "z_rmse_p90": float(rmse[m].quantile(0.9)) if n_ok else None,
             "rel_change_err_median": float(rel[m].median()) if n_ok else None}
      sp, sr = Sp[i], S[k]
      if "pelvis_height" in sl:
        row["pelvis_height_m"] = float((sp[:, sl["pelvis_height"]] - sr[:, sl["pelvis_height"]])
                                       .abs().squeeze(-1)[m].median())
      if "projected_gravity" in sl:
        row["tilt_deg"] = float((_tilt_deg(sp[:, sl["projected_gravity"]])
                                 - _tilt_deg(sr[:, sl["projected_gravity"]])).abs()[m].median())
      if "joint_pos" in sl:
        row["joint_pos_rad"] = float((sp[:, sl["joint_pos"]] - sr[:, sl["joint_pos"]])
                                     .abs().mean(-1)[m].median())
      if "joint_vel" in sl:
        row["joint_vel_rad_s"] = float((sp[:, sl["joint_vel"]] - sr[:, sl["joint_vel"]])
                                       .abs().mean(-1)[m].median())
      if "base_lin_vel" in sl:
        row["base_lin_vel_m_s"] = float((sp[:, sl["base_lin_vel"]] - sr[:, sl["base_lin_vel"]])
                                        .norm(dim=-1)[m].median())
      if "base_ang_vel" in sl:
        row["base_ang_vel_rad_s"] = float((sp[:, sl["base_ang_vel"]] - sr[:, sl["base_ang_vel"]])
                                          .norm(dim=-1)[m].median())
      row["g_abs_err_median"] = float((gp[i] - G[i]).abs()[m].median()) if n_ok else None
      # the fall verdict within k steps, over every episode
      fell_r = (gmin_r[i] < 0) | (ended <= k)
      fell_p = (gmin_p[i] < 0) | (p_by[i] > 0.5)
      row["verdict_agreement"] = float((fell_r == fell_p).float().mean())
      row["real_fell"] = float(fell_r.float().mean())
      row["false_safe"] = float((fell_r & ~fell_p).float().mean())   # model: fine; real: fell
      row["false_fail"] = float((~fell_r & fell_p).float().mean())
      rows_out[str(k)] = row
    rep["modes"][mode] = rows_out

  # the per-term picture at 1 and 32 steps, open loop: which parts of the state are hard
  Zp, _gp, _pp = roll("open loop")
  per = {}
  for name, (st, sz, _c) in layout.items():
    idx = torch.arange(st, st + sz, device=dev)
    d = dyn[idx]
    if not bool(d.any()):
      continue
    e = {}
    for k in (1, 8, 32):
      if k > H:
        continue
      m = ALIVE[k - 1]
      dz = (Zp[k - 1][:, idx] - Zr[k - 1][:, idx])[:, d]
      ch = (Zr[k - 1][:, idx] - z0[:, idx])[:, d].norm(dim=-1)
      if float(ch[m].median()) < 1e-4:
        continue                      # a term that does not move (the zeroed command)
      e[str(k)] = float((dz.norm(dim=-1) / ch.clamp_min(1e-6))[m].median())
    if e:
      per[name] = e
  rep["open_loop_rel_change_err_by_term"] = per
  rep["wall_s"] = time.time() - t0

  print(f"\n[wm-acc] {a.tag}: {N} kicked held-out starts, |dv| ~ U{tuple(a.dv_range)} m/s, "
        f"{H} steps; real SHAC noise {'on' if a.obs_noise else 'off'}; real falls within "
        f"{a.horizon_s:g} s {rep['real_fell_within_horizon']:.3f}; actor obs rebuilt from the "
        f"state within {rebuilt_gap:.2e}")
  cols = ("n", "z_rmse_median", "rel_change_err_median", "pelvis_height_m", "tilt_deg",
          "joint_pos_rad", "joint_vel_rad_s", "base_lin_vel_m_s", "g_abs_err_median",
          "verdict_agreement", "false_safe", "false_fail")
  for mode in MODES:
    print(f"  {mode}")
    print("    " + f"{'k':>4s} " + " ".join(f"{c[:13]:>13s}" for c in cols))
    for k, row in rep["modes"][mode].items():
      print("    " + f"{k:>4s} " + " ".join(
        f"{row.get(c):13.4f}" if isinstance(row.get(c), float) else f"{str(row.get(c)):>13s}"
        for c in cols))
  print("  open-loop relative error of the predicted change, by critic-obs term (k = 1 / 8 / 32):")
  for name, e in per.items():
    print(f"    {name:22s} " + "  ".join(f"{e.get(k, float('nan')):.3f}" for k in ("1", "8", "32")))
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[wm-acc] wrote {a.out} ({rep['wall_s']:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
