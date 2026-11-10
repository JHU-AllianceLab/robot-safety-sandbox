"""Does the real noisy trajectory stay inside the hull of R noisy rollouts, and how big is the hull?

    python examples/rollout_hull_coverage.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip --tag shac_s0

From --episodes kicked held-out states (a spawn on a start_ok row of the
held-out clip library, then the eval's kick: |dv| ~ U(--dv-range) m/s,
horizontal, uniform angle), SHAC drives the REAL env with its sensor noise ON
for --horizon-s, and from the same state R = --rollouts rollouts are run with
SHAC reading each through the same sensor-noise law (a fresh draw per rollout
and step, from the first step on):

  cpu  exact CPU MuJoCo (``cpu_rollout.Roller``: the exact state and the
       robot's DR draw), run on past a fall (the hull needs every rollout);
  wm   SHAC's world model on the GPU, from the true privileged state.

Per step k, over the episodes the real env has not ended by step k, in a few
low-dimensional spaces (a hull of 32 points in the 71-D physics state is a
31-D sliver no new point lands in):

  1-D  g (the fall margin), pelvis height (m), tilt (deg)  -- the hull is [min, max]
  2-D  horizontal body velocity (the IMU velocimeter, body frame, m/s);
       lean (projected gravity x, y)
  3-D  lean x, lean y, pelvis height

COVERAGE: the share of episodes whose real point lies in the hull of the R
rollouts (R = 8 / 16 / 32: the first R of them), and the share whose real
trajectory stayed inside at EVERY step so far. The BENCHMARK, for a rollout
distribution that is exactly the real one: the real point is then one more
exchangeable sample, so it lies in a 1-D range of R samples with probability
(R - 1) / (R + 1); for 2-D / 3-D it is estimated leave-one-out -- each rollout
against the hull of the other R - 1, next to the real point against the same
R - 1 hulls (``loo_real`` vs ``loo_self``: equal in expectation iff calibrated).
SIZE: the hull's width (1-D), area (2-D) or volume (3-D), median over
episodes.

Writes ``--out`` (json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay, QhullError

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
import mujoco  # noqa: E402
from eval_sonic_qcbf import write_episode_bank  # noqa: E402

from robot_safety_sandbox.base import MjlabTensorSafetyEnv  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import use_sonic_hip_pitch  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES, RecoverSpawn  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import KIND_MOCAP, load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402

#: The spaces: name -> (dim, feature columns). Features per step, in this order:
FEATURES = ("g", "pelvis_z", "tilt_deg", "vel_x", "vel_y", "lean_x", "lean_y")
SPACES = {"g": ["g"], "pelvis height": ["pelvis_z"], "tilt": ["tilt_deg"],
          "velocity xy": ["vel_x", "vel_y"], "lean xy": ["lean_x", "lean_y"],
          "lean xy + height": ["lean_x", "lean_y", "pelvis_z"]}
SUBSETS = (8, 16, 32)
MARKS = (1, 2, 5, 10, 20, 35, 50, 75, 100, 150)


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="a SafetySHAC1P final_model.zip")
  p.add_argument("--tag", required=True)
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"))
  p.add_argument("--hip", choices=("mjlab", "sonic"), default="mjlab")
  p.add_argument("--episodes", type=int, default=128)
  p.add_argument("--rollouts", type=int, default=32)
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--dv-range", type=float, nargs=2, default=[0.0, 5.0])
  p.add_argument("--predictors", nargs="+", choices=("cpu", "wm"), default=["cpu", "wm"])
  p.add_argument("--out", default=None,
                 help="default runs/shac_vs_ppo_sonic/rollout_hull/<tag>.json")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  a.out = a.out or os.path.join(_ZOO, "runs", "shac_vs_ppo_sonic", "rollout_hull", f"{a.tag}.json")
  return a


def _spawn_term(mj) -> RecoverSpawn:
  for cfgs in getattr(mj.event_manager, "_mode_term_cfgs", {}).values():
    for c in cfgs:
      if isinstance(getattr(c, "func", None), RecoverSpawn):
        return c.func
  raise SystemExit("[hull] the env has no RecoverSpawn reset term")


def _tilt(pg: np.ndarray) -> np.ndarray:
  c = np.clip(-pg[..., 2] / np.maximum(np.linalg.norm(pg, axis=-1), 1e-9), -1.0, 1.0)
  return np.degrees(np.arccos(c))


def _features(g, z, pg, vel) -> np.ndarray:
  """[..., 7] in FEATURES order."""
  return np.stack([g, z, _tilt(pg), vel[..., 0], vel[..., 1], pg[..., 0], pg[..., 1]], -1)


def _inside(points: np.ndarray, q: np.ndarray) -> bool:
  """``q`` in the convex hull of ``points`` [n, d] (d = 1: the [min, max] range)."""
  if points.shape[1] == 1:
    return bool(points.min() <= q[0] <= points.max())
  try:
    return bool(Delaunay(points, qhull_options="QJ").find_simplex(q[None])[0] >= 0)
  except (QhullError, ValueError):
    return False


def _size(points: np.ndarray) -> float:
  if points.shape[1] == 1:
    return float(points.max() - points.min())
  try:
    return float(ConvexHull(points, qhull_options="QJ").volume)
  except (QhullError, ValueError):
    return 0.0


@torch.no_grad()
def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  rng = np.random.default_rng(a.seed)
  dev = a.device
  t0 = time.time()
  N, R = int(a.episodes), int(a.rollouts)
  subsets = [r for r in SUBSETS if r <= R]
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)

  # --- the env (sensor noise ON), on kicked held-out spawns
  lib = load_bank(a.library, "cpu")
  pool = (lib["start_ok"] & (lib["kind"] == KIND_MOCAP)).nonzero(as_tuple=False).squeeze(-1)
  gen = torch.Generator().manual_seed(a.seed)
  rows = pool[torch.randint(0, pool.numel(), (N,), generator=gen)]
  out_dir = os.path.dirname(a.out)
  work = os.path.join(out_dir, f"{a.tag}_work")
  os.makedirs(work, exist_ok=True)
  bank_path = os.path.join(work, "spawn_bank.pt")
  write_episode_bank(lib, {"spawn": {"row": rows}}, bank_path)
  mag = a.dv_range[0] + (a.dv_range[1] - a.dv_range[0]) * torch.rand(N, generator=gen)
  ang = 2 * math.pi * torch.rand(N, generator=gen)
  s = spec(a.task)

  def builder(play: bool = False, **kw):
    cfg = s.cfg_builder(play=play, **kw)
    cfg.events.pop("push_robot", None)
    if a.hip == "sonic":
      use_sonic_hip_pitch(cfg)
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
  low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
  high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
  om = mj.observation_manager
  names, dims = om._group_obs_term_names[env.obs_key], om._group_obs_term_dim[env.obs_key]
  sizes = [int(np.prod(d)) for d in dims]
  i_act = names.index("actions")
  sl = slice(sum(sizes[:i_act]), sum(sizes[:i_act + 1]))

  def pi(obs):
    o = obs.clone()
    o[:, sl] = torch.maximum(torch.minimum(o[:, sl], high), low)
    return torch.clamp(model.policy._predict(norm(o), deterministic=True), low, high)

  # the kick
  robot = mj.scene["robot"]
  ids = torch.arange(N, device=dev)
  vel = robot.data.root_link_vel_w.clone()
  vel[:, 0] += mag.to(dev) * torch.cos(ang.to(dev))
  vel[:, 1] += mag.to(dev) * torch.sin(ang.to(dev))
  robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
  mj.sim.forward()
  qpos0, qvel0, ws0 = cr.live_state(env, ids)
  s0 = torch.nan_to_num(om.compute_group(env.critic_obs_key, update_history=False).float())
  obs = torch.nan_to_num(om.compute_group(env.obs_key, update_history=False).float())  # noisy
  layout = env._obs_layout(env.critic_obs_key)

  def crit(name, x):
    st, sz, _ = layout[name]
    return x[..., st:st + sz]

  # --- the real trajectory (noisy SHAC), its features per step
  feat_real = np.full((H, N, len(FEATURES)), np.nan)
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  ALIVE = np.zeros((H, N), dtype=bool)
  for t in range(H):
    obs, g, dones, _to, _l = env.step_tensor(pi(obs))
    c = env.critic_obs().float()
    f = _features(g.cpu().numpy(), crit("pelvis_height", c).squeeze(-1).cpu().numpy(),
                  crit("projected_gravity", c).cpu().numpy(), crit("base_lin_vel", c).cpu().numpy())
    a_np = (alive & ~dones).cpu().numpy()
    feat_real[t, a_np] = f[a_np]
    # the step that ends the episode still counts (its g is the fall)
    ended = (alive & dones).cpu().numpy()
    feat_real[t, ended, 0] = g.cpu().numpy()[ended]
    ALIVE[t] = a_np
    alive = alive & ~dones
  real_fell = (~ALIVE[-1]).mean()

  rep = {"tag": a.tag, "policy": os.path.abspath(a.policy), "episodes": N, "rollouts": R,
         "horizon": H, "dv_range": list(a.dv_range), "real_fell_within_horizon": float(real_fell),
         "spaces": {k: v for k, v in SPACES.items()},
         "benchmark_1d": {str(r): (r - 1) / (r + 1) for r in subsets}, "predictors": {}}

  # --- the predictors' R noisy rollouts, features [H, N, R, F]
  preds = {}
  if "cpu" in a.predictors:
    m_cpu, n_cpu = load_twin(a.policy, "cpu")
    m_cpu.policy.set_training_mode(False)
    actor = os.path.join(work, "actor.pt")
    cr.export_actor(m_cpu.policy, n_cpu, int(obs.shape[1]), sl, env.action_space.low,
                    env.action_space.high, env.action_space.low, env.action_space.high, actor)
    sp = cr.build_spec(env, actor, work)
    roller = cr.Roller(sp, list(range(N)), seed=a.seed)
    m0 = roller.models[0]
    v_adr = int(m0.sensor("robot/imu_lin_vel").adr[0])
    lo, span = roller.noise_lo, roller.noise_span
    F = np.full((H, N, R, len(FEATURES)), np.nan)
    ob = np.zeros((N * R, sp.obs_dim))
    t_c = time.time()
    items = [(i, r) for i in range(N) for r in range(R)]
    for j, (i, r) in enumerate(items):
      roller.seed(i, qpos0[i], qvel0[i], ws0[i], r)
      roller.observe(i, np.asarray(obs[i, sl].cpu().numpy(), dtype=np.float64), ob[j], r)
    # the first observation's last action is the spawn's (the live env's own)
    for t in range(H):
      x = ob + lo + span * rng.random(ob.shape)
      act = roller.actor(torch.from_numpy(x).float()).double().numpy()
      for j, (i, r) in enumerate(items):
        g = roller.control_step(i, act[j], r)
        roller.observe(i, act[j], ob[j], r)
        d = roller._d(i, r)
        q = d.qpos[sp.root_qadr:sp.root_qadr + 7]
        pg = cr._quat_rotate_inv(q[3:7], np.array([0.0, 0.0, -1.0]))
        F[t, i, r] = _features(np.float64(g), np.float64(q[2]), pg,
                               d.sensordata[v_adr:v_adr + 3])
      if t % 25 == 0:
        print(f"[hull] cpu step {t}/{H} ({time.time() - t_c:.0f}s)", flush=True)
    preds["cpu"] = F
  if "wm" in a.predictors:
    wm = model.world_model
    z_clip = getattr(model, "wm_z_clip", None)
    aidx = model._actor_idx_t.to(dev)
    n_lo, n_hi = model._noise_lo_t, model._noise_hi_t
    cn = wm.encode_context(env.sim_context().repeat_interleave(R, 0))
    z = wm.encode(s0.repeat_interleave(R, 0))
    F = np.full((H, N, R, len(FEATURES)), np.nan)
    for t in range(H):
      o = wm.decode(z).index_select(-1, aidx)
      o = o + n_lo + (n_hi - n_lo) * torch.rand_like(o)
      st = wm.step(z, pi(o), cn, z_clip=z_clip)
      z = st.z_next
      sd = wm.decode(z)
      f = _features(st.g.cpu().numpy(), crit("pelvis_height", sd).squeeze(-1).cpu().numpy(),
                    crit("projected_gravity", sd).cpu().numpy(), crit("base_lin_vel", sd).cpu().numpy())
      F[t] = f.reshape(N, R, -1)
    preds["wm"] = F

  # --- coverage and size
  col = {f: i for i, f in enumerate(FEATURES)}
  marks = [k for k in MARKS if k <= H]
  for pname, F in preds.items():
    out = {}
    for space, cols in SPACES.items():
      ci = [col[c] for c in cols]
      res = {}
      for r in subsets:
        inside_all = np.ones(N, dtype=bool)        # inside at every step so far
        per_k = {}
        for k in range(1, H + 1):
          i_k = k - 1
          m = ALIVE[i_k]
          ins = np.zeros(N, dtype=bool)
          loo_real, loo_self, size = [], [], []
          for e in np.flatnonzero(m):
            P = F[i_k, e, :r][:, ci]
            qv = feat_real[i_k, e, ci]
            ins[e] = _inside(P, qv)
            if k in marks:
              size.append(_size(P))
              if len(ci) > 1:
                for j in range(r):
                  rest = np.delete(P, j, axis=0)
                  loo_real.append(_inside(rest, qv))
                  loo_self.append(_inside(rest, P[j]))
          inside_all &= ins | ~m
          if k in marks:
            row = {"n": int(m.sum()), "coverage": float(ins[m].mean()) if m.any() else None,
                   "inside_every_step": float(inside_all[m].mean()) if m.any() else None,
                   "size_median": float(np.median(size)) if size else None}
            if len(ci) > 1 and loo_real:
              row["loo_real"] = float(np.mean(loo_real))
              row["loo_self"] = float(np.mean(loo_self))
            per_k[str(k)] = row
        res[str(r)] = per_k
      out[space] = res
    rep["predictors"][pname] = out
  rep["wall_s"] = time.time() - t0

  print(f"\n[hull] {a.tag}: {N} kicked held-out starts, {R} noisy rollouts each, {H} steps; "
        f"real falls within {a.horizon_s:g} s {real_fell:.3f}")
  for pname, out in rep["predictors"].items():
    for space, res in out.items():
      print(f"  {pname} / {space}")
      for r, per_k in res.items():
        bench = f"benchmark {(int(r) - 1) / (int(r) + 1):.2f}" if len(SPACES[space]) == 1 else ""
        print(f"    R={r:>2s} {bench}")
        for k, row in per_k.items():
          extra = (f"  loo real {row['loo_real']:.2f} / self {row['loo_self']:.2f}"
                   if "loo_real" in row else "")
          print(f"      k={k:>3s} n={row['n']:4d} coverage {row['coverage']:.2f}  inside every "
                f"step {row['inside_every_step']:.2f}  size {row['size_median']:.4g}{extra}")
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[hull] wrote {a.out} ({rep['wall_s']:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
