"""Record a fixed policy's rollouts step by step, for bootstrapped (TD) values.

    python examples/record_policy_transitions.py --policy <run>/final_model.zip \
        --states data/policy_eval_gfall/train_v.pt --rows 50000 --task g1_recover_lafan \
        --out data/policy_eval_gfall/td_train_v_traj_<tag>.pt
    python examples/record_policy_transitions.py --policy <run>/final_model.zip \
        --states data/policy_eval_gfall/train_q.pt --mode q --horizon-s 0.02 \
        --task g1_recover_lafan --out data/policy_eval_gfall/td_train_q_step_<tag>.pt

The rollouts of ``label_policy_values.py`` -- the same env (``--task``, no push,
no rest truncation, domain randomization and observation noise on), the policy's
mean action clamped to the action box, and under ``--mode q`` the row's
``q_action`` played first -- but where the labeller keeps one number per start
row (the min of g over the rollout), this keeps every state of the first
``T = horizon / dt`` steps, so ``fit_policy_value_td.py`` can bootstrap
``V(s_t) <- min(g(s_t), V(s_{t+1}))`` along the trajectory and at the
truncation, instead of stopping at T.

Each selected row (``--rows`` random rows of ``--states``, written next to
``--out`` as ``<out stem>_states.pt``; every row when omitted) gets ONE rollout.
Per row r and step t = 0..T:

    x         [R, T + 1, 97] float16   the value's input at s_t (policy_eval.features:
                                       phi(s_t) and the last applied action, clamped
                                       to the box like every bank row); 0 past the end
    g         [R, T + 1]               g(s_t), the task's clamped margin, computed
                                       before the auto-reset (a fall state keeps its
                                       g, anchored <= -0.1); NaN past the end
    n_states  [R]                      states with an x: T + 1 when the rollout
                                       reached step T, else the fall step L (x up to
                                       s_{L-1}, g up to s_L)
    fell_at   [R]                      the fall step L <= T, or -1
    rows      [R]                      each row's index in --states
    meta                               policy, task, states, mode, T, dt, ...

``--q-center pi`` (``--mode q``) plays pi's own first action plus the row's
noise ``q_action - sonic_action`` instead, as ``label_policy_values.py
--q-center pi`` does, and adds the played actions as ``q_action`` [R, 29].

``--no-features`` keeps g only (no x): a longer-horizon check of the labels.
The episode is capped two steps past T, so s_T is observed before the
timeout's auto-reset; whatever happens after step T is discarded.
"""

from __future__ import annotations

import argparse
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
from robot_safety_sandbox.envs.g1_recover.eta_net import phi_of_bank  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import save_bank, snapshot  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402

#: The G1 tasks' control step (sim 200 Hz, decimation 4); checked against the env.
DT = 0.02


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True)
  p.add_argument("--states", required=True)
  p.add_argument("--mode", choices=("v", "q"), default="v")
  p.add_argument("--q-center", choices=("row", "pi"), default="row",
                 help="--mode q: play the row's q_action (row), or pi's own action plus "
                      "the row's noise q_action - sonic_action (pi)")
  p.add_argument("--out", required=True)
  p.add_argument("--rows", type=int, default=None,
                 help="record this many random rows of --states (default: all)")
  p.add_argument("--task", default="g1_recover_lafan_k")
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--no-features", action="store_true", help="keep g only, no x")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  if a.q_center == "pi" and a.mode != "q":
    p.error("--q-center pi needs --mode q")
  return a


def _subset(raw: dict, idx: torch.Tensor) -> dict:
  """Rows ``idx`` of a states file: every per-row tensor subset, the rest kept."""
  n = int(raw["root_z"].numel())
  return {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == n else v)
          for k, v in raw.items()}


def record(env, model, norm, margin_fn, R: int, T: int, q_action, keep_x: bool,
           q_noise=None) -> dict:
  """``q_noise`` [R, 29] in place of ``q_action``: the first action is the
  policy's own plus the row's noise, returned as ``q_action``."""
  mj, dev, n = env.mj, env.device, env.num_envs
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)

  def feats() -> torch.Tensor:
    """policy_eval.features of every env's current state, on the device."""
    s = snapshot(mj)
    a = s["action"].float()
    a = torch.maximum(torch.minimum(a, high[:a.shape[1]]), low[:a.shape[1]])
    return torch.cat([phi_of_bank(s, device=dev).float(), a], dim=1)

  x0 = feats()
  X = torch.zeros(R, T + 1, x0.shape[1], dtype=torch.float16, device=dev) if keep_x else None
  G = torch.full((R, T + 1), float("nan"), device=dev)
  n_states = torch.zeros(R, dtype=torch.long, device=dev)
  fell_at = torch.full((R,), -1, dtype=torch.long, device=dev)
  done_row = torch.zeros(R, dtype=torch.bool, device=dev)

  obs = torch.nan_to_num(env.obs_groups()[env.obs_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
  cur_row = mj._zoo_spawn_row.clone()
  seq_len = int(mj._zoo_spawn_seq_len)
  ar = torch.arange(n, device=dev)
  first = (int(mj._zoo_spawn_cursor) - n + ar) < seq_len
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  # an env records while it runs a row's FIRST pass, until the fall or step T
  active = first & (cur_row >= 0)

  def start(e: torch.Tensor, x_all: torch.Tensor) -> None:
    r = cur_row[e]
    G[r, 0] = margin_fn(mj)[0][e]                # g at the restored state
    if X is not None:
      X[r, 0] = x_all[e].half()

  start(ar[active], x0)
  played = (torch.full((R, q_noise.shape[1]), float("nan"), device=dev)
            if q_noise is not None else None)
  reset_policy_memory(model.policy)
  remaining, it, since, t0 = R, 0, 0, time.time()
  stall = 3 * int(mj.max_episode_length) + 10
  while remaining > 0:
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    if q_action is not None or q_noise is not None:
      over = elapsed == 0
      if q_noise is not None:
        first_act = act + q_noise[cur_row.clamp_min(0)]
        mine = over & active                       # a row's recorded (first) pass
        played[cur_row[mine]] = first_act[mine]
      else:
        first_act = q_action[cur_row.clamp_min(0)]
      act = torch.where(over.unsqueeze(1), first_act, act)
      reset_policy_memory(model.policy, over)    # the policy takes over at s'
    cursor = int(mj._zoo_spawn_cursor)
    obs, g, dones, timeouts, _l = env.step_tensor(act)
    elapsed += 1
    it += 1
    since += 1
    # after the step: s_t for an env that goes on, the NEW episode's s_0 for one
    # that ended in it (mjlab auto-resets inside the step); g is pre-reset
    x_all = feats() if X is not None else None
    rec = active & (elapsed <= T)
    if bool(rec.any()):
      e = rec.nonzero(as_tuple=False).squeeze(-1)
      r, t, de = cur_row[e], elapsed[e], dones[e]
      G[r, t] = g[e]
      alive = ~de
      if X is not None and bool(alive.any()):
        X[r[alive], t[alive]] = x_all[e[alive]].half()
      fell = de & ~timeouts[e]
      fell_at[r[fell]] = t[fell]
      end = de | (t == T)
      if bool(end.any()):
        n_states[r[end]] = torch.where(de, t, t + 1)[end]
        done_row[r[end]] = True
        active[e[end]] = False
        remaining = int((~done_row).sum())
        since = 0
    if bool(dones.any()):
      d = dones.nonzero(as_tuple=False).squeeze(-1)
      cur_row[d] = mj._zoo_spawn_row[d]
      first[d] = (cursor + ar[:d.numel()]) < seq_len
      reset_policy_memory(model.policy, dones)
      elapsed[d] = 0
      new = first[d] & (cur_row[d] >= 0)
      new &= ~done_row[cur_row[d].clamp_min(0)]
      active[d] = new
      if bool(new.any()):
        if x_all is None:
          x_all = x0                               # unused without x
        start(d[new], x_all)
    if since > stall:
      raise SystemExit(f"[record] no rollout finished in {since} steps; {remaining} rows short")
    if it % 250 == 0:
      el = max(time.time() - t0, 1e-6)
      print(f"[record] step {it}  rows done {R - remaining}/{R}  "
            f"({it * n / el:.0f} env-steps/s)", flush=True)
  out = {"x": X.cpu() if X is not None else None, "g": G.cpu(), "n_states": n_states.cpu(),
         "fell_at": fell_at.cpu(), "wall_s": time.time() - t0, "env_steps": it * n}
  if played is not None:
    out["q_action"] = played.cpu()
  return out


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  raw = torch.load(a.states, map_location="cpu", weights_only=False)
  n_all = int(raw["root_z"].numel())
  if a.rows is not None and a.rows < n_all:
    gen = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(n_all, generator=gen)[:a.rows].sort().values
    raw = _subset(raw, idx)
    states = os.path.splitext(a.out)[0] + "_states.pt"
    save_bank(raw, states)
  else:
    idx, states = torch.arange(n_all), a.states
  R = int(raw["root_z"].numel())
  q_action = q_noise = None
  if a.mode == "q":
    if "q_action" not in raw:
      raise SystemExit(f"{a.states} has no q_action column (use a *_q.pt file)")
    q_action = raw["q_action"].float().to(a.device)
    if a.q_center == "pi":
      if "sonic_action" not in raw:
        raise SystemExit(f"{a.states} has no sonic_action column (build_policy_eval_states.py)")
      q_noise, q_action = q_action - raw["sonic_action"].float().to(a.device), None
  T = int(round(a.horizon_s / DT))
  model, norm = load_twin(a.policy, a.device)
  model.policy.set_training_mode(False)
  overrides = dict(dataset=states, sampler="sequential", sequential_repeats=1,
                   episode_length_s=(T + 1.5) * DT, rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_tensor(a.task, a.num_envs, a.device, cfg_overrides=overrides, random_push=push_kw)
  dt = float(env.mj.step_dt)
  if abs(dt - DT) > 1e-9 or int(env.mj.max_episode_length) < T + 1:
    raise SystemExit(f"[record] control step {dt} s / cap {env.mj.max_episode_length} steps: "
                     f"expected {DT} s and >= {T + 1}")
  print(f"[record] {a.policy} on {a.task} x {a.num_envs}: {R} rows of {a.states}, mode "
        f"{a.mode}{' (pi-centred)' if q_noise is not None else ''}, T = {T} steps "
        f"({T * dt:.2f} s){', g only' if a.no_features else ''}")
  out = record(env, model, norm, spec(a.task).margin_fn, R, T, q_action, not a.no_features,
               q_noise)
  if "q_action" in out and bool(torch.isnan(out["q_action"]).any()):
    raise SystemExit(f"[record] {int(torch.isnan(out['q_action']).any(1).sum())} rows have no "
                     f"played action")
  out["rows"] = idx
  out["meta"] = {"source": "record_policy_transitions.py", "policy": os.path.abspath(a.policy),
                 "states": os.path.abspath(a.states), "states_used": os.path.abspath(states),
                 "mode": a.mode, "q_center": a.q_center, "task": a.task, "T": T, "dt": dt,
                 "horizon_s": T * dt,
                 "rows": R, "env_steps": out["env_steps"], "wall_s": out["wall_s"],
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  torch.save(out, a.out)
  fell = out["fell_at"] >= 0
  v = torch.where(torch.isnan(out["g"]), torch.full_like(out["g"], float("inf")), out["g"]).amin(1)
  print(f"[record] {R} rollouts: fell within T {float(fell.float().mean()):.3f}, "
        f"min g >= 0 {float((v >= 0).float().mean()):.3f}, mean states "
        f"{float(out['n_states'].float().mean()):.1f} -> {a.out} ({out['wall_s']:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
