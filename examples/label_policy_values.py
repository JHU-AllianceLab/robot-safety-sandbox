"""Exact (gamma = 1) safety values of a fixed policy, by rollout.

    python examples/label_policy_values.py --policy <run>/final_model.zip \
        --states data/policy_eval/train_v.pt --k 2
    python examples/label_policy_values.py --policy <run>/final_model.zip \
        --states data/policy_eval/train_q.pt --mode q --k 1

For every row of a states file (``build_policy_eval_states.py``) and each of
``--k`` rollouts, the env restores the row in the ``--task`` env (default
``g1_recover_lafan_k``: K is the margin and the termination) and the policy
runs deterministically for ``--horizon-s`` seconds (default 3; the episode cap,
no rest truncation, no push). The rollout's label is

    v = min over the episode of g_k: at the restored state, after every step,
        and at the step that leaves K (g_k <= -0.1 there),

i.e. V^pi(s) for ``--mode v`` (``policy_eval`` module docstring). ``--mode
q`` plays the row's ``q_action`` (env action units, not clamped) for the
first step and the policy afterwards, so v = min(g(s), V^pi_{T-1}(s')) =
Q^pi(s, a); a recurrent policy's memory then starts at s', where it takes
over. A row's k rollouts differ by the env's domain randomization (foot
friction, torso CoM, encoder bias: one draw per env).

The spawn walks the file with the SEQUENTIAL sampler, each row k times in a
row, and an ending episode is booked on the row it started from (the
attribution of ``label_start_states.py``: only first-pass episodes, the first
k per row).

OUTPUT (``--out``, default ``<states stem>_labels_<policy tag>.pt``): ``v``
``[R, k]`` (NaN where a row got fewer), ``safe`` = v >= 0, ``ttf`` (s to the
K exit, NaN if it stayed), and ``meta`` (policy, task, horizon, mode, k).
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
from robot_safety_sandbox.envs.g1_recover.states import load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True)
  p.add_argument("--states", required=True)
  p.add_argument("--mode", choices=("v", "q"), default="v")
  p.add_argument("--out", default=None)
  p.add_argument("--k", type=int, default=1, help="rollouts per row")
  p.add_argument("--task", default="g1_recover_lafan_k")
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--max-rows", type=int, default=None, help="label only the first n rows")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  if a.out is None:
    tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.policy)))) or "policy"
    a.out = f"{os.path.splitext(a.states)[0]}_labels_{tag}.pt"
  return a


def _rank_in_group(r: torch.Tensor) -> torch.Tensor:
  """For each entry of ``r``, how many EARLIER entries hold the same value."""
  n = r.numel()
  if n == 0:
    return r
  order = torch.argsort(r, stable=True)
  rs = r[order]
  ar = torch.arange(n, device=r.device)
  new = torch.ones(n, dtype=torch.bool, device=r.device)
  new[1:] = rs[1:] != rs[:-1]
  first = torch.cummax(torch.where(new, ar, torch.zeros_like(ar)), 0).values
  out = torch.empty_like(ar)
  out[order] = ar - first
  return out


def rollouts(env, model, norm, margin_fn, R: int, k: int, q_action, max_rows) -> dict:
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  target = torch.ones(R, dtype=torch.bool, device=dev)
  if max_rows is not None:
    target[int(max_rows):] = False
  n_target = int(target.sum())
  counts = torch.zeros(R, dtype=torch.long, device=dev)
  v_out = torch.full((R, k), float("nan"), device=dev)
  t_out = torch.full((R, k), float("nan"), device=dev)
  obs = torch.nan_to_num(env.obs_groups()[env.obs_key].float(), nan=0.0, posinf=0.0, neginf=0.0)
  cur_row = mj._zoo_spawn_row.clone()
  seq_len = int(mj._zoo_spawn_seq_len)
  ar = torch.arange(n, device=dev)
  first = (int(mj._zoo_spawn_cursor) - n + ar) < seq_len
  vmin = margin_fn(mj)[0].clone()              # g at the restored state
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  reset_policy_memory(model.policy)
  remaining, it, since, t0 = n_target, 0, 0, time.time()
  stall = 3 * int(mj.max_episode_length) + 10
  while remaining > 0:
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    if q_action is not None:
      over = elapsed == 0
      act = torch.where(over.unsqueeze(1), q_action[cur_row.clamp_min(0)], act)
      reset_policy_memory(model.policy, over)    # the policy takes over at s'
    cursor = int(mj._zoo_spawn_cursor)
    obs, g, dones, timeouts, _l = env.step_tensor(act)
    vmin = torch.minimum(vmin, g)
    elapsed += 1
    it += 1
    since += 1
    if bool(dones.any()):
      d = dones.nonzero(as_tuple=False).squeeze(-1)
      rows = cur_row[d]
      valid = (rows >= 0) & first[d]
      safe = rows.clamp_min(0)
      valid &= target[safe]
      rank = _rank_in_group(torch.where(valid, safe, torch.full_like(safe, -1)))
      acc = valid & (counts[safe] + rank < k)
      if bool(acc.any()):
        ra, slot = rows[acc], (counts[safe] + rank)[acc]
        v_out[ra, slot] = vmin[d][acc]
        fell = ~timeouts[d][acc]
        t_out[ra, slot] = torch.where(fell, elapsed[d][acc].float() * dt,
                                      torch.full_like(slot, 0).float() * float("nan"))
        counts.index_add_(0, ra, torch.ones_like(ra))
        since = 0
        remaining = int((target & (counts < k)).sum())
      cur_row[d] = mj._zoo_spawn_row[d]
      first[d] = (cursor + ar[:d.numel()]) < seq_len
      reset_policy_memory(model.policy, dones)
      vmin[d] = margin_fn(mj)[0][d]             # g at the newly restored states
      elapsed[d] = 0
    if since > stall:
      raise SystemExit(f"[values] no episode booked in {since} steps; {remaining} rows short")
    if it % 250 == 0:
      el = max(time.time() - t0, 1e-6)
      print(f"[values] step {it}  rows done {n_target - remaining}/{n_target}  "
            f"({it * n / el:.0f} env-steps/s)", flush=True)
  return {"v": v_out.cpu(), "ttf": t_out.cpu(), "n_rows": n_target,
          "wall_s": time.time() - t0, "env_steps": it * n}


def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  bank = load_bank(a.states, "cpu")
  raw = torch.load(a.states, map_location="cpu", weights_only=False)
  R = int(bank["root_z"].numel())
  q_action = None
  if a.mode == "q":
    if "q_action" not in raw:
      raise SystemExit(f"{a.states} has no q_action column (use a *_q.pt file)")
    q_action = raw["q_action"].float().to(dev)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  overrides = dict(dataset=a.states, sampler="sequential", sequential_repeats=int(a.k),
                   episode_length_s=float(a.horizon_s), rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_tensor(a.task, a.num_envs, dev, cfg_overrides=overrides, random_push=push_kw)
  print(f"[values] {a.policy} on {a.task} x {a.num_envs}: {R} rows x {a.k} rollouts, "
        f"mode {a.mode}, horizon {a.horizon_s} s")
  out = rollouts(env, model, norm, spec(a.task).margin_fn, R, a.k, q_action, a.max_rows)
  v = out["v"]
  res = {"v": v, "safe": v >= 0.0, "ttf": out["ttf"],
         "meta": {"source": "label_policy_values.py", "policy": os.path.abspath(a.policy),
                  "states": os.path.abspath(a.states), "mode": a.mode, "task": a.task,
                  "horizon_s": a.horizon_s, "k": a.k, "rows": R, "rows_labeled": out["n_rows"],
                  "env_steps": out["env_steps"], "wall_s": out["wall_s"],
                  "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
  torch.save(res, a.out)
  lab = ~torch.isnan(v)
  print(f"[values] {int(lab.sum())} rollouts: safe {float((v[lab] >= 0).float().mean()):.3f}, "
        f"mean v {float(v[lab].mean()):+.3f} -> {a.out} ({out['wall_s']:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
