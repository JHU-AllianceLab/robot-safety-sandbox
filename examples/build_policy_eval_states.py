"""Query states for evaluating a fallback policy, from SONIC's trajectories.

    python examples/build_policy_eval_states.py \
        --train-traj data/g1_sonic_push_train.pt --test-traj data/g1_sonic_push_test.pt \
        --out-dir data/policy_eval

The states a safety monitor is asked about are the ones SONIC visits:
every step of its harvested trajectories (``harvest_g1_sonic_trajectories.py``),
before and after the kick. Each becomes a start-state row
(``policy_eval.trajectory_rows``: pose, twist, joints, and SONIC's target
INTO the state as the last action) that the K env can restore, for
``label_policy_values.py``.

SPLITS, by SONIC episode (the trajectories of one episode are correlated):

    train   the train file's episodes minus the calibration ones; every
            ``--stride``-th step (1 <= j <= length - 2: the last recorded
            step of a survivor has no next action for Q), all of them
    calib   ``--calib-frac`` of the train file's episodes, or every episode of
            ``--calib-traj`` (a dedicated harvest on the train clips: the
            false-safe threshold at alpha = 0.1 %, beta = 1e-3 needs ~12k
            ACCEPTED calibration states with no error); ONE uniformly random
            step per episode (exchangeable samples for conformal)
    test    the test file (held-out clips); one random step per episode

Q ACTIONS. Every state also gets actions for Q^pi(s, a): SONIC's own next
action (variant 0; the target it applies FROM the state, in env action units,
not clamped) and SONIC's plus Gaussian noise of each ``--q-stds`` (variants
1, 2, ..., env action units, i.e. joint radians over the env's per-joint
scale). ``<split>_q.pt`` holds one row per (state, variant) for train and ONE
random variant per state for calib / test, with ``q_action`` and ``variant``
columns and ``v_row`` (the state's row in ``<split>_v.pt``).

Output per split: ``<split>_v.pt`` and ``<split>_q.pt`` in the start-bank
schema (``states.py``; every row KIND_HANDOVER, no pending push) plus the
evaluation columns ``step`` / ``t_rel`` (s from the kick) / ``dv`` /
``sonic_action``, which ``load_bank`` ignores and the evaluation scripts read
from the file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

from robot_safety_sandbox.envs.g1_recover.policy_eval import trajectory_rows  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import FIELDS, save_bank  # noqa: E402


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--train-traj", default=os.path.join(_ZOO, "data", "g1_sonic_push_train.pt"))
  p.add_argument("--test-traj", default=os.path.join(_ZOO, "data", "g1_sonic_push_test.pt"))
  p.add_argument("--out-dir", default=os.path.join(_ZOO, "data", "policy_eval"))
  p.add_argument("--stride", type=int, default=5, help="train split: every n-th step")
  p.add_argument("--calib-frac", type=float, default=0.2,
                 help="calibration episodes carved from --train-traj (unless --calib-traj)")
  p.add_argument("--calib-traj", default=None,
                 help="a separate trajectory file whose every episode calibrates "
                      "(then --train-traj is all train)")
  p.add_argument("--max-train-episodes", type=int, default=None)
  p.add_argument("--q-stds", type=float, nargs="*", default=[0.3, 1.0],
                 help="Gaussian action noise around SONIC's action, one variant per std")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def _q_rows(rows: dict, stds, all_variants: bool, g: torch.Generator) -> dict:
  n = int(rows["root_z"].numel())
  nv = 1 + len(stds)
  if all_variants:
    v_row = torch.arange(n).repeat_interleave(nv)
    variant = torch.arange(nv).repeat(n)
  else:
    v_row = torch.arange(n)
    variant = torch.randint(0, nv, (n,), generator=g)
  out = {k: (v[v_row] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v)
         for k, v in rows.items()}
  std = torch.tensor([0.0] + list(stds))[variant]
  base = rows["sonic_action"][v_row]
  out["q_action"] = base + std[:, None] * torch.randn(base.shape, generator=g)
  out["variant"] = variant
  out["v_row"] = v_row
  return out


def main(argv=None):
  a = _parse(argv)
  t0 = time.time()
  os.makedirs(a.out_dir, exist_ok=True)
  g = torch.Generator().manual_seed(a.seed)
  rng = np.random.default_rng(a.seed)
  tr = torch.load(a.train_traj, map_location="cpu", weights_only=False)
  te = torch.load(a.test_traj, map_location="cpu", weights_only=False)
  E = int(tr["ep/length"].numel())
  perm = rng.permutation(E)
  ca = tr
  if a.calib_traj:
    ca = torch.load(a.calib_traj, map_location="cpu", weights_only=False)
    cal_e, train_e = np.arange(int(ca["ep/length"].numel())), np.sort(perm)
  else:
    n_cal = int(round(a.calib_frac * E))
    cal_e, train_e = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
  if a.max_train_episodes:
    # a RANDOM subset: the trajectory file is in emission order (falls first)
    train_e = np.sort(rng.permutation(train_e)[:a.max_train_episodes])

  def one_step(res, eps):
    eps = torch.as_tensor(eps, dtype=torch.long)
    L = res["ep/length"][eps]
    j = 1 + (torch.rand(len(eps), generator=g) * (L - 2).float()).long()
    return eps, j

  def strided(res, eps):
    es, js = [], []
    for e in eps:
      L = int(res["ep/length"][e])
      j = torch.arange(1, L - 1, a.stride)
      es.append(torch.full_like(j, int(e)))
      js.append(j)
    return torch.cat(es), torch.cat(js)

  splits = {"train": (tr, strided(tr, train_e), True),
            "calib": (ca, one_step(ca, cal_e), False),
            "test": (te, one_step(te, np.arange(int(te["ep/length"].numel()))), False)}
  summary = {}
  for name, (res, (e, j), all_var) in splits.items():
    rows = trajectory_rows(res, e, j)
    assert all(f in rows for f in FIELDS)
    meta = {"source": "build_policy_eval_states.py", "split": name,
            "trajectories": os.path.abspath(a.train_traj if res is tr else
                                            a.calib_traj if res is ca else a.test_traj),
            "library": res["meta"]["library"], "clips": list(res["meta"]["clips"]),
            "categories": list(res["meta"]["categories"]), "task": res["meta"]["task"],
            "stride": a.stride if name == "train" else None, "q_stds": list(a.q_stds),
            "one_state_per_episode": name != "train", "seed": a.seed,
            "env_action_scale": res["meta"]["env_action_scale"],
            "env_action_offset": res["meta"]["env_action_offset"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    vb = dict(rows, meta=dict(meta, kind="v", rows=int(e.numel())))
    save_bank(vb, os.path.join(a.out_dir, f"{name}_v.pt"))
    qr = _q_rows(rows, a.q_stds, all_var, g)
    qb = dict(qr, meta=dict(meta, kind="q", rows=int(qr["root_z"].numel())))
    save_bank(qb, os.path.join(a.out_dir, f"{name}_q.pt"))
    summary[name] = {"episodes": int(torch.unique(e).numel()), "v_rows": int(e.numel()),
                     "q_rows": int(qr["root_z"].numel()),
                     "pre_kick_frac": float((rows["t_rel"] < 0).float().mean())}
    print(f"[eval-states] {name}: {summary[name]}")
  print(f"[eval-states] wrote {a.out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
  main()
