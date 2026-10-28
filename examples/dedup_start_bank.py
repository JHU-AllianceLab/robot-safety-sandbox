"""Thin a start-state bank to its greedy eta-net (the ViaLasso pool's nodes).

    python examples/dedup_start_bank.py --bank data/g1_sonic_push_train_bank.pt
    python examples/dedup_start_bank.py --bank data/g1_sonic_push_test_bank.pt \
        --W-from data/g1_sonic_push_train_bank_net.pt

Keeps a row iff no row kept before it lies within eta (default 0.1) in
``d(x, y) = ||W (phi(x) - phi(y))||``, visiting the rows in a seeded random
order (``envs/g1_recover/eta_net.py``). W is the bank's own pooled-std W, or,
with ``--W-from``, the W stored in another net's meta (so a test bank is
thinned in its train bank's units). The output is the bank restricted to the
kept rows, every column kept (``traj_index`` still names each row's SONIC
trajectory); its ``meta`` is the input's plus ``eta_net`` (eta, W, the order
seed, the kept row ids in the input, counts per handover delay and category).
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

from robot_safety_sandbox.envs.g1_recover.eta_net import (  # noqa: E402
  ETA, GROUP_WEIGHTS, compute_W, eta_net, phi_of_bank)
from robot_safety_sandbox.envs.g1_recover.states import load_bank, save_bank  # noqa: E402


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--bank", required=True)
  p.add_argument("--out", default=None, help="default: <bank without .pt>_net.pt")
  p.add_argument("--eta", type=float, default=ETA)
  p.add_argument("--W-from", default=None,
                 help="a thinned bank whose meta['eta_net']['W'] to use")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args(argv)


def _counts(bank: dict, rows: np.ndarray) -> dict:
  out = {}
  if "handover_delay_s" in bank:
    d = bank["handover_delay_s"][rows]
    out["per_delay"] = {f"{float(v):g}": int((d == v).sum()) for v in torch.unique(d)}
  cats = bank["meta"].get("categories")
  if cats and "category" in bank:
    c = bank["category"][rows]
    out["per_category"] = {name: int((c == i).sum()) for i, name in enumerate(cats)}
  return out


def main(argv=None):
  a = _parse(argv)
  t0 = time.time()
  bank = load_bank(a.bank, "cpu")
  raw = torch.load(a.bank, map_location="cpu", weights_only=False)
  n = int(bank["root_z"].numel())
  phi = phi_of_bank(bank)
  if a.W_from:
    W = torch.as_tensor(torch.load(a.W_from, map_location="cpu", weights_only=False)
                        ["meta"]["eta_net"]["W"], dtype=torch.float64)
    if W.numel() != phi.shape[1]:
      raise SystemExit(f"--W-from has {W.numel()} weights, phi here has {phi.shape[1]}")
  else:
    W = compute_W(phi)
  order = np.random.default_rng(a.seed).permutation(n)
  kept = eta_net(phi * W, a.eta, order=order, device=a.device)
  rows = np.sort(kept)
  print(f"[eta-net] {a.bank}: {n} rows -> {len(rows)} nodes at eta {a.eta:g} "
        f"({100 * len(rows) / n:.1f}%; {time.time() - t0:.0f}s)")
  idx = torch.as_tensor(rows)
  out = {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v)
         for k, v in raw.items() if k != "meta"}
  meta = dict(raw.get("meta", {}))
  meta["eta_net"] = {
    "source": os.path.abspath(a.bank), "eta": a.eta, "seed": a.seed,
    "order": "seeded random permutation", "W": W.tolist(),
    "W_from": os.path.abspath(a.W_from) if a.W_from else None,
    "group_weights": dict(GROUP_WEIGHTS), "phi_dim": int(phi.shape[1]),
    "rows_in": n, "rows_kept": int(len(rows)), "kept_rows": rows.tolist(),
    "input": _counts(bank, np.arange(n)), "kept": _counts(bank, rows),
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
  }
  meta["rows"] = int(len(rows))
  if "fell" in out:
    meta["fell_rows"] = int(out["fell"].sum())
  out["meta"] = meta
  dst = a.out or os.path.splitext(a.bank)[0] + "_net.pt"
  save_bank(out, dst)
  for k in ("per_delay", "per_category"):
    if k in meta["eta_net"]["kept"]:
      print(f"[eta-net] kept {k}: {meta['eta_net']['kept'][k]} "
            f"(of {meta['eta_net']['input'][k]})")
  print(f"[eta-net] wrote {dst}")


if __name__ == "__main__":
  main()
