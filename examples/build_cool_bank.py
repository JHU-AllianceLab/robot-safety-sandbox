"""Derive the g1_cool spawn banks (only ``start_ok`` changes; every other column
is the source's):

* HANDOVER: the sysID SONIC handover bank with only its MINOR-kick rows
  spawnable (``start_ok`` = kick |dv| <= --max-dv m/s, the kick's impulse over
  the robot's mass).
* MOCAP (``--mocap``): the LAFAN TRAIN clip library without the categories the
  push eval also leaves out (--exclude-categories, default fallAndGetUp sprint:
  on the floor / too fast to stand from), for g1_cool_thermal70_mix's 30 % of
  mocap spawns (the eval spawns on the held-out test clips).

    python examples/build_cool_bank.py                 # -> data/g1_cool_handover_bank_dv1p5.pt
    python examples/build_cool_bank.py --max-dv 1.0    # -> data/g1_cool_handover_bank_dv1p0.pt
    python examples/build_cool_bank.py --mocap         # -> data/g1_cool_mocap_bank.pt

CPU, ~100 MB in and out. Never overwrites an existing output (--force to).
"""

from __future__ import annotations

import argparse
import os

import torch

SRC = "data/g1_sonic_push_gfall_sysid316_train_bank_net.pt"
MOCAP_SRC = "data/g1_lafan1_clips_train.pt"
#: the G1's mass in the push accounting (random_push.py: 200 N x 0.2 s ~ 1.2 m/s)
MASS_KG = 33.0


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--src", default=None, help=f"default {SRC} ({MOCAP_SRC} with --mocap)")
  p.add_argument("--max-dv", type=float, default=1.5, help="m/s")
  p.add_argument("--mocap", action="store_true", help="build the mocap bank instead")
  p.add_argument("--exclude-categories", nargs="*", default=["fallAndGetUp", "sprint"])
  p.add_argument("--out", default=None, help="default data/g1_cool_handover_bank_dv<X>.pt "
                                             "(data/g1_cool_mocap_bank.pt with --mocap)")
  p.add_argument("--force", action="store_true")
  a = p.parse_args()
  if a.mocap:
    out = a.out or "data/g1_cool_mocap_bank.pt"
  else:
    out = a.out or f"data/g1_cool_handover_bank_dv{str(a.max_dv).replace('.', 'p')}.pt"
  if os.path.exists(out) and not a.force:
    raise SystemExit(f"{out} exists (--force to overwrite)")
  if a.mocap:
    _mocap(a.src or MOCAP_SRC, out, list(a.exclude_categories))
    return
  a.src = a.src or SRC
  d = torch.load(a.src, map_location="cpu", weights_only=False)
  dv = d["push_impulse"].nan_to_num(0.0) / MASS_KG
  ok = dv <= float(a.max_dv)
  if "start_ok" in d:
    ok &= d["start_ok"].bool()
  d["start_ok"] = ok
  meta = dict(d.get("meta") or {})
  meta["g1_cool"] = {"src": os.path.abspath(a.src), "max_dv_m_s": float(a.max_dv),
                     "mass_kg": MASS_KG, "rows_ok": int(ok.sum()), "rows": int(ok.numel())}
  d["meta"] = meta
  torch.save(d, out)
  print(f"[cool-bank] {out}: {int(ok.sum())}/{ok.numel()} rows spawnable "
        f"(kick dv <= {a.max_dv} m/s; SONIC fell after {float(d['fell'][ok].float().mean()):.2f} "
        f"of them, base speed median {float(d['speed'][ok].float().median()):.2f} m/s)")


def _mocap(src: str, out: str, exclude: list) -> None:
  d = torch.load(src, map_location="cpu", weights_only=False)
  cats = list(d["meta"]["categories"])
  unknown = [c for c in exclude if c not in cats]
  if unknown:
    raise SystemExit(f"--exclude-categories {unknown}: the library's categories are {cats}")
  ex = torch.as_tensor([cats.index(c) for c in exclude], dtype=d["category"].dtype)
  ok = d["start_ok"].bool() & ~torch.isin(d["category"], ex)
  d["start_ok"] = ok
  meta = dict(d["meta"])
  meta["g1_cool"] = {"src": os.path.abspath(src), "exclude_categories": exclude,
                     "rows_ok": int(ok.sum()), "rows": int(ok.numel())}
  d["meta"] = meta
  torch.save(d, out)
  per = {c: int((ok & (d["category"] == i)).sum()) for i, c in enumerate(cats)}
  print(f"[cool-bank] {out}: {int(ok.sum())}/{ok.numel()} mocap rows spawnable "
        f"(without {exclude}; per category {per}; speed median "
        f"{float(d['speed'][ok].float().median()):.2f} m/s)")


if __name__ == "__main__":
  main()
