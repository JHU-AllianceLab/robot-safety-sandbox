"""Cut a LAFAN clip library (build_g1_recover_mocap_states.py --library) down to a few clips.

    python examples/subset_clip_library.py --library data/g1_lafan1_clips.pt \
        --clips dance1_subject1 dance2_subject1 --out data/g1_lafan1_clips_dance.pt

Every per-row tensor (first dim = the library's row count) keeps only the rows of the named clips; the
meta (clip / category name lists, csv_dir, ...) is kept as is, so clip / category indices stay valid, and
gets ``subset_clips`` plus recounted ``rows_per_category`` / ``start_ok_rows``. Used for the SAILR
dance / fight-and-sport calibration datasets (the user, 2026-11-25): SONIC tracks exactly the task clips.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--library", default="data/g1_lafan1_clips.pt")
  p.add_argument("--clips", nargs="+", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  lib = torch.load(a.library, map_location="cpu", weights_only=False)
  meta = dict(lib["meta"])
  names = list(meta["clips"])
  missing = [c for c in a.clips if c not in names]
  if missing:
    raise SystemExit(f"{missing} not in {a.library}")
  n = int(lib["clip"].shape[0])
  keep = torch.isin(lib["clip"], torch.as_tensor([names.index(c) for c in a.clips]))
  out = {}
  for k, v in lib.items():
    if k == "meta":
      continue
    out[k] = v[keep] if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n else v
  cats = list(meta["categories"])
  meta["subset_clips"] = list(a.clips)
  meta["subset_of"] = a.library
  meta["rows_per_category"] = {c: int((out["category"] == i).sum()) for i, c in enumerate(cats)}
  if "start_ok" in out:
    meta["start_ok_rows"] = int(out["start_ok"].sum())
    meta["start_ok_per_category"] = {c: int((out["start_ok"] & (out["category"] == i)).sum())
                                     for i, c in enumerate(cats)}
  out["meta"] = meta
  torch.save(out, a.out)
  per = {c: int((out["clip"] == names.index(c)).sum()) for c in a.clips}
  print(f"[subset] {a.out}: {int(keep.sum())} of {n} rows; per clip {per}; "
        f"start_ok {meta.get('start_ok_rows')}")


if __name__ == "__main__":
  main()
