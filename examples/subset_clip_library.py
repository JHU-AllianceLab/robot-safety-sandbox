"""A clip library restricted to some clips, optionally cut at a time.

    python examples/subset_clip_library.py --library data/g1_lafan1_clips.pt \
        --clips walk1_subject1 run1_subject2 dance1_subject1 \
        --end-s dance1_subject1=105 --out data/g1_lafan1_clips_bench_train.pt

The SONIC push benchmark draws its start states from three motions only, and
dance1_subject1 only up to 105 s (its reference leaves K after 106.6 s). The
spawn draws from a library's ``start_ok`` mocap rows and a harvest episode
needs ``next_ok`` frames after its row, so this keeps the rows of ``--clips``
and, for a cut clip, drops the rows past the cut and caps ``next_ok`` so a
window plus SONIC's reference preview (``--preview`` frames) ends by the cut;
rows left with no room lose ``start_ok``. ``meta`` keeps the clip and category
names (row indices into them stay valid) and records the subset.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips.pt"))
  p.add_argument("--clips", nargs="+", required=True)
  p.add_argument("--end-s", nargs="*", default=[], metavar="NAME=SECONDS")
  p.add_argument("--preview", type=int, default=45, help="SONIC's reference preview, frames")
  p.add_argument("--fps", type=float, default=50.0)
  p.add_argument("--out", required=True)
  return p.parse_args(argv)


def subset(lib: dict, clips, end_s: dict, preview: int, fps: float) -> dict:
  names = list(lib["meta"]["clips"])
  unknown = [c for c in clips if c not in names]
  if unknown:
    raise SystemExit(f"clips {unknown} are not in the library")
  ids = torch.as_tensor([names.index(c) for c in clips])
  keep = torch.isin(lib["clip"], ids)
  next_ok, start_ok = lib["next_ok"].clone(), lib["start_ok"].clone()
  for name, sec in end_s.items():
    if name not in clips:
      raise SystemExit(f"--end-s {name}: not among --clips")
    c = names.index(name)
    end = int(round(float(sec) * fps))
    rows = lib["clip"] == c
    keep &= ~(rows & (lib["frame"] > end))
    room = (end - preview - lib["frame"]).clamp_min(0)
    next_ok = torch.where(rows, torch.minimum(next_ok, room), next_ok)
    start_ok = torch.where(rows & (next_ok <= 0), torch.zeros_like(start_ok), start_ok)
  idx = keep.nonzero(as_tuple=False).squeeze(-1)
  out = {k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v)
         for k, v in lib.items() if k != "meta"}
  out["next_ok"], out["start_ok"] = next_ok[idx], start_ok[idx]
  out["meta"] = dict(lib["meta"], subset=dict(
    clips=list(clips), end_s={k: float(v) for k, v in end_s.items()}, preview_frames=preview,
    rows=int(idx.numel()), start_ok_rows=int(out["start_ok"].sum()),
    created=time.strftime("%Y-%m-%d %H:%M:%S")))
  return out


def main(argv=None):
  a = _parse(argv)
  sys.path.insert(0, _ZOO)
  lib = torch.load(a.library, map_location="cpu", weights_only=False)
  end_s = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.end_s}
  out = subset(lib, a.clips, end_s, a.preview, a.fps)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
  torch.save(out, a.out)
  per = {c: int((out["clip"] == list(lib["meta"]["clips"]).index(c)).sum()) for c in a.clips}
  print(f"[subset] {a.out}: {out['meta']['subset']['rows']} rows {per}, "
        f"{out['meta']['subset']['start_ok_rows']} start_ok")


if __name__ == "__main__":
  main()
