"""Hold out LAFAN1 clips per category: the train / test split of the mocap banks.

    python examples/split_lafan_clips.py            # frac 0.15, seed 0
    python examples/split_lafan_clips.py --banks data/g1_handover_x.pt \
        --split-json data/lafan1_split.json         # reuse the split

The recovery policy trains on start states derived from mocap clips, and the
evaluation must ask whether it recovers from MOTIONS it has not seen, not from
neighbouring frames of the clips it trained on (frames 0.1 s apart are nearly
the same state). So the split is by CLIP, never by row: per category (the clip
name's prefix: walk / run / dance / ...), ``max(1, round(frac * n_clips))``
clips are held out (Python's round, half to even), chosen by a per-category
RNG seeded with ``"<seed>:<category>"`` over the sorted clip names -- the same
clips for the same seed whatever order the banks list them in, and adding a
category does not reshuffle the others. A category with a single clip loses
it to the test side (reported).

Clips are identified by NAME. Every bank indexes ``clip`` into its OWN
``meta["clips"]`` (the stride bank and the clip library are built separately
and need not agree on the order), so names are resolved per bank and the
split carries over to any bank built from the same clips, including ones
built later (``--split-json``: read an existing split instead of drawing one;
a clip in neither list is refused rather than silently put on one side).

Outputs: the split, ``data/lafan1_split.json`` (``{"test": [names], "train":
[names], "frac", "seed", "categories": {category: {"test", "train"}}}``),
and next to each input bank ``<stem>_train.pt`` / ``<stem>_test.pt``: the row
subsets, EVERY column kept (not only the schema's: the file is subset, not
re-read through ``load_bank``), rows in their original order so a library's
clips stay contiguous (its ``frame`` / ``next_ok`` stay valid, since a clip is
kept or dropped whole), ``meta["clips"]`` / ``meta["categories"]`` unchanged
so the row indices still resolve, and ``meta["split"]`` recording the side,
the held-out names and the split file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

from robot_safety_sandbox.envs.g1_recover.states import save_bank  # noqa: E402

DEFAULT_BANKS = (os.path.join(_ZOO, "data", "g1_recover_states_lafan1.pt"),
                 os.path.join(_ZOO, "data", "g1_lafan1_clips.pt"))


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--banks", nargs="+", default=None,
                 help="banks to split (default: the LAFAN1 stride bank and the "
                      "clip library, those that exist)")
  p.add_argument("--frac", type=float, default=0.15,
                 help="share of each category's clips held out")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--split-json", default=None,
                 help="reuse this existing split instead of drawing one")
  p.add_argument("--split-out", default=os.path.join(_ZOO, "data", "lafan1_split.json"),
                 help="where a newly drawn split is written")
  p.add_argument("--out-dir", default=None,
                 help="directory of the <stem>_{train,test}.pt files (default: "
                      "next to each bank)")
  return p.parse_args(argv)


def _prefix(name: str) -> str:
  """The category of a clip name, as the mocap builder derives it."""
  m = re.match(r"([A-Za-z]+)\d*_subject\d+", name)
  return m.group(1) if m else name


def clip_categories(bank: dict) -> dict[str, str]:
  """``{clip name: category}`` of every clip in ``bank["meta"]["clips"]``:
  the category its rows carry, or the name's prefix for a clip with no rows
  (every frame filtered out)."""
  meta = bank.get("meta", {}) or {}
  names = list(meta.get("clips", []))
  cats = list(meta.get("categories", []))
  if not names or "clip" not in bank:
    raise SystemExit("not a clip bank: needs a 'clip' column and meta['clips']")
  out = {nm: _prefix(nm) for nm in names}
  if "category" in bank and cats:
    clip = bank["clip"].reshape(-1).long()
    cat = bank["category"].reshape(-1).long()
    ok = (clip >= 0) & (cat >= 0)
    pairs = torch.unique(torch.stack([clip[ok], cat[ok]], dim=1), dim=0).tolist()
    for c, k in pairs:
      out[names[c]] = cats[k]
  return out


def make_split(clips: dict[str, str], frac: float = 0.15, seed: int = 0) -> dict:
  """Hold out ``max(1, round(frac * n))`` of each category's clips.

  ``clips`` maps clip name -> category. Deterministic in (the set of names,
  ``frac``, ``seed``): each category draws from its sorted names with its own
  ``random.Random(f"{seed}:{category}")``.
  """
  if not 0.0 <= frac <= 1.0:
    raise ValueError(f"frac must be in [0, 1], got {frac}")
  by_cat: dict[str, list[str]] = {}
  for nm, c in clips.items():
    by_cat.setdefault(c, []).append(nm)
  test, per_cat = [], {}
  for c in sorted(by_cat):
    names = sorted(by_cat[c])
    k = min(len(names), max(1, round(frac * len(names))))
    held = sorted(random.Random(f"{seed}:{c}").sample(names, k))
    test += held
    per_cat[c] = {"test": held, "train": [n for n in names if n not in held]}
  train = sorted(n for n in clips if n not in set(test))
  return {"test": sorted(test), "train": train, "frac": float(frac),
          "seed": int(seed), "categories": per_cat}


def split_bank(raw: dict, test_names: set, side: str, info: dict) -> dict:
  """The rows of ``raw`` on ``side`` ("train" / "test") of the split, every
  per-row tensor subset, everything else copied, ``meta["split"]`` added."""
  meta = dict(raw.get("meta", {}) or {})
  names = list(meta["clips"])
  clip = raw["clip"].reshape(-1).long()
  n = int(clip.numel())
  table = torch.tensor([nm in test_names for nm in names] + [False],
                       dtype=torch.bool)
  in_test = table[torch.where(clip >= 0, clip, len(names))]   # -1: not a clip row
  keep = in_test if side == "test" else ~in_test
  idx = keep.nonzero(as_tuple=False).squeeze(-1)
  out = {}
  for k, v in raw.items():
    if k == "meta":
      continue
    if torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == n:
      out[k] = v[idx]
    else:
      out[k] = v
  if "rows_per_category" in meta and "category" in out:
    cats = list(meta.get("categories", []))
    meta["rows_per_category"] = {c: int((out["category"] == i).sum())
                                 for i, c in enumerate(cats)}
  meta["split"] = {"part": side, "test": sorted(test_names), "frac": info.get("frac"),
                   "seed": info.get("seed"), "split_json": info.get("path"),
                   "rows": int(idx.numel()), "rows_before_split": n}
  out["meta"] = meta
  return out


def main(argv=None):
  args = _parse(argv)
  banks = args.banks or [p for p in DEFAULT_BANKS if os.path.exists(p)]
  if not banks:
    raise SystemExit("no bank to split (build data/g1_recover_states_lafan1.pt first)")
  raws = {p: torch.load(p, map_location="cpu", weights_only=False) for p in banks}
  clips: dict[str, str] = {}
  for p, raw in raws.items():
    for nm, c in clip_categories(raw).items():
      if clips.get(nm, c) != c:
        raise SystemExit(f"clip {nm!r} is {clips[nm]!r} in one bank and {c!r} in {p}")
      clips[nm] = c

  if args.split_json:
    with open(args.split_json) as f:
      split = json.load(f)
    path = os.path.abspath(args.split_json)
    unknown = sorted(set(clips) - set(split["test"]) - set(split["train"]))
    if unknown:
      raise SystemExit(f"{args.split_json} assigns no side to clips {unknown}")
    print(f"[split] reusing {args.split_json}: {len(split['test'])} held-out clips")
  else:
    split = make_split(clips, args.frac, args.seed)
    path = os.path.abspath(args.split_out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      json.dump(split, f, indent=2)
    for c, s in split["categories"].items():
      note = "  (no train clip left)" if not s["train"] else ""
      print(f"[split] {c:15s} {len(s['train']) + len(s['test']):3d} clips, "
            f"held out {s['test']}{note}")
    print(f"[split] wrote {path}: {len(split['test'])} test / "
          f"{len(split['train'])} train clips (frac {args.frac}, seed {args.seed})")

  test = set(split["test"])
  info = {"frac": split.get("frac"), "seed": split.get("seed"), "path": path}
  for p, raw in raws.items():
    stem = os.path.splitext(os.path.basename(p))[0]
    d = args.out_dir or os.path.dirname(os.path.abspath(p))
    for side in ("train", "test"):
      sub = split_bank(raw, test, side, info)
      out = os.path.join(d, f"{stem}_{side}.pt")
      save_bank(sub, out)
      print(f"[split] {os.path.basename(p)} -> {out}: {sub['meta']['split']['rows']} rows")
  return split


if __name__ == "__main__":
  main()
