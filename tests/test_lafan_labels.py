"""Start-state label helpers (envs/g1_recover/labels.py) and the LAFAN clip
split (examples/split_lafan_clips.py).

Simulator-free (CPU):

  * AUC -- the Mann-Whitney statistic with ties counted half, against a
    brute-force pairwise count; NaN with an empty class; non-finite scores
    left out;
  * SPEARMAN -- average ranks on ties (scipy's convention, checked by hand),
    +-1 on monotone data, NaN on a constant side;
  * FIT_LOGISTIC -- recovers known (a, b) from Bernoulli samples and from
    exact fractional targets; stays finite on separable data;
  * FRONTIER_WEIGHT and ROW_NAMES (single-file and multi-part meta);
  * the SPLIT -- deterministic per seed, train/test disjoint and covering,
    max(1, round(frac * n)) held-out clips per category, and the per-bank row
    subsets keep every column, the clips' contiguity and the meta indices.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from robot_safety_sandbox.envs.g1_recover.labels import (  # noqa: E402
  auc, fit_logistic, frontier_weight, rankdata, row_names, spearman)


def _brute_auc(score, pos):
  p = [float(s) for s, q in zip(score, pos) if q]
  n = [float(s) for s, q in zip(score, pos) if not q]
  tot = 0.0
  for a in p:
    for b in n:
      tot += 1.0 if a > b else (0.5 if a == b else 0.0)
  return tot / (len(p) * len(n))


def test_rankdata_averages_ties():
  r = rankdata(torch.tensor([3.0, 1.0, 3.0, 2.0, 3.0]))
  assert r.tolist() == [4.0, 1.0, 4.0, 2.0, 4.0]


def test_auc_matches_brute_force_with_ties():
  g = torch.Generator().manual_seed(0)
  for trial in range(20):
    n = 5 + trial * 7
    score = torch.randint(0, 6, (n,), generator=g).float()   # many ties
    pos = torch.rand(n, generator=g) < 0.4
    if int(pos.sum()) in (0, n):
      continue
    assert auc(score, pos) == pytest.approx(_brute_auc(score.tolist(), pos.tolist()))
  assert auc(torch.tensor([1.0, 2.0]), torch.tensor([False, True])) == 1.0
  assert auc(torch.tensor([1.0, 1.0]), torch.tensor([False, True])) == 0.5
  assert auc(torch.tensor([1.0, 2.0]), torch.tensor([True, True])) != \
    auc(torch.tensor([1.0, 2.0]), torch.tensor([True, True]))       # NaN
  # a NaN score is dropped, not ranked
  assert auc(torch.tensor([float("nan"), 0.0, 1.0]),
             torch.tensor([False, False, True])) == 1.0


def test_spearman():
  x = torch.arange(10.0)
  assert spearman(x, x ** 3) == pytest.approx(1.0)
  assert spearman(x, -x) == pytest.approx(-1.0)
  # by hand: ranks a = [1, 2.5, 2.5, 4], b = [1, 2, 3, 4]
  a = torch.tensor([0.0, 1.0, 1.0, 2.0])
  b = torch.tensor([0.0, 1.0, 2.0, 3.0])
  ra = torch.tensor([1.0, 2.5, 2.5, 4.0]) - 2.5
  rb = torch.tensor([1.0, 2.0, 3.0, 4.0]) - 2.5
  want = float((ra * rb).sum() / torch.sqrt((ra * ra).sum() * (rb * rb).sum()))
  assert spearman(a, b) == pytest.approx(want)
  assert spearman(torch.ones(5), x[:5]) != spearman(torch.ones(5), x[:5])   # NaN
  assert spearman(torch.tensor([0.0, float("nan"), 2.0]),
                  torch.tensor([0.0, 5.0, 1.0])) == pytest.approx(1.0)


def test_fit_logistic_recovers_known_parameters():
  g = torch.Generator().manual_seed(1)
  a, b = 2.5, -0.7
  x = torch.randn(40000, generator=g) * 1.5 + 0.3
  y = (torch.rand(40000, generator=g) < torch.sigmoid(a * x + b)).float()
  fa, fb = fit_logistic(x, y)
  assert fa == pytest.approx(a, rel=0.05) and fb == pytest.approx(b, abs=0.05)
  # exact fractional targets (a row's p_surv) recover the parameters exactly
  xf = torch.linspace(-3.0, 3.0, 61)
  fa, fb = fit_logistic(xf * 10.0 + 4.0, torch.sigmoid(-1.2 * xf + 0.4))
  assert fa == pytest.approx(-0.12, abs=1e-4)          # x scaled by 10, shifted
  assert fb == pytest.approx(0.4 + 1.2 * 0.4, abs=1e-3)
  # separable labels stay finite and order the classes
  fa, fb = fit_logistic(torch.tensor([0.0, 1.0, 2.0, 3.0]),
                        torch.tensor([0.0, 0.0, 1.0, 1.0]))
  assert fa > 0 and abs(fa) < 1e6 and abs(fb) < 1e7
  # constant x: slope 0, logit of the mean
  fa, fb = fit_logistic(torch.ones(4), torch.tensor([1.0, 1.0, 1.0, 0.0]))
  assert fa == 0.0 and fb == pytest.approx(float(torch.logit(torch.tensor(0.75))))
  # one outcome only (every row survived): no MLE, slope 0, clamped logit
  fa, fb = fit_logistic(torch.arange(5.0), torch.ones(5))
  assert fa == 0.0 and fb == pytest.approx(float(torch.logit(torch.tensor(1.0 - 1e-6,
                                                                       dtype=torch.float64))))


def test_frontier_weight():
  p = torch.tensor([0.0, 0.25, 0.5, 1.0])
  assert torch.allclose(frontier_weight(p), torch.tensor([0.0, 0.1875, 0.25, 0.0]))


def test_row_names_single_and_multi_part():
  bank = {"category": torch.tensor([1, 0, -1]), "kind": torch.tensor([3, 3, 0]),
          "meta": {"categories": ["walk", "run"]}}
  assert row_names(bank) == ["run", "walk", "-"]
  assert row_names(bank, fallback=("clean", "pushed")) == ["run", "walk", "clean"]
  multi = {"category": torch.tensor([0, 0, 1]), "part": torch.tensor([0, 1, 1]),
           "kind": torch.tensor([3, 3, 3]),
           "meta": {"parts": [{"categories": ["dance"]},
                              {"categories": ["jumps", "sprint"]}],
                    "paths": ["a.pt", "b.pt"]}}
  assert row_names(multi) == ["dance", "jumps", "sprint"]


# --- the clip split ----------------------------------------------------------

def _split_module():
  path = os.path.join(_ROOT, "examples", "split_lafan_clips.py")
  spec = importlib.util.spec_from_file_location("split_lafan_clips", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


CATS = {"walk": 12, "run": 4, "dance": 8, "fight": 3, "sprint": 2}


def _clip_bank(frames_per_clip=3, shuffle_names=False):
  """A library-like bank: every clip's frames contiguous, its own meta order."""
  names, cats = [], list(CATS)
  for c, n in CATS.items():
    names += [f"{c}{i + 1}_subject{i % 3 + 1}" for i in range(n)]
  if shuffle_names:
    names = names[::-1]
  clip, cat, frame = [], [], []
  for i, nm in enumerate(names):
    c = cats.index(next(k for k in CATS if nm.startswith(k)))
    clip += [i] * frames_per_clip
    cat += [c] * frames_per_clip
    frame += list(range(frames_per_clip))
  n = len(clip)
  return {"root_z": torch.rand(n), "joint_pos": torch.randn(n, 29),
          "kind": torch.full((n,), 3), "clip": torch.tensor(clip),
          "category": torch.tensor(cat), "frame": torch.tensor(frame),
          "extra_col": torch.arange(n),
          "meta": {"clips": names, "categories": cats, "fps": 50}}


def test_split_is_deterministic_disjoint_and_counts_per_category():
  mod = _split_module()
  bank = _clip_bank()
  clips = mod.clip_categories(bank)
  s1 = mod.make_split(clips, frac=0.15, seed=0)
  s2 = mod.make_split(dict(reversed(list(clips.items()))), frac=0.15, seed=0)
  assert s1["test"] == s2["test"] and s1["train"] == s2["train"]
  assert not set(s1["test"]) & set(s1["train"])
  assert set(s1["test"]) | set(s1["train"]) == set(clips)
  for c, n in CATS.items():
    held = [nm for nm in s1["test"] if clips[nm] == c]
    assert len(held) == max(1, round(0.15 * n)), c
  other = mod.make_split(clips, frac=0.15, seed=1)
  assert other["test"] != s1["test"]                    # the seed matters
  big = mod.make_split(clips, frac=0.5, seed=0)
  assert sum(1 for nm in big["test"] if clips[nm] == "walk") == 6


def test_split_bank_keeps_columns_contiguity_and_meta(tmp_path):
  mod = _split_module()
  lib = _clip_bank(frames_per_clip=4)
  stride = _clip_bank(frames_per_clip=2, shuffle_names=True)  # its own clip order
  lib_path, stride_path = str(tmp_path / "lib.pt"), str(tmp_path / "stride.pt")
  torch.save(lib, lib_path)
  torch.save(stride, stride_path)
  split_json = str(tmp_path / "split.json")
  mod.main(["--banks", stride_path, lib_path, "--split-out", split_json,
            "--frac", "0.15", "--seed", "0"])
  split = json.load(open(split_json))
  assert split["frac"] == 0.15 and split["seed"] == 0
  test = set(split["test"])
  for path, src in ((lib_path, lib), (stride_path, stride)):
    stem = os.path.splitext(path)[0]
    tr = torch.load(stem + "_train.pt", weights_only=False)
    te = torch.load(stem + "_test.pt", weights_only=False)
    n = src["clip"].numel()
    assert tr["clip"].numel() + te["clip"].numel() == n
    for part, want_test in ((tr, False), (te, True)):
      names = [part["meta"]["clips"][int(c)] for c in part["clip"]]
      assert all((nm in test) == want_test for nm in names)
      assert part["meta"]["clips"] == src["meta"]["clips"]   # indices stay valid
      assert part["meta"]["split"]["part"] == ("test" if want_test else "train")
      assert set(part) >= set(src)                          # every column kept
      assert part["joint_pos"].shape[1] == 29
      # each clip's frames stay contiguous and in order
      c = part["clip"]
      change = (c[1:] != c[:-1]).sum().item()
      assert change == len(set(c.tolist())) - 1
      assert torch.equal(part["extra_col"], torch.sort(part["extra_col"]).values)
  # reusing the split for a new bank reproduces the same assignment
  again = str(tmp_path / "again.pt")
  torch.save(_clip_bank(frames_per_clip=1), again)
  mod.main(["--banks", again, "--split-json", split_json])
  te = torch.load(str(tmp_path / "again_test.pt"), weights_only=False)
  assert {te["meta"]["clips"][int(c)] for c in te["clip"]} == test
