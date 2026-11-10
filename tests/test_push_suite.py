"""The push suite's pure half (robot_safety_sandbox/envs/g1_recover/push_suite.py) and the
simulator-free parts of examples/eval_push_suite.py.

Simulator-free (CPU):

  * UNITS -- impulse <-> base velocity change at the G1's 33.34 kg, and a
    push's force rounded to whole control steps delivers exactly its impulse;
  * CELLS -- every (impulse, duration, direction, delay) once, directions in
    the heading frame from behind, counterclockwise; round-robin assignment
    is balanced to within one episode on every prefix of the issue order;
  * F-LEVELS -- linear interpolation at the first crossing, the ``<min`` /
    ``>max`` sentinels (NaN + flag), unsorted and NaN input;
  * SUMMARY -- survival curves per setting x policy, per-direction F50s,
    handover replays that fell before the takeover left out of survival,
    behaviour statistics and the value AUC on synthetic records;
  * THE FROZEN GRID -- F90..F10 of the first policy's pooled E-lafan curve,
    one duration when the per-duration curves overlap, the 1.3 x F10 ceiling;
  * ARMS -- grouping by ``_s<seed>``, seed gaps and the decision rule;
  * THE SCRIPT -- grid precedence (defaults < frozen file < --quick <
    explicit flags), the report tables, and ``--merge`` of saved evals.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

from robot_safety_sandbox.envs.g1_recover import push_suite as ps  # noqa: E402


# --- units ---------------------------------------------------------------------

def test_impulse_and_velocity_change_at_the_g1_mass():
  assert ps.G1_MASS_KG == pytest.approx(33.34)
  # the plan's two anchors: 260 N x 0.2 s ~ 1.56 m/s, 100 N x 0.2 s ~ 0.6 m/s
  assert ps.impulse_to_dv(260.0 * 0.2) == pytest.approx(1.56, abs=0.005)
  assert ps.impulse_to_dv(100.0 * 0.2) == pytest.approx(0.6, abs=0.005)
  assert ps.dv_to_impulse(ps.impulse_to_dv(37.0)) == pytest.approx(37.0)
  t = torch.tensor([0.0, 33.34])
  assert torch.allclose(ps.impulse_to_dv(t), torch.tensor([0.0, 1.0]))


def test_push_force_delivers_the_impulse_in_whole_steps():
  dt = 0.02
  assert ps.push_steps(0.2, dt) == 10 and ps.push_steps(0.3, dt) == 15
  assert ps.push_steps(0.001, dt) == 1                  # at least one step
  for j, t in ((40.0, 0.2), (12.0, 0.1), (60.0, 0.3), (5.0, 0.031)):
    f = ps.push_force(j, t, dt)
    assert f * ps.push_steps(t, dt) * dt == pytest.approx(j)
  assert ps.push_force(40.0, 0.2) == pytest.approx(200.0)   # no dt: nominal T


def test_direction_labels_name_the_side_the_push_comes_from():
  # rel 0 = force along the heading = from behind; +90 deg = toward the
  # robot's left = from its right (random_push.SIDES order for 4)
  assert [ps.direction_label(i, 8) for i in (0, 2, 4, 6)] == [
    "behind", "right", "front", "left"]
  assert ps.direction_label(1, 8) == "rear-right"
  assert [ps.direction_label(i, 4) for i in range(4)] == [
    "behind", "right", "front", "left"]
  assert ps.direction_label(1, 6) == "60deg"
  assert ps.direction_label(0, 1) == "any"          # pre: the rows' own pushes


# --- cells ---------------------------------------------------------------------

def test_build_cells_is_the_full_product_in_the_heading_frame():
  cells = ps.build_cells([0, 8, 16], [0.1, 0.2], 8, delays=[0.0, 0.1])
  assert len(cells) == 3 * 2 * 8 * 2
  assert [c["cell"] for c in cells] == list(range(len(cells)))
  combos = {(c["impulse"], c["duration"], c["dir_idx"], c["delay"]) for c in cells}
  assert len(combos) == len(cells)
  for c in cells:
    assert c["rel"] == pytest.approx(2.0 * math.pi * c["dir_idx"] / 8)
  # impulse 0 is kept for every duration and direction (each curve's baseline)
  assert sum(c["impulse"] == 0.0 for c in cells) == 2 * 8 * 2
  tab = ps.cell_table(cells)
  assert tab["impulse"].shape == (len(cells),) and tab["dir_idx"].dtype == torch.long
  with pytest.raises(ValueError):
    ps.build_cells([-1.0], [0.2], 8)
  with pytest.raises(ValueError):
    ps.build_cells([1.0], [0.0], 8)
  with pytest.raises(ValueError):
    ps.build_cells([1.0], [0.2], 0)


def test_round_robin_assignment_is_balanced_on_every_prefix():
  C = 7
  seq = torch.cat([ps.assign_cells(C, 0, 5), ps.assign_cells(C, 5, 9),
                   ps.assign_cells(C, 14, 30)])
  assert torch.equal(seq, torch.arange(44) % C)
  for k in range(1, seq.numel() + 1):
    counts = torch.bincount(seq[:k], minlength=C)
    assert int(counts.max() - counts.min()) <= 1


# --- F-levels -------------------------------------------------------------------

def test_f_level_interpolates_the_first_crossing():
  lv = [0, 10, 20, 30, 40]
  s = [1.0, 0.9, 0.6, 0.2, 0.0]
  assert ps.f50(lv, s) == (pytest.approx(22.5), "ok")      # 0.6 -> 0.2 at 0.5
  assert ps.f_level(lv, s, 0.9) == (pytest.approx(10.0), "ok")  # exact hit
  assert ps.f_level(lv, s, 0.1) == (pytest.approx(35.0), "ok")
  # order and NaN do not matter
  v, flag = ps.f50([30, np.nan, 0, 20, 10, 40], [0.2, 0.5, 1.0, 0.6, 0.9, 0.0])
  assert flag == "ok" and v == pytest.approx(22.5)
  # a noisy upturn above the first crossing does not move it
  assert ps.f50(lv, [1.0, 0.4, 0.7, 0.2, 0.0])[0] == pytest.approx(
    10 * (1.0 - 0.5) / (1.0 - 0.4))


def test_f_level_sentinels():
  v, flag = ps.f50([0, 10, 20], [0.4, 0.2, 0.1])
  assert math.isnan(v) and flag == "<min"
  v, flag = ps.f50([0, 10, 20], [1.0, 0.9, 0.7])
  assert math.isnan(v) and flag == ">max"
  assert ps.f50([0, 10], [0.5, 0.2]) == (0.0, "ok")         # exactly q at the minimum
  v, flag = ps.f50([], [])
  assert math.isnan(v) and flag == "empty"


# --- records and the summary ---------------------------------------------------

def _records(policy, setting, impulses, surv_of, n_dirs=4, per=40,
             durations=(0.2,), prefix_rate=0.0, value=True, seed=0):
  """Synthetic records: for each (impulse, duration, direction) ``per``
  episodes, the first round(surv * per) of them survive."""
  g = torch.Generator().manual_seed(seed)
  cols = {k: [] for k in ("impulse", "duration", "dir_idx", "delay", "survived",
                          "prefix_fell", "peak_speed", "dist", "value0",
                          "t_reach_tight", "stay_frac_tight",
                          "fell_after_reach_tight", "t_reach_loose",
                          "stay_frac_loose", "fell_after_reach_loose")}
  cats = []
  for j in impulses:
    for t in durations:
      for d in range(n_dirs):
        s = surv_of(j, t, d)
        k = int(round(s * per))
        for i in range(per):
          ok = i < k
          pre = (not ok) and i < k + int(round(prefix_rate * per))
          cols["impulse"].append(j)
          cols["duration"].append(t)
          cols["dir_idx"].append(d)
          cols["delay"].append(0.0)
          cols["survived"].append(ok)
          cols["prefix_fell"].append(pre)
          cols["peak_speed"].append(0.5 + j / 40.0)
          cols["dist"].append(0.1 * j / 10.0)
          # V separates survivors perfectly
          cols["value0"].append((1.0 if ok else -1.0) + 0.01 * float(
            torch.rand(1, generator=g)) if value else float("nan"))
          # survivors reach the tight set at 1.0 s and stay 80 % of the time;
          # the loose set at 0.5 s; fallers never reach
          cols["t_reach_tight"].append(1.0 if ok else float("nan"))
          cols["stay_frac_tight"].append(0.8 if ok else float("nan"))
          cols["fell_after_reach_tight"].append(False)
          cols["t_reach_loose"].append(0.5 if ok else float("nan"))
          cols["stay_frac_loose"].append(1.0 if ok else float("nan"))
          cols["fell_after_reach_loose"].append(False)
          cats.append("walk" if d % 2 == 0 else "run")
  rec = {k: torch.tensor(v, dtype=torch.bool if k in ("survived", "prefix_fell")
                         or k.startswith("fell_after") else torch.float32)
         for k, v in cols.items()}
  n = rec["impulse"].numel()
  rec["n_dirs"] = torch.full((n,), n_dirs)
  rec["policy"] = [policy] * n
  rec["setting"] = [setting] * n
  rec["category"] = cats if setting != "std" else [""] * n
  rec["clip"] = [""] * n
  return rec


def _linear(j, t, d):
  return max(0.0, 1.0 - j / 40.0)            # F50 = 20, F90 = 4, F10 = 36


def test_summarize_curves_flevels_and_splits():
  rec = ps.concat_records([
    _records("a", "lafan", [0, 8, 16, 24, 32, 40], _linear),
    _records("b", "lafan", [0, 8, 16, 24, 32, 40],
             lambda j, t, d: max(0.0, 1.0 - j / (20.0 if d == 0 else 60.0))),
    _records("a", "std", [0, 8, 16, 24, 32, 40], lambda j, t, d: 1.0 if j < 30 else 0.0),
  ])
  s = ps.summarize(rec)
  assert s["meta"]["settings"] == ["lafan", "std"]
  assert s["meta"]["policies"] == ["a", "b"]
  assert set(s["settings"]["lafan"]) == {"a", "b"} and set(s["settings"]["std"]) == {"a"}
  a = s["settings"]["lafan"]["a"]
  assert a["curve"]["levels"] == [0, 8, 16, 24, 32, 40]
  assert a["curve"]["surv"] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
  assert a["curve"]["n"] == [160] * 6
  assert a["curve"]["fell"] == [0, 32, 64, 96, 128, 160]
  assert a["f"]["f50"][0] == pytest.approx(20.0) and a["f"]["f50"][1] == "ok"
  assert a["f"]["f90"][0] == pytest.approx(4.0)
  assert a["f"]["f10"][0] == pytest.approx(36.0)
  assert a["mean_surv_levels"] == pytest.approx(0.5)
  # per direction: b is weak from behind (dir 0) only
  b = s["settings"]["lafan"]["b"]
  assert set(b["by_dir"]) == {"behind", "right", "front", "left"}
  assert b["by_dir"]["behind"]["f50"][0] == pytest.approx(10.0)
  assert b["by_dir"]["front"]["f50"][0] == pytest.approx(30.0, abs=1.0)
  # one duration -> no by_duration split; two categories -> a by_category one
  assert "by_duration" not in a and set(a["by_category"]) == {"run", "walk"}
  assert "by_category" not in s["settings"]["std"]["a"]
  # behaviour: survivors reach (tight at 1.0 s, loose at 0.5 s), fallers never
  beh = a["behaviour"]
  assert beh["tight_reach_rate"] == pytest.approx(0.5)
  assert beh["tight_reach_rate_within"] == pytest.approx(0.5)
  assert beh["tight_t_reach_median"] == pytest.approx(1.0)
  assert beh["tight_stay_frac_mean"] == pytest.approx(0.8)
  assert beh["loose_t_reach_median"] == pytest.approx(0.5)
  assert beh["tight_fell_after_reach"] == 0.0
  bi = a["behaviour_by_impulse"]
  assert bi["levels"] == a["curve"]["levels"]
  assert bi["tight_reach_rate"] == pytest.approx(a["curve"]["surv"])
  # a 0.5 s budget excludes the 1.0 s tight reaches but keeps the loose ones
  s2 = ps.summarize(rec, reach_s=0.5)
  assert s2["settings"]["lafan"]["a"]["behaviour"]["tight_reach_rate_within"] == 0.0
  assert s2["settings"]["lafan"]["a"]["behaviour"]["loose_reach_rate_within"] == pytest.approx(0.5)
  # V separates survivors perfectly
  assert a["value_auc"] == pytest.approx(1.0)


def test_summarize_leaves_replay_falls_out_of_survival():
  rec = _records("a", "handover", [0, 20, 40], lambda j, t, d: 0.5,
                 prefix_rate=0.25, value=False)
  g = ps.summarize(rec)["settings"]["handover"]["a"]
  assert g["prefix_fall_rate"] == pytest.approx(0.25)
  assert g["n_valid"] == int(0.75 * g["n"])
  # of the valid episodes 0.5 / 0.75 survive
  assert g["survival"] == pytest.approx(0.5 / 0.75)
  assert "value_auc" not in g                       # no finite value0


def test_concat_records_checks_columns():
  a = _records("a", "std", [0], _linear)
  b = _records("b", "std", [0], _linear)
  ab = ps.concat_records([a, b])
  assert ab["impulse"].numel() == a["impulse"].numel() * 2
  assert ab["policy"][:1] == ["a"] and ab["policy"][-1:] == ["b"]
  with pytest.raises(ValueError):
    ps.concat_records([a, {k: v for k, v in b.items() if k != "dist"}])


# --- the frozen grid ------------------------------------------------------------

def test_propose_frozen_grid_spans_f90_to_f10_of_the_first_policy():
  lv = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
  rec = ps.concat_records([
    _records("first", "lafan", lv, _linear, durations=(0.1, 0.2, 0.3)),
    _records("second", "lafan", lv, lambda j, t, d: 1.0),
  ])
  prop = ps.propose_frozen_grid(ps.summarize(rec), n_levels=5)
  assert prop["source"]["policy"] == "first"
  assert prop["impulses"] == [4.0, 12.0, 20.0, 28.0, 36.0]   # linspace(F90, F10)
  # identical duration curves -> the training duration alone
  assert prop["durations_overlap"] is True and prop["durations"] == [0.2]
  assert prop["force_max_suggested"] == pytest.approx(1.3 * 36.0 / 0.2)
  assert prop["warnings"] == []
  # the other policy explicitly: survival never falls -> a degenerate grid + warnings
  p2 = ps.propose_frozen_grid(ps.summarize(rec), policy="second")
  assert len(p2["warnings"]) == 2


def test_propose_frozen_grid_keeps_all_durations_when_curves_differ():
  lv = [0, 10, 20, 30, 40, 50, 60]
  rec = _records("p", "lafan", lv, lambda j, t, d: max(0.0, 1.0 - j / (100.0 * t)),
                 durations=(0.1, 0.2, 0.3))
  prop = ps.propose_frozen_grid(ps.summarize(rec), n_levels=3, round_to=1.0)
  assert prop["durations_overlap"] is False
  assert prop["durations"] == [0.1, 0.2, 0.3]
  # the ceiling reads the 0.2 s curve: F10 at 0.2 s = 18 N s
  assert prop["force_max_suggested"] == pytest.approx(1.3 * 18.0 / 0.2)
  with pytest.raises(ValueError):
    ps.propose_frozen_grid(ps.summarize(rec), setting="handover")


def test_propose_frozen_grid_falls_back_to_the_grid_ends():
  rec = _records("p", "lafan", [0, 10, 20], lambda j, t, d: 0.8 - j / 100.0)
  prop = ps.propose_frozen_grid(ps.summarize(rec), n_levels=3)
  # F90 '<min' -> 0; F10 '>max' -> 20
  assert prop["impulses"] == [0.0, 10.0, 20.0]
  assert len(prop["warnings"]) == 2


# --- arms -----------------------------------------------------------------------

def test_compare_arms_groups_seeds_and_the_decision_rule():
  assert ps.arm_of("S3_s0") == ("S3", 0)
  assert ps.arm_of("P-pre-self_s12") == ("P-pre-self", 12)
  assert ps.arm_of("push_rear") == ("push_rear", None)
  lv = [0, 10, 20, 30, 40]
  rec = ps.concat_records([
    _records("A_s0", "handover", lv, lambda j, t, d: max(0.0, 1.0 - j / 40.0)),
    _records("A_s1", "handover", lv, lambda j, t, d: max(0.0, 1.0 - j / 44.0)),
    _records("B_s0", "handover", lv, lambda j, t, d: max(0.0, 1.0 - j / 80.0)),
    _records("B_s1", "handover", lv, lambda j, t, d: max(0.0, 1.0 - j / 84.0)),
    _records("C_s0", "handover", lv, lambda j, t, d: max(0.0, 1.0 - j / 42.0)),
  ])
  arms = ps.compare_arms(ps.summarize(rec), "handover")
  assert list(arms) == ["A", "B", "C"]
  assert arms["A"]["seeds"] == [0, 1] and arms["C"]["seeds"] == [0]
  a, b, c = (arms[k]["mean_surv_levels"] for k in "ABC")
  assert a["gap"] > 0 and c["gap"] == 0.0
  assert ps.significant(a, b)             # far apart vs the seed gaps
  assert not ps.significant(a, c)         # inside A's seed gap
  assert arms["B"]["f50_behind"]["mean"] > arms["A"]["f50_behind"]["mean"]


# --- the script's simulator-free parts --------------------------------------------

def _script():
  import eval_push_suite
  return eval_push_suite


def test_grid_precedence_defaults_frozen_quick_flags(tmp_path):
  m = _script()
  out = str(tmp_path / "out")
  g = m.resolve_grid(m._parse(["--out", out]))
  assert g["impulses"] == list(ps.CALIBRATION_IMPULSES)
  assert g["durations"] == [0.1, 0.2, 0.3] and g["directions"] == 8
  assert g["handover_delays"] == [0.0, 0.1, 0.2]
  assert g["episodes_per_cell"] == 64 and g["eval_s"] == 4.0 and g["num_envs"] == 4096
  grid = tmp_path / "grid.yaml"
  grid.write_text("impulses: [10, 20]\ndurations: [0.2]\neval_s: 6.0\n")
  g = m.resolve_grid(m._parse(["--out", out, "--frozen", "--grid-file", str(grid)]))
  assert g["impulses"] == [10.0, 20.0] and g["durations"] == [0.2] and g["eval_s"] == 6.0
  assert g["directions"] == 8                      # not in the file: the default
  g = m.resolve_grid(m._parse(["--out", out, "--frozen", "--grid-file", str(grid),
                               "--quick", "--impulses", "5"]))
  assert g["impulses"] == [5.0]                    # explicit flag wins
  assert g["num_envs"] == 256 and g["directions"] == 4   # --quick over the file
  with pytest.raises(SystemExit):
    m.resolve_grid(m._parse(["--out", out, "--frozen", "--grid-file",
                             str(tmp_path / "missing.yaml")]))


def test_report_tables_and_merge(tmp_path):
  m = _script()
  lv = [0, 10, 20, 30, 40]
  for name, scale in (("X_s0", 40.0), ("X_s1", 44.0)):
    rec = ps.concat_records([
      _records(name, "handover", lv, lambda j, t, d, s=scale: max(0.0, 1.0 - j / s)),
      _records(name, "std", lv, _linear)])
    d = tmp_path / name
    d.mkdir()
    torch.save({"records": rec, "meta": {}}, d / "episodes.pt")
  lines = m.report_lines(ps.summarize(rec))
  text = "\n".join(lines)
  assert "=== handover" in text and "F50 N s by direction" in text
  assert "replay fell before takeover" in text and "AUC(V at takeover" in text
  out = tmp_path / "report"
  m.main(["--merge", f"A_s0={tmp_path / 'X_s0'}", f"A_s1={tmp_path / 'X_s1'}",
          f"B_s0={tmp_path / 'never_ran'}", "--out", str(out)])
  import json
  summ = json.loads((out / "summary.json").read_text())
  assert summ["meta"]["policies"] == ["A_s0", "A_s1"]
  assert summ["meta"]["missing"] == [f"B_s0={tmp_path / 'never_ran'}"]
  rep = (out / "report.txt").read_text()
  assert "arms (mean over seeds" in rep and "left out" in rep
  # NaN became null: strict JSON
  assert "NaN" not in (out / "summary.json").read_text()
