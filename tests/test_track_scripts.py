"""CPU checks of the squat-tracking scripts (examples/track_*.py, eval_track.py,
robot_safety_sandbox/track_stats.py, configs/sweeps/track_squat/make_jobs.py).

Everything is loaded by file path: no sandbox package, no mjlab.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ZOO, "examples"))


def _load(rel: str, name: str):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ZOO, rel))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


tc = _load("examples/track_common.py", "track_common")
ts = _load("robot_safety_sandbox/track_stats.py", "_track_stats_test")
ev = _load("examples/eval_track.py", "_eval_track_test")
rp = _load("examples/track_report.py", "_track_report_test")
mj = _load("configs/sweeps/track_squat/make_jobs.py", "_make_jobs_test")
pr = _load("examples/track_promote.py", "_track_promote_test")


# --- arms and paths ------------------------------------------------------------------

def test_arms_cover_both_worlds_bare_and_two_dwells():
  assert set(tc.ARMS) == {"t70_bare", "t70_wm_d0", "t70_wm_d1", "cbf_bare", "cbf_wm_d0",
                          "cbf_wm_d1"}
  for arm, (world, kind, dwell) in tc.ARMS.items():
    assert tc.arm_name(world, kind, dwell) == arm
    assert tc.world_of_task(tc.task_of(world)) == world
  assert tc.PROPOSAL_BOUND == 6.5 and tc.PILOT_SEED == 2
  assert tc.run_dir("pilot", "t70_bare", 2) == "runs/track_squat/pilot/t70_bare/s2"
  assert tc.tube_tag("thermal_cbf", 1) == "thermalcbf_s1"
  assert tc.condition(0.9) == "wmmin:1:0.9"


def test_fallback_and_tube_files_exist():
  for w in tc.WORLDS:
    for s in tc.SEEDS:
      assert os.path.isfile(os.path.join(tc.fallback_run(w, s), "final_model.zip"))
      assert os.path.isfile(tc.tube_json(w, s))


# --- TrackStatsEnv ------------------------------------------------------------------------

def _bus(n, *, done, fail, fail_step, t, cause, phase_bin, start_rsi=None, tw=40.0):
  z = torch.zeros(n)
  return {
    "step": {"err_body": z + 0.05, "err_q": z + 0.1, "err_ori": z + 0.2, "lost": z.bool(),
             "tw_max": z + tw, "tau2": z + 10.0, "t": torch.as_tensor(t),
             "phase_bin": torch.as_tensor(phase_bin)},
    "done": torch.as_tensor(done),
    "ep": {"fail": torch.as_tensor(fail, dtype=torch.int8), "fail_step": torch.as_tensor(fail_step),
           "cause": torch.as_tensor(cause, dtype=torch.int8),
           "start_rsi": torch.as_tensor(start_rsi if start_rsi is not None else [False] * n),
           "completion": z + 0.5, "any_fail": torch.as_tensor(fail) != 0,
           "tw_peak": z + 50.0, "dtw_peak": z + 5.0, "heat_dose": z + 2.0,
           "tf_steps": z + 10.0, "steps": z + 20.0},
  }


class _Inner:
  num_envs, device = 2, "cpu"

  def metrics(self):
    return {"safety/failures_total": 1.0}


def test_track_stats_counts_first_failures_by_cause_and_phase():
  env = ts.TrackStatsEnv(_Inner(), check_codes=False)
  # step 1: env 0 overheats (non-terminal, as under the barrier) at t=3 in bin 2
  env.fold(_bus(2, done=[False, False], fail=[1, 0], fail_step=[3, -1], t=[3, 3], cause=[5, 5],
                phase_bin=[2, 2]))
  # step 2: env 0 now falls (contact) and ends; its FIRST failure stays the overheat.
  #         env 1 completes.
  env.fold(_bus(2, done=[True, True], fail=[1, 0], fail_step=[3, -1], t=[4, 30], cause=[2, 0],
                phase_bin=[3, 9], start_rsi=[False, True]))
  m = env.metrics()
  assert m["safety/failures_total"] == 1.0
  assert m["track/steps"] == 4 and m["track/episodes"] == 2
  assert m["track/fail_overheat_per_1k"] == pytest.approx(250.0)
  assert m["track/fail_contact_per_1k"] == 0.0
  assert m["track/fail_overheat_per_ep"] == pytest.approx(0.5)
  assert m["track/end_contact_per_ep"] == pytest.approx(0.5)
  assert m["track/end_complete_per_ep"] == pytest.approx(0.5)
  assert m["track/hazard_bin2"] == pytest.approx(1000.0 / 2)
  assert m["track/hazard_bin3"] == 0.0
  assert m["track/fail_early_per_1k"] == pytest.approx(1000.0 / 3)   # t <= 25: 3 of 4 steps
  assert m["track/start0_any_fail_per_ep"] == 1.0 and m["track/rsi_any_fail_per_ep"] == 0.0
  assert m["track/cum_fail_overheat"] == 1.0
  # a new window starts empty but keeps the cumulative count
  env.fold(_bus(2, done=[False, False], fail=[0, 0], fail_step=[-1, -1], t=[1, 1], cause=[5, 5],
                phase_bin=[0, 0]))
  m2 = env.drain()
  assert m2["track/cum_fail_overheat"] == 1.0 and m2["track/steps"] == 2


# --- eval statistics -----------------------------------------------------------------------

def test_wilson_and_condition_names():
  lo, hi = ev.wilson(50, 100)
  assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
  assert ev._parse_condition("bare") == ("bare", 0.0, None)
  assert ev._parse_condition("wm_d1") == ("wm", 1.0, 0.95)
  assert ev._parse_condition("wm_d0_c90") == ("wm", 0.0, 0.9)
  assert ev._parse_condition("wmcalib")[0] == "wmcalib"
  with pytest.raises(ValueError):
    ev._parse_condition("rollout:1")


def test_state_hash_ignores_last_bits():
  x = torch.randn(8, 3)
  assert ev.state_hash(x) == ev.state_hash(x + 1e-7)
  assert ev.state_hash(x) != ev.state_hash(x + 1e-3)


def test_summarize_fake_records():
  n, T = 4, 100
  rec = {k: torch.zeros(n) for k in ev.EP_KEYS}
  rec["cause"] = torch.tensor([0, 2, 1, 0], dtype=torch.int8)
  rec["fail"] = torch.tensor([0, 2, 1, 1], dtype=torch.int8)        # env 3: overheat, survived
  rec["any_fail"] = rec["fail"] != 0
  rec["overheat"] = rec["fail"] == 1
  rec["fail_bin"] = torch.tensor([-1, 5, 2, 7])
  rec["start_f"] = torch.zeros(n)
  rec["tf_steps"] = torch.tensor([100.0, 10.0, 20.0, 50.0])
  rec["steps"] = torch.tensor([100.0, 50.0, 30.0, 100.0])
  rec["heat_dose"] = torch.tensor([2.0, 0.0, 4.0, 2.0])
  rec["tw_start_max"] = torch.tensor([40.0, 61.0, 67.0, 69.9])
  rec["cbf_clip_mean"] = torch.full((n,), float("nan"))
  s = ev.summarize(rec, T, 0.02)
  assert s["survival"] == 0.5 and s["any_fail"] == 0.75 and s["overheat_incidence"] == 0.5
  assert s["fail_contact"] == 0.25 and s["end_overheat"] == 0.25
  assert s["tf"] == pytest.approx(np.mean([1.0, 0.1, 0.2, 0.5]))
  assert s["heat_dose_per_T"] == pytest.approx(np.mean([2.0, 0, 4.0, 2.0]) / (T * 0.02))
  assert s["S_phase"][0] == 1.0 and s["S_phase"][3] == 0.75 and s["S_phase"][9] == 0.25
  assert s["bands"]["66-70"]["n"] == 2 and s["cbf_clip_mean"] is None


def test_checkpoint_grid_picks_nearest(tmp_path):
  ck = tmp_path / "checkpoints"
  ck.mkdir()
  for i in range(1, 13):
    (ck / f"model_{i * 6103 * 4096}_steps.zip").write_bytes(b"")
  got = [n for n, _ in ev._checkpoints(str(tmp_path), 100_000_000)]
  assert got == [4 * 6103 * 4096, 8 * 6103 * 4096, 12 * 6103 * 4096]


# --- report statistics ---------------------------------------------------------------------

def test_mcnemar_and_bootstrap():
  assert rp.mcnemar_exact(0, 0) == 1.0
  assert rp.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
  assert rp.mcnemar_exact(3, 3) == 1.0
  lo, hi = rp.bootstrap_mean_ci([1.0, 2.0, 3.0])
  assert (lo, hi) == (1.0, 3.0)        # all 27 resamples; the 2.5/97.5% ends are the extremes
  m, sd, n = rp.mean_std([1.0, 2.0, 3.0])
  assert (m, n) == (2.0, 3) and sd == pytest.approx(1.0)


def test_report_contrasts_on_fake_evals(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  base = tmp_path / "runs" / "track_squat" / "eval" / "pilot"

  def put(arm, seed, world, conds):
    d = base / f"{arm}_s{seed}" / world
    d.mkdir(parents=True)
    summ = {"meta": {"eval_seed": 0, "suite": "full"},
            "conditions": {k: {**v, "hashes": {"start_temps": "x"}} for k, v in conds.items()}}
    (d / "summary.json").write_text(json.dumps(summ))
    recs = {k: {"any_fail": torch.tensor([v["any_fail"] > 0.5] * 4)} for k, v in conds.items()}
    torch.save({"records": recs}, d / "records.pt")

  for s in (0, 1, 2):
    for arm, af in (("t70_bare", 0.7), ("t70_wm_d1", 0.2)):
      conds = {c: {"any_fail": af - (0.1 if c != "bare" else 0.0), "tf": 0.5, "dtw_peak": 3.0}
               for c in ("bare", f"wm_d0@fb{s}", f"wm_d1@fb{s}")}
      put(arm, s, "thermal70", conds)
  evals = rp.load_evals("pilot")
  assert len(evals) == 6
  c = rp.contrasts(evals, [0, 1, 2])
  p1 = c["P1_t70_d1"]["summary"]["any_fail"]
  assert p1["mean"] == pytest.approx(-0.5) and p1["n"] == 3
  assert c["P2_t70_bare_d1"]["summary"]["any_fail"]["mean"] == pytest.approx(-0.1)
  mc = c["P1_t70_d1"]["per_seed"][0]["mcnemar_any_fail"]
  assert mc == {"a_only": 0, "b_only": 4, "p": pytest.approx(2 / 16)}
  arms = rp.arm_summaries(evals, [0, 1, 2])
  assert arms["t70_wm_d1/all"]["any_fail"]["mean"] == pytest.approx(0.1)
  assert rp.hash_check(evals)["thermal70"]["seed0_full"]["distinct"] == 1


# --- job files and promotion ------------------------------------------------------------------

def test_job_files_cover_every_arm_and_seed(tmp_path, monkeypatch):
  monkeypatch.setattr(mj, "HERE", str(tmp_path))
  mj.main(["--budget", "600000000"])
  read = lambda f: [l for l in (tmp_path / f).read_text().splitlines()  # noqa: E731
                    if l.strip() and not l.startswith("#")]
  pilot = read("pilot_jobs.txt")
  assert len(pilot) == 6 and all("--seed 2 --stage pilot" in l for l in pilot)
  final = read("jobs.txt")
  assert len(final) == 18
  assert sum("track_promote.py" in l and "# creates:" in l for l in final) == 6
  assert all("--steps 600000000" in l for l in final if "train.py" in l)
  evals = read("eval_jobs.txt")
  assert len(evals) == 6 * 3 * 2 + 2 + 1 + 1 and evals[-1].endswith("# always")
  mj.main(["--budget", "1500000000"])
  assert sum("track_promote.py" in l for l in read("jobs.txt")) == 0


def test_promote_picks_the_checkpoint_at_the_budget(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  src = tmp_path / tc.run_dir("pilot", "t70_bare", 2) / "checkpoints"
  src.mkdir(parents=True)
  for i in (1, 2, 3):
    n = i * 6103 * 4096
    for f in (f"model_{n}_steps.zip", f"tensornorm_{n}.pt", f"amp_disc_{n}.pt"):
      (src / f).write_text(str(n))
  pr.main(["--arm", "t70_bare", "--seed", "2", "--budget", "50000000"])
  dst = tmp_path / tc.run_dir("final", "t70_bare", 2)
  assert (dst / "final_model.zip").read_text() == str(2 * 6103 * 4096)
  assert json.loads((dst / "promoted.json").read_text())["checkpoint_steps"] == 2 * 6103 * 4096
  with pytest.raises(SystemExit):
    pr.main(["--arm", "t70_bare", "--seed", "2", "--budget", "200000000"])


def test_budget_rule():
  flat = [(i * 1e7, 1.0) for i in range(100)]
  dec = [(i * 1e7, 2.0 - i / 50.0 if i < 50 else 1.0) for i in range(100)]
  b = rp.budget_rule({"env/track/err_q": {"a": flat, "b": dec}})
  assert b["still_improving"] == [] and 4e8 <= b["B"] <= 1.5e9 and b["B"] % 100_000_000 == 0
  late = [(i * 1e7, 2.0 - i / 100.0) for i in range(100)]
  assert rp.budget_rule({"env/track/err_q": {"c": late}})["B"] == 1_500_000_000
