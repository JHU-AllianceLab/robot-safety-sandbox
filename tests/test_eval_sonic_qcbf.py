"""The Q-CBF evaluation's pure parts (examples/qcbf_eval_lib.py) and the
cross-seed summarizer (examples/summarize_sonic_qcbf.py), on fabricated
per-episode records. The driver itself (examples/eval_sonic_qcbf.py) needs
the simulator, SONIC and a GPU; what decides its numbers is here:

  * STATISTICS -- Wilson intervals against known values, the exact McNemar
    test against an exact rational computation, rescued / broken pairing,
    |dv| bins (the last one closed), categories;
  * CONDITIONS -- the grammar, canonical names, every refusal;
  * EPISODE LISTS -- seeded, uniform over the ELIGIBLE spawn rows, with a
    rejection rule that never shifts the onset / push draws;
  * TELEMETRY -- the per-env book split pre / post at each episode's split
    step, the outcome rule, the pooled rates;
  * CONTROLLER -- what each condition applies and records (fake
    interventions and monitors);
  * SUMMARIZER -- two fake run directories (SHAC, PPO) written in the
    driver's JSON shape, grouped, pooled and tabulated, and the two groups
    paired against each other on shared episodes.

CPU only, torch and the standard library (no mjlab import): seconds after
torch's import. Run with OMP_NUM_THREADS=1 on a login node.
"""

from __future__ import annotations

import json
import math
import os
import sys
from fractions import Fraction
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(os.path.dirname(_HERE), "examples")
sys.path.insert(0, _EX)

import qcbf_eval_lib as ql  # noqa: E402
import summarize_sonic_qcbf as sq  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
  old = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(old)


def B(*xs):
  return torch.tensor([bool(x) for x in xs])


# --- statistics ------------------------------------------------------------------------

def test_wilson_known_values():
  lo, hi = ql.wilson_ci(8, 10)
  assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
  lo, hi = ql.wilson_ci(0, 10)
  assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
  lo, hi = ql.wilson_ci(10, 10)
  assert lo == pytest.approx(0.7225, abs=1e-4) and hi == pytest.approx(1.0, abs=1e-12)
  for k, n in ((3, 17), (500, 1234)):
    a, b = ql.wilson_ci(k, n)
    c, d = ql.wilson_ci(n - k, n)
    assert a == pytest.approx(1.0 - d) and b == pytest.approx(1.0 - c)
    assert a < k / n < b
  assert all(math.isnan(v) for v in ql.wilson_ci(0, 0))


def _mcnemar_rational(b, c):
  n, m = b + c, min(b, c)
  tail = Fraction(sum(math.comb(n, i) for i in range(m + 1)), 2 ** n)
  return min(1.0, float(2 * tail))


def test_mcnemar_exact():
  assert ql.mcnemar_exact_p(0, 0) == 1.0
  assert ql.mcnemar_exact_p(1, 9) == pytest.approx(22 / 1024, rel=1e-12)
  assert ql.mcnemar_exact_p(9, 1) == ql.mcnemar_exact_p(1, 9)
  assert ql.mcnemar_exact_p(5, 5) == 1.0
  assert ql.mcnemar_exact_p(0, 7) == pytest.approx(2 / 128)
  for b, c in ((3, 11), (40, 61), (500, 600), (2000, 2210)):
    assert ql.mcnemar_exact_p(b, c) == pytest.approx(_mcnemar_rational(b, c), rel=1e-9)
  assert ql.binom_half_cdf(-1, 5) == 0.0 and ql.binom_half_cdf(5, 5) == 1.0
  assert ql.binom_half_cdf(2, 4) == pytest.approx(11 / 16)


def test_paired_rescued_broken():
  ref = B(1, 1, 0, 0, 0, 1)
  surv = B(1, 0, 1, 0, 1, 1)
  p = ql.paired(ref, surv)
  assert (p["rescued"], p["broken"], p["both_survived"], p["both_fell"]) == (2, 1, 2, 1)
  assert p["n"] == 6 and p["delta_sr"] == pytest.approx(1 / 6)
  assert p["mcnemar_p"] == pytest.approx(ql.mcnemar_exact_p(2, 1))
  m = B(1, 1, 1, 0, 0, 0)
  q = ql.paired(ref, surv, m)
  assert (q["n"], q["rescued"], q["broken"]) == (3, 1, 1)
  with pytest.raises(ValueError):
    ql.paired(ref, surv[:5])


def test_survival_and_bins():
  s = ql.survival(B(1, 1, 0, 1))
  assert (s["n"], s["k"], s["sr"]) == (4, 3, 0.75)
  assert s["ci"] == list(ql.wilson_ci(3, 4))
  assert math.isnan(ql.survival(B(1, 0), B(0, 0))["sr"])
  dv = torch.tensor([0.0, 1.49, 1.5, 2.5, 3.49, 5.0, 5.01, -0.1, float("nan")])
  assert ql.bin_index(dv, ql.DV_BINS).tolist() == [0, 0, 1, 2, 2, 3, -1, -1, -1]
  assert ql.bin_labels(ql.DV_BINS) == ["0-1.5", "1.5-2.5", "2.5-3.5", "3.5-5"]
  surv = B(1, 0, 1, 1, 0, 0, 1, 1, 1)
  bins = ql.by_bins(dv, surv, ql.DV_BINS, ref=B(1, 1, 1, 1, 1, 1, 1, 1, 1))
  assert [bins[k]["n"] for k in bins] == [2, 1, 2, 1]
  assert [bins[k]["k"] for k in bins] == [1, 1, 1, 0]
  assert bins["0-1.5"]["paired"]["broken"] == 1
  cats = ql.by_category(torch.tensor([0, 0, 2, -1]), B(1, 0, 1, 1), ["walk", "run", "dance"])
  assert set(cats) == {"walk", "dance", "-"}
  assert cats["walk"]["sr"] == 0.5 and cats["-"]["n"] == 1


# --- conditions ----------------------------------------------------------------------------

def test_parse_conditions():
  c = ql.parse_condition("qcbf:exact:0.70")
  assert (c.name, c.kind, c.variant, c.kappa, c.solver) == (
    "qcbf:exact:0.7", "qcbf", "exact", 0.7, "gd+segment")
  assert ql.parse_condition("qcbf:td_g099:0:segment").name == "qcbf:td_g099:0.0:segment"
  assert ql.parse_condition("qcbf:exact:0.7:gd+segment").name == "qcbf:exact:0.7"
  assert ql.parse_condition("lrsf:td_g099").variant == "td_g099"
  s, f = ql.parse_condition("sonic"), ql.parse_condition("fallback")
  assert not s.needs_values and not s.needs_fallback
  assert not f.needs_values and f.needs_fallback
  assert ql.parse_condition("lrsf:exact").needs_values
  r = ql.parse_condition("rollout:2.0")
  assert (r.name, r.kind, r.horizon_s) == ("rollout:2", "rollout", 2.0)
  assert r.needs_fallback and not r.needs_values
  assert ql.parse_condition("rollout:0.5").name == "rollout:0.5"
  r16 = ql.parse_condition("rollout:1:16")
  assert (r16.name, r16.horizon_s, r16.n_rollouts) == ("rollout:1:16", 1.0, 16)
  assert r.n_rollouts is None and ql.parse_condition("wmrollout:3:8").n_rollouts == 8
  tb = ql.parse_condition("wmtube:1.0:16:0.950")
  assert (tb.name, tb.kind, tb.horizon_s, tb.n_rollouts, tb.coverage) == (
    "wmtube:1:16:0.95", "wmtube", 1.0, 16, 0.95)
  wc = ql.parse_condition("wmcalib:clean")
  assert (wc.name, wc.kind, wc.n_rollouts) == ("wmcalib:clean", "wmcalib", None)
  mn = ql.parse_condition("wmmin:0.10:0.95")
  assert (mn.name, mn.kind, mn.horizon_s, mn.coverage, mn.n_rollouts) == (
    "wmmin:0.1:0.95", "wmmin", 0.1, 0.95, None)
  cb = ql.parse_condition("wmcalib:16")
  assert (cb.name, cb.kind, cb.n_rollouts) == ("wmcalib:16", "wmcalib", 16) and cb.needs_fallback
  w = ql.parse_condition("wmrollout:0.20")
  assert (w.name, w.kind, w.horizon_s) == ("wmrollout:0.2", "wmrollout", 0.2)
  assert w.needs_fallback and not w.needs_values
  lt = ql.parse_condition("wmmin:0.40:0.95+latch")
  assert (lt.name, lt.kind, lt.horizon_s, lt.coverage, lt.latch) == (
    "wmmin:0.4:0.95+latch", "wmmin", 0.4, 0.95, True)
  assert ql.parse_condition("rollout:1+latch").latch and not mn.latch
  assert ql.parse_condition("wmrollout:1:8+latch").name == "wmrollout:1:8+latch"
  dw = ql.parse_condition("wmmin:0.4:0.95+dwell0.50")
  assert (dw.name, dw.dwell_s, dw.latch, dw.coverage) == ("wmmin:0.4:0.95+dwell0.5", 0.5, False, 0.95)
  assert ql.parse_condition("rollout:1+dwell1").dwell_s == 1.0 and lt.dwell_s is None
  for bad in ("qcbf:exact:1.5", "qcbf:exact:-0.1", "qcbf:exact:nan", "qcbf:exact:x",
              "qcbf:exact", "qcbf::0.7", "qcbf:exact:0.7:adam", "lrsf", "lrsf:",
              "rollout", "rollout:", "rollout:0", "rollout:-1", "rollout:inf",
              "wmrollout", "wmrollout:0", "rollout:1:0", "rollout:1:x", "rollout:1:2.5",
              "rollout:1:8:2", "wmtube:1:16", "wmtube:1:16:1.5", "wmtube:0:16:0.9",
              "wmtube:1:0:0.9", "wmtube:1:16:0.9:2", "wmcalib", "wmcalib:0", "wmcalib:x",
              "wmmin", "wmmin:1", "wmmin:1:1.5", "wmmin:0:0.9", "wmmin:1:0.9:2",
              "sonic:exact", "cbf:exact:0.7", "", "sonic+latch", "fallback+latch",
              "qcbf:exact:0.7+latch", "wmcalib:clean+latch", "wmmin:1:0.9+latch+latch",
              "+latch", "wmmin:1:0.9+dwell0", "wmmin:1:0.9+dwell", "wmmin:1:0.9+dwellx",
              "sonic+dwell1", "wmmin:1:0.9+latch+dwell1", "wmmin:1:0.9+dwell1+latch",
              "wmmin:1:0.9+dwell-1"):
    with pytest.raises(ValueError):
      ql.parse_condition(bad)
  with pytest.raises(ValueError, match="twice"):
    ql.parse_conditions(["qcbf:exact:0.70", "qcbf:exact:0.7:gd+segment"])
  for lst in (ql.PUSH_CONDITIONS, ql.BADREF_CONDITIONS, ql.QUICK_PUSH_CONDITIONS,
              ql.QUICK_BADREF_CONDITIONS):
    names = [c.name for c in ql.parse_conditions(lst)]
    assert names == list(lst) and names[0] == ql.REFERENCE
  # the driver runs SONIC first whatever the order given (the pairs need it)
  conds = ql.reference_first(ql.parse_conditions(["qcbf:exact:0.7", "lrsf:exact", "sonic"]))
  assert [c.name for c in conds] == ["sonic", "qcbf:exact:0.7", "lrsf:exact"]
  assert [c.name for c in ql.reference_first(ql.parse_conditions(["lrsf:exact"]))] == ["lrsf:exact"]


# --- episode lists ---------------------------------------------------------------------------

def test_draw_rows_uniform_over_eligible():
  pool = torch.arange(100, 110)
  room = torch.tensor([5, 50, 10, 80, 20, 0, 60, 30, 90, 40])
  need = torch.tensor([45, 45, 45, 85, 0])
  # eligible sorted by room: 45 -> rooms 50, 60, 80, 90 (rows 101, 106, 103, 108)
  u = torch.tensor([0.0, 0.26, 0.999, 0.5, 0.0])
  rows = ql.draw_rows(pool, room, need, u)
  assert rows.tolist() == [101, 106, 108, 108, 105]
  gen = torch.Generator().manual_seed(0)
  big = ql.draw_rows(pool, room, torch.full((20000,), 45), torch.rand(20000, generator=gen))
  counts = torch.bincount(big - 100, minlength=10)
  assert set(torch.nonzero(counts).squeeze(-1).tolist()) == {1, 3, 6, 8}
  assert float(counts[[1, 3, 6, 8]].float().std() / 5000) < 0.05
  with pytest.raises(ValueError, match="room"):
    ql.draw_rows(pool, room, torch.tensor([91]), torch.tensor([0.5]))


def test_draw_episodes_seeded_and_eligible():
  pool = torch.arange(1000)
  room = (torch.arange(1000) * 7) % 400
  kw = dict(dt=0.02, onset_s=(0.5, 1.5), window=200, look_ahead=46, dv_range=(0.0, 5.0))
  a = ql.draw_episodes(pool, room, 512, torch.Generator().manual_seed(3), **kw)
  b = ql.draw_episodes(pool, room, 512, torch.Generator().manual_seed(3), **kw)
  c = ql.draw_episodes(pool, room, 512, torch.Generator().manual_seed(4), **kw)
  for key in ("row", "on", "dv", "rel", "dv_h"):
    assert torch.equal(a[key], b[key])
  assert not torch.equal(a["row"], c["row"])
  assert int(a["on"].min()) >= 25 and int(a["on"].max()) <= 75
  assert bool((room[a["row"]] >= a["on"] + 200 + 46).all())
  assert float(a["dv"].min()) >= 0.0 and float(a["dv"].max()) <= 5.0
  assert float(a["rel"].abs().max()) <= math.pi
  assert torch.allclose(a["dv_h"][:, :2].norm(dim=1), a["dv"], atol=1e-5)
  assert bool((a["dv_h"][:, 2] == 0).all())
  # a rejection rule redraws rows only; the onsets and pushes stay put
  odd = ql.draw_episodes(pool, room, 512, torch.Generator().manual_seed(3),
                         fits=lambda rows, on: rows % 2 == 1, **kw)
  assert bool((odd["row"] % 2 == 1).all()) and odd["redraw_rounds"] >= 1
  for key in ("on", "dv", "rel"):
    assert torch.equal(odd[key], a[key])
  assert bool((room[odd["row"]] >= odd["on"] + 246).all())
  with pytest.raises(ValueError, match="redraws"):
    ql.draw_episodes(pool, room, 8, torch.Generator().manual_seed(0),
                     fits=lambda rows, on: torch.zeros_like(rows, dtype=torch.bool),
                     max_rounds=3, **kw)


# --- telemetry -------------------------------------------------------------------------------

def _tel(n, engaged, dev, status, v=1.0, q=0.5, q_nom=0.2, iters=0):
  return dict(engaged=B(*engaged), deviation=torch.tensor(dev, dtype=torch.float32),
              status=torch.tensor(status), v=torch.full((n,), v), q=torch.full((n,), q),
              q_nom=torch.full((n,), q_nom), iters=torch.full((n,), iters))


def test_step_outcomes():
  track = B(1, 1, 1, 1, 0)
  dones = B(1, 1, 0, 0, 1)
  timeouts = B(0, 1, 0, 0, 0)
  k = torch.tensor([10, 10, 30, 29, 5])
  on = torch.tensor([12, 5, 10, 10, 1])
  oc = ql.step_outcomes(track, dones, timeouts, k, on, window=20)
  assert oc["fell"].tolist() == [True, False, False, False, False]
  assert oc["cut"].tolist() == [False, True, False, False, False]
  assert oc["up"].tolist() == [False, False, True, False, False]
  assert oc["after"].tolist() == [-2, 5, 20, 19, 4]


def test_episode_book_splits_and_books():
  book = ql.EpisodeBook(num_envs=3, episodes=4, dt=0.02)
  ep = torch.tensor([2, 0, -1])                    # env 2 idle
  on = torch.tensor([2, 1, 0])
  split = on.clone()
  book.start(ep >= 0)
  ids = torch.tensor([0, 1])
  k = torch.zeros(3, dtype=torch.long)
  # step 0: env 0 pre, env 1 pre; env 1 engaged by gd
  steps = [((False, True), (0.0, 2.0), (0, 1)),
           ((True, False), (1.5, 0.0), (3, 0)),    # k=1: env 0 pre, env 1 post
           ((True, True), (0.5, 4.0), (2, 1))]     # k=2: both post
  for eng, dev, st in steps:
    phase = (k[ids] >= split[ids]).long()
    book.step(ids, phase, _tel(2, eng, dev, st, iters=7), jerr=torch.tensor([0.1, 0.2]),
              rel_steps=k[ids] - on[ids])
    k[ids] += 1
  book.finish(B(1, 0, 0), ep, True, k, on)         # env 0: survived
  book.finish(B(0, 1, 0), ep, False, k, on)        # env 1: fell, 2 steps after its onset
  with pytest.raises(RuntimeError):
    book.finish(B(0, 1, 0), ep, False, k, on)      # booked twice
  with pytest.raises(RuntimeError):
    book.finish(B(0, 0, 1), ep, False, k, on)      # an idle env
  with pytest.raises(RuntimeError, match="never finished"):
    book.records()
  # the other two episodes, on env 2, one before the onset
  ep2 = torch.tensor([-1, -1, 1])
  book.start(B(0, 0, 1))
  book.step(torch.tensor([2]), torch.tensor([0]), _tel(1, (False,), (0.0,), (0,)))
  book.finish(B(0, 0, 1), ep2, False, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 3]))
  ep3 = torch.tensor([-1, -1, 3])
  book.start(B(0, 0, 1))
  book.finish(B(0, 0, 1), ep3, True, torch.tensor([0, 0, 9]), torch.tensor([0, 0, 2]))
  r = book.records()
  # episode 2 (env 0, onset 2): pre at k = 0, 1; post at k = 2
  assert r["tel_steps"][2].tolist() == [2.0, 1.0]
  assert r["tel_engaged"][2].tolist() == [1.0, 1.0]
  assert r["tel_dev_sum"][2].tolist() == [1.5, 0.5]
  assert r["tel_dev_max"][2].tolist() == [1.5, 0.5]
  assert r["tel_status"][2].tolist() == [[1, 0, 0, 1], [0, 0, 1, 0]]
  assert float(r["first_engaged_s"][2]) == pytest.approx(-1 * 0.02)
  # episode 0 (env 1, onset 1): pre at k = 0, post at k = 1, 2
  assert r["tel_steps"][0].tolist() == [1.0, 2.0]
  assert r["tel_engaged"][0].tolist() == [1.0, 1.0]
  assert r["tel_dev_sum"][0].tolist() == [2.0, 4.0]
  assert r["tel_iters_sum"][0].tolist() == [7.0, 14.0]
  assert r["tel_jerr_sum"][0].tolist() == pytest.approx([0.2, 0.4])
  assert bool(torch.isnan(r["tel_jerr_bad_sum"][0]).all())
  assert float(r["first_engaged_s"][0]) == pytest.approx(-0.02)
  assert r["survived"].tolist() == [False, False, True, True]
  assert r["fell_pre"].tolist() == [False, True, False, False]
  assert float(r["time_to_fall"][0]) == pytest.approx(2 * 0.02)
  assert float(r["time_to_fall"][1]) == pytest.approx(-2 * 0.02)
  assert math.isnan(r["time_to_fall"][2]) and math.isnan(r["time_to_fall"][3])
  assert r["env_id"].tolist() == [1, 2, 0, 2] and r["steps"].tolist() == [3, 1, 3, 9]
  assert math.isnan(r["first_engaged_s"][1])


def test_episode_book_solver_keys():
  book = ql.EpisodeBook(num_envs=2, episodes=2, dt=0.02)
  book.start(B(1, 1))
  tel = _tel(2, (True, False), (1.0, 0.0), (1, 0))
  tel.update(q_safe=torch.tensor([0.4, float("nan")]), fb_infeasible=B(1, 0), tr_bound=B(1, 0),
             seg_gap=torch.tensor([0.5, float("nan")]))
  book.step(torch.tensor([0, 1]), torch.ones(2, dtype=torch.long), tel)
  book.finish(B(1, 1), torch.tensor([0, 1]), True, torch.tensor([1, 1]), torch.tensor([0, 0]))
  r = book.records()
  assert r["tel_q_safe_sum"][:, 1].tolist() == pytest.approx([0.4, 0.0])   # engaged steps only
  assert r["tel_fb_infeasible"][:, 1].tolist() == [1.0, 0.0]
  assert r["tel_tr_bound"][:, 1].tolist() == [1.0, 0.0]
  assert r["tel_seg_gap_sum"][:, 1].tolist() == pytest.approx([0.5, 0.0])
  assert r["tel_seg_gap_n"][:, 1].tolist() == [1.0, 0.0]


def _records(E, survived, steps, engaged, dev, status, first=None, fell_pre=None):
  """Fabricated per-episode records in EpisodeBook.records' shape."""
  st = torch.tensor(steps, dtype=torch.float32)
  rec = {"survived": B(*survived),
         "fell_pre": B(*(fell_pre or [0] * E)),
         "time_to_fall": torch.tensor([float("nan") if s else 1.0 for s in survived]),
         "steps": st.sum(1).long(), "env_id": torch.arange(E),
         "first_engaged_s": torch.tensor(first or [float("nan")] * E),
         "tel_status": torch.tensor(status), "tel_steps": st,
         "tel_engaged": torch.tensor(engaged, dtype=torch.float32),
         "tel_dev_sum": torch.tensor(dev, dtype=torch.float32),
         "tel_dev_max": torch.tensor(dev, dtype=torch.float32) / 2}
  for key in ("v_sum", "q_sum", "q_nom_sum", "iters_sum", "jerr_sum"):
    rec["tel_" + key] = st * 0.5
  rec["tel_jerr_bad_sum"] = st * float("nan")
  return rec


def test_intervention_stats():
  rec = _records(
    3, survived=(1, 0, 1), steps=[[10, 20], [10, 20], [0, 5]],
    engaged=[[2, 10], [0, 20], [0, 0]], dev=[[4.0, 30.0], [0.0, 10.0], [0.0, 0.0]],
    status=[[[8, 2, 0, 0], [10, 6, 2, 2]], [[10, 0, 0, 0], [0, 0, 0, 20]],
            [[0, 0, 0, 0], [5, 0, 0, 0]]],
    first=[-0.3, 0.1, float("nan")])
  iv = ql.intervention_stats(rec, ref=B(1, 0, 0))
  assert iv["rate_pre"] == pytest.approx(2 / 20)
  assert iv["rate_post"] == pytest.approx(30 / 45)
  assert iv["dev_per_step_post"] == pytest.approx(40 / 45)
  assert iv["dev_per_engaged_post"] == pytest.approx(40 / 30)
  assert iv["dev_per_engaged_pre"] == pytest.approx(4 / 2)
  assert iv["best_effort_rate_post"] == pytest.approx(22 / 45)
  assert iv["best_effort_per_engaged_post"] == pytest.approx(22 / 30)
  assert iv["status_post"]["segment"] == pytest.approx(2 / 45)
  assert iv["status_pre"]["pass"] == pytest.approx(18 / 20)
  # false alarms over the episodes with pre steps (the third has none)
  assert iv["false_alarm_episodes"] == pytest.approx(1 / 2)
  assert iv["intervened_post_episodes"] == pytest.approx(2 / 3)
  assert iv["intervened_episodes"] == pytest.approx(2 / 3)
  assert iv["first_engaged_s_median"] == pytest.approx(-0.3)
  assert iv["rate_post_sonic_survived"] == pytest.approx(10 / 20)
  assert iv["rate_post_sonic_fell"] == pytest.approx(20 / 25)
  assert iv["v_mean_post"] == pytest.approx(0.5)
  assert math.isnan(iv["jerr_bad_post"])
  assert iv["dev_max_post"] == pytest.approx(15.0)
  assert iv["dev_max_mean_post"] == pytest.approx((15.0 + 5.0) / 2)
  assert "trust_region_per_engaged_post" not in iv          # records without the solver keys
  rec.update(tel_q_safe_sum=torch.tensor([[1.0, 5.0], [0.0, -2.0], [0.0, 0.0]]),
             tel_fb_infeasible=torch.tensor([[0.0, 3.0], [0.0, 12.0], [0.0, 0.0]]),
             tel_tr_bound=torch.tensor([[1.0, 6.0], [0.0, 0.0], [0.0, 0.0]]),
             tel_seg_gap_sum=torch.tensor([[0.0, 2.0], [0.0, 1.0], [0.0, 0.0]]),
             tel_seg_gap_n=torch.tensor([[0.0, 4.0], [0.0, 2.0], [0.0, 0.0]]))
  iv = ql.intervention_stats(rec)
  assert iv["q_safe_mean_engaged_post"] == pytest.approx(3.0 / 30)
  assert iv["fallback_infeasible_per_engaged_post"] == pytest.approx(15 / 30)
  assert iv["trust_region_per_engaged_post"] == pytest.approx(6 / 30)
  assert iv["trust_region_per_engaged_pre"] == pytest.approx(1 / 2)
  assert iv["seg_gap_mean_post"] == pytest.approx(3.0 / 6)


# --- the controller ---------------------------------------------------------------------------

class _Mon:
  """A fake PolicyValueMonitor: Q(x, u) = x0 - |u|_1 / 10, V(x) = x0."""

  def q_fn(self, action, x=None, **_):
    return x[:, 0] - action.abs().sum(-1) / 10.0

  def v_fn(self, x=None, **_):
    return x[:, 0].clone()

  def __call__(self, a, fallback=None, **ctx):
    with torch.no_grad():
      return self.q_fn(action=a, **ctx)


class _Switch:
  """LeastRestrictiveIntervention's rule, verbatim."""

  def __call__(self, a_nom, fallback, monitor, **ctx):
    value = monitor(a_nom, fallback=fallback, **ctx)
    engaged = value <= 0.0
    action = torch.where(engaged.unsqueeze(-1), fallback(**ctx), a_nom)
    return action, SimpleNamespace(engaged=engaged, value=value)


class _FakeQCBF:
  def __call__(self, a_nom, fallback, monitor, **ctx):
    n = a_nom.shape[0]
    status = torch.tensor([0, 1, 2, 3][:n])
    out = torch.where((status == 3).unsqueeze(-1), fallback(**ctx), a_nom + status.unsqueeze(-1))
    self.last = {"status": status, "deviation": (out - a_nom).norm(dim=-1),
                 "v": torch.ones(n), "q": torch.full((n,), 0.3), "q_nom": torch.full((n,), -0.1),
                 "gd_iters_used": torch.tensor([0, 12, 0, 60][:n]),
                 "q_safe": torch.tensor([0.5, 0.5, -1.0, -1.0][:n]),
                 "tau": torch.tensor([0.0, 0.2, 0.2, 0.2][:n]),
                 "dist_seg": torch.tensor([math.inf, 9.0, 1.5, math.inf][:n]),
                 "tr_bound": torch.tensor([True, True, False, True][:n])}
    return out, SimpleNamespace(engaged=status > 0, value=self.last["q"])


def test_controller_kinds():
  a_nom = torch.zeros(4, 29)
  a_safe = torch.ones(4, 29)
  x = torch.tensor([[1.0], [5.0], [-1.0], [3.0]]).expand(4, 97).clone()
  u, t = ql.Controller(ql.parse_condition("sonic"))(a_nom, None, None)
  assert u is a_nom and not bool(t["engaged"].any()) and bool(torch.isnan(t["v"]).all())
  u, t = ql.Controller(ql.parse_condition("fallback"))(a_nom, a_safe, None)
  assert torch.equal(u, a_safe) and bool(t["engaged"].all())
  assert t["status"].tolist() == [ql.BEST_EFFORT] * 4
  assert float(t["deviation"][0]) == pytest.approx(math.sqrt(29))
  with pytest.raises(ValueError, match="does not match"):
    ql.Controller(ql.parse_condition("fallback"))(a_nom, a_safe[:3], None)
  with pytest.raises(ValueError, match="intervention"):
    ql.Controller(ql.parse_condition("qcbf:exact:0.7"))
  q = ql.Controller(ql.parse_condition("qcbf:exact:0.7"), _FakeQCBF(), _Mon())
  u, t = q(a_nom, a_safe, x)
  assert t["status"].tolist() == [0, 1, 2, 3] and t["engaged"].tolist() == [False, True, True, True]
  assert t["iters"].tolist() == [0, 12, 0, 60] and torch.equal(u[3], a_safe[3])
  assert t["fb_infeasible"].tolist() == [False, False, True, True]
  assert t["tr_bound"].tolist() == [False, True, False, True]      # engaged steps only
  assert float(t["seg_gap"][1]) == pytest.approx(9.0 - math.sqrt(29))   # the descent won
  assert bool(torch.isnan(t["seg_gap"][[0, 2, 3]]).all())
  assert t["q_safe"].tolist() == [0.5, 0.5, -1.0, -1.0]
  with pytest.raises(ValueError, match="features"):
    q(a_nom, a_safe, x[:2])
  # the switch: Q(x, u_SONIC) = x0 (u_SONIC = 0) -> envs 2 hands over
  s = ql.Controller(ql.parse_condition("lrsf:exact"), _Switch(), _Mon())
  u, t = s(a_nom, a_safe, x)
  assert t["engaged"].tolist() == [False, False, True, False]
  assert t["status"].tolist() == [0, 0, ql.BEST_EFFORT, 0]
  assert torch.equal(u[2], a_safe[2]) and torch.equal(u[0], a_nom[0])
  assert t["q_nom"].tolist() == pytest.approx([1.0, 5.0, -1.0, 3.0])
  assert float(t["q"][2]) == pytest.approx(-1.0 - 2.9) and float(t["q"][1]) == pytest.approx(5.0)
  assert t["v"].tolist() == pytest.approx([1.0, 5.0, -1.0, 3.0])
  assert t["deviation"].tolist() == pytest.approx([0.0, 0.0, math.sqrt(29), 0.0])
  assert t["fb_infeasible"].tolist() == [False, False, True, False]   # Q(x, pi) = -3.9 <= 0
  assert not bool(t["tr_bound"].any()) and bool(torch.isnan(t["seg_gap"]).all())
  _, t = ql.Controller(ql.parse_condition("sonic"))(a_nom, None, None)
  assert not bool(t["fb_infeasible"].any()) and bool(torch.isnan(t["q_safe"]).all())
  # the rollout filter: the verdict (the rollout's min g) < 0 hands over; g = 0 passes
  r = ql.Controller(ql.parse_condition("rollout:1"))
  u, t = r(a_nom, a_safe, None, torch.tensor([0.4, -0.2, 0.0, -3.0]))
  assert t["engaged"].tolist() == [False, True, False, True]
  assert t["status"].tolist() == [0, ql.BEST_EFFORT, 0, ql.BEST_EFFORT]
  assert torch.equal(u[1], a_safe[1]) and torch.equal(u[2], a_nom[2])
  assert t["v"].tolist() == pytest.approx([0.4, -0.2, 0.0, -3.0])
  assert t["deviation"].tolist() == pytest.approx([0.0, math.sqrt(29), 0.0, math.sqrt(29)])
  assert bool(torch.isnan(t["q"]).all()) and not bool(t["fb_infeasible"].any())
  with pytest.raises(ValueError, match="verdict"):
    r(a_nom, a_safe, None)
  with pytest.raises(ValueError, match="verdict"):
    r(a_nom, a_safe, None, torch.zeros(3))
  _u, t = ql.Controller(ql.parse_condition("wmmin:1:0.9"))(a_nom, a_safe, None,
                                                          torch.tensor([0.1, -0.1, 0.0, 2.0]))
  assert t["engaged"].tolist() == [False, True, False, False]
  u, t = ql.Controller(ql.parse_condition("wmrollout:1"))(a_nom, a_safe, None,
                                                          torch.tensor([1.0, -1.0, 0.0, -0.1]))
  assert t["engaged"].tolist() == [False, True, False, True] and torch.equal(u[1], a_safe[1])


# --- summaries ------------------------------------------------------------------------------

def _suite(seed: int, E: int = 40, shift: float = 0.0):
  """A fabricated suite: episode columns and sonic / qcbf / lrsf records."""
  g = torch.Generator().manual_seed(seed)
  eps = {"dv": torch.rand(E, generator=g) * 5.0, "category": torch.randint(0, 3, (E,), generator=g),
         "ref_out_k": torch.rand(E, generator=g) < 0.1}
  sonic = (torch.rand(E, generator=g) > eps["dv"] / 5.0)
  recs = {}
  for name, p in (("sonic", None), ("qcbf:exact:0.7", 0.5 + shift), ("qcbf:td_g099:0.7", 0.4),
                  ("lrsf:exact", 0.2), ("lrsf:td_g099", 0.25)):
    surv = sonic.clone()
    if p is not None:                                    # rescues some, breaks a few
      surv = (sonic | (torch.rand(E, generator=g) < p)) & ~(torch.rand(E, generator=g) < 0.05)
    st = torch.stack([torch.full((E,), 50.0), torch.full((E,), 200.0)], 1)
    eng = torch.zeros(E, 2) if p is None else (st * torch.tensor([0.02, 0.3])).round()
    status = torch.zeros(E, 2, 4, dtype=torch.long)
    status[:, :, 0] = (st - eng).long()
    status[:, :, 1] = eng.long()
    rec = {"survived": surv, "fell_pre": torch.zeros(E, dtype=torch.bool),
           "time_to_fall": torch.where(surv, torch.full((E,), float("nan")), torch.full((E,), 1.2)),
           "steps": st.sum(1).long(), "env_id": torch.arange(E),
           "first_engaged_s": torch.full((E,), float("nan")), "tel_status": status,
           "tel_steps": st, "tel_engaged": eng, "tel_dev_sum": eng * 2.0, "tel_dev_max": eng.clamp_max(3.0),
           "tel_v_sum": st * 0.4, "tel_q_sum": st * 0.3, "tel_q_nom_sum": st * 0.1,
           "tel_iters_sum": eng * 20, "tel_jerr_sum": st * 0.05, "tel_jerr_bad_sum": st * float("nan")}
    recs[name] = rec
  return eps, recs


def test_summarize_suite_end_to_end():
  eps, recs = _suite(0)
  conds = ql.parse_conditions(list(recs))
  out = ql.summarize_suite(recs, eps, conds, ["walk", "run", "dance"], ql.DV_BINS)
  assert list(out) == list(recs)
  s, q = out["sonic"], out["qcbf:exact:0.7"]
  assert "paired" not in s and q["paired"]["n"] == 40
  assert q["survival"]["k"] == int(recs["qcbf:exact:0.7"]["survived"].sum())
  ok = ~eps["ref_out_k"]
  assert q["survival_excl_ref_out_k"]["n"] == int(ok.sum())
  assert q["paired_excl_ref_out_k"]["n"] == int(ok.sum())
  ref, sv = recs["sonic"]["survived"], recs["qcbf:exact:0.7"]["survived"]
  assert q["paired"]["rescued"] == int((~ref & sv).sum())
  assert q["paired"]["broken"] == int((ref & ~sv).sum())
  assert sum(b["n"] for b in q["by_dv"].values()) == 40
  assert set(q["by_category"]) <= {"walk", "run", "dance"}
  assert q["condition"]["kappa"] == 0.7 and q["intervention"]["rate_post"] == pytest.approx(0.3)
  assert "survival_excl_bad_out_k" not in q
  eps_b = dict(eps, bad_out_k=torch.zeros(40, dtype=torch.bool))
  qb = ql.summarize_condition(recs["qcbf:exact:0.7"], eps_b, conds[1], ref, [], None)
  assert qb["survival_excl_bad_out_k"]["n"] == 40 and "by_dv" not in qb
  text = json.dumps(ql.clean_json(out), allow_nan=False)     # strict JSON: NaN -> null
  assert json.loads(text)["sonic"]["intervention"]["dev_per_engaged_post"] is None
  assert "SR" in ql.headline(q) and "McNemar" in ql.headline(q)
  both = ql.concat_records([recs["sonic"], recs["sonic"]])
  assert both["tel_status"].shape == (80, 2, 4) and both["survived"].shape == (80,)
  with pytest.raises(ValueError, match="rows"):
    ql.summarize_condition(recs["sonic"], {k: v[:10] for k, v in eps.items()})


def _write_run(root, tag, seed, shift=0.0, partial=False, push=True):
  d = os.path.join(root, tag)
  os.makedirs(d, exist_ok=True)
  if push:
    eps, recs = _suite(seed, shift=shift)
    conds = ql.parse_conditions(list(recs))
    summ = ql.summarize_suite(recs, eps, conds, ["walk", "run", "dance"], ql.DV_BINS)
    with open(os.path.join(d, "push.json"), "w") as f:
      json.dump(ql.clean_json({"meta": {"tag": tag, "seed": seed}, "conditions": summ}), f)
  scen, pooled_recs, pooled_eps = {}, {}, []
  for i, name in enumerate(("hf_osc_4hz_0.6", "teleport")):
    eps, recs = _suite(100 + 10 * seed + i, E=24, shift=shift)
    eps["bad_out_k"] = torch.zeros(24, dtype=torch.bool)
    eps["family"] = torch.full((24,), i)
    conds = ql.parse_conditions(list(recs))
    scen[name] = {"family": ("hf_osc", "teleport")[i], "after": "return", "doc": "-",
                  "episodes": 24,
                  "conditions": ql.summarize_suite(recs, eps, conds, [], None)}
    pooled_eps.append(eps)
    for k, v in recs.items():
      pooled_recs.setdefault(k, []).append(v)
  out = {"meta": {"tag": tag, "seed": seed}, "scenarios": scen}
  if not partial:
    all_eps = ql.concat_records(pooled_eps)
    ref = ql.concat_records(pooled_recs["sonic"])["survived"]
    out["pooled"], out["by_family"] = {}, {}
    for c in ql.parse_conditions(list(pooled_recs)):
      rec = ql.concat_records(pooled_recs[c.name])
      r = None if c.name == "sonic" else ref
      out["pooled"][c.name] = ql.summarize_condition(rec, all_eps, c, r, [], None)
      for fi, fam in enumerate(("hf_osc", "teleport")):
        m = all_eps["family"] == fi
        out["by_family"].setdefault(fam, {})[c.name] = ql.summarize_condition(
          {k: v[m] for k, v in rec.items()}, {k: v[m] for k, v in all_eps.items()}, c,
          None if r is None else r[m], [], None)
  name = "badref.partial.json" if partial else "badref.json"
  with open(os.path.join(d, name), "w") as f:
    json.dump(ql.clean_json(out), f)
  return d


def test_summarizer_on_fake_runs(tmp_path):
  root = str(tmp_path / "qcbf_eval")
  _write_run(root, "shac_s0", 0, shift=0.2)
  _write_run(root, "shac_s1", 1, shift=0.2, partial=True)
  _write_run(root, "ppo_s0", 2)
  os.makedirs(os.path.join(root, "notes"))                    # no evaluation: ignored
  stem = str(tmp_path / "out" / "summary")
  res = sq.main(["--root", root, "--out", stem])
  assert res["groups"] == {"PPO": ["ppo_s0"], "SHAC": ["shac_s0", "shac_s1"]}
  assert res["partial_badref"] == ["shac_s1"]
  # pooled push SR = the two SHAC seeds' episodes together
  runs = {t: json.load(open(os.path.join(root, t, "push.json"))) for t in ("shac_s0", "shac_s1")}
  agg = res["push"]["qcbf:exact:0.7"]["SHAC"]
  k = sum(r["conditions"]["qcbf:exact:0.7"]["survival"]["k"] for r in runs.values())
  assert agg["seeds"] == 2 and agg["pooled"]["k"] == k and agg["pooled"]["n"] == 80
  srs = [r["conditions"]["qcbf:exact:0.7"]["survival"]["sr"] for r in runs.values()]
  assert agg["sr_mean"] == pytest.approx(sum(srs) / 2)
  assert agg["sr_std"] == pytest.approx(abs(srs[0] - srs[1]) / math.sqrt(2))
  b = sum(r["conditions"]["qcbf:exact:0.7"]["paired"]["rescued"] for r in runs.values())
  c = sum(r["conditions"]["qcbf:exact:0.7"]["paired"]["broken"] for r in runs.values())
  assert agg["vs_sonic"]["rescued"] == b and agg["vs_sonic"]["broken"] == c
  # different seeds: independent episodes, so the summed pairs are a valid test
  assert agg["pooled"]["method"] == "independent"
  assert agg["vs_sonic"]["p"] == pytest.approx(ql.mcnemar_exact_p(b, c))
  assert sum(v["n"] for v in agg["by_dv"].values()) == 80
  assert res["push"]["qcbf:exact:0.7"]["PPO"]["seeds"] == 1
  assert math.isnan(res["push"]["qcbf:exact:0.7"]["PPO"]["sr_std"])
  assert "vs_sonic" not in res["push"]["sonic"]["SHAC"]
  # exact vs td_g099: the qcbf pair at kappa 0.7 and the switch pair
  keys = {r["key"]: r for r in res["exact_vs_td"]}
  assert set(keys) == {"qcbf kappa 0.7", "lrsf (switch)"}
  assert keys["qcbf kappa 0.7"]["conditions"] == {"exact": "qcbf:exact:0.7",
                                                  "td_g099": "qcbf:td_g099:0.7"}
  assert keys["lrsf (switch)"]["groups"]["SHAC"]["td_g099"]["seeds"] == 2
  # bad references: per scenario both seeds; pooled / by family only the complete run
  assert res["badref"]["teleport"]["conditions"]["lrsf:exact"]["SHAC"]["seeds"] == 2
  assert res["badref_pooled"]["qcbf:exact:0.7"]["SHAC"]["seeds"] == 1
  assert res["badref_pooled"]["qcbf:exact:0.7"]["SHAC"]["pooled"]["n"] == 48
  assert set(res["badref_by_family"]) == {"hf_osc", "teleport"}
  md = open(stem + ".md").read()
  for needle in ("SHAC SR", "PPO SR", "Exact (3 s rollout labels) vs td_g099", "hf_osc_4hz_0.6",
                 "**all scenarios**", "family teleport", "Partial", "|dv|"):
    assert needle in md
  saved = json.load(open(stem + ".json"))
  assert saved["groups"]["SHAC"] == ["shac_s0", "shac_s1"]
  # explicit groups; a missing tag is skipped
  res2 = sq.main(["--root", root, "--out", stem + "_g", "--group", "S", "shac_s0", "shac_s9",
                  "--group", "P", "ppo_s0"])
  assert res2["groups"] == {"S": ["shac_s0"], "P": ["ppo_s0"]}
  with pytest.raises(SystemExit):
    sq.main(["--root", str(tmp_path / "nothing")])


def _shared_runs(root, outcomes, write_pt=(True, True, True)):
  """Three SHAC runs of ONE episode list (one --seed): SONIC's outcomes are
  the same in every run, the filter's ``outcomes[i]``; push.json and (where
  ``write_pt``) push.pt in the driver's shapes."""
  eps, recs = _suite(0)
  E = int(eps["dv"].numel())
  eps = dict(eps, row=torch.arange(E) * 7, on=torch.full((E,), 50))
  for i, surv in enumerate(outcomes):
    tag = f"shac_s{i}"
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    r = {k: dict(v) for k, v in recs.items()}
    r["qcbf:exact:0.7"]["survived"] = surv
    conds = ql.parse_conditions(list(r))
    summ = ql.summarize_suite(r, eps, conds, ["walk", "run", "dance"], ql.DV_BINS)
    meta = {"tag": tag, "seed": 0}
    with open(os.path.join(d, "push.json"), "w") as f:
      json.dump(ql.clean_json({"meta": meta, "conditions": summ}), f)
    if write_pt[i]:
      torch.save({"meta": meta, "episodes": eps, "conditions": r, "summary": summ},
                 os.path.join(d, "push.pt"))
  return eps, recs


def test_summarizer_clusters_shared_episodes(tmp_path):
  """One episode list run by three policies is three looks at the SAME
  episodes: identical runs must give one run's CI and one run's p, not a
  sqrt(3) narrower CI and the p of the tripled counts."""
  root = str(tmp_path / "shared")
  eps, recs = _suite(0)
  one = recs["qcbf:exact:0.7"]["survived"]
  ref = recs["sonic"]["survived"]
  _shared_runs(root, [one, one, one])
  res = sq.main(["--root", root, "--out", str(tmp_path / "s1")])
  agg = res["push"]["qcbf:exact:0.7"]["SHAC"]
  k1, E = int(one.sum()), int(one.numel())
  assert agg["pooled"]["method"] == "clustered" and agg["pooled"]["n"] == 3 * E
  assert agg["pooled"]["k"] == 3 * k1 and agg["pooled"]["n_eff"] == pytest.approx(E)
  assert agg["pooled"]["ci"] == pytest.approx(list(ql.wilson_ci(k1, E)))
  b1, c1 = int((~ref & one).sum()), int((ref & ~one).sum())
  vs = agg["vs_sonic"]
  assert (vs["rescued"], vs["broken"]) == (3 * b1, 3 * c1)
  assert (vs["episodes_better"], vs["episodes_worse"]) == (b1, c1)
  assert vs["p"] == pytest.approx(ql.mcnemar_exact_p(b1, c1))
  assert vs["p"] > ql.mcnemar_exact_p(3 * b1, 3 * c1)
  assert sum(v["n"] for v in agg["by_dv"].values()) == 3 * E
  assert agg["pooled_excl_ref_out_k"]["episodes"] == int((~eps["ref_out_k"]).sum())
  # SONIC itself: three identical runs are one run
  s = res["push"]["sonic"]["SHAC"]["pooled"]
  assert s["ci"] == pytest.approx(list(ql.wilson_ci(int(ref.sum()), E)))
  # runs that disagree: the effective size lies between E and 3 E
  g = torch.Generator().manual_seed(9)
  outs = [one ^ (torch.rand(E, generator=g) < 0.3) for _ in range(3)]
  root2 = str(tmp_path / "shared2")
  _shared_runs(root2, outs)
  agg2 = sq.main(["--root", root2, "--out", str(tmp_path / "s2")])["push"]["qcbf:exact:0.7"]["SHAC"]
  assert E < agg2["pooled"]["n_eff"] <= 3 * E
  Y = torch.stack(outs)
  d = (Y.double() - ref.double()).mean(0)
  assert agg2["vs_sonic"]["episodes_better"] == int((d > 0).sum())
  # a run without its push.pt: conservative (one seed's n, the largest per-seed p)
  root3 = str(tmp_path / "shared3")
  _shared_runs(root3, outs, write_pt=(True, False, True))
  agg3 = sq.main(["--root", root3, "--out", str(tmp_path / "s3")])["push"]["qcbf:exact:0.7"]["SHAC"]
  assert agg3["pooled"]["method"] == "conservative"
  k3 = sum(int(o.sum()) for o in outs)
  assert agg3["pooled"]["ci"] == pytest.approx(list(sq.wilson_p(k3 / (3 * E), E)))
  assert agg3["vs_sonic"]["p"] == pytest.approx(max(agg3["vs_sonic"]["p_seeds"]))


def test_summarizer_pairs_the_groups_on_shared_episodes(tmp_path):
  """SHAC and PPO runs of ONE episode list: the group contrast pairs the
  episodes (seed means, then the sign test), on the push suite and per
  bad-reference scenario; SONIC's row (the same controller in both groups) is
  the zero floor; a run without its .pt drops the contrast instead of
  pretending the episodes are independent."""
  root = str(tmp_path / "pairs")
  eps, recs = _suite(0)
  E = int(eps["dv"].numel())
  eps = dict(eps, row=torch.arange(E) * 7, on=torch.full((E,), 50))
  eps_b = dict(eps, bad_out_k=torch.zeros(E, dtype=torch.bool))
  g = torch.Generator().manual_seed(4)
  outs = {}
  for tag, p in (("ppo_s0", 0.4), ("shac_s0", 0.8), ("shac_s1", 0.7)):
    d = os.path.join(root, tag)
    os.makedirs(d)
    r = {k: dict(v) for k, v in recs.items()}
    r["qcbf:exact:0.7"]["survived"] = outs[tag] = torch.rand(E, generator=g) < p
    conds = ql.parse_conditions(list(r))
    meta = {"tag": tag, "seed": 0}
    summ = ql.summarize_suite(r, eps, conds, ["walk", "run", "dance"], ql.DV_BINS)
    with open(os.path.join(d, "push.json"), "w") as f:
      json.dump(ql.clean_json({"meta": meta, "conditions": summ}), f)
    torch.save({"meta": meta, "episodes": eps, "conditions": r}, os.path.join(d, "push.pt"))
    bs = ql.summarize_suite(r, eps_b, conds, [], None)
    with open(os.path.join(d, "badref.json"), "w") as f:
      json.dump(ql.clean_json({"meta": meta, "scenarios": {"teleport": {
        "family": "teleport", "after": "continue", "doc": "-", "episodes": E, "conditions": bs}},
        "pooled": bs, "by_family": {"teleport": bs}}), f)
    torch.save({"meta": meta, "scenarios": {"teleport": {"episodes": eps_b, "conditions": r}}},
               os.path.join(d, "badref.pt"))
  stem = str(tmp_path / "s")
  res = sq.main(["--root", root, "--out", stem])
  assert list(res["groups"]) == ["PPO", "SHAC"]
  diff = outs["ppo_s0"].double() - (outs["shac_s0"].double() + outs["shac_s1"].double()) / 2
  b, c = int((diff > 0).sum()), int((diff < 0).sum())
  for r in (res["push_contrast"]["PPO vs SHAC"]["qcbf:exact:0.7"],
            res["badref_contrast"]["PPO vs SHAC"]["teleport"]["qcbf:exact:0.7"],
            res["badref_contrast"]["PPO vs SHAC"]["all scenarios"]["qcbf:exact:0.7"]):
    assert (r["episodes_better"], r["episodes_worse"]) == (b, c)
    assert r["delta_sr"] == pytest.approx(float(diff.mean())) and r["seeds"] == [1, 2]
    assert r["p"] == pytest.approx(ql.mcnemar_exact_p(b, c)) and r["episodes"] == E
  s = res["push_contrast"]["PPO vs SHAC"]["sonic"]
  assert (s["delta_sr"], s["episodes_better"], s["episodes_worse"], s["p"]) == (0.0, 0, 0, 1.0)
  md = open(stem + ".md").read()
  for needle in ("PPO vs SHAC on the same episodes", "PPO vs SHAC per scenario",
                 "intervention rate pre / post"):
    assert needle in md
  os.remove(os.path.join(root, "ppo_s0", "push.pt"))
  res = sq.main(["--root", root, "--out", stem])
  assert res["push_contrast"]["PPO vs SHAC"] == {}
  assert "qcbf:exact:0.7" in res["badref_contrast"]["PPO vs SHAC"]["teleport"]


def test_clustered_reduces_to_wilson_for_one_run():
  g = torch.Generator().manual_seed(3)
  y = torch.rand(1, 200, generator=g) < 0.6
  c = sq.clustered(y)
  assert c["n_eff"] == pytest.approx(200) and c["ci"] == pytest.approx(
    list(ql.wilson_ci(int(y.sum()), 200)))
  assert sq.wilson_p(0.3, 50) == pytest.approx(ql.wilson_ci(15, 50))
  ind = torch.rand(4, 5000, generator=g) < 0.5            # independent runs: ~S E
  assert sq.clustered(ind)["n_eff"] > 0.9 * 4 * 5000
  t = sq.sign_test(y, ~y)
  assert t["episodes_better"] == int(y.sum()) and t["episodes_worse"] == 200 - int(y.sum())


def test_default_groups():
  assert sq.default_group("shac_s0") == "SHAC" and sq.default_group("ppo_s12") == "PPO"
  assert sq.default_group("sac_s1") == "SAC" and sq.default_group("privcrit") == "PRIVCRIT"
  assert sq.mean_std([1.0, None, float("nan"), 3.0]) == (2.0, math.sqrt(2.0), 2)
