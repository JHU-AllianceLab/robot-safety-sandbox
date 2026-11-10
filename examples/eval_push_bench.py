"""The SONIC push benchmark in the sandbox: his trial tables, our safety filters.

    python examples/eval_push_bench.py --table data/zz_test/trials.csv \
        --policy runs/push_bench/shac_s0/g1_recover_bench_safetyshac1p/final_model.zip \
        --tag shac_s0 --sonic-dir $HOME/gear_sonic_hf \
        --conditions sonic fallback wmmin:1:0.95 --tube-calib <min_tube.json> \
        --out runs/push_bench/eval/dev/shac_s0
    # then his paired metrics, per condition:
    python -m tools.sonic_bench.metrics wmmin=<out>/wmmin_1_0.95   # (in his repo)

THE WORLD is ``g1_recover_bench`` (``envs/g1_recover/push_bench.py``): his
physics, K as the failure, SONIC's hip, foot friction 1, no randomization,
the 70-D exact observation.

THE TRIALS (his ``push_bench.load_table``): every row of ``--table`` (his CSV
schema; ``examples/make_push_table.py`` writes more) is a push, and one
no-push control per context is appended. A trial:

1. SPAWN on the clip library's row of (motion, start_frame): the reference's
   pose, twist and joint state (soles on the floor, as the library stores
   them), at a random heading. SONIC tracks the clip from that frame.
2. WARM-UP: ``--warm`` ticks (100) of SONIC, run ONCE per trial; its state at
   the onset (MjData's qpos / qvel / solver warm start / time, the action
   buffers, SONIC's memory and clock, the contact sensors' air-time and force
   histories) is restored for every condition, so the arms of a trial differ
   only from the onset on (his runner copies the warm state the same way). A K
   exit here is ``warm_fail``; the trial is excluded, as his.
3. SCORED: ``--ticks`` ticks (510) from the onset; the condition acts from
   tick 0 (his shell's monitor starts there too). For ticks 0-9 the push acts:
   ``push_bench.PointPush`` at the row's body and body-frame point, force
   ``m dv / (10 dt)`` (m = this robot's mass). Its direction: the row's
   ``world_force_N`` turned by SONIC's delta heading (``direction_frame``
   ``reference``, his dev table: the force keeps its angle to the reference's
   heading at the onset) or the row's ``direction`` from the robot's actual
   heading at the onset (``actual``, his training-motion table).
4. OUTCOME: the trial survives if the robot stays in K through every scored
   tick. ``exit_tick`` is his: t + 1 for the first scored tick t whose step
   left K, -1 if none.

THE CONDITIONS are ``qcbf_eval_lib``'s ``sonic``, ``fallback`` (the policy
from the onset), ``rollout:<s>`` (the exact-sim filter: CPU MuJoCo rollouts
of "u_SONIC now, the policy after", K's margin), ``wmrollout:<s>`` /
``wmmin:<s>:<c>`` / ``wmtube:<s>:<R>:<c>`` (the world-model filters; ``wmmin``
needs ``--tube-calib``) with ``+latch`` / ``+dwell<s>``, and ``wmcalib:clean`` /
``wmcalib:<R>`` (calibration records: SONIC to a decision tick ~ U{0 ..
--calib-max-delay-s} after the onset, the policy after it; the world model's
prediction there and the real margins after it, for
``calibrate_wm_min_tube.py`` / ``calibrate_wm_tube.py``). Every condition runs
the same trials in the same env slots from the same warm state, so the arms
are paired; GPU contact accumulation is not bit-deterministic, which
``--repeat-sonic`` measures.

OUTPUT in ``--out``:

    episodes_bank.pt        the spawn rows, in trial order (the env's dataset)
    bench.pt                {"meta", "trials": per-trial columns, "conditions":
                             {name: per-trial records}} (wmcalib: also the
                             calibrate_wm_*_tube.py layout: "episodes" with dv)
    summary.json            per condition: survival (Wilson 95 %), by dv and by
                            motion, rescued / harmed against sonic, the no-push
                            trigger rate, policy share of scored ticks, hand-overs
    <condition>/trials.npz  his schema (first n = the sonic arm, last n = this
                            condition's), for tools.sonic_bench.metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

_EX = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_EX)
sys.path.insert(0, _ZOO)
sys.path.insert(0, _EX)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get("SAFETY_SB3_PATH",
                         os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import eval_sonic_qcbf as eq  # noqa: E402
import qcbf_eval_lib as ql  # noqa: E402
from harvest_g1_sonic_trajectories import make_env  # noqa: E402

from robot_safety_sandbox.envs.g1_recover import push_bench as pb  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_value_filter import (  # noqa: E402
  ACTION_DIM, record_applied_action)
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SonicPolicy, SonicReference, SonicTracker)
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_MOCAP, OPTIONAL_FIELDS, load_bank, save_bank)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402

DT = 0.02
TASK = "g1_recover_bench"
#: the conditions this runner drives (qcbf / lrsf need fitted values: eval_sonic_qcbf.py)
BENCH_KINDS = ("sonic", "fallback", "rollout", "wmrollout", "wmtube", "wmmin", "wmcalib")


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--table", required=True, help="trials.csv (his schema)")
  p.add_argument("--direction-frame", choices=("reference", "actual"), default=None,
                 help="override the table's direction_frame column (his files have none: "
                      "default reference)")
  p.add_argument("--policy", required=True, help="the fallback's final_model.zip")
  p.add_argument("--tag", default=None)
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips.pt"),
                 help="a clip library holding every motion of the table (all frames)")
  p.add_argument("--sonic-dir", default=DEFAULT_SONIC_DIR)
  p.add_argument("--conditions", nargs="+", default=["sonic", "fallback"])
  p.add_argument("--tube-calib", default=None)
  p.add_argument("--out", required=True)
  p.add_argument("--warm", type=int, default=pb.WARM_TICKS)
  p.add_argument("--ticks", type=int, default=pb.SCORED_TICKS)
  p.add_argument("--no-controls", action="store_true", help="no no-push control per context")
  p.add_argument("--limit", type=int, default=None, help="the first N trials only")
  p.add_argument("--sample", type=int, default=None,
                 help="N trials drawn at random (seeded by --seed) instead of all")
  p.add_argument("--repeat-sonic", type=int, default=0,
                 help="diagnostic: rerun the sonic condition this many more times in this "
                      "process and report how many outcomes change (the simulator's noise floor)")
  p.add_argument("--calib-horizon-s", type=float, default=3.0)
  p.add_argument("--calib-max-delay-s", type=float, default=1.0)
  p.add_argument("--rollout-workers", type=int,
                 default=int(os.environ.get("SLURM_CPUS_PER_TASK",
                                            max(1, (os.cpu_count() or 2) - 2))))
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  try:
    a.conds = ql.reference_first(ql.parse_conditions(a.conditions))
  except ValueError as e:
    p.error(str(e))
  bad = [c.name for c in a.conds if c.kind not in BENCH_KINDS]
  if bad:
    p.error(f"conditions {bad}: this runner drives {BENCH_KINDS}")
  if a.conds[0].name != ql.REFERENCE and not all(c.kind == "wmcalib" for c in a.conds):
    p.error("the paired statistics need the sonic condition")
  a.tag = a.tag or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.policy))))
  # what eval_sonic_qcbf.Evaluator reads off its args
  a.task, a.hip, a.sonic_history = TASK, "sonic", "applied"
  a.pi_obs_clamp = "box"
  a.qcbf_margin, a.qcbf_iters, a.best_effort = 0.0, 60, "fallback"
  a.qcbf_max_dev, a.lrsf_eps = math.sqrt(ACTION_DIM), 0.0
  return a


# --- trials ------------------------------------------------------------------------------

def load_trials(a) -> dict:
  """The table's pushes plus one no-push control per context, as columns."""
  rows = list(csv.DictReader(open(a.table)))
  if a.sample is not None:
    g = np.random.default_rng(np.random.SeedSequence([a.seed, 7]))
    rows = [rows[i] for i in sorted(g.choice(len(rows), min(a.sample, len(rows)), replace=False))]
  if a.limit is not None:
    rows = rows[:a.limit]
  ctx = {}
  for r in rows:
    ctx.setdefault(int(r["context_id"]), r)
  trials = [dict(context_id=int(r["context_id"]), motion=r["motion"], start_frame=int(r["start_frame"]),
                 site=r["site"], direction=r["direction"], dv=float(r["dv_mps"]),
                 force_w=json.loads(r["world_force_N"]), point=json.loads(r["point_body"]),
                 frame_kind=(a.direction_frame or r.get("direction_frame") or "reference"),
                 trial_id=int(r["trial_id"])) for r in rows]
  if not a.no_controls:
    for c, r in sorted(ctx.items()):
      trials.append(dict(context_id=c, motion=r["motion"], start_frame=int(r["start_frame"]),
                         site="pelvis", direction="none", dv=0.0, force_w=[0.0, 0.0, 0.0],
                         point=[0.0, 0.0, 0.0], frame_kind="reference", trial_id=-1))
  kinds = {t["frame_kind"] for t in trials if t["dv"] > 0}
  if not kinds <= {"reference", "actual"}:
    raise SystemExit(f"[bench] direction_frame {kinds}: reference | actual")
  names = list(pb.PUSH_SITES)
  return {
    "context_id": torch.tensor([t["context_id"] for t in trials]),
    "motion": [t["motion"] for t in trials],
    "start_frame": torch.tensor([t["start_frame"] for t in trials]),
    "site": torch.tensor([names.index(t["site"]) for t in trials]),
    "site_name": [t["site"] for t in trials],
    "direction": [t["direction"] for t in trials],
    "angle": torch.tensor([pb.DIRECTIONS.get(t["direction"], 0.0) for t in trials]),
    "dv": torch.tensor([t["dv"] for t in trials], dtype=torch.float32),
    "force_w": torch.tensor([t["force_w"] for t in trials], dtype=torch.float32),
    "point": torch.tensor([t["point"] for t in trials], dtype=torch.float32),
    "actual": torch.tensor([t["frame_kind"] == "actual" for t in trials]),
    "trial_id": torch.tensor([t["trial_id"] for t in trials]),
  }


def library_rows(lib: dict, trials: dict) -> torch.Tensor:
  """The library row of each trial's (motion, start_frame)."""
  names = list(lib["meta"]["clips"])
  out = torch.empty(len(trials["motion"]), dtype=torch.long)
  for m in sorted(set(trials["motion"])):
    if m not in names:
      raise SystemExit(f"[bench] motion {m} is not in the library")
    rows = (lib["clip"] == names.index(m)).nonzero(as_tuple=False).squeeze(-1)
    frames = lib["frame"][rows]
    order = torch.argsort(frames)
    rows, frames = rows[order], frames[order]
    sel = torch.tensor([i for i, mm in enumerate(trials["motion"]) if mm == m])
    want = trials["start_frame"][sel]
    pos = torch.searchsorted(frames, want)
    if bool((pos >= frames.numel()).any()) or bool((frames[pos.clamp_max(frames.numel() - 1)] != want).any()):
      raise SystemExit(f"[bench] {m}: a start frame has no library row")
    out[sel] = rows[pos]
  return out


def write_bank(lib: dict, rows: torch.Tensor, path: str) -> None:
  """The trials' spawn rows in order (every one spawnable: the library's
  start_ok filter is the training spawn's, not the benchmark's)."""
  bank = {f: lib[f][rows] for f in FIELDS + tuple(OPTIONAL_FIELDS) if f in lib}
  bank["next_ok"] = torch.zeros_like(bank["next_ok"])
  if not bool((bank["kind"] == KIND_MOCAP).all()):
    raise SystemExit("[bench] a trial spawns on a row that is not a mocap row")
  bank["start_ok"] = torch.ones_like(bank["start_ok"])
  bank["meta"] = dict(lib["meta"], source="eval_push_bench.py (episode bank)",
                      library_rows=rows.clone(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
  save_bank(bank, path)


# --- the runner -------------------------------------------------------------------------

def _t(x):
  """The torch view of an mjlab sim array (``TorchArray``) or a tensor."""
  return getattr(x, "_tensor", x)


#: MjData fields of a warm state (``time`` is the contact sensors' air-time clock)
SIM_FIELDS = ("qpos", "qvel", "qacc_warmstart", "time")


class BenchRunner:
  """One condition over every trial (module docstring), on an
  ``eval_sonic_qcbf.Evaluator``'s env, SONIC, policy and filter helpers.

  :meth:`warmup` runs every trial's SONIC warm-up ONCE and keeps its state at
  the onset (his runner's shared warm state); :meth:`run` restores it for each
  condition, so the arms of a trial differ only from the onset on."""

  def __init__(self, ev: eq.Evaluator, a, tr_cols: dict, clip: torch.Tensor):
    self.ev, self.a = ev, a
    self.dev, self.n = ev.dev, ev.n
    self.cols = {k: (v.to(ev.dev) if torch.is_tensor(v) else v) for k, v in tr_cols.items()}
    self.clip = clip.to(ev.dev)
    self.pp = pb.PointPush(ev.robot, ev.dev)
    self.mass = float(ev.mj.sim.mj_model.body_mass.sum())
    self.snap: dict | None = None
    self.warm_fail: torch.Tensor | None = None

  # --- the shared warm state ---------------------------------------------------------
  def _live(self) -> dict:
    """Every per-env tensor a warm state is made of, fetched now (several are
    replaced, not updated in place, as the env and SONIC step)."""
    mj, tr = self.ev.mj, self.ev.tr
    d = mj.sim.data
    out = {f: _t(getattr(d, f)) for f in SIM_FIELDS}
    am = mj.action_manager
    out.update(action=am._action, prev_action=am._prev_action,
               prev_prev_action=am._prev_prev_action, episode_length=mj.episode_length_buf,
               sonic_last=tr.last_action, sonic_frame=tr.frame, sonic_dq=tr.dq)
    out.update({f"sonic_hist{i}": h for i, h in enumerate(tr.hist)})
    for name, sen in mj.scene.sensors.items():
      st = getattr(sen, "_air_time_state", None)
      if st is not None:
        for f in ("current_air_time", "last_air_time", "current_contact_time",
                  "last_contact_time", "last_time"):
          out[f"{name}/air/{f}"] = getattr(st, f)
      for k, v in (getattr(sen, "_history_state", None) or {}).items():
        out[f"{name}/hist/{k}"] = v
    return out

  def _capture(self, ids: torch.Tensor, e: torch.Tensor) -> None:
    live = self._live()
    if self.snap is None:
      E = int(self.cols["dv"].numel())
      self.snap = {k: torch.zeros((E,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                   for k, v in live.items()}
    for k, v in live.items():
      self.snap[k][e] = v[ids]

  def _restore(self, ids: torch.Tensor, e: torch.Tensor) -> None:
    for k, v in self._live().items():
      v[ids] = self.snap[k][e]
    mj = self.ev.mj
    mj.sim.forward()
    for sen in mj.scene.sensors.values():
      sen._invalidate_cache()

  def _start_wave(self, w0: int, w1: int, label: str):
    """Reset the env onto the wave's trials and start SONIC on their spawn
    frames. Returns each env's trial (-1: idle), the mask and the clamped index."""
    ev, c, n, dev = self.ev, self.cols, self.n, self.dev
    torch.manual_seed(eq._seed(self.a.seed, f"bench/wave{w0}"))
    ev._set_cursor(w0)
    ev.env.reset()
    reset_policy_memory(ev.model.policy)
    ep = ev._wave_envs(label, 0, w0, w1)
    take = ep >= 0
    epc = ep.clamp_min(0)
    everyone = torch.ones(n, dtype=torch.bool, device=dev)
    ev.tr.stop(everyone)
    ev.tr.clip.zero_()
    ev.tr.frame.zero_()
    ev.tr.start(take, self.clip[epc], c["start_frame"][epc])
    self.pp.stop(everyone)
    return ep, take, epc

  @torch.no_grad()
  def warmup(self) -> None:
    """SONIC's warm-up of every trial (``--warm`` ticks from the spawn), once:
    ``warm_fail`` where it left K, the state at the onset otherwise."""
    ev, a, dev, n = self.ev, self.a, self.dev, self.n
    tr = ev.tr
    E, W = int(self.cols["dv"].numel()), int(a.warm)
    self.warm_fail = torch.zeros(E, dtype=torch.bool, device=dev)
    t0 = time.time()
    for w0 in range(0, E, n):
      w1 = min(E, w0 + n)
      ep, take, epc = self._start_wave(w0, w1, "bench/warmup")
      track = take.clone()
      for _ in range(W):
        act = tr.act()
        record_applied_action(tr, act, mask=track)
        _obs, _g, dones, timeouts, _l = ev.env.step_tensor(act)
        tr.advance()
        if bool((track & dones & timeouts).any()):
          raise SystemExit("[bench] the env's time cap cut a warm-up")
        fell = track & dones
        self.warm_fail[ep[fell]] = True
        track &= ~fell
        tr.stop(dones)
      ids = track.nonzero(as_tuple=False).squeeze(-1)
      self._capture(ids, ep[ids])
    print(f"[bench] warm-up: {E} trials, {int(self.warm_fail.sum())} left K "
          f"({time.time() - t0:.0f}s)", flush=True)

  def _dir_world(self, ids: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Unit world force directions at the onset of trials ``e`` in envs ``ids``."""
    c = self.cols
    dq = self.ev.tr.dq[ids]
    w, x, y, z = dq.unbind(-1)
    psi = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))   # robot - reference heading
    fw = c["force_w"][e]
    ref_ang = torch.atan2(fw[:, 1], fw[:, 0]) + psi
    act_ang = self.ev.robot.data.heading_w.reshape(-1)[ids] + c["angle"][e]
    ang = torch.where(c["actual"][e], act_ang, ref_ang)
    return ang

  @torch.no_grad()
  def run(self, cond: ql.Condition) -> dict:
    ev, a, dev, n = self.ev, self.a, self.dev, self.n
    mj, tr, c = ev.mj, ev.tr, self.cols
    E = int(c["dv"].numel())
    W, T = int(a.warm), int(a.ticks)
    ctl = ev.controller(cond)
    kind = cond.kind
    policy = ev.model.policy
    latch, dwell = bool(cond.latch), (eq._steps(cond.dwell_s) if cond.dwell_s is not None else 0)
    calib = kind == "wmcalib"
    long = dict(dtype=torch.long, device=dev)
    rec = {k: torch.full((E,), -1, **long) for k in ("exit_tick", "first_trigger", "exit_mode")}
    rec.update({k: torch.zeros(E, **long) for k in ("triggers", "resumes", "policy_ticks",
                                                     "task_ticks", "mode", "env_id")})
    if calib:
      Hc, Kc = eq._steps(a.calib_horizon_s), eq._steps(a.calib_max_delay_s)
      if Kc + Hc > T:
        raise SystemExit(f"[bench] {cond.name}: decision delay + horizon exceed the scored ticks")
      gen_c = torch.Generator().manual_seed(eq._seed(a.seed, "bench/calib"))
      k_off = torch.randint(0, Kc + 1, (E,), generator=gen_c).to(dev)
      R = cond.n_rollouts or 1
      c_pred = torch.full((E, R, Hc), math.nan, device=dev)
      c_real = torch.full((E, Hc), math.nan, device=dev)
      rec_st = getattr(ev, "wm_state_sl", None) is not None
      if rec_st:
        c_pred_st = torch.full((E, R, Hc, 2), math.nan, device=dev)
        c_real_st = torch.full((E, Hc, 2), math.nan, device=dev)
    t0, rollout_s, loop = time.time(), 0.0, 0
    nefc = getattr(mj.sim.data, "nefc", None)
    nefc_max = 0
    horizon = eq._steps(cond.horizon_s) if kind in ql.ROLLOUT_KINDS else 0
    waves = (E + n - 1) // n
    if self.snap is None:
      raise SystemExit("[bench] run warmup() first")
    rec["warm_fail"] = self.warm_fail.clone()
    rec["done"] = self.warm_fail.clone()
    for w, w0 in enumerate(range(0, E, n)):
      w1 = min(E, w0 + n)
      ep, take, epc = self._start_wave(w0, w1, f"bench/{cond.name}")
      # every trial resumes from its shared warm state, at the onset
      track = take & ~self.warm_fail[epc]
      ids0 = track.nonzero(as_tuple=False).squeeze(-1)
      self._restore(ids0, ep[ids0])
      obs = torch.nan_to_num(mj.observation_manager.compute_group(
        ev.env.obs_key, update_history=False).float(), nan=0.0, posinf=0.0, neginf=0.0)
      k = torch.full((n,), W, **long)
      eng_prev = torch.zeros(n, dtype=torch.bool, device=dev)
      fired = torch.zeros(n, dtype=torch.bool, device=dev)
      hold_until = torch.zeros(n, **long)
      rec["env_id"][ep[take]] = take.nonzero(as_tuple=False).squeeze(-1)
      for _ in range(T + 2):
        if not bool(track.any()):
          break
        ids = track.nonzero(as_tuple=False).squeeze(-1)
        a_nom = tr.act()
        on = k[ids] >= W                                  # scored ticks: the condition acts
        aid = ids[on]
        act = a_nom.clone()
        eng_now = torch.zeros(n, dtype=torch.bool, device=dev)
        if aid.numel() and kind != "sonic":
          pobs, _b = ev._pi_obs(obs)
          a_safe = torch.clamp(policy._predict(ev.norm(pobs), deterministic=True), ev.low, ev.high)
          verdict = None
          t_r = time.time()
          if calib:
            k_dec = W + k_off[ep[aid]]
            now = k[aid] == k_dec
            if bool(now.any()):
              d_ids = aid[now]
              if rec_st:
                m_p, s_p = ev.wm_margins(d_ids, a_nom[d_ids].float(), Hc, cond.n_rollouts,
                                         states=True)
                c_pred[ep[d_ids]], c_pred_st[ep[d_ids]] = m_p, s_p
              else:
                c_pred[ep[d_ids]] = ev.wm_margins(d_ids, a_nom[d_ids].float(), Hc, cond.n_rollouts)
            verdict = torch.where(k[aid] > k_dec, -1.0, 1.0).float()
          elif kind == "wmtube":
            verdict = ev.tube_verdict(aid, a_nom[aid].float(), cond)
          elif kind == "wmmin":
            verdict = ev.min_verdict(aid, a_nom[aid].float(), cond)
          elif kind == "wmrollout":
            verdict = ev.wm_verdict(aid, a_nom[aid].float(), horizon, cond.n_rollouts)
          elif kind == "rollout":
            qpos, qvel, ws = eq.cr.live_state(ev.env, aid)
            gmin, _tf = ev.pool.rollout(aid.cpu().numpy(), qpos, qvel, ws,
                                        a_nom[aid].double().cpu().numpy(), horizon,
                                        n_rollouts=cond.n_rollouts or 1,
                                        noise=cond.n_rollouts is not None)
            verdict = torch.as_tensor(gmin, dtype=torch.float32, device=dev)
          rollout_s += time.time() - t_r
          if verdict is not None:
            if latch:
              verdict = torch.where(fired[aid], torch.full_like(verdict, -1.0), verdict)
            if dwell:
              verdict = torch.where(k[aid] < hold_until[aid], torch.full_like(verdict, -1.0), verdict)
          u, tel = ctl(a_nom[aid], a_safe[aid], None, verdict)
          act[aid] = u.to(act.dtype)
          e_on = tel["engaged"].bool()
          eng_now[aid] = e_on
          if latch:
            fired[aid] |= e_on
          if dwell:
            start = e_on & ~eng_prev[aid]
            hold_until[aid] = torch.where(start, k[aid] + dwell, hold_until[aid])
        # the hand-over bookkeeping, scored ticks only (his: triggers / resumes)
        if aid.numel():
          e = ep[aid]
          en, pr = eng_now[aid], eng_prev[aid]
          rec["triggers"][e] += (en & ~pr).long()
          rec["resumes"][e] += (~en & pr).long()
          rec["policy_ticks"][e] += en.long()
          rec["task_ticks"][e] += (~en).long()
          t_now = k[aid] - W
          first = en & (rec["first_trigger"][e] < 0)
          rec["first_trigger"][e[first]] = t_now[first]
          eng_prev[aid] = en
        record_applied_action(tr, act, mask=track)
        # the push: armed at the onset tick, held 10 ticks
        due = track & (k == W) & (c["dv"][epc] > 0)
        if bool(due.any()):
          d_ids = due.nonzero(as_tuple=False).squeeze(-1)
          e = ep[d_ids]
          ang = torch.zeros(n, device=dev)
          ang[d_ids] = self._dir_world(d_ids, e)
          steps = torch.full((n,), pb.PUSH_TICKS, **long)
          f_w = pb.push_force(self.mass, c["dv"][epc], steps, DT, ang)
          self.pp.start(due, c["site"][epc], c["point"][epc], f_w, steps)
        self.pp.apply()
        obs, g, dones, timeouts, _l = ev.env.step_tensor(act)
        if nefc is not None:
          nefc_max = max(nefc_max, int(eq.cr._np(nefc).max()))
        k += 1
        self.pp.tick()
        self.pp.stop(dones)
        if calib:
          j = k - (W + k_off[epc])
          rm = track & (j >= 1) & (j <= Hc) & (k > W)
          if bool(rm.any()):
            r_ids = rm.nonzero(as_tuple=False).squeeze(-1)
            c_real[ep[r_ids], j[r_ids] - 1] = g[r_ids].float()
            live = rm & ~dones
            if rec_st and bool(live.any()):
              l_ids = live.nonzero(as_tuple=False).squeeze(-1)
              cs = torch.nan_to_num(mj.observation_manager.compute_group(
                "critic", update_history=False).float())[l_ids]
              c_real_st[ep[l_ids], j[l_ids] - 1] = ev.margin_states(cs)
        tr.advance()
        fell = track & dones & ~timeouts
        if bool((track & dones & timeouts).any()):
          raise SystemExit(f"[bench] {cond.name}: the env's time cap cut a trial")
        warm = fell & (k <= W)
        ex = fell & (k > W)
        up = track & ~dones & (k >= W + T)
        if bool(warm.any()):
          rec["warm_fail"][ep[warm]] = True
        if bool(ex.any()):
          x_ids = ex.nonzero(as_tuple=False).squeeze(-1)
          rec["exit_tick"][ep[x_ids]] = k[x_ids] - W
          rec["exit_mode"][ep[x_ids]] = eng_prev[x_ids].long()
        if bool(up.any()):
          u_ids = up.nonzero(as_tuple=False).squeeze(-1)
          rec["mode"][ep[u_ids]] = eng_prev[u_ids].long()
          ev._finish(up)
        ended = warm | ex | up
        rec["done"][ep[ended]] = True
        track &= ~ended
        tr.stop(dones)
        if bool(dones.any()):
          reset_policy_memory(policy, dones)
        loop += 1
      if bool(track.any()):
        raise SystemExit(f"[bench] {cond.name}: {int(track.sum())} trials still running")
      el = max(time.time() - t0, 1e-6)
      print(f"[bench]   {cond.name}: wave {w + 1}/{waves} ({int(rec['done'].sum())}/{E} trials, "
            f"{loop * n / el:.0f} env-steps/s)", flush=True)
    if not bool(rec["done"].all()):
      raise SystemExit(f"[bench] {cond.name}: {int((~rec['done']).sum())} trials never finished")
    out = {k: v.cpu() for k, v in rec.items() if k != "done"}
    out["_run"] = {"wall_s": time.time() - t0, "loop_steps": loop, "verdict_wall_s": rollout_s,
                   "nefc_max": nefc_max, "njmax": getattr(mj.cfg.sim, "njmax", None)}
    print(f"[bench]   {cond.name}: most constraints in a world {nefc_max} (njmax "
          f"{getattr(mj.cfg.sim, 'njmax', None)})", flush=True)
    if calib:
      out.update(calib_pred=c_pred.cpu(), calib_real=c_real.cpu(), calib_decision=k_off.cpu())
      if rec_st:
        out.update(calib_pred_states=c_pred_st.cpu(), calib_real_states=c_real_st.cpu(),
                   calib_state_names=["pelvis_height_m", "tilt_deg"])
    return out


# --- statistics and his schema --------------------------------------------------------------

def summarize(rec: dict, ref: dict | None, cols: dict) -> dict:
  ok = ~rec["warm_fail"] if ref is None else ~(rec["warm_fail"] | ref["warm_fail"])
  surv = rec["exit_tick"] < 0
  push = cols["dv"] > 0
  m = ok & push

  def sr(mask):
    k, nn = int((surv & mask).sum()), int(mask.sum())
    lo, hi = ql.wilson_ci(k, nn)
    return {"n": nn, "sr": (k / nn) if nn else None, "lo": lo, "hi": hi}

  out = {"pushed": sr(m), "warm_fail": int(rec["warm_fail"].sum()),
         "by_dv": {f"{d:g}": sr(m & (cols["dv"] == d)) for d in sorted(set(cols["dv"][push].tolist()))},
         "by_motion": {mo: sr(m & torch.tensor([x == mo for x in cols["motion"]]))
                       for mo in sorted(set(cols["motion"]))},
         "policy_share": float(rec["policy_ticks"][m].sum()) / max(
           float((rec["policy_ticks"] + rec["task_ticks"])[m].sum()), 1.0),
         "handovers_per_trial": float(rec["triggers"][m].float().mean()) if bool(m.any()) else None,
         "survived_with_sonic_at_end": (float((surv & (rec["mode"] == 0) & m).sum()) / max(int(m.sum()), 1))}
  nop = ok & ~push
  if bool(nop.any()):
    out["no_push"] = {"n": int(nop.sum()), "triggered": float((rec["triggers"][nop] > 0).float().mean()),
                      "survived": float(surv[nop].float().mean())}
  if ref is not None:
    rs = ref["exit_tick"] < 0
    out["paired"] = {"rescued": int((m & surv & ~rs).sum()), "harmed": int((m & ~surv & rs).sum()),
                     "sonic_sr": float(rs[m].float().mean()) if bool(m.any()) else None,
                     "mcnemar_p": ql.mcnemar_exact_p(int((m & surv & ~rs).sum()),
                                                     int((m & ~surv & rs).sum()))}
  return out


def his_npz(path: str, ref: dict, rec: dict, cols: dict) -> None:
  """His trials.npz: every per-trial array twice, the sonic arm then this one."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  keys = ("warm_fail", "exit_tick", "triggers", "first_trigger", "mode", "exit_mode",
          "resumes", "policy_ticks", "task_ticks")
  arr = {k: np.concatenate([ref[k].numpy(), rec[k].numpy()]) for k in keys}
  arr["warm_fail"] = arr["warm_fail"].astype(bool)
  arr["conn_ticks"] = arr.pop("policy_ticks")
  np.savez_compressed(path, **arr, dv=cols["dv"].numpy(), motion=np.array(cols["motion"]),
                      site=np.array(cols["site_name"]), direction=np.array(cols["direction"]),
                      context_id=cols["context_id"].numpy(),
                      cpu_survival=np.zeros(len(cols["motion"]), dtype=bool))


def main(argv=None):
  a = _parse(argv)
  dev = a.device
  t_start = time.time()
  os.makedirs(a.out, exist_ok=True)
  torch.manual_seed(a.seed)
  cols = load_trials(a)
  lib = load_bank(a.library, "cpu")
  # only the table's motions: SONIC's reference is built (and checked against
  # the library) for those; a few fallAndGetUp frames of the full library fail
  # that check and no table uses them
  from subset_clip_library import subset
  lib = subset(lib, sorted(set(cols["motion"])), {}, pb.PREVIEW_FRAMES, 50.0)
  rows = library_rows(lib, cols)
  clip = lib["clip"][rows]
  bank_path = os.path.join(a.out, "episodes_bank.pt")
  write_bank(lib, rows, bank_path)
  E = int(rows.numel())
  print(f"[bench] {a.table}: {E} trials ({int((cols['dv'] > 0).sum())} pushes, "
        f"{int((cols['dv'] == 0).sum())} no-push controls); policy {a.policy}", flush=True)

  csv_dir = lib["meta"]["csv_dir"]
  if not os.path.isdir(csv_dir):
    csv_dir = os.path.join(_ZOO, "data", "lafan1_g1")
  base = SonicReference.build(csv_dir, lib["meta"]["clips"], dev, library=lib)
  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  overrides = dict(dataset=bank_path, sampler="sequential", sequential_repeats=1, weights=weights,
                   episode_length_s=float((a.warm + a.ticks + 5) * DT + 1.0), rest_terminate=False)
  env_args = types.SimpleNamespace(task=TASK, hip="sonic", num_envs=min(a.num_envs, E),
                                   device=dev)
  env = make_env(env_args, overrides, None)
  mj = env.mj
  if abs(float(mj.step_dt) - DT) > 1e-9:
    raise SystemExit(f"[bench] control step {mj.step_dt} s, SONIC needs {DT} s")
  spawn = eq._spawn_term(mj)
  seq = getattr(spawn, "_seq_rows", None)
  if (spawn.sampler != "sequential" or seq is None or int(seq.numel()) != E
          or not bool((seq.cpu() == torch.arange(E)).all())):
    raise SystemExit("[bench] the spawn does not walk the episode bank row by row")
  sonic = SonicPolicy.from_onnx(a.sonic_dir).to(dev).eval()
  tracker = SonicTracker(env, sonic, base)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  # the bench's actor group has no last-action term: nothing to clamp
  layout = env._obs_layout(env.obs_key)
  if not any(getattr(cfg.func, "__name__", "") == "last_action" for _s, _z, cfg in layout.values()):
    a.pi_obs_clamp = "none"
  ev = eq.Evaluator(env, tracker, base, model, norm, {}, spawn, a)
  rollout_meta = wm_meta = None
  if any(c.kind == "rollout" for c in a.conds):
    rollout_meta = ev.setup_rollouts(margin="k", k_joint_scale=pb.JOINT_SCALE)
  if any(c.kind in eq.WM_KINDS for c in a.conds):
    wm_meta = ev.setup_world_model()
  if any(c.kind in ("wmtube", "wmmin") for c in a.conds):
    if a.tube_calib is None:
      raise SystemExit("[bench] wmtube / wmmin need --tube-calib")
    with open(a.tube_calib) as f:
      ev.tube_calib = json.load(f)
    if ev.tube_calib.get("tag") != a.tag:
      raise SystemExit(f"[bench] {a.tube_calib} was calibrated for {ev.tube_calib.get('tag')!r}, "
                       f"not {a.tag!r}")
  runner = BenchRunner(ev, a, cols, clip)
  print(f"[bench] env {env.num_envs} slots, mass {runner.mass:.3f} kg, actor obs "
        f"{env.observation_space.shape[0]}-D, pi_obs_clamp {a.pi_obs_clamp}; setup "
        f"{time.time() - t_start:.0f}s", flush=True)
  meta = {"source": "eval_push_bench.py", "table": os.path.abspath(a.table), "task": TASK,
          "policy": os.path.abspath(a.policy), "tag": a.tag, "library": os.path.abspath(a.library),
          "sonic_dir": os.path.abspath(a.sonic_dir), "dt": DT, "warm_ticks": a.warm,
          "scored_ticks": a.ticks, "push_ticks": pb.PUSH_TICKS, "mass_kg": runner.mass,
          "conditions": [c.name for c in a.conds], "num_envs": env.num_envs, "seed": a.seed,
          "sample": a.sample, "limit": a.limit, "controls": not a.no_controls,
          "rollout": rollout_meta, "wm": wm_meta,
          "tube_calib": os.path.abspath(a.tube_calib) if a.tube_calib else None,
          "categories": list(lib["meta"].get("categories", [])),
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  recs, runs, summary = {}, {}, {}
  ref = None
  runner.warmup()
  meta["warm_fail"] = int(runner.warm_fail.sum())
  for cond in a.conds:
    rec = runner.run(cond)
    runs[cond.name] = rec.pop("_run")
    recs[cond.name] = rec
    if cond.name == ql.REFERENCE:
      ref = rec
    if cond.kind == "wmcalib":
      continue
    s = summarize(rec, None if cond.name == ql.REFERENCE else ref, cols)
    summary[cond.name] = s
    his_npz(os.path.join(a.out, cond.name.replace(":", "_").replace("+", "_"), "trials.npz"),
            ref, rec, cols)
    p = s["pushed"]
    print(f"[bench] {cond.name}: survival {p['sr']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
          f"(n {p['n']}, warm fails {s['warm_fail']}); "
          + (f"rescued {s['paired']['rescued']} / harmed {s['paired']['harmed']}; "
             if "paired" in s else "")
          + (f"no-push triggered {s['no_push']['triggered']:.3f}; " if "no_push" in s else "")
          + f"policy share {s['policy_share']:.3f} ({runs[cond.name]['wall_s']:.0f}s)", flush=True)
  if a.repeat_sonic and ref is not None:
    # GPU contact accumulation is not bit-deterministic: the same trials, rerun
    noise = []
    for i in range(a.repeat_sonic):
      rep = runner.run(a.conds[0])
      rep.pop("_run")
      ok = ~(ref["warm_fail"] | rep["warm_fail"])
      s0, s1 = ref["exit_tick"] < 0, rep["exit_tick"] < 0
      noise.append({"trials": int(ok.sum()), "survival_flips": int((ok & (s0 != s1)).sum()),
                    "exit_tick_changes": int((ok & (ref["exit_tick"] != rep["exit_tick"])).sum()),
                    "warm_fail_changes": int((ref["warm_fail"] != rep["warm_fail"]).sum())})
      print(f"[bench] sonic rerun {i + 1}: {noise[-1]}", flush=True)
    meta["sonic_repeat_noise"] = noise
  meta["runs"] = runs
  episodes = {"dv": cols["dv"], "context_id": cols["context_id"], "start_frame": cols["start_frame"],
              "site": cols["site"], "angle": cols["angle"], "actual": cols["actual"],
              "trial_id": cols["trial_id"], "row": rows,
              "category": lib["category"][rows]}
  torch.save({"meta": meta, "trials": dict(episodes, motion=cols["motion"],
                                            direction=cols["direction"]),
              "episodes": episodes, "conditions": recs}, os.path.join(a.out, "bench.pt"))
  with open(os.path.join(a.out, "summary.json"), "w") as f:
    json.dump(ql.clean_json({"meta": meta, "conditions": summary}), f, indent=1)
  print(f"[bench] wrote {a.out} ({time.time() - t_start:.0f}s)")
  ev.close()
  env.close()


if __name__ == "__main__":
  main()
