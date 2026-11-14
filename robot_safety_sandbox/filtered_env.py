"""Training a task policy INSIDE a safety filter -- the PORL setting of
"Provably Optimal Reinforcement Learning under Safety Filtering" (Oh, Nguyen,
Hu, Fisac; arXiv:2510.18082): the env wrapper the filtered (treatment) arm needs.

The thing that happens here and nowhere else in the stack is the executed-action
contract.

**The executed action is the one that gets learned from.** The filter may
replace the policy's proposal with the fallback's action, and it is the
EXECUTED action that must enter the replay buffer -- storing the proposal
against the resulting transition would train the critic on a transition that
never occurred. Off-policy learners make this exactly correct with no
importance correction, which is why filtered training is run with SAC. The
wrapper publishes what it executed on ``self.executed_action``; the collector
reads it back (``safety_sb3.sac_base._collect_rollouts_tensor``) instead of
trusting the action it passed in.

**The failure/episode/margin counting is NOT here.** It lives in the base
tensor env (``MjlabTensorSafetyEnv``, drained as ``safety/*`` through
``metrics()``) so that EVERY training env -- filtered treatment arm, unfiltered
control arm, or any other run -- counts failures the same way, off one shared
accounting path. If the two arms counted failures differently the headline plot
would be measuring the instrumentation. This wrapper only ADDS
``safety/engaged_frac`` (the fraction of env-steps the fallback drove), which is
meaningful for the filtered arm alone.

The filtered arm's ``ep_len_mean`` (logged by the learner, not here) is the
experiment's sanity check: it must sit at the episode limit essentially from
step one. If it does not, the filter is not doing its job and nothing
downstream is worth reading.

:class:`WMTubeFilteredEnv` is the same contract for the world-model tube filter
(filters/wm_tube.py), which reads the env's own cached ``critic`` / ``actor``
groups and sim context rather than one normalized twin obs, and which an
ON-policy learner trains inside (PPO stores its proposal: the filter is part of
the env).
"""

from __future__ import annotations

import collections
import math
import time

import numpy as np
import torch

# obs terms the fallback twin never saw vary, and which therefore have to be
# neutralized before its critic is asked anything. See TwinObsAdapter.
_TWIN_BLIND_TERMS = ("command", "phase")


class TwinObsAdapter:
  """Present the live obs to a twin trained at a DIFFERENT command.

  The fallback here is a stabilizer: it was trained on the same cfg with the
  twist command pinned to zero, so across its whole training set the command
  channel was ``[0, 0, 0]`` and the gait-clock ``phase`` was frozen at
  ``[0, 0]`` (measured, both tasks, 2026-07-25). Feeding it the walker's live
  ``command=[1, 0, 0]`` / ``phase=[0.87, 0.50]`` would ask a critic about a
  region of its input space it has never seen and read the answer as a safety
  certificate.

  Zeroing both restores the twin's own distribution AND states the right
  question: "from this physical state, can the stabilizer still bring the robot
  to a safe stand?" -- which is what a fallback is for. It is emphatically not
  "can it track 1 m/s", which this twin cannot do at all (E053).

  The slices are DERIVED from the observation manager's term table rather than
  hardcoded, so a cfg that reorders or resizes terms cannot silently shift them.
  """

  def __init__(self, env, obs_key: str, blind_terms=_TWIN_BLIND_TERMS):
    om = env.mj.observation_manager
    names = list(om.active_terms[obs_key])
    dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[obs_key]]
    self.slices, off = [], 0
    for name, width in zip(names, dims):
      if name in blind_terms:
        self.slices.append((off, off + width))
      off += width
    missing = [t for t in blind_terms if t not in names]
    if missing:
      raise ValueError(
        f"obs group '{obs_key}' has no term(s) {missing}; the twin adapter was "
        f"written for the velocity cfg's layout {names}. Update blind_terms "
        "rather than letting the twin see a channel it never trained on.")
    self.total = off

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    for a, b in self.slices:
      out[:, a:b] = 0.0
    return out

  def __repr__(self) -> str:
    return (f"TwinObsAdapter(zeroing {self.slices} of {self.total} dims)")


class FilteredTensorEnv:
  """A ``MjlabTensorSafetyEnv`` stepped through a safety filter.

  Delegates everything it does not own, so it substitutes for the bridge
  anywhere a tensor env is expected.

  :param env: the bridge, built with ``dense_margins=True`` (the filter's
      monitor needs the task margins; the base env's safety counters read them
      too).
  :param filt: a :class:`~robot_safety_sandbox.filters.SafetyFilter`. May be
      None (a transparent pass-through -- every transition is the policy's own
      action), but the unfiltered control arm normally just uses the bare env,
      which already counts failures on its own.
  :param norm: the twin's frozen obs normalizer (from ``load_twin``); required
      when ``filt`` is given.
  :param obs_adapter: raw obs -> the obs the twin should see, applied BEFORE
      ``norm``. Defaults to :class:`TwinObsAdapter`.
  """

  def __init__(self, env, filt=None, *, norm=None, obs_adapter=None):
    if filt is not None and norm is None:
      raise ValueError("a filtered env needs the twin's obs normalizer (norm=)")
    if not getattr(env, "dense_margins", False):
      raise ValueError(
        "FilteredTensorEnv needs an env built with dense_margins=True: the "
        "monitor and the failure counters both read the task margins, and a "
        "plain cumulative env computes none.")
    self.env, self.filt, self.norm = env, filt, norm
    self.obs_adapter = (
      obs_adapter if obs_adapter is not None
      else TwinObsAdapter(env, env.obs_key)) if filt is not None else None
    self.num_envs = int(env.num_envs)
    self.device = str(env.mj.device)
    self.executed_action: torch.Tensor | None = None
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool,
                             device=self.device)
    # Engagement window (the one metric the filter adds on top of the base env's
    # safety/* counters). Reset on each metrics() drain, aligned with the base.
    self._engaged = 0.0
    self._eng_steps = 0
    if filt is not None:
      print(f"[safety-filter] filtered env: {type(filt.intervention).__name__} "
            f"over {type(filt.monitor).__name__}; twin obs via {self.obs_adapter}")
    else:
      print("[safety-filter] pass-through env: no filter")

  # --- delegation ------------------------------------------------------------

  def __getattr__(self, name):
    # __getattr__ runs only when normal lookup fails, INCLUDING during __init__
    # before self.env exists -- so miss cleanly instead of raising KeyError.
    try:
      env = self.__dict__["env"]
    except KeyError:
      raise AttributeError(name) from None
    return getattr(env, name)

  # --- engagement ------------------------------------------------------------

  def _twin_obs(self, raw: torch.Tensor) -> torch.Tensor:
    return self.norm(self.obs_adapter(raw.float()))

  # --- the env API -----------------------------------------------------------

  def reset(self) -> torch.Tensor:
    obs = self.env.reset()
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool,
                             device=self.device)
    if self.filt is not None:
      self.filt.reset(self._fresh)
    return obs

  def step_tensor(self, a_nom: torch.Tensor):
    if self.filt is None:
      action = a_nom
    else:
      raw = self.env.obs_groups()[self.env.obs_key]
      speed = torch.norm(
        self.env.mj.scene["robot"].data.root_link_lin_vel_w[:, :2], dim=1)
      action, info = self.filt(a_nom, speed=speed, fresh=self._fresh,
                               s_obs=self._twin_obs(raw))
      self._engaged += float(info.engaged.float().sum())
      self._eng_steps += self.num_envs
    # THE contract: what was executed is what the learner must store. The base
    # env (self.env.step_tensor) counts failures/episodes/margins off this same
    # executed action, so both arms share one accounting path.
    self.executed_action = action

    obs, reward, dones, timeouts, l = self.env.step_tensor(action)

    self._fresh = dones.bool()
    if self.filt is not None:
      self.filt.reset(dones.bool())
    return obs, reward, dones, timeouts, l

  def metrics(self) -> dict[str, float]:
    """The base env's metrics (incl. its safety/* counters), plus the one metric
    only a filter has: the fraction of env-steps the fallback drove."""
    out = dict(self.env.metrics() or {})
    if self.filt is not None and self._eng_steps:
      out["safety/engaged_frac"] = self._engaged / self._eng_steps
    self._engaged, self._eng_steps = 0.0, 0
    return out

  def close(self) -> None:
    self.env.close()


def unwrap_env(env, cls=None):
  """Walk a wrapper chain inward through each layer's own ``venv`` / ``env``
  attribute, read from its ``__dict__`` so ``__getattr__`` delegation cannot hop
  over a layer (TensorVecNormalize has no ``__getattr__`` at all, so code above
  it reaches a wrapper only this way). Returns the first instance of ``cls``,
  or None when the chain has none; with ``cls=None``, the innermost env."""
  e, seen = env, set()
  while e is not None and id(e) not in seen:
    if cls is not None and isinstance(e, cls):
      return e
    seen.add(id(e))
    d = getattr(e, "__dict__", {})
    nxt = d.get("venv", d.get("env"))
    if nxt is None:
      break
    e = nxt
  return None if cls is not None else e


class WMTubeFilteredEnv:
  """A tensor env stepped through the world-model tube filter
  (:func:`~robot_safety_sandbox.filters.wm_tube.wm_tube_filter`), or through
  nothing (``filt=None``: the BARE arm, a pass-through with the same interface).

  FilteredTensorEnv's contract -- ``executed_action``, ``reset(done)`` routing
  every step, ``safety/engaged_frac``, delegation of everything else -- for a
  filter whose monitor reads the env's own CACHED groups by name: the raw
  ``critic`` group (the world-model state), the ``actor`` group (the
  fallback's input) and ``sim_context()``, all as the step that produced them
  left them (no recompute: see filters/wm_tube.py).

  Per step: the proposal is clamped to the monitor's proposal box (idempotent
  after PPO's clamp to the same box), the filter decides from the state the
  action will be applied in, the env steps on the executed action, then the
  filter's per-env state is reset on ``dones``.

  Published after every step (all ``[N]`` tensors on the env device; shared,
  do not write into them):

  * ``executed_action`` -- what the env applied;
  * ``last_info`` -- this step's decision: ``engaged`` (the fallback acted;
    "would have" under ``shadow``), ``held`` (forced by the dwell), ``new_handover``,
    ``switch``, ``verdict`` (min g^ - eps; NaN where not rolled out) and
    ``pred_min`` (min g^; NaN likewise). BARE: all False / NaN;
  * ``finished`` -- the episodes that ended this step: ``done`` plus per-env
    ``steps``, ``engaged``, ``held``, ``switches``, ``handovers`` counts, valid on
    ``done`` rows (no host sync). None in BARE;
  * ``filter_obs()`` -- ``[N, 2]`` = [engaged, forced-hold seconds left] of the
    state the next obs belongs to: zero on done rows, and always zero in BARE.
    ``critic_obs()`` appends it to the env's critic group (``append_filter_state``)
    and ``critic_observation_space`` is widened by 2, in EVERY arm, so the
    learner's critic input has one width across arms: the dwell timer is hidden
    state of the filtered MDP, and the actor never sees it.

  ``metrics()`` adds ``safety/engaged_frac`` and ``filter/*``: engagement, holds,
  rollouts, hand-overs and switches per 1k steps, raw verdict statistics,
  failures split by who drove the failing step, per-episode means, the actor's
  pinned command / phase columns (must stay 0), the fallback's saturation, the
  filter's wall time, and, in arms whose hold covers the tube's horizon (dwell
  1 s = H), the COVERAGE probe.

  Coverage probe: a hand-over at step k >= 1 of an episode means the verdict at
  k-1 was rolled out and passed, so the proposal ran at k-1 and the fallback
  holds from k for >= H steps -- exactly the "proposal, then pi" path the
  verdict at k-1 predicted (the wmcalib:clean protocol, E:786-797, 875-889).
  The probe compares that prediction's min g^ with the min real g over the H
  steps from k-1 (``safety_margins()``, pre-reset): a failure inside the window
  closes it with the terminal value (calibration's forward fill), a time-out
  CENSORS it (counted, excluded). Its score r = min g^ - min g; ``cov_rate`` =
  P(r <= eps), the tube's claim, on hand-over states only (selection-biased:
  "agreement" elsewhere is not measured here).

  :param verify_cache_every: every n-th step, recompute the critic group from
      its term functions and require it equal to the cached group (debug; 0 off).
  :param shadow: decide but always execute the proposal (eval telemetry of what
      the filter would do).
  """

  def __init__(self, env, filt=None, *, actor_group: str = "actor",
               critic_group: str = "critic", append_filter_state: bool = True,
               coverage_probe: bool = True, verify_cache_every: int = 0,
               shadow: bool = False):
    self.env, self.filt = env, filt
    self.actor_group, self.critic_group = actor_group, critic_group
    self.append_filter_state = bool(append_filter_state)
    self.verify_cache_every = int(verify_cache_every)
    self.shadow = bool(shadow)
    self.num_envs = N = int(env.num_envs)
    self.device = dev = str(env.device)
    cs = getattr(env, "critic_observation_space", None)
    if cs is not None and self.append_filter_state:
      from gymnasium import spaces
      cs = spaces.Box(-np.inf, np.inf, shape=(int(np.prod(cs.shape)) + 2,), dtype=np.float32)
    self.critic_observation_space = cs
    self.executed_action: torch.Tensor | None = None
    self.finished: dict | None = None
    self._fobs = torch.zeros(N, 2, device=dev)
    no = torch.zeros(N, dtype=torch.bool, device=dev)
    nan = torch.full((N,), math.nan, device=dev)
    self._idle = {"engaged": no, "held": no, "new_handover": no, "switch": no,
                  "verdict": nan, "pred_min": nan}
    self.last_info = self._idle
    self._calls = 0
    self._probe = False
    if filt is None:
      print("[wm-tube] pass-through env (BARE): no filter"
            + ("; critic obs + 2 zero filter-state columns" if cs is not None else ""))
      return
    mon, itv = filt.monitor, filt.intervention
    self._H, self._eps, self._dt = int(mon.horizon), float(mon.eps), float(mon.b.step_dt)
    self._p_lo, self._p_hi = mon.p_lo, mon.p_hi
    self._fb_bound = float(mon.b.fallback_bound)
    # the probe needs the real margin of every step: the dense tasks' safety_margins(),
    # or the reward slot of a safety env; a plain cumulative env has none
    self._margins = bool(getattr(env, "dense_margins", False))
    has_g = self._margins or not bool(getattr(env, "dense_reward", False))
    # a hand-over must hold the fallback for >= H steps: a dwell of >= H, or a latch
    self._probe = (bool(coverage_probe) and not self.shadow and has_g
                   and (int(itv.dwell_steps) >= self._H or bool(getattr(itv, "latch", False))))
    if self.verify_cache_every and getattr(
        env.mj.observation_manager.cfg[critic_group], "enable_corruption", False):
      raise ValueError(f"verify_cache_every: the {critic_group!r} group is noisy, a recompute "
                       f"cannot equal it")
    lay = getattr(env, "_obs_layout", None)
    cols = []
    if lay is not None:
      L = lay(actor_group)
      for name in ("command", "phase"):
        if name in L:
          cols.extend(range(L[name][0], L[name][0] + L[name][1]))
    self._cmd_cols = torch.as_tensor(cols, dtype=torch.long, device=dev) if cols else None
    f64 = lambda: torch.zeros((), dtype=torch.float64, device=dev)  # noqa: E731
    self._acc_keys = ("engaged", "held", "computed", "handovers", "switches", "finite", "trig",
                      "nan", "v_sum", "pm_sum", "dclamp", "fail_eng", "fail_task", "sat",
                      "ep_n", "ep_steps", "ep_eng", "ep_sw", "ep_ho", "ep_any", "ep_frac",
                      "cov_n", "cov_hit", "cov_miss", "cov_fell", "cov_cens")
    self._acc = {k: f64() for k in self._acc_keys}
    self._acc["v_min"] = torch.full((), math.inf, dtype=torch.float64, device=dev)
    self._acc["cmdphase"] = f64()
    self._steps = 0
    # the last steps' raw verdicts / closed probe scores, for the window quantiles
    self._verdicts = collections.deque(maxlen=64)
    self._scores = collections.deque(maxlen=64)
    self._times: list = []
    self._ep = {k: torch.zeros(N, device=dev) for k in ("steps", "engaged", "held", "switches",
                                                         "handovers")}
    self._cov = {"on": no.clone(), "pred": nan.clone(), "gmin": nan.clone(),
                 "n": torch.zeros(N, dtype=torch.long, device=dev),
                 "cand_pred": nan.clone(), "cand_g": nan.clone()}
    print(f"[wm-tube] filtered env: {type(itv).__name__} over {type(mon).__name__}; reads "
          f"{critic_group!r} + {actor_group!r} + sim_context; coverage probe "
          f"{'on' if self._probe else 'off'}" + ("; SHADOW (proposals executed)"
                                                 if self.shadow else ""))

  # --- delegation ------------------------------------------------------------

  def __getattr__(self, name):
    # as FilteredTensorEnv: miss cleanly while __init__ has not set self.env yet
    try:
      env = self.__dict__["env"]
    except KeyError:
      raise AttributeError(name) from None
    return getattr(env, name)

  # --- what the learner and the stats read -----------------------------------

  def filter_obs(self) -> torch.Tensor:
    """``[N, 2]`` = [engaged, forced-hold seconds left] (class docstring)."""
    return self._fobs

  def critic_obs(self):
    c = self.env.critic_obs()
    if c is None or not self.append_filter_state:
      return c
    return torch.cat([c, self._fobs], dim=1)

  @property
  def last_engaged(self) -> torch.Tensor:
    """``last_info["engaged"]``: the fallback drove the step just executed."""
    return self.last_info["engaged"]

  # --- the env API -----------------------------------------------------------

  def reset(self) -> torch.Tensor:
    obs = self.env.reset()
    self._fobs = torch.zeros_like(self._fobs)        # a fresh tensor: callers may hold the old one
    self.last_info = self._idle
    if self.filt is not None:
      self.filt.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
      for t in self._ep.values():
        t.zero_()
      self._cov["on"].zero_()
      self._cov["cand_pred"].fill_(math.nan)
      self.finished = None
    return obs

  def step_tensor(self, a_prop: torch.Tensor):
    filt = self.filt
    if filt is None:
      self.executed_action = a_prop
      return self.env.step_tensor(a_prop)
    if self.verify_cache_every and self._calls % self.verify_cache_every == 0:
      self._verify_cache()
    timed = self._calls % 20 == 0
    self._calls += 1
    og = self.env.obs_groups()
    S = og[self.critic_group].float()
    A = torch.nan_to_num(og[self.actor_group].float(), nan=0.0, posinf=0.0, neginf=0.0)
    C = self.env.sim_context()
    a_p = torch.clamp(a_prop.float(), self._p_lo, self._p_hi)
    t0 = self._mark() if timed else None
    with torch.no_grad():
      a_f, info = filt(a_p, wm_state=S, actor_obs=A, sim_ctx=C)
    if timed:
      self._times.append((t0, self._mark()))
    il, ml = filt.intervention.last, filt.monitor.last
    action = a_p if self.shadow else a_f
    # THE contract (FilteredTensorEnv): the executed action is what the env is
    # stepped with and what the base env's failure counters see
    self.executed_action = action
    obs, reward, dones, timeouts, l = self.env.step_tensor(action)
    d = dones.bool()
    tout = timeouts.bool()
    failed = d & ~tout
    eng = info.engaged.bool()
    self._book(eng, il, ml, A, d, failed)
    if self._probe:
      g = self.env.safety_margins()[0] if self._margins else reward
      self._probe_step(il["new_handover"], ml["pred_min"], g.float(), d, failed, tout)
    filt.reset(d)
    self._fobs = torch.stack([(eng & ~d).float(),
                              filt.intervention.hold_left().float() * self._dt], dim=1)
    self.last_info = {"engaged": eng, "held": il["held"], "new_handover": il["new_handover"],
                      "switch": il["switch"], "verdict": il["verdict"],
                      "pred_min": ml["pred_min"]}
    return obs, reward, dones, timeouts, l

  # --- bookkeeping (device-side; one host sync per metrics() drain) ----------

  def _mark(self):
    if torch.device(self.device).type == "cuda":
      ev = torch.cuda.Event(enable_timing=True)
      ev.record()
      return ev
    return time.perf_counter()

  def _book(self, eng, il, ml, A, d, failed) -> None:
    acc, e = self._acc, self._ep
    held, new, sw, v = il["held"], il["new_handover"], il["switch"], il["verdict"]
    fin = torch.isfinite(v)
    self._steps += self.num_envs
    acc["engaged"] += eng.sum()
    acc["held"] += held.sum()
    acc["computed"] += ml["computed"].sum()
    acc["handovers"] += new.sum()
    acc["switches"] += sw.sum()
    acc["finite"] += fin.sum()
    acc["trig"] += (fin & (v < 0.0)).sum()
    acc["nan"] += (ml["computed"] & torch.isnan(v)).sum()
    acc["v_sum"] += torch.where(fin, v, 0.0).sum()
    acc["pm_sum"] += torch.where(fin, ml["pred_min"], 0.0).sum()
    acc["v_min"] = torch.minimum(acc["v_min"], torch.where(fin, v, math.inf).amin().double())
    acc["dclamp"] += ml["done_clamped"].sum()
    acc["fail_eng"] += (failed & eng).sum()
    acc["fail_task"] += (failed & ~eng).sum()
    acc["sat"] += ((il["a_safe"].abs() >= self._fb_bound) & eng.unsqueeze(-1)).sum()
    if self._cmd_cols is not None:
      acc["cmdphase"] = torch.maximum(acc["cmdphase"],
                                      A.index_select(1, self._cmd_cols).abs().amax().double())
    self._verdicts.append(v)
    e["steps"] += 1.0
    e["engaged"] += eng.float()
    e["held"] += held.float()
    e["switches"] += sw.float()
    e["handovers"] += new.float()
    self.finished = {"done": d, **{k: t.clone() for k, t in e.items()}}
    df = d.double()
    acc["ep_n"] += df.sum()
    acc["ep_steps"] += (df * e["steps"]).sum()
    acc["ep_eng"] += (df * e["engaged"]).sum()
    acc["ep_sw"] += (df * e["switches"]).sum()
    acc["ep_ho"] += (df * e["handovers"]).sum()
    acc["ep_any"] += (d & (e["engaged"] > 0)).sum()
    acc["ep_frac"] += (df * e["engaged"] / e["steps"].clamp_min(1.0)).sum()
    keep = (~d).float()
    for t in e.values():
      t.mul_(keep)

  def _probe_step(self, new, pred_min, g, d, failed, tout) -> None:
    p, acc, H = self._cov, self._acc, self._H
    # open at a hand-over whose previous step (same episode) passed a rolled-out verdict
    opn = new & torch.isfinite(p["cand_pred"]) & ~p["on"]
    p["pred"] = torch.where(opn, p["cand_pred"], p["pred"])
    p["gmin"] = torch.where(opn, p["cand_g"], p["gmin"])
    p["n"] = torch.where(opn, torch.ones_like(p["n"]), p["n"])
    on = p["on"] | opn
    p["gmin"] = torch.where(on, torch.minimum(p["gmin"], g), p["gmin"])
    p["n"] = p["n"] + on.long()
    full = on & (p["n"] >= H)
    fell = on & failed & ~full        # forward fill: the terminal g stands for the rest
    cens = on & tout & ~full          # the env cut the episode inside the window
    close = full | fell
    r = p["pred"] - p["gmin"]
    acc["cov_n"] += close.sum()
    acc["cov_hit"] += (close & (r <= self._eps)).sum()
    acc["cov_miss"] += (close & (p["gmin"] < 0.0)).sum()
    acc["cov_fell"] += fell.sum()
    acc["cov_cens"] += cens.sum()
    self._scores.append(torch.where(close, r, math.nan))
    p["on"] = on & ~(close | cens | d)
    # the candidate of the next step: this step's prediction and real margin
    p["cand_pred"] = torch.where(d, torch.full_like(pred_min, math.nan), pred_min)
    p["cand_g"] = g

  def _verify_cache(self) -> None:
    """The cached critic group equals a recompute from its term functions
    (mjlab's compute_group arithmetic, observation_manager.py:345-357; not
    compute_group itself, which returns the stored frame at history 1)."""
    mj = self.env.mj
    om = mj.observation_manager
    parts = []
    for cfg in om._group_obs_term_cfgs[self.critic_group]:
      x = cfg.func(mj, **cfg.params).clone()
      if cfg.clip:
        x = x.clip(min=cfg.clip[0], max=cfg.clip[1])
      if cfg.scale is not None:
        x = x * cfg.scale
      parts.append(x.reshape(x.shape[0], -1))
    cached = self.env.obs_groups()[self.critic_group]
    rec = torch.cat(parts, dim=-1)
    if not torch.equal(rec, cached):
      bad = (rec != cached).any(0).nonzero().flatten().tolist()
      raise RuntimeError(f"[wm-tube] cached {self.critic_group!r} group differs from a "
                         f"recompute in columns {bad[:20]}: something wrote sim state between "
                         f"the env step and the filter")

  def _drain(self) -> dict[str, float]:
    keys = self._acc_keys + ("v_min", "cmdphase")
    vals = dict(zip(keys, torch.stack([self._acc[k] for k in keys]).tolist()))
    n = float(self._steps)
    out = {}
    if n:
      out["safety/engaged_frac"] = vals["engaged"] / n
      out["filter/engaged_frac"] = vals["engaged"] / n
      out["filter/hold_frac"] = vals["held"] / n
      out["filter/rollout_frac"] = vals["computed"] / n
      out["filter/handovers_per_1k"] = 1000.0 * vals["handovers"] / n
      out["filter/switches_per_1k"] = 1000.0 * vals["switches"] / n
      out["filter/fail_while_engaged"] = vals["fail_eng"]
      out["filter/fail_while_task"] = vals["fail_task"]
      out["filter/nan_verdicts"] = vals["nan"]
      if self._cmd_cols is not None:
        out["filter/cmd_phase_absmax"] = vals["cmdphase"]
    if vals["engaged"]:
      out["filter/hold_share"] = vals["held"] / vals["engaged"]
      out["filter/fallback_sat_frac"] = vals["sat"] / (vals["engaged"] * self._p_lo.numel())
    if vals["finite"]:
      out["filter/raw_trigger_frac"] = vals["trig"] / vals["finite"]
      out["filter/verdict_mean"] = vals["v_sum"] / vals["finite"]
      out["filter/verdict_min"] = vals["v_min"]
      out["filter/pred_min_mean"] = vals["pm_sum"] / vals["finite"]
    if vals["computed"]:
      out["filter/done_clamp_frac"] = vals["dclamp"] / vals["computed"]
    if self._verdicts:
      out["filter/verdict_p05"] = float(torch.nanquantile(torch.cat(list(self._verdicts)), 0.05))
    if vals["ep_n"]:
      m = vals["ep_n"]
      out["filter/episodes"] = m
      out["filter/engaged_steps_per_ep"] = vals["ep_eng"] / m
      out["filter/switches_per_ep"] = vals["ep_sw"] / m
      out["filter/handovers_per_ep"] = vals["ep_ho"] / m
      out["filter/ep_engaged_frac"] = vals["ep_frac"] / m
      out["filter/ep_any_engaged"] = vals["ep_any"] / m
      out["filter/ep_len_mean"] = vals["ep_steps"] / m
    if self._probe:
      out["filter/cov_n"] = vals["cov_n"]
      out["filter/cov_fell"] = vals["cov_fell"]
      out["filter/cov_censored"] = vals["cov_cens"]
      if vals["cov_n"]:
        out["filter/cov_rate"] = vals["cov_hit"] / vals["cov_n"]
        out["filter/cov_miss_rate"] = vals["cov_miss"] / vals["cov_n"]
        sc = torch.cat(list(self._scores))
        if bool(torch.isfinite(sc).any()):
          out["filter/cov_score_p95"] = float(torch.nanquantile(sc, 0.95))
    if self._times:
      if isinstance(self._times[0][0], float):
        ms = [1e3 * (b - a) for a, b in self._times]
      else:
        self._times[-1][1].synchronize()
        ms = [a.elapsed_time(b) for a, b in self._times]
      out["filter/ms_per_step"] = sum(ms) / len(ms)
    for t in self._acc.values():
      t.zero_()
    self._acc["v_min"].fill_(math.inf)
    self._steps = 0
    self._verdicts.clear()
    self._scores.clear()
    self._times = []
    return out

  def metrics(self) -> dict[str, float]:
    """The env's metrics (its safety/* counters included), plus the filter's
    window (class docstring); BARE adds nothing."""
    out = dict(self.env.metrics() or {})
    if self.filt is not None:
      out.update(self._drain())
    return out

  def close(self) -> None:
    self.env.close()
