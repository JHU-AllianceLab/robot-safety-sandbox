"""TrackStatsEnv: the squat-tracking task's training metrics, from its episode bus.

The task env's step event (envs/g1_track/mdp.TrackEpisodeStats) publishes
``mj._zoo_track`` every step, BEFORE the auto-reset, so done rows still hold the
episode that just ended. This wrapper folds it into device-side window
accumulators (no host sync per step) and drains them as ``track/*`` in
``metrics()``, which the PPO collector logs once per rollout (``env/track/*``).

One failure taxonomy in both worlds (AMENDMENTS C3): an episode's FIRST failure
event -- an overheat (hottest winding >= 70 C, terminal in thermal70, non-terminal
under the barrier) or a fall (tilt / non-foot contact termination). Logged per
window:

  steps, episodes          the window's env-steps and finished episodes
  fail_<c>_per_1k          first-failure events of cause c per 1000 env-steps
  fail_<c>_per_ep          finished episodes whose first failure was c, per episode
  cum_fail_<c>             cumulative first-failure events since the run started
  end_<c>_per_ep           how episodes ENDED (complete / overheat / contact / tilt /
                           other / cut): a cbf overheat event ends nothing
  any_fail_per_ep, completion, complete_frac
  start0_* / rsi_*         the same three for frame-0 and random-phase starts (C7)
  fail_early_per_1k        first-failure events within early_s of the reset, per 1000
  fail_late_per_1k         early steps / later steps (spawn artefacts vs learning, C7)
  hazard_bin<b>            first-failure events per 1000 at-risk steps in phase bin b
  err_body / err_q / err_ori, tracked_frac (err_body < tau), lost_frac (per step)
  tw_max_mean / tw_max_max, tau2_mean (per step); tw_peak / dtw_peak / heat_dose /
  tf_frac (per finished episode)
"""

from __future__ import annotations

import torch

#: the task bus's cause codes (envs/g1_track/mdp.py CAUSES); checked at bind time
CAUSES = ("complete", "overheat", "contact", "tilt", "other", "cut")
FAIL_CAUSES = ("overheat", "contact", "tilt", "other")
TF_TAU_M = 0.1


def _innermost(env):
  e, seen = env, set()
  while e is not None and id(e) not in seen:
    seen.add(id(e))
    d = getattr(e, "__dict__", {})
    nxt = d.get("venv", d.get("env"))
    if nxt is None:
      return e
    e = nxt
  return e


class TrackStatsEnv:
  """See the module docstring. Delegates everything it does not define.

  :param n_bins: phase bins of the bus (``mdp.N_PHASE_BINS``)
  :param early_s: "early" = within this long of the reset (C7)
  :param check_codes: compare :data:`CAUSES` with the task module's (imports it)
  """

  def __init__(self, venv, *, n_bins: int = 10, early_s: float = 0.5, step_dt: float = 0.02,
               check_codes: bool = True):
    self.venv = venv
    self.num_envs, self.device = int(venv.num_envs), str(venv.device)
    self.n_bins = int(n_bins)
    self.early_steps = int(round(float(early_s) / float(step_dt)))
    self._base = _innermost(venv)
    if check_codes:
      from robot_safety_sandbox.envs.g1_track import mdp
      if tuple(mdp.CAUSES) != CAUSES or int(mdp.N_PHASE_BINS) != self.n_bins:
        raise ValueError(f"track bus codes {mdp.CAUSES} / {mdp.N_PHASE_BINS} bins, this "
                         f"wrapper expects {CAUSES} / {self.n_bins}")
    self._cum = {c: 0.0 for c in FAIL_CAUSES}
    self._acc = None
    self._zero()

  def __getattr__(self, name):
    try:
      venv = self.__dict__["venv"]
    except KeyError:
      raise AttributeError(name) from None
    return getattr(venv, name)

  def _zero(self) -> None:
    dev = self.device
    keys = ["steps", "err_body", "err_q", "err_ori", "tracked", "lost", "tw_max", "tau2",
            "steps_early", "fail_early", "fail_late", "ep", "ep_completion", "ep_complete",
            "ep_anyfail", "ep_tw_peak", "ep_dtw", "ep_dose", "ep_tf", "ep0", "ep0_anyfail",
            "ep0_completion", "ep0_complete", "epr", "epr_anyfail", "epr_completion",
            "epr_complete"]
    keys += [f"ev_{c}" for c in FAIL_CAUSES] + [f"epf_{c}" for c in FAIL_CAUSES]
    keys += [f"end_{c}" for c in CAUSES]
    self._acc = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in keys}
    self._acc["tw_max_max"] = torch.full((), -1e9, dtype=torch.float64, device=dev)
    self._acc["bin_steps"] = torch.zeros(self.n_bins, dtype=torch.float64, device=dev)
    self._acc["bin_fail"] = torch.zeros(self.n_bins, dtype=torch.float64, device=dev)

  def _bus(self):
    mj = getattr(self._base, "mj", self._base)
    return getattr(mj, "_zoo_track", None)

  # --- the env API -------------------------------------------------------------

  def reset(self):
    return self.venv.reset()

  def step_tensor(self, actions):
    out = self.venv.step_tensor(actions)
    bus = self._bus()
    if bus is not None:
      self.fold(bus)
    return out

  def fold(self, bus: dict) -> None:
    """Add one step of the bus to the window (all device ops)."""
    s, d, e, A = bus["step"], bus["done"].bool(), bus["ep"], self._acc
    f64 = torch.float64
    A["steps"] += float(self.num_envs)
    A["err_body"] += s["err_body"].sum(dtype=f64)
    A["err_q"] += s["err_q"].sum(dtype=f64)
    A["err_ori"] += s["err_ori"].sum(dtype=f64)
    A["tracked"] += (s["err_body"] < TF_TAU_M).sum(dtype=f64)
    A["lost"] += s["lost"].sum(dtype=f64)
    A["tw_max"] += s["tw_max"].sum(dtype=f64)
    A["tw_max_max"] = torch.maximum(A["tw_max_max"], s["tw_max"].max().to(f64))
    A["tau2"] += s["tau2"].sum(dtype=f64)
    early = s["t"] <= self.early_steps
    A["steps_early"] += early.sum(dtype=f64)
    pb = s["phase_bin"].long().clamp(0, self.n_bins - 1)
    A["bin_steps"] += torch.zeros_like(A["bin_steps"]).scatter_add_(
      0, pb, torch.ones_like(pb, dtype=f64))
    fail = e["fail"].long()
    new = (fail != 0) & (e["fail_step"] == s["t"])      # the first failure happened now
    nf = new.to(f64)
    A["bin_fail"] += torch.zeros_like(A["bin_fail"]).scatter_add_(0, pb, nf)
    A["fail_early"] += (nf * early).sum()
    A["fail_late"] += (nf * ~early).sum()
    for code, c in enumerate(CAUSES):
      if c in FAIL_CAUSES:
        A[f"ev_{c}"] += (new & (fail == code)).sum(dtype=f64)
    # finished episodes
    df = d.to(f64)
    cause = e["cause"].long()
    rsi = e["start_rsi"].bool()
    comp = e["completion"].to(f64)
    complete = (cause == 0) & d
    anyf = e["any_fail"].bool() & d
    A["ep"] += df.sum()
    A["ep_completion"] += (comp * df).sum()
    A["ep_complete"] += complete.sum(dtype=f64)
    A["ep_anyfail"] += anyf.sum(dtype=f64)
    for code, c in enumerate(CAUSES):
      A[f"end_{c}"] += ((cause == code) & d).sum(dtype=f64)
      if c in FAIL_CAUSES:
        A[f"epf_{c}"] += ((fail == code) & d).sum(dtype=f64)
    A["ep_tw_peak"] += (e["tw_peak"].to(f64) * df).sum()
    A["ep_dtw"] += (e["dtw_peak"].to(f64) * df).sum()
    A["ep_dose"] += (e["heat_dose"].to(f64) * df).sum()
    A["ep_tf"] += (e["tf_steps"].to(f64) / e["steps"].to(f64).clamp_min(1.0) * df).sum()
    for pre, m in (("ep0", d & ~rsi), ("epr", d & rsi)):
      mf = m.to(f64)
      A[pre] += mf.sum()
      A[f"{pre}_anyfail"] += (anyf & m).sum(dtype=f64)
      A[f"{pre}_completion"] += (comp * mf).sum()
      A[f"{pre}_complete"] += (complete & m).sum(dtype=f64)

  def drain(self) -> dict[str, float]:
    """The window as ``track/*`` (one host sync), then start a new window."""
    A = {k: v.cpu() for k, v in self._acc.items()}
    self._zero()
    steps = float(A["steps"])
    if steps == 0:
      return {}

    def div(x, y):
      x, y = float(x), float(y)
      return x / y if y > 0 else float("nan")

    out = {"steps": steps, "episodes": float(A["ep"])}
    for k in ("err_body", "err_q", "err_ori"):
      out[k] = div(A[k], steps)
    out["tracked_frac"] = div(A["tracked"], steps)
    out["lost_frac"] = div(A["lost"], steps)
    out["tw_max_mean"] = div(A["tw_max"], steps)
    out["tw_max_max"] = float(A["tw_max_max"])
    out["tau2_mean"] = div(A["tau2"], steps)
    for c in FAIL_CAUSES:
      n = float(A[f"ev_{c}"])
      self._cum[c] += n
      out[f"fail_{c}_per_1k"] = 1000.0 * n / steps
      out[f"cum_fail_{c}"] = self._cum[c]
    out["fail_per_1k"] = sum(out[f"fail_{c}_per_1k"] for c in FAIL_CAUSES)
    out["cum_fail"] = sum(self._cum.values())
    out["fail_early_per_1k"] = 1000.0 * div(A["fail_early"], A["steps_early"])
    out["fail_late_per_1k"] = 1000.0 * div(A["fail_late"], steps - float(A["steps_early"]))
    for b in range(self.n_bins):
      out[f"hazard_bin{b}"] = 1000.0 * div(A["bin_fail"][b], A["bin_steps"][b])
    if float(A["ep"]) > 0:
      ep = A["ep"]
      out["any_fail_per_ep"] = div(A["ep_anyfail"], ep)
      out["completion"] = div(A["ep_completion"], ep)
      out["complete_frac"] = div(A["ep_complete"], ep)
      for c in FAIL_CAUSES:
        out[f"fail_{c}_per_ep"] = div(A[f"epf_{c}"], ep)
      for c in CAUSES:
        out[f"end_{c}_per_ep"] = div(A[f"end_{c}"], ep)
      out["tw_peak"] = div(A["ep_tw_peak"], ep)
      out["dtw_peak"] = div(A["ep_dtw"], ep)
      out["heat_dose"] = div(A["ep_dose"], ep)
      out["tf_frac"] = div(A["ep_tf"], ep)
      for pre, name in (("ep0", "start0"), ("epr", "rsi")):
        if float(A[pre]) > 0:
          out[f"{name}_episodes"] = float(A[pre])
          out[f"{name}_any_fail_per_ep"] = div(A[f"{pre}_anyfail"], A[pre])
          out[f"{name}_completion"] = div(A[f"{pre}_completion"], A[pre])
          out[f"{name}_complete_frac"] = div(A[f"{pre}_complete"], A[pre])
    return {f"track/{k}": v for k, v in out.items()}

  def metrics(self) -> dict[str, float]:
    out = dict(self.venv.metrics() or {})
    out.update(self.drain())
    return out
