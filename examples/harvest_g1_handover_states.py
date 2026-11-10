"""Harvest P-pre-nominal start states: a replayed clip, a push, a handover.

    python examples/harvest_g1_handover_states.py \
        --library data/g1_lafan1_clips_train.pt \
        --policy runs/g1_recover_push_fast_privcrit/g1_recover_push_fast/final_model.zip \
        --samples 100000 --out data/g1_handover_train.pt

A safety filter does not hand over in a quiet state. A NOMINAL controller is
running a motion, a push lands, and the filter switches to the fallback some
time into the push -- possibly while it is still acting. These are the starts
the fallback has to recover from in deployment, and neither the walker bank
nor a kinematic mocap frame is one. This script makes them, per env, per
episode, in the ``g1_recover_lafan`` env spawning on the clip LIBRARY
(``build_g1_recover_mocap_states.py --library``, every 50 fps frame; the
spawn's ``mocap`` source only, ``sampler="uniform"``):

1. REPLAY. The episode starts on a library frame r; the nominal is the clip
   itself, replayed open loop as PD targets (``envs/g1_recover/playback.py``:
   frames r+1, r+2, ..., one per control step).
2. PUSH. ``onset`` (U(--onset-s) s) into the replay a scripted horizontal push
   lands (``RandomPush.schedule``, the bridge's torso push channel): impulse
   J ~ U(--impulse-range) N*s over a duration T drawn from --durations, i.e.
   a constant J / T newtons for round(T / dt) control steps, at a direction
   drawn uniformly in the ROBOT's heading frame at the onset (``rel`` in
   [-pi, pi): 0 = from behind). No random push is ever drawn in this env
   (``max_pushes=0``); the scripted one is the only push.
3. HANDOVER. Under ``--handover-rule delay`` the replay stops a delay
   U(--handover-s) s after the onset; under ``monitor`` it stops at the first
   state, from the onset on, where the monitor's value V_monitor < --monitor-
   delta (``--monitor-policy``, default --policy) -- the switch a real
   value-based filter makes -- or at the --handover-s maximum if it never
   drops. The state at that instant is the row (``states.snapshot``: its
   ``action`` is the last replayed PD target, which the spawn seeds as
   ``last_action``, so a restored row continues exactly as the handover did).
4. LABEL. The fallback (--policy, deterministic) takes over in the same env
   for --window-s; the row is ``fell`` (with ``time_to_fall`` from the
   handover) if the episode ends on a fall, survived otherwise. ``--no-label``
   stops at the snapshot. Either way the episode is then ended (its
   ``episode_length_buf`` is set to the cap, so the next step truncates it
   through the env's own ``time_out``) and the env starts the next one.

THE PENDING PUSH. A handover can land mid-push. With t = control steps
replayed, ``onset`` and ``D`` = round(T / dt) in steps:

* t == onset  -- the push has NOT started: it acts on the very next step,
  the first one of the fallback's episode (schedule's onset semantics). It is
  pending in full: ``push_left_steps = D`` at J / T newtons, direction
  ``rel`` resolved against the robot's heading at that instant;
* onset < t < onset + D -- it is acting: ``push_left_steps`` = the push's own
  ``t_left`` (= onset + D - t, checked), force = the force it applied at the
  last step (``RandomPush.cur_force_w``);
* t >= onset + D -- it is over: no pending push (0, zero force).

The force is stored in the row's HEADING frame (``push_force_h``: rotated by
minus the yaw of the snapshot's pelvis, the frame of ``lin_vel_h``), so the
spawn rotates it by its own fresh yaw together with the twist and the push
keeps its angle to the body; the bridge then resumes it on the episode's
first step for its remaining steps, and the row's ``KIND_HANDOVER`` blocks
any random push after it. ``handover_delay_s`` = (t - onset) dt;
``push_impulse`` = J (the whole push, pending part included);
``push_severity`` = J / the top of --impulse-range.

Replays that FALL before the handover (a fast clip drifts open loop: nothing
closes the loop on the base) are counted, per category, and discarded: that
is the replay failing, not the fallback. The library's ``start_ok`` already
guarantees --min-follow frames after each spawnable row (default 25 = 0.5 s,
the default onset + handover maximum); a longer prefix is clamped to the clip
(delay first, then onset; counted).

Rows: ``states.FIELDS`` (``kind = KIND_HANDOVER``, command / phase_step 0,
``difficulty`` 0, ``env_id``) plus ``clip`` / ``category`` (indices into the
library's names, copied into this file's meta), ``frame`` / ``t`` (the
library frame and clip time at the handover), ``push_force_h``,
``push_left_steps``, ``push_impulse``, ``handover_delay_s``, ``value0`` (the
fallback's value at the handover, asymmetric-critic aware), and for analysis
``push_duration_s``, ``push_onset_s``, ``push_rel``, ``src_row`` (the library
row the episode spawned on) and ``monitor_value`` (monitor rule). The training
spawn draws them as its ``handover`` source; ``label_start_states.py`` gives
them K-rollout labels (``<out>_labeled``). The report: survival by impulse,
by handover delay, by duration and pending-or-not, per category, and the
prefix-fall rate.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat  # noqa: E402

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.labels import PolicyValue, auc  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.playback import PlaybackDriver  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_HANDOVER, KIND_MOCAP, load_bank, save_bank, snapshot, speed_of)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402

PLAY, WINDOW, WAIT = 0, 1, 2


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_train.pt"),
                 help="the clip library (build_g1_recover_mocap_states.py --library)")
  p.add_argument("--policy", default=None,
                 help="the fallback that takes over and labels the row (and "
                      "gives value0)")
  p.add_argument("--no-label", action="store_true",
                 help="stop at the snapshot (fell False, time_to_fall NaN)")
  p.add_argument("--out", default=os.path.join(_ZOO, "data", "g1_handover_train.pt"))
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--samples", type=int, default=100000)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--impulse-range", type=float, nargs=2, default=[4.0, 48.0],
                 help="push impulse J, N*s (uniform)")
  p.add_argument("--durations", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                 help="push durations T, s (one drawn per push; force = J / T)")
  p.add_argument("--onset-s", type=float, nargs=2, default=[0.02, 0.1],
                 help="push onset after the replay starts, s (uniform; at least "
                      "one control step)")
  p.add_argument("--handover-s", type=float, nargs=2, default=[0.0, 0.2],
                 help="handover delay after the push onset, s (uniform; the "
                      "monitor rule's maximum is the upper end)")
  p.add_argument("--handover-rule", choices=("delay", "monitor"), default="delay")
  p.add_argument("--monitor-policy", default=None,
                 help="the monitor's checkpoint (default: --policy)")
  p.add_argument("--monitor-delta", type=float, default=0.1,
                 help="monitor rule: hand over at the first V_monitor < delta")
  p.add_argument("--window-s", type=float, default=2.0,
                 help="label window after the handover, s")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args(argv)
  if not args.no_label and not args.policy:
    p.error("--policy is required unless --no-label")
  if args.handover_rule == "monitor" and not (args.monitor_policy or args.policy):
    p.error("--handover-rule monitor needs --monitor-policy (or --policy)")
  for name in ("impulse_range", "onset_s", "handover_s"):
    lo, hi = getattr(args, name)
    if not 0.0 <= lo <= hi:
      p.error(f"--{name.replace('_', '-')} needs 0 <= lo <= hi")
  if args.impulse_range[1] <= 0.0 or min(args.durations) <= 0.0:
    p.error("impulses and durations must be > 0")
  return args


class Harvest:
  """Per-env episode bookkeeping of the harvest (module docstring)."""

  def __init__(self, env, lib: dict, args, policy=None, value=None, monitor=None):
    self.env, self.mj, self.args = env, env.mj, args
    self.dev, self.n = env.device, env.num_envs
    self.dt = float(self.mj.step_dt)
    self.push = getattr(env, "_push", None)
    if self.push is None or not hasattr(self.push, "schedule"):
      raise SystemExit(f"task {args.task!r} has no scriptable push process")
    if not hasattr(self.mj, "_zoo_spawn_row"):
      raise SystemExit(f"task {args.task!r}: its spawn publishes no _zoo_spawn_row")
    self.policy, self.value, self.monitor = policy, value, monitor
    dev, n = self.dev, self.n
    self.driver = PlaybackDriver(lib, n, dev)
    self.lib = {k: lib[k].to(dev) for k in ("clip", "category", "frame", "t")}
    self.robot = self.mj.scene["robot"]

    def steps(s):
      return int(round(float(s) / self.dt))

    self.dl_hi = steps(args.handover_s[1])          # the monitor rule's latest handover
    self.dur =torch.tensor([max(1, steps(s)) for s in args.durations], device=dev)
    long = dict(dtype=torch.long, device=dev)
    self.mode = torch.zeros(n, **long)
    self.row0 = torch.full((n,), -1, **long)
    self.on = torch.zeros(n, **long)
    self.D = torch.ones(n, **long)
    self.P = torch.zeros(n, **long)
    self.J = torch.zeros(n, device=dev)
    self.F = torch.zeros(n, device=dev)
    self.rel = torch.zeros(n, device=dev)
    self.count = torch.zeros(n, **long)
    self.since = torch.zeros(n, **long)
    self.pend: dict[str, torch.Tensor] = {}
    self.rows: dict[str, list] = {}
    self.got = 0
    self.started = 0
    ncat = len(lib["meta"].get("categories", [])) + 1
    self.ep_cat = torch.zeros(ncat, **long)       # episodes started, per category
    self.prefix_cat = torch.zeros(ncat, **long)   # of them fell during the replay
    self.clamped = 0
    self.cut = 0
    self.mismatch = 0

  def _cat_bucket(self, rows: torch.Tensor) -> torch.Tensor:
    """The category of library rows; -1 (no category) -> the last bucket."""
    cat = self.lib["category"][rows]
    return torch.where(cat < 0, torch.full_like(cat, self.ep_cat.numel() - 1), cat)

  # --- a new episode: draw the push and the handover, start the replay -----
  def begin(self, mask: torch.Tensor) -> None:
    k = int(mask.sum())
    if k == 0:
      return
    a, dev, dt = self.args, self.dev, self.dt
    rows = self.mj._zoo_spawn_row[mask].clone()
    if bool((rows < 0).any()):
      raise SystemExit("an env spawned in the nominal stance: the library has no "
                       "start_ok mocap rows, or the spawn ignored weights/sampler")
    jlo, jhi = a.impulse_range
    J = jlo + (jhi - jlo) * torch.rand(k, device=dev)
    D = self.dur[torch.randint(0, self.dur.numel(), (k,), device=dev)]
    rel = (torch.rand(k, device=dev) * 2.0 - 1.0) * math.pi
    on_s = a.onset_s[0] + (a.onset_s[1] - a.onset_s[0]) * torch.rand(k, device=dev)
    on = torch.round(on_s / dt).long().clamp_min(1)
    if a.handover_rule == "monitor":
      delay = torch.full((k,), self.dl_hi, dtype=torch.long, device=dev)
    else:
      d_s = a.handover_s[0] + (a.handover_s[1] - a.handover_s[0]) * torch.rand(k, device=dev)
      delay = torch.round(d_s / dt).long().clamp_min(0)
    # the replay must stay inside its clip: shorten the delay, then the onset
    room = self.driver.next_ok[rows]
    P = torch.minimum(on + delay, room)
    self.clamped += int((P < on + delay).sum())
    on = torch.minimum(on, P)
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    self.row0[ids], self.on[ids], self.D[ids], self.P[ids] = rows, on, D, P
    self.J[ids], self.F[ids], self.rel[ids] = J, J / (D.float() * dt), rel
    self.mode[ids] = PLAY
    self.driver.start(mask, self.row0, self.P)
    self.push.schedule(mask, force_n=self.F, steps=self.D, onset=self.on, rel=self.rel)
    cat = self._cat_bucket(rows)
    self.ep_cat.index_add_(0, cat, torch.ones_like(cat))
    self.started += k

  # --- the handover ----------------------------------------------------------
  def handover(self, mask: torch.Tensor, obs: torch.Tensor,
               mon_val: torch.Tensor | None) -> None:
    dev, dt = self.dev, self.dt
    s = snapshot(self.mj)
    d = self.robot.data
    t_play = self.driver.k.clone()
    started = t_play > self.on
    left_book = (self.on + self.D - t_play).clamp(0, None)
    in_prog = started & (left_book > 0)
    t_left = self.push.t_left
    self.mismatch += int((mask & in_prog & (t_left != left_book)).sum())
    pend_steps = torch.where(started, torch.where(in_prog, t_left, 0), self.D)
    h = d.heading_w.reshape(-1)
    ang = self.rel + h
    f_next = self.F.unsqueeze(1) * torch.stack(
      [torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=1)
    force_w = torch.where(in_prog.unsqueeze(1), self.push.cur_force_w,
                          torch.where(started.unsqueeze(1), torch.zeros_like(f_next),
                                      f_next))
    force_w = torch.where((pend_steps > 0).unsqueeze(1), force_w, torch.zeros_like(force_w))
    force_h = quat_apply_inverse(yaw_quat(d.root_link_quat_w), force_w)
    if self.value is not None:
      crit = self.env.critic_obs() if self.value.asymmetric else None
      v0 = self.value(obs, crit)
    else:
      v0 = torch.full((self.n,), float("nan"), device=dev)
    lib = self.lib
    extra = {
      "push_force_h": force_h, "push_left_steps": pend_steps,
      "push_impulse": self.J.clone(),
      "handover_delay_s": (t_play - self.on).float() * dt,
      "value0": v0,
      "clip": lib["clip"][self.row0], "category": lib["category"][self.row0],
      "frame": lib["frame"][self.row0] + t_play,
      "t": lib["t"][self.row0] + t_play.float() * dt,
      "push_duration_s": self.D.float() * dt, "push_onset_s": self.on.float() * dt,
      "push_rel": self.rel.clone(), "src_row": self.row0.clone(),
      "monitor_value": (mon_val if mon_val is not None
                        else torch.full((self.n,), float("nan"), device=dev)),
    }
    s["command"] = torch.zeros(self.n, 3, device=dev)
    s["phase_step"] = torch.zeros(self.n, dtype=torch.long, device=dev)
    for key, v in list(s.items()) + list(extra.items()):
      if key not in self.pend:
        self.pend[key] = torch.zeros_like(v)
      self.pend[key][mask] = v[mask]
    self.since[mask] = 0
    if self.args.no_label:
      self.emit(mask, torch.zeros(self.n, dtype=torch.bool, device=dev))
      self.finish(mask)
    else:
      self.mode[mask] = WINDOW
      self.count[mask] = max(1, int(round(self.args.window_s / dt)))

  def emit(self, mask: torch.Tensor, fell: torch.Tensor) -> None:
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    m = int(idx.numel())
    if m == 0:
      return
    dev, dt = self.dev, self.dt
    out = {k: v[idx] for k, v in self.pend.items()}
    out["kind"] = torch.full((m,), KIND_HANDOVER, dtype=torch.long, device=dev)
    out["fell"] = fell[idx]
    out["time_to_fall"] = torch.where(
      fell[idx], self.since[idx].float() * dt,
      torch.full((m,), float("nan"), device=dev))
    out["push_severity"] = out["push_impulse"] / float(self.args.impulse_range[1])
    out["difficulty"] = torch.zeros(m, device=dev)
    out["speed"] = speed_of(out["lin_vel_h"])
    out["env_id"] = idx.clone()
    for k, v in out.items():
      self.rows.setdefault(k, []).append(v)
    self.got += m

  def finish(self, mask: torch.Tensor) -> None:
    """End these episodes on the next step (a truncation through the env's own
    time_out term), so the env moves on to a new start."""
    self.mode[mask] = WAIT
    buf = self.mj.episode_length_buf
    buf[mask] = torch.clamp_min(buf[mask], int(self.mj.max_episode_length) - 1)

  def begin_capped(self, mask: torch.Tensor) -> None:
    """:meth:`begin` for at most as many of ``mask`` as the bank still needs.

    A fall is labeled within a few tenths of a second, a survivor only at the
    end of its window, so stopping the moment ``samples`` rows exist would drop
    the captures still in flight -- nearly all survivors -- and bias the bank
    toward falls (with ``samples`` close to ``num_envs``, to all falls). So a
    capture starts only while labeled + in-flight rows fall short of
    ``samples``, and the loop runs until exactly ``samples`` are labeled;
    replays lost before their handover free their slot for a new start. The
    envs left out idle under the fallback until they reset."""
    inflight = int((((self.mode == PLAY) | (self.mode == WINDOW)) & ~mask).sum())
    need = self.args.samples - self.got - inflight
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    take = torch.zeros_like(mask)
    if need > 0:
      take[ids[:need]] = True
    self.mode[mask & ~take] = WAIT
    self.begin(take)

  # --- the loop ----------------------------------------------------------------
  def run(self) -> dict:
    a, dev, n = self.args, self.dev, self.n
    env = self.env
    low = torch.as_tensor(env.action_space.low, device=dev)
    high = torch.as_tensor(env.action_space.high, device=dev)
    obs = env.reset()
    self.begin_capped(torch.ones(n, dtype=torch.bool, device=dev))
    t0, it = time.time(), 0
    zeros = torch.zeros(n, dtype=torch.bool, device=dev)
    while self.got < a.samples:
      # 1. handovers due at the current state
      play = self.mode == PLAY
      due = play & ~self.driver.active               # the replay just ended
      mon_val = None
      if self.monitor is not None:
        cand = play & self.driver.active & (self.driver.k >= self.on)
        if bool(cand.any()):
          crit = env.critic_obs() if self.monitor.asymmetric else None
          mon_val = self.monitor(obs, crit)
          trig = cand & (mon_val < a.monitor_delta)
          self.driver.stop(trig)
          due = due | trig
      if bool(due.any()):
        self.handover(due, obs, mon_val)
      # 2. act: the replay where it runs, the fallback elsewhere
      if self.policy is not None:
        model, norm = self.policy
        # the fallback's memory starts at its handover: held fresh while the
        # replay drives the env (a no-op for a memoryless policy)
        reset_policy_memory(model.policy, self.mode != WINDOW)
        with torch.no_grad():
          pol = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
      else:
        pol = None
      obs, _g, dones, timeouts, _l = env.step_tensor(self.driver.actions(pol))
      self.driver.advance()
      it += 1
      # 3. labels
      win = self.mode == WINDOW
      self.since += win.long()
      self.count -= win.long()
      fell = dones & ~timeouts
      self.emit(win & fell, fell)                          # fell in the window
      cut = win & dones & ~fell                            # the cap: no label
      self.cut += int(cut.sum())
      up = win & ~dones & (self.count <= 0)                # upright at the end
      self.emit(up, zeros)
      self.finish(up)
      # 4. replays that fell before the handover, then the new episodes
      lost = (self.mode == PLAY) & dones
      if bool(lost.any()):
        cat = self._cat_bucket(self.row0[lost])
        self.prefix_cat.index_add_(0, cat, torch.ones_like(cat))
      if bool(dones.any()):
        self.driver.stop(dones)
        self.begin_capped(dones)
      if it % 250 == 0:
        el = max(time.time() - t0, 1e-6)
        pf = int(self.prefix_cat.sum()) / max(self.started, 1)
        print(f"[handover] step {it}  rows {self.got}/{a.samples}  episodes "
              f"{self.started}  prefix falls {pf:.3f}  ({it * n / el:.0f} env-steps/s)",
              flush=True)
    return {k: torch.cat(v)[:a.samples] for k, v in self.rows.items()}


def _report(bank: dict, h: Harvest, lib_meta: dict, args) -> dict:
  fell = bank["fell"]
  stats: dict = {"rows": int(fell.numel()), "episodes_started": h.started,
                 "prefix_falls": int(h.prefix_cat.sum()),
                 "prefix_fall_rate": int(h.prefix_cat.sum()) / max(h.started, 1),
                 "window_cut_by_cap": h.cut, "prefix_clamped_to_clip": h.clamped,
                 "pending_push_mismatch": h.mismatch}
  print(f"[handover] {stats['rows']} rows from {h.started} episodes; replays that "
        f"fell before the handover: {stats['prefix_falls']} "
        f"({100 * stats['prefix_fall_rate']:.1f}%)")
  if h.mismatch:
    print(f"[handover] WARNING: {h.mismatch} handovers whose push t_left disagreed "
          f"with the schedule (the push's own count was stored)")
  if args.no_label:
    return stats
  surv = ~fell

  def table(name, key, edges, unit, fmt="{:.2f}"):
    x = bank[key]
    out = {}
    print(f"[handover] survival by {name}:")
    for lo, hi in zip(edges[:-1], edges[1:]):
      m = (x >= lo) & ((x < hi) if hi != edges[-1] else (x <= hi))
      if int(m.sum()):
        s = float(surv[m].float().mean())
        out[f"{fmt.format(lo)}_{fmt.format(hi)}"] = s
        print(f"    {fmt.format(lo)}-{fmt.format(hi)} {unit}: {int(m.sum()):7d} rows, "
              f"survival {s:.3f}")
    return out

  jlo, jhi = args.impulse_range
  stats["survival"] = float(surv.float().mean())
  stats["by_impulse"] = table("impulse", "push_impulse",
                              torch.linspace(jlo, jhi, 5).tolist(), "N*s", "{:.1f}")
  hlo, hhi = args.handover_s
  stats["by_delay"] = table("handover delay", "handover_delay_s",
                            torch.linspace(hlo, max(hhi, hlo + 1e-6), 5).tolist(), "s")
  by_dur = {}
  for T in sorted(set(bank["push_duration_s"].round(decimals=3).tolist())):
    m = (bank["push_duration_s"] - T).abs() < 1e-3
    by_dur[f"{T:.2f}"] = float(surv[m].float().mean())
  stats["by_duration"] = by_dur
  pend = bank["push_left_steps"] > 0
  stats["survival_pending_push"] = float(surv[pend].float().mean()) if bool(pend.any()) else None
  stats["survival_push_over"] = float(surv[~pend].float().mean()) if bool((~pend).any()) else None
  stats["frac_pending_push"] = float(pend.float().mean())
  print(f"[handover] by duration {by_dur}; pending push at the handover in "
        f"{100 * stats['frac_pending_push']:.1f}% of rows (survival "
        f"{stats['survival_pending_push']}), push over: {stats['survival_push_over']}")
  cats = list(lib_meta.get("categories", [])) + ["-"]
  per = {}
  print("[handover] per category (survival after the handover / replay falls):")
  for i, c in enumerate(cats):
    m = bank["category"] == (i if i < len(cats) - 1 else -1)
    started = int(h.ep_cat[i])
    if started == 0:
      continue
    s = float(surv[m].float().mean()) if bool(m.any()) else float("nan")
    pf = int(h.prefix_cat[i]) / started
    per[c] = {"rows": int(m.sum()), "survival": s, "episodes": started,
              "prefix_fall_rate": pf}
    print(f"    {c:15s} {int(m.sum()):7d} rows  survival {s:.3f}  replay falls {pf:.3f}")
  stats["per_category"] = per
  v = bank["value0"]
  if bool(torch.isfinite(v).any()):
    stats["auc_value"] = auc(v, surv)
    print(f"[handover] fallback value at the handover: AUC {stats['auc_value']:.3f} "
          f"(survived mean {float(v[surv].mean()):+.3f}, fell mean "
          f"{float(v[fell].mean()) if bool(fell.any()) else float('nan'):+.3f})")
  return stats


def main(argv=None):
  args = _parse(argv)
  torch.manual_seed(args.seed)
  dev = args.device
  lib = load_bank(args.library, "cpu")
  if not bool((lib["next_ok"] > 0).any()):
    raise SystemExit(f"{args.library} is not a clip library (no next_ok); build "
                     f"one with build_g1_recover_mocap_states.py --library")
  policy = value = monitor = None
  if args.policy:
    model, norm = load_twin(args.policy, dev)
    model.policy.set_training_mode(False)
    policy, value = (model, norm), PolicyValue(model, args.policy, dev)
  if args.handover_rule == "monitor":
    mpath = args.monitor_policy or args.policy
    if mpath == args.policy and value is not None:
      monitor = value
    else:
      mmodel, _ = load_twin(mpath, dev)
      mmodel.policy.set_training_mode(False)
      monitor = PolicyValue(mmodel, mpath, dev)
  need_crit = any(v is not None and v.asymmetric for v in (value, monitor))
  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  prefix_max = args.onset_s[1] + args.handover_s[1]
  overrides = dict(dataset=args.library, sampler="uniform", weights=weights,
                   episode_length_s=float(args.window_s + prefix_max + 0.5),
                   rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_tensor(args.task, args.num_envs, dev, cfg_overrides=overrides,
                    random_push=push_kw, critic_obs_key="critic" if need_crit else None)
  dt = float(env.mj.step_dt)
  ok = lib["start_ok"] & (lib["kind"] == KIND_MOCAP)
  room = lib["next_ok"][ok]
  need = int(round(prefix_max / dt))
  if room.numel() and int(room.min()) < need:
    print(f"[handover] WARNING: {int((room < need).sum())} spawnable rows have fewer "
          f"than {need} frames after them; their prefix is clamped to the clip")
  print(f"[handover] {args.library}: {int(ok.sum())} spawnable frames; push "
        f"{args.impulse_range[0]:g}-{args.impulse_range[1]:g} N*s over {args.durations} s, "
        f"onset {args.onset_s} s, handover {args.handover_rule} "
        f"{args.handover_s} s" + (f" (V < {args.monitor_delta:g})"
                                  if args.handover_rule == "monitor" else "")
        + ("; no label" if args.no_label else f"; label window {args.window_s} s"))
  h = Harvest(env, lib, args, policy=policy, value=value, monitor=monitor)
  t0 = time.time()
  bank = h.run()
  assert all(f in bank for f in FIELDS)
  stats = _report(bank, h, lib["meta"], args)
  lm = lib["meta"]
  bank["meta"] = {
    "source": "harvest_g1_handover_states.py", "library": os.path.abspath(args.library),
    "clips": list(lm.get("clips", [])), "categories": list(lm.get("categories", [])),
    "split": lm.get("split"), "policy": os.path.abspath(args.policy) if args.policy else None,
    "labeled": not args.no_label, "task": args.task,
    "impulse_range": list(args.impulse_range), "durations": list(args.durations),
    "onset_s": list(args.onset_s), "handover_s": list(args.handover_s),
    "handover_rule": args.handover_rule, "monitor_policy": args.monitor_policy,
    "monitor_delta": args.monitor_delta, "window_s": args.window_s, "seed": args.seed,
    "rows": int(bank["kind"].numel()), "fell_rows": int(bank["fell"].sum()),
    "pending_push": "push_force_h (N, heading frame of the row) acting for "
                    "push_left_steps control steps from the episode's first step",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats,
  }
  save_bank(bank, args.out)
  print(f"[handover] wrote {bank['meta']['rows']} rows ({bank['meta']['fell_rows']} "
        f"fell) to {args.out} ({time.time() - t0:.0f}s)")
  env.close()


if __name__ == "__main__":
  main()
