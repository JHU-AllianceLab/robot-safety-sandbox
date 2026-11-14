"""Evaluate a squat-tracking policy (or the SONIC anchor) in one world, under filter conditions.

    # a trained arm in its own world and in the other one (C2), every condition
    python examples/eval_track.py --stage pilot --arm t70_wm_d1 --seed 2 --world thermal70
    python examples/eval_track.py --stage pilot --arm t70_wm_d1 --seed 2 --world thermal_cbf
    # the no-learning anchor: SONIC tracking the same segment (action clamped to +-6.5)
    python examples/eval_track.py --sonic --world thermal70 --out runs/track_squat/eval/sonic/thermal70
    # learning curves: every checkpoint on a 100M grid, native condition, 256 episodes (C12)
    python examples/eval_track.py --stage pilot --arm t70_bare --seed 2 --world thermal70 \
        --curve-every 100000000 --episodes 256

One env of ``--episodes`` envs, one episode each: the first episode of every env
is recorded at its end (the task's episode bus, envs/g1_track/mdp.py), later ones
are ignored. The policy is deterministic (the actor mean, clamped to the +-6.5
task box), observation and spawn noise are OFF, and every condition starts from
the SAME states: the global torch RNG is reseeded with ``--eval-seed`` before the
env is built (startup domain randomization, the 33-D sim context) and before
every condition's reset (start temperatures, yaw / xy, RSI phases), and the
summary records hashes of the start state, the start temperatures and the sim
context so the report can verify the pairing (C11). The sim itself is not
bit-reproducible run to run, so paired episodes can still diverge.

Conditions (``--conditions``), each in THIS world:
  bare        no filter (in thermal_cbf the barrier is still under every action)
  wm_d0       the world's tube filter, no dwell       } once per --fb-seeds seed
  wm_d1       the same, 1 s dwell                     } (cross-pairing, C4)
  wm_d0_c90 / wm_d1_c90   the c = 0.90 sensitivity tube (C4)
  wmcalib     coverage (C5): the policy acts until a per-env decision step, the
              tube's verdict rollout ("proposal, then pi") is recorded there, the
              proposal runs, then the fallback is forced for the rest of the 1 s
              window; score = min predicted g - min real g over the window
              (a failure closes it with its terminal g, a time-out censors it);
              coverage = P(score <= eps)
Suites: ``full`` (every episode starts at frame 0 and runs to the segment's end:
the primary suite, C6) or ``rsi`` (paired uniform starts leaving >= 1 s).

Writes ``<out>/summary.json`` (per condition: the metrics below, Wilson
intervals, bands of the hottest start winding [30,60) [60,66) [66,70] (C3),
per-phase-bin curves, filter engagement, hashes) and ``<out>/records.pt``
(per-episode records, for the report's paired tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import track_common as tc  # noqa: E402

#: the bus's codes (envs/g1_track/mdp.py; checked at run time)
CAUSES = ("complete", "overheat", "contact", "tilt", "other", "cut")
FAIL_CAUSES = ("overheat", "contact", "tilt", "other")
BANDS = ((30.0, 60.0), (60.0, 66.0), (66.0, 70.0 + 1e-6))
EP_KEYS = ("cause", "fail", "fail_step", "fail_bin", "any_fail", "overheat", "overheat_step",
           "steps", "start_f", "end_f", "completion", "err_ori_mean", "err_body_mean",
           "err_q_mean", "err_anchor_z_mean", "tf_steps", "lost_any", "tw_start_max", "tw_peak",
           "dtw_peak", "heat_dose", "tau2_sum", "tw_over_max", "cbf_clip_mean")
FILTER_KEYS = ("engaged", "held", "switches", "handovers")


# --- statistics (pure; tests/test_track_eval.py) ------------------------------------

def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
  """Wilson score interval of a binomial proportion k / n."""
  if n <= 0:
    return (math.nan, math.nan)
  p = k / n
  den = 1.0 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (c - h, c + h)


def state_hash(*tensors, decimals: int = 4) -> str:
  """A digest of tensors rounded to ``decimals``: equal starts hash equal even when
  the last float bits of a rebuilt env differ."""
  import torch
  h = hashlib.sha256()
  for t in tensors:
    if t is None:
      h.update(b"none")
      continue
    x = torch.round(t.detach().double().cpu() * 10 ** decimals).to(torch.int64)
    h.update(x.numpy().tobytes())
  return h.hexdigest()[:16]


def summarize(rec: dict, T: int, dt: float, n_bins: int = 10) -> dict:
  """Per-condition metrics from per-episode records (numpy/torch arrays on CPU)."""
  import numpy as np
  r = {k: np.asarray(v.cpu() if hasattr(v, "cpu") else v) for k, v in rec.items()}
  n = int(r["cause"].shape[0])
  span = (T - r["start_f"]).clip(min=1)                 # frames the episode could track
  out = {"n": n}

  def rate(mask, name):
    k = float(np.sum(mask))
    out[name] = k / n if n else math.nan
    out[f"{name}_ci"] = wilson(k, n)

  rate(r["cause"] == CAUSES.index("complete"), "survival")
  rate(r["any_fail"].astype(bool), "any_fail")
  rate(r["overheat"].astype(bool), "overheat_incidence")
  for c in FAIL_CAUSES:
    out[f"fail_{c}"] = float(np.mean(r["fail"] == CAUSES.index(c))) if n else math.nan
  for c in CAUSES:
    out[f"end_{c}"] = float(np.mean(r["cause"] == CAUSES.index(c))) if n else math.nan
  out["tf"] = float(np.mean(r["tf_steps"] / span))
  out["completion"] = float(np.mean(r["completion"]))
  out["err_body_alive"] = float(np.mean(r["err_body_mean"]))
  out["err_q_alive"] = float(np.mean(r["err_q_mean"]))
  out["dtw_peak"] = float(np.mean(r["dtw_peak"]))
  out["tw_peak"] = float(np.mean(r["tw_peak"]))
  out["tw_start_max"] = float(np.mean(r["tw_start_max"]))
  out["heat_dose_per_T"] = float(np.mean(r["heat_dose"] / (span * dt)))
  out["tau2_per_step"] = float(np.mean(r["tau2_sum"] / r["steps"].clip(min=1)))
  out["tw_over_max"] = float(np.max(r["tw_over_max"])) if n else math.nan
  clip = r["cbf_clip_mean"]
  out["cbf_clip_mean"] = float(np.nanmean(clip)) if np.isfinite(clip).any() else None
  if "engaged" in r:
    out["engaged_frac"] = float(np.sum(r["engaged"]) / max(1.0, float(np.sum(r["steps"]))))
    out["handovers_per_ep"] = float(np.mean(r["handovers"]))
    out["switches_per_ep"] = float(np.mean(r["switches"]))
    out["held_frac"] = float(np.sum(r["held"]) / max(1.0, float(np.sum(r["steps"]))))
  bands = {}
  for lo, hi in BANDS:
    m = (r["tw_start_max"] >= lo) & (r["tw_start_max"] < hi)
    k = int(m.sum())
    bands[f"{lo:g}-{min(hi, 70.0):g}"] = {
      "n": k,
      "survival": float(np.mean(r["cause"][m] == 0)) if k else math.nan,
      "any_fail": float(np.mean(r["any_fail"][m])) if k else math.nan,
      "overheat": float(np.mean(r["overheat"][m])) if k else math.nan}
  out["bands"] = bands
  fb = r["fail_bin"]
  out["S_phase"] = [float(np.mean((fb < 0) | (fb >= b))) for b in range(n_bins)]
  if "bin_steps" in r:
    st = r["bin_steps"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
      out["err_body_phase"] = (r["bin_err"].sum(0) / st).tolist()
      out["tw_phase"] = (r["bin_tw"].sum(0) / st).tolist()
      out["engaged_phase"] = (r["bin_eng"].sum(0) / st).tolist()
  return out


# --- the run ---------------------------------------------------------------------------

def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--stage", default="pilot")
  p.add_argument("--arm", default=None, choices=sorted(tc.ARMS))
  p.add_argument("--seed", type=int, default=None, help="the policy's task seed")
  p.add_argument("--policy-dir", default=None, help="a run dir (default: from stage/arm/seed)")
  p.add_argument("--checkpoint", type=int, default=None, help="evaluate model_<N>_steps.zip")
  p.add_argument("--curve-every", type=int, default=None,
                 help="evaluate every checkpoint on this step grid (native condition only)")
  p.add_argument("--sonic", action="store_true", help="the SONIC anchor instead of a policy")
  p.add_argument("--sonic-dir", default=os.path.expanduser("~/gear_sonic_hf"))
  p.add_argument("--world", required=True, choices=sorted(tc.WORLDS))
  p.add_argument("--conditions", nargs="+", default=["bare", "wm_d0", "wm_d1"])
  p.add_argument("--fb-seeds", type=int, nargs="+", default=list(tc.SEEDS))
  p.add_argument("--suite", choices=("full", "rsi"), default="full")
  p.add_argument("--episodes", type=int, default=1024)
  p.add_argument("--eval-seed", type=int, default=0)
  p.add_argument("--repeat", type=int, default=1,
                 help="run every condition this many times (2: the nondeterminism floor)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None)
  a = p.parse_args(argv)
  if not a.sonic and a.policy_dir is None and (a.arm is None or a.seed is None):
    p.error("give --policy-dir, or --arm and --seed, or --sonic")
  return a


def _parse_condition(name: str):
  """'bare' | 'wm_d<dwell>[_c<pct>]' | 'wmcalib' -> (kind, dwell_s, conf)."""
  if name == "bare":
    return ("bare", 0.0, None)
  if name == "wmcalib":
    return ("wmcalib", 0.0, 0.95)
  if name.startswith("wm_d"):
    rest = name[4:]
    conf = 0.95
    if "_c" in rest:
      rest, pct = rest.split("_c", 1)
      conf = int(pct) / 100.0
    return ("wm", float(rest), conf)
  raise ValueError(f"unknown condition {name!r}")


class _Policy:
  """A trained task policy: deterministic actor mean on its own frozen normalizer."""

  def __init__(self, zip_path: str, device: str):
    # a plain CumulativePPO1P load (AMPPPO keeps D out of the zip); eval.policies'
    # load_twin only tries the safety / reach-avoid twin classes
    import torch
    from safety_sb3 import CumulativePPO1P

    from robot_safety_sandbox.eval.policies import find_obs_stats
    self.model = CumulativePPO1P.load(zip_path, device=device)
    pt = find_obs_stats(zip_path, "tensornorm", "tensornormalize.pt")
    if not pt:
      raise SystemExit(f"[eval-track] no tensornormalize.pt / tensornorm_<n>.pt beside {zip_path}")
    st = torch.load(pt, map_location=device, weights_only=True)
    mean, var = st["obs_mean"].to(device), st["obs_var"].to(device)
    self.norm = lambda o: torch.clamp((o - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
    print(f"[eval-track] policy {zip_path} + {os.path.basename(pt)}")
    self.pol = self.model.policy
    self.pol.set_training_mode(False)

  def __call__(self, obs):
    import torch
    with torch.no_grad():
      x = self.norm(obs.float())
      if hasattr(self.pol, "_actor_latent"):        # asymmetric: the actor stack only
        return self.pol.action_net(self.pol._actor_latent(x))
      return self.pol._predict(x, deterministic=True)


def _checkpoints(run: str, every: int) -> list[tuple[int, str]]:
  import re
  ck = os.path.join(run, "checkpoints")
  out = []
  for f in os.listdir(ck) if os.path.isdir(ck) else []:
    m = re.fullmatch(r"model_(\d+)_steps\.zip", f)
    if m:
      out.append((int(m.group(1)), os.path.join(ck, f)))
  # the checkpoint NEAREST each multiple of ``every``: SB3 saves every
  # floor(checkpoint_every / num_envs) calls, so checkpoints sit just below the
  # round numbers (6103 x 4096 = 24,997,888 for 25M)
  best = {}
  for n, path in sorted(out):
    k = round(n / every)
    if k >= 1 and (k not in best or abs(n - k * every) < abs(best[k][0] - k * every)):
      best[k] = (n, path)
  return [best[k] for k in sorted(best)]


def main(argv=None):
  a = _parse(argv)
  import numpy as np  # noqa: F401
  import torch

  from robot_safety_sandbox.envs.g1_track import mdp
  from robot_safety_sandbox.filtered_env import WMTubeFilteredEnv
  from robot_safety_sandbox.filters.wm_tube import check_binding, wm_tube_filter
  if tuple(mdp.CAUSES) != CAUSES:
    raise SystemExit(f"[eval-track] bus codes {mdp.CAUSES} != {CAUSES}")

  run = a.policy_dir or (tc.run_dir(a.stage, a.arm, a.seed) if not a.sonic else None)
  who = "sonic" if a.sonic else f"{a.arm or os.path.basename(run)}_s{a.seed}"
  out = a.out
  if out is None and a.arm:
    out = tc.eval_dir(a.stage, a.arm, a.seed, a.world)
    if a.curve_every:
      out = os.path.join(out, "curve")
  if out is None:
    raise SystemExit("[eval-track] pass --out")
  os.makedirs(out, exist_ok=True)
  task = tc.task_of(a.world)
  dev, N = a.device, int(a.episodes)

  def noise_off(cfg):
    for g in ("task_policy", "actor"):
      if g in cfg.observations:
        cfg.observations[g].enable_corruption = False

  torch.manual_seed(a.eval_seed)                      # startup DR: paired across worlds / arms
  env = tc.make_task_env(task, N, dev, cfg_fn=noise_off)
  env.reset()
  mj = env.mj
  cmd = mj.command_manager.get_term("ref")
  T, dt = int(cmd.T), float(mj.step_dt)
  cmd.spawn_noise = False
  if a.suite == "full":
    cmd.start_override = torch.zeros((), dtype=torch.long, device=dev)
  else:
    g = torch.Generator(device=dev).manual_seed(a.eval_seed + 1)
    cmd.start_override = torch.randint(0, T - 1 - int(round(1.0 / dt)) + 1, (N,), device=dev,
                                       generator=g)
  max_steps = int(round(float(mj.cfg.episode_length_s) / dt)) + 5

  # the controller(s) to evaluate
  if a.sonic:
    from robot_safety_sandbox.envs.g1_recover.sonic import (SonicPolicy, SonicReference,
                                                            SonicTracker)
    from robot_safety_sandbox.envs.g1_track.reference import DEFAULT_CSV_DIR
    meta = cmd.ref.meta
    sref = SonicReference.build(DEFAULT_CSV_DIR, [meta["clip"]], dev)
    tracker = SonicTracker(env, SonicPolicy.from_onnx(a.sonic_dir).to(dev).eval(), sref)
    f0 = int(meta["f0"])
    controllers = [("final", None)]
  elif a.curve_every:
    controllers = [(n, path) for n, path in _checkpoints(run, a.curve_every)]
    if not controllers:
      raise SystemExit(f"[eval-track] no checkpoints on a {a.curve_every} grid in {run}")
  else:
    zip_path = (os.path.join(run, "checkpoints", f"model_{a.checkpoint}_steps.zip")
                if a.checkpoint else os.path.join(run, "final_model.zip"))
    controllers = [("final" if not a.checkpoint else a.checkpoint, zip_path)]

  # the filter conditions; one bundle per (fallback seed, coverage), bound once
  conds = []
  native = None
  if a.curve_every:
    world, kind, dwell = tc.ARMS[a.arm]
    native = tc.arm_name(world, kind, dwell)
    a.conditions = ["bare" if kind is None else f"wm_d{int(dwell)}"]
    a.fb_seeds = [a.seed]
  bundles = {}
  lay_path = tc.layout_json(a.world)
  layout = json.load(open(lay_path)) if os.path.isfile(lay_path) else None
  for name in a.conditions:
    kind, dwell, conf = _parse_condition(name)
    seeds = [None] if kind == "bare" else a.fb_seeds
    for s in seeds:
      if s is not None and (s, conf) not in bundles:
        b = tc.load_bundle(a.world, s, conf=conf, device=dev)
        check_binding(b, env, layout_ref=layout, verbose=False)
        bundles[(s, conf)] = b
      conds.append((name if s is None else f"{name}@fb{s}", kind, dwell, conf, s))

  results, records = {}, {}
  t_start = time.time()
  for tag, ctrl_path in controllers:
    policy = None if a.sonic else _Policy(ctrl_path, dev)
    for cname, kind, dwell, conf, s in conds:
      for rep in range(int(a.repeat)):
        key = cname if a.repeat == 1 else f"{cname}#r{rep}"
        if a.curve_every:
          key = f"{tag}"
        filt, mon, bundle = None, None, None
        if kind in ("wm", "wmcalib"):
          bundle = bundles[(s, conf)]
          filt, _ = wm_tube_filter(env, dwell_s=dwell, proposal_bound=tc.PROPOSAL_BOUND,
                                   condition=tc.condition(conf), bundle=bundle)
          mon = filt.monitor
        fenv = WMTubeFilteredEnv(env, filt if kind == "wm" else None, coverage_probe=False)
        rec, cov = _run(env, fenv, mj, policy, tracker if a.sonic else None,
                        f0 if a.sonic else 0, kind, mon, bundle, a.eval_seed, N, T, max_steps,
                        dev)
        summ = summarize(rec, T, dt)
        if cov is not None:
          summ["coverage"] = cov
        summ["condition"] = {"name": cname, "kind": kind, "dwell_s": dwell, "conf": conf,
                             "fb_seed": s, "eps": None if bundle is None else bundle.eps}
        summ["hashes"] = rec.pop("_hashes")
        results[key] = summ
        records[key] = rec
        print(f"[eval-track] {who} {a.world} {key}: survival {summ['survival']:.3f} any-fail "
              f"{summ['any_fail']:.3f} overheat {summ['overheat_incidence']:.3f} TF "
              f"{summ['tf']:.3f} dTw {summ['dtw_peak']:.2f} C"
              + (f" engaged {summ['engaged_frac']:.3f}" if "engaged_frac" in summ else "")
              + (f" coverage {cov['coverage']:.3f} ({cov['n']} windows)" if cov else "")
              + f" ({time.time() - t_start:.0f} s)")

  meta = {"who": who, "run": run, "world": a.world, "task": task, "suite": a.suite,
          "episodes": N, "eval_seed": a.eval_seed, "T": T, "dt": dt, "obs_noise": False,
          "spawn_noise": False, "proposal_bound": tc.PROPOSAL_BOUND,
          "sonic_native_clamp": 20.0 if a.sonic else None, "native": native,
          "fallbacks": {f"fb{s}_c{c:g}": b.meta for (s, c), b in bundles.items()},
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  torch.save({"meta": meta, "records": records}, os.path.join(out, "records.pt"))
  with open(os.path.join(out, "summary.json"), "w") as f:
    json.dump({"meta": meta, "conditions": results}, f, indent=1, default=repr)
  print(f"[eval-track] wrote {out}/summary.json ({len(results)} conditions)")


def _run(env, fenv, mj, policy, tracker, f0, kind, mon, bundle, seed, N, T, max_steps, dev):
  """One condition: every env's first episode. Returns (records, coverage | None)."""
  import torch
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  n_bins = 10
  torch.manual_seed(seed)
  obs = fenv.reset()
  robot = mj.scene["robot"].data
  hashes = {"start_state": state_hash(robot.root_link_pos_w - mj.scene.env_origins,
                                      robot.root_link_quat_w, robot.joint_pos),
            "start_temps": state_hash(thermal_state(mj).Tw),
            "sim_context": state_hash(env.sim_context())}
  cmd = mj.command_manager.get_term("ref")
  if tracker is not None:
    tracker.start(torch.ones(N, dtype=torch.bool, device=dev),
                  torch.zeros(N, dtype=torch.long, device=dev), f0 + cmd.start_f)
  done_rec = torch.zeros(N, dtype=torch.bool, device=dev)
  rec = None
  fin = {k: torch.zeros(N, device=dev) for k in FILTER_KEYS}
  bins = {k: torch.zeros(N, n_bins, device=dev) for k in ("bin_steps", "bin_err", "bin_tw",
                                                          "bin_eng")}
  # wmcalib: one decision step per env, then the forced fallback window
  H = 50
  if kind == "wmcalib":
    g = torch.Generator(device=dev).manual_seed(seed + 2)
    k_dec = torch.randint(1, max(2, T - 1 - H), (N,), device=dev, generator=g)
    pred_min = torch.full((N,), math.nan, device=dev)
    real_min = torch.full((N,), math.inf, device=dev)
    win_end = torch.full((N,), -1, dtype=torch.long, device=dev)
    state = torch.zeros(N, dtype=torch.long, device=dev)    # 0 before, 1 in window, 2 closed, 3 censored
  for t in range(max_steps):
    if tracker is not None:
      a = tracker.act()
    else:
      a = policy(obs)
    a = torch.clamp(a, -tc.PROPOSAL_BOUND, tc.PROPOSAL_BOUND)
    if kind == "wmcalib":
      step_t = cmd.f - cmd.start_f                   # steps into the episode before this one
      dec = (state == 0) & (step_t == k_dec) & ~done_rec
      if bool(dec.any()):
        ids = torch.nonzero(dec).squeeze(1)
        pred_min[ids] = mon.margins(ids, a[ids], H).amin(1)
        state[ids] = 1
        win_end[ids] = step_t[ids] + H
      forced = (state == 1) & ~dec
      if bool(forced.any()):
        A_obs = torch.nan_to_num(env.obs_groups()["actor"].float())
        a = torch.where(forced[:, None], bundle.pi(A_obs), a)
    obs, _, d, tout, _ = fenv.step_tensor(a)
    if tracker is not None:
      tracker.advance()
    d = d.bool()
    bus = mj._zoo_track
    s, e = bus["step"], bus["ep"]
    live = ~done_rec
    pb = s["phase_bin"].long().clamp(0, n_bins - 1)
    oh = torch.nn.functional.one_hot(pb, n_bins).float() * live.float()[:, None]
    bins["bin_steps"] += oh
    bins["bin_err"] += oh * s["err_body"][:, None]
    bins["bin_tw"] += oh * s["tw_max"][:, None]
    bins["bin_eng"] += oh * fenv.last_info["engaged"].float()[:, None]
    if kind == "wmcalib":
      inwin = state == 1
      g_now = env.safety_margins()[0].float()
      real_min = torch.where(inwin & live, torch.minimum(real_min, g_now), real_min)
      step_after = cmd.f - cmd.start_f
      closed = inwin & live & ((d & ~tout.bool()) | (step_after >= win_end))
      cens = inwin & live & d & tout.bool() & (step_after < win_end)
      state = torch.where(cens, torch.full_like(state, 3), state)
      state = torch.where(closed & ~cens, torch.full_like(state, 2), state)
    new = d & live
    if bool(new.any()):
      if rec is None:
        rec = {k: torch.zeros_like(e[k]) for k in EP_KEYS
               if k in e and e[k] is not None and e[k].dim() == 1}
      for k in rec:
        rec[k] = torch.where(new, e[k], rec[k])
      fd = getattr(fenv, "finished", None)
      if fd is not None:
        for k in FILTER_KEYS:
          if k in fd:
            fin[k] = torch.where(new, fd[k].float(), fin[k])
      done_rec |= new
    if bool(done_rec.all()):
      break
  if not bool(done_rec.all()):
    print(f"[eval-track] WARNING: {int((~done_rec).sum())} envs never finished in {max_steps} "
          f"steps; their records are zeros")
  out = {k: v.cpu() for k, v in rec.items()}
  if kind == "wm":
    out.update({k: v.cpu() for k, v in fin.items()})
  else:
    out.update({k: torch.zeros(N) for k in FILTER_KEYS})
    out["engaged"] = bins["bin_eng"].sum(1).cpu()
  out.update({k: v.cpu() for k, v in bins.items()})
  out["_hashes"] = hashes
  cov = None
  if kind == "wmcalib":
    ok = state == 2
    score = (pred_min - real_min)[ok]
    n = int(ok.sum())
    cov = {"n": n, "censored": int((state == 3).sum()), "never_decided": int((state == 0).sum()),
           "eps": bundle.eps,
           "coverage": float((score <= bundle.eps).float().mean()) if n else math.nan,
           "score_q": {q: float(torch.quantile(score, q)) for q in (0.5, 0.9, 0.95, 0.99)}
           if n else {}}
  return out, cov


if __name__ == "__main__":
  main()
