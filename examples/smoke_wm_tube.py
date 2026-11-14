"""GPU smoke of the train-time world-model tube filter (filters/wm_tube.py).

    # 1. the layout references, from the fallbacks' own tasks (once per world)
    python examples/smoke_wm_tube.py --dump-layout --world thermal70
    python examples/smoke_wm_tube.py --dump-layout --world thermal_cbf
    # 2. live checks on the task env, then the bench
    python examples/smoke_wm_tube.py --world thermal70 --task g1_track_squat_thermal70 --bench

``--dump-layout`` builds ``g1_recover_<world>`` (64 envs), checks every seed's
fallback binds it (check_binding: the rebuilt world-model spec equals each
zip's saved one, the thermal world equals its g1_sysid.json) and writes
``fallback_layout(env)`` to ``configs/track_squat/fallback_layout_<world>.json``,
the reference every training / eval env is held to.

Live (``--task``; any env with the fallback's ``actor`` / ``critic`` groups, the
reference task included): check_binding against that reference, then for dwell
0 and 1 s, ``--steps`` filtered steps of ``--num-envs`` envs under scripted
proposals (half the envs at rest, half random in the task box), checking every
step that

  * the cached critic group equals a recompute from its term functions;
  * the verdicts of the rolled-out envs are torch.equal to the eval's
    ``Evaluator.min_verdict`` on the same ids (E built on the live env through
    a view naming the ``actor`` group, E:640), every ``--parity-every`` steps;
  * the executed action is E's fallback action (E:868-870) exactly where
    engaged and the clamped proposal elsewhere; ``_actor_mean`` = ``_predict``;
  * every hand-over is followed by min(dwell, steps to the episode's end)
    engaged steps.

``--bench``: on live states of a ``max(--bench-envs)``-env task env, the
rollout's ms per call for N in ``--bench-envs`` x {fp32, tf32, bf16} x
{eager, eager with 25% held, graph}, and against eager fp32 the max and p99.9
|delta verdict| and the decision flip rate. A mode may replace eager fp32 in
training only if flips <= 1e-3 and p99.9 <= 0.1 * min eps (0.0066).

Prints ``[smoke] ... OK`` per check, exits non-zero on the first failure,
writes ``<--out>/smoke_<world>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

_EX = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_EX)
sys.path.insert(0, _ZOO)
sys.path.insert(0, _EX)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.filtered_env import WMTubeFilteredEnv  # noqa: E402
from robot_safety_sandbox.filters import wm_tube as wt  # noqa: E402

#: The frozen fallbacks and tubes per world (tags as min_tube.json has them).
WORLDS = {
  "thermal70": dict(task="g1_recover_thermal70",
                    run="runs/shac_thermal70/shac_s{s}/g1_recover_thermal70_safetyshac1p",
                    tube="runs/shac_thermal70/wmreport/calib/thermal70_s{s}/min_tube.json",
                    tag="thermal70_s{s}"),
  "thermal_cbf": dict(task="g1_recover_thermal_cbf",
                      run="runs/shac_thermal_cbf/shac_s{s}/g1_recover_thermal_cbf_safetyshac1p",
                      tube="runs/shac_thermal_cbf/wmreport/calib/thermalcbf_s{s}/min_tube.json",
                      tag="thermalcbf_s{s}"),
}
ALIASES = {"cbf": "thermal_cbf", "t70": "thermal70"}
LAYOUT = os.path.join(_ZOO, "configs", "track_squat", "fallback_layout_{world}.json")
#: the bench's switching rule (spec_wm-filter.md section 7)
MAX_FLIP, MAX_P999 = 1e-3, 0.0066


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--world", required=True, choices=sorted(WORLDS) + sorted(ALIASES))
  p.add_argument("--dump-layout", action="store_true")
  p.add_argument("--task", default=None, help="the env to check live (default: none)")
  p.add_argument("--seed", type=int, default=2, help="fallback / tube seed of the live checks")
  p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="--dump-layout seeds")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=300)
  p.add_argument("--parity-every", type=int, default=10)
  p.add_argument("--proposal-bound", type=float, default=None, help="default: the env's box")
  p.add_argument("--bench", action="store_true")
  p.add_argument("--bench-envs", type=int, nargs="+", default=[1024, 4096])
  p.add_argument("--bench-reps", type=int, default=5)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs", "track_squat", "smoke_wm_tube"))
  p.add_argument("--wm-thermal", default="nn", choices=list(wt.WM_THERMAL),
                 help="hybrid: the sysID heat model in the rollout (thermal70 only; its tube is "
                      "the hybrid step-0 calibration)")
  p.add_argument("--holds", nargs="+", default=["0", "1"],
                 help="live checks per hold: dwell seconds, or 'latch'")
  p.add_argument("--root", default=None,
                 help="a release (examples/export_safety_filters.py): load the fallback, tube "
                      "and layout from it through filters/pretrained.py")
  a = p.parse_args(argv)
  a.world = ALIASES.get(a.world, a.world)
  if a.wm_thermal == "hybrid" and a.world != "thermal70":
    p.error("--wm-thermal hybrid needs --world thermal70")
  return a


#: the hybrid step-0 tubes (examples/calibrate_wm_min_tube.py on --wm-thermal hybrid runs)
HYBRID_TUBE = "runs/shac_thermal70/hybrid_step0/calib/hybrid/thermal70_s{s}/min_tube.json"
#: (world, wm_thermal) -> the release's variant
VARIANT = {("thermal70", "nn"): "joint", ("thermal70", "hybrid"): "separate",
           ("thermal_cbf", "nn"): "fall_only_cbf"}


def _paths(world: str, s: int, wm_thermal: str = "nn", root: str | None = None):
  if root is not None:
    from robot_safety_sandbox.filters.pretrained import variant_paths
    p = variant_paths(root, VARIANT[(world, wm_thermal)], s)
    return p["run_dir"], p["tube_json"], p["tag"]
  w = WORLDS[world]
  tube = HYBRID_TUBE if wm_thermal == "hybrid" else w["tube"]
  return (os.path.join(_ZOO, w["run"].format(s=s)), os.path.join(_ZOO, tube.format(s=s)),
          w["tag"].format(s=s))


def _bundle(world, s, dev, cond="wmmin:1:0.95", wm_thermal="nn", root=None):
  run, tube, tag = _paths(world, s, wm_thermal, root)
  return wt.load_wm_tube(run, tube, tag=tag, cond=cond, device=dev, wm_thermal=wm_thermal)


def ok(msg: str) -> None:
  print(f"[smoke] {msg} OK", flush=True)


def fail(msg: str) -> None:
  raise SystemExit(f"[smoke] FAILED: {msg}")


def _sync(dev):
  if str(dev).startswith("cuda"):
    torch.cuda.synchronize(dev)


# --- 1. layout references --------------------------------------------------------

def dump_layout(a) -> dict:
  task = WORLDS[a.world]["task"]
  env = make_tensor(task, 64, a.device, critic_obs_key="critic")
  env.reset()
  lay = wt.fallback_layout(env)
  binds = {}
  for s in a.seeds:
    b = _bundle(a.world, s, a.device)
    binds[s] = wt.check_binding(b, env, layout_ref=lay)
    ok(f"{task}: fallback s{s} binds its own task")
  path = LAYOUT.format(world=a.world)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(lay, f, indent=1, sort_keys=True)
  dims = {g: sum(math.prod(t["dims"]) for t in lay["groups"][g]["terms"])
          for g in ("actor", "critic")}
  print(f"[smoke] layout of {task} -> {path} (actor {dims['actor']}-D, critic "
        f"{dims['critic']}-D, context {lay['context']['dim']}-D)")
  env.close()
  return {"layout": path, "world": binds[a.seeds[0]]["world"]}


# --- 2. live checks --------------------------------------------------------------

def _eval_stub(eq, env, bundle):
  """E's Evaluator on the live env, through a view naming the fallback's
  ``actor`` group (on the track tasks env.obs_key is the task group, E:640)."""
  ev = object.__new__(eq.Evaluator)
  ev.mj = env.mj
  ev.env = SimpleNamespace(obs_key="actor", mj=env.mj, sim_context=env.sim_context)
  ev.model, ev.norm = bundle.model, bundle.norm
  ev.dev, ev.n = torch.device(env.device), env.num_envs
  ev.low, ev.high = bundle.lo, bundle.hi          # the fallback box E ran in
  ev.pi_slice, ev.pi_obs_dim = eq.Evaluator._last_action_slice(ev)
  ev.wm_actor_idx = bundle.actor_idx
  ev.wm_z_clip = bundle.z_clip
  ev.wm_state_sl = None
  ev.wm_tw_sl = None
  ev.hyb = None
  if bundle.wm_thermal == "hybrid":
    # E:561-564, --wm-thermal hybrid with the calibration's settings
    from robot_safety_sandbox.filters.hybrid_thermal import AnalyticThermal
    ev.hyb = AnalyticThermal(env, bundle.wm, "hybrid", cold_c=float(bundle.thermal["cold_c"]),
                             clamp=float(bundle.thermal["clamp"]))
  with open(bundle.meta["tube"]) as f:
    ev.tube_calib = json.load(f)
  if ev.pi_slice != bundle.la:
    fail(f"E's last-action columns {ev.pi_slice} vs the bundle's {bundle.la}")
  return ev


def live(a, eq, ql) -> dict:
  dev = a.device
  env = make_tensor(a.task, a.num_envs, dev)
  env.reset()
  bundle = _bundle(a.world, a.seed, dev, wm_thermal=a.wm_thermal, root=a.root)
  ref_path = (os.path.join(a.root, "layouts", f"{a.world}.json") if a.root
              else LAYOUT.format(world=a.world))
  layout_ref = None
  if os.path.isfile(ref_path):
    with open(ref_path) as f:
      layout_ref = json.load(f)
  else:
    print(f"[smoke] WARNING: no {ref_path} (run --dump-layout first): layout not compared")
  binding = wt.check_binding(bundle, env, layout_ref=layout_ref)
  ok(f"{a.task}: check_binding (layout {'compared' if layout_ref else 'NOT compared'}; "
     f"{bundle.wm_thermal} world model{'; release ' + a.root if a.root else ''})")
  pb = a.proposal_bound if a.proposal_bound is not None else float(env.action_space.high.max())
  ev = _eval_stub(eq, env, bundle)
  cond = ql.parse_condition(bundle.cond.name)
  n = env.num_envs
  out = {"binding": binding, "proposal_bound": pb, "wm_thermal": bundle.wm_thermal,
         "release": a.root, "dwell": {}}
  for hold in a.holds:
    latch = hold == "latch"
    dwell_s = 0.0 if latch else float(hold)
    filt, _ = wt.wm_tube_filter(env, dwell_s=dwell_s, latch=latch, proposal_bound=pb,
                                bundle=bundle)
    w = WMTubeFilteredEnv(env, filt, verify_cache_every=1)
    w.reset()
    D = filt.intervention.dwell_steps
    gen = torch.Generator(device=dev).manual_seed(0)
    moving = torch.arange(n, device=dev) % 2 == 1
    owe = torch.zeros(n, dtype=torch.long, device=dev)     # hold steps still owed per env
    n_par = n_ids = 0
    for t in range(a.steps):
      og = env.obs_groups()
      A = torch.nan_to_num(og["actor"].float(), nan=0.0, posinf=0.0, neginf=0.0)
      a_prop = 0.5 * pb * (2 * torch.rand(n, 29, generator=gen, device=dev) - 1)
      a_prop = torch.where(moving.unsqueeze(-1), a_prop, torch.zeros_like(a_prop))
      a_p = a_prop.clamp(-pb, pb)
      itv = filt.intervention
      if latch:
        active = ~itv.fired
      else:
        active = ~(itv.k < itv.hold_until) if D else torch.ones(n, dtype=torch.bool, device=dev)
      ref_v = None
      with torch.no_grad():
        pobs, _ = eq.Evaluator._pi_obs(ev, A)
        a_safe = torch.clamp(bundle.model.policy._predict(bundle.norm(pobs), deterministic=True),
                             ev.low, ev.high)
        if not torch.equal(a_safe, bundle.model._actor_mean(bundle.norm(pobs)).clamp(ev.low,
                                                                                   ev.high)):
          fail(f"step {t}: _actor_mean != _predict(deterministic=True)")
        if t % a.parity_every == 0:
          ids = active.nonzero(as_tuple=False).squeeze(-1)
          ref_v = (ids, eq.Evaluator.min_verdict(ev, ids, a_p[ids], cond))
      w.step_tensor(a_prop)
      info = w.last_info
      eng = info["engaged"]
      ex = w.executed_action
      if not (torch.equal(ex[eng], a_safe[eng]) and torch.equal(ex[~eng], a_p[~eng])):
        fail(f"dwell {dwell_s}: step {t}: executed action is not where(engaged, E's a_safe, "
             f"proposal)")
      if ref_v is not None:
        ids, rv = ref_v
        if not torch.equal(info["verdict"][ids], rv):
          d = (info["verdict"][ids] - rv).abs()
          fail(f"dwell {dwell_s}: step {t}: verdicts differ from E's min_verdict (max |d| "
               f"{float(d.max()):.3g}, {int((d > 0).sum())} of {ids.numel()})")
        n_par += 1
        n_ids += int(ids.numel())
      done = w.finished["done"]
      # the steps an earlier hand-over still holds must be engaged (an episode end clears them)
      if bool(((owe > 0) & ~eng).any()):
        fail(f"hold {hold}: step {t}: a hand-over released before its hold")
      owe = torch.where(info["new_handover"],
                        torch.full_like(owe, 10 ** 9 if latch else max(D, 1)), owe) - 1
      owe = torch.where(done, torch.zeros_like(owe), owe.clamp_min(0))
    m = w.metrics()
    ok(f"hold {hold}: {a.steps} steps x {n} envs: cache = recompute, executed = "
       f"where(engaged, a_safe, proposal), holds served, verdicts = E on {n_ids} rollouts "
       f"({n_par} steps); engaged {m.get('filter/engaged_frac', 0):.3f}, hand-overs/1k "
       f"{m.get('filter/handovers_per_1k', 0):.2f}, {m.get('filter/ms_per_step', 0):.1f} ms/step")
    out["dwell"][hold] = {k: v for k, v in m.items() if k.startswith(("filter/", "safety/"))}
  env.close()
  return out


# --- 3. bench --------------------------------------------------------------------

def bench(a) -> dict:
  dev, N = a.device, max(a.bench_envs)
  env = make_tensor(a.task, N, dev)
  env.reset()
  bundle = _bundle(a.world, a.seed, dev)
  pb = a.proposal_bound if a.proposal_bound is not None else float(env.action_space.high.max())
  gen = torch.Generator(device=dev).manual_seed(1)
  for _ in range(25):                               # move off the spawn distribution a little
    env.step_tensor(0.3 * pb * (2 * torch.rand(N, 29, generator=gen, device=dev) - 1))
  S = torch.nan_to_num(env.obs_groups()["critic"].float()).clone()
  C = env.sim_context().clone()
  P = pb * (2 * torch.rand(N, 29, generator=gen, device=dev) - 1)
  env.close()
  rows = []
  for n in sorted(a.bench_envs):
    s, c, p = S[:n], C[:n], P[:n]
    held = torch.rand(n, generator=gen, device=dev) < 0.25
    ref = None
    for prec in wt.PRECISIONS:
      for name, mode, active in (("eager", "compact", None), ("eager_held25", "compact", ~held),
                                 ("graph", "graph", None)):
        mon = wt.WMTubeMonitor(n, dev, bundle, proposal_bound=pb, mode=mode, precision=prec,
                               action_dim=29)
        call = lambda: mon(p, wm_state=s, sim_ctx=c, active=active)  # noqa: E731
        try:
          for _ in range(2):
            v = call()
          times = []
          for _ in range(a.bench_reps):
            _sync(dev)
            t0 = time.perf_counter()
            v = call()
            _sync(dev)
            times.append(1e3 * (time.perf_counter() - t0))
        except Exception as e:                     # noqa: BLE001 - one mode failing is a result
          err = f"{type(e).__name__}: {e}"
          rows.append({"N": n, "precision": prec, "mode": name, "error": err})
          print(f"[bench] N {n:5d} {prec:4s} {name:12s} FAILED: {err}", flush=True)
          continue
        row = {"N": n, "precision": prec, "mode": name, "ms": statistics.median(times)}
        if ref is None:
          ref = v.clone()
        fin = torch.isfinite(v) & torch.isfinite(ref)
        d = (v - ref)[fin].abs()
        flips = ((v < 0) != (ref < 0))[fin].float()
        row.update(max_abs=float(d.max()) if d.numel() else 0.0,
                   p999_abs=float(torch.quantile(d, 0.999)) if d.numel() else 0.0,
                   flip_rate=float(flips.mean()) if d.numel() else 0.0)
        row["switch_ok"] = row["flip_rate"] <= MAX_FLIP and row["p999_abs"] <= MAX_P999
        rows.append(row)
        print(f"[bench] N {n:5d} {prec:4s} {name:12s} {row['ms']:8.2f} ms  max|d| "
              f"{row['max_abs']:.2e}  p99.9|d| {row['p999_abs']:.2e}  flips "
              f"{row['flip_rate']:.2e}  {'ok' if row['switch_ok'] else 'NO'}", flush=True)
  return {"rows": rows, "rule": {"max_flip_rate": MAX_FLIP, "max_p999_abs": MAX_P999},
          "gpu": torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu"}


def main(argv=None):
  a = _parse(argv)
  report = {"world": a.world, "seed": a.seed, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  if a.dump_layout:
    report["dump_layout"] = dump_layout(a)
  if a.task is not None:
    import eval_sonic_qcbf as eq
    import qcbf_eval_lib as ql
    report["live"] = live(a, eq, ql)
    if a.bench:
      report["bench"] = bench(a)
  elif a.bench:
    raise SystemExit("[smoke] --bench needs --task (the live states come from it)")
  os.makedirs(a.out, exist_ok=True)
  path = os.path.join(a.out, f"smoke_{a.world}"
                      + ("" if a.wm_thermal == "nn" else f"_{a.wm_thermal}") + ".json")
  with open(path, "w") as f:
    json.dump(report, f, indent=1, default=str)
  print(f"[smoke] all checks passed -> {path}")


if __name__ == "__main__":
  main()
