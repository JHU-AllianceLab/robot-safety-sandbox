"""Wall-clock cost of an H-step rollout in the G1 sim: what a rollout filter pays per check.

    python examples/bench_rollout_latency.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip

GPU (mjlab / MuJoCo Warp, the env every eval here runs): for each batch size N
the time of one control step -- the env step, SHAC's forward, SONIC's step
(reference window, encoder, decoder, history roll) -- and of a whole H-step
rollout of N envs in lockstep, driven by SHAC and by SONIC. Steps in time are
sequential, so that is the LATENCY of N parallel rollouts; envs that fall are
reset by the env and keep stepping (the cost of a step does not depend on it).

CPU (plain MuJoCo, ONE env, the same compiled model and solver options, one
thread): the physics of H control steps (``decimation`` substeps each) from
the GPU env's state, the actuators holding that pose, plus SHAC's and SONIC's
forward at batch 1 -- the onboard-computer view.

Prints a table and writes ``--out`` (json).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SonicPolicy, SonicReference, SonicTracker)
from robot_safety_sandbox.envs.g1_recover.states import KIND_MOCAP, load_bank  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="the fallback (SHAC / PPO final_model.zip)")
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"))
  p.add_argument("--sonic-dir", default=DEFAULT_SONIC_DIR)
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--num-envs", type=int, nargs="+", default=[1, 8, 64, 512, 4096])
  p.add_argument("--reps", type=int, default=3, help="timed repeats (the median is kept)")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs", "rollout_filter", "bench_latency.json"))
  p.add_argument("--device", default="cuda:0")
  return p.parse_args(argv)


def _sync(dev):
  if str(dev).startswith("cuda"):
    torch.cuda.synchronize(dev)


def timed(fn, n: int, reps: int, dev) -> float:
  """Median over ``reps`` of the wall time of ``n`` calls of ``fn`` (s)."""
  out = []
  for _ in range(reps):
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
      fn()
    _sync(dev)
    out.append(time.perf_counter() - t0)
  return statistics.median(out)


def gpu_case(a, N: int, sonic: SonicPolicy, base: SonicReference, lib: dict, H: int) -> dict:
  dev = a.device
  env = make_tensor(a.task, N, dev, cfg_overrides=dict(episode_length_s=600.0, rest_terminate=False),
                    random_push=dict(interval_s=(1.0, 2.0), duration_s=0.2,
                                     first_delay_s=(0.1, 0.4), horizontal=True, max_pushes=0))
  env.reset()
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)

  def obs():
    return torch.nan_to_num(env.obs_groups()[env.obs_key].float())

  def shac_act():
    with torch.no_grad():
      return torch.clamp(model.policy._predict(norm(obs()), deterministic=True), low, high)

  tracker = SonicTracker(env, sonic, base)
  pool = (lib["start_ok"] & (lib["kind"] == KIND_MOCAP)).nonzero(as_tuple=False).squeeze(-1)
  pick = pool[torch.arange(N) % pool.numel()]
  tracker.start(torch.ones(N, dtype=torch.bool, device=dev),
                lib["clip"][pick].to(dev), lib["frame"][pick].to(dev))
  a_fixed = shac_act()

  def env_step():
    env.step_tensor(a_fixed)

  def sonic_step():
    tracker.act()
    tracker.advance()

  def shac_rollout_step():
    env.step_tensor(shac_act())

  def sonic_rollout_step():
    env.step_tensor(tracker.act())
    tracker.advance()

  for fn in (env_step, shac_act, sonic_step, shac_rollout_step, sonic_rollout_step):
    for _ in range(10):                                        # warm-up
      fn()
  r = {"num_envs": N,
       "env_step_ms": 1e3 * timed(env_step, H, a.reps, dev) / H,
       "shac_fwd_ms": 1e3 * timed(shac_act, H, a.reps, dev) / H,
       "sonic_step_ms": 1e3 * timed(sonic_step, H, a.reps, dev) / H,
       "rollout_shac_s": timed(shac_rollout_step, H, a.reps, dev),
       "rollout_sonic_s": timed(sonic_rollout_step, H, a.reps, dev)}
  r["rollouts_per_s_shac"] = N / r["rollout_shac_s"]
  meta = {"physics_dt": float(env.mj.sim.mj_model.opt.timestep),
          "decimation": int(env.mj.cfg.decimation), "control_dt": float(env.mj.step_dt)}
  try:
    state = (env.mj.sim.data.qpos[0].detach().cpu().double().numpy(),
             env.mj.sim.data.qvel[0].detach().cpu().double().numpy())
  except Exception as e:                                       # noqa: BLE001
    print(f"[bench] could not read the GPU state for the CPU run ({e}); using qpos0")
    state = None
  cpu_model = env.mj.sim.mj_model
  # batch-1 inputs for the CPU forward passes
  x_shac = norm(obs())[:1].detach().cpu()
  x_dec = tracker.obs[:1].detach().cpu()
  env.close()
  return r, meta, cpu_model, state, x_shac, x_dec


def cpu_case(a, m, state, decimation: int, H: int, x_shac, x_dec) -> dict:
  import mujoco
  torch.set_num_threads(1)
  d = mujoco.MjData(m)
  if state is not None:
    d.qpos[:], d.qvel[:] = state
  mujoco.mj_forward(m, d)
  for i in range(m.nu):                                        # hold the pose
    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
      d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
  q0, v0 = d.qpos.copy(), d.qvel.copy()

  def physics_rollout():
    d.qpos[:], d.qvel[:] = q0, v0
    d.time = 0.0
    for _ in range(H * decimation):
      mujoco.mj_step(m, d)

  physics_rollout()
  t_phys = timed(physics_rollout, 1, a.reps, "cpu")
  model_cpu, _ = load_twin(a.policy, "cpu")
  model_cpu.policy.set_training_mode(False)
  sonic_cpu = SonicPolicy.from_onnx(a.sonic_dir).eval()

  def shac_fwd():
    with torch.no_grad():
      model_cpu.policy._predict(x_shac, deterministic=True)

  def sonic_dec():
    with torch.no_grad():
      sonic_cpu.decode(x_dec)

  pos, vel, anchor = torch.zeros(1, 10, 29), torch.zeros(1, 10, 29), torch.zeros(1, 10, 6)

  def sonic_enc():
    with torch.no_grad():
      sonic_cpu.encode(pos, vel, anchor)

  for fn in (shac_fwd, sonic_dec, sonic_enc):
    for _ in range(20):
      fn()
  return {"physics_rollout_s": t_phys, "physics_step_ms": 1e3 * t_phys / H,
          "nbody": int(m.nbody), "nv": int(m.nv), "ncon_end": int(d.ncon),
          "shac_fwd_ms": 1e3 * timed(shac_fwd, 200, a.reps, "cpu") / 200,
          "sonic_fwd_ms": 1e3 * (timed(sonic_dec, 200, a.reps, "cpu")
                                 + timed(sonic_enc, 200, a.reps, "cpu")) / 200}


def main(argv=None):
  a = _parse(argv)
  dev = a.device
  lib = load_bank(a.library, "cpu")
  lm = lib["meta"]
  csv_dir = lm["csv_dir"]
  if not os.path.isdir(csv_dir):
    csv_dir = os.path.join(_ZOO, "data", "lafan1_g1")
  base = SonicReference.build(csv_dir, lm["clips"], dev, library=lib)
  sonic = SonicPolicy.from_onnx(a.sonic_dir).to(dev).eval()
  gpu = torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu"
  rows, meta, cpu_in = [], None, None
  H = None
  for N in a.num_envs:
    if H is None:
      # the control step is the env's; build once to read it
      probe = make_tensor(a.task, 1, dev)
      H = int(round(a.horizon_s / float(probe.mj.step_dt)))
      probe.close()
    t0 = time.time()
    r, meta, m, state, x_shac, x_dec = gpu_case(a, N, sonic, base, lib, H)
    r["build_and_time_s"] = time.time() - t0
    rows.append(r)
    if cpu_in is None:
      cpu_in = (m, state, x_shac, x_dec)
    print(f"[bench] N {N:5d}: env step {r['env_step_ms']:.2f} ms, SHAC fwd {r['shac_fwd_ms']:.2f} ms, "
          f"SONIC step {r['sonic_step_ms']:.2f} ms | {H}-step rollout: SHAC {r['rollout_shac_s']:.2f} s, "
          f"SONIC {r['rollout_sonic_s']:.2f} s", flush=True)
  cpu = cpu_case(a, cpu_in[0], cpu_in[1], meta["decimation"], H, cpu_in[2], cpu_in[3])
  rep = {"gpu": gpu, "task": a.task, "horizon_s": a.horizon_s, "horizon_steps": H, **meta,
         "gpu_rows": rows, "cpu_1env": cpu, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  print(f"\n[bench] {gpu}; control dt {meta['control_dt']} s = {meta['decimation']} x "
        f"{meta['physics_dt']} s physics; a {a.horizon_s} s rollout = {H} control steps "
        f"({H * meta['decimation']} physics steps)")
  print(f"  {'N':>5s} {'env step':>9s} {'SHAC fwd':>9s} {'SONIC':>7s} {'rollout SHAC':>13s} "
        f"{'rollout SONIC':>14s} {'rollouts/s':>11s}")
  for r in rows:
    print(f"  {r['num_envs']:5d} {r['env_step_ms']:7.2f}ms {r['shac_fwd_ms']:7.2f}ms "
          f"{r['sonic_step_ms']:5.2f}ms {r['rollout_shac_s']:12.2f}s {r['rollout_sonic_s']:13.2f}s "
          f"{r['rollouts_per_s_shac']:11.0f}")
  print(f"  CPU MuJoCo, 1 env, 1 thread: {H}-step physics {cpu['physics_rollout_s']:.3f} s "
        f"({cpu['physics_step_ms']:.2f} ms per control step; {cpu['ncon_end']} contacts at the end); "
        f"SHAC fwd {cpu['shac_fwd_ms']:.3f} ms, SONIC fwd {cpu['sonic_fwd_ms']:.3f} ms")
  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, "w") as f:
    json.dump(rep, f, indent=1)
  print(f"[bench] wrote {a.out}")


if __name__ == "__main__":
  main()
