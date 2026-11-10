"""GPU check of the thermal task on the MuJoCo 3.16 stack (run from the mjc316 worktree)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import robot_safety_sandbox  # noqa: F401,E402
from robot_safety_sandbox.registry import make_tensor  # noqa: E402

BANK = "data/g1_sonic_push_gfall_arch_train_bank_net.pt"   # the old bank, only to exercise the env
OV = dict(dataset=BANK, weights={"handover": 1.0, "nominal": 0.0, "clean": 0.0, "pushed": 0.0,
                                 "fell": 0.0, "synthetic": 0.0, "mocap": 0.0},
          sampler="uniform", episode_length_s=3.0, rest_terminate=True, rest_dwell_s=0.4,
          rest_arm_s=0.8)


def run(task, n=1024, steps=150):
  print(f"\n===== {task} ({n} envs)", flush=True)
  t0 = time.time()
  env = make_tensor(task, n, "cuda:0", critic_obs_key="critic", cfg_overrides=dict(OV))
  print(f"built in {time.time() - t0:.0f}s", flush=True)
  mj = env.mj
  env.reset()
  obs = env.obs_groups()[env.obs_key]
  crit = env.critic_obs()
  spec = env.world_model_spec()
  ctx = env.sim_context()
  print("actor obs", tuple(obs.shape), "critic", tuple(crit.shape), "wm actor idx",
        len(spec.actor_obs_index), "last_action", spec.last_action_index is not None,
        "sim context", None if ctx is None else tuple(ctx.shape))
  print("actor terms", mj.observation_manager.active_terms[env.obs_key])
  print("critic terms", mj.observation_manager.active_terms["critic"])
  print("terminations", mj.termination_manager.active_terms)
  st = None
  if task.endswith("thermal"):
    from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
    st = thermal_state(mj)
    tw = st.Tw
    print(f"start: hottest {tw.amax(1).mean():.1f} (min {tw.amax(1).min():.1f}, max "
          f"{tw.amax(1).max():.1f}); median motor {tw.median(1).values.mean():.1f}; "
          f"gap Tw-Th mean {(st.Tw - st.Th).mean():.1f}", flush=True)
    # critic twin equals the state
    lay = env._obs_layout("critic")
    s0, sz, _ = lay["motor_temp"]
    assert torch.allclose(crit[:, s0:s0 + sz], st.Tw), "critic motor_temp != Tw"
    a0, asz, _ = env._obs_layout(env.obs_key)["motor_temp"]
    print("actor motor_temp - Tw (noise):", float((obs[:, a0:a0 + asz] - st.Tw).abs().max()))
    tw0 = st.Tw.clone()
  act = torch.zeros(n, env.action_space.shape[0], device="cuda:0")
  done_n, fails, gsum = 0, {}, []
  torch.cuda.synchronize()
  t1 = time.time()
  for k in range(steps):
    obs, g, dones, trunc, _l = env.step_tensor(act)
    done_n += int(dones.sum())
    tm = mj.termination_manager
    for name in tm.active_terms:
      fails[name] = fails.get(name, 0) + int(tm.get_term(name).sum())
    gsum.append(float(g.mean()))
    if st is not None and k in (0, 24, 49, 99, 149):
      live = ~dones
      rise = (st.Tw - tw0)[live]
      print(f"  t={0.02 * (k + 1):.2f}s hottest {st.Tw.amax(1).mean():.2f} | rise in non-reset "
            f"envs mean {rise.mean():.3f} max {rise.max():.2f} C | g mean {float(g.mean()):.3f}",
            flush=True)
  torch.cuda.synchronize()
  dt = time.time() - t1
  print(f"{steps} steps x {n} envs: {n * steps / dt:.0f} env-steps/s; dones {done_n}; "
        f"termination counts {fails}; g mean {sum(gsum) / len(gsum):.3f}")
  log = mj.extras.get("log", {})
  print("metrics:", {k: round(float(v), 2) for k, v in log.items() if "Metrics" in k})
  if st is not None:
    import numpy as np
    names = st.names
    rise = (st.Tw - tw0).mean(0).cpu().numpy()
    order = np.argsort(rise)[::-1][:6]
    print("mean winding rise by motor (top 6; resets included):",
          {names[i]: round(float(rise[i]), 2) for i in order})
  env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
  for task in sys.argv[1:] or ["g1_recover_noheight", "g1_recover_thermal"]:
    run(task)
