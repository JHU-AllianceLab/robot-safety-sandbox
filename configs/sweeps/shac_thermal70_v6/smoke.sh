#!/bin/bash
# Hybrid Beat-It on sysID v0.6, step 2: smoke of the HYBRID world model before the long run.
#  1. the analytic heat block vs the simulator, one control step from REAL states (true next
#     joint states, so only the torque rebuild and the heat model are tested), and the head
#     targets (fall margin / fall termination apart from the overheat);
#  2. a short SHAC run with --wm-hybrid at 4096 envs (throughput for the long run's estimate);
#  3. the checkpoint reloads and its world model rebuilds the analytic block without the env.
# Needs the bank of prep.sh. safety_sb3 from ~/ssb-wt-hybrid (PYTHONPATH).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h100,l40s,a100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=01:30:00 --job-name=v6_smoke --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70_v6/smoke.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH=$HOME/ssb-wt-hybrid${PYTHONPATH:+:$PYTHONPATH}
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import safety_sb3; print('[smoke] safety_sb3 from', safety_sb3.__file__)" | tee /dev/stderr | grep -q ssb-wt-hybrid
CFG=configs/g1_recover_sonic_thermal70_v6_hybrid_safetyshac1p.yaml
S=runs/shac_thermal70_v6/smoke
rm -rf $S && mkdir -p $S

# 1. the heat block vs the simulator
python - <<'PY'
import torch, yaml
from robot_safety_sandbox import make_tensor
from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
from robot_safety_sandbox.filters.hybrid_thermal_model import HybridThermalModel
cfg = yaml.safe_load(open("configs/g1_recover_sonic_thermal70_v6_hybrid_safetyshac1p.yaml"))
env = make_tensor(cfg["task"], 512, "cuda:0", cfg_overrides=cfg["env_overrides"], critic_obs_key="critic")
spec = env.analytic_model_spec()
blk = HybridThermalModel(**spec["config"]).to("cuda:0")
st = thermal_state(env.mj)
print("[smoke] critic", env.critic_observation_space.shape, "| block owns", len(blk.index), "dims",
      "| bias ctx cols", sum(i >= 0 for i in spec["config"]["bias_ctx_idx"]), "of 29")
obs = env.reset()
g_ = torch.Generator(device="cuda:0").manual_seed(0)
errs, rel, n_oh, n_fall, n_bad = [], [], 0, 0, 0
for k in range(150):
  s0 = env.critic_obs().clone()
  c = env.sim_context()
  a = 0.5 * torch.randn(512, env.action_space.shape[0], device="cuda:0", generator=g_)
  obs, g, dones, timeouts, _ = env.step_tensor(a)
  g_head, term_head = env.wm_head_targets()
  assert bool((g_head >= g - 1e-5).all()), "the fall margin is below the full margin"
  tm = env.mj.termination_manager
  oh = tm.get_term("motor_overheat").bool() if "motor_overheat" in tm.active_terms else torch.zeros_like(dones)
  n_oh += int((oh & ~term_head).sum()); n_fall += int(term_head.sum())
  n_bad += int((term_head & ~(dones & ~timeouts)).sum())
  s1 = env.critic_obs()
  ok = ~dones.bool()
  if ok.any():
    pred = blk.advance(s0[ok], s1[ok], a[ok], c[ok])
    true = torch.cat([s1[ok][:, blk.tw_idx], s1[ok][:, blk.th_idx], s1[ok][:, blk.ts_idx]], -1)
    d_true = s1[ok][:, blk.tw_idx] - s0[ok][:, blk.tw_idx]
    d_pred = pred[:, :29] - s0[ok][:, blk.tw_idx]
    errs.append((pred - true).abs().amax(-1))
    big = d_true.abs() > 1e-3
    if big.any():
      rel.append((d_pred[big] / d_true[big]))
e = torch.cat(errs); r = torch.cat(rel)
print(f"[smoke] one-step |T_pred - T_sim| (max over 87 dims): median {e.median():.2e} C, p99 "
      f"{e.quantile(0.99):.2e}, max {e.max():.2e} | winding increment ratio pred/sim median "
      f"{r.median():.3f}, p10 {r.quantile(0.1):.3f}, p90 {r.quantile(0.9):.3f}")
print(f"[smoke] head targets: {n_fall} fall terminations, {n_oh} overheat-only (not head "
      f"terminations), {n_bad} head terminations outside an episode end")
assert n_bad == 0
assert 0.9 < float(r.median()) < 1.1, "the heat block's winding increment departs from the sim's"
# eager vs compiled: same numbers, and the time of an imagination's worth of steps (fwd + bwd)
import os, time
B = 4096
S0 = s0[:1].expand(B, -1).clone(); C = c[:1].expand(B, -1).clone()
def run(compiled):
  os.environ["ZOO_HYBRID_COMPILE"] = "1" if compiled else "0"
  A = (0.5 * torch.randn(B, a.shape[1], device="cuda:0", generator=torch.Generator(device="cuda:0").manual_seed(1))).requires_grad_(True)
  s = S0
  outs = []
  for _ in range(32):
    t = blk.advance(s, s, A, C)
    s = s.index_copy(-1, torch.as_tensor(blk.index, device="cuda:0"), t)
    outs.append(t)
  torch.stack(outs).sum().backward()
  return torch.stack(outs).detach(), A.grad.detach()
for compiled in (False, True):
  out, gr = run(compiled)          # warm-up / compile
  torch.cuda.synchronize(); t0 = time.time()
  for _ in range(5):
    out, gr = run(compiled)
  torch.cuda.synchronize()
  print(f"[smoke] heat block x32 steps fwd+bwd, B={B}, compiled={compiled}: {(time.time() - t0) / 5 * 1e3:.1f} ms")
  if compiled:
    print(f"[smoke] compiled vs eager: max |dT| {(out - ref_out).abs().max():.2e} C, max |dgrad| {(gr - ref_gr).abs().max():.2e}")
    assert (out - ref_out).abs().max() < 1e-3
  else:
    ref_out, ref_gr = out, gr
os.environ["ZOO_HYBRID_COMPILE"] = "1"
PY

# 2. a short hybrid SHAC run at 4096 envs
python examples/train.py --config $CFG --seed 0 --out $S/train --num-envs 4096 --steps 5000000 \
  --wm-warmup-iters 50 --actor-delay-iters 10 --lr-decay-iters 600 --checkpoint-every 1000000000 \
  --video-interval 1000000000 --no-wandb 2>&1 | tee $S/train.log | grep -v "linesearch" | tail -80
grep -q "state = the 'critic' group (200 dims); actor obs 127 dims" $S/train.log \
  || { echo "[smoke] FAILED: the world-model state is not the 200-D critic group / actor 127-D"; exit 1; }
grep -q "g_head_min" $S/train.log || { echo "[smoke] FAILED: no hybrid imagination logs"; exit 1; }
P=$S/train/g1_recover_thermal70_v6_safetyshac1p/final_model.zip
[[ -f $P ]] || { echo "[smoke] FAILED: no $P"; exit 1; }

# 3. the checkpoint rebuilds the analytic block without the env
python - <<PY
import torch
from safety_sb3 import SafetySHAC1P
m = SafetySHAC1P.load("$P", device="cuda:0")
wm = m.world_model
assert wm.hybrid, "the loaded world model has no analytic block"
print("[smoke] reloaded: hybrid", wm.hybrid, "| analytic dims", int(wm.analytic_index.numel()),
      "| sysID", wm.analytic.sysid_version)
z = torch.zeros(8, wm.state_dim, device="cuda:0")
a = torch.zeros(8, wm.action_dim, device="cuda:0")
cn = torch.zeros(8, wm.context_dim, device="cuda:0") if wm.context_dim else None
st = wm.step(z, a, cn)
assert st.g_head is not None and torch.isfinite(st.g).all()
PY
echo "[smoke] PASSED"
