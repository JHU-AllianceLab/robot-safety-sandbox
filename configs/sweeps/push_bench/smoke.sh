#!/bin/bash
# Smoke test of the SONIC push benchmark rebuilt in mjlab: every stage at toy scale.
#   sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
#       --job-name=push_bench_smoke --output=logs/slurm/%x_%j.out configs/sweeps/push_bench/smoke.sh
# FROM=N (sbatch --export=ALL,FROM=5 ...) reruns stages N.. on the outputs of an earlier run.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
S=runs/push_bench_smoke
SON="--sonic-dir $HOME/gear_sonic_hf"
PPO=$S/ppo/g1_recover_bench/final_model.zip
SHAC=$S/shac/g1_recover_bench_safetyshac1p/final_model.zip
FROM=${FROM:-1}
[[ $FROM -le 1 ]] && rm -rf $S
mkdir -p $S

if [[ $FROM -le 1 ]]; then
echo "[smoke] 1. the world: dimensions, physics, friction, the point push"
python - <<'PY'
import math, types, torch, mujoco
import sys; sys.path.insert(0, "examples")
from harvest_g1_sonic_trajectories import make_env
from robot_safety_sandbox.envs.g1_recover import push_bench as pb
env = make_env(types.SimpleNamespace(task="g1_recover_bench", hip="sonic", num_envs=64, device="cuda:0"),
               dict(episode_length_s=10.0, rest_terminate=False), None)
mj = env.mj; m = mj.sim.mj_model
og = env.obs_groups()
print("[smoke] actor", tuple(og["actor"].shape), "critic", tuple(og["critic"].shape),
      "step_dt", mj.step_dt, "opt", m.opt.timestep, m.opt.iterations, m.opt.tolerance, m.opt.ccd_iterations,
      "integrator", m.opt.integrator, "mass", float(m.body_mass.sum()))
assert og["actor"].shape[1] == 70 and abs(mj.step_dt - 0.02) < 1e-9 and m.opt.iterations == 100
feet = [g for g in range(m.ngeom) if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
ground = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
print("[smoke] foot friction", sorted({float(m.geom_friction[g, 0]) for g in feet}), "priority",
      sorted({int(m.geom_priority[g]) for g in feet}), "ground", [float(m.geom_friction[g, 0]) for g in ground])
assert all(abs(m.geom_friction[g, 0] - 1.0) < 1e-9 for g in feet)
print("[smoke] events", list(mj.cfg.events), "terminations", list(mj.cfg.terminations))
print("[smoke] sim_context", env.sim_context())
robot = mj.scene["robot"]
pp = pb.PointPush(robot, env.device)
n = env.num_envs
ids = torch.arange(n, device=env.device)
f = torch.zeros(n, 3, device=env.device); f[:, 0] = 100.0
pt = torch.zeros(n, 3, device=env.device); pt[:, 2] = 0.1
pp.start(ids < 8, torch.full((n,), 1, device=env.device), pt, f, torch.full((n,), 2, device=env.device))
pp.apply()
x = env.mj.sim.data.xfrc_applied
torso = robot.indexing.body_ids[pp.body_ids[1]]
d = robot.data
lever = d.body_link_pos_w[0, pp.body_ids[1]] + torch.tensor([0., 0., 0.]).to(env.device)
print("[smoke] xfrc torso env0", x[0, torso].tolist(), "env9", x[9, torso].tolist())
assert torch.allclose(x[0, torso, :3], f[0]) and float(x[9, torso].abs().sum()) == 0.0
a = torch.zeros(n, 29, device=env.device)
for _ in range(3):
  pp.apply(); env.step_tensor(a); pp.tick()
pp.apply()
assert float(env.mj.sim.data.xfrc_applied.abs().sum()) == 0.0, "the push did not end"
print("[smoke] point push ok")
env.close()
PY

fi
if [[ $FROM -le 2 ]]; then
echo "[smoke] 2. tables and the training clip library"
python examples/subset_clip_library.py --clips walk1_subject1 run1_subject2 dance1_subject1 --end-s dance1_subject1=105 --out $S/clips.pt
python examples/make_push_table.py --preset calib --out $S/calib_table

fi
if [[ $FROM -le 3 ]]; then
echo "[smoke] 3. harvest (force push) and its bank"
python examples/harvest_g1_sonic_trajectories.py --task g1_recover_bench --library $S/clips.pt $SON --push force --push-ticks 5 15 --dv-range 0.5 2.5 --onset-s 1.0 2.0 --episodes 2048 --num-envs 512 --seed 0 --out $S/harvest.pt
python - <<PY
import torch
r = torch.load("$S/harvest.pt", weights_only=False); b = torch.load("$S/harvest_bank.pt", weights_only=False)
ps = r["ep/push_steps"]; print("[smoke] push ticks", int(ps.min()), int(ps.max()), "sites", torch.bincount(r["ep/push_site"]).tolist(),
      "dv", float(r["ep/dv"].min()), float(r["ep/dv"].max()), "survival", float((~r["ep/fell"]).float().mean()))
assert int(ps.min()) >= 5 and int(ps.max()) <= 15 and int(r["ep/push_site"].min()) >= 0
since_end = b["handover_delay_s"] - b["push_duration_s"]
print("[smoke] bank rows", int(b["kind"].numel()), "delays after the push end", sorted(set(round(float(x), 2) for x in since_end)))
assert set(round(float(x), 2) for x in since_end) <= {0.0, 0.1, 0.2, 0.3}
assert float(b["push_left_steps"].abs().sum()) == 0
PY

fi
if [[ $FROM -le 4 ]]; then
echo "[smoke] 4. tiny trainings"
python examples/train.py --config configs/g1_recover_bench_safetyppo1p.yaml --env-override dataset=$S/harvest_bank.pt --num-envs 512 --steps 1000000 --no-wandb --out $S/ppo
python examples/train.py --config configs/g1_recover_bench_safetyshac1p.yaml --env-override dataset=$S/harvest_bank.pt --num-envs 512 --steps 400000 --wm-warmup-iters 20 --actor-delay-iters 5 --learning-starts 8 --lr-decay-iters 400 --imagine-batch 512 --replay-size 100 --no-wandb --out $S/shac
ls -la $PPO $SHAC

fi
if [[ $FROM -le 5 ]]; then
echo "[smoke] 5. the CPU rollout against the live env (observation, one step, rollouts; K margin)"
python examples/validate_cpu_rollout.py --policy $PPO --task g1_recover_bench --hip sonic --num-envs 64 --rounds 2 --horizon-s 1.0 --workers 8 --out $S/validate.json
python -c "import json; r = json.load(open('$S/validate.json')); print('[smoke] validate', json.dumps({k: r[k] for k in ('obs', 'one_step', 'rollout') if k in r})[:1500])"

fi
if [[ $FROM -le 6 ]]; then
echo "[smoke] 6. the benchmark: SafePPO (exact-sim filter), SHAC (calibration, tube, filter)"
python examples/eval_push_bench.py --table data/zz_test/trials.csv --sample 96 --policy $PPO --tag smoke_ppo $SON --conditions sonic fallback rollout:1 --rollout-workers 12 --num-envs 128 --out $S/eval_ppo
python examples/eval_push_bench.py --table $S/calib_table/trials.csv --sample 128 --no-controls --seed 1 --policy $SHAC --tag smoke_shac $SON --conditions wmcalib:clean --num-envs 128 --out $S/calib_shac
python examples/calibrate_wm_min_tube.py --calib $S/calib_shac/bench.pt --tag smoke_shac
python examples/eval_push_bench.py --table data/zz_test/trials.csv --sample 96 --policy $SHAC --tag smoke_shac $SON --conditions sonic fallback wmmin:1:0.95 wmmin:1:0.95+latch --tube-calib $S/calib_shac/min_tube.json --num-envs 128 --out $S/eval_shac
python - <<PY
import numpy as np, json
for d in ("$S/eval_ppo/rollout_1", "$S/eval_shac/wmmin_1_0.95", "$S/eval_shac/fallback"):
  z = np.load(d + "/trials.npz"); n = len(z["dv"])
  print("[smoke]", d, {k: z[k].shape for k in ("exit_tick", "warm_fail", "triggers", "mode")},
        "sonic surv", float((z["exit_tick"][:n] < 0).mean()), "arm surv", float((z["exit_tick"][n:] < 0).mean()))
print(json.dumps(json.load(open("$S/eval_shac/summary.json"))["conditions"], indent=1)[:2500])
PY
fi
echo "[job] smoke ok"
