#!/bin/bash
# Hybrid Beat-It on sysID v0.6, step 1: check the v0.6 robot / world, then the start bank
# (data_jobs.txt). Conda env mjlab316 (MuJoCo 3.16).
#   sbatch -A hhu49_alliance1 --comment=accept_cost -p rtx6000,h100,l40s,a100 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=02:00:00 --job-name=v6_prep --output=logs/slurm/%x_%j.out configs/sweeps/shac_thermal70_v6/prep.sh
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate "${ZOO_CONDA_ENV:-mjlab316}"
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1
NV_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f $NV_EGL ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=$NV_EGL
echo "[slurm] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 0. the v0.6 robot / world vs the v0.4 one
python - <<'PY'
import warnings
warnings.simplefilter("error", DeprecationWarning)   # a sim_heat_scale request on v0.6 must not slip by
import torch
from robot_safety_sandbox.base import MjlabTensorSafetyEnv
from robot_safety_sandbox.registry import CUMULATIVE, spec
from robot_safety_sandbox.envs.assets import g1_sysid
from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
for v in ("0.4.0", "0.6.0"):
  print("[check] provenance", {k: x for k, x in g1_sysid.provenance(v).items() if k != "sha256"})
out = {}
for task in ("g1_recover_thermal70", "g1_recover_thermal70_v6", "g1_recover_lafan_sysid_v6"):
  s = spec(task)
  kw = dict(s.kwargs); kw.setdefault("dense_reward", s.mode == CUMULATIVE)
  env = MjlabTensorSafetyEnv(8, "cuda:0", cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
                             ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=False,
                             end_criterion=s.end_criterion,
                             critic_obs_key=None if "sysid_v6" in task else "critic", **kw)
  obs = env.reset()
  m = env.mj.sim.mj_model
  mass = float(m.body_mass.sum())
  crit = env.critic_obs()
  r = dict(actor=int(obs.shape[1]), critic=None if crit is None else int(crit.shape[1]),
           mass_kg=round(mass, 3), ntendon=int(m.ntendon),
           frictionloss_sum=round(float(m.dof_frictionloss.sum()), 3))
  try:
    st = thermal_state(env.mj)
    r.update(sysid=st.cfg.sysid_version, sim_heat_scale=st.cfg.sim_heat_scale,
             a_ankle_roll=float(st.th.a[5]))
  except Exception as e:   # the harvest twin has no thermal state
    r.update(thermal=None)
  for _ in range(25):
    obs, *_ = env.step_tensor(torch.zeros(8, s.ctrl_dim, device="cuda:0"))
  if crit is not None and r.get("sysid"):
    st = thermal_state(env.mj)
    lay = env._obs_layout("critic")
    r["critic_terms_tail"] = [k for k in lay][-3:]
    r["Tw_max_after_0p5s"] = float(st.Tw.amax())
  print(f"[check] {task}: {r}", flush=True)
  out[task] = r
  env.close() if hasattr(env, "close") else None
  del env
assert out["g1_recover_thermal70"]["critic"] == 171 and out["g1_recover_thermal70"]["actor"] == 127
assert out["g1_recover_thermal70_v6"]["critic"] == 200 and out["g1_recover_thermal70_v6"]["actor"] == 127
assert abs(out["g1_recover_thermal70_v6"]["mass_kg"] - 37.40) < 0.05, "v0.6 mass is not the weighed 37.40 kg"
assert out["g1_recover_thermal70_v6"]["sim_heat_scale"] is False
assert out["g1_recover_thermal70"]["sim_heat_scale"] is True
assert out["g1_recover_thermal70_v6"]["a_ankle_roll"] < out["g1_recover_thermal70"]["a_ankle_roll"]
print("[check] PASSED", flush=True)
PY

# 1. the bank (skipped when it exists)
examples/run_jobs.sh configs/sweeps/shac_thermal70_v6/data_jobs.txt
python - <<'PY'
import torch
b = torch.load("data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt", map_location="cpu", weights_only=False)
m = b["meta"]
print("[prep] bank rows", m.get("rows"), "eta_net", {k: v for k, v in m.get("eta_net", {}).items()
      if k in ("eta", "kept", "per_delay", "per_category")})
fell = b.get("fell")
if fell is not None:
  print("[prep] SONIC fell in", float(fell.float().mean()), "of rows")
PY
echo "[prep] DONE"
