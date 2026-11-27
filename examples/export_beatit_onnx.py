"""Export a Beat-It safety filter (fallback + 1 s monitor) as ONNX graphs, filter only.

    MUJOCO_GL=egl PYTHONPATH=~/ssb-wt-hybrid python examples/export_beatit_onnx.py \\
        --run-dir runs/v061_study/beatit_thermal_s0/g1_recover_thermal70_v61_safetyshac1p \\
        --tube runs/v061_study/eval/calib/beatit_thermal/min_tube.json --tag beatit_thermal_s0 \\
        --threshold 0.665 --out <folder>

Graphs (batch 1, opset 18, float32):
  beatit_fallback.onnx  fb_obs [1, F] -> actions [1, A]
      the fallback's deterministic action (obs normalizer inside; its last-action columns clamped
      to the fallback box first; output clamped to +- fallback_bound), Beat-It's action units.
  beatit_monitor.onnx   fb_state [1, S], context [1, C], proposal [1, A]
                        -> margin_min [1], survival [1], tw_max_c [1, 29]
      one noise-free 1 s imagined rollout: the proposal (clamped to +- proposal_bound) for one
      step, then the fallback, in the world model (its analytic SysID heat block inside).
      margin_min = the lowest predicted margin (a step is pushed to <= -0.1 once the predicted
      P(terminated) passes 0.5), survival = predicted P(alive after 1 s), tw_max_c = the hottest
      predicted winding per motor (fb_state's motor_temp order). Reject when margin_min < threshold.

The run's OWN training env (its config.yaml: task + env_overrides) gives the layouts, so a student
world model with a smaller state (e.g. no foot terms, 188-D) exports the same way.

Parity: N envs of that env, the fallback driving from the task's start bank (pushed hand-over
states, hot starts); every step the proposals are the fallback's action, half the envs with
N(0, sigma) added. The deploy modules are checked against the sandbox's calibrated monitor
(filters/wm_tube.WMTubeMonitor, fp32) and recorded to parity.npz for check_onnx.py (onnxruntime).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)


def sha256(p: str) -> str:
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def git_rev(path: str) -> dict:
  try:
    rev = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    br = subprocess.check_output(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", path, "status", "--porcelain", "-uno"], text=True).strip())
    return {"path": path, "commit": rev, "branch": br, "dirty": dirty}
  except Exception as e:  # noqa: BLE001
    return {"path": path, "error": str(e)}


class FallbackDeploy(nn.Module):
  def __init__(self, bundle):
    super().__init__()
    self.b = bundle
    am = bundle.model._actor_mean
    self.actor_mean = am if isinstance(am, nn.Module) else nn.Identity()

  def forward(self, fb_obs):
    return self.b.pi(fb_obs)


class MonitorDeploy(nn.Module):
  """wm_tube.WMTubeMonitor._rollout (the "nn" path: a natively hybrid world model carries its
  heat block inside step()), with the proposal clamp and the state sanitation, no in-place writes."""

  def __init__(self, bundle, proposal_bound: float, tw_cols: torch.Tensor):
    super().__init__()
    self.b, self.wm = bundle, bundle.wm
    am = bundle.model._actor_mean
    self.actor_mean = am if isinstance(am, nn.Module) else nn.Identity()
    self.H = int(bundle.horizon)
    A = int(bundle.wm.action_dim)
    dev = bundle.device
    self.register_buffer("p_lo", torch.full((A,), -float(proposal_bound), device=dev))
    self.register_buffer("p_hi", torch.full((A,), float(proposal_bound), device=dev))
    self.register_buffer("tw", tw_cols.long().clone())

  def forward(self, fb_state, context, proposal):
    b, wm = self.b, self.wm
    s = torch.nan_to_num(fb_state)
    act = torch.clamp(proposal, self.p_lo, self.p_hi)
    cn = wm.encode_context(context)
    z = wm.encode(s)
    alive = torch.ones_like(s[:, 0])
    tmax = s.index_select(-1, self.tw)
    gmin = None
    for t in range(self.H):
      st = wm.step(z, act, cn, z_clip=b.z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      g = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      gmin = g if gmin is None else torch.minimum(gmin, g)
      dec = wm.decode(z)
      tmax = torch.maximum(tmax, dec.index_select(-1, self.tw))
      if t + 1 < self.H:
        act = b.pi(dec.index_select(-1, b.actor_idx))
    return gmin.reshape(-1), alive.reshape(-1), tmax


def layout_of(om, group: str):
  out, s = [], 0
  for name, dims, cfg in zip(om.active_terms[group], om.group_obs_term_dim[group],
                             om._group_obs_term_cfgs[group]):
    w = int(np.prod(dims))
    noise = getattr(cfg, "noise", None)
    nz = [float(noise.n_min), float(noise.n_max)] if noise is not None and hasattr(noise, "n_min") else None
    params = {k: v for k, v in (cfg.params or {}).items() if isinstance(v, (int, float, str, bool))}
    out.append({"name": name, "start": s, "width": w, "func": getattr(cfg.func, "__name__", str(cfg.func)),
                "params": params, "scale": None if cfg.scale is None else (
                  cfg.scale if isinstance(cfg.scale, (int, float)) else str(cfg.scale)),
                "train_noise_uniform": nz})
    s += w
  return out, s


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run-dir", required=True, help="the directory holding final_model.zip + config.yaml")
  p.add_argument("--tube", required=True, help="its min_tube.json")
  p.add_argument("--tag", required=True)
  p.add_argument("--threshold", type=float, default=None,
                 help="the Wang-calibrated reject threshold on margin_min (None: not calibrated)")
  p.add_argument("--threshold-note", default="")
  p.add_argument("--what", default="")
  p.add_argument("--out", required=True)
  p.add_argument("--proposal-bound", type=float, default=6.5)
  p.add_argument("--fallback-bound", type=float, default=3.0)
  p.add_argument("--dwell-s", type=float, default=1.0)
  p.add_argument("--envs", type=int, default=8)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--sigma", type=float, default=1.5, help="proposal noise on half the envs")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  import yaml
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.filters import wm_tube as wt

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.manual_seed(a.seed)
  dev = a.device
  run_dir = os.path.abspath(a.run_dir)
  out = os.path.abspath(a.out)
  os.makedirs(out, exist_ok=True)
  cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
  task, ov = cfg["task"], dict(cfg.get("env_overrides") or {})
  print(f"[export] {run_dir}\n[export] task {task}, env_overrides {ov}", flush=True)

  env = make_tensor(task, a.envs, dev, cfg_overrides=ov, critic_obs_key="critic")
  env.reset()
  base = wt._base(env)
  mj = base.mj
  dt = float(mj.step_dt)
  bundle = wt.load_wm_tube(run_dir, os.path.abspath(a.tube), tag=a.tag, cond="wmmin:1:0.95",
                           device=dev, fallback_bound=a.fallback_bound, step_dt=dt, wm_thermal="nn")
  wm = bundle.wm
  print(f"[export] world model: state {wm.state_dim}, action {wm.action_dim}, context {wm.context_dim}, "
        f"hybrid {wm.hybrid}; H {bundle.horizon}, tube eps {bundle.eps:.4f}", flush=True)
  binding = wt.check_binding(bundle, env)
  print(f"[export] check_binding ok: {sorted(binding)[:6]}", flush=True)
  ref = wt.WMTubeMonitor(a.envs, dev, bundle, proposal_bound=a.proposal_bound, mode="compact",
                         precision="fp32", action_dim=int(wm.action_dim))

  om = mj.observation_manager
  fb_obs_lay, F_dim = layout_of(om, "actor")
  fb_state_lay, S_dim = layout_of(om, "critic")
  if S_dim != int(wm.state_dim) or F_dim != bundle.actor_dim:
    raise SystemExit(f"layout {F_dim}/{S_dim} vs bundle {bundle.actor_dim}/{wm.state_dim}")
  mt = [t for t in fb_state_lay if t["name"] == "motor_temp"]
  if len(mt) != 1 or mt[0]["width"] != 29:
    raise SystemExit(f"no 29-wide motor_temp term in the critic group: {[t['name'] for t in fb_state_lay]}")
  tw_cols = torch.arange(mt[0]["start"], mt[0]["start"] + 29, device=dev)
  pg = [t for t in fb_obs_lay if t["name"] == "projected_gravity"]
  gz_col = pg[0]["start"] + 2 if pg else None

  fb = FallbackDeploy(bundle).eval()
  md = MonitorDeploy(bundle, a.proposal_bound, tw_cols).eval()
  A, C_dim = int(wm.action_dim), int(wm.context_dim)

  # --- parity episode
  noisy = (torch.arange(a.envs, device=dev) % 2 == 1).float().unsqueeze(-1)
  keys = ("fb_obs", "fb_state", "context", "proposal", "a_fallback", "ref_margin_min", "ref_survival",
          "margin_min", "survival", "tw_max_c", "ref_verdict")
  R = {k: [] for k in keys}
  errs = {"fallback": 0.0, "margin_vs_ref": 0.0, "survival_vs_ref": 0.0, "verdict_vs_ref_call": 0.0}
  t0 = time.time()
  with torch.no_grad():
    for k in range(a.steps):
      og = base.obs_groups()
      F = og["actor"].float()
      S = og["critic"].float()
      Cx = base.sim_context()
      a_fb = bundle.pi(F)
      prop = a_fb + noisy * a.sigma * torch.randn_like(a_fb)
      with wt.matmul_precision("fp32"):
        r_out, r_alive = ref._rollout(torch.nan_to_num(S), Cx,
                                      torch.clamp(prop, ref.p_lo, ref.p_hi), ref.horizon)
      r_min = r_out.amin(dim=1)
      verdict = ref(prop, wm_state=S, sim_ctx=Cx)            # = pred_min - eps
      mm, sv, tw = md(S, Cx, prop)
      a_dep = fb(F)
      errs["fallback"] = max(errs["fallback"], float((a_dep - a_fb).abs().max()))
      errs["margin_vs_ref"] = max(errs["margin_vs_ref"], float((mm - r_min).abs().max()))
      errs["survival_vs_ref"] = max(errs["survival_vs_ref"], float((sv - r_alive).abs().max()))
      errs["verdict_vs_ref_call"] = max(errs["verdict_vs_ref_call"],
                                        float((verdict - (r_min - ref.eps)).abs().nan_to_num(0).max()))
      for name, v in (("fb_obs", F), ("fb_state", S), ("context", Cx), ("proposal", prop),
                      ("a_fallback", a_fb), ("ref_margin_min", r_min), ("ref_survival", r_alive),
                      ("margin_min", mm), ("survival", sv), ("tw_max_c", tw), ("ref_verdict", verdict)):
        R[name].append(v.detach().float().cpu())
      env.step_tensor(a_fb)
  print(f"[export] parity: {a.steps} steps x {a.envs} envs in {time.time() - t0:.0f} s; deploy modules vs "
        f"the sandbox monitor (max abs): { {k: f'{v:.2e}' for k, v in errs.items()} }", flush=True)
  if errs["margin_vs_ref"] > 1e-4 or errs["fallback"] > 1e-5:
    raise SystemExit("[export] the deploy modules do not reproduce the sandbox monitor")
  R = {k: torch.stack(v, 1).numpy() for k, v in R.items()}   # (E, K, ...)
  thr = a.threshold if a.threshold is not None else float(bundle.eps)
  acc = R["ref_margin_min"] >= thr
  print(f"[export] parity accept rate at {thr:.4f}: clean envs {acc[0::2].mean():.3f}, noisy envs "
        f"{acc[1::2].mean():.3f}", flush=True)

  # --- export
  z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
  ex = dict(export_params=True, opset_version=18, dynamic_axes={}, dynamo=False)
  torch.onnx.export(fb, (z(1, F_dim),), os.path.join(out, "beatit_fallback.onnx"),
                    input_names=["fb_obs"], output_names=["actions"], **ex)
  torch.onnx.export(md, (z(1, S_dim), z(1, C_dim), z(1, A)), os.path.join(out, "beatit_monitor.onnx"),
                    input_names=["fb_state", "context", "proposal"],
                    output_names=["margin_min", "survival", "tw_max_c"], **ex)
  import onnx
  for f in ("beatit_fallback.onnx", "beatit_monitor.onnx"):
    onnx.checker.check_model(onnx.load(os.path.join(out, f)))
  print("[export] wrote and checked the two ONNX graphs", flush=True)
  np.savez_compressed(os.path.join(out, "parity.npz"), **{k: v.astype(np.float32) for k, v in R.items()},
                      dt=np.float32(dt))

  # --- spec
  am = mj.action_manager
  term = am.get_term(am.active_terms[0])
  robot = mj.scene["robot"]
  jids = getattr(term, "_joint_ids", None)
  jn = list(robot.joint_names)
  if isinstance(jids, torch.Tensor):
    act_names = [jn[int(i)] for i in jids.tolist()]
  elif isinstance(jids, (list, tuple)):
    act_names = [jn[int(i)] for i in jids]
  else:
    act_names = list(getattr(term, "_joint_names", jn))

  def row(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x[0] if x.dim() == 2 else x
    return (x.expand(A) if x.dim() == 0 else x).cpu().tolist()

  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  motor_names = list(thermal_state(mj).names)
  lay = wt.fallback_layout(env)
  files = {n: os.path.join(run_dir, n) for n in ("final_model.zip", "tensornormalize.pt", "config.yaml",
                                                 "g1_sysid.json")}
  files["min_tube.json"] = os.path.abspath(a.tube)
  spec = {
    "what": a.what or f"Beat-It {a.tag}: fallback + 1 s monitor",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "control": {"step_dt_s": dt, "rate_hz": 1.0 / dt},
    "action": {"joint_names (action order)": act_names,
               "joint_pos_target": "offset + scale * action (Beat-It's action units; the monitor's "
                                   "proposal is the task policy's target expressed in these units: "
                                   "(q_target - offset) / scale)",
               "offset": row(term._offset), "scale": row(term._scale)},
    "motor_temp_names (order of the 29 temperatures in fb_obs / fb_state / tw_max_c)": motor_names,
    "fallback": {"onnx": "beatit_fallback.onnx", "fb_obs_layout": fb_obs_lay, "fb_obs_width": F_dim,
                 "fallback_bound": a.fallback_bound,
                 "note": "fb_obs is the fallback's actor group, raw (its normalizer is inside the graph); "
                         "trained with the per-term noise listed, no history"},
    "monitor": {
      "onnx": "beatit_monitor.onnx", "fb_state_layout": fb_state_lay, "fb_state_width": S_dim,
      "context_columns": lay["context"]["columns"], "context_width": C_dim,
      "context_nominal": wm.c_mean.detach().cpu().tolist() if C_dim else [],
      "context_note": "simulator domain randomization the world model reads (foot friction, torso COM "
                      "offset, encoder biases). On the robot use context_nominal (the training mean).",
      "horizon_steps": int(bundle.horizon), "proposal_bound": a.proposal_bound,
      "tube_eps_1s_95": float(bundle.eps),
      "threshold": a.threshold, "threshold_note": a.threshold_note,
      "rule": "reject the proposal when margin_min < threshold (or NaN); once rejected the fallback "
              f"drives >= {a.dwell_s:g} s, then the monitor decides again every step on the task "
              "policy's proposal",
      "dwell_s": a.dwell_s,
    },
    "parity": {"file": "parity.npz", "envs": a.envs, "steps": a.steps, "task": task, "env_overrides": ov,
               "proposals": f"the fallback's action; odd envs + N(0, {a.sigma})",
               "torch_vs_sandbox_monitor_max_abs": errs},
    "provenance": {"run_dir": run_dir, "tag": a.tag, "tube": os.path.abspath(a.tube),
                   "sha256": {k: sha256(v) for k, v in files.items() if os.path.isfile(v)},
                   "sandbox": git_rev(_ZOO),
                   "safety_sb3": git_rev(os.path.dirname(os.path.dirname(
                     __import__("safety_sb3").__file__)))},
    "onnx_sha256": {f: sha256(os.path.join(out, f)) for f in ("beatit_fallback.onnx", "beatit_monitor.onnx")},
    "upright_col": gz_col,
  }
  with open(os.path.join(out, "spec.json"), "w") as f:
    json.dump(spec, f, indent=1)
  print(f"[export] spec.json written; done -> {out}", flush=True)


if __name__ == "__main__":
  main()
