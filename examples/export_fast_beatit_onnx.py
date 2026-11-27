"""Export the FAST Beat-It monitor (k imagined steps + a terminal value V_(50-k) learned from real rollouts), the
fallback policy and the full 1 s monitor as ONNX graphs, with per-task thresholds (the user, 2026-11-26).

    MUJOCO_GL=egl PYTHONPATH=~/ssb-wt-hybrid python examples/export_fast_beatit_onnx.py \\
        --volume runs/fast_beatit/volume --ks 10 15 25 --out <folder>

Graphs (batch 1, opset 18, float32), in <out>:
  beatit_fallback.onnx      fb_obs [1, F] -> actions [1, A]                     (= examples/export_beatit_onnx.py)
  beatit_monitor_full.onnx  fb_state [1, S], context [1, C], proposal [1, A]
                            -> margin_min [1], survival [1], tw_max_c [1, 29]   (= examples/export_beatit_onnx.py)
  beatit_monitor_k<k>.onnx  fb_state, context, proposal
                            -> sigma [1], margin_min_k [1], value [1], survival_k [1], tw_max_c [1, 29]
      sigma = min(margin_min_k, value): margin_min_k = the lowest imagined margin over the first k steps (the
      proposal for one step, then the fallback, in the world model with its SysID heat block); value =
      V_(50-k)(predicted state after k steps, context) ~ the lowest REAL margin over the rest of the 1 s.
      Reject when sigma < the task's threshold (spec.json "thresholds"), or NaN.

Parity (recorded to parity.npz for the onnxruntime check): N envs of Beat-It's own world, the fallback driving from
its start bank, proposals = the fallback's action (+ N(0, sigma) on half the envs). The full graph is checked against
the sandbox's calibrated monitor (filters/wm_tube.WMTubeMonitor); each fast graph against the offline scorer the
thresholds were calibrated with (examples/truncated_monitor_volume.Fallback + the same V).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_beatit_onnx as E  # noqa: E402
from truncated_monitor_volume import MLP, Fallback  # noqa: E402


class FastMonitorDeploy(nn.Module):
  def __init__(self, bundle, V: nn.Module, k: int, proposal_bound: float, tw_cols: torch.Tensor):
    super().__init__()
    self.b, self.wm, self.V, self.k = bundle, bundle.wm, V, int(k)
    am = bundle.model._actor_mean
    self.actor_mean = am if isinstance(am, nn.Module) else nn.Identity()
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
    mk = torch.full_like(alive, 1e9)
    dec = s
    for t in range(self.k):
      st = wm.step(z, act, cn, z_clip=b.z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      g = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      mk = torch.minimum(mk, g)
      dec = wm.decode(z)
      tmax = torch.maximum(tmax, dec.index_select(-1, self.tw))
      if t + 1 < self.k:
        act = b.pi(dec.index_select(-1, b.actor_idx))
    v = self.V(torch.cat([dec, context], -1))
    return torch.minimum(mk, v).reshape(-1), mk.reshape(-1), v.reshape(-1), alive.reshape(-1), tmax


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--volume", default="", help="(old) runs/fast_beatit/volume*: values.pt + volume.json with the thresholds")
  p.add_argument("--values", default="", help="the value functions (values.pt); thresholds then from --calib-key")
  p.add_argument("--variant", default="pooled")
  p.add_argument("--calib-key", default="", help="thresholds.json key with the fast thresholds per task (beatit_fast, "
                 "beatit_student, ...); its 'full' entry gives the full monitor's for a student")
  p.add_argument("--full-key", default="beatit_thermal", help="beatit_thermal: the trained thresholds; or the calib key")
  p.add_argument("--tasks", nargs="+", default=["dance", "fight", "bentknee"])
  p.add_argument("--what", default="Beat-It v0.6.1 heat-aware (beatit_thermal_s0)")
  p.add_argument("--ks", type=int, nargs="+", default=[10, 15, 25])
  p.add_argument("--run-dir", default="runs/v061_study/beatit_thermal_s0/g1_recover_thermal70_v61_safetyshac1p")
  p.add_argument("--tube", default="runs/v061_study/eval/calib/beatit_thermal/min_tube.json")
  p.add_argument("--tag", default="beatit_thermal_s0")
  p.add_argument("--out", required=True)
  p.add_argument("--proposal-bound", type=float, default=6.5)
  p.add_argument("--fallback-bound", type=float, default=3.0)
  p.add_argument("--envs", type=int, default=8)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--sigma", type=float, default=1.5)
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
  run_dir, out = os.path.abspath(a.run_dir), os.path.abspath(a.out)
  os.makedirs(out, exist_ok=True)
  vol = json.load(open(os.path.join(a.volume, "volume.json"))) if a.volume else None
  vpath = a.values or os.path.join(a.volume, "values.pt")
  vblob = torch.load(vpath, map_location=dev, weights_only=False)
  H = int(vblob.get("H", 50))
  calib = {t: json.load(open(os.path.join("runs/sailr_df/calib", t, "thresholds.json"))) for t in a.tasks}
  cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
  task, ov = cfg["task"], dict(cfg.get("env_overrides") or {})
  env = make_tensor(task, a.envs, dev, cfg_overrides=ov, critic_obs_key="critic")
  env.reset()
  base = wt._base(env)
  mj = base.mj
  dt = float(mj.step_dt)
  bundle = wt.load_wm_tube(run_dir, os.path.abspath(a.tube), tag=a.tag, cond="wmmin:1:0.95", device=dev,
                           fallback_bound=a.fallback_bound, step_dt=dt, wm_thermal="nn")
  wm = bundle.wm
  assert int(bundle.horizon) == H, (bundle.horizon, H)
  wt.check_binding(bundle, env)
  ref = wt.WMTubeMonitor(a.envs, dev, bundle, proposal_bound=a.proposal_bound, mode="compact", precision="fp32",
                         action_dim=int(wm.action_dim))
  om = mj.observation_manager
  fb_obs_lay, F_dim = E.layout_of(om, "actor")
  fb_state_lay, S_dim = E.layout_of(om, "critic")
  mt = [t for t in fb_state_lay if t["name"] == "motor_temp"]
  tw_cols = torch.arange(mt[0]["start"], mt[0]["start"] + 29, device=dev)
  A, C_dim = int(wm.action_dim), int(wm.context_dim)

  def V_of(h):
    sd = vblob[a.variant][str(h)]
    m = MLP(sd["mean"].numel(), vblob["hidden"], vblob["layers"], sd["mean"], sd["std"]).to(dev)
    m.load_state_dict(sd)
    return m.eval()

  fb = E.FallbackDeploy(bundle).eval()
  full = E.MonitorDeploy(bundle, a.proposal_bound, tw_cols).eval()
  fasts = {k: FastMonitorDeploy(bundle, V_of(H - k), k, a.proposal_bound, tw_cols).eval() for k in a.ks}
  # the offline scorer the thresholds were calibrated with (its fallback box = the dump's low / high)
  lab = vol["tasks"][next(iter(vol["tasks"]))]["test"] if vol else calib[a.tasks[0]][a.calib_key]["labels"]
  dump = torch.load(lab, map_location="cpu", weights_only=False)["settings"]["dump"]
  off = Fallback(dump["policy"], dump, dev)

  noisy = (torch.arange(a.envs, device=dev) % 2 == 1).float().unsqueeze(-1)
  R = {k: [] for k in ("fb_obs", "fb_state", "context", "proposal", "a_fallback", "ref_margin_min", "margin_min",
                       "survival", "tw_max_c")}
  for k in a.ks:
    for n in ("sigma", "margin_min_k", "value", "survival_k", "tw_max_c_k", "offline_sigma"):
      R[f"k{k}_{n}"] = []
  errs = {"fallback": 0.0, "full_vs_ref": 0.0} | {f"k{k}_vs_offline": 0.0 for k in a.ks}
  t0 = time.time()
  with torch.no_grad():
    for step in range(a.steps):
      og = base.obs_groups()
      F, S, Cx = og["actor"].float(), og["critic"].float(), base.sim_context()
      a_fb = bundle.pi(F)
      prop = a_fb + noisy * a.sigma * torch.randn_like(a_fb)
      with wt.matmul_precision("fp32"):
        r_out, _ = ref._rollout(torch.nan_to_num(S), Cx, torch.clamp(prop, ref.p_lo, ref.p_hi), ref.horizon)
      r_min = r_out.amin(dim=1)
      mm, sv, tw = full(S, Cx, prop)
      errs["fallback"] = max(errs["fallback"], float((fb(F) - a_fb).abs().max()))
      errs["full_vs_ref"] = max(errs["full_vs_ref"], float((mm - r_min).abs().max()))
      pc = torch.clamp(prop, -a.proposal_bound, a.proposal_bound)
      o_out, o_at = off.rollout(torch.nan_to_num(S), Cx, pc, steps=max(a.ks), keep=set(a.ks))
      pre = o_out.cummin(1).values
      for k, m in fasts.items():
        sg, mk, v, svk, twk = m(S, Cx, prop)
        ov_ = fasts[k].V(torch.cat([o_at[k][0], Cx], -1))
        osg = torch.minimum(pre[:, k - 1], ov_) if k > 0 else ov_
        errs[f"k{k}_vs_offline"] = max(errs[f"k{k}_vs_offline"], float((sg - osg).abs().max()))
        for n, x in (("sigma", sg), ("margin_min_k", mk), ("value", v), ("survival_k", svk), ("tw_max_c_k", twk),
                     ("offline_sigma", osg)):
          R[f"k{k}_{n}"].append(x.float().cpu())
      for n, x in (("fb_obs", F), ("fb_state", S), ("context", Cx), ("proposal", prop), ("a_fallback", a_fb),
                   ("ref_margin_min", r_min), ("margin_min", mm), ("survival", sv), ("tw_max_c", tw)):
        R[n].append(x.float().cpu())
      env.step_tensor(a_fb)
  print(f"[export-fast] parity {a.steps} steps x {a.envs} envs ({time.time() - t0:.0f} s), max abs: "
        f"{ {k: f'{v:.2e}' for k, v in errs.items()} }", flush=True)
  if errs["full_vs_ref"] > 1e-4 or errs["fallback"] > 1e-5 or any(errs[f"k{k}_vs_offline"] > 1e-3 for k in a.ks):
    raise SystemExit("[export-fast] deploy modules do not reproduce the references")
  R = {k: torch.stack(v, 1).numpy() for k, v in R.items()}
  z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
  ex = dict(export_params=True, opset_version=18, dynamic_axes={}, dynamo=False)
  torch.onnx.export(fb, (z(1, F_dim),), os.path.join(out, "beatit_fallback.onnx"), input_names=["fb_obs"],
                    output_names=["actions"], **ex)
  torch.onnx.export(full, (z(1, S_dim), z(1, C_dim), z(1, A)), os.path.join(out, "beatit_monitor_full.onnx"),
                    input_names=["fb_state", "context", "proposal"], output_names=["margin_min", "survival", "tw_max_c"],
                    **ex)
  for k, m in fasts.items():
    torch.onnx.export(m, (z(1, S_dim), z(1, C_dim), z(1, A)), os.path.join(out, f"beatit_monitor_k{k}.onnx"),
                      input_names=["fb_state", "context", "proposal"],
                      output_names=["sigma", "margin_min_k", "value", "survival_k", "tw_max_c"], **ex)
  import onnx
  files = ["beatit_fallback.onnx", "beatit_monitor_full.onnx"] + [f"beatit_monitor_k{k}.onnx" for k in a.ks]
  for f in files:
    onnx.checker.check_model(onnx.load(os.path.join(out, f)))
  np.savez_compressed(os.path.join(out, "parity.npz"), **{k: v.astype(np.float32) for k, v in R.items()},
                      dt=np.float32(dt))

  # thresholds: the trained full-monitor threshold per task (SAILR runs) and the fast monitors' Wang eps-1 % deltas
  if a.full_key == "beatit_thermal":
    full_thr = {"walk": 0.665} | {t: c["beatit_thermal"]["raw_threshold_eps1"] for t, c in calib.items()}
    note = "the thresholds the SAILR trackers trained with (Wang eps 1 % on each task's SONIC push bank)"
  else:
    full_thr = {t: c[a.full_key]["full"]["delta"] for t, c in calib.items()}
    note = "Wang eps 1 % on each task's SONIC push bank"
  thr = {"beatit_monitor_full.onnx": {"threshold_on": "margin_min", "by_task": full_thr, "note": note}}
  for k in a.ks:
    if vol:
      by = {t: vol["tasks"][t]["scores"][f"{a.variant}_k{k}"]["wang_0.01"] for t in vol["tasks"]}
      fv = {t: vol["tasks"][t]["scores"]["full"]["wang_0.01"]["volume"] for t in vol["tasks"]}
    else:
      by = {t: c[a.calib_key]["by_k"][str(k)] for t, c in calib.items()}
      fv = {t: c[a.calib_key]["full"]["volume"] for t, c in calib.items()}
    thr[f"beatit_monitor_k{k}.onnx"] = {
      "threshold_on": "sigma", "by_task": {t: d["delta"] for t, d in by.items()},
      "volume_by_task": {t: d["volume"] for t, d in by.items()},
      "soundness_by_task": {t: d["soundness"] for t, d in by.items()},
      "full_monitor_volume_by_task": fv,
      "note": "Wang et al. Prop. 1, eps 1 %, beta 1e-10, on each task's SONIC push bank (K 4 real 3 s fallback "
              "rollouts per row, windings soaked U(30, 60) C)"}
  am = mj.action_manager
  term = am.get_term(am.active_terms[0])
  robot = mj.scene["robot"]
  jn = list(robot.joint_names)
  jids = getattr(term, "_joint_ids", None)
  act_names = [jn[int(i)] for i in (jids.tolist() if isinstance(jids, torch.Tensor) else (jids or range(len(jn))))]

  def row(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x[0] if x.dim() == 2 else x
    return (x.expand(A) if x.dim() == 0 else x).cpu().tolist()

  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  lay = wt.fallback_layout(env)
  spec = {
    "what": f"{a.what}: fallback, full 1 s monitor, fast monitors (k imagined steps + V_(50-k)) for k in {a.ks}",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "control": {"step_dt_s": dt, "rate_hz": 1.0 / dt},
    "action": {"joint_names (action order)": act_names, "offset": row(term._offset), "scale": row(term._scale),
               "joint_pos_target": "offset + scale * action"},
    "motor_temp_names": list(thermal_state(mj).names),
    "fallback": {"onnx": "beatit_fallback.onnx", "fb_obs_layout": fb_obs_lay, "fb_obs_width": F_dim,
                 "fallback_bound": a.fallback_bound},
    "monitor_inputs": {"fb_state_layout": fb_state_lay, "fb_state_width": S_dim,
                       "context_columns": lay["context"]["columns"], "context_width": C_dim,
                       "context_nominal": wm.c_mean.detach().cpu().tolist() if C_dim else [],
                       "proposal_bound": a.proposal_bound, "horizon_steps": H},
    "thresholds": thr,
    "dwell_s": 1.0,
    "rule": "reject when the score < the task's threshold (or NaN); the fallback then drives >= 1 s, after which the "
            "monitor judges the task policy's proposal again every step",
    "value_functions": {"variant": a.variant, "hidden": vblob["hidden"], "layers": vblob["layers"],
                        "file": os.path.abspath(vpath)},
    "parity": {"file": "parity.npz", "envs": a.envs, "steps": a.steps, "task": task, "max_abs": errs},
    "provenance": {"run_dir": run_dir, "tube_for_layout": os.path.abspath(a.tube), "values": os.path.abspath(vpath),
                   "sandbox": E.git_rev(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))},
    "onnx_sha256": {f: E.sha256(os.path.join(out, f)) for f in files},
  }
  json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
  print(f"[export-fast] wrote {files} + parity.npz + spec.json -> {out}", flush=True)


if __name__ == "__main__":
  main()
