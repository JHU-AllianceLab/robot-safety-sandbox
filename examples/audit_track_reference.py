"""Audit of the squat-tracking reference in one world, before any training.

    python examples/audit_track_reference.py --task g1_track_squat_thermal70 \
        --out runs/track_squat/audit/thermal70
    python examples/audit_track_reference.py --task g1_track_squat_thermal_cbf \
        --out runs/track_squat/audit/thermal_cbf

Three passes over the segment (``--clip/--t0/--t1`` override the task's):

KINEMATIC (CPU forward kinematics on the identified G1, envs/g1_track/reference.py),
per frame: the lowest sole and non-sole points and which collision geoms are
within 1 cm of the floor, the grounding shift, the pelvis tilt and the margin's
tilt term against the eps of each of the six tubes (frames where the tilt term
alone is below eps engage any tube filter), the per-joint action needed to hold
the reference, ``(q - default) / scale``, against the +-6.5 task box and the +-3
fallback box with the PD torque deficit ``kp (|a| - box) scale``, and the
reference's hard- and soft-limit clipping.

DYNAMIC: noise-free spawns at frame 0 played open loop -- each step's action is
the PD target of the frame it aims at, clamped to +-6.5 or +-3 -- with every
winding and housing set to 30 / 50 / 65 C after the spawn. Per condition: how
the episode ends (complete / overheat / contact / tilt) and when, the margin
g(t), the hottest winding and the body tracking error.

ORACLE REPLAY (AMENDMENTS C1), per tube of this world (``--fb-seeds``): every frame
f in 1 .. T-2 is reached by one env step from the reference state of frame f-1
under the reference action toward f (so contact forces and sensor histories are
the simulator's), then the tube's world model verdicts the reference action
toward f+1 clamped to +-6.5 (``WMTubeMonitor.margins``, H = 50 steps, minus eps):
the frames where even perfect tracking engages the filter, i.e. the structural
engagement floor. The real margin of the same states is recorded alongside, and
every frame is flagged when the reference tilts more than 40 deg (C1's exclusion).
Windings and housings sit at ``--oracle-temp`` C.

Writes ``--out``/audit.json (summary and per-condition traces), frames.csv (one row
per frame) and audit.png. Exits 2 when some non-sole collision geom comes within
``--contact-clear-m`` of the floor on some frame (the frozen fall definition
would count perfect tracking as a fall there), else 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)

#: the tubes of both worlds: world -> (run dir, tube json, tag), each a format of the seed
TUBES = {
  "thermal70": ("runs/shac_thermal70/shac_s{s}/g1_recover_thermal70_safetyshac1p",
                "runs/shac_thermal70/wmreport/calib/thermal70_s{s}/min_tube.json",
                "thermal70_s{s}"),
  "thermal_cbf": ("runs/shac_thermal_cbf/shac_s{s}/g1_recover_thermal_cbf_safetyshac1p",
                  "runs/shac_thermal_cbf/wmreport/calib/thermalcbf_s{s}/min_tube.json",
                  "thermalcbf_s{s}"),
}
H_STEPS = 50
COVERAGE = "0.95"
TASK_BOX, FALLBACK_BOX = 6.5, 3.0
TILT_EXCLUDE_DEG = 40.0
#: g_fall_noheight's tilt term: (cos tilt - cos 50 deg) / (1 - cos 50 deg)
TILT_COS = math.cos(math.radians(50.0))


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--task", default="g1_track_squat_thermal70",
                 choices=("g1_track_squat_thermal70", "g1_track_squat_thermal_cbf"))
  p.add_argument("--clip", default=None)
  p.add_argument("--t0", type=float, default=None)
  p.add_argument("--t1", type=float, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--temps", type=float, nargs="+", default=[30.0, 50.0, 65.0])
  p.add_argument("--fb-seeds", type=int, nargs="*", default=[0, 1, 2])
  p.add_argument("--oracle-temp", type=float, default=40.0)
  p.add_argument("--no-oracle", action="store_true", help="skip the world-model replay")
  p.add_argument("--contact-clear-m", type=float, default=0.01)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", required=True)
  return p.parse_args(argv)


def tube_eps(world: str, seed: int) -> float:
  path = os.path.join(_ZOO, TUBES[world][1].format(s=seed))
  with open(path) as fh:
    return float(json.load(fh)["min_eps"][COVERAGE][str(H_STEPS)])


def kinematic_pass(cmd, model, R, a_joint, kp, scale, soft, names, contact_clear):
  """Per-frame geometry, tilt and box tables of the prepared reference; the
  per-joint arrays (``a_joint [T, J]``, ``kp``, ``scale``, ``soft [J, 2]``) are in
  the robot's joint order, ``names``."""
  import mujoco
  ref = cmd.ref
  sole, other = R.collision_geoms(model)
  gnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sole + other]
  _, _, low = R.fk_frames(model, ref.root_pos.numpy(), ref.root_quat.numpy(), ref.q.numpy(),
                          ref.joint_names, ("pelvis",), sole + other)
  ns = len(sole)
  q = ref.root_quat.double()
  up = (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)
  tilt = torch.rad2deg(torch.acos(up)).numpy()
  tilt_term = ((up - TILT_COS) / (1 - TILT_COS)).numpy()
  a = a_joint.abs().double().numpy()
  rows = []
  for i in range(ref.T):
    nf = int(np.argmin(low[i, ns:]))
    near = [gnames[ns + j] for j in np.nonzero(low[i, ns:] < contact_clear)[0]]
    rows.append({
      "frame": i, "t": ref.seg.t0 + i / ref.seg.fps, "pelvis_z": float(ref.root_pos[i, 2]),
      "tilt_deg": float(tilt[i]), "tilt_term": float(tilt_term[i]),
      "lowest_sole": float(low[i, :ns].min()), "lowest_nonfoot": float(low[i, ns:].min()),
      "nonfoot_geom": gnames[ns + nf], "nonfoot_near_floor": " ".join(near),
      "dz": float(ref.dz[i]), "max_abs_a": float(a[i].max()),
      "n_out_3": int((a[i] > FALLBACK_BOX).sum()), "n_out_6.5": int((a[i] > TASK_BOX).sum()),
      "deficit_3_Nm": float((kp * np.clip(a[i] - FALLBACK_BOX, 0, None) * scale).max()),
      "deficit_6.5_Nm": float((kp * np.clip(a[i] - TASK_BOX, 0, None) * scale).max()),
    })
  qn = cmd.q.cpu().numpy()
  joints = {}
  for j, nm in enumerate(names):
    joints[nm] = {
      "hard_clipped": int(ref.meta["clipped_low"].get(nm, 0) + ref.meta["clipped_high"].get(nm, 0)),
      "soft_violations": int(((qn[:, j] < soft[j, 0]) | (qn[:, j] > soft[j, 1])).sum()),
      "box_3_frames": int((a[:, j] > FALLBACK_BOX).sum()),
      "box_6.5_frames": int((a[:, j] > TASK_BOX).sum()),
      "max_abs_a": float(a[:, j].max()),
    }
  return rows, joints, tilt_term


def dynamic_pass(env, cmd, a_ref, temps, thermal_state, causes):
  """Open-loop playback from frame 0 per (start temperature, box)."""
  mj, dev, n = env.mj, env.device, env.num_envs
  conds = [(t, b) for t in temps for b in (TASK_BOX, FALLBACK_BOX)]
  cond = torch.arange(n, device=dev) % len(conds)
  temp = torch.tensor([c[0] for c in conds], device=dev)[cond]
  box = torch.tensor([c[1] for c in conds], device=dev)[cond]
  cmd.spawn_noise, cmd.start_override = False, 0
  env.reset()
  tw0 = temp[:, None].expand(n, 29).contiguous()
  thermal_state(mj).set(None, tw0, tw0.clone(), tw0.clone())
  steps = cmd.T - 1
  nan = float("nan")
  g_tr = torch.full((steps, n), nan, device=dev)
  tw_tr = torch.full((steps, n), nan, device=dev)
  err_tr = torch.full((steps, n), nan, device=dev)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  end_cause = torch.full((n,), -1, dtype=torch.long, device=dev)
  end_step = torch.full((n,), -1, dtype=torch.long, device=dev)
  first_fail = torch.zeros(n, dtype=torch.long, device=dev)
  for k in range(steps):
    act = torch.maximum(torch.minimum(a_ref[cmd.f], box[:, None]), -box[:, None])
    env.step_tensor(act)
    bus = mj._zoo_track
    g = env.safety_margins()[0]
    g_tr[k] = torch.where(alive, g, nan)
    tw_tr[k] = torch.where(alive, bus["step"]["tw_max"], nan)
    err_tr[k] = torch.where(alive, bus["step"]["err_body"], nan)
    ended = bus["done"] & alive
    end_cause = torch.where(ended, bus["ep"]["cause"].long(), end_cause)
    first_fail = torch.where(ended, bus["ep"]["fail"].long(), first_fail)
    end_step = torch.where(ended, k + 1, end_step)
    alive &= ~bus["done"]
  out = []
  for c, (t, b) in enumerate(conds):
    m = cond == c
    ec = end_cause[m]
    fs = end_step[m & (end_cause > 0)].float()
    out.append({
      "temp_c": t, "box": b, "episodes": int(m.sum()),
      "ends": {causes[i]: int((ec == i).sum()) for i in range(len(causes))},
      "still_running": int((ec < 0).sum()),
      "any_fail": float((first_fail[m] != 0).float().mean()),
      "median_fail_s": float(fs.median() * mj.step_dt) if fs.numel() else None,
      "g_min": float(torch.nan_to_num(g_tr[:, m], nan=math.inf).min()),
      "tw_peak": float(torch.nan_to_num(tw_tr[:, m], nan=-math.inf).max()),
      "err_body_mean": float(torch.nanmean(err_tr[:, m])),
      "trace_g": torch.nanmean(g_tr[:, m], dim=1).tolist(),
      "trace_tw_max": torch.nanmean(tw_tr[:, m], dim=1).tolist(),
      "trace_err_body": torch.nanmean(err_tr[:, m], dim=1).tolist(),
    })
  return out


def oracle_pass(env, cmd, a_ref, world, seeds, temp, thermal_state):
  """Reference states pushed through each tube's verdict (module docstring)."""
  try:
    from robot_safety_sandbox.filters.wm_tube import wm_tube_filter
  except ImportError as e:
    raise SystemExit(f"[audit] the oracle replay needs robot_safety_sandbox.filters.wm_tube "
                     f"({e}); rerun with --no-oracle to skip it") from e
  from robot_safety_sandbox.envs.g1_recover.states import to_heading_frame, write_states
  mj, dev, n = env.mj, env.device, env.num_envs
  mons = {}
  for s in seeds:
    run, tube, tag = (x.format(s=s) for x in TUBES[world])
    filt, _ = wm_tube_filter(env, os.path.join(_ZOO, run), os.path.join(_ZOO, tube), tag=tag,
                             condition=f"wmmin:{H_STEPS * mj.step_dt:g}:{COVERAGE}",
                             dwell_s=0.0, device=dev, proposal_bound=TASK_BOX,
                             fallback_bound=FALLBACK_BOX)
    mons[tag] = filt.monitor
  T = cmd.T
  frames = torch.arange(1, T - 1, device=dev)
  g_real = torch.full((T,), float("nan"), device=dev)
  pred = {tag: torch.full((T,), float("nan"), device=dev) for tag in mons}
  invalid = 0
  for c0 in range(0, int(frames.numel()), n):
    f = frames[c0:c0 + n]
    k = int(f.numel())
    ids = torch.arange(k, device=dev)
    cmd.spawn_noise, cmd.start_override = False, 0
    env.reset()
    p = f - 1
    quat_h, lin_h, ang_h = to_heading_frame(cmd.ref_root_quat[p], cmd.ref_root_lin[p],
                                            cmd.ref_root_ang[p])
    rows = {"root_z": cmd.ref_root_pos[p, 2], "root_quat_h": quat_h, "lin_vel_h": lin_h,
            "ang_vel_h": ang_h, "joint_pos": cmd.q[p], "joint_vel": cmd.qd[p]}
    write_states(mj, ids, rows, torch.zeros(k, device=dev), torch.zeros(k, 2, device=dev))
    tw = torch.full((n, 29), float(temp), device=dev)
    thermal_state(mj).set(None, tw, tw.clone(), tw.clone())
    act = torch.zeros(n, a_ref.shape[1], device=dev)
    act[:k] = a_ref[f].clamp(-TASK_BOX, TASK_BOX)
    env.step_tensor(act)
    ok = ~mj._zoo_track["done"][:k]
    invalid += int((~ok).sum())
    g_real[f] = torch.where(ok, env.safety_margins()[0][:k], float("nan"))
    a_first = a_ref[f + 1].clamp(-TASK_BOX, TASK_BOX)
    with torch.no_grad():
      for tag, mon in mons.items():
        pm = mon.margins(ids, a_first, H_STEPS).amin(1)
        pred[tag][f] = torch.where(ok, pm, float("nan"))
  eps = {tag: float(mon.eps) for tag, mon in mons.items()}
  return g_real, pred, eps, invalid


def plot(path, rows, eps_all, dyn, g_real, pred, eps):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  t = np.array([r["t"] for r in rows])
  fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
  ax[0].plot(t, [r["tilt_term"] for r in rows], "k", label="tilt term")
  for tag, e in eps_all.items():
    ax[0].axhline(e, lw=0.8, ls="--", label=f"eps {tag} {e:.3f}")
  ax[0].set_ylabel("tilt margin term")
  ax[0].legend(fontsize=7, ncol=4)
  ax[1].plot(t, [r["lowest_nonfoot"] for r in rows], label="lowest non-sole point")
  ax[1].plot(t, [r["pelvis_z"] for r in rows], label="pelvis z")
  ax[1].plot(t, [r["max_abs_a"] / 10 for r in rows], label="max |a| / 10")
  ax[1].axhline(0.65, color="grey", lw=0.6, ls=":")
  ax[1].axhline(0.30, color="grey", lw=0.6, ls=":")
  ax[1].set_ylabel("m")
  ax[1].legend(fontsize=7)
  if g_real is not None:
    ax[2].plot(t, g_real.cpu().numpy(), "k", label="real g")
    for tag, v in pred.items():
      ax[2].plot(t, (v - eps[tag]).cpu().numpy(), lw=0.8, label=f"verdict {tag}")
    ax[2].axhline(0.0, color="r", lw=0.6)
    ax[2].legend(fontsize=7, ncol=4)
  ax[2].set_ylabel("oracle replay")
  td = t[1:]
  for d in dyn:
    ax[3].plot(td, d["trace_g"], lw=0.8, label=f"{d['temp_c']:g} C, +-{d['box']:g}")
  ax[3].set_ylabel("open-loop g (alive mean)")
  ax[3].set_xlabel("CSV time, s")
  ax[3].legend(fontsize=7, ncol=3)
  fig.tight_layout()
  fig.savefig(path, dpi=110)


def main(argv=None) -> int:
  a = _parse(argv)
  import robot_safety_sandbox as rss
  from robot_safety_sandbox.envs.g1_recover.thermal import thermal_state
  from robot_safety_sandbox.envs.g1_track import reference as R
  from robot_safety_sandbox.envs.g1_track.mdp import CAUSES

  world = a.task.removeprefix("g1_track_squat_")
  overrides = {k: v for k, v in (("clip", a.clip), ("t0", a.t0), ("t1", a.t1)) if v is not None}
  torch.manual_seed(a.seed)
  env = rss.make_tensor(a.task, num_envs=a.num_envs, device=a.device,
                        cfg_overrides=overrides or None)
  mj = env.mj
  cmd = mj.command_manager.get_term("ref")
  robot = cmd.robot
  term = mj.action_manager.get_term("joint_pos")
  tid = term.target_ids
  a_ref = (cmd.q[:, tid] - term.offset[0]) / term.scale[0]       # [T, 29] action order
  # the same per joint, in the robot's joint order (cmd.q's)
  names = list(robot.joint_names)
  a_joint = torch.zeros_like(cmd.q)
  a_joint[:, tid] = a_ref
  scale = np.zeros(len(names))
  scale[tid.cpu().numpy()] = term.scale[0].cpu().numpy()
  kp = np.full(len(names), np.nan)
  for act in robot.actuators:
    for nm in act.target_names:
      kp[names.index(nm)] = float(act.cfg.stiffness)
  soft = robot.data.soft_joint_pos_limits[0].cpu().numpy()
  rows, joints, tilt_term = kinematic_pass(
    cmd, R.sysid_model(), R, a_joint.cpu(), kp, scale, soft, names, a.contact_clear_m)
  eps_all = {TUBES[w][2].format(s=s): tube_eps(w, s) for w in TUBES for s in (0, 1, 2)}
  tilt_floor = {tag: int((tilt_term < e).sum()) for tag, e in eps_all.items()}
  kneel = [r["frame"] for r in rows if r["lowest_nonfoot"] < a.contact_clear_m]
  print(f"[audit] {cmd.ref.meta}")
  print(f"[audit] kinematic: lowest non-sole point {min(r['lowest_nonfoot'] for r in rows):.3f} m, "
        f"{len(kneel)} frames within {a.contact_clear_m:g} m; max tilt "
        f"{max(r['tilt_deg'] for r in rows):.1f} deg; frames whose tilt term is below eps "
        f"{tilt_floor}")
  print(f"[audit] action box: frames with a joint past +-3 "
        f"{sum(r['n_out_3'] > 0 for r in rows)}, past +-6.5 "
        f"{sum(r['n_out_6.5'] > 0 for r in rows)} of {cmd.T}")

  dyn = dynamic_pass(env, cmd, a_ref, a.temps, thermal_state, CAUSES)
  for d in dyn:
    print(f"[audit] open loop {d['temp_c']:g} C, box +-{d['box']:g}: ends {d['ends']}, "
          f"running {d['still_running']}, median failure {d['median_fail_s']} s, "
          f"g min {d['g_min']:.3f}, Tw peak {d['tw_peak']:.1f} C")

  oracle = None
  g_real, pred, eps = None, {}, {}
  if not a.no_oracle and a.fb_seeds:
    g_real, pred, eps, invalid = oracle_pass(env, cmd, a_ref, world, a.fb_seeds,
                                             a.oracle_temp, thermal_state)
    tilt = torch.tensor([r["tilt_deg"] for r in rows], device=g_real.device)
    keep = tilt <= TILT_EXCLUDE_DEG
    valid = ~torch.isnan(g_real)
    oracle = {"temp_c": a.oracle_temp, "invalid_frames": invalid,
              "frames_tilt_gt_40": int((~keep).sum()), "real_g_min": float(g_real[valid].min()),
              "real_g_negative": int((g_real[valid] < 0).sum()), "tubes": {}}
    for tag, v in pred.items():
      eng = (v - eps[tag] < 0) & valid
      oracle["tubes"][tag] = {
        "eps": eps[tag], "floor_frac": float(eng.sum() / valid.sum()),
        "floor_frac_tilt_le_40": float((eng & keep).sum() / (valid & keep).sum()),
        "engaged_frames": eng.nonzero().flatten().tolist(),
      }
      print(f"[audit] oracle {tag}: eps {eps[tag]:.3f}, perfect tracking engages on "
            f"{100 * oracle['tubes'][tag]['floor_frac']:.1f}% of frames "
            f"({100 * oracle['tubes'][tag]['floor_frac_tilt_le_40']:.1f}% with tilt <= 40 deg)")
    for i, r in enumerate(rows):
      r["g_oracle"] = float(g_real[i])
      for tag, v in pred.items():
        r[f"verdict_{tag}"] = float(v[i] - eps[tag])

  os.makedirs(a.out, exist_ok=True)
  with open(os.path.join(a.out, "frames.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  summary = {
    "task": a.task, "reference": cmd.ref.meta, "kneel_frames": kneel,
    "kinematic": {
      "lowest_nonfoot_min": min(r["lowest_nonfoot"] for r in rows),
      "tilt_max_deg": max(r["tilt_deg"] for r in rows),
      "tilt_below_eps_frames": tilt_floor, "eps": eps_all,
      "frames_box_3": sum(r["n_out_3"] > 0 for r in rows),
      "frames_box_6.5": sum(r["n_out_6.5"] > 0 for r in rows),
      "deficit_3_max_Nm": max(r["deficit_3_Nm"] for r in rows),
      "deficit_6.5_max_Nm": max(r["deficit_6.5_Nm"] for r in rows),
      "joints": joints,
    },
    "dynamic": dyn, "oracle": oracle,
  }
  with open(os.path.join(a.out, "audit.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
  plot(os.path.join(a.out, "audit.png"), rows, eps_all, dyn, g_real, pred, eps)
  print(f"[audit] wrote {a.out}/audit.json, frames.csv, audit.png")
  if kneel:
    print(f"[audit] {len(kneel)} frames bring a non-sole geom within {a.contact_clear_m:g} m "
          f"of the floor: perfect tracking is a fall there")
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main())
