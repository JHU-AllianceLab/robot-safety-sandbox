"""Harvest the start-state bank for the G1 recovery task (g1_recover*).

    python examples/harvest_g1_recover_states.py \
        --walker runs/g1_walker_flat/final_model.zip \
        --samples 200000 --synthetic 60000

Writes ``data/g1_recover_states.pt`` (fields: envs/g1_recover/states.py), which
the g1_recover spawn reads. Three stages, one env, one process:

1. WALKER. The walker (a mode="cumulative" pi_task and its VecNormalize) walks
   in its own env on the task's command box (default: the fast box, vx in
   [-0.5, 1.5], vy in [-0.5, 0.5], yaw rate in [-1, 1], resampled as in
   training). Each env loops: settle for a while, then either snapshot the
   state as it is (CLEAN) or kick the root velocity (up to --max-push-v m/s
   horizontal, --max-push-w rad/s roll/pitch) and snapshot 20-300 ms later
   (PUSHED). The walker then keeps control for --window-s, and the row is
   labelled with whether it fell and when. Rows whose window hits the episode
   timeout are dropped (no label).
2. REPLAY CHECK. Rows are restored, through the same ``write_states`` the
   training spawn uses, into the SAME env they were harvested from (so the
   startup domain randomization -- friction, COM, encoder bias -- matches),
   with the command, the last action and the gait phase put back; the walker
   runs the window again and the fall outcome is compared with the label.
   Every row is replayed TWICE: replay-vs-replay agreement is the noise floor
   (observation noise, one-sample labels), and replay-vs-label close to it
   means the restore itself loses nothing.
3. SYNTHETIC. The default stance plus random tilt, body twist, joint offsets
   and joint velocities, each scaled by a difficulty d in [0, 1]. Height is set
   by forward kinematics so the lowest sole point sits 2 mm above the ground
   (tilt or bent knees otherwise push a foot into the floor, and the contact
   solver throws the robot).
"""

from __future__ import annotations

import argparse
import math
import os
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

from mjlab.utils.lab_api.math import (  # noqa: E402
  quat_apply, quat_from_euler_xyz, yaw_quat)

from robot_safety_sandbox import make_tensor, spec  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_CLEAN, KIND_PUSHED, KIND_SYNTHETIC, SoleGeometry, snapshot,
  speed_of, tilt_deg, write_states)
from robot_safety_sandbox.eval import NominalPolicy, load_nominal  # noqa: E402

_NEVER = 1e9
# Synthetic perturbation at difficulty d = 1 (each component is d * max * U(-1, 1)).
SYN_TILT_DEG = 25.0            # roll and pitch, each
SYN_LIN_VEL = (1.5, 0.8, 0.3)  # vx, vy, vz (m/s, heading frame)
SYN_ANG_VEL = (1.5, 1.5, 1.5)  # roll, pitch, yaw rates (rad/s)
SYN_JOINT_POS = 0.3            # rad offset from the default pose
SYN_JOINT_VEL = 2.0            # rad/s
SOLE_CLEARANCE = 0.002         # m between the lowest sole point and the ground


def _parse():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--walker", required=True,
                 help="the walker's final_model.zip (its vecnormalize.pkl is "
                      "found next to it)")
  p.add_argument("--task", default="g1_walker_flat_fast",
                 help="the walker env (sets the command box). The fast box by "
                      "default, whichever walker is loaded")
  p.add_argument("--out", default=os.path.join(_ZOO, "data",
                                               "g1_recover_states.pt"))
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--samples", type=int, default=200000,
                 help="labelled walker rows to collect (clean + pushed)")
  p.add_argument("--synthetic", type=int, default=60000)
  p.add_argument("--push-frac", type=float, default=0.6,
                 help="fraction of walker events that are pushes")
  p.add_argument("--max-push-v", type=float, default=1.5,
                 help="largest horizontal velocity kick, m/s")
  p.add_argument("--max-push-w", type=float, default=1.0,
                 help="largest roll/pitch rate kick, rad/s")
  p.add_argument("--window-s", type=float, default=2.0,
                 help="label window after the snapshot, s")
  p.add_argument("--replay-rounds", type=int, default=2,
                 help="replay-check rounds (each restores up to one row per env)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args()


def _randint(lo: int, hi: int, n: int, device) -> torch.Tensor:
  """Uniform integers in [lo, hi], shape (n,)."""
  return torch.randint(lo, hi + 1, (n,), device=device)


def _u(n: int, device, lo=-1.0, hi=1.0) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(n, device=device)


def _phase_period_steps(mj) -> int:
  for group in mj.cfg.observations.values():
    term = group.terms.get("phase")
    if term is not None:
      return max(1, round(term.params["period"] / mj.step_dt))
  return 1


def _freeze_commands(mj) -> None:
  """Stop the twist command from resampling (so a restored command stays)."""
  term = mj.command_manager.get_term("twist")
  term.cfg.resampling_time_range = (_NEVER, _NEVER)
  if hasattr(term.cfg, "rel_standing_envs"):
    term.cfg.rel_standing_envs = 0.0
  if hasattr(term, "time_left"):
    term.time_left[:] = _NEVER
  if hasattr(term, "is_standing_env"):
    term.is_standing_env[:] = False


# --- stage 1: the walker ------------------------------------------------------

WAIT, DELAY, WINDOW = 0, 1, 2


def harvest_walker(env, walker, args) -> dict[str, torch.Tensor]:
  mj, dev, n = env.mj, env.device, env.num_envs
  robot = mj.scene["robot"]
  dt = mj.step_dt
  period = _phase_period_steps(mj)
  window = max(1, round(args.window_s / dt))
  settle = (round(1.0 / dt), round(2.0 / dt))    # after a (re)spawn
  gap = (round(0.5 / dt), round(2.0 / dt))       # between two events
  delay = (1, max(1, round(0.3 / dt)))           # push -> snapshot

  mode = torch.full((n,), WAIT, dtype=torch.long, device=dev)
  count = _randint(*settle, n, dev)
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  pend_kind = torch.zeros(n, dtype=torch.long, device=dev)
  pend_sev = torch.zeros(n, device=dev)
  pend: dict[str, torch.Tensor] = {}
  rows: dict[str, list] = {f: [] for f in FIELDS}
  ids_all = torch.arange(n, device=dev)
  got, t0, it = 0, time.time(), 0

  def take(mask):
    """Store the current state of the envs in ``mask`` as pending rows."""
    s = snapshot(mj)
    s["command"] = mj.command_manager.get_command("twist").clone()
    s["phase_step"] = mj.episode_length_buf % period
    for k, v in s.items():
      if k not in pend:
        pend[k] = torch.zeros_like(v)
      pend[k][mask] = v[mask]
    elapsed[mask] = 0

  def emit(mask, fell):
    nonlocal got
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
      return
    for k in ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos",
              "joint_vel", "action", "command", "phase_step"):
      rows[k].append(pend[k][idx])
    rows["kind"].append(pend_kind[idx])
    rows["fell"].append(fell[idx])
    rows["time_to_fall"].append(torch.where(
      fell[idx], elapsed[idx].float() * dt,
      torch.full_like(elapsed[idx], float("nan"), dtype=torch.float)))
    rows["push_severity"].append(pend_sev[idx])
    rows["difficulty"].append(torch.zeros(idx.numel(), device=dev))
    rows["speed"].append(speed_of(pend["lin_vel_h"][idx]))
    rows["env_id"].append(ids_all[idx])
    got += int(idx.numel())

  obs = env.reset()
  while got < args.samples:
    # 1. events due now: clean snapshot, or push (snapshot after the delay)
    due = (mode == WAIT) & (count <= 0)
    if bool(due.any()):
      push = due & (torch.rand(n, device=dev) < args.push_frac)
      clean = due & ~push
      if bool(clean.any()):
        take(clean)
        pend_kind[clean] = KIND_CLEAN
        pend_sev[clean] = 0.0
        mode[clean] = WINDOW
        count[clean] = window
      if bool(push.any()):
        pid = push.nonzero(as_tuple=False).squeeze(-1)
        k = pid.numel()
        sev = torch.rand(k, device=dev)
        heading = _u(k, dev, 0.0, 2.0 * math.pi)
        tilt_dir = _u(k, dev, 0.0, 2.0 * math.pi)
        d = robot.data
        dv = torch.stack([sev * args.max_push_v * torch.cos(heading),
                          sev * args.max_push_v * torch.sin(heading),
                          sev * 0.3 * _u(k, dev)], dim=1)
        dw_h = torch.stack([sev * args.max_push_w * torch.cos(tilt_dir),
                            sev * args.max_push_w * torch.sin(tilt_dir),
                            sev * 1.0 * _u(k, dev)], dim=1)
        # the roll/pitch kick is drawn in the robot's heading frame -> world
        yaw_q = yaw_quat(d.root_link_quat_w[pid])
        vel = torch.cat([d.root_link_lin_vel_w[pid] + dv,
                         d.root_link_ang_vel_w[pid] + quat_apply(yaw_q, dw_h)],
                        dim=1)
        robot.write_root_link_velocity_to_sim(vel, env_ids=pid)
        pend_kind[push] = KIND_PUSHED
        pend_sev[pid] = sev
        mode[push] = DELAY
        count[push] = _randint(*delay, n, dev)[push]
    snap = (mode == DELAY) & (count <= 0)
    if bool(snap.any()):
      take(snap)
      mode[snap] = WINDOW
      count[snap] = window

    # 2. the walker acts, the env steps
    with torch.no_grad():
      obs, _g, dones, timeouts, _l = env.step_tensor(walker(obs))
    terminated = dones & ~timeouts
    count -= 1
    elapsed += 1

    # 3. labels: a fall inside the window labels the row; anything else that
    #    ends the episode (a timeout, or a fall before the snapshot) drops it
    in_win = mode == WINDOW
    emit(in_win & terminated, torch.ones(n, dtype=torch.bool, device=dev))
    survived = in_win & ~dones & (count <= 0)
    emit(survived, torch.zeros(n, dtype=torch.bool, device=dev))
    mode[dones] = WAIT
    count[dones] = _randint(*settle, n, dev)[dones]
    mode[survived] = WAIT
    count[survived] = _randint(*gap, n, dev)[survived]

    it += 1
    if it % 250 == 0:
      rate = it / max(time.time() - t0, 1e-6)
      print(f"[harvest] step {it}  rows {got}/{args.samples}  "
            f"({rate:.1f} steps/s)", flush=True)
  out = {k: torch.cat(v)[:args.samples] for k, v in rows.items()}
  return out


# --- stage 2: replay check ------------------------------------------------------

def replay_check(env, walker, bank, args) -> dict:
  """Restore rows into the env they came from; rerun the window; compare."""
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = mj.step_dt
  window = max(1, round(args.window_s / dt))
  _freeze_commands(mj)
  env_id = bank["env_id"]
  order = torch.argsort(env_id, stable=True)
  counts = torch.bincount(env_id, minlength=n)
  starts = torch.cumsum(counts, 0) - counts
  def rerun(ids, rows_idx):
    """Restore ``rows_idx`` into ``ids``, run the window, return who fell."""
    rows = {k: bank[k][rows_idx] for k in ("root_z", "root_quat_h", "lin_vel_h",
                                           "ang_vel_h", "joint_pos",
                                           "joint_vel")}
    mj.scene.reset(ids)                       # clear wrenches + contact history
    write_states(mj, ids, rows, yaw=_u(ids.numel(), dev, -math.pi, math.pi),
                 xy=torch.zeros(ids.numel(), 2, device=dev))
    mj.command_manager.get_command("twist")[ids] = bank["command"][rows_idx]
    mj.action_manager._action[ids] = bank["action"][rows_idx]
    mj.action_manager._prev_action[ids] = bank["action"][rows_idx]
    mj.episode_length_buf[ids] = bank["phase_step"][rows_idx]
    mj.sim.forward()
    mj.obs_buf = mj.observation_manager.compute()
    obs = mj.obs_buf[env.obs_key].float()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(window):
      with torch.no_grad():
        obs, _g, dones, timeouts, _l = env.step_tensor(walker(obs))
      fell |= dones & ~timeouts & ~ended
      ended |= dones
    return fell[ids]

  label, replay, replay2, kinds = [], [], [], []
  for r in range(args.replay_rounds):
    has = counts > r
    ids = has.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      break
    rows_idx = order[starts[ids] + r]
    # Twice: replay-vs-replay agreement is the noise floor (observation noise,
    # one-sample labels) that replay-vs-label cannot be expected to beat.
    replay.append(rerun(ids, rows_idx))
    replay2.append(rerun(ids, rows_idx))
    label.append(bank["fell"][rows_idx])
    kinds.append(bank["kind"][rows_idx])
  label, replay, kinds = torch.cat(label), torch.cat(replay), torch.cat(kinds)
  replay2 = torch.cat(replay2)
  res = {"rows": int(label.numel()),
         "agreement": float((label == replay).float().mean()),
         "replay_vs_replay": float((replay == replay2).float().mean())}
  for name, code in (("clean", KIND_CLEAN), ("pushed", KIND_PUSHED)):
    m = kinds == code
    if not bool(m.any()):
      continue
    lf, rf = label[m], replay[m]
    res[f"{name}_rows"] = int(m.sum())
    res[f"{name}_agreement"] = float((lf == rf).float().mean())
    res[f"{name}_replay_vs_replay"] = float((rf == replay2[m]).float().mean())
    res[f"{name}_label_fell"] = float(lf.float().mean())
    res[f"{name}_replay_fell"] = float(rf.float().mean())
    if bool(lf.any()):
      res[f"{name}_replay_fell_given_label_fell"] = float(rf[lf].float().mean())
    if bool((~lf).any()):
      res[f"{name}_replay_fell_given_label_ok"] = float(rf[~lf].float().mean())
  return res


# --- stage 3: synthetic ---------------------------------------------------------

def synthetic_states(env, count: int) -> dict[str, torch.Tensor]:
  mj, dev, n = env.mj, env.device, env.num_envs
  robot = mj.scene["robot"]
  sole = SoleGeometry(mj)
  default_q = robot.data.default_joint_pos[0]
  lim = robot.data.soft_joint_pos_limits[0]
  z0 = float(robot.data.default_root_state[0, 2])
  nj = default_q.numel()
  ids = torch.arange(n, device=dev)
  rows: dict[str, list] = {f: [] for f in FIELDS}
  got = 0
  while got < count:
    d = torch.rand(n, device=dev)
    roll = torch.deg2rad(d * SYN_TILT_DEG * _u(n, dev))
    pitch = torch.deg2rad(d * SYN_TILT_DEG * _u(n, dev))
    quat_h = quat_from_euler_xyz(roll, pitch, torch.zeros(n, device=dev))
    lin_h = torch.stack([d * m * _u(n, dev) for m in SYN_LIN_VEL], dim=1)
    ang_h = torch.stack([d * m * _u(n, dev) for m in SYN_ANG_VEL], dim=1)
    jpos = default_q + d[:, None] * SYN_JOINT_POS * (
      2.0 * torch.rand(n, nj, device=dev) - 1.0)
    jpos = torch.maximum(torch.minimum(jpos, lim[:, 1]), lim[:, 0])
    jvel = d[:, None] * SYN_JOINT_VEL * (2.0 * torch.rand(n, nj, device=dev) - 1.0)
    rows_now = {"root_z": torch.full((n,), z0, device=dev), "root_quat_h": quat_h,
                "lin_vel_h": torch.zeros(n, 3, device=dev),
                "ang_vel_h": torch.zeros(n, 3, device=dev),
                "joint_pos": jpos, "joint_vel": torch.zeros_like(jvel)}
    write_states(mj, ids, rows_now, yaw=torch.zeros(n, device=dev),
                 xy=torch.zeros(n, 2, device=dev))
    mj.sim.forward()
    root_z = z0 - sole.min_z() + SOLE_CLEARANCE
    k = min(n, count - got)
    sel = slice(0, k)
    rows["root_z"].append(root_z[sel])
    rows["root_quat_h"].append(quat_h[sel])
    rows["lin_vel_h"].append(lin_h[sel])
    rows["ang_vel_h"].append(ang_h[sel])
    rows["joint_pos"].append(jpos[sel])
    rows["joint_vel"].append(jvel[sel])
    rows["action"].append(torch.zeros(k, nj, device=dev))
    rows["command"].append(torch.zeros(k, 3, device=dev))
    rows["phase_step"].append(torch.zeros(k, dtype=torch.long, device=dev))
    rows["kind"].append(torch.full((k,), KIND_SYNTHETIC, dtype=torch.long,
                                   device=dev))
    rows["fell"].append(torch.zeros(k, dtype=torch.bool, device=dev))
    rows["time_to_fall"].append(torch.full((k,), float("nan"), device=dev))
    rows["push_severity"].append(torch.zeros(k, device=dev))
    rows["difficulty"].append(d[sel])
    rows["speed"].append(speed_of(lin_h[sel]))
    rows["env_id"].append(torch.full((k,), -1, dtype=torch.long, device=dev))
    got += k
  return {k: torch.cat(v) for k, v in rows.items()}


def _summary(bank) -> str:
  lines = []
  for name, code in (("clean", KIND_CLEAN), ("pushed", KIND_PUSHED),
                     ("synthetic", KIND_SYNTHETIC)):
    m = bank["kind"] == code
    if not bool(m.any()):
      continue
    sp, tl = bank["speed"][m], tilt_deg(bank["root_quat_h"][m])
    line = (f"  {name:9s} {int(m.sum()):7d} rows | speed mean {sp.mean():.2f} "
            f"max {sp.max():.2f} m/s | tilt mean {tl.mean():.1f} max "
            f"{tl.max():.1f} deg | pelvis z {bank['root_z'][m].mean():.3f} m")
    if code != KIND_SYNTHETIC:
      line += f" | walker fell {bank['fell'][m].float().mean():.1%}"
    lines.append(line)
  return "\n".join(lines)


def main():
  args = _parse()
  torch.manual_seed(args.seed)
  dev = args.device
  s = spec(args.task)
  env = make_tensor(args.task, args.num_envs, dev)
  walker = NominalPolicy(*load_nominal(args.walker, dev), device=dev,
                         bound=float(s.kwargs.get("action_bound", 1.0)))

  t0 = time.time()
  bank = harvest_walker(env, walker, args)
  print(f"[harvest] walker stage: {bank['kind'].numel()} rows in "
        f"{time.time() - t0:.0f} s")
  check = replay_check(env, walker, bank, args)
  print("[replay] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float)
                                else f"{k}={v}" for k, v in check.items()))
  syn = synthetic_states(env, args.synthetic) if args.synthetic > 0 else None
  if syn is not None:
    bank = {k: torch.cat([bank[k], syn[k]]) for k in FIELDS}
  bank = {k: v.cpu() for k, v in bank.items()}
  bank["meta"] = {
    "walker": os.path.abspath(args.walker), "task": args.task,
    "num_envs": args.num_envs, "push_frac": args.push_frac,
    "max_push_v": args.max_push_v, "max_push_w": args.max_push_w,
    "window_s": args.window_s, "seed": args.seed, "replay_check": check,
    "synthetic_max": {"tilt_deg": SYN_TILT_DEG, "lin_vel": SYN_LIN_VEL,
                      "ang_vel": SYN_ANG_VEL, "joint_pos": SYN_JOINT_POS,
                      "joint_vel": SYN_JOINT_VEL},
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  torch.save(bank, args.out)
  print(f"[harvest] wrote {args.out}\n{_summary(bank)}")


if __name__ == "__main__":
  main()
