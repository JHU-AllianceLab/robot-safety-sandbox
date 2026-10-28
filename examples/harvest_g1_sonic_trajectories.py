"""Harvest SONIC trajectories on the LAFAN clip library under a velocity push.

    python examples/harvest_g1_sonic_trajectories.py \
        --library data/g1_lafan1_clips_train.pt --episodes 50000 \
        --out data/g1_sonic_push_train.pt
    python examples/harvest_g1_sonic_trajectories.py \
        --from-traj data/g1_sonic_push_train.pt --bank-delays-s 0 0.1 0.2 0.3

SONIC (``envs/g1_recover/sonic.py``) is the nominal controller. Per env, per
episode, in the ``g1_recover_lafan_k`` env (the ViaLasso safe set K is its
margin and its termination, ``envs/g1_recover/k_set.py``: an episode ends, and
``fell`` reads True, exactly when the state leaves K) spawning on the clip
LIBRARY (the spawn's ``mocap`` source only, ``sampler="uniform"``):

1. TRACK. The episode starts on a library frame r (pose, twist, joint state
   of the clip). SONIC tracks the clip from frame r, closed loop, its
   reference turned to the robot's spawn heading.
2. PUSH. ``onset`` (U(--onset-s) s, late enough to fill SONIC's 10-tick
   histories) into the episode the pelvis velocity is SET: the root link's
   world twist gets a horizontal dv with |dv| ~ U(--dv-range) m/s, at an angle
   ``rel`` ~ U(-pi, pi) in the robot's heading frame (0 = along the heading,
   i.e. a shove from behind), plus optional vertical (--dz-range) and angular
   (--dw-range, per world axis) parts. Joint velocities are untouched, so the
   whole body takes the kick. This is Isaac Lab's / mjlab's
   ``push_by_setting_velocity``, the push SONIC was trained with (+-0.5 m/s per
   horizontal axis, every 4-6 s). It lands after the env step and before
   SONIC's next observation, where Isaac Lab's interval events land.
   ``impulse`` = m |dv| with m the robot's total mass, for comparison with the
   force pushes of the other harvesters (a velocity kick moves every link
   alike and adds no rotation, so it is the gentler push per N*s).
3. RECORD. Every control step from ``--pre-s`` before the push to
   ``--window-s`` after it (the state at the onset index is the one just after
   the kick), or to the fall. The episode then ends (its
   ``episode_length_buf`` is set to the cap, so the next step truncates it
   through the env's own ``time_out``) and the env starts the next one.

Episodes whose clip cannot be tracked through the whole window (fewer frames
left after r than onset + window) are dropped at the start, and so are
episodes that fall BEFORE the push (SONIC failing on the clip itself). Both
are counted per category. Episodes whose REFERENCE itself leaves K are kept
but flagged (``ep/ref_out_k``): a non-foot body on the floor (``ep/ref_floor``),
the pelvis below 0.45 m or a tilt over 1 rad at some frame of the episode.
That is most of the fallAndGetUp clips and a few kneels and deep crouches in
walk / fight / dance / jumps. When SONIC tracks such a stretch the episode
ends on K whatever the push did. The report's second survival column leaves
them out.

THE ROBOT. ``--hip sonic`` (default) drives the hip pitch with SONIC's
training actuator (7520-22), ``--hip mjlab`` keeps the sandbox G1's (7520-14,
the robot the g1_recover fallbacks train on). mjlab's own interval
``push_robot`` event (velocity kicks every few seconds) is removed, and the
bridge's random force push is off (``max_pushes=0``). The rest of
g1_recover's domain randomization (foot friction, torso CoM, encoder bias)
stays.

OUTPUT (one ``torch.save`` dict, CPU tensors; E episodes, T = pre + window + 1
steps, rows past a fall NaN, ``-1`` for the frame):

    traj/root_pos    (E, T, 3)   pelvis position, world, minus the spawn's x/y
    traj/root_quat   (E, T, 4)   pelvis orientation, world (w, x, y, z)
    traj/lin_vel     (E, T, 3)   root-link linear velocity, world
    traj/ang_vel     (E, T, 3)   root-link angular velocity, world
    traj/joint_pos   (E, T, 29)  joint positions (mjlab entity order, true)
    traj/joint_vel   (E, T, 29)  joint velocities
    traj/target      (E, T, 29)  SONIC's joint targets applied FROM this state
                                 (action-term order; NaN at the last step)
    traj/ref_frame   (E, T)      the clip frame SONIC tracks at this step
    traj/g           (E, T)      the task's fall margin g at this state
    traj/obs         (E, T, 994) SONIC's decoder input (--store-obs only)
    ep/row ep/clip ep/category ep/frame0      the spawn's library row
    ep/onset_s ep/dv_w (E, 3) ep/dw_w (E, 3) ep/dv_h (E, 3) ep/dv
    ep/rel           the kick's angle in the heading frame at the onset (0 =
                     along the heading = from behind, +pi/2 = toward the
                     robot's left = from its right; random_push's convention)
    ep/impulse       m |dv_h| (horizontal part), N*s
    ep/fell          the env ended the episode on a fall within the window
    ep/time_to_fall  s from the push; NaN if it survived
    ep/length        valid steps of each trajectory
    ep/pre_jerr      mean |q - q_ref| over the pre-push steps, rad (how well
                     SONIC was tracking when the push landed)
    ep/env_id        the harvest env
    ep/encoder_bias  (E, 29) the env's joint encoder bias (entity order)
    ep/dq            (E, 4) SONIC's delta heading
    ep/hist_*        (E, 10, d) SONIC's five histories at trajectory step 0
    ep/ref_clearance the lowest point of the reference's non-foot collision
                     geoms over the episode's frames (spawn to window end), m
    ep/ref_floor     ref_clearance < 0: the reference itself touches the floor
    ep/ref_out_k     the reference leaves K somewhere in the episode (floor
                     contact, pelvis < 0.45 m or tilt > 1 rad)
    meta             names (clips, categories, joint orders), the env's action
                     term, ranges, stats

``traj`` index ``pre`` is the push (``meta["onset_index"]``). SONIC's exact
inputs at any step (its histories, token and decoder input, for a failure
monitor) are rebuilt from a trajectory with ``sonic.rebuild_inputs``.

THE START BANK (``--bank-delays-s``; also from an existing trajectory file
with ``--from-traj``). Each trajectory gives one start-state row per delay d
(s after the push) it is still standing at: the state at index pre + d / dt
in the bank schema (``envs/g1_recover/states.py``, ``kind =
KIND_HANDOVER``), for the g1_recover spawn and ``label_start_states.py``.
``action`` is the env action of the target SONIC applied INTO that state
(index - 1), clamped to the bridge's +-action_bound like the mocap rows (the
fallback's own action box; SONIC's arms and wrists leave it). The kick is
instantaneous, so no push is pending. ``fell`` / ``time_to_fall`` are
SONIC's fate from that state if it KEEPS control, not the fallback's (label
those with label_start_states.py); ``handover_delay_s`` = d,
``push_impulse`` = m |dv|, ``push_severity`` = |dv| / the top of --dv-range,
``push_duration_s`` = 0, ``push_rel`` / ``push_onset_s`` / ``src_row`` /
``clip`` / ``category`` / ``frame`` / ``t`` as in the handover harvester.
``traj_index`` (a ``states.OPTIONAL_FIELDS`` column, so labeled and merged
copies keep it) is the row's episode in the trajectory file named by the
bank's ``meta["trajectories"]``. Join on it for ``ep/ref_floor`` and the
rest.
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

from mjlab.utils.lab_api.math import quat_apply  # noqa: E402

from robot_safety_sandbox.base import MjlabTensorSafetyEnv  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SONIC_JOINTS, SonicPolicy, SonicReference, SonicTracker,
  sonic_action_scale, use_sonic_hip_pitch)
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_HANDOVER, KIND_MOCAP, SOLE_GEOMS, load_bank, save_bank, speed_of,
  to_heading_frame, write_states)
from robot_safety_sandbox.random_push import SIDES  # noqa: E402
from robot_safety_sandbox.registry import spec  # noqa: E402

TRACK, WAIT = 0, 1
TRAJ = {"root_pos": 3, "root_quat": 4, "lin_vel": 3, "ang_vel": 3,
        "joint_pos": 29, "joint_vel": 29, "target": 29}


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_train.pt"))
  p.add_argument("--sonic-dir", default=DEFAULT_SONIC_DIR,
                 help="the GEAR-SONIC release (model_encoder.onnx, model_decoder.onnx)")
  p.add_argument("--out", default=os.path.join(_ZOO, "data", "g1_sonic_push_train.pt"))
  p.add_argument("--task", default="g1_recover_lafan_k")
  p.add_argument("--episodes", type=int, default=8192)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--hip", choices=("sonic", "mjlab"), default="sonic",
                 help="hip-pitch actuator: SONIC's training one or the sandbox G1's")
  p.add_argument("--dv-range", type=float, nargs=2, default=[0.0, 5.0],
                 help="horizontal |dv| of the push, m/s (uniform); SONIC survives "
                      "~half at 2.8")
  p.add_argument("--dz-range", type=float, nargs=2, default=[0.0, 0.0],
                 help="vertical dv, m/s (uniform)")
  p.add_argument("--dw-range", type=float, default=0.0,
                 help="angular dv, U(-x, x) rad/s per world axis")
  p.add_argument("--onset-s", type=float, nargs=2, default=[0.5, 1.5],
                 help="push time after the spawn, s (uniform; >= 0.2 fills the "
                      "histories)")
  p.add_argument("--pre-s", type=float, default=0.5,
                 help="record this long before the push, s (<= the onset minimum)")
  p.add_argument("--window-s", type=float, default=2.5,
                 help="record (and label) this long after the push, s")
  p.add_argument("--store-obs", action="store_true",
                 help="also store SONIC's live decoder input per step (994 floats; "
                      "small runs, to check sonic.rebuild_inputs)")
  p.add_argument("--bank-delays-s", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3],
                 help="start-bank rows at these delays after the push, s (none: no bank)")
  p.add_argument("--bank-out", default=None,
                 help="default: <out without .pt>_bank.pt")
  p.add_argument("--from-traj", default=None,
                 help="no harvest: write the start bank of this trajectory file")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args(argv)
  lo, hi = args.dv_range
  if not 0.0 <= lo <= hi:
    p.error("--dv-range needs 0 <= lo <= hi")
  if not 0.0 <= args.pre_s <= args.onset_s[0] <= args.onset_s[1]:
    p.error("need 0 <= --pre-s <= onset lo <= onset hi")
  if any(d < 0.0 or d > args.window_s for d in args.bank_delays_s):
    p.error("--bank-delays-s must lie in [0, --window-s]")
  return args


def make_env(args, overrides: dict, push_kw: dict) -> MjlabTensorSafetyEnv:
  """The task's env with mjlab's interval velocity push removed and, under
  ``--hip sonic``, SONIC's hip-pitch actuator."""
  s = spec(args.task)

  def builder(play: bool = False, **kw):
    cfg = s.cfg_builder(play=play, **kw)
    cfg.events.pop("push_robot", None)
    if args.hip == "sonic":
      use_sonic_hip_pitch(cfg)
    return cfg

  kw = dict(s.kwargs, random_push=push_kw)
  return MjlabTensorSafetyEnv(
    args.num_envs, args.device, cfg_builder=builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=False,
    end_criterion=s.end_criterion, cfg_overrides=overrides, **kw)


class NonFootClearance:
  """The G1's non-foot collision geoms (capsules and spheres), for the
  reference's own floor contact. :meth:`min_z` is the lowest point of any of
  them, per env, after a ``sim.forward()``: ``states.SoleGeometry``'s rule,
  a sphere being a capsule of half-length 0."""

  def __init__(self, env):
    import mujoco
    robot = env.scene["robot"]
    local, names = robot.find_geoms(".*_collision")
    keep = [i for i, nm in zip(local, names) if nm not in SOLE_GEOMS]
    self._robot = robot
    self._local = torch.as_tensor(keep, device=env.device)
    gid = robot.indexing.geom_ids[self._local].cpu().numpy()
    m = env.sim.mj_model
    kind = m.geom_type[gid]
    ok = (kind == mujoco.mjtGeom.mjGEOM_CAPSULE) | (kind == mujoco.mjtGeom.mjGEOM_SPHERE)
    if not ok.all() or len(keep) == 0:
      raise RuntimeError("expected the G1's non-foot collision geoms to be "
                         "capsules and spheres")
    size = m.geom_size[gid]
    half = size[:, 1] * (kind == mujoco.mjtGeom.mjGEOM_CAPSULE)
    self._radius = torch.as_tensor(size[:, 0], dtype=torch.float32, device=env.device)
    self._half = torch.as_tensor(half, dtype=torch.float32, device=env.device)

  def min_z(self) -> torch.Tensor:
    d = self._robot.data
    pos = d.geom_pos_w[:, self._local]
    quat = d.geom_quat_w[:, self._local]
    axis = quat_apply(quat.reshape(-1, 4),
                      torch.tensor([0.0, 0.0, 1.0], device=pos.device)
                      .expand(quat.shape[0] * quat.shape[1], 3))
    axis_z = axis[:, 2].reshape(quat.shape[0], quat.shape[1])
    return (pos[..., 2] - (self._half * axis_z).abs() - self._radius).amin(dim=1)


def reference_clearance(env, lib: dict) -> torch.Tensor:
  """The lowest non-foot point of every library row's pose (the clip's
  kinematic reference, soles snapped to the floor as the library stores it),
  m, ``[rows]`` on the env's device. Writes the rows into the env batch by
  batch: reset the env afterwards."""
  mj, dev, n = env.mj, env.device, env.num_envs
  clr = NonFootClearance(mj)
  ids = torch.arange(n, device=dev)
  zero_yaw, zero_xy = torch.zeros(n, device=dev), torch.zeros(n, 2, device=dev)
  fields = ("root_z", "root_quat_h", "lin_vel_h", "ang_vel_h", "joint_pos", "joint_vel")
  M = int(lib["root_z"].numel())
  out = torch.empty(M, device=dev)
  for b0 in range(0, M, n):
    k = min(n, M - b0)
    pad = torch.arange(n, device=dev) % k
    batch = {f: lib[f][b0:b0 + k].to(dev)[pad] for f in fields}
    write_states(mj, ids, batch, zero_yaw, zero_xy)
    mj.sim.forward()
    out[b0:b0 + k] = clr.min_z()[:k]
  return out


def reference_out_k(lib: dict, clearance: torch.Tensor) -> torch.Tensor:
  """Library rows whose pose is outside K by itself: a non-foot geom below
  the floor (``clearance < 0``), the pelvis below K's 0.45 m or a tilt over
  its 1 rad (``k_set``). Every library joint lies within its range."""
  from robot_safety_sandbox.envs.g1_recover import k_set
  dev = clearance.device
  z = lib["root_z"].to(dev)
  tl = k_set.tilt(lib["root_quat_h"].to(dev))
  return (clearance < 0.0) | (z < k_set.H_MIN) | (tl > k_set.THETA_MAX)


class Harvest:
  """Per-env episode bookkeeping (module docstring)."""

  def __init__(self, env, lib: dict, tracker: SonicTracker, args, margin_fn,
               clearance: torch.Tensor, ref_out: torch.Tensor):
    self.env, self.mj, self.args, self.tr = env, env.mj, args, tracker
    self.margin_fn, self.clear, self.ref_out = margin_fn, clearance, ref_out
    self.dev, self.n = env.device, env.num_envs
    self.dt = float(self.mj.step_dt)
    if not hasattr(self.mj, "_zoo_spawn_row"):
      raise SystemExit(f"task {args.task!r}: its spawn publishes no _zoo_spawn_row")
    self.robot = self.mj.scene["robot"]
    self.mass = float(self.mj.sim.mj_model.body_mass.sum())   # the robot's (the floor has none)
    dev, n = self.dev, self.n
    self.lib = {k: lib[k].to(dev) for k in ("clip", "category", "frame", "next_ok")}
    steps = lambda s: int(round(float(s) / self.dt))  # noqa: E731
    self.pre, self.win = steps(args.pre_s), steps(args.window_s)
    if steps(args.onset_s[0]) < 1:
      # the push lands after a step: an onset of 0 steps would never fire
      raise SystemExit(f"--onset-s must round to >= 1 control step ({self.dt} s)")
    self.T = self.pre + self.win + 1
    long = dict(dtype=torch.long, device=dev)
    self.mode = torch.full((n,), WAIT, **long)
    self.k = torch.zeros(n, **long)          # steps since the spawn
    self.on = torch.zeros(n, **long)         # the push step
    self.row = torch.full((n,), -1, **long)
    self.xy0 = torch.zeros(n, 2, device=dev)
    self.dv_w = torch.zeros(n, 3, device=dev)
    self.dw_w = torch.zeros(n, 3, device=dev)
    self.dv_h = torch.zeros(n, 3, device=dev)
    self.rel = torch.zeros(n, device=dev)
    self.jerr = torch.zeros(n, device=dev)
    self.pushed = torch.zeros(n, dtype=torch.bool, device=dev)
    self.ref_clear = torch.zeros(n, device=dev)
    self.ref_bad = torch.zeros(n, dtype=torch.bool, device=dev)
    dims = dict(TRAJ, obs=994) if args.store_obs else TRAJ
    self.buf = {k: torch.full((n, self.T, d), float("nan"), device=dev)
                for k, d in dims.items()}
    self.buf["ref_frame"] = torch.full((n, self.T), -1, **long)
    self.buf["g"] = torch.full((n, self.T), float("nan"), device=dev)
    self.snap: dict[str, torch.Tensor] = {}
    self.out: dict[str, list] = {}
    self.got = 0
    self.started = 0
    ncat = len(lib["meta"].get("categories", [])) + 1
    self.ep_cat = torch.zeros(ncat, **long)
    self.prefix_cat = torch.zeros(ncat, **long)
    self.short = 0

  def _cat_bucket(self, rows: torch.Tensor) -> torch.Tensor:
    cat = self.lib["category"][rows]
    return torch.where(cat < 0, torch.full_like(cat, self.ep_cat.numel() - 1), cat)

  def _index(self) -> torch.Tensor:
    """Each env's trajectory index now (k - onset + pre)."""
    return self.k - self.on + self.pre

  # --- a new episode -----------------------------------------------------------
  def begin(self, mask: torch.Tensor) -> None:
    """Envs in ``mask`` were just (re)spawned: draw the push, start SONIC on
    the spawn's clip frame. The rest of the envs are left alone."""
    a, dev, dt = self.args, self.dev, self.dt
    rows = self.mj._zoo_spawn_row.clone()
    if bool((rows[mask] < 0).any()):
      raise SystemExit("an env spawned in the nominal stance: the library has no "
                       "start_ok mocap rows, or the spawn ignored weights/sampler")
    on_s = a.onset_s[0] + (a.onset_s[1] - a.onset_s[0]) * torch.rand(self.n, device=dev)
    on = torch.round(on_s / dt).long()
    room = self.lib["next_ok"][rows.clamp_min(0)]
    fits = room >= on + self.win
    go = mask & fits
    # capped: start only while labeled + in-flight episodes fall short
    inflight = int(((self.mode == TRACK) & ~mask).sum())
    need = self.args.episodes - self.got - inflight
    if need > 0:
      self.short += int((mask & ~fits).sum())
    ids = go.nonzero(as_tuple=False).squeeze(-1)
    take = torch.zeros_like(go)
    if need > 0:
      take[ids[:need]] = True
    self.mode[mask] = WAIT
    self._finish(mask & ~take)
    if not bool(take.any()):
      return
    self.mode[take] = TRACK
    self.k[take] = 0
    self.on = torch.where(take, on, self.on)
    self.row = torch.where(take, rows, self.row)
    self.xy0[take] = self.robot.data.root_link_pos_w[take, :2]
    for b in self.buf.values():
      b[take] = -1 if b.dtype == torch.long else float("nan")
    self.jerr[take] = 0.0
    self.pushed[take] = False
    # the reference's own floor contact over the frames this episode tracks
    # (spawn to window end; fits guarantees they stay inside the clip)
    r0, span = rows[take], (on + self.win)[take]
    steps_ = torch.arange(int(span.max()) + 1, device=dev)
    idx = torch.where(steps_ <= span.unsqueeze(1), r0.unsqueeze(1) + steps_,
                      r0.unsqueeze(1))
    self.ref_clear[take] = self.clear[idx].amin(1)
    self.ref_bad[take] = self.ref_out[idx].any(1)
    # the push, drawn now
    lo, hi = a.dv_range
    mag = lo + (hi - lo) * torch.rand(self.n, device=dev)
    rel = (torch.rand(self.n, device=dev) * 2.0 - 1.0) * math.pi
    zlo, zhi = a.dz_range
    dz = zlo + (zhi - zlo) * torch.rand(self.n, device=dev)
    dv_h = torch.stack([mag * torch.cos(rel), mag * torch.sin(rel), dz], 1)
    dw = (torch.rand(self.n, 3, device=dev) * 2.0 - 1.0) * a.dw_range
    self.dv_h = torch.where(take.unsqueeze(1), dv_h, self.dv_h)
    self.rel = torch.where(take, rel, self.rel)
    self.dw_w = torch.where(take.unsqueeze(1), dw, self.dw_w)
    self.tr.start(take, self.lib["clip"][rows.clamp_min(0)],
                  self.lib["frame"][rows.clamp_min(0)])
    cat = self._cat_bucket(rows[take])
    self.ep_cat.index_add_(0, cat, torch.ones_like(cat))
    self.started += int(take.sum())

  def _finish(self, mask: torch.Tensor) -> None:
    """End these episodes on the next step (a truncation through the env's own
    time_out term), so the env moves on to a new start."""
    self.mode[mask] = WAIT
    self.tr.stop(mask)
    buf = self.mj.episode_length_buf
    buf[mask] = torch.clamp_min(buf[mask], int(self.mj.max_episode_length) - 1)

  # --- the push ----------------------------------------------------------------
  def push(self, mask: torch.Tensor) -> None:
    if not bool(mask.any()):
      return
    d = self.robot.data
    h = d.heading_w.reshape(-1)
    c, s = torch.cos(h), torch.sin(h)
    dv_w = torch.stack([c * self.dv_h[:, 0] - s * self.dv_h[:, 1],
                        s * self.dv_h[:, 0] + c * self.dv_h[:, 1], self.dv_h[:, 2]], 1)
    self.dv_w = torch.where(mask.unsqueeze(1), dv_w, self.dv_w)
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    vel = d.root_link_vel_w[ids].clone()
    vel[:, :3] += dv_w[ids]
    vel[:, 3:] += self.dw_w[ids]
    self.robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
    self.pushed |= mask
    # the twist accessors read cvel: refresh it, so SONIC's next observation
    # and the recorded state see the kick (Isaac Lab's buffers do at once)
    self.mj.sim.forward()

  # --- recording -----------------------------------------------------------------
  def record(self, mask: torch.Tensor, g: torch.Tensor) -> None:
    """Write the current state of the envs in ``mask`` at their trajectory
    index; at index 0 also snapshot SONIC's histories (the rebuild's seed)."""
    j = self._index()
    m = mask & (j >= 0) & (j < self.T)
    if not bool(m.any()):
      return
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    jj = j[ids]
    d = self.robot.data
    pos = d.root_link_pos_w[ids].clone()
    pos[:, :2] -= self.xy0[ids]
    vals = {"root_pos": pos, "root_quat": d.root_link_quat_w[ids],
            "lin_vel": d.root_link_lin_vel_w[ids], "ang_vel": d.root_link_ang_vel_w[ids],
            "joint_pos": d.joint_pos[ids], "joint_vel": d.joint_vel[ids]}
    for key, v in vals.items():
      self.buf[key][ids, jj] = v
    self.buf["ref_frame"][ids, jj] = self.tr.frame[ids]
    self.buf["g"][ids, jj] = g[ids]
    first = ids[jj == 0]
    if first.numel():
      for key, v in self.tr.snapshot(first).items():
        if key not in self.snap:
          self.snap[key] = torch.zeros((self.n,) + v.shape[1:], device=self.dev)
        self.snap[key][first] = v
      if "encoder_bias" not in self.snap:
        self.snap["encoder_bias"] = torch.zeros(self.n, 29, device=self.dev)
      self.snap["encoder_bias"][first] = self.robot.data.encoder_bias[first]

  def record_action(self, mask: torch.Tensor) -> None:
    """The target SONIC applies from each env's current state (and, with
    --store-obs, the decoder input it read)."""
    j = self._index()
    m = mask & (j >= 0) & (j < self.T)
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    self.buf["target"][ids, j[ids]] = self.tr.target[ids]
    if "obs" in self.buf:
      self.buf["obs"][ids, j[ids]] = self.tr.obs[ids]

  def emit(self, mask: torch.Tensor, fell: torch.Tensor) -> None:
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    dt, lib = self.dt, self.lib
    after = (self.k[ids] - self.on[ids]).clamp_min(0)
    ep = {
      "row": self.row[ids], "clip": lib["clip"][self.row[ids]],
      "category": lib["category"][self.row[ids]], "frame0": lib["frame"][self.row[ids]],
      "onset_s": self.on[ids].float() * dt, "dv_w": self.dv_w[ids], "dw_w": self.dw_w[ids],
      "dv_h": self.dv_h[ids], "rel": self.rel[ids], "dv": self.dv_h[ids, :2].norm(dim=1),
      "impulse": self.mass * self.dv_h[ids, :2].norm(dim=1),
      "fell": fell[ids],
      "time_to_fall": torch.where(fell[ids], after.float() * dt,
                                  torch.full(after.shape, float("nan"), device=after.device)),
      # a fall's own step is not recorded (mjlab has already respawned the env)
      "length": (self.pre + after + (~fell[ids]).long()).clamp_max(self.T),
      "pre_jerr": self.jerr[ids] / max(self.pre, 1),
      "env_id": ids.clone(),
      "ref_clearance": self.ref_clear[ids], "ref_floor": self.ref_clear[ids] < 0.0,
      "ref_out_k": self.ref_bad[ids],
    }
    if not bool(self.pushed[ids].all()):
      raise RuntimeError("an episode was emitted without its push")
    ep.update({k: v[ids] for k, v in self.snap.items()})
    for k, v in ep.items():
      self.out.setdefault("ep/" + k, []).append(v.cpu())
    for k, v in self.buf.items():
      self.out.setdefault("traj/" + k, []).append(v[ids].cpu())
    self.got += int(ids.numel())

  # --- the loop ------------------------------------------------------------------
  def run(self) -> dict:
    a, dev, n = self.args, self.dev, self.n
    env, tr = self.env, self.tr
    env.reset()
    self.begin(torch.ones(n, dtype=torch.bool, device=dev))
    # g of the spawn states, for the episodes whose index 0 is the spawn
    # (onset == pre); every later index records the step's own g
    self.record(self.mode == TRACK, self.margin_fn(self.mj)[0])
    t0, it = time.time(), 0
    while self.got < a.episodes:
      track = self.mode == TRACK
      act = tr.act()
      self.record_action(track)
      _obs, g, dones, timeouts, _l = env.step_tensor(act)
      self.k += 1
      fell = dones & ~timeouts
      # the push lands after the step, before SONIC's next observation
      due = track & ~dones & (self.k == self.on)
      self.push(due)
      tr.advance()
      live = track & ~dones
      self.record(live, g)
      r = tr.ref.rows(tr.clip, tr.frame)
      jerr = (self.robot.data.joint_pos[:, tr.to_sonic] - tr.ref.joint_pos[r]).abs().mean(1)
      self.jerr += torch.where(live & (self.k > self.on - self.pre) & (self.k <= self.on),
                               jerr, 0.0)
      # outcomes (a fall in the step that reaches the onset is before the push:
      # the push lands after a step, and only on an env still standing)
      after = self.k - self.on
      before = track & dones & (after <= 0)                  # fell on the clip itself
      if bool(before.any()):
        cat = self._cat_bucket(self.row[before])
        self.prefix_cat.index_add_(0, cat, torch.ones_like(cat))
      self.emit(track & fell & (after > 0), fell)
      up = live & (after >= self.win)
      self.emit(up, torch.zeros_like(fell))
      self._finish(up)
      self.mode[track & dones] = WAIT
      tr.stop(dones)
      # new episodes on the envs mjlab just respawned
      if bool(dones.any()):
        self.begin(dones)
        self.record(dones & (self.mode == TRACK), self.margin_fn(self.mj)[0])
      it += 1
      if it % 250 == 0:
        el = max(time.time() - t0, 1e-6)
        print(f"[sonic] step {it}  episodes {self.got}/{a.episodes}  started "
              f"{self.started}  falls before the push {int(self.prefix_cat.sum())}  "
              f"({it * n / el:.0f} env-steps/s)", flush=True)
    return {k: torch.cat(v)[:a.episodes] for k, v in self.out.items()}


def _report(res: dict, h: Harvest, lib_meta: dict, args) -> dict:
  fell = res["ep/fell"]
  surv = ~fell
  E = int(fell.numel())
  stats = {"episodes": E, "started": h.started, "survival": float(surv.float().mean()),
           "falls_before_push": int(h.prefix_cat.sum()),
           "dropped_short_clip": h.short, "mass_kg": h.mass}
  print(f"[sonic] {E} episodes, survival after the push {stats['survival']:.3f}; "
        f"{stats['falls_before_push']} episodes fell before their push "
        f"({100 * stats['falls_before_push'] / max(h.started, 1):.1f}%), "
        f"{h.short} spawns dropped (clip too short)")
  lo, hi = args.dv_range
  edges = torch.linspace(lo, hi, 11).tolist()
  cats = list(lib_meta.get("categories", []))
  # the episodes whose fall can be the push's alone: neither a fallAndGetUp
  # clip nor a window where the reference itself leaves K
  real = ((res["ep/category"] != (cats.index("fallAndGetUp") if "fallAndGetUp" in cats else -2))
          & ~res["ep/ref_out_k"])
  stats["ref_floor_episodes"] = int(res["ep/ref_floor"].sum())
  stats["ref_out_k_episodes"] = int(res["ep/ref_out_k"].sum())
  by_dv = {}
  print("[sonic] survival by |dv| (impulse = m |dv|, m = %.1f kg; all clips / "
        "without fallAndGetUp and a reference outside K, %d episodes left out):"
        % (h.mass, int((~real).sum())))
  for e0, e1 in zip(edges[:-1], edges[1:]):
    m = (res["ep/dv"] >= e0) & ((res["ep/dv"] < e1) if e1 != edges[-1] else (res["ep/dv"] <= e1))
    if int(m.sum()):
      s = float(surv[m].float().mean())
      s2 = float(surv[m & real].float().mean()) if bool((m & real).any()) else float("nan")
      by_dv[f"{e0:.2f}_{e1:.2f}"] = {"all": s, "push_only": s2}
      print(f"    {e0:.2f}-{e1:.2f} m/s ({h.mass * e0:5.1f}-{h.mass * e1:5.1f} N*s): "
            f"{int(m.sum()):6d} episodes, survival {s:.3f} / {s2:.3f}")
  stats["by_dv"] = by_dv
  # the side the kick comes FROM, random_push's sectors (0 = from behind,
  # counterclockwise: right, front, left)
  side = (torch.remainder(res["ep/rel"] + math.pi / 4, 2.0 * math.pi)
          // (math.pi / 2)).long().clamp_(0, len(SIDES) - 1)
  moved = {"behind": "pushed forward", "right": "pushed left",
           "front": "pushed back", "left": "pushed right"}
  stats["by_side"] = {}
  for i, name in enumerate(SIDES):
    m = side == i
    if int(m.sum()):
      stats["by_side"][name] = float(surv[m].float().mean())
  print(f"[sonic] survival by the side the kick comes from: "
        + ", ".join(f"{k} ({moved[k]}) {v:.3f}" for k, v in stats["by_side"].items()))
  names = cats + ["-"]
  per = {}
  print("[sonic] per category (survival after the push / falls before it):")
  for i, c in enumerate(names):
    started = int(h.ep_cat[i])
    if started == 0:
      continue
    m = res["ep/category"] == (i if i < len(names) - 1 else -1)
    s = float(surv[m].float().mean()) if bool(m.any()) else float("nan")
    pf = int(h.prefix_cat[i]) / started
    nf = int((m & res["ep/ref_out_k"]).sum())
    per[c] = {"episodes": int(m.sum()), "survival": s, "falls_before_push": pf,
              "ref_out_k": nf}
    print(f"    {c:15s} {int(m.sum()):6d} episodes  survival {s:.3f}  "
          f"fell before the push {pf:.3f}  reference outside K {nf}")
  stats["per_category"] = per
  stats["pre_jerr_mean"] = float(res["ep/pre_jerr"].mean())
  return stats


# --- the start bank --------------------------------------------------------------

def trajectories_to_bank(res: dict, delays_s, traj_path: str) -> dict:
  """Start-state rows of trajectory file ``res`` at ``delays_s`` after each
  push (module docstring, THE START BANK)."""
  meta = res["meta"]
  dt, pre = float(meta["dt"]), int(meta["onset_index"])
  length, fell = res["ep/length"], res["ep/fell"]
  scale = torch.as_tensor(meta["env_action_scale"])
  offset = torch.as_tensor(meta["env_action_offset"])
  bound = float(meta["action_bound"])
  dv_hi = float(meta["dv_range"][1]) or 1.0
  cols: dict[str, list] = {}
  for d in delays_s:
    j = pre + int(round(float(d) / dt))
    # standing at j, and j >= 1 so the target INTO it exists
    ok = (length > j) & (j >= 1)
    e = ok.nonzero(as_tuple=False).squeeze(-1)
    if e.numel() == 0:
      continue
    q = res["traj/root_quat"][e, j]
    quat_h, lin_h, ang_h = to_heading_frame(q, res["traj/lin_vel"][e, j],
                                            res["traj/ang_vel"][e, j])
    act = ((res["traj/target"][e, j - 1] - offset) / scale).clamp(-bound, bound)
    ttf = res["ep/time_to_fall"][e] - float(d)
    clip, frame = res["ep/clip"][e], res["traj/ref_frame"][e, j]
    m = int(e.numel())
    rows = {
      "root_z": res["traj/root_pos"][e, j, 2], "root_quat_h": quat_h,
      "lin_vel_h": lin_h, "ang_vel_h": ang_h,
      "joint_pos": res["traj/joint_pos"][e, j], "joint_vel": res["traj/joint_vel"][e, j],
      "action": act, "command": torch.zeros(m, 3),
      "phase_step": torch.zeros(m, dtype=torch.long),
      "kind": torch.full((m,), KIND_HANDOVER, dtype=torch.long),
      "fell": fell[e], "time_to_fall": ttf,
      "push_severity": res["ep/dv"][e] / dv_hi, "difficulty": torch.zeros(m),
      "speed": speed_of(lin_h), "env_id": res["ep/env_id"][e],
      "clip": clip, "category": res["ep/category"][e], "frame": frame,
      "t": frame.float() / 50.0,
      "push_force_h": torch.zeros(m, 3), "push_left_steps": torch.zeros(m, dtype=torch.long),
      "push_impulse": res["ep/impulse"][e],
      "handover_delay_s": torch.full((m,), float(d)),
      "push_duration_s": torch.zeros(m), "push_onset_s": res["ep/onset_s"][e],
      "push_rel": res["ep/rel"][e], "src_row": res["ep/row"][e],
      "traj_index": e,
    }
    for k, v in rows.items():
      cols.setdefault(k, []).append(v)
  bank = {k: torch.cat(v) for k, v in cols.items()}
  assert all(f in bank for f in FIELDS)
  n = int(bank["kind"].numel())
  bank["meta"] = {
    "source": "harvest_g1_sonic_trajectories.py (start bank)",
    "trajectories": os.path.abspath(traj_path), "delays_s": [float(d) for d in delays_s],
    "library": meta.get("library"), "clips": list(meta.get("clips", [])),
    "categories": list(meta.get("categories", [])), "split": meta.get("split"),
    "hip": meta.get("hip"), "dv_range": meta.get("dv_range"),
    "labels": "fell / time_to_fall are SONIC's (the nominal keeps control from the "
              "row), not the fallback's; label the fallback with label_start_states.py",
    "action": f"env action of SONIC's target into the row, clamped to +-{bound:g}",
    "pending_push": "none (the velocity kick is instantaneous)",
    "rows": n, "fell_rows": int(bank["fell"].sum()),
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
  }
  return bank


def write_bank(res: dict, args, traj_path: str) -> None:
  if not args.bank_delays_s:
    return
  out = args.bank_out or (os.path.splitext(traj_path)[0] + "_bank.pt")
  bank = trajectories_to_bank(res, args.bank_delays_s, traj_path)
  save_bank(bank, out)
  per = {f"{d:g}": int((bank["handover_delay_s"] == float(d)).sum()) for d in args.bank_delays_s}
  print(f"[sonic] start bank: {bank['meta']['rows']} rows ({bank['meta']['fell_rows']} "
        f"SONIC falls) at delays {per} -> {out}")


def main(argv=None):
  args = _parse(argv)
  if args.from_traj:
    res = torch.load(args.from_traj, map_location="cpu", weights_only=False)
    write_bank(res, args, args.from_traj)
    return
  torch.manual_seed(args.seed)
  dev = args.device
  lib = load_bank(args.library, "cpu")
  if not bool((lib["next_ok"] > 0).any()):
    raise SystemExit(f"{args.library} is not a clip library (no next_ok); build "
                     f"one with build_g1_recover_mocap_states.py --library")
  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  overrides = dict(dataset=args.library, sampler="uniform", weights=weights,
                   episode_length_s=float(args.onset_s[1] + args.window_s + 1.0),
                   rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  env = make_env(args, overrides, push_kw)
  t0 = time.time()
  policy = SonicPolicy.from_onnx(args.sonic_dir).to(dev).eval()
  ref = SonicReference.build(lib["meta"]["csv_dir"], lib["meta"]["clips"], dev, library=lib)
  tracker = SonicTracker(env, policy, ref)
  ok = lib["start_ok"] & (lib["kind"] == KIND_MOCAP)
  print(f"[sonic] {args.library}: {int(ok.sum())} spawnable frames; hip pitch "
        f"{args.hip}; push |dv| {args.dv_range} m/s, dz {args.dz_range}, dw +-{args.dw_range} "
        f"rad/s, onset {args.onset_s} s; record {args.pre_s} s before to "
        f"{args.window_s} s after ({time.time() - t0:.0f}s setup)")
  t1 = time.time()
  clearance = reference_clearance(env, lib)
  print(f"[sonic] reference floor clearance of {clearance.numel()} library rows "
        f"({time.time() - t1:.0f}s): {float((clearance < 0).float().mean()):.3f} "
        f"have a non-foot geom below the floor")
  ref_out = reference_out_k(lib, clearance)
  print(f"[sonic] library rows outside K by themselves: {float(ref_out.float().mean()):.3f}")
  h = Harvest(env, lib, tracker, args, spec(args.task).margin_fn, clearance, ref_out)
  res = h.run()
  stats = _report(res, h, lib["meta"], args)
  robot = env.mj.scene["robot"]
  lm = lib["meta"]
  tnames = [robot.joint_names[i] for i in tracker.act_ids.tolist()]
  res["meta"] = {
    "source": "harvest_g1_sonic_trajectories.py", "library": os.path.abspath(args.library),
    "sonic_dir": os.path.abspath(args.sonic_dir), "sonic_variant": "default (sonic_release)",
    "clips": list(lm.get("clips", [])), "categories": list(lm.get("categories", [])),
    "split": lm.get("split"), "csv_dir": lm.get("csv_dir"), "task": args.task,
    "hip": args.hip, "dv_range": list(args.dv_range), "dz_range": list(args.dz_range),
    "dw_range": args.dw_range, "onset_s": list(args.onset_s), "pre_s": args.pre_s,
    "window_s": args.window_s, "dt": h.dt, "onset_index": h.pre, "T": h.T,
    "joint_names": list(robot.joint_names), "target_joint_names": tnames,
    "sonic_joint_names": list(SONIC_JOINTS),
    "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
    "sonic_action_scale": sonic_action_scale(tnames),
    "env_action_scale": tracker.env_scale.tolist(),
    "env_action_offset": tracker.env_offset.tolist(),
    "action_bound": float(env.action_bound), "mass_kg": h.mass,
    "store_obs": bool(args.store_obs), "seed": args.seed,
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
  torch.save(res, args.out)
  print(f"[sonic] wrote {stats['episodes']} trajectories ({int(res['ep/fell'].sum())} "
        f"fell) to {args.out} ({time.time() - t0:.0f}s)")
  env.close()
  write_bank(res, args, args.out)


if __name__ == "__main__":
  main()
