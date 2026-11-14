"""``TrackRefCommand`` ("ref"): the squat reference as an mjlab command term.

It holds per env the frame ``f`` of the reference the robot should reach with
its next action, and where that env's copy of the reference is placed in the
world (a yaw and a planar offset). It duck-types mjlab's tracking
``MotionCommand`` (``body_pos_w``, ``robot_body_pos_w``, ``body_pos_relative_w``,
``cfg.body_names`` ...), so mjlab's tracking rewards and observations run on it
unchanged -- they only ``cast`` the term.

TIMING (mjlab's step: actions, physics, terminations, rewards, step events,
auto-reset, forward, command update, observations). A reset spawns the robot ON
frame s and the command update that follows moves every env, the reset ones
included, to f = s + 1, the frame the first action aims at. Terminations and
rewards of step k therefore grade the state reached by action k against frame
f = s + k, and ``reference_finished`` (mdp.py) fires once f = T - 1: a start at
s lasts T - 1 - s steps.

THE SPAWN lives in this term's resample (mjlab resets the command manager after
the scene, the reset events, the action manager and the metrics -- so the start
temperatures are already drawn and stay untouched, and the action buffer is
already zeroed for the seed below). ``resampling_time_range`` is (1e9, 1e9): a
resample mid-episode would teleport the robot without a done. Per reset env:

1. start frame s: ``start_override`` if set (eval), else 0 with probability
   ``p_start`` and otherwise uniform over 0 .. T-1-``rsi_min_left_s``/dt;
2. heading and offset: ``states.sample_yaw_xy`` (yaw U(+-pi), xy U(+-0.5 m));
3. the reference state of frame s at that heading (``states.write_states``);
   with ``spawn_noise`` (training): roll / pitch +-0.05 rad, linear velocity
   +-0.1 m/s, angular velocity +-0.2 rad/s, joints +-0.05 rad clamped into the
   hard range shrunk by 0.01 rad, and the robot raised ``spawn_lift_m`` so the
   tilt rarely pushes a sole into the floor. Noise-free spawns are the
   reference state exactly;
4. ``last_action`` seeded with the PD target that holds the spawned pose,
   ``(q - offset) / scale`` clamped to the task's action box (the bank builder's
   rule), as ``RecoverSpawn`` seeds it;
5. f = start_f = s, and the reference placed so its root at s is where the robot is.

Every draw uses torch's global RNG after the thermal draw, so a seeded
evaluation pairs start frames, headings and temperatures across conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat)

from robot_safety_sandbox.envs.g1_recover.states import (
  sample_yaw_xy, to_heading_frame, write_states)
from robot_safety_sandbox.envs.g1_track.reference import (
  ANCHOR_BODY, TRACK_BODIES, SegmentSpec, build_reference, placement,
  sample_start_frames)

#: the task policy's action box (bridge ``action_bound``), the user's +-6.5
#: (AMENDMENTS A2): the frozen action term's knee reaches 0.669 + 6.5 * 0.351 =
#: 2.95 rad, past the reference's 2.82; the fallback keeps its +-3
TASK_ACTION_BOUND = 6.5
#: joints the joint-position reward and error leave out: the wrist pitch / yaw,
#: whose +-0.22 rad box leaves most of their reference out of reach
ERR_JOINT_EXCLUDE = r".*_wrist_(pitch|yaw)_joint"


@dataclass(kw_only=True)
class TrackRefCommandCfg(CommandTermCfg):
  """The reference command (module docstring)."""

  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  entity_name: str = "robot"
  segment: SegmentSpec = field(default_factory=SegmentSpec)
  body_names: tuple[str, ...] = TRACK_BODIES
  anchor_body_name: str = ANCHOR_BODY
  #: probability of a start at frame 0; else a uniform frame leaving >= rsi_min_left_s
  p_start: float = 0.5
  rsi_min_left_s: float = 1.0
  spawn_noise: bool = True
  spawn_lift_m: float = 0.005
  noise_roll_pitch: float = 0.05
  noise_lin_vel: float = 0.1
  noise_ang_vel: float = 0.2
  noise_joint: float = 0.05
  #: spawned joints stay this far inside the hard range
  joint_margin: float = 0.01
  xy_half: float = 0.5
  action_name: str = "joint_pos"
  seed_action_bound: float = TASK_ACTION_BOUND

  def build(self, env) -> "TrackRefCommand":
    return TrackRefCommand(self, env)


class TrackRefCommand(CommandTerm):
  """The squat reference, per env: frame ``f``, start frame ``start_f``, and the
  world placement ``yaw_q`` / ``offset`` (module docstring).

  Eval controls: ``start_override`` (None, an int, or a LongTensor
  ``[num_envs]``) fixes the start frame of every later reset, and
  ``spawn_noise`` (bool) switches the spawn noise; both are read at each reset.
  """

  cfg: TrackRefCommandCfg

  def __init__(self, cfg: TrackRefCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    dev = self.device
    ref = build_reference(cfg.segment, body_names=tuple(cfg.body_names))
    jids = self.robot.find_joints(ref.joint_names, preserve_order=True)[0]
    nj = len(self.robot.joint_names)
    if len(jids) != nj:
      raise ValueError(f"the reference drives {len(jids)} of the robot's {nj} joints")
    rng = self.robot.data.joint_pos_limits[0, jids].float().cpu()
    if not torch.allclose(rng, ref.joint_range, atol=1e-5):
      raise ValueError("the reference's FK model is not this env's robot (joint "
                       "ranges differ): it is built on envs/assets/g1_sysid's G1")
    self.ref = ref
    self.T = ref.T
    m = ref.meta
    print(f"[g1_track] {m['clip']} {m['t0']:g}-{m['t1']:g} s = frames {m['f0']}..{m['f1']} "
          f"(T {m['T']}, {m['grid']} grid), clipped {m['clipped_low'] | m['clipped_high']}, "
          f"airborne {m['airborne_frames']}, lowest non-sole clearance "
          f"{m['nonfoot_clear_min']:.3f} m")
    # entity joint order (the reference is in CSV order; equal on the G1, mapped anyway)
    jids = torch.as_tensor(jids, device=dev)
    self.q = torch.zeros(self.T, nj, device=dev)
    self.qd = torch.zeros(self.T, nj, device=dev)
    self.q[:, jids] = ref.q.to(dev)
    self.qd[:, jids] = ref.qd.to(dev)
    self.ref_body_pos = ref.body_pos.to(dev)
    self.ref_body_quat = ref.body_quat.to(dev)
    self.ref_body_lin = ref.body_lin.to(dev)
    self.ref_body_ang = ref.body_ang.to(dev)
    self.ref_root_pos = ref.root_pos.to(dev)
    self.ref_root_quat = ref.root_quat.to(dev)
    self.ref_root_lin = ref.root_lin.to(dev)
    self.ref_root_ang = ref.root_ang.to(dev)
    # the root twist in the reference's own heading frame (placement-free)
    qy = yaw_quat(self.ref_root_quat.contiguous())
    self.ref_root_lin_h = quat_apply_inverse(qy, self.ref_root_lin)
    self.ref_root_ang_h = quat_apply_inverse(qy, self.ref_root_ang)
    self.body_indexes = torch.as_tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], device=dev)
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = list(cfg.body_names).index(cfg.anchor_body_name)
    names = list(self.robot.joint_names)
    self.err_joint_ids = torch.as_tensor(
      [i for i, nm in enumerate(names) if not re.fullmatch(ERR_JOINT_EXCLUDE, nm)],
      device=dev)
    lim = self.robot.data.joint_pos_limits.float()
    self._q_lo = lim[..., 0] + cfg.joint_margin
    self._q_hi = lim[..., 1] - cfg.joint_margin
    n, nb = self.num_envs, len(cfg.body_names)
    long = dict(dtype=torch.long, device=dev)
    self.f = torch.zeros(n, **long)
    self.start_f = torch.zeros(n, **long)
    self.yaw_q = torch.zeros(n, 4, device=dev)
    self.yaw_q[:, 0] = 1.0
    self.offset = torch.zeros(n, 3, device=dev)
    self.body_pos_relative_w = torch.zeros(n, nb, 3, device=dev)
    self.body_quat_relative_w = torch.zeros(n, nb, 4, device=dev)
    self.body_quat_relative_w[..., 0] = 1.0
    self.start_override = None
    self.spawn_noise = bool(cfg.spawn_noise)
    self._min_left = int(round(cfg.rsi_min_left_s / env.step_dt))
    self._action_term = None      # the action manager is built after the commands

  # --- MotionCommand's interface ------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.q[self.f]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.qd[self.f]

  def _rot(self, v: torch.Tensor) -> torch.Tensor:
    """Rotate per-env vectors ``[n, ..., 3]`` by each env's placement yaw."""
    q = self.yaw_q.view(-1, *([1] * (v.dim() - 2)), 4).expand(*v.shape[:-1], 4)
    return quat_apply(q, v)

  def _rot_quat(self, quat: torch.Tensor) -> torch.Tensor:
    q = self.yaw_q.view(-1, *([1] * (quat.dim() - 2)), 4).expand(*quat.shape)
    return quat_mul(q.contiguous(), quat.contiguous())

  def placed_bodies(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose of the tracked bodies at per-env ``frames``: ``([n, B, 3], [n, B, 4])``."""
    return (self._rot(self.ref_body_pos[frames]) + self.offset[:, None, :],
            self._rot_quat(self.ref_body_quat[frames]))

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._rot(self.ref_body_pos[self.f]) + self.offset[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._rot_quat(self.ref_body_quat[self.f])

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._rot(self.ref_body_lin[self.f])

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._rot(self.ref_body_ang[self.f])

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    a = self.motion_anchor_body_index
    return self._rot(self.ref_body_pos[self.f, a]) + self.offset

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._rot_quat(self.ref_body_quat[self.f, self.motion_anchor_body_index])

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._rot(self.ref_body_lin[self.f, self.motion_anchor_body_index])

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._rot(self.ref_body_ang[self.f, self.motion_anchor_body_index])

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def ref_root_quat_w(self, frames: torch.Tensor) -> torch.Tensor:
    """The placed reference root orientation at per-env ``frames`` ``[n]``."""
    return self._rot_quat(self.ref_root_quat[frames])

  def update_relative_body_poses(self) -> None:
    """MotionCommand's: the reference bodies moved under the robot's anchor
    (its xy and yaw, the reference's anchor height)."""
    nb = len(self.cfg.body_names)
    anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
    anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
    robot_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
    robot_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
    delta_pos = robot_pos
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_quat, quat_inv(anchor_quat)))
    self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
    self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)

  # --- CommandTerm hooks ----------------------------------------------------------

  def _update_metrics(self) -> None:
    pass    # the episode statistics live in mdp.TrackEpisodeStats

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    last = self.T - 1
    if env_ids is None:
      self.f = (self.f + 1).clamp_max(last)
    else:
      self.f[env_ids] = (self.f[env_ids] + 1).clamp_max(last)
    self.update_relative_body_poses()

  def _start_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
    ov = self.start_override
    if ov is None:
      return sample_start_frames(len(env_ids), self.T, self.cfg.p_start,
                                 self._min_left, self.device)
    ov = torch.as_tensor(ov, dtype=torch.long, device=self.device)
    s = ov.expand(self.num_envs)[env_ids] if ov.dim() == 0 else ov[env_ids]
    return s.clamp(0, self.T - 2)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg, dev, n = self.cfg, self.device, len(env_ids)
    s = self._start_frames(env_ids)
    yaw, xy = sample_yaw_xy(n, dev, cfg.xy_half)
    quat_h, lin_h, ang_h = to_heading_frame(self.ref_root_quat[s], self.ref_root_lin[s],
                                            self.ref_root_ang[s])
    rows = {"root_z": self.ref_root_pos[s, 2].clone(), "root_quat_h": quat_h,
            "lin_vel_h": lin_h, "ang_vel_h": ang_h,
            "joint_pos": self.q[s].clone(), "joint_vel": self.qd[s].clone()}
    if self.spawn_noise:
      def u(*shape, half):
        return (torch.rand(*shape, device=dev) * 2.0 - 1.0) * half
      rp = u(n, 2, half=cfg.noise_roll_pitch)
      tilt = quat_from_euler_xyz(rp[:, 0], rp[:, 1], torch.zeros(n, device=dev))
      rows["root_quat_h"] = quat_mul(tilt, rows["root_quat_h"])
      rows["lin_vel_h"] = rows["lin_vel_h"] + u(n, 3, half=cfg.noise_lin_vel)
      rows["ang_vel_h"] = rows["ang_vel_h"] + u(n, 3, half=cfg.noise_ang_vel)
      jp = rows["joint_pos"] + u(n, self.q.shape[1], half=cfg.noise_joint)
      rows["joint_pos"] = torch.minimum(torch.maximum(jp, self._q_lo[env_ids]),
                                        self._q_hi[env_ids])
      rows["root_z"] = rows["root_z"] + cfg.spawn_lift_m
    write_states(self._env, env_ids, rows, yaw, xy)
    yaw_q, offset = placement(self.ref_root_pos[s], self.ref_root_quat[s], yaw, xy,
                              self._env.scene.env_origins[env_ids])
    self.yaw_q[env_ids] = yaw_q
    self.offset[env_ids] = offset
    self.f[env_ids] = s
    self.start_f[env_ids] = s
    term = self._action_term
    if term is None:
      term = self._action_term = self._env.action_manager.get_term(cfg.action_name)
    off, sc = term.offset, term.scale
    off = off if isinstance(off, float) else off[env_ids]
    sc = sc if isinstance(sc, float) else sc[env_ids]
    a = (rows["joint_pos"][:, term.target_ids] - off) / sc
    b = float(cfg.seed_action_bound)
    self._env.action_manager._action[env_ids] = a.clamp(-b, b)
