"""Observation, reward and termination terms of the squat-tracking task, and its
per-episode statistics bus (:class:`TrackEpisodeStats`).

Every term reads the reference through the ``"ref"`` command
(:class:`~robot_safety_sandbox.envs.g1_track.command.TrackRefCommand`); mjlab's
own tracking terms (``mjlab.tasks.tracking.mdp``) run on that command unchanged.
"""

from __future__ import annotations

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import (
  matrix_from_quat, quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul,
  yaw_quat)

from robot_safety_sandbox.envs.g1_recover.thermal import LIMIT70_C, thermal_state

#: phase bins of the per-step bus (``frame * N_PHASE_BINS // T``)
N_PHASE_BINS = 10
#: a step is "tracked" when the mean body position error is below this, m (C6's tau)
TF_TAU_M = 0.1
#: BeyondMimic's termination thresholds, kept as a "lost" flag only: anchor height
#: error, m, and the anchor's projected-gravity z error
LOST_ANCHOR_Z_M = 0.25
LOST_GRAVITY_Z = 0.8
#: episode end / failure codes of the bus (``CAUSES[code]``)
COMPLETE, OVERHEAT, CONTACT, TILT, OTHER, CUT = range(6)
CAUSES = ("complete", "overheat", "contact", "tilt", "other", "cut")
#: the bodies the AMP root height is measured from (AMENDMENTS B12)
AMP_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _ref(env, command_name: str):
  return env.command_manager.get_term(command_name)


# --- observations ------------------------------------------------------------------

def ref_phase(env, command_name: str = "ref") -> torch.Tensor:
  """``[f / (T-1), 1 - f / (T-1)]``: how far into the segment the target is."""
  cmd = _ref(env, command_name)
  ph = cmd.f.float() / float(cmd.T - 1)
  return torch.stack([ph, 1.0 - ph], dim=1)


def ref_future(env, command_name: str = "ref",
               offsets: tuple = (0, 2, 5, 10, 25)) -> torch.Tensor:
  """For each k in ``offsets`` the reference at frame f + k (clamped to the
  last): joint targets minus the default pose (29), the 6-D rotation (first two
  columns) of the robot root to the placed reference root (6), the reference
  root height (1) and its root twist in its own heading frame (6) -- 42 per k."""
  cmd = _ref(env, command_name)
  d = cmd.robot.data
  q_inv = quat_inv(d.root_link_quat_w.contiguous())
  out = []
  for k in offsets:
    fk = (cmd.f + int(k)).clamp_max(cmd.T - 1)
    rot = matrix_from_quat(quat_mul(q_inv, cmd.ref_root_quat_w(fk)))
    out += [cmd.q[fk] - d.default_joint_pos, rot[..., :2].reshape(-1, 6),
            cmd.ref_root_pos[fk, 2:3], cmd.ref_root_lin_h[fk], cmd.ref_root_ang_h[fk]]
  return torch.cat(out, dim=1)


def anchor_pos_err_h(env, command_name: str = "ref") -> torch.Tensor:
  """Reference anchor minus robot anchor, in the robot anchor's heading frame (3)."""
  cmd = _ref(env, command_name)
  qy = yaw_quat(cmd.robot_anchor_quat_w.contiguous())
  return quat_apply_inverse(qy, cmd.anchor_pos_w - cmd.robot_anchor_pos_w)


def rel_body_pos_err_h(env, command_name: str = "ref") -> torch.Tensor:
  """The relative-body-position reward's error (reference bodies under the
  robot's anchor minus the robot's), in the robot anchor's heading frame (3 B)."""
  cmd = _ref(env, command_name)
  err = cmd.body_pos_relative_w - cmd.robot_body_pos_w
  nb = err.shape[1]
  qy = yaw_quat(cmd.robot_anchor_quat_w.contiguous())[:, None, :].expand(-1, nb, -1)
  return quat_apply_inverse(qy, err).reshape(err.shape[0], -1)


class AmpObs(ManagerTermBase):
  """The ``amp`` group's term: ``amp.features.amp_features`` of the robot's
  state, with the root height measured from the lower ankle-roll link
  (pelvis z - min ankle_roll_link z; the expert builder measures it the same
  way on the same robot, so a grounding offset is no tell)."""

  def __init__(self, cfg, env):
    super().__init__(env)
    from robot_safety_sandbox.amp.features import amp_features
    self._features = amp_features
    self.robot = env.scene[cfg.params.get("asset_name", "robot")]
    self.feet = torch.as_tensor(
      self.robot.find_bodies(AMP_FOOT_BODIES, preserve_order=True)[0], device=env.device)

  def __call__(self, env, asset_name: str = "robot") -> torch.Tensor:
    d = self.robot.data
    root_h = d.root_link_pos_w[:, 2] - d.body_link_pos_w[:, self.feet, 2].amin(1)
    return self._features(root_h, d.root_link_quat_w, d.root_link_lin_vel_b,
                          d.root_link_ang_vel_b, d.joint_pos, d.joint_vel)


# --- reward and termination -----------------------------------------------------------

def ref_joint_pos_exp(env, command_name: str = "ref", std: float = 0.5) -> torch.Tensor:
  """``exp(-mean_j (q_j - q_ref_j)^2 / std^2)`` over the command's error joints
  (all but the wrist pitch / yaw)."""
  cmd = _ref(env, command_name)
  j = cmd.err_joint_ids
  err = (cmd.robot.data.joint_pos[:, j] - cmd.joint_pos[:, j]).square().mean(1)
  return torch.exp(-err / std ** 2)


def reference_finished(env, command_name: str = "ref") -> torch.Tensor:
  """Time-out termination: the state just reached was graded against the last frame."""
  cmd = _ref(env, command_name)
  return cmd.f >= cmd.T - 1


# --- the per-episode statistics bus ---------------------------------------------------

class TrackEpisodeStats(ManagerTermBase):
  """Event term (``mode="step"``): per-step tracking / thermal / failure
  statistics and per-episode records, published every step as
  ``env._zoo_track`` -- step events run after terminations and rewards and
  BEFORE the auto-reset, so the done rows still hold their terminal state. The
  bus is rebuilt (cloned) every step; a consumer copies what it needs in the
  same step. ``None`` until the first step.

  ``_zoo_track["step"]`` (all envs, this step):
    frame (long: the frame the state was graded against), start_f, start_rsi
    (start_f > 0), t (steps into the episode, this one included), phase_bin
    (frame * N_PHASE_BINS // T), err_ori (anchor orientation, rad), err_body
    (mean relative body position, m), err_q (joint RMSE over the error joints,
    rad), err_anchor_z (m), lost (BeyondMimic's thresholds), tw_max (hottest
    winding, C), tw_over (tw_max - overheat_c, >= 0), tau2 (sum of squared
    actuator forces), overheat (the episode's FIRST step with tw_max >=
    overheat_c), cbf_clip (the barrier's clipped share; None without a barrier).
  ``_zoo_track["done"]``: bool, this step ended the episode.
  ``_zoo_track["ep"]`` (meaningful on done rows only):
    cause (int8: how it ended -- COMPLETE = reference_finished, OVERHEAT /
    CONTACT / TILT = that termination, overheat > contact > tilt on a tie,
    OTHER = another termination, CUT = another time-out), fail (int8: the FIRST
    failure EVENT, one taxonomy for both worlds -- an overheat event is
    tw_max >= overheat_c whether or not it terminates, a fall is a tilt or
    contact termination; 0 = none), fail_step, fail_bin (-1 = none), any_fail,
    overheat, overheat_step, steps, start_f, end_f, start_rsi, completion
    ((end_f - start_f) / (T-1 - start_f)), err_{ori,body,q,anchor_z}_mean,
    tf_steps (steps with err_body < TF_TAU_M), lost_any, tw_start_max, tw_peak,
    dtw_peak (peak - start of the motor with the highest peak), heat_dose
    (integral of (tw_max - tw_start_max)+ dt, C s), tau2_sum, tw_over_max,
    cbf_clip_mean (NaN without a barrier), tw_start_motor / tw_peak_motor [N, 29].
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    p = cfg.params
    self.command_name = str(p.get("command_name", "ref"))
    self.overheat_c = float(p.get("overheat_c", LIMIT70_C))
    self.robot = env.scene["robot"]
    self._cmd = None              # the command / action managers are built later
    n, dev = self.num_envs, self.device

    def zf(*s):
      return torch.zeros(n, *s, device=dev)

    def zl(fill=0):
      return torch.full((n,), fill, dtype=torch.long, device=dev)

    def zb():
      return torch.zeros(n, dtype=torch.bool, device=dev)

    self.n = zl()
    self.sum_ori, self.sum_body, self.sum_q, self.sum_az = zf(), zf(), zf(), zf()
    self.tf, self.lost_any = zl(), zb()
    self.tw_start, self.tw_peak = zf(29), zf(29)
    self.tw_start_max, self.dose, self.tau2 = zf(), zf(), zf()
    self.over_max, self.clip_sum = zf(), zf()
    self.overheat, self.overheat_step = zb(), zl(-1)
    self.fail, self.fail_step, self.fail_bin = zl(), zl(-1), zl(-1)
    env._zoo_track = None

  def _bind(self, env) -> None:
    self._cmd = env.command_manager.get_term(self.command_name)
    self._thermal = thermal_state(env)
    term = env.action_manager.get_term("joint_pos")
    self._barrier = term if hasattr(term, "clip_frac") else None
    self._tm_names = set(env.termination_manager.active_terms)

  def _term(self, tm, name: str) -> torch.Tensor:
    if name in self._tm_names:
      return tm.get_term(name)
    return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids=None) -> None:
    """Runs after the metrics reset (the start temperatures are drawn) and the
    spawn: start a fresh record."""
    ids = slice(None) if env_ids is None else env_ids
    for acc in (self.n, self.sum_ori, self.sum_body, self.sum_q, self.sum_az, self.tf,
                self.dose, self.tau2, self.over_max, self.clip_sum, self.fail):
      acc[ids] = 0
    self.lost_any[ids] = False
    self.overheat[ids] = False
    self.overheat_step[ids] = -1
    self.fail_step[ids] = -1
    self.fail_bin[ids] = -1
    tw = thermal_state(self._env).Tw[ids]
    self.tw_start[ids] = tw
    self.tw_peak[ids] = tw
    self.tw_start_max[ids] = tw.amax(1)

  def __call__(self, env, env_ids=None, command_name: str = "ref",
               overheat_c: float = LIMIT70_C) -> None:
    if self._cmd is None:
      self._bind(env)
    cmd, tm, d = self._cmd, env.termination_manager, self.robot.data
    # tracking errors of the state just reached, against frame f
    err_ori = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)
    err_body = (cmd.body_pos_relative_w - cmd.robot_body_pos_w).norm(dim=-1).mean(-1)
    j = cmd.err_joint_ids
    err_q = (d.joint_pos[:, j] - cmd.joint_pos[:, j]).square().mean(1).sqrt()
    err_az = (cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2]).abs()
    g_ref = quat_apply_inverse(cmd.anchor_quat_w, d.gravity_vec_w)[:, 2]
    g_rob = quat_apply_inverse(cmd.robot_anchor_quat_w.contiguous(), d.gravity_vec_w)[:, 2]
    lost = (err_az > LOST_ANCHOR_Z_M) | ((g_ref - g_rob).abs() > LOST_GRAVITY_Z)
    # thermal and load
    tw = self._thermal.Tw
    tw_max = tw.amax(1)
    tw_over = (tw_max - self.overheat_c).clamp_min(0.0)
    tau2 = d.actuator_force.square().sum(1)
    clip = None if self._barrier is None else self._barrier.clip_frac
    self.n += 1
    self.sum_ori += err_ori
    self.sum_body += err_body
    self.sum_q += err_q
    self.sum_az += err_az
    self.tf += (err_body < TF_TAU_M).long()
    self.lost_any |= lost
    torch.maximum(self.tw_peak, tw, out=self.tw_peak)
    self.dose += (tw_max - self.tw_start_max).clamp_min(0.0) * float(env.step_dt)
    self.tau2 += tau2
    torch.maximum(self.over_max, tw_over, out=self.over_max)
    if clip is not None:
      self.clip_sum += clip
    hot = tw_max >= self.overheat_c
    first_hot = hot & ~self.overheat
    self.overheat |= hot
    self.overheat_step = torch.where(first_hot, self.n, self.overheat_step)
    # failures: the first EVENT (one taxonomy in both worlds) and the episode end
    term = tm.terminated
    contact, tilt = self._term(tm, "illegal_contact"), self._term(tm, "fell_over")
    oh_term = self._term(tm, "motor_overheat")
    other = term & ~(contact | tilt | oh_term)
    frame = cmd.f
    phase_bin = (frame * N_PHASE_BINS // cmd.T).clamp_max(N_PHASE_BINS - 1)

    def code(c_oh, fallback):
      return torch.where(c_oh, OVERHEAT, torch.where(contact, CONTACT, torch.where(
        tilt, TILT, torch.where(other, OTHER, fallback))))

    zero = torch.zeros_like(self.fail)
    event = code(hot | oh_term, zero)
    new_fail = (self.fail == 0) & (event != 0)
    self.fail = torch.where(new_fail, event, self.fail)
    self.fail_step = torch.where(new_fail, self.n, self.fail_step)
    self.fail_bin = torch.where(new_fail, phase_bin, self.fail_bin)
    ended = torch.where(self._term(tm, "reference_finished") & ~term,
                        torch.full_like(zero, COMPLETE), torch.full_like(zero, CUT))
    cause = code(oh_term, ended)
    steps = self.n.clamp_min(1).float()
    start_f = cmd.start_f
    hot_m = self.tw_peak.argmax(1, keepdim=True)
    dtw = (self.tw_peak.gather(1, hot_m) - self.tw_start.gather(1, hot_m)).squeeze(1)
    nan = torch.full_like(tw_max, float("nan"))
    env._zoo_track = {
      "step": {
        "frame": frame.clone(), "start_f": start_f.clone(), "start_rsi": start_f > 0,
        "t": self.n.clone(), "phase_bin": phase_bin, "err_ori": err_ori,
        "err_body": err_body, "err_q": err_q, "err_anchor_z": err_az, "lost": lost,
        "tw_max": tw_max, "tw_over": tw_over, "tau2": tau2, "overheat": first_hot,
        "cbf_clip": None if clip is None else clip.clone(),
      },
      "done": tm.dones.clone(),
      "ep": {
        "cause": cause.to(torch.int8), "fail": self.fail.to(torch.int8),
        "fail_step": self.fail_step.clone(), "fail_bin": self.fail_bin.clone(),
        "any_fail": self.fail != 0, "overheat": self.overheat.clone(),
        "overheat_step": self.overheat_step.clone(), "steps": self.n.clone(),
        "start_f": start_f.clone(), "end_f": frame.clone(), "start_rsi": start_f > 0,
        "completion": (frame - start_f).float() / (cmd.T - 1 - start_f).clamp_min(1).float(),
        "err_ori_mean": self.sum_ori / steps, "err_body_mean": self.sum_body / steps,
        "err_q_mean": self.sum_q / steps, "err_anchor_z_mean": self.sum_az / steps,
        "tf_steps": self.tf.clone(), "lost_any": self.lost_any.clone(),
        "tw_start_max": self.tw_start_max.clone(), "tw_peak": self.tw_peak.amax(1),
        "dtw_peak": dtw, "heat_dose": self.dose.clone(), "tau2_sum": self.tau2.clone(),
        "tw_over_max": self.over_max.clone(),
        "cbf_clip_mean": nan if clip is None else self.clip_sum / steps,
        "tw_start_motor": self.tw_start.clone(), "tw_peak_motor": self.tw_peak.clone(),
      },
    }
