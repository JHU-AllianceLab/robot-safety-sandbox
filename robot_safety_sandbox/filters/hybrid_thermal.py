"""Analytic motor temperatures inside a SHAC world-model rollout: the HYBRID model.

The 70 C-margin SHAC (``g1_recover*_thermal70``) learned ONE world model: an MLP
over the 171-D critic state, the 29 winding and 29 housing temperatures included,
whose head learns the whole margin ``min(tilt, non-foot contact, (70 C - hottest
winding) / 10 C)``. Here the temperatures and the thermal margin come from the
sysID heat model instead (the env's own ``MotorThermal``, the equations the
simulator integrates), and the network keeps the rest:

* TORQUE. Per model step (one control step), the joint torque of each of the
  ``decimation`` physics substeps is rebuilt from the model's joint positions and
  velocities, linearly interpolated between the step's two states (substep k
  reads the state a fraction k / D into the step, as the dcmotor reads the state
  at the substep's start), and the step's joint targets
  ``offset + scale * (ctrl_gain * a) - encoder_bias`` (mjlab's
  JointPositionAction), through the identified drive law: ``kp (q* - q) - kd dq``,
  capped by the back-EMF limit ``kt (+-V - ke dq) / R`` and the peak torque
  (``G1-robot-model-with-SysID`` README, "Plain MuJoCo": MuJoCo's ``<dcmotor>`` in
  position mode with back-EMF compensation);
* HEAT. ``MotorThermal.step`` integrates winding / housing / structure over those
  substeps, with each substep's motor torque and its end-of-substep motor speed,
  as ``thermal.ThermalIntegrator`` does after every ``mj_step``. The state holds no
  structure temperature: it rides alongside, starting at the env's true one;
* STATE. The winding / housing dims of the predicted state are overwritten with
  the result before pi reads it and before the next model step;
* MARGIN. ``mode="hybrid"``: ``min(fall margin, (limit - hottest winding) /
  scale)`` clamped to +-CLAMP, the fall margin from the learned head read with the
  temperatures shifted COLD (every winding / housing of the pair moved down so the
  hottest winding sits at ``cold_c``, floored at the start-draw floor), where its
  learned thermal term saturates above the fall terms; the head's termination
  logit is read the same way (an overheat is the analytic margin's job).
  ``mode="hybrid-temps"``: the analytic temperatures only, the head's own margin
  and termination on them (does the head's thermal term work once the
  temperatures are right?).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

MODES = ("nn", "hybrid", "hybrid-temps")


def _row(x, ids: torch.Tensor, n_cols: int, device) -> torch.Tensor:
  """A per-action scalar / [A] / [num_envs, A] constant -> ``[len(ids), A]``."""
  if not torch.is_tensor(x):
    return torch.full((int(ids.numel()), n_cols), float(x), device=device)
  x = x.to(device=device, dtype=torch.float32)
  if x.dim() == 0:
    return x.expand(int(ids.numel()), n_cols).clone()
  if x.dim() == 1:
    return x.unsqueeze(0).expand(int(ids.numel()), -1).clone()
  return x[ids]


class AnalyticThermal:
  """The analytic half of the hybrid world model, for one eval env (module
  docstring). ``env`` is the sandbox wrapper (``.mj`` the mjlab env), ``wm`` the
  SHAC world model whose critic-group state this reads and writes.
  ``critic_group`` names that group in ``env`` (another name when the env's
  own ``critic`` group belongs to a task policy, e.g. g1_safety_porl's);
  ``floor_c`` is the cold shift's floor (None: the env's start-draw floor;
  pass the calibration's when the env draws its start temperatures elsewhere)."""

  def __init__(self, env, wm, mode: str = "hybrid", cold_c: float = 50.0,
               clamp: Optional[float] = None, critic_group: str = "critic",
               floor_c: Optional[float] = None):
    if mode not in MODES or mode == "nn":
      raise ValueError(f"AnalyticThermal mode must be one of {MODES[1:]}, got {mode!r}")
    from ..envs.g1_recover.thermal import has_thermal, thermal_state
    from .wm_tube import thermal_world_info
    mj = env.mj
    if not has_thermal(mj):
      raise ValueError("the hybrid world model needs a *_thermal task (no motor thermal state)")
    info = thermal_world_info(env)
    if info.get("barrier") is not None:
      raise ValueError("the hybrid world model rebuilds the dcmotor's torque; the barrier "
                       "arm's clip is not modelled")
    mt = info.get("margin_thermal")
    if not mt:
      raise ValueError("the env's margin has no thermal term (not a *_thermal70 task?)")
    if clamp is None:
      from ..margins import CLAMP as clamp
    self.mode, self.cold_c, self.clamp = mode, float(cold_c), float(clamp)
    self.limit_c, self.scale_c = float(mt["limit_c"]), float(mt["scale_c"])
    self.env, self.mj, self.wm = env, mj, wm
    self.state = thermal_state(mj)
    self.th = self.state.th
    self.floor_c = float(self.state.cfg.start_floor_c if floor_c is None else floor_c)
    self.dev = torch.device(env.device)
    self.dt = float(mj.physics_dt)
    self.D = int(mj.cfg.decimation)
    self.ctrl_gain = float(getattr(env, "ctrl_gain", 1.0))
    self.robot = mj.scene["robot"]
    names = list(self.state.names)
    joints = list(self.robot.joint_names)
    # --- the critic-group state: joints (robot order), temperatures (motor order)
    layout = env._obs_layout(critic_group)
    need = ("joint_pos", "joint_vel", "motor_temp", "motor_temp_housing")
    miss = [k for k in need if k not in layout]
    if miss:
      raise ValueError(f"the critic group has no {miss} term(s): {sorted(layout)}")
    for k in need:
      cfg = layout[k][2]
      if getattr(cfg, "noise", None) is not None or (getattr(cfg, "scale", None) not in (None, 1.0)):
        raise ValueError(f"critic term {k} is noisy or scaled; the hybrid reads raw values")
    if bool((getattr(layout["joint_pos"][2], "params", None) or {}).get("biased", False)):
      raise ValueError("critic joint_pos is the biased (encoder) reading; expected the true one")
    for k, n in (("joint_pos", len(joints)), ("joint_vel", len(joints)),
                 ("motor_temp", len(names)), ("motor_temp_housing", len(names))):
      if layout[k][1] != n:
        raise ValueError(f"critic term {k} has {layout[k][1]} dims, expected {n}")
    jperm = self.state.jnt_perm.to(self.dev).long()
    self.jp_idx = layout["joint_pos"][0] + jperm            # motor i -> its joint's column
    self.jv_idx = layout["joint_vel"][0] + jperm
    self.tw_sl = slice(layout["motor_temp"][0], layout["motor_temp"][0] + len(names))
    self.th_sl = slice(layout["motor_temp_housing"][0],
                       layout["motor_temp_housing"][0] + len(names))
    self.jperm = jperm
    # --- actions -> joint targets (mjlab JointPositionAction), motor order
    term = mj.action_manager.get_term("joint_pos")
    if hasattr(term, "safe_targets"):
      raise ValueError("the action term is the temperature barrier: not the thermal70 world")
    if getattr(term.cfg, "clip", None) is not None:
      raise ValueError("the joint_pos action clips its targets; not modelled")
    tids = [int(i) for i in term.target_ids.reshape(-1).tolist()]
    tnames = [joints[i] for i in tids]
    self.m2t = torch.tensor([tnames.index(n + "_joint") for n in names], device=self.dev)
    if [tids[int(c)] for c in self.m2t.tolist()] != jperm.tolist():
      raise ValueError("action columns and the thermal state's joints disagree")
    self.term, self.n_act = term, len(tids)
    # --- the dcmotor of every motor (the actuator configs the sim was built from)
    vals = {k: [math.nan] * len(names) for k in ("kp", "kd", "kt", "ke", "R", "V", "peak")}
    for act in self.robot.actuators:
      c = act.cfg
      mp = getattr(c, "motor_params", None)
      for tn in act.target_names:
        nm = tn[:-len("_joint")] if tn.endswith("_joint") else tn
        if nm not in names:
          continue
        i = names.index(nm)
        if mp is None or not hasattr(mp, "kt"):
          raise ValueError(f"actuator of {tn} is not a dcmotor with physical params "
                           f"({type(c).__name__})")
        if float(getattr(c, "gear", 1.0)) != 1.0 or float(getattr(c, "integral_gain", 0.0)) != 0.0 \
                or float(getattr(c, "slew_rate", 0.0)) != 0.0:
          raise ValueError(f"actuator of {tn}: gear / integral / slew not modelled")
        vals["kp"][i], vals["kd"][i] = float(c.stiffness), float(c.damping)
        vals["kt"][i], vals["ke"][i], vals["R"][i] = float(mp.kt), float(mp.ke), float(mp.resistance)
        vals["V"][i] = float(c.voltage_limit)
        vals["peak"][i] = float(c.effort_limit) if c.effort_limit is not None else math.inf
    bad = [k for k, v in vals.items() if any(math.isnan(x) for x in v)]
    if bad:
      raise ValueError(f"no dcmotor value for {bad} on some motors")
    for k, v in vals.items():
      setattr(self, k, torch.tensor(v, device=self.dev, dtype=torch.float32))

  # --- pieces ---------------------------------------------------------------------
  def meta(self) -> dict:
    return {"mode": self.mode, "limit_c": self.limit_c, "scale_c": self.scale_c,
            "clamp": self.clamp, "cold_c": self.cold_c, "floor_c": self.floor_c,
            "substeps": self.D, "physics_dt": self.dt,
            "torque": "kp (q* - q) - kd dq, capped by kt (+-V - ke dq) / R and the peak; "
                      "q, dq linear between the model's two states",
            "structure": "carried alongside, from the env's true value"}

  def start(self, ids: torch.Tensor, repeat: int = 1):
    """The env's true ``(Tw, Th, Ts)`` and per-env constants for envs ``ids``
    (each repeated ``repeat`` times, as the rollouts are)."""
    st = self.state
    T = tuple(x[ids].float().clone() for x in (st.Tw, st.Th, st.Ts))
    off = _row(self.term.offset, ids, self.n_act, self.dev)[:, self.m2t]
    scl = _row(self.term.scale, ids, self.n_act, self.dev)[:, self.m2t]
    bias = self.robot.data.encoder_bias[ids][:, self.jperm].float()
    q_def = self.robot.data.default_joint_pos[ids][:, self.jperm].float()
    ctx = {"off": off, "scale": scl, "bias": bias, "q_def": q_def}
    if repeat > 1:
      T = tuple(x.repeat_interleave(repeat, dim=0) for x in T)
      ctx = {k: v.repeat_interleave(repeat, dim=0) for k, v in ctx.items()}
    return T, ctx

  def torque(self, tgt, q, qd) -> torch.Tensor:
    """The dcmotor's joint torque, motor order (module docstring)."""
    tau = self.kp * (tgt - q) - self.kd * qd
    hi = torch.minimum(self.kt * (self.V - self.ke * qd) / self.R, self.peak)
    lo = torch.maximum(self.kt * (-self.V - self.ke * qd) / self.R, -self.peak)
    return torch.maximum(torch.minimum(tau, hi), lo)

  def advance(self, T, ctx: dict, s0: torch.Tensor, s1: torch.Tensor, a: torch.Tensor):
    """``(Tw, Th, Ts)`` after one control step from raw critic state ``s0`` to
    ``s1`` under env action ``a`` (``[B, A]``, the action term's column order)."""
    q0 = s0[:, self.jp_idx] + ctx["q_def"]
    q1 = s1[:, self.jp_idx] + ctx["q_def"]
    v0, v1 = s0[:, self.jv_idx], s1[:, self.jv_idx]
    tgt = ctx["off"] + ctx["scale"] * (self.ctrl_gain * a[:, self.m2t].float()) - ctx["bias"]
    D = self.D
    for k in range(D):
      f0, f1 = k / D, (k + 1) / D
      tau = self.torque(tgt, q0 + f0 * (q1 - q0), v0 + f0 * (v1 - v0))
      T = self.th.step(T, self.th.motor_torque(tau), self.th.motor_speed(v0 + f1 * (v1 - v0)),
                       self.dt)
    return T

  def write(self, s: torch.Tensor, T) -> torch.Tensor:
    """``s`` with its winding / housing dims replaced by ``T``'s."""
    s = s.clone()
    s[:, self.tw_sl] = T[0]
    s[:, self.th_sl] = T[1]
    return s

  def thermal_margin(self, Tw: torch.Tensor) -> torch.Tensor:
    g = (self.limit_c - Tw.amax(dim=1)) / self.scale_c
    return g.clamp(-self.clamp, self.clamp)

  def cold_pair(self, s0: torch.Tensor, s1: torch.Tensor):
    """Both states with every winding / housing shifted down by one amount per
    rollout, so the pair's hottest winding sits at ``cold_c`` (none moves up)."""
    hot = torch.maximum(s0[:, self.tw_sl].amax(1), s1[:, self.tw_sl].amax(1))
    d = (hot - self.cold_c).clamp_min(0.0).unsqueeze(1)
    out = []
    for s in (s0, s1):
      s = s.clone()
      tw = (s[:, self.tw_sl] - d).clamp_min(self.floor_c)
      s[:, self.tw_sl] = tw
      s[:, self.th_sl] = torch.minimum((s[:, self.th_sl] - d).clamp_min(self.floor_c), tw)
      out.append(s)
    return out[0], out[1]

  def heads(self, s0: torch.Tensor, s1: torch.Tensor, cn) -> tuple[torch.Tensor, torch.Tensor]:
    """The learned head's ``(g, done_logit)`` on raw states ``s0 -> s1``."""
    wm = self.wm
    h = [wm.encode(s0), wm.encode(s1)] + ([] if cn is None else [cn])
    out = wm.heads(torch.cat(h, -1))
    return out[..., 0], out[..., 4 if wm.predict_l else 2]

  def model_step(self, T, ctx, s0: torch.Tensor, st, a: torch.Tensor, cn):
    """One hybrid step after ``wm.step`` gave ``st`` from raw ``s0`` under ``a``:
    ``(T', s1, g, done_logit)`` with ``s1`` the raw next state (analytic
    temperatures written in)."""
    wm = self.wm
    s1 = wm.decode(st.z_next)
    T = self.advance(T, ctx, s0, s1, a)
    s1 = self.write(s1, T)
    if self.mode == "hybrid":
      c0, c1 = self.cold_pair(s0, s1)
      g_fall, done_logit = self.heads(c0, c1, cn)
      g = torch.minimum(g_fall, self.thermal_margin(T[0]))
    else:
      g, done_logit = self.heads(s0, s1, cn)
    return T, s1, g, done_logit
