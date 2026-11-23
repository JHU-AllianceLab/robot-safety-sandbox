"""The analytic block of a HYBRID SHAC world model: motor temperatures from the
sysID heat model, for TRAINING (and every later use of the trained model).

The user's design (2026-11-12, decided 2026-11-21): the learned network predicts
the body (it never reads the temperatures: in simulation they do not change the
motion), the identified 3-node heat model predicts the winding / housing /
structure temperatures from the torque the drive makes, the learned head learns
the FALL margin and termination, and the 70 C margin and overheat termination
are computed from the predicted temperatures. ``safety_sb3.world_model.WorldModel``
takes this module as its ``analytic`` block (``wm_hybrid=True`` in SHAC); the env
supplies its plain-data config (:func:`analytic_spec_from_env`), which is pickled
with the model, so a checkpoint rebuilds the block without the env.

Per model step (one control step), as :class:`~.hybrid_thermal.AnalyticThermal`
(the filter-time version validated in hybrid step 0: one-step heat 1.02-1.05x the
simulator's) and the env's ``thermal.ThermalIntegrator``:

* TORQUE of each of the ``decimation`` physics substeps from the joint positions
  and velocities linearly interpolated between the step's two states (substep k
  reads the state k / D into the step), the step's joint targets
  ``offset + scale * (ctrl_gain * a) - encoder_bias`` (mjlab's
  JointPositionAction; the bias from the sim context, the per-env DR draw) and
  the identified drive law ``kp (q* - q) - kd dq`` capped by the back-EMF limit
  ``kt (+-V - ke dq) / R`` and the peak torque;
* HEAT: ``MotorThermal.step`` (the pinned sysID version's class) over those
  substeps, motor torque through the ankle / waist linkage split, end-of-substep
  motor speed;
* MARGIN ``(limit - hottest winding) / scale`` clamped to +-``clamp``;
  TERMINATION ``hottest winding >= limit``, graded at ``terminal_margin`` (the
  env's failure anchor), as the env's ``motor_overheat`` term and margin hook.

Everything is torch and differentiable: the actor's gradient reaches the thermal
margin through the torque.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

#: MotorThermal fields that are per-motor arrays (the rest are scalars / metadata)
_ARRAY_FIELDS = ("Cw", "Rwh", "Ch", "Rha", "Cs", "Rsa", "P_idle", "rho_off", "a", "b2", "b1",
                 "stall_s", "stall_w0")


class HybridThermalModel(nn.Module):
  """The analytic block (module docstring). Built from plain data only
  (:func:`analytic_spec_from_env` makes it); ``index`` lists the model-state dims
  it owns: the winding, housing and structure temperatures, in that order."""

  def __init__(self, *, sysid_version: str, n_motors: int, jp_idx, jv_idx, tw_idx, th_idx,
               ts_idx, q_def, act_offset, act_scale, m2t, bias_ctx_idx, bias_const,
               ctrl_gain: float, physics_dt: float, decimation: int, dcmotor: dict,
               thermal: dict, limit_c: float, scale_c: float, clamp: float,
               terminal_margin: float):
    super().__init__()
    from robot_safety_sandbox.envs.assets.g1_sysid import motor_thermal_cls
    M = int(n_motors)
    self.sysid_version = str(sysid_version)
    self._cls = motor_thermal_cls(self.sysid_version)
    for k, v in (("jp_idx", jp_idx), ("jv_idx", jv_idx), ("tw_idx", tw_idx),
                 ("th_idx", th_idx), ("ts_idx", ts_idx), ("m2t", m2t),
                 ("bias_ctx_idx", bias_ctx_idx)):
      t = torch.as_tensor(v, dtype=torch.long)
      if t.numel() != M:
        raise ValueError(f"{k}: {t.numel()} entries for {M} motors")
      self.register_buffer(k, t, persistent=False)
    for k, v in (("q_def", q_def), ("act_offset", act_offset), ("act_scale", act_scale),
                 ("bias_const", bias_const)):
      self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32).reshape(M), persistent=False)
    for k in ("kp", "kd", "kt", "ke", "R", "V", "peak"):
      self.register_buffer(f"dc_{k}", torch.as_tensor(dcmotor[k], dtype=torch.float32).reshape(M),
                           persistent=False)
    self._th_scalars = {}
    for f, v in thermal.items():
      if f in _ARRAY_FIELDS:
        if v is None:
          self._th_scalars[f] = None
        else:
          self.register_buffer(f"th_{f}", torch.as_tensor(v, dtype=torch.float32).reshape(M),
                               persistent=False)
      else:
        self._th_scalars[f] = tuple(v) if isinstance(v, list) else v
    self.ctrl_gain, self.dt, self.D = float(ctrl_gain), float(physics_dt), int(decimation)
    self.limit_c, self.scale_c = float(limit_c), float(scale_c)
    self.clamp, self.terminal_margin = float(clamp), float(terminal_margin)
    #: the model-state dims this block writes (WorldModel reads it)
    self.index = [int(i) for i in list(tw_idx) + list(th_idx) + list(ts_idx)]
    self.n_motors = M

  def heat_model(self):
    """The pinned ``MotorThermal`` on this module's device / dtype."""
    kw = dict(self._th_scalars)
    for f in _ARRAY_FIELDS:
      if f in kw and kw[f] is None:
        continue
      kw[f] = getattr(self, f"th_{f}")
    return self._cls(**kw)

  def torque(self, tgt, q, qd) -> torch.Tensor:
    """The dcmotor's joint torque, motor order."""
    tau = self.dc_kp * (tgt - q) - self.dc_kd * qd
    hi = torch.minimum(self.dc_kt * (self.dc_V - self.dc_ke * qd) / self.dc_R, self.dc_peak)
    lo = torch.maximum(self.dc_kt * (-self.dc_V - self.dc_ke * qd) / self.dc_R, -self.dc_peak)
    return torch.maximum(torch.minimum(tau, hi), lo)

  def encoder_bias(self, c: Optional[torch.Tensor], B: int) -> torch.Tensor:
    const = self.bias_const.expand(B, -1)
    if c is None:
      return const
    got = c.index_select(-1, self.bias_ctx_idx.clamp_min(0))
    return torch.where(self.bias_ctx_idx >= 0, got, const)

  def advance(self, s0: torch.Tensor, s1: torch.Tensor, a: torch.Tensor,
              c: Optional[torch.Tensor]) -> torch.Tensor:
    """Temperatures ``[B, 3 * 29]`` (winding, housing, structure; ``index``'s
    order) after one control step from raw state ``s0`` to raw predicted ``s1``
    under raw env action ``a`` (action-term columns) and raw context ``c``.

    On CUDA the step is ``torch.compile``d (its ~200 small elementwise kernels
    per step made imagination launch-bound: 2x slower SHAC); the eager path is
    the reference and the fallback ($ZOO_HYBRID_COMPILE=0, or a failed compile)."""
    if s0.is_cuda and self._compile_ok():
      try:
        return self._compiled(s0, s1, a, c)
      except Exception as e:   # noqa: BLE001 -- fall back to the reference path
        import warnings
        warnings.warn(f"hybrid heat block: torch.compile failed ({type(e).__name__}: {e}); "
                      "running eager")
        self.__dict__["_compile_failed"] = True
    return self._advance_eager(s0, s1, a, c)

  def _compile_ok(self) -> bool:
    import os
    if self.__dict__.get("_compile_failed") or os.environ.get("ZOO_HYBRID_COMPILE", "1") == "0":
      return False
    if self.__dict__.get("_compiled") is None:
      self.__dict__["_compiled"] = torch.compile(self._advance_eager, dynamic=None)
    return True

  def _advance_eager(self, s0: torch.Tensor, s1: torch.Tensor, a: torch.Tensor,
                     c: Optional[torch.Tensor]) -> torch.Tensor:
    th = self.heat_model()
    T = (s0.index_select(-1, self.tw_idx), s0.index_select(-1, self.th_idx),
         s0.index_select(-1, self.ts_idx))
    q0 = s0.index_select(-1, self.jp_idx) + self.q_def
    q1 = s1.index_select(-1, self.jp_idx) + self.q_def
    v0, v1 = s0.index_select(-1, self.jv_idx), s1.index_select(-1, self.jv_idx)
    bias = self.encoder_bias(c, s0.shape[0])
    tgt = self.act_offset + self.act_scale * (self.ctrl_gain * a.index_select(-1, self.m2t)) - bias
    D = self.D
    for k in range(D):
      f0, f1 = k / D, (k + 1) / D
      tau = self.torque(tgt, q0 + f0 * (q1 - q0), v0 + f0 * (v1 - v0))
      T = th.step(T, th.motor_torque(tau), th.motor_speed(v0 + f1 * (v1 - v0)), self.dt)
    return torch.cat(T, dim=-1)

  def margin(self, s: torch.Tensor) -> torch.Tensor:
    g = (self.limit_c - s.index_select(-1, self.tw_idx).amax(-1)) / self.scale_c
    return g.clamp(-self.clamp, self.clamp)

  def terminated(self, s: torch.Tensor) -> torch.Tensor:
    return (s.index_select(-1, self.tw_idx).amax(-1) >= self.limit_c).float()


def env_margin_fn(env):
  """The margin_fn of a sandbox env (its reward hook's)."""
  rewards = getattr(env.mj.cfg, "rewards", None) or {}
  hook = rewards.get("zoo_safety_hook") or rewards.get("zoo_margin_probe")
  return (getattr(hook, "params", None) or {}).get("margin_fn")


def analytic_spec_from_env(env, critic_group: Optional[str] = None) -> dict:
  """The plain-data spec of :class:`HybridThermalModel` for a live sandbox env
  (``base.MjlabTensorSafetyEnv``) of a ``*_thermal70_v6`` task: the critic group
  must hold noise-free, unscaled ``joint_pos`` (relative, true), ``joint_vel``,
  ``motor_temp``, ``motor_temp_housing`` and ``motor_temp_structure``; the
  margin must carry a thermal term; no barrier action."""
  import dataclasses

  from robot_safety_sandbox.base import TERMINAL_MARGIN
  from robot_safety_sandbox.envs.g1_recover.thermal import has_thermal, thermal_state
  from robot_safety_sandbox.margins import CLAMP
  mj = env.mj
  if not has_thermal(mj):
    raise ValueError("the hybrid world model needs a *_thermal task (no motor thermal state)")
  group = critic_group or env.critic_obs_key
  if group is None:
    raise ValueError("the hybrid world model's state is the critic group; the env has none")
  margin_fn = env_margin_fn(env)
  avoid = getattr(margin_fn, "avoid_margin", None)
  mt = getattr(avoid, "thermal", None)
  if not mt or getattr(avoid, "fall_margin", None) is None:
    raise ValueError("the env's margin has no thermal term / fall margin (not a *_thermal70 task?)")
  st = thermal_state(mj)
  names = list(st.names)
  M = len(names)
  robot = mj.scene["robot"]
  joints = list(robot.joint_names)
  layout = env._obs_layout(group)
  need = ("joint_pos", "joint_vel", "motor_temp", "motor_temp_housing", "motor_temp_structure")
  miss = [k for k in need if k not in layout]
  if miss:
    raise ValueError(f"the critic group has no {miss} term(s): {sorted(layout)}")
  for k in need:
    cfg = layout[k][2]
    if getattr(cfg, "noise", None) is not None or getattr(cfg, "scale", None) not in (None, 1.0):
      raise ValueError(f"critic term {k} is noisy or scaled; the hybrid reads raw values")
  if bool((getattr(layout["joint_pos"][2], "params", None) or {}).get("biased", False)):
    raise ValueError("critic joint_pos is the biased (encoder) reading; expected the true one")
  for k, n in (("joint_pos", len(joints)), ("joint_vel", len(joints)), ("motor_temp", M),
               ("motor_temp_housing", M), ("motor_temp_structure", M)):
    if layout[k][1] != n:
      raise ValueError(f"critic term {k} has {layout[k][1]} dims, expected {n}")
  jperm = [int(i) for i in st.jnt_perm.reshape(-1).tolist()]   # motor i -> robot joint index
  tw0, th0, ts0 = (layout[k][0] for k in ("motor_temp", "motor_temp_housing",
                                          "motor_temp_structure"))
  # actions -> joint targets (mjlab JointPositionAction), motor order
  term = mj.action_manager.get_term("joint_pos")
  if hasattr(term, "safe_targets"):
    raise ValueError("the action term is the temperature barrier: not modelled")
  if getattr(term.cfg, "clip", None) is not None:
    raise ValueError("the joint_pos action clips its targets; not modelled")
  tids = [int(i) for i in term.target_ids.reshape(-1).tolist()]
  tnames = [joints[i] for i in tids]
  m2t = [tnames.index(n + "_joint") for n in names]
  if [tids[c] for c in m2t] != jperm:
    raise ValueError("action columns and the thermal state's joints disagree")

  def per_motor_const(x, what):
    """An action-term / robot constant, motor order, the same in every env."""
    t = x if torch.is_tensor(x) else torch.tensor(float(x))
    t = t.detach().float()
    if t.dim() == 0:
      return [float(t)] * M
    if t.dim() == 2:
      if not torch.allclose(t, t[:1].expand_as(t)):
        raise ValueError(f"{what} differs across envs; the hybrid block holds one value")
      t = t[0]
    return t.cpu().tolist()

  off = per_motor_const(term.offset, "action offset")
  scl = per_motor_const(term.scale, "action scale")
  off = [off[c] for c in m2t] if len(off) == len(tids) else off
  scl = [scl[c] for c in m2t] if len(scl) == len(tids) else scl
  qd = per_motor_const(robot.data.default_joint_pos, "default joint pos")
  q_def = [qd[j] for j in jperm]
  # encoder bias: the sim-context column of each motor's joint (the DR draw)
  bias_now = robot.data.encoder_bias.detach().float()                 # [N, J]
  ctx = env.sim_context()
  bias_ctx_idx = [-1] * M
  if ctx is not None:
    for i, j in enumerate(jperm):
      col = bias_now[:, j]
      hit = [k for k in range(ctx.shape[1]) if torch.equal(ctx[:, k].float(), col)]
      if hit:
        bias_ctx_idx[i] = hit[0]
  bias_const = []
  for i, j in enumerate(jperm):
    col = bias_now[:, j]
    if bias_ctx_idx[i] < 0 and not torch.allclose(col, col[:1].expand_as(col)):
      raise ValueError(f"encoder bias of {joints[j]} varies across envs but is not in the "
                       "sim context")
    bias_const.append(float(col[0]))
  # the dcmotor of every motor (the actuator configs the sim was built from)
  vals = {k: [math.nan] * M for k in ("kp", "kd", "kt", "ke", "R", "V", "peak")}
  for act in robot.actuators:
    c = act.cfg
    mp = getattr(c, "motor_params", None)
    for tn in act.target_names:
      nm = tn[:-len("_joint")] if tn.endswith("_joint") else tn
      if nm not in names:
        continue
      i = names.index(nm)
      if mp is None or not hasattr(mp, "kt"):
        raise ValueError(f"actuator of {tn} is not a dcmotor with physical params")
      if float(getattr(c, "gear", 1.0)) != 1.0 or float(getattr(c, "integral_gain", 0.0)) != 0.0 \
              or float(getattr(c, "slew_rate", 0.0)) != 0.0:
        raise ValueError(f"actuator of {tn}: gear / integral / slew not modelled")
      vals["kp"][i], vals["kd"][i] = float(c.stiffness), float(c.damping)
      vals["kt"][i], vals["ke"][i], vals["R"][i] = float(mp.kt), float(mp.ke), float(mp.resistance)
      vals["V"][i] = float(c.voltage_limit)
      vals["peak"][i] = float(c.effort_limit) if c.effort_limit is not None else 1e9
  bad = [k for k, v in vals.items() if any(math.isnan(x) for x in v)]
  if bad:
    raise ValueError(f"no dcmotor value for {bad} on some motors")
  thermal = {}
  for f in dataclasses.fields(st.th):
    v = getattr(st.th, f.name)
    if torch.is_tensor(v):
      thermal[f.name] = v.detach().float().reshape(-1).cpu().tolist()
    elif v is None or isinstance(v, (int, float, str)):
      thermal[f.name] = v
    elif isinstance(v, dict):
      thermal[f.name] = {str(k): (int(x) if isinstance(x, (int, bool)) else float(x))
                         for k, x in v.items()}
    else:
      thermal[f.name] = list(v)
  return dict(
    factory="robot_safety_sandbox.filters.hybrid_thermal_model:HybridThermalModel",
    config=dict(
      sysid_version=st.cfg.sysid_version, n_motors=M,
      jp_idx=[layout["joint_pos"][0] + j for j in jperm],
      jv_idx=[layout["joint_vel"][0] + j for j in jperm],
      tw_idx=list(range(tw0, tw0 + M)), th_idx=list(range(th0, th0 + M)),
      ts_idx=list(range(ts0, ts0 + M)),
      q_def=q_def, act_offset=off, act_scale=scl, m2t=m2t,
      bias_ctx_idx=bias_ctx_idx, bias_const=bias_const,
      ctrl_gain=float(getattr(env, "ctrl_gain", 1.0)), physics_dt=float(mj.physics_dt),
      decimation=int(mj.cfg.decimation), dcmotor=vals, thermal=thermal,
      limit_c=float(mt["limit_c"]), scale_c=float(mt["scale_c"]), clamp=float(CLAMP),
      terminal_margin=float(getattr(margin_fn, "terminal_margin", TERMINAL_MARGIN))))
