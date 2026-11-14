"""Motor temperatures for the G1 recovery tasks: the sysID heat model in the loop.

The ``g1_recover*_thermal`` tasks add one constraint to the no-pelvis-height
fallback: the hottest motor winding must stay below ``LIMIT_C`` = 85 C, the
temperature at which the G1's own controller switches the drives to damping
(the robot would collapse). Temperatures come from the lab's identified heat
model (``envs/assets/g1_sysid.py`` -> ``g1_sysid.MotorThermal`` v0.4): per
motor a winding (the robot's ``temperature[1]`` sensor, the one the limit is
on), a housing (``temperature[0]``) and the structure the motor is bolted to,
driven by the motor's torque and speed. ``sim_heat_scale=True``: the simulated
robot makes less torque than the real one for the same motion (lighter links,
no joint friction yet), and the sysID repo's per-motor scale corrects the
copper heat; it is the repo's rule for simulated torques.

THE STATE. :class:`MotorThermalState`, one per env (``thermal_state(env)``),
``(num_envs, 29)`` tensors ``Tw``, ``Th``, ``Ts`` in Unitree MOTOR order
(``MotorThermalState.names``). The terms below all read the same object.

INTEGRATION, once per PHYSICS substep, by :class:`ThermalIntegrator`, a metrics
term with ``per_substep=True``: mjlab calls it right after every ``mj_step``,
inside the decimation loop, so the heat of each 5 ms substep uses that
substep's own actuator force (the force the step just applied) and joint
speed, and the state is current before terminations and the margin hook read
it. (The sysID repo's own mjlab term integrates once per 20 ms env step from
the last substep's force.) Its episode metric is the peak winding temperature.
The motor torque is split across the ankle / waist linkages and the speed
taken per motor as the heat model requires (``motor_torque`` /
``motor_speed``).

START TEMPERATURES, drawn at every reset (the user's choice, 2026-11-09):

* a per-robot level ``T0 ~ U(30, 80)`` C; each motor's winding
  ``Tw ~ U(30, T0)``. The hottest motor then sits near T0, spread evenly from
  a cold to a hot robot, the others below it; which motor is hottest changes
  from episode to episode. (Real sessions: hottest motor 51-74 C while the
  median motor is at 37-44 C.)
* each motor's housing ``Th = max(Tw - U(0, 20), 30)``: a winding that was
  recently loaded sits above its housing (recordings: 0-2 C typically, up to
  ~27 C when hot), which is the path its heat leaves by;
* the structure ``Ts = Th`` (not measured; it does not move in a 3 s episode).

The draws use torch's global RNG on the env's device, so an evaluation that
reseeds torch before a reset (``eval_sonic_qcbf.py`` does, per wave, with a
condition-independent key) gives every condition the same start temperatures.
:meth:`MotorThermalState.set` overwrites them (e.g. from a recorded state).

OBSERVATIONS. ``motor_winding_temp`` (Tw, 29) is an actor term with
``+-0.5`` C uniform noise (the robot reports whole degrees) and, noise-free,
a critic term; ``motor_housing_temp`` (Th, 29) is a critic term only. The
critic group is a SHAC run's world-model state, so the model predicts both and
the margin head can read the thermal term. ``Ts`` stays out: unmeasured, and
it does not change within an episode.

MARGIN AND TERMINATION (``*_thermal`` tasks). ``g_thermal = (LIMIT_C - max
Tw) / SCALE_C``: 1 at 75 C, 0 at the limit, so it competes with the fall terms
(tilt, non-foot contact; both 1 when comfortably safe) only in the last ~10 C.
``motor_overheat`` ends the episode, as a failure, once ``max Tw >= LIMIT_C``.
The ``*_thermal70`` tasks (2026-11-11) are the same design at the barrier's
limit, ``LIMIT70_C`` = 70 C (margin and termination; start levels U(30, 70)),
so the two ways of holding 70 C -- learned through the margin, or a barrier
below the policy -- compare like for like.

THE TEMPERATURE BARRIER (``*_thermal_cbf`` tasks, :class:`ThermalBarrierAction`;
the user's design, 2026-11-10). Thermal safety becomes a layer BELOW whatever
drives the robot (SHAC, SONIC, a filter), decoupled from the task's margin: a
control barrier function per motor on ``h = T_cbf - Tw`` (``T_cbf`` = 70 C)
clips the motor torque. The winding dynamics are affine in ``u = tau_m^2``
(``MotorThermal.affine_terms``: ``dTw/dt = f(T, w) + g(Tw, w) u``, ``g > 0``),
so ``dh/dt >= -gamma h`` is ``u <= u_bar = (gamma (T_cbf - Tw) - f) / g`` and
``|tau_m| <= sqrt(max(u_bar, 0))``. Every physics substep, the term takes the
nominal joint targets, computes the PD torque the dcmotor will apply,
``kp (q* - q) - kd dq``, splits it into motor torques (ankle / waist
linkages), clips each motor to its bound, maps back to joint torques and
rewrites only the targets that changed, ``q* = q + (tau + kd dq) / kp``. With
``u_bar < 0`` (the winding heats or fails to cool even at zero torque, e.g.
speed heating) the motor gets zero torque and the barrier can still be
crossed: the term counts those cases. The margin of these tasks is the fall
margin alone (``g_fall_noheight``) and there is no overheat termination: the
policy meets the temperatures only through the torque it loses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.manager_base import ManagerTermBase

from robot_safety_sandbox.envs.assets.g1_sysid import linkages, motor_thermal_cls

#: winding temperature limit, C: the G1 controller's damping threshold
LIMIT_C = 85.0
#: the temperature barrier's limit, C, and its class-K rate, 1/s (*_thermal_cbf tasks)
CBF_LIMIT_C = 70.0
CBF_GAMMA = 1.0
#: margin scale, C: g_thermal = 1 at LIMIT_C - SCALE_C
SCALE_C = 10.0
#: the margin / termination limit of the *_thermal70 tasks, C: the barrier's limit
LIMIT70_C = CBF_LIMIT_C
#: attribute of the mjlab env that holds its MotorThermalState
_KEY = "_zoo_motor_thermal"


@dataclass(frozen=True)
class ThermalCfg:
  """Start-temperature draw and heat-model options (module docstring)."""

  #: per-robot level T0 ~ U(lo, hi), C; the hottest winding sits near it
  start_level_c: tuple[float, float] = (30.0, 80.0)
  #: each winding ~ U(start_floor_c, T0); the housing is floored here too
  start_floor_c: float = 30.0
  #: winding - housing ~ U(lo, hi) per motor, C
  housing_gap_c: tuple[float, float] = (0.0, 20.0)
  #: scale the copper heat of simulated torques to the real robot's (sysID repo)
  sim_heat_scale: bool = True


class MotorThermalState:
  """Per-env winding / housing / structure temperatures ``(num_envs, 29)`` in
  Unitree motor order, and the heat model that moves them."""

  def __init__(self, env, cfg: ThermalCfg, asset_name: str = "robot"):
    self.cfg = cfg
    self.robot = env.scene[asset_name]
    self.device = env.device
    like = torch.zeros(1, device=env.device)
    self.th = motor_thermal_cls().from_json(like=like, sim_heat_scale=cfg.sim_heat_scale)
    #: motor names, Unitree order (joint ``<name>_joint``)
    self.names = tuple(self.th.names)
    acts, joints = list(self.robot.actuator_names), list(self.robot.joint_names)
    missing = [n for n in self.names if n + "_joint" not in acts or n + "_joint" not in joints]
    if missing:
      raise ValueError(f"the robot has no actuator / joint for motors {missing}")
    self.act_perm = torch.tensor([acts.index(n + "_joint") for n in self.names],
                                 device=env.device)
    self.jnt_perm = torch.tensor([joints.index(n + "_joint") for n in self.names],
                                 device=env.device)
    n = int(env.num_envs)
    self.Tw = torch.zeros(n, len(self.names), device=env.device)
    self.Th = torch.zeros_like(self.Tw)
    self.Ts = torch.zeros_like(self.Tw)
    self.reset(None)

  @property
  def T(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(Tw, Th, Ts)``, e.g. for ``th.affine_terms(T, w)``."""
    return self.Tw, self.Th, self.Ts

  def draw(self, n: int, generator: torch.Generator | None = None):
    """``n`` start states ``(Tw, Th, Ts)``, each ``(n, 29)`` (module docstring)."""
    c = self.cfg
    k = len(self.names)

    def u(*shape):
      return torch.rand(*shape, device=self.device, generator=generator)

    lo, hi = c.start_level_c
    level = lo + (hi - lo) * u(n, 1)
    floor = float(c.start_floor_c)
    Tw = floor + (level - floor).clamp_min(0.0) * u(n, k)
    g_lo, g_hi = c.housing_gap_c
    Th = (Tw - (g_lo + (g_hi - g_lo) * u(n, k))).clamp_min(floor)
    Th = torch.minimum(Th, Tw)
    return Tw, Th, Th.clone()

  def reset(self, env_ids=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    n = int(self.Tw[ids].shape[0])
    if n == 0:
      return
    Tw, Th, Ts = self.draw(n)
    self.set(ids, Tw, Th, Ts)

  def set(self, env_ids, Tw, Th, Ts) -> None:
    """Overwrite the temperatures of ``env_ids`` (slice / index tensor)."""
    ids = slice(None) if env_ids is None else env_ids
    self.Tw[ids] = Tw
    self.Th[ids] = Th
    self.Ts[ids] = Ts

  def motor_inputs(self):
    """Per-motor torque and speed now, ``(num_envs, 29)`` each (linkage split)."""
    d = self.robot.data
    tau = self.th.motor_torque(d.actuator_force[:, self.act_perm])
    w = self.th.motor_speed(d.joint_vel[:, self.jnt_perm])
    return tau, w

  def step(self, dt: float) -> None:
    """Advance every env by ``dt`` seconds of the current torque and speed."""
    tau, w = self.motor_inputs()
    Tw, Th, Ts = self.th.step(self.T, tau, w, float(dt))
    self.Tw.copy_(Tw)
    self.Th.copy_(Th)
    self.Ts.copy_(Ts)


def thermal_state(env, cfg: ThermalCfg | None = None) -> MotorThermalState:
  """The env's :class:`MotorThermalState`; the first caller (with a ``cfg``)
  creates it. The observation manager is built first, so its terms do."""
  st = getattr(env, _KEY, None)
  if st is None:
    if cfg is None:
      raise RuntimeError("this env has no motor thermal state (not a *_thermal task?)")
    st = MotorThermalState(env, cfg)
    setattr(env, _KEY, st)
  elif cfg is not None and cfg != st.cfg:
    raise ValueError(f"thermal terms disagree on the ThermalCfg: {cfg} vs {st.cfg}")
  return st


def has_thermal(env) -> bool:
  return getattr(env, _KEY, None) is not None


class ThermalIntegrator(ManagerTermBase):
  """Metrics term (``per_substep=True``): integrates the heat model over one
  physics substep and returns the hottest winding (the episode metric, with
  ``reduce="max"``). Its reset draws the episode's start temperatures."""

  def __init__(self, cfg, env):
    super().__init__(env)
    self.state = thermal_state(env, cfg.params.get("thermal_cfg"))
    self.dt = float(env.physics_dt)

  def reset(self, env_ids=None) -> None:
    self.state.reset(env_ids)

  def __call__(self, env, thermal_cfg: ThermalCfg | None = None) -> torch.Tensor:
    del thermal_cfg
    self.state.step(self.dt)
    return self.state.Tw.amax(dim=1)


def motor_winding_temp(env, thermal_cfg: ThermalCfg | None = None) -> torch.Tensor:
  """Winding temperatures, C, ``(num_envs, 29)``, motor order."""
  return thermal_state(env, thermal_cfg).Tw.clone()


def motor_housing_temp(env, thermal_cfg: ThermalCfg | None = None) -> torch.Tensor:
  """Housing temperatures, C, ``(num_envs, 29)``, motor order."""
  return thermal_state(env, thermal_cfg).Th.clone()


def motor_overheat(env, limit_c: float = LIMIT_C,
                   thermal_cfg: ThermalCfg | None = None) -> torch.Tensor:
  """Termination: some winding reached ``limit_c`` (the robot goes limp)."""
  return thermal_state(env, thermal_cfg).Tw.amax(dim=1) >= float(limit_c)


def g_thermal(env, limit_c: float = LIMIT_C) -> torch.Tensor:
  """Thermal margin ``(limit_c - max winding) / SCALE_C``; < 0 == over the limit."""
  return (float(limit_c) - thermal_state(env).Tw.amax(dim=1)) / SCALE_C


def g_fall_noheight_thermal(env) -> torch.Tensor:
  """``g_fall_noheight`` (pelvis tilt, non-foot contact) and ``g_thermal``:
  the margin of the ``*_thermal`` tasks. Unclamped; ``compose`` clamps."""
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall_noheight
  return torch.minimum(g_fall_noheight(env), g_thermal(env))


g_fall_noheight_thermal.fall_height_term = False
#: what a CPU rollout must add to the fall margin (cpu_rollout.build_spec)
g_fall_noheight_thermal.thermal = dict(limit_c=LIMIT_C, scale_c=SCALE_C)


def g_fall_noheight_thermal70(env) -> torch.Tensor:
  """:func:`g_fall_noheight_thermal` with the limit at ``LIMIT70_C`` = 70 C:
  the margin of the ``*_thermal70`` tasks."""
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall_noheight
  return torch.minimum(g_fall_noheight(env), g_thermal(env, LIMIT70_C))


g_fall_noheight_thermal70.fall_height_term = False
g_fall_noheight_thermal70.thermal = dict(limit_c=LIMIT70_C, scale_c=SCALE_C)


# --- the temperature barrier (*_thermal_cbf tasks) ---------------------------------

def barrier_bound(state: MotorThermalState, w: torch.Tensor, limit_c: float,
                  gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
  """``(tau_bar, u_bar)``, ``(num_envs, 29)``: each motor's torque bound from the
  barrier ``h = limit_c - Tw`` at motor speeds ``w`` -- ``u_bar = (gamma h - f) /
  g`` (``dh/dt >= -gamma h`` with ``dTw/dt = f + g tau_m^2``) and ``tau_bar =
  sqrt(max(u_bar, 0))``. ``u_bar < 0``: no torque keeps the condition."""
  f, g = state.th.affine_terms(state.T, w)
  u_bar = (float(gamma) * (float(limit_c) - state.Tw) - f) / g
  return torch.sqrt(u_bar.clamp_min(0.0)), u_bar


def joint_torque(th, tau_m: torch.Tensor, pairs: dict) -> torch.Tensor:
  """Per-motor torques ``(..., 29)`` -> joint torques, both in motor order: the
  inverse of ``MotorThermal.motor_torque`` (a linkage pair's motors carry
  ``(tau_p +- s tau_r) / 2``, so ``tau_p = m_p + m_r``, ``tau_r = s (m_p - m_r)``)."""
  out = tau_m.clone()
  for name, (p, r) in pairs.items():
    s = float(th.linkage_sign[name])
    out[..., p] = tau_m[..., p] + tau_m[..., r]
    out[..., r] = s * (tau_m[..., p] - tau_m[..., r])
  return out


@dataclass(kw_only=True)
class ThermalBarrierActionCfg(JointPositionActionCfg):
  """``JointPositionActionCfg`` whose targets the temperature barrier clips
  (module docstring); same fields, plus the barrier's limit and rate."""

  limit_c: float = CBF_LIMIT_C
  gamma: float = CBF_GAMMA
  thermal_cfg: ThermalCfg = field(default_factory=ThermalCfg)

  def build(self, env) -> ThermalBarrierAction:
    return ThermalBarrierAction(self, env)


class ThermalBarrierAction(JointPositionAction):
  """Joint position targets, then the per-motor temperature barrier, every
  physics substep (``apply_actions`` runs inside the decimation loop, before
  each ``mj_step``; the thermal state is the previous substep's). It acts on
  whatever writes the actions -- a policy, SONIC, a filter -- as a low-level
  guard on the robot would."""

  cfg: ThermalBarrierActionCfg

  def __init__(self, cfg: ThermalBarrierActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self.state = thermal_state(env, cfg.thermal_cfg)
    ent = self._entity
    joints = list(ent.joint_names)
    tnames = [joints[i] for i in self._target_ids.tolist()]
    missing = [n for n in self.state.names if n + "_joint" not in tnames]
    if missing:
      raise ValueError(f"the barrier action does not drive the joints of motors {missing}")
    #: motor i -> its joint's column among the action targets
    self.m2t = torch.tensor([tnames.index(n + "_joint") for n in self.state.names],
                            device=self.device)
    kp = [math.nan] * len(tnames)
    kd = [math.nan] * len(tnames)
    for act in ent.actuators:
      st, dp = getattr(act.cfg, "stiffness", None), getattr(act.cfg, "damping", None)
      for n in act.target_names:
        if n in tnames and st is not None and dp is not None:
          kp[tnames.index(n)], kd[tnames.index(n)] = float(st), float(dp)
    if any(not (k > 0.0) for k in kp) or any(math.isnan(k) for k in kd):
      raise ValueError("the barrier needs every target joint driven by a PD actuator "
                       "(stiffness > 0, damping)")
    self.kp = torch.tensor(kp, device=self.device)
    self.kd = torch.tensor(kd, device=self.device)
    self.limit_c, self.gamma = float(cfg.limit_c), float(cfg.gamma)
    self.pairs = linkages()
    n = self.num_envs
    self._clip_acc = torch.zeros(n, device=self.device)
    self._neg_acc = torch.zeros(n, device=self.device)
    self._n_sub = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._clip_acc.zero_()
    self._neg_acc.zero_()
    self._n_sub = 0

  def safe_targets(self, target: torch.Tensor, q: torch.Tensor, qd: torch.Tensor):
    """``(target', clipped, u_bar)`` for joint targets / positions / velocities
    ``(num_envs, n_targets)`` in the action's order; ``clipped`` and ``u_bar``
    per motor ``(num_envs, 29)``."""
    th = self.state.th
    tau = self.kp * (target - q) - self.kd * qd                 # the PD torque the dcmotor applies
    tau_m = th.motor_torque(tau[:, self.m2t])
    w = th.motor_speed(qd[:, self.m2t])
    tau_bar, u_bar = barrier_bound(self.state, w, self.limit_c, self.gamma)
    clipped = tau_m.abs() > tau_bar
    safe_m = torch.minimum(torch.maximum(tau_m, -tau_bar), tau_bar)
    tau_j = joint_torque(th, safe_m, self.pairs)                # motor order
    moved = clipped.clone()                                     # a clipped linkage motor moves both joints
    for p, r in self.pairs.values():
      both = clipped[:, p] | clipped[:, r]
      moved[:, p], moved[:, r] = both, both
    tau_safe = tau.clone()
    tau_safe[:, self.m2t] = tau_j
    mask = torch.zeros_like(tau, dtype=torch.bool)
    mask[:, self.m2t] = moved
    target = torch.where(mask, q + (tau_safe + self.kd * qd) / self.kp, target)
    return target, clipped, u_bar

  def apply_actions(self) -> None:
    ent = self._entity
    ids = self._target_ids
    target = self._processed_actions - ent.data.encoder_bias[:, ids]
    target, clipped, u_bar = self.safe_targets(target, ent.data.joint_pos[:, ids],
                                               ent.data.joint_vel[:, ids])
    ent.set_joint_position_target(target, joint_ids=ids)
    self._clip_acc += clipped.float().mean(1)
    self._neg_acc += (u_bar < 0.0).float().mean(1)
    self._n_sub += 1

  @property
  def clip_frac(self) -> torch.Tensor:
    """Share of motors clipped, averaged over this env step's substeps, ``(num_envs,)``."""
    return self._clip_acc / max(self._n_sub, 1)

  @property
  def negative_bound_frac(self) -> torch.Tensor:
    """Share of motors whose bound was negative (no torque keeps the barrier)."""
    return self._neg_acc / max(self._n_sub, 1)


def cbf_clip_frac(env, action_name: str = "joint_pos") -> torch.Tensor:
  """Metric: share of motors the barrier clipped this step."""
  return env.action_manager.get_term(action_name).clip_frac


def cbf_negative_bound_frac(env, action_name: str = "joint_pos") -> torch.Tensor:
  """Metric: share of motors with no torque that keeps the barrier this step."""
  return env.action_manager.get_term(action_name).negative_bound_frac


def cbf_overshoot_c(env, limit_c: float = CBF_LIMIT_C) -> torch.Tensor:
  """Metric: how far the hottest winding is above the barrier's limit, C (>= 0)."""
  return (thermal_state(env).Tw.amax(dim=1) - float(limit_c)).clamp_min(0.0)
