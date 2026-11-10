"""CPU MuJoCo rollouts from a live G1 env's state: the rollout filter off the GPU.

    spec = build_spec(env, actor_path, out_dir)       # once, from the live env
    pool = CpuRolloutPool(spec, workers=32)
    gmin, t_fail = pool.rollout(env_ids, qpos, qvel, qacc_ws, a_first, horizon)

The rollout filter (``eval_sonic_qcbf.py`` condition ``rollout:<seconds>``)
asks, every control step and for every tracked env: if this action (SONIC's)
is played now and SHAC drives for the rest of H, does the fall margin g stay
>= 0? This module answers in plain MuJoCo on the CPU, one MjData per robot, in
worker processes -- the deployable form of ``filters.monitor.RolloutMonitor``,
which steps a GPU shadow env instead.

What a rollout reproduces of the live env (``validate_cpu_rollout.py``
measures how well):

* the ROBOT: the live env's compiled MjModel (the --hip actuators, solver
  options and sensors included) with each env slot's domain-randomization draw
  copied in -- ``filters.rollout.dr_identity_fields``: foot friction, torso COM
  and the constants recomputed from them -- and its encoder bias, which moves
  the joint targets (mjlab's JointPositionAction commands
  ``offset + scale * a - bias``);
* the STATE: ``qpos``, ``qvel`` and ``qacc_warmstart`` (the solver's warm
  start; without it the first step already departs);
* a CONTROL STEP as mjlab takes one: ctrl held for ``decimation`` mj_steps,
  g from the post-step kinematics and the peak non-foot contact force over
  those substeps (the margin reads ``force_history``), then the actor's
  observation after one mj_forward (mjlab's single post-step forward);
* g = ``compose(g_fall)``: pelvis height, tilt and non-foot ground force,
  clamped to +-CLAMP (on a ``*_thermal`` task also the hottest motor winding,
  ``(85 C - max Tw) / 10 C``, the episode ending at 85 C as in mjlab; the
  winding / housing / structure temperatures are part of the seeded state and
  the sysID heat model is integrated after every mj_step, as
  ``envs/g1_recover/thermal.py`` does); or, on a task that ends on leaving K (``k_failure``,
  e.g. ``g1_recover_bench``), K's margin (``k_set.margin_from_terms``: pelvis
  height, tilt, every joint within its range + 0.1 rad, no non-foot contact,
  read as mjlab's termination reads it: the derived pose and contact
  sensors of the last substep, the joints current) with the task's joint scale;
* SHAC's input: the actor group rebuilt from MjData term by term, noise-free
  by default; a NOISY check (``n_rollouts`` > 1 or ``noise=True``) runs R
  rollouts per robot, each reading it through the env's own sensor-noise law
  (a fresh uniform draw per rollout and step), and takes the worst. SHAC
  itself is a TorchScript export of the live call (:func:`export_actor`).

What it does not: the live env's future pushes (a filter cannot know them),
MuJoCo Warp's own collision and solver arithmetic (a rollout is CPU MuJoCo's
answer and drifts from the live trajectory at the system's own rate), and
episode bookkeeping (a rollout is H control steps of physics, cut short only
when g < 0).

Workers are spawned processes that import numpy, mujoco and torch only, which
is why this module's top-level imports are just those: the sandbox package
pulls in mjlab and warp on import.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

#: The actor-group terms a rollout can rebuild from MjData.
OBS_TERMS = ("base_ang_vel", "projected_gravity", "command", "phase", "joint_pos",
             "joint_vel", "actions", "pelvis_height", "lin_vel_heading", "foot_contact",
             "motor_temp")
#: K's constants (``envs/g1_recover/k_set.py``), copied: workers import no sandbox code.
K_H_MIN, K_H_SCALE, K_THETA_MAX, K_JOINT_OVERSHOOT, K_ANCHOR = 0.45, 0.1, 1.0, 0.1, -0.1


@dataclass
class Spec:
  """Everything a worker needs, picklable (``build_spec``)."""
  mjb_path: str
  actor_path: str
  n_slots: int
  decimation: int
  ctrl_gain: float
  #: per actuator: the action index driving it (-1: none), the term's scale / offset
  act_index: np.ndarray
  act_scale: np.ndarray
  act_offset: np.ndarray
  #: [n_slots, nu]: each slot's encoder bias on its actuators' joints
  ctrl_bias: np.ndarray
  #: model field -> [n_slots, ...]: each slot's DR draw
  dr: dict = field(default_factory=dict)
  #: (term, start, size, extra) of the actor group, in order
  obs_terms: list = field(default_factory=list)
  obs_dim: int = 0
  gyro_adr: int = -1
  #: the robot's joints in the actor's order: qpos / qvel addresses, defaults
  joint_qadr: np.ndarray = None
  joint_vadr: np.ndarray = None
  default_jpos: np.ndarray = None
  default_jvel: np.ndarray = None
  #: [n_slots, n_joints]: encoder bias as the joint_pos term reads it (biased terms only)
  obs_bias: np.ndarray = None
  root_qadr: int = 0
  root_vadr: int = 0
  #: the root body (its xpos / xquat: the pose mjlab's terminations read)
  root_body: int = 1
  #: sensordata start of every non-foot ground-contact force sensor (3 values each)
  force_adr: np.ndarray = None
  #: [obs_dim]: the actor group's additive uniform sensor-noise law (None: none)
  obs_noise_low: np.ndarray = None
  obs_noise_high: np.ndarray = None
  z_min: float = 0.45
  z_scale: float = 0.25
  tilt_cos: float = math.cos(math.radians(50.0))
  force_threshold: float = 10.0
  clamp: float = 3.0
  #: "fall" (compose(g_fall)) or "k" (K's margin, k_set.margin_from_terms)
  margin: str = "fall"
  k_joint_scale: float = 0.1
  #: [n_joints, 2]: the joints' ranges in the actor's joint order (K's margin)
  joint_range: np.ndarray = None
  #: sensordata of every non-foot contact "found" sensor (K's margin)
  nonfoot_found_adr: np.ndarray = None
  #: sensordata of the two feet's "found" sensors (the foot_contact term)
  foot_found_adr: np.ndarray = None
  #: the motor heat model of a *_thermal task (None: no temperatures): the
  #: sysID repo path, MotorThermal's fields as numpy, per motor the model's
  #: actuator index and the joint's qvel address, the limit and margin scale
  thermal: dict = None


# --- the parent side: building the spec from a live mjlab env -------------------

def _np(x) -> np.ndarray:
  t = getattr(x, "_tensor", x)
  return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def export_actor(policy, norm, obs_dim: int, pi_slice, box_low, box_high, low, high,
                 path: str) -> None:
  """TorchScript of the live fallback call, on CPU: ``obs`` [B, obs_dim] (raw
  actor group) -> the clamped action. ``pi_slice`` (or None): the last-action
  columns, clamped to [box_low, box_high] first (``--pi-obs-clamp box``).
  ``policy`` / ``norm`` must already live on the CPU. Checked against the
  direct call before it is written."""
  lo, hi = torch.as_tensor(low).float(), torch.as_tensor(high).float()
  blo = None if pi_slice is None else torch.as_tensor(box_low).float()
  bhi = None if pi_slice is None else torch.as_tensor(box_high).float()

  def call(obs):
    if pi_slice is not None:
      a, b = pi_slice.start, pi_slice.stop
      obs = torch.cat([obs[:, :a], torch.maximum(torch.minimum(obs[:, a:b], bhi), blo),
                       obs[:, b:]], 1)
    return torch.clamp(policy._predict(norm(obs), deterministic=True), lo, hi)

  class _Actor(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.policy = policy

    def forward(self, obs):
      return call(obs)

  gen = torch.Generator().manual_seed(0)
  x = torch.randn(64, obs_dim, generator=gen) * 2.0
  with torch.no_grad():
    traced = torch.jit.trace(_Actor().eval(), x, check_trace=False)
    for b in (1, 7, 64):
      xb = torch.randn(b, obs_dim, generator=gen) * 3.0
      err = float((traced(xb) - call(xb)).abs().max())
      if err > 1e-5:
        raise RuntimeError(f"the traced fallback departs from the live call by {err:.2e}")
  traced.save(path)


def _sensor_adr(m, prefix: str, suffix: str) -> list:
  out = []
  for i in range(m.nsensor):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
    if nm.startswith(prefix) and nm.endswith(suffix):
      out.append(int(m.sensor_adr[i]))
  return out


def build_spec(env, actor_path: str, out_dir: str, margin: str = "auto",
               k_joint_scale: float | None = None) -> Spec:
  """The rollout spec of a live ``MjlabTensorSafetyEnv`` (any env slot can be
  rolled out). Writes ``<out_dir>/rollout_model.mjb``. Fails loudly on an
  actor term, actuator or sensor it does not know how to rebuild.

  ``margin``: ``"fall"`` (compose(g_fall)), ``"k"`` (K's margin with joint
  scale ``k_joint_scale``: default the task margin's own, else k_set's 0.1) or
  ``"auto"``: ``"k"`` when the env ends episodes on leaving K (a ``k_failure``
  termination term)."""
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
    CLAMP, CONTACT_FORCE_THRESHOLD, TILT_COS, Z_MIN, Z_SCALE)
  from robot_safety_sandbox.filters.rollout import dr_identity_fields
  mj = env.mj
  m = mj.sim.mj_model
  n = int(mj.num_envs)
  os.makedirs(out_dir, exist_ok=True)
  mjb = os.path.join(out_dir, "rollout_model.mjb")
  mujoco.mj_saveModel(m, mjb, None)

  # each slot's DR draw (fields never expanded per world are shared: skip)
  dr = {}
  model_fields, _entity = dr_identity_fields(mj)
  for f in model_fields:
    x = getattr(mj.sim.model, f, None)
    t = getattr(x, "_tensor", x)
    if t is None or not torch.is_tensor(t) or not t.numel() or t.shape[0] != n or t.stride(0) == 0:
      continue
    if not hasattr(m, f) or tuple(np.shape(getattr(m, f))) != tuple(t.shape[1:]):
      raise SystemExit(f"[rollout] DR field {f!r} {tuple(t.shape)} has no matching MjModel field")
    dr[f] = t.detach().cpu().numpy().copy()

  robot = mj.scene["robot"]
  idx = robot.indexing
  qadr = _np(idx.joint_q_adr).astype(np.int64)
  vadr = _np(idx.joint_v_adr).astype(np.int64)
  fq = _np(idx.free_joint_q_adr).astype(np.int64)
  fv = _np(idx.free_joint_v_adr).astype(np.int64)
  if fq.size != 7:
    raise SystemExit("[rollout] the robot has no free joint")
  bias = _np(robot.data.encoder_bias).astype(np.float64)          # [n, joints]

  # actions -> actuators: ctrl = offset + scale * (ctrl_gain * a) - bias
  term = mj.action_manager.get_term("joint_pos")
  tids = _np(term.target_ids).astype(np.int64).reshape(-1)

  def per_action(x) -> np.ndarray:      # a scalar, [A] or [n, A] (the same on every env)
    x = _np(x).astype(np.float64)
    return np.full(tids.size, float(x)) if x.size == 1 else x.reshape(-1, tids.size)[0].copy()

  scale, offset = per_action(term.scale), per_action(term.offset)
  jnt_of_act = {}
  for k, j in enumerate(tids):
    jnt_of_act[int(qadr[j])] = k
  nu = int(m.nu)
  act_index = np.full(nu, -1, dtype=np.int64)
  for i in range(nu):
    if m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
      raise SystemExit(f"[rollout] actuator {i} is not a joint actuator")
    # a position actuator, or a <dcmotor> in POSITION mode (the identified G1:
    # ctrl is the joint target either way; it keeps no activation state, na = 0)
    dc_pos = (m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_DCMOTOR
              and int(m.actuator_ctrlspec[i]) & int(mujoco.mjtCtrlInput.mjINPUT_POS))
    if m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE and not dc_pos:
      raise SystemExit(f"[rollout] actuator {i} is not a position actuator")
    jid = int(m.actuator_trnid[i, 0])
    act_index[i] = jnt_of_act.get(int(m.jnt_qposadr[jid]), -1)
  if m.na:
    raise SystemExit(f"[rollout] the model has {m.na} actuator states; a rollout seeds none")
  if (act_index < 0).any():
    raise SystemExit(f"[rollout] {int((act_index < 0).sum())} actuators are driven by no action")
  a_scale, a_off = scale[act_index], offset[act_index]
  ctrl_bias = bias[:, tids][:, act_index]

  therm = _thermal_spec(mj, robot)

  # the actor group, term by term
  om, key = mj.observation_manager, env.obs_key
  names = om._group_obs_term_names[key]
  dims = om._group_obs_term_dim[key]
  cfgs = om._group_obs_term_cfgs[key]
  obs_now = _np(om.compute_group(key, update_history=False)).astype(np.float64)
  terms, start, gyro_adr, obs_bias = [], 0, -1, None
  for nm, dm, cfg in zip(names, dims, cfgs):
    size = int(np.prod(dm))
    fn = getattr(getattr(cfg, "func", None), "__name__", "")
    extra = {}
    if nm not in OBS_TERMS:
      raise SystemExit(f"[rollout] actor term {nm!r} ({fn}) cannot be rebuilt on the CPU")
    if getattr(cfg, "scale", None) is not None or getattr(cfg, "clip", None) is not None:
      raise SystemExit(f"[rollout] actor term {nm!r} is scaled / clipped")
    params = dict(getattr(cfg, "params", None) or {})
    if nm == "base_ang_vel":
      gyro_adr = int(m.sensor(params["sensor_name"]).adr[0])
    elif nm in ("command", "phase"):
      # the recover task pins the twist command to zero, which also zeroes the phase
      col = obs_now[:, start:start + size]
      if np.abs(col).max() > 0:
        raise SystemExit(f"[rollout] actor term {nm!r} is not identically zero")
    elif nm == "joint_pos":
      if params.get("biased", False):
        obs_bias = bias.copy()
      ac = params.get("asset_cfg")
      if ac is not None and getattr(ac, "joint_ids", slice(None)) not in (slice(None), None):
        raise SystemExit("[rollout] joint_pos on a joint subset")
    elif nm == "motor_temp" and therm is None:
      raise SystemExit("[rollout] actor term 'motor_temp' but the env has no motor thermal state")
    elif nm == "foot_contact" and params.get("sensor_name") != "feet_ground_contact":
      raise SystemExit(f"[rollout] foot_contact reads {params.get('sensor_name')!r}, not "
                       "feet_ground_contact")
    if size != {"base_ang_vel": 3, "projected_gravity": 3, "command": 3, "phase": 2,
                "joint_pos": qadr.size, "joint_vel": vadr.size, "actions": tids.size,
                "pelvis_height": 1, "lin_vel_heading": 3, "foot_contact": 2,
                "motor_temp": 29}[nm]:
      raise SystemExit(f"[rollout] actor term {nm!r} is {size}-D")
    terms.append((nm, start, size, extra))
    start += size

  # the sensor-noise law of the actor group (cfg.noise per term; drawn by a
  # noisy check whether or not the live env has its corruption on)
  n_lo, n_hi = np.zeros(start), np.zeros(start)
  for nm, (st, sz, cfg) in env._obs_layout(key).items():
    lo_t, hi_t = env._term_noise_bounds(key, nm, sz, cfg)
    n_lo[st:st + sz], n_hi[st:st + sz] = lo_t.numpy(), hi_t.numpy()
  fadr = _sensor_adr(m, "nonfoot_ground_touch_", "_force")
  if not fadr:
    raise SystemExit("[rollout] no nonfoot_ground_touch_*_force sensors in the model")
  if margin == "auto":
    margin = "k" if "k_failure" in getattr(mj.cfg, "terminations", {}) else "fall"
  hook = getattr(mj.cfg, "rewards", {}).get("zoo_safety_hook")
  avoid = getattr((getattr(hook, "params", None) or {}).get("margin_fn"), "avoid_margin", None)
  if k_joint_scale is None:
    # the task's K margin says its joint scale (k_set.g_k / push_bench.g_bench)
    k_joint_scale = getattr(avoid, "k_joint_scale", None)
  # g_fall_noheight (the *_noheight tasks) drops g_fall's pelvis-height term:
  # z_min = -inf makes that term +inf, so it never binds
  z_min = Z_MIN if getattr(avoid, "fall_height_term", True) else -math.inf
  if margin not in ("fall", "k"):
    raise SystemExit(f"[rollout] margin {margin!r}: fall | k | auto")
  t_margin = getattr(avoid, "thermal", None)
  if therm is not None:
    if margin != "fall" or t_margin is None:
      raise SystemExit("[rollout] a motor thermal state needs the fall margin with its thermal "
                       "term (g_fall_noheight_thermal)")
    therm.update(limit_c=float(t_margin["limit_c"]), scale_c=float(t_margin["scale_c"]))
  elif t_margin is not None:
    raise SystemExit("[rollout] the margin has a thermal term but the env no thermal state")
  k_extra = {}
  if margin == "k":
    found = _sensor_adr(m, "nonfoot_ground_touch_", "_found")
    if not found:
      raise SystemExit("[rollout] the K margin needs the nonfoot_ground_touch_*_found sensors")
    k_extra = dict(margin="k", k_joint_scale=float(0.1 if k_joint_scale is None else k_joint_scale),
                   joint_range=_np(robot.data.joint_pos_limits)[0].astype(np.float64),
                   nonfoot_found_adr=np.asarray(found))
  if any(t[0] == "foot_contact" for t in terms):
    feet = [_sensor_adr(m, "feet_ground_contact_", f"{side}_ankle_roll_link_found")
            for side in ("left", "right")]
    if any(len(f) != 1 for f in feet):
      raise SystemExit("[rollout] foot_contact needs one feet_ground_contact_<left|right>_"
                       "ankle_roll_link_found sensor per foot")
    k_extra["foot_found_adr"] = np.asarray([f[0] for f in feet])
  return Spec(mjb_path=mjb, actor_path=actor_path, n_slots=n,
              decimation=int(mj.cfg.decimation), ctrl_gain=float(getattr(env, "ctrl_gain", 1.0)),
              act_index=act_index, act_scale=a_scale, act_offset=a_off, ctrl_bias=ctrl_bias,
              dr=dr, obs_terms=terms, obs_dim=start, gyro_adr=gyro_adr,
              joint_qadr=qadr, joint_vadr=vadr,
              default_jpos=_np(robot.data.default_joint_pos)[0].astype(np.float64),
              default_jvel=_np(robot.data.default_joint_vel)[0].astype(np.float64),
              obs_bias=obs_bias, root_qadr=int(fq[0]), root_vadr=int(fv[0]),
              root_body=int(_np(idx.root_body_id).reshape(-1)[0]),
              force_adr=np.asarray(fadr),
              obs_noise_low=n_lo, obs_noise_high=n_hi,
              z_min=z_min, z_scale=Z_SCALE, tilt_cos=TILT_COS,
              force_threshold=CONTACT_FORCE_THRESHOLD, clamp=CLAMP, thermal=therm, **k_extra)


def _thermal_spec(mj, robot) -> dict | None:
  """The ``Spec.thermal`` of a live env with a motor thermal state, else None."""
  import dataclasses

  from robot_safety_sandbox.envs.assets.g1_sysid import SYSID_DIR
  from robot_safety_sandbox.envs.g1_recover.thermal import has_thermal, thermal_state
  if not has_thermal(mj):
    return None
  st = thermal_state(mj)
  params = {}
  for f in dataclasses.fields(st.th):
    v = getattr(st.th, f.name)
    params[f.name] = _np(v).astype(np.float64).copy() if torch.is_tensor(v) else copy.deepcopy(v)
  # robot.data.actuator_force / joint_vel index the model's actuators / dofs
  # through these: motor i -> model actuator, joint dof
  ctrl_ids = _np(robot.indexing.ctrl_ids).astype(np.int64).reshape(-1)
  vadr = _np(robot.indexing.joint_v_adr).astype(np.int64).reshape(-1)
  return dict(sysid_dir=str(SYSID_DIR), params=params,
              act=ctrl_ids[_np(st.act_perm).astype(np.int64)],
              vadr=vadr[_np(st.jnt_perm).astype(np.int64)], dt=float(mj.physics_dt))


def live_state(env, ids: torch.Tensor):
  """``(qpos, qvel, qacc_warmstart)`` of the live envs ``ids``, as float64 numpy."""
  d = env.mj.sim.data
  return tuple(_np(getattr(d, f))[ids.cpu().numpy()].astype(np.float64)
               for f in ("qpos", "qvel", "qacc_warmstart"))


def live_thermal(env, ids: torch.Tensor) -> np.ndarray | None:
  """``[k, 3, 29]`` winding / housing / structure temperatures of the live envs
  ``ids`` (float64), or None when the env has no motor thermal state."""
  from robot_safety_sandbox.envs.g1_recover.thermal import has_thermal, thermal_state
  if not has_thermal(env.mj):
    return None
  st = thermal_state(env.mj)
  return np.stack([_np(x[ids]).astype(np.float64) for x in st.T], axis=1)


# --- the worker side -----------------------------------------------------------

def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """``R(q)^T v`` for a unit quaternion ``q`` = (w, x, y, z)."""
  w, u = q[0], q[1:]
  return v * (2.0 * w * w - 1.0) - np.cross(u, v) * 2.0 * w + u * (2.0 * np.dot(u, v))


class Roller:
  """One worker's robots: an MjModel (with its DR draw) per slot, MjData per
  slot and rollout (``R`` of them for a noisy check, made on demand), and the
  exported fallback. Usable in-process too (the validation does)."""

  def __init__(self, spec: Spec, slots, seed: int = 0):
    self.s = spec
    base = mujoco.MjModel.from_binary_path(spec.mjb_path)
    self.models, self.datas = {}, {}
    for slot in slots:
      m = copy.deepcopy(base)
      for f, rows in spec.dr.items():
        getattr(m, f)[...] = rows[slot]
      self.models[slot] = m
      self.datas[slot] = [mujoco.MjData(m)]
    self.actor = torch.jit.load(spec.actor_path)
    self.actor.eval()
    self.tilt_norm = 1.0 - spec.tilt_cos
    self.rng = np.random.default_rng(seed)
    lo, hi = spec.obs_noise_low, spec.obs_noise_high
    self.noise_lo = np.zeros(spec.obs_dim) if lo is None else np.asarray(lo, dtype=np.float64)
    self.noise_span = (np.zeros(spec.obs_dim) if hi is None
                       else np.asarray(hi, dtype=np.float64) - self.noise_lo)
    #: (slot, r) -> (Tw, Th, Ts) of a *_thermal task's rollout
    self.temps, self.th = {}, None
    if spec.thermal is not None:
      if spec.thermal["sysid_dir"] not in sys.path:
        sys.path.insert(0, spec.thermal["sysid_dir"])
      from g1_sysid.thermal import MotorThermal   # numpy only
      self.th = MotorThermal(**copy.deepcopy(spec.thermal["params"]))

  def _d(self, slot: int, r: int = 0) -> mujoco.MjData:
    ds = self.datas[slot]
    while len(ds) <= r:
      ds.append(mujoco.MjData(self.models[slot]))
    return ds[r]

  def seed(self, slot: int, qpos, qvel, qacc_ws, r: int = 0, temps=None) -> mujoco.MjData:
    """``temps`` ``[3, 29]``: winding / housing / structure (*_thermal tasks)."""
    d = self._d(slot, r)
    mujoco.mj_resetData(self.models[slot], d)
    d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = qpos, qvel, qacc_ws
    if self.th is not None:
      if temps is None:
        raise ValueError("a *_thermal rollout needs the start temperatures")
      self.temps[(slot, r)] = tuple(np.asarray(t, dtype=np.float64).copy() for t in temps)
    return d

  def _heat(self, d: mujoco.MjData, key) -> None:
    """One physics substep of the heat model (``thermal.MotorThermalState.step``)."""
    th, t = self.th, self.s.thermal
    tau = th.motor_torque(d.actuator_force[t["act"]].astype(np.float64))
    w = th.motor_speed(d.qvel[t["vadr"]].astype(np.float64))
    self.temps[key] = th.step(self.temps[key], tau, w, t["dt"])

  def control_step(self, slot: int, a: np.ndarray, r: int = 0) -> float:
    """``a`` (env action units) for one control step; returns g after it."""
    s, m, d = self.s, self.models[slot], self._d(slot, r)
    d.ctrl[:] = s.act_offset + s.act_scale * (s.ctrl_gain * a[s.act_index]) - s.ctrl_bias[slot]
    if s.margin == "k":
      for _ in range(s.decimation):
        mujoco.mj_step(m, d)
      # mjlab reads its terminations, its margin and (sensor cache) the contact
      # flags of the next observation BEFORE its post-step forward: the derived
      # pose and the sensors are those of the last substep's start, the joints
      # current. Read them the same way, and latch the foot flags for observe().
      if s.foot_found_adr is not None:
        self._latch = ((slot, r), d.sensordata[s.foot_found_adr].copy())
      return self.margin_k(d)
    peak = 0.0
    for _ in range(s.decimation):
      mujoco.mj_step(m, d)
      if self.th is not None:
        self._heat(d, (slot, r))
      f = d.sensordata[s.force_adr[:, None] + np.arange(3)]
      peak = max(peak, float(np.sqrt((f * f).sum(1)).max()))
    g = self.margin(d, peak)
    if self.th is not None:
      t = s.thermal
      hottest = float(self.temps[(slot, r)][0].max())
      g = min(g, float(np.clip((t["limit_c"] - hottest) / t["scale_c"], -s.clamp, s.clamp)))
      if hottest >= t["limit_c"]:     # mjlab's motor_overheat termination: the hook's anchor
        g = min(g, K_ANCHOR)
    return g

  def margin_k(self, d: mujoco.MjData) -> float:
    """K's margin (``k_set.margin_from_terms``) as mjlab's termination reads it
    after a step: the root body's derived pose and the contact sensors of the
    last substep (call before any forward), the joints from qpos."""
    s = self.s
    quat = d.xquat[s.root_body]
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    tilt = math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))))
    jq = d.qpos[s.joint_qadr]
    lo, hi = s.joint_range[:, 0] - K_JOINT_OVERSHOOT, s.joint_range[:, 1] + K_JOINT_OVERSHOOT
    joint = float(np.minimum(jq - lo, hi - jq).min())
    nonfoot = bool((d.sensordata[s.nonfoot_found_adr] > 0).any())
    z = float(d.xpos[s.root_body][2])
    g = min((z - K_H_MIN) / K_H_SCALE, (K_THETA_MAX - tilt) / K_THETA_MAX, joint / s.k_joint_scale)
    g = float(np.clip(g, -s.clamp, s.clamp))
    if nonfoot:
      g = -1.0
    if z < K_H_MIN or tilt > K_THETA_MAX or joint < 0.0 or nonfoot:
      return min(g, K_ANCHOR)
    return max(g, 0.0)

  def margin(self, d: mujoco.MjData, peak_force: float) -> float:
    s = self.s
    q = d.qpos[s.root_qadr:s.root_qadr + 7]
    up = -_quat_rotate_inv(q[3:7], np.array([0.0, 0.0, -1.0]))[2]
    g = min((q[2] - s.z_min) / s.z_scale, (up - s.tilt_cos) / self.tilt_norm,
            (s.force_threshold - peak_force) / s.force_threshold)
    return float(np.clip(g, -s.clamp, s.clamp))

  def observe(self, slot: int, last_action: np.ndarray, out: np.ndarray, r: int = 0) -> None:
    """The actor group at the current state (after ``mj_forward``) into ``out``."""
    s, m, d = self.s, self.models[slot], self._d(slot, r)
    mujoco.mj_forward(m, d)
    latch = getattr(self, "_latch", None)
    self._latch = None
    q = d.qpos[s.root_qadr:s.root_qadr + 7]
    for nm, st, sz, _x in s.obs_terms:
      seg = out[st:st + sz]
      if nm == "base_ang_vel":
        seg[:] = d.sensordata[s.gyro_adr:s.gyro_adr + 3]
      elif nm == "projected_gravity":
        seg[:] = _quat_rotate_inv(q[3:7], np.array([0.0, 0.0, -1.0]))
      elif nm in ("command", "phase"):
        seg[:] = 0.0
      elif nm == "joint_pos":
        seg[:] = d.qpos[s.joint_qadr] - s.default_jpos
        if s.obs_bias is not None:
          seg += s.obs_bias[slot]
      elif nm == "joint_vel":
        seg[:] = d.qvel[s.joint_vadr] - s.default_jvel
      elif nm == "actions":
        seg[:] = s.ctrl_gain * last_action
      elif nm == "pelvis_height":
        seg[:] = q[2]
      elif nm == "lin_vel_heading":
        w, x, y, z = q[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, sn = math.cos(yaw), math.sin(yaw)
        v = d.qvel[s.root_vadr:s.root_vadr + 3]       # free joint: world linear velocity
        seg[:] = (c * v[0] + sn * v[1], -sn * v[0] + c * v[1], v[2])
      elif nm == "motor_temp":
        seg[:] = self.temps[(slot, r)][0]
      elif nm == "foot_contact":
        # after a control step, the flags of its last substep (mjlab's sensor
        # cache); from a seeded state, the current ones
        found = latch[1] if latch is not None and latch[0] == (slot, r) else \
            d.sensordata[s.foot_found_adr]
        seg[:] = (found > 0).astype(np.float64)

  def rollout(self, slots, qpos, qvel, qacc_ws, a_first, horizon: int,
              n_rollouts: int = 1, noise: bool = False, trace: bool = False, temps=None):
    """Every slot's ``n_rollouts`` rollouts in lockstep: ``a_first`` [k, A]
    for one control step, then the fallback, ``horizon`` control steps in all;
    with ``noise`` the fallback reads its observation through the env's sensor
    noise law (a fresh uniform draw per rollout and step). A slot's verdict is
    its WORST rollout, so a slot stops as soon as one of its rollouts fails.
    Returns ``gmin`` [k] (min g over the horizon and the rollouts) and
    ``t_fail`` [k] (the control step g first went < 0 in any rollout, counted
    from 1; ``horizon + 1`` if never); with ``trace`` (one rollout only) also
    the per-step g and qpos ([k, horizon], [k, horizon, nq]; NaN after a
    failure). ``temps`` [k, 3, 29]: the start temperatures of a *_thermal
    task (``live_thermal``)."""
    R = max(1, int(n_rollouts))
    if trace and R != 1:
      raise ValueError("trace is for single rollouts")
    k = len(slots)
    owner = np.repeat(np.arange(k), R)                 # item -> slot index
    rep = np.tile(np.arange(R), k)                     # item -> rollout index
    n = owner.size
    gmin = np.full(k, np.inf)
    t_fail = np.full(k, horizon + 1, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    act = np.repeat(np.asarray(a_first, dtype=np.float64), R, axis=0)
    obs = np.zeros((n, self.s.obs_dim))
    g_tr = np.full((k, horizon), np.nan) if trace else None
    q_tr = None
    for j in range(n):
      i = owner[j]
      self.seed(slots[i], qpos[i], qvel[i], qacc_ws[i], rep[j],
                None if temps is None else temps[i])
    if trace:
      q_tr = np.full((k, horizon, self._d(slots[0]).qpos.size), np.nan)
    for t in range(horizon):
      for j in np.flatnonzero(alive):
        i, r = owner[j], rep[j]
        if t_fail[i] <= horizon:                       # a sibling already failed
          alive[j] = False
          continue
        g = self.control_step(slots[i], act[j], r)
        gmin[i] = min(gmin[i], g)
        if trace:
          g_tr[i, t] = g
          q_tr[i, t] = self._d(slots[i]).qpos
        if g < 0.0:
          alive[j] = False
          t_fail[i] = t + 1
        elif t + 1 < horizon:
          self.observe(slots[i], act[j], obs[j], r)
      alive &= t_fail[owner] > horizon
      if t + 1 >= horizon or not alive.any():
        break
      live = np.flatnonzero(alive)
      x = obs[live]
      if noise:
        x = x + self.noise_lo + self.noise_span * self.rng.random(x.shape)
      with torch.no_grad():
        act[live] = self.actor(torch.from_numpy(x).float()).double().numpy()
    out = (gmin, t_fail)
    return out + (g_tr, q_tr) if trace else out


def _worker(conn, spec: Spec, slots, seed: int) -> None:
  torch.set_num_threads(1)
  roller = Roller(spec, slots, seed)
  conn.send("ready")
  while True:
    msg = conn.recv()
    if msg is None:
      break
    sl, qpos, qvel, ws, a0, horizon, n_rollouts, noise, temps = msg
    conn.send(roller.rollout(list(sl), qpos, qvel, ws, a0, horizon, n_rollouts, noise,
                             temps=temps))
  conn.close()


class CpuRolloutPool:
  """``workers`` spawned processes; env slot j lives in worker ``j % workers``.
  :meth:`rollout` scatters the requested slots, gathers in order."""

  def __init__(self, spec: Spec, workers: int, seed: int = 0):
    self.spec = spec
    self.workers = max(1, min(int(workers), spec.n_slots))
    ctx = mp.get_context("spawn")
    self.conns, self.procs = [], []
    # A spawned child re-imports the parent's main script unless it cannot find
    # it; the eval script imports the whole sandbox (mjlab, warp), which the
    # workers do not need, so hide it while they start.
    main = sys.modules["__main__"]
    hidden = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None              # spawn reads the attribute; None = no module name
    try:
      for w in range(self.workers):
        parent, child = ctx.Pipe()
        slots = list(range(w, spec.n_slots, self.workers))
        p = ctx.Process(target=_worker, args=(child, spec, slots, seed * 1000 + w),
                        daemon=True)
        p.start()
        self.conns.append(parent)
        self.procs.append(p)
    finally:
      main.__dict__.update(hidden)
    for c in self.conns:
      if c.recv() != "ready":
        raise RuntimeError("a rollout worker failed to start")

  def rollout(self, slots, qpos, qvel, qacc_ws, a_first, horizon: int,
              n_rollouts: int = 1, noise: bool = False, temps=None):
    """``slots`` [k] env ids (numpy / list); state and ``a_first`` [k, ...];
    ``temps`` [k, 3, 29] on a *_thermal task (``live_thermal``). Returns
    ``gmin`` [k], ``t_fail`` [k] (see :meth:`Roller.rollout`)."""
    slots = np.asarray(slots, dtype=np.int64)
    k = slots.size
    gmin = np.empty(k)
    t_fail = np.empty(k, dtype=np.int64)
    asked = []
    for w, c in enumerate(self.conns):
      sel = np.flatnonzero(slots % self.workers == w)
      if sel.size:
        c.send((slots[sel], qpos[sel], qvel[sel], qacc_ws[sel], a_first[sel], int(horizon),
                int(n_rollouts), bool(noise), None if temps is None else temps[sel]))
        asked.append((c, sel))
    for c, sel in asked:
      g, t = c.recv()
      gmin[sel], t_fail[sel] = g, t
    return gmin, t_fail

  def close(self) -> None:
    for c in self.conns:
      try:
        c.send(None)
      except (BrokenPipeError, OSError):
        pass
    for p in self.procs:
      p.join(timeout=5)
      if p.is_alive():
        p.terminate()
