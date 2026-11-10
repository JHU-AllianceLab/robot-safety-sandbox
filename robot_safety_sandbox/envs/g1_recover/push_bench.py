"""The SONIC push benchmark's world, rebuilt in mjlab.

The benchmark (``robot_states_reachability_checker/tools/sonic_bench``) pushes
SONIC while it tracks LAFAN clips and scores a safety mechanism on whether the
robot stays in the safe set K. This module is that world as an mjlab G1 env,
so a recovery policy trains, its start states are harvested and every safety
filter is scored under the same conditions:

* ``g1_recover_bench`` (``tasks/g1_recover.py``) is the task:
  :func:`bench_env_cfg`, margin :func:`g_bench`, no in-episode disturbance;
* ``examples/harvest_g1_sonic_trajectories.py --push force`` harvests start
  states with :class:`PointPush` (his state pool's push);
* ``examples/make_push_table.py`` writes trial tables in his CSV schema, and
  ``examples/eval_push_bench.py`` runs them.

WHAT MATCHES HIS WORLD

* FAILURE. The safe set K (``k_set``): pelvis >= 0.45 m, tilt <= 1 rad, every
  joint within its range widened by 0.1 rad, no non-foot ground contact. The
  episode ends on leaving K. The margin is his (``k_set.margin_from_terms``
  with joint scale :data:`JOINT_SCALE`, 0.2): g < 0 exactly when outside K.
* PHYSICS. 2 ms steps, 10 per 20 ms control tick, 100 solver iterations,
  tolerance 1e-10, 300 CCD iterations, line search at MuJoCo's default 50
  iterations and serial (mjlab's default is MuJoCo Warp's parallel one; his
  sim turns it off), up to 1024 constraints per world, ``implicitfast``. Ground friction 1 (MuJoCo's default for the
  plane) and foot friction :data:`FOOT_FRICTION` (1.0): the soles carry
  contact priority 1, so their friction is the foot contacts'.
* ROBOT. SONIC's hip-pitch motor (7520-22), so every joint's kp / kd / torque
  limit / armature is his. No domain randomization (foot friction, torso COM,
  encoder bias and mjlab's interval push are removed), no sensor noise.
* OBSERVATION. The actor group is his connector's input, 70-D and exact
  (:data:`ACTOR_TERMS`): pelvis height 1, gravity in the pelvis frame 3, joint
  positions 29 (minus the default pose), pelvis linear velocity in the heading
  frame 3, pelvis angular velocity 3, joint velocities 29, foot contacts 2.
  The critic group (the world model's state) keeps its terms and gains
  ``lin_vel_heading``, so every actor term has its noise-free twin there.
* PUSH. :class:`PointPush`: a world-horizontal force on one of
  :data:`PUSH_SITES` at a point on its collision surface, with the torque
  ``(p - com) x F``, recomputed at every control tick and held through its 10
  physics steps.

WHAT DOES NOT

* The robot file is mjlab's G1 (same total mass, 33.341142 kg), not GR00T's
  ``g1_29dof_rev_1_0.xml``, so collision shapes and surface points differ.
* Actuators are MuJoCo position actuators under ``implicitfast`` (implicit
  damping); his PD torque is explicit.
* K is read once per control tick; he checks every physics step.
* SONIC is the sandbox's port (``sonic.py``) with its own reference
  conventions (window one frame ahead, raw action clipped to +-20).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import numpy as np
import torch

# mjlab is imported inside the functions that need it: the CPU tools
# (examples/make_push_table.py) load this file alone, by path, on machines
# where importing mjlab (warp) is slow.

#: The 12 pushed bodies, his ``SITES`` order (row ``site`` of a trial table).
PUSH_SITES = ("pelvis", "torso_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
              "left_elbow_link", "right_elbow_link", "left_hip_yaw_link", "right_hip_yaw_link",
              "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link")
#: His push directions: the force's angle from the heading, rad (a FORCE
#: direction: "forward" pushes the robot forward).
DIRECTIONS = {"forward": 0.0, "backward": math.pi, "left": math.pi / 2, "right": -math.pi / 2}
#: His margin's joint normalization (the 29-DoF pipeline's; K itself uses 0.1 rad).
JOINT_SCALE = 0.2
#: His physics.
TIMESTEP = 0.002
DECIMATION = 10
SOLVER_ITERATIONS = 100
SOLVER_TOLERANCE = 1e-10
LS_ITERATIONS = 50
CCD_ITERATIONS = 300
#: his serial line search (mjlab defaults to MuJoCo Warp's parallel one, a
#: different solver step) and his per-world constraint capacity
LS_PARALLEL = False
NJMAX = 1024
#: The soles' friction (priority 1: it is the foot contacts' friction).
FOOT_FRICTION = 1.0
#: His trial protocol: SONIC warm-up, push length, scored ticks after the onset.
WARM_TICKS = 100
PUSH_TICKS = 10
SCORED_TICKS = 510
#: SONIC's reference preview past the tracked frame (10 frames, 5 apart).
PREVIEW_FRAMES = 45
#: mjlab G1 events that randomize the robot or push it.
DR_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com")
#: The actor group, in his feature order (all exact).
ACTOR_TERMS = ("pelvis_height", "projected_gravity", "joint_pos", "lin_vel_heading",
               "base_ang_vel", "joint_vel", "foot_contact")


# --- margin and observation terms ----------------------------------------------------------

def g_bench(env) -> torch.Tensor:
  """His avoid margin: K's, joint term over 0.2 rad (g < 0 iff outside K)."""
  from robot_safety_sandbox.envs.g1_recover import k_set
  return k_set.g_k(env, joint_scale=JOINT_SCALE)


g_bench.k_joint_scale = JOINT_SCALE     # cpu_rollout.build_spec reads it


def lin_vel_heading(env) -> torch.Tensor:
  """Pelvis (root link) linear velocity in the heading (yaw-only) frame, ``(n, 3)``."""
  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
  d = env.scene["robot"].data
  return quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)


# --- the env cfg -----------------------------------------------------------------------------

def _foot_friction(cfg, mu: float) -> None:
  robot = cfg.scene.entities["robot"]
  cols = []
  for c in robot.collisions:
    fr = c.friction
    if isinstance(fr, dict):
      fr = {k: ((float(mu),) if "foot" in k else v) for k, v in fr.items()}
      if not any("foot" in k for k in fr):
        raise ValueError(f"collision cfg {c.geom_names_expr}: no foot friction entry to set")
    elif fr is not None:
      fr = (float(mu),)
    cols.append(replace(c, friction=fr))
  cfg.scene.entities["robot"] = replace(robot, collisions=tuple(cols))


def _observations(cfg) -> None:
  from mjlab.managers.observation_manager import ObservationTermCfg
  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  critic.terms["lin_vel_heading"] = ObservationTermCfg(func=lin_vel_heading)
  missing = [t for t in ACTOR_TERMS if t not in critic.terms]
  if missing:
    raise ValueError(f"the critic group has no {missing}: the actor terms need their twins")
  actor.terms = {name: replace(critic.terms[name], noise=None) for name in ACTOR_TERMS}
  actor.enable_corruption = False


def use_bench_world(cfg, foot_friction: float = FOOT_FRICTION) -> None:
  """Turn a ``g1_recover`` env cfg into the benchmark's world (module
  docstring). Mutates ``cfg``; the spawn, the episode length and the rest
  truncation are left to the caller."""
  from robot_safety_sandbox.envs.g1_recover.k_set import use_k_terminations
  from robot_safety_sandbox.envs.g1_recover.sonic import use_sonic_hip_pitch
  use_k_terminations(cfg)
  use_sonic_hip_pitch(cfg)
  for name in DR_EVENTS:
    cfg.events.pop(name, None)
  _foot_friction(cfg, foot_friction)
  m = cfg.sim.mujoco
  m.timestep = TIMESTEP
  m.integrator = "implicitfast"
  m.iterations = SOLVER_ITERATIONS
  m.tolerance = SOLVER_TOLERANCE
  m.ls_iterations = LS_ITERATIONS
  m.ccd_iterations = CCD_ITERATIONS
  cfg.sim.ls_parallel = LS_PARALLEL
  cfg.sim.njmax = NJMAX
  cfg.decimation = DECIMATION
  _observations(cfg)


def bench_env_cfg(play: bool = False, foot_friction: float = FOOT_FRICTION, **kwargs):
  """``g1_recover_env_cfg(**kwargs)`` in the benchmark's world."""
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  cfg = g1_recover_env_cfg(play=play, **kwargs)
  use_bench_world(cfg, foot_friction=foot_friction)
  return cfg


# --- surface points (CPU) ----------------------------------------------------------------

def collision_geoms(model, body_id: int) -> list[int]:
  """The collision geoms of one body of a compiled MjModel."""
  return [g for g in range(model.ngeom)
          if int(model.geom_bodyid[g]) == int(body_id)
          and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]


def sample_surface_points(model, body_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
  """``n`` points drawn uniformly by area on the collision surface of a body
  (capsules and spheres, the G1's), in the BODY frame, ``(n, 3)``. A shared
  surface (two geoms overlapping) is not deduplicated, as in his sampler."""
  import mujoco
  geoms = collision_geoms(model, body_id)
  if not geoms:
    raise ValueError(f"body {body_id} has no collision geoms")
  parts = []                         # (geom, kind, r, h, area)
  for g in geoms:
    t, (r, h) = int(model.geom_type[g]), (float(model.geom_size[g, 0]), float(model.geom_size[g, 1]))
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
      parts.append((g, "sphere", r, 0.0, 4.0 * math.pi * r * r))
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
      parts.append((g, "side", r, h, 2.0 * math.pi * r * 2.0 * h))
      parts.append((g, "caps", r, h, 4.0 * math.pi * r * r))
    else:
      raise ValueError(f"geom {g} of body {body_id}: unsupported collision type {t}")
  area = np.array([p[4] for p in parts])
  pick = rng.choice(len(parts), size=n, p=area / area.sum())
  out = np.empty((n, 3))
  for i, j in enumerate(pick):
    g, kind, r, h, _a = parts[j]
    if kind == "side":
      z, phi = rng.uniform(-h, h), rng.uniform(0.0, 2.0 * math.pi)
      local = np.array([r * math.cos(phi), r * math.sin(phi), z])
    else:
      d = rng.normal(size=3)
      d /= np.linalg.norm(d)
      local = r * d
      if kind == "caps":
        local[2] += h if d[2] >= 0.0 else -h
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    out[i] = model.geom_pos[g] + rot.reshape(3, 3) @ local
  return out


def push_force(mass: float, dv: torch.Tensor, steps: torch.Tensor, dt: float,
               angle_w: torch.Tensor) -> torch.Tensor:
  """His force: ``mass * dv / (steps * dt)``, horizontal at world angle
  ``angle_w``, ``(n, 3)`` N (``dv`` = equivalent free-COM velocity change)."""
  mag = float(mass) * dv / (steps.float() * float(dt))
  return torch.stack([mag * torch.cos(angle_w), mag * torch.sin(angle_w),
                      torch.zeros_like(mag)], dim=-1)


# --- the push (GPU) ------------------------------------------------------------------------

class PointPush:
  """Force pushes on :data:`PUSH_SITES` of an mjlab robot entity.

  ``start(mask, site, point_b, force_w, steps)`` arms a push per env: body
  ``PUSH_SITES[site]``, point ``point_b`` in its body frame, world force
  ``force_w`` (N), for ``steps`` control ticks. Call :meth:`apply` right
  before every env step: it writes each pushing env's force at the body COM
  with the torque ``(p - com) x F`` of its point ``p`` (the body pose of the
  tick start, held through the tick's physics steps), and zeroes the pushed
  bodies of every other env. :meth:`tick` after the step counts the tick
  off; :meth:`stop` ends pushes early (an episode that ended)."""

  def __init__(self, robot, device, sites: Sequence[str] = PUSH_SITES):
    ids, names = robot.find_bodies(list(sites), preserve_order=True)
    if list(names) != list(sites):
      raise ValueError(f"push bodies {list(sites)}: the robot has {list(names)}")
    self.robot, self.dev = robot, device
    self.body_ids = list(ids)
    self._bt = torch.as_tensor(self.body_ids, dtype=torch.long, device=device)
    n = int(robot.data.root_link_pos_w.shape[0])
    self.n = n
    self.site = torch.zeros(n, dtype=torch.long, device=device)
    self.point = torch.zeros(n, 3, device=device)
    self.force = torch.zeros(n, 3, device=device)
    self.left = torch.zeros(n, dtype=torch.long, device=device)
    self._written = True             # zero once at the first apply

  @property
  def active(self) -> torch.Tensor:
    return self.left > 0

  def start(self, mask: torch.Tensor, site: torch.Tensor, point_b: torch.Tensor,
            force_w: torch.Tensor, steps: torch.Tensor) -> None:
    m = mask.reshape(-1).bool()
    self.site = torch.where(m, site.long(), self.site)
    self.point = torch.where(m.unsqueeze(1), point_b.float(), self.point)
    self.force = torch.where(m.unsqueeze(1), force_w.float(), self.force)
    self.left = torch.where(m, steps.long(), self.left)

  def stop(self, mask: torch.Tensor) -> None:
    self.left = torch.where(mask.reshape(-1).bool(), torch.zeros_like(self.left), self.left)

  def tick(self) -> None:
    self.left = (self.left - 1).clamp_min(0)

  def apply(self) -> None:
    from mjlab.utils.lab_api.math import quat_apply
    act = self.active
    on = bool(act.any())
    if not on and not self._written:
      return
    nb = len(self.body_ids)
    F = torch.zeros(self.n, nb, 3, device=self.dev)
    T = torch.zeros(self.n, nb, 3, device=self.dev)
    if on:
      ids = act.nonzero(as_tuple=False).squeeze(-1)
      s = self.site[ids]
      b = self._bt[s]
      d = self.robot.data
      p = d.body_link_pos_w[ids, b] + quat_apply(d.body_link_quat_w[ids, b], self.point[ids])
      f = self.force[ids]
      F[ids, s] = f
      T[ids, s] = torch.cross(p - d.body_com_pos_w[ids, b], f, dim=-1)
    self.robot.write_external_wrench_to_sim(F, T, body_ids=self.body_ids)
    self._written = on
