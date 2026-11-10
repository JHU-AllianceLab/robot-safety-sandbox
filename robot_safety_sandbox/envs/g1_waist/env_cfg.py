"""Unitree G1 reach-avoid-stay to a waist-posture + body-velocity goal.

The goal is a waist (roll, pitch) and a body-frame velocity (vx, vy, wz): the
``waist_target`` command (``command.py``, one draw per episode, magnitude ramped
by a curriculum) plus the ``twist`` command g1_stabilize already pins to zero.
Both are in the actor observation, and so is the base linear velocity, which the
goal's velocity half is measured in.

Reach-avoid-stay is split into two avoid/reach-avoid problems and two policies,
switched at deploy time (``filters/switch.py``):

  STAY   ``g1_waist_stay_arm_random``, mode="safety" (avoid). The constraint is
         "safe AND inside the goal set": h = min(g, soft_tail(l_box)), no
         reach target. V_stay > 0 == the stay policy can keep the robot upright and
         inside the goal set from here. Spawns ON the goal: the waist placed
         within ``spawn_waist_noise`` of this episode's target, the base moving
         at up to ``spawn_vel_noise`` (half the tolerances, so a spawn is inside
         the goal set, not on its edge), the legs in a random standing stance
         (``stance.py``: feet flat, CoM over the feet; ``spawn_stance_scale``).
  REACH  ``g1_waist_reach_arm_random``, mode="reach-avoid". Same g; the target
         is the stay policy's certified set, l = min(l_box, V_stay) with V_stay
         read off the FROZEN stay twin (``stay_value.py``) -- so "reached"
         means "somewhere the stay policy can hold", not merely "inside the
         box". The min with l_box keeps a V_stay that overestimates far from
         the goal (states its training never visited) from declaring arrival
         there. Stock g1_stabilize spawn: at rest in the default pose.

The STAY task is therefore trained first; the reach cfg loads its checkpoint.

g (both): g1_stabilize's fall margin, unchanged -- pelvis height, torso tilt
within 50 deg, no non-foot ground contact.

l_box (both): the min over five normalized terms, each "fraction of the way to
the edge of the goal set",
  (Q_TOL - |q_waist_roll  - roll*|)  / Q_TOL
  (Q_TOL - |q_waist_pitch - pitch*|) / Q_TOL
  (V_TOL - |v_x - vx*|) / V_TOL,  (V_TOL - |v_y - vy*|) / V_TOL   (m/s)
  (V_TOL - |w_z - wz*|) / V_TOL                                   (rad/s)
with the velocities the pelvis's, in its body frame. The three velocity terms
are the three the twist command can set; vz, wx and wy are not part of the goal.

WHY THE STAY CONSTRAINT COMPRESSES l_box. The avoid value is the worst h along
the trajectory, and a fall ENDS the trajectory. With h = min(g, l_box) raw, a
fall cost about -0.84 (g's tilt term when ``fell_over`` fires at 70 deg) while
staying up and drifting more than 0.185 out of the box anywhere in the episode
cost more, down to -CLAMP. Falling early was the optimum, and the first stay
run learned exactly that: every episode fell within ~0.5 s, goal or no goal.
So the stay constraint ranks the two:
  * a failure termination is valued at -CLAMP (``stay_margins.terminal_margin``,
    read by ``base.safety_margin_hook``) instead of the zoo-wide -0.1 anchor,
    which an ``illegal_contact`` touchdown, where g is barely negative, would
    otherwise turn into a cheap way out;
  * l_box is compressed below zero, L asinh(l/L) (identity for l_box >= 0): the
    goal set, and so V_stay's zero level set, is unchanged and the margin near
    the box's edge keeps its full slope, but an upright robot stays well above
    -CLAMP (-1.75 with the waist at the far stop from its goal) while a large
    violation still scores worse than a small one. A FLAT tail (L tanh(l/L),
    the second run) removed that last part: with every far-off waist angle
    scoring the same, the policy parked both waist joints at one stop whatever
    the goal.

WHY THE WAIST IS STIFFER. The upper body is ~15 kg of the robot's 33; with the
pelvis level, gravity's torque at waist roll and pitch grows ~20 Nm per rad of
lean, which cancelled ~70% of mjlab's stock waist stiffness (28.5 Nm/rad): a
target of +-0.17 rad already put roll on its +-0.52 stop, so ~80% of the action
range only pressed a joint into its stop. The waist roll/pitch PD here is
``WAIST_GAIN_MULT`` x stock (kp and kd both, which keeps the damping ratio on the
real torso inertia about what it was), and their action scale is
``WAIST_ACTION_SCALE`` rad per unit, so +-3 spans the targets that hold the whole
joint range. Applied to both tasks: the reach and stay twins share one robot.

The arms swing at random (the g1_stabilize_arm_random generator) at a lower
magnitude than the arm-random stands: see ``tasks/g1_waist.py``.
"""

from __future__ import annotations

import dataclasses

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
  CLAMP, _g_fall, g1_stabilize_env_cfg)
from robot_safety_sandbox.envs.g1_waist.command import (
  WAIST_JOINTS, WaistTargetCommandCfg)
from robot_safety_sandbox.envs.g1_waist.stance import reset_leg_stance
from robot_safety_sandbox.envs.g1_waist.stay_value import StayValue, stay_value
from robot_safety_sandbox.margins import avoid_only

#: The waist goal's command term (and observation term) name.
COMMAND = "waist_target"
#: Goal tolerance on each waist joint, rad.
Q_TOL = 0.1
#: Goal tolerance on each body-frame velocity component, m/s (vx, vy) and
#: rad/s (wz).
V_TOL = 0.1
#: Actor-observation noise on the base linear velocity, m/s (half-width). Half
#: the goal's velocity tolerance: an estimator that good would resolve its edge.
LIN_VEL_NOISE = 0.05
#: Stay spawn defaults: waist offset from the goal (rad) and base velocity
#: (m/s and rad/s), uniform half-widths. Half the tolerances: at the tolerance
#: itself a spawn lands on the box's edge and leaves it on the first step.
SPAWN_WAIST_NOISE = 0.05
SPAWN_VEL_NOISE = 0.05
#: Stay spawn default: the leg-stance distribution's scale (``stance.py``;
#: 0 = every episode on the stock knees-bent stance).
SPAWN_STANCE_SCALE = 1.0
#: Scale of the stay constraint's compressed tail (see the module docstring):
#: l_box below zero becomes L_TAIL * asinh(l_box / L_TAIL).
L_TAIL = 0.5
#: Waist roll/pitch PD gains as a multiple of mjlab's G1 defaults, and their
#: action scale in rad per unit action (stock: 0.4386). See the module docstring.
WAIST_GAIN_MULT = 3.0
WAIST_ACTION_SCALE = 0.2


def target_margin(env, q_tol: float = Q_TOL, v_tol: float = V_TOL) -> torch.Tensor:
  """l_box: >= 0 iff the waist and the body-frame velocity are both within
  tolerance of the goal. Unclamped."""
  d = env.scene["robot"].data
  term = env.command_manager.get_term(COMMAND)
  q = d.joint_pos[:, term.joint_ids]
  q_err = (q - term.command).abs()
  v = torch.cat([d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]],
                dim=1)
  v_err = (v - env.command_manager.get_command("twist")[:, :3]).abs()
  return torch.cat([(q_tol - q_err) / q_tol, (v_tol - v_err) / v_tol],
                   dim=1).amin(dim=1)


def soft_tail(l: torch.Tensor, scale: float = L_TAIL) -> torch.Tensor:
  """l where l >= 0; scale * asinh(l / scale) below. Same sign, value, slope
  and curvature at 0; below, still strictly increasing but only logarithmically
  (slope 0.12 at l = -4), so a large violation stays well above -CLAMP."""
  return torch.where(l >= 0, l, scale * torch.asinh(l / scale))


def _stay_constraint(env):
  """(h, l_box) with h = min(g, soft_tail(l_box)): safe AND inside the goal
  set."""
  l_box = target_margin(env)
  h = torch.minimum(_g_fall(env), soft_tail(l_box))
  return h.clamp(-CLAMP, CLAMP), l_box.clamp(-CLAMP, CLAMP)


#: The STAY margins: (min(g, soft_tail(l_box)), 0), avoid-only. A failure
#: termination is valued at -CLAMP, the bottom of h's range.
stay_margins = avoid_only(_stay_constraint)
stay_margins.terminal_margin = -CLAMP


def reach_margins(env):
  """The REACH margins: (g, min(l_box, V_stay))."""
  l = torch.minimum(target_margin(env), stay_value(env))
  return _g_fall(env).clamp(-CLAMP, CLAMP), l.clamp(-CLAMP, CLAMP)


def reset_waist_near_target(env, env_ids, noise: float = SPAWN_WAIST_NOISE,
                            command_name: str = COMMAND) -> None:
  """Reset event (after the stock joint reset): draw this episode's goal now
  and place the waist within ``noise`` rad of it, inside the soft limits."""
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  term = env.command_manager.get_term(command_name)
  term.sample(env_ids)
  robot = env.scene["robot"]
  jid = term.joint_ids
  off = (2.0 * torch.rand(len(env_ids), len(jid), device=env.device) - 1.0)
  q = term.command[env_ids] + float(noise) * off
  lim = robot.data.soft_joint_pos_limits[env_ids][:, jid]
  q = torch.clamp(q, lim[..., 0], lim[..., 1])
  robot.write_joint_position_to_sim(q, joint_ids=jid, env_ids=env_ids)


def _stiffen_waist(cfg, gain_mult: float, action_scale: float) -> None:
  """Waist roll/pitch: PD gains x ``gain_mult``, action scale ``action_scale``.
  The articulation and the scale dict are mjlab module-level objects shared by
  every G1 task, so both are REPLACED here, never mutated."""
  robot = cfg.scene.entities["robot"]
  acts, hit = [], 0
  for a in robot.articulation.actuators:
    if set(a.target_names_expr) == set(WAIST_JOINTS):
      a = dataclasses.replace(a, stiffness=a.stiffness * gain_mult,
                              damping=a.damping * gain_mult)
      hit += 1
    acts.append(a)
  if hit != 1:
    raise RuntimeError("expected one G1 actuator cfg for exactly the waist roll "
                       f"and pitch joints, found {hit}")
  robot.articulation = dataclasses.replace(robot.articulation,
                                           actuators=tuple(acts))
  action = cfg.actions["joint_pos"]
  action.scale = {**action.scale, **{j: float(action_scale) for j in WAIST_JOINTS}}


def _waist_env_cfg(play: bool, episode_length_s: float | None) -> ManagerBasedRlEnvCfg:
  """g1_stabilize (flat, twist pinned to zero) + the waist goal command and the
  two observation terms the goal needs. Both appended at the END of the actor
  group, after g1_stabilize's terms, so its column order is unchanged."""
  cfg = g1_stabilize_env_cfg(play=play)
  _stiffen_waist(cfg, WAIST_GAIN_MULT, WAIST_ACTION_SCALE)
  if episode_length_s is not None and not play:
    cfg.episode_length_s = float(episode_length_s)
  cfg.commands[COMMAND] = WaistTargetCommandCfg()
  actor = cfg.observations["actor"].terms
  actor["base_lin_vel"] = ObservationTermCfg(
    func=envs_mdp.base_lin_vel,
    noise=Unoise(n_min=-LIN_VEL_NOISE, n_max=LIN_VEL_NOISE))
  for group in ("actor", "critic"):
    cfg.observations[group].terms[COMMAND] = ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": COMMAND})
  return cfg


def g1_waist_stay_env_cfg(play: bool = False,
                          spawn_waist_noise: float = SPAWN_WAIST_NOISE,
                          spawn_vel_noise: float = SPAWN_VEL_NOISE,
                          spawn_stance_scale: float = SPAWN_STANCE_SCALE,
                          episode_length_s: float | None = None
                          ) -> ManagerBasedRlEnvCfg:
  """The STAY task's cfg. Keyword arguments can be set from a run config's
  ``env_overrides``."""
  cfg = _waist_env_cfg(play, episode_length_s)
  v = float(spawn_vel_noise)
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-v, v), "y": (-v, v), "yaw": (-v, v)}
  if float(spawn_stance_scale) > 0.0:
    # After reset_base (it keeps that event's position, heading and twist).
    cfg.events["reset_leg_stance"] = EventTermCfg(
      func=reset_leg_stance, mode="reset",
      params={"scale": float(spawn_stance_scale)})
  cfg.events["reset_waist_near_target"] = EventTermCfg(
    func=reset_waist_near_target, mode="reset",
    params={"noise": float(spawn_waist_noise)})
  return cfg


def g1_waist_reach_env_cfg(play: bool = False, stay_run: str | None = None,
                           episode_length_s: float | None = None
                           ) -> ManagerBasedRlEnvCfg:
  """The REACH task's cfg. ``stay_run``: the trained stay run dir or model zip
  (None -> ``$SAFE_ZOO_G1_WAIST_STAY_RUN``, else
  ``runs/g1_waist_stay_arm_random``)."""
  cfg = _waist_env_cfg(play, episode_length_s)
  cfg.events["stay_value"] = EventTermCfg(
    func=StayValue, mode="startup", params={"run": stay_run})
  return cfg
