"""Unitree G1 flat-ground stabilization / locomotion vs an adversarial force —
the HUMANOID twin of ``go2_stabilize``, and like it a "no-machinery" task:

  * flat terrain, stock velocity-task spawns — no custom spawn distribution
  * NO curriculum of any kind
  * no staged pipeline — trainable from scratch in one run
  * margins are a handful of body-frame inequalities

Boundary experience comes from the ADVERSARY (a worst-case wrench on the torso,
which replaces the random push the bridge drops), not from a widened spawn
distribution. See ``PORTING.md`` — this task is the first row of that table.

Tasks:
  stabilize: zero command; g = don't fall (pelvis height, tilt, non-foot ground
             contact); l = stance band (pelvis in [0.55, 0.85], nearly upright,
             |v| and |w| small)
             -> V > 0 == "can return to a stable stand despite the adversary"
  locomote:  constant forward command; same g; l = tracking that command
             -> V > 0 == "can keep walking at the command despite the adversary"

WHY THE MARGINS DIFFER FROM THE GO2's. The quadruped's g is the minimum height
of the eight trunk corners: a Go2 trunk is a box close to the ground and a fall
IS the box touching it. A G1 is 1.3 m of articulated linkage whose pelvis stays
high in poses that are already a fall (a kneel, a hand plant), so height alone
does not decide it. Hence the three-term g below — the contact term is the one
that actually catches a kneel, and it is read from the SAME sensor as the env's
``illegal_contact`` termination, so g < 0 whenever the episode ends on a fall.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg

from robot_safety_sandbox.envs.velocity.g1 import unitree_g1_flat_env_cfg

# --- g (safety): don't fall ---------------------------------------------------
# Pelvis height. Nominal stance (the knees-bent keyframe) is 0.76 m; a fall puts
# it under ~0.35 m. 0.45 leaves room for a deep squat, and the 0.25 m scale makes
# the nominal stance g_height ~ 1.2, i.e. O(1) like every other margin here.
Z_MIN = 0.45
Z_SCALE = 0.25
# Torso tilt, as the cosine of the angle from vertical: cos(50 deg). The env's
# own ``fell_over`` termination fires at 70 deg, so g crosses zero BEFORE the
# episode ends — the learner sees the boundary, not just the aftermath.
TILT_COS = math.cos(math.radians(50.0))
TILT_NORM = 1.0 - TILT_COS
# Non-foot ground contact, in newtons: the same threshold the env's
# ``illegal_contact`` termination uses, so the two agree on what a fall is.
CONTACT_FORCE_THRESHOLD = 10.0
# Both margins are clamped to +-CLAMP before they leave this module, the way
# ``margins.compose`` clamps the shared library's terms. The contact term is the
# reason it is not optional here: a hard body slam reads several hundred newtons,
# which would hand the value regression a reward of -150 for one step and drown
# every other transition in the batch.
CLAMP = 3.0

# --- l (reach): the stance band ----------------------------------------------
# EVERY l term below is divided by its own tolerance, so each reads "fraction of
# the way to the edge of the target set" and the min over them is not decided by
# whichever term happens to be measured in the smallest unit. (The Go2's stance
# margin leaves its terms in raw metres and m/s; it gets away with it because a
# Go2 trunk sits in a 0.30 m band. A standing G1 pelvis has ~0.09 m of headroom
# under its ceiling, so the raw height slack would swamp every other term.)
PELVIS_MID = 0.70        # band centre -> the band is [0.55, 0.85] m
PELVIS_BAND = 0.15
# cos(25 deg): "roughly upright", loose enough that the target is reachable from
# a recovery step rather than only from a statue-perfect pose.
STANCE_COS = math.cos(math.radians(25.0))
LIN_TOL = 0.25
ANG_TOL = 0.30

# --- l (reach): command tracking ---------------------------------------------
LOCO_VEL_TOL = 0.5

#: Default forward command for the locomotion task (m/s). A comfortable G1 walk;
#: mjlab's own flat G1 play config sweeps up to 1.5-2.0 m/s.
DEFAULT_CMD_VX = 1.0


def _nonfoot_contact_margin(env) -> torch.Tensor | None:
  """(threshold - peak non-foot ground force) / threshold, or None if the task
  cfg carries no such sensor (a variant that dropped it stays trainable)."""
  try:
    sensor = env.scene["nonfoot_ground_touch"]
  except (KeyError, AttributeError):
    return None
  hist = sensor.data.force_history
  if getattr(env, "_zoo_instantaneous_contact", False):
    # Rollout shadow sims that could not round-trip the contact-force history
    # set this flag (filters/rollout.py): a half-filled window under-reports the
    # peak force and would hand back a bogus margin. Read the instant instead.
    hist = None
  force = hist if hist is not None else sensor.data.force
  if force is None:
    return None
  mag = torch.norm(force, dim=-1)
  while mag.dim() > 1:
    mag = mag.amax(dim=-1)
  return (CONTACT_FORCE_THRESHOLD - mag) / CONTACT_FORCE_THRESHOLD


def _g_fall(env) -> torch.Tensor:
  """g: pelvis height, torso tilt, and non-foot ground contact — the humanoid
  failure margin. g < 0 == fallen."""
  d = env.scene["robot"].data
  base_z = d.root_link_pos_w[:, 2]
  up = -d.projected_gravity_b[:, 2]        # cos(angle from vertical)
  terms = [(base_z - Z_MIN) / Z_SCALE, (up - TILT_COS) / TILT_NORM]
  contact = _nonfoot_contact_margin(env)
  if contact is not None:
    terms.append(contact)
  return torch.stack(terms, dim=0).amin(dim=0)


def stance_margins(env):
  """(g, l): fall safety + stabilize-to-stance target.

  l >= 0 == standing: pelvis inside the height band, roughly upright, and both
  velocities small. A G1 spawned on the knees-bent keyframe already satisfies it
  (as a Go2 does in ``go2_stabilize``), so the reach-avoid value here answers
  "can it hold / RETURN to that stance despite the adversary", not "can it ever
  touch it" — which is why the task keeps the default ``end_criterion="failure"``
  and never ends an episode on reach.
  """
  d = env.scene["robot"].data
  base_z = d.root_link_pos_w[:, 2]
  up = -d.projected_gravity_b[:, 2]
  v_b, w_b = d.root_link_lin_vel_b, d.root_link_ang_vel_b
  l = torch.stack([
    (PELVIS_BAND - (base_z - PELVIS_MID).abs()) / PELVIS_BAND,
    (up - STANCE_COS) / (1.0 - STANCE_COS),
    (LIN_TOL - v_b[:, 0].abs()) / LIN_TOL,
    (LIN_TOL - v_b[:, 1].abs()) / LIN_TOL,
    (LIN_TOL - v_b[:, 2].abs()) / LIN_TOL,
    (ANG_TOL - w_b[:, 0].abs()) / ANG_TOL,
    (ANG_TOL - w_b[:, 1].abs()) / ANG_TOL,
    (ANG_TOL - w_b[:, 2].abs()) / ANG_TOL,
  ], dim=0).amin(dim=0)
  return _g_fall(env).clamp(-CLAMP, CLAMP), l.clamp(-CLAMP, CLAMP)


def locomote_margins(env):
  """(g, l): fall safety + velocity-command-tracking target.

  l >= 0 == walking at the command (planar body-frame velocity within
  ``LOCO_VEL_TOL`` of it), normalized by that tolerance so l = 1 at perfect
  tracking — the same O(1) scale as g, which matters because the reach-avoid
  backup compares the two directly in ``min(g, max(l, gamma V'))``.
  """
  d = env.scene["robot"].data
  v_b = d.root_link_lin_vel_b[:, :2]
  cmd = env.command_manager.get_command("twist")[:, :2]
  err = torch.linalg.norm(v_b - cmd, dim=1)
  l = (LOCO_VEL_TOL - err) / LOCO_VEL_TOL
  return _g_fall(env).clamp(-CLAMP, CLAMP), l.clamp(-CLAMP, CLAMP)


def _pin_twist(cfg, vx: float) -> None:
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False
    if getattr(twist.ranges, "heading", None) is not None:
      twist.ranges.heading = None
  # The inherited "command_vel" curriculum term (the generic walker cfg) mutates
  # these SAME ranges in place on the first env reset after step 0, silently
  # reopening them to the full locomotion range regardless of the pin above.
  # Remove it here so the pin actually holds for the run.
  cfg.curriculum.pop("command_vel", None)


def g1_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _pin_twist(cfg, 0.0)     # zero command: the target is a stable stand
  return cfg


def g1_locomote_env_cfg(play: bool = False,
                        cmd_vx: float = DEFAULT_CMD_VX) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _pin_twist(cfg, cmd_vx)  # constant forward drive: the target is tracking it
  return cfg
