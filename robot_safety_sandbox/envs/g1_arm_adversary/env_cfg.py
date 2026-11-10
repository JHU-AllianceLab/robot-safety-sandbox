"""Unitree G1 flat-ground stabilization / locomotion vs an ARM adversary — the
next robustification step past ``g1_stabilize``, where the adversary was a
worst-case wrench on the torso.

Here the adversary is not an external force at all: it OWNS THE ARMS. The 29
actuated joints are split between the two players,

    action dims  0..11   legs   (12)  \\
    action dims 12..14   waist  (3)    > CONTROL — the actor
    action dims 15..28   arms   (14)  -> DISTURBANCE — the adversary

so the boundary experience the value function sees comes from a limb the robot
actually has, driven adversarially, rather than from a disembodied push. A G1's
arms are ~10 kg of the ~35 kg body at the end of a long lever: swung worst-case
they are a far more expressive attack than a 3-dim wrench, and one the deployed
robot can genuinely suffer (a mistrained or hijacked arm controller, a carried
load that swings, an arm policy trained for a different task).

THE ACTOR KEEPS A 29-DIM HEAD. The bridge (``dstb_mode="joint"``) overwrites the
actor's 14 arm outputs with the adversary's before the action reaches mjlab, so
the actor's arm outputs are simply discarded. The alternative — a 15-dim actor —
is tidier on paper but throws away the action head and log_std of an already
trained ``g1_stabilize`` adversarial policy, and warm-starting from that policy
is the entire point of this task. Note the actor still OBSERVES the applied arm
targets: mjlab's ``last_action`` observation term reads the 29-dim vector the
bridge submitted, which is the adversary's arm command, not the actor's discarded
one. The actor therefore sees what is about to be done to it.

WHY THE ADVERSARY IS PENALIZED FOR SELF-COLLISION
-------------------------------------------------
An arm adversary with a free hand wins trivially and uselessly: it drives the
hands through the torso and the thighs, where MuJoCo's contact solver hands it
impulses no physical arm could produce, and the actor learns to survive an attack
that cannot happen. The adversary must therefore be made to attack with arm
motions the robot could actually execute — which means it pays for hitting
itself.

This is NOT a term in ``g``. The two-player learners are zero-sum (``ppo_2p``
negates the disturbance advantage on the SAME targets; ``sac_2p`` shares one
joint-action critic), so anything folded into ``g`` or into the shared ``l`` is
something the MINIMIZING player wants MORE of — a self-collision term in ``g``
would pay the adversary to smash its own arms into the torso, the exact opposite
of the intent.

The penalty is delivered instead as a DISTURBANCE-PHASE reach margin. The bridge
is told which player is currently training (``callbacks.DstbPhaseCallback`` reads
the learner's own ``_phase()`` at each rollout start), and while the disturbance
player is collecting its rollout the margin function returns

    l_dstb = l + frac * (CLAMP - l),   frac = min(arm self-contact force / F, 1)

i.e. ``l`` interpolated toward the ceiling ``+CLAMP`` as the arms load up against
the body. The disturbance MINIMIZES ``min(g, max(l, gamma V'))``, so a spike in
``l`` is a spike in the value it is trying to push down: it learns to keep the
arms clear. At zero self-contact ``frac = 0`` and ``l_dstb == l`` exactly, so the
control player's problem — and every 1P run of this task — is bit-identical to
``g1_stabilize``'s.

``g`` IS NEVER SHAPED. It stays the shared physical safety margin, which is what
keeps the failure counters, the ``_ever_gneg`` league bookkeeping and the
certificate's meaning intact.

Known interaction: with ``end_criterion="reach-avoid"`` a shaped ``l`` would end
the episode on the adversary's self-collision, which is nonsense — so the
shaping is skipped outright in that mode (the bridge stamps the criterion onto
the mjlab env as ``_zoo_end_criterion``). Under the default ``"failure"``
criterion the shaping never touches a termination.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from robot_safety_sandbox.envs.g1_stabilize.env_cfg import (
  CLAMP,
  DEFAULT_CMD_VX,
  _pin_twist,
  locomote_margins,
  stance_margins,
)
from robot_safety_sandbox.envs.velocity.g1 import unitree_g1_flat_env_cfg

# --- the joint split ----------------------------------------------------------
# The G1's actuated joints come out of the model in one contiguous order (the
# floating base is joint 0 and is not actuated):
#     0..11  left/right hip pitch,roll,yaw + knee + ankle pitch,roll
#    12..14  waist yaw, roll, pitch
#    15..28  left/right shoulder pitch,roll,yaw + elbow + wrist roll,pitch,yaw
# so "the arms" is a slice, not a lookup. ``tests/test_g1_arm_adversary.py``
# pins this against the compiled model so a mjlab asset reshuffle fails loudly
# instead of handing the adversary somebody's knee.
N_ACTUATED = 29
ARM_SLICE = slice(15, 29)
#: number of arm joints = the disturbance action dimension
ARM_DIM = ARM_SLICE.stop - ARM_SLICE.start   # 14

#: The arm collision geoms — the only ones on the four arm links that carry any
#: (shoulder_pitch_link and shoulder_roll_link have no collision geometry).
ARM_GEOM_PATTERN = r"^(left|right)_(shoulder_yaw|elbow_yaw|wrist|hand)_collision$"
ARM_GEOMS = tuple(
  f"{side}_{part}_collision"
  for side in ("left", "right")
  for part in ("shoulder_yaw", "elbow_yaw", "wrist", "hand")
)
#: The OTHER side of the contact: the pelvis SUBTREE, which on the G1 is the
#: whole robot. It has to be a subtree rather than a geom regex — mjlab's
#: ``secondary_policy`` collapses a multi-match secondary pattern to ONE name
#: ("first") or drops the filter entirely ("any"), and neither is what this
#: sensor wants: the first would watch the arms against ``pelvis_collision``
#: alone (it reads zero forever, which looks exactly like a well-behaved
#: adversary), and no filter at all would also catch arm-versus-TERRAIN, i.e. it
#: would charge the adversary for the robot falling — the outcome it is being
#: trained to cause. A subtree secondary is passed through whole.
ROBOT_SUBTREE = "pelvis"

#: Arm self-contact force, in newtons, at which the disturbance pays the FULL
#: penalty. The same 10 N the fall margin and the ``illegal_contact``
#: termination use, so "a real contact" means one thing across this task.
ARM_CONTACT_THRESHOLD = 10.0

#: Sensor name — read by :func:`_arm_selfcol_frac`, written by
#: :func:`_add_arm_selfcol_sensor`.
ARM_SELFCOL_SENSOR = "arm_self_collision"


def _add_arm_selfcol_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the arm-vs-body contact sensor the disturbance penalty reads.

  Distinct from the cfg's existing ``self_collision`` sensor, which is the whole
  pelvis subtree against itself: that one also fires on thigh-against-thigh,
  which is the ACTOR's doing and must not be charged to the adversary. This one
  is ARM GEOMS (primary) against the robot subtree (secondary) — arm vs torso,
  arm vs thigh, hand vs hand — and nothing that does not involve an arm.

  ``history_length=4`` matches the other contact sensors here: a policy step is
  several physics substeps and a swing can clip the body inside one of them, so
  the window is what catches it. ``reduce="none"`` keeps it cheap — the penalty
  wants the PEAK over the window, which :func:`_arm_selfcol_frac` takes itself.
  """
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=ARM_SELFCOL_SENSOR,
      primary=ContactMatch(mode="geom", entity="robot",
                           pattern=ARM_GEOM_PATTERN),
      secondary=ContactMatch(mode="subtree", entity="robot",
                             pattern=ROBOT_SUBTREE),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )
  # The flat cfg budgets 128 contact-sensor matches for nonfoot_ground_touch's
  # 19 geoms against the terrain. Eight more primaries against every robot
  # collision geom does not fit in what is left; an overflowing sensor silently
  # drops matches, which here would read as "the adversary is behaving".
  cfg.sim.contact_sensor_maxmatch = 256


# --- the disturbance-phase penalty -------------------------------------------

def _arm_selfcol_frac(env) -> torch.Tensor | None:
  """Peak arm self-contact force this step as a fraction of
  :data:`ARM_CONTACT_THRESHOLD`, saturated at 1. ``None`` when the cfg carries
  no such sensor, so a variant that drops it stays trainable (the penalty then
  simply never fires — same graceful-degradation contract as the fall margin's
  contact term)."""
  try:
    sensor = env.scene[ARM_SELFCOL_SENSOR]
  except (KeyError, AttributeError):
    return None
  hist = sensor.data.force_history
  if getattr(env, "_zoo_instantaneous_contact", False):
    # Rollout shadow sims that could not round-trip the force history set this
    # flag (filters/rollout.py); a half-filled window under-reports the peak.
    hist = None
  force = hist if hist is not None else sensor.data.force
  if force is None:
    return None
  mag = torch.norm(force, dim=-1)
  while mag.dim() > 1:
    mag = mag.amax(dim=-1)
  return (mag / ARM_CONTACT_THRESHOLD).clamp(0.0, 1.0)


def _shape_l_for_dstb(env, l: torch.Tensor) -> torch.Tensor:
  """``l`` as the DISTURBANCE player sees it: interpolated toward ``+CLAMP`` in
  proportion to how hard the arms are loading against the body.

  Returns ``l`` unchanged — same tensor values, bit for bit — whenever the
  control player is the one training, the env is in ``end_criterion="reach-avoid"``
  mode (where a shaped ``l`` would end episodes), or the sensor is absent. That
  is what makes a 1P run of this task identical to ``g1_stabilize``.

  The interpolation ``l + frac * (CLAMP - l)`` is monotone in ``frac`` and lands
  exactly on ``l`` at ``frac = 0`` and on ``CLAMP`` at ``frac = 1``; because it
  is a lerp toward the ceiling rather than a ``max``, a disturbance that is
  winning (``l`` deeply negative, the robot far from its target) still sees a
  gradient out of a grazing contact instead of a flat penalty plateau.
  """
  if not getattr(env, "_zoo_dstb_phase", False):
    return l
  if getattr(env, "_zoo_end_criterion", "failure") == "reach-avoid":
    return l
  frac = _arm_selfcol_frac(env)
  if frac is None:
    return l
  return l + frac * (CLAMP - l)


def arm_stance_margins(env):
  """(g, l) for the stabilize task: ``g1_stabilize``'s stance margins, with the
  disturbance-phase self-collision penalty folded into ``l``. ``g`` is untouched."""
  g, l = stance_margins(env)
  return g, _shape_l_for_dstb(env, l)


def arm_locomote_margins(env):
  """(g, l) for the locomotion task: ``g1_stabilize``'s command-tracking margins,
  with the disturbance-phase self-collision penalty folded into ``l``."""
  g, l = locomote_margins(env)
  return g, _shape_l_for_dstb(env, l)


# --- cfg builders -------------------------------------------------------------

def g1_arm_adv_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat ground, zero command, arm adversary: hold a stable stand while the
  arms are driven against you."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  _add_arm_selfcol_sensor(cfg)
  _pin_twist(cfg, 0.0)
  return cfg


def g1_arm_adv_locomote_env_cfg(play: bool = False,
                                cmd_vx: float = DEFAULT_CMD_VX
                                ) -> ManagerBasedRlEnvCfg:
  """Flat ground, constant forward command, arm adversary: keep walking at the
  command while the arms are driven against you."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  _add_arm_selfcol_sensor(cfg)
  _pin_twist(cfg, cmd_vx)
  return cfg
