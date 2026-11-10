"""Ill-posed reference motions for SONIC: the bad-reference suite.

    factory = BadReferenceFactory(ref, joint_low, joint_high)   # SONIC order
    bad = factory.build(clip, frame0, onset, length, "speedup_5x", generator)
    tracker = SonicTracker(env, policy, bad)                    # or tracker.ref = bad
    tracker.start(mask, clip=episode_ids, frame=zeros)          # clip e = episode e

``SonicTracker.start`` and ``act`` look up EVERY env's clip id in the
tracker's reference, the envs outside ``mask`` included. So when
``tracker.ref`` is swapped to a corrupted reference, every env's clip id must
be valid in it (one reference covering all envs' episodes, or the other envs'
ids reset first).

The push suite asks whether a safety filter saves SONIC from a push. This
module serves the other question of ``examples/eval_sonic_qcbf.py``. SONIC is
a tracker, and a tracker does what its reference says. When the reference is
wrong, can the Q-CBF filter (SONIC as the nominal, a SafetyPPO / SHAC
fallback's Q and V as the certificate) refuse to follow it? Each
:data:`SCENARIOS` entry turns a clean LAFAN episode into a reference that is
KINEMATICALLY valid (every frame a pose inside the joint limits, unit
quaternions) but DYNAMICALLY bad: too fast, too shaky, discontinuous, or
heading for a pose no controller can hold. Each scenario's ``doc`` says what
makes it ill-posed.

WHAT SONIC READS. Its encoder sees the reference's joint positions, joint
velocities and the root orientation relative to the robot (``anchor_6d`` of
``dq * root_quat``, where ``dq`` is the yaw offset fixed at
``SonicTracker.start``). It never sees the root POSITION. So every
corruption here lives in those three channels: joint_pos (oscillation, noise,
poses, mirror, discontinuities), joint_vel (always the forward difference of
joint_pos, see below) and root_quat (lean, yaw, the root of a pose or of a
teleport target). A corrupted root position would be invisible to SONIC.

THE CONTRACT WITH THE TRACKER. :meth:`BadReferenceFactory.build` returns a
new ``SonicReference`` with one clip per episode, ``length`` frames each.
Clip e frame f is the base clip's frame ``frame0[e] + f`` for
f < ``onset[e]``, bit for bit. So frame 0 is the spawn row:
``SonicTracker.start(mask, e, 0)`` fixes its delta heading from frame 0's yaw
and fills its histories from the robot, exactly as if it tracked the base
clip from ``frame0[e]``. Frame ``onset[e]`` is the first frame the corruption
changes (``onset >= 1``). SONIC's encoder reads frames t + 1 + 5k, k < 10, at
the robot's nominal frame t. It therefore SEES the corruption from
t = onset - :func:`look_ahead` (46 frames, 0.92 s before the robot's nominal
frame reaches it), and a pre / post-onset split of the telemetry has to
allow for that. A clip must be long enough for that window at the last
tracked step (:func:`episode_length`). Past its end, ``SonicReference.rows``
holds the last frame.

VELOCITIES. ``joint_vel`` is recomputed from the corrupted positions with
SONIC's own ``forward_difference``, per episode clip, as SONIC's motion lib
would compute it for that clip. A position jump therefore shows up as a
one-frame velocity spike, and a sped-up clip as k-fold velocities: what the
encoder reads is self-consistent. One consequence: the velocity at frame
onset - 1 is the difference INTO the first corrupted frame, so it already
differs from the base. Positions and orientations before the onset do not.

AFTER THE WINDOW. A scenario corrupts ``duration`` frames from the onset
(the build's ``duration``, 1.5 s by default, unless the scenario fixes its
own) and then does one of three things:

    return     the reference is the time-aligned original again (a burst)
    hold       the corruption stays to the end of the clip
    continue   index scenarios: the clip plays on at 1x from the frame the
               corruption reached, with no second jump

:class:`Corruption` (``ref.corruption``) records per episode what was done:
the corrupted frames, the base rows of the TRUE (time-aligned) reference for
tracking-error read-outs, the source index of every frame, the pelvis tilt
the corrupted reference asks for and the random draws.

LIMITS. In a corrupted frame, every joint the family EDITED is clamped to
the joint limits the driver passes (the env's soft limits, SONIC order),
widened to the source frame's own value. A third of the LAFAN frames have
some joint (ankle roll, waist pitch) past the 0.9 soft limits, though every
one lies inside the hard MJCF ranges. So a joint the family leaves alone (a
lean's or a yaw's joints, hf_osc's arms, every frame an index family reads
from the base) stays the base's own bit for bit, and an edit never steps at
its onset. Frames outside the corruption are the base's own, unclamped, so
the pre-onset identity is exact.

RANDOMNESS. Every random draw comes from the ``generator`` passed to
``build``, drawn on that generator's device. The same seed therefore gives
the same corruption for every filter condition (common random numbers). A
random scenario without a generator raises. Draw shapes do not depend on the
onset or the duration, so changing those does not shift the stream.

Pure torch. ``sonic.py`` (SONIC_JOINTS, forward_difference, SonicReference)
is imported on first use, because it imports mjlab, and mjlab imports
mujoco_warp. ``tests/test_bad_reference.py`` checks the contract on synthetic
clips: pre-onset identity, lengths, limits, unit quaternions, velocities, and
that each family changes what it claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

#: SONIC's reference rate: ``SonicReference.build`` resamples every clip to
#: the 50 Hz control rate, and its velocities are differences at this rate.
FPS = 50.0
#: The corruption window when neither the build nor the scenario sets one.
DEFAULT_DURATION_S = 1.5
#: Blend-in time of the lean and pose scenarios (cosine ramp).
BLEND_S = 0.3
#: Ramp in and out of the oscillation bursts: a short taper, so that the
#: oscillation's spectrum is not dominated by a step at the onset.
OSC_RAMP_S = 0.1

RETURN, HOLD, CONTINUE = "return", "hold", "continue"
AFTER = (RETURN, HOLD, CONTINUE)

_sonic_module = None


def _sonic():
  """``envs/g1_recover/sonic.py``, imported on first use. Its top-level
  mjlab import pulls in mujoco_warp, which takes minutes on a login node, and
  this module's own import must not."""
  global _sonic_module
  if _sonic_module is None:
    from robot_safety_sandbox.envs.g1_recover import sonic
    _sonic_module = sonic
  return _sonic_module


def look_ahead() -> int:
  """How many frames past the robot's nominal frame SONIC's encoder reads
  (``REF_LEAD + (NUM_FUTURE - 1) * FUTURE_STEP`` = 46)."""
  s = _sonic()
  return int(s.REF_LEAD + (s.NUM_FUTURE - 1) * s.FUTURE_STEP)


def episode_length(onset, window: int) -> int:
  """Frames a corrupted clip needs so that SONIC's window never runs past its
  end, when every episode is tracked for ``window`` steps after its onset:
  the last step's nominal frame is onset + window - 1, and the encoder reads
  up to :func:`look_ahead` frames past it."""
  return int(torch.as_tensor(onset).max()) + int(window) + look_ahead()


# --- quaternions (w-first, unit) -----------------------------------------------

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  """The inverse of a unit quaternion."""
  return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quat_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Rotation by ``angle`` ``[...]`` about the unit ``axis`` ``[..., 3]``."""
  h = 0.5 * angle
  return torch.cat([torch.cos(h).unsqueeze(-1), axis * torch.sin(h).unsqueeze(-1)], -1)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
  """The heading (z-y-x yaw) of ``q``, rad, as mjlab's ``yaw_quat`` reads it."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  """The pure-yaw part of ``q``."""
  return z_quat(yaw_of(q))


def z_quat(psi: torch.Tensor) -> torch.Tensor:
  """Rotation by ``psi`` about the world z axis."""
  h = 0.5 * psi
  zero = torch.zeros_like(h)
  return torch.stack([torch.cos(h), zero, zero, torch.sin(h)], -1)


def tilt_of(q: torch.Tensor) -> torch.Tensor:
  """The angle between the body's z axis and the world's, rad."""
  _, x, y, _ = q.unbind(-1)
  return torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Shortest-arc interpolation, ``t`` ``[..., 1]`` in [0, 1]."""
  d = (q0 * q1).sum(-1, keepdim=True)
  q1 = torch.where(d < 0, -q1, q1)
  d = d.abs().clamp(max=1.0)
  theta = torch.acos(d)
  s = torch.sin(theta)
  near = s < 1e-6
  s = s.clamp_min(1e-6)
  w0 = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * theta) / s)
  w1 = torch.where(near, t, torch.sin(t * theta) / s)
  return F.normalize(w0 * q0 + w1 * q1, dim=-1)


# --- left / right --------------------------------------------------------------

def _other_side(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_"):]
  if name.startswith("right_"):
    return "left_" + name[len("right_"):]
  return name


def mirror_map(names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
  """``(perm, sign)`` of the sagittal mirror of a joint vector in ``names``
  order: ``mirrored[j] = sign[j] * q[perm[j]]``.

  Derived from the names alone. ``left_X`` and ``right_X`` swap, and
  centre joints (the waist) map to themselves. Roll and yaw joints change
  sign and pitch joints (knee and elbow included) keep it. That holds
  because every G1 left/right pair shares its joint axis (the MJCF: pitch
  about y, roll about x, yaw about z on both sides), and the reflection
  y -> -y keeps a rotation about y and reverses rotations about x and z. The
  map is an involution, and the G1's roll ranges are mirror images
  (left hip roll [-0.52, 2.97], right [-2.97, 0.52]), so a pose inside the
  limits mirrors to one inside them."""
  idx = {n: i for i, n in enumerate(names)}
  perm, sign = [], []
  for n in names:
    m = _other_side(n)
    if m not in idx:
      raise ValueError(f"joint {n!r} has no mirror twin {m!r} in the joint list")
    perm.append(idx[m])
    toks = n.split("_")
    sign.append(-1.0 if ("roll" in toks or "yaw" in toks) else 1.0)
  return torch.tensor(perm, dtype=torch.long), torch.tensor(sign)


def mirror_joints(pos: torch.Tensor, names: Optional[Sequence[str]] = None) -> torch.Tensor:
  """The sagittal mirror of joint vectors ``[..., J]`` (:func:`mirror_map`;
  ``names`` defaults to SONIC_JOINTS)."""
  perm, sign = mirror_map(_sonic().SONIC_JOINTS if names is None else names)
  return pos[..., perm.to(pos.device)] * sign.to(pos)


def mirror_quat(q: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
  """The root attitude reflected across the vertical plane through heading
  ``yaw0`` (broadcast against ``q[..., 0]``). In that heading frame the
  reflection y -> -y negates the x and z rotation components, so roll and
  yaw relative to ``yaw0`` change sign and pitch is kept. An involution for a
  fixed ``yaw0``."""
  y0 = z_quat(yaw0)
  qh = quat_mul(quat_conj(y0), q) * q.new_tensor([1.0, -1.0, 1.0, -1.0])
  return quat_mul(y0, qh)


def _is_leg_or_waist(name: str) -> bool:
  toks = name.split("_")
  return any(t in toks for t in ("hip", "knee", "ankle", "waist"))


# --- poses ---------------------------------------------------------------------

def _pair(joint: str, left: float, right: Optional[float] = None) -> dict:
  return {f"left_{joint}_joint": left,
          f"right_{joint}_joint": left if right is None else right}


#: The pose's starting point for every joint a pose does not name: mjlab's
#: G1 KNEES_BENT_KEYFRAME (``asset_zoo/robots/unitree_g1/g1_constants.py``),
#: the default pose mjlab and SONIC's training G1 share. Rad.
NEUTRAL = {**_pair("hip_pitch", -0.312), **_pair("knee", 0.669),
           **_pair("ankle_pitch", -0.363), **_pair("elbow", 0.6),
           **_pair("shoulder_pitch", 0.2), **_pair("shoulder_roll", 0.2, -0.2)}


@dataclass(frozen=True)
class Pose:
  """A whole-body target: ``joints`` (SONIC joint name -> rad, the rest from
  :data:`NEUTRAL`), and the pelvis attitude in the heading frame,
  ``root_pitch`` (+ = forward, about the heading's y axis) then
  ``root_roll`` (+ = toward the robot's right, about its x axis). It
  replaces the clip's own tilt, and the heading stays the clip's. A ``sided``
  pose is written for the LEFT side. Each episode draws left or its mirror.

  Sign conventions (the G1 MJCF, every pitch joint about +y): a negative hip
  or shoulder pitch swings the thigh / arm FORWARD, a positive knee bends it,
  a positive ankle pitch points the toe down, and a positive waist pitch
  bends the trunk forward. The ELBOW is offset: the forearm runs along the
  elbow link's +x, so elbow 0 is a right angle (forearm horizontal ahead of a
  hanging upper arm), a positive elbow EXTENDS it (~1.5 is straight, the
  LAFAN T-pose has 1.2-1.5) and a negative one flexes it further. A positive
  left hip / shoulder roll lifts the limb out to the left (the right side's
  mirror is negative). The sole's pitch is the root pitch plus hip, knee and
  ankle pitch (the hip-roll and knee links' -10 / +10 deg offsets cancel), so
  a flat foot needs that sum at 0. The knee's soft limit is 0.061, so a
  "straight" knee here is 0.07 with the ankle at -0.07."""
  joints: dict
  root_pitch: float = 0.0
  root_roll: float = 0.0
  sided: bool = False
  doc: str = ""


POSES = {
  "knee_up": Pose(
    joints={**_pair("hip_pitch", -1.9, 0.0),      # left thigh ~20 deg above horizontal
            **_pair("knee", 2.2, 0.1),            # left shank folded under it; right straight
            **_pair("ankle_pitch", -0.2, -0.1),   # the stance sole flat (0 + 0.1 - 0.1)
            **_pair("hip_roll", 0.0, 0.0),        # no lateral weight shift
            **_pair("ankle_roll", 0.0, 0.0),
            "waist_pitch_joint": -0.2,            # trunk leaning back
            **_pair("shoulder_pitch", -2.6),      # both arms overhead
            **_pair("elbow", 1.5)},               # and straight
    sided=True,
    doc="single-leg stance, the other knee pulled high, arms overhead, trunk "
        "back: single support on a straight leg with zero hip and ankle roll, "
        "so the pose keeps the pelvis centred between where both feet were and "
        "the COM stays ~0.1 m inside the stance foot's edge; the lifted thigh "
        "and the raised arms put more mass high and forward."),
  "bend_fwd": Pose(
    joints={**_pair("hip_pitch", -0.8),           # legs vertical under a pelvis pitched 0.8 fwd
            **_pair("knee", 0.07),                # locked straight (the soft limit is 0.061)
            **_pair("ankle_pitch", -0.07),        # soles flat (0.8 - 0.8 + 0.07 - 0.07)
            "waist_pitch_joint": 0.5,             # trunk bent forward to the waist's limit
            **_pair("shoulder_pitch", -1.2),      # arms hanging from shoulders ahead of the toes
            **_pair("elbow", 1.5)},               # straight
    root_pitch=0.8,
    doc="deep forward bend with straight knees: pelvis pitched 0.8 rad, the "
        "trunk ~1.3 rad from vertical, arms hanging: the upper body's mass is "
        "ahead of the toes over locked knees and neutral ankles; holding it "
        "needs more ankle torque than the foot's lever can carry, and the only "
        "recoveries (bend the knees, step) are what the pose forbids."),
  "split": Pose(
    joints={**_pair("hip_roll", 0.6, -0.6),       # each leg 34 deg out (feet ~0.9 m apart)
            **_pair("hip_pitch", 0.0),
            **_pair("knee", 0.07),                # straight
            **_pair("ankle_pitch", -0.07),        # soles flat in pitch
            **_pair("ankle_roll", -0.26, 0.26),   # the most counter-roll the ankles have
            **_pair("shoulder_roll", 1.5, -1.5),  # arms out sideways
            **_pair("elbow", 1.5)},               # and straight
    doc="wide lateral split with straight knees: the ankles' 0.24 rad of roll "
        "cannot flatten soles rolled 0.6 rad, so each foot stands on its inner "
        "edge, and the splayed straight legs need a horizontal friction force "
        "of ~0.35 x the weight at each foot to keep the pelvis up; nothing in "
        "the pose can catch a lateral slip."),
  "arch_back": Pose(
    joints={**_pair("hip_pitch", 0.5),            # legs vertical under a pelvis pitched 0.5 back
            **_pair("knee", 0.07),                # locked straight
            **_pair("ankle_pitch", -0.07),        # soles flat (-0.5 + 0.5 + 0.07 - 0.07)
            "waist_pitch_joint": -0.5,            # trunk arched back to the waist's limit
            **_pair("shoulder_pitch", -2.8),      # arms overhead, i.e. up and back
            **_pair("elbow", 1.5)},               # and straight
    root_pitch=-0.5,
    doc="backward arch: pelvis pitched 0.5 rad back, the trunk ~1 rad back and "
        "the arms overhead: the COM is behind the heels over locked knees, and "
        "the heel's short lever cannot hold it."),
}


# --- the scenarios -------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
  """One ill-posed reference: a family (:data:`FAMILIES`), its ``params``,
  what happens after the window (``after``: return / hold / continue), an
  optional ``duration_s`` that overrides the build's window, and ``doc``:
  what makes it ill-posed."""
  family: str
  after: str
  doc: str
  params: dict = field(default_factory=dict)
  duration_s: Optional[float] = None


def _osc_doc(f_hz: float, amp: float) -> str:
  w = 2.0 * math.pi * f_hz
  return (f"a {f_hz:g} Hz, {amp:g} rad sinusoid on every leg and waist joint for "
          f"the window: peak target velocity {amp * w:.0f} rad/s and acceleration "
          f"{amp * w * w:.0f} rad/s^2. The joint loop (Kp = armature x (2 pi 10)^2, "
          "damping ratio 2) has a dominant pole near 2.7 Hz for the bare rotor, "
          "lower with the links, so it cannot follow; the torque spent trying "
          "shakes the stance foot loose. Every frame is a valid pose.")


def _lean_doc(deg: float, way: str) -> str:
  beyond = (" 60 deg is also past K's 1 rad tilt limit (the driver's bad_out_k "
            "flag), though the episode only ends at the task's 70 deg fell_over "
            "termination or a non-foot contact." if deg > 57.3 else "")
  return (f"the root attitude tilted {deg:g} deg {way} (heading-frame axis, on "
          "top of the clip's own tilt) over 0.3 s and held, joints unchanged: "
          "SONIC tracks the pelvis attitude relative to the robot, so it is told "
          "to lean the whole body with the feet planted, putting the COM beyond "
          "the support polygon." + beyond)


SCENARIOS: dict[str, Scenario] = {
  # high-frequency movement
  "hf_osc_4hz_0.3": Scenario("hf_osc", RETURN, _osc_doc(4.0, 0.3), dict(freq_hz=4.0, amp=0.3)),
  "hf_osc_4hz_0.6": Scenario("hf_osc", RETURN, _osc_doc(4.0, 0.6), dict(freq_hz=4.0, amp=0.6)),
  "hf_osc_8hz_0.3": Scenario("hf_osc", RETURN, _osc_doc(8.0, 0.3), dict(freq_hz=8.0, amp=0.3)),
  "jitter_0.1": Scenario(
    "jitter", RETURN,
    "i.i.d. N(0, 0.1^2) rad per frame on all 29 joints for the window: white "
    "noise up to the 25 Hz Nyquist rate, with forward-difference velocities "
    "of 7 rad/s RMS; no motion has this spectrum, and SONIC reads the noise in "
    "its velocity channel.", dict(sigma=0.1)),
  "jitter_0.3": Scenario(
    "jitter", RETURN,
    "i.i.d. N(0, 0.3^2) rad per frame on all 29 joints for the window: "
    "21 rad/s RMS velocity noise, every frame a different random pose around "
    "the clip.", dict(sigma=0.3)),
  "speedup_3x": Scenario(
    "speedup", CONTINUE,
    "the clip played 3x for the window, then on at 1x from where that "
    "left it: joint velocities and the yaw rate tripled (a walk becomes a "
    "run with a walk's foot placement, a run a sprint the G1 cannot "
    "power), then a sudden deceleration.", dict(rate=3)),
  "speedup_5x": Scenario(
    "speedup", CONTINUE,
    "the clip played 5x for the window, then 1x: 7.5 s of motion in 1.5 s, "
    "past the actuators' speed and torque limits.", dict(rate=5)),
  # knowingly unstable poses and attitudes
  "lean_fwd_30": Scenario("lean", HOLD, _lean_doc(30, "forward"), dict(axis="pitch", deg=30.0, side=1.0)),
  "lean_fwd_60": Scenario("lean", HOLD, _lean_doc(60, "forward"), dict(axis="pitch", deg=60.0, side=1.0)),
  "lean_back_30": Scenario("lean", HOLD, _lean_doc(30, "backward"), dict(axis="pitch", deg=30.0, side=-1.0)),
  "lean_back_60": Scenario("lean", HOLD, _lean_doc(60, "backward"), dict(axis="pitch", deg=60.0, side=-1.0)),
  "lean_roll_30": Scenario("lean", HOLD, _lean_doc(30, "sideways (side drawn per episode)"),
                           dict(axis="roll", deg=30.0, side=0.0)),
  "lean_roll_60": Scenario("lean", HOLD, _lean_doc(60, "sideways (side drawn per episode)"),
                           dict(axis="roll", deg=60.0, side=0.0)),
  **{f"pose_{k}": Scenario("pose", HOLD, f"blend in 0.3 s and hold: {p.doc}", dict(pose=k))
     for k, p in POSES.items()},
  # discontinuities
  "teleport": Scenario(
    "teleport", CONTINUE,
    "at the onset the reference jumps to a random frame of a DIFFERENT clip "
    "among the factory's teleport_clips (the driver passes the held-out ones), "
    "its heading aligned to the current one, so that only the pose and the "
    "tilt jump, and plays that clip on: an arbitrary whole-body pose "
    "discontinuity (a walk into a fight, a dance or a fall-and-get-up frame), "
    "a one-frame velocity spike of the jump x 50."),
  "skip_0.5s": Scenario(
    "skip", CONTINUE,
    "the clip jumps 0.5 s ahead at the onset and plays on: a pose jump the "
    "size of what the clip does in 0.5 s (about a step in a walk), with the "
    "matching velocity spike.", dict(skip_s=0.5)),
  "skip_1.0s": Scenario(
    "skip", CONTINUE,
    "the clip jumps 1 s ahead at the onset and plays on: about a gait cycle "
    "in a walk, anything in a dance or a fight.", dict(skip_s=1.0)),
  "reverse": Scenario(
    "reverse", CONTINUE,
    "the clip played BACKWARDS for the window, then forward again from there: "
    "every joint velocity and the yaw rate flip sign in one frame, twice; the "
    "reference asks for instantaneous momentum reversals."),
  "freeze_jump": Scenario(
    "freeze", RETURN,
    "the last clean frame held for 0.5 s, then a jump back to the "
    "time-aligned frame: the robot is told to stop dead mid-motion (zero "
    "target velocity against the clip's momentum), then to be 0.5 s ahead at "
    "once.", duration_s=0.5),
  "freeze_jump_1.0s": Scenario(
    "freeze", RETURN,
    "as freeze_jump, frozen for 1 s: a longer stop and a bigger jump.",
    duration_s=1.0),
  "mirror": Scenario(
    "mirror", HOLD,
    "left and right swapped at the onset (roll / yaw joints negated, the root "
    "attitude reflected across the heading plane) and held: the mirrored clip "
    "is a valid motion of the symmetric G1, but the switch is instantaneous, "
    "so stance and swing legs exchange in one frame and in a walk the "
    "reference puts the loaded foot in the air."),
  "yaw_90": Scenario(
    "yaw", HOLD,
    "the reference root yaw turned 90 deg (side drawn per episode) in one "
    "frame and held: SONIC reads the anchor orientation relative to the "
    "robot, so it is told it faces 90 deg off the clip and can only comply "
    "by spinning on the spot mid-motion.", dict(deg=90.0, random_sign=True)),
  "yaw_180": Scenario(
    "yaw", HOLD,
    "the reference root yaw turned 180 deg in one frame and held: the clip "
    "now runs backwards relative to where the robot faces.",
    dict(deg=180.0, random_sign=False)),
}

#: A small cross-section for smoke runs: frequency, attitude, discontinuity.
QUICK_SCENARIOS = ("hf_osc_4hz_0.6", "lean_fwd_30", "teleport")


# --- the batch a family edits ----------------------------------------------------

class _Batch:
  """E episodes x L frames: per-frame indices of the time-aligned original
  (``true_clip`` / ``true_frame``), the source index a family may remap
  (``src_*``), and the arrays it edits (``pos`` ``[E, L, J]``, ``quat``
  ``[E, L, 4]``, starting as the original ``pos0`` / ``quat0``)."""

  def __init__(self, base, names, clip, frame0, onset, dur, L, gen, scenario, targets=None):
    self.base, self.names, self.scenario, self.gen = base, tuple(names), scenario, gen
    self.targets = targets                    # the clips a teleport may land in (sorted)
    self.idx = {n: i for i, n in enumerate(self.names)}
    dev = self.dev = clip.device
    self.E, self.L = int(clip.numel()), int(L)
    self.clip, self.frame0, self.onset, self.dur = clip, frame0, onset, dur
    f = torch.arange(self.L, device=dev)
    self.rel = f.unsqueeze(0) - onset.unsqueeze(1)             # frames since the onset
    self.true_clip = clip.unsqueeze(1).expand(self.E, self.L)
    self.true_frame = frame0.unsqueeze(1) + f
    self.src_clip, self.src_frame = self.true_clip, self.true_frame
    self.draws: dict = {}

  @property
  def after_onset(self) -> torch.Tensor:
    return self.rel >= 0

  @property
  def window(self) -> torch.Tensor:
    return (self.rel >= 0) & (self.rel < self.dur.unsqueeze(1))

  def gather(self) -> None:
    self.pos0, self.quat0 = self.take(self.true_clip, self.true_frame)
    self.pos, self.quat = self.pos0.clone(), self.quat0.clone()

  def take(self, clip: torch.Tensor, frame: torch.Tensor):
    """Joint positions and root quaternions of base ``(clip, frame)`` (any
    shape; frames clamp to the clip, as the tracker's do)."""
    rows = self.base.rows(clip.reshape(-1), frame.reshape(-1)).view(clip.shape)
    return self.base.joint_pos[rows], self.base.root_quat[rows]

  def at_onset(self, x: torch.Tensor) -> torch.Tensor:
    """Each episode's ``x[e, onset[e]]``."""
    return x[torch.arange(self.E, device=self.dev), self.onset]

  def _need_gen(self) -> torch.Generator:
    if self.gen is None:
      raise ValueError(f"scenario {self.scenario!r} draws random numbers: pass a "
                       "torch.Generator (the same seed for every condition)")
    return self.gen

  def rand(self, *shape) -> torch.Tensor:
    g = self._need_gen()
    return torch.rand(shape, generator=g, device=g.device).to(self.dev)

  def randn(self, *shape) -> torch.Tensor:
    g = self._need_gen()
    return torch.randn(shape, generator=g, device=g.device).to(self.dev)

  def joint_ids(self, pred: Callable[[str], bool]) -> torch.Tensor:
    return torch.as_tensor([i for i, n in enumerate(self.names) if pred(n)],
                           dtype=torch.long, device=self.dev)

  def vector(self, angles: dict) -> torch.Tensor:
    """A joint vector from :data:`NEUTRAL` updated by ``angles`` (by name)."""
    v = torch.zeros(len(self.names), device=self.dev)
    for n, a in {**NEUTRAL, **angles}.items():
      if n not in self.idx:
        raise ValueError(f"pose joint {n!r} is not a SONIC joint")
      v[self.idx[n]] = float(a)
    return v


def _cos_ramp(x: torch.Tensor) -> torch.Tensor:
  return 0.5 * (1.0 - torch.cos(math.pi * x.clamp(0.0, 1.0)))


def _blend_in(b: _Batch, ramp_s: float) -> torch.Tensor:
  """``[E, L]``: 0 up to the last clean frame, > 0 at the onset, 1 from
  ``onset + R - 1`` on (R = the ramp in frames)."""
  R = max(1, round(ramp_s * FPS))
  return _cos_ramp((b.rel + 1).float() / R)


# --- the families --------------------------------------------------------------

def _hf_osc(b: _Batch, freq_hz: float, amp: float, ramp_s: float = OSC_RAMP_S) -> None:
  """``amp * sin(2 pi freq t + phi)`` added to every leg and waist joint over
  the window (``phi ~ U(0, 2 pi)`` per episode and joint; t from the last
  clean frame), tapered by cosine ramps of ``ramp_s`` at both ends. The
  arms keep the clip's motion."""
  ids = b.joint_ids(_is_leg_or_waist)
  phase = b.rand(b.E, int(ids.numel())) * (2.0 * math.pi)
  R = max(1, round(ramp_s * FPS))
  t = (b.rel + 1).float() / FPS
  env = (_cos_ramp((b.rel + 1).float() / R)
         * _cos_ramp((b.dur.unsqueeze(1) - b.rel).float() / R) * b.window)
  wave = amp * torch.sin(2.0 * math.pi * freq_hz * t.unsqueeze(-1) + phase.unsqueeze(1))
  b.pos[..., ids] = b.pos[..., ids] + wave * env.unsqueeze(-1)
  b.draws["phase"] = phase


def _jitter(b: _Batch, sigma: float) -> None:
  """i.i.d. ``N(0, sigma^2)`` per frame and joint (all joints) over the
  window. The noise is drawn for all L frames, so the stream does not depend
  on the onset."""
  noise = b.randn(b.E, b.L, len(b.names)) * sigma
  b.pos = b.pos + noise * b.window.unsqueeze(-1)


def _rate_offset(b: _Batch, rate: int) -> torch.Tensor:
  # source frame = frame0 + f + (rate - 1) * (frames of fast play so far):
  # slope ``rate`` from the last clean frame for ``dur`` frames, then slope 1
  return (int(rate) - 1) * torch.minimum((b.rel + 1).clamp_min(0), b.dur.unsqueeze(1))


def _speedup(b: _Batch, rate: int) -> None:
  """The clip played at ``rate`` x (an integer) for the window, starting
  from the last clean frame, then on at 1x from the frame it reached."""
  b.src_frame = b.true_frame + _rate_offset(b, rate)
  b.pos, b.quat = b.take(b.src_clip, b.src_frame)


def _reverse_offset(b: _Batch) -> torch.Tensor:
  return _rate_offset(b, -1)


def _reverse(b: _Batch) -> None:
  """The clip played backwards from the last clean frame for the window,
  then forward again at 1x from the frame it reached (rate -1)."""
  _speedup(b, -1)


def _skip_offset(b: _Batch, skip_s: float) -> torch.Tensor:
  return round(skip_s * FPS) * (b.rel >= 0).long()


def _skip(b: _Batch, skip_s: float) -> None:
  """At the onset the clip jumps ``skip_s`` ahead and plays on."""
  b.src_frame = b.true_frame + _skip_offset(b, skip_s)
  b.pos, b.quat = b.take(b.src_clip, b.src_frame)


def _freeze_offset(b: _Batch) -> torch.Tensor:
  held = -torch.minimum((b.rel + 1).clamp_min(0), b.dur.unsqueeze(1))
  return torch.where(b.rel < b.dur.unsqueeze(1), held, torch.zeros_like(held))


def _freeze(b: _Batch) -> None:
  """The last clean frame (onset - 1) held for the window, then the
  time-aligned frame again."""
  b.src_frame = b.true_frame + _freeze_offset(b)
  b.pos, b.quat = b.take(b.src_clip, b.src_frame)


def _teleport(b: _Batch) -> None:
  """From the onset, another clip (uniform over the target clips, the
  factory's ``teleport_clips`` minus the episode's own) from a uniform frame
  that leaves it room for the rest of the episode when it has that room
  (from frame 0 otherwise). Its root is turned so that its heading at the
  jump is the original's at the onset."""
  T = b.targets
  own = T.unsqueeze(0) == b.clip.unsqueeze(1)                # [E, M]
  has_own = own.any(1)
  count = int(T.numel()) - has_own.long()
  if bool((count < 1).any()):
    raise ValueError("teleport needs a target clip other than the episode's own")
  u = b.rand(b.E, 2)
  # the j-th target that is not the own clip (T is sorted, so own sits at
  # searchsorted(T, clip) when present)
  j = torch.minimum((u[:, 0] * count.float()).long(), count - 1)
  j = j + (has_own & (j >= torch.searchsorted(T, b.clip))).long()
  other = T[j]
  room = (b.base.length[other] - (b.L - b.onset)).clamp_min(0)
  g0 = torch.minimum((u[:, 1] * (room + 1).float()).long(), room)
  after = b.after_onset
  b.src_clip = torch.where(after, other.unsqueeze(1), b.true_clip)
  b.src_frame = torch.where(after, g0.unsqueeze(1) + b.rel, b.true_frame)
  pos, quat = b.take(b.src_clip, b.src_frame)
  q_to = b.base.root_quat[b.base.rows(other, g0)]
  dq = quat_mul(yaw_quat(b.at_onset(b.quat0)), quat_conj(yaw_quat(q_to)))
  b.pos = pos
  b.quat = torch.where(after.unsqueeze(-1), quat_mul(dq.unsqueeze(1).expand_as(quat), quat), quat)
  b.draws.update(other_clip=other, other_frame=g0)


def _lean(b: _Batch, axis: str, deg: float, side: float = 1.0,
          ramp_s: float = BLEND_S) -> None:
  """The root attitude tilted by ``side * deg`` about the heading frame's
  ``axis`` ("pitch": y, + = forward; "roll": x, + = toward the right) on top
  of the clip's own attitude, blended in over ``ramp_s``. ``side`` 0 draws
  +-1 per episode."""
  if axis not in ("pitch", "roll"):
    raise ValueError(f"lean axis {axis!r}: 'pitch' or 'roll'")
  if side == 0:
    sgn = torch.where(b.rand(b.E) < 0.5, -1.0, 1.0)
  else:
    sgn = torch.full((b.E,), float(side), device=b.dev)
  ang = math.radians(deg) * sgn.unsqueeze(1) * _blend_in(b, ramp_s) * b.after_onset
  ax = torch.tensor((0.0, 1.0, 0.0) if axis == "pitch" else (1.0, 0.0, 0.0), device=b.dev)
  r = quat_axis_angle(ax.expand(b.E, b.L, 3), ang)
  y = yaw_quat(b.quat)
  b.quat = quat_mul(quat_mul(y, r), quat_mul(quat_conj(y), b.quat))
  b.draws["side"] = sgn


def _pose(b: _Batch, pose: str, ramp_s: float = BLEND_S) -> None:
  """Joints blended linearly, and the root attitude by slerp, to
  :data:`POSES` ``[pose]`` over ``ramp_s`` (cosine ramp), then held. The
  target attitude is the clip's heading times the pose's pitch and roll."""
  p = POSES[pose]
  left = b.vector(p.joints)
  if p.sided:
    right_side = b.rand(b.E) < 0.5
    vec = torch.where(right_side.unsqueeze(1), mirror_joints(left, b.names), left)
    roll = torch.where(right_side, -p.root_roll, p.root_roll)
  else:
    right_side = torch.zeros(b.E, dtype=torch.bool, device=b.dev)
    vec = left.expand(b.E, -1)
    roll = torch.full((b.E,), p.root_roll, device=b.dev)
  s = (_blend_in(b, ramp_s) * b.after_onset).unsqueeze(-1)
  b.pos = (1.0 - s) * b.pos + s * vec.unsqueeze(1)
  ex = torch.tensor((1.0, 0.0, 0.0), device=b.dev)
  ey = torch.tensor((0.0, 1.0, 0.0), device=b.dev)
  tilt = quat_mul(quat_axis_angle(ey, torch.tensor(p.root_pitch, device=b.dev)).expand(b.E, 4),
                  quat_axis_angle(ex.expand(b.E, 3), roll))
  target = quat_mul(yaw_quat(b.quat0), tilt.unsqueeze(1).expand(b.E, b.L, 4))
  b.quat = slerp(b.quat, target, s)
  b.draws["right_side"] = right_side


def _yaw(b: _Batch, deg: float, random_sign: bool = True) -> None:
  """The root turned by ``deg`` about the world z axis from the onset on
  (``random_sign``: the turn's side drawn per episode)."""
  sgn = (torch.where(b.rand(b.E) < 0.5, -1.0, 1.0) if random_sign
         else torch.ones(b.E, device=b.dev))
  rz = z_quat(math.radians(deg) * sgn)
  turned = quat_mul(rz.unsqueeze(1).expand_as(b.quat), b.quat)
  b.quat = torch.where(b.after_onset.unsqueeze(-1), turned, b.quat)
  b.draws["sign"] = sgn


def _mirror(b: _Batch) -> None:
  """From the onset, the sagittal mirror of the clip: joints by
  :func:`mirror_map`, the root by :func:`mirror_quat` about the original's
  heading at the onset (so the heading is continuous there)."""
  after = b.after_onset.unsqueeze(-1)
  b.pos = torch.where(after, mirror_joints(b.pos, b.names), b.pos)
  yaw0 = yaw_of(b.at_onset(b.quat0)).unsqueeze(1)
  b.quat = torch.where(after, mirror_quat(b.quat, yaw0), b.quat)


@dataclass(frozen=True)
class Family:
  """A corruption: ``apply(batch, **params)`` edits the batch. Index families
  also give their frame ``offset(batch, **params)`` (``fits`` reads it).
  ``own_clip`` is False when the family reads another clip after the onset."""
  apply: Callable
  offset: Optional[Callable] = None
  own_clip: bool = True


FAMILIES: dict[str, Family] = {
  "hf_osc": Family(_hf_osc),
  "jitter": Family(_jitter),
  "speedup": Family(_speedup, _rate_offset),
  "reverse": Family(_reverse, _reverse_offset),
  "skip": Family(_skip, _skip_offset),
  "freeze": Family(_freeze, _freeze_offset),
  "teleport": Family(_teleport, own_clip=False),
  "lean": Family(_lean),
  "pose": Family(_pose),
  "yaw": Family(_yaw),
  "mirror": Family(_mirror),
}


def _scenario(s: Union[str, Scenario]) -> tuple[str, Scenario]:
  if isinstance(s, Scenario):
    name, sc = f"custom_{s.family}", s
  elif s in SCENARIOS:
    name, sc = s, SCENARIOS[s]
  else:
    raise ValueError(f"unknown bad-reference scenario {s!r}; known: {', '.join(SCENARIOS)}")
  if sc.family not in FAMILIES:
    raise ValueError(f"scenario {name!r}: unknown family {sc.family!r}")
  if sc.after not in AFTER:
    raise ValueError(f"scenario {name!r}: after={sc.after!r}, not one of {AFTER}")
  return name, sc


# --- the factory ---------------------------------------------------------------

@dataclass
class Corruption:
  """What :meth:`BadReferenceFactory.build` did, per episode e (clip e of the
  returned reference, ``ref.corruption``). ``[E, L]`` tensors are per frame."""
  scenario: str
  family: str
  after: str
  onset: torch.Tensor       # [E] the first corrupted frame
  end: torch.Tensor         # [E] first frame past the corruption (L when it never returns)
  corrupted: torch.Tensor   # [E, L] bool: frames the scenario changed (edited joints clamped)
  true_rows: torch.Tensor   # [E, L] base rows of the time-aligned original: the TRUE reference
  src_clip: torch.Tensor    # [E, L] base clip each frame was read from
  src_frame: torch.Tensor   # [E, L] its frame, before clamping to that clip
  tilt_max: torch.Tensor    # [E] the largest pelvis tilt the corrupted reference asks for, rad
  draws: dict = field(default_factory=dict)   # the scenario's random draws, [E, ...]


class BadReferenceFactory:
  """Per-episode corrupted copies of a base ``SonicReference``.

  :param base: the clean reference (``SonicReference``, SONIC joint order).
  :param joint_low, joint_high: ``[29]`` joint limits in SONIC order (the
      env's soft limits, ``robot.data.soft_joint_pos_limits[0, to_sonic]``);
      the joints a family edits are clamped to them (module docstring,
      LIMITS).
  :param teleport_clips: the base clip ids a teleport may jump into (default
      every clip). The driver passes the held-out library's clips, the only
      ones whose frames it has K flags for.
  """

  def __init__(self, base, joint_low, joint_high, teleport_clips=None):
    self.base = base
    names = tuple(_sonic().SONIC_JOINTS)
    J = len(names)
    if base.joint_pos.dim() != 2 or base.joint_pos.shape[1] != J:
      raise ValueError(f"base joint_pos {tuple(base.joint_pos.shape)}: expected [F, {J}] "
                       "in SONIC_JOINTS order")
    if base.root_quat.shape != (base.joint_pos.shape[0], 4):
      raise ValueError(f"base root_quat {tuple(base.root_quat.shape)}: expected [F, 4]")
    self.names = names
    dev = self.dev = base.joint_pos.device
    self.low = torch.as_tensor(joint_low, dtype=torch.float32, device=dev).reshape(-1)
    self.high = torch.as_tensor(joint_high, dtype=torch.float32, device=dev).reshape(-1)
    if self.low.numel() != J or self.high.numel() != J:
      raise ValueError(f"joint limits must have {J} entries (SONIC order)")
    if bool((self.low > self.high).any()):
      raise ValueError("joint_low > joint_high for some joint")
    C = int(base.length.numel())
    tc = (torch.arange(C) if teleport_clips is None
          else torch.as_tensor(teleport_clips).long().reshape(-1))
    tc = torch.unique(tc.cpu()).to(dev)                 # sorted, as _teleport needs
    if tc.numel() == 0 or bool(((tc < 0) | (tc >= C)).any()):
      raise ValueError(f"teleport_clips must be clip ids in [0, {C})")
    self.teleport_clips = tc

  def _batch(self, clip, frame0, onset, length, sc: Scenario, duration, gen, name) -> _Batch:
    dev = self.dev
    as_long = lambda x: torch.as_tensor(x, device=dev).long().reshape(-1)  # noqa: E731
    clip, frame0, onset = as_long(clip), as_long(frame0), as_long(onset)
    E, L = int(clip.numel()), int(length)
    if E == 0 or frame0.numel() != E or onset.numel() != E:
      raise ValueError(f"clip / frame0 / onset: {clip.numel()} / {frame0.numel()} / "
                       f"{onset.numel()} episodes (need the same, > 0)")
    if L < 3:
      raise ValueError(f"length {L}: need >= 3 frames")
    if bool((onset < 1).any()) or bool((onset >= L).any()):
      raise ValueError(f"onset must lie in [1, length - 1] = [1, {L - 1}]: frame 0 is "
                       "the spawn and stays the base frame, and an onset past the "
                       "clip would corrupt nothing")
    C = int(self.base.length.numel())
    if bool(((clip < 0) | (clip >= C)).any()):
      raise ValueError(f"clip ids must lie in [0, {C})")
    n = self.base.length[clip]
    if bool(((frame0 < 0) | (frame0 >= n)).any()):
      raise ValueError("frame0 outside its clip")
    if sc.duration_s is not None:
      dur = round(sc.duration_s * FPS)
    elif duration is None:
      dur = round(DEFAULT_DURATION_S * FPS)
    else:
      dur = duration
    dur = as_long(dur)
    dur = dur.expand(E).clone() if dur.numel() == 1 else dur
    if dur.numel() != E or bool((dur < 1).any()):
      raise ValueError("duration: one value >= 1 frame, or one per episode")
    return _Batch(self.base, self.names, clip, frame0, onset, dur, L, gen, name,
                  self.teleport_clips)

  def build(self, clip, frame0, onset, length: int, scenario: Union[str, Scenario],
            generator: Optional[torch.Generator] = None,
            duration: Union[int, torch.Tensor, None] = None):
    """A ``SonicReference`` of E clips of ``length`` frames: clip e is base
    clip ``clip[e]`` from frame ``frame0[e]``, corrupted by ``scenario`` from
    frame ``onset[e]`` (frames; >= 1). ``duration`` is the window in frames
    (int or ``[E]``, default 1.5 s), overridden by a scenario that fixes its
    own. Random scenarios draw from ``generator`` only. The result carries
    :class:`Corruption` as ``.corruption``. Start the tracker on it with
    ``clip = e, frame = 0``."""
    name, sc = _scenario(scenario)
    b = self._batch(clip, frame0, onset, length, sc, duration, generator, name)
    b.gather()
    FAMILIES[sc.family].apply(b, **sc.params)
    mask = b.window if sc.after == RETURN else b.after_onset
    m = mask.unsqueeze(-1)
    # untouched frames are the base's own, bit for bit; corrupted ones are
    # clamped into the limits widened to the source frame's value (so a
    # joint the family did not edit keeps it exactly) and renormalized
    src = b.take(b.src_clip, b.src_frame)[0]
    lo, hi = torch.minimum(self.low, src), torch.maximum(self.high, src)
    pos = torch.where(m, torch.maximum(torch.minimum(b.pos, hi), lo), b.pos0)
    quat = torch.where(m, F.normalize(b.quat, dim=-1), b.quat0)
    S = _sonic()
    vel = S.forward_difference(pos.transpose(0, 1), FPS).transpose(0, 1)
    E, L, J = b.E, b.L, len(self.names)
    names = [f"{name}:{self.base.names[c]}@{f}"
             for c, f in zip(b.clip.tolist(), b.frame0.tolist())]
    ref = S.SonicReference(pos.reshape(E * L, J).contiguous(), vel.reshape(E * L, J).contiguous(),
                           quat.reshape(E * L, 4).contiguous(), [L] * E, names)
    end = b.onset + b.dur if sc.after == RETURN else torch.full_like(b.onset, L)
    ref.corruption = Corruption(
      scenario=name, family=sc.family, after=sc.after, onset=b.onset.clone(),
      end=torch.minimum(end, torch.full_like(end, L)), corrupted=mask,
      true_rows=self.base.rows(b.true_clip.reshape(-1), b.true_frame.reshape(-1)).view(E, L),
      src_clip=b.src_clip.clone(), src_frame=b.src_frame.clone(),
      tilt_max=tilt_of(quat).amax(1), draws=b.draws)
    return ref

  def fits(self, clip, frame0, onset, length: int, scenario: Union[str, Scenario],
           duration: Union[int, torch.Tensor, None] = None) -> torch.Tensor:
    """``[E]`` bool: every base frame the scenario reads lies inside the
    episode's clip, so nothing is held at a clip end. For teleport only the
    frames before the jump are checked; its target frame is drawn to fit
    when the target clip is long enough. Cheap (indices only): use it to pick
    spawn rows for a scenario, e.g. speedup_5x reads 7.5 s of clip in 1.5 s."""
    name, sc = _scenario(scenario)
    b = self._batch(clip, frame0, onset, length, sc, duration, None, name)
    fam = FAMILIES[sc.family]
    frame = b.true_frame
    if fam.offset is not None:
      frame = frame + fam.offset(b, **sc.params)
    n = self.base.length[b.clip].unsqueeze(1)
    ok = (frame >= 0) & (frame < n)
    if not fam.own_clip:
      ok = ok | b.after_onset
    return ok.all(1)
