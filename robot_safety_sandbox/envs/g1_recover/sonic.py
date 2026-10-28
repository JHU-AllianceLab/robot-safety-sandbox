"""SONIC (NVIDIA GEAR-SONIC, arXiv 2511.07820) as a closed-loop G1 motion tracker.

The handover harvesters need a NOMINAL controller running a motion when the
push lands. ``playback.py`` replays the clip open loop, which drifts on fast
clips because nothing closes the loop on the base. SONIC is a pretrained
whole-body tracker for the 29-DoF G1. Here it runs batched on the GPU inside
the sandbox's own mjlab G1 envs and tracks the LAFAN clip library
(``examples/build_g1_recover_mocap_states.py --library``).

THE RELEASE. ``huggingface.co/nvidia/GEAR-SONIC``, default variant, files
``model_encoder.onnx`` and ``model_decoder.onnx`` (the deploy stack's
``observation_config.yaml``: encoder input 1762, decoder input 994). Both are
exported at batch 1. :class:`SonicPolicy` lifts their weights into torch, so
4096 envs run in one call. ``tests/test_sonic.py`` checks the port against
``onnx.reference``. Only the encoder's ``g1`` branch (robot joint-space
reference) is kept. The teleop and smpl branches are multiplied by a zero
one-hot in mode 0, so they never reach the token.

THE INPUTS, as the training env builds them (``gear_sonic/envs/manager_env/
mdp``). The C++ deploy (``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/
g1_deploy_onnx_ref.cpp``) builds the same layout. Joints are in IsaacLab order
(:data:`SONIC_JOINTS`). Every history is 10 control ticks, oldest first.

* encoder (g1 mode, 640 = 10 frames x 64): reference frames t + 1 + 5k,
  k = 0..9, 0.1 s apart and clamped to the clip's last frame, where t is the
  frame the robot nominally is at. The window starts one frame ahead
  (:data:`REF_LEAD`) because the training env's command term advances its
  clock before every observation, a reset's included. The C++ deploy reads
  t + 5k; on G1 starts the difference does not change the fall rate.
  ``cat(joint_pos[10x29], joint_vel[10x29]).view(10, 58)``, each frame then
  followed by the 6-d anchor orientation (first two columns of
  R(q_robot^-1 * dq * q_ref), row-major). Joint positions are ABSOLUTE.
  Joint velocities are forward differences at 50 fps, as SONIC's motion lib
  computes them (mjlab's MotionLoader uses central ones). The pelvis is the
  anchor. ``dq`` is the delta heading (yaw of the robot at the episode start
  times the inverse yaw of the reference start frame), so the clip is played
  from wherever the robot faces.
  The output is an FSQ token (2 x 32, :meth:`SonicPolicy.encode`).
* decoder (994): token 64, then the histories of pelvis angular velocity
  (body frame) 30, joint_pos - default 290, joint_vel 290, last action 290, and
  projected gravity 30.
* action: the net output clipped to +-20 (:data:`ACTION_CLIP`, the training
  env's ``action_clip_value``; the deploy does not clip, and the clip only
  binds in states that are already lost). The joint target is
  ``default + action * SONIC_scale`` with SONIC's own per-joint scale,
  0.25 * effort / stiffness of ITS actuators (:func:`sonic_action_scale`).

THE ROBOT. mjlab's G1 and SONIC's training G1 share the BeyondMimic
actuator recipe (10 Hz natural frequency, damping ratio 2, the 2x ankle and
waist roll/pitch drives) and the knees-bent default pose. They differ at the
hip pitch: SONIC's training robot (and the deploy's ``policy_parameters.hpp``,
"old is 7520_14 new is 7520_22") drives it with the 7520-22 motor (Kp 99.1,
139 N*m), mjlab with the 7520-14 (Kp 40.2, 88 N*m).
:func:`use_sonic_hip_pitch` swaps an env cfg's hip-pitch actuator to SONIC's.
Either way, SONIC's targets reach the env through :meth:`SonicTracker.env_action`,
which inverts the env's own action term, so the env applies exactly SONIC's
joint targets.

RESET. Every history is filled with the state at the reset, and the last
action is zero. Isaac Lab's history buffers do the same on the first append
after a reset, and its action manager zeroes the last action.

OFFLINE. The tracker and :func:`rebuild_inputs` build every input through the
same functions (:func:`reference_window`, :func:`anchor_6d`,
:func:`proprio_entry`, :func:`decoder_obs`), so a harvested trajectory plus
its step-0 histories gives SONIC's inputs at every step. On a live check, the
histories agree to 1e-6. On about 1e-4 of steps one FSQ token entry rounds to
the neighbouring level (float noise at a quantization boundary).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)

#: Where ``huggingface.co/nvidia/GEAR-SONIC`` was downloaded (override with
#: ``SONIC_DIR`` or the harvester's ``--sonic-dir``).
DEFAULT_SONIC_DIR = os.environ.get(
  "SONIC_DIR", os.path.expanduser("~/Documents/gear_sonic_hf"))

#: The 29 joints in IsaacLab order, the order of every SONIC joint vector.
SONIC_JOINTS = (
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
  "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
  "left_knee_joint", "right_knee_joint",
  "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
  "left_shoulder_roll_joint", "right_shoulder_roll_joint",
  "left_ankle_roll_joint", "right_ankle_roll_joint",
  "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
  "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint",
  "left_wrist_pitch_joint", "right_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_wrist_yaw_joint")

#: Frames per reference window, the spacing between them (50 Hz control
#: ticks) and the proprioceptive history length.
NUM_FUTURE = 10
FUTURE_STEP = 5
HISTORY = 10
TOKEN_DIM = 64
#: The reference window starts this many frames past the robot's nominal frame
#: (the training env's clock advances before each observation).
REF_LEAD = 1
#: The encoder's mode id for a robot joint-space reference.
MODE_G1 = 0

# SONIC's actuators (gear_sonic/envs/manager_env/robots/g1.py): the armatures
# the deploy hard-codes and the BeyondMimic gains derived from them.
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520,
             "7520_22": 0.025101925, "4010": 0.00425}
_OMEGA = 10.0 * 2.0 * 3.1415926535
_ZETA = 2.0
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}


def _stiffness(motor: str) -> float:
  return _ARMATURE[motor] * _OMEGA ** 2


def _damping(motor: str) -> float:
  return 2.0 * _ZETA * _ARMATURE[motor] * _OMEGA


def sonic_action_scale(joint_names: Sequence[str]) -> list[float]:
  """SONIC's per-joint action scale, 0.25 * effort / stiffness of its training
  actuators (``G1_MODEL_12_ACTION_SCALE``), for ``joint_names``. The 2x drives
  (ankles, waist roll/pitch) double effort and stiffness alike."""
  def motor(n: str) -> str:
    if any(k in n for k in ("hip_pitch", "hip_roll", "knee")):
      return "7520_22"
    if "hip_yaw" in n or n == "waist_yaw_joint":
      return "7520_14"
    if "wrist_pitch" in n or "wrist_yaw" in n:
      return "4010"
    return "5020"
  return [0.25 * _EFFORT[motor(n)] / _stiffness(motor(n)) for n in joint_names]


def use_sonic_hip_pitch(cfg) -> None:
  """Drive an mjlab G1 env cfg's hip pitch with SONIC's 7520-22 actuator
  (Kp 99.1, Kd 6.31, armature 0.0251, 139 N*m) instead of mjlab's 7520-14.
  Every other actuator is already SONIC's. Replaces ``cfg``'s robot entity cfg
  with a copy. mjlab hands the same articulation object to every G1 cfg, so
  mutating it would change the G1 of every env built later in the process.
  Idempotent."""
  from dataclasses import replace
  hip = ".*_hip_pitch_joint"
  robot = cfg.scene.entities["robot"]
  acts = list(robot.articulation.actuators)
  out = []
  for a in acts:
    names = tuple(a.target_names_expr)
    if hip not in names:
      out.append(a)
      continue
    rest = tuple(n for n in names if n != hip)
    if rest:
      out.append(replace(a, target_names_expr=rest))
    out.append(replace(
      a, target_names_expr=(hip,),
      stiffness=_stiffness("7520_22"), damping=_damping("7520_22"),
      effort_limit=_EFFORT["7520_22"], armature=_ARMATURE["7520_22"]))
  if not any(hip in a.target_names_expr for a in acts):
    raise ValueError(f"no actuator group drives {hip!r}")
  cfg.scene.entities["robot"] = replace(
    robot, articulation=replace(robot.articulation, actuators=tuple(out)))


# --- the networks -------------------------------------------------------------

def _onnx_tensors(model) -> dict:
  """Every constant of an ONNX graph: initializers and Constant node outputs."""
  from onnx import numpy_helper
  vals = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
  for n in model.graph.node:
    if n.op_type == "Constant":
      vals[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
  return vals


class SonicPolicy(torch.nn.Module):
  """SONIC's g1-mode encoder (MLP + FSQ) and its decoder, batched.

  Both MLPs use SiLU. The encoder's FSQ maps each of the 2 x 32 latents through
  ``round(tanh(z + shift) * half_l - offset) / half_width``. Build it with
  :meth:`from_onnx`.
  """

  def __init__(self, enc: list, fsq: dict, dec: list):
    super().__init__()
    self.enc = torch.nn.ModuleList(enc)
    self.dec = torch.nn.ModuleList(dec)
    for k, v in fsq.items():
      self.register_buffer(k, v)

  @classmethod
  def from_onnx(cls, sonic_dir: str = DEFAULT_SONIC_DIR) -> "SonicPolicy":
    import onnx
    enc_m = onnx.load(os.path.join(sonic_dir, "model_encoder.onnx"))
    dec_m = onnx.load(os.path.join(sonic_dir, "model_decoder.onnx"))
    if [d.dim_value for d in enc_m.graph.input[0].type.tensor_type.shape.dim] != [1, 1762]:
      raise ValueError(f"{sonic_dir}: not the default SONIC encoder (input 1762)")
    ev = _onnx_tensors(enc_m)

    def linear(w, b, transposed):
      w = torch.tensor(w if transposed else w.T, dtype=torch.float32)
      lin = torch.nn.Linear(w.shape[1], w.shape[0])
      lin.weight.data.copy_(w)
      lin.bias.data.copy_(torch.tensor(b, dtype=torch.float32))
      return lin

    enc = [linear(ev[n.input[1]], ev[n.input[2]], True)
           for n in enc_m.graph.node
           if n.op_type == "Gemm" and n.output[0].startswith("/g1/")]
    if [l.in_features for l in enc] != [640, 2048, 1024, 512, 512]:
      raise ValueError("unexpected SONIC g1 encoder shapes")
    q = {n.output[0].split("/")[-1]: n for n in enc_m.graph.node
         if n.output[0].startswith("/quantizer/")}

    def const(node):
      c = [i for i in node.input if i in ev and ev[i].size == 32]
      return torch.tensor(ev[c[0]], dtype=torch.float32).reshape(-1)

    fsq = {"shift": const(q["Add_output_0"]), "half_l": const(q["Mul_output_0"]),
           "offset": const(q["Sub_output_0"]), "half_width": const(q["Div_output_0"])}
    dv = _onnx_tensors(dec_m)
    nodes = list(dec_m.graph.node)
    dec = []
    for i, n in enumerate(nodes):
      if n.op_type == "MatMul":
        add = nodes[i + 1]
        bias = next(x for x in add.input if x in dv)
        dec.append(linear(dv[n.input[1]], dv[bias], False))
    if dec[0].in_features != 994 or dec[-1].out_features != 29:
      raise ValueError("unexpected SONIC decoder shapes")
    return cls(enc, fsq, dec)

  def encode(self, pos: torch.Tensor, vel: torch.Tensor,
             anchor: torch.Tensor) -> torch.Tensor:
    """``pos`` / ``vel`` ``[B, 10, 29]`` (IsaacLab order), ``anchor`` ``[B, 10, 6]``
    -> the token ``[B, 64]``."""
    B = pos.shape[0]
    x = torch.cat([pos.reshape(B, -1), vel.reshape(B, -1)], 1).view(B, NUM_FUTURE, 58)
    x = torch.cat([x, anchor], -1).view(B, -1)
    for i, lin in enumerate(self.enc):
      x = lin(x)
      if i < len(self.enc) - 1:
        x = F.silu(x)
    z = x.view(B, 2, 32)
    z = torch.tanh(z + self.shift) * self.half_l - self.offset
    return (torch.round(z) / self.half_width).view(B, TOKEN_DIM)

  def decode(self, obs: torch.Tensor) -> torch.Tensor:
    x = obs
    for i, lin in enumerate(self.dec):
      x = lin(x)
      if i < len(self.dec) - 1:
        x = F.silu(x)
    return x


# --- the reference -------------------------------------------------------------

class SonicReference:
  """Every clip at 50 fps, world frame, in SONIC's joint order: ``joint_pos``,
  ``joint_vel`` (``[F, 29]``, IsaacLab order) and ``root_quat`` (``[F, 4]``,
  w-first), all clips concatenated. Clip ``c`` frame ``f`` is row
  ``start[c] + f``. Built with mjlab's ``MotionLoader`` exactly like the clip
  library, so library row ``r`` is frame ``library["frame"][r]`` of clip
  ``library["clip"][r]`` here (``build`` checks that the yaw-free orientation
  agrees)."""

  def __init__(self, joint_pos, joint_vel, root_quat, lengths, names):
    self.joint_pos, self.joint_vel, self.root_quat = joint_pos, joint_vel, root_quat
    self.length = torch.as_tensor(lengths, dtype=torch.long, device=joint_pos.device)
    self.start = torch.cumsum(self.length, 0) - self.length
    self.names = list(names)

  @classmethod
  def build(cls, csv_dir: str, clip_names: Sequence[str], device,
            input_fps: int = 30, library: Optional[dict] = None) -> "SonicReference":
    from mjlab.scripts.csv_to_npz import MotionLoader
    from robot_safety_sandbox.envs.g1_recover.states import to_heading_frame
    csv_order = _csv_joints()
    perm = torch.as_tensor([csv_order.index(n) for n in SONIC_JOINTS])
    pos, vel, quat, lengths = [], [], [], []
    for name in clip_names:
      ml = MotionLoader(os.path.join(csv_dir, f"{name}.csv"), int(input_fps), 50, "cpu")
      p = ml.motion_dof_poss[:, perm]
      pos.append(p)
      vel.append(forward_difference(p, 50.0))
      quat.append(ml.motion_base_rots)
      lengths.append(int(ml.output_frames))
    ref = cls(torch.cat(pos).float().to(device), torch.cat(vel).float().to(device),
              torch.cat(quat).float().to(device), lengths, clip_names)
    if library is not None:
      lib_len = library["meta"].get("clip_frames")
      if lib_len is not None and list(lib_len) != lengths:
        raise ValueError("clip lengths differ from the library's clip_frames")
      rows = ref.rows(library["clip"].to(device), library["frame"].to(device))
      qh = to_heading_frame(ref.root_quat[rows],
                            torch.zeros(rows.numel(), 3, device=device),
                            torch.zeros(rows.numel(), 3, device=device))[0]
      lib_q = library["root_quat_h"].to(device)
      err = 1.0 - (qh * lib_q).sum(1).abs()
      if float(err.max()) > 1e-4:
        raise ValueError(f"reference and library disagree (max 1-|<q,q'>| "
                         f"{float(err.max()):.2e})")
    return ref

  def rows(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Global rows of ``(clip, frame)``, the frame clamped to the clip's end."""
    f = torch.minimum(frame, self.length[clip] - 1).clamp_min(0)
    return self.start[clip] + f


def forward_difference(pos: torch.Tensor, fps: float) -> torch.Tensor:
  """Joint velocities of a ``[F, J]`` clip as SONIC's motion lib computes them
  (``gear_sonic/utils/motion_lib/torch_humanoid_batch.py``): forward
  differences, the last frame repeating the SECOND-to-last difference."""
  d = (pos[1:] - pos[:-1]) * fps
  return torch.cat([d, d[-2:-1]], 0)


def _csv_joints() -> tuple:
  # Unitree's G1 CSV joint order = mjlab's entity (MuJoCo) order.
  return (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint")


# --- SONIC's inputs, shared by the live tracker and the offline rebuild ------

def reference_window(ref: SonicReference, clip: torch.Tensor, frame: torch.Tensor,
                     dq: torch.Tensor):
  """``(pos, vel, quat_w)`` of frames ``frame + 5k`` of ``clip`` (per row,
  ``[B]``): ``[B, 10, 29]`` x2 and ``[B, 10, 4]``, the quaternion turned by the
  delta heading ``dq`` ``[B, 4]``."""
  B = clip.shape[0]
  offsets = torch.arange(NUM_FUTURE, device=clip.device) * FUTURE_STEP
  f = frame.unsqueeze(1) + offsets
  c = clip.unsqueeze(1).expand_as(f)
  rows = ref.rows(c.reshape(-1), f.reshape(-1))
  pos = ref.joint_pos[rows].view(B, NUM_FUTURE, 29)
  vel = ref.joint_vel[rows].view(B, NUM_FUTURE, 29)
  q = ref.root_quat[rows].view(B, NUM_FUTURE, 4)
  q = quat_mul(dq.unsqueeze(1).expand_as(q).reshape(-1, 4),
               q.reshape(-1, 4)).view(B, NUM_FUTURE, 4)
  return pos, vel, q


def anchor_6d(quat_w: torch.Tensor, qref: torch.Tensor) -> torch.Tensor:
  """The anchor orientation of each reference frame relative to the robot's
  pelvis ``quat_w`` ``[B, 4]``: first two columns of R(q^-1 q_ref), row-major,
  ``[B, 10, 6]``."""
  B = quat_w.shape[0]
  rel = quat_mul(quat_inv(quat_w).unsqueeze(1).expand_as(qref).reshape(-1, 4),
                 qref.reshape(-1, 4))
  return matrix_from_quat(rel)[..., :2].reshape(B, NUM_FUTURE, 6)


def proprio_entry(quat_w: torch.Tensor, ang_vel_w: torch.Tensor,
                  joint_pos_rel: torch.Tensor, joint_vel: torch.Tensor,
                  last_action: torch.Tensor):
  """One history entry: ``(ang_vel_b, joint_pos - default, joint_vel,
  last_action, projected gravity)``, joints in SONIC order. ``quat_w`` /
  ``ang_vel_w`` are the pelvis's world orientation and angular velocity."""
  down = torch.tensor([0.0, 0.0, -1.0], device=quat_w.device).expand(quat_w.shape[:-1] + (3,))
  return (quat_apply_inverse(quat_w, ang_vel_w), joint_pos_rel, joint_vel,
          last_action, quat_apply_inverse(quat_w, down))


def decoder_obs(token: torch.Tensor, hist: Sequence[torch.Tensor]) -> torch.Tensor:
  """The decoder's 994-d input from the token ``[B, 64]`` and the five
  histories ``[B, 10, d]`` (ang_vel, joint_pos, joint_vel, action, gravity)."""
  B = token.shape[0]
  return torch.cat([token] + [h.reshape(B, -1) for h in hist], 1)


#: The applied action is clipped here, as in SONIC's training env
#: (``action_clip_value``); the clipped action is also the last action it sees.
ACTION_CLIP = 20.0
#: The five history groups, in the decoder's order, and their widths.
HISTORY_KEYS = (("ang_vel", 3), ("joint_pos", 29), ("joint_vel", 29),
                ("action", 29), ("gravity", 3))


# --- the tracker ---------------------------------------------------------------

class SonicTracker:
  """Per-env SONIC state in an mjlab G1 env: the clip cursor, the delta heading
  and the 10-tick histories.

  Call order each control step (envs not tracking get the action of whatever
  else drives them)::

      tracker.start(mask, clip, frame)       # after the env (re)spawned them
      a_env = tracker.act()                  # SONIC's targets as env actions
      env.step_tensor(a_env)
      tracker.advance()                      # cursor += 1, histories roll

  ``act`` reads the CURRENT robot state, so ``start`` must run after the reset
  wrote the spawn state, and ``advance`` after the step.
  """

  def __init__(self, env, policy: SonicPolicy, ref: SonicReference):
    self.env, self.mj = env, env.mj
    self.robot = self.mj.scene["robot"]
    self.policy, self.ref = policy, ref
    dev = self.dev = env.device
    n = self.n = env.num_envs
    names = list(self.robot.joint_names)
    self.to_sonic = torch.as_tensor([names.index(j) for j in SONIC_JOINTS], device=dev)
    term = self.mj.action_manager.get_term("joint_pos")
    ids = torch.as_tensor(term.target_ids, device=dev).long().reshape(-1)
    if ids.numel() != 29:
      raise ValueError("the env's joint_pos action term must drive all 29 joints")
    self.act_ids = ids
    self.env_scale = self._per_joint(term.scale)
    self.env_offset = self._per_joint(term.offset)
    act_names = [names[i] for i in self.act_ids.tolist()]
    # SONIC targets, in the action term's joint order
    self.sonic_scale = torch.as_tensor(sonic_action_scale(act_names), device=dev)
    self.default = self.robot.data.default_joint_pos[0, self.act_ids].clone()
    pos_in_act = {j: i for i, j in enumerate(act_names)}
    self.sonic_to_act = torch.as_tensor([pos_in_act[j] for j in SONIC_JOINTS], device=dev)
    self.default_sonic = self.robot.data.default_joint_pos[0, self.to_sonic].clone()
    long = dict(dtype=torch.long, device=dev)
    self.clip = torch.zeros(n, **long)
    self.frame = torch.zeros(n, **long)
    self.active = torch.zeros(n, dtype=torch.bool, device=dev)
    self.dq = torch.zeros(n, 4, device=dev)
    self.dq[:, 0] = 1.0
    self.hist = [torch.zeros(n, HISTORY, d, device=dev) for _, d in HISTORY_KEYS]
    self.last_action = torch.zeros(n, 29, device=dev)       # SONIC order

  def _per_joint(self, x) -> torch.Tensor:
    t = torch.as_tensor(x, device=self.dev, dtype=torch.float32)
    if t.dim() == 0:
      t = t.expand(29)
    return t.reshape(-1, 29)[0] if t.numel() >= 29 else t

  # --- robot state, SONIC's way --------------------------------------------
  def _entry(self):
    d = self.robot.data
    # the encoder reading (mjlab's encoder_bias event), as a robot would see it
    jp = d.joint_pos_biased[:, self.to_sonic] - self.default_sonic
    return proprio_entry(d.root_link_quat_w, d.root_link_ang_vel_w, jp,
                         d.joint_vel[:, self.to_sonic], self.last_action)

  def _push_history(self, mask: Optional[torch.Tensor] = None, fill: bool = False):
    for buf, x in zip(self.hist, self._entry()):
      if fill:
        m = mask.view(-1, 1, 1)
        buf.copy_(torch.where(m, x.unsqueeze(1).expand_as(buf), buf))
      else:
        rolled = torch.cat([buf[:, 1:], x.unsqueeze(1)], 1)
        if mask is None:
          buf.copy_(rolled)
        else:
          buf.copy_(torch.where(mask.view(-1, 1, 1), rolled, buf))

  # --- episode bookkeeping --------------------------------------------------
  def start(self, mask: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor) -> None:
    """Envs in ``mask`` start tracking clip ``clip[i]`` from frame ``frame[i]``
    (per-env tensors), from the state the env is in now."""
    m = mask.reshape(-1).bool()
    if not bool(m.any()):
      return
    self.clip = torch.where(m, clip.long(), self.clip)
    self.frame = torch.where(m, frame.long(), self.frame)
    self.active |= m
    rows = self.ref.rows(self.clip, self.frame)
    q_r = self.robot.data.root_link_quat_w
    dq = quat_mul(yaw_quat(q_r), quat_inv(yaw_quat(self.ref.root_quat[rows])))
    self.dq = torch.where(m.unsqueeze(1), dq, self.dq)
    self.last_action = torch.where(m.unsqueeze(1), torch.zeros_like(self.last_action),
                                   self.last_action)
    self._push_history(m, fill=True)

  def stop(self, mask: torch.Tensor) -> None:
    self.active &= ~mask.reshape(-1).bool()

  def advance(self) -> None:
    """The env stepped: move the clip cursor and roll the histories (every
    env; an env that is not tracking is refilled by its next ``start``)."""
    self.frame = self.frame + self.active.long()
    self._push_history()

  # --- the policy -----------------------------------------------------------
  @torch.no_grad()
  def act(self) -> torch.Tensor:
    """SONIC's action for every env, as the env's own action vector
    (``(target - offset) / scale`` in the action term's order). Also kept:
    ``self.obs`` (the decoder input), ``self.raw`` (the clipped action, SONIC
    order) and ``self.target`` (joint targets, action order)."""
    pos, vel, qref = reference_window(self.ref, self.clip, self.frame + REF_LEAD, self.dq)
    anchor = anchor_6d(self.robot.data.root_link_quat_w, qref)
    self.obs = decoder_obs(self.policy.encode(pos, vel, anchor), self.hist)
    raw = self.policy.decode(self.obs).clamp(-ACTION_CLIP, ACTION_CLIP)
    self.raw = raw
    self.last_action = raw
    raw_act = torch.empty_like(raw)
    raw_act[:, self.sonic_to_act] = raw
    self.target = self.default + raw_act * self.sonic_scale
    return self.env_action(self.target)

  def env_action(self, target: torch.Tensor) -> torch.Tensor:
    """The env action whose joint targets are ``target`` (action order)."""
    return (target - self.env_offset) / self.env_scale

  def snapshot(self, ids: torch.Tensor) -> dict:
    """What the offline rebuild needs besides the trajectory, for envs ``ids``:
    the five histories (as of now) and the delta heading."""
    out = {f"hist_{k}": h[ids].clone() for (k, _), h in zip(HISTORY_KEYS, self.hist)}
    out["dq"] = self.dq[ids].clone()
    return out


# --- the offline rebuild -------------------------------------------------------

@torch.no_grad()
def rebuild_inputs(policy: SonicPolicy, ref: SonicReference, traj: dict, ep: dict,
                   meta: dict, with_token: bool = True) -> dict:
  """SONIC's exact inputs at every step of harvested trajectories
  (``examples/harvest_g1_sonic_trajectories.py``), for a BATCH of episodes.

  ``traj`` holds the ``traj/*`` tensors ``[B, T, ...]`` and ``ep`` the ``ep/*``
  ones ``[B, ...]`` of the same episodes, without their prefixes, on one
  device. Returns ``hist_<key>`` ``[B, T, 10, d]`` for the five history groups
  (:data:`HISTORY_KEYS`) and, with ``with_token``, ``token`` ``[B, T, 64]``
  and ``obs`` ``[B, T, 994]``: what the decoder read before applying
  ``traj["target"][:, j]``. Steps past an episode's ``length`` are garbage.

  Step 0's histories are the ones the harvester stored (``ep["hist_*"]``);
  each later step appends the entry of its own recorded state. The last
  action is the previous step's target mapped back
  (``(target - default) / scale``). The reference window starts at
  ``ref_frame + REF_LEAD`` like the live tracker's."""
  dev = traj["joint_pos"].device
  B, T = traj["joint_pos"].shape[:2]
  names = list(meta["joint_names"])
  to_s = torch.as_tensor([names.index(j) for j in SONIC_JOINTS], device=dev)
  tnames = list(meta["target_joint_names"])
  t_to_s = torch.as_tensor([tnames.index(j) for j in SONIC_JOINTS], device=dev)
  default_e = torch.as_tensor(meta["default_joint_pos"], device=dev)
  scale_s = torch.as_tensor(sonic_action_scale(SONIC_JOINTS), device=dev)
  jp = traj["joint_pos"] + ep["encoder_bias"].unsqueeze(1)
  jp_rel = (jp - default_e)[..., to_s]
  tgt = traj["target"][..., t_to_s]
  last = torch.zeros_like(tgt)
  last[:, 1:] = (tgt[:, :-1] - default_e[to_s]) / scale_s
  entry = proprio_entry(traj["root_quat"], traj["ang_vel"], jp_rel,
                        traj["joint_vel"][..., to_s], last)
  out = {}
  for (k, d), e in zip(HISTORY_KEYS, entry):
    seq = torch.cat([ep[f"hist_{k}"], e[:, 1:]], 1)           # [B, 10 + T - 1, d]
    out[f"hist_{k}"] = seq.unfold(1, HISTORY, 1).permute(0, 1, 3, 2)
  if with_token:
    clip = ep["clip"].unsqueeze(1).expand(B, T).reshape(-1)
    frame = traj["ref_frame"].clamp_min(0).reshape(-1) + REF_LEAD
    dq = ep["dq"].unsqueeze(1).expand(B, T, 4).reshape(-1, 4)
    pos, vel, qref = reference_window(ref, clip, frame, dq)
    quat = traj["root_quat"].reshape(-1, 4)
    quat = torch.where(torch.isnan(quat), torch.tensor([1.0, 0, 0, 0], device=dev), quat)
    token = policy.encode(pos, vel, anchor_6d(quat, qref))
    out["token"] = token.view(B, T, TOKEN_DIM)
    hist = [out[f"hist_{k}"].reshape(B * T, HISTORY, d) for k, d in HISTORY_KEYS]
    out["obs"] = decoder_obs(token, hist).view(B, T, -1)
  return out
