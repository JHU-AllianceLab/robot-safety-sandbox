"""The lab's Unitree G1 as identified on the real robot, and its motor heat model.

Both come from the sysID repo (``G1-robot-model-with-SysID``, JHU Alliance Lab;
``$G1_SYSID_DIR``, default ``~/G1-robot-model-with-SysID``), which is not a pip
package. This module reads it in place, without copying, and PINS it: the files
that set the robot and the temperatures must hash to the values below (v0.4.0,
commit 69f2fc6). The repo is a living one ("values will change"); a pull that
changes them would otherwise change a run silently between its seeds, or
between training and evaluation. To move to a newer model, update ``PINNED``
(and say so in the experiment's notes).

What the robot is (``get_g1_sysid_robot_cfg``, the sysID repo's
``integrations/mjlab/g1_sysid_constants.py`` re-built here, because that file
expects to be copied into mjlab's asset zoo):

* mjlab's G1 spec with the waist ranges measured on the robot and the collision
  capsules refitted to the measured joint stops (a chest capsule, slimmer hip /
  wrist / hand capsules);
* every joint driven by MuJoCo's ``<dcmotor>`` (``BuiltinDcMotorActuatorCfg``,
  POSITION mode) with the drive's PD law at mjlab's gains, K, R, the peak
  torque, and the bus voltage as the voltage limit. Below the voltage limit it
  is exactly the PD torque of mjlab's position actuator; near it, back-EMF caps
  the torque as on the robot. MuJoCo's own dcmotor heating stays OFF: the
  temperatures come from :class:`MotorThermal` (the sysID repo's ``g1_sysid``
  package), fitted to the robot's temperature sensors.

The heat model's arrays are in Unitree MOTOR order (0..28, ``MotorThermal.names``);
mjlab orders actuators by motor type. Map by name (``<motor>_joint``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

#: where the sysID repo lives (override with $G1_SYSID_DIR)
SYSID_DIR = Path(os.environ.get("G1_SYSID_DIR",
                                os.path.expanduser("~/G1-robot-model-with-SysID")))
#: sysID repo version this sandbox is pinned to
PINNED_VERSION = "0.4.0"
#: sha256 of every file that sets the robot or the temperatures (v0.4.0, 69f2fc6)
PINNED = {
  "models/thermal_params.json":
    "144bc09b89e3f6f586f32a57942db585c992fa7463e8fc17ae1f6466fdcef104",
  "models/sim_heat_scale.json":
    "4241be965cb4e5e3578f99ef33d536867ce42726279012ece5e046b31ad014d2",
  "integrations/mjlab/g1_sysid_params.py":
    "dd9f1ed3b3576cfeb52a975f795168ce2e9f588d9da7d71d1bc666f2b5dc2b1e",
  "g1_sysid/thermal.py":
    "6cd9fb2baf1ff8efa71e067d64dba45678ffefa6d1e025e26597aa105a6daa20",
}
#: battery voltage the drives see by default (the sysID repo's NOMINAL_V: the
#: voltage Unitree's torque-speed curves are taken at)
DEFAULT_V_BUS = 52.0


def _check_pins(root: Path = SYSID_DIR) -> None:
  if not root.is_dir():
    raise FileNotFoundError(
      f"the G1 sysID repo is not at {root}; set $G1_SYSID_DIR to its checkout")
  bad = []
  for rel, want in PINNED.items():
    p = root / rel
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
    if got != want:
      bad.append(f"{rel}: {got[:12]} (pinned {want[:12]})")
  if bad:
    raise RuntimeError(
      f"the G1 sysID repo at {root} is not the pinned v{PINNED_VERSION} model:\n  "
      + "\n  ".join(bad) + "\nCheck out its commit 69f2fc6, or update PINNED in "
      f"{__file__} on purpose.")


@lru_cache(maxsize=1)
def sysid_params():
  """The sysID repo's generated parameter module (``g1_sysid_params.py``):
  JOINT_RANGE, NOMINAL_V, MOTOR, GEOM_EDITS, MOTOR_OF_JOINT."""
  _check_pins()
  path = SYSID_DIR / "integrations" / "mjlab" / "g1_sysid_params.py"
  spec = importlib.util.spec_from_file_location("_g1_sysid_params", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


@lru_cache(maxsize=1)
def _g1_sysid_pkg():
  _check_pins()
  root = str(SYSID_DIR)
  if root not in sys.path:
    sys.path.insert(0, root)
  import g1_sysid  # noqa: E402  (the sysID repo's package)
  if g1_sysid.__version__ != PINNED_VERSION:
    raise RuntimeError(f"g1_sysid {g1_sysid.__version__} at {g1_sysid.__file__}: "
                       f"pinned {PINNED_VERSION}")
  return g1_sysid


def motor_thermal_cls():
  """``g1_sysid.MotorThermal`` (v0.4 heat model) of the pinned sysID repo."""
  return _g1_sysid_pkg().MotorThermal


def linkages() -> dict:
  """``{name: (pitch slot, roll slot)}`` of the motor pairs behind the ankle /
  waist linkages, in Unitree motor order (``g1_sysid.LINKAGES``)."""
  return dict(_g1_sysid_pkg().LINKAGES)


def get_spec():
  """mjlab's G1 spec with the measured waist ranges and the refitted collision
  geometry (the sysID repo's ``get_spec``)."""
  import mujoco
  import numpy as np
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
  p = sysid_params()
  spec = g1_constants.get_spec()
  for joint, (lo, hi) in p.JOINT_RANGE.items():
    spec.joint(joint).range[:] = np.array([lo, hi])
  for name, edit in p.GEOM_EDITS.items():
    if name == "torso_chest_collision":  # new geom, same class as the other capsules
      geom = spec.body("torso_link").add_geom(
        spec.find_default("collision"), name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE)
    else:
      geom = spec.geom(name)
    if "size0" in edit:
      geom.size[0] = edit["size0"]
    if "fromto" in edit:
      geom.fromto[:] = np.array(edit["fromto"])
  return spec


def dcmotor_actuators(v_bus: float = DEFAULT_V_BUS) -> tuple:
  """One ``BuiltinDcMotorActuatorCfg`` per motor type (the sysID repo's
  ``dcmotor_actuators``). ``v_bus``: battery voltage at the drives (52.8 V full,
  49 V at half charge, measured); it caps the drive voltage."""
  from mjlab.actuator import (BuiltinDcMotorActuatorCfg, DcMotorInputMode,
                              DcMotorPhysicalParams)
  p = sysid_params()
  out = []
  for mtype, m in p.MOTOR.items():
    joints = tuple(j for j, t in p.MOTOR_OF_JOINT.items() if t == mtype)
    if not joints:
      continue
    out.append(BuiltinDcMotorActuatorCfg(
      target_names_expr=joints,
      motor_params=DcMotorPhysicalParams(kt=m["K"], ke=m["K"], resistance=m["R"]),
      mode=DcMotorInputMode.POSITION,
      stiffness=m["kp"], damping=m["kd"], voltage_limit=float(v_bus),
      effort_limit=m["peak_torque"], armature=m["armature"]))
  return tuple(out)


def get_g1_sysid_robot_cfg(v_bus: float = DEFAULT_V_BUS):
  """The lab's G1: measured joint ranges and collision geometry, dcmotor
  actuators at battery voltage ``v_bus``. Same body, joint, geom and site names
  as mjlab's G1, and the same kp / kd / peak torque / armature per joint, so
  ``G1_ACTION_SCALE`` (0.25 x peak / kp) is unchanged."""
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
  cfg = g1_constants.get_g1_robot_cfg()
  cfg.spec_fn = get_spec
  cfg.articulation = dataclasses.replace(
    g1_constants.G1_ARTICULATION, actuators=dcmotor_actuators(v_bus))
  return cfg


def provenance() -> dict:
  """What a run log should record about the robot / heat model it used."""
  _check_pins()
  return {"sysid_dir": str(SYSID_DIR), "version": PINNED_VERSION,
          "commit": "69f2fc6", "sha256": dict(PINNED)}
