"""The lab's Unitree G1 as identified on the real robot, and its motor heat model.

Both come from the sysID repo (``G1-robot-model-with-SysID``, JHU Alliance Lab),
which is not a pip package. This module reads it in place, without copying, and
PINS it: the files that set the robot and the temperatures must hash to the
values of a known VERSION. The repo is a living one ("values will change"); a
pull that changes them would otherwise change a run silently between its
seeds, or between training and evaluation.

Two versions are pinned, side by side (each loaded under its own module name,
so one process may hold both):

* ``"0.4.0"`` (commit 69f2fc6; the DEFAULT, every task built before 2026-11-21):
  mjlab's catalogue body, heat model v0.4 with ``sim_heat_scale`` (the copper
  heat of simulated torques scaled per motor, ankles x2.8-6.2). Read from
  ``$G1_SYSID_V04_DIR`` (default ``~/G1-robot-model-with-SysID-v0.4``, a
  ``git archive`` of that commit, made when the main checkout moved on).
* ``"0.6.0"`` (commit 4259dc6, the ``*_v6`` tasks): the IDENTIFIED BODY
  (``g1_sysid.body.apply_identified_body``: limb masses / COMs fitted to
  measured gravity torques, joint friction measured in motion, the ankle
  linkage's corner stops, foot contact, 37.40 kg weighed) and NO heat scale
  (removed upstream: simulated torques go into the heat model as they are).
  The heat parameters are v0.4's. Read from ``$G1_SYSID_DIR`` (default
  ``~/G1-robot-model-with-SysID``, the lab repo's main checkout).

To move to a newer model, add its pins to :data:`VERSIONS` (and say so in the
experiment's notes).

What the robot is (``get_g1_sysid_robot_cfg``, the sysID repo's
``integrations/mjlab/g1_sysid_constants.py`` re-built here, because that file
expects to be copied into mjlab's asset zoo):

* mjlab's G1 spec with the waist ranges measured on the robot and the collision
  capsules refitted to the measured joint stops (a chest capsule, slimmer hip /
  wrist / hand capsules); from v0.6 also the identified body (above);
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
import functools
import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

#: every pinned sysID version: where it lives, its commit, the sha256 of every
#: file that sets the robot or the temperatures, and what it changes
VERSIONS = {
  "0.4.0": dict(
    commit="69f2fc6", dir_env="G1_SYSID_V04_DIR",
    default_dir="~/G1-robot-model-with-SysID-v0.4",
    identified_body=False, sim_heat_scale=True,
    pins={
      "models/thermal_params.json":
        "144bc09b89e3f6f586f32a57942db585c992fa7463e8fc17ae1f6466fdcef104",
      "models/sim_heat_scale.json":
        "4241be965cb4e5e3578f99ef33d536867ce42726279012ece5e046b31ad014d2",
      "integrations/mjlab/g1_sysid_params.py":
        "dd9f1ed3b3576cfeb52a975f795168ce2e9f588d9da7d71d1bc666f2b5dc2b1e",
      "g1_sysid/thermal.py":
        "6cd9fb2baf1ff8efa71e067d64dba45678ffefa6d1e025e26597aa105a6daa20",
    }),
  "0.6.0": dict(
    commit="4259dc6", dir_env="G1_SYSID_DIR",
    default_dir="~/G1-robot-model-with-SysID",
    identified_body=True, sim_heat_scale=False,
    pins={
      "models/thermal_params.json":
        "65ab3917a5edec204308ba3f4cbae889821f02ae43f9317f38a08cbbb65ef0ce",
      "integrations/mjlab/g1_sysid_params.py":
        "dd9f1ed3b3576cfeb52a975f795168ce2e9f588d9da7d71d1bc666f2b5dc2b1e",
      "g1_sysid/thermal.py":
        "4c0f935c988865d04b9d1516fd2bcf061e5ccd71bbf699e8db9a35f852c25505",
      "g1_sysid/body.py":
        "fedce3ce6ea465670beee6012e4b379b523650422455cb1cc6962fae862586f6",
      "models/body_params.json":
        "287011c5d60c99d0f464563225bc17a4a920986f2da1ea05927ab55ef2c703a7",
      "g1_sysid/__init__.py":
        "5f24fa30bef5d04f3c8c6aa644c3ccf519fdb6bf6c24dea83728d4b8d5aaf741",
    }),
  # v0.6.1 (2026-11-22): only the heat model's waist linkage sign changed (-1 -> +1: which waist
  # motor carries (pitch + roll) / 2); the body files hash as in v0.6.0. Same pins as CoolMimic's
  # coolmimic/sailr/sysid.py, which defers to this entry once it exists.
  "0.6.1": dict(
    commit="8758984", dir_env="G1_SYSID_V061_DIR",
    default_dir="~/G1-robot-model-with-SysID-v0.6.1",
    identified_body=True, sim_heat_scale=False,
    pins={
      "models/thermal_params.json":
        "bf2dfdccf1958d111e2cbfe477f1b19dda50b7fe22eca3cabaa43b398f8151b1",
      "integrations/mjlab/g1_sysid_params.py":
        "dd9f1ed3b3576cfeb52a975f795168ce2e9f588d9da7d71d1bc666f2b5dc2b1e",
      "g1_sysid/thermal.py":
        "3cf8b7de6d3634f64fa7231aa857d250020ced02fb6aa0b18f0f94b4db8ddf77",
      "g1_sysid/body.py":
        "fedce3ce6ea465670beee6012e4b379b523650422455cb1cc6962fae862586f6",
      "models/body_params.json":
        "287011c5d60c99d0f464563225bc17a4a920986f2da1ea05927ab55ef2c703a7",
      "g1_sysid/__init__.py":
        "8de437b8ced317014db3db703cfad98f224093dc488f6850cdef26c72c0299a5",
    }),
}
#: the version a builder uses when none is named (every pre-v6 task)
DEFAULT_VERSION = "0.4.0"
#: backward compatible names: the default version's pins and directory
PINNED_VERSION = DEFAULT_VERSION
PINNED = VERSIONS[DEFAULT_VERSION]["pins"]
#: battery voltage the drives see by default (the sysID repo's NOMINAL_V: the
#: voltage Unitree's torque-speed curves are taken at)
DEFAULT_V_BUS = 52.0


def _version(version: str | None) -> str:
  v = DEFAULT_VERSION if version is None else str(version)
  if v not in VERSIONS:
    raise ValueError(f"no pinned G1 sysID version {v!r} (pinned: {sorted(VERSIONS)})")
  return v


def sysid_dir(version: str | None = None) -> Path:
  """Where the sysID repo of ``version`` is read from."""
  info = VERSIONS[_version(version)]
  return Path(os.environ.get(info["dir_env"], os.path.expanduser(info["default_dir"])))


#: the default version's directory (backward compatible)
SYSID_DIR = sysid_dir(DEFAULT_VERSION)


def version_info(version: str | None = None) -> dict:
  """``{commit, identified_body, sim_heat_scale, ...}`` of a pinned version."""
  return dict(VERSIONS[_version(version)])


def _check_pins(root: Path | None = None, version: str | None = None) -> None:
  v = _version(version)
  info = VERSIONS[v]
  root = sysid_dir(v) if root is None else Path(root)
  if not root.is_dir():
    raise FileNotFoundError(
      f"the G1 sysID repo v{v} is not at {root}; set ${info['dir_env']} to a checkout "
      f"of its commit {info['commit']}")
  bad = []
  for rel, want in info["pins"].items():
    p = root / rel
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
    if got != want:
      bad.append(f"{rel}: {got[:12]} (pinned {want[:12]})")
  if bad:
    raise RuntimeError(
      f"the G1 sysID repo at {root} is not the pinned v{v} model:\n  "
      + "\n  ".join(bad) + f"\nCheck out its commit {info['commit']}, or update "
      f"VERSIONS in {__file__} on purpose.")


def _tag(version: str) -> str:
  return version.replace(".", "")


@lru_cache(maxsize=None)
def sysid_params(version: str | None = None):
  """The sysID repo's generated parameter module (``g1_sysid_params.py``) of
  ``version``: JOINT_RANGE, NOMINAL_V, MOTOR, GEOM_EDITS, MOTOR_OF_JOINT."""
  v = _version(version)
  _check_pins(version=v)
  path = sysid_dir(v) / "integrations" / "mjlab" / "g1_sysid_params.py"
  spec = importlib.util.spec_from_file_location(f"_g1_sysid_params_v{_tag(v)}", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


@lru_cache(maxsize=None)
def _g1_sysid_pkg(version: str | None = None):
  """The sysID repo's ``g1_sysid`` package of ``version``, imported under its own
  module name (``_g1_sysid_v040`` ...) so two versions can live in one process."""
  v = _version(version)
  _check_pins(version=v)
  root = sysid_dir(v) / "g1_sysid"
  name = f"_g1_sysid_v{_tag(v)}"
  if name in sys.modules:
    mod = sys.modules[name]
  else:
    spec = importlib.util.spec_from_file_location(
      name, root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
      spec.loader.exec_module(mod)
    except BaseException:
      sys.modules.pop(name, None)
      raise
  if mod.__version__ != v:
    raise RuntimeError(f"g1_sysid {mod.__version__} at {mod.__file__}: pinned {v}")
  return mod


def motor_thermal_cls(version: str | None = None):
  """``g1_sysid.MotorThermal`` (the 3-node heat model) of the pinned sysID repo."""
  return _g1_sysid_pkg(version).MotorThermal


def motor_thermal(version: str | None = None, like=None):
  """The heat model of ``version`` as simulated torques need it: v0.4 with its
  per-motor ``sim_heat_scale``, v0.6+ unscaled (the scale was removed upstream)."""
  v = _version(version)
  return motor_thermal_cls(v).from_json(like=like,
                                        sim_heat_scale=VERSIONS[v]["sim_heat_scale"])


def linkages(version: str | None = None) -> dict:
  """``{name: (pitch slot, roll slot)}`` of the motor pairs behind the ankle /
  waist linkages, in Unitree motor order (``g1_sysid.LINKAGES``)."""
  return dict(_g1_sysid_pkg(version).LINKAGES)


def get_spec(version: str | None = None):
  """mjlab's G1 spec with the measured waist ranges and the refitted collision
  geometry (the sysID repo's ``get_spec``); for a version with the identified
  body (v0.6+) also ``g1_sysid.body.apply_identified_body`` (its
  ``get_identified_spec``)."""
  import mujoco
  import numpy as np
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
  v = _version(version)
  p = sysid_params(v)
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
  if VERSIONS[v]["identified_body"]:
    _g1_sysid_pkg(v).apply_identified_body(spec)
  return spec


def dcmotor_actuators(v_bus: float = DEFAULT_V_BUS, version: str | None = None) -> tuple:
  """One ``BuiltinDcMotorActuatorCfg`` per motor type (the sysID repo's
  ``dcmotor_actuators``). ``v_bus``: battery voltage at the drives (52.8 V full,
  49 V at half charge, measured); it caps the drive voltage."""
  from mjlab.actuator import (BuiltinDcMotorActuatorCfg, DcMotorInputMode,
                              DcMotorPhysicalParams)
  p = sysid_params(version)
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


def get_g1_sysid_robot_cfg(v_bus: float = DEFAULT_V_BUS, version: str | None = None):
  """The lab's G1 of sysID ``version`` (default :data:`DEFAULT_VERSION`):
  measured joint ranges and collision geometry (+ the identified body from
  v0.6), dcmotor actuators at battery voltage ``v_bus``. Same body, joint, geom
  and site names as mjlab's G1, and the same kp / kd / peak torque / armature
  per joint, so ``G1_ACTION_SCALE`` (0.25 x peak / kp) is unchanged."""
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
  v = _version(version)
  cfg = g1_constants.get_g1_robot_cfg()
  cfg.spec_fn = get_spec if v == DEFAULT_VERSION else functools.partial(get_spec, v)
  cfg.articulation = dataclasses.replace(
    g1_constants.G1_ARTICULATION, actuators=dcmotor_actuators(v_bus, v))
  return cfg


def provenance(version: str | None = None) -> dict:
  """What a run log should record about the robot / heat model it used."""
  v = _version(version)
  _check_pins(version=v)
  info = VERSIONS[v]
  return {"sysid_dir": str(sysid_dir(v)), "version": v, "commit": info["commit"],
          "identified_body": info["identified_body"],
          "sim_heat_scale": info["sim_heat_scale"], "sha256": dict(info["pins"])}
