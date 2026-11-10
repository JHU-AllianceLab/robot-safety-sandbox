"""Unitree G1 velocity environment configurations.

The humanoid sibling of :mod:`robot_safety_sandbox.envs.velocity.go2`: same
vendored velocity task (``envs/velocity/env_cfg.py``), different robot. Two
things differ from the quadruped file and both are structural, not cosmetic:

* the ROBOT ASSET is stock mjlab (``mjlab.asset_zoo.robots.get_g1_robot_cfg``).
  The Go2 is vendored into ``envs/assets/go2`` because mjlab's asset zoo ships
  only the Go1; the G1 it ships natively, so there is nothing to vendor.
* the ACTION SCALE is PER-JOINT (``G1_ACTION_SCALE`` = 0.25 * effort / stiffness
  per actuator group) rather than the quadruped's single 0.25. A G1 knee gets
  0.35 rad per unit action, a hip pitch 0.55 — so the bridge's ``ctrl_gain``
  stays at 1.0 for G1 tasks and the mjlab scale IS the joint authority (see
  ``tasks/g1_stabilize.py``).

The reward/sensor wiring follows mjlab's own G1 velocity config, retargeted onto
the terms this fork actually defines (it predates mjlab's ``upright`` /
``foot_swing_height`` / ``air_time`` split and keeps ``body_orientation_l2``).
"""

from typing import Literal

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from robot_safety_sandbox.envs.velocity import mdp
from robot_safety_sandbox.envs.velocity.env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

#: The two foot sites (foot-height / clearance / slip terms read these).
FOOT_SITES = ("left_foot", "right_foot")
#: The sole collision geoms: left_foot1..7_collision, right_foot1..7_collision.
FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis (the floating base).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    # Subtree, not geom: the sole is 7 geoms per foot and the gait terms want
    # ONE contact per foot, so let the sensor reduce the whole ankle subtree.
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # ANY non-foot body touching the ground is a fall for a humanoid — a knee, a
  # hand, the torso. This sensor is the contact term of the safety margin g
  # (envs/g1_stabilize) AND the illegal_contact termination, so the two agree
  # on what "fell" means.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Every collision geom...
      pattern=r".*_collision$",
      # ...except the soles.
      exclude=FOOT_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # Per-joint authority (0.25 * effort_limit / stiffness), NOT the quadruped's
  # flat 0.25. Pair with ctrl_gain=1.0 on the bridge.
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0
  # 720p offscreen clips (mjlab's ViewerConfig default is 320x240, which is a
  # thumbnail for judging humanoid footwork). Set on the cfg the way
  # envs/car_goal/env_cfg.py does, because the OffscreenRenderer reads width and
  # height ONCE, when the env is constructed -- a callback cannot raise it after
  # the fact the way it can with max_extra_envs.
  #
  # Cost: 1.1 ms/frame at this size on an RTX 5080, i.e. a 20 s clip in ~1 s --
  # but ONLY with a GPU GL context. MuJoCo picks its backend at import: with
  # MUJOCO_GL unset it may bind GLFW and land on Mesa's llvmpipe SOFTWARE
  # rasterizer, where the same frame takes ~450 ms (a 400x difference, and the
  # reason a 20 s clip can cost 8 minutes). Check with glGetString(GL_RENDERER);
  # if it says llvmpipe, export:
  #   MUJOCO_GL=egl
  #   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  # (plain MUJOCO_GL=egl may select an integrated GPU -- ~22 ms/frame here.)
  cfg.viewer.width, cfg.viewer.height = 1280, 720

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15  # draw the command arrow above the head

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = FOOT_SITES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Posture stds, from mjlab's G1 velocity config:
  # - knees/hip_pitch loosest (natural leg bending during stride);
  # - hip roll/yaw tighter (no lateral sway);
  # - ankle roll very tight (balance), ankle pitch looser (foot clearance);
  # - waist roll/pitch tight (torso upright);
  # - shoulders/elbows moderate (arm swing), wrists loose (balance-irrelevant).
  # Running values are ~1.5-2x walking to accommodate the larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  # Biped gait: the two feet alternate (the quadruped's trot is [0, .5, .5, 0]).
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  # 64 (the Go2 flat budget) is not enough here: the G1 carries 14 sole geoms
  # plus 19 non-foot collision geoms, and nonfoot_ground_touch matches all 19
  # against the terrain.
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


#: Forward command box of the fast walker, m/s: back to 0.5, forward to 1.5.
G1_FAST_LIN_VEL_X = (-0.5, 1.5)
#: Lateral box, m/s: the stock stage-1 value, i.e. what g1_walker_flat trains on.
G1_FAST_LIN_VEL_Y = (-0.5, 0.5)


def unitree_g1_flat_fast_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 flat terrain with the forward command widened to (-0.5, 1.5) m/s.

  The stock flat cfg only reaches its wide box through the ``command_vel``
  curriculum stage at ``common_step_counter > 5000 * 24`` -- PPO iteration 5000
  at n_steps=24, which the 5e8-step g1_walker_flat budget (~5087 iterations at
  4096 envs) barely passes, so that walker effectively trains on vx in
  (-0.5, 1.0). Here the curriculum is dropped and the box is fixed from the
  first reset, so every iteration samples it regardless of the budget.

  Yaw rate is left alone: (-1, 1) in training (the base cfg and stage 1 agree),
  and the play cfg's gentler (-0.7, 0.7).
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.curriculum.pop("command_vel", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = G1_FAST_LIN_VEL_X
  twist_cmd.ranges.lin_vel_y = G1_FAST_LIN_VEL_Y

  return cfg
