"""robot_safety_sandbox: mjlab safety-benchmark environments for safety_sb3.

    from robot_safety_sandbox import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)

Tasks register lazily on import; phase-1 compat tasks additionally need their
source repo on sys.path (see tasks/*.py + MIGRATION.md).
"""

from .base import MjlabNumpySafetyEnv, MjlabTensorSafetyEnv, build_task_cfg
from .registry import (
  AVOID, CUMULATIVE, FAMILIES, MODES, REACH_AVOID, SAFETY_MODES,
  TaskSpec, algo_name, list_tasks, make_numpy, make_tensor, register, spec)

from .tasks import digit_safety as _digit_safety
from .tasks import go2_crawl as _go2_crawl
from .tasks import go2_gap as _go2_gap
from .tasks import go2_stabilize as _go2_stabilize
from .tasks import g1_stabilize as _g1_stabilize
from .tasks import g1_arm_adversary as _g1_arm_adversary
from .tasks import g1_arm_random as _g1_arm_random
from .tasks import g1_recover as _g1_recover
from .tasks import g1_waist as _g1_waist
from .tasks import go2_crawl_twins as _go2_crawl_twins
from .tasks import go2_gap_brake_or_jump as _go2_gap_brake_or_jump
from .tasks import go2_filtered as _go2_filtered
from .tasks import go2_low_bar as _go2_low_bar
from .tasks import go2_tunnel as _go2_tunnel
from .tasks import car_goal as _car_goal
from .tasks import go2_velocity as _go2_velocity
from .tasks import g1_velocity as _g1_velocity

# safety tasks (margins + safety_sb3 learners)
_go2_gap.register_all()
_go2_crawl.register_all()
_go2_stabilize.register_all()
_g1_stabilize.register_all()  # the humanoid twin: G1 stand / walk vs an adversarial torso force
_g1_arm_adversary.register_all()  # same, but the adversary COMMANDS THE ARMS (and pays for self-collision)
_g1_arm_random.register_all()  # same arm split, single-player: the arms swing at random (curriculum on magnitude)
_g1_recover.register_all()  # avoid-only fallback spawned from harvested walker / pushed / synthetic states
_g1_waist.register_all()  # reach-avoid-stay to a waist posture + velocity goal: stay (avoid) and reach (RA) twins
_digit_safety.register_all()
_go2_crawl_twins.register_all()
_go2_gap_brake_or_jump.register_all()  # split test: harvested-state RA vs avoid twins
_go2_filtered.register_all()  # velocity walker trained inside a safety filter (PORL)
_go2_low_bar.register_all()  # 2nd RA-liveness benchmark: virtual low-bar crawl twins
_go2_tunnel.register_all()  # crawl campaign new formulation: uniform randomized tunnel twins
_car_goal.register_all()  # bicycle5d analog: diff-drive car reach-avoid (tutorial)
# mode="cumulative" task policies (dense reward + stock SB3) — what filters wrap.
# go2_crawl_walk{,_video} register alongside their safety twins in go2_crawl.
_go2_velocity.register_all()  # go2_walker_flat: the blind flat-terrain pi_task
_g1_velocity.register_all()  # g1_walker_flat: the same, on the humanoid

__all__ = [
  "MjlabTensorSafetyEnv", "MjlabNumpySafetyEnv", "build_task_cfg",
  "TaskSpec", "register", "spec", "list_tasks", "make_tensor", "make_numpy",
  "algo_name",
  "AVOID", "REACH_AVOID", "CUMULATIVE", "MODES", "SAFETY_MODES", "FAMILIES",
]
