"""Unitree G1 blind flat-terrain velocity walker (mode=CUMULATIVE pi_task).

The humanoid counterpart of ``go2_walker_flat``, and the policy the G1 safety
filter wraps: plain reward-maximizing RL on the env's dense reward stack (the
CUMULATIVE backup, reward + gamma*V'), i.e. stock SB3 PPO. It is deliberately
safety-oblivious — proprioception only, no margins, no notion of falling beyond
what the dense terms already penalize. The reach-avoid twins (``g1_stabilize`` /
``g1_locomote``) supply the certificate V(s) and the fallback at deploy time
(``examples/eval.py --nominal``).

Env cfg: envs/velocity/g1.py.
"""

from __future__ import annotations

from ..registry import CUMULATIVE, TaskSpec, register
from .g1_stabilize import G1_BRIDGE_KWARGS, G1_CTRL_DIM


def register_all() -> None:
  from robot_safety_sandbox.envs.velocity.g1 import (
    unitree_g1_flat_env_cfg, unitree_g1_flat_fast_env_cfg)

  register(TaskSpec(
    task_id="g1_walker_flat", cfg_builder=unitree_g1_flat_env_cfg,
    mode=CUMULATIVE, ctrl_dim=G1_CTRL_DIM,
    kwargs=dict(G1_BRIDGE_KWARGS),
    description="Blind flat-terrain G1 velocity walker (dense reward, stock "
                "SB3 PPO). Task policy pi_task for the G1 filter experiments; "
                "train with train.py --family on_policy."))
  # Same env and obs/action spaces as the walker above; only the command box
  # differs: vx fixed to (-0.5, 1.5) from step 0 instead of a curriculum that
  # the stock budget barely reaches.
  register(TaskSpec(
    task_id="g1_walker_flat_fast", cfg_builder=unitree_g1_flat_fast_env_cfg,
    mode=CUMULATIVE, ctrl_dim=G1_CTRL_DIM,
    kwargs=dict(G1_BRIDGE_KWARGS),
    description="g1_walker_flat with the forward command fixed to vx in "
                "(-0.5, 1.5) m/s, vy in (-0.5, 0.5) from the first iteration "
                "(no command curriculum)."))
