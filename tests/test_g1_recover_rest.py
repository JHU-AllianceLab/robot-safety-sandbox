"""The rest termination (envs/g1_recover/rest.py) and the push_fast task.

Simulator-free (CPU):

  * RestReached fires only after ``l >= 0`` has held for ``dwell_s`` without a
    break, never before ``arm_s`` of episode time, and a reset clears the dwell;
  * the recover cfg registers it as a TRUNCATION (``time_out=True``) only when
    asked;
  * g1_recover_push_fast registers avoid-only, single-player, with the push
    landing 0.1-0.4 s after the spawn.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox.envs.g1_recover.rest import RestReached  # noqa: E402

DT = 0.02


def _term(n=4, dwell_s=0.1, arm_s=0.0):
  l = torch.zeros(n)
  env = SimpleNamespace(step_dt=DT, num_envs=n, device="cpu",
                        episode_length_buf=torch.zeros(n, dtype=torch.long))
  cfg = SimpleNamespace(params=dict(dwell_s=dwell_s, arm_s=arm_s,
                                    margin_fn=lambda e: (torch.ones(n), l)))
  return RestReached(cfg, env), env, l


def test_fires_after_an_unbroken_dwell_only():
  term, env, l = _term(dwell_s=0.1)            # 5 steps
  l[:] = 1.0
  for _ in range(4):
    assert not bool(term(env).any())
    env.episode_length_buf += 1
  assert bool(term(env).all())                 # 5th consecutive step inside
  # a single step outside restarts the dwell
  l[0] = -0.1
  out = term(env)
  assert not bool(out[0]) and bool(out[1:].all())
  l[0] = 1.0
  for _ in range(4):
    assert not bool(term(env)[0])
  assert bool(term(env)[0])


def test_arm_time_and_reset():
  term, env, l = _term(dwell_s=0.1, arm_s=0.2)  # arm at step 10
  l[:] = 1.0
  for step in range(10):
    env.episode_length_buf[:] = step
    assert not bool(term(env).any()), step
  env.episode_length_buf[:] = 10
  assert bool(term(env).all())
  term.reset(torch.tensor([1, 2]))
  out = term(env)
  assert bool(out[0]) and not bool(out[1]) and not bool(out[2]) and bool(out[3])
  term.reset(None)
  assert not bool(term(env).any())


def test_recover_cfg_registers_rest_as_a_truncation_only_when_asked():
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  cfg = g1_recover_env_cfg(rest_terminate=True, rest_dwell_s=0.3, rest_arm_s=0.5,
                           episode_length_s=3.0)
  t = cfg.terminations["recovered"]
  assert t.time_out is True and t.func is RestReached
  assert t.params == dict(dwell_s=0.3, arm_s=0.5)
  assert cfg.episode_length_s == 3.0
  assert "recovered" not in g1_recover_env_cfg().terminations


def test_push_fast_task_registers():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.registry import AVOID, algo_name, spec
  s = spec("g1_recover_push_fast")
  assert s.mode == AVOID and not s.supports_adversary
  assert s.kwargs["random_push"]["first_delay_s"] == (0.1, 0.4)
  assert algo_name("g1_recover_push_fast", adversary=False, family="on_policy") == "SafetyPPO1P"
  with pytest.raises(ValueError):
    algo_name("g1_recover_push_fast", adversary=True, family="on_policy")


def test_recover_critic_group_carries_pelvis_height_and_actor_is_untouched():
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg, pelvis_height
  cfg = g1_recover_env_cfg()
  assert cfg.observations["critic"].terms["pelvis_height"].func is pelvis_height
  assert "pelvis_height" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
    root_link_pos_w=torch.tensor([[0.1, 0.2, 0.76], [0.0, 0.0, 0.40]])))})
  assert torch.allclose(pelvis_height(env), torch.tensor([[0.76], [0.40]]))
