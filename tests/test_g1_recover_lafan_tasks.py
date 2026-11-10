"""The LAFAN recovery tasks: their reach target, their registration, and the
trainer's random_push passthrough that configures their push.

Simulator-free (CPU):

  * the TARGET -- ``targets.make_l_stance`` is g1_stabilize's ``l_stance`` value
    for value at its tolerances, moves with the tolerances, and with
    ``gate_on_push`` reads -1 wherever the bridge's ``_zoo_push_quiet`` is False
    (a missing flag reads as quiet);
  * the REGISTRATION -- g1_recover_lafan is avoid-only, the two _ra twins are
    reach-avoid with end_criterion "failure", every one single-player with
    push_fast's push timing capped at one push per episode; each RA task's rest
    truncation dwells in the SAME gated target as its margin_fn;
  * the RUN CONFIG -- ``random_push:`` / ``--random-push KEY=VAL`` resolve onto
    ``args.random_push`` (CLI wins per key), dump and reload unchanged, merge
    onto the registration, and are refused for a task without pushes, for a key
    RandomPush does not take, and by a trainer that does not read them.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

from robot_safety_sandbox.envs.g1_recover.targets import GATED_L, make_l_stance  # noqa: E402

RA_TASKS = {"g1_recover_lafan_ra": (0.25, 0.30),
            "g1_recover_lafan_ra_loose": (0.5, 0.6)}
LAFAN_TASKS = ("g1_recover_lafan", *RA_TASKS)


def _env(pos, grav, lin, ang, quiet=None):
  """A fake mjlab env carrying only what g_fall / l_stance read."""
  data = SimpleNamespace(
    root_link_pos_w=torch.as_tensor(pos, dtype=torch.float32),
    projected_gravity_b=torch.as_tensor(grav, dtype=torch.float32),
    root_link_lin_vel_b=torch.as_tensor(lin, dtype=torch.float32),
    root_link_ang_vel_b=torch.as_tensor(ang, dtype=torch.float32))
  env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
  if quiet is not None:
    env._zoo_push_quiet = torch.as_tensor(quiet)
  return env


def _random_env(n=256, seed=0, quiet=None):
  g = torch.Generator().manual_seed(seed)
  pos = torch.rand(n, 3, generator=g) * torch.tensor([2.0, 2.0, 0.6]) + torch.tensor(
    [-1.0, -1.0, 0.4])
  grav = torch.randn(n, 3, generator=g) * 0.3 + torch.tensor([0.0, 0.0, -1.0])
  grav = grav / grav.norm(dim=1, keepdim=True)
  lin = torch.randn(n, 3, generator=g) * 0.4
  ang = torch.randn(n, 3, generator=g) * 0.5
  # the first half near the stance (in the band, upright, slow), so the target
  # is met by some states and missed by others
  h = n // 2
  pos[:h, 2] = 0.70 + 0.05 * torch.randn(h, generator=g)
  grav[:h] = torch.tensor([0.0, 0.0, -1.0])
  lin[:h] *= 0.25
  ang[:h] *= 0.25
  return _env(pos, grav, lin, ang, quiet)


# --- the target --------------------------------------------------------------------

def test_ungated_target_at_the_stance_tolerances_is_l_stance_value_for_value():
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import ANG_TOL, LIN_TOL, l_stance
  env = _random_env()
  l = make_l_stance(LIN_TOL, ANG_TOL, gate_on_push=False)(env)
  assert torch.equal(l, l_stance(env))
  assert torch.equal(make_l_stance(gate_on_push=False)(env), l)   # the defaults
  assert bool((l >= 0).any()) and bool((l < 0).any())             # both sides seen


def test_target_moves_with_its_tolerances():
  # standing in the band, upright, moving 0.3 m/s forward and pitching 0.4 rad/s
  env = _env([[0.0, 0.0, 0.70]], [[0.0, 0.0, -1.0]], [[0.3, 0.0, 0.0]],
             [[0.0, 0.4, 0.0]])
  strict = make_l_stance(0.25, 0.30, gate_on_push=False)(env)
  loose = make_l_stance(0.5, 0.6, gate_on_push=False)(env)
  # min over the terms: the pitch rate binds in both, (tol - 0.4) / tol
  assert strict.item() == pytest.approx((0.30 - 0.4) / 0.30)
  assert loose.item() == pytest.approx((0.6 - 0.4) / 0.6)
  for bad in (dict(lin_tol=0.0), dict(ang_tol=-0.1)):
    with pytest.raises(ValueError):
      make_l_stance(**bad)
  fn = make_l_stance(0.5, 0.6, gate_on_push=True)
  assert (fn.lin_tol, fn.ang_tol, fn.gate_on_push) == (0.5, 0.6, True)


def test_gate_holds_the_target_out_of_reach_until_the_push_is_over():
  quiet = torch.zeros(256, dtype=torch.bool)
  quiet[::3] = True
  env = _random_env(quiet=quiet)
  free = make_l_stance(0.5, 0.6, gate_on_push=False)(env)
  gated = make_l_stance(0.5, 0.6, gate_on_push=True)(env)
  assert torch.equal(gated[quiet], free[quiet])
  assert bool((gated[~quiet] == GATED_L).all()) and GATED_L == -1.0
  # no flag (a task without a push process) reads as quiet
  del env._zoo_push_quiet
  assert torch.equal(make_l_stance(0.5, 0.6, gate_on_push=True)(env), free)


# --- the registration ---------------------------------------------------------------

def test_lafan_tasks_register_single_player_with_one_push_per_episode():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.random_push import RandomPush
  from robot_safety_sandbox.registry import AVOID, REACH_AVOID, algo_name, spec
  fast = spec("g1_recover_push_fast")
  expect = {"g1_recover_lafan": (AVOID, "SafetyPPO1P"),
            "g1_recover_lafan_ra": (REACH_AVOID, "ReachAvoidPPO1P"),
            "g1_recover_lafan_ra_loose": (REACH_AVOID, "ReachAvoidPPO1P")}
  pushes = []
  for tid, (mode, algo) in expect.items():
    s = spec(tid)
    assert s.mode == mode and s.end_criterion == "failure", tid
    assert not s.supports_adversary and s.ctrl_dim == 29 and s.dstb_dim == 3, tid
    assert algo_name(tid, adversary=False, family="on_policy") == algo
    with pytest.raises(ValueError):
      algo_name(tid, adversary=True, family="on_policy")
    rp = dict(s.kwargs["random_push"])
    assert rp.pop("max_pushes") == 1, tid
    assert rp == fast.kwargs["random_push"], tid      # push_fast's timing
    assert {k: v for k, v in s.kwargs.items() if k != "random_push"} == \
           {k: v for k, v in fast.kwargs.items() if k != "random_push"}, tid
    assert s.kwargs["regime_force_buckets"] == 4 and s.kwargs["dstb_mode"] == "wrench"
    RandomPush(4, 0.02, "cpu", **s.kwargs["random_push"])     # the kwargs construct
    pushes.append(s.kwargs["random_push"])
  # each task owns its push dict (a mutation in one run never leaks to another)
  assert len({id(p) for p in pushes}) == len(pushes)


def test_avoid_twin_declares_no_target_and_ra_twins_the_gated_stance():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.envs.g1_recover.env_cfg import g1_recover_env_cfg
  from robot_safety_sandbox.envs.g1_stabilize.env_cfg import g_fall
  from robot_safety_sandbox.registry import spec
  s = spec("g1_recover_lafan")
  assert s.margin_fn.has_target is False and s.margin_fn.avoid_margin is g_fall
  assert s.cfg_builder is g1_recover_env_cfg
  for tid, (lin, ang) in RA_TASKS.items():
    s = spec(tid)
    reach = s.margin_fn.reach_margin
    assert s.margin_fn.has_target is True and s.margin_fn.avoid_margin is g_fall
    assert (reach.lin_tol, reach.ang_tol, reach.gate_on_push) == (lin, ang, True)
    b = s.cfg_builder
    assert isinstance(b, functools.partial) and b.func is g1_recover_env_cfg
    assert b.keywords == dict(rest_terminate=True, rest_lin_tol=lin,
                              rest_ang_tol=ang, rest_gate_on_push=True)


_HAS_REST_TARGET = "rest_lin_tol" in inspect.signature(
  __import__("robot_safety_sandbox.envs.g1_recover.env_cfg",
             fromlist=["g1_recover_env_cfg"]).g1_recover_env_cfg).parameters


@pytest.mark.skipif(not _HAS_REST_TARGET,
                    reason="g1_recover_env_cfg has no rest-target kwargs yet")
def test_ra_rest_truncation_dwells_in_the_tasks_own_target():
  import robot_safety_sandbox  # noqa: F401
  from robot_safety_sandbox.envs.g1_recover.rest import RestReached
  from robot_safety_sandbox.registry import spec
  quiet = torch.zeros(256, dtype=torch.bool)
  quiet[1::2] = True
  env = _random_env(seed=3, quiet=quiet)
  for tid, (lin, ang) in RA_TASKS.items():
    s = spec(tid)
    t = s.cfg_builder(play=False).terminations["recovered"]
    assert t.func is RestReached and t.time_out is True, tid
    rest_fn = t.params["margin_fn"]
    reach = rest_fn.reach_margin
    assert (reach.lin_tol, reach.ang_tol, reach.gate_on_push) == (lin, ang, True)
    # the same (g, l), state for state, gate included
    g0, l0 = s.margin_fn(env)
    g1, l1 = rest_fn(env)
    assert torch.equal(g0, g1) and torch.equal(l0, l1), tid
    # a run config's env_overrides still reach the partial's baked kwargs
    t = s.cfg_builder(play=False, rest_lin_tol=0.4).terminations["recovered"]
    assert t.params["margin_fn"].reach_margin.lin_tol == 0.4
  # the avoid reference keeps today's cfg: no rest truncation unless asked
  assert "recovered" not in spec("g1_recover_lafan").cfg_builder(
    play=False).terminations


# --- the run config's random_push passthrough ----------------------------------------

def _parser(reads_push=True):
  p = argparse.ArgumentParser()
  p.add_argument("--config", default=None)
  p.add_argument("--env-override", action="append", default=None)
  if reads_push:
    p.add_argument("--random-push", action="append", default=None,
                   dest="random_push_override")
  p.add_argument("--task", required=True)
  p.add_argument("--num-envs", type=int, default=2048)
  return p


def _merge(monkeypatch, argv, reads_push=True):
  import _run_config
  monkeypatch.setattr(sys, "argv", ["train_on_policy.py", *argv])
  return _run_config.merge_config(_parser(reads_push))


def test_random_push_config_and_cli_resolve_dump_and_reload(tmp_path, monkeypatch):
  import _run_config
  cfg = tmp_path / "run.yaml"
  cfg.write_text(yaml.safe_dump({
    "task": "g1_recover_lafan", "num_envs": 64,
    "random_push": {"max_pushes": 0, "first_delay_s": [0.2, 0.3]},
    "env_overrides": {"rest_terminate": True}}))
  args = _merge(monkeypatch, ["--config", str(cfg), "--random-push", "max_pushes=2",
                              "--random-push", "duration_s=0.1"])
  assert args.random_push == {"max_pushes": 2, "first_delay_s": [0.2, 0.3],
                              "duration_s": 0.1}
  assert args.env_overrides == {"rest_terminate": True} and args.num_envs == 64
  out = tmp_path / "out"
  out.mkdir()
  _run_config.dump_config(str(out), args)
  dumped = yaml.safe_load((out / "config.yaml").read_text())
  assert dumped["random_push"] == args.random_push
  assert "random_push_override" not in dumped
  again = _merge(monkeypatch, ["--config", str(out / "config.yaml")])
  assert again.random_push == args.random_push and again.task == "g1_recover_lafan"
  # nothing set: an empty dict, omitted from the dump
  plain = _merge(monkeypatch, ["--task", "g1_recover_lafan"])
  assert plain.random_push == {}
  out2 = tmp_path / "out2"
  out2.mkdir()
  _run_config.dump_config(str(out2), plain)
  assert "random_push" not in yaml.safe_load((out2 / "config.yaml").read_text())
  with pytest.raises(SystemExit, match="random-push"):
    _merge(monkeypatch, ["--task", "t", "--random-push", "max_pushes"])


def test_a_trainer_without_the_flag_refuses_a_random_push_config(tmp_path, monkeypatch):
  cfg = tmp_path / "run.yaml"
  cfg.write_text(yaml.safe_dump({"task": "g1_recover_lafan",
                                 "random_push": {"max_pushes": 1}}))
  with pytest.raises(SystemExit, match="on-policy"):
    _merge(monkeypatch, ["--config", str(cfg)], reads_push=False)
  # and leaves args untouched when the config has no such key (or an empty one)
  for doc in ({"task": "g1_recover_lafan"},
              {"task": "g1_recover_lafan", "random_push": {}}):
    cfg.write_text(yaml.safe_dump(doc))
    args = _merge(monkeypatch, ["--config", str(cfg)], reads_push=False)
    assert not hasattr(args, "random_push") and args.task == "g1_recover_lafan"
  cfg.write_text(yaml.safe_dump({"task": "x", "random_push": [1, 2]}))
  with pytest.raises(SystemExit, match="mapping"):
    _merge(monkeypatch, ["--config", str(cfg)])


def test_resolve_random_push_merges_onto_the_registration():
  import robot_safety_sandbox  # noqa: F401
  from _run_config import resolve_random_push
  from robot_safety_sandbox.registry import spec
  reg = spec("g1_recover_lafan").kwargs["random_push"]
  before = dict(reg)
  assert resolve_random_push("g1_recover_lafan", reg, {}) is None
  merged = resolve_random_push("g1_recover_lafan", reg,
                               {"max_pushes": None, "first_delay_s": [0.3, 0.3]})
  assert merged == {**before, "max_pushes": None, "first_delay_s": [0.3, 0.3]}
  assert reg == before                                    # registration untouched
  with pytest.raises(SystemExit, match="without random pushes"):
    resolve_random_push("g1_recover", spec("g1_recover").kwargs.get("random_push"),
                        {"max_pushes": 1})
  with pytest.raises(SystemExit, match="not all accepted"):
    resolve_random_push("g1_recover_lafan", reg, {"max_push": 1})
