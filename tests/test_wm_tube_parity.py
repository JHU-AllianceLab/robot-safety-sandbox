"""Parity of the train-time world-model tube filter (filters/wm_tube.py) with
the eval it was lifted from (examples/eval_sonic_qcbf.py, E), on the real
thermal70_s0 and thermalcbf_s0 checkpoints, CPU, no simulator:

  * verdicts ``torch.equal`` to ``eq.Evaluator.min_verdict`` on identical id
    batches (the full batch and a compacted subset), at the training tube
    (0.95) and the eval's sensitivity tube (0.90); the compacted subset equals
    the full batch to rounding only;
  * the per-step trace equal to ``Evaluator.wm_margins``;
  * the live fallback action equal to E:868-870 (``_pi_obs`` + ``_predict`` +
    the box);
  * the whole filter's action and engagement equal to ``qcbf_eval_lib.Controller``
    on E's verdict (no dwell, the eval's NaN policy).

The Evaluator is built without its constructor and handed an env VIEW whose
``obs_key`` is ``"actor"`` (the fallback's group: on the track tasks the env's
own obs_key is the task group, E:640) and whose observation manager carries the
fallback's actor layout, so E's own ``_last_action_slice`` finds the columns.
States are drawn from each world model's normalizer, the last-action dims
anywhere in the task box +-6.5, proposals in +-3 and +-6.5. The same check runs
on live states in the GPU smoke (examples/smoke_wm_tube.py).

Importing E pulls in robot_safety_sandbox, mjlab and warp (minutes on a login
node), so this is opt-in:

    ZOO_SLOW_TESTS=1 OMP_NUM_THREADS=1 pytest tests/test_wm_tube_parity.py -q
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
  os.environ.get("ZOO_SLOW_TESTS") != "1",
  reason="imports eval_sonic_qcbf (robot_safety_sandbox, mjlab, warp); set ZOO_SLOW_TESTS=1")

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
sys.path.insert(0, os.path.join(_ZOO, "examples"))

CASES = {
  "thermal70": ("runs/shac_thermal70/shac_s0/g1_recover_thermal70_safetyshac1p",
                "runs/shac_thermal70/wmreport/calib/thermal70_s0/min_tube.json", "thermal70_s0"),
  "thermal_cbf": ("runs/shac_thermal_cbf/shac_s0/g1_recover_thermal_cbf_safetyshac1p",
                  "runs/shac_thermal_cbf/wmreport/calib/thermalcbf_s0/min_tube.json",
                  "thermalcbf_s0"),
}
#: The fallback's actor group (velocity cfg + motor_temp; envs/g1_recover/env_cfg.py:267):
#: what E's _last_action_slice reads off the observation manager.
ACTOR_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3), ("phase", 2),
               ("joint_pos", 29), ("joint_vel", 29), ("actions", 29), ("motor_temp", 29))
N = 64


def last_action():
  """Stands in for mjlab's term function: E's rule matches it by __name__."""


def _other():
  pass


@pytest.fixture(scope="module")
def mods():
  eq = importlib.import_module("eval_sonic_qcbf")
  import qcbf_eval_lib as ql
  from robot_safety_sandbox.filters import wm_tube as wt
  return eq, ql, wt


def _evaluator(eq, bundle, S, C, tube):
  om = SimpleNamespace(
    _group_obs_term_names={"actor": [n for n, _ in ACTOR_TERMS]},
    _group_obs_term_dim={"actor": [(d,) for _, d in ACTOR_TERMS]},
    _group_obs_term_cfgs={"actor": [SimpleNamespace(func=last_action if n == "actions" else _other,
                                                    scale=None, clip=None)
                                    for n, _ in ACTOR_TERMS]},
    compute_group=lambda group, update_history=False: {"critic": S}[group])
  ev = object.__new__(eq.Evaluator)
  ev.mj = SimpleNamespace(observation_manager=om)
  ev.env = SimpleNamespace(obs_key="actor", mj=ev.mj, sim_context=lambda: C)
  ev.model, ev.norm = bundle.model, bundle.norm
  ev.dev, ev.n = torch.device("cpu"), int(S.shape[0])
  # the box E ran in: the fallback task's +-3
  ev.low = torch.full((29,), -3.0)
  ev.high = torch.full((29,), 3.0)
  ev.pi_slice, ev.pi_obs_dim = eq.Evaluator._last_action_slice(ev)
  ev.wm_actor_idx = bundle.model._actor_idx_t
  ev.wm_z_clip = getattr(bundle.model, "wm_z_clip", None)
  ev.wm_state_sl = None
  ev.wm_tw_sl = None
  ev.hyb = None            # --wm-thermal nn (the hybrid's parity runs on live envs: smoke_wm_tube)
  ev.tube_calib = tube
  return ev


def _states(bundle, n, seed):
  g = torch.Generator().manual_seed(seed)
  wm = bundle.wm
  S = wm.s_mean + 0.5 * wm.s_std * torch.randn(n, int(wm.state_dim), generator=g)
  last = bundle.actor_idx[bundle.la]
  S[:, last] = 6.5 * (2 * torch.rand(n, last.numel(), generator=g) - 1)
  S[:, 6:11] = 0.0
  C = wm.c_mean + 0.5 * wm.c_std * torch.randn(n, int(wm.context_dim), generator=g)
  A = S[:, bundle.actor_idx].clone()
  A[:, bundle.la] = 6.5 * (2 * torch.rand(n, 29, generator=g) - 1)
  props = {box: box * (2 * torch.rand(n, 29, generator=g) - 1) for box in (3.0, 6.5)}
  return S, C, A, props


@pytest.mark.parametrize("world", sorted(CASES))
def test_verdicts_equal_the_evals(mods, world):
  import json
  from gymnasium import spaces
  eq, ql, wt = mods
  run, tube_json, tag = (os.path.join(_ZOO, CASES[world][0]), os.path.join(_ZOO, CASES[world][1]),
                         CASES[world][2])
  if not os.path.isfile(os.path.join(run, "final_model.zip")):
    pytest.skip(f"{run} not on this machine")
  with open(tube_json) as f:
    tube = json.load(f)
  for cond_text in ("wmmin:1:0.95", "wmmin:1:0.9"):
    bundle = wt.load_wm_tube(run, tube_json, tag=tag, cond=cond_text, device="cpu")
    cond = ql.parse_condition(cond_text)
    S, C, A, props = _states(bundle, N, 11 if world == "thermal70" else 12)
    ev = _evaluator(eq, bundle, S, C, tube)
    assert ev.pi_slice == bundle.la and ev.pi_obs_dim == bundle.actor_dim
    mon = wt.WMTubeMonitor(N, "cpu", bundle, proposal_bound=6.5, action_dim=29)
    ids = torch.arange(N)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[1::3] = True
    sub = mask.nonzero().squeeze(-1)
    for box, a in props.items():
      ref = eq.Evaluator.min_verdict(ev, ids, a, cond)
      ours = mon(a, wm_state=S, sim_ctx=C)
      assert torch.equal(ours, ref), (world, cond_text, box)
      ref_sub = eq.Evaluator.min_verdict(ev, sub, a[sub], cond)
      ours_sub = mon(a, wm_state=S, sim_ctx=C, active=mask)
      assert torch.equal(ours_sub[sub], ref_sub), (world, cond_text, box)
      assert torch.isnan(ours_sub[~mask]).all()
      assert torch.allclose(ours_sub[sub], ours[sub], atol=1e-5, rtol=0)
      if cond_text == "wmmin:1:0.95":
        assert torch.equal(mon.margins(ids, a, 50, wm_state=S, sim_ctx=C),
                           eq.Evaluator.wm_margins(ev, ids, a, 50)[:, 0])
    # the live fallback action, E:868-870
    with torch.no_grad():
      pobs, _ = eq.Evaluator._pi_obs(ev, A)
      a_safe = torch.clamp(bundle.model.policy._predict(bundle.norm(pobs), deterministic=True),
                           ev.low, ev.high)
      assert torch.equal(bundle.pi(A), a_safe)
    # the whole filter (no dwell, NaN passes as in E) = E's Controller on E's verdict
    fenv = SimpleNamespace(mj=SimpleNamespace(step_dt=0.02), num_envs=N, device="cpu",
                           obs_groups=lambda: {"critic": S, "actor": A}, sim_context=lambda: C,
                           action_space=spaces.Box(-6.5, 6.5, shape=(29,)))
    filt, _ = wt.wm_tube_filter(fenv, condition=cond_text, dwell_s=0.0, proposal_bound=6.5,
                                bundle=bundle, nan_engages=False)
    a = props[6.5]
    act, info = filt(a, wm_state=S, actor_obs=A, sim_ctx=C)
    u, tel = ql.Controller(cond)(a, a_safe, None, eq.Evaluator.min_verdict(ev, ids, a, cond))
    assert torch.equal(act, u) and torch.equal(info.engaged, tel["engaged"])
    print(f"[parity] {world} {cond_text}: eps {bundle.eps:.4f}, engaged "
          f"{float(info.engaged.float().mean()):.3f} of {N} synthetic states")
