"""The frozen STAY certificate V_stay(x), read inside the reach task's margin.

The reach task's target is not the goal box itself but the set of states from
which the trained stay policy can remain inside it: ``l = min(l_box,
V_stay)``. ``l_box`` alone would let the reach-avoid policy count as "arrived"
at a state that is inside the box but already leaving it, which the stay policy
cannot hold. ``V_stay`` is that policy's own avoid value, frozen.

Two pieces:

:class:`StayValue`
  A ``mode="startup"`` event term. It loads the stay twin (policy + its obs
  normalizer) once, when the env is built, and publishes itself as
  ``env._zoo_stay_value`` for the margin to call. It is an event term only
  because that is where mjlab lets a cfg hand an object to the env it builds;
  its ``__call__`` does nothing.

:func:`current_obs`
  The actor observation of the state the physics has JUST produced, rebuilt
  from the observation terms. The margin runs inside mjlab's reward manager,
  before the observation manager has computed that step's observation (and
  before auto-reset), and the observation manager cannot be asked early:
  ``compute_group(update_history=False)`` returns the history buffer, which
  still holds the PREVIOUS step. So the terms are evaluated directly, with
  their clip and scale and WITHOUT their noise (the certificate is read on the
  clean state). ``tests/test_g1_waist.py`` pins it against the manager.
"""

from __future__ import annotations

import os

import torch

from mjlab.managers.manager_base import ManagerTermBase

#: Where the stay run lands under the trainer's default ``--out``.
DEFAULT_STAY_RUN = "runs/g1_waist_stay_arm_random"
#: Environment override for the stay run (a run dir or a model zip).
STAY_RUN_ENV = "SAFE_ZOO_G1_WAIST_STAY_RUN"

_ZOO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.abspath(__file__)))))


def resolve_stay_run(run: str | None) -> str:
  """A stay run dir or model zip -> the model zip. None -> $SAFE_ZOO_G1_WAIST_
  STAY_RUN, else :data:`DEFAULT_STAY_RUN`. Relative paths resolve against the
  repo root."""
  path = run or os.environ.get(STAY_RUN_ENV) or DEFAULT_STAY_RUN
  path = os.path.expanduser(path)
  if not os.path.isabs(path):
    path = os.path.join(_ZOO, path)
  if os.path.isdir(path):
    path = os.path.join(path, "final_model.zip")
  if not os.path.exists(path):
    raise FileNotFoundError(
      f"g1_waist reach task: no stay policy at {path}. The reach target is the "
      f"stay policy's certified set, so train g1_waist_stay_arm_random first "
      f"(configs/g1_waist_stay_armrandom_safetyppo1p.yaml), then point "
      f"env_overrides.stay_run (or ${STAY_RUN_ENV}) at its run dir or a "
      f"checkpoint zip.")
  return path


def current_obs(env, group: str = "actor") -> torch.Tensor:
  """``group``'s concatenated observation of the current sim state, noise-free.

  Valid for groups without observation delay and with at most one frame of
  history, which is every G1 group in this repo; anything else is refused
  rather than approximated.
  """
  om = env.observation_manager
  if not om._group_obs_concatenate[group]:
    raise ValueError(f"current_obs: group {group!r} is not concatenated")
  parts = []
  for name, tc in zip(om._group_obs_term_names[group],
                      om._group_obs_term_cfgs[group]):
    if tc.delay_max_lag > 0 or tc.history_length > 1:
      raise ValueError(
        f"current_obs: term {group}/{name} has delay {tc.delay_max_lag} / "
        f"history {tc.history_length}; only undelayed single-frame terms can "
        f"be rebuilt from the current state")
    x = tc.func(env, **tc.params).clone()
    if tc.clip:
      x = x.clip(min=tc.clip[0], max=tc.clip[1])
    if tc.scale is not None:
      x = x * tc.scale
    parts.append(x.reshape(x.shape[0], -1))
  return torch.cat(parts, dim=om._group_obs_concatenate_dim[group])


class StayValue(ManagerTermBase):
  """Startup event term: load the frozen stay twin, publish it on the env.

  Params: ``run`` (stay run dir or model zip; None -> :func:`resolve_stay_run`
  defaults), ``obs_group`` (the observation group the twin reads, default
  ``"actor"``).
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    from robot_safety_sandbox.eval.policies import load_twin
    self.path = resolve_stay_run(cfg.params.get("run"))
    self.group = cfg.params.get("obs_group", "actor")
    model, self.norm = load_twin(self.path, str(env.device), quiet=True)
    self.policy = model.policy
    self.policy.set_training_mode(False)
    self.obs_dim = int(self.policy.observation_space.shape[0])
    env._zoo_stay_value = self
    print(f"[g1_waist] reach target = certified set of the frozen stay policy "
          f"{self.path} ({type(model).__name__})")

  def __call__(self, env, env_ids, **_params) -> None:
    pass

  def _evaluate(self, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      return self.policy.predict_values(self.norm(obs)).squeeze(-1)

  def value(self, env) -> torch.Tensor:
    """V_stay of the current state, (N,)."""
    obs = current_obs(env, self.group).float()
    if obs.shape[1] != self.obs_dim:
      raise ValueError(
        f"g1_waist: the stay policy at {self.path} reads {self.obs_dim}-dim "
        f"observations but this env's {self.group!r} group is {obs.shape[1]}; "
        f"it was trained on a different observation layout.")
    return self._evaluate(obs)


def stay_value(env) -> torch.Tensor:
  """V_stay of the current state. Needs the cfg's ``stay_value`` startup term."""
  sv = getattr(env, "_zoo_stay_value", None)
  if sv is None:
    raise RuntimeError(
      "g1_waist reach margin: no frozen stay value on this env. The reach cfg "
      "(g1_waist_reach_env_cfg) adds it as the 'stay_value' startup event.")
  return sv.value(env)
