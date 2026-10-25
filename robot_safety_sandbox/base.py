"""Base classes: run any mjlab task with safety_sb3 (tensor or numpy path).

The zoo's env contract (matches what every safety_sb3 learner consumes):

    reward   = g(s)   the physical safety margin  (NEVER normalize/reshape it)
    l_x      = l(s)   the target margin           (zeros for avoid-only tasks)
    dones    = terminated | truncated  (mjlab auto-resets internally)
    timeouts = truncated & ~terminated (no value bootstrap: g is absolute)

A task is fully specified by two callables (see :mod:`registry`):

    cfg_builder(play: bool) -> ManagerBasedRlEnvCfg   # the mjlab env
    margin_fn(env)          -> (g, l) batched tensors # the reach-avoid margins

Everything else here is plumbing: :class:`MjlabTensorSafetyEnv` exposes the
task as a safety_sb3 ``TensorVecEnv`` (GPU-resident, ~50k steps/s on a 12GB
card at 2048 envs); :class:`MjlabNumpySafetyEnv` as a classic SB3 ``VecEnv``
(for the SAC family / plain SB3 tooling). Curriculum levels and task metrics
(mjlab ``extras['log']``) are forwarded via ``metrics()`` so training logs
always show curriculum progression — silent curriculum stalls were the single
biggest source of lost weeks in the reference tasks.

Porting a new task = write a cfg_builder (spawn events + curricula, plain
mjlab) + a margin_fn (compose from :mod:`margins`), then ``register()`` it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from gymnasium import spaces

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

# The tensor bridge pairs with safety_sb3 learners; the NUMPY bridge (cumulative
# task policies, vanilla SB3) must work without safety_sb3 installed.
try:
  from safety_sb3.tensor_env import TensorVecEnv
  _HAS_SAFETY_SB3 = True
except ImportError:  # cumulative-only install: tensor path disabled
  TensorVecEnv = object
  _HAS_SAFETY_SB3 = False
# The model-based learners' env contract (safety_sb3 >= SHAC). An older
# safety_sb3 simply has no SHAC to call world_model_spec().
try:
  from safety_sb3.tensor_env import WorldModelSpec
except ImportError:
  WorldModelSpec = None

TERMINAL_MARGIN = -0.1  # g anchor on failure terminations (pre-reset hook)
# A margin_fn may carry its own anchor as a ``terminal_margin`` attribute (read
# with getattr, like ``has_target``): g1_waist's stay margins anchor at -CLAMP so
# that a fall is never cheaper than a goal-box violation.

#: name of the uniform success DoneTerm added by end_criterion="reach-avoid"
SUCCESS_TERM = "zoo_reach_success"


def _cached_margins(env, margin_fn):
  """(g, l) for THIS step. mjlab computes terminations (line 148 of
  ManagerBasedRlEnv.step) BEFORE rewards (line 152), both after the single
  ``common_step_counter`` bump — so the reach-avoid success DoneTerm runs first
  and stashes the margins it computed under a step token; the reward hook then
  reuses them instead of calling the (geometry) margin_fn a second time. The
  token guards against reading a stale cache when no DoneTerm ran this step
  (failure/timeout mode), in which case we compute fresh — identical result,
  since margin_fn is a pure function of the (unchanged between 148 and 152)
  scene state."""
  cache = getattr(env, "_zoo_gl_cache", None)
  if cache is not None and cache[2] == env.common_step_counter:
    return cache[0], cache[1]
  g, l = margin_fn(env)
  env._zoo_gl_cache = (g, l, env.common_step_counter)
  return g, l


def zoo_reach_success(env, margin_fn=None) -> torch.Tensor:
  """Uniform reach-avoid success termination (added ONLY when the env's
  end_criterion is "reach-avoid"): fire when the target is reached while safe,
  i.e. g >= 0 AND l >= 0. Registered as a mjlab DoneTerm (non-time_out) so
  mjlab's internal auto-reset fires on the SAME step the target is reached —
  a bridge-side augmentation of ``terminated`` would reset one step late
  (mjlab already auto-reset during self.mj.step) and desync the rollout buffer.
  Computes (and caches) the margins here so the later reward hook need not
  recompute them (see :func:`_cached_margins`)."""
  g, l = _cached_margins(env, margin_fn)
  return (g >= 0.0) & (l >= 0.0)


def safety_margin_hook(env, margin_fn=None) -> torch.Tensor:
  """Reward-term hook: computes (g, l) BEFORE mjlab's auto-reset (terminal-
  correct) and stashes them in ``env.extras``. g is anchored to the terminal
  failure value on real terminations: ``margin_fn.terminal_margin`` if it has
  one, else TERMINAL_MARGIN."""
  g, l = _cached_margins(env, margin_fn)
  anchor = float(getattr(margin_fn, "terminal_margin", TERMINAL_MARGIN))
  failed = env.termination_manager.terminated
  # A reach-avoid SUCCESS is a terminated step too (its DoneTerm sets terminated),
  # but it must NOT be anchored to the failure value: its terminal reach-avoid
  # target min(l, g) is >= 0 (a win), and clamping g to -0.1 would flip it to a
  # loss. Exclude it. In failure/timeout mode the term is absent, so this is a
  # no-op and the anchor is bit-identical to before.
  if SUCCESS_TERM in env.termination_manager.active_terms:
    failed = failed & ~env.termination_manager.get_term(SUCCESS_TERM)
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, anchor)), g)
  # NaN SANITATION: a physics NaN makes margins NaN in the SAME step the
  # nan_detection termination fires; the reset happens after the reward is
  # recorded, so one NaN margin can poison the rollout buffer and NaN the
  # policy weights (box stage-2 run 10d774v3 died this way at ~75M despite
  # nan_term). Grade corrupted states as failures, never as poison.
  g = torch.nan_to_num(g, nan=anchor, posinf=3.0, neginf=-3.0)
  l = torch.nan_to_num(l, nan=-3.0, posinf=3.0, neginf=-3.0)
  env.extras["zoo_g"] = g
  env.extras["zoo_l"] = l
  return g


def margin_probe_hook(env, margin_fn=None) -> torch.Tensor:
  """Compute (g, l) and stash them WITHOUT contributing to the reward.

  The dense-reward + margins case (mode="cumulative" tasks that a safety filter
  wraps at TRAIN time — see the PORL experiment): the learner's reward must stay
  the env's own dense stack, but the filter's monitor and the failure counters
  still need the task's margins, computed at the same instant and under the same
  contract as a safety run.

  Why this is a REWARD term returning zeros rather than a zero-weight one: mjlab
  skips zero-weight terms outright (``RewardManager.compute``, "if
  term_cfg.weight == 0.0: continue"), so a weight-0 hook would silently never
  run. A weight-1 term whose func returns zeros DOES run and contributes exactly
  ``0 * 1.0 * dt = 0``. And it has to run here, inside the reward manager,
  because that is the only place margins are computed BEFORE mjlab's in-step
  auto-reset — reading them afterwards would grade the post-reset state.
  """
  safety_margin_hook(env, margin_fn=margin_fn)   # stashes zoo_g / zoo_l
  return torch.zeros_like(env.extras["zoo_g"])


def build_task_cfg(cfg_builder: Callable, margin_fn: Callable, num_envs: int,
                   drop_events: tuple[str, ...] = ("push_robot",),
                   dense: bool = False, end_criterion: str = "failure",
                   cfg_overrides: dict | None = None,
                   dense_margins: bool = False):
  """Assemble an mjlab cfg for the zoo: task cfg + the margin hook.

  ``drop_events`` removes events a learned adversary replaces (default: the
  random push). ``dense=True`` is the STAGE-1 locomotion mode: the reward stays
  the env's own dense reward stack (`_r`) and the safety hook is NOT added, so a
  stock PPO shapes a proper gait (reach-avoid g/l are not used here).

  ``end_criterion`` (see :data:`registry.END_CRITERIA`) sets WHEN episodes end,
  UNIFORMLY across tasks. The task cfg's own terminations already encode the
  FAILURE set (fell_over / illegal_contact / ... -> the reward hook anchors g<0
  there) plus the env timeout, so:
    * "failure"     : add nothing. The task's existing terminations stand as-is
                      -> bit-identical to pre-end_criterion behavior. Reaching
                      the target does not end the episode; the agent reaches
                      deeper (l keeps climbing to the g ceiling).
    * "reach-avoid" : ALSO add the uniform success DoneTerm (g>=0 AND l>=0) on
                      top of the failure terminations -> reach-and-stop.
    * "timeout"     : strip every non-time_out (failure) termination so ONLY the
                      env timeout ends episodes, and add no success term (pure
                      value-learning / diagnostic). g is still anchored nowhere,
                      so failures live on in the margin, they just don't reset.
  """
  # cfg_overrides: experiment-level env/task params forwarded to the cfg_builder
  # (partial call-kwargs override the task registration's baked values, e.g.
  # {"gate_close_rate": 0.003}). Fail loud on a param the cfg_builder rejects.
  try:
    cfg = cfg_builder(play=False, **(cfg_overrides or {}))
  except TypeError as e:
    if cfg_overrides:
      raise SystemExit(
        f"[cfg_overrides] {list(cfg_overrides)} not all accepted by the task's "
        f"cfg_builder: {e}") from e
    raise
  cfg.scene.num_envs = int(num_envs)
  if cfg.events is not None:
    for e in drop_events:
      cfg.events.pop(e, None)
  if dense and dense_margins:
    if margin_fn is None:
      raise ValueError(
        "dense_margins=True needs a margin_fn: it exists to give a dense-reward "
        "task the margins a train-time safety filter monitors. Register the "
        "task with the margin_fn of the twin that will be its fallback.")
    cfg.rewards["zoo_margin_probe"] = RewardTermCfg(
      func=margin_probe_hook, weight=1.0, params={"margin_fn": margin_fn})
  if not dense:
    if margin_fn is None:
      raise ValueError(
        "margin_fn=None with dense=False: a mode='cumulative' task trains on "
        "the dense env reward — build it in dense mode (make_tensor/make_numpy "
        "do this automatically from the task's mode).")
    cfg.rewards["zoo_safety_hook"] = RewardTermCfg(
      func=safety_margin_hook, weight=1.0, params={"margin_fn": margin_fn})
    if end_criterion == "reach-avoid":
      cfg.terminations[SUCCESS_TERM] = TerminationTermCfg(
        func=zoo_reach_success, params={"margin_fn": margin_fn})
    elif end_criterion == "timeout":
      # Keep only the env timeout (time_out=True); drop the failure set so the
      # episode always runs to the horizon (diagnostic / pure value-learning).
      for name in [n for n, t in cfg.terminations.items()
                   if not getattr(t, "time_out", False)]:
        cfg.terminations.pop(name)
  return cfg


class _MjlabCore:
  """Shared mjlab plumbing for both bridges."""

  def _init_core(self, num_envs, device, cfg_builder, margin_fn, *,
                 ctrl_dim, dstb_dim, ctrl_gain, force_max, adversary,
                 adversary_body, render_mode, obs_key=None, dense_reward=False,
                 dstb_mode="wrench", dstb_gain=0.25, dstb_joint_slice=None,
                 joint_noise=None, hybrid_skill=None,
                 latch_margin_fn=None, end_criterion="failure",
                 cfg_overrides=None, dense_margins=False, action_bound=1.0,
                 critic_obs_key=None, random_push=None, regime_force_buckets=0):
    self.obs_key = obs_key  # resolved after first reset (auto-detect)
    # Privileged (asymmetric) critic obs group. OPT-IN: surfaced to the learner
    # ONLY when critic_obs_key is explicitly requested (the factory passes it when
    # the spec's interface.observation.privileged is set). The one-group (symmetric)
    # path stays the DEFAULT — a cfg that merely HAPPENS to emit a distinct "critic"
    # group (e.g. the vendored velocity cfgs) is left untouched, so existing runs
    # are byte-identical. Resolved after the first reset.
    self._critic_obs_key_req = critic_obs_key
    self.critic_obs_key = None
    self.critic_observation_space = None
    self.end_criterion = str(end_criterion)
    self.ctrl_dim = int(ctrl_dim)
    self.dstb_dim = int(dstb_dim)
    self.ctrl_gain = float(ctrl_gain)
    # The learner samples an UNBOUNDED Gaussian action; the bridge's action space is the box the
    # rollout clamps it to before it reaches the env. The mjlab JointPositionAction has NO clip of
    # its own (rsl_rl does not clamp either), so THIS +-action_bound is the only limit on the
    # per-joint position delta = scale * action. The historical default 1.0 caps a G1 knee at
    # scale*ctrl_gain = 0.53 rad (a stride cap that made the walker shuffle, T006); a task whose
    # mjlab action term has no clip should pass a WIDE bound so the policy can use the stride it
    # needs. Kept at 1.0 by default so every existing task is byte-for-byte unchanged.
    self.action_bound = float(action_bound)
    self.force_max = float(force_max)
    # dstb channel: "wrench" = external force on adversary_body (legged tasks);
    # "action" = ACTION-ADDITIVE disturbance, ctrl += dstb_gain * a_dstb — the
    # Robust-Gymnasium / classic-control two-player convention (e.g. hopper +-25%
    # of the ctrl bound). MuJoCo clamps ctrl to ctrlrange afterward.
    # "joint" = JOINT-PARTITION disturbance: the adversary OWNS a slice of the
    # robot's own actuators (envs/g1_arm_adversary — the adversary commands the
    # arms, the actor the legs and waist). The ctrl vector stays the env's full
    # action width and the bridge OVERWRITES the disturbance's slice in it, so
    # the actor keeps its original action head (and a trained 1P checkpoint warm-
    # starts whole) while its outputs on that slice are discarded. dstb_gain is
    # the adversary's joint AUTHORITY, on top of whatever per-joint scale the
    # mjlab action term already carries; ramp it with DstbGainRampCallback the
    # way force_max is ramped for a wrench.
    self.dstb_mode = str(dstb_mode)
    self.dstb_gain = float(dstb_gain)
    self._dstb_slice = None
    # The joint slice has ONE owner: a joint-mode adversary, or the scripted
    # joint_noise below. joint_noise may also run under a WRENCH adversary
    # (g1_recover_arm_random_wrench: random arms + a learned torso push), since
    # the two never share a channel; dstb_dim is then the wrench's width, 3.
    if self.dstb_mode == "joint" and dstb_joint_slice is None:
      raise ValueError(
        "dstb_mode='joint' needs dstb_joint_slice=(start, stop): which action "
        "dims the adversary owns (e.g. (15, 29) for the G1's 14 arm joints).")
    if joint_noise is not None and dstb_joint_slice is None:
      raise ValueError(
        "joint_noise drives a joint slice, so it needs dstb_joint_slice="
        "(start, stop): which action dims the random process owns.")
    if dstb_joint_slice is not None:
      if self.dstb_mode != "joint" and joint_noise is None:
        raise ValueError(
          f"dstb_joint_slice is only meaningful for dstb_mode='joint' or a "
          f"joint_noise task, not dstb_mode={self.dstb_mode!r} alone.")
      lo, hi = (int(x) for x in dstb_joint_slice)
      if not 0 <= lo < hi <= self.ctrl_dim:
        raise ValueError(
          f"dstb_joint_slice=({lo}, {hi}) is not a sub-range of the env's "
          f"{self.ctrl_dim} action dims.")
      if self.dstb_mode == "joint" and hi - lo != self.dstb_dim:
        raise ValueError(
          f"dstb_joint_slice=({lo}, {hi}) spans {hi - lo} joints but dstb_dim="
          f"{self.dstb_dim}; the adversary's action width must match the slice "
          f"it drives.")
      self._dstb_slice = (lo, hi)
    # Per-env multiplier on the joint slice's command (a joint-mode adversary's
    # or joint_noise's), (num_envs,) or None. Written by
    # PerEnvForceScaleCallback(name="dstb_scale"); read only by
    # _write_joint_slice (a wrench adversary's per-env knob is force_scale,
    # read in _apply_dstb).
    self.dstb_scale = None
    self.adversary = bool(adversary)
    # SCRIPTED joint-partition disturbance (envs/g1_arm_random): the joint
    # slice is driven by joint_noise.SmoothJointNoise instead of a learned
    # adversary, in a SINGLE-player run. joint_noise is that generator's kwargs
    # ({} = its defaults); None = off. Same slice, same dstb_gain x dstb_scale
    # authority, same overwrite as the adversary, so the actor's head and obs
    # are those of the joint-adversary task and checkpoints move between the two.
    # With dstb_mode="wrench" a learned wrench adversary may play on top: it
    # pushes adversary_body while the noise keeps the slice.
    # Built after the mjlab env below (it needs the control dt).
    self._joint_noise = None
    if joint_noise is not None:
      if self.dstb_mode not in ("joint", "wrench"):
        raise ValueError(
          f"joint_noise runs alone (dstb_mode='joint') or under a wrench "
          f"adversary (dstb_mode='wrench'), not dstb_mode={self.dstb_mode!r}: "
          f"an action-additive adversary would add onto the slice the noise "
          f"overwrites.")
      if self.adversary and self.dstb_mode == "joint":
        raise ValueError(
          "joint_noise and adversary=True would both drive the same joint "
          "slice; with dstb_mode='joint' a scripted-disturbance task is "
          "single-player.")
    # SCRIPTED random pushes (random_push.RandomPush): the single-player stand-in
    # for a wrench adversary. The same channel -- an external force on
    # adversary_body at force_max x force_scale, so the force ramp and the
    # per-env survival scale act on it unchanged -- driven by a random process
    # (direction, onset, duration) instead of a learned player. random_push is
    # that process's kwargs ({} = its defaults); None = off. Built after the
    # mjlab env below (it needs the control dt).
    self._push = None
    if random_push is not None:
      if self.adversary:
        raise ValueError(
          "random_push and adversary=True would both write the wrench on "
          "adversary_body; a random-push task is single-player.")
      if self.dstb_mode != "wrench":
        raise ValueError(
          f"random_push is an external force on adversary_body and needs "
          f"dstb_mode='wrench', not dstb_mode={self.dstb_mode!r}.")
    # Per-episode REGIMES for a multi-gamma learner (safety_sb3 gamma ladder):
    # spawn source (published by the task's spawn as mj._zoo_spawn_source) x
    # the per-env force scale quantized into this many buckets (0 = no force
    # component). See regime().
    self._regime_buckets = int(regime_force_buckets)
    if self._regime_buckets < 0:
      raise ValueError(f"regime_force_buckets must be >= 0, got {regime_force_buckets}")
    self.dense_reward = bool(dense_reward)  # Stage-1: train on the env reward
    # dense reward AND margins: the reward stays the env's dense stack, while
    # (g, l) are computed alongside it for a train-time filter / failure
    # counters. Only meaningful with dense_reward=True.
    self.dense_margins = bool(dense_margins) and self.dense_reward
    self._safety_g = None
    self._safety_l = None
    # Training-safety counters (drained through metrics() -> safety/* logging).
    # This is a safety codebase, so EVERY training env reports how often it
    # fails DURING training -- no flag, no wrapper. ``_failures_total`` is
    # cumulative over the whole run (the headline y axis); the rest are window
    # counters zeroed on each drain. See _count_step / _drain_counts / metrics.
    self._failures_total = 0
    self._zero_counts()
    # Reach accounting (safety/reach_rate, safety/time_to_reach_s) only where l
    # is a real target: an avoid task's l is a zero placeholder, on which every
    # safe step would read as a reach. Same has_target reading as the registry.
    self._track_reach = (margin_fn is not None
                         and bool(getattr(margin_fn, "has_target", True))
                         and (self.dense_margins or not self.dense_reward))
    self._reach_ep = None     # per-env episode state, see _count_reach
    self.render_mode = render_mode
    self.mj = ManagerBasedRlEnv(
      cfg=build_task_cfg(cfg_builder, margin_fn, num_envs, dense=dense_reward,
                         end_criterion=self.end_criterion,
                         cfg_overrides=cfg_overrides,
                         dense_margins=self.dense_margins),
      device=device, render_mode="rgb_array" if render_mode else None)
    self._robot = self.mj.scene["robot"]
    # Stamp the criterion onto the mjlab env so a margin_fn can see it. The
    # arm-adversary task reads it: its disturbance-phase l shaping must not run
    # under end_criterion="reach-avoid", where a shaped l would end the episode
    # (see envs/g1_arm_adversary/env_cfg.py).
    self.mj._zoo_end_criterion = self.end_criterion
    if joint_noise is not None:
      from .joint_noise import SmoothJointNoise
      lo, hi = self._dstb_slice
      self._joint_noise = SmoothJointNoise(
        int(num_envs), hi - lo, float(self.mj.step_dt), device,
        **joint_noise)
      # The HOLD glide (see _reset_joint_noise); stays off unless the task's
      # spawn publishes the action each env was restored with.
      shape = (int(num_envs), hi - lo)
      self._hold = torch.zeros(shape, device=device)
      self._hold_t = torch.zeros(shape, device=device)
      self._hold_T = torch.ones(shape, device=device)
      self._hold_on = False
    if self.dstb_mode == "wrench":
      body_ids, _ = self._robot.find_bodies(adversary_body)
      self._adv_body_ids = list(body_ids)
    else:
      self._adv_body_ids = []  # action-mode dstb needs no body
    self._all_ids = torch.arange(int(num_envs), device=device)
    self._zero_wrench = torch.zeros((int(num_envs), 1, 3), device=device)
    if random_push is not None:
      from .random_push import RandomPush
      self._push = RandomPush(int(num_envs), float(self.mj.step_dt), device,
                              **random_push)
    self._log: dict = {}
    # HYBRID ROLLOUTS (funnel composition): when hybrid_skill is set, any env
    # whose LAST-step reach margin l went positive is latched to the FROZEN
    # skill's deterministic actions until its episode ends. The RA learner
    # then optimizes the value of the COMPOSED system (approach policy ->
    # frozen skill), which is exactly what the l = V_hat certificate certifies.
    self.hybrid_skill = hybrid_skill
    # dense-mode latch source (v10.3): geometric proposal margin computed for
    # the hybrid latch ONLY — never a reward term (anti-Goodhart split).
    self.latch_margin_fn = latch_margin_fn
    self._hyb = None       # lazy (policy, obs_mean, obs_var)
    self._hyb_latch = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    self._last_l = None
    obs_dict, _ = self.mj.reset()
    if self._push is not None:
      # The process was reset when it was built; this reset's spawn may have
      # handed some envs a pending push (see _apply_pending_push).
      self._apply_pending_push()
      self.mj._zoo_push_quiet = self._push.quiet()
    # The FULL obs dict of the last transition. The bridges expose one group
    # (self.obs_key) because a learner consumes one; an EVALUATION composes two
    # policies that may read DIFFERENT groups (a blind nominal walker off
    # "actor", a safety twin off "proprioception"), so the dict is kept rather
    # than re-derived by stepping the raw env a second time. Read-only.
    self._obs_dict = obs_dict
    if self.obs_key is None:
      # Obs group naming is per-task-family ("proprioception" in the parkour
      # cfgs, "actor"/"policy" in velocity-style cfgs). Auto-detect the actor
      # group; pass obs_key explicitly to override.
      for cand in ("proprioception", "actor", "policy"):
        if cand in obs_dict:
          self.obs_key = cand
          break
      else:
        self.obs_key = next(iter(obs_dict.keys()))
    obs0 = obs_dict[self.obs_key]
    # Cache the CURRENT raw actor obs on the mjlab env so margin fns that need
    # the observation (e.g. l_vhat) can read it INSIDE the reward hook, where
    # this step's obs are not computed yet (1-step lag, see l_vhat).
    self.mj._zoo_last_obs = obs0.float()
    obs_space = spaces.Box(-np.inf, np.inf, shape=(int(obs0.shape[1]),),
                           dtype=np.float32)
    # Privileged critic group (asymmetric critic). The cfg's "critic" group is
    # computed every step and normally discarded (the learner consumes one group).
    # Surface it to the learner ONLY when explicitly requested (critic_obs_key set
    # by the factory when the spec asks for a privileged critic) — never merely
    # because a distinct "critic" group exists, so the one-group path is the
    # default and existing runs are unchanged.
    if self._critic_obs_key_req is not None:
      ck = self._critic_obs_key_req
      if ck not in obs_dict:
        raise ValueError(
          f"critic_obs_key={ck!r} not among the cfg's observation groups "
          f"{list(obs_dict)}; the privileged critic needs that group in the env cfg.")
      if ck == self.obs_key:
        raise ValueError(
          f"critic_obs_key={ck!r} is the actor group — the privileged critic needs a "
          f"DISTINCT group (add base_lin_vel etc. to it).")
      self.critic_obs_key = ck
      self.critic_observation_space = spaces.Box(
        -np.inf, np.inf, shape=(int(obs_dict[ck].shape[1]),), dtype=np.float32)
    act_dim = self.ctrl_dim + (self.dstb_dim if self.adversary else 0)
    b = self.action_bound
    act_space = spaces.Box(-b, b, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space

  def _apply_dstb(self, a_dstb: torch.Tensor) -> None:
    unit = a_dstb / a_dstb.norm(dim=1, keepdim=True).clamp_min(1e-6)
    mag = self.force_max
    scale = getattr(self, "force_scale", None)  # per-env (survival curriculum)
    if scale is not None:
      mag = mag * scale.reshape(-1, 1)
    forces = (unit * mag).reshape(-1, 1, 3)
    self._robot.write_external_wrench_to_sim(
      forces, self._zero_wrench, body_ids=self._adv_body_ids,
      env_ids=self._all_ids)

  def _hyb_load(self):
    """Lazy-load the frozen hybrid skill (policy + its frozen obs normalizer).
    Relative paths resolve against the zoo repo root."""
    if self._hyb is None:
      import os
      from safety_sb3 import SafetyPPO1P
      root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      d = os.path.expanduser(self.hybrid_skill)
      if not os.path.isabs(d):
        d = os.path.join(root, d)
      device = self.mj.device
      pol = SafetyPPO1P.load(os.path.join(d, "final_model.zip"), device=device,
                           custom_objects={"tensorboard_log": None}).policy
      pol.set_training_mode(False)
      st = torch.load(os.path.join(d, "tensornormalize.pt"),
                      map_location=device, weights_only=True)
      self._hyb = (pol, st["obs_mean"].to(device), st["obs_var"].to(device))
      print(f"[hybrid] frozen skill loaded from {d}")
    return self._hyb

  def _write_joint_slice(self, ctrl: torch.Tensor,
                         a_dstb: torch.Tensor) -> torch.Tensor:
    """Joint partition: the disturbance OWNS the dstb_joint_slice action dims.
    Overwrite (not add) the actor's outputs there — the actor's values on the
    slice are discarded, which is what lets it keep its full-width head. The
    actor still SEES the result: mjlab's last_action observation reads the
    vector submitted to mj.step, so next step's obs carries the disturbance's
    joint command, not the actor's own discarded one. Shared by the learned
    adversary and the scripted joint_noise, so both mean the same authority."""
    lo, hi = self._dstb_slice
    ctrl = ctrl.clone()
    a_dstb = self.dstb_gain * a_dstb
    # Per-env authority (survival curriculum, PerEnvForceScaleCallback with
    # name="dstb_scale"): the joint-mode twin of force_scale. None -> every
    # env at the global dstb_gain, which is the behavior before it existed.
    if self.dstb_scale is not None:
      a_dstb = a_dstb * self.dstb_scale.reshape(-1, 1)
    ctrl[:, lo:hi] = a_dstb
    return ctrl

  def _reset_joint_noise(self, mask: Optional[torch.Tensor] = None) -> None:
    """Restart the scripted joint slice for the envs in ``mask`` (None = all).

    The generator restarts at 0, the default pose. A task whose spawn restores
    a state mid-motion (g1_recover's harvested walker rows) publishes the
    action each env was restored with as ``mj._zoo_reset_action``; the slice
    then starts from THAT command instead: a hold term glides from it to zero
    over the generator's first segment with the same minimum-jerk profile, so
    the slice's command is one smooth glide, restored command -> first
    waypoint, and the arms do not snap at the spawn. Tasks that publish
    nothing keep the hold off, exactly as before.
    """
    self._joint_noise.reset(mask)
    src = getattr(self.mj, "_zoo_reset_action", None)
    if src is None:
      return
    lo, hi = self._dstb_slice
    if mask is None:
      mask = torch.ones(self._hold.shape[0], dtype=torch.bool,
                        device=self._hold.device)
    m = mask.reshape(-1, 1).bool()
    self._hold.copy_(torch.where(m, src[:, lo:hi], self._hold))
    self._hold_t.masked_fill_(m, 0.0)
    self._hold_T.copy_(torch.where(m, self._joint_noise._T, self._hold_T))
    self._hold_on = True

  def _hold_glide(self) -> torch.Tensor:
    """This step's hold term, then advance by one control step (the same
    evaluate-then-advance order as ``SmoothJointNoise.step``)."""
    from .joint_noise import _min_jerk
    u = (self._hold_t / self._hold_T).clamp(0.0, 1.0)
    out = self._hold * (1.0 - _min_jerk(u))
    self._hold_t += float(self.mj.step_dt)
    return out

  def _apply_pending_push(self, mask: Optional[torch.Tensor] = None) -> None:
    """Hand the push process what each RESTARTED env's start state carries
    (``mask`` = the envs just reset; None = all). Call it right after
    ``self._push.reset(mask)``.

    A start state captured at a handover (g1_recover's KIND_HANDOVER rows) may
    come with a push that was still acting when it was captured; the spawn
    publishes it for the episode it just started as ``mj._zoo_pending_push``
    (``force_w`` [n, 3] N world frame, ``steps`` [n] control steps left,
    ``consume`` [n]: the row's push already happened). Here a pending push
    becomes a scripted push of that exact force, resumed on the FIRST step of
    the new episode (onset 0) for its remaining steps, and a consumed row gets
    no random push for the rest of its episode. A task whose spawn publishes
    nothing is unchanged."""
    push = getattr(self, "_push", None)
    pend = getattr(self.mj, "_zoo_pending_push", None)
    if push is None or pend is None:
      return
    m = (torch.ones(push.n, dtype=torch.bool, device=push.device) if mask is None
         else mask.reshape(-1).bool())
    push.exhaust(m & pend["consume"].reshape(-1).bool())
    force_w = pend["force_w"].reshape(-1, 3).float()
    mag = force_w.norm(dim=1)
    steps = pend["steps"].reshape(-1).long()
    push.schedule(m & (steps > 0), force_n=mag, steps=steps, onset=0,
                  dir_w=force_w / mag.clamp_min(1e-9).unsqueeze(1))

  def _mj_step(self, actions: torch.Tensor):
    ctrl = actions[:, :self.ctrl_dim] * self.ctrl_gain
    if self.adversary:
      if self.dstb_mode == "action":
        ctrl = ctrl + self.dstb_gain * actions[:, self.ctrl_dim:]
      elif self.dstb_mode == "joint":
        ctrl = self._write_joint_slice(ctrl, actions[:, self.ctrl_dim:])
      else:
        self._apply_dstb(actions[:, self.ctrl_dim:])
    # Not an elif: the noise may run under a WRENCH adversary (a joint-mode one
    # is refused at init, so the two never write the slice in the same step).
    if self._joint_noise is not None:
      ctrl = self._write_joint_slice(ctrl, self._joint_noise.step())
      if getattr(self, "_hold_on", False):
        lo, hi = self._dstb_slice
        ctrl[:, lo:hi] += self._hold_glide()
    if self.hybrid_skill is not None and self._last_l is not None:
      # latch on LAST step's l (computed inside the previous env.step); once
      # latched, the frozen skill controls the env until its episode ends.
      self._hyb_latch |= self._last_l > 0.0
      if bool(self._hyb_latch.any()):
        pol, h_mean, h_var = self._hyb_load()
        with torch.no_grad():
          nobs = torch.clamp(
            (self.mj._zoo_last_obs - h_mean) / torch.sqrt(h_var + 1e-8),
            -10.0, 10.0)
          a_f = torch.clamp(pol._predict(nobs, deterministic=True), -1.0, 1.0)
        ctrl = torch.where(self._hyb_latch.unsqueeze(-1),
                           a_f * self.ctrl_gain, ctrl)
    push = getattr(self, "_push", None)
    if push is not None:
      # The random push writes the adversary's channel (adversary=True is
      # refused with it): force_max x per-env force_scale, zeros where quiet.
      # The pelvis heading places rear-biased directions and labels each push
      # by the side it comes from.
      scale = getattr(self, "force_scale", None)
      f, active = push.forces(self.force_max, scale, self._robot.data.heading_w)
      self._robot.write_external_wrench_to_sim(
        f.reshape(-1, 1, 3), self._zero_wrench, body_ids=self._adv_body_ids,
        env_ids=self._all_ids)
      # Read INSIDE mj.step by a push-gated target margin (the termination and
      # reward managers grade the state this step produces): quiet = no push
      # left this episode, and the push applied in THIS step counts as active.
      self.mj._zoo_push_quiet = push.quiet()
      self._log["push/force_max_N"] = float(self.force_max)
      self._log["push/active_frac"] = float(active.float().mean())
      if scale is not None:
        self._log["push/scale_mean"] = float(scale.mean())
    obs_dict, _r, terminated, truncated, extras = self.mj.step(ctrl)
    self._obs_dict = obs_dict
    obs = obs_dict[self.obs_key].float()
    self.mj._zoo_last_obs = obs
    if self.dense_reward:
      # Stage-1 locomotion: the reward IS the env's dense reward stack (_r); no
      # safety g/l (a stock PPO with standard GAE shapes the gait).
      g = _r.float()
      if self.latch_margin_fn is not None:
        l = self.latch_margin_fn(self.mj).float()
      else:
        l = torch.zeros_like(g)
      if self.dense_margins:
        # The margin probe ran inside the reward manager (pre-auto-reset), so
        # these are terminal-correct. They stay OFF the (g, l) the learner sees
        # -- its reward must remain the dense stack -- and are read through
        # safety_margins() by the filter and the failure counters.
        self._safety_g = extras["zoo_g"].float()
        self._safety_l = extras["zoo_l"].float()
    else:
      g = extras["zoo_g"].float()
      l = extras["zoo_l"].float()
    if self.hybrid_skill is not None:
      done = terminated | truncated
      self._hyb_latch &= ~done
      self._last_l = l
      self._log["hybrid/latched_frac"] = float(self._hyb_latch.float().mean())
    # mjlab auto-resets INSIDE self.mj.step, which clears the termination
    # manager's buffers — so the only correct source for "what ended this step"
    # is the step's own return, kept here for callers that do not consume it
    # (see MjlabTensorSafetyEnv.step_tensor's collapsed dones/timeouts).
    self._last_terminated, self._last_truncated = terminated, truncated
    if self._joint_noise is not None:
      # mjlab has already auto-reset these envs; their next command restarts
      # from the pose the reset put the joints in (see _reset_joint_noise).
      self._reset_joint_noise(terminated | truncated)
    if push is not None:
      self._log.update(push.record(terminated | truncated, terminated))
      push.reset(terminated | truncated)   # a new episode starts quiet
      self._apply_pending_push(terminated | truncated)
      self.mj._zoo_push_quiet = push.quiet()
    # Always-on training-safety accounting. The margin is the task's g: the
    # returned reward-slot on a safety/reach-avoid env, the alongside-computed
    # _safety_g on a dense-reward+margins env, and absent on a plain cumulative
    # env (which has no margin to report).
    count_g = (self._safety_g if self.dense_margins
               else (None if self.dense_reward else g))
    count_l = (self._safety_l if self.dense_margins
               else (None if self.dense_reward else l))
    self._count_step(terminated, truncated, count_g, count_l)
    log = extras.get("log", {})
    for k, v in log.items():
      try:
        self._log[k] = float(v)
      except (TypeError, ValueError):
        pass
    return obs, g, terminated, truncated, l

  def obs_groups(self) -> dict:
    """The FULL obs dict of the last transition (every group the cfg emits).

    ``step``/``reset`` return only ``self.obs_key``'s group because a learner
    consumes one; an evaluation composing two policies over different groups
    reads the rest here. See :mod:`robot_safety_sandbox.eval.envs`.
    """
    return self._obs_dict

  def critic_obs(self):
    """The RAW privileged critic obs of the last transition, or None.

    The asymmetric-critic value net reads this (base_lin_vel etc.); the actor
    never does, and it is discarded at deploy. Returns None unless the cfg emits
    a distinct privileged "critic" group (see :meth:`_init_core`). The group is
    already recomputed every step in ``_obs_dict`` — this only exposes it, adding
    no new computation. NaN-sanitized to match the actor obs path."""
    if self.critic_obs_key is None:
      return None
    c = self._obs_dict[self.critic_obs_key].float()
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

  # --- model-based learners (safety_sb3 SHAC) ---------------------------------
  # A SHAC learner fits a world model on the RAW privileged critic group and
  # rolls the actor inside it, so it must rebuild, from a predicted critic obs,
  # the actor obs the deployed policy would see. Both hooks below are derived
  # from the env's own managers -- term names, noise cfgs, DR events -- so no
  # task has to describe its observation layout by hand.

  def _obs_layout(self, group: str) -> dict:
    """``{term: (start, size, cfg)}`` of a concatenated obs group, in order."""
    om = self.mj.observation_manager
    if not om.group_obs_concatenate[group]:
      raise ValueError(f"obs group {group!r} is not concatenated")
    out, start = {}, 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
      size = int(np.prod(dims))
      out[name] = (start, size, om.get_term_cfg(group, name))
      start += size
    return out

  @staticmethod
  def _term_noise_bounds(group: str, name: str, size: int, cfg):
    """Additive uniform noise of one obs term as (low, high) per dim, in OBS
    units (the term's scale applied, which mjlab applies after the noise)."""
    from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
    if (cfg.history_length or 0) > 1 or cfg.delay_max_lag:   # 1 = this frame
      raise ValueError(
        f"obs term {group}/{name} keeps history / delay: a world-model state "
        "of one frame cannot rebuild it")
    zeros = torch.zeros(size)
    noise = cfg.noise
    if noise is None:
      return zeros, zeros.clone()
    if not (isinstance(noise, UniformNoiseCfg) and noise.operation == "add"):
      raise ValueError(
        f"obs term {group}/{name} has {type(noise).__name__}"
        f"(operation={getattr(noise, 'operation', None)!r}); the world-model "
        "spec describes additive uniform sensor noise only")

    def per_dim(v):
      # CPU: mjlab swaps cfg.scale for a tensor on the env device at init
      t = torch.as_tensor(v, dtype=torch.float32).detach().cpu().flatten()
      if t.numel() == 1:
        return t.expand(size).clone()
      if t.numel() != size:
        raise ValueError(f"obs term {group}/{name}: noise bound of {t.numel()} "
                         f"values for a {size}-dim term")
      return t
    lo, hi = per_dim(noise.n_min), per_dim(noise.n_max)
    if cfg.scale is not None:
      sc = per_dim(cfg.scale)
      lo, hi = torch.minimum(lo * sc, hi * sc), torch.maximum(lo * sc, hi * sc)
    return lo, hi

  def world_model_spec(self):
    """How a model-based learner reads the actor obs out of the model state.

    The model state is the privileged critic group when one is surfaced
    (``critic_obs_key``), else the actor obs itself. Each actor term is matched
    BY NAME to its noise-free twin in the critic group; its sensor noise comes
    from the actor term's ``UniformNoiseCfg``; the previous-action term (mjlab's
    ``last_action``) is named so the learner can write the action it took
    exactly -- but only when the obs holds exactly that action (unit
    ``ctrl_gain``, nothing overwriting a slice of it, no adversary sharing the
    head). See ``safety_sb3.tensor_env.WorldModelSpec``.
    """
    if WorldModelSpec is None:
      return None
    actor = self._obs_layout(self.obs_key)
    obs_dim = sum(sz for _, sz, _ in actor.values())
    lo, hi = torch.zeros(obs_dim), torch.zeros(obs_dim)
    for name, (st, sz, cfg) in actor.items():
      lo[st:st + sz], hi[st:st + sz] = self._term_noise_bounds(
        self.obs_key, name, sz, cfg)
    if self.critic_obs_key is None:
      index, state = None, actor
    else:
      state = self._obs_layout(self.critic_obs_key)
      index = []
      for name, (_, sz, _) in actor.items():
        twin = state.get(name)
        if twin is None or twin[1] != sz:
          raise ValueError(
            f"actor obs term {name!r} ({sz} dims) has no same-size twin in the "
            f"critic group {self.critic_obs_key!r}; a model-based learner "
            "rebuilds the actor obs from it")
        if twin[2].noise is not None:
          raise ValueError(
            f"critic term {name!r} is noisy; the model state must be the "
            "noise-free twin of the actor obs")
        index.extend(range(twin[0], twin[0] + sz))
    last = None
    if (self.ctrl_gain == 1.0 and not self.adversary
            and self._joint_noise is None and self.hybrid_skill is None):
      for name, (st, sz, cfg) in state.items():
        if (getattr(cfg.func, "__name__", "") == "last_action"
                and not (cfg.params or {}).get("action_name")
                and cfg.scale is None and sz == self.ctrl_dim):
          last = list(range(st, st + sz))
          break
    return WorldModelSpec(actor_obs_index=index, last_action_index=last,
                          obs_noise_low=lo.tolist(), obs_noise_high=hi.tolist())

  def _context_source(self, kind: str, field: str):
    from .filters.rollout import _native
    if kind == "model":
      x = getattr(self.mj.sim.model, field, None)
    else:
      x = getattr(self._robot.data, field, None)
    return None if x is None else _native(x)

  #: MuJoCo constants mj_setConst RECOMPUTES from the randomized fields. The DR
  #: field list carries them (a shadow must copy them), but as context they only
  #: restate the draw: on G1 they are 90 of 123 columns, all functions of the
  #: 3-dim torso COM offset.
  _DERIVED_CONSTANTS = ("body_invweight0", "dof_invweight0", "tendon_invweight0",
                        "body_subtreemass", "actuator_acc0", "actuator_length0",
                        "tendon_length0")

  def _context_columns(self) -> list:
    """Which columns of which DR buffers differ across envs (deduplicated):
    the realized draw, without the per-world copies of shared constants."""
    from .filters.rollout import dr_identity_fields
    model_fields, entity_fields = dr_identity_fields(self.mj)
    model_fields = [f for f in model_fields if f not in self._DERIVED_CONSTANTS]
    n = int(self.mj.num_envs)
    cols, seen = [], []
    if n < 2:
      return cols
    for kind, fields in (("model", model_fields), ("entity", entity_fields)):
      for f in fields:
        x = self._context_source(kind, f)
        if x is None or not x.numel() or x.shape[0] != n:
          continue
        x = x.reshape(n, -1).float()
        keep = []
        for c in (x.std(0) > 1e-7).nonzero().flatten().tolist():
          col = x[:, c]
          if any(torch.equal(col, prev) for prev in seen):
            continue
          seen.append(col)
          keep.append(c)
        if keep:
          cols.append((kind, f, torch.as_tensor(keep, device=x.device)))
    return cols

  def sim_context(self):
    """The realized domain-randomization draw of each env, ``[num_envs, C]``,
    or None when nothing is randomized per env.

    Derived from the env's own DR events (filters.rollout.dr_identity_fields,
    the same list a shadow sim copies to model the SAME robot): every column
    of those buffers that differs across envs, deduplicated, minus the
    constants MuJoCo derives from them -- on the G1 tasks the shared foot
    friction, the torso COM offset and the per-joint encoder biases (33 dims).
    All of them are ``startup`` draws, fixed per env for the run.
    """
    if getattr(self, "_ctx_cols", None) is None:
      self._ctx_cols = self._context_columns()
    if not self._ctx_cols:
      return None
    return torch.cat([self._context_source(kind, f).reshape(
      int(self.mj.num_envs), -1)[:, idx].float()
      for kind, f, idx in self._ctx_cols], dim=1)

  # --- per-episode regimes (safety_sb3 multi-gamma ladder) --------------------
  @property
  def n_regimes(self) -> int:
    """Spawn sources x force buckets (1 when the task exposes neither)."""
    n_src = int(getattr(self.mj, "_zoo_n_spawn_sources", 1) or 1)
    return n_src * max(int(getattr(self, "_regime_buckets", 0)), 1)

  def regime(self) -> Optional[torch.Tensor]:
    """The regime of each env's CURRENT episode, ``[num_envs]`` long, or None.

    ``source * buckets + bucket``: the spawn source the task's spawn published
    for this episode (``mj._zoo_spawn_source``, written at the reset, so it
    already names the new episode right after an auto-reset) and the per-env
    force scale (PerEnvForceScaleCallback's ``force_scale``, moved only at
    episode ends) quantized into ``regime_force_buckets`` equal buckets of
    [0, 1]. Both are fixed for the episode, as the learner's contract asks
    (safety_sb3.tensor_env.TensorVecEnv.regime). The learner copies the tensor.
    """
    buckets = int(getattr(self, "_regime_buckets", 0))
    src = getattr(self.mj, "_zoo_spawn_source", None)
    if src is None and buckets == 0:
      return None
    n = int(self.mj.num_envs)
    dev = self.mj.device
    r = (src.reshape(-1).long() if src is not None
         else torch.zeros(n, dtype=torch.long, device=dev))
    if buckets > 0:
      scale = getattr(self, "force_scale", None)
      if scale is None:
        b = torch.zeros(n, dtype=torch.long, device=dev)
      else:
        b = (scale.reshape(-1).clamp(0.0, 1.0) * buckets).long().clamp_(max=buckets - 1)
      r = r * buckets + b
    return r

  def safety_margins(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The task's (g, l) for the last step, on a DENSE-reward env.

    Only available when the env was built with ``dense_margins=True`` -- a
    plain cumulative env computes no margins at all, and returning zeros there
    would read as "perfectly safe" to a monitor or a failure counter.
    """
    if not self.dense_margins:
      raise RuntimeError(
        "safety_margins() needs an env built with dense_margins=True (the "
        "dense-reward + margins mode). A mode='cumulative' task gets it by "
        "registering kwargs={'dense_margins': True} and a margin_fn.")
    if self._safety_g is None:
      raise RuntimeError("safety_margins() called before the first step")
    return self._safety_g, self._safety_l

  # --- training-safety counters (always on; drained via metrics) -------------

  def _zero_counts(self) -> None:
    """Reset the WINDOW counters. ``_failures_total`` is cumulative over the
    whole run and is never zeroed here."""
    self._c_steps = 0
    self._c_failures = 0
    self._c_episodes = 0
    self._c_g_sum = 0.0
    self._c_g_min = float("inf")
    # [episodes ended, of which safely reached, sum of steps to first reach],
    # on the device (no per-step sync); None until a reach-tracking step.
    self._c_reach = None

  def _count_step(self, terminated, truncated, margin_g,
                  margin_l=None) -> None:
    """Fold one step into the training-safety counters. A failure is a real
    termination (NOT a timeout); ``margin_g`` is None when the env computes no
    task margin (a plain cumulative env). ``margin_l`` feeds the reach
    counters on a task with a real target."""
    dones = terminated | truncated
    timeouts = truncated & ~terminated
    failed = dones & ~timeouts
    n_failed = int(failed.sum())
    self._c_steps += int(terminated.numel())
    self._c_failures += n_failed
    self._failures_total += n_failed
    self._c_episodes += int(dones.sum())
    if margin_g is not None:
      self._c_g_sum += float(margin_g.sum())
      self._c_g_min = min(self._c_g_min, float(margin_g.min()))
    if (getattr(self, "_track_reach", False) and margin_g is not None
        and margin_l is not None):
      self._count_reach(dones, margin_g, margin_l)

  def _count_reach(self, dones, g, l) -> None:
    """Per-episode SAFE reach, the reach-avoid outcome: the episode reached
    ``l >= 0`` at a step where ``g >= 0``, with ``g >= 0`` at every earlier step
    of the episode too. A later fall does not undo it (reach-avoid asks nothing
    after the reach); a ``g < 0`` BEFORE it does, even one the env did not
    terminate on (e.g. the G1's tilt between g's 50 deg and fell_over's 70 deg).
    ``g`` arrives anchored on failure terminations, so a fall is never a reach.

    Margins are those of the state each step produced, so the first reached
    state of an episode is step k >= 1 and its time is k * step_dt.
    """
    ep = self._reach_ep
    if ep is None:
      zf, zb = torch.zeros_like(g), torch.zeros_like(g, dtype=torch.bool)
      ep = self._reach_ep = {"t": zf, "t_hit": zf.clone(), "hit": zb,
                             "bad": zb.clone()}
    ep["t"] = ep["t"] + 1.0
    new = (l >= 0.0) & (g >= 0.0) & ~ep["bad"] & ~ep["hit"]
    ep["t_hit"] = torch.where(new, ep["t"], ep["t_hit"])
    ep["hit"] = ep["hit"] | new
    ep["bad"] = ep["bad"] | (g < 0.0)
    ended = dones & ep["hit"]
    add = torch.stack([dones.sum().float(), ended.sum().float(),
                       (ep["t_hit"] * ended).sum()])
    self._c_reach = add if self._c_reach is None else self._c_reach + add
    keep = ~dones
    ep["t"] = torch.where(keep, ep["t"], 0.0)
    ep["t_hit"] = torch.where(keep, ep["t_hit"], 0.0)
    ep["hit"] = ep["hit"] & keep
    ep["bad"] = ep["bad"] & keep

  def _drain_counts(self) -> dict[str, float]:
    """The training-safety counters as safety/* metrics, then reset the window.

      safety/failures_total   cumulative real terminations (NOT timeouts)
      safety/failures_per_1k  the same, per 1000 env-steps in the window
      safety/failure_rate     fraction of episodes this window ending in failure
      safety/episodes         episodes ended this window
      safety/margin_mean/_min the task margin g (only when the env computes one)
      safety/reach_rate       fraction of episodes ended this window that
                              SAFELY reached the target (see _count_reach);
                              tasks with a real target only
      safety/time_to_reach_s  mean time from episode start to that first safe
                              reach, over those episodes

    Like failure_rate, the reach counters see an episode only once it ENDS.
    Early in a run, before the first timeouts, only falls have ended, so both
    read pessimistic until then.
    """
    out = {"safety/failures_total": float(self._failures_total)}
    if self._c_steps:
      out["safety/failures_per_1k"] = 1000.0 * self._c_failures / self._c_steps
      out["safety/failure_rate"] = (self._c_failures / self._c_episodes
                                    if self._c_episodes else 0.0)
      out["safety/episodes"] = float(self._c_episodes)
      if self._c_g_min != float("inf"):
        out["safety/margin_mean"] = self._c_g_sum / self._c_steps
        out["safety/margin_min"] = self._c_g_min
    if self._c_reach is not None:
      n_ep, n_hit, t_sum = self._c_reach.tolist()
      if n_ep > 0:
        out["safety/reach_rate"] = n_hit / n_ep
      if n_hit > 0:
        out["safety/time_to_reach_s"] = t_sum / n_hit * float(self.mj.step_dt)
    self._zero_counts()
    return out

  def metrics(self) -> dict[str, float]:
    out, self._log = self._log, {}
    out.update(self._drain_counts())
    return out

  def render(self):
    return self.mj.render()

  def close(self):
    try:
      self.mj.close()
    except Exception:
      pass


class MjlabTensorSafetyEnv(_MjlabCore, TensorVecEnv):
  """GPU-resident bridge (primary): torch end-to-end, no numpy bounce.
  Pair with a safety_sb3 1P learner, SafetyPPO1P / ReachAvoidPPO1P
  (auto-detected)."""

  def __init__(self, num_envs=2048, device="cuda:0", *, cfg_builder,
               margin_fn, ctrl_dim=12, dstb_dim=3, ctrl_gain=3.0,
               force_max=50.0, adversary=False, adversary_body="base_link",
               render_mode=None, obs_key=None, dense_reward=False,
               dstb_mode="wrench", dstb_gain=0.25, dstb_joint_slice=None,
               joint_noise=None, hybrid_skill=None,
                 latch_margin_fn=None, end_criterion="failure",
               cfg_overrides=None, dense_margins=False, action_bound=1.0,
               critic_obs_key=None, random_push=None, regime_force_buckets=0):
    if not _HAS_SAFETY_SB3:
      raise ImportError(
        "safety_sb3 is required for the tensor bridge (pip install it or put "
        "the safety-stable-baselines repo on sys.path); the numpy bridge "
        "(make_numpy) works with vanilla stable_baselines3 only.")
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key, dense_reward=dense_reward,
      dstb_mode=dstb_mode, dstb_gain=dstb_gain,
      dstb_joint_slice=dstb_joint_slice, joint_noise=joint_noise,
      hybrid_skill=hybrid_skill,
      latch_margin_fn=latch_margin_fn, end_criterion=end_criterion,
      cfg_overrides=cfg_overrides, dense_margins=dense_margins,
      action_bound=action_bound, critic_obs_key=critic_obs_key,
      random_push=random_push, regime_force_buckets=regime_force_buckets)
    TensorVecEnv.__init__(self, int(num_envs), obs_space, act_space, device)

  def reset(self) -> torch.Tensor:
    obs_dict, _ = self.mj.reset()
    self._obs_dict = obs_dict
    obs = obs_dict[self.obs_key].float()
    self.mj._zoo_last_obs = obs
    if self._joint_noise is not None:
      self._reset_joint_noise()
    if self._push is not None:
      self._push.reset()
      self._apply_pending_push()
      self.mj._zoo_push_quiet = self._push.quiet()
    self._hyb_latch.zero_()
    self._last_l = None
    self._reach_ep = None      # every episode restarts: no reach carried over
    return obs

  def step_tensor(self, actions: torch.Tensor):
    obs, g, terminated, truncated, l = self._mj_step(actions)
    # NaN sanitation (see safety_margin_hook): a NaN obs entering the rollout
    # buffer NaNs the policy update even though the env resets via nan_term.
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    dones = terminated | truncated
    timeouts = truncated & ~terminated
    return obs, g, dones, timeouts, l


from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class MjlabNumpySafetyEnv(_MjlabCore, VecEnv):
  """Classic numpy SB3 ``VecEnv`` bridge (for the SAC family / stock SB3
  tooling). Slower (device<->host each step); prefer the tensor bridge for
  on-policy training."""

  def __init__(self, num_envs=64, device="cuda:0", *, cfg_builder, margin_fn,
               ctrl_dim=12, dstb_dim=3, ctrl_gain=3.0, force_max=50.0,
               adversary=False, adversary_body="base_link", render_mode=None,
               obs_key=None, dense_reward=False, dstb_mode="wrench",
               dstb_gain=0.25, dstb_joint_slice=None, joint_noise=None,
               hybrid_skill=None, latch_margin_fn=None,
               end_criterion="failure", cfg_overrides=None,
               dense_margins=False, action_bound=1.0, critic_obs_key=None,
               random_push=None, regime_force_buckets=0):
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key, dense_reward=dense_reward,
      dstb_mode=dstb_mode, dstb_gain=dstb_gain,
      dstb_joint_slice=dstb_joint_slice, joint_noise=joint_noise,
      hybrid_skill=hybrid_skill,
      latch_margin_fn=latch_margin_fn, end_criterion=end_criterion,
      cfg_overrides=cfg_overrides, dense_margins=dense_margins,
      action_bound=action_bound, critic_obs_key=critic_obs_key,
      random_push=random_push, regime_force_buckets=regime_force_buckets)
    self._device = device
    VecEnv.__init__(self, int(num_envs), obs_space, act_space)
    self._actions = None

  def reset(self):
    obs_dict, _ = self.mj.reset()
    self._obs_dict = obs_dict
    if self._joint_noise is not None:
      self._reset_joint_noise()
    if self._push is not None:
      self._push.reset()
      self._apply_pending_push()
      self.mj._zoo_push_quiet = self._push.quiet()
    self._reach_ep = None      # every episode restarts: no reach carried over
    return obs_dict[self.obs_key].float().cpu().numpy()

  def step_async(self, actions):
    self._actions = np.asarray(actions, dtype=np.float32)

  def step_wait(self):
    a = torch.as_tensor(self._actions, device=self._device)
    obs, g, terminated, truncated, l = self._mj_step(a)
    term = terminated.cpu().numpy()
    trunc = truncated.cpu().numpy()
    dones = np.logical_or(term, trunc)
    obs_np = obs.cpu().numpy()
    l_np = l.cpu().numpy()
    infos = []
    for i in range(self.num_envs):
      info = {"l_x": float(l_np[i])}
      if dones[i]:
        info["terminal_observation"] = obs_np[i]
        info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
      infos.append(info)
    return obs_np, g.cpu().numpy(), dones, infos

  # VecEnv boilerplate
  def _indices(self, indices):
    if indices is None:
      return range(self.num_envs)
    return [indices] if isinstance(indices, int) else indices

  def get_attr(self, attr_name, indices=None):
    return [getattr(self, attr_name, None) for _ in self._indices(indices)]

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *args, indices=None, **kwargs):
    return [None for _ in self._indices(indices)]

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._indices(indices)]
