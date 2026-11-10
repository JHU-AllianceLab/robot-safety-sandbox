"""Training callbacks for zoo runs: wandb videos, normalizer persistence,
adversary force ramp, warm-start normalizer freeze.

All battle-tested on the Go2 gap pipeline; see PORTING.md for when to use
each. Everything logs through SB3's logger (wandb via sync_tensorboard).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StdFloorCallback(BaseCallback):
  """Bound the policy action log_std into [min_std, max_std] at each rollout
  start (right after the prior update). The floor prevents the premature std
  COLLAPSE that freezes a policy in a sub-optimal gait; the ceiling prevents the
  entropy-bonus log-std RUNAWAY that inflates std into aggressive thrashing
  (robots fall over -> ep_len crashes). Together = sustained BOUNDED exploration."""

  def __init__(self, min_std=0.2, max_std=None):
    super().__init__()
    self.min_log = math.log(min_std)
    self.max_log = math.log(max_std) if max_std is not None else None

  def _on_rollout_start(self):
    import torch as th
    ls = getattr(self.model.policy, "log_std", None)
    if ls is not None:
      with th.no_grad():
        ls.clamp_(min=self.min_log, max=self.max_log)

  def _on_step(self):
    return True


def _clip_path(outdir, step):
  """``<run>/videos/step_<N>.mp4`` — created on demand."""
  d = os.path.join(outdir, "videos")
  os.makedirs(d, exist_ok=True)
  return os.path.join(d, f"step_{int(step):012d}.mp4")


def _record_clip(env, predict, n_frames, fps, path, bound):
  """Roll ``predict`` on ``env`` for ``n_frames`` and STREAM them into an mp4.

  Frames go straight to the encoder instead of being accumulated: a 20 s 720p
  clip is ~2.8 GB of raw frames, and the array the old path built (list ->
  np.stack -> transpose) held that twice before handing it to wandb, which then
  re-encoded it. One frame is in flight here, and the finished FILE is what
  wandb uploads.

  ``bound`` is the action clamp, read from the render env's own action space —
  NOT a hardcoded +-1. A task whose bridge action_bound is wider (the G1 tasks
  use 3.0) would otherwise be rendered stride-capped, i.e. the clip would show a
  policy that was never trained (the T006 failure mode, at video time).
  """
  import imageio
  writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=6,
                              macro_block_size=1)
  try:
    obs = env.reset()
    for _ in range(n_frames):
      obs = predict(obs, bound)
      writer.append_data(np.asarray(env.render()))
  finally:
    writer.close()
  return path


class _ClipRecorder:
  """Shared plumbing for the two video callbacks: the render env, the clip
  length (in seconds of robot motion or in frames), and the mp4 + optional
  upload. Subclasses supply ``_predict_step``."""

  def _init_clip(self, eval_env_fn, interval, video_len, video_seconds, fps,
                 outdir, to_wandb, herd):
    self.eval_env_fn, self.interval = eval_env_fn, interval
    self.video_len, self.video_seconds, self.fps = video_len, video_seconds, fps
    self.to_wandb = bool(to_wandb)
    self.herd = herd
    self._outdir = outdir or tempfile.mkdtemp(prefix="zoo_video_")
    self._env = None
    self._last = -10**15
    self._bound = 1.0

  def _start_clip_env(self):
    # interval <= 0 means "no clips": do not even BUILD the render env. It costs
    # a second mjlab construction plus the GPU state for 8 extra envs, which a
    # run that switched videos off is explicitly trying not to pay. (Before the
    # recorder moved out of the wandb block, --no-wandb skipped this for free.)
    if self.interval <= 0:
      return
    self._env = self.eval_env_fn()
    # mjlab renders the tracked env + its nearest neighbors in ONE scene (a herd,
    # like the unitree-rl-mjlab videos). herd=None keeps that: draw every eval
    # env. An explicit herd (0 = one robot, the car_goal showcase framing) is
    # both cheaper -- the per-frame cost is dominated by the scene rebuild, not
    # the pixels -- and easier to read for a single-robot gait.
    n_extra = (max(1, int(self._env.num_envs) - 1) if self.herd is None
               else int(self.herd))
    try:
      self._env.mj.cfg.viewer.max_extra_envs = n_extra
    except Exception:
      pass
    # The clamp the clip must use; see _record_clip.
    try:
      self._bound = float(self._env.action_space.high[0])
    except Exception:
      self._bound = 1.0
    # video_seconds is in seconds of ROBOT MOTION, resolved against this env's
    # control dt so it means the same thing whatever the decimation is.
    if self.video_seconds is not None:
      try:
        dt = float(self._env.mj.step_dt)
        self.video_len = max(1, int(round(self.video_seconds / dt)))
      except Exception:
        pass

  def _close_clip_env(self):
    """Close the render env at the end of training. Without this, mujoco's
    Renderer.__del__ runs during interpreter shutdown, where making the EGL
    context current fails -- printing an 'Exception ignored in __del__'
    EGLError traceback after the run has already succeeded."""
    env, self._env = self._env, None
    if env is not None:
      try:
        env.close()
      except Exception:
        pass

  def _mirror_push(self):
    """A random-push clip env is built bare (force_max at the bridge default,
    no per-env scale), so without this a clip would show weak pushes whatever
    the curriculum has reached. Copy the training bridge's force_max and its
    MEAN per-env scale onto the clip env, so the clip shows the push the
    policy is being trained against right now."""
    clip = self._env
    if getattr(clip, "_push", None) is None:
      return
    train = getattr(self, "model", None)
    train = getattr(train, "env", None)
    while hasattr(train, "venv"):
      train = train.venv
    fmax = getattr(train, "force_max", None)
    if fmax is None:
      return
    clip.force_max = float(fmax)
    scale = getattr(train, "force_scale", None)
    if scale is not None:
      import torch as th
      clip.force_scale = th.full((int(clip.num_envs),), float(scale.mean()),
                                 device=clip.device)

  def _emit_clip(self):
    self._mirror_push()
    from robot_safety_sandbox.eval.policies import reset_policy_memory
    reset_policy_memory(self.model.policy)          # a clip starts fresh episodes
    path = _record_clip(self._env, self._predict_step, self.video_len,
                        self.fps, _clip_path(self._outdir, self.num_timesteps),
                        self._bound)
    print(f"[video] {path} ({self.video_len} frames @ {self.fps} fps)",
          flush=True)
    if self.to_wandb:
      import wandb
      if wandb.run is not None:
        # Hand wandb the FILE, not a frame array: no second copy, no re-encode.
        wandb.log({"eval/video": wandb.Video(path, format="mp4")},
                  step=self.num_timesteps)


class VideoWandbCallback(_ClipRecorder, BaseCallback):
  """Record eval clips of the current deterministic policy.

  Uses a SEPARATE small render env (built with adversary off — show the
  deployable ctrl policy). First clip fires immediately (warm-start
  behavior), then every ``interval`` env-steps.

  Every clip is written to ``<outdir>/videos/step_<N>.mp4``; ``to_wandb`` then
  ALSO uploads that file. The local write is unconditional on purpose — a run
  keeps its own footage whether or not wandb is in the picture, and a 720p clip
  is the kind of thing you do not want to re-render to look at twice.
  """

  def __init__(self, eval_env_fn, interval=25_000_000, video_len=300,
               video_seconds=None, fps=30, outdir=None, to_wandb=True,
               herd=None):
    super().__init__()
    self._init_clip(eval_env_fn, interval, video_len, video_seconds, fps,
                    outdir, to_wandb, herd)

  def _on_training_start(self):
    self._start_clip_env()

  def _on_training_end(self):
    self._close_clip_env()

  def _on_step(self):
    # interval <= 0 DISABLES the clips. Without this the comparison reads
    # "0 >= 0" every step and the callback renders a video PER STEP -- the
    # natural way to spell "no videos" silently became the most expensive
    # possible setting, on the runs (long, shared GPU) that most need it off.
    if (self.interval > 0 and self._env is not None
        and self.num_timesteps - self._last >= self.interval):
      self._last = self.num_timesteps
      try:
        self._emit_clip()
      except Exception as e:  # a bad clip must never kill a long train
        print(f"[video] skipped: {e}", flush=True)
    return True

  def _predict_step(self, obs, bound):
    import torch as th
    tvn = self.model.env if hasattr(self.model.env, "normalize_obs") else None
    o = obs if tvn is None else tvn.normalize_obs(obs)
    with th.no_grad():
      act = self.model.policy._predict(o, deterministic=True)
    obs, _g, _d, _t, _l = self._env.step_tensor(th.clamp(act, -bound, bound))
    if bool(_d.any()):
      from robot_safety_sandbox.eval.policies import reset_policy_memory
      reset_policy_memory(self.model.policy, _d)
    return obs


class DenseMetricsCallback(BaseCallback):
  """Forward the numpy bridge's ``env.metrics()`` (mjlab per-term reward /
  termination / curriculum logs) to the SB3 logger -> wandb, once per rollout.

  Stock SB3 PPO knows nothing about the bridge's ``metrics()`` (the tensor-path
  the safety learners drain it themselves), so without this the per-term reward
  breakdown is
  collected and silently dropped. Reachable through the VecMonitor/VecNormalize
  wrappers via VecEnvWrapper attribute forwarding."""

  def _on_rollout_end(self):
    fn = getattr(self.model.env, "metrics", None)
    if not callable(fn):
      return
    for k, v in (fn() or {}).items():
      self.logger.record(f"env/{k}", float(v))

  def _on_step(self):
    return True


class DenseVideoWandbCallback(_ClipRecorder, BaseCallback):
  """eval/video for the STAGE-1 dense walker (stock SB3 PPO on the NUMPY bridge).

  Renders on a SEPARATE tensor env built in DENSE mode (no force / no safety
  hook), so the clip shows the UNAIDED gait. Bounces the render env's tensor obs
  -> numpy, normalizes with the training VecNormalize, predicts with the stock
  policy, steps the tensor env.

  The trigger fires on ROLLOUT boundaries (once per policy update), not per env
  step: within a rollout the policy is frozen, so a clip can only change after an
  update -> one-per-rollout is the finest MEANINGFUL cadence. At N envs a rollout
  advances num_timesteps by N*n_steps, so any ``interval`` below that resolves to
  "every rollout"."""

  def __init__(self, eval_env_fn, interval=10_000_000, video_len=300,
               video_seconds=None, fps=30, outdir=None, to_wandb=True,
               herd=None):
    super().__init__()
    self._init_clip(eval_env_fn, interval, video_len, video_seconds, fps,
                    outdir, to_wandb, herd)

  def _on_training_start(self):
    self._start_clip_env()

  def _on_training_end(self):
    self._close_clip_env()

  def _on_step(self):
    return True

  def _on_rollout_end(self):
    # interval <= 0 disables the clips; see VideoWandbCallback._on_step.
    if (self.interval > 0 and self._env is not None
        and self.num_timesteps - self._last >= self.interval):
      self._last = self.num_timesteps
      try:
        self._emit_clip()
      except Exception as e:  # a bad clip must never kill a long train
        print(f"[dense-video] skipped: {e}", flush=True)

  def _predict_step(self, obs, bound):
    import torch as th
    vn = self.model.env if hasattr(self.model.env, "normalize_obs") else None
    o = obs.detach().cpu().numpy()
    if vn is not None:
      o = vn.normalize_obs(o)
    act, _ = self.model.predict(o, deterministic=True)  # stock PPO, numpy
    obs, _g, _d, _t, _l = self._env.step_tensor(
      th.as_tensor(np.clip(act, -bound, bound), dtype=th.float32,
                   device=obs.device))
    return obs


class VecNormSaveCallback(BaseCallback):
  """Save the numpy VecNormalize obs stats alongside periodic checkpoints.

  The stock CheckpointCallback saves only the policy; without the matching
  obs-normalizer a mid-run checkpoint can't be evaluated (or handed to Stage 2)
  faithfully. Writes ``vecnormalize_<step>.pkl`` next to the model checkpoints."""

  def __init__(self, outdir, save_freq_steps=10_000_000):
    super().__init__()
    self.outdir, self.freq = outdir, save_freq_steps
    self._last = 0

  def _on_step(self):
    if self.num_timesteps - self._last >= self.freq:
      self._last = self.num_timesteps
      if hasattr(self.model.env, "save"):
        os.makedirs(self.outdir, exist_ok=True)
        self.model.env.save(os.path.join(
          self.outdir, f"vecnormalize_{self.num_timesteps}.pkl"))
    return True


class TensorNormSaveCallback(BaseCallback):
  """Save TensorVecNormalize stats alongside periodic checkpoints (needed for
  faithful mid-run eval / warm-starting)."""

  def __init__(self, outdir, save_freq_steps=25_000_000):
    super().__init__()
    self.outdir, self.freq = outdir, save_freq_steps
    self._last = 0

  def _on_step(self):
    if self.num_timesteps - self._last >= self.freq:
      self._last = self.num_timesteps
      if hasattr(self.model.env, "save"):
        os.makedirs(self.outdir, exist_ok=True)
        self.model.env.save(os.path.join(
          self.outdir, f"tensornorm_{self.num_timesteps}.pt"))
    return True


class CriticWarmupCallback(BaseCallback):
  """Freeze the ACTOR for the first ``warmup_steps`` so the critic can fit the
  inherited policy before any policy gradient is taken.

  For a WARM-STARTED run whose value net was reset (``--reset-value``), the first
  updates compute advantages from a randomly-initialized critic: the policy is
  pushed in a direction that carries no information, which is exactly the gait
  the warm-start was meant to preserve. This blocks gradients into every actor
  parameter (the policy MLP, the action head and ``log_std``) while the value
  loss trains normally, so the critic converges
  toward V^pi for the policy that is actually collecting the data -- on-policy
  and correct -- and the first real policy gradient points somewhere meaningful.

  The actor keeps ACTING throughout; only its parameters are frozen. Freezing it
  and stopping it from acting would leave nothing to learn a value for.

  Two interactions worth knowing. ``approx_kl`` reads ~0 while frozen (the PPO
  ratio is exactly 1), so a ``target_kl`` early-stop never fires and every epoch
  runs -- maximum critic data per rollout.

  And ``--max-std`` is NOT blocked by the freeze: that callback clamps
  ``log_std`` under ``no_grad``, writing the tensor directly, so a warm-started
  policy whose std exceeds the cap is pulled down to it on the first rollout and
  held there (measured: a walker at std 0.507 lands exactly on log(0.4) under the
  reach-avoid recipe). This is the behavior to want -- the critic then fits the
  policy that will actually keep training -- but it does mean "frozen actor" is
  about GRADIENTS, not about every byte of the actor staying put.

  Watch ``train/explained_variance``: once it climbs past ~0.7 the critic is
  tracking returns and the warmup has done its job.

  Sibling of :class:`NormFreezeCallback` (which freezes the OBS NORMALIZER for
  the same reason, on the same kind of run) -- the two are complementary and
  usually both want to be on for a warm-start.
  """

  #: Parameter-name fragments that belong to the VALUE pathway of an SB3
  #: ActorCriticPolicy; everything else in the policy is the actor.
  _VALUE_KEYS = ("value_net", "memory_c")

  def __init__(self, warmup_steps: int = 0):
    super().__init__()
    self.warmup_steps = int(warmup_steps)
    self._frozen: list = []
    self._released = False

  def _actor_params(self):
    return [(n, p) for n, p in self.model.policy.named_parameters()
            if not any(k in n for k in self._VALUE_KEYS)]

  def _on_training_start(self):
    if self.warmup_steps <= 0:
      self._released = True
      return
    pol = self.model.policy
    if not hasattr(pol, "value_net"):
      # Not an ActorCriticPolicy (the SAC family keeps .actor / .critic instead).
      # Fail loud rather than "freeze" nothing and report a warmup that never ran.
      raise SystemExit(
        "[critic-warmup] this policy has no .value_net, so the actor/critic split "
        f"this callback assumes does not apply ({type(pol).__name__}). It targets "
        "the PPO family (SB3 ActorCriticPolicy and the safety_sb3 subclasses).")
    self._frozen = [p for _n, p in self._actor_params() if p.requires_grad]
    for p in self._frozen:
      p.requires_grad_(False)
    n_val = sum(1 for n, _ in pol.named_parameters()
                if any(k in n for k in self._VALUE_KEYS))
    print(f"[critic-warmup] actor FROZEN for the first {self.warmup_steps:,} env-steps "
          f"({len(self._frozen)} actor tensors held, {n_val} value tensors training)",
          flush=True)

  def _release(self):
    for p in self._frozen:
      p.requires_grad_(True)
    self._frozen = []
    self._released = True
    print(f"[critic-warmup] actor RELEASED at {self.num_timesteps:,} env-steps "
          f"-- normal PPO resumes", flush=True)

  def _on_step(self):
    if not self._released and self.num_timesteps >= self.warmup_steps:
      self._release()
    self.logger.record("warmup/actor_frozen", float(not self._released))
    return True

  def _on_training_end(self):
    # A run that ends INSIDE the warmup window must not leave the actor frozen in
    # the saved model -- a later warm-start from it would load parameters whose
    # requires_grad is False and silently train nothing.
    if not self._released:
      self._release()


class NormFreezeCallback(BaseCallback):
  """Freeze obs-normalizer updates for the first ``freeze_steps`` of a
  WARM-STARTED run: the inherited policy expects the loaded stats; letting
  them drift while the policy adapts to a new task distribution destabilizes
  the transfer."""

  def __init__(self, freeze_steps=5_000_000):
    super().__init__()
    self.freeze_steps = freeze_steps
    self._released = False

  def _on_training_start(self):
    if hasattr(self.model.env, "training"):
      self.model.env.training = False

  def _on_step(self):
    if not self._released and self.num_timesteps >= self.freeze_steps:
      self._released = True
      if hasattr(self.model.env, "training"):
        self.model.env.training = True
    return True


class PerEnvForceScaleCallback(BaseCallback):
  """rsl_rl-inc1b anti-collapse lever for two-player (2P) training: scale
  the adversary's force PER ENV by a survival curriculum — envs that fail get
  a weaker adversary (+recover), envs that survive get a stronger one. This is
  what stabilized the reference two-player game after pinned curricula and the
  global force ramp were not enough.

  ``name`` is the bridge attribute the per-env scale is written to, and the
  stem of its log keys. ``"force_scale"`` (default) multiplies a WRENCH
  adversary's force magnitude (``base._MjlabCore._apply_dstb``);
  ``"dstb_scale"`` multiplies a JOINT-partition adversary's joint command
  (``dstb_mode="joint"``, e.g. the G1 arm adversary) on top of the global
  ``dstb_gain`` ramp. Same survival rule either way: each env settles near the
  adversary strength at which it survives about half its episodes, clamped to
  ``[lo, hi]``.

  Logs ``game/<name>_mean`` and ``game/<name>_at_floor`` (the fraction of envs
  pinned at ``lo``). A mean that climbs to ``hi`` says the controller has caught
  up with the full-strength adversary; an at_floor fraction near 1 says it is
  losing even at the weakest allowed adversary — lower ``lo``.
  """

  def __init__(self, step=0.05, lo=0.3, hi=1.0, init=0.5, name="force_scale"):
    super().__init__()
    self.step, self.lo, self.hi, self.init = step, lo, hi, init
    self.name = str(name)
    self._bridge = None
    self._scale = None

  def _on_training_start(self):
    import torch as th
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e
    self._scale = th.full((e.num_envs,), self.init, device=e.device)
    setattr(e, self.name, self._scale)

  def _on_step(self):
    import torch as th
    dones = self.locals.get("dones")
    timeouts = self.locals.get("timeouts")
    if dones is None or not th.is_tensor(dones) or not bool(dones.any()):
      return True
    d = dones.bool()
    survived = d & timeouts.bool()
    failed = d & ~timeouts.bool()
    self._scale += self.step * survived.float() - self.step * failed.float()
    self._scale.clamp_(self.lo, self.hi)
    self.logger.record(f"game/{self.name}_mean", float(self._scale.mean()))
    self.logger.record(f"game/{self.name}_at_floor",
                       float((self._scale <= self.lo + 1e-6).float().mean()))
    return True


class FwdForceAnnealCallback(BaseCallback):
  """Anneal the crawl constant forward-current force (bootstraps forward
  locomotion) from ``start`` N -> 0 over the run: hold ``start`` for ``hold_steps``
  (let the gait form), then linearly decay to 0 over ``anneal_steps``, then 0.
  Sets env._fwd_force_scale, read by crawl_duck_margins. Final policy crawls
  unaided (the 0-force ablation showed it already partly can)."""

  def __init__(self, start=15.0, hold_steps=40_000_000, anneal_steps=160_000_000):
    super().__init__()
    self.start, self.hold, self.anneal = start, hold_steps, anneal_steps
    self._mj = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)

  def _on_step(self):
    t = self.num_timesteps
    if t < self.hold:
      val = self.start
    elif t < self.hold + self.anneal:
      val = self.start * (1.0 - (t - self.hold) / self.anneal)
    else:
      val = 0.0
    if self._mj is not None:
      self._mj._fwd_force_scale = val
    self.logger.record("crawl/fwd_force", float(val))
    return True


class GaitThreshRampCallback(BaseCallback):
  """Trot-threshold curriculum for the crawl gait-l. Keeps env._gait_thresh just
  above the achieved trot-score (EMA) so the gait sub-margin stays a mild-negative,
  reachable gradient (a fixed high threshold is masked by gamma*V' -> no learning).
  Monotonic-up; creeps toward trot_ema + gap, capped at `cap`."""

  def __init__(self, start=0.56, cap=0.88, gap=0.05, ema=0.01, step=0.002):
    super().__init__()
    self.start, self.cap, self.gap, self.ema_a, self.step = start, cap, gap, ema, step
    self._mj = None
    self._ema = start

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)
    if self._mj is not None:
      self._mj._gait_thresh = self.start

  def _on_step(self):
    if self._mj is None:
      return True
    tr = getattr(self._mj, "_gait_trot_last", None)
    if tr is not None:
      self._ema += self.ema_a * (float(tr) - self._ema)
      cur = getattr(self._mj, "_gait_thresh", self.start)
      target = min(self.cap, self._ema + self.gap)
      if target > cur:
        self._mj._gait_thresh = min(self.cap, cur + self.step)
    self.logger.record("crawl/gait_thresh",
                       float(getattr(self._mj, "_gait_thresh", self.start)))
    self.logger.record("crawl/trot_ema", float(self._ema))
    return True


class ForceRampCallback(BaseCallback):
  """Two-player (2P) runs: ramp the adversary force from ``force_start``
  to ``force_max`` over ``ramp_steps`` so ctrl adapts to a strengthening
  attacker instead of collapsing under the full bound from step one."""

  def __init__(self, force_max=50.0, ramp_steps=1_000_000_000, force_start=8.0):
    super().__init__()
    self.fmax, self.ramp, self.fstart = force_max, ramp_steps, force_start
    self._bridge = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e

  def _on_step(self):
    frac = min(1.0, self.num_timesteps / max(1, self.ramp))
    self._bridge.force_max = self.fstart + (self.fmax - self.fstart) * frac
    return True


class LAnnealCallback(BaseCallback):
  """Reach-set curriculum for warm-started reach-avoid: ramp ``env._l_alpha``
  from 0 -> 1 so the reach target ``l`` contracts from LOOSE (alpha=0, ~= the
  non-fallen set the avoid base already satisfies -> reach-avoid ~= avoid-only,
  no OOD jump) to STRICT (alpha=1, the true stay-in-place set). Holds alpha=0
  for ``hold_steps`` first (let the re-inflated policy re-settle onto the warm
  base), then linearly to 1 over ``anneal_steps``, then strict. Sets the attr
  the margin reads via ``getattr(env, '_l_alpha', 1.0)``. Same env-reach pattern
  as FwdForceAnnealCallback (through the bridge's ``.mj``)."""

  def __init__(self, anneal_steps=120_000_000, hold_steps=20_000_000):
    super().__init__()
    self.anneal = max(1, anneal_steps)
    self.hold = hold_steps
    self._mj = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)
    if self._mj is not None:
      self._mj._l_alpha = 0.0

  def _on_step(self):
    t = self.num_timesteps
    if t < self.hold:
      a = 0.0
    else:
      a = min(1.0, (t - self.hold) / self.anneal)
    if self._mj is not None:
      self._mj._l_alpha = a
    self.logger.record("curriculum/l_alpha", float(a))
    return True


class DstbPhaseCallback(BaseCallback):
  """Tell the env which PLAYER is currently collecting a rollout, so a task's
  margin function can hand the two players different targets.

  The two-player PPO learner is zero-sum by construction: ``ppo_2p`` negates the
  disturbance advantage on the SAME reach-avoid targets the control player was
  scored on, so the env's single (g, l) channel is all either player ever sees.
  A task that needs the disturbance to be penalized for something of its OWN —
  ``g1_arm_adversary``, where the adversary drives the robot's arms and must not
  win by driving them through the torso — has nowhere to put that penalty
  without also paying the control player for it, which is the wrong sign.

  This callback opens the one channel that is missing: it reads the learner's
  own ``_phase()`` at each rollout start (it is constant for the whole rollout,
  which is why this hook and not ``_on_step`` — the latter would mislabel the
  first step after every phase switch) and stamps it on the mjlab env as
  ``_zoo_dstb_phase``. A margin_fn reads it with
  ``getattr(env, "_zoo_dstb_phase", False)`` and shapes accordingly; every task
  that does not read it is completely unaffected.

  Same env-reach pattern as :class:`LAnnealCallback` (through the bridge's
  ``.mj``). On a 1P learner there is no ``_phase`` and the flag stays False, so
  attaching this to a single-player run is a no-op rather than an error.
  """

  def __init__(self):
    super().__init__()
    self._mj = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)
    if self._mj is not None:
      self._mj._zoo_dstb_phase = False

  def _on_rollout_start(self):
    if self._mj is None:
      return
    phase = getattr(self.model, "_phase", None)
    is_dstb = bool(phase() == "dstb") if callable(phase) else False
    self._mj._zoo_dstb_phase = is_dstb
    self.logger.record("game/dstb_phase", float(is_dstb))

  def _on_step(self):
    return True


class DstbGainRampCallback(BaseCallback):
  """Ramp a JOINT-partition adversary's authority from ``gain_start`` to
  ``gain_max`` over ``ramp_steps`` — the ``dstb_mode="joint"`` twin of
  :class:`ForceRampCallback`.

  Same reason as the force ramp: an adversary handed its full authority at step
  zero simply throws the robot down every episode, the control player never sees
  a survivable state, and the game freezes with no gradient. Here the authority
  is ``bridge.dstb_gain``, the multiplier on the adversary's joint command
  (which mjlab's action term then scales per joint), so 1.0 means "the adversary
  drives its joints with exactly the authority the actor has over its own".
  """

  def __init__(self, gain_max=1.0, ramp_steps=1_000_000_000, gain_start=0.1):
    super().__init__()
    self.gmax, self.ramp, self.gstart = gain_max, ramp_steps, gain_start
    self._bridge = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e

  def _on_step(self):
    frac = min(1.0, self.num_timesteps / max(1, self.ramp))
    self._bridge.dstb_gain = self.gstart + (self.gmax - self.gstart) * frac
    self.logger.record("game/dstb_gain", float(self._bridge.dstb_gain))
    return True


class TargetScaleRampCallback(BaseCallback):
  """Ramp a goal command's sampling MAGNITUDE from ``scale_start`` to 1.0 over
  ``ramp_steps`` -- the goal-side twin of :class:`DstbGainRampCallback`.

  The command term exposes ``scale`` as a fraction of its full goal range (e.g.
  ``envs/g1_waist/command.py``: goals drawn from ``+-scale * max_angle``), and
  sets ``ramped = True`` so the trainer knows to attach this. Only the TRAINING
  env is ramped; an env built elsewhere (eval, play, the training video) keeps
  the term's default scale of 1.0, the full goal range.

  Logs ``curriculum/<term>_scale``.
  """

  def __init__(self, term_name: str, ramp_steps: int = 1_000_000_000,
               scale_start: float = 0.0):
    super().__init__()
    self.term_name = str(term_name)
    self.ramp, self.start = int(ramp_steps), float(scale_start)
    self._term = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._term = e.mj.command_manager.get_term(self.term_name)
    self._term.scale = self.start

  def _on_step(self):
    frac = min(1.0, self.num_timesteps / max(1, self.ramp))
    self._term.scale = self.start + (1.0 - self.start) * frac
    self.logger.record(f"curriculum/{self.term_name}_scale",
                       float(self._term.scale))
    return True
