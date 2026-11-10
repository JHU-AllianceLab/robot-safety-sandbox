"""A G1 fallback's fitted safety values at runtime: the Q-CBF filter's monitor.

``examples/fit_policy_value.py`` and ``examples/fit_policy_value_td.py`` fit,
for each fallback policy pi (``runs/shac_vs_ppo_sonic/{shac,ppo}_s*``), a state
value V^pi(s) and a state-action value Q^pi(s, a) of the g_fall margin, on
states of SONIC's harvested trajectories. Each fit comes in VARIANTS:

    exact      {v,q}_<tag>.pt             gamma = 1, regressed on 3 s rollout
                                          min-g labels
    td_g099    {v,q}_td_g099_<tag>.pt     temporal differences, gamma = 0.99,
                                          bootstrapped (fit_policy_value_td.py
                                          --variant td_g099)

A Q-CBF filter (Oh, Lidard et al., "Safety with Agency", arXiv 2504.11717,
Def. 2) reads them at EVERY control step of the live env, with SONIC as the
nominal controller and pi as the best-effort action:

    u*(x) = argmin_u ||u - u_SONIC(x)||^2   s.t.   Q(x, u) >= kappa * V(x)

A value net is only as good as the match between its runtime input and its
training input. Three details decide that match:

* THE STATE. ``policy_eval.features`` is the 68-D phi of the state
  (``eta_net.phi_of_bank``, x / y / yaw invariant) and the LAST APPLIED env
  action, clamped to the bridge's box +-action_bound = +-3
  (``trajectory_rows`` clamps SONIC's previous target this way). Live, the
  last applied action is ``mj.action_manager.action``, read by
  ``states.snapshot`` together with the state. Under a filter that is the
  FILTERED action, not SONIC's. :func:`runtime_features` is
  ``record_policy_transitions.py``'s ``feats`` op for op, and that function
  reproduced the labeller's values.
* THE ACTION. Q's action input is in env action units and is NOT clamped. Its
  training actions were SONIC's own next action (up to +-20 on the arms) and
  that action plus N(0, 0.3^2) or N(0, 1^2) noise. So Q is trustworthy near
  SONIC's action, and is extrapolated further away from it. The fallback's
  action, clamped to the box, can lie off that support.
* SONIC'S HISTORY. SONIC reads its own last action in its proprioceptive
  history, in raw decoder units and SONIC joint order. When the filter changes
  the applied action, the history can carry the APPLIED one, which is what
  SONIC's training env would have recorded (:func:`record_applied_action`,
  through :func:`env_action_to_sonic_raw`, the inverse of
  ``SonicTracker.act``'s map). It can also keep SONIC's own output, which is
  what the tracker does when left alone. SONIC's raw action is clipped to
  +-ACTION_CLIP, so a filtered action outside :func:`sonic_action_box` would
  be recorded clamped; the driver keeps the filter inside that box.

``tests/test_policy_value_filter.py`` checks the nets against the fits' own
inputs on the held-out test split, and the action map against a real
``SonicTracker``.
"""

from __future__ import annotations

import os

import torch

from robot_safety_sandbox.envs.g1_recover.eta_net import phi_of_bank
from robot_safety_sandbox.envs.g1_recover.policy_eval import ValueNet
from robot_safety_sandbox.envs.g1_recover.sonic import ACTION_CLIP
from robot_safety_sandbox.envs.g1_recover.states import snapshot
from robot_safety_sandbox.filters.monitor import CriticMonitor

_ZOO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.abspath(__file__)))))

#: Where the g_fall fits of the SHAC-vs-PPO policies live.
DEFAULT_VALUE_DIR = os.path.join(_ZOO, "data", "policy_eval_gfall")
#: The rollout-label fit. Every other variant name is a TD fit's ``--variant``.
VARIANT_EXACT = "exact"
#: Widths: phi of the state, the G1's action, V's input (phi + last action)
#: and Q's input (V's + the action).
PHI_DIM = 68
ACTION_DIM = 29
STATE_DIM = PHI_DIM + ACTION_DIM
Q_DIM = STATE_DIM + ACTION_DIM


# --- the frozen nets ---------------------------------------------------------------

def value_path(value_dir: str, kind: str, tag: str, variant: str = VARIANT_EXACT) -> str:
  """``<value_dir>/<kind>_<tag>.pt`` for the exact fit, ``<kind>_<variant>_<tag>.pt``
  for a TD one. ``kind`` is "v" or "q"."""
  if kind not in ("v", "q"):
    raise ValueError(f"value_path: kind must be 'v' or 'q', got {kind!r}")
  stem = f"{kind}_{tag}" if variant == VARIANT_EXACT else f"{kind}_{variant}_{tag}"
  return os.path.join(value_dir, stem + ".pt")


def load_value_checkpoint(path: str, device="cpu") -> tuple[ValueNet, dict]:
  """A fit's network, frozen (eval mode, no parameter gradients), and the rest
  of its checkpoint (``in_dim``, ``hidden``, ``eps``, ``report``, and for a TD
  fit ``gamma`` and ``init``), plus ``path``."""
  if not os.path.exists(path):
    raise FileNotFoundError(f"no value checkpoint {path}")
  ck = torch.load(path, map_location="cpu", weights_only=False)
  missing = [k for k in ("state_dict", "in_dim", "hidden", "eps") if k not in ck]
  if missing:
    raise ValueError(f"{path}: not a policy_eval value checkpoint (missing {missing})")
  net = ValueNet(int(ck["in_dim"]), tuple(int(h) for h in ck["hidden"]))
  net.load_state_dict(ck["state_dict"])
  net = net.to(device).eval()
  net.requires_grad_(False)
  info = {k: v for k, v in ck.items() if k != "state_dict"}
  info["path"] = os.path.abspath(path)
  return net, info


def load_value_net(path: str, device="cpu") -> ValueNet:
  """:func:`load_value_checkpoint`'s network only."""
  return load_value_checkpoint(path, device)[0]


def _check_state(x: torch.Tensor) -> torch.Tensor:
  if x.dim() != 2 or int(x.shape[1]) != STATE_DIM:
    raise ValueError(f"value input must be (N, {STATE_DIM}) = phi {PHI_DIM} + last action "
                     f"{ACTION_DIM}, got {tuple(x.shape)}")
  return x.float()


class PolicyValues:
  """V^pi and Q^pi of one fallback policy (``tag``, e.g. ``ppo_s0``) under one
  fit ``variant`` ("exact" or a TD variant such as "td_g099"), frozen.

  ``v(x)`` and ``q(x, u)`` take :func:`state_features`' 97-D ``x``. ``q`` is
  differentiable in ``u``, and the nets themselves carry no gradient. ``u`` may
  carry leading candidate dimensions ``(*B, N, 29)`` against ``x`` ``(N, 97)``,
  which gives ``(*B, N)``. ``eps`` holds each net's calibrated thresholds
  (``{"v": {...}, "q": {...}}``, keys ``false_safe``, ``residual_0.05``, ...).
  ``gamma`` is the fit's discount (1.0 for the exact fit). ``info`` holds the
  rest of both checkpoints (reports, paths)."""

  def __init__(self, tag: str, variant: str = VARIANT_EXACT,
               value_dir: str = DEFAULT_VALUE_DIR, device="cpu"):
    self.tag, self.variant, self.device = tag, variant, device
    self.v_net, v_info = load_value_checkpoint(value_path(value_dir, "v", tag, variant), device)
    self.q_net, q_info = load_value_checkpoint(value_path(value_dir, "q", tag, variant), device)
    for kind, info, want in (("V", v_info, STATE_DIM), ("Q", q_info, Q_DIM)):
      if int(info["in_dim"]) != want:
        raise ValueError(f"{info['path']}: {kind}'s input is {info['in_dim']}-D, expected {want} "
                         f"(phi {PHI_DIM} + last action {ACTION_DIM}"
                         f"{f' + action {ACTION_DIM}' if kind == 'Q' else ''})")
    gv, gq = float(v_info.get("gamma", 1.0)), float(q_info.get("gamma", 1.0))
    if gv != gq:
      raise ValueError(f"{tag} / {variant}: V was fit with gamma {gv}, Q with {gq}")
    self.gamma = gv
    self.info = {"v": v_info, "q": q_info}
    self.eps = {"v": dict(v_info["eps"]), "q": dict(q_info["eps"])}

  def v(self, x: torch.Tensor) -> torch.Tensor:
    """V(x), ``(N,)``."""
    return self.v_net(_check_state(x))

  def q(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Q(x, u), ``(*B, N)`` for ``u`` ``(*B, N, 29)`` in env action units (not
    clamped, as in training)."""
    x = _check_state(x)
    if u.dim() < 2 or int(u.shape[-1]) != ACTION_DIM or int(u.shape[-2]) != int(x.shape[0]):
      raise ValueError(f"Q's action must be (*B, {x.shape[0]}, {ACTION_DIM}) env actions, "
                       f"got {tuple(u.shape)}")
    if u.dim() > 2:
      x = x.expand(*u.shape[:-1], STATE_DIM)
    return self.q_net(torch.cat([x, u.to(x.dtype)], dim=-1))

  def monitor(self, num_envs: int | None = None, **kw) -> "PolicyValueMonitor":
    """A :class:`PolicyValueMonitor` on these values."""
    return PolicyValueMonitor(self, num_envs, **kw)

  def __repr__(self) -> str:
    return f"PolicyValues({self.tag!r}, {self.variant!r}, gamma={self.gamma}, device={self.device!r})"


# --- the state input -----------------------------------------------------------------

def _bound(b, a: torch.Tensor) -> torch.Tensor:
  t = torch.as_tensor(b, dtype=a.dtype, device=a.device)
  return t if t.dim() == 0 else t.reshape(-1)[:a.shape[1]]


def state_features(s: dict, low, high, device=None) -> torch.Tensor:
  """The value nets' 97-D state input ``(N, 97)`` float32, from a state dict in
  the bank schema: a live ``states.snapshot`` or bank rows (``load_bank``). It
  is phi of the state and the last applied action clamped to ``[low, high]``,
  the env's action box (``env.action_space.low / high``, per joint or scalar;
  the fits' rows were clamped to +-3). This is ``record_policy_transitions.py``'s
  ``feats`` op for op. On bank rows whose action is already in the box it
  equals ``policy_eval.features`` bit for bit."""
  dev = s["action"].device if device is None else device
  a = s["action"].to(dev).float()
  if a.dim() != 2 or int(a.shape[1]) != ACTION_DIM:
    raise ValueError(f"the last action must be (N, {ACTION_DIM}), got {tuple(a.shape)}")
  a = torch.maximum(torch.minimum(a, _bound(high, a)), _bound(low, a))
  return torch.cat([phi_of_bank(s, device=dev).float(), a], dim=1)


def runtime_features(mj, low, high, device=None) -> torch.Tensor:
  """:func:`state_features` of every env of the live mjlab env ``mj``
  (``env.mj``), now. Read it after the step AND after any push (the kick
  changes the twist): it is the state the next action is applied from."""
  return state_features(snapshot(mj), low, high, device)


# --- the monitor ------------------------------------------------------------------------

class PolicyValueMonitor(CriticMonitor):
  """The Q-CBF filter's monitor on one policy's values.

  It follows the call convention of the interventions in
  ``filters/intervention.py``, which forward their ``**ctx`` to the monitor
  and to the fallback alike:

    q_fn(action, x=None, **ctx) -> (N,)   Q(x, action), differentiable in the
                                          action (``(*B, N)`` for candidate
                                          batches ``(*B, N, 29)``)
    v_fn(x=None, **ctx)          -> (N,)  V(x)
    monitor(a, **ctx)            -> (N,)  no-grad Q(x, a) (``CriticMonitor``),
                                          which the least-restrictive switch reads

  ``x`` is this step's :func:`runtime_features`. Pass it in the ctx
  (``x=...``, preferred, because it then travels with the fallback's obs), or
  set it once per step with :meth:`set_state`. Other ctx keys are ignored.
  ``q_offset`` / ``v_offset`` are subtracted from Q / V (e.g. a calibrated
  eps, for a conservative constraint); both default to 0.
  """

  def __init__(self, values: PolicyValues, num_envs: int | None = None, *,
               q_offset: float = 0.0, v_offset: float = 0.0):
    super().__init__(num_envs, values.device, self.q_fn, ACTION_DIM)
    self.values = values
    self.q_offset, self.v_offset = float(q_offset), float(v_offset)
    self._x: torch.Tensor | None = None

  def set_state(self, x: torch.Tensor) -> None:
    """The features every later call without ``x`` uses."""
    self._x = self._state(x)

  def _state(self, x: torch.Tensor | None) -> torch.Tensor:
    if x is None:
      if self._x is None:
        raise RuntimeError("PolicyValueMonitor: no state; pass x=... in the ctx or call set_state(x)")
      return self._x
    x = _check_state(x)
    if self.num_envs is not None and int(x.shape[0]) != int(self.num_envs):
      raise ValueError(f"PolicyValueMonitor: {x.shape[0]} states for {self.num_envs} envs")
    # an inference-mode tensor cannot enter an autograd graph; a clone made
    # outside inference mode can
    return (x.clone() if x.is_inference() else x).detach()

  def q_fn(self, action: torch.Tensor, x: torch.Tensor | None = None, **ctx) -> torch.Tensor:
    del ctx
    return self.values.q(self._state(x), action) - self.q_offset

  def v_fn(self, x: torch.Tensor | None = None, **ctx) -> torch.Tensor:
    del ctx
    return self.values.v(self._state(x)) - self.v_offset


# --- the applied action in SONIC's history ------------------------------------------

def env_action_to_sonic_raw(tracker, a_env: torch.Tensor) -> torch.Tensor:
  """The raw SONIC action ``(N, 29)`` (decoder units, SONIC joint order,
  clamped +-ACTION_CLIP) whose joint targets the env action ``a_env``
  ``(N, 29)`` applies. It inverts ``SonicTracker.act``, where
  ``target = default + raw * sonic_scale`` (action-term order) and
  ``a_env = env_action(target) = (target - env_offset) / env_scale``. In SONIC
  order this is the offline rebuild's ``(target - default) / scale``
  (``sonic.rebuild_inputs``)."""
  if a_env.dim() != 2 or int(a_env.shape[1]) != ACTION_DIM:
    raise ValueError(f"env actions must be (N, {ACTION_DIM}), got {tuple(a_env.shape)}")
  target = tracker.env_offset + a_env * tracker.env_scale           # action-term order
  raw_act = (target - tracker.default) / tracker.sonic_scale
  return raw_act[:, tracker.sonic_to_act].clamp(-ACTION_CLIP, ACTION_CLIP)


def sonic_action_box(tracker) -> tuple[torch.Tensor, torch.Tensor]:
  """``(low, high)``, ``(29,)`` env actions: the box of joint targets SONIC's
  own raw action can command (raw within +-ACTION_CLIP, action-term order).
  SONIC's env action always lies inside it. A filter that keeps its modified
  actions inside (``QCBFDescentIntervention(action_low=, action_high=)``)
  only applies actions that SONIC's history can carry exactly:
  :func:`env_action_to_sonic_raw` never has to clamp them."""
  ends = [tracker.env_action(tracker.default + s * ACTION_CLIP * tracker.sonic_scale)
          for s in (-1.0, 1.0)]
  return torch.minimum(*ends), torch.maximum(*ends)


def record_applied_action(tracker, a_env: torch.Tensor,
                          mask: torch.Tensor | None = None) -> None:
  """Make SONIC's next history entry carry the APPLIED env action ``a_env``
  instead of its own output, for the envs in ``mask`` (all by default). Call it
  after ``tracker.act()``, which sets ``last_action`` to SONIC's output, and
  before ``tracker.advance()``, which rolls ``last_action`` into the history.

  An env whose ``a_env`` is SONIC's own env action bit for bit (a pass-through)
  keeps SONIC's raw exactly. The inverse map is exact only up to rounding, and
  this way a filter that never intervenes leaves SONIC's inputs identical to
  the unfiltered run's. ``tracker.raw`` and ``tracker.target`` stay SONIC's
  own, nominal action."""
  raw = env_action_to_sonic_raw(tracker, a_env)
  own = (a_env == tracker.env_action(tracker.target)).all(dim=1)
  keep = own if mask is None else own | ~mask.reshape(-1).bool()
  tracker.last_action = torch.where(keep.unsqueeze(1), tracker.last_action, raw)
