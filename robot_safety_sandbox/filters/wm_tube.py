"""The world-model tube filter at TRAIN time: the eval's ``wmmin:1:0.95`` and
``wmmin:1:0.95+dwell1`` (``examples/eval_sonic_qcbf.py``, E below) lifted out of
the eval driver into the house triple, so a task policy can learn inside it.

    filt, bundle = wm_tube_filter(env, run_dir, tube_json, tag="thermal70_s2",
                                  dwell_s=1.0, proposal_bound=6.5)
    env = WMTubeFilteredEnv(env, filt)       # robot_safety_sandbox.filtered_env

The triple (core.py: a new filter is a new triple, not a new type):

  fallback      PolicyFallback over the frozen SHAC fallback pi: deterministic, on
                the live ``actor`` group with its last-action columns clamped to
                the fallback box, through its own frozen normalizer, clamped to
                the box (E:868-870)
  monitor       WMTubeMonitor: one noise-free H-step rollout in the fallback's own
                world model from the TRUE ``critic`` state and sim context -- the
                proposal for one model step, then pi on the decoded state -- and
                verdict = min_j g^_j - eps_H(c), the conformal worst-margin bound
                of calibrate_wm_min_tube.py (E:557-627)
  intervention  DwellSwitchIntervention: pi iff verdict < 0, STRICT as the eval's
                Controller (qcbf_eval_lib.py:331); each hand-over holds pi for
                ``dwell_s`` (E:917-933); held envs skip the rollout

The verdicts are the eval's bit for bit on identical id batches
(tests/test_wm_tube_parity.py: torch.equal against ``Evaluator.min_verdict`` on
the real checkpoints). The deliberate differences:

  * the first imagined action is the proposal clamped to ``proposal_bound``
    (6.5 for the track tasks; the eval's first action was SONIC's, clamped to
    +-20). The imagined fallback steps, the fallback's last-action input columns
    and the executed fallback action stay in ``fallback_bound`` (+-3, the box the
    fallback trained and was calibrated in);
  * a NaN verdict engages (``nan_engages=True``); the eval let it pass, NaN < 0
    being False;
  * held envs are not rolled out (the eval computed the rollout and overrode it
    with -1). A noise-free rollout draws no random numbers, so skipping it moves
    no stream; only the batch size of the GEMMs changes, which can flip a
    decision only when |verdict| is at float rounding;
  * pi is SHAC's ``_actor_mean``: bit-identical to the eval's
    ``policy._predict(deterministic=True)``, without building a torch Normal per
    call, whose argument validation syncs the host 50 times a control step.

Every per-step input is a cached group the env step already computed, read by
explicit group name: ``obs_groups()["critic"]`` (the world-model state),
``obs_groups()["actor"]`` (the fallback's input) and ``sim_context()``. Never
``compute_group``: on the actor group it redraws the sensor noise (consuming
torch's RNG), and with ``history_length=1`` it returns the stored frame anyway.
Never ``env.obs_key`` / ``critic_obs()`` / ``world_model_spec()`` either: on the
track tasks those name the TASK policy's groups.

Nothing here imports mjlab at module level; :func:`load_wm_tube` imports
safety_sb3 (through ``eval.policies.load_twin``) when called.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from .core import SafetyFilter
from .fallback import PolicyFallback
from .intervention import Intervention
from .monitor import Monitor
from .telemetry import FilterInfo

#: The calibrated conditions a training or eval filter may use: the frozen
#: 1 s / 0.95 tube, 0.90 for the eval's sensitivity condition, and the 0.4 s /
#: 0.95 tube the filter report evaluated (the 3 s tube is NOT offered: on two of
#: three seeds its world model predicts a fall from almost every state).
ALLOWED_CONDITIONS = ("wmmin:0.4:0.95", "wmmin:1:0.95", "wmmin:1:0.9")
DWELL_SUFFIX = "+dwell"
#: ``<condition>+latch``: once the fallback takes over it keeps the episode
#: (qcbf_eval_lib.LATCH_SUFFIX; E:977-983).
LATCH_SUFFIX = "+latch"
#: The world models a tube may have been calibrated with (min_tube.json's
#: ``wm_thermal``; absent = "nn"): the learned one, or the HYBRID whose
#: temperatures and thermal margin come from the sysID heat model
#: (filters/hybrid_thermal.py, mode "hybrid").
WM_THERMAL = ("nn", "hybrid")
#: The released bundle's per-run provenance file (examples/export_safety_filters.py).
MANIFEST = "manifest.json"
MANIFEST_FORMAT = "g1-thermal-safety-filter/1"
#: ``matmul_precision`` modes (the smoke's bench); "fp32" is the calibration's.
PRECISIONS = ("fp32", "tf32", "bf16")
#: WMTubeMonitor batch modes: "compact" rolls out the active envs only (the
#: eval's, and the default); "full" rolls out every env and masks; "graph" is
#: "full" replayed from a captured CUDA graph (static shapes need the full batch).
MODES = ("compact", "full", "graph")


# --- conditions ----------------------------------------------------------------

@dataclass(frozen=True)
class WMTubeCondition:
  """``wmmin:<horizon_s>:<coverage>[+dwell<dwell_s>|+latch]``; ``name`` is the
  eval's canonical name (qcbf_eval_lib.parse_condition)."""
  name: str
  horizon_s: float = 1.0
  coverage: float = 0.95
  dwell_s: Optional[float] = None
  latch: bool = False

  @property
  def base(self) -> str:
    """The name without the dwell / latch suffix (the calibrated tube it reads)."""
    return self.name.split("+", 1)[0]


def parse_wm_condition(text: str) -> WMTubeCondition:
  """The eval's grammar restricted to ``wmmin`` with an optional ``+dwell`` or
  ``+latch`` (qcbf_eval_lib.py:150-174, 189-200); ValueError on anything else."""
  usage = "wmmin:<seconds>:<c>[+dwell<seconds>|+latch]"
  text = str(text).strip()
  head, sep, tail = text.rpartition("+")
  dwell, latch = None, False
  if sep:
    if "+" + tail == LATCH_SUFFIX:
      latch = True
    elif not ("+" + tail).startswith(DWELL_SUFFIX):
      raise ValueError(f"condition {text!r}: {usage}")
    else:
      try:
        dwell = float(tail[len(DWELL_SUFFIX) - 1:])
      except ValueError:
        raise ValueError(f"condition {text!r}: {usage}") from None
      if not (math.isfinite(dwell) and dwell > 0.0):
        raise ValueError(f"condition {text!r}: the dwell must be > 0 s")
    text = head
  parts = text.split(":")
  if len(parts) != 3 or parts[0] != "wmmin":
    raise ValueError(f"condition {text!r}: {usage}")
  try:
    h, c = float(parts[1]), float(parts[2])
  except ValueError:
    raise ValueError(f"condition {text!r}: {usage}") from None
  if not (math.isfinite(h) and h > 0.0 and 0.0 < c < 1.0):
    raise ValueError(f"condition {text!r}: horizon > 0, 0 < c < 1")
  name = (f"wmmin:{h:g}:{c:g}" + ("" if dwell is None else f"{DWELL_SUFFIX}{dwell:g}")
          + (LATCH_SUFFIX if latch else ""))
  return WMTubeCondition(name, h, c, dwell, latch)


# --- precision and RNG ---------------------------------------------------------

@contextlib.contextmanager
def matmul_precision(mode: str = "fp32"):
  """CUDA fp32 matmuls in IEEE fp32 ("fp32", what the tubes were calibrated
  in), in TF32 ("tf32"), or under bf16 autocast ("bf16"), restored after.

  Only torch's new API (``fp32_precision``) is read and written: it reports a
  TF32 switch made through the legacy flags too, and writing back the old value
  restores the state whichever API made it. A setting that is already right is
  left untouched, because once the new API is written torch refuses the legacy
  getters (``torch.get_float32_matmul_precision``) until it is written back."""
  if mode not in PRECISIONS:
    raise ValueError(f"precision {mode!r}: one of {PRECISIONS}")
  m = torch.backends.cuda.matmul
  prev = m.fp32_precision
  want_tf32 = mode == "tf32"
  touch = (prev == "tf32") != want_tf32
  if touch:
    m.fp32_precision = "tf32" if want_tf32 else "ieee"
  try:
    if mode == "bf16" and torch.cuda.is_available():
      # no weight-cast cache: a captured CUDA graph must not hold stale casts
      with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        yield
    else:
      yield
  finally:
    if touch:
      m.fp32_precision = prev


@contextlib.contextmanager
def preserve_rng():
  """Leave python's random, numpy, torch (CPU and every CUDA device) and the
  cudnn flags as they were.

  SB3's load reseeds all of them (``set_random_seed`` in SHAC's _setup_model,
  safety_sb3/shac.py:396; stable_baselines3 common/utils.py:28-45). The arms that
  load a fallback would otherwise draw different start temperatures (torch's
  global RNG, thermal.py:42-45) from the arm that does not. ``fork_rng`` alone
  covers torch only."""
  py, npy, cpu = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
  cuda = None
  if torch.cuda.is_available():
    # run any lazily queued seed now, so it is part of the state put back
    torch.cuda.init()
    cuda = torch.cuda.get_rng_state_all()
  cudnn = torch.backends.cudnn
  flags = (cudnn.deterministic, cudnn.benchmark)
  try:
    yield
  finally:
    random.setstate(py)
    np.random.set_state(npy)
    torch.random.set_rng_state(cpu)
    if cuda is not None:
      torch.cuda.set_rng_state_all(cuda)
    cudnn.deterministic, cudnn.benchmark = flags


# --- loading -------------------------------------------------------------------

def _sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _last_action_cols(actor_idx: list, last: Optional[list], action_dim: int) -> slice:
  """The actor-obs columns holding the previous action: the positions of the
  spec's critic-state last-action dims inside the actor->state index, which must
  be one contiguous block in action order."""
  if not last or len(last) != action_dim:
    raise ValueError(f"the checkpoint's world-model spec names no {action_dim}-D last action "
                     f"({last}); the fallback's last-action clamp needs it")
  pos = {int(i): p for p, i in enumerate(actor_idx)}
  if any(int(i) not in pos for i in last):
    raise ValueError("the last-action state dims are not all in the actor obs index")
  cols = [pos[int(i)] for i in last]
  if cols != list(range(cols[0], cols[0] + action_dim)):
    raise ValueError(f"the last action sits in actor-obs columns {cols}, not one block")
  return slice(cols[0], cols[0] + action_dim)


@dataclass(eq=False)
class WMTubeBundle:
  """A frozen fallback, its world model and its calibrated tube
  (:func:`load_wm_tube`), plus the pi both the rollout and the live fallback use."""
  model: Any                 # the SHAC learner, inference only
  norm: Callable             # the fallback's frozen actor-obs normalizer (load_twin)
  cond: WMTubeCondition
  horizon: int               # H, model steps
  eps: float                 # eps_H(c) of min_tube.json
  step_dt: float
  fallback_bound: float
  meta: dict
  #: the calibration's thermal-model record (push.json meta.wmrollout.thermal:
  #: mode, limit_c, scale_c, clamp, cold_c, ...); None = the learned model ("nn")
  thermal: Optional[dict] = None

  @property
  def wm_thermal(self) -> str:
    return "nn" if not self.thermal else str(self.thermal.get("mode", "nn"))

  def __post_init__(self):
    m = self.model
    self.wm = m.world_model
    self.device = self.wm.s_mean.device
    self.actor_idx = m._actor_idx_t.to(self.device)
    self.z_clip = m.wm_z_clip
    A = int(self.wm.action_dim)
    spec = getattr(m, "_wm_spec", {}) or {}
    self.la = _last_action_cols(self.actor_idx.tolist(), spec.get("last_action_index"), A)
    b = float(self.fallback_bound)
    self.lo = torch.full((A,), -b, dtype=torch.float32, device=self.device)
    self.hi = torch.full((A,), b, dtype=torch.float32, device=self.device)

  @property
  def actor_dim(self) -> int:
    return int(self.actor_idx.numel())

  def pi_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """``obs`` with the last-action columns clamped to the fallback box: the
    fallback never saw a larger previous action (E:665-677)."""
    seg = obs[:, self.la]
    out = obs.clone()
    out[:, self.la] = torch.maximum(torch.minimum(seg, self.hi), self.lo)
    return out

  def pi(self, obs: torch.Tensor) -> torch.Tensor:
    """The fallback's deterministic action on RAW actor obs ``[n, 127]``, clamped
    to the fallback box (E:600, E:868-870)."""
    return torch.clamp(self.model._actor_mean(self.norm(self.pi_obs(obs))), self.lo, self.hi)


def _manifest_calibration(run_dir: str, zip_path: str, tube_json: str) -> tuple[dict, str]:
  """A released run directory (``manifest.json``, written by
  examples/export_safety_filters.py): the checkpoint, its normalizer and the
  tube must hash to the manifest's sums, and the manifest's calibration record
  (copied from the calibration's push.json at export, where its policy path was
  checked against this zip) must name this zip's sha256. Returns that record
  and the manifest path."""
  path = os.path.join(run_dir, MANIFEST)
  with open(path) as f:
    man = json.load(f)
  if man.get("format") != MANIFEST_FORMAT:
    raise ValueError(f"{path}: format {man.get('format')!r}, expected {MANIFEST_FORMAT!r}")
  files = man.get("files", {})
  for name in ("final_model.zip", "tensornormalize.pt"):
    want = files.get(name)
    p = zip_path if name == "final_model.zip" else os.path.join(run_dir, name)
    if want is None or not os.path.isfile(p) or _sha256(p) != want:
      raise ValueError(f"{p}: missing or its sha256 differs from {path}")
  # abspath, not realpath: a Hugging Face cache links every file to a content blob
  rel = os.path.relpath(os.path.abspath(tube_json), os.path.abspath(run_dir))
  rec = man.get("tubes", {}).get(rel)
  if rec is None:
    raise ValueError(f"{tube_json} is not one of {path}'s tubes {sorted(man.get('tubes', {}))}")
  if _sha256(tube_json) != rec.get("sha256"):
    raise ValueError(f"{tube_json}: sha256 differs from {path}")
  cal = dict(rec.get("calibration") or {})
  if cal.get("policy_sha256") != files["final_model.zip"]:
    raise ValueError(f"{tube_json} was calibrated on a different checkpoint "
                     f"({cal.get('policy_sha256')})")
  return cal, path


def load_wm_tube(run_dir: str, tube_json: str, *, tag: str, cond="wmmin:1:0.95",
                 device: str = "cuda:0", zip_name: str = "final_model.zip",
                 fallback_bound: float = 3.0, step_dt: float = 0.02,
                 wm_thermal: str = "nn") -> WMTubeBundle:
  """Load a SHAC fallback run and its min-tube calibration, refusing anything
  that is not the frozen calibration of THIS checkpoint.

  * the tube's ``tag`` must be ``tag`` (E:1332; the tags are
    ``thermal70_s<i>`` / ``thermalcbf_s<i>``), its ``dt`` the control step, and
    ``min_eps[c][H]`` must exist (E:622-627);
  * the tube's world model (``wm_thermal``: "nn", or "hybrid" = the sysID heat
    model for temperatures and the thermal margin) must be ``wm_thermal``, and
    a hybrid calibration must record its heat-model settings;
  * its calibration run must have evaluated this very zip, noise-free, with the
    box clamp on pi's last action: on this machine through ``push.json`` beside
    ``calib`` (the zip's path); in a released run directory (a ``manifest.json``
    beside the zip) through the manifest's sha256 sums;
  * the world model must be fitted and the policy's box +-``fallback_bound``.

  The checkpoint loads inside :func:`preserve_rng`. ``meta`` records the paths,
  sha256 sums, eps, the run's task and its ``g1_sysid.json``."""
  if isinstance(cond, str):
    cond = parse_wm_condition(cond)
  if cond.base not in ALLOWED_CONDITIONS:
    raise ValueError(f"condition {cond.name!r}: the frozen calibrations are {ALLOWED_CONDITIONS}")
  if wm_thermal not in WM_THERMAL:
    raise ValueError(f"wm_thermal {wm_thermal!r}: one of {WM_THERMAL}")
  zip_path = os.path.join(run_dir, zip_name)
  if not os.path.isfile(zip_path):
    raise FileNotFoundError(f"no fallback checkpoint {zip_path}")
  with open(tube_json) as f:
    tube = json.load(f)
  if tube.get("tag") != tag:
    raise ValueError(f"{tube_json} was calibrated for {tube.get('tag')!r}, not {tag!r}")
  if abs(float(tube.get("dt", math.nan)) - step_dt) > 1e-9:
    raise ValueError(f"{tube_json}: dt {tube.get('dt')} is not the control step {step_dt}")
  tube_mode = str(tube.get("wm_thermal", "nn"))
  if tube_mode != wm_thermal:
    raise ValueError(f"{tube_json} was calibrated with the {tube_mode!r} world model, not "
                     f"{wm_thermal!r}")
  H = int(round(cond.horizon_s / step_dt))
  tab = tube.get("min_eps", {}).get(f"{cond.coverage:g}", {})
  if str(H) not in tab:
    raise ValueError(f"{tube_json} has no min_eps at coverage {cond.coverage:g} for {H} steps")
  eps = float(tab[str(H)])
  manifest = None
  if os.path.isfile(os.path.join(run_dir, MANIFEST)):
    pmeta, manifest = _manifest_calibration(run_dir, zip_path, tube_json)
    push_json = manifest
    thermal = pmeta.get("thermal")
  else:
    push_json = os.path.join(os.path.dirname(str(tube.get("calib", ""))), "push.json")
    if not os.path.isfile(push_json):
      raise FileNotFoundError(f"{tube_json}: its calibration record {push_json} is missing")
    with open(push_json) as f:
      pmeta = json.load(f).get("meta", {})
    if os.path.realpath(str(pmeta.get("policy"))) != os.path.realpath(zip_path):
      raise ValueError(f"{tube_json} calibrated {pmeta.get('policy')}, not {zip_path}")
    thermal = (pmeta.get("wmrollout") or {}).get("thermal")
  if pmeta.get("obs_noise") is not False or pmeta.get("pi_obs_clamp") != "box":
    raise ValueError(f"{push_json}: calibrated with obs_noise={pmeta.get('obs_noise')}, "
                     f"pi_obs_clamp={pmeta.get('pi_obs_clamp')}; this filter is noise-free with "
                     f"the box clamp")
  rec_mode = "nn" if not thermal else str(thermal.get("mode", "nn"))
  if rec_mode != wm_thermal:
    raise ValueError(f"{push_json}: the calibration ran the {rec_mode!r} world model, the tube "
                     f"says {wm_thermal!r}")
  if wm_thermal == "hybrid":
    miss = [k for k in ("limit_c", "scale_c", "clamp", "cold_c") if k not in thermal]
    if miss:
      raise ValueError(f"{push_json}: the hybrid calibration does not record {miss}")
  else:
    thermal = None
  from ..eval.policies import load_twin
  with preserve_rng():
    model, norm = load_twin(zip_path, device, quiet=True)
  model.policy.set_training_mode(False)
  model.policy.requires_grad_(False)
  model.world_model.eval()
  model.world_model.requires_grad_(False)
  # inference only: the Adam moments of the world model are its largest tensors
  model.pi_optimizer = model.vf_optimizer = model.wm_optimizer = None
  wm = model.world_model
  if not bool(wm.fitted):
    raise ValueError(f"{zip_path} has no fitted world model")
  lo, hi = model._act_low, model._act_high
  if not (bool((lo == -fallback_bound).all()) and bool((hi == fallback_bound).all())):
    raise ValueError(f"{zip_path} acts in [{float(lo.min())}, {float(hi.max())}], not "
                     f"+-{fallback_bound}")
  sysid_path = os.path.join(run_dir, "g1_sysid.json")
  sysid = None
  if os.path.isfile(sysid_path):
    with open(sysid_path) as f:
      sysid = json.load(f)
  task = None
  cfg_path = os.path.join(run_dir, "config.yaml")
  if os.path.isfile(cfg_path):
    import yaml
    with open(cfg_path) as f:
      task = (yaml.safe_load(f) or {}).get("task")
  norm_path = os.path.join(run_dir, "tensornormalize.pt")
  meta = {"run_dir": os.path.abspath(run_dir), "zip": os.path.abspath(zip_path),
          "zip_sha256": _sha256(zip_path),
          "normalizer_sha256": _sha256(norm_path) if os.path.isfile(norm_path) else None,
          "tube": os.path.abspath(tube_json), "tube_sha256": _sha256(tube_json), "tag": tag,
          "condition": cond.name, "horizon": H, "coverage": cond.coverage, "eps": eps,
          "step_dt": step_dt, "z_clip": model.wm_z_clip, "state_dim": int(wm.state_dim),
          "context_dim": int(wm.context_dim), "task": task, "sysid": sysid,
          "calibration": {k: pmeta.get(k) for k in ("policy", "policy_sha256", "tag", "task",
                                                    "obs_noise", "pi_obs_clamp", "created")},
          "wm_thermal": wm_thermal, "thermal": thermal, "manifest": manifest}
  return WMTubeBundle(model, norm, cond, H, eps, float(step_dt), float(fallback_bound), meta,
                      thermal)


# --- binding to an env ---------------------------------------------------------

def _base(env):
  """The bridge under any wrappers (the first object in the chain with ``mj``)."""
  e, seen = env, set()
  while e is not None and id(e) not in seen:
    seen.add(id(e))
    d = getattr(e, "__dict__", {})
    if "mj" in d:
      return e
    e = d.get("venv", d.get("env"))
  raise ValueError(f"{type(env).__name__}: no mjlab bridge (an object with .mj) in the chain")


def _qualname(f) -> str:
  t = f if (callable(f) and hasattr(f, "__qualname__")) else type(f)
  return f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', type(t).__name__)}"


def _jsonable(x):
  """A JSON-exact image of a cfg value: tensors as lists, dataclasses as
  dicts, callables and other objects by qualified name (never a repr carrying
  a memory address), non-finite floats as strings (NaN != NaN)."""
  if isinstance(x, np.generic):
    x = x.item()
  if isinstance(x, float):
    return x if math.isfinite(x) else repr(x)
  if x is None or isinstance(x, (bool, int, str)):
    return x
  if isinstance(x, (torch.Tensor, np.ndarray)):
    return _jsonable(x.tolist() if isinstance(x, np.ndarray) else x.detach().cpu().tolist())
  if isinstance(x, (list, tuple)):
    return [_jsonable(v) for v in x]
  if isinstance(x, dict):
    return {str(k): _jsonable(v) for k, v in x.items()}
  if isinstance(x, slice):
    return repr(x)
  if dataclasses.is_dataclass(x) and not isinstance(x, type):
    return {"__type__": _qualname(type(x)),
            **{f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}}
  return _qualname(x)


def _row0(x):
  """Env 0's row of a per-env action-term tensor (scale / offset), or the scalar."""
  if isinstance(x, torch.Tensor):
    return _jsonable(x[0] if x.dim() > 1 else x)
  return _jsonable(x)


def fallback_layout(env, actor_group: str = "actor", critic_group: str = "critic") -> dict:
  """Everything the frozen fallback and its world model read of an env, as
  JSON: per group each term's name, dims, function, params, noise, scale, clip,
  history and delay; the sim-context columns (``_ctx_cols``: which DR buffer
  columns, base.py:827-871); the action terms (class, dim, env 0's scale and
  offset) and the bridge's ctrl gain and dim. Two envs the fallback can run in
  give equal layouts (noise aside: an eval may switch the actor's off)."""
  base = _base(env)
  mj = base.mj
  om = mj.observation_manager
  out = {"groups": {}}
  for g in (actor_group, critic_group):
    terms = []
    for name, dims, cfg in zip(om.active_terms[g], om.group_obs_term_dim[g],
                               om._group_obs_term_cfgs[g]):
      terms.append({"name": name, "dims": list(dims), "func": _qualname(cfg.func),
                    "params": _jsonable(cfg.params), "noise": _jsonable(cfg.noise),
                    "scale": _jsonable(cfg.scale), "clip": _jsonable(cfg.clip),
                    "history_length": int(cfg.history_length or 0),
                    "delay_max_lag": int(cfg.delay_max_lag or 0)})
    out["groups"][g] = {"concatenate": bool(om.group_obs_concatenate[g]),
                        "enable_corruption": bool(getattr(om.cfg[g], "enable_corruption", False)),
                        "terms": terms}
  ctx = base.sim_context()                        # also fills base._ctx_cols
  out["context"] = {"dim": 0 if ctx is None else int(ctx.shape[1]),
                    "columns": [[k, f, idx.tolist()] for k, f, idx in (base._ctx_cols or [])]}
  am = mj.action_manager
  out["actions"] = {}
  for name in am.active_terms:
    t = am.get_term(name)
    out["actions"][name] = {"class": _qualname(type(t)), "dim": int(t.action_dim),
                            "scale": _row0(getattr(t, "scale", None)),
                            "offset": _row0(getattr(t, "offset", None))}
  out["ctrl_gain"] = float(base.ctrl_gain)
  out["ctrl_dim"] = int(base.ctrl_dim)
  return json.loads(json.dumps(out))


def thermal_world_info(env) -> Optional[dict]:
  """The env's thermal world as ``train_model_based._record_thermal`` writes it
  to ``g1_sysid.json`` (train_model_based.py:139-158), or None without a
  thermal model. The margin is read from whichever hook the env carries: the
  safety envs' ``zoo_safety_hook`` or the dense tasks' ``zoo_margin_probe``
  (base.py:185-199)."""
  mj = _base(env).mj
  from ..envs.g1_recover.thermal import LIMIT_C, has_thermal, thermal_state
  if not has_thermal(mj):
    return None
  from ..envs.assets.g1_sysid import provenance
  info = dict(provenance(), thermal_cfg=dataclasses.asdict(thermal_state(mj).cfg))
  bar = mj.action_manager.get_term("joint_pos")
  info["barrier"] = ({"limit_c": bar.limit_c, "gamma": bar.gamma}
                     if hasattr(bar, "safe_targets") else None)
  tm = mj.termination_manager
  info["overheat_termination"] = "motor_overheat" in tm.active_terms
  info["overheat_c"] = (float(tm.get_term_cfg("motor_overheat").params.get("limit_c", LIMIT_C))
                        if info["overheat_termination"] else None)
  rewards = getattr(mj.cfg, "rewards", None) or {}
  hook = rewards.get("zoo_safety_hook") or rewards.get("zoo_margin_probe")
  avoid = getattr((getattr(hook, "params", None) or {}).get("margin_fn"), "avoid_margin", None)
  info["margin_thermal"] = getattr(avoid, "thermal", None)
  return json.loads(json.dumps(info))


#: g1_sysid.json keys the world check compares (the rest is provenance prose).
_WORLD_KEYS = ("version", "commit", "sha256", "thermal_cfg", "barrier", "overheat_termination",
               "overheat_c", "margin_thermal")


def _actor_last_action_cols(om, group: str, action_dim: int) -> slice:
  """E:636-663's rule on an env group: one ``last_action`` term (else one
  named ``actions``), ``action_dim`` wide, with no scale or clip."""
  names = om.active_terms[group]
  cfgs = om._group_obs_term_cfgs[group]
  sizes = [int(np.prod(d)) for d in om.group_obs_term_dim[group]]
  hits = [i for i, c in enumerate(cfgs)
          if getattr(getattr(c, "func", None), "__name__", "") == "last_action"]
  hits = hits or [i for i, nm in enumerate(names) if nm == "actions"]
  if len(hits) != 1:
    raise ValueError(f"group {group!r} has {len(hits)} last-action terms, not one")
  i, c = hits[0], cfgs[hits[0]]
  if sizes[i] != action_dim or c.scale is not None or c.clip is not None:
    raise ValueError(f"group {group!r}: last-action term {names[i]!r} is {sizes[i]}-D with scale "
                     f"{c.scale} / clip {c.clip}, not the raw {action_dim}-D action")
  start = sum(sizes[:i])
  return slice(start, start + action_dim)


#: what :func:`check_binding`'s ``allow`` may exempt (each one a stated
#: difference between the env and the fallback's world, recorded in the result):
#:   "thermal_start": the start-temperature draw (ThermalCfg start_level_c /
#:       start_floor_c / housing_gap_c) -- an initial-state distribution, not
#:       the heat model;
#:   "overheat_termination": an overheat termination the fallback's world lacks
#:       (or the reverse) -- when an episode ends, not the dynamics;
#:   "action_class": a subclass of the fallback world's action term (e.g. one
#:       that runs the filter inside the env and then the same targets / clip).
CHECK_ALLOW = ("thermal_start", "overheat_termination", "action_class")
_START_KEYS = ("start_level_c", "start_floor_c", "housing_gap_c")


def _soften(d, allow):
  """``d`` (a fallback_layout / thermal-world JSON image) without what ``allow`` exempts."""
  if isinstance(d, list):
    return [_soften(v, allow) for v in d]
  if not isinstance(d, dict):
    return d
  out = {k: _soften(v, allow) for k, v in d.items()}
  if "thermal_start" in allow and "ThermalCfg" in str(out.get("__type__", "")):
    for k in _START_KEYS:
      out.pop(k, None)
  return out


def check_binding(bundle: WMTubeBundle, env, *, actor_group: str = "actor",
                  critic_group: str = "critic", layout_ref: Optional[dict] = None,
                  verbose: bool = True, allow: tuple = ()) -> dict:
  """Refuse an env the fallback and its world model cannot read exactly
  (ValueError, every failure listed). Checks:

  * widths: critic group = WM state (171), sim context = WM context (33),
    actor group = the actor index (127);
  * the world-model spec rebuilt from the env BY GROUP NAME (base.py:762-805's
    rule: actor terms matched by name to noise-free critic twins; the critic's
    unscaled ``last_action``) equals the checkpoint's saved ``_wm_spec``
    (safety_sb3/shac.py:405-417). Differing sensor-noise bounds only warn: the
    calibration and the eval ran noise-free;
  * the actor group's last-action columns (E:636-663) are the bundle's;
  * no history beyond the current frame and no delay in either group; the
    critic group uncorrupted;
  * the actor's ``command`` / ``phase`` columns read 0 now (call after a reset);
  * ctrl gain 1 (the last-action obs is the action), 29 actions, and an action
    box that holds the fallback's;
  * the thermal world (:func:`thermal_world_info`) equals the run's
    ``g1_sysid.json`` on every key it records;
  * ``fallback_layout(env) == layout_ref`` (noise compared softly), the env's
    two groups compared as the reference's ``actor`` / ``critic``.

  ``allow``: differences to accept, from :data:`CHECK_ALLOW` (default none).

  Returns what was checked, for the run meta."""
  allow = tuple(allow)
  bad_allow = [a for a in allow if a not in CHECK_ALLOW]
  if bad_allow:
    raise ValueError(f"check_binding allow {bad_allow}: not in {CHECK_ALLOW}")
  base = _base(env)
  om = base.mj.observation_manager
  wm = bundle.wm
  errs, warns = [], []
  og = base.obs_groups()
  S, A, C = og[critic_group], og[actor_group], base.sim_context()
  if int(S.shape[1]) != int(wm.state_dim):
    errs.append(f"critic group {critic_group!r} is {S.shape[1]}-D, the world model's state "
                f"{wm.state_dim}-D")
  if (0 if C is None else int(C.shape[1])) != int(wm.context_dim):
    errs.append(f"sim context is {None if C is None else C.shape[1]}-D, the world model's "
                f"{wm.context_dim}-D")
  if int(A.shape[1]) != bundle.actor_dim:
    errs.append(f"actor group {actor_group!r} is {A.shape[1]}-D, the fallback's {bundle.actor_dim}")
  for g in (actor_group, critic_group):
    for name, cfg in zip(om.active_terms[g], om._group_obs_term_cfgs[g]):
      if (cfg.history_length or 0) > 1 or (cfg.delay_max_lag or 0) > 0:
        errs.append(f"{g}/{name} keeps history {cfg.history_length} / delay {cfg.delay_max_lag}")
  if getattr(om.cfg[critic_group], "enable_corruption", False):
    errs.append(f"critic group {critic_group!r} is corrupted (sensor noise on the WM state)")
  saved = getattr(bundle.model, "_wm_spec", {}) or {}
  if not errs:
    actor, state = base._obs_layout(actor_group), base._obs_layout(critic_group)
    index, lo, hi = [], [], []
    for name, (st, sz, cfg) in actor.items():
      twin = state.get(name)
      if twin is None or twin[1] != sz or twin[2].noise is not None:
        errs.append(f"actor term {name!r} has no same-size noise-free twin in {critic_group!r}")
        continue
      index.extend(range(twin[0], twin[0] + sz))
      l_, h_ = base._term_noise_bounds(actor_group, name, sz, cfg)
      lo.extend(l_.tolist())
      hi.extend(h_.tolist())
    last = None
    for name, (st, sz, cfg) in state.items():
      if (getattr(cfg.func, "__name__", "") == "last_action"
              and not (cfg.params or {}).get("action_name") and cfg.scale is None
              and sz == base.ctrl_dim):
        last = list(range(st, st + sz))
        break
    if index != list(saved.get("actor_obs_index") or []):
      errs.append("the actor->state index rebuilt from the env differs from the checkpoint's")
    if last != saved.get("last_action_index"):
      errs.append(f"last-action state dims {last} differ from the checkpoint's "
                  f"{saved.get('last_action_index')}")
    if (saved.get("obs_noise_low") is not None
            and (lo != saved["obs_noise_low"] or hi != saved["obs_noise_high"])):
      warns.append(f"actor sensor-noise bounds differ from the checkpoint's (actor corruption "
                   f"{getattr(om.cfg[actor_group], 'enable_corruption', None)})")
    try:
      la = _actor_last_action_cols(om, actor_group, int(wm.action_dim))
      if la != bundle.la:
        errs.append(f"actor last-action columns {la} are not the bundle's {bundle.la}")
    except ValueError as e:
      errs.append(str(e))
    for name in ("command", "phase"):
      if name not in actor:
        errs.append(f"actor group has no {name!r} term (the fallback reads it pinned to 0)")
        continue
      st, sz, _ = actor[name]
      if float(A[:, st:st + sz].abs().max()) != 0.0:
        errs.append(f"actor {name!r} columns {st}:{st + sz} are not 0")
  if float(base.ctrl_gain) != 1.0 or int(base.ctrl_dim) != int(wm.action_dim):
    errs.append(f"ctrl_gain {base.ctrl_gain} / ctrl_dim {base.ctrl_dim}: the fallback needs 1.0 / "
                f"{wm.action_dim}")
  if float(base.action_space.high.min()) < bundle.fallback_bound:
    errs.append(f"the env's action box {float(base.action_space.high.min())} cannot hold the "
                f"fallback's +-{bundle.fallback_bound}")
  world = thermal_world_info(base)
  rec = bundle.meta.get("sysid")
  unchecked = []
  if rec is None:
    if world is not None:
      errs.append("the env has a thermal model, the fallback run recorded none (g1_sysid.json)")
  elif world is None:
    errs.append("the fallback run is thermal (g1_sysid.json), the env has no thermal model")
  else:
    for k in _WORLD_KEYS:
      if k not in rec:
        # older records: no overheat_c when the run had no overheat termination
        if k == "overheat_c" and rec.get("overheat_termination") is False:
          want = None
        else:
          unchecked.append(k)
          continue
      else:
        want = rec[k]
      have = world.get(k)
      if k in ("overheat_termination", "overheat_c") and "overheat_termination" in allow:
        if have != want:
          warns.append(f"thermal world {k}: env {have!r}, fallback run {want!r} (allowed)")
        continue
      if k == "thermal_cfg" and "thermal_start" in allow:
        if have != want:
          warns.append(f"thermal world start draw: env {have!r}, fallback run {want!r} (allowed)")
        have = {kk: vv for kk, vv in (have or {}).items() if kk not in _START_KEYS}
        want = {kk: vv for kk, vv in (want or {}).items() if kk not in _START_KEYS}
      if have != want:
        errs.append(f"thermal world {k}: env {have!r}, fallback run {want!r}")
  if layout_ref is not None:
    lay = fallback_layout(base, actor_group, critic_group)
    lay["groups"] = {"actor": lay["groups"][actor_group], "critic": lay["groups"][critic_group]}
    if "action_class" in allow:
      for k in ("actions",):
        for name, t in (lay.get(k) or {}).items():
          ref_t = (layout_ref.get(k) or {}).get(name)
          if ref_t is not None and t.get("class") != ref_t.get("class"):
            warns.append(f"action term {name!r}: {t.get('class')} in place of "
                         f"{ref_t.get('class')} (allowed)")
            t["class"] = ref_t.get("class")
    soft = ("noise", "enable_corruption")

    def strip(d):
      d = _soften(d, allow)
      if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in soft}
      if isinstance(d, list):
        return [strip(v) for v in d]
      return d
    if strip(lay) != strip(layout_ref):
      diff = [k for k in sorted(set(lay) | set(layout_ref)) if strip(lay.get(k)) != strip(
        layout_ref.get(k))]
      errs.append(f"fallback_layout differs from the reference in {diff}")
    elif lay != layout_ref:
      warns.append("fallback_layout: sensor noise differs from the reference")
  if errs:
    raise ValueError("[wm-tube] the env does not bind the fallback:\n  " + "\n  ".join(errs))
  for w in warns:
    print(f"[wm-tube] WARNING: {w}")
  out = {"state_dim": int(S.shape[1]), "context_dim": 0 if C is None else int(C.shape[1]),
         "actor_dim": int(A.shape[1]), "last_action_cols": [bundle.la.start, bundle.la.stop],
         "groups": [actor_group, critic_group], "world": world, "world_unchecked": unchecked,
         "layout_checked": layout_ref is not None, "warnings": warns, "allowed": list(allow)}
  if verbose:
    wtxt = f"equal but {unchecked} unrecorded" if unchecked else "equal"
    print(f"[wm-tube] binding OK: {actor_group} {out['actor_dim']}-D, {critic_group} "
          f"{out['state_dim']}-D, context {out['context_dim']}-D; last action "
          f"{bundle.la.start}:{bundle.la.stop}; world {wtxt}; layout "
          f"{'equal' if layout_ref is not None else 'not given'}")
  return out


# --- the triple ----------------------------------------------------------------

class WMTubeMonitor(Monitor):
  """Delta(x, u) = min_{j<H} g^_j - eps_H(c): the world model's predicted margin
  over one noise-free rollout of "u for one model step, then pi", step j's value
  pushed to <= -0.1 once the predicted P(terminated by j) passes 0.5 (E:557-603,
  619-627). NaN where an env was not rolled out.

  ``__call__(a, fallback=None, *, wm_state, sim_ctx, active=None)``: ``a`` the
  proposals ``[N, A]`` (clamped to ``proposal_bound`` here), ``wm_state`` the raw
  critic group ``[N, S]``, ``sim_ctx`` ``[N, C]``, ``active`` the envs to roll
  out (None: all). ``.last`` holds this call's ``pred_min`` (min g^ before eps),
  ``done_clamped`` and ``computed``.

  :meth:`margins` is the per-step trace for the eval's wmcalib and the audit's
  oracle replay; :attr:`eps` and :attr:`horizon` are the tube's.

  :param mode: "compact" (roll out the active envs only), "full" (all, masked
      after) or "graph" ("full" replayed from a CUDA graph captured per batch
      size and precision).
  :param precision: :func:`matmul_precision` mode of the rollout.
  :param hybrid: an ``AnalyticThermal`` (filters/hybrid_thermal.py, mode
      "hybrid") on the SAME env the states come from: the rollout's
      temperatures and thermal margin come from the sysID heat model, started
      from the env's true winding / housing / structure temperatures of the
      rolled-out envs (E:612-620, the eval's ``--wm-thermal hybrid``). Not with
      mode "graph" (the start reads live env buffers by index).
  """

  def __init__(self, num_envs: int, device: str, bundle: WMTubeBundle, *,
               proposal_bound: float, mode: str = "compact", precision: str = "fp32",
               action_dim: int | None = None, hybrid=None):
    super().__init__(num_envs, device, action_dim)
    if mode not in MODES:
      raise ValueError(f"mode {mode!r}: one of {MODES}")
    if precision not in PRECISIONS:
      raise ValueError(f"precision {precision!r}: one of {PRECISIONS}")
    if mode == "graph" and torch.device(device).type != "cuda":
      raise ValueError("mode='graph' needs a CUDA device")
    if (hybrid is None) != (bundle.wm_thermal == "nn"):
      raise ValueError(f"the tube was calibrated with the {bundle.wm_thermal!r} world model; "
                       f"hybrid={'given' if hybrid is not None else 'None'}")
    if hybrid is not None:
      if mode == "graph":
        raise ValueError("mode='graph' does not support the hybrid world model")
      if getattr(hybrid, "mode", None) != "hybrid":
        raise ValueError(f"hybrid.mode {getattr(hybrid, 'mode', None)!r}, expected 'hybrid'")
    self.hyb = hybrid
    self.b = bundle
    self.eps, self.horizon = float(bundle.eps), int(bundle.horizon)
    self.proposal_bound = float(proposal_bound)
    A = int(bundle.wm.action_dim)
    self.p_lo = torch.full((A,), -self.proposal_bound, dtype=torch.float32, device=device)
    self.p_hi = torch.full((A,), self.proposal_bound, dtype=torch.float32, device=device)
    self.mode, self.precision = mode, precision
    self._source = None
    self._graphs: dict = {}
    self.last: dict = {}

  def bind(self, env, critic_group: str = "critic") -> "WMTubeMonitor":
    """Make :meth:`margins` read ``env``'s cached critic group and sim context
    when no state is passed."""
    self._source = (env, critic_group)
    return self

  def _read_source(self):
    if self._source is None:
      raise ValueError("WMTubeMonitor.margins: pass wm_state / sim_ctx or bind() an env")
    env, g = self._source
    return torch.nan_to_num(env.obs_groups()[g].float()), env.sim_context()

  def _rollout(self, s: torch.Tensor, c: Optional[torch.Tensor], a_first: torch.Tensor, H: int,
               ids: Optional[torch.Tensor] = None):
    """E:557-603 with one noise-free rollout: ``[n, H]`` margins and the
    predicted survival after H steps. ``ids``: the env rows ``s`` came from
    (None: all, in order); only the hybrid reads them (the true temperatures)."""
    b, wm, hyb = self.b, self.b.wm, self.hyb
    cn = wm.encode_context(c)
    z = wm.encode(s)
    alive = torch.ones(int(s.shape[0]), device=s.device)
    out = torch.empty(int(s.shape[0]), H, device=s.device)
    act = a_first
    if hyb is not None:
      if ids is None:
        ids = torch.arange(int(s.shape[0]), device=s.device)
      T, hctx = hyb.start(ids)
      s_cur = s
    for t in range(H):
      st = wm.step(z, act, cn, z_clip=b.z_clip)
      if hyb is not None:
        # E:615-618: the heat model advances the temperatures over the step's
        # substeps, writes them into the decoded state, and gives the margin
        T, s_cur, g, done_logit = hyb.model_step(T, hctx, s_cur, st, act.float(), cn)
        z = wm.encode(s_cur)
      else:
        z, g, done_logit = st.z_next, st.g, st.done_logit
      alive = alive * (1.0 - torch.sigmoid(done_logit))
      out[:, t] = torch.where(alive < 0.5, torch.clamp_max(g, -0.1), g)
      if t + 1 < H:
        dec = s_cur if hyb is not None else wm.decode(z)
        act = b.pi(dec.index_select(-1, b.actor_idx))
    return out, alive

  def _graph_rollout(self, s, c, a_first, H):
    """:meth:`_rollout` captured once per (batch, H, precision) as a CUDA graph
    and replayed on copies of the inputs (static shapes, hence the full batch)."""
    key = (int(s.shape[0]), H, self.precision)
    g = self._graphs.get(key)
    if g is None:
      st = {"s": s.clone(), "c": None if c is None else c.clone(), "a": a_first.clone()}
      side = torch.cuda.Stream()
      side.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(side):
        for _ in range(2):                     # warm up allocator and kernels off the graph
          self._rollout(st["s"], st["c"], st["a"], H)
      torch.cuda.current_stream().wait_stream(side)
      graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
        st["out"], st["alive"] = self._rollout(st["s"], st["c"], st["a"], H)
      g = self._graphs[key] = (graph, st)
    graph, st = g
    st["s"].copy_(s)
    if c is not None:
      st["c"].copy_(c)
    st["a"].copy_(a_first)
    graph.replay()
    return st["out"].clone(), st["alive"].clone()

  @torch.no_grad()
  def margins(self, ids: Optional[torch.Tensor], a_first: torch.Tensor, H: Optional[int] = None,
              *, wm_state: Optional[torch.Tensor] = None,
              sim_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The predicted margins ``[n, H]`` of envs ``ids`` (None: all) for
    ``a_first`` ``[n, A]`` (clamped to the proposal box) then pi: E's
    ``wm_margins(ids, a_first, H)[:, 0]``. States come from ``wm_state`` /
    ``sim_ctx`` (full-batch tensors ``ids`` indexes) or the bound env."""
    H = self.horizon if H is None else int(H)
    S, C = (self._read_source() if wm_state is None
            else (torch.nan_to_num(wm_state.float()), sim_ctx))
    if ids is not None:
      S, C = S[ids], (None if C is None else C[ids])
    a = torch.clamp(a_first.float(), self.p_lo, self.p_hi)
    with matmul_precision(self.precision):
      return self._rollout(S, C, a, H, ids=ids)[0]

  @torch.no_grad()
  def __call__(self, a: torch.Tensor, fallback=None, *, wm_state: torch.Tensor,
               sim_ctx: Optional[torch.Tensor], active: Optional[torch.Tensor] = None,
               **_ctx) -> torch.Tensor:
    N, H = int(a.shape[0]), self.horizon
    a_p = torch.clamp(a.float(), self.p_lo, self.p_hi)
    S = torch.nan_to_num(wm_state.float())            # E:572's sanitation, idempotent
    pred = torch.full((N,), math.nan, device=a.device)
    dclamp = torch.zeros(N, dtype=torch.bool, device=a.device)
    computed = (torch.ones(N, dtype=torch.bool, device=a.device) if active is None
                else active.bool())
    with matmul_precision(self.precision):
      if self.mode == "compact" and active is not None:
        ids = computed.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
          out, alive = self._rollout(S[ids], None if sim_ctx is None else sim_ctx[ids],
                                     a_p[ids], H, ids=ids)
          pred[ids] = out.amin(dim=1)
          dclamp[ids] = alive < 0.5
      else:
        roll = self._graph_rollout if self.mode == "graph" else self._rollout
        out, alive = roll(S, sim_ctx, a_p, H)
        pm = out.amin(dim=1)
        pred = pm if active is None else torch.where(computed, pm, pred)
        dclamp = computed & (alive < 0.5)
    self.last = {"pred_min": pred, "done_clamped": dclamp, "computed": computed}
    return pred - self.eps


class DwellSwitchIntervention(Intervention):
  """The eval's hard switch with its ``+dwell`` hold (E:917-933; the rule of
  qcbf_eval_lib.Controller, :331): the fallback iff verdict < 0 -- STRICT,
  where LeastRestrictiveIntervention engages at <= eps -- and every hand-over
  to the fallback (engaged now, not on the env's previous step) holds it for
  ``dwell_steps`` steps, the hand-over's own included. When the hold has run
  out the verdict decides again step by step; a new hold starts only at the
  next release -> engage edge.

  Held envs are not monitored: the monitor gets ``active=~held`` (the eval
  computed their rollout and overrode it with -1 before the controller saw it,
  so the actions are the same). A NaN verdict engages when ``nan_engages``.

  ``latch=True`` (the eval's ``+latch``, E:977-983): the first hand-over holds
  the fallback until the episode ends (``fired``); no dwell with it. Under a
  latch :meth:`hold_left` is 0 (the hold has no length to count down; the
  engaged flag, set for the rest of the episode, carries it).

  Per-env state ``k`` (steps since the episode began), ``hold_until``,
  ``fired`` and ``prev_eng`` -- cleared by ``reset(done)``, which the caller
  MUST call every step (core.py). ``.last`` holds this step's ``held``,
  ``new_handover``, ``switch``, raw ``verdict`` (NaN where held) and the
  fallback's ``a_safe``.
  """

  def __init__(self, num_envs: int, device: str, *, dwell_steps: int = 0,
               nan_engages: bool = True, action_dim: int | None = None,
               latch: bool = False):
    super().__init__(num_envs, device, action_dim)
    if int(dwell_steps) < 0:
      raise ValueError(f"dwell_steps {dwell_steps} < 0")
    if latch and int(dwell_steps):
      raise ValueError("a latch holds to the episode's end: no dwell with it")
    self.dwell_steps, self.nan_engages = int(dwell_steps), bool(nan_engages)
    self.latch = bool(latch)
    self.k = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.hold_until = torch.zeros_like(self.k)
    self.prev_eng = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.fired = torch.zeros_like(self.prev_eng)
    self.last: dict = {}

  def hold_left(self) -> torch.Tensor:
    """Forced steps still to come, ``[N]`` long (0 outside a hold; 0 under a latch)."""
    if self.latch:
      return torch.zeros_like(self.k)
    return (self.hold_until - self.k).clamp_min(0)

  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    D = self.dwell_steps
    holds = bool(D) or self.latch
    if self.latch:
      held = self.fired.clone()
    else:
      held = self.k < self.hold_until if D else torch.zeros_like(self.prev_eng)
    verdict = monitor(a_nom, fallback=fallback, active=~held if holds else None, **ctx)
    v = torch.where(held, torch.full_like(verdict, -1.0), verdict)
    eng = v < 0.0
    if self.nan_engages:
      eng = eng | torch.isnan(v)
    a_safe = fallback(**ctx)
    action = torch.where(eng.unsqueeze(-1), a_safe.to(a_nom.dtype), a_nom)
    new = eng & ~self.prev_eng
    if D:
      # the hold counts from THIS step's k: pi holds through k + D - 1
      self.hold_until = torch.where(new, self.k + D, self.hold_until)
    if self.latch:
      self.fired = self.fired | eng
    switch = eng != self.prev_eng
    self.prev_eng = eng
    self.k = self.k + 1
    self.last = {"held": held, "new_handover": new, "switch": switch, "verdict": verdict,
                 "a_safe": a_safe}
    return action, FilterInfo(engaged=eng, value=verdict,
                              intervention=(action - a_nom).norm(dim=-1))

  def reset(self, done: torch.Tensor) -> None:
    d = done.bool()
    self.k = torch.where(d, torch.zeros_like(self.k), self.k)
    self.hold_until = torch.where(d, torch.zeros_like(self.hold_until), self.hold_until)
    self.prev_eng = self.prev_eng & ~d
    self.fired = self.fired & ~d


class _FallbackPi:
  """PolicyFallback's policy_fn: the bundle's pi on the live raw actor group,
  always IEEE fp32 (it sets the executed action, as in the eval)."""

  def __init__(self, bundle: WMTubeBundle):
    self.b = bundle

  @torch.no_grad()
  def __call__(self, *, actor_obs: torch.Tensor, **_ctx) -> torch.Tensor:
    with matmul_precision("fp32"):
      return self.b.pi(actor_obs)


def wm_tube_filter(env, run_dir: Optional[str] = None, tube_json: Optional[str] = None, *,
                   tag: Optional[str] = None, condition: str = "wmmin:1:0.95",
                   dwell_s: float = 0.0, latch: bool = False,
                   proposal_bound: float, fallback_bound: float = 3.0,
                   device: Optional[str] = None, nan_engages: bool = True,
                   mode: str = "compact", precision: str = "fp32", critic_group: str = "critic",
                   actor_group: str = "actor", wm_thermal: str = "nn",
                   bundle: Optional[WMTubeBundle] = None) -> tuple[SafetyFilter, WMTubeBundle]:
  """PolicyFallback + WMTubeMonitor + DwellSwitchIntervention on ``env``.

  ``dwell_s`` must be a whole number of control steps (0: no hold; 1.0: the
  eval's ``+dwell1``); ``condition`` may carry the same dwell as a suffix.
  ``latch`` (or a ``+latch`` condition): the first hand-over holds the fallback
  to the episode's end. ``proposal_bound`` is the TASK policy's box (the first
  imagined action and the executed non-engaged action are clamped to it),
  ``fallback_bound`` the fallback's. ``wm_thermal``: the world model the tube
  was calibrated with ("hybrid": the sysID heat model gives the rollout's
  temperatures and thermal margin; the env must be the 70 C margin world).
  ``bundle`` reuses a :func:`load_wm_tube` result (then ``run_dir``,
  ``tube_json``, ``tag`` and ``wm_thermal`` are not read). Checks the widths of
  the env's groups; :func:`check_binding` is the full check. Returns
  ``(filter, bundle)``."""
  cond = parse_wm_condition(condition)
  base = _base(env)
  dt = float(base.mj.step_dt)
  D = int(round(float(dwell_s) / dt))
  if dwell_s < 0 or abs(D * dt - float(dwell_s)) > 1e-9:
    raise ValueError(f"dwell_s {dwell_s} is not a whole number of {dt} s control steps")
  if cond.dwell_s is not None and abs(cond.dwell_s - float(dwell_s)) > 1e-9:
    raise ValueError(f"condition {cond.name!r} and dwell_s={dwell_s} disagree")
  latch = bool(latch) or cond.latch
  if latch and D:
    raise ValueError("latch or dwell, not both")
  dev = str(env.device if device is None else device)
  if bundle is None:
    if run_dir is None or tube_json is None or tag is None:
      raise ValueError("wm_tube_filter needs run_dir, tube_json and tag, or a bundle")
    bundle = load_wm_tube(run_dir, tube_json, tag=tag, cond=cond, device=dev,
                          fallback_bound=fallback_bound, step_dt=dt, wm_thermal=wm_thermal)
  elif bundle.cond.base != cond.base or abs(bundle.step_dt - dt) > 1e-9:
    raise ValueError(f"bundle {bundle.cond.name!r} at {bundle.step_dt} s is not {cond.base!r} "
                     f"at {dt} s")
  wm = bundle.wm
  og = base.obs_groups()
  ctx = base.sim_context()
  bad = []
  if int(og[critic_group].shape[1]) != int(wm.state_dim):
    bad.append(f"{critic_group} {og[critic_group].shape[1]}-D vs state {wm.state_dim}-D")
  if (0 if ctx is None else int(ctx.shape[1])) != int(wm.context_dim):
    bad.append(f"context {None if ctx is None else ctx.shape[1]}-D vs {wm.context_dim}-D")
  if int(og[actor_group].shape[1]) != bundle.actor_dim:
    bad.append(f"{actor_group} {og[actor_group].shape[1]}-D vs {bundle.actor_dim}-D")
  if int(env.action_space.shape[0]) != int(wm.action_dim):
    bad.append(f"action {env.action_space.shape[0]}-D vs {wm.action_dim}-D")
  if bad:
    raise ValueError("[wm-tube] env does not fit the fallback: " + "; ".join(bad))
  env_bound = float(env.action_space.high.max())
  if abs(env_bound - float(proposal_bound)) > 1e-9:
    print(f"[wm-tube] note: proposals clamped to +-{proposal_bound}, the env's box is "
          f"+-{env_bound}")
  N, A = int(env.num_envs), int(wm.action_dim)
  hyb = None
  if bundle.wm_thermal == "hybrid":
    from .hybrid_thermal import AnalyticThermal
    th = bundle.thermal
    hyb = AnalyticThermal(base, wm, "hybrid", cold_c=float(th["cold_c"]),
                          clamp=float(th["clamp"]), critic_group=critic_group,
                          floor_c=th.get("floor_c"))
    if (hyb.limit_c, hyb.scale_c) != (float(th["limit_c"]), float(th["scale_c"])):
      raise ValueError(f"the env's thermal margin ({hyb.limit_c} C, {hyb.scale_c} C) is not the "
                       f"calibration's ({th['limit_c']} C, {th['scale_c']} C)")
    if hyb.D != int(th.get("substeps", hyb.D)) or \
            abs(hyb.dt - float(th.get("physics_dt", hyb.dt))) > 1e-12:
      raise ValueError(f"the env's physics ({hyb.D} x {hyb.dt} s) is not the calibration's")
  mon = WMTubeMonitor(N, dev, bundle, proposal_bound=proposal_bound, mode=mode,
                      precision=precision, action_dim=A, hybrid=hyb).bind(base, critic_group)
  itv = DwellSwitchIntervention(N, dev, dwell_steps=D, nan_engages=nan_engages, action_dim=A,
                                latch=latch)
  fb = PolicyFallback(N, dev, _FallbackPi(bundle), A)
  name = cond.base + ("" if D == 0 else f"{DWELL_SUFFIX}{D * dt:g}") + (
    LATCH_SUFFIX if latch else "")
  print(f"[wm-tube] {name} on {bundle.meta.get('tag')} ({bundle.wm_thermal} world model): H "
        f"{bundle.horizon} steps, eps {bundle.eps:.4f}, hold "
        f"{'to the episode end' if latch else f'{D} steps'}, proposal +-{proposal_bound:g}, "
        f"fallback +-{bundle.fallback_bound:g}, NaN {'engages' if nan_engages else 'passes'}, "
        f"{mode} {precision}")
  return SafetyFilter(fb, mon, itv), bundle
