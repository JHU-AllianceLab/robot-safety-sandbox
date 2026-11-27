"""Label every start state of a bank with K rollouts of a recovery policy.

    python examples/label_start_states.py \
        --policy runs/g1_recover_push_fast_privcrit/g1_recover_push_fast/final_model.zip \
        --bank data/g1_recover_states_lafan1_train.pt \
        --out data/g1_recover_states_lafan1_train_labeled.pt --k 8

    python examples/label_start_states.py --rl-dh \
        --policy runs/sonic_baselines/rldh_s0/g1_recover/final_model.zip \
        --task g1_recover --no-push --bank runs/shac_vs_ppo_sonic/eval/ppo_s0.pt \
        --k 4 --out runs/sonic_baselines/eval/rldh_s0.pt

    python examples/label_start_states.py \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip \
        --task g1_recover --no-push --no-obs-noise \
        --bank runs/shac_vs_ppo_sonic/eval/ppo_s0.pt --k 1 --wm-tube wmmin:1:0.95 \
        --tube-calib runs/shac_vs_ppo_sonic/wmmin_calib/shac_s0/min_tube.json \
        --out runs/sonic_baselines/eval/shac_s0_tube.pt

A row's LABEL is how often the policy survives an episode started from it
under the push distribution the training env uses: ``p_surv`` = survived / K.
It is what the frontier samplers weight with (``p (1 - p)``, spawn.py
``sampler="weighted"``) and what the E2 question is asked against: does the
policy's own value V(s) at the start rank the rows like the rollouts do? If
it does (AUC >= 0.9), the value can stand in for rollout labels and the
value-weighted arm (S5) needs no labeling pass at all.

THE ROLLOUTS. One env, ``--task`` (default ``g1_recover_lafan``), spawning
with the SEQUENTIAL sampler (``envs/g1_recover/spawn.py``): a cursor walks
every ``start_ok`` row in bank order, each ``--k`` times in a row, so a pass
over the bank issues each row exactly K episodes and no env idles. The push
is the training push: one horizontal random push per episode (``max_pushes``
1), its onset 0.1-0.4 s after the spawn, 0.2 s long, at ``--force-max x s``
newtons with ``s ~ U(--force-min-frac, 1)`` redrawn for every episode (the
bridge's per-env ``force_scale``); ``--no-push`` turns it off. A
``KIND_HANDOVER`` row brings its own pending push and gets no random one (the
spawn's ``consume`` flag; nothing to do here), resumed at the episode's first
step even under ``--no-push`` (which only removes the random one). The
policy acts deterministically, on the actor group's noisy observations unless
``--no-obs-noise``. An episode ends on a FALL (a termination) or survives:
back at rest (the ``recovered`` truncation, ``rest_terminate``) or the
``--episode-s`` cap.

ATTRIBUTION. mjlab resets an env INSIDE the step its episode ends in, so the
spawn's ``_zoo_spawn_row`` already names the NEXT episode when the step
returns. The labeler keeps its own copy per env, taken when each episode
starts, and books an ending episode's outcome on the OLD copy. The outcomes
of a row are exactly its K FIRST-PASS episodes: the spawn hands out entries
in env-id order, so the cursor before a step says which pass each new episode
belongs to, and episodes of a later pass (the cursor cycles) are ignored.
Keeping instead "the first K to end" would bias a small bank toward falls: once
the pass wraps while a row's earlier episodes still run, its falls end first
(a survivor needs the rest dwell or the cap). The run stops once every target
row has K. The env is NOT
reset again after it is built: its construction reset already issued the
first wave of sequential entries (and handed out their pending pushes), and a
second reset would discard that wave until the cursor came round again.

THE VALUE. ``value0`` of an episode is V(its first observation), on the input
the critic was trained on (``labels.PolicyValue``: the privileged critic group
for an asymmetric critic). A row's ``value0`` is the mean over its K episodes
(the start differs only by yaw, planar offset and the per-episode
randomization), and ``p_value = sigmoid(a value0 + b)`` with (a, b) fit by
``labels.fit_logistic`` on the labeled rows (targets ``p_surv``, weighted by
``n_label``), so value-weighted sampling works on a calibrated probability.

OUTPUT. ``--out``: the input bank, every field ``load_bank`` returns (all
``OPTIONAL_FIELDS``, plus ``part`` for a multi-file input), with ``p_surv``,
``n_label``, ``value0``, ``ttf_mean`` (mean time to fall of the failed
episodes, s) and ``p_value`` written on the labeled rows (others keep their
input values), and ``meta["label"]`` (settings + report). The report is also
printed and written to ``<out stem>.report.json``: overall survival, the
episode AUC of V (P(V of a surviving episode > V of a falling one), ties
half), the row Spearman of value0 vs p_surv, survival and AUC per category,
the shares of rows with p_surv in [0, 0.1], (0.1, 0.9) and [0.9, 1], survival
by push force, and the E2 decision line. Also the value read as a certificate
(``cert_frac``, ``cert_precision`` = P(survive | V(s0) >= 0), ``cert_recall`` =
P(V(s0) >= 0 | survive)) and the policy's action statistics over every
rollout step (``action_rate`` ||a_t - a_{t-1}||^2, ``action_mag`` ||a_t||^2,
``action_sat_frac``), for comparing differently regularized learners.
``--episodes-out`` (default ``<out stem>.episodes.pt``, always written): the
per-episode records as 1-D CPU tensors (``row``, ``value0``, ``fell``,
``ttf``, ``force``, ``pending_push`` (the start resumed a stored push),
``score0`` [, ``score0_ref``, ``score0_exec``]), ``score_kind``, the score
settings and the label meta.

SCORES (``score0``: soundness / calibrated-volume read-outs). Next to
``value0`` an episode can carry a safety score sigma(x0) of its start, taken
at the same moment (the construction wave's first obs; for a respawned env
right after the step that reset it: mjlab resets inside the step and builds
every obs group afterwards, so the critic group read there is the new
episode's state). NaN without a score.

* ``--wm-tube wmmin:<seconds>:<c> --tube-calib min_tube.json`` (Beat-It: a
  SHAC policy with a fitted world model): ``eval_sonic_qcbf.py``'s ``wmmin``
  bound, ``sigma = min_{j < H} ghat_j - eps_H(c)``, H = round(seconds /
  step_dt), eps_H(c) from the calibration's ``min_eps`` table
  (``calibrate_wm_min_tube.py``). ghat is ONE noise-free world-model rollout
  (``Evaluator.wm_margins``, ``n_rollouts=None``) from the true critic-group
  state and sim context: pi's own deterministic action on the episode's first
  real observation (the action the real rollout takes; the calibration used
  u_SONIC there), then pi on the decoded state (its last-action term clamped
  to the action box); a step's ghat is pushed to <= -0.1 once the predicted
  P(terminated) passes 0.5. It refuses what the calibration did not cover:
  another checkpoint (the policy in the calibration run's meta, else its tag
  vs ``--tube-tag`` / the run directory above the zip), another observation
  noise (that meta's ``obs_noise`` vs ``--no-obs-noise``: eps_H(c) was fit on
  noise-free rollouts when it says false), another dt, and a random push
  (``--no-push`` is required: the world model never sees one). It certifies
  the first H steps only, while ``fell`` covers the whole episode: the report
  adds the survival of those H steps (``survival_nonneg_within_H``).
* ``--rl-dh`` (not ``--sonic-alone``; automatic): the best-action margin of
  the mean hyperplane the filter applies at the first step, ``sigma = max_{u
  in U} a^T u - b`` (``safety_sb3.hyperplane.box_max``; < 0: the half-space
  misses U), i.e. the margin of argmax_{u in U} a^T u, not of the control the
  rollout executes. Also ``score0_ref = a^T clip_U(u_ref) - b``, u_ref SONIC's
  proposal for that step (>= 0: SONIC's action passes unchanged), and
  ``score0_exec = a^T u - b`` at the executed control u (the box QP's
  projection of u_ref: = score0 when infeasible, = score0_ref when inactive,
  ~0 up to round-off when active).

Per episode is the faithful record (the tube's statement holds per start,
sim context and first action; the episodes file). A row's ``score0`` (and
``score0_ref`` / ``score0_exec``) is the mean over its K episodes, as
``value0``, and ``score0_min`` (``score0_ref_min``, ``score0_exec_min``) the
conservative min (sigma >= 0 only if every episode's is), written only when a
score is active. ``load_bank`` drops
these columns (not ``OPTIONAL_FIELDS``): read them with ``torch.load`` or from
the episodes file. The report adds ``score`` (settings, share of episodes with
sigma >= 0, survival among them, episode AUC, episodes that resumed a stored
push, which no score sees: ``pending_push`` marks them).

RL-DH (``--rl-dh``). The policy is a learned half-space filter of SONIC
(``train_on_policy.py --rl-dh``, ``envs/g1_recover/rl_dh.py``): SONIC resumes
each episode's row (its clip at its frame) and the env executes the min-norm
projection of SONIC's action onto the policy's MEAN half-space ``{u in U :
a^T u >= b}``, with no exploration (``delta = 0``). The env is built exactly as
above and wrapped in the run's filter env (``make_sonic_dh_env``: box ``U``,
SONIC's action history and release directory, ``u_ref`` appended to the actor
/ critic obs, reward; from the run's ``config.yaml`` beside the zip unless
``--dh-box`` / ``--sonic-history`` / ``--sonic-dir`` say otherwise). Without a
readable RL-DH run config the labeler needs ``--sonic-history`` and
``--dh-box`` (they change the executed control; nothing else would notice a
wrong guess) and warns with the defaults it falls back on; where the eval
departs from the run (task, SONIC directory, an overriding flag, the critic)
it warns and records the departure (``overrides_vs_run``). One ``--bank``
only: its meta names the clip library SONIC resumes. The wrapper steps and
observes; the bare env keeps the spawn bookkeeping, push knobs and episode
cap, so rows,
attribution and outcomes mean what they mean for any other policy. It is not
reset either: the filter env's ``prime`` starts SONIC on every env from the
construction wave's rows. The checkpoint is a ``CumulativePPO1P`` learner
(``load_rl_dh``; ``load_twin``'s safety cells refuse its rollout buffer), so
``value0`` is the discounted return of the [DH] Eq. 7 survival reward on the
critic obs (+ ``u_ref``), not a margin: the certificate read-out is not a
certificate there. The action statistics are taken on the EXECUTED control
(the filter's ``applied_control``, env units; ``action_sat_frac`` counts dims
at or beyond the bridge's action box, the box the other policies are clamped
to), and the report adds the filter's read-outs under ``rl_dh``, over every
rollout step: intervention rate, mean correction ``||u - clip_U(u_ref)||``
over the intervened steps, infeasible / invalid rates, SONIC's non-finite
actions, SONIC's restarts against the episodes started and SONIC's resumed
clip / frame against the bank's for the labeler's row, plus the settings. The
last two are BOOKKEEPING checks (SONIC restarted on exactly the episodes the
labeler books, from the rows it books them on; both must agree); they do not
validate the bank's clip / frame values, which ``SonicReferenceFn`` checks
itself (it refuses a row without a clip of its library and a frame >= 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

from robot_safety_sandbox import make_tensor  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.labels import (  # noqa: E402
  PolicyValue, auc, fit_logistic, row_names, spearman)
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  KIND_HANDOVER, load_bank, save_bank)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402

#: Names for rows without a category of their own (walker / synthetic rows),
#: by ``kind``.
KIND_NAMES = ("clean", "pushed", "synthetic", "mocap", "handover")
#: The per-episode safety scores (module docstring SCORES) a run may carry.
SCORE_KEYS = ("score0", "score0_ref", "score0_exec")
#: The E2 threshold: an episode AUC of V at or above it makes the
#: value-weighted sampling arm (S5) viable without rollout labels.
AUC_VIABLE = 0.9


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True,
                 help="the labeling policy's safety_sb3 checkpoint zip")
  p.add_argument("--bank", nargs="+",
                 default=[os.path.join(_ZOO, "data", "g1_recover_states_lafan1_train.pt")],
                 help="the bank(s) to label (several are concatenated in order)")
  p.add_argument("--out", default=None,
                 help="default: <first bank stem>_labeled.pt")
  p.add_argument("--k", type=int, default=8, help="rollouts per row")
  p.add_argument("--task", default="g1_recover_lafan")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--force-max", type=float, default=200.0,
                 help="push magnitude at scale 1, N")
  p.add_argument("--force-min-frac", type=float, default=0.2,
                 help="per-episode scale drawn in [this, 1]")
  p.add_argument("--no-push", action="store_true",
                 help="no random push (a row's stored pending push still resumes)")
  p.add_argument("--no-obs-noise", action="store_true",
                 help="the actor obs without their corruption noise (eval_sonic_qcbf.py "
                      "--no-obs-noise; a wmcalib run with it fits the tube's eps)")
  p.add_argument("--episode-s", type=float, default=3.0, help="episode cap, s")
  p.add_argument("--max-rows", type=int, default=None,
                 help="label only the first M start_ok rows (smoke tests)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  # RL-DH (module docstring): the policy filters SONIC
  p.add_argument("--rl-dh", action="store_true",
                 help="the policy is an RL-DH half-space filter of SONIC "
                      "(train_on_policy.py --rl-dh); evaluated with its mean "
                      "hyperplane and no exploration")
  p.add_argument("--sonic-dir", default=None,
                 help="--rl-dh: GEAR-SONIC release directory (default $SONIC_DIR, "
                      "else ~/gear_sonic_hf, as train_on_policy.py)")
  p.add_argument("--sonic-history", choices=("applied", "nominal"), default=None,
                 help="--rl-dh: SONIC's action history (default: the run's "
                      "config.yaml, else applied)")
  p.add_argument("--dh-box", choices=("sonic", "action"), default=None,
                 help="--rl-dh: the control box U (default: the run's config.yaml, "
                      "else sonic)")
  p.add_argument("--sonic-alone", action="store_true",
                 help="--rl-dh: label SONIC ALONE in the same filter env: the "
                      "policy is ignored and the half-space is the whole space "
                      "(a = 0, b = the box's lowest b), so the env executes "
                      "clip_U(u_ref) = SONIC's own action (U is SONIC's reachable "
                      "box) and SONIC's history records it")
  # SCORES (module docstring)
  p.add_argument("--wm-tube", default=None, metavar="wmmin:<seconds>:<c>",
                 help="score each episode's start with the SHAC policy's world-model "
                      "worst-margin bound (eval_sonic_qcbf.py's wmmin): score0 = min "
                      "over the first round(seconds / step_dt) steps of one noise-free "
                      "predicted g, minus min_eps[c][H] of --tube-calib; needs --no-push")
  p.add_argument("--tube-calib", default=None,
                 help="--wm-tube: the min_tube.json of calibrate_wm_min_tube.py")
  p.add_argument("--dump-wm", action="store_true",
                 help="--wm-tube: also record each episode's world-model inputs at its start "
                      "(wm_S the true critic state, wm_C the sim context, wm_a pi's first action) "
                      "in the episodes file, with the scorer's action box and last-action slice "
                      "(settings['dump']): the truncated-monitor study recomputes the score offline")
  p.add_argument("--tube-tag", default=None,
                 help="--wm-tube: the policy's tag in --tube-calib (default: the run "
                      "directory above the zip, as eval_sonic_qcbf.py --tag)")
  p.add_argument("--episodes-out", default=None,
                 help="per-episode records (row, value0, fell, ttf, force, pending_push, "
                      "score0, ...); default <out stem>.episodes.pt")
  p.add_argument("--thermal-start-c", type=float, nargs="+", default=None, metavar="C",
                 help="thermal tasks: a SOAKED robot, every motor's winding, housing and "
                      "structure at T0 C (one value) or at a per-episode T0 ~ U(LO, HI) (two "
                      "values); replaces the task's start draw. The episodes file then also "
                      "carries overheat (the motor_overheat termination ended it) and "
                      "tw_peak (hottest winding seen, C)")
  args = p.parse_args(argv)
  if not args.rl_dh and (args.sonic_dir or args.sonic_history or args.dh_box
                         or args.sonic_alone):
    p.error("--sonic-dir / --sonic-history / --dh-box / --sonic-alone need --rl-dh")
  if (args.wm_tube is None) != (args.tube_calib is None):
    p.error("--wm-tube and --tube-calib go together")
  if args.tube_tag is not None and args.wm_tube is None:
    p.error("--tube-tag needs --wm-tube")
  if args.wm_tube is not None:
    if args.rl_dh:
      p.error("--wm-tube scores a SHAC policy's world model; --rl-dh has its own "
              "score (the hyperplane's best-action margin)")
    if not args.no_push:
      p.error("--wm-tube needs --no-push: the world model never sees a push, and "
              "the random one lands 0.1-0.4 s into the scored episode")
    try:
      _parse_wm_tube(args.wm_tube)
    except ValueError as e:
      p.error(str(e))
  if args.rl_dh and len(args.bank) != 1:
    p.error("--rl-dh labels one SONIC bank (its meta names the clip library "
            "SONIC resumes; a multi-file bank has none)")
  if args.k < 1:
    p.error("--k must be >= 1")
  if not 0.0 <= args.force_min_frac <= 1.0:
    p.error("--force-min-frac must be in [0, 1]")
  if args.out is None:
    args.out = os.path.splitext(args.bank[0])[0] + "_labeled.pt"
  if args.episodes_out is None:
    args.episodes_out = os.path.splitext(args.out)[0] + ".episodes.pt"
  taken = {os.path.realpath(x) for x in [args.out, report_path(args.out), *args.bank]}
  if os.path.realpath(args.episodes_out) in taken:
    p.error(f"--episodes-out {args.episodes_out} is --out, its report or a --bank")
  # the calibration is checked here, before a policy is loaded or an env built
  args.tube_cal = _tube_calib(args) if args.wm_tube is not None else None
  return args


def report_path(out: str) -> str:
  return os.path.splitext(out)[0] + ".report.json"


def _parse_wm_tube(text: str) -> tuple[float, float]:
  """``(seconds, coverage)`` of ``wmmin:<seconds>:<c>`` (qcbf_eval_lib's
  ``wmmin`` condition); ValueError on anything else."""
  parts = str(text).strip().split(":")
  try:
    if len(parts) != 3 or parts[0] != "wmmin":
      raise ValueError
    h, c = float(parts[1]), float(parts[2])
  except ValueError:
    raise ValueError(f"--wm-tube {text!r}: wmmin:<seconds>:<coverage>") from None
  if not (math.isfinite(h) and h > 0.0 and 0.0 < c < 1.0):
    raise ValueError(f"--wm-tube {text!r}: seconds > 0, 0 < coverage < 1")
  return h, c


def _tube_calib(args) -> dict:
  """``--tube-calib`` read and checked against ``--wm-tube`` and the labeling
  run: eps_H(c) at the calibration's own dt, and the calibration run's meta
  (``eval_sonic_qcbf.py``'s json beside the records ``min_tube.json`` names):
  its policy must be ``--policy`` and its ``obs_noise`` this run's. Its tag
  must be ``--tube-tag`` (default the run directory above the zip), as
  ``eval_sonic_qcbf.py`` refuses. SystemExit on any mismatch; the env's dt is
  checked in :class:`WMTube`."""
  h_s, cov = _parse_wm_tube(args.wm_tube)
  path = args.tube_calib
  try:
    with open(path, "rb") as f:
      raw = f.read()
    calib = json.loads(raw)
    dt = float(calib["dt"])
  except (OSError, ValueError, KeyError, TypeError) as e:
    raise SystemExit(f"[label] --tube-calib {path}: {type(e).__name__}: {e}") from None
  H = int(round(h_s / dt))
  tab = (calib.get("min_eps") or {}).get(f"{cov:g}") or {}
  if H < 1 or str(H) not in tab:
    raise SystemExit(f"[label] --tube-calib {path} has no min_eps at coverage {cov:g} for "
                     f"{H} steps ({h_s:g} s at {dt:g} s; coverages "
                     f"{list(calib.get('min_eps') or {})}, steps {list(tab)})")
  tag = args.tube_tag or os.path.basename(
    os.path.dirname(os.path.dirname(os.path.abspath(args.policy))))
  if calib.get("tag") != tag:
    raise SystemExit(f"[label] --tube-calib {path} was calibrated for {calib.get('tag')!r}, "
                     f"not {tag!r} (--tube-tag names the policy's tag)")
  meta, side = None, None
  if calib.get("calib"):
    side = os.path.splitext(calib["calib"])[0] + ".json"
    try:
      with open(side) as f:
        meta = json.load(f).get("meta")
    except (OSError, ValueError, AttributeError):
      meta = None
  if not isinstance(meta, dict):
    meta = {}
    print(f"[label] WARNING: no calibration run meta at {side}: the calibrated policy "
          f"and observation noise are unchecked")
  pol = meta.get("policy")
  if pol and os.path.realpath(pol) != os.path.realpath(args.policy):
    raise SystemExit(f"[label] --tube-calib {path} was fit on {pol} ({side}), not "
                     f"{args.policy}: another world model")
  noise = meta.get("obs_noise")
  if noise is not None and bool(noise) != (not args.no_obs_noise):
    raise SystemExit(
      f"[label] --tube-calib {path} was fit on rollouts with obs_noise {bool(noise)} "
      f"({side}); this run's actor obs are {'noise-free' if args.no_obs_noise else 'noisy'}: "
      + ("pass --no-obs-noise" if not noise else "drop --no-obs-noise")
      + " (and label every compared method so), or recalibrate")
  return {"raw": raw, "calib": calib, "dt": dt, "H": H, "eps": float(tab[str(H)]),
          "horizon_s": h_s, "coverage": cov, "tag": tag, "meta_path": side,
          "policy": pol, "obs_noise": None if noise is None else bool(noise)}


def _last_action_slice(mj, key: str, action_dim: int) -> tuple[slice, int]:
  """The columns of observation group ``key`` that hold mjlab's ``last_action``
  term, and the group's width (``eval_sonic_qcbf.Evaluator._last_action_slice``).
  Fails loudly when they cannot be found."""
  om = mj.observation_manager
  names = getattr(om, "_group_obs_term_names", {}).get(key)
  dims = getattr(om, "_group_obs_term_dim", {}).get(key)
  cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(key)
  if not (names and dims and cfgs) or not len(names) == len(dims) == len(cfgs):
    raise SystemExit(f"[label] --wm-tube: cannot read the terms of observation group {key!r}")
  hits = [i for i, c in enumerate(cfgs)
          if getattr(getattr(c, "func", None), "__name__", "") == "last_action"]
  hits = hits or [i for i, nm in enumerate(names) if nm == "actions"]
  sizes = [int(math.prod(d)) for d in dims]
  if len(hits) != 1:
    raise SystemExit(f"[label] --wm-tube: observation group {key!r} ({names}) has "
                     f"{len(hits)} last-action terms, not one")
  i, c = hits[0], cfgs[hits[0]]
  if (sizes[i] != action_dim or getattr(c, "scale", None) is not None
          or getattr(c, "clip", None) is not None):
    raise SystemExit(f"[label] --wm-tube: the last-action term {names[i]!r} is {sizes[i]}-D "
                     f"with scale {getattr(c, 'scale', None)} / clip {getattr(c, 'clip', None)}: "
                     f"the box clamp expects the raw {action_dim}-D action")
  start = sum(sizes[:i])
  return slice(start, start + action_dim), sum(sizes)


class WMTube:
  """Beat-It's score of a start state (module docstring SCORES): the
  ``wmmin:<seconds>:<c>`` bound of ``eval_sonic_qcbf.py`` with pi's own first
  action. ``tube(obs, ids)``: sigma of the envs ``ids`` from their CURRENT
  state, ``obs`` the raw actor obs of every env. Refuses a policy without a
  fitted world model on this env's critic group and sim context
  (``Evaluator.setup_world_model``) and a calibration without eps_H(c)."""

  def __init__(self, env, model, norm, args):
    wm = getattr(model, "world_model", None)
    if wm is None or not bool(wm.fitted):
      raise SystemExit(f"[label] --wm-tube: {args.policy} has no fitted world model "
                       f"(a SHAC checkpoint is needed)")
    if getattr(model.policy, "reset_memory", None) is not None:
      raise SystemExit("[label] --wm-tube: a recurrent policy (the imagined steps "
                       "would advance its memory)")
    mj, dev = env.mj, env.device
    om = mj.observation_manager
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get("critic")
    if cfgs is None:
      raise SystemExit(f"[label] --wm-tube: task {args.task!r} has no critic group")
    # compute_group would advance a delay buffer (the critic obs would move) or
    # draw noise (the true state is the model's input; torch's RNG would move)
    if any(getattr(c, "delay_max_lag", 0) > 0 or getattr(c, "noise", None) is not None
           for c in cfgs):
      raise SystemExit("[label] --wm-tube: the critic group has delayed or noisy terms")
    s = om.compute_group("critic", update_history=False)
    c = env.sim_context()
    if int(s.shape[1]) != int(wm.state_dim) or (0 if c is None else int(c.shape[1])) != int(
        wm.context_dim):
      raise SystemExit(f"[label] --wm-tube: the env's critic group ({s.shape[1]}-D) / sim "
                       f"context ({None if c is None else c.shape[1]}-D) are not the world "
                       f"model's {wm.state_dim}-D / {wm.context_dim}-D")
    self.low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    self.high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    self.sl, width = _last_action_slice(mj, env.obs_key, int(self.low.numel()))
    self.actor_idx = model._actor_idx_t.to(dev)
    if int(self.actor_idx.numel()) != width:
      raise SystemExit(f"[label] --wm-tube: the model rebuilds a {self.actor_idx.numel()}-D "
                       f"actor obs, the env's {env.obs_key!r} group is {width}-D")
    self.z_clip = getattr(model, "wm_z_clip", None)
    dt = float(mj.step_dt)
    cal = args.tube_cal                     # _tube_calib, checked in _parse
    if abs(cal["dt"] - dt) > 1e-6:
      raise SystemExit(f"[label] --tube-calib {args.tube_calib} was calibrated at dt "
                       f"{cal['dt']}, the env steps {dt}")
    h_s, cov = cal["horizon_s"], cal["coverage"]
    self.H, self.eps = int(cal["H"]), float(cal["eps"])
    self.env, self.mj, self.model, self.norm, self.wm = env, mj, model, norm, wm
    self.settings = {
      "kind": "wm_tube", "cond": f"wmmin:{h_s:g}:{cov:g}", "horizon_s": h_s,
      "coverage": cov, "H_steps": self.H, "step_dt": dt, "eps": self.eps,
      "calib": os.path.abspath(args.tube_calib),
      "calib_sha256": hashlib.sha256(cal["raw"]).hexdigest(),
      "calib_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(os.path.getmtime(args.tube_calib))),
      "calib_tag": cal["calib"].get("tag"), "policy_tag": cal["tag"],
      "calib_meta": cal["meta_path"], "calib_policy": cal["policy"],
      "calib_obs_noise": cal["obs_noise"], "obs_noise": not args.no_obs_noise,
      "no_push": bool(args.no_push),
      "sigma": "min_{j < H} ghat_j - min_eps[c][H]; ghat_j <= -0.1 once predicted "
               "P(terminated) > 0.5",
      "first_action": "pi on the real first observation",
      "pi_input": "decoded predicted critic state, noise-free, last action clamped "
                  "to the action box",
      "state": f"critic group ({wm.state_dim}-D, true)", "context_dim": int(wm.context_dim),
      "z_clip": self.z_clip,
      # what the score certifies vs what fell / survival count
      "certified_s": self.H * dt, "outcome_horizon_s": float(args.episode_s),
      "outcome": "fell: a termination before the --episode-s cap or the rest truncation; "
                 "survival_nonneg_within_H: no fall in the first H steps"}
    print(f"[label] score: Beat-It world-model tube {self.settings['cond']}: H {self.H} "
          f"steps, eps {self.eps:+.4f} ({args.tube_calib}, tag {cal['tag']}, calibration "
          f"obs_noise {cal['obs_noise']}); first action pi's own; pi's last-action obs "
          f"columns {self.sl.start}:{self.sl.stop} of {width} clamped to the box; "
          f"certifies {self.H * dt:g} s of a <= {args.episode_s:g} s episode")

  @torch.no_grad()
  def margins(self, ids: torch.Tensor, a_first: torch.Tensor) -> torch.Tensor:
    """``[n, H]``: ``eval_sonic_qcbf.Evaluator.wm_margins(ids, a_first, H)``,
    one noise-free rollout."""
    wm, policy, H = self.wm, self.model.policy, self.H
    s = torch.nan_to_num(self.mj.observation_manager.compute_group(
      "critic", update_history=False).float())[ids]
    ctx = self.env.sim_context()            # None: nothing is randomized per env
    c = None if ctx is None else ctx[ids]
    self.last = {"S": s, "C": c, "a": a_first}
    cn = wm.encode_context(c)
    z = wm.encode(s)
    n = int(s.shape[0])
    alive = torch.ones(n, device=s.device)
    out = torch.empty(n, H, device=s.device)
    act = a_first
    for t in range(H):
      st = wm.step(z, act, cn, z_clip=self.z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      out[:, t] = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      if t + 1 < H:
        o = wm.decode(z).index_select(-1, self.actor_idx)
        o[:, self.sl] = torch.maximum(torch.minimum(o[:, self.sl], self.high), self.low)
        act = torch.clamp(policy._predict(self.norm(o), deterministic=True), self.low, self.high)
    return out

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # the action the real rollout takes first (rollouts' loop)
    a_first = torch.clamp(self.model.policy._predict(self.norm(obs[ids]), deterministic=True),
                          self.low, self.high)
    return self.margins(ids, a_first).amin(dim=1) - self.eps


def _rank_in_group(r: torch.Tensor) -> torch.Tensor:
  """For each entry of ``r``, how many EARLIER entries hold the same value."""
  n = r.numel()
  if n == 0:
    return r
  order = torch.argsort(r, stable=True)
  rs = r[order]
  ar = torch.arange(n, device=r.device)
  new = torch.ones(n, dtype=torch.bool, device=r.device)
  new[1:] = rs[1:] != rs[:-1]
  first = torch.cummax(torch.where(new, ar, torch.zeros_like(ar)), 0).values
  out = torch.empty_like(ar)
  out[order] = ar - first
  return out


def _construction_obs(env) -> torch.Tensor:
  """The raw actor obs of the episodes the env's construction reset started
  (sanitized as ``step_tensor`` does)."""
  obs = env.obs_groups()[env.obs_key].float()
  return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def _run_config(policy: str) -> tuple[dict, str | None]:
  """The training run's resolved ``config.yaml`` (train_on_policy.py's dump)
  beside the policy zip, or one level up for a ``checkpoints/`` zip, and its
  path; ``({}, None)`` if there is none, or (with a warning naming the file)
  if it does not parse to a mapping."""
  d = os.path.dirname(os.path.abspath(policy))
  for c in (os.path.join(d, "config.yaml"),
            os.path.join(os.path.dirname(d), "config.yaml")):
    if os.path.isfile(c):
      try:
        import yaml
        with open(c) as f:
          cfg = yaml.safe_load(f)
      except Exception as e:  # noqa: BLE001 -- the run config is optional here
        print(f"[label] WARNING: cannot read the run config {c} "
              f"({type(e).__name__}: {e}); ignored")
        return {}, None
      if not isinstance(cfg, dict):
        print(f"[label] WARNING: the run config {c} is not a mapping "
              f"({type(cfg).__name__}); ignored")
        return {}, None
      return cfg, c
  return {}, None


def load_rl_dh(zip_path: str, device):
  """An RL-DH checkpoint (``train_on_policy.py --rl-dh``: a ``CumulativePPO1P``
  learner with a ``HyperplanePolicy``) and its actor-obs normalizer, as
  ``load_twin`` returns a twin. ``load_twin`` cannot load it: its candidates
  are the safety learners, whose mode check refuses the cumulative rollout
  buffer the checkpoint was saved with."""
  import safety_sb3
  from robot_safety_sandbox.eval.policies import find_obs_stats
  if not hasattr(safety_sb3, "HyperplanePolicy"):
    raise SystemExit(f"--rl-dh: this safety_sb3 ({os.path.dirname(safety_sb3.__file__)}) "
                     f"has no RL-DH (safety_sb3.hyperplane); put the "
                     f"feat/lambda-reach-rldh worktree first on PYTHONPATH")
  model = safety_sb3.CumulativePPO1P.load(zip_path, device=device)
  if not hasattr(model.policy, "hyperplane"):
    raise SystemExit(f"{zip_path}: --rl-dh needs a HyperplanePolicy checkpoint, "
                     f"got {type(model.policy).__name__}")
  pt = find_obs_stats(zip_path, "tensornorm", "tensornormalize.pt")
  if not pt:
    raise SystemExit(f"obs-norm stats not found next to {zip_path} (expected "
                     f"tensornormalize.pt, or tensornorm_<step>.pt for a checkpoint)")
  st = torch.load(pt, map_location=device, weights_only=True)
  mean, var = st["obs_mean"].to(device), st["obs_var"].to(device)
  print(f"[twin] {zip_path} ({type(model).__name__}, "
        f"{type(model.policy).__name__}) + {os.path.basename(pt)}")

  def norm(obs):
    return torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)

  return model, norm


def wrap_rl_dh(env, model, value, run_cfg: dict, cfg_path, args):
  """``(filter env, settings)``: ``env`` inside the RL-DH filter env the run
  trained in (``make_sonic_dh_env``, SONIC the reference), with the run's
  settings (``config.yaml``, its ``sonic_dir`` included where it exists here;
  train_on_policy.py's defaults where it is silent, ``--sonic-dir`` /
  ``--sonic-history`` / ``--dh-box`` over both) and no exploration. Without
  an RL-DH run config (none, unreadable, or no ``rl_dh: true``) it requires
  ``--sonic-history`` and ``--dh-box`` and warns with every value it uses.
  Where the eval departs from the run (task, SONIC directory, history, box,
  critic) it warns; ``settings["overrides_vs_run"]`` records each departure
  and ``settings["run_config_defaults"]`` the keys the config lacked. Refuses
  a filter env whose obs / action shapes are not the checkpoint's."""
  if not args.task.startswith("g1_recover"):
    raise SystemExit(f"--rl-dh: the SONIC reference needs a g1_recover task, "
                     f"not {args.task!r}")
  from robot_safety_sandbox.envs.g1_recover.rl_dh import make_sonic_dh_env

  dh_cfg = cfg_path is not None and run_cfg.get("rl_dh") is True
  if not dh_cfg:
    why = ("no readable config.yaml" if cfg_path is None
           else f"{cfg_path} is not an RL-DH run's (no rl_dh: true)")
    if not (args.sonic_history and args.dh_box):
      raise SystemExit(
        f"--rl-dh: no RL-DH run config beside {args.policy} ({why}); pass "
        f"--sonic-history and --dh-box (and --sonic-dir) as the run was trained, "
        f"or the filter env would be a guess")
    run_cfg = {}
  defaulted: list[str] = []

  def cfg(key, default):
    v = run_cfg.get(key)
    if v is None:
      defaulted.append(key)
      return default
    return v

  # SONIC makes u_ref, half the evaluated controller: --sonic-dir, else the
  # run's own release if it is on this node, else train_on_policy.py's default
  # (a departure from the run's is warned below)
  run_sonic = run_cfg.get("sonic_dir")
  if args.sonic_dir:
    sonic_dir = args.sonic_dir
  elif run_sonic and os.path.isdir(run_sonic):
    sonic_dir = run_sonic
  else:
    sonic_dir = os.environ.get("SONIC_DIR", os.path.expanduser("~/gear_sonic_hf"))
    if dh_cfg and not run_sonic:
      defaulted.append("sonic_dir")
  kw = dict(
    sonic_dir=sonic_dir,
    history=args.sonic_history or str(cfg("sonic_history", "applied")),
    box=args.dh_box or str(cfg("dh_box", "sonic")),
    explore_prob=0.0,
    append_reference=bool(cfg("dh_append_reference", True)),
    critic_reference=bool(cfg("dh_critic_reference", True)),
    reward=str(cfg("dh_reward", "paper")),
    bonus=float(cfg("dh_bonus", 0.0)),
    fail_reward=float(cfg("dh_fail_reward", -1.0)))
  if not dh_cfg:
    print(f"[rl-dh] WARNING: no RL-DH run config beside {args.policy} ({why}); "
          f"the filter env uses the flags and train_on_policy.py's defaults: "
          + ", ".join(f"{k}={v}" for k, v in kw.items()))
  elif defaulted:
    print(f"[rl-dh] WARNING: the run config {cfg_path} has no "
          f"{', '.join(defaulted)}; train_on_policy.py's default used")
  over: dict = {}
  if dh_cfg:
    for key, run_v, eval_v in (
        ("task", run_cfg.get("task"), args.task),
        ("sonic_history", run_cfg.get("sonic_history"), kw["history"]),
        ("dh_box", run_cfg.get("dh_box"), kw["box"]),
        ("privileged_critic", run_cfg.get("privileged_critic"), value.asymmetric)):
      if run_v is not None and run_v != eval_v:
        over[key] = {"run": run_v, "eval": eval_v}
    if run_sonic and os.path.realpath(run_sonic) != os.path.realpath(sonic_dir):
      over["sonic_dir"] = {"run": run_sonic, "eval": sonic_dir}
    for key, d in over.items():
      gone = (" (not on this node)" if key == "sonic_dir"
              and not os.path.isdir(run_sonic) else "")
      print(f"[rl-dh] WARNING: {key} {d['eval']!r} is not the run's "
            f"{d['run']!r}{gone} ({cfg_path}): not the env it was trained in")
  denv, ref = make_sonic_dh_env(env, seed=args.seed, **kw)
  pol = model.policy
  if tuple(denv.observation_space.shape) != tuple(pol.observation_space.shape):
    raise SystemExit(
      f"--rl-dh: the filter env's actor obs is {denv.observation_space.shape} but "
      f"the policy reads {pol.observation_space.shape} (append_reference "
      f"{kw['append_reference']}, task {args.task}): not the run's env")
  if tuple(denv.action_space.shape) != tuple(pol.action_space.shape):
    raise SystemExit(
      f"--rl-dh: the filter env's action (a, b) is {denv.action_space.shape} but "
      f"the policy outputs {pol.action_space.shape}")
  if value.asymmetric:
    cs = denv.critic_observation_space
    want = tuple(pol.critic_observation_space.shape)
    if cs is None or tuple(cs.shape) != want:
      raise SystemExit(
        f"--rl-dh: the filter env's critic obs is "
        f"{None if cs is None else cs.shape} but the value net reads {want} "
        f"(critic_reference {kw['critic_reference']})")
  settings = dict(
    kw, run_config=cfg_path if dh_cfg else None, overrides_vs_run=over,
    run_config_defaults=defaulted, library=ref.library, learner=type(model).__name__,
    policy_class=type(pol).__name__,
    action="mean hyperplane (a, b), deterministic",
    actor_obs_dim=int(denv.observation_space.shape[0]),
    critic_obs_dim=(int(denv.critic_observation_space.shape[0])
                    if denv.critic_observation_space is not None else None),
    u_low=[float(x) for x in denv.low.reshape(-1).tolist()],
    u_high=[float(x) for x in denv.high.reshape(-1).tolist()])
  print(f"[rl-dh] SONIC reference ({kw['sonic_dir']}, library {ref.library}, "
        f"history {kw['history']}); U = {kw['box']} box "
        f"[{float(denv.low.min()):.2f}, {float(denv.high.max()):.2f}]; delta 0; "
        f"action (a, b) {denv.action_space.shape[0]}-D; actor obs "
        f"{settings['actor_obs_dim']}-D, critic obs {settings['critic_obs_dim']}-D; "
        f"run config {settings['run_config']}, departures from it "
        f"{sorted(over) or 'none'}")
  return denv, settings


def _add_action_stats(st: torch.Tensor, act: torch.Tensor, prev: torch.Tensor,
                      has_prev: torch.Tensor, low: torch.Tensor,
                      high: torch.Tensor) -> None:
  """Accumulate the action read-outs of one step into ``st`` (``rollouts``)."""
  st[0] += ((act - prev).pow(2).sum(-1) * has_prev).sum()
  st[1] += has_prev.sum()
  st[2] += act.pow(2).sum(-1).sum()
  st[3] += act.shape[0]
  st[4] += ((act >= high - 1e-4) | (act <= low + 1e-4)).float().mean(-1).sum()
  st[5] += act.shape[0]


def _sonic_row_mismatch(ref, rows: torch.Tensor, idx=None) -> torch.Tensor:
  """Of the envs ``idx`` (all by default), which ones' SONIC resumed another
  clip / frame than their spawn row's (``rows``, the labeler's copy). Valid
  right after the filter env started them (``prime`` or the step whose reset
  they had), before SONIC advances; should be none. A bookkeeping check (did
  SONIC restart from the row the labeler books the episode on?): both sides
  read the same bank columns, so it says nothing of their values
  (``SonicReferenceFn`` refuses a row without a usable clip / frame)."""
  tr = ref.tracker
  ids = slice(None) if idx is None else idx
  r = rows[ids].clamp_min(0)
  return (tr.clip[ids] != ref.clip[r]) | (tr.frame[ids] != ref.frame[r])


def rollouts(env, model, norm, value, target: torch.Tensor, kinds: torch.Tensor,
             args, dh=None, tube=None) -> dict[str, torch.Tensor]:
  """Run until every ``target`` row has ``args.k`` ended episodes; return the
  per-episode records (row, value0, fell, ttf, force, pending_push, score0 [,
  score0_ref, score0_exec]), ``score_kind`` and ``score_settings``.

  ``dh``: the RL-DH filter env around ``env`` (``--rl-dh``, :func:`wrap_rl_dh`).
  It then steps and observes (obs and critic obs + ``u_ref``) while ``env``
  keeps the rest (mj, the spawn's rows / cursor, the push knobs, the episode
  cap); the action statistics are taken on the control it executed, and its
  read-outs come back under ``"dh"``. Without ``--sonic-alone`` each episode
  is scored with its first hyperplane's best-action margin. ``tube``: a
  :class:`WMTube` (``--wm-tube``) scoring each episode's start (module
  docstring SCORES)."""
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  if not hasattr(mj, "_zoo_spawn_row"):
    raise SystemExit(f"task {args.task!r}: its spawn publishes no _zoo_spawn_row "
                     f"(a g1_recover task is needed)")
  push = getattr(env, "_push", None)
  if push is None and not args.no_push:
    print(f"[label] WARNING: task {args.task!r} has no push process; labeling "
          f"without pushes")
  target = target.to(dev)
  kinds = kinds.to(dev)
  R = int(target.numel())
  k = int(args.k)
  counts = torch.zeros(R, dtype=torch.long, device=dev)
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  fmax = 0.0 if args.no_push else float(args.force_max)
  env.force_max = fmax
  env.force_scale = torch.ones(n, device=dev)

  def draw(mask):
    m = int(mask.sum())
    if m:
      env.force_scale[mask] = (args.force_min_frac + (1.0 - args.force_min_frac)
                               * torch.rand(m, device=dev))

  draw(torch.ones(n, dtype=torch.bool, device=dev))
  obs = _construction_obs(env)
  senv = env if dh is None else dh       # what steps and observes
  if dh is not None:
    # still no reset: SONIC starts on every env from its construction-wave row
    obs = dh.prime(obs)
    ref = dh.reference_fn
    hp_low = torch.as_tensor(dh.action_space.low, device=dev)
    hp_high = torch.as_tensor(dh.action_space.high, device=dev)
    # control dims at U (sum, n), SONIC starts on another clip / frame than the
    # labeler's row, episodes that ended
    dh_acc = torch.zeros(4, dtype=torch.float64, device=dev)
  cur_row = mj._zoo_spawn_row.clone()
  if dh is not None:
    dh_acc[2] += _sonic_row_mismatch(ref, cur_row).float().sum()
  # which sequential entry each running episode took (the last reset issued
  # the n entries before the cursor, in env-id order): only first-pass ones count
  seq_len = int(mj._zoo_spawn_seq_len)
  ar = torch.arange(n, device=dev)
  first = (int(mj._zoo_spawn_cursor) - n + ar) < seq_len
  crit = senv.critic_obs() if value.asymmetric else None
  v0 = value(obs, crit)
  # the score of each running episode's start (module docstring SCORES), taken
  # where value0 is: here, and for the envs a step just reset after that step
  dh_score = dh is not None and not getattr(args, "sonic_alone", False)
  # --dump-wm: the scorer's inputs of every env's current episode start
  wm_dump = None
  if tube is not None and getattr(args, "dump_wm", False):
    cdim = int(tube.wm.context_dim)
    wm_dump = {"S": torch.zeros(n, int(tube.wm.state_dim), device=dev),
               "a": torch.zeros(n, int(tube.low.numel()), device=dev)}
    if cdim:
      wm_dump["C"] = torch.zeros(n, cdim, device=dev)
  score_kind = "wm_tube" if tube is not None else ("rl_dh_best_action" if dh_score else "none")
  extra = ("score0_ref", "score0_exec") if dh_score else ()
  if dh_score:
    from safety_sb3.hyperplane import box_max, project_halfspace

  @torch.no_grad()
  def score(obs_, ids):
    """(sigma, {extra key: score}) of the episodes envs ``ids`` just started,
    ``obs_`` the (filter) env's current obs of every env."""
    if tube is not None:
      sc_ = tube(obs_, ids)
      if wm_dump is not None:
        for key_, buf_ in wm_dump.items():
          buf_[ids] = tube.last[key_]
      return sc_, {}
    # the mean hyperplane the loop applies on this obs, clamped as there
    a_, b_ = model.policy.hyperplane(norm(obs_[ids]), deterministic=True)
    ab = torch.clamp(torch.cat([a_, b_.unsqueeze(-1)], dim=-1), hp_low, hp_high)
    a, b = ab[:, :-1], ab[:, -1]
    lo, hi = dh.low, dh.high
    u_ref = dh.reference[ids]                      # SONIC's proposal
    # the control the env executes at that step (delta 0: _sample's box QP, clipped)
    u_exe = torch.clamp(project_halfspace(u_ref, a, b, lo, hi)[0], lo, hi)
    return box_max(a, lo, hi) - b, {
      "score0_ref": (a * torch.clamp(u_ref, lo, hi)).sum(-1) - b,
      "score0_exec": (a * u_exe).sum(-1) - b}

  # a start that resumes a stored push (base._apply_pending_push; also under
  # --no-push), which no score sees: marked per episode
  pend = getattr(mj, "_zoo_pending_push", None)

  def pending(ids):
    if push is None or pend is None:
      return torch.zeros(int(ids.numel()), dtype=torch.bool, device=dev)
    return pend["steps"][ids] > 0

  pp0 = pending(ar)
  s0 = torch.full((n,), float("nan"), device=dev)
  s0x = {key: torch.full((n,), float("nan"), device=dev) for key in extra}
  if score_kind != "none":
    sc, scx = score(obs, ar)
    s0.copy_(sc)
    for key in extra:
      s0x[key].copy_(scx[key])
  elapsed = torch.zeros(n, dtype=torch.long, device=dev)
  # What the deterministic policy's actions look like (a smoothness read-out to
  # compare policies trained with different regularizers): the action rate
  # ||a_t - a_{t-1}||^2 (the first step of an episode has no own predecessor
  # and is left out), the magnitude ||a_t||^2 (summed over action dims), and
  # the fraction of action dims at the box. Every step of every env counts.
  # (RL-DH: of the executed control; at or beyond the box, which it may leave.)
  prev_act = torch.zeros(n, int(low.numel()), device=dev)
  a_stats = torch.zeros(6, device=dev)  # rate sum, n, mag sum, n, sat sum, n
  ep: dict[str, list] = {"row": [], "value0": [], "fell": [], "ttf": [], "force": [],
                         "pending_push": [], "score0": []}
  ep.update({key: [] for key in extra})
  if wm_dump is not None:
    ep.update({f"wm_{key}": [] for key in wm_dump})
  # thermal tasks: which ended episodes the overheat termination ended, and the
  # hottest winding each episode reached (its last step is lost to the in-step
  # reset; an overheat episode's peak is at least the limit)
  from robot_safety_sandbox.envs.g1_recover import thermal as _thermal
  th = _thermal.thermal_state(mj) if _thermal.has_thermal(mj) else None
  oh_term = None
  if th is not None:
    ep.update(overheat=[], tw_peak=[])
    tm = mj.termination_manager
    oh_term = "motor_overheat" if "motor_overheat" in tm.active_terms else None
    tw_peak = th.Tw.amax(1).clone()
  n_target = int(target.sum())
  remaining = n_target
  stall_limit = 3 * int(mj.max_episode_length) + 10
  since_accept, it, t0, episodes = 0, 0, time.time(), 0
  while remaining > 0:
    with torch.no_grad():
      if dh is None:
        act = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
      elif getattr(args, "sonic_alone", False):
        # SONIC alone: a = 0, b = the lowest b the box allows -> a^T u >= b for
        # every u, no intervention, the env executes clip_U(u_ref)
        act = torch.zeros(n, int(hp_low.numel()), device=dev)
        act[:, -1] = hp_low[-1]
      else:
        # the mean hyperplane, clamped to the (a, b) box as the training
        # rollout clamps its samples (inert: +-1e6)
        a_, b_ = model.policy.hyperplane(norm(obs), deterministic=True)
        act = torch.clamp(torch.cat([a_, b_.unsqueeze(-1)], dim=-1), hp_low, hp_high)
    has_prev = (elapsed > 0).float()
    if dh is None:
      _add_action_stats(a_stats, act, prev_act, has_prev, low, high)
      prev_act = act
    cursor = int(mj._zoo_spawn_cursor)
    obs, _g, dones, timeouts, _l = senv.step_tensor(act)
    if dh is not None:
      ctrl = dh.applied_control                    # what the env executed
      _add_action_stats(a_stats, ctrl, prev_act, has_prev, low, high)
      prev_act = ctrl
      dh_acc[0] += ((ctrl >= dh.high - 1e-4) | (ctrl <= dh.low + 1e-4)).float().mean(-1).sum()
      dh_acc[1] += n
      dh_acc[3] += dones.float().sum()
    reset_policy_memory(model.policy, dones)       # a recurrent policy restarts
    if th is not None:
      tw_now = th.Tw.amax(1)
      tw_peak = torch.where(dones, tw_peak, torch.maximum(tw_peak, tw_now))
    elapsed += 1
    it += 1
    since_accept += 1
    if bool(dones.any()):
      d = dones.nonzero(as_tuple=False).squeeze(-1)
      rows = cur_row[d]
      fell = ~timeouts[d]
      valid = (rows >= 0) & first[d]
      safe = rows.clamp_min(0)
      valid &= target[safe]
      # first k ENDED episodes per row; two envs ending on the same row in
      # this step are ranked so the cap is exact
      rank = _rank_in_group(torch.where(valid, safe, torch.full_like(safe, -1)))
      acc = valid & (counts[safe] + rank < k)
      if bool(acc.any()):
        ra = rows[acc]
        counts.index_add_(0, ra, torch.ones_like(ra))
        fa = fell[acc]
        ep["row"].append(ra)
        ep["value0"].append(v0[d][acc])
        ep["pending_push"].append(pp0[d][acc])
        ep["score0"].append(s0[d][acc])
        if wm_dump is not None:
          for key_, buf_ in wm_dump.items():
            ep[f"wm_{key_}"].append(buf_[d][acc])
        for key in extra:
          ep[key].append(s0x[key][d][acc])
        ep["fell"].append(fa)
        if th is not None:
          oh = (mj.termination_manager.get_term(oh_term)[d][acc].bool() if oh_term
                else torch.zeros_like(fa))
          ep["overheat"].append(oh)
          ep["tw_peak"].append(tw_peak[d][acc])
        ep["ttf"].append(torch.where(fa, elapsed[d][acc].float() * dt,
                                     torch.full_like(fa, float("nan"), dtype=torch.float)))
        force = env.force_scale[d][acc] * fmax
        ep["force"].append(torch.where(kinds[ra] == KIND_HANDOVER,
                                       torch.full_like(force, float("nan")), force))
        episodes += int(acc.sum())
        since_accept = 0
        remaining = int((target & (counts < k)).sum())
      # the episodes that just started (mjlab already reset these envs); they
      # took entries cursor, cursor + 1, ... in env-id order (d is ascending)
      cur_row[d] = mj._zoo_spawn_row[d]
      if dh is not None:
        dh_acc[2] += _sonic_row_mismatch(ref, cur_row, d).float().sum()
      first[d] = (cursor + ar[:d.numel()]) < seq_len
      crit = senv.critic_obs() if value.asymmetric else None
      v0[d] = value(obs[d], crit[d] if crit is not None else None)
      pp0[d] = pending(d)
      if score_kind != "none":
        # obs[d], the critic group and (RL-DH) u_ref already hold the new episodes
        sc, scx = score(obs, d)
        s0[d] = sc
        for key in extra:
          s0x[key][d] = scx[key]
      elapsed[d] = 0
      if th is not None:
        tw_peak[d] = tw_now[d]                       # the new episodes' start
      draw(dones)
    if since_accept > stall_limit:
      raise SystemExit(
        f"[label] no labeled episode in {since_accept} steps with {remaining} "
        f"rows still short of {k}: is the spawn sampler sequential and the bank "
        f"the one the env spawns from?")
    if it % 250 == 0:
      el = max(time.time() - t0, 1e-6)
      print(f"[label] step {it}  rows done {n_target - remaining}/{n_target}  "
            f"episodes {episodes}/{n_target * k}  ({episodes / el:.0f} ep/s, "
            f"{it * n / el:.0f} env-steps/s)", flush=True)
  el = max(time.time() - t0, 1e-6)
  print(f"[label] done: {episodes} episodes in {it} steps, {el:.0f}s "
        f"({episodes / el:.0f} ep/s, {it * n / el:.0f} env-steps/s)")
  out = {key: torch.cat(v) for key, v in ep.items()}
  out["action_stats"] = a_stats.cpu()
  out["score_kind"] = score_kind
  if tube is not None:
    out["score_settings"] = dict(tube.settings)
    if wm_dump is not None:   # what the offline recomputation needs (WMTube.margins)
      out["score_settings"]["dump"] = {
        "low": tube.low.cpu().tolist(), "high": tube.high.cpu().tolist(),
        "last_action_slice": [int(tube.sl.start), int(tube.sl.stop)],
        "actor_idx": tube.actor_idx.cpu().tolist(), "z_clip": tube.z_clip, "H": tube.H,
        "eps": tube.eps, "policy": os.path.abspath(args.policy)}
  elif dh_score:
    out["score_settings"] = {
      "kind": score_kind,
      "sigma": "max_{u in U} a^T u - b (safety_sb3.hyperplane.box_max): the margin of "
               "argmax_{u in U} a^T u, not of the executed control",
      "sigma_ref": "a^T clip_U(u_ref) - b, u_ref = SONIC's proposal for the first step",
      "sigma_exec": "a^T u - b, u = the executed control (project_halfspace of u_ref onto "
                    "the half-space in U, clipped); ~0 up to round-off when the filter acts",
      "hyperplane": "the mean (a, b) on the episode's first obs, clamped to the (a, b) box",
      "U": "the filter env's box (rl_dh settings: box, u_low, u_high)"}
  else:
    out["score_settings"] = {}
  if dh is not None:
    out["dh"] = _dh_readout(dh, dh_acc, n)
  return out


def _dh_readout(dh, acc: torch.Tensor, n: int) -> dict:
  """The RL-DH filter's read-outs over every rollout step (``rollouts``): its
  own ``dh/*`` metrics (fractions of env-steps; ``correction_norm`` is the mean
  ``||u - clip_U(u_ref)||`` over the steps it intervened), the share of control
  dims at ``U``, and the SONIC bookkeeping checks."""
  out = {k[3:]: float(v) for k, v in dh.metrics().items() if k.startswith("dh/")}
  ref = dh.reference_fn
  out["control_at_u_frac"] = float(acc[0] / acc[1].clamp_min(1.0))
  out["ref_nonfinite"] = int(ref.n_nonfinite)
  out["sonic_restarts"] = int(ref.n_restarts)
  out["episodes_started"] = n + int(acc[3])
  out["sonic_row_mismatch"] = int(acc[2])
  return out


def write_labels(bank: dict, ep: dict, target: torch.Tensor) -> tuple[float, float]:
  """Fold the episodes into per-row columns of ``bank`` (in place, CPU) and
  fill ``p_value``; returns the logistic fit (a, b)."""
  R = int(bank["kind"].numel())
  row = ep["row"].cpu()
  fell = ep["fell"].cpu().float()
  one = torch.ones_like(fell)
  cnt = torch.zeros(R).index_add_(0, row, one)
  surv = torch.zeros(R).index_add_(0, row, 1.0 - fell)
  vsum = torch.zeros(R).index_add_(0, row, ep["value0"].cpu().float())
  nfell = torch.zeros(R).index_add_(0, row, fell)
  ttf = torch.nan_to_num(ep["ttf"].cpu().float(), nan=0.0)
  tsum = torch.zeros(R).index_add_(0, row, ttf * fell)
  lab = target.cpu() & (cnt > 0)
  nan = torch.full((R,), float("nan"))
  bank["n_label"] = torch.where(lab, cnt.long(), bank["n_label"].cpu())
  bank["p_surv"] = torch.where(lab, surv / cnt.clamp_min(1.0), bank["p_surv"].cpu())
  bank["value0"] = torch.where(lab, vsum / cnt.clamp_min(1.0), bank["value0"].cpu())
  ttf_mean = torch.where(nfell > 0, tsum / nfell.clamp_min(1.0), nan)
  bank["ttf_mean"] = torch.where(lab, ttf_mean, bank["ttf_mean"].cpu())
  if ep.get("score_kind", "none") != "none":
    # the row's mean score over its K episodes, as value0, and its min (>= 0
    # only if every episode's is; NaN propagates); NaN: unlabeled
    for key in SCORE_KEYS:
      if key in ep:
        s = ep[key].cpu().float()
        ssum = torch.zeros(R).index_add_(0, row, s)
        bank[key] = torch.where(lab, ssum / cnt.clamp_min(1.0), nan)
        smin = torch.full((R,), float("inf")).scatter_reduce_(0, row, s, reduce="amin")
        bank[key + "_min"] = torch.where(lab, smin, nan)
  a, b = fit_logistic(bank["value0"][lab], bank["p_surv"][lab], weight=cnt[lab])
  if a == a:
    p_value = torch.sigmoid(a * bank["value0"] + b)
    bank["p_value"] = torch.where(lab, p_value, bank["p_value"].cpu())
  return a, b


def _score_readout(s: torch.Tensor, fell: torch.Tensor, ttf: torch.Tensor,
                   pend: torch.Tensor, within_s: float | None = None) -> dict:
  """A score read as a certificate ("safe iff sigma >= 0"), per episode: the
  share it certifies, the survival among those (soundness), the share of
  survivors it certifies, its episode AUC, and how many of its episodes
  resumed a stored push. ``within_s``: also the survival of the first
  ``within_s`` seconds (a fall with ``ttf <= within_s`` counts), for a score
  that certifies only that long (the tube's H steps)."""
  cert, surv = s >= 0.0, ~fell
  n, n_cert, n_surv = int(s.numel()), int(cert.sum()), int(surv.sum())
  out = {"frac_nonneg": float(cert.float().mean()) if n else float("nan"),
         "survival_nonneg": (float((cert & surv).sum()) / n_cert if n_cert
                             else float("nan")),
         "recall_nonneg": (float((cert & surv).sum()) / n_surv if n_surv
                           else float("nan")),
         "auc_episodes": auc(s, surv), "n_nonfinite": int((~torch.isfinite(s)).sum()),
         "mean": float(s[torch.isfinite(s)].mean()) if bool(torch.isfinite(s).any())
         else float("nan"),
         "n_pending_push": int(pend.sum()), "n_pending_push_nonneg": int((pend & cert).sum())}
  if within_s is not None:
    # ttf = steps * dt in float32: 1e-3 s of slack, far below a control step
    early = fell & (torch.nan_to_num(ttf, nan=float("inf")) <= within_s + 1e-3)
    out["within_s"] = float(within_s)
    out["survival_nonneg_within_H"] = (float((cert & ~early).sum()) / n_cert if n_cert
                                       else float("nan"))
    out["auc_episodes_within_H"] = auc(s, ~early)
  return out


def report(bank: dict, ep: dict, target: torch.Tensor, ab: tuple, args) -> dict:
  row = ep["row"].cpu()
  fell = ep["fell"].cpu()
  v0 = ep["value0"].cpu()
  lab = target.cpu() & (bank["n_label"] > 0)
  p = bank["p_surv"][lab]
  out: dict = {"rows_labeled": int(lab.sum()), "episodes": int(row.numel()),
               "k": int(args.k), "logistic_a": ab[0], "logistic_b": ab[1]}
  out["survival"] = 1.0 - float(fell.float().mean()) if row.numel() else float("nan")
  out["auc_value_episodes"] = auc(v0, ~fell)
  out["spearman_value_psurv_rows"] = spearman(bank["value0"][lab], p)
  out["frac_p_le_0.1"] = float((p <= 0.1).float().mean()) if p.numel() else float("nan")
  out["frac_p_mid"] = float(((p > 0.1) & (p < 0.9)).float().mean()) if p.numel() else float("nan")
  out["frac_p_ge_0.9"] = float((p >= 0.9).float().mean()) if p.numel() else float("nan")
  # The value as a CERTIFICATE ("safe iff V >= 0"), per episode, at its start:
  # how much it certifies, and how often a certified start then fell.
  cert, surv = v0 >= 0.0, ~fell
  n_cert, n_surv = int(cert.sum()), int(surv.sum())
  out["cert_frac"] = float(cert.float().mean()) if row.numel() else float("nan")
  out["cert_precision"] = (float((cert & surv).sum()) / n_cert if n_cert
                           else float("nan"))
  out["cert_recall"] = (float((cert & surv).sum()) / n_surv if n_surv
                        else float("nan"))
  st = ep.get("action_stats")
  if st is not None:
    out["action_rate"] = float(st[0] / st[1].clamp_min(1.0))
    out["action_mag"] = float(st[2] / st[3].clamp_min(1.0))
    out["action_sat_frac"] = float(st[4] / st[5].clamp_min(1.0))
  print(f"[label] {out['rows_labeled']} rows x {args.k} = {out['episodes']} episodes; "
        f"survival {out['survival']:.3f}")
  print(f"[label] value vs outcome: episode AUC {out['auc_value_episodes']:.3f} "
        f"(P(V_survived > V_fell)), row Spearman(value0, p_surv) "
        f"{out['spearman_value_psurv_rows']:.3f}; p_value = sigmoid({ab[0]:.3f} V "
        f"+ {ab[1]:.3f})")
  print(f"[label] rows by p_surv: [0, 0.1] {out['frac_p_le_0.1']:.3f}, (0.1, 0.9) "
        f"{out['frac_p_mid']:.3f}, [0.9, 1] {out['frac_p_ge_0.9']:.3f}")
  print(f"[label] certificate V(s0) >= 0: certifies {out['cert_frac']:.3f} of starts, "
        f"precision P(survive | V>=0) {out['cert_precision']:.3f}, recall "
        f"P(V>=0 | survive) {out['cert_recall']:.3f}")
  if "action_rate" in out:
    print(f"[label] actions: rate {out['action_rate']:.4f}, magnitude "
          f"{out['action_mag']:.3f} (sum over dims), at the box "
          f"{out['action_sat_frac']:.4f} of dims")
  sk = ep.get("score_kind", "none")
  if sk != "none":
    sc = dict(ep.get("score_settings") or {}, kind=sk)
    ttf_all = ep["ttf"].cpu().float()
    pend = ep.get("pending_push")
    pend = torch.zeros_like(fell) if pend is None else pend.cpu().bool()
    for key, name in (("score0", None), ("score0_ref", "ref"), ("score0_exec", "exec")):
      if key not in ep:
        continue
      r = _score_readout(ep[key].cpu().float(), fell, ttf_all, pend,
                         within_s=sc.get("certified_s") if name is None else None)
      if name is None:
        sc.update(r)
      else:
        sc[name] = r
      within = (f" (first {r['within_s']:g} s: {r['survival_nonneg_within_H']:.3f})"
                if "within_s" in r else "")
      print(f"[label] score {sk}{'' if name is None else ' (' + key + ')'}"
            f"{' ' + sc['cond'] if 'cond' in sc else ''}: sigma(s0) >= 0 on "
            f"{r['frac_nonneg']:.3f} of starts, survival there {r['survival_nonneg']:.3f}"
            f"{within}, recall P(sigma>=0 | survive) {r['recall_nonneg']:.3f}, episode AUC "
            f"{r['auc_episodes']:.3f}, mean {r['mean']:+.3f}, non-finite {r['n_nonfinite']}")
    if int(pend.sum()):
      print(f"[label] WARNING: {int(pend.sum())} scored episodes resumed a stored push "
            f"at their first step, which no score sees (pending_push in the episodes file)")
    out["score"] = sc
  dh = ep.get("dh")
  if dh is not None:
    out["mode"] = "sonic_alone" if getattr(args, "sonic_alone", False) else "rl_dh"
    out["action_source"] = ("executed control (the RL-DH filter's applied_control, "
                            "env units); action_sat_frac: at or beyond the bridge's box")
    out["value_kind"] = ("CumulativePPO1P return of the [DH] Eq. 7 survival reward, "
                         "not a margin: cert_* is no certificate")
    out["rl_dh"] = dict(dh)
    print(f"[label] RL-DH: actions above are the EXECUTED control; filter "
          f"intervened on {dh.get('intervention_rate', float('nan')):.4f} of env-steps "
          f"(mean ||u - clip_U(u_ref)|| {dh.get('correction_norm', float('nan')):.3f} "
          f"there), half-space missed U on {dh.get('infeasible_rate', float('nan')):.4f}, "
          f"invalid {dh.get('invalid_rate', float('nan')):.4f}, control at U "
          f"{dh['control_at_u_frac']:.4f} of dims; SONIC non-finite actions "
          f"{dh['ref_nonfinite']}, restarts {dh['sonic_restarts']} vs episodes "
          f"started {dh['episodes_started']}, row mismatches {dh['sonic_row_mismatch']}")
    if dh["sonic_restarts"] != dh["episodes_started"] or dh["sonic_row_mismatch"]:
      print("[label] WARNING: SONIC's restarts / resumed rows disagree with the "
            "labeler's episodes: the reference did not track the labeled rows")
    print(f"[label] RL-DH: V is {out['value_kind']}")
  # per category (the row's own file resolves the name)
  names = row_names(bank, fallback=KIND_NAMES)
  uniq = sorted(set(names))
  code = {c: i for i, c in enumerate(uniq)}
  code_row = torch.tensor([code[x] for x in names], dtype=torch.long)
  code_ep = code_row[row]
  per = {}
  print("[label] per category:")
  for c in sorted(set(code_row[lab].tolist())):
    em = code_ep == c
    rm = lab & (code_row == c)
    c = uniq[c]
    s = 1.0 - float(fell[em].float().mean()) if bool(em.any()) else float("nan")
    a_c = auc(v0[em], ~fell[em])
    per[c] = {"rows": int(rm.sum()), "episodes": int(em.sum()), "survival": s,
              "auc_value": a_c, "p_mean": float(bank["p_surv"][rm].mean())}
    print(f"    {c:15s} {int(rm.sum()):7d} rows  survival {s:.3f}  AUC {a_c:.3f}")
  out["per_category"] = per
  # by push force (random pushes only; handover rows carry their own)
  force = ep["force"].cpu()
  fm = torch.isfinite(force)
  if not args.no_push and bool(fm.any()):
    lo, hi = args.force_min_frac * args.force_max, args.force_max
    edges = torch.linspace(lo, hi, 5)
    byf = {}
    print("[label] survival by push force:")
    for a_, b_ in zip(edges[:-1].tolist(), edges[1:].tolist()):
      m = fm & (force >= a_) & (force <= b_)
      if bool(m.any()):
        s = 1.0 - float(fell[m].float().mean())
        byf[f"{a_:.0f}_{b_:.0f}"] = s
        print(f"    {a_:5.0f}-{b_:5.0f} N: {int(m.sum()):7d} episodes, survival {s:.3f}")
    out["survival_by_force"] = byf
  ttf = ep["ttf"].cpu()[fell]
  if ttf.numel():
    q = torch.quantile(ttf, torch.tensor([0.1, 0.5, 0.9])).tolist()
    out["time_to_fall_q10_q50_q90"] = q
    print(f"[label] time to fall: q10 {q[0]:.2f} s, q50 {q[1]:.2f} s, q90 {q[2]:.2f} s")
  a_ep = out["auc_value_episodes"]
  if a_ep != a_ep:
    out["e2_decision"] = ("AUC undefined (every labeled episode survived, or every "
                          "one fell): no E2 decision from this pass")
  elif a_ep >= AUC_VIABLE:
    out["e2_decision"] = (f"AUC {a_ep:.3f} >= {AUC_VIABLE}: the value ranks starts like "
                          f"the rollouts -> the value-weighted arm S5 is viable")
  else:
    out["e2_decision"] = (f"AUC {a_ep:.3f} < {AUC_VIABLE}: the value does not rank "
                          f"starts well enough -> weight with rollout labels (S3/S4)")
  print(f"[label] E2: {out['e2_decision']}")
  return out


def make_env(args, overrides: dict, push_kw: dict, critic_obs_key):
  """``make_tensor`` on ``args.task``; under ``--no-obs-noise`` with every
  observation group's corruption off (the actor's: ``eval_sonic_qcbf.py
  --no-obs-noise``), by wrapping the task's cfg builder for this one
  construction (the env reads it only then)."""
  kw = dict(cfg_overrides=overrides, random_push=push_kw, critic_obs_key=critic_obs_key)
  if not args.no_obs_noise:
    return make_tensor(args.task, args.num_envs, args.device, **kw)
  from robot_safety_sandbox import spec
  s = spec(args.task)
  build = s.cfg_builder

  def quiet(play: bool = False, **cfg_kw):
    cfg = build(play=play, **cfg_kw)
    for g in cfg.observations.values():
      if g is not None:
        g.enable_corruption = False
    return cfg

  s.cfg_builder = quiet
  try:
    return make_tensor(args.task, args.num_envs, args.device, **kw)
  finally:
    s.cfg_builder = build


def main(argv=None):
  args = _parse(argv)
  torch.manual_seed(args.seed)
  dev = args.device
  paths = list(args.bank)
  bank = load_bank(paths[0] if len(paths) == 1 else paths, "cpu")
  R = int(bank["kind"].numel())
  target = bank["start_ok"].clone()
  if args.max_rows is not None:
    idx = target.nonzero(as_tuple=False).squeeze(-1)[:int(args.max_rows)]
    target = torch.zeros(R, dtype=torch.bool)
    target[idx] = True
  n_target = int(target.sum())
  if n_target == 0:
    raise SystemExit("no start_ok row to label")

  run_cfg, cfg_path = _run_config(args.policy)
  if run_cfg.get("rl_dh") and not args.rl_dh:
    raise SystemExit(f"{args.policy}: its run trained RL-DH ({cfg_path}: rl_dh: "
                     f"true); label it with --rl-dh")
  if args.rl_dh:
    model, norm = load_rl_dh(args.policy, dev)
  else:
    model, norm = load_twin(args.policy, dev)
  model.policy.set_training_mode(False)
  value = PolicyValue(model, args.policy, dev)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0 if args.no_push else 1)
  overrides = dict(dataset=paths[0] if len(paths) == 1 else paths,
                   sampler="sequential", sequential_repeats=int(args.k),
                   episode_length_s=float(args.episode_s), rest_terminate=True,
                   rest_dwell_s=0.4, rest_arm_s=0.8)
  if args.thermal_start_c is not None:
    tc = [float(x) for x in args.thermal_start_c]
    if len(tc) not in (1, 2) or tc[0] > tc[-1]:
      raise SystemExit("--thermal-start-c takes T0 or LO HI (LO <= HI)")
    overrides.update(thermal_start_level_c=(tc[0], tc[-1]), thermal_start_soaked=True)
  env = make_env(args, overrides, push_kw, value.critic_obs_key)
  seq_len = getattr(env.mj, "_zoo_spawn_seq_len", None)
  print(f"[label] {args.policy} ({'asymmetric' if value.asymmetric else 'symmetric'} "
        f"critic) on {args.task} x {args.num_envs}: {n_target} rows x {args.k}, "
        f"push {'off' if args.no_push else f'{args.force_min_frac * args.force_max:.0f}-{args.force_max:.0f} N'}, "
        f"obs noise {'off' if args.no_obs_noise else 'on'}, "
        f"episode cap {args.episode_s} s; sequential pass {seq_len} entries")
  dh, dh_settings = None, None
  if args.rl_dh:
    dh, dh_settings = wrap_rl_dh(env, model, value, run_cfg, cfg_path, args)
  tube = WMTube(env, model, norm, args) if args.wm_tube is not None else None
  ep = rollouts(env, model, norm, value, target, bank["kind"], args, dh=dh, tube=tube)
  ab = write_labels(bank, ep, target)
  stats = report(bank, ep, target, ab, args)
  if dh_settings is not None:
    stats["rl_dh"]["settings"] = dh_settings
  label_meta = {
    "source": "label_start_states.py", "policy": os.path.abspath(args.policy),
    "task": args.task, "k": args.k, "force_max": args.force_max,
    "force_min_frac": args.force_min_frac, "no_push": bool(args.no_push),
    "obs_noise": not args.no_obs_noise,
    "episode_s": args.episode_s, "max_rows": args.max_rows, "seed": args.seed,
    "banks": [os.path.abspath(p) for p in paths],
    "critic": "asymmetric" if value.asymmetric else "symmetric",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "report": stats,
    "score_kind": ep["score_kind"], "wm_tube": args.wm_tube,
    "tube_calib": os.path.abspath(args.tube_calib) if args.tube_calib else None,
    "tube_tag": args.tube_cal["tag"] if args.tube_cal else None,
    "episodes_out": os.path.abspath(args.episodes_out),
    "thermal_start_c": args.thermal_start_c,
  }
  if dh_settings is not None:
    label_meta.update(mode="sonic_alone" if args.sonic_alone else "rl_dh",
                      rl_dh=dh_settings)
  bank["meta"] = dict(bank.get("meta", {}) or {}, label=label_meta)
  save_bank(bank, args.out)
  with open(report_path(args.out), "w") as f:
    json.dump(stats, f, indent=2)
  # one entry per labeled episode, 1-D CPU tensors (the soundness / volume tables)
  epi = {k: v.detach().cpu() for k, v in ep.items()
         if torch.is_tensor(v) and k != "action_stats"}
  epi.update(score_kind=ep["score_kind"], settings=ep["score_settings"], label=label_meta)
  os.makedirs(os.path.dirname(os.path.abspath(args.episodes_out)), exist_ok=True)
  torch.save(epi, args.episodes_out)
  print(f"[label] wrote {args.out}, {report_path(args.out)} and {args.episodes_out}")
  env.close()


if __name__ == "__main__":
  main()
