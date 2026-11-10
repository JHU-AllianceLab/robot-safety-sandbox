"""Q-CBF safety filter on SONIC: the push suite and the bad-reference suite.

    python examples/eval_sonic_qcbf.py --sonic-dir $HOME/gear_sonic_hf \
        --policy runs/shac_vs_ppo_sonic/shac_s0/g1_recover_safetyshac1p/final_model.zip
    python examples/eval_sonic_qcbf.py --sonic-dir $HOME/gear_sonic_hf --quick \
        --policy runs/shac_vs_ppo_sonic/ppo_s0/g1_recover/final_model.zip      # smoke

THE FILTER. SONIC (``envs/g1_recover/sonic.py``) tracks held-out LAFAN clips
and is the nominal controller, u_SONIC. A fallback policy pi (one of the
SafetyPPO / SafetySHAC policies of runs/shac_vs_ppo_sonic, trained from
SONIC's post-kick states) has fitted safety values V^pi(x) and Q^pi(x, u) of
the g_fall margin (``envs/g1_recover/policy_value_filter.py``): the ``exact``
fit (gamma 1, regressed on 3 s rollout labels) and the ``td_g099`` fit (TD,
gamma 0.99). The Q-CBF filter (Oh, Lidard et al., "Safety with Agency",
arXiv 2504.11717, Def. 2 / Prop. 1) runs at EVERY control step from the
spawn. It does not know when the push comes. At each step it applies

    u*(x) = argmin_u ||u - u_SONIC(x)||^2   s.t.   Q(x, u) >= kappa V(x)

solved by gradient descent (``filters.QCBFDescentIntervention``: an augmented
Lagrangian, with the paper's line search toward pi(x) as a net and pi(x) as
the best effort when nothing is feasible). Two limits keep it honest. A
TRUST REGION ||u - u_SONIC|| <= --qcbf-max-dev (default sqrt(29), the 1-sigma
radius of Q's widest training noise N(0, I)) refuses "certificates" that only
exist because Q is extrapolated, and hands such a step to pi. SONIC's ACTION
BOX (``policy_value_filter.sonic_action_box``, the targets its raw +-20 can
command) keeps every applied action one SONIC's history can record. The
conditions
(``qcbf_eval_lib``) are ``sonic`` (unfiltered), ``fallback`` (pi alone),
``qcbf:<variant>:<kappa>[:<solver>]``, ``lrsf:<variant>``, the paper's
last-resort switch (SONIC while Q(x, u_SONIC) > 0, else pi), and
``rollout:<seconds>``, the ROLLOUT FILTER: every control step, each tracked
env's exact state is copied into CPU MuJoCo (``cpu_rollout.py``, worker
processes, --rollout-workers), SONIC's action is played for one step and pi
for the rest of <seconds>, and SONIC's action goes through only if g stayed
>= 0 throughout; otherwise pi's action is applied. The rollout knows the
robot's domain-randomization draw but not the push to come, and pi drives it
on noise-free observations; it ends at the first g < 0. ``wmrollout:<seconds>``
is the same filter with the future IMAGINED in pi's own learned world model
(a SHAC checkpoint's, on GPU, batched): from the true privileged state and
the robot's sim context, u_SONIC for one model step, then pi on the model's
predicted state (noise-free, last action clamped to the box as for the live
pi); SONIC's action goes through only if the predicted g stayed >= 0 and the
predicted P(terminated) stayed <= 0.5. One process
evaluates one fallback (``--policy``, its fits found by ``--tag``). It builds
the env, SONIC and pi once and runs every condition on fixed episode lists.

THE SUITES.

1. push. The held-out clip library (``--library``,
   data/g1_lafan1_clips_test.pt). Each episode spawns on a start_ok mocap
   row with room for the whole episode and SONIC's look-ahead (next_ok >=
   onset + window + 46 frames, uniform over such rows). SONIC tracks the
   clip. ``onset`` ~ U(--onset-s) s after the spawn, the pelvis gets the
   harvester's velocity kick: |dv| ~ U(--dv-range) m/s at a uniform angle in
   the heading frame (``harvest_g1_sonic_trajectories.Harvest.push``). The
   episode SURVIVES when nothing terminates it (fell_over 70 deg,
   illegal_contact, task g1_recover_lafan) from the spawn to --window-s
   after the onset.
2. badref. The same spawns, the onset ~ U(--badref-onset-s), no push unless
   --badref-push. From the onset the REFERENCE is corrupted by one of
   ``bad_reference.SCENARIOS`` (high-frequency oscillation, jitter, speed-up,
   lean, an unstable pose, teleport / skip / reverse / freeze, mirror, yaw;
   --duration-s, then the scenario's own return / hold / continue). Each
   scenario has its own episode list and one corrupted reference with a clip
   per episode. SONIC tracks it from frame 0 (the spawn, identical to the
   clean clip before the onset). Survival counts to --badref-window-s after
   the onset.

COMMON RANDOM NUMBERS. Every list (spawn rows, onsets, pushes, corruption
draws) is drawn once from its own seeded generator, keyed by --seed and the
list's name, and is shared by every condition and by every policy run with
the same seed. The episodes run in WAVES of --num-envs. Before each wave the
spawn's sequential cursor is set to that wave's block of a per-run episode
bank (``<out>/episodes_bank.pt``, the lists' rows end to end; the env is
built on it with ``sampler="sequential"``), torch's global RNG is reseeded
from the list and the wave, and the env is fully reset. So episode e runs in
the SAME env slot under every condition, with the same startup domain
randomization (foot friction, torso CoM, encoder bias: drawn per env at
startup) and the same spawn yaw and offset. What the seed does not pin is
the per-step observation noise after the first mid-wave reset, and the
simulator's own float determinism. SONIC, V and Q read the robot's state,
not the noisy observation, and only pi reads it. An env whose episode ended
idles until the wave ends (its respawns are ignored), which costs ~25 % of
the env-steps against a streaming loop. That is the price of the fixed
slots.

ONE CONTROL STEP (the harvester's order; the kick lands after the step and
before anyone's next observation):

    x = runtime_features(state)               # V / Q input; last action = the APPLIED one
    u_SONIC = tracker.act(); pi(obs)          # pi every step (a recurrent pi sees every obs)
    u = condition(u_SONIC, pi(obs), x)        # the filter, tracked envs only
    record_applied_action(tracker, u)         # --sonic-history applied (default)
    env.step_tensor(u); k += 1
    push at k == onset; tracker.advance()     # pi's obs is recomputed for kicked envs

``--sonic-history applied`` writes the applied action into SONIC's action
history (what its training env would have recorded). ``nominal`` keeps
SONIC's own output there.

PI'S OBSERVATION. pi's actor observation holds the env's last applied action
(mjlab's ``last_action`` term), and SONIC's targets leave the +-3 action box
on most steps (85 % of SONIC's actions on the held-out test rows have an
entry past 3, up to 20). pi never saw that: in training its own actions were
clamped to the box, and so were the spawn rows' last actions and the last
action of every V / Q label's start. ``--pi-obs-clamp box`` (default) clamps
that term to the box before pi's normalizer, as in training and labelling
(V's features clamp it too); each run records how often the clamp bound.

TELEMETRY is split per episode at its ``split`` step. On the push suite that
is the push (PRE = normal tracking, where any intervention is a false alarm).
On badref it is onset - 47, the first step whose 46-frame look-ahead window
holds a corrupted frame or its velocity. Per phase it holds steps, engaged
steps, ||u - u_SONIC|| (sum and max), solver status counts (pass / gd /
segment / best effort), the sums of V, Q(x, u), Q(x, u_SONIC) and descent
iterations, and the joint tracking error |q - q_ref| (SONIC order, against
the TRUE reference; on badref also against the corrupted one).

OUTPUT in ``--out`` (default runs/shac_vs_ppo_sonic/qcbf_eval/<tag>, ``--quick``
under qcbf_eval_smoke/):

    episodes_bank.pt   the spawn rows of every list, end to end (the env's dataset)
    push.pt            {"meta", "episodes": per-episode columns, "conditions":
                        {name: per-episode records}, "summary"}
    push.json          {"meta", "conditions": {name: summary}}
    badref.pt          {"meta", "scenarios": {name: {"episodes", "corruption",
                        "conditions", "summary"}}}
    badref.json        {"meta", "scenarios": {name: {family, after, doc,
                        "conditions"}}, "pooled": {name: summary},
                        "by_family": {family: {name: summary}}}

A summary (``qcbf_eval_lib.summarize_condition``) holds SR with its Wilson
95 % CI, SR without the episodes whose reference leaves K by itself
(``ref_out_k``, the harvester's flag; on badref also ``bad_out_k``: the
corrupted reference's source frames or its pelvis tilt leave K), SR by |dv|
bin and by category, the pairs against ``sonic`` (rescued / broken, exact
McNemar p), intervention rates and deviations pre / post, false alarms,
best-effort rate, solver status mix and tracking error. badref.json is
written once every scenario has run; badref.partial.json holds the
scenarios done so far. ``summarize_sonic_qcbf.py`` pools the six policies.

K FLAGS. ``ref_out_k`` / ``bad_out_k`` are the harvester's K rule (non-foot
clearance < 0, pelvis below k_set.H_MIN = 0.45 m, tilt over k_set.THETA_MAX
= 1 rad), not this task's terminations (fell_over at 70 deg, a 10 N non-foot
contact). They mark episodes whose reference itself leaves the safe set K
(a deep crouch, a 57-70 deg lean), which the task may well survive; read
the excluding SRs as "without the episodes whose reference leaves K", not
"without those the reference makes fail". On badref, teleport jumps only
into the held-out clips, the only ones whose frames have K flags.

CAVEATS. ``--hip mjlab`` (default) drives the hip pitch with the sandbox G1's
7520-14 actuator (Kp 40.2, 88 N m), the robot the fallbacks were trained, and
their values labelled, on; ``--hip sonic`` with SONIC's 7520-22 (Kp 99.1,
139 N m): 2.5x the stiffness and 1.6x the torque. So with ``--hip mjlab``
SONIC runs off-model, and with ``--hip sonic`` pi and its Q / V do. The
diagnostics (qcbf_diag_jobs.txt, 512 push episodes) put the sizes at: SONIC
0.45 vs 0.52 survival (mjlab vs sonic hip), ppo_s0 alone 0.67 vs 0.31 (0.98
vs 0.66 without a kick), shac_s0 alone 0.85 vs 0.83. The variants exact /
td_g099 fit Q on SONIC's own next action plus N(0, 0.3^2) / N(0, 1) noise
only, so Q(x, pi(x)) is an extrapolation there; exact_pi / td_g099_pi
(policy_eval_qpi_jobs.txt) add pi's own action plus the same noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

import torch

_EX = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_EX)
sys.path.insert(0, _ZOO)
sys.path.insert(0, _EX)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import cpu_rollout as cr  # noqa: E402
import qcbf_eval_lib as ql  # noqa: E402
from harvest_g1_sonic_trajectories import (  # noqa: E402
  make_env, reference_clearance, reference_out_k)

from robot_safety_sandbox.envs.g1_recover import bad_reference as br  # noqa: E402
from robot_safety_sandbox.envs.g1_recover import k_set  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.policy_value_filter import (  # noqa: E402
  ACTION_DIM, DEFAULT_VALUE_DIR, PolicyValues, record_applied_action, runtime_features,
  sonic_action_box)
from robot_safety_sandbox.envs.g1_recover.sonic import (  # noqa: E402
  DEFAULT_SONIC_DIR, SonicPolicy, SonicReference, SonicTracker)
from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES, RecoverSpawn  # noqa: E402
from robot_safety_sandbox.envs.g1_recover.states import (  # noqa: E402
  FIELDS, KIND_MOCAP, OPTIONAL_FIELDS, load_bank, save_bank)
from robot_safety_sandbox.eval.policies import load_twin, reset_policy_memory  # noqa: E402
from robot_safety_sandbox.filters import (  # noqa: E402
  LeastRestrictiveIntervention, QCBFDescentIntervention)

#: SONIC's reference rate is the G1 tasks' control rate: a frame is a step.
DT = 1.0 / br.FPS
#: the conditions that roll SHAC's world model (they read the critic group's state)
WM_KINDS = ("wmrollout", "wmtube", "wmcalib", "wmmin")
#: The bridge's action box the value fits' last actions were clamped to.
FIT_ACTION_BOUND = 3.0


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="the fallback's final_model.zip")
  p.add_argument("--tag", default=None,
                 help="its value fits' tag (default: the run directory, e.g. shac_s0)")
  p.add_argument("--value-dir", default=DEFAULT_VALUE_DIR,
                 help="{v,q}_<tag>.pt and {v,q}_<variant>_<tag>.pt")
  p.add_argument("--library", default=os.path.join(_ZOO, "data", "g1_lafan1_clips_test.pt"),
                 help="the held-out clip library")
  p.add_argument("--sonic-dir", default=DEFAULT_SONIC_DIR,
                 help="the GEAR-SONIC release (model_encoder.onnx, model_decoder.onnx)")
  p.add_argument("--out", default=None,
                 help="default runs/shac_vs_ppo_sonic/qcbf_eval/<tag> (qcbf_eval_smoke/ with --quick)")
  p.add_argument("--task", default="g1_recover_lafan",
                 help="the env, margin and terminations (the values' g_fall task)")
  p.add_argument("--hip", choices=("sonic", "mjlab"), default="mjlab",
                 help="hip-pitch actuator: the sandbox G1's (pi's and its values') or "
                      "SONIC's training one")
  p.add_argument("--suites", nargs="+", choices=("push", "badref"), default=["push", "badref"])
  p.add_argument("--conditions", nargs="+", default=None,
                 help="push-suite conditions (default: " + " ".join(ql.PUSH_CONDITIONS) + ")")
  p.add_argument("--badref-conditions", nargs="+", default=None,
                 help="bad-reference conditions (default: " + " ".join(ql.BADREF_CONDITIONS) + ")")
  p.add_argument("--scenarios", nargs="+", default=None,
                 help="bad-reference scenarios (default: all of bad_reference.SCENARIOS)")
  p.add_argument("--episodes", type=int, default=16384, help="push-suite episodes")
  p.add_argument("--badref-episodes", type=int, default=4096, help="episodes per scenario")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--onset-s", type=float, nargs=2, default=[0.5, 1.5],
                 help="push time after the spawn, s (uniform)")
  p.add_argument("--window-s", type=float, default=4.0, help="survival window after the push, s")
  p.add_argument("--dv-range", type=float, nargs=2, default=[0.0, 5.0],
                 help="horizontal |dv| of the push, m/s (uniform)")
  p.add_argument("--dv-bins", type=float, nargs="+", default=list(ql.DV_BINS),
                 help="|dv| bin edges of the survival breakdown, m/s")
  p.add_argument("--badref-onset-s", type=float, nargs=2, default=[1.0, 2.0],
                 help="corruption onset after the spawn, s (uniform)")
  p.add_argument("--badref-window-s", type=float, default=3.0,
                 help="survival window after the corruption onset, s")
  p.add_argument("--duration-s", type=float, default=br.DEFAULT_DURATION_S,
                 help="corruption window, s (unless the scenario fixes its own)")
  p.add_argument("--badref-push", action="store_true",
                 help="also kick the robot at the corruption onset (the push suite's dv draws)")
  p.add_argument("--sonic-history", choices=("applied", "nominal"), default="applied",
                 help="SONIC's action history holds the applied (filtered) action or its own")
  p.add_argument("--qcbf-iters", type=int, default=60, help="QCBFDescentIntervention n_iter")
  p.add_argument("--qcbf-margin", type=float, default=0.0, help="tau = kappa V + margin")
  p.add_argument("--qcbf-max-dev", type=float, default=math.sqrt(ACTION_DIM),
                 help="trust region: certify only ||u - u_SONIC|| <= this, else hand over "
                      "to pi (default sqrt(29), Q's N(0, I) training noise; inf: no limit)")
  p.add_argument("--pi-obs-clamp", choices=("box", "none"), default="box",
                 help="clamp the last-action term of pi's observation to the action box, "
                      "as in its training (box) or not (none)")
  p.add_argument("--best-effort", choices=("fallback", "argmax"), default="fallback")
  p.add_argument("--tube-calib", default=None,
                 help="wmtube:<seconds>:<R>:<c>: the json of calibrate_wm_tube.py (lower_eps); "
                      "wmmin:<seconds>:<c>: that of calibrate_wm_min_tube.py (min_eps)")
  p.add_argument("--calib-horizon-s", type=float, default=3.0,
                 help="wmcalib: steps of the recorded prediction and real margins")
  p.add_argument("--calib-max-delay-s", type=float, default=1.0,
                 help="wmcalib: the decision step ~ U{0 .. this} after the onset")
  p.add_argument("--exclude-categories", nargs="+", default=[], metavar="NAME",
                 help="draw no episode from these clip categories (e.g. fallAndGetUp sprint)")
  p.add_argument("--no-obs-noise", action="store_true",
                 help="turn the actor group's sensor noise off (pi alone and every filter's pi "
                      "then act on exact observations, as a rollout's pi does)")
  p.add_argument("--rollout-workers", type=int,
                 default=int(os.environ.get("SLURM_CPUS_PER_TASK", max(1, (os.cpu_count() or 2) - 2))),
                 help="CPU worker processes of the rollout:<seconds> conditions")
  p.add_argument("--video-episodes", type=int, nargs="+", default=None,
                 help="push suite: render these episodes (indices into its list) under "
                      "--video-conditions, in their own env slots (filter_video.py)")
  p.add_argument("--video-conditions", nargs="+", default=None,
                 help="the video's panels, in order (default: every push condition)")
  p.add_argument("--video-dir", default=None, help="default <out>/videos")
  p.add_argument("--video-size", type=int, nargs=2, default=[480, 360], metavar=("W", "H"),
                 help="panel size, px")
  p.add_argument("--video-cols", type=int, default=0,
                 help="panels per row of the video (0: all in one row; 2 with 4 panels: 2x2)")
  p.add_argument("--lrsf-eps", type=float, default=0.0,
                 help="the switch hands over when Q(x, u_SONIC) <= eps")
  p.add_argument("--quick", action="store_true",
                 help="smoke: 256 envs, 512 push episodes, 4 push conditions, 3 scenarios x "
                      "3 conditions x 256 episodes")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args(argv)
  if a.quick:
    a.num_envs, a.episodes, a.badref_episodes = 256, 512, 256
    a.conditions = a.conditions or list(ql.QUICK_PUSH_CONDITIONS)
    a.badref_conditions = a.badref_conditions or list(ql.QUICK_BADREF_CONDITIONS)
    a.scenarios = a.scenarios or list(br.QUICK_SCENARIOS)
  a.conditions = a.conditions or list(ql.PUSH_CONDITIONS)
  a.badref_conditions = a.badref_conditions or list(ql.BADREF_CONDITIONS)
  a.scenarios = a.scenarios or list(br.SCENARIOS)
  try:
    # sonic first: every other condition's pairs need its outcomes
    a.push_conds = ql.reference_first(ql.parse_conditions(a.conditions))
    a.badref_conds = ql.reference_first(ql.parse_conditions(a.badref_conditions))
  except ValueError as e:
    p.error(str(e))
  if not a.qcbf_max_dev > 0:
    p.error("--qcbf-max-dev must be > 0 (inf: no trust region)")
  unknown = [s for s in a.scenarios if s not in br.SCENARIOS]
  if unknown:
    p.error(f"unknown scenarios {unknown}; known: {', '.join(br.SCENARIOS)}")
  if len(set(a.scenarios)) != len(a.scenarios):
    p.error("a scenario is listed twice")
  for name, (lo, hi) in (("--onset-s", a.onset_s), ("--badref-onset-s", a.badref_onset_s)):
    if not 0.0 < lo <= hi or int(round(lo / DT)) < 1:
      p.error(f"{name} needs 0 < lo <= hi and lo >= one control step ({DT} s)")
  if not 0.0 <= a.dv_range[0] <= a.dv_range[1]:
    p.error("--dv-range needs 0 <= lo <= hi")
  if any(b <= c for b, c in zip(a.dv_bins[1:], a.dv_bins[:-1])) or len(a.dv_bins) < 2:
    p.error("--dv-bins must be at least two increasing edges")
  if min(a.window_s, a.badref_window_s, a.duration_s) <= 0.0:
    p.error("windows and --duration-s must be > 0")
  if min(a.episodes, a.badref_episodes, a.num_envs, a.qcbf_iters) < 1:
    p.error("episode counts, --num-envs and --qcbf-iters must be >= 1")
  if a.video_episodes is not None:
    if "push" not in a.suites:
      p.error("--video-episodes renders push-suite episodes: add push to --suites")
    if min(a.video_episodes) < 0 or max(a.video_episodes) >= a.episodes:
      p.error(f"--video-episodes must lie in [0, {a.episodes})")
    names = [c.name for c in a.push_conds]
    try:
      vc = [ql.parse_condition(t).name for t in (a.video_conditions or names)]
    except ValueError as e:
      p.error(str(e))
    missing = [v for v in vc if v not in names]
    if missing:
      p.error(f"--video-conditions {missing} are not among --conditions")
    a.video_conditions = vc
    if min(a.video_size) < 16:
      p.error("--video-size must be at least 16 px each way")
    if a.video_cols < 0:
      p.error("--video-cols must be >= 0")
  if a.tag is None:
    a.tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.policy))))
  if a.out is None:
    sub = "qcbf_eval_smoke" if a.quick else "qcbf_eval"
    a.out = os.path.join(_ZOO, "runs", "shac_vs_ppo_sonic", sub, a.tag)
  if a.video_episodes is not None and a.video_dir is None:
    a.video_dir = os.path.join(a.out, "videos")
  return a


def _seed(seed: int, key: str) -> int:
  """A generator seed per (--seed, list / wave name): a list's draws do not
  depend on which other lists the run holds."""
  return (int(seed) * 1_000_003 + zlib.crc32(key.encode())) % (2 ** 62)


def _spawn_term(mj) -> RecoverSpawn:
  """The env's RecoverSpawn event term (it owns the sequential cursor)."""
  for cfgs in getattr(mj.event_manager, "_mode_term_cfgs", {}).values():
    for c in cfgs:
      if isinstance(getattr(c, "func", None), RecoverSpawn):
        return c.func
  raise SystemExit("[qcbf] the env has no RecoverSpawn reset term: the episode "
                   "waves need g1_recover's sequential spawn")


# --- the episode lists ----------------------------------------------------------------

def _steps(s: float) -> int:
  return int(round(float(s) / DT))


def episode_columns(lst: dict, window: int, lib: dict, base: SonicReference,
                    out_by_ref: torch.Tensor, clear_by_ref: torch.Tensor, dev) -> dict:
  """The per-episode columns of a list on ``dev``: its library row, clip,
  frame, category, onset, push, and whether the clean reference leaves K
  somewhere between the spawn and the window's end (``ref_out_k``, the
  harvester's flag; ``ref_clearance``: its lowest non-foot point)."""
  rows = lst["row"].to(dev)
  clip, frame0 = lib["clip"].to(dev)[rows], lib["frame"].to(dev)[rows]
  on = lst["on"].to(dev)
  span = on + int(window)
  f = torch.arange(int(span.max()) + 1, device=dev)
  fr = torch.where(f <= span.unsqueeze(1), frame0.unsqueeze(1) + f, frame0.unsqueeze(1))
  rr = base.rows(clip.unsqueeze(1).expand_as(fr), fr)
  return {"row": rows, "clip": clip, "frame0": frame0,
          "category": lib["category"].to(dev)[rows], "on": on, "onset_s": on.float() * DT,
          "split": on.clone(), "dv": lst["dv"].to(dev), "rel": lst["rel"].to(dev),
          "dv_h": lst["dv_h"].to(dev), "ref_out_k": out_by_ref[rr].any(1),
          "ref_clearance": clear_by_ref[rr].amin(1),
          # SONIC's start: the clip and frame of the spawn row
          "spawn_clip": clip, "spawn_frame": frame0}


def bad_columns(eps: dict, bad, window: int, base: SonicReference,
                out_by_ref: torch.Tensor) -> dict:
  """Extra columns of a bad-reference list once its reference is built:
  SONIC starts clip e at frame 0, telemetry splits where SONIC first sees the
  corruption, and ``bad_out_k`` says the corrupted reference itself leaves K
  (the harvester's rule, module docstring K FLAGS) over the tracked frames.
  Its source frames are out of K (exact for the index families, teleport
  included as long as the factory's targets are library clips, the only
  frames ``out_by_ref`` knows; the frames a value family was built from
  otherwise), or its pelvis tilt passes K's 1 rad."""
  cor = bad.corruption
  E, dev = int(eps["on"].numel()), eps["on"].device
  span = eps["on"] + int(window)
  f = torch.arange(int(span.max()) + 1, device=dev)
  inside = f <= span.unsqueeze(1)
  fm = torch.where(inside, f, torch.zeros_like(f))
  ar = torch.arange(E, device=dev).unsqueeze(1)
  src = base.rows(cor.src_clip[ar, fm], cor.src_frame[ar, fm])
  tilt = br.tilt_of(bad.root_quat[bad.rows(ar.expand_as(fm), fm)])
  tilt_max = torch.where(inside, tilt, torch.zeros_like(tilt)).amax(1)
  out = dict(eps)
  out.update({"spawn_clip": torch.arange(E, device=dev),
              "spawn_frame": torch.zeros(E, dtype=torch.long, device=dev),
              "split": (eps["on"] - (br.look_ahead() + 1)).clamp_min(0),
              "tilt_max": tilt_max,
              "bad_out_k": out_by_ref[src].any(1) | (tilt_max > k_set.THETA_MAX)})
  return out


def write_episode_bank(lib: dict, blocks: dict, path: str) -> dict:
  """The lists' spawn rows end to end, the env's dataset. Returns each
  block's first entry. ``next_ok`` is zeroed (the rows are not contiguous)."""
  keys = list(blocks)
  rows = torch.cat([blocks[k]["row"].cpu() for k in keys])
  bank = {f: lib[f][rows] for f in FIELDS + tuple(OPTIONAL_FIELDS) if f in lib}
  bank["next_ok"] = torch.zeros_like(bank["next_ok"])
  if not bool((bank["start_ok"] & (bank["kind"] == KIND_MOCAP)).all()):
    raise SystemExit("[qcbf] an episode spawns on a row that is not a start_ok mocap row")
  first, off = {}, 0
  for k in keys:
    first[k] = off
    off += int(blocks[k]["row"].numel())
  bank["meta"] = dict(lib["meta"], source="eval_sonic_qcbf.py (episode bank)",
                      library_rows=rows.clone(), blocks={k: [first[k], int(blocks[k]["row"].numel())]
                                                         for k in keys},
                      created=time.strftime("%Y-%m-%d %H:%M:%S"))
  save_bank(bank, path)
  return first


# --- the evaluator ---------------------------------------------------------------------

class Evaluator:
  """Runs one condition over one episode list (module docstring: waves,
  one control step)."""

  def __init__(self, env, tracker: SonicTracker, base: SonicReference, model, norm,
               values: dict, spawn: RecoverSpawn, args):
    self.env, self.mj, self.tr, self.base = env, env.mj, tracker, base
    self.model, self.norm, self.values, self.spawn, self.a = model, norm, values, spawn, args
    self.dev, self.n = env.device, env.num_envs
    self.robot = self.mj.scene["robot"]
    self.low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.dev)
    self.high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.dev)
    self._fresh_obs = self._can_refresh_obs()
    # the filter's actions stay inside what SONIC's raw +-ACTION_CLIP can command
    self.box = sonic_action_box(tracker)
    self.pi_slice, self.pi_obs_dim = (self._last_action_slice() if args.pi_obs_clamp == "box"
                                      else (None, None))
    self.pool = None
    #: the last control step, for on_step hooks (filter_video.py): the tracked
    #: envs ``ids``, the verdict before any latch (None without one), whether
    #: pi acted, and after the step the margin g of every env
    self.trace: dict = {}

  def setup_rollouts(self, **spec_kw) -> dict:
    """The CPU rollout pool of the ``rollout:<seconds>`` conditions: pi
    exported as the live call (the same last-action clamp, normalizer and
    action clamp) and the env's model with every slot's DR draw. ``spec_kw``
    go to ``cpu_rollout.build_spec`` (its margin). Returns its meta."""
    a = self.a
    work = os.path.join(a.out, "rollout")
    os.makedirs(work, exist_ok=True)
    m_cpu, n_cpu = load_twin(a.policy, "cpu")
    m_cpu.policy.set_training_mode(False)
    obs_dim = int(self.mj.observation_manager.compute_group(
      self.env.obs_key, update_history=False).shape[1])
    actor = os.path.join(work, "actor.pt")
    cr.export_actor(m_cpu.policy, n_cpu, obs_dim, self.pi_slice, self.low.cpu(), self.high.cpu(),
                    self.low.cpu(), self.high.cpu(), actor)
    spec = cr.build_spec(self.env, actor, work, **spec_kw)
    t0 = time.time()
    self.pool = cr.CpuRolloutPool(spec, a.rollout_workers, seed=a.seed)
    print(f"[qcbf] rollout pool: {self.pool.workers} CPU workers over {spec.n_slots} env slots "
          f"({time.time() - t0:.0f}s to start); pi's input {[t[0] for t in spec.obs_terms]}, "
          f"DR fields {sorted(spec.dr)}", flush=True)
    return {"backend": "CPU MuJoCo (cpu_rollout.py)", "workers": self.pool.workers,
            "first_action": "u_SONIC, then pi (deterministic, noise-free observation)",
            "verdict": ("min over the horizon of compose(g_fall) >= 0" if spec.margin == "fall"
                        else f"min over the horizon of K's margin (joint scale "
                             f"{spec.k_joint_scale:g}) >= 0"),
            "dr_fields": sorted(spec.dr), "state": ["qpos", "qvel", "qacc_warmstart"]}

  def close(self) -> None:
    if self.pool is not None:
      self.pool.close()
      self.pool = None

  def setup_world_model(self) -> dict:
    """Checks for the ``wmrollout:<seconds>`` conditions: pi is a SHAC
    checkpoint whose world model is fitted and reads this env's privileged
    critic group and sim context. Returns its meta."""
    wm = getattr(self.model, "world_model", None)
    if wm is None or not bool(wm.fitted):
      raise SystemExit(f"[qcbf] wmrollout: {self.a.policy} has no fitted world model")
    s = self.mj.observation_manager.compute_group("critic", update_history=False)
    c = self.env.sim_context()
    if int(s.shape[1]) != int(wm.state_dim) or (0 if c is None else int(c.shape[1])) != int(
        wm.context_dim):
      raise SystemExit(f"[qcbf] wmrollout: the env's critic group ({s.shape[1]}-D) / sim context "
                       f"({None if c is None else c.shape[1]}-D) are not the world model's "
                       f"{wm.state_dim}-D / {wm.context_dim}-D")
    self.wm_actor_idx = self.model._actor_idx_t.to(self.dev)
    self.wm_z_clip = getattr(self.model, "wm_z_clip", None)
    # the critic-state terms behind the margin's height and tilt (wmcalib records them)
    layout = self.env._obs_layout("critic")
    self.wm_state_sl = {name: slice(layout[name][0], layout[name][0] + layout[name][1])
                        for name in ("pelvis_height", "projected_gravity") if name in layout}
    if (len(self.wm_state_sl) != 2 or self.wm_state_sl["pelvis_height"].stop
            - self.wm_state_sl["pelvis_height"].start != 1
            or self.wm_state_sl["projected_gravity"].stop
            - self.wm_state_sl["projected_gravity"].start != 3):
      print(f"[qcbf] the critic group has no 1-D pelvis_height / 3-D projected_gravity "
            f"({sorted(layout)}): wmcalib records no margin states", flush=True)
      self.wm_state_sl = None
    return {"state": f"critic group ({wm.state_dim}-D, true)", "context_dim": int(wm.context_dim),
            "z_clip": self.wm_z_clip, "pi_input": "rebuilt from the predicted state, noise-free",
            "verdict": "min predicted g >= 0 and P(terminated) <= 0.5 over the horizon"}

  def margin_states(self, s: torch.Tensor) -> torch.Tensor:
    """Pelvis height (m) and tilt from vertical (deg) of critic states ``s``
    ``[..., D]`` -> ``[..., 2]`` (the margin's height and tilt inputs)."""
    sl = self.wm_state_sl
    h = s[..., sl["pelvis_height"]][..., 0]
    pg = s[..., sl["projected_gravity"]]
    cos = (-pg[..., 2] / pg.norm(dim=-1).clamp_min(1e-9)).clamp(-1.0, 1.0)
    return torch.stack([h, torch.rad2deg(torch.acos(cos))], dim=-1)

  @torch.no_grad()
  def wm_margins(self, ids: torch.Tensor, a_first: torch.Tensor, horizon: int,
                 n_rollouts: Optional[int] = None, states: bool = False):
    """The world model's predicted margins for envs ``ids``, ``[n, R, horizon]``:
    ``a_first`` for one model step, then pi on the predicted state. Step j's
    value is the predicted g, pushed to <= -0.1 once the predicted
    P(terminated) by step j passes 0.5 (a predicted fall stays one).
    ``n_rollouts`` R: R rollouts per env, pi reading the predicted state
    through the sensor-noise law (a fresh draw per rollout and step, as SHAC's
    imagination draws it); None: one noise-free rollout. ``states``: also
    return the decoded state's pelvis height and tilt after each step,
    ``[n, R, horizon, 2]`` (:meth:`margin_states`)."""
    wm, policy = self.model.world_model, self.model.policy
    R = 1 if n_rollouts is None else int(n_rollouts)
    noisy = n_rollouts is not None and self.model._noise_lo_t is not None
    s = torch.nan_to_num(self.mj.observation_manager.compute_group(
      "critic", update_history=False).float())[ids]
    ctx = self.env.sim_context()            # None: nothing is randomized per env
    c = None if ctx is None else ctx[ids]
    if R > 1:
      s, a_first = (x.repeat_interleave(R, dim=0) for x in (s, a_first))
      c = None if c is None else c.repeat_interleave(R, dim=0)
    cn = wm.encode_context(c)
    z = wm.encode(s)
    alive = torch.ones(int(s.shape[0]), device=self.dev)
    out = torch.empty(int(s.shape[0]), horizon, device=self.dev)
    sts = (torch.full((int(s.shape[0]), horizon, 2), math.nan, device=self.dev)
           if states and self.wm_state_sl is not None else None)
    act = a_first
    lo, hi = self.model._noise_lo_t, self.model._noise_hi_t
    for t in range(horizon):
      st = wm.step(z, act, cn, z_clip=self.wm_z_clip)
      z = st.z_next
      alive = alive * (1.0 - torch.sigmoid(st.done_logit))
      out[:, t] = torch.where(alive < 0.5, torch.clamp_max(st.g, -0.1), st.g)
      dec = wm.decode(z) if (sts is not None or t + 1 < horizon) else None
      if sts is not None:
        sts[:, t] = self.margin_states(dec)
      if t + 1 < horizon:
        o = dec.index_select(-1, self.wm_actor_idx)
        if noisy:
          o = o + lo + (hi - lo) * torch.rand_like(o)
        o, _bound = self._pi_obs(o)
        act = torch.clamp(policy._predict(self.norm(o), deterministic=True), self.low, self.high)
    if states:
      return out.view(-1, R, horizon), (None if sts is None else sts.view(-1, R, horizon, 2))
    return out.view(-1, R, horizon)

  def wm_verdict(self, ids: torch.Tensor, a_first: torch.Tensor, horizon: int,
                 n_rollouts: Optional[int] = None) -> torch.Tensor:
    """The world-model filter's verdict: the min over steps and rollouts of
    :meth:`wm_margins` (the worst rollout decides)."""
    return self.wm_margins(ids, a_first, horizon, n_rollouts).amin(dim=(1, 2))

  def tube_verdict(self, ids: torch.Tensor, a_first: torch.Tensor, cond) -> torch.Tensor:
    """The calibrated tube's verdict: min over steps j <= H of (the lowest of
    the R rollouts' predicted margins at j, minus eps_j at coverage c)."""
    H = _steps(cond.horizon_s)
    eps = self.tube_eps(cond.coverage, H)
    m = self.wm_margins(ids, a_first, H, cond.n_rollouts).amin(dim=1)     # [n, H]
    return (m - eps).amin(dim=1)

  def min_verdict(self, ids: torch.Tensor, a_first: torch.Tensor, cond) -> torch.Tensor:
    """The conformal worst-margin bound: min over j <= H of the one
    noise-free rollout's predicted margin, minus eps_H at coverage c."""
    H = _steps(cond.horizon_s)
    tab = self.tube_calib.get("min_eps", {}).get(f"{cond.coverage:g}", {})
    if str(H) not in tab:
      raise SystemExit(f"[qcbf] --tube-calib has no min_eps at coverage {cond.coverage:g} for "
                       f"{H} steps")
    return self.wm_margins(ids, a_first, H, None)[:, 0].amin(dim=1) - float(tab[str(H)])

  def tube_eps(self, coverage: float, H: int) -> torch.Tensor:
    tab = self.tube_calib.get("lower_eps", {}).get(f"{coverage:g}")
    if tab is None or len(tab) < H:
      raise SystemExit(f"[qcbf] --tube-calib has no lower_eps at coverage {coverage:g} for "
                       f"{H} steps")
    return torch.as_tensor(tab[:H], dtype=torch.float32, device=self.dev)

  def _last_action_slice(self) -> tuple[slice, int]:
    """The columns of pi's observation group that hold mjlab's ``last_action``
    term, and the group's width. Fails loudly when they cannot be found (run
    with --pi-obs-clamp none to go without the clamp)."""
    om, key = self.mj.observation_manager, self.env.obs_key
    names = getattr(om, "_group_obs_term_names", {}).get(key)
    dims = getattr(om, "_group_obs_term_dim", {}).get(key)
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(key)
    if not (names and dims and cfgs) or not len(names) == len(dims) == len(cfgs):
      raise SystemExit(f"[qcbf] cannot read the terms of observation group {key!r}; "
                       f"pass --pi-obs-clamp none")
    hits = [i for i, c in enumerate(cfgs)
            if getattr(getattr(c, "func", None), "__name__", "") == "last_action"]
    hits = hits or [i for i, nm in enumerate(names) if nm == "actions"]
    sizes = [int(math.prod(d)) for d in dims]
    if len(hits) != 1:
      raise SystemExit(f"[qcbf] observation group {key!r} ({names}) has {len(hits)} last-action "
                       f"terms, not one; pass --pi-obs-clamp none")
    i, c = hits[0], cfgs[hits[0]]
    if (sizes[i] != ACTION_DIM or getattr(c, "scale", None) is not None
            or getattr(c, "clip", None) is not None):
      raise SystemExit(f"[qcbf] the last-action term {names[i]!r} is {sizes[i]}-D with scale "
                       f"{getattr(c, 'scale', None)} / clip {getattr(c, 'clip', None)}: the box "
                       f"clamp expects the raw {ACTION_DIM}-D action; pass --pi-obs-clamp none")
    start = sum(sizes[:i])
    print(f"[qcbf] pi's last-action term {names[i]!r}: observation columns "
          f"{start}:{start + ACTION_DIM} of {sum(sizes)}, clamped to the action box")
    return slice(start, start + ACTION_DIM), sum(sizes)

  def _pi_obs(self, obs: torch.Tensor):
    """pi's input: ``obs`` with the last-action term clamped to the action box
    (``--pi-obs-clamp box``), and per env whether the clamp bound."""
    if self.pi_slice is None:
      return obs, None
    if int(obs.shape[1]) != self.pi_obs_dim:
      raise SystemExit(f"[qcbf] pi's observation is {obs.shape[1]}-D, its group's terms "
                       f"sum to {self.pi_obs_dim}")
    seg = obs[:, self.pi_slice]
    clamped = torch.maximum(torch.minimum(seg, self.high), self.low)
    out = obs.clone()
    out[:, self.pi_slice] = clamped
    return out, (clamped != seg).any(1)

  def _can_refresh_obs(self) -> bool:
    """Whether pi's observation group can be recomputed after a kick without
    side effects: no delayed or history terms (those buffers would advance)."""
    om = self.mj.observation_manager
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(self.env.obs_key)
    if cfgs is None:
      print(f"[qcbf] cannot inspect observation group {self.env.obs_key!r}: pi sees a "
            f"kick one step late")
      return False
    lagged = [c for c in cfgs if getattr(c, "delay_max_lag", 0) > 0
              or getattr(c, "history_length", 0) > 0]
    if lagged:
      print(f"[qcbf] observation group {self.env.obs_key!r} has delayed / history terms: "
            f"pi sees a kick one step late")
    return not lagged

  def controller(self, c: ql.Condition) -> ql.Controller:
    itv = None
    if c.kind == "qcbf":
      itv = QCBFDescentIntervention(self.n, self.dev, kappa=c.kappa, margin=self.a.qcbf_margin,
                                    solver=c.solver, n_iter=self.a.qcbf_iters,
                                    best_effort=self.a.best_effort, action_dim=ACTION_DIM,
                                    max_dev=self.a.qcbf_max_dev, action_low=self.box[0],
                                    action_high=self.box[1])
    elif c.kind == "lrsf":
      itv = LeastRestrictiveIntervention(self.n, self.dev, eps=self.a.lrsf_eps,
                                         action_dim=ACTION_DIM)
    mon = self.values[c.variant].monitor() if c.needs_values else None
    return ql.Controller(c, itv, mon)

  def _set_cursor(self, entry: int) -> None:
    self.spawn._cursor = int(entry)
    self.mj._zoo_spawn_cursor = int(entry)

  def _push(self, mask: torch.Tensor, dv_h: torch.Tensor) -> None:
    """``Harvest.push``: the root link's world twist += dv (heading frame)."""
    d = self.robot.data
    h = d.heading_w.reshape(-1)
    c, s = torch.cos(h), torch.sin(h)
    dv_w = torch.stack([c * dv_h[:, 0] - s * dv_h[:, 1],
                        s * dv_h[:, 0] + c * dv_h[:, 1], dv_h[:, 2]], 1)
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    vel = d.root_link_vel_w[ids].clone()
    vel[:, :3] += dv_w[ids]
    self.robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
    # the twist accessors read cvel: refresh it so every next reading sees the kick
    self.mj.sim.forward()

  def _finish(self, mask: torch.Tensor) -> None:
    """``Harvest._finish``: end these episodes on the next step (a truncation
    through the env's own time_out)."""
    self.tr.stop(mask)
    buf = self.mj.episode_length_buf
    buf[mask] = torch.clamp_min(buf[mask], int(self.mj.max_episode_length) - 1)

  def _wave_envs(self, label: str, block0: int, w0: int, w1: int) -> torch.Tensor:
    """After the wave's reset: each env's episode (-1: idle). An episode the
    spawn issued twice (a bank shorter than the env count wraps around) runs
    in its lowest env slot only."""
    ep = self.mj._zoo_spawn_row.clone() - block0
    cand = ((ep >= w0) & (ep < w1)).nonzero(as_tuple=False).squeeze(-1)
    e_c = ep[cand]
    order = torch.argsort(e_c, stable=True)
    es = e_c[order]
    first = torch.ones_like(es, dtype=torch.bool)
    first[1:] = es[1:] != es[:-1]
    keep = cand[order[first]]
    if int(keep.numel()) != w1 - w0:
      raise SystemExit(f"[qcbf] {label}: the spawn issued {int(keep.numel())} of the wave's "
                       f"{w1 - w0} episodes (cursor {block0 + w0}); the sequential sampler "
                       f"did not walk the episode bank in order")
    out = torch.full_like(ep, -1)
    out[keep] = ep[keep]
    return out

  @torch.no_grad()
  def run(self, label: str, ctl: ql.Controller, block0: int, eps: dict, ref: SonicReference,
          window: int, push: bool, true_rows: torch.Tensor | None = None,
          on_step=None) -> dict:
    """Every episode of ``eps`` under ``ctl`` (module docstring), tracking
    ``ref``; ``true_rows`` (``[E, L]`` base rows) marks a corrupted reference.
    Returns the per-episode records (``EpisodeBook.records``) plus
    ``"_run"``: wall time and step counts. Runs under no_grad (the descent
    turns autograd back on for the action only). ``on_step(ep, k, dones)``,
    when given, is called after each wave's reset (``dones`` None) and after
    every control step, kick included: ``ep`` is each env's episode (-1:
    idle), ``k`` its steps since the spawn, and an env whose ``dones`` is
    True has already respawned (``record_sonic_push_video.py`` renders
    there)."""
    a, dev, n, mj, tr = self.a, self.dev, self.n, self.mj, self.tr
    E = int(eps["on"].numel())
    book = ql.EpisodeBook(n, E, DT, dev)
    tr.ref = ref
    bad = true_rows is not None
    L = int(true_rows.shape[1]) if bad else 0
    applied = a.sonic_history == "applied"
    policy = self.model.policy
    t0, loop, env_steps = time.time(), 0, 0
    pi_steps = pi_clamped = 0
    rollout_s, rollouts = 0.0, 0
    if ctl.cond.kind == "rollout" and self.pool is None:
      raise SystemExit(f"[qcbf] {ctl.cond.name}: no rollout pool (Evaluator.setup_rollouts)")
    horizon = _steps(ctl.cond.horizon_s) if ctl.cond.kind in ql.ROLLOUT_KINDS else 0
    kind = ctl.cond.kind
    latch = bool(ctl.cond.latch)
    dwell = _steps(ctl.cond.dwell_s) if ctl.cond.dwell_s is not None else 0
    calib = kind == "wmcalib"
    if calib:
      Hc, Kc = _steps(a.calib_horizon_s), _steps(a.calib_max_delay_s)
      if Kc + Hc > window:
        raise SystemExit(f"[qcbf] {ctl.cond.name}: decision delay + horizon ({Kc + Hc} steps) "
                         f"exceeds the survival window ({window})")
      gen_c = torch.Generator().manual_seed(_seed(a.seed, f"{label}/calib"))
      k_off = torch.randint(0, Kc + 1, (E,), generator=gen_c).to(dev)   # decision - onset
      c_pred = torch.full((E, ctl.cond.n_rollouts or 1, Hc), math.nan, device=dev)
      c_real = torch.full((E, Hc), math.nan, device=dev)
      # the margin's height / tilt inputs, predicted (decoded) and real; NaN once the robot fell
      rec_states = getattr(self, "wm_state_sl", None) is not None
      if rec_states:
        c_pred_st = torch.full((E, ctl.cond.n_rollouts or 1, Hc, 2), math.nan, device=dev)
        c_real_st = torch.full((E, Hc, 2), math.nan, device=dev)
    waves = (E + n - 1) // n
    for w, w0 in enumerate(range(0, E, n)):
      w1 = min(E, w0 + n)
      torch.manual_seed(_seed(a.seed, f"{label}/wave{w0}"))
      self._set_cursor(block0 + w0)
      self.env.reset()
      obs = torch.nan_to_num(self.env.obs_groups()[self.env.obs_key].float(),
                             nan=0.0, posinf=0.0, neginf=0.0)
      reset_policy_memory(policy)
      ep = self._wave_envs(label, block0, w0, w1)
      take = ep >= 0
      epc = ep.clamp_min(0)
      on, split, dv_h = eps["on"][epc], eps["split"][epc], eps["dv_h"][epc]
      everyone = torch.ones(n, dtype=torch.bool, device=dev)
      tr.stop(everyone)
      tr.clip.zero_()
      tr.frame.zero_()
      tr.start(take, eps["spawn_clip"][epc], eps["spawn_frame"][epc])
      book.start(take)
      # pi acted (this episode, under +latch: ever; the last step: for the switches)
      fired = torch.zeros(n, dtype=torch.bool, device=dev)
      prev_eng = torch.zeros(n, dtype=torch.bool, device=dev)
      # +dwell: pi holds every step before this one (set at each hand-over to pi)
      hold_until = torch.zeros(n, dtype=torch.long, device=dev)
      track = take.clone()
      k = torch.zeros(n, dtype=torch.long, device=dev)
      if on_step is not None:
        on_step(ep, k, None)
      cap = int(on[take].max()) + window + 2
      for _ in range(cap):
        if not bool(track.any()):
          break
        ids = track.nonzero(as_tuple=False).squeeze(-1)
        # --- the state the next action is applied from
        q_now = self.robot.data.joint_pos[ids][:, tr.to_sonic]
        jerr = (q_now - ref.joint_pos[ref.rows(tr.clip[ids], tr.frame[ids])]).abs().mean(1)
        jerr_bad = None
        if bad:
          f = tr.frame[ids].clamp(0, L - 1)
          jerr, jerr_bad = (q_now - self.base.joint_pos[true_rows[ep[ids], f]]).abs().mean(1), jerr
        x = (runtime_features(mj, self.low, self.high, dev)[ids] if ctl.needs_values else None)
        a_nom = tr.act()
        a_safe = None
        if ctl.needs_fallback:
          pobs, bound = self._pi_obs(obs)
          a_safe = torch.clamp(policy._predict(self.norm(pobs), deterministic=True),
                               self.low, self.high)
          pi_steps += int(ids.numel())
          pi_clamped += 0 if bound is None else int(bound[ids].sum())
          if tuple(a_safe.shape) != tuple(a_nom.shape):
            raise SystemExit(f"[qcbf] pi's action {tuple(a_safe.shape)} vs SONIC's "
                             f"{tuple(a_nom.shape)}")
          a_safe = a_safe[ids]
        verdict = None
        if calib:
          k_dec = on[ids] + k_off[epc[ids]]
          now = k[ids] == k_dec
          if bool(now.any()):
            d_ids = ids[now]
            if rec_states:
              m_p, s_p = self.wm_margins(d_ids, a_nom[d_ids].float(), Hc, ctl.cond.n_rollouts,
                                         states=True)
              c_pred[ep[d_ids]], c_pred_st[ep[d_ids]] = m_p, s_p
            else:
              c_pred[ep[d_ids]] = self.wm_margins(d_ids, a_nom[d_ids].float(), Hc,
                                                  ctl.cond.n_rollouts)
          # SONIC up to and at the decision step, pi from the next one
          verdict = torch.where(k[ids] > k_dec, -1.0, 1.0).float()
        elif kind == "wmtube":
          t_r = time.time()
          verdict = self.tube_verdict(ids, a_nom[ids].float(), ctl.cond)
          rollout_s += time.time() - t_r
          rollouts += int(ids.numel()) * ctl.cond.n_rollouts
        elif kind == "wmmin":
          t_r = time.time()
          verdict = self.min_verdict(ids, a_nom[ids].float(), ctl.cond)
          rollout_s += time.time() - t_r
          rollouts += int(ids.numel())
        if horizon:
          t_r = time.time()
          R = ctl.cond.n_rollouts
          if ctl.cond.kind == "wmrollout":
            verdict = self.wm_verdict(ids, a_nom[ids].float(), horizon, R)
          else:
            qpos, qvel, ws = cr.live_state(self.env, ids)
            gmin, _t_fail = self.pool.rollout(ids.cpu().numpy(), qpos, qvel, ws,
                                              a_nom[ids].double().cpu().numpy(), horizon,
                                              n_rollouts=R or 1, noise=R is not None)
            verdict = torch.as_tensor(gmin, dtype=torch.float32, device=dev)
          rollout_s += time.time() - t_r
          rollouts += int(ids.numel()) * (ctl.cond.n_rollouts or 1)
        # --- the condition, on the tracked envs only (idle ones keep SONIC's)
        raw_verdict = verdict
        if latch:
          verdict = torch.where(fired[ids], torch.full_like(verdict, -1.0), verdict)
        if dwell:
          verdict = torch.where(k[ids] < hold_until[ids], torch.full_like(verdict, -1.0), verdict)
        u, tel = ctl(a_nom[ids], a_safe, x, verdict)
        eng_now = tel["engaged"].bool()
        if latch:
          fired[ids] |= eng_now
        if dwell:
          start = eng_now & ~prev_eng[ids]
          hold_until[ids] = torch.where(start, k[ids] + dwell, hold_until[ids])
        if latch or dwell:
          tel["v"] = raw_verdict.float()
        tel["switch"] = eng_now != prev_eng[ids]
        prev_eng[ids] = eng_now
        self.trace = {"ids": ids, "verdict": raw_verdict, "engaged": eng_now}
        act = a_nom.clone()
        act[ids] = u.to(act.dtype)
        book.step(ids, (k[ids] >= split[ids]).long(), tel, jerr, jerr_bad, k[ids] - on[ids])
        if applied:
          record_applied_action(tr, act, mask=track)
        # --- the step, then the kick, then SONIC's clock
        obs, _g, dones, timeouts, _l = self.env.step_tensor(act)
        self.trace["g"] = _g
        k += 1
        if calib:
          # step j = k - decision (1..Hc) of the real "u_SONIC, then pi"
          j = k - (on + k_off[epc])
          rec_m = track & (j >= 1) & (j <= Hc)
          if bool(rec_m.any()):
            r_ids = rec_m.nonzero(as_tuple=False).squeeze(-1)
            c_real[ep[r_ids], (j[r_ids] - 1)] = _g[r_ids].float()
            live = rec_m & ~dones            # a done env has respawned: its state is not the fall's
            if rec_states and bool(live.any()):
              l_ids = live.nonzero(as_tuple=False).squeeze(-1)
              cs = torch.nan_to_num(self.mj.observation_manager.compute_group(
                "critic", update_history=False).float())[l_ids]
              c_real_st[ep[l_ids], (j[l_ids] - 1)] = self.margin_states(cs)
        if push:
          due = track & ~dones & (k == on)
          if bool(due.any()):
            self._push(due, dv_h)
            if kind in WM_KINDS:
              # the critic group keeps a one-frame history that env.step filled BEFORE the
              # kick; refresh it so the world model's rollouts (wm_margins) start from the
              # kicked state, as the exact rollout filter's live state does
              mj.observation_manager.compute_group("critic", update_history=True)
            if ctl.needs_fallback and self._fresh_obs:
              # mjlab's own interval pushes land before the observation; so
              # does this one for pi (SONIC and x read the robot directly)
              fresh = mj.observation_manager.compute_group(self.env.obs_key, update_history=False)
              if torch.is_tensor(fresh) and fresh.shape == obs.shape:
                obs = torch.where(due.unsqueeze(1), torch.nan_to_num(
                  fresh.float(), nan=0.0, posinf=0.0, neginf=0.0), obs)
        tr.advance()
        if on_step is not None:
          on_step(ep, k, dones)
        oc = ql.step_outcomes(track, dones, timeouts, k, on, window)
        if bool(oc["cut"].any()):
          raise SystemExit(f"[qcbf] {label}: {int(oc['cut'].sum())} episodes hit the env's "
                           f"time cap ({mj.max_episode_length} steps) inside their window")
        book.finish(oc["fell"], ep, False, k, on)
        book.finish(oc["up"], ep, True, k, on)
        self._finish(oc["up"])
        tr.stop(dones)
        track &= ~(oc["fell"] | oc["up"])
        if bool(dones.any()):
          reset_policy_memory(policy, dones)
        loop += 1
        env_steps += int(ids.numel())
      if bool(track.any()):
        raise SystemExit(f"[qcbf] {label}: {int(track.sum())} episodes still running after "
                         f"{cap} steps")
      el = max(time.time() - t0, 1e-6)
      print(f"[qcbf]   {label} {ctl.cond.name}: wave {w + 1}/{waves} done "
            f"({int(book.done.sum())}/{E} episodes, {loop} steps, "
            f"{loop * n / el:.0f} env-steps/s)", flush=True)
    rec = book.records()
    if calib:
      rec.update(calib_pred=c_pred.cpu(), calib_real=c_real.cpu(), calib_decision=k_off.cpu())
      if rec_states:
        rec.update(calib_pred_states=c_pred_st.cpu(), calib_real_states=c_real_st.cpu(),
                   calib_state_names=["pelvis_height_m", "tilt_deg"])
    rec["_run"] = {"wall_s": time.time() - t0, "loop_steps": loop,
                   "tracked_env_steps": env_steps, "env_steps": loop * n,
                   "env_steps_per_s": loop * n / max(time.time() - t0, 1e-6),
                   # the share of pi's tracked steps whose last action left the box
                   "pi_obs_clamped": (pi_clamped / pi_steps) if pi_steps else None}
    if horizon:
      rec["_run"].update(rollout_horizon_steps=horizon, rollouts=rollouts, rollout_wall_s=rollout_s,
                         rollout_ms_per_step=1e3 * rollout_s / max(loop, 1))
    if pi_steps and self.pi_slice is not None:
      print(f"[qcbf]   {label} {ctl.cond.name}: pi's last-action input clamped on "
            f"{pi_clamped / pi_steps:.3f} of its steps", flush=True)
    return rec


# --- outputs ----------------------------------------------------------------------------

def _cpu(d: dict) -> dict:
  return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


def _write_json(obj, path: str) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(ql.clean_json(obj), f, indent=1)
  os.replace(tmp, path)


def _strip(rec: dict) -> dict:
  return {k: v for k, v in rec.items() if not k.startswith("_")}


def run_push(ev: Evaluator, a, conds, block0: int, eps: dict, base: SonicReference,
             cats: list, meta: dict, clips: list | None = None) -> None:
  win = _steps(a.window_s)
  recs, runs, summary = {}, {}, {}
  ref = None
  vid = None
  if a.video_episodes is not None:
    import filter_video as fv
    vid = fv.FilterVideo(ev.env, a.video_episodes, a.video_dir, _cpu(eps), cats, clips or [],
                         DT, win, tuple(a.video_size), cols=a.video_cols)
  for c in conds:
    hook = vid.hook(c.name, ev) if vid is not None and c.name in a.video_conditions else None
    rec = ev.run("push", ev.controller(c), block0, eps, base, win, push=True, on_step=hook)
    runs[c.name] = rec.pop("_run")
    recs[c.name] = rec
    if c.name == ql.REFERENCE:
      ref = rec["survived"]
    s = ql.summarize_condition(rec, _cpu(eps), c, None if c.name == ql.REFERENCE else ref,
                               cats, a.dv_bins)
    summary[c.name] = s
    print(f"[qcbf] push {c.name}: {ql.headline(s)} ({runs[c.name]['wall_s']:.0f}s)", flush=True)
  m = dict(meta, suite="push", episodes=int(eps["on"].numel()),
           conditions=[c.name for c in conds], runs=runs, onset_s=list(a.onset_s),
           window_s=a.window_s, dv_range=list(a.dv_range), dv_bins=list(a.dv_bins),
           split="the push step (pre = normal tracking before the kick)")
  torch.save({"meta": m, "episodes": _cpu(eps), "conditions": recs, "summary": summary},
             os.path.join(a.out, "push.pt"))
  _write_json({"meta": m, "conditions": summary}, os.path.join(a.out, "push.json"))
  print(f"[qcbf] wrote {os.path.join(a.out, 'push.json')}", flush=True)
  if vid is not None:
    vid.write(a.video_conditions, recs, m)


def run_badref(ev: Evaluator, a, conds, firsts: dict, lists: dict, lib: dict,
               base: SonicReference, out_by_ref, clear_by_ref, cats: list, meta: dict) -> None:
  dev = ev.dev
  win = _steps(a.badref_window_s)
  dur = _steps(a.duration_s)
  robot = ev.robot
  lim = robot.data.soft_joint_pos_limits[0, ev.tr.to_sonic]
  # teleport lands in held-out clips only: the frames out_by_ref has K flags for
  factory = br.BadReferenceFactory(base, lim[:, 0], lim[:, 1],
                                   teleport_clips=torch.unique(lib["clip"]))
  dv_edges = a.dv_bins if a.badref_push else None
  scen_out, scen_json, pooled_recs, pooled_eps = {}, {}, {c.name: [] for c in conds}, []
  for i, name in enumerate(a.scenarios):
    key = f"badref/{name}"
    sc = br.SCENARIOS[name]
    eps = episode_columns(lists[key], win, lib, base, out_by_ref, clear_by_ref, dev)
    L = br.episode_length(eps["on"], win)
    gen = torch.Generator(device=dev).manual_seed(_seed(a.seed, key + "/corruption"))
    t0 = time.time()
    bad = factory.build(eps["clip"], eps["frame0"], eps["on"], L, name, generator=gen,
                        duration=dur)
    eps = bad_columns(eps, bad, win, base, out_by_ref)
    print(f"[qcbf] badref {i + 1}/{len(a.scenarios)} {name} ({sc.family}, {sc.after}): "
          f"{int(eps['on'].numel())} episodes, {L} frames each, reference built in "
          f"{time.time() - t0:.1f}s; corrupted reference out of K "
          f"{float(eps['bad_out_k'].float().mean()):.3f}", flush=True)
    recs, runs, summary, ref = {}, {}, {}, None
    for c in conds:
      rec = ev.run(key, ev.controller(c), firsts[key], eps, bad, win, push=a.badref_push,
                   true_rows=bad.corruption.true_rows)
      runs[c.name] = rec.pop("_run")
      recs[c.name] = rec
      if c.name == ql.REFERENCE:
        ref = rec["survived"]
      s = ql.summarize_condition(rec, _cpu(eps), c, None if c.name == ql.REFERENCE else ref,
                                 cats, dv_edges)
      summary[c.name] = s
      pooled_recs[c.name].append(rec)
      print(f"[qcbf] badref {name} {c.name}: {ql.headline(s)} ({runs[c.name]['wall_s']:.0f}s)",
            flush=True)
    cor = bad.corruption
    fam_eps = dict(_cpu(eps), family=torch.full((int(eps["on"].numel()),),
                                                list(br.FAMILIES).index(sc.family)))
    pooled_eps.append(fam_eps)
    scen_out[name] = {"episodes": _cpu(eps), "conditions": recs, "summary": summary, "runs": runs,
                      "corruption": {"onset": cor.onset.cpu(), "end": cor.end.cpu(),
                                     "tilt_max": cor.tilt_max.cpu(),
                                     "draws": {k: v.cpu() for k, v in cor.draws.items()
                                               if torch.is_tensor(v)}}}
    scen_json[name] = {"family": sc.family, "after": sc.after, "doc": sc.doc,
                       "episodes": int(eps["on"].numel()), "runs": runs, "conditions": summary}
    _write_json({"meta": dict(meta, suite="badref", complete=False), "scenarios": scen_json},
                os.path.join(a.out, "badref.partial.json"))
    del bad
  # pooled over the scenarios, and per family
  all_eps = ql.concat_records(pooled_eps)
  pooled, by_family = {}, {}
  ref_all = (ql.concat_records(pooled_recs[ql.REFERENCE])["survived"]
             if ql.REFERENCE in pooled_recs else None)
  for c in conds:
    rec = ql.concat_records(pooled_recs[c.name])
    ref = None if c.name == ql.REFERENCE else ref_all
    pooled[c.name] = ql.summarize_condition(rec, all_eps, c, ref, cats, dv_edges)
    for fi, fam in enumerate(br.FAMILIES):
      m = all_eps["family"] == fi
      if not bool(m.any()):
        continue
      sub = {k: v[m] for k, v in rec.items()}
      sub_eps = {k: v[m] for k, v in all_eps.items()}
      by_family.setdefault(fam, {})[c.name] = ql.summarize_condition(
        sub, sub_eps, c, None if ref is None else ref[m], cats, None)
  m = dict(meta, suite="badref", complete=True, conditions=[c.name for c in conds],
           scenarios=list(a.scenarios), episodes_per_scenario=a.badref_episodes,
           onset_s=list(a.badref_onset_s), window_s=a.badref_window_s, duration_s=a.duration_s,
           push=bool(a.badref_push), dv_range=list(a.dv_range),
           split=f"onset - {br.look_ahead() + 1} frames: the first step whose look-ahead "
                 "window holds a corrupted frame (or its velocity)")
  torch.save({"meta": m, "scenarios": scen_out, "pooled": pooled, "by_family": by_family},
             os.path.join(a.out, "badref.pt"))
  _write_json({"meta": m, "scenarios": scen_json, "pooled": pooled, "by_family": by_family},
              os.path.join(a.out, "badref.json"))
  part = os.path.join(a.out, "badref.partial.json")
  if os.path.exists(part):
    os.remove(part)
  print(f"[qcbf] wrote {os.path.join(a.out, 'badref.json')}", flush=True)


# --- main ----------------------------------------------------------------------------------

def main(argv=None):
  a = _parse(argv)
  torch.manual_seed(a.seed)
  dev = a.device
  t_start = time.time()
  os.makedirs(a.out, exist_ok=True)
  lib = load_bank(a.library, "cpu")
  if not bool((lib["next_ok"] > 0).any()):
    raise SystemExit(f"{a.library} is not a clip library (no next_ok)")
  lm = lib["meta"]
  cats = list(lm.get("categories", []))
  # A library built on another machine names that machine's CSV directory; the
  # clip CSVs also live in this checkout's data/lafan1_g1.
  csv_dir = lm["csv_dir"]
  if not os.path.isdir(csv_dir):
    local = os.path.join(_ZOO, "data", "lafan1_g1")
    print(f"[qcbf] the library's csv_dir {csv_dir} does not exist here; "
          f"reading the clip CSVs from {local}")
    csv_dir = local
  base = SonicReference.build(csv_dir, lm["clips"], dev, library=lib)
  la = br.look_ahead()

  # --- the episode lists (before the env: they are its dataset)
  allowed = torch.ones_like(lib["start_ok"], dtype=torch.bool)
  if a.exclude_categories:
    unknown = [c for c in a.exclude_categories if c not in cats]
    if unknown:
      raise SystemExit(f"[qcbf] --exclude-categories {unknown}: the library's categories are {cats}")
    ex = torch.as_tensor([cats.index(c) for c in a.exclude_categories],
                         dtype=lib["category"].dtype)
    allowed = ~torch.isin(lib["category"], ex)
  pool = (lib["start_ok"] & (lib["kind"] == KIND_MOCAP) & allowed).nonzero(
    as_tuple=False).squeeze(-1)
  if a.exclude_categories:
    print(f"[qcbf] spawn pool without {a.exclude_categories}: {int(pool.numel())} rows", flush=True)
  room = lib["next_ok"][pool]
  lists = {}
  if "push" in a.suites:
    gen = torch.Generator().manual_seed(_seed(a.seed, "push"))
    lists["push"] = ql.draw_episodes(pool, room, a.episodes, gen, DT, a.onset_s,
                                     _steps(a.window_s), la, a.dv_range)
  if "badref" in a.suites:
    win_b, dur = _steps(a.badref_window_s), _steps(a.duration_s)
    # fits() reads indices only; the real factory (with the env's joint
    # limits) builds the references later
    pick = br.BadReferenceFactory(base, torch.full((29,), -math.inf), torch.full((29,), math.inf))
    lib_clip, lib_frame = lib["clip"].to(dev), lib["frame"].to(dev)
    for name in a.scenarios:
      key = f"badref/{name}"
      gen = torch.Generator().manual_seed(_seed(a.seed, key))

      def fits(rows, on, name=name):
        # the clip length the build will use (the onsets are fixed by now)
        return pick.fits(lib_clip[rows.to(dev)], lib_frame[rows.to(dev)], on.to(dev),
                         br.episode_length(on, win_b), name, duration=dur).cpu()

      lists[key] = ql.draw_episodes(pool, room, a.badref_episodes, gen, DT, a.badref_onset_s,
                                    win_b, la, a.dv_range, fits=fits)
  bank_path = os.path.join(a.out, "episodes_bank.pt")
  firsts = write_episode_bank(lib, lists, bank_path)
  total = sum(int(v["row"].numel()) for v in lists.values())
  print(f"[qcbf] {len(lists)} episode lists, {total} episodes -> {bank_path}")

  # --- the env, on that bank, walked in order
  weights = {s: 0.0 for s in SOURCES}
  weights["mocap"] = 1.0
  longest = max([a.onset_s[1] + a.window_s] * ("push" in a.suites)
                + [a.badref_onset_s[1] + a.badref_window_s] * ("badref" in a.suites))
  overrides = dict(dataset=bank_path, sampler="sequential", sequential_repeats=1,
                   weights=weights, episode_length_s=float(longest + 1.0), rest_terminate=False)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  video = a.video_episodes is not None

  def cfg_fn(cfg):
    if a.no_obs_noise:
      cfg.observations["actor"].enable_corruption = False
    if video:
      import filter_video as fv
      fv.video_cfg(cfg, *a.video_size)

  env = make_env(a, overrides, push_kw, render_mode="rgb_array" if video else None,
                 cfg_fn=cfg_fn if (a.no_obs_noise or video) else None)
  mj = env.mj
  if abs(float(mj.step_dt) - DT) > 1e-9:
    raise SystemExit(f"[qcbf] control step {mj.step_dt} s: SONIC's references need {DT} s")
  if abs(float(env.action_bound) - FIT_ACTION_BOUND) > 1e-9:
    print(f"[qcbf] WARNING: the env's action box is +-{env.action_bound}, the value fits' "
          f"last actions were clamped to +-{FIT_ACTION_BOUND}")
  if int(env.action_space.shape[0]) != ACTION_DIM:
    raise SystemExit(f"[qcbf] the env's action is {env.action_space.shape[0]}-D, not {ACTION_DIM}")
  need = max(int(v["on"].max()) for v in lists.values()) + max(
    [_steps(a.window_s)] * ("push" in a.suites) + [_steps(a.badref_window_s)] * ("badref" in a.suites))
  if int(mj.max_episode_length) < need + 2:
    raise SystemExit(f"[qcbf] the env caps episodes at {mj.max_episode_length} steps, "
                     f"an episode needs {need + 2}")
  spawn = _spawn_term(mj)
  seq = getattr(spawn, "_seq_rows", None)
  if (spawn.sampler != "sequential" or seq is None or int(seq.numel()) != total
          or not bool((seq.cpu() == torch.arange(total)).all())):
    raise SystemExit("[qcbf] the spawn does not walk the episode bank row by row "
                     "(sequential sampler over every row)")
  policy = SonicPolicy.from_onnx(a.sonic_dir).to(dev).eval()
  tracker = SonicTracker(env, policy, base)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  # only the suites that run: a push-only run needs no fits for the badref defaults
  variants = sorted({c.variant for c in list(a.push_conds) * ("push" in a.suites)
                     + list(a.badref_conds) * ("badref" in a.suites) if c.needs_values})
  values = {}
  for v in variants:
    pv = PolicyValues(a.tag, v, a.value_dir, dev)
    for kind in ("v", "q"):
      rep = pv.info[kind].get("report")
      t = rep.get("tag") if isinstance(rep, dict) else None
      if t is not None and t != a.tag:
        raise SystemExit(f"[qcbf] {pv.info[kind]['path']} was fit for {t!r}, not {a.tag!r}")
    values[v] = pv
    print(f"[qcbf] values {pv}: eps V {pv.eps['v']}, Q {pv.eps['q']}")
  t1 = time.time()
  clearance = reference_clearance(env, lib)
  ref_out = reference_out_k(lib, clearance)
  F = int(base.joint_pos.shape[0])
  lib_ref = base.rows(lib["clip"].to(dev), lib["frame"].to(dev))
  out_by_ref = torch.zeros(F, dtype=torch.bool, device=dev)
  out_by_ref[lib_ref] = ref_out
  clear_by_ref = torch.full((F,), math.inf, device=dev)
  clear_by_ref[lib_ref] = clearance
  print(f"[qcbf] {a.library}: library rows outside K by themselves "
        f"{float(ref_out.float().mean()):.3f} ({time.time() - t1:.0f}s); policy {a.policy} "
        f"(tag {a.tag}); hip {a.hip}; SONIC history {a.sonic_history}; "
        f"setup {time.time() - t_start:.0f}s", flush=True)

  ev = Evaluator(env, tracker, base, model, norm, values, spawn, a)
  rollout_meta = None
  if any(c.kind == "rollout" for c in list(a.push_conds) * ("push" in a.suites)
         + list(a.badref_conds) * ("badref" in a.suites)):
    rollout_meta = ev.setup_rollouts()
  wm_meta = None
  wm_kinds = WM_KINDS
  if any(c.kind in wm_kinds for c in list(a.push_conds) * ("push" in a.suites)
         + list(a.badref_conds) * ("badref" in a.suites)):
    wm_meta = ev.setup_world_model()
  if any(c.kind in ("wmtube", "wmmin") for c in a.push_conds):
    if a.tube_calib is None:
      raise SystemExit("[qcbf] wmtube / wmmin conditions need --tube-calib")
    with open(a.tube_calib) as f:
      ev.tube_calib = json.load(f)
    if ev.tube_calib.get("tag") != a.tag:
      raise SystemExit(f"[qcbf] {a.tube_calib} was calibrated for {ev.tube_calib.get('tag')!r}, "
                       f"not {a.tag!r}")
    wm_meta = dict(wm_meta, tube_calib=os.path.abspath(a.tube_calib))
  meta = {"source": "eval_sonic_qcbf.py", "policy": os.path.abspath(a.policy), "tag": a.tag,
          "task": a.task, "hip": a.hip, "library": os.path.abspath(a.library),
          "sonic_dir": os.path.abspath(a.sonic_dir), "value_dir": os.path.abspath(a.value_dir),
          "values": {v: {"gamma": pv.gamma, "v": pv.info["v"]["path"], "q": pv.info["q"]["path"],
                         "eps": pv.eps} for v, pv in values.items()},
          "sonic_history": a.sonic_history, "num_envs": env.num_envs, "seed": a.seed,
          "quick": bool(a.quick), "dt": DT, "look_ahead": la, "categories": cats,
          "phases": list(ql.PHASES), "status": list(ql.STATUS),
          # every solver setting (kappa and the solver are per condition)
          "qcbf": dict({k: v for k, v in QCBFDescentIntervention(
            1, "cpu", margin=a.qcbf_margin, n_iter=a.qcbf_iters, best_effort=a.best_effort,
            max_dev=a.qcbf_max_dev).settings().items() if k not in ("kappa", "solver")},
            action_box="policy_value_filter.sonic_action_box: SONIC's raw +-ACTION_CLIP"),
          "pi_obs_clamp": a.pi_obs_clamp,
          "pi_obs_columns": (None if ev.pi_slice is None
                             else [ev.pi_slice.start, ev.pi_slice.stop]),
          "hip_actuator": {"sonic": "7520-22, Kp 99.1, 139 N m", "mjlab": "7520-14, Kp 40.2, 88 N m",
                           "fallbacks_and_values_made_on": "mjlab"},
          "k_flags": "ref_out_k / bad_out_k: the harvester's K rule (non-foot clearance < 0, "
                     f"pelvis < {k_set.H_MIN} m, tilt > {k_set.THETA_MAX} rad), not the task's "
                     "terminations (fell_over 70 deg, illegal_contact)",
          "lrsf_eps": a.lrsf_eps, "rollout": rollout_meta, "wmrollout": wm_meta,
          "obs_noise": not a.no_obs_noise, "exclude_categories": list(a.exclude_categories),
          "episode_bank": os.path.abspath(bank_path),
          "common_random_numbers": "every condition runs the same episodes (spawn rows, "
          "onsets, pushes, corruptions) in the same env slots (startup domain randomization) "
          "with the same spawn yaw; per-step observation noise after a mid-wave reset and the "
          "simulator's float determinism are not pinned",
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
  if "push" in a.suites:
    eps = episode_columns(lists["push"], _steps(a.window_s), lib, base, out_by_ref,
                          clear_by_ref, dev)
    print(f"[qcbf] push suite: {a.episodes} episodes x {len(a.push_conds)} conditions; "
          f"reference out of K {float(eps['ref_out_k'].float().mean()):.3f}", flush=True)
    run_push(ev, a, a.push_conds, firsts["push"], eps, base, cats, meta,
             clips=list(lm.get("clips", [])))
  if "badref" in a.suites:
    print(f"[qcbf] bad-reference suite: {len(a.scenarios)} scenarios x {a.badref_episodes} "
          f"episodes x {len(a.badref_conds)} conditions", flush=True)
    run_badref(ev, a, a.badref_conds, firsts, lists, lib, base, out_by_ref, clear_by_ref,
               cats, meta)
  print(f"[qcbf] done in {time.time() - t_start:.0f}s -> {a.out}")
  ev.close()
  env.close()


if __name__ == "__main__":
  main()
