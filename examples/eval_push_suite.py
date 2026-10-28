"""The push suite: E1's push-range calibration and the frozen eval of every arm.

    # E1: calibrate on the current policies; writes the frozen grid file
    python examples/eval_push_suite.py --calibrate \
        --policy push_fast_privcrit=runs/g1_recover_push_fast_privcrit/g1_recover_push_fast/final_model.zip \
        --policy push_rear=runs/g1_recover_push_rear/g1_recover_push_rear/final_model.zip \
        --policy push_fast_cont=runs/g1_recover_push_fast_cont/g1_recover_push_fast/final_model.zip \
        --out runs/lafan_recovery/E1_calibration

    # every later experiment: the frozen grid on one trained arm
    python examples/eval_push_suite.py --frozen \
        --policy S3_s0=runs/lafan_recovery/E3/S3_s0/g1_recover_lafan/final_model.zip \
        --out runs/lafan_recovery/E3/S3_s0/eval_frozen

    # compare arms: merge evaluated runs (NAME=DIR renames a one-policy eval)
    python examples/eval_push_suite.py --merge S3_s0=runs/.../eval_frozen \
        S3_s1=runs/.../eval_frozen --out runs/lafan_recovery/E3/report

    # smoke: tiny grid, 256 envs, a 4 s window
    python examples/eval_push_suite.py --quick --policy p=<zip> --out runs/smoke/suite

The pure half (cells, records, F-levels, the summary and the frozen-grid
proposal) is ``robot_safety_sandbox/envs/g1_recover/push_suite.py``; its docstring
defines the settings, cells and record columns. This script fills the records
in the simulator.

PROTOCOL. One env per setting, ``g1_recover_lafan`` (the reference task's
cfg and head) with the rest truncation OFF, an episode cap just past the
window, and the task's random pushes switched off (``max_pushes=0``): the
only push is the SCRIPTED one, ``RandomPush.schedule`` with the cell's
absolute force (J / (steps dt)), duration and heading-frame direction,
issued for every env right after it spawns (after ``env.reset()`` for all,
then after every step for the envs whose episode ended: mjlab has already
respawned them inside the step, so the schedule replaces whatever the
bridge armed for the new episode).

    std       weights {nominal: 1}; push ``--onset-s`` after the spawn
    lafan     the held-out stride bank (``--lafan-bank``), weights {mocap: 1},
              sampler uniform; push ``--onset-s`` after the spawn
    handover  the held-out clip LIBRARY (``--library``), weights {mocap: 1},
              sampler uniform; the clip is replayed open-loop from the spawn
              frame (``PlaybackDriver``), the push lands U(``--handover-onset-s``)
              into the replay and the policy takes over ``delay`` s after
              the onset -- the push keeps acting through the takeover if it
              has not ended. Library rows without ``onset + delay`` frames left
              in their clip are marked not spawnable (a filtered copy of the
              library under ``--out``), so every replay stays inside its clip.
    walker    (opt-in) the walker bank (``--walker-bank``) at the push_rear
              mixture, ``levels 0`` / ``band_width 1`` so every source
              samples all its rows whatever the curriculum does; pushed like
              std
    pre       (opt-in) collected handover states (``--pre-bank``), weights
              {handover: 1}, sampler uniform, NO scripted push: the bridge
              replays each row's pending push. Run with the same grid as the
              others, it issues as many episodes as an E-lafan run

The policy acts deterministically (``policy._predict``, clamped to the action
box) with the checkpoint's own obs normalizer (``load_twin``). Its value at
the takeover state is read with ``labels.PolicyValue`` (the privileged
critic's obs and normalizer where it has one).

THE WINDOW. A record opens at the spawn and closes ``--eval-s`` after the
takeover (the spawn for std / lafan): survived, or at the fall (a failure
termination), or -- handover only -- at a fall of the REPLAY before the
takeover (``prefix_fell``; not the policy's, left out of survival). The
episode cap is set just past the longest window, so an env whose record has
closed simply runs on, unrecorded, until its cap (at most the longest
replay prefix + 0.1 s): simpler than force-resetting it from outside the
bridge, and no record is cut short. Episodes are issued to cells round-robin
in spawn order (``push_suite.assign_cells``) until ``episodes_per_cell x
cells`` have been issued, and every issued episode is recorded exactly once
whatever its outcome, so slow-to-fail cells are not under-sampled. Each
policy runs the same cells from the same seed, and the handover push onset
is a fixed table over the issue order, so the k-th episode gets the same
cell and onset under every policy (the spawn row itself is drawn by the
spawn's RNG and diverges once outcomes differ).

THE STANCE COLUMNS. For the tight (0.25 m/s, 0.30 rad/s) and loose (0.5 m/s,
0.6 rad/s) targets, from the robot data with ``targets.make_l_stance(...,
gate_on_push=False)``, counted from the later of the push END and the
takeover: time to the first step inside, the share of the later steps
spent inside, and whether the robot fell after reaching. For E5's staying
check (3 s after the first reach) run with ``--eval-s 6`` or more.

OUTPUTS (``--out``): ``episodes.pt`` (``{"records": columnar records,
"meta": ...}``), ``summary.json`` (``push_suite.summarize``; NaN -> null),
``report.txt`` (the printed tables), ``grid.json`` (the resolved grid);
``--calibrate`` also writes ``--grid-file`` (the frozen grid E1 proposes:
``push_suite.propose_frozen_grid``).
"""

from __future__ import annotations

import argparse
import datetime
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

from robot_safety_sandbox.envs.g1_recover.push_suite import (  # noqa: E402
  ALL_SETTINGS, CALIBRATION_DURATIONS, CALIBRATION_IMPULSES, HANDOVER_DELAYS,
  N_DIRECTIONS, PRE_IMPULSE_BIN, SETTINGS, TOLERANCES, assign_cells,
  build_cells, cell_table, compare_arms, concat_records, direction_label,
  impulse_to_dv, propose_frozen_grid, push_steps, summarize)

DEFAULT_GRID_FILE = os.path.join("configs", "sweeps", "lafan_recovery", "eval_grid.yaml")
#: E-handover push onset into the replay, s (uniform). With the 0.2 s maximum
#: takeover delay the open-loop replay lasts <= 0.3 s: longer, its drift (no
#: balance feedback) rather than the push makes the takeover state hard
#: (E1, 2026-10-23: 0.27 survival at zero push with up to 0.4 s of replay vs
#: 0.49 from the same clips' frames).
HANDOVER_ONSET_S = (0.02, 0.1)
#: --quick: a grid small enough to finish in a few windows on 256 envs.
QUICK = dict(impulses=[0.0, 20.0, 40.0], durations=[0.2], directions=4,
             handover_delays=[0.0, 0.2], episodes_per_cell=8, num_envs=256)
DEFAULTS = dict(impulses=list(CALIBRATION_IMPULSES),
                durations=list(CALIBRATION_DURATIONS), directions=N_DIRECTIONS,
                handover_delays=list(HANDOVER_DELAYS), episodes_per_cell=64,
                num_envs=4096, eval_s=4.0, onset_s=0.1)
#: The suite's push process: the reference task's timing with every random
#: onset switched off; the scripted push is the only one.
SUITE_RANDOM_PUSH = dict(interval_s=(1.0, 2.0), duration_s=0.2,
                         first_delay_s=(0.1, 0.4), horizontal=True, max_pushes=0)
#: Episode cap slack past the longest window, s.
CAP_SLACK_S = 0.1
#: The walker setting's mixture (the push_rear / push_fast walker weights).
WALKER_WEIGHTS = {"nominal": 0.10, "clean": 0.20, "pushed": 0.25, "fell": 0.30,
                  "synthetic": 0.15}
#: Row category names by bank ``kind`` (states.KIND_*), for banks without one.
KIND_NAMES = ("clean", "pushed", "synthetic", "mocap", "handover")


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", action="append", default=[], metavar="NAME=PATH",
                 help="a safety_sb3 checkpoint to evaluate (repeatable; its obs "
                      "stats are found next to it). The first one is the policy "
                      "--calibrate reads the frozen grid off.")
  p.add_argument("--setting", action="append", choices=ALL_SETTINGS, default=None,
                 help="std / lafan / handover (repeatable; default all three), "
                      "or the opt-in walker / pre")
  p.add_argument("--task", default="g1_recover_lafan",
                 help="the env the suite runs in (a random_push recover task)")
  p.add_argument("--lafan-bank", default=os.path.join(
    "data", "g1_recover_states_lafan1_test.pt"), help="E-lafan start states")
  p.add_argument("--library", default=os.path.join("data", "g1_lafan1_clips_test.pt"),
                 help="E-handover clip library (every 50 fps frame, next_ok column)")
  p.add_argument("--walker-bank", default=os.path.join("data", "g1_recover_states.pt"),
                 help="walker setting: the walker start-state bank")
  p.add_argument("--pre-bank", default=os.path.join("data", "g1_handover_test.pt"),
                 help="pre setting: held-out collected handover states")
  p.add_argument("--impulses", type=float, nargs="+", default=None,
                 help="push impulses, N s (default: the calibration grid 0, 4, ..., 60)")
  p.add_argument("--durations", type=float, nargs="+", default=None,
                 help="push durations, s (default 0.1 0.2 0.3)")
  p.add_argument("--directions", type=int, default=None,
                 help="evenly spaced push directions (default 8; 0 = from behind)")
  p.add_argument("--handover-delays", type=float, nargs="+", default=None,
                 help="handover: takeover delay after the push onset, s "
                      "(default 0 0.1 0.2)")
  p.add_argument("--handover-onset-s", type=float, nargs=2, default=None,
                 metavar=("LO", "HI"),
                 help="handover: push onset into the replay, uniform, s (default "
                      "the grid file's, else 0.02 0.1: the replay drifts without "
                      "feedback, so it is kept to <= 0.3 s with the 0.2 s delay)")
  p.add_argument("--onset-s", type=float, default=None,
                 help="std / lafan: push onset after the spawn, s (default 0.1)")
  p.add_argument("--episodes-per-cell", type=int, default=None,
                 help="episodes per cell per setting per policy (default 64)")
  p.add_argument("--eval-s", type=float, default=None,
                 help="window after the takeover, s (default 4.0)")
  p.add_argument("--reach-s", type=float, default=3.0,
                 help="the reach-within budget of the behaviour stats, s")
  p.add_argument("--num-envs", type=int, default=None, help="default 4096")
  p.add_argument("--grid-file", default=DEFAULT_GRID_FILE,
                 help="the frozen grid (--frozen reads it, --calibrate writes it)")
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--frozen", action="store_true",
                    help="evaluate on the grid in --grid-file (E1's output)")
  mode.add_argument("--calibrate", action="store_true",
                    help="E1: after the run, propose the frozen grid from the "
                         "first policy's pooled E-lafan curve and write --grid-file")
  mode.add_argument("--merge", nargs="+", default=None, metavar="[NAME=]DIR",
                    help="no simulation: summarize the episodes.pt of earlier "
                         "evals together (NAME= renames a one-policy eval)")
  p.add_argument("--quick", action="store_true",
                 help="smoke: a tiny grid on 256 envs (explicit flags still win)")
  p.add_argument("--no-values", action="store_true",
                 help="do not read the policy's value at the takeover")
  p.add_argument("--levels", type=int, default=5,
                 help="--calibrate: impulse levels of the frozen grid")
  p.add_argument("--out", required=True, help="output directory")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  return p.parse_args(argv)


def _named(items, what):
  out = []
  for item in items:
    if "=" not in item:
      raise SystemExit(f"--{what} expects NAME=PATH, got {item!r}")
    name, path = item.split("=", 1)
    out.append((name.strip(), path.strip()))
  names = [n for n, _ in out]
  if len(set(names)) != len(names):
    raise SystemExit(f"--{what}: names must be unique, got {names}")
  return out


def resolve_grid(args) -> dict:
  """defaults < --frozen grid file < --quick < explicit flags."""
  g = dict(DEFAULTS)
  if args.frozen:
    import yaml
    if not os.path.exists(args.grid_file):
      raise SystemExit(f"--frozen: no grid file at {args.grid_file}; run E1 with "
                       f"--calibrate first (or pass --grid-file)")
    with open(args.grid_file) as f:
      frozen = yaml.safe_load(f) or {}
    for k in ("impulses", "durations", "directions", "handover_delays",
              "episodes_per_cell", "eval_s", "onset_s", "handover_onset_s"):
      if frozen.get(k) is not None:
        g[k] = frozen[k]
    g["frozen_from"] = os.path.abspath(args.grid_file)
  if args.quick:
    g.update(QUICK)
  for k in ("impulses", "durations", "directions", "handover_delays",
            "episodes_per_cell", "eval_s", "onset_s", "num_envs"):
    v = getattr(args, k)
    if v is not None:
      g[k] = v
  g["impulses"] = [float(x) for x in g["impulses"]]
  g["durations"] = [float(x) for x in g["durations"]]
  g["handover_delays"] = [float(x) for x in g["handover_delays"]]
  g["directions"] = int(g["directions"])
  if args.handover_onset_s is not None:
    g["handover_onset_s"] = args.handover_onset_s
  g["handover_onset_s"] = [float(x) for x in g.get("handover_onset_s",
                                                   HANDOVER_ONSET_S)]
  g["reach_s"] = float(args.reach_s)
  return g


# --- the simulator half --------------------------------------------------------

def _weights(source: str) -> dict:
  """Every source of the spawn at 0 but ``source`` (the spawn fills missing
  keys from its DEFAULT_WEIGHTS, so all of them are spelled out)."""
  from robot_safety_sandbox.envs.g1_recover.spawn import SOURCES
  w = {s: 0.0 for s in SOURCES}
  w[source] = 1.0
  return w


def _filtered_library(path: str, need: int, out_dir: str) -> str:
  """A copy of the library with ``start_ok`` False on every row with fewer
  than ``need`` frames left in its clip (rows, order and every column kept,
  so spawn rows index the original). Returns the copy's path."""
  raw = torch.load(path, map_location="cpu", weights_only=False)
  if "next_ok" not in raw:
    raise SystemExit(f"{path} has no next_ok column: not a clip library (build it "
                     f"with examples/build_g1_recover_mocap_states.py --library)")
  n = int(raw["next_ok"].shape[0])
  ok = raw.get("start_ok", torch.ones(n, dtype=torch.bool)).bool()
  raw["start_ok"] = ok & (raw["next_ok"].long() >= int(need))
  print(f"[suite] handover library {path}: {int(raw['start_ok'].sum())}/{n} rows "
        f"have >= {need} frames left in their clip")
  os.makedirs(out_dir, exist_ok=True)
  dst = os.path.join(out_dir, "_handover_library.pt")
  torch.save(raw, dst)
  return dst


def _row_names(bank: dict, setting: str) -> tuple[list[str], list[str]]:
  """Per-row clip and category names ('' where a row has none); walker rows
  are named by their source (kind, or ``fell``)."""
  from robot_safety_sandbox.envs.g1_recover.labels import row_names
  if setting == "walker":
    kind, fell = bank["kind"].tolist(), bank["fell"].tolist()
    cats = ["fell" if f else (KIND_NAMES[k] if 0 <= k < len(KIND_NAMES) else "-")
            for k, f in zip(kind, fell)]
    return [""] * len(cats), cats
  return (row_names(bank, "clip", "clips"),
          row_names(bank, "category", "categories", fallback=KIND_NAMES))


def suite_cells(setting: str, grid: dict) -> tuple[list[dict], int]:
  """A setting's cells and the episodes it issues. ``pre`` has no scripted
  push, so ONE impulse-0 cell carries as many episodes as an E-lafan run."""
  n_lafan = (len(grid["impulses"]) * len(grid["durations"]) * grid["directions"]
             * int(grid["episodes_per_cell"]))
  if setting == "pre":
    return build_cells([0.0], grid["durations"][:1], 1), n_lafan
  cells = build_cells(grid["impulses"], grid["durations"], grid["directions"],
                      grid["handover_delays"] if setting == "handover" else (0.0,))
  return cells, len(cells) * int(grid["episodes_per_cell"])


def build_env(setting: str, grid: dict, args, need_critic: bool):
  """The suite env of one setting and its context: the bank (the replay's
  targets, the pending pushes of ``pre``) and the per-row names."""
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.envs.g1_recover.states import load_bank
  dt_guess = 0.02          # the G1 control step; checked against the env below
  prefix_s = (max(grid["handover_onset_s"]) + max(grid["handover_delays"])
              if setting == "handover" else 0.0)
  episode_s = float(grid["eval_s"]) + prefix_s + CAP_SLACK_S
  ov = dict(episode_length_s=episode_s, rest_terminate=False)
  ctx = {"bank": None, "clip": [], "category": []}
  if setting == "std":
    ov.update(weights=_weights("nominal"))
  else:
    src = {"lafan": args.lafan_bank, "handover": args.library,
           "walker": args.walker_bank, "pre": args.pre_bank}[setting]
    if not os.path.exists(src):
      raise SystemExit(f"[suite] {setting}: no bank at {src}; build it first (see "
                       f"docs/experiments/g1-lafan-recovery.md, 'How to run')")
    if setting == "handover":
      need = int(math.ceil(prefix_s / dt_guess)) + 1
      src = _filtered_library(src, need, args.out)
    if setting == "walker":
      w = _weights("nominal")
      w.update(WALKER_WEIGHTS)
      # level 0 with a full-width band: every source samples ALL its rows,
      # whatever the curriculum does, so the starts are policy-independent
      ov.update(dataset=src, weights=w, levels=0, band_width=1.0, init_level=0)
    else:
      ov.update(dataset=src, sampler="uniform",
                weights=_weights("handover" if setting == "pre" else "mocap"))
    bank = load_bank(src, "cpu")       # names + the replay's targets (moved by the driver)
    ctx["bank"] = bank
    ctx["clip"], ctx["category"] = _row_names(bank, setting)
  kw = {"critic_obs_key": "critic"} if need_critic else {}
  env = make_tensor(args.task, int(grid["num_envs"]), args.device, cfg_overrides=ov,
                    random_push=dict(SUITE_RANDOM_PUSH), **kw)
  if abs(float(env.mj.step_dt) - dt_guess) > 1e-9:
    print(f"[suite] WARNING: control dt {env.mj.step_dt} != {dt_guess}; the "
          f"library filter assumed {dt_guess}")
  if getattr(env, "_push", None) is None:
    raise SystemExit(f"[suite] task {args.task!r} has no push process")
  return env, ctx


def run_policy(env, setting, cells, total, grid, pol, ctx, args) -> dict:
  """Run ``total`` episodes of one policy over ``cells``; return their
  columnar records (see push_suite.RECORD_FIELDS)."""
  from robot_safety_sandbox.envs.g1_recover.playback import PlaybackDriver
  from robot_safety_sandbox.envs.g1_recover.targets import make_l_stance
  from robot_safety_sandbox.eval.policies import reset_policy_memory
  name, _path, model, norm, vfn = pol
  reset_policy_memory(model.policy)                # every policy starts fresh
  mj, dev, n = env.mj, env.device, env.num_envs
  dt = float(mj.step_dt)
  push = env._push
  robot = mj.scene["robot"]
  C = len(cells)
  bank = ctx["bank"]
  pre = setting == "pre"
  if pre:
    # the pending push of each collected row: how long it still acts after the
    # spawn (stance counting starts after it), what produced the row
    pend_steps = bank["push_left_steps"].long().to(dev)
    pend_force = bank["push_force_h"].float().norm(dim=1).to(dev)
    row_imp = torch.nan_to_num(bank["push_impulse"].float(), nan=0.0).to(dev)
    row_imp = torch.round(row_imp / PRE_IMPULSE_BIN) * PRE_IMPULSE_BIN
    row_delay = torch.round(torch.nan_to_num(bank["handover_delay_s"].float(),
                                             nan=0.0) * 10.0).to(dev) / 10.0
    # the row's own push duration (the single pre cell's is a placeholder)
    row_dur = bank["push_duration_s"].float().to(dev)
  tab = cell_table(cells, dev)
  T_steps = torch.tensor([push_steps(c["duration"], dt) for c in cells], device=dev)
  force = tab["impulse"] / (T_steps.float() * dt)
  delay_steps = torch.round(tab["delay"] / dt).long()
  eval_steps = int(round(float(grid["eval_s"]) / dt))
  onset_fixed = int(round(float(grid["onset_s"]) / dt))
  handover = setting == "handover"
  # the handover push onset of the k-th issued episode: one table per run, so
  # every policy gets the same onset for the same (episode, cell)
  gen = torch.Generator().manual_seed(int(args.seed) + 7919)
  lo, hi = grid["handover_onset_s"]
  onset_tab = torch.round((lo + (hi - lo) * torch.rand(total, generator=gen)) / dt
                          ).long().clamp_min(1).to(dev)
  driver = PlaybackDriver(bank, n, dev) if handover else None
  stance = {k: make_l_stance(lin, ang, gate_on_push=False)
            for k, (lin, ang) in TOLERANCES.items()}
  low = torch.as_tensor(env.action_space.low, device=dev)
  high = torch.as_tensor(env.action_space.high, device=dev)
  cap = int(mj.max_episode_length)
  longest = int((onset_tab.max() + delay_steps.max()) if handover else 0) + eval_steps
  if longest >= cap:
    raise SystemExit(f"[suite] episode cap {cap} steps < window {longest}: raise "
                     f"CAP_SLACK_S or check the control dt")

  L = dict(dtype=torch.long, device=dev)
  open_ = torch.zeros(n, dtype=torch.bool, device=dev)
  t = torch.zeros(n, **L)
  cell = torch.zeros(n, **L)
  k_on, k_end, k_h, k_close = (torch.zeros(n, **L) for _ in range(4))
  row = torch.full((n,), -1, **L)
  peak = torch.zeros(n, device=dev)
  value0 = torch.full((n,), float("nan"), device=dev)
  pos0, posh, last = (torch.zeros(n, 2, device=dev) for _ in range(3))
  t_first = {k: torch.full((n,), -1, **L) for k in stance}
  n_after = {k: torch.zeros(n, **L) for k in stance}
  n_in = {k: torch.zeros(n, **L) for k in stance}
  parts = []
  issued, recorded = 0, 0
  clip_names, cat_names = ctx["clip"], ctx["category"]

  def per_env(ids, x, dtype):
    """``x`` (one value per env of ``ids``) as an ``[n]`` tensor, 0 elsewhere."""
    return torch.zeros(n, dtype=dtype, device=dev).index_copy_(0, ids, x.to(dtype))

  def begin(mask):
    """Issue the next episodes to the envs that just spawned (ascending ids)."""
    nonlocal issued
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    m = min(int(idx.numel()), total - issued)
    open_[idx[m:]] = False
    if m == 0:
      return
    new = idx[:m]
    order = issued + torch.arange(m, device=dev)
    issued += m
    c = assign_cells(C, int(order[0]), m, dev)
    sp = getattr(mj, "_zoo_spawn_row", None)
    row[new] = sp[new].long() if sp is not None else -1
    if pre:                      # no scripted push: the row's own plays out
      kon, kh = torch.zeros(m, **L), torch.zeros(m, **L)
      kend = pend_steps[row[new].clamp_min(0)]
    else:
      kon = onset_tab[order] if handover else torch.full((m,), onset_fixed, **L)
      kh = kon + delay_steps[c] if handover else torch.zeros(m, **L)
      kend = kon + T_steps[c]
    cell[new], open_[new], t[new] = c, True, 0
    k_on[new], k_end[new], k_h[new] = kon, kend, kh
    k_close[new] = kh + eval_steps
    p0 = robot.data.root_link_pos_w[new, :2]
    pos0[new], posh[new], last[new] = p0, p0, p0
    peak[new] = 0.0
    value0[new] = float("nan")
    for k in stance:
      t_first[k][new] = -1
      n_after[k][new] = 0
      n_in[k][new] = 0
    full = torch.zeros(n, dtype=torch.bool, device=dev)
    full[new] = force[c] > 0.0                   # impulse 0: no push at all
    if bool(full.any()):
      push.schedule(full, force_n=per_env(new, force[c], torch.float32),
                    steps=per_env(new, T_steps[c], torch.long),
                    onset=per_env(new, kon, torch.long),
                    rel=per_env(new, tab["rel"][c], torch.float32))
    if handover:
      start = torch.zeros(n, dtype=torch.bool, device=dev)
      start[new] = True
      if bool((row[new] < 0).any()):
        raise SystemExit("[suite] handover: an env spawned without a library row "
                         "(env._zoo_spawn_row = -1); check the spawn weights")
      driver.start(start, rows=row.clamp_min(0), steps=torch.where(start, k_h, 0))

  def emit(close, fell):
    nonlocal recorded
    idx = close.nonzero(as_tuple=False).squeeze(-1)
    c = cell[idx]
    pfx = fell[idx] & (t[idx] <= k_h[idx])      # the replay fell, not the policy
    pol_fell = fell[idx] & ~pfx
    start_count = torch.maximum(k_end[idx], k_h[idx])
    nan = torch.full((idx.numel(),), float("nan"), device=dev)
    r = row[idx].clamp_min(0)
    rec = {
      "cell": c, "impulse": row_imp[r] if pre else tab["impulse"][c],
      "duration": row_dur[r] if pre else tab["duration"][c],
      "force": pend_force[r] if pre else force[c],
      "dir_idx": tab["dir_idx"][c],
      "n_dirs": torch.full((idx.numel(),), 1 if pre else int(grid["directions"]), **L),
      "delay": row_delay[r] if pre else tab["delay"][c],
      "onset_s": k_on[idx].float() * dt,
      "handover_s": k_h[idx].float() * dt, "row": row[idx],
      "survived": ~fell[idx], "prefix_fell": pfx,
      "fall_t": torch.where(pol_fell, (t[idx] - k_h[idx]).float() * dt, nan),
      "peak_speed": peak[idx],
      "dist": torch.where(pfx, nan, (last[idx] - posh[idx]).norm(dim=1)),
      "dist_spawn": (last[idx] - pos0[idx]).norm(dim=1),
      "value0": value0[idx],
    }
    for k in stance:
      tf = t_first[k][idx]
      rec[f"t_reach_{k}"] = torch.where(tf >= 0, (tf - start_count).float() * dt, nan)
      na = n_after[k][idx]
      rec[f"stay_frac_{k}"] = torch.where(
        na > 0, n_in[k][idx].float() / na.clamp_min(1).float(), nan)
      rec[f"fell_after_reach_{k}"] = (tf >= 0) & pol_fell
    rec = {k: v.detach().cpu() for k, v in rec.items()}
    rows = rec["row"].tolist()
    rec["clip"] = [clip_names[r] if 0 <= r < len(clip_names) else "" for r in rows]
    rec["category"] = [cat_names[r] if 0 <= r < len(cat_names) else "" for r in rows]
    rec["policy"] = [name] * idx.numel()
    rec["setting"] = [setting] * idx.numel()
    parts.append(rec)
    recorded += int(idx.numel())

  torch.manual_seed(int(args.seed))
  obs = env.reset()
  every = torch.ones(n, dtype=torch.bool, device=dev)
  if driver is not None:
    driver.stop(every)
  begin(every)
  max_iters = (total // n + 3) * (cap + 1) * 2
  t0, it = time.time(), 0
  while recorded < total:
    d = robot.data
    pos = d.root_link_pos_w[:, :2]
    speed = d.root_link_lin_vel_w[:, :2].norm(dim=1)
    take = open_ & (t == k_h)
    if bool(take.any()):
      posh[take] = pos[take]
      if vfn is not None:
        v = vfn(obs, env.critic_obs() if vfn.asymmetric else None)
        value0[take] = v[take]
    peak = torch.where(open_ & (t >= k_on), torch.maximum(peak, speed), peak)
    counting = open_ & (t >= torch.maximum(k_end, k_h))
    if bool(counting.any()):
      for k, fn in stance.items():
        inside = fn(mj).reshape(-1) >= 0.0
        first = counting & inside & (t_first[k] < 0)
        t_first[k] = torch.where(first, t, t_first[k])
        after = counting & (t_first[k] >= 0) & ~first
        n_after[k] += after.long()
        n_in[k] += (after & inside).long()
    last = torch.where(open_.unsqueeze(1), pos, last)
    if driver is not None:
      # the policy's memory starts at the handover: hold it fresh while the
      # replay still drives the env (a no-op for a memoryless policy)
      reset_policy_memory(model.policy, driver.active)
    with torch.no_grad():
      act = torch.clamp(model.policy._predict(norm(obs), deterministic=True), low, high)
    if driver is not None:
      act = driver.actions(act)
    obs, _g, dones, timeouts, _l = env.step_tensor(act)
    reset_policy_memory(model.policy, dones)
    if driver is not None:
      driver.advance()
    t += 1
    term = dones & ~timeouts
    fell = open_ & term
    cut = open_ & timeouts & ~term & (t < k_close)
    if bool(cut.any()):
      raise SystemExit("[suite] an episode was truncated before its window closed "
                       "(rest_terminate on, or the cap too short)")
    close = fell | (open_ & ~term & (t >= k_close))
    if bool(close.any()):
      emit(close, fell)
      open_ &= ~close
    if bool(dones.any()):
      if driver is not None:
        driver.stop(dones)
      begin(dones)
    it += 1
    if it % 500 == 0:
      print(f"[suite] {setting} {name}: step {it}, {recorded}/{total} episodes "
            f"({it / max(time.time() - t0, 1e-6):.0f} steps/s)", flush=True)
    if it > max_iters:
      raise SystemExit(f"[suite] {setting} {name}: no progress after {it} steps "
                       f"({recorded}/{total} episodes)")
  print(f"[suite] {setting} {name}: {recorded} episodes in {it} steps, "
        f"{time.time() - t0:.0f}s", flush=True)
  return concat_records(parts)


# --- reporting -------------------------------------------------------------------

def _fmt(v, w=7, p=2):
  if v is None or (isinstance(v, float) and not math.isfinite(v)):
    return "-".rjust(w)
  return f"{v:{w}.{p}f}"


def _flevel(pair, w=7):
  v, flag = pair
  return _fmt(v, w, 1) if flag == "ok" else flag.rjust(w)


def report_lines(summary: dict) -> list[str]:
  """The printed tables, per setting with one column per policy: survival by
  impulse, the F-levels, F50 by direction / duration / delay / category, and
  the behaviour statistics."""
  out = []
  lw = 28                                         # the label column
  for setting, groups in summary["settings"].items():
    gs = list(groups.values())
    w = max(9, *(len(p) + 2 for p in groups))

    def row(label, cells):
      out.append(f"{label:<{lw}}" + "".join(cells))

    out.append("")
    out.append(f"=== {setting}: survival by impulse (valid episodes; dv = J / m)")
    row("J N s (dv m/s)", (p.rjust(w) for p in groups))
    for lv in sorted({x for g in gs for x in g["curve"]["levels"]}):
      row(f"{lv:g} ({impulse_to_dv(lv):.2f})",
          (_fmt(g["curve"]["surv"][g["curve"]["levels"].index(lv)], w)
           if lv in g["curve"]["levels"] else "-".rjust(w) for g in gs))
    row("episodes (valid)", (str(g["n_valid"]).rjust(w) for g in gs))
    row("survival, pooled", (_fmt(g["survival"], w) for g in gs))
    row("survival, mean over levels", (_fmt(g["mean_surv_levels"], w) for g in gs))
    for k in ("f90", "f50", "f10"):
      row(f"{k.upper()} N s", (_flevel(g["f"][k], w) for g in gs))
    if setting == "handover":
      row("replay fell before takeover", (_fmt(g["prefix_fall_rate"], w) for g in gs))
    for key, title in (("by_dir", "F50 N s by direction (push from)"),
                       ("by_duration", "F50 N s by duration s"),
                       ("by_delay", "F50 N s by takeover delay s"),
                       ("by_category", "F50 N s by clip category")):
      labels = list(dict.fromkeys(lab for g in gs for lab in g.get(key, {})))
      if not labels:
        continue
      out.append(f"--- {title}")
      for lab in labels:
        row(lab, (_flevel(g[key][lab]["f50"], w) if lab in g.get(key, {})
                  else "-".rjust(w) for g in gs))
    out.append(f"--- behaviour (valid episodes; reach counted after the push "
               f"ends; 'within' = {summary['meta']['reach_s']:g} s)")
    keys = ["peak_speed_median", "dist_median"]
    for tol in TOLERANCES:
      keys += [f"{tol}_reach_rate", f"{tol}_reach_rate_within",
               f"{tol}_t_reach_median", f"{tol}_stay_frac_mean",
               f"{tol}_fell_after_reach"]
    for k in keys:
      row(k, (_fmt(g["behaviour"].get(k), w) for g in gs))
    if any("value_auc" in g for g in gs):
      row("AUC(V at takeover, survived)", (_fmt(g.get("value_auc"), w) for g in gs))
  return out


def arm_lines(summary: dict) -> list[str]:
  """Per-arm means and within-arm seed gaps (``push_suite.compare_arms``)."""
  out = []
  metrics = ("mean_surv_levels", "f50", "f50_behind")
  for setting in summary["settings"]:
    arms = compare_arms(summary, setting, metrics)
    if not any(len(a["seeds"]) > 1 for a in arms.values()):
      continue
    out.append("")
    out.append(f"=== {setting}: arms (mean over seeds, gap = max - min across seeds)")
    out.append(f"{'arm':<20}{'seeds':>6}" + "".join(f"{m:>24}" for m in metrics))
    for arm, a in arms.items():
      out.append(f"{arm:<20}{len(a['seeds']):>6}" + "".join(
        f"{_fmt(a[m]['mean'], 9, 3)} +- {_fmt(a[m]['gap'], 8, 3)}  " for m in metrics))
    out.append("a difference counts only if it exceeds the LARGER of the two arms' "
               "gaps (push_suite.significant); otherwise the simpler arm wins")
  return out


def _jsonable(x):
  if isinstance(x, float):
    return x if math.isfinite(x) else None
  if isinstance(x, dict):
    return {str(k): _jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_jsonable(v) for v in x]
  return x


def write_outputs(out_dir: str, summary: dict, lines: list[str], extra_meta: dict):
  os.makedirs(out_dir, exist_ok=True)
  summary["meta"].update(extra_meta)
  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(_jsonable(summary), f, indent=1)
  with open(os.path.join(out_dir, "report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
  print("\n".join(lines))
  print(f"\n[suite] wrote {out_dir}/summary.json and report.txt")


def merge(args):
  parts, missing = [], []
  for item in args.merge:
    name, d = item.split("=", 1) if "=" in item else (None, item)
    path = os.path.join(d, "episodes.pt")
    if not os.path.exists(path):
      # a run still training (or a gated arm never run): report the rest
      missing.append(item)
      print(f"[suite] --merge: no episodes.pt in {d}; left out")
      continue
    rec = torch.load(path, map_location="cpu", weights_only=False)["records"]
    if name is not None:
      pols = sorted(set(rec["policy"]))
      if len(pols) != 1:
        raise SystemExit(f"--merge {item}: {d} holds {len(pols)} policies {pols}; "
                         f"NAME= renames a one-policy eval only")
      rec = dict(rec, policy=[name] * len(rec["policy"]))
    parts.append(rec)
  if not parts:
    raise SystemExit("--merge: none of the given evals has an episodes.pt yet")
  records = concat_records(parts)
  summary = summarize(records, reach_s=args.reach_s)
  lines = report_lines(summary) + arm_lines(summary)
  if missing:
    lines += ["", f"left out (no episodes.pt yet): {' '.join(missing)}"]
  write_outputs(args.out, summary, lines, {"merged": list(args.merge),
                                           "missing": missing})


def main(argv=None):
  args = _parse(argv)
  if args.merge:
    return merge(args)
  if not args.policy:
    raise SystemExit("--policy NAME=PATH is required (repeatable)")
  from robot_safety_sandbox.envs.g1_recover.labels import PolicyValue
  from robot_safety_sandbox.eval.policies import load_twin
  pols_in = _named(args.policy, "policy")
  grid = resolve_grid(args)
  settings = list(dict.fromkeys(args.setting or SETTINGS))
  if args.calibrate and "lafan" not in settings:
    raise SystemExit("--calibrate reads the frozen grid off the lafan setting; "
                     "include --setting lafan")
  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, "grid.json"), "w") as f:
    json.dump(dict(grid, settings=settings, policies=dict(pols_in),
                   seed=args.seed, task=args.task), f, indent=1)
  pols = []
  for name, path in pols_in:
    model, norm = load_twin(path, args.device)
    model.policy.set_training_mode(False)
    vfn = None if args.no_values else PolicyValue(model, path, args.device)
    pols.append((name, path, model, norm, vfn))
  need_critic = any(p[4] is not None and p[4].asymmetric for p in pols)
  print(f"[suite] settings {settings}; impulses {grid['impulses']} N s, durations "
        f"{grid['durations']} s, {grid['directions']} directions, handover delays "
        f"{grid['handover_delays']} s; {grid['episodes_per_cell']} episodes per "
        f"cell; window {grid['eval_s']} s; {grid['num_envs']} envs", flush=True)
  parts = []
  for setting in settings:
    cells, total = suite_cells(setting, grid)
    env, ctx = build_env(setting, grid, args, need_critic)
    print(f"[suite] {setting}: {len(cells)} cells, {total} episodes per policy",
          flush=True)
    for pol in pols:
      parts.append(run_policy(env, setting, cells, total, grid, pol, ctx, args))
    env.close()
    del env
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  records = concat_records(parts)
  torch.save({"records": records, "meta": dict(grid, settings=settings,
                                               policies=dict(pols_in), seed=args.seed,
                                               task=args.task)},
             os.path.join(args.out, "episodes.pt"))
  summary = summarize(records, reach_s=grid["reach_s"])
  lines = report_lines(summary)
  meta = {"grid": grid, "policy_paths": dict(pols_in), "seed": args.seed,
          "task": args.task, "created": datetime.datetime.now().isoformat(timespec="seconds"),
          "directions": {str(i): direction_label(i, grid["directions"])
                         for i in range(grid["directions"])}}
  if args.calibrate:
    prop = propose_frozen_grid(summary, n_levels=args.levels, setting="lafan")
    frozen = {
      "impulses": prop["impulses"], "durations": prop["durations"],
      "directions": grid["directions"], "handover_delays": grid["handover_delays"],
      "episodes_per_cell": grid["episodes_per_cell"], "eval_s": grid["eval_s"],
      "onset_s": grid["onset_s"], "handover_onset_s": grid["handover_onset_s"],
      "force_max_suggested": round(prop["force_max_suggested"], 1),
      "durations_overlap": prop["durations_overlap"],
      "source": prop["source"], "f50_by_duration": prop["f50_by_duration"],
      "warnings": prop["warnings"], "calibration": os.path.abspath(args.out),
      "created": meta["created"],
    }
    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(args.grid_file)), exist_ok=True)
    with open(args.grid_file, "w") as f:
      f.write("# The FROZEN push-suite grid, written by examples/eval_push_suite.py "
              "--calibrate (E1).\n# Every later eval reads it with --frozen. Edit "
              "only to record a deliberate decision.\n")
      yaml.safe_dump(_jsonable(frozen), f, sort_keys=False, default_flow_style=None)
    meta["frozen_grid"] = frozen
    lines += ["", f"=== E1 proposal -> {args.grid_file}",
              f"impulses {prop['impulses']} N s (F90..F10 of {prop['source']['policy']} "
              f"on lafan), durations {prop['durations']} s "
              f"(curves overlap: {prop['durations_overlap']})",
              f"suggested training force_max {prop['force_max_suggested']:.0f} N "
              f"(1.3 x F10 at 0.2 s)"] + [f"WARNING {w}" for w in prop["warnings"]]
  write_outputs(args.out, summary, lines, meta)


if __name__ == "__main__":
  main()
