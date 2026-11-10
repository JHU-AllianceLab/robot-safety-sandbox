"""Generate a training sweep: one trainer config per arm, plus the job lists.

    python examples/make_sweep.py                       # configs/sweeps/lafan_recovery/
    python examples/make_sweep.py --smoke               # configs/sweeps/lafan_recovery_smoke/
    python examples/make_sweep.py --only E3 --only E5   # just those experiments' jobs
    python examples/make_sweep.py --force-max 260       # E1's ceiling, into every arm

then run the lists one job at a time (locally, or one line per cluster job):

    examples/run_jobs.sh configs/sweeps/lafan_recovery/prep_jobs.txt
    examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt
    examples/run_jobs.sh configs/sweeps/lafan_recovery/eval_jobs.txt
    examples/run_jobs.sh configs/sweeps/lafan_recovery/report_jobs.txt

THE SWEEP FILE (``--sweep``, default ``configs/sweeps/lafan_recovery.yaml``)
holds ONE reference config (``base``) and, per experiment, arms that override
it (``experiments.<exp>.arms.<arm>.config``). An arm resolves as the base with
its override on top (:func:`merge_arm`): top-level keys replace, the two
passthrough mappings ``env_overrides`` and ``random_push`` merge PER KEY (each
value replaces the base's whole, so an arm's ``weights`` is the complete
mixture, not a patch). Arms whose resolved configs are identical train ONCE:
the first one in file order owns the run directory and the later ones
(``same_as``) reference it, so the reference arm shared by several
experiments (S3 = P-env = O0) is one run per seed, compared in each.

WHAT IS CHECKED, before anything is written:

* every key of every generated config is a dest of the on-policy trainer's
  argparse (read from ``examples/train_on_policy.py`` by AST, so it tracks the
  trainer without importing it) or a reserved passthrough (``family``,
  ``env_overrides``, ``random_push``);
* the task is registered, every ``env_overrides`` key is a parameter of its
  cfg builder, and ``random_push`` overrides are accepted by ``RandomPush``
  (``_run_config.resolve_random_push``) -- skipped with ``--no-check-env``,
  since it imports the package (and mjlab);
* every job line's script exists and takes every ``--flag`` the line passes
  (read from the script's argparse by AST: a prep step written against a
  script's old CLI fails here);
* every data / checkpoint path a config or a job reads either exists or is in
  some prep step's ``creates`` (a typo in a bank name shows up here, not after
  a cluster job has queued for a day).

OUTPUTS (in the sweep's ``config_dir``, or ``--out-dir``):

    <exp>_<arm>.yaml   the arm's trainer config (``train.py --config``)
    jobs.txt           one training job per unique (arm, seed):
                       ``python examples/train.py --config <cfg> --out
                       <runs_root>/<exp>/<arm>_s<seed> --seed <seed>``; seeds
                       outermost, so all of seed 0 comes first
    jobs_gated.txt     the same for arms with a ``gate`` (a condition on an
                       earlier read-out, e.g. S5 on E2): run them only if it
                       holds
    eval_jobs.txt      the frozen push suite on every trained run's final
                       model and on the baselines
    report_jobs.txt    one merged report per experiment (``eval_push_suite.py
                       --merge``: per-arm means and seed gaps)
    prep_jobs.txt      the prerequisites (data, labels, E1, E2), in order
    manifest.json      everything above, resolved (arm -> config, run dirs,
                       final models, eval dirs, canonical arm)

The trainer writes a run into ``<--out>/<task>/`` (its tag), so the final
model of ``<runs_root>/E3/S3_s0`` is ``<runs_root>/E3/S3_s0/g1_recover_lafan/
final_model.zip``. Prep commands may use ``${force_max}`` (the base's),
``${load}`` (the base's warm start, also the labeling policy) and any
``${name}`` of the sweep's ``prep_vars``.

``--smoke`` applies the sweep's ``smoke`` block: its ``config`` on top of
every arm (short run, small batch, no checkpoints or videos inside it), its
``train_flags`` / ``eval_flags`` on the command lines (``--no-wandb``,
``--quick``), its own runs root, config dir and seeds.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import os
import string
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.dirname(_HERE)

DEFAULT_SWEEP = os.path.join("configs", "sweeps", "lafan_recovery.yaml")
TRAINER = os.path.join(_HERE, "train_on_policy.py")
#: Config keys the trainer accepts that are not argparse dests: the train.py
#: router's ``family`` and _run_config's reserved passthrough mappings.
RESERVED = ("family", "env_overrides", "random_push")
#: Keys merged per key (one level) instead of replaced.
MERGED = ("env_overrides", "random_push")


def trainer_dests(path: str = TRAINER) -> set[str]:
  """The argparse dests of every ``add_argument`` call in ``path`` (an explicit
  ``dest=`` or the first long option, dashes -> underscores), minus help and
  --config. Read from the source so the check needs no import."""
  with open(path) as f:
    tree = ast.parse(f.read(), filename=path)
  dests = set()
  for node in ast.walk(tree):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"):
      continue
    dest = next((kw.value.value for kw in node.keywords if kw.arg == "dest"
                 and isinstance(kw.value, ast.Constant)), None)
    if dest is None:
      opts = [a.value for a in node.args
              if isinstance(a, ast.Constant) and isinstance(a.value, str)]
      longs = [o for o in opts if o.startswith("--")]
      if longs:
        dest = longs[0][2:].replace("-", "_")
      elif opts and not opts[0].startswith("-"):
        dest = opts[0]
    if dest:
      dests.add(dest)
  return dests - {"help", "config"}


def script_flags(path: str) -> set[str]:
  """Every long option (``--x``) an ``add_argument`` call in ``path`` defines."""
  with open(path) as f:
    tree = ast.parse(f.read(), filename=path)
  return {a.value for node in ast.walk(tree)
          if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
          and node.func.attr == "add_argument"
          for a in node.args
          if isinstance(a, ast.Constant) and isinstance(a.value, str)
          and a.value.startswith("--")}


def check_command(line: str, root: str = _ZOO) -> list[str]:
  """A job line calling ``examples/<script>.py``: the script exists and takes
  every ``--flag`` the line passes (train.py forwards to the trainer, so it
  takes the trainer's). Other commands pass unchecked."""
  import shlex
  toks = shlex.split(line, comments=True)
  script = next((t for t in toks if t.startswith("examples/") and t.endswith(".py")), None)
  if script is None:
    return []
  path = os.path.join(root, script)
  if not os.path.exists(path):
    return [f"{script} does not exist"]
  flags = script_flags(path)
  if os.path.basename(script) == "train.py":
    flags |= script_flags(TRAINER) | {"--family"}
  bad = sorted({t.split("=", 1)[0] for t in toks if t.startswith("--")} - flags)
  return [f"{script} does not take {bad}"] if bad else []


def merge_arm(base: dict, override: dict | None) -> dict:
  """``base`` with ``override`` on top: top-level keys replace, ``MERGED``
  mappings merge one level (each value replacing the base's whole). Neither
  input is modified."""
  out = copy.deepcopy(base)
  for k, v in (override or {}).items():
    if k in MERGED and isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = {**out[k], **copy.deepcopy(v)}
    else:
      out[k] = copy.deepcopy(v)
  return out


def _key(cfg: dict) -> str:
  return json.dumps(cfg, sort_keys=True, default=str)


def load_sweep(path: str) -> dict:
  with open(path) as f:
    sw = yaml.safe_load(f) or {}
  for k in ("base", "experiments", "seeds", "runs_root", "config_dir"):
    if k not in sw:
      raise SystemExit(f"[sweep] {path}: missing '{k}'")
  return sw


def resolve(sw: dict, smoke: bool = False, force_max: float | None = None) -> dict:
  """Every arm of every experiment, resolved: its config, the canonical arm it
  trains as, its run / eval directories per seed. Returns ``{"arms": [...],
  "runs_root", "config_dir", "seeds", "train_flags", "eval_flags", "base"}``."""
  base = copy.deepcopy(sw["base"])
  if force_max is not None:
    base["force_max"] = base["force_start"] = float(force_max)
  sm = sw.get("smoke") or {}
  runs_root = sm.get("runs_root", sw["runs_root"]) if smoke else sw["runs_root"]
  config_dir = sm.get("config_dir", sw["config_dir"]) if smoke else sw["config_dir"]
  seeds = [int(s) for s in (sm.get("seeds", sw["seeds"]) if smoke else sw["seeds"])]
  arms, owner = [], {}
  for exp, e in sw["experiments"].items():
    for arm, a in (e.get("arms") or {}).items():
      a = a or {}
      cfg = merge_arm(base, a.get("config"))
      if smoke:
        cfg = merge_arm(cfg, sm.get("config"))
      key = _key(cfg)
      canon = owner.setdefault(key, (exp, arm))
      c_exp, c_arm = canon
      task = cfg.get("task")
      runs = {s: f"{runs_root}/{c_exp}/{c_arm}_s{s}" for s in seeds}
      arms.append({
        "exp": exp, "arm": arm, "config": cfg, "note": a.get("note", ""),
        "gate": a.get("gate"), "canonical": f"{c_exp}/{c_arm}",
        "same_as": None if canon == (exp, arm) else f"{c_exp}/{c_arm}",
        "config_path": f"{config_dir}/{c_exp}_{c_arm}.yaml",
        "own_config_path": f"{config_dir}/{exp}_{arm}.yaml",
        "runs": runs,
        "final_models": {s: f"{d}/{task}/final_model.zip" for s, d in runs.items()},
        "eval_dirs": {s: f"{d}/eval_frozen" for s, d in runs.items()},
      })
  # a gated arm that is also the canonical of an ungated one is not gated
  ungated = {a["canonical"] for a in arms if not a["gate"]}
  for a in arms:
    if a["gate"] and a["canonical"] in ungated:
      a["gate"] = None
  return {"arms": arms, "runs_root": runs_root, "config_dir": config_dir,
          "seeds": seeds, "base": base,
          "train_flags": list(sm.get("train_flags", [])) if smoke else [],
          "eval_flags": list(sm.get("eval_flags", [])) if smoke else []}


# --- checks ----------------------------------------------------------------------

def check_keys(cfg: dict, dests: set[str]) -> list[str]:
  """Config keys the on-policy trainer would refuse."""
  bad = sorted(k for k in cfg if k not in dests and k not in RESERVED)
  errs = [f"unknown trainer keys {bad}"] if bad else []
  if cfg.get("random_push") and "random_push_override" not in dests:
    errs.append("random_push: set but this trainer has no --random-push")
  for k in MERGED:
    if k in cfg and not isinstance(cfg[k], dict):
      errs.append(f"{k} must be a mapping, got {type(cfg[k]).__name__}")
  return errs


def check_env(cfg: dict) -> list[str]:
  """The task exists, its cfg builder takes every env_overrides key, and
  RandomPush takes every random_push key."""
  sys.path.insert(0, _ZOO)
  sys.path.insert(0, _HERE)
  from robot_safety_sandbox import spec
  errs = []
  task = cfg.get("task")
  try:
    s = spec(task)
  except KeyError:
    return [f"task {task!r} is not registered"]
  sig = inspect.signature(s.cfg_builder)
  takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                  for p in sig.parameters.values())
  bad = sorted(k for k in (cfg.get("env_overrides") or {})
               if k not in sig.parameters and not takes_any)
  if bad:
    errs.append(f"env_overrides {bad} are not parameters of {task}'s cfg builder")
  if cfg.get("random_push"):
    from _run_config import resolve_random_push
    try:
      resolve_random_push(task, s.kwargs.get("random_push"), cfg["random_push"])
    except SystemExit as e:
      errs.append(str(e))
  return errs


def _paths_read(cfg: dict) -> list[str]:
  ds = (cfg.get("env_overrides") or {}).get("dataset")
  out = [ds] if isinstance(ds, str) else list(ds or [])
  if cfg.get("load"):
    out.append(cfg["load"])
  return out


def check_paths(paths, creates: set[str], root: str = _ZOO) -> list[str]:
  """Paths that neither exist (relative to the repo root) nor are created by a
  prep step."""
  return sorted({p for p in paths
                 if p not in creates and not os.path.exists(os.path.join(root, p))})


# --- writing ----------------------------------------------------------------------

def _py(cmd: str, python: str) -> str:
  return python + cmd[len("python"):] if cmd.startswith("python ") else cmd


def prep_lines(sw: dict, base: dict, python: str) -> list[str]:
  out = []
  for step in sw.get("prep") or []:
    cmd = " ".join(str(step["cmd"]).split())
    cmd = string.Template(cmd).substitute(
      {k: str(v) for k, v in (sw.get("prep_vars") or {}).items()},
      force_max=f"{float(base['force_max']):g}", load=base.get("load") or "")
    note = f"# [{step['name']}] {step.get('note', '')}".rstrip()
    out += [note, _py(cmd, python) + (
      f"  # creates: {' '.join(step['creates'])}" if step.get("creates") else "")]
  return out


def build_jobs(res: dict, sw: dict, only: list[str] | None, python: str) -> dict:
  """The job lists (each a list of lines) for the selected experiments."""
  sel = [a for a in res["arms"] if not only or a["exp"] in only]
  by_name = {f"{a['exp']}/{a['arm']}": a for a in res["arms"]}
  needed = list(dict.fromkeys(a["canonical"] for a in sel))
  tf = " ".join(res["train_flags"])
  ef = " ".join(res["eval_flags"]) or "--frozen"
  ev = sw.get("eval") or {}
  grid = ev.get("grid_file")
  grid_only = f" --grid-file {grid}" if grid and ef == "--frozen" else ""
  grid_flag = grid_only + "".join(f" --setting {s}" for s in ev.get("settings") or [])
  jobs, gated, evals = [], [], []
  for seed in res["seeds"]:
    for c in needed:
      a = by_name[c]
      line = (f"{python} examples/train.py --config {a['config_path']} "
              f"--out {a['runs'][seed]} --seed {seed}" + (f" {tf}" if tf else ""))
      ev_line = (f"{python} examples/eval_push_suite.py {ef}{grid_flag} --policy "
                 f"{a['arm']}_s{seed}={a['final_models'][seed]} --out {a['eval_dirs'][seed]}")
      if a["gate"]:
        gated += [f"# [{c}] gated on {a['gate']}", line, ev_line]
      else:
        jobs.append(line)
        evals.append(ev_line)
  base_dirs = {}
  for name, path in (ev.get("baselines") or {}).items():
    d = f"{res['runs_root']}/baselines/{name}/eval_frozen"
    base_dirs[name] = d
    evals.append(f"{python} examples/eval_push_suite.py {ef}{grid_flag} --policy "
                 f"{name}={path} --out {d}")
  reports = []

  def report(exp, sub, dirs_of, bases):
    parts = [f"{a['arm']}_s{s}={dirs_of(a, s)}" for a in sel if a["exp"] == exp
             for s in res["seeds"]]
    parts += [f"{n}={d}" for n, d in bases.items()]
    # "# always": run_jobs.sh never skips it -- a report is redone as evals land
    reports.append(f"{python} examples/eval_push_suite.py --merge {' '.join(parts)} "
                   f"--out {res['runs_root']}/{exp}/{sub}  # always")

  for exp in dict.fromkeys(a["exp"] for a in sel):
    report(exp, "report", lambda a, s: a["eval_dirs"][s], base_dirs)
  # extra eval PASSES over some experiments' runs (e.g. E5's staying check: a
  # longer window): their own flags, eval dirs <run>/eval_<pass>, reports
  # <exp>/report_<pass>. A pass that names --setting replaces the default list.
  for pname, pas in (ev.get("passes") or {}).items():
    exps = [e for e in pas.get("experiments", []) if any(a["exp"] == e for a in sel)]
    if not exps:
      continue
    pf = " ".join(str(x) for x in pas.get("flags", []))
    gf = grid_only if "--setting" in pf else grid_flag
    sub = f"eval_{pname}"
    for seed in res["seeds"]:
      for c in dict.fromkeys(a["canonical"] for a in sel if a["exp"] in exps):
        a = by_name[c]
        line = (f"{python} examples/eval_push_suite.py {ef}{gf} {pf} --policy "
                f"{a['arm']}_s{seed}={a['final_models'][seed]} "
                f"--out {a['runs'][seed]}/{sub}")
        (gated if a["gate"] else evals).append(line)
    pbase = {}
    for name, path in (ev.get("baselines") or {}).items():
      d = f"{res['runs_root']}/baselines/{name}/{sub}"
      pbase[name] = d
      evals.append(f"{python} examples/eval_push_suite.py {ef}{gf} {pf} --policy "
                   f"{name}={path} --out {d}")
    for exp in exps:
      report(exp, f"report_{pname}", lambda a, s: f"{a['runs'][s]}/{sub}", pbase)
  return {"jobs": jobs, "gated": gated, "evals": evals, "reports": reports,
          "selected": sel, "needed": needed}


def _header(title: str, sweep_path: str, smoke: bool) -> list[str]:
  return [f"# {title} -- generated by examples/make_sweep.py from {sweep_path}"
          f"{' (--smoke)' if smoke else ''}; do not edit.",
          "# One job per line; run with examples/run_jobs.sh <this file> "
          "(from the repo root)."]


def write_config(path: str, cfg: dict, arm: dict, sweep_path: str):
  head = [f"# {arm['exp']}/{arm['arm']}: {arm['note']}".rstrip(),
          f"# generated by examples/make_sweep.py from {sweep_path} -- edit the "
          f"sweep file, not this one."]
  if arm.get("same_as"):
    head.append(f"# identical to {arm['same_as']}: trained once, as that arm")
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w") as f:
    f.write("\n".join(head) + "\n")
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=100)


def generate(sweep_path: str = DEFAULT_SWEEP, smoke: bool = False,
             only: list[str] | None = None, out_dir: str | None = None,
             force_max: float | None = None, python: str = "python",
             check_env_on: bool = True, write: bool = True,
             root: str = _ZOO) -> dict:
  """Resolve, check and (``write``) emit the sweep. Returns the resolution
  plus the job lists and every problem found; raises SystemExit on a config
  the trainer would refuse."""
  sw = load_sweep(sweep_path if os.path.isabs(sweep_path)
                  else os.path.join(root, sweep_path))
  if only:
    unknown = sorted(set(only) - set(sw["experiments"]))
    if unknown:
      raise SystemExit(f"[sweep] --only {unknown}: not in {sorted(sw['experiments'])}")
  res = resolve(sw, smoke=smoke, force_max=force_max)
  if out_dir:
    res["config_dir"] = out_dir
    for a in res["arms"]:
      a["config_path"] = f"{out_dir}/{os.path.basename(a['config_path'])}"
      a["own_config_path"] = f"{out_dir}/{os.path.basename(a['own_config_path'])}"
  dests = trainer_dests()
  creates = {p for st in (sw.get("prep") or []) for p in (st.get("creates") or [])}
  errors, warnings = [], []
  jobs = build_jobs(res, sw, only, python)
  checked = {}
  for a in jobs["selected"] + [a for a in res["arms"]
                               if f"{a['exp']}/{a['arm']}" in jobs["needed"]]:
    cfg = a["config"]
    k = _key(cfg)
    if k in checked:
      continue
    probs = check_keys(cfg, dests)
    if check_env_on:
      probs += check_env(cfg)
    checked[k] = probs
    errors += [f"{a['exp']}/{a['arm']}: {p}" for p in probs]
    miss = check_paths(_paths_read(cfg), creates, root)
    warnings += [f"{a['exp']}/{a['arm']}: {p} does not exist and no prep step creates it"
                 for p in miss]
  for name, path in ((sw.get("eval") or {}).get("baselines") or {}).items():
    for p in check_paths([path], creates, root):
      warnings.append(f"baseline {name}: {p} does not exist")
  prep = prep_lines(sw, res["base"], python)
  for line in [l for l in prep if not l.startswith("#")] + jobs["jobs"] + [
      l for l in jobs["gated"] if not l.startswith("#")] + jobs["evals"] + jobs["reports"]:
    errors += [f"job `{line[:60]}...`: {p}" for p in check_command(line, root)]
  if errors:
    raise SystemExit("[sweep] refused:\n  " + "\n  ".join(dict.fromkeys(errors)))
  out = dict(res, **jobs, prep=prep, errors=errors, warnings=warnings)
  if not write:
    return out
  cdir = res["config_dir"] if os.path.isabs(res["config_dir"]) else os.path.join(
    root, res["config_dir"])
  os.makedirs(cdir, exist_ok=True)
  written = set()
  for a in jobs["selected"] + [a for a in res["arms"]
                               if f"{a['exp']}/{a['arm']}" in jobs["needed"]]:
    p = os.path.join(root, a["own_config_path"]) if not os.path.isabs(
      a["own_config_path"]) else a["own_config_path"]
    if p not in written:
      write_config(p, a["config"], a, sweep_path)
      written.add(p)
  files = {"jobs.txt": ("training jobs", jobs["jobs"]),
           "eval_jobs.txt": ("frozen push-suite evals", jobs["evals"]),
           "report_jobs.txt": ("per-experiment reports", jobs["reports"]),
           "prep_jobs.txt": ("prerequisites, in order", out["prep"])}
  if jobs["gated"]:
    files["jobs_gated.txt"] = ("GATED training jobs: run only if the gate holds",
                               jobs["gated"])
  else:
    stale = os.path.join(cdir, "jobs_gated.txt")
    if os.path.exists(stale):
      os.remove(stale)
  for fname, (title, lines) in files.items():
    with open(os.path.join(cdir, fname), "w") as f:
      f.write("\n".join(_header(title, sweep_path, smoke) + lines) + "\n")
  manifest = {
    "sweep": sweep_path, "smoke": smoke, "only": only, "runs_root": res["runs_root"],
    "seeds": res["seeds"],
    "arms": [{k: a[k] for k in ("exp", "arm", "note", "gate", "canonical", "same_as",
                                "config_path", "runs", "final_models", "eval_dirs")}
             for a in res["arms"] if not only or a["exp"] in only],
    "warnings": warnings,
  }
  with open(os.path.join(cdir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1, default=str)
  out["written"] = sorted(written) + [os.path.join(cdir, f) for f in files]
  return out


def _parse(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--sweep", default=DEFAULT_SWEEP, help="the sweep YAML")
  p.add_argument("--smoke", action="store_true",
                 help="apply the sweep's smoke block (short runs, no wandb, own dirs)")
  p.add_argument("--only", action="append", default=None, metavar="EXP",
                 help="only this experiment's arms and jobs (repeatable)")
  p.add_argument("--out-dir", default=None,
                 help="write the configs and job lists here instead of config_dir")
  p.add_argument("--force-max", type=float, default=None,
                 help="set base force_max = force_start (E1's ceiling) for every arm")
  p.add_argument("--python", default="python",
                 help="the interpreter the job lines call")
  p.add_argument("--no-check-env", action="store_true",
                 help="skip the task / cfg-builder / RandomPush checks (no import)")
  p.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
  return p.parse_args(argv)


def main(argv=None):
  args = _parse(argv)
  out = generate(args.sweep, smoke=args.smoke, only=args.only, out_dir=args.out_dir,
                 force_max=args.force_max, python=args.python,
                 check_env_on=not args.no_check_env, write=not args.dry_run)
  print(f"[sweep] {args.sweep}{' (smoke)' if args.smoke else ''}: runs under "
        f"{out['runs_root']}, seeds {out['seeds']}, force_max "
        f"{out['base'].get('force_max')}")
  for a in out["arms"]:
    if args.only and a["exp"] not in args.only:
      continue
    tag = (f"= {a['same_as']}" if a["same_as"] else
           f"GATED ({a['gate']})" if a["gate"] else "trains")
    print(f"  {a['exp']}/{a['arm']:<14} {tag:<14} {a['note']}")
  print(f"[sweep] {len(out['jobs'])} training jobs, {len(out['gated']) // 3} gated "
        f"(x seeds), {len(out['evals'])} evals, {len(out['reports'])} reports, "
        f"{len(out['prep']) // 2} prep steps")
  for w in out["warnings"]:
    print(f"[sweep] WARNING {w}")
  if not args.dry_run:
    print(f"[sweep] wrote {out['config_dir']}/ (jobs.txt, eval_jobs.txt, "
          f"report_jobs.txt, prep_jobs.txt, manifest.json, configs)")


if __name__ == "__main__":
  main()
