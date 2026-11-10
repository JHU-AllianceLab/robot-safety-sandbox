"""Shared ``--config`` plumbing for the trainers (train_on_policy.py / train_off_policy.py).

A YAML config file sets argparse DEFAULTS, so precedence is:

    argparse defaults  <  --config file  <  explicit CLI flags

i.e. a config is a reusable recipe you can still override one knob at a time on
the command line. Config keys must be argparse ``dest`` names (e.g. ``num_envs``,
``gamma_schedule``) — argparse stays the single source of truth for the schema.

Every run also DUMPS its fully-resolved config to ``<outdir>/config.yaml``, so any
run is exactly reproducible: re-run with ``--config <that file>``.
"""

from __future__ import annotations

import argparse
import os

import yaml


# A reserved config key: a dict of ENV/TASK params forwarded to the task's
# cfg_builder (overriding the values baked into its registration). It is NOT
# validated against the argparse flags -- it is a passthrough, so an experiment
# can tune the env from YAML without adding a trainer flag. See make_tensor's
# ``cfg_overrides``. On the CLI, ``--env-override KEY=VAL`` (repeatable) sets
# individual entries, overriding the config's dict per-key.
_ENV_OVERRIDES_KEY = "env_overrides"

# A second reserved config key: the ``safety_filter:`` block that trains a task
# policy INSIDE a safety filter (the off-policy trainer reads it). Like
# ``env_overrides`` it is a nested-dict passthrough, resolved onto
# ``args.safety_filter``, with a per-key CLI override ``--safety-filter KEY=VAL``
# (repeatable, wins per-key). Schema (every key optional; presence of
# ``safety_policy`` selects the filtered arm, absence = the unfiltered control):
#     safety_filter:
#       safety_policy: path/to/twin.zip   # the fallback twin; omit -> control
#       filter: critic                    # value | critic | qcbf
#       eps: 0.1                           # switching threshold
#       smoothing: false                  # HeuristicSmoothing vs canonical switch
_SAFETY_FILTER_KEY = "safety_filter"

# A third reserved config key: ``random_push:``, per-key overrides of the TASK's
# registered random_push kwargs (random_push.RandomPush: interval_s, duration_s,
# first_delay_s, max_pushes, rear_frac, ...). Those are bridge kwargs baked into
# the TaskSpec, which ``env_overrides`` (cfg_builder params) cannot reach. Same
# passthrough shape: resolved onto ``args.random_push`` (config dict, then the
# per-key CLI ``--random-push KEY=VAL``, repeatable, wins per-key); the trainer
# merges it onto the registration (resolve_random_push) and passes the result
# to make_tensor. Only a trainer that defines ``--random-push`` (dest
# ``random_push_override``: train_on_policy.py) reads it; for any other trainer
# a config carrying the key is refused rather than silently ignored.
_RANDOM_PUSH_KEY = "random_push"
_RANDOM_PUSH_DEST = "random_push_override"


def _parse_kv(pairs, flag="env-override"):
    """Parse ``["k=v", ...]`` into a dict, interpreting each value as YAML
    (so ``0.003`` -> float, ``true`` -> bool, ``foo`` -> str, ``[0.1, 0.4]``
    -> list). ``flag`` names the CLI flag in the error message."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"[{flag}] expected KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def merge_config(parser):
    """Parse args with an optional ``--config`` YAML applied as defaults.

    Two-phase so a config can supply even REQUIRED args (e.g. ``--task``): a
    throwaway pre-parser extracts ``--config`` from the command line first, its
    keys are validated + applied as defaults on ``parser``, then ``parser`` does
    the strict parse. Precedence: argparse defaults < config < explicit CLI flags.

    The reserved ``env_overrides:`` and ``safety_filter:`` config keys (each a
    dict) are PASSTHROUGHS — not validated against the flags — resolved (each
    merged with its per-key CLI overrides, CLI winning per-key) onto
    ``args.env_overrides`` / ``args.safety_filter``. ``random_push:`` is the
    same, onto ``args.random_push``, for a parser that defines
    ``--random-push`` (dest ``random_push_override``); with any other parser a
    config carrying it is refused.

    Call this INSTEAD of ``parser.parse_args()``. Returns the args namespace.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    path = pre.parse_known_args()[0].config
    cfg_env_overrides = {}
    cfg_safety_filter = {}
    cfg_random_push = {}
    reads_push = any(a.dest == _RANDOM_PUSH_DEST for a in parser._actions)
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"[config] file not found: {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise SystemExit(
                f"[config] {path} must be a YAML mapping (key: value), got "
                f"{type(cfg).__name__}")
        cfg_env_overrides = cfg.pop(_ENV_OVERRIDES_KEY, {}) or {}   # reserved passthrough
        if not isinstance(cfg_env_overrides, dict):
            raise SystemExit(
                f"[config] '{_ENV_OVERRIDES_KEY}' must be a mapping, got "
                f"{type(cfg_env_overrides).__name__}")
        cfg_safety_filter = cfg.pop(_SAFETY_FILTER_KEY, {}) or {}   # reserved passthrough
        if not isinstance(cfg_safety_filter, dict):
            raise SystemExit(
                f"[config] '{_SAFETY_FILTER_KEY}' must be a mapping, got "
                f"{type(cfg_safety_filter).__name__}")
        if _RANDOM_PUSH_KEY in cfg:
            if not reads_push and cfg[_RANDOM_PUSH_KEY]:
                raise SystemExit(
                    f"[config] '{_RANDOM_PUSH_KEY}' in {path} overrides a task's "
                    f"random-push kwargs, which only the on-policy trainer "
                    f"(train_on_policy.py) applies; this trainer would ignore it.")
            cfg_random_push = cfg.pop(_RANDOM_PUSH_KEY) or {}   # reserved passthrough
            if not isinstance(cfg_random_push, dict):
                raise SystemExit(
                    f"[config] '{_RANDOM_PUSH_KEY}' must be a mapping, got "
                    f"{type(cfg_random_push).__name__}")
        cfg.pop("family", None)   # reserved: consumed by the train.py router, not a trainer arg
        valid = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
        bad = [k for k in cfg if k not in valid]
        if bad:
            raise SystemExit(
                f"[config] unknown keys in {path}: {sorted(bad)}\n"
                f"  valid keys: {sorted(valid)} (env/task params go under "
                f"'{_ENV_OVERRIDES_KEY}:')")
        parser.set_defaults(**cfg)
        # A `required=True` arg (e.g. --task) ignores a default, so clear the
        # requirement for anything the config supplies.
        for a in parser._actions:
            if a.dest in cfg and getattr(a, "required", False):
                a.required = False
        print(f"[config] loaded {path} ({len(cfg)} keys"
              f"{f' + {len(cfg_env_overrides)} env_overrides' if cfg_env_overrides else ''}"
              f"{f' + {len(cfg_safety_filter)} safety_filter' if cfg_safety_filter else ''}"
              f"{f' + {len(cfg_random_push)} random_push' if cfg_random_push else ''}"
              f"); CLI flags override it")
    args = parser.parse_args()   # strict parse: config = defaults, CLI overrides
    # Resolve env_overrides: config dict, then CLI --env-override entries (win per-key).
    cli_env = _parse_kv(getattr(args, "env_override", None))
    args.env_overrides = {**cfg_env_overrides, **cli_env}
    # Resolve safety_filter the same way (config dict, then --safety-filter CLI).
    cli_sf = _parse_kv(getattr(args, "safety_filter_override", None))
    args.safety_filter = {**cfg_safety_filter, **cli_sf}
    # And random_push (config dict, then --random-push CLI), where it is read.
    if reads_push:
        cli_rp = _parse_kv(getattr(args, _RANDOM_PUSH_DEST, None), "random-push")
        args.random_push = {**cfg_random_push, **cli_rp}
    return args


def resolve_random_push(task_id, registered, overrides):
    """The random-push kwargs a run builds its env with: the task's registered
    ``random_push`` kwargs with ``overrides`` (``args.random_push``) on top,
    per key. Returns None when there is nothing to pass (no overrides), so the
    env is built exactly from the registration.

    Refused (SystemExit) rather than ignored: overrides for a task registered
    without random pushes, and a key ``RandomPush`` does not take (checked
    against its signature here, before any output dir or env exists)."""
    if not overrides:
        return None
    if registered is None:
        raise SystemExit(
            f"[random-push] '{task_id}' is registered without random pushes, so "
            f"'{_RANDOM_PUSH_KEY}' overrides {sorted(overrides)} have nothing to "
            f"override. Use a push task (e.g. g1_recover_push_fast, "
            f"g1_recover_lafan).")
    import inspect

    from robot_safety_sandbox.random_push import RandomPush
    merged = {**registered, **overrides}
    try:
        inspect.signature(RandomPush).bind(1, 0.02, "cpu", **merged)
    except TypeError as e:
        raise SystemExit(f"[random-push] {sorted(overrides)} not all accepted by "
                         f"RandomPush: {e}") from e
    return merged


def dump_config(outdir, args):
    """Write the fully-resolved run config to ``<outdir>/config.yaml``.

    Drops ``config`` and the raw ``env_override`` / ``safety_filter_override``
    / ``random_push_override`` CLI lists, keeping the resolved
    ``env_overrides`` / ``safety_filter`` / ``random_push`` dicts, so the dump
    round-trips via ``--config``. Reproduce the run with
    ``--config <that file>``."""
    drop = {"config", "env_override", "safety_filter_override", _RANDOM_PUSH_DEST}
    d = {k: v for k, v in vars(args).items() if k not in drop}
    if not d.get("env_overrides"):
        d.pop("env_overrides", None)   # omit an empty dict for tidiness
    if not d.get("safety_filter"):
        d.pop("safety_filter", None)   # omit an empty dict for tidiness
    if not d.get(_RANDOM_PUSH_KEY):
        d.pop(_RANDOM_PUSH_KEY, None)   # omit an empty dict for tidiness
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)
    print(f"[config] resolved run config -> {path}")
