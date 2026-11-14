"""The released G1 thermal safety filters: load one onto a task env.

The release (a Hugging Face model repo, or any local copy of it) is written by
``examples/export_safety_filters.py``::

    <root>/
      README.md
      layouts/thermal70.json, layouts/thermal_cbf.json   fallback_layout() references
      thermal70/s{0,1,2}/                                the 70 C margin world's SHAC runs
        final_model.zip  tensornormalize.pt  config.yaml  g1_sysid.json  manifest.json
        tubes/joint.json  tubes/separate.json
      thermal_cbf/s{0,1,2}/                              the barrier world's SHAC runs
        ...  tubes/fall_only.json

Three filters, one per row of the report (all three: PolicyFallback +
WMTubeMonitor + DwellSwitchIntervention, filters/wm_tube.py):

  joint          the 70 C margin world's learned world model; its head learns the
                 whole margin min(fall, (70 C - hottest winding) / 10 C)
  fall_only_cbf  the barrier world's learned world model, fall margin only; the
                 70 C limit is the env's torque clip (the env MUST be the
                 barrier world: g1_recover_thermal_cbf's joint_pos action term)
  separate       the 70 C margin world's network for the motion, the sysID heat
                 model for the temperatures and the thermal margin

Usage, on an env that carries the fallback's ``actor`` / ``critic`` groups, sim
context, robot, thermal model and action term (built from the fallback's own
world cfg, as envs/g1_track does; :func:`load_pretrained_filter` checks it)::

    from robot_safety_sandbox.filters.pretrained import download, load_pretrained_filter
    from robot_safety_sandbox.filtered_env import WMTubeFilteredEnv

    root = download("<owner>/g1-thermal-safety-filters")       # or a local path
    filt, bundle = load_pretrained_filter(env, root, "separate", seed=0,
                                          condition="wmmin:1:0.95+dwell1",
                                          proposal_bound=float(env.action_space.high.max()))
    env = WMTubeFilteredEnv(env, filt)

Nothing here imports mjlab, torch or safety_sb3 at module level.
"""

from __future__ import annotations

import json
import os
from typing import Optional

#: variant -> the world (run directory family), its tube file, the world model
#: the tube was calibrated with
VARIANTS = {
  "joint": {"world": "thermal70", "tube": "tubes/joint.json", "wm_thermal": "nn"},
  "fall_only_cbf": {"world": "thermal_cbf", "tube": "tubes/fall_only.json", "wm_thermal": "nn"},
  "separate": {"world": "thermal70", "tube": "tubes/separate.json", "wm_thermal": "hybrid"},
}
#: the conditions the release offers (filters/wm_tube.ALLOWED_CONDITIONS plus a hold)
CONDITIONS = ("wmmin:0.4:0.95", "wmmin:1:0.95", "wmmin:1:0.95+dwell1", "wmmin:1:0.95+dwell2",
              "wmmin:1:0.95+latch")


def variant_paths(root: str, variant: str, seed: int = 0) -> dict:
  """The run directory, tube, tag, world model and layout reference of
  ``variant`` / ``seed`` in a release at ``root``."""
  if variant not in VARIANTS:
    raise ValueError(f"variant {variant!r}: one of {sorted(VARIANTS)}")
  v = VARIANTS[variant]
  run_dir = os.path.join(root, v["world"], f"s{int(seed)}")
  man_path = os.path.join(run_dir, "manifest.json")
  if not os.path.isfile(man_path):
    raise FileNotFoundError(f"{man_path}: not a release run directory (seed {seed}?)")
  with open(man_path) as f:
    man = json.load(f)
  tube = os.path.join(run_dir, v["tube"])
  if not os.path.isfile(tube):
    raise FileNotFoundError(f"{tube}: the {variant} tube is missing from the release")
  return {"run_dir": run_dir, "tube_json": tube, "tag": man["tag"],
          "wm_thermal": v["wm_thermal"], "world": v["world"],
          "layout": os.path.join(root, "layouts", f"{v['world']}.json")}


def load_pretrained_filter(env, root: str, variant: str, seed: int = 0, *,
                           condition: str = "wmmin:1:0.95", proposal_bound: float,
                           device: Optional[str] = None, check: bool = True,
                           actor_group: str = "actor", critic_group: str = "critic",
                           allow: tuple = (), **kw):
  """``(filter, bundle)`` of ``variant`` / ``seed`` from the release at
  ``root`` on ``env`` (wrap with ``filtered_env.WMTubeFilteredEnv``).

  ``condition``: ``wmmin:<0.4|1>:0.95`` with an optional ``+dwell<s>`` or
  ``+latch`` (``wmmin:1:0.9`` is the eval's sensitivity tube). ``check``:
  :func:`~robot_safety_sandbox.filters.wm_tube.check_binding` against the
  release's layout reference first (call after ``env.reset()``); it refuses an
  env whose groups, context, action term or thermal world differ from the
  fallback's, except the differences ``allow`` names (wm_tube.CHECK_ALLOW).
  Other keywords go to ``wm_tube_filter`` (``mode``,
  ``precision``, ``nan_engages``, ...)."""
  from .wm_tube import _base, check_binding, load_wm_tube, parse_wm_condition, wm_tube_filter
  p = variant_paths(root, variant, seed)
  cond = parse_wm_condition(condition)
  base = _base(env)
  dt = float(base.mj.step_dt)
  dev = str(env.device if device is None else device)
  bundle = load_wm_tube(p["run_dir"], p["tube_json"], tag=p["tag"], cond=cond, device=dev,
                        step_dt=dt, wm_thermal=p["wm_thermal"])
  if check:
    with open(p["layout"]) as f:
      layout = json.load(f)
    check_binding(bundle, env, actor_group=actor_group, critic_group=critic_group,
                  layout_ref=layout, allow=allow)
  return wm_tube_filter(env, condition=cond.name, dwell_s=cond.dwell_s or 0.0,
                        latch=cond.latch, proposal_bound=proposal_bound, device=dev,
                        actor_group=actor_group, critic_group=critic_group, bundle=bundle, **kw)


def download(repo_id: str, local_dir: Optional[str] = None, *, revision: Optional[str] = None,
             token: Optional[str] = None) -> str:
  """A local copy of the release ``repo_id`` (Hugging Face model repo; a
  private one needs ``huggingface-cli login`` or ``token``). Returns its path."""
  try:
    from huggingface_hub import snapshot_download
  except ImportError as e:
    raise ImportError("pip install huggingface_hub (or pass a local copy as root)") from e
  return snapshot_download(repo_id, repo_type="model", local_dir=local_dir, revision=revision,
                           token=token)
