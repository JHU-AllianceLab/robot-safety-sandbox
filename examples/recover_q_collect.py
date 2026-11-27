"""Policy-evaluation data for a fallback's Q^pi, in the fallback's OWN training setting (v0.6 port).

    PYTHONPATH=$HOME/safety-stable-baselines-baselines \
    G1_SYSID_V061_DIR=$HOME/G1-robot-model-with-SysID-v0.6.1 \
    python examples/recover_q_collect.py \
        --policy runs/v061_study/safetyppo_noheight_s0/g1_recover_noheight_v6/final_model.zip \
        --out runs/v061_study/q_eval/data_safetyppo.pt

Port of the main checkout's ``examples/recover_q_collect.py`` (mjlab 1.2, task g1_recover, 2026-11-21)
to the sysID v0.6 world of the heat-blind v0.6.1-study fallbacks (SafetyPPO
``runs/v061_study/safetyppo_noheight_s0``, lambda-Reach ``runs/thermal65/lambdareach_s0``). The recipe
is unchanged; what the v0.6 task changes:

  * task ``g1_recover_noheight_v6`` (sysID v0.6 body, margin g_fall_noheight, NO pelvis height in the
    critic group: actor 98-D, critic 113-D instead of 114-D), the fallbacks' training task
    (``configs/g1_recover_noheight_v6_{safetyppo,lambdareach}1p.yaml``);
  * bank ``data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt``, both configs' bank;
  * the last-action columns are READ from the env's observation layout (the ``actions`` term of each
    group, ``_obs_layout``), not hard-coded, and must be one 29-D slice; on this task they are 69:98
    in both groups, as before (the dropped pelvis height was the critic group's LAST column, 113);
  * new outputs (additive): ``pi0`` (the fallback's own action at s_0, for every rollout, so the fit
    can score Q(s_0, pi(s_0)) on bank start states whatever the first action was), and in ``meta``
    the observation layouts, the exact sanitize, the safety_sb3 path and the per-column count of
    non-finite RAW observation values (the NaN-prone columns, if any).

The safety filter's Q monitor executes a task proposal a iff Q^pi(s, a) >= delta, pi the filter's
fallback. Q^pi belongs to the filter, not to a task: it is evaluated where the fallback was trained:
start states the SONIC hand-over rows of the bank (weights handover 1, sampler uniform), the task's
startup domain randomization, 3 s episodes with the rest truncation (dwell 0.4 s, arm 0.8 s), falls
(fell_over, illegal_contact) terminal, no in-episode push. The observation noise is OFF (every
group): the calibration condition and the walk filter's fallback inputs.

Every episode is one rollout: at the restored hand-over state s_0 the env plays a FIRST action
a_0 drawn from a task-agnostic mixture (``--mix``, weights of):

  0 pi      pi(s_0), the fallback's own action (its rollout's value is V^pi(s_0))
  1 pi+0.3  pi(s_0) + N(0, 0.3^2)
  2 pi+1    pi(s_0) + N(0, 1^2)
  3 sonic   SONIC's last target (the row's last-action columns, the action INTO s_0, as sanitized:
            clamped to +-3) + N(0, 0.3^2): a nominal tracker's action
  4 unif    U(-pb, pb)^29

every one clamped to +-``--proposal-bound`` (6.5, the walk filter's proposal box; env action
units, the bridge does not clamp), then the fallback pi for the rest of the episode
(deterministic, clamped to +-3, on its actor obs with the last-action columns clamped to
+-3 -- the walk filter's ``fallback_obs_clamp``).

Output (row r = one episode, step k = 0 .. H-1, H = 150 at 3 s):

  s      [R, H, 113] fp16  the critic group at the state step k starts from, sanitized as the
                           filter's input (nan/+-inf -> 0, last action cols clamped to +-3)
  a      [R, H, 29]  fp16  the executed action (k = 0: a_0; k >= 1: pi's)
  pi0    [R, 29]     fp16  pi(s_0) (clamped to +-3), whatever a_0 was
  g      [R, H]            the task margin clamp(g_fall_noheight, +-3) the step returns (of s_{k+1},
                           before mjlab's auto-reset: safety_sb3's reward slot)
  done   [R, H]  bool      the episode ended at step k (fall, time-out or rest: all terminal,
                           as in the fallbacks' training -- no time-out bootstrap)
  v      [R, H]            the fallback's critic V(s_k) (labels.PolicyValue, the V monitor)
  len, fell, timeout [R]   valid steps; ended by a fall; ended by the time-out / rest
  first_kind [R]           the mixture component of a_0 (meta first_kinds)
  row    [R]               the bank row of s_0;   ctx [R, C] the env's realized DR draw
                           (sim_context)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get("SAFETY_SB3_PATH",
                         os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import torch  # noqa: E402

KINDS = ["pi", "pi+0.3", "pi+1", "sonic+0.3", "unif"]
A_DIM = 29


def _sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def make_quiet_env(task, num_envs, device, overrides, push_kw, critic_obs_key):
  """make_tensor with every observation group's corruption off (label_start_states.make_env
  --no-obs-noise)."""
  from robot_safety_sandbox import make_tensor, spec
  s = spec(task)
  build = s.cfg_builder

  def quiet(play: bool = False, **cfg_kw):
    cfg = build(play=play, **cfg_kw)
    for g in cfg.observations.values():
      if g is not None:
        g.enable_corruption = False
    return cfg

  s.cfg_builder = quiet
  try:
    return make_tensor(task, num_envs, device, cfg_overrides=overrides, random_push=push_kw,
                       critic_obs_key=critic_obs_key)
  finally:
    s.cfg_builder = build


def last_action_slice(env, group: str) -> tuple[slice, dict]:
  """The ``actions`` (mdp.last_action) term's columns of a concatenated obs group, and the group's
  layout ``{term: [start, size]}``."""
  lay = env._obs_layout(group)
  layout = {k: [int(v[0]), int(v[1])] for k, v in lay.items()}
  if "actions" not in lay:
    raise SystemExit(f"obs group {group!r} has no 'actions' term: {sorted(lay)}")
  st, sz, _ = lay["actions"]
  if int(sz) != A_DIM:
    raise SystemExit(f"obs group {group!r}: 'actions' is {sz}-D, expected {A_DIM}")
  return slice(int(st), int(st) + int(sz)), layout


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--policy", required=True, help="the fallback's final_model.zip")
  p.add_argument("--bank", default="data/g1_sonic_push_gfall_sysid316v6_train_bank_net.pt")
  p.add_argument("--task", default="g1_recover_noheight_v6")
  p.add_argument("--rollouts", type=int, default=120000)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--horizon-s", type=float, default=3.0)
  p.add_argument("--mix", type=float, nargs=5, default=[0.2] * 5, metavar=tuple(KINDS))
  p.add_argument("--proposal-bound", type=float, default=6.5)
  p.add_argument("--fallback-bound", type=float, default=3.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", required=True)
  a = p.parse_args(argv)
  torch.manual_seed(a.seed)
  dev = a.device

  import safety_sb3 as _ssb

  from robot_safety_sandbox.envs.g1_recover.labels import PolicyValue
  from robot_safety_sandbox.eval.policies import load_twin

  print(f"[recover-q] safety_sb3 from {os.path.dirname(os.path.abspath(_ssb.__file__))}; "
        f"G1_SYSID_DIR={os.environ.get('G1_SYSID_DIR', '(default)')} "
        f"G1_SYSID_V061_DIR={os.environ.get('G1_SYSID_V061_DIR', '(default)')}", flush=True)
  model, norm = load_twin(a.policy, dev)
  model.policy.set_training_mode(False)
  value = PolicyValue(model, a.policy, dev)
  if not value.asymmetric:
    raise SystemExit(f"{a.policy}: expected a privileged (asymmetric) critic")
  overrides = dict(dataset=a.bank, sampler="uniform",
                   weights={"handover": 1.0, "nominal": 0.0, "clean": 0.0, "pushed": 0.0,
                            "fell": 0.0, "synthetic": 0.0, "mocap": 0.0},
                   episode_length_s=float(a.horizon_s), rest_terminate=True, rest_dwell_s=0.4,
                   rest_arm_s=0.8)
  push_kw = dict(interval_s=(1.0, 2.0), duration_s=0.2, first_delay_s=(0.1, 0.4),
                 horizontal=True, max_pushes=0)
  t0 = time.time()
  env = make_quiet_env(a.task, a.num_envs, dev, overrides, push_kw, value.critic_obs_key)
  mj = env.mj
  N = env.num_envs
  dt = float(mj.step_dt)
  H = int(round(a.horizon_s / dt))
  pb, fb = float(a.proposal_bound), float(a.fallback_bound)
  ctx_all = env.sim_context()

  # --- observation layout: read, check against the policy, never assume -----------------------
  LA_A, lay_a = last_action_slice(env, env.obs_key)
  LA_C, lay_c = last_action_slice(env, value.critic_obs_key)
  a_dim_obs = int(sum(v[1] for v in lay_a.values()))
  c_dim_obs = int(sum(v[1] for v in lay_c.values()))
  pol_a = int(model.observation_space.shape[0])
  pol_c = int(value.mean.shape[0])
  if (a_dim_obs, c_dim_obs) != (pol_a, pol_c):
    raise SystemExit(f"{a.task}: actor {a_dim_obs}-D / critic {c_dim_obs}-D, but {a.policy} reads "
                     f"actor {pol_a}-D / critic {pol_c}-D")
  print(f"[recover-q] obs layout: actor ({env.obs_key}) {a_dim_obs}-D {lay_a}", flush=True)
  print(f"[recover-q] obs layout: critic ({value.critic_obs_key}) {c_dim_obs}-D {lay_c}", flush=True)
  print(f"[recover-q] last action: actor cols {LA_A.start}:{LA_A.stop}, critic cols "
        f"{LA_C.start}:{LA_C.stop} (clamped to +-{fb:g})", flush=True)
  print(f"[recover-q] {a.policy} on {a.task} x {N}: bank {a.bank}, H {H} steps, mix "
        f"{dict(zip(KINDS, a.mix))}, proposal +-{pb:g}, ctx "
        f"{None if ctx_all is None else tuple(ctx_all.shape)} ({time.time() - t0:.0f} s)",
        flush=True)
  mix = torch.tensor(a.mix, device=dev, dtype=torch.float)
  mix = mix / mix.sum()

  def sanitize(x, la):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([x[:, :la.start], x[:, la].clamp(-fb, fb), x[:, la.stop:]], dim=-1)

  # non-finite RAW observation values per column (before any sanitation), both groups
  nf_a = torch.zeros(a_dim_obs, dtype=torch.long, device=dev)
  nf_c = torch.zeros(c_dim_obs, dtype=torch.long, device=dev)

  def count_nonfinite():
    raw = env.obs_groups()
    nf_a.add_((~torch.isfinite(raw[env.obs_key].float())).sum(0))
    nf_c.add_((~torch.isfinite(raw[value.critic_obs_key].float())).sum(0))

  rows = {k: [] for k in ("s", "a", "pi0", "g", "done", "v", "len", "fell", "timeout",
                          "first_kind", "row", "ctx")}
  got, n_steps = 0, 0
  ar = torch.arange(N, device=dev)
  t1 = time.time()
  with torch.no_grad():
    count_nonfinite()
    obs = env.obs_groups()[env.obs_key].float()
    crit = env.critic_obs()
    k = torch.zeros(N, dtype=torch.long, device=dev)
    kind = torch.multinomial(mix, N, replacement=True)
    row = mj._zoo_spawn_row.clone()
    S = A = None
    P0 = torch.zeros(N, A_DIM, dtype=torch.float16, device=dev)
    G = torch.zeros(N, H, device=dev)
    D = torch.zeros(N, H, dtype=torch.bool, device=dev)
    V = torch.zeros(N, H, device=dev)
    while got < a.rollouts:
      ao, co = sanitize(obs, LA_A), sanitize(crit, LA_C)
      if S is None:
        S = torch.zeros(N, H, co.shape[1], dtype=torch.float16, device=dev)
        A = torch.zeros(N, H, A_DIM, dtype=torch.float16, device=dev)
      pi = torch.clamp(model.policy._predict(norm(ao), deterministic=True), -fb, fb)
      v = value(ao, co).float().reshape(N)
      first = k == 0
      noise = torch.randn_like(pi)
      cand = torch.stack([pi, pi + 0.3 * noise, pi + 1.0 * noise,
                          co[:, LA_C] + 0.3 * noise,
                          (torch.rand_like(pi) * 2 - 1) * pb], dim=1)         # [N, 5, 29]
      a0 = torch.clamp(cand[ar, kind], -pb, pb)
      act = torch.where(first.unsqueeze(-1), a0, pi)
      P0 = torch.where(first.unsqueeze(-1), pi.half(), P0)
      kk = k.clamp_max(H - 1)
      S[ar, kk] = co.half()
      A[ar, kk] = act.half()
      V[ar, kk] = v
      obs, g, dones, timeouts, _l = env.step_tensor(act)
      count_nonfinite()
      crit = env.critic_obs()
      n_steps += 1
      d = dones.bool()
      G[ar, kk] = g.float()
      D[ar, kk] = d
      if bool((~d & (k == H - 1)).any()):
        raise RuntimeError("an episode passed the horizon without a time-out")
      if bool(d.any()):
        ids = d.nonzero(as_tuple=False).squeeze(-1)
        take = ids[: max(0, a.rollouts - got)]
        if take.numel():
          L = k[take] + 1
          m = torch.arange(H, device=dev).unsqueeze(0) < L.unsqueeze(1)
          rows["s"].append((S[take] * m.unsqueeze(-1)).cpu())
          rows["a"].append((A[take] * m.unsqueeze(-1)).cpu())
          rows["pi0"].append(P0[take].cpu())
          rows["g"].append((G[take] * m).cpu())
          rows["done"].append((D[take] & m).cpu())
          rows["v"].append((V[take] * m).cpu())
          rows["len"].append(L.cpu())
          to = timeouts.bool()[take]
          rows["fell"].append((~to).cpu())
          rows["timeout"].append(to.cpu())
          rows["first_kind"].append(kind[take].cpu())
          rows["row"].append(row[take].cpu())
          if ctx_all is not None:
            rows["ctx"].append(ctx_all[take].float().cpu())
          got += int(take.numel())
        k = torch.where(d, torch.zeros_like(k), k + 1)
        kind = torch.where(d, torch.multinomial(mix, N, replacement=True), kind)
        row = torch.where(d, mj._zoo_spawn_row, row)
        G[d] = 0.0
        D[d] = False
      else:
        k = k + 1
      if n_steps % 500 == 0:
        print(f"[recover-q] step {n_steps}: {got}/{a.rollouts} rollouts "
              f"({time.time() - t1:.0f} s)", flush=True)

  out = {kk_: torch.cat(v) for kk_, v in rows.items() if v}
  R = int(out["len"].numel())
  nf_a_d = {int(i): int(nf_a[i]) for i in nf_a.nonzero(as_tuple=False).flatten().tolist()}
  nf_c_d = {int(i): int(nf_c[i]) for i in nf_c.nonzero(as_tuple=False).flatten().tolist()}
  git = subprocess.run(["git", "-C", _ZOO, "rev-parse", "HEAD"], capture_output=True,
                       text=True).stdout.strip()
  out["meta"] = {"source": "recover_q_collect.py (mjc316 v0.6 port)",
                 "policy": os.path.abspath(a.policy),
                 "policy_sha256": _sha(a.policy), "bank": os.path.abspath(a.bank),
                 "task": a.task, "H": H, "step_dt": dt, "horizon_s": a.horizon_s,
                 "first_kinds": KINDS, "mix": a.mix, "proposal_bound": pb, "fallback_bound": fb,
                 "obs_noise": False, "push": False, "rows": R, "seed": a.seed,
                 "s_dim": int(out["s"].shape[-1]),
                 "ctx_dim": int(out["ctx"].shape[1]) if "ctx" in out else 0,
                 "actor_obs_dim": a_dim_obs, "critic_obs_dim": c_dim_obs,
                 "obs_layout": {"actor": lay_a, "critic": lay_c},
                 "actor_last_action_cols": [LA_A.start, LA_A.stop],
                 "critic_last_action_cols": [LA_C.start, LA_C.stop],
                 "sanitize": {"group": value.critic_obs_key, "dim": c_dim_obs,
                              "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                              "clamp_cols": [[LA_C.start, LA_C.stop]], "clamp": fb},
                 "actor_sanitize": {"group": env.obs_key, "dim": a_dim_obs,
                                    "nan_to_num": {"nan": 0.0, "posinf": 0.0, "neginf": 0.0},
                                    "clamp_cols": [[LA_A.start, LA_A.stop]], "clamp": fb},
                 "nonfinite_raw_obs": {"actor": nf_a_d, "critic": nf_c_d,
                                       "obs_rows_checked": (n_steps + 1) * N},
                 "unique_bank_rows": int(torch.unique(out["row"]).numel()),
                 "safety_sb3": os.path.dirname(os.path.abspath(_ssb.__file__)),
                 "fell_frac": float(out["fell"].float().mean()),
                 "env_steps": n_steps * N, "wall_s": time.time() - t1, "git": git}
  print(f"[recover-q] non-finite RAW obs values: actor {nf_a_d or 'none'}, critic "
        f"{nf_c_d or 'none'} over {(n_steps + 1) * N} obs rows; {out['meta']['unique_bank_rows']} "
        f"distinct bank rows", flush=True)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  torch.save(out, a.out)
  for i, nm in enumerate(KINDS):
    m = out["first_kind"] == i
    print(f"[recover-q] first {nm:10s}: {int(m.sum()):6d} rollouts, fell "
          f"{float(out['fell'][m].float().mean()):.3f}, mean len "
          f"{float(out['len'][m].float().mean()) * dt:.2f} s", flush=True)
  print(f"[recover-q] {R} rollouts, fell {out['meta']['fell_frac']:.3f} -> {a.out} "
        f"({time.time() - t1:.0f} s)", flush=True)
  env.close()


if __name__ == "__main__":
  main()
