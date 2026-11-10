# Which privileged inputs does SHAC's world model need? Runbook and results

This page records one line of work: which inputs the learned world model behind
the SHAC safety monitor needs, so that the monitor can run on the real G1. It
covers the setup (software, environment, observations, domain randomization,
data), the three experiments (a no-pelvis-height retrain, world-model
distillation into smaller states, and the calibration noise floor), the
commands to reproduce each one, the results, and the traps hit along the way.

It builds on [SHAC vs SafetyPPO from SONIC's post-kick states](shac-ppo-sonic.md),
which trained the original SHAC policies (`runs/shac_vs_ppo_sonic`) and
documents the start-state harvest in more detail.

**Status (2026-11-09).** Everything below has run on the JHU ARCH cluster and
the results are final. **None of it is committed yet.** The work sits
uncommitted on branch `feat/sonic-push-bench-mjlab` (on top of `a09ed3e`); the
files are listed in [§11](#11-where-everything-is). Commit them before relying
on this page from another checkout.

## 0. Results in one paragraph

Pelvis height is not needed: retraining SHAC without it (margin, critic and
world model) moves the 1 s tube from 0.115 to 0.138, within the seed spread,
with unchanged false alarms (0.5 %) and policy survival 97.3 % vs 98.1 %. The
foot terms (height, air time, contact flags, contact forces) are not needed: a
world model distilled without them has the same tube as one distilled with
them (1 s: 0.171 vs 0.173). Base linear velocity **is** needed by a
single-frame model: without it the 0.4 s tube is 2.4× wider (0.449 vs 0.186)
and at 1 s the monitor would flag 71 % of the episodes in which the robot does
not fall, against 5 % with it. Distillation itself costs about 25 % of tube
width (1 s: 0.17 vs 0.14 for the jointly trained model). Every world model here
still takes the simulator's 33-D domain-randomization draw as an input; that
has not been ablated.

## 1. The question and the design

The SHAC fallback (`safety_sb3.shac`) learns a deterministic world model of the
G1 and trains its actor by backpropagating through it. The same model powers
the online safety monitor of `eval_sonic_qcbf.py`. From the current state, it
imagines "SONIC's action now, then the fallback π", and SONIC's action goes
through only if the predicted margin minus a conformal tube width stays ≥ 0.
The original model's state was the 114-D privileged critic group. That group
includes quantities a real robot does not measure directly: pelvis height, foot
heights, contact flags and forces, and the base linear velocity.

Three experiments, run in order:

| Stage | What changes | What stays | Output |
|---|---|---|---|
| A. No-height retrain | Pelvis height out of the margin, the critic and the world-model state (114 → 113-D). Policy **and** world model retrained from scratch. | Everything else of `runs/shac_vs_ppo_sonic`'s SHAC arm | `runs/shac_noheight/` |
| B. Distillation | The no-height policy is **frozen**. A fresh world model is fitted to its data with a smaller state: 113-D (control), 101-D (no foot terms), 98-D (no foot terms, no base_lin_vel). | Policy, recipe, data, sim context | `runs/shac_student/` |
| C. Calibration | Each world model gets a conformal tube on the same 2,048 pushed episodes. The teacher's own calibration is repeated twice as a noise floor. | Eval protocol, seeds | `runs/shac_student/tube_compare*.log` |

Why distil instead of retraining SHAC on the smaller state: in SHAC the critic
reads decoded world-model states (`shac.py`, `_value(wm.decode(z))`), so a
smaller world-model state would also shrink the critic and change the policy.
The privileged model is only a problem for the **online** monitor; training
happens in simulation anyway. Freezing the policy isolates the world model.

The 113-D control student uses the same recipe as the smaller students. So
"control vs student" isolates the dropped observations, and "teacher vs
control" isolates the cost of the distillation recipe.

## 2. Software environment

| What | Where / value |
|---|---|
| Cluster | JHU ARCH. Sessions run on a login node **without a GPU**. All GPU work goes through SLURM. |
| SLURM account | `-A hhu49_alliance1` (QOS jhu). The default account `hhu49` is scavenger QOS and preemptible: always pass `-A`. |
| GPU partitions used | `rtx6000` (RTX PRO 6000 Blackwell), `l40s`, `h100`, `a100`, `h200`. All have a 3-day limit. Listing several (`-p rtx6000,l40s,h100,a100,h200`) starts jobs much sooner. |
| Python | conda env `mjlab`: `source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab` |
| This repository | `/home/jhu/yzeng52/robot-safety-sandbox` (also visible as `/weka/home/jhu/yzeng52/...`), branch `feat/sonic-push-bench-mjlab` |
| safety-stable-baselines | `../safety-stable-baselines`, editable install of `safety_sb3`, branch **`feat/supermimic`** (commit `f92a02f`, "Add SHAC"). `main` has no SHAC. If `train_model_based.py` says it "does not export 'SafetySHAC1P'", check this branch. |
| GEAR-SONIC release | `~/gear_sonic_hf` on the cluster. Pass `--sonic-dir $HOME/gear_sonic_hf` (the code's default is the workstation's `~/Documents/gear_sonic_hf`). |
| Job runner | `examples/run_jobs.sh JOBS --only N` runs job N of a jobs file. It skips a job whose `# creates: PATH` outputs exist, or whose `--out` holds a `final_model.zip`. Resubmitting a whole array therefore reruns only what is missing. `--list` shows done / todo. |
| Generic SLURM driver | `examples/slurm_jobs.sbatch JOBS` (array task N = job N). Experiment-specific drivers sit next to their jobs files. |
| Rendering | Every GPU script exports `MUJOCO_GL=egl` and points `__EGL_VENDOR_LIBRARY_FILENAMES` at the NVIDIA EGL vendor file. |

Login-node habits:

- Importing torch and the sandbox there takes minutes. `compare_wm_tubes.py`
  over five arms took more than 5 minutes there. Run it in the background (`&`,
  `nohup`), or on a GPU node at the end of a job (as the drivers below do).
- CPU-only work (pytest, reading checkpoints, the comparison scripts) is fine
  on the login node.

Throughput (measured):

| Job | Speed | Wall time |
|---|---|---|
| SHAC training, 1.5B steps, 4096 envs | ~31k fps on RTX PRO 6000 | ~13 h |
| World-model-only fit (student), 300M steps | ~44k fps | 1 h 50 – 2 h 10 |
| Calibration, 2,048 episodes, 256 envs | — | ~3 min (+ queue) |

## 3. The SHAC environment

### 3.1 Tasks

All tasks share g1_recover's world: flat ground, the twist command pinned to
zero, 50 Hz control (0.02 s), physics at 0.005 s, and the same terminations
and domain randomization. They differ only in the margin and in the critic
group, which is the world-model state.

| Task (train / eval twin) | Margin | Critic group = world-model state | Used by |
|---|---|---|---|
| `g1_recover` / `g1_recover_lafan` | `g_fall` (pelvis height, tilt, non-foot contact) | 114-D | the original SHAC (`runs/shac_vs_ppo_sonic`) |
| `g1_recover_noheight` / `g1_recover_lafan_noheight` | `g_fall_noheight` (tilt, non-foot contact) | 113-D: no `pelvis_height` | stage A teacher; the 113-D control student |
| `g1_recover_noheight_nofoot` / `g1_recover_lafan_noheight_nofoot` | `g_fall_noheight` | 101-D: actor terms + `base_lin_vel` | the 101-D student |
| `g1_recover_noheight_nofoot_nolinvel` / `g1_recover_lafan_noheight_nofoot_nolinvel` | `g_fall_noheight` | 98-D: exactly the actor terms | the 98-D student |

The train task is where SHAC trains and the world model is fitted. The `lafan`
twin is what `eval_sonic_qcbf.py` builds, because SONIC needs the LAFAN clip
library. The critic groups are built by `g1_recover_env_cfg` kwargs
(`robot_safety_sandbox/envs/g1_recover/env_cfg.py`):

| kwarg | Default | When `False` |
|---|---|---|
| `critic_pelvis_height` | True | no `pelvis_height` term |
| `critic_foot_terms` | True | drops `CRITIC_FOOT_TERMS` = foot_height, foot_air_time, foot_contact, foot_contact_forces |
| `critic_base_lin_vel` | True | drops `base_lin_vel` |

The builders `g1_recover_noheight_env_cfg`, `..._nofoot_env_cfg` and
`..._nofoot_nolinvel_env_cfg` fix these flags and refuse to have them turned
back on. Registration is in `robot_safety_sandbox/tasks/g1_recover.py`. Tests:
`pytest tests/test_g1_margins.py -k "noheight or nofoot or nolinvel"`.

### 3.2 Observations

**Actor (π's input), 98-D, one frame, no history.** Noise is uniform additive
and on only in the actor group (`enable_corruption=True`):

| # | Term | Dims | Noise | Notes |
|---|---|---|---|---|
| 0 | base_ang_vel | 3 | ±0.2 rad/s | IMU gyro, pelvis frame |
| 1 | projected_gravity | 3 | ±0.05 | gravity in the pelvis frame. The margin's tilt term reads it. |
| 2 | command | 3 | — | twist command, pinned to **zero** in g1_recover |
| 3 | phase | 2 | — | gait phase. Reads **zero** because the command is zero. |
| 4 | joint_pos | 29 | ±0.01 rad | relative to the default pose. Includes the encoder bias (§3.4). |
| 5 | joint_vel | 29 | ±1.5 rad/s | |
| 6 | actions | 29 | — | previous action (joint position targets) |

`command` and `phase` are constant zeros, so 93 of the 98 dims carry
information.

**Critic / world-model state.** The critic group is the actor terms
**noise-free** (`enable_corruption=False`; mjlab drops the terms' noise), plus
privileged terms:

| Term | Dims | In 114-D | 113-D | 101-D | 98-D |
|---|---|---|---|---|---|
| the 7 actor terms | 98 | ✓ | ✓ | ✓ | ✓ |
| base_lin_vel (pelvis frame) | 3 | ✓ | ✓ | ✓ | — |
| foot_height (world-frame foot sites) | 2 | ✓ | ✓ | — | — |
| foot_air_time | 2 | ✓ | ✓ | — | — |
| foot_contact | 2 | ✓ | ✓ | — | — |
| foot_contact_forces | 6 | ✓ | ✓ | — | — |
| pelvis_height | 1 | ✓ | — | — | — |

`base.py world_model_spec` rebuilds π's observation from a world-model state
by matching each actor term **by name** to a same-size, noise-free twin in the
critic group, and adds the actor's sensor noise in imagination. Any critic group
that contains the 7 actor terms works, including one that is only those terms.

### 3.3 Margin and terminations

`g_fall_noheight` (`envs/g1_stabilize/env_cfg.py`) is the min of:

- **tilt**: `(up − cos 50°) / (1 − cos 50°)`, where `up = −projected_gravity_b[2]`
  of the root link. The root link is the **pelvis**, so this is pelvis tilt,
  not torso tilt. It is 0 at 50°.
- **non-foot contact**: `(10 N − peak non-foot ground force) / 10 N`.

`g_fall` adds a pelvis-height term (`(z − 0.45 m) / 0.25`). `compose` clamps
the margin to ±3. The episode terminates at a **70° tilt** or a **10 N
non-foot contact**; neither reads the height. Episodes last 3 s with **rest
truncation**: the episode ends as a truncation after the robot dwells 0.4 s in
the stance set, armed after 0.8 s.

### 3.4 Domain randomization and the sim context

Three `startup` events, drawn **once per simulated robot** when the simulation
is built and fixed for the whole run:

| Event | Field | Dims | Range |
|---|---|---|---|
| `foot_friction` | `geom_friction` (shared by both feet) | 1 | U(0.3, 1.2), absolute |
| `base_com` | `body_ipos` of `torso_link` | 3 | x, y ±0.025 m; z ±0.03 m, added |
| `encoder_bias` | `entity.data.encoder_bias` | 29 | ±0.015 rad per joint. The observed `joint_pos` is true + bias (`joint_pos_biased`). |

The only `reset` event is `recover_spawn`, which restores a bank row
(§4.1). **There is no in-episode push in training.** The kick lives in the
start states. A push in the middle of an episode would be a force the world
model cannot see coming, so model-based runs keep it out of the dynamics.

**The sim context.** `base.py sim_context()` returns these 33 realized values
per env (1 + 3 + 29, deduplicated, minus constants MuJoCo derives from them).
The world model takes them as a normalized **conditioning input** at every
step (`WorldModel.encode_context`). It does not predict them. In training and in
calibration it gets the simulator's true values. **Every world model on this
page uses it.** It is privileged: a real robot does not know its friction, CoM
offset or encoder biases.

## 4. Data

### 4.1 Start states (the bank)

Each training episode starts from a row of
`data/g1_sonic_push_gfall_arch_train_bank_net.pt`: 199,102 rows, drawn
uniformly (`weights: {handover: 1.0, others: 0}`, `sampler: uniform`). A row is
a full robot state recorded 0 / 0.1 / 0.2 / 0.3 s after SONIC was kicked:

1. SONIC (`envs/g1_recover/sonic.py`) tracks a LAFAN **train** clip from a
   library frame.
2. 0.5–1.5 s in, the pelvis velocity gets a horizontal kick, |Δv| ~ U(0, 5)
   m/s in a random direction.
3. Rows are kept only if SONIC is still up at that moment, under
   g1_recover_lafan's fall terminations. The ViaLasso set K is not involved.
4. η-net thinning with η = 0.1 (`dedup_start_bank.py`), which keeps almost
   everything.

The cluster bank is a rebuild of the workstation's
`data/g1_sonic_push_gfall_train_bank_net.pt` (199,103 rows), which is not on
the cluster. Commands:

```bash
# the SONIC harvest (cluster GPU; configs/sweeps/shac_vs_ppo_sonic/policy_eval_states.txt line 16)
python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan \
    --library data/g1_lafan1_clips_train.pt --sonic-dir $HOME/gear_sonic_hf \
    --episodes 50000 --seed 0 --bank-delays-s --out data/g1_sonic_push_gfall_arch_train.pt
# the bank + thinning (configs/sweeps/shac_noheight/data_jobs.txt; prep_and_smoke.sh runs it)
python examples/harvest_g1_sonic_trajectories.py --from-traj data/g1_sonic_push_gfall_arch_train.pt \
    --bank-delays-s 0 0.1 0.2 0.3 --bank-out data/g1_sonic_push_gfall_arch_train_bank.pt
python examples/dedup_start_bank.py --bank data/g1_sonic_push_gfall_arch_train_bank.pt
```

Caveat: clip libraries built on the workstation carry `meta["csv_dir"]`
pointing at `/home/yuanhong/...`. If a harvest fails with FileNotFoundError,
rewrite it to the repo's `data/lafan1_g1/`.

### 4.2 World-model training data

SHAC collects on-policy into `SequenceReplay`, a device-resident ring of
**raw** world-model states (`replay_size: 500` slots × 4096 envs, ~1.1 GiB).
The collection policy is π's mean action plus N(0, 0.1²) noise
(`action_noise_std: 0.1`).

- **Teacher (stage A):** the replay follows the policy as it learns over
  1.5B steps, so it includes the early, falling policy.
- **Students (stage B):** the replay holds only the **finished, frozen**
  teacher policy (plus action noise), for 300M steps.

The world model is fitted on contiguous windows of `wm_window: 8` steps.

### 4.3 Calibration data

`eval_sonic_qcbf.py --conditions sonic wmcalib:clean` with `--suites push`
generates the episodes every tube is fitted on:

1. Spawn on a held-out LAFAN clip (`data/g1_lafan1_clips_test.pt`, excluding
   `fallAndGetUp` and `sprint`). SONIC tracks it.
2. At onset 0.02 s, the pelvis gets a kick |Δv| ~ U(0, 4) m/s.
3. SONIC keeps control until a **decision step** drawn U{0 … 1 s} after the
   kick (`--calib-max-delay-s 1.0`, saved as `calib_decision`).
4. At the decision step, from the **true** world-model state and the true sim
   context, the world model imagines "u_SONIC for one step, then π" for 3 s,
   noise-free (`--no-obs-noise`). The real robot runs the same
   "u_SONIC, then π" for 3 s. Both margin sequences are recorded.

The settings are `--episodes 2048 --num-envs 256 --seed 1`, with
`--dv-bins 0 1 2 3 4`. About 1,875–1,878 of the 2,048 episodes per run have a
valid record.

**Common random numbers.** The episode lists (spawn rows, onsets, kicks) come
from seeded generators keyed by `--seed`, and each episode runs in a fixed env
slot. So every run with `--seed 1` sees the same episodes, robots and kicks.
**But the simulator is not bit-reproducible run to run**, even on the same GPU
type. SONIC alone survives 1,243–1,252 of 2,048 episodes across runs with the
same seed. After a kick, tiny float differences become different falls, so two
calibration runs never record identical real trajectories (§10).

## 5. The world model

`safety_sb3/world_model.py` (branch `feat/supermimic`):

- **Dynamics.** Deterministic and residual in normalized coordinates
  `z = (s − mean)/std`: `z' = z + delta_scale · net(z, a/a_scale, c)`.
  - The state statistics are fitted once from replay data and then
    **frozen**.
  - `delta_scale` is the per-dim std of the one-step change.
  - The last layer starts at zero, so an untrained model predicts "nothing
    moves".
- **Previous-action dims are copied, not predicted.** The 29
  previous-action dims (`exact_index`) are written from `a_t` exactly and carry
  no loss. The model therefore predicts 84 / 72 / 69 dims for the 113 / 101 /
  98-D states.
- **Heads**, which read `(z_t, z_{t+1}, c)` and not the action:
  - `g`: margin, given the step continues
  - `g_term`: margin, given it terminates
  - `done_logit`
- **Recipe**:

  | Setting | Value |
  |---|---|
  | `wm_net` | 1024×3, LayerNorm, SiLU |
  | `wm_head_net` | 256×2 |
  | `lr` | 2e-4 (the world model uses SHAC's LR; `--wm-lr` is unset), cosine down to 3 % over `lr_decay_iters` |
  | `wm_window` | 8 (k-step autoregressive fit) |
  | `wm_batch` | 2048 |
  | `wm_gradient_steps` | 10 per iteration |
  | `tf32` | on |
- **Imagination**: horizon 32 for SHAC's actor. In the monitor, up to 3 s
  (150 steps) during calibration.

## 6. Stage A: the no-pelvis-height teacher

Config `configs/g1_recover_sonic_noheight_safetyshac1p.yaml`. It is the
original SHAC arm (`configs/g1_recover_sonic_safetyshac1p.yaml`) verbatim, but
for the task (`g1_recover_noheight`) and the cluster bank.

| Setting | Value |
|---|---|
| Steps | 1.5B per seed, 4096 envs, seeds 0 / 1 / 2 |
| Policy | MLP 512-256-128, privileged critic, gamma 0.99 |
| Penalties | action rate 0.05, magnitude 0.005 |
| SHAC | horizon 32, `wm_warmup_iters` 1000 (~10M model-only steps), `actor_delay_iters` 200, `lr_decay_iters` 146,500 (= steps / 10,240) |
| Checkpoints | every 50M |

```bash
cd ~/robot-safety-sandbox && mkdir -p logs/slurm
# 0. bank + smoke test (data_jobs.txt, a short SHAC run, the report pipeline on 64 episodes)
sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100 --gres=gpu:1 --cpus-per-task=32 --mem=96G \
    --time=02:00:00 --job-name=noheight_prep --output=logs/slurm/%x_%j.out \
    configs/sweeps/shac_noheight/prep_and_smoke.sh
# 1. training, 3 seeds (~13 h each)
sbatch -A hhu49_alliance1 -p rtx6000 --array=1-3 --time=20:00:00 --mem=64G \
    --job-name=shac_noheight examples/slurm_jobs.sbatch configs/sweeps/shac_noheight/jobs.txt
# 2. per seed: calibration (2048 eps) + tube fit + filter evaluation (1024 eps); then the report
sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --array=1-3 --gres=gpu:1 --cpus-per-task=32 \
    --mem=96G --time=01:30:00 --job-name=wmreport_noheight --output=logs/slurm/%x_%A_%a.out \
    configs/sweeps/shac_noheight/wmreport_seed.sh
examples/run_jobs.sh configs/sweeps/shac_noheight/wmreport_jobs.txt --only 10   # if the last seed did not
```

Outputs:

- runs: `runs/shac_noheight/shac_s{0,1,2}/g1_recover_noheight_safetyshac1p/final_model.zip`
- calibrations: `runs/shac_noheight/wmreport/calib/noheight_s*/{push.pt,min_tube.json}`
- report: `runs/shac_noheight/wmreport/report.{json,log}`

The filter evaluation runs these conditions:

```
sonic fallback rollout:1 wmmin:0.4:0.95 wmmin:1:0.95 wmmin:3:0.95 wmmin:1:0.95+dwell1 wmmin:1:0.95+dwell2 wmmin:1:0.95+latch
```

## 7. Stage B: distilling student world models

### 7.1 Recipe

Config: `configs/g1_recover_sonic_noheight_student_wm.yaml`. It is the
teacher's config with these changes:

| Setting | Student | Why |
|---|---|---|
| `--load <teacher final_model.zip>` | yes | the teacher's actor and its actor obs normalizer. The critic does not load (shape mismatch), and is never used. |
| `--no-load-world-model` | yes | **new flag** in `examples/train_model_based.py`: start a fresh world model even when its shapes match (needed by the 113-D control) |
| `wm_warmup_iters` | 1e9 (past the end) | SHAC never imagines, so the actor and critic **never update**. Only the world model trains. |
| `actor_delay_iters` | 0 | |
| `norm_freeze_steps` | 1e15 (never) | the actor's obs normalizer stays the teacher's |
| `steps` | 300M (vs 1.5B) | `lr_decay_iters` 29,297 |
| `checkpoint_every` / `video_interval` | 100M / never | |
| `--task` | the arm's task (§3.1) | 113-D control: `g1_recover_noheight`; 101-D: `g1_recover_noheight_nofoot`; 98-D: `g1_recover_noheight_nofoot_nolinvel` |

Everything else (world-model architecture, LR, bank, env, action noise,
replay) is the teacher's.

**`check_frozen.py`** (`configs/sweeps/shac_student/`) runs after each
training job. It:

1. confirms the actor tensors are **bit-identical** to the teacher's;
2. confirms the world-model state dim;
3. checks the actor obs normalizer moved by at most one reset batch:
   - the count grew by ≤ `--max-fold` (= num_envs);
   - the mean moved < 1e-3 std, and the variance < 1e-3 relative;
4. then writes the teacher's exact `obs_mean / obs_var / count` into the
   student's `tensornormalize.pt`.

After step 4, the student checkpoint acts exactly like the teacher in every
eval. Why step 3 is needed: SB3 resets the env in `_setup_learn` before the
freeze callback runs, so one batch is folded into the statistics.

### 7.2 Jobs

`configs/sweeps/shac_student/`:

| File | What |
|---|---|
| `jobs.txt` | 9 training jobs: 1–3 `nofoot_s*` (101-D), 4–6 `full_s*` (113-D control), 7–9 `nolinvel_s*` (98-D), seed s ↔ teacher `shac_s{s}` |
| `calib_jobs.txt` | 18 lines: for job N, lines 2N−1 / 2N = calibration + tube fit, with the teacher's calibration flags verbatim except policy, tag, `--task`, `--out` |
| `student_seed.sh` | array driver for task N: train → `check_frozen.py` → calibrate → fit. The last of tasks 1–6 writes `runs/shac_student/tube_compare.*`; the last of 7–9 writes `tube_compare_nolinvel.*`. |
| `smoke.sh`, `smoke_nolinvel.sh` | ~10 min end-to-end checks: 3M-step student, state-dim grep, "world model NOT restored" grep, no actor-loss lines, `check_frozen`, 64-episode calibration + fit + compare |
| `teacher_rerun_jobs.txt` | the noise floor (§8.2) |
| `check_frozen.py` | §7.1 |

```bash
P=rtx6000,l40s,h100,a100,h200
# smoke first; the array waits on it (afterok)
SM=$(sbatch --parsable -A hhu49_alliance1 -p $P --gres=gpu:1 --cpus-per-task=16 --mem=64G \
     --time=01:00:00 --job-name=student_smoke --output=logs/slurm/%x_%j.out \
     configs/sweeps/shac_student/smoke.sh)
sbatch -A hhu49_alliance1 -p $P --dependency=afterok:$SM --array=1-9 --gres=gpu:1 \
    --cpus-per-task=16 --mem=64G --time=04:00:00 --job-name=shac_student \
    --output=logs/slurm/%x_%A_%a.out configs/sweeps/shac_student/student_seed.sh
```

Outputs:

- students: `runs/shac_student/{nofoot,full,nolinvel}_s{0,1,2}/<task>_safetyshac1p/final_model.zip`
- calibrations: `runs/shac_student/calib/{nofoot,full,nolinvel}_s*/`
- logs: `logs/shac_student_jobs/`, `logs/shac_student_calib_jobs/`, `logs/slurm/shac_student_*`

## 8. Stage C: tubes, the noise floor, the comparison

### 8.1 The tube

`examples/calibrate_wm_min_tube.py --calib <push.pt> --tag <tag>` writes
`min_tube.json`. For horizon H ∈ {0.1, 0.4, 1, 3} s:

- predicted worst margin `Ĝ = min_{j≤H} ĝ_j`;
- real worst margin `G = min_{j≤H} g_j` (a fall holds its value);
- score `r = Ĝ − G`.

`r > 0` means reality was worse than predicted.

The fit uses split conformal prediction. On the even rows, `min_eps[c][H]`
is the ⌈(n+1)c⌉-th smallest score, so `P(G ≥ Ĝ − ε) ≥ c`; this is the one-sided
bound the filter uses. On the odd rows it reports:

- coverage, and coverage among real falls;
- **fall recall** and **false alarm**: the fraction of non-falling episodes
  with `Ĝ − ε < 0`, i.e. episodes the monitor would flag although the robot
  does not fall within H;
- score quantiles.

The final tables are refit on all rows. A tube can be narrow while the
false-alarm rate is high: the one-sided width only guards against
over-optimism. **Always read both.**

### 8.2 The noise floor

Since the simulator is not reproducible (§4.3), each arm's tube is a quantile
over its own run's trajectories, so arms are **not paired**. To see what a
difference is worth, the teacher's own calibration was rerun twice per seed,
verbatim but for `--tag` and `--out`:

```bash
sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100,a100,h200 --array=1-6 --cpus-per-task=16 \
    --mem=64G --time=01:00:00 --job-name=teacher_rerun examples/slurm_jobs.sbatch \
    configs/sweeps/shac_student/teacher_rerun_jobs.txt
```

→ `runs/shac_student/calib/teacher_rerun_{a,b}_s{0,1,2}/`. The three
teacher runs agree within ±0.004 on the mean width and ≤ 0.007 per seed
(0.1–1 s, c = 0.95). The seed-to-seed spread of world-model training is
larger (~0.03), so read differences against both.

### 8.3 The comparison

`examples/compare_wm_tubes.py --arm NAME DIR... [--arm ...] --out X.json`. The
k-th DIR of each arm must be seed k. Per arm it reports:

- the mean over seeds, and per seed, of `min_eps` and the two-sided width;
- the split-half test: coverage, coverage among falls, recall, false alarm;
- the score quantiles;
- world-model accuracy (`wm_filter_report.accuracy`: |ĝ − g| at 1 / 10 steps,
  0.4 / 1 / 3 s, median |r|, median signed r);
- the **score by decision delay** after the kick (0–0.1 / 0.1–0.3 / 0.3–0.6 /
  0.6–1 s, pooled over seeds).

`real_g_identical` checks whether two runs recorded the same real margins.
Given §4.3, expect False.

```bash
A="--arm teacher runs/shac_noheight/wmreport/calib/noheight_s0 runs/shac_noheight/wmreport/calib/noheight_s1 runs/shac_noheight/wmreport/calib/noheight_s2"
python examples/compare_wm_tubes.py $A \
    --arm teacher_rerun runs/shac_student/calib/teacher_rerun_a_s{0,1,2} \
    --arm full_113 runs/shac_student/calib/full_s{0,1,2} \
    --arm nofoot_101 runs/shac_student/calib/nofoot_s{0,1,2} \
    --arm nolinvel_98 runs/shac_student/calib/nolinvel_s{0,1,2} \
    --out runs/shac_student/tube_compare_nolinvel.json   # run in the background on the login node
# pelvis height: the original 114-D teacher vs the no-height one
python examples/compare_wm_tubes.py --arm original_114 runs/shac_vs_ppo_sonic/wmreport/calib/shac_s{0,1,2} \
    --arm noheight_113 runs/shac_noheight/wmreport/calib/noheight_s{0,1,2} \
    --out runs/shac_student/tube_compare_height.json
```

## 9. Results

All numbers are means over 3 seeds unless marked, at c = 0.95, from the files
in `runs/shac_student/` and `runs/*/wmreport/`.

### 9.1 Tube width (`min_eps`)

| Arm | State | 0.1 s | 0.4 s | 1 s | 3 s |
|---|---|---|---|---|---|
| original teacher (co-trained, margin `g_fall`) | 114-D | 0.067 | 0.133 | 0.115 | 0.031 |
| no-height teacher (co-trained) | 113-D | 0.065 | 0.140 | 0.138 | −0.010 |
| teacher reruns a / b | 113-D | 0.064 / 0.064 | 0.140 / 0.138 | 0.134 / 0.135 | −0.015 / −0.013 |
| distilled control | 113-D | 0.135 | 0.223 | 0.173 | 0.138 |
| distilled, no foot terms | 101-D | 0.137 | 0.186 | 0.171 | 0.156 |
| distilled, no foot terms, no base_lin_vel | 98-D | 0.142 | 0.449 | 0.576 | 0.255 |

Per seed at 1 s:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| original teacher | 0.096 | 0.138 | 0.112 |
| no-height teacher | 0.110 | 0.142 | 0.160 |
| control (113-D) | 0.181 | 0.195 | 0.142 |
| 101-D | 0.169 | 0.172 | 0.171 |
| 98-D | 0.183 | 0.131 | 1.414 |

At 0.4 s, 101-D is tighter than 113-D in all three seeds:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 101-D | 0.177 | 0.204 | 0.177 |
| 113-D | 0.196 | 0.234 | 0.240 |

At c = 0.99, every arm sits at ~1.9–2.4: falls that no model foresees.

### 9.2 What the monitor would do (split-half test, c = 0.95)

| Arm | False alarm 0.4 s | False alarm 1 s | Fall recall 1 s | Median score r, 1 s |
|---|---|---|---|---|
| original teacher | 0.006 | 0.005 | 0.926 | −0.032 |
| no-height teacher | 0.007 | 0.005 | 0.910 | −0.032 |
| control 113-D | 0.008 | 0.054 | 0.947 | −0.055 |
| 101-D | 0.020 | 0.046 | 0.946 | −0.027 |
| 98-D | 0.378 (0.076 / 0.059 / 1.00) | 0.710 (0.523 / 0.607 / 1.00) | 0.982 | −0.340 (−0.481 / −0.490 / −0.051) |

In brackets: per seed. The median score is the median of r = Ĝ − G on the
calibration half, averaged over seeds; negative means the model predicts a
lower margin than reality, i.e. it is pessimistic.

At 3 s, the jointly trained teachers are very pessimistic: their 3 s
predictions include many falls that do not happen. The median score is −1.30
for the no-height teacher and −0.43 for the original, and the no-height
teacher's false-alarm rate is 0.76. The distilled 101-D model is not (median
−0.04, false alarm 0.23).

### 9.3 Survival with the filter (stage A; 3 × 1,024 episodes, kick bins 0–1 / 1–2 / 2–3 / 3–4 m/s / overall)

| Condition | Original (114-D) | No-height (113-D) |
|---|---|---|
| SONIC alone | 66.9 | 66.5 |
| SHAC alone | 98.1 | 97.3 |
| exact-sim filter 1 s | 98.6 | 98.4 |
| WM tube 1 s | 66.5 | 66.4 |
| WM tube 1 s, latch 1 s / 2 s / forever | 84.8 / 88.2 / 90.1 | 84.5 / 87.0 / 88.9 |
| WM tube 3 s | 67.8 (SHAC 48 % of steps) | 82.9 (SHAC 79 %, from pessimism, not a better model) |

The students have **not** been run through the filter evaluation yet; only
their tubes exist.

### 9.4 Reading

- **Pelvis height**: a small cost, within the seed spread, with no change in
  false alarms. The comparison is not like-for-like: the margin changed too
  (`g_fall` → `g_fall_noheight`) and the policy was retrained.
- **Foot terms**: free. The 101-D model is as good as the 113-D control, and
  tighter and more consistent across seeds at 0.4 s. The control is biased
  pessimistic (median score −0.055 vs −0.027), with about twice the median
  error (|r| 0.073 vs 0.039). One plausible reason is that the contact-force
  dims are hard to predict; this hasn't been tested.
- **Distillation recipe**: about 25 % wider than the co-trained teacher at 1 s
  (0.17 vs 0.14), and 10× the false alarms at 1 s (0.05 vs 0.005), in both the
  control and the 101-D arm. This is 5–10× the noise floor, so it is real.
  - It is not explained by fit quality. The control's final training losses
    (err_1step / err_8step / state, seed 0: 0.033 / 0.038 / 0.034) beat the
    teacher's (0.038 / 0.045 / 0.040).
  - Two candidate causes, not yet tested:
    1. **Data coverage.** The student only sees the finished, 98 %-surviving
       policy.
    2. **Co-adaptation.** The teacher's policy learned to stay where its own
       model predicts well.
- **base_lin_vel**: essential for a single-frame model. The 98-D model's error
  compounds over the rollout:

  | Horizon | 98-D | 101-D |
  |---|---|---|
  | 1 step | 0.102 | 0.085 |
  | 10 steps | 0.048 | 0.030 |
  | 0.4 s | 0.106 | 0.049 |
  | 1 s | 0.661 | 0.081 |

  (|ĝ − g| while the robot is up.)
  - The 8-step training losses look normal (0.0095–0.0102 / 0.0139–0.0151),
    so the training metrics do not reveal it.
  - Two failure modes: seeds 0 and 1 drift into imagined falls (median r
    −0.48 at 1 s → 52–61 % false alarms); seed 2 has an optimistic tail
    (width 1.41 → flags everything).
  - The degradation shows at every decision delay, not just right after the
    kick. The guess that the model is merely blind to the kick's velocity jump
    was wrong.

## 10. Traps hit along the way

1. **The simulator is not reproducible run to run.**
   - Identical `--seed` runs differ: SONIC alone survives 1,243 vs 1,252 of
     2,048, even on two L40S runs.
   - Never assume paired trajectories across runs. Compare against a rerun
     noise floor (§8.2).
2. **The first `check_frozen.py` failed twice on tolerance, not on a real
   change.**
   - The mean, measured relative to |mean| near zero, inflated a 3e-6-std move.
   - The variance moved 1.6e-4 from one 4096-env reset batch (1024 in the
     smoke test, which passed).
   - The current check allows exactly one reset batch.
3. **`kill_invalid_depend` is on.** A failed `afterok` dependency **kills**
   the dependent array; it does not leave it pending. Resubmit it. `run_jobs`
   skips finished outputs, so a resubmitted student array only re-checks and
   re-calibrates.
4. **Queue time** dominated some days (a 5 h start estimate on
   `rtx6000,l40s,h100`). `scontrol update JobId=<id> Partition=rtx6000,l40s,h100,a100,h200`
   on a pending job helped at once.
5. **The training log shows two spawn mixtures.** The second, default one
   ("handover 0.00 (0 rows)") belongs to an 8-env video side env, which never
   renders here. The training envs use the 199,102-row handover bank.
6. **Critic terms carry noise configs but are noise-free**
   (`enable_corruption=False`). Test the group flag, not `term.noise`.
7. **Margin-state recording** (pelvis height / tilt prediction error) needs
   `pelvis_height` in the world-model state, so the no-height arms do not
   record it. Their `wm_filter_report` says `states recorded: False`.
8. **`compare_wm_tubes.py` on the login node** can take more than 5 minutes.
   Run it in the background or at the end of a GPU job.

## 11. Where everything is

| What | Path |
|---|---|
| Env config, critic-group kwargs | `robot_safety_sandbox/envs/g1_recover/env_cfg.py` (`critic_pelvis_height`, `critic_foot_terms`, `critic_base_lin_vel`, the three builders) |
| Margins, constants | `robot_safety_sandbox/envs/g1_stabilize/env_cfg.py` (`g_fall`, `g_fall_noheight`) |
| Task registrations | `robot_safety_sandbox/tasks/g1_recover.py` |
| Tests | `tests/test_g1_margins.py` |
| `--load-world-model` flag | `examples/train_model_based.py` |
| Eval + calibration | `examples/eval_sonic_qcbf.py`, `examples/calibrate_wm_min_tube.py`, `examples/wm_filter_report.py` (`calib_arrays` now also returns `decision`) |
| Comparison | `examples/compare_wm_tubes.py` |
| Configs | `configs/g1_recover_sonic_noheight_safetyshac1p.yaml`, `configs/g1_recover_sonic_noheight_student_wm.yaml` |
| Sweeps | `configs/sweeps/shac_noheight/`, `configs/sweeps/shac_student/` |
| Teacher runs / report | `runs/shac_noheight/shac_s*/`, `runs/shac_noheight/wmreport/report.log` |
| Original runs / report | `runs/shac_vs_ppo_sonic/shac_s*/`, `runs/shac_vs_ppo_sonic/wmreport/report.log` |
| Students | `runs/shac_student/{full,nofoot,nolinvel}_s*/` |
| Calibrations | `runs/shac_student/calib/{full,nofoot,nolinvel,teacher_rerun_a,teacher_rerun_b}_s*/` |
| Comparisons | `runs/shac_student/tube_compare.log` (teacher / 113 / 101), `tube_compare_noise.log` (+ reruns), `tube_compare_nolinvel.log` (all arms + delay breakdown), `tube_compare_height.log` (original vs no-height) |

SLURM job IDs, for finding logs in `logs/slurm/`:

| Job | ID |
|---|---|
| teacher training | 1678450 |
| teacher wmreport | 1678451 |
| student training (113 / 101) | 1709505 |
| student calibration rerun | 1715633 |
| teacher reruns | 1726000 |
| 98-D smoke | 1750789 |
| 98-D array | 1750790 |

## 12. Open questions and next steps

1. **The sim context.** Train a 101-D student without the 33-D DR input. It
   needs a switch: the context size is derived from the env's DR events in
   `train_model_based.py` and the eval. If the tube holds, nothing robot-specific
   needs estimating on hardware.
2. **How accurate must base_lin_vel be?** Add estimator-like noise to the
   101-D model's `base_lin_vel` input at calibration. This needs a small eval
   change and no retraining, and gives the accuracy a mocap system or
   estimator must deliver.
3. **A history-input student.** Give the model the last few frames of π's
   observation instead of base_lin_vel. This is the fully deployable route,
   but needs a stacked-state or recurrent world model.
4. **Close the distillation gap.** For example, initialise the student from
   the teacher's world-model weights with the dropped columns removed. If the
   gap closes, it was data coverage.
5. **Closed-loop filter evaluation of the students** (the stage-A conditions
   of §6 with `--policy` = a student). The tubes predict the outcome; this
   would measure it.
