# G1 recovery from LAFAN motions and pushes: experiment plan

This page is the plan for training the **fallback** of a G1 safety filter, the
recovery policy the filter hands control to, so that it recovers from
dangerous LAFAN mocap poses and from pushes. It is the canonical record of the
plan: the framing, the reference configuration, each experiment's question,
arms, read-outs and decision rule, the schedule, how to run it, and what is
built so far ([Status](#status)).

Every training arm is one override of one reference config, kept in a single
sweep file (`configs/sweeps/lafan_recovery.yaml`). `examples/make_sweep.py`
turns it into trainer configs and job lists, and `examples/run_jobs.sh` runs
the lists one job at a time, on the local RTX 5080 or on a cluster.

## Framing

The fallback should train on the filter's **handover distribution**: the
states where the monitor fires, along the nominal controller's trajectories,
together with the disturbance that caused them. Three levers act on that
distribution, and each one is its own experiment:

- **Sampling** (E3) decides where in the distribution training episodes are
  spent.
- **Noise and switching** (E4) decide whether the distribution is generated
  at all: does the push land inside the episode, or does the episode start
  from a state captured after a push?
- **The reach-avoid target** (E5) is the filter's hand-back condition: the
  set the fallback must reach before control returns to the nominal.

## What the existing results say

- **The regularized fallbacks are weak on LAFAN starts even without a push.**
  On the first 2000 rows of the LAFAN train split (all dance), with no push,
  4 rollouts per row and a 3 s cap, the action-regularized push policies
  survive 49–68 %: push_rear 0.49, push_fast 0.57, push_fast_privcrit 0.65,
  push_fast_cont 0.68 (2026-10-23, `examples/label_start_states.py
  --no-push --max-rows 2000 --k 4`). The rows split almost in two (for
  privcrit 32 % of rows have p_surv ≤ 0.1 and 62 % ≥ 0.9), and the falls
  follow spawn speed (AUC 0.79; failing rows average 0.71 m/s against
  0.34 m/s) and the hip/ankle roll joints (side steps), not tilt. So uniform
  LAFAN sampling is not wasted for these policies; what matters is which
  rows carry signal, which is what the rollout labels measure.
  (`runs/g1_recover`, which survives 98.5 % of the LAFAN starts, was trained
  WITHOUT action regularization and cannot transfer to hardware; it is not a
  baseline, warm start or labeling policy anywhere in this plan.)
- **The value cannot yet be trusted as a label.** The symmetric critic was
  blind to kicks (AUC 0.47). The privileged critic has not been tested on
  LAFAN.
- **Avoid policies never stop.** Only 0.2 % of their steps are inside the
  stance set, and their median base speed is 1.5–1.9 m/s.

## Three clarifications

### The push range is measured, not assumed

E1 measures it. A push is described by its impulse $J = F T$, or by the base
velocity change it causes, $\Delta v = J / m$ with $m = 33.3$ kg (the G1
model's mass, 33.34 kg): 260 N for 0.2 s is about 1.56 m/s, and 100 N for
0.2 s about 0.6 m/s.

### The reach-avoid velocity condition is a tolerance, not a target

Reach-avoid needs a velocity condition, but as a tolerance around zero, not a
value to track. `l_stance` already has one: every body-frame linear velocity
component within 0.25 m/s, every angular one within 0.30 rad/s, the pelvis at
0.55–0.85 m, and the tilt at most 25°.

The capture point sits $v / \omega_0$ ahead of the pelvis, with
$\omega_0 = \sqrt{g / z} \approx 3.7\ \mathrm{s}^{-1}$. At 0.25 m/s that is
about 7 cm, and at 0.5 m/s about 13 cm, roughly half a foot length (not yet
checked on the model). Beyond that the robot has to step, so the useful sweep
is 0.25–0.5 m/s.

- If the hand-back controller is a walker, the target becomes "velocity near
  the walker's command" (like `l_track`), not "velocity near zero".
- With a push inside the episode, the target may only count once the push has
  ended. A slow LAFAN start can already sit in the target before its push
  lands.
- Reach-avoid says nothing about what happens after the first reach. It
  certifies getting into the target without falling first, not staying in
  it. Staying has to come from handing back to a controller that keeps the
  target safe. E5 measures staying with a separate check; the objective does
  not promise it.

### Push placement is its own experiment (E4)

| | Push inside the episode | Collected post-push starts | Hybrid |
|---|---|---|---|
| Trains reacting while the force still acts | yes | no | yes |
| Sees a push that hit before the handover | no | yes | yes |
| Critic | mixes in the risk of future pushes it cannot see | clean: "recoverable from here" | clean |
| Per-row curricula | no | yes | yes |
| Goes stale as the policy changes | no | yes | yes |

The hybrid stores the **pending** push with each handover state: the force
that is still acting and how many steps it has left. After the spawn, the
remainder is replayed.

## The reference configuration

Every training arm is one override of this configuration (the `base:` block
of `configs/sweeps/lafan_recovery.yaml`).

| | |
|---|---|
| Recipe | the push_fast family with the privileged critic, fixed γ = 0.99, a 3 s episode cap, and the rest truncation for the avoid arms |
| Task | `g1_recover_lafan`: exactly **one** push per episode |
| Steps | 500M, warm-started from push_fast_privcrit's final model with a 20M-step critic warmup |
| Starts | 30 % the walker-bank mixture, 70 % LAFAN **train** frames weighted by their rollout labels (S3) |
| Push | inside the episode, uniform horizontal direction, onset 0.1–0.4 s after the spawn, the ceiling from E1; the per-env survival scale finds each env's frontier below it |
| Objective | avoid |
| Seeds | 0 and 1 |

- **Primary metric:** E-handover survival averaged over the frozen E1 grid,
  plus the F50 pooled over directions and the F50 from behind.
- **Guardrail:** the E-std F50 must not drop by more than 5 %.
- **Decision rule:** a difference counts only if it is larger than the bigger
  of the two arms' within-arm seed gaps. Otherwise the simpler arm wins.
- **Frozen after E1:** the held-out clip split and the eval grid.

## E0: infrastructure

| | What | Where |
|---|---|---|
| E0.1 | the `mocap` spawn source with per-row weights (and a `handover` source for collected handover states) | `envs/g1_recover/spawn.py` (samplers `ladder`, `uniform`, `weighted`, `online`, `sequential`); `envs/g1_recover/states.py` (`KIND_MOCAP`, `KIND_HANDOVER`, optional columns, multi-file banks) |
| E0.2 | clip-level train / test split, about 15 % of the clips of each category held out | `examples/split_lafan_clips.py` → `data/lafan1_split.json`, `*_train.pt` / `*_test.pt` |
| E0.3 | the push suite: E-std (standing), E-lafan (a held-out frame, pushed 0.1 s later), E-handover (a held-out clip replayed, pushed, then handed over); impulse × duration × 8 directions; survival, F50 per direction, peak speed, distance, time to target | `examples/eval_push_suite.py`, `robot_safety_sandbox/envs/g1_recover/push_suite.py` ([below](#the-push-suite-e03)) |
| E0.4 | the rollout labeler: K = 8 rollouts per row → `p_surv`, plus V(s0) (about 85M steps, about 20 min) | `examples/label_start_states.py`, `envs/g1_recover/labels.py` |
| E0.5 | the replay-handover generator, storing the pending push | `examples/harvest_g1_handover_states.py`, `envs/g1_recover/playback.py`; the policy's own post-push states: `examples/harvest_g1_frontier_states.py --kind handover` |
| E0.6 | the reach-avoid tasks: tolerance knobs, the target counts only after the push, the episode ends once the robot dwells in the target | `g1_recover_lafan_ra`, `g1_recover_lafan_ra_loose` (`tasks/g1_recover.py`, `envs/g1_recover/targets.py`) |
| E0.7 | the sweep launcher | `configs/sweeps/lafan_recovery.yaml`, `examples/make_sweep.py`, `examples/run_jobs.sh` |

!!! note "Why the reach-avoid tasks end through the rest truncation"
    `end_criterion="reach-avoid"` adds a success termination that is not a
    time-out. The force-scale curriculum, `RecoverCurriculum` and the safety
    counters would all count it as a fall. So the reach-avoid tasks keep
    `end_criterion="failure"` and end an episode with the rest truncation
    (`envs/g1_recover/rest.py`, `time_out=True`), whose margin is the
    reach-avoid target itself: the same tolerances, gated on the push. The
    dwell only decides when the episode ends; it adds nothing to what the
    value certifies.

### The push suite (E0.3)

One protocol for every checkpoint: the same start states, the same pushes,
the same seed. It runs in the `g1_recover_lafan` env with the rest truncation
off and the task's random pushes switched off (`max_pushes=0`). The only push
is a **scripted** one (`RandomPush.schedule`), armed for every env right after
it spawns.

| Setting | Start | Push | Policy in control |
|---|---|---|---|
| `std` | the nominal stance at rest | 0.1 s after the spawn | from the spawn |
| `lafan` | a held-out LAFAN frame (`data/g1_recover_states_lafan1_test.pt`) | 0.1 s after the spawn | from the spawn |
| `handover` | a held-out clip (`data/g1_lafan1_clips_test.pt`), replayed open-loop from the spawn frame | U(0.02, 0.1) s into the replay (the grid file's `handover_onset_s`) | from `delay` s after the push onset; a push that has not ended keeps acting through the takeover |
| `walker` (opt-in) | the walker bank, every source sampled over all its rows | 0.1 s after the spawn | from the spawn |
| `pre` (opt-in) | held-out **collected** handover states (`data/g1_handover_test.pt`) | none scripted: each row's own pending push plays out | from the spawn |

- **Cells.** A cell is impulse J × duration T × direction × handover delay.
  The force is J / (steps · dt) with steps = round(T / dt), so the delivered
  impulse is exactly J. Directions are in the robot's heading frame at the
  onset: 0 = a push from behind, then counterclockwise (90° = from the right).
  Episodes are assigned to cells round-robin in the order they are issued, so
  the cells stay balanced to within one episode, and every issued episode is
  recorded once whatever its outcome.
- **The window.** A record closes `eval_s` (default 4 s) after the takeover:
  survived, or at the fall. In `handover`, a replay that falls **before** the
  takeover is the open-loop replay failing, not the fallback: it is counted in
  `prefix_fall_rate` and left out of every survival number.
- **Per episode:** survival and fall time; peak planar speed after the push
  onset; planar distance; the policy's own V at the takeover state (the
  privileged critic's inputs where it has them); and, for the tight
  (0.25 m/s, 0.30 rad/s) and loose (0.5 m/s, 0.6 rad/s) stance targets,
  measured from the robot data only after the push has **ended**: time to the
  first reach, the share of the later steps spent inside, and whether the
  robot fell after reaching.
- **Per setting × policy:** survival per impulse level (pooled, and per
  direction, duration, delay and clip category), F90 / F50 / F10 by linear
  interpolation at the first crossing, behaviour statistics, and the AUC of V
  against survival. An F-level outside the measured grid reads `<min` or
  `>max`, not a number.

## E1: push-range calibration (eval only)

- **Question.** Which push sizes take the current policies from 90 % to 10 %
  survival, from standing, from LAFAN frames, and at a handover? Does the
  impulse alone predict survival?
- **Checkpoints.** The action-regularized push policies only:
  push_fast_privcrit, push_rear and push_fast_cont (`runs/g1_recover` is
  unregularized and is not used).
- **Grid.** J = 0–60 N·s in steps of 4, durations 0.1 / 0.2 / 0.3 s,
  8 directions, handover delays 0 / 0.1 / 0.2 s. At the suite's default of 64
  episodes per (J, T, direction) cell, that is 512 starts per (J, T) pooled
  over directions and about 26M env-steps per checkpoint over the three
  settings. (The plan budgeted about 300 starts per cell, about 50M steps and
  15 min per checkpoint; raise `--episodes-per-cell` if the per-direction
  curves are too noisy.)
- **Read off.** Survival against J by setting, direction and clip category;
  F90 / F50 / F10.
- **Decisions.**
    - Describe pushes by impulse alone if the per-duration curves overlap.
    - Set the training ceiling to about 1.3 × the E-lafan F10, as a force at
      the training duration of 0.2 s. Raise it if `game/force_scale_mean` sits
      near 1 during training.
    - Freeze the eval grid at 5 levels from F90 to F10 of the pooled E-lafan
      curve.
    - Sanity-check the range against hardware pushes.

`--calibrate` writes this proposal to
`configs/sweeps/lafan_recovery/eval_grid.yaml`: the impulses, the durations
(one if the curves overlap), and `force_max_suggested`.

### E1 result (2026-10-23, `runs/lafan_recovery/E1_calibration/report.txt`)

Survival by impulse, 512 episodes per (J, T) pooled over 8 directions:

| Setting, policy | J = 0 | 20 N·s | 40 N·s | 60 N·s | F90 | F50 |
|---|---|---|---|---|---|---|
| standing, push_fast_privcrit | 1.00 | 0.88 | 0.67 | 0.12 | 19.1 | 48.4 (from behind 18.8) |
| standing, push_fast_cont | 1.00 | 0.81 | 0.63 | 0.11 | 16.1 | 47.0 |
| standing, push_rear | 1.00 | 0.63 | 0.01 | 0.00 | 13.9 | 22.5 |
| held-out LAFAN, push_fast_privcrit | 0.49 | 0.42 | 0.36 | 0.15 | < 0 | < 0 |
| held-out LAFAN, push_fast_cont | 0.50 | 0.44 | 0.38 | 0.14 | < 0 | < 0 |
| handover (replay ≤ 0.4 s), push_fast_privcrit | 0.27 | 0.27 | 0.17 | 0.07 | < 0 | < 0 |

- **Impulse alone describes a push.** From standing, F50 moves about ±10 %
  between 0.1, 0.2 and 0.3 s pushes of the same impulse (privcrit: 43, 49,
  54 N·s).
- **On LAFAN starts the start state dominates.** About half the held-out
  LAFAN starts fall with no push, for every regularized policy, and the curve
  is flat to about 30 N·s. So the rule "1.3 × the E-lafan F10" has nothing
  to read, and its 390 N suggestion is a grid-edge fallback.
- **Decisions taken.**
    - Training ceiling `force_max` = 300 N (60 N·s at 0.2 s): past privcrit's
      standing F50 toward its F10, so it does not bind as the policy improves.
    - The frozen grid is the one E1 wrote: 0, 15, 30, 45, 60 N·s × 0.1 /
      0.2 / 0.3 s × 8 directions, takeover delays 0 / 0.1 / 0.2 s. It spans
      the standing F90 to past F10 and the whole LAFAN decline.
    - The P-pre-nominal pushes span the same 4–60 N·s.
    - The E-handover replay is kept to ≤ 0.3 s: push onset U(0.02, 0.1) s
      into the replay (E1 used U(0.05, 0.2) s), takeover ≤ 0.2 s after the
      onset. Without balance feedback the replay drifts: E1 measured 0.27
      survival at zero push after up to 0.4 s of replay, against 0.49 from the
      same clips' frames. E1's handover row is on the older protocol.

## E2: the value as a label (eval only)

- **Read off.** The AUC of V(s0) against falls, and the rank correlation of V
  with `p_surv`, overall and per category (from the labeling pass,
  `data/g1_recover_states_lafan1_train_labeled.report.json`).
- **Decisions.**
    - AUC ≥ 0.9: run S5 and allow value screening in E6. Otherwise drop S5.
    - If only a tiny fraction of rows has `p_surv` in (0.1, 0.9), there is
      little frontier to sample from: prioritize E4 over E3.

## E3: start-state sampling (12 runs)

All arms keep the reference push; S1–S5 keep the 30 % walker-bank mixture.

| Arm | Starts |
|---|---|
| S0 | the walker bank only (the push_rear mixture), no LAFAN |
| S1 | + LAFAN, rows drawn uniformly |
| S2 | + LAFAN, the per-env level ladder over rows sorted by speed |
| S3 | + LAFAN, fixed p(1 − p) of the rollout label (the reference) |
| S4 | + LAFAN, an online per-row EMA of the training outcomes, initialised from the S3 labels |
| S5 | + LAFAN, p(1 − p) of the value-calibrated label (only if E2 passes) |

- **Hypothesis.** S1 ≈ S0; S3 and S4 beat S1; S4 pulls ahead late.
- **Read off.** The primary metric at 250M and 500M steps; walker-bank
  survival and E-std (forgetting); the `Curriculum/recover/surv_mocap`
  curves.

## E4: push placement (8 runs)

Fixed: S3 sampling. Each bank is labeled under its own setup.

| Arm | Starts | Push in the episode |
|---|---|---|
| P-env | LAFAN frames (the reference) | one random push |
| P-pre-self | the warm-start policy's own states 0–20 steps after a push on LAFAN starts, harvested once | none |
| P-pre-nominal | a clip replayed, pushed, and handed over 0–0.2 s after the push starts, with the pending push carried | none other |
| P-mix | half P-pre-nominal handovers, half P-env LAFAN frames | P-env's push on the LAFAN half only |

- **Hypothesis.** The collected-state arms learn faster per step and give a
  better monitor; P-env handles ongoing pushes better; P-mix is best overall.
- **Read off.** The primary metric (E-handover); blind spots via E-lafan and
  E-pre (held-out post-push starts, no further push); monitor quality, the
  AUC of each arm's own V on the E-handover starts.

## E5: the objective (6–8 runs)

| Arm | Objective |
|---|---|
| O0 | avoid (the reference) |
| O1 | reach-avoid to `l_stance`, tight: 0.25 m/s, 0.30 rad/s (`g1_recover_lafan_ra`) |
| O2 | reach-avoid to `l_stance`, loose: 0.5 m/s, 0.6 rad/s (`g1_recover_lafan_ra_loose`) |
| O3 | optional: reach-avoid to {V_handback ≥ δ}, if a standing controller with a safety critic exists (δ calibrated by rollouts) |

All warm-start the actor from the avoid policy and refit the critic
(`reset_value`).

- **Read off.** E-handover survival against O0 at every level; the rate of
  reaching the target within 3 s; the median time to reach; peak speed;
  distance travelled. Then the **staying check**: after the first reach the
  same policy keeps running at least 3 s more, and the suite reports the fall
  rate and the share of time spent back outside the target. Reach-avoid does
  not promise staying, so this measures it.
- **Decision.** Pick the tightest tolerance whose survival cost is at most
  about 5 points at every level of the grid.

## E6: follow-ups on the winners (wave 2, 2 arms × 2 seeds each)

- **E6a, the handover rule:** a fixed random delay, or handing over when the
  monitor's V drops below δ. Applies to P-pre-nominal; the monitor rule
  matches how the real filter switches.
- **E6b, refreshing the bank:** harvest and label once, or re-harvest and
  re-label every 100M steps (chained jobs).
- **E6c, a velocity kick:** a small random base velocity at the spawn, or
  none.
- **E6d, frontier placement:** a random τ, or τ bisected so that the start
  sits at p̂ ≈ 0.5.

## E7: confirmation

- **Setup.** The best of E3 × E4 × E5 plus any E6 wins, 3 seeds, about 1.5B
  steps.
- **Evaluation.** The full suite, the per-direction push eval, and videos.
- **Baselines.** push_fast_privcrit and push_rear.
- **Later.** The closed-loop filter test, a real tracker plus the monitor,
  once a tracker exists. There is none in the repo yet.

## Schedule

| Wave | Contents | Jobs | GPU-h (approx.) |
|---|---|---|---|
| 0 | E0 code; E1 and E2 eval jobs | ~6 evals | ~2 |
| 1 | E3, E4, E5 in parallel (they depend only on E0 and E1) | ~28 runs | ~50 |
| 2 | E6 | ~16 runs | ~28 |
| 3 | E7 | ~3–6 runs | ~15–30 |

One 500M-step run takes about 1.75 h at the roughly 80k steps/s of the local
RTX 5080; cluster GPUs may differ. Wave 1 is the big parallel batch, and every
arm in it is one override of the reference config. With the reference shared
(S3 = P-env = O0), wave 1 is 10 distinct arms × 2 seeds = 20 runs, plus S5's
2 if E2 passes.

## Launching the sweep (runbook)

The step-by-step path from where the work stands (2026-10-23) to a running
sweep. [How to run](#how-to-run) below is the reference for each tool.

**Where it stands.** On the local RTX 5080, every prerequisite has run at full
size: the clip library and split, E1, the labeled banks and E2, and the
handover banks. Every arm has passed a 2M-step smoke run. The decisions are
in the sweep file: `force_max` 300 N, the frozen grid in
`configs/sweeps/lafan_recovery/eval_grid.yaml`, replay ≤ 0.3 s, and S5 gated
off (E2 AUC 0.72). What is left is wave 1: **20 training runs** (10 arms × 2
seeds), then their evaluations.

### Step 1. Code on the cluster

1. This repository, branch `g1-experiments`.
2. **safety-stable-baselines, branch `feat/action-penalties`** (commit
   `f465126` or later). Every arm sets `action_rate_coef: 0.05` and
   `action_mag_coef: 0.005`, and those two arguments exist only on that branch.
   On `main` (through `f26fb8e`) every arm would fail at model construction.
   The branch also carries the multi-gamma heads, which this sweep does not
   use. Install it editable (`pip install -e .`), or put the checkout next to
   this repository (or point `$SAFETY_SB3_PATH` at it).
3. The local environment the smoke runs used: torch 2.11.0+cu128, mjlab
   1.2.0, mujoco and mujoco-warp 3.6.0, warp-lang 1.12.0, stable_baselines3
   2.9.0, gymnasium 1.3.0, numpy 2.2.6, Python 3.10. Match them where you can;
   a different mjlab can change the simulation.

### Step 2. Data and checkpoints (copy, never regenerate)

`data/` and `runs/` are git-ignored. Copy them from the 5080 machine rather
than rebuilding them on the cluster. The labels come from one policy's
rollouts and one random stream, and every arm must train on the same banks.

```bash
# from the 5080 machine, repository root; CLUSTER = user@host:/path/to/robot-safety-sandbox
rsync -av data/ "$CLUSTER/data/"                                       # ~0.9 GB
rsync -av runs/g1_recover_push_fast_privcrit runs/g1_recover_push_rear \
      runs/g1_recover_push_fast_cont "$CLUSTER/runs/"                  # warm start + baselines, ~0.9 GB
rsync -av runs/lafan_recovery/E1_calibration "$CLUSTER/runs/lafan_recovery/"   # E1's record
```

The warm start is `runs/g1_recover_push_fast_privcrit` and needs its
`tensornormalize.pt`. The other two checkpoints are evaluation baselines.

### Step 3. Check before launching

From the repository root on the cluster, with the environment active (the
job lines call plain `python`):

```bash
python -c "import safety_sb3, inspect; from safety_sb3 import SafetyPPO1P; \
  assert 'action_rate_coef' in inspect.signature(SafetyPPO1P).parameters"   # step 1.2
python examples/make_sweep.py                  # regenerates the configs and job lists (byte-identical to the committed ones)
examples/run_jobs.sh configs/sweeps/lafan_recovery/prep_jobs.txt --list   # all 9 'done': every bank was copied (a 'todo' means one is missing)
examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt --list        # 20 'todo'
```

If `make_sweep.py` changed any generated file compared with the committed
one, the sweep file or the code differs from what the smoke runs tested.
Stop and find out why before launching. A one-arm smoke on the cluster
(about 4 minutes) checks the installation end to end:

```bash
python examples/make_sweep.py --smoke
examples/run_jobs.sh configs/sweeps/lafan_recovery_smoke/jobs.txt --only 4   # S3, the reference
python examples/make_sweep.py                  # back to the full job lists
```

### Step 4. Launch the 20 training jobs

`jobs.txt` holds seed 0 of every arm first, then seed 1:

| Job | Arm | Job | Arm |
|---|---|---|---|
| 1 / 11 | E3 S0 (walker bank only) | 6 / 16 | E4 P-pre-self |
| 2 / 12 | E3 S1 (+ LAFAN, uniform) | 7 / 17 | E4 P-pre-nominal |
| 3 / 13 | E3 S2 (+ LAFAN, speed ladder) | 8 / 18 | E4 P-mix |
| 4 / 14 | E3 S3 = E4 P-env = E5 O0 (the reference) | 9 / 19 | E5 O1 (reach-avoid, tight) |
| 5 / 15 | E3 S4 (+ LAFAN, online) | 10 / 20 | E5 O2 (reach-avoid, loose) |

Each job is one GPU for about 1.75 h at the 5080's ~80k env-steps/s (4096
envs, 500M steps). Run them one at a time:

```bash
examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt --only 1
```

or as a scheduler array (SLURM shown; adapt the partition and environment
lines to the cluster):

```bash
#!/bin/bash
#SBATCH --job-name=lafan_recovery
#SBATCH --array=1-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%A_%a.out
cd /path/to/robot-safety-sandbox
source activate rl                     # the environment of step 1
examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt --only "$SLURM_ARRAY_TASK_ID"
```

To get one seed of every arm before spending the second, submit `1-10` first.
Every run warm-starts from the same checkpoint, so no job depends on another.
Every run is named `g1_recover_lafan` (or `..._ra`, `..._ra_loose`) in wandb.
Tell them apart by the `--out` directory, `runs/lafan_recovery/<exp>/<arm>_s<seed>/`.

### Step 5. The first minutes of a run

Its log (`logs/<jobs dir>_jobs/NNN_*.log`) should show these lines.

All arms:

- `[algo] ... -> SafetyPPO1P` (`ReachAvoidPPO1P` for jobs 9, 10, 19, 20);
- `[g1_recover] spawn mixture:` with `mocap 0.70 (62550 rows)`: 0.35 + `handover 0.35` for P-mix, `handover 0.70 (100000 rows)` for the P-pre arms, no LAFAN for S0;
- `[g1_recover] sampler <weighted|uniform|online>` (none printed for S2, the ladder);
- `[random-push] kwargs ... 'max_pushes': 1` (`0` for the P-pre arms);
- `[critic] privileged: value net reads the 'critic' group (114 dims)`;
- `[warm-start] policy weights loaded: 29/29 tensors` and `+ critic obs stats (114 dims)`.

The two reach-avoid arms only:

- `[warm-start] value net RESET`.

Early metrics, from the smoke runs:

- `Curriculum/recover/surv_mocap` around 0.4–0.5, `surv_handover` around 0.45–0.6 (P-pre arms);
- `safety/reach_rate` around 0.07 (O1) and 0.35 (O2);
- `force_scale_mean` moving off 0.5;
- no NaN in the losses.

A job that dies has no `final_model.zip`. Rerun the same `--only N`; it
starts from scratch, because the trainer does not resume.

### Step 6. Evaluate and compare

When the runs are done (a missing run is reported, not fatal):

```bash
examples/run_jobs.sh configs/sweeps/lafan_recovery/eval_jobs.txt --list   # 30 jobs: one per run + baselines + E5's staying pass
examples/run_jobs.sh configs/sweeps/lafan_recovery/eval_jobs.txt --only N # or as an array, one GPU each
examples/run_jobs.sh configs/sweeps/lafan_recovery/report_jobs.txt        # the merged comparison per experiment (cheap; rerunnable)
```

Then read `runs/lafan_recovery/E3/report/report.txt`, `E4/report`, `E5/report`
and `E5/report_stay` with the decision rules of
[What to read off](#what-to-read-off-and-the-decision-rules). Leave
`jobs_gated.txt` (S5) unrun, because E2 failed its gate.

## How to run

Everything runs from the repository root, with the `rl` environment's
`python` on the path (`make_sweep.py --python <interpreter>` writes another
one into the job lines).

### 1. Data and checkpoints

`data/` and `runs/` are git-ignored, so a fresh clone (a cluster) has neither.
The inputs are:

- `data/lafan1_g1/*.csv`, Unitree's G1 retargeting of LAFAN1;
- `data/g1_recover_states.pt` and `data/g1_recover_states_lafan1.pt`, the
  walker bank and the LAFAN stride bank;
- the checkpoints `runs/g1_recover_push_fast_privcrit/g1_recover_push_fast/`
  (the warm start and labeling policy), `runs/g1_recover_push_rear/` and
  `runs/g1_recover_push_fast_cont/` (E1 and the baselines), each with its
  `tensornormalize.pt`.

Either copy `data/` and those run directories over, or copy the inputs and
run the prerequisites there. Arms that are compared should train on the same
`data/`: make it once and copy it, rather than regenerating it per machine.

### 2. Generate the sweep

```bash
python examples/make_sweep.py            # configs/sweeps/lafan_recovery/
python examples/make_sweep.py --smoke    # configs/sweeps/lafan_recovery_smoke/
```

It writes one trainer config per arm (`<exp>_<arm>.yaml`), `jobs.txt` (one
training job per distinct arm and seed, all of seed 0 first),
`jobs_gated.txt` (S5, until E2 decides), `eval_jobs.txt`, `report_jobs.txt`,
`prep_jobs.txt` and `manifest.json`. Before it writes anything it checks that
every config key is one the trainer accepts, that every task, env override
and push override exists, that every job's script takes every flag the job
passes, and that every bank a config reads either exists or is made by a prep
step. **Edit the sweep file, never the generated configs**, and regenerate.
`--only E4` limits the jobs to one experiment; it overwrites the job lists.

### 3. Prerequisites (wave 0)

```bash
examples/run_jobs.sh configs/sweeps/lafan_recovery/prep_jobs.txt
```

In order: the clip library, the clip split, **E1** (writes
`configs/sweeps/lafan_recovery/eval_grid.yaml`), the LAFAN train labels
(**E2**, read `data/g1_recover_states_lafan1_train_labeled.report.json`), the
P-pre-self harvest and labels, the P-pre-nominal harvest and labels, and the
held-out handover bank for E-pre. A step whose outputs exist is skipped.

Two numbers come from E1, so the steps after it run with them. Both are set
in the sweep file since the E1 run of 2026-10-23 (`base.force_max` 300 N,
`prep_vars.handover_impulse_range` 4–60 N·s; see [E1 result](#e1-result-2026-10-23-runslafan_recoverye1_calibrationreporttxt)).
To re-decide them:

1. Set `base.force_max` in the sweep file (`force_start` follows it) and
   `prep_vars.handover_impulse_range`.
2. Regenerate (`python examples/make_sweep.py`) and run the rest of
   `prep_jobs.txt`, from `--from 4` on (job 4 is the first labeling step).

After E2, if the AUC is at least 0.9, run `jobs_gated.txt` (S5's training and
evaluation) along with the rest.

### 4. Smoke test on the local RTX 5080

```bash
python examples/make_sweep.py --smoke
examples/run_jobs.sh configs/sweeps/lafan_recovery_smoke/jobs.txt --keep-going
examples/run_jobs.sh configs/sweeps/lafan_recovery_smoke/eval_jobs.txt --keep-going
examples/run_jobs.sh configs/sweeps/lafan_recovery_smoke/report_jobs.txt
```

The smoke sweep uses the same arms, data and warm start: 2M steps on
1024 envs, no critic warmup, no checkpoint inside the run, no wandb, seed 0
only, outputs under `runs/smoke/lafan_recovery/`, and `--quick` evals (a tiny
grid on 256 envs). It checks that every arm builds and trains a few rollouts
(configs, banks, samplers, push overrides, warm start, the reach-avoid
tasks), not that it learns. It needs the prerequisites, which it shares with
the full sweep. The first video clip of each run still renders at step 0.

### 5. Training on the cluster, one job at a time

Each line of `jobs.txt` is one self-contained job:

```bash
python examples/train.py --config configs/sweeps/lafan_recovery/E3_S1.yaml \
    --out runs/lafan_recovery/E3/S1_s0 --seed 0
```

Run the lines one by one, or let a scheduler's array index pick one:

```bash
examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt --list            # number, done/todo, command
examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt --only "$SLURM_ARRAY_TASK_ID"
```

`run_jobs.sh` numbers the jobs 1, 2, ... in file order, skipping blank and
comment lines, so `--only N` is stable for a given file. It skips a job whose
outputs exist (a training run with a `final_model.zip`, an eval with a
`summary.json`), writes one log per job under `logs/`, and stops at the first
failure unless `--keep-going` is given. A run that died has no final model and
reruns from scratch. Every arm is named `g1_recover_lafan` (or `..._ra`) in
wandb, the trainer's tag; tell runs apart by their `--out` directory, or pass
`--no-wandb`.

### 6. Evaluation and reports

```bash
examples/run_jobs.sh configs/sweeps/lafan_recovery/eval_jobs.txt
examples/run_jobs.sh configs/sweeps/lafan_recovery/report_jobs.txt
```

`eval_jobs.txt` runs the frozen suite on every run's final model and on the
two baselines, over all five settings (`std`, `lafan`, `handover`, `walker`,
`pre`), plus E5's staying pass (E-handover with a 6 s window). Each eval runs
one policy from the same seed and cells, so evals of different runs merge
into one comparison. `report_jobs.txt` merges each experiment's evals, a
missing one is left out and named, and is never skipped, so it can be rerun
as runs finish.

To evaluate a checkpoint mid-run (E3's read-off at 250M), name it with the
step before the seed suffix so the seeds still group:

```bash
python examples/eval_push_suite.py --frozen \
    --policy S3@250M_s0=runs/lafan_recovery/E3/S3_s0/g1_recover_lafan/checkpoints/model_<steps>_steps.zip \
    --out runs/lafan_recovery/E3/S3_s0/eval_250M
```

### Where outputs land

```text
configs/sweeps/lafan_recovery/       generated configs + job lists; eval_grid.yaml (E1, frozen)
data/                                library, split, labeled banks; *_labeled.report.json (E2)
runs/lafan_recovery/
  E1_calibration/                    summary.json, report.txt, episodes.pt
  E3/S1_s0/g1_recover_lafan/         final_model.zip, tensornormalize.pt, checkpoints/, config.yaml
  E3/S1_s0/eval_frozen/              summary.json, report.txt, episodes.pt, grid.json
  E3/report/                         the merged E3 comparison (report.txt, summary.json)
  E5/O1_s0/g1_recover_lafan_ra/      a reach-avoid run (the trainer's tag is the task)
  E5/O1_s0/eval_stay/                E5's staying pass; E5/report_stay/ merges them
  baselines/push_fast_privcrit/      eval_frozen/, eval_stay/
runs/smoke/lafan_recovery/           the smoke sweep
logs/<jobs dir>_<jobs file>/         one log per job
```

### What to read off, and the decision rules

Every eval's `report.txt` holds, per setting and policy: survival per impulse
level (with Δv), pooled survival and its mean over levels, F90 / F50 / F10,
F50 per direction, duration, takeover delay and clip category, the
replay-fall rate (handover), the behaviour statistics, and the AUC of V. A
merged report adds, per arm, the mean over seeds and the within-arm seed gap
of the mean survival over levels, the pooled F50 and the F50 from behind.

| Experiment | Read | Rule |
|---|---|---|
| E1 | `E1_calibration/report.txt`; `eval_grid.yaml` | impulse alone if the duration curves overlap; ceiling about 1.3 × E-lafan F10; grid = 5 levels F90..F10 of pooled E-lafan, then frozen |
| E2 | `data/g1_recover_states_lafan1_train_labeled.report.json` | AUC ≥ 0.9 → run S5 (`jobs_gated.txt`); tiny (0.1, 0.9) share → E4 before E3 |
| E3 | `E3/report`: handover mean over levels and F50s; walker and std (forgetting); `Curriculum/recover/surv_mocap` in training | seed-gap rule; E-std F50 within 5 % of the reference |
| E4 | `E4/report`: handover; lafan and pre (blind spots); AUC(V) on handover (monitor quality) | seed-gap rule |
| E5 | `E5/report`: survival per level against O0, reach-within-3 s, time to reach, speed, distance; `E5/report_stay`: fall-after-reach and stay share | the tightest tolerance whose survival cost is ≤ ~5 points at every level |

During training, watch `Curriculum/recover/surv_mocap` / `surv_handover`, the
frontier statistics `p_mean_mocap` / `front_mocap` (weighted and online
samplers), `push/surv_from_<side>`, `game/force_scale_mean` (near 1 means the
ceiling binds) and `env/Episode_Termination/recovered`.

## Status

**Built (E0).** The `mocap` and `handover` spawn sources and their samplers,
multi-file banks with per-row label columns, the pending-push channel through
the bridge, the one-push-per-episode push process with scripted pushes, the
three tasks `g1_recover_lafan`, `g1_recover_lafan_ra` and
`g1_recover_lafan_ra_loose`, the clip library and split, the rollout labeler,
the two handover harvesters, the push suite, and the sweep tooling
(`make_sweep.py`, `run_jobs.sh`, the sweep file). The sweep's configs are
checked against the trainer and the task registry.

**Run on the RTX 5080 (2026-10-23).** Every prerequisite in `prep_jobs.txt`
ran at full size, in about 45 min after E1: the clip library and split, E1,
the LAFAN train labels (E2), the self-handover and replay-handover banks with
their labels, and the held-out handover bank. Every training arm then trained
a 2M-step smoke run (`configs/sweeps/lafan_recovery_smoke/jobs.txt`, about 4
min each) with the right learner, spawn mixture, sampler and push settings.
The real 500M-step sweep has not been run.

| Bank | Rows | Survival (privcrit) | Rows with 0.1 < p < 0.9 | AUC of V(s0) |
|---|---|---|---|---|
| LAFAN train, labeled | 62,550 | 0.36 | 78 % | 0.72 |
| self-handover, labeled (P-pre-self) | 100k | 0.44 | 6 % | 0.995 |
| replay handover, labeled (P-pre-nominal) | 100k | 0.22 | 9 % | 0.86 |

**E2 result.** The episode AUC of V on the LAFAN train labels is 0.72 (< 0.9),
so S5 stays gated and the arms weight by rollout labels (S3, S4).

**Open.**

- **S5 is gated off by E2** (AUC 0.72). Its jobs stay in `jobs_gated.txt`.
- **O3** needs a hand-back controller with a safety critic (a standing
  controller whose own V defines the target). None exists yet, so it is not
  in the sweep.
- **E6 and E7 are not generated yet.** They depend on the winners of E3–E5;
  add them as experiments to the sweep file then. E6b's periodic re-harvest
  needs chained jobs, which the job lists do not express yet.
- **The closed-loop filter test** needs a tracking controller; there is none
  in the repo. Until then the E-handover nominal is the clip replayed
  open-loop, which drifts on fast clips.
- **The warm start is weak on LAFAN frames before any push** (see "What the
  existing results say"): push_fast_privcrit survives 65 % of the first 2000
  dance rows with no push. Its rollout labels, and so S3's weights, reflect
  that weakness, which is what they are for; E1 measures it on the held-out
  split.
