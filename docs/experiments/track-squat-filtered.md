# Learning inside the thermal safety filter: squat tracking

**Question.** A task policy learns a held-out LAFAN motion either inside the trained
fall + motor-temperature rollout safety filter or with no filter at all. How do the
two compare on motor temperatures, failure counts and task performance?

**Status (2026-11-12).** The pipeline is built and the CPU tests pass. The GPU smoke
(`configs/sweeps/track_squat/smoke.sh`) has not run yet, and no training has started.

The design record is in `runs/track_squat/design/`: `plan.md`, the four component
specs, the two critiques, and `AMENDMENTS.md`, which wins wherever the documents disagree.

## The setup

| | |
|---|---|
| Target | `fightAndSports1_subject1`, 166.0–179.0 s: 651 frames at 50 fps, four deep squat bounces done on the feet, then a walk-off. It is a held-out clip. The pelvis goes down to 0.24 m above the soles and the knees reach 2.8 rad. Nothing but the feet comes within 3.8 cm of the ground. The pelvis tilts up to 46°, a tilt margin of 0.14. |
| Learner | PPO from the tensor path of safety_sb3, plus AMP (Peng et al. 2021). The tracking reward follows BeyondMimic; the style reward is learned from the 31 train clips. |
| Filter | The world-model tube filter `wmmin:1:0.95`, with no dwell (`d0`) or a 1 s dwell (`d1`). Its fallback is a frozen thermal SHAC policy. |
| Arms (6) | `t70_bare`, `t70_wm_d0`, `t70_wm_d1`, `cbf_bare`, `cbf_wm_d0`, `cbf_wm_d1` |
| Seeds | A pilot at seed 2, then seeds 0–2. Task seed *i* uses fallback and tube seed *i*. |

There are two worlds, one per fallback:

- **thermal70** (`g1_track_squat_thermal70`). No barrier. The margin is min(tilt, non-foot contact, (70 °C − hottest winding) / 10 °C), and a winding reaching 70 °C ends the episode as a failure. `t70_bare` is the user's "no safety filter at all".
- **thermal_cbf** (`g1_track_squat_thermal_cbf`). A 70 °C torque-clipping barrier sits under every controller. The margin covers falls only, and overheating does not end the episode.

### Decisions the user made

These are settled; do not reopen them.

- The filter counts as part of the environment. PPO stores its own proposed action and that action's log-prob. Both the task reward and the style reward are paid on the transitions that were actually executed. There is no penalty for interventions and no loss masking.
- The tubes are the frozen calibrations under `runs/shac_thermal{70,_cbf}/wmreport/calib/`.
- **Starts.** Half the episodes start at frame 0. The other half start at a uniformly random frame that leaves at least 1 s of the segment. Start temperatures are drawn as in fallback training.
- **Action box.** The task policy acts in **±6.5**; the ±3 box caps the knee's PD target at 1.72 rad. The fallback keeps ±3.
- **AMP.** The discriminator's policy samples are only "free" pairs: transitions in which the task policy produced both states.
- **Segment.** The user first picked `fallAndGetUp1_subject4` 30–39.6 s. Forward kinematics showed it is a kneel: a shin is within 3 cm of the soles or below them on about 230 of 289 frames, so perfect tracking would count as a fall. Every deep "squat" in that clip turned out to be a kneel or a get-up.

## How to run

All commands run from the repo root in the `mjlab316` env.

1. **AMP expert set.** Run once on CPU, about 15 s: `python examples/build_amp_expert.py`. It produces 339,177 pairs from the 31 train clips. 24,796 rows have the waist clamped into the robot's range.
2. **GPU smoke.** Submit `configs/sweeps/track_squat/smoke.sh`; its header has the sbatch line. It covers:
   - the layout references and both reference audits (including the oracle replay through all six tubes);
   - the task-env checks and the filter's parity and benchmark on the task env;
   - all 6 arms for 2M steps each, with self-checks;
   - an eval, the SONIC anchor and the report.
3. **Pilot.** `pilot_jobs.txt` holds the 6 arms at seed 2, 1e9 steps each. `ckpt_eval_jobs.txt` produces the learning curves and `pilot_eval_jobs.txt` the evaluations. `python examples/track_report.py --stage pilot` writes the report, including the **budget rule**.
   - For each run, t95 is the first step at which the EMA of `err_q` is within 5% of its final value.
   - B = clamp(roundup_100M(1.25 · max t95), 4e8, 1.5e9).
   - If any run still improved `err_q` by more than 5% over its last 20% of steps, B = 1.5e9.
4. **Final runs.** Run `python configs/sweeps/track_squat/make_jobs.py --budget B`, then submit `jobs.txt`.
   - Seeds 0 and 1 train for B steps.
   - When B ≤ 1e9, seed 2 is promoted from its pilot checkpoint at B (`examples/track_promote.py`). This is valid because no schedule depends on the total step count: the learning rate is KL-adaptive, gamma is constant and the std clamp is fixed.
5. **Evals and report.** Submit `eval_jobs.txt`; its last line runs `track_report.py --stage final`.

## What is measured

### Training (logged every rollout by `TrackStatsEnv`)

Failures use one taxonomy in both worlds: an episode's **first failure event** is either an overheat (hottest winding ≥ 70 °C, whether or not it ends the episode) or a fall (a tilt or non-foot-contact termination). The trainer logs:

- `track/fail_<cause>_per_1k`, `track/fail_<cause>_per_ep` and `track/cum_fail_<cause>`;
- `track/end_<cause>_per_ep`: how episodes actually ended;
- a hazard per phase bin;
- the same counts split by start type (frame 0 vs random phase) and by time since reset (before or after 0.5 s);
- thermal metrics: per-step `tw_max_mean`, plus per-episode `tw_peak`, `dtw_peak` and `heat_dose`;
- task metrics: `err_body`, `err_q`, `tracked_frac`, `completion`, `tf_frac`, and `amp/ep_task_return`, which is the task-only return;
- from the filter: `filter/*`, including `engaged_frac`, hand-overs, holds and the coverage probe at hand-overs in the d1 arms;
- from AMP: `amp/*`, including the style reward on free vs engaged steps and the discriminator's statistics.

### Eval (`examples/eval_track.py`)

The primary eval is the Full suite: frame-0 starts, run to the end of the segment, observation and spawn noise off, deterministic actor clamped to ±6.5. Every condition starts from the same states (hashes are recorded and checked).

- Each policy is evaluated in **both** worlds, bare and under the d0 and d1 filters, each at all **3** fallback seeds.
- In its own world it also gets the c = 0.90 sensitivity tube and the `wmcalib` forced-hand-over coverage condition.
- The SONIC anchor tracks the same segment with its action clamped to ±6.5 (its native clamp is ±20).

Metrics:

- survival and any-failure (with Wilson intervals);
- first-failure incidence per cause;
- **TF**: the share of the segment's frames tracked to within 0.1 m mean body error, so frames after a failure count against it;
- completion;
- `dtw_peak`: the rise of the motor that peaked highest;
- heat dose / T: the mean excess of the hottest winding over its start temperature, over the whole segment;
- Σ τ² per step;
- S(phase), error, temperature and engagement curves by phase;
- all of the above stratified by the hottest start winding: [30, 60), [60, 66) and [66, 70] °C;
- engagement in excess of the audit's **oracle floor**: the share of frames on which each tube engages even under perfect tracking.

### Report (`examples/track_report.py`)

The unit of inference is the training run. Three primary endpoints, any-failure, TF and `dtw_peak`, feed three pre-registered contrasts:

- **P1, learning effect**: the `W_wm_dX` policy minus the `W_bare` policy, both deployed with the same dX filter.
- **P2, deployment effect**: each policy with the filter minus the same policy bare.
- **P3, world contrast**: descriptive only.

Each contrast reports per-seed differences, a bootstrap interval over seeds, and a McNemar test of any-failure on paired starts within each seed. Every cbf summary is shown with and without fallback seed 0. The report also gives the nondeterminism floor: one policy and condition evaluated twice. Training curves are saved as PNGs.

## Caveats

- **The filter engages even on perfect tracking.** Near the 46° lean, the tilt margin falls below some tubes' eps. On the deepest 3 s, the oracle replay engages on these shares of frames:

  | Tube seed | thermal70 | cbf |
  |---|---|---|
  | s0 | 8.1% | 29.5% |
  | s1 | 2.0% | 10.1% |
  | s2 | 6.7% | 1.3% |

  Engagement is reported as the excess over this floor.
- **The fallback and world model are off-distribution here.** They were trained on 3 s post-kick recoveries, never on deep squats. The tubes' 95% coverage was calibrated on SONIC's post-kick states, so it does not carry over to this task; the `wmcalib` condition and the d1 hand-over probe measure what coverage actually holds.
- **The proposal can leave the box the world model was trained on.** The world model was trained on actions within ±3, while the task policy proposes up to ±6.5. The first imagined action is therefore out of distribution, as SONIC's ±20 was in the earlier filter evals.
- **The clip was in the tube-calibration set.** Calibration used the held-out clips except fallAndGetUp and sprint. Describe the segment as "outside fallback and world-model training", not as "outside calibration".
- **Under AMP, style acts as an implicit intervention penalty.** The discriminator scores the fallback's motion as fake. `amp/r_style_gap` logs the size of this effect, and no performance metric uses style.
- **The two worlds differ in five ways at once**: dynamics, termination, margin, fallback and eps. P3 is descriptive, which is why `cbf_bare` exists.
- **The sim is not bit-reproducible**, so paired episodes can diverge. The repeat eval measures that floor.
- **Hot starts.** In a start drawn near 70 °C an episode can be infeasible in every arm. Results are stratified by the hottest start winding.

## Files

| Path | What |
|---|---|
| `robot_safety_sandbox/envs/g1_track/`, `tasks/g1_track.py` | the task: segment, command, rewards, episode bus |
| `robot_safety_sandbox/filters/wm_tube.py`, `filtered_env.py` (`WMTubeFilteredEnv`) | the training-time filter, matching the eval's verdicts bit for bit (`tests/test_wm_tube_parity.py`) |
| `robot_safety_sandbox/amp/` | AMP features, discriminator, env wrapper, `AMPPPO` |
| `robot_safety_sandbox/track_stats.py` | the training metrics |
| `examples/track_common.py` | arms, paths, the wrapper stack |
| `examples/train_track_filtered.py` (`--family track`) | the trainer; recipe in `configs/g1_track_squat_ampppo.yaml` |
| `examples/eval_track.py`, `examples/track_report.py`, `examples/track_promote.py` | eval, report, promotion |
| `examples/build_amp_expert.py`, `examples/audit_track_reference.py`, `examples/smoke_g1_track.py`, `examples/smoke_wm_tube.py` | data and checks |
| `configs/sweeps/track_squat/` | `smoke.sh`, `make_jobs.py`, the job files, `quiet.sh` |
| `tests/test_amp.py`, `tests/test_track_scripts.py`, `tests/test_g1_track_*.py`, `tests/test_wm_tube_*.py` | CPU tests |
