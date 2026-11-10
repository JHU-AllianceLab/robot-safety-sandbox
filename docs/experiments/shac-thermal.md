# SHAC with a motor-temperature limit: setup and runbook

This page sets up one experiment: the no-pelvis-height SHAC fallback of
[the world-model ablation](shac-world-model-ablation.md) (stage A, 113-D
world-model state), retrained on the lab's identified G1 with its motor heat
model in the loop and one more constraint: **every motor winding stays below
85 °C**. It covers what changed (software stack, robot, heat model, task),
the decisions behind each change, the commands, and the checks that have run.

**Status (2026-11-10).** Code and configs are written, unit-tested,
checked on a GPU, and the end-to-end smoke test passed (job 1768194, §7;
two earlier ones failed on the migration traps of §8). The 3-seed training
(array 1768777; started 05:25 / 08:24 on an A100 / two L40S, 21k / 29k fps)
and the per-seed report (array 1768778, `aftercorr` on training) are
running. Everything is committed on `feat/shac-thermal-mjc316` (§1); the
cluster has no GitHub credentials, so push it from a terminal that does.

## 1. Software stack: a new branch and a new conda env

The heat model needs the robot's real motor torques. The identified robot
drives every joint with MuJoCo's `<dcmotor>` actuator, which needs MuJoCo ≥
3.16 and mjlab main. The old stack (conda env `mjlab`: MuJoCo 3.6, mjlab
1.2.0) stays untouched; everything here runs on a new one.

| What | Value |
|---|---|
| Branch | `feat/shac-thermal-mjc316`, from `origin/mjc-316` (Zizhe's port of the sandbox to MuJoCo / Warp 3.16, mjlab main `dcdf6ce5`, commit `7720479`) |
| Checkout | `~/robot-safety-sandbox-mjc316` (a `git worktree` of the main checkout). `data/` and `runs/` are symlinks to the same scratch directories as the main checkout's. |
| What it holds | `origin/mjc-316`, then: a merge of `feat/sonic-push-bench-mjlab`'s 6 commits (`a09ed3e`, also on `origin/feat/sonic-shac-vs-ppo`; only `README.md` conflicted, on the Python requirement line); one commit with that branch's work that was uncommitted on 2026-11-09 (the no-height / student-WM experiments, the push bench, their configs and docs); one commit with the migration fixes and this experiment. |
| Conda env | `mjlab316`: Python 3.12, torch 2.14.1+cu130, mujoco 3.16.0, mujoco-warp 3.16.0, mjlab main `dcdf6ce5` (reports 1.6.0), warp-lang from NVIDIA's index; `safety_sb3` editable from `../safety-stable-baselines` (branch `feat/supermimic`, the same checkout the old env uses); the sandbox editable from the worktree |
| Cluster drivers | 580.178 (CUDA 13.0), so the cu130 wheels run on every GPU partition, the RTX PRO 6000 (sm_120) included. torch 2.14 has no cu128 build, and its cu126 build has no Blackwell kernels. |
| SLURM | `examples/slurm_jobs.sbatch` on this branch activates `${ZOO_CONDA_ENV:-mjlab316}`. The older experiments' own drivers under `configs/sweeps/` still activate `mjlab`: they describe runs made on the old stack. |

Recreate the env:

```bash
conda create -y -n mjlab316 python=3.12 && conda activate mjlab316
pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
pip install --extra-index-url https://pypi.nvidia.com \
  "mjlab @ git+https://github.com/mujocolab/mjlab.git@dcdf6ce519abfd2ea45e9503ae6fb650b75bb869" \
  "mujoco==3.16.0" "mujoco-warp==3.16.0"
pip install wandb tensorboard imageio imageio-ffmpeg moviepy onnx scipy matplotlib pytest \
  pyyaml tqdm trimesh gymnasium "stable_baselines3>=2.0"
pip install -e ~/safety-stable-baselines --no-deps        # branch feat/supermimic (SHAC)
pip install -e ~/robot-safety-sandbox-mjc316 --no-deps
```

Results on this stack are not bit-comparable with the old one (Zizhe's port
notes: mjlab's terrain columns, actuator defaults and reset semantics
changed). Compare the thermal runs with a no-height rerun on this stack, not
with `runs/shac_noheight` (§9).

## 2. The robot and the heat model

Both come from the lab's sysID repo, `~/G1-robot-model-with-SysID`
(`$G1_SYSID_DIR`), v0.4.0, commit `69f2fc6`. `envs/assets/g1_sysid.py` reads
it in place and **pins** it: the four files that set the robot and the
temperatures (`thermal_params.json`, `sim_heat_scale.json`,
`g1_sysid_params.py`, `g1_sysid/thermal.py`) must match recorded SHA-256
hashes, or every task that uses them refuses to build. A `git pull` of the
sysID repo therefore cannot change a run between seeds, or between training
and evaluation. To move to a newer model, update `PINNED` on purpose.
Training writes the provenance to `<run>/<task>_safetyshac1p/g1_sysid.json`.

**The robot** (`get_g1_sysid_robot_cfg`; the sysID repo's
`g1_sysid_constants.py` rebuilt here, since that file expects to be copied
into mjlab):

- mjlab's G1 with the measured waist ranges (yaw ±2.708, roll −0.674 / +0.650,
  pitch −0.632 / +0.503 rad) and refitted collision capsules (a new chest
  capsule, slimmer hip / wrist / hand capsules);
- every joint a `<dcmotor>` in POSITION mode: mjlab's kp / kd, peak torque and
  armature, Unitree's K and R, 52 V bus. Below the voltage limit it applies
  exactly the PD torque of mjlab's position actuator (checked: 5e-6 N·m at
  rest). It keeps no activation state (`na = 0`).
- the hip pitch is the identified 7520-14. `use_sonic_hip_pitch` (`--hip
  sonic`) now refuses a dcmotor robot.

**The heat model** (`g1_sysid.MotorThermal`, v0.4): per motor, winding Tw
(the robot's `temperature[1]`), housing Th (`temperature[0]`) and the
structure Ts it is bolted to. Heat comes from torque² (with standstill
heating) and from speed. `sim_heat_scale=True`: the simulated robot makes
less torque than the real one, and the sysID repo's per-motor scale corrects
the copper heat; this is the repo's rule for simulated torques. Accuracy and
limits (sysID README): ~0.8 °C median on held-out real data; up to ~9 °C too
cool for loaded ankle / waist motors near 60–70 °C; simulated hip-roll and
ankle temperatures are still low after the scale (1–4 °C on a walk).

How a few torques heat a winding at 80 °C (housing = winding, standing
still, scale on): 10 N·m on one ankle linkage motor ≈ 6 °C/s; 20 N·m on a knee
≈ 0.8 °C/s; 40 N·m on a hip pitch ≈ 3.6 °C/s. A 30 °C winding–housing gap
cools an unloaded winding at 1.3–2.4 °C/s. A 3 s episode can therefore cross
the limit from a hot start.

## 3. The task

`g1_recover_thermal` (train) / `g1_recover_lafan_thermal` (the eval twin,
with the LAFAN push kwargs) = `g1_recover_noheight` / `g1_recover_lafan_noheight`
on the identified G1, plus (`envs/g1_recover/thermal.py`):

| Piece | What |
|---|---|
| State | `MotorThermalState` per env: Tw, Th, Ts, `(num_envs, 29)`, Unitree motor order. mjlab's actuator / joint orders are mapped by name. |
| Integration | a metrics term with `per_substep=True` (`ThermalIntegrator`): mjlab calls it after **every physics substep** (5 ms), before terminations and the margin read the state. Each substep's heat uses that substep's actuator force and joint speed, split over the ankle / waist linkages (`motor_torque`, `motor_speed`). Its episode metric `motor_temp_max` is the peak winding temperature. |
| Start temperatures | at every reset: per robot a level T0 ~ U(30, 80) °C; each winding ~ U(30, T0); each housing = winding − U(0, 20) °C, floored at 30; structure = housing (§4) |
| Margin | `g_fall_noheight_thermal` = min(tilt, non-foot contact, (85 − max Tw) / 10), clamped ±3 by `compose`. The thermal term is 1 at 75 °C and 0 at 85 °C. |
| Termination | `motor_overheat`: max Tw ≥ 85 °C ends the episode as a failure (the G1's own controller switches to damping at 85 °C). The margin hook anchors g at −0.1, as for a fall. |
| Actor obs | the 98 terms of the no-height task + `motor_temp` (29 winding temperatures, ±0.5 °C uniform noise: the robot reports whole degrees) = **127-D** |
| Critic group = world-model state | the 113-D no-height group + `motor_temp` and `motor_temp_housing` (noise-free) = **171-D**. Ts is left out: it is not measured and does not move within an episode. Both added terms are sensors on the real robot. |
| Sim context | unchanged (33-D: foot friction, torso COM, encoder biases) |

`g1_recover_lafan_sysid` is `g1_recover_lafan` (margin `g_fall`, its
terminations) on the identified G1 without temperatures; it exists to
harvest SONIC's start states on that robot.

The rest of the world is the no-height task's: flat ground, zero command, 3 s
episodes with rest truncation (0.4 s dwell, armed at 0.8 s), no in-episode
push, the same domain randomization.

## 4. Decisions, and why

| Decision | Choice | Why |
|---|---|---|
| Stack | mjc-316 branch + new env `mjlab316` | the dcmotor robot needs MuJoCo 3.16 / mjlab main; the old setup stays intact |
| Robot | the full identified G1 (dcmotor) | the user's choice; same gains, peak torques and armature as mjlab's G1 |
| Policy sees temperatures | yes, the 29 windings (±0.5 °C) | measured on the real G1, so still deployable; without them the policy cannot act differently hot vs cold |
| Margin / limit | (85 − max Tw) / 10; terminate at 85 °C | competes with the fall terms only in the last ~10 °C; 85 °C is where the robot goes limp |
| Start windings | level T0 ~ U(30, 80) per robot, each motor U(30, T0) | the hottest motor lands near T0, spread evenly from cold to hot; real sessions have a few hot motors (hottest 51–74 °C) while the median motor sits at 37–44 °C. Independent per-motor draws would put the hottest motor at ~78 °C in almost every episode. |
| Start housings | winding − U(0, 20) °C, ≥ 30; structure = housing | the recordings show winding − housing of 0–2 °C typically and up to ~27 °C right after heavy load. Housing = winding would leave no cooling path, so from 80 °C a loaded standing ankle crosses 85 °C in about a second. |
| Thermal model | sysID v0.4, `sim_heat_scale` on, pinned | the repo's rules for simulated torques; pinned for reproducibility |
| Bank | re-harvested on the identified G1 with its own hip | the start states must come from the robot that trains on them |

Not done (possible follow-ups): randomizing the battery voltage (49–52.8 V)
and `sim_heat_scale` (±30 %), both suggested by the sysID README. Either
would add a privileged input to the world model's context.

## 5. Data: the start bank

`configs/sweeps/shac_thermal/data_jobs.txt` reruns the no-height bank recipe
on `g1_recover_lafan_sysid`: the SONIC harvest (train clips, seed 0, 50k
episodes, kick |Δv| ~ U(0, 5) m/s), rows 0 / 0.1 / 0.2 / 0.3 s after the kick,
then the η = 0.1 net → `data/g1_sonic_push_gfall_sysid316_train_bank_net.pt`.
One difference from the earlier banks: those were harvested with SONIC's
7520-22 hip actuator (the harvest's default `--hip sonic`), which the
identified robot does not have; this one runs `--hip mjlab`, i.e. the robot's
own hip. The bank carries no temperatures; they are drawn at every reset.

## 6. Training, calibration and evaluation

Config `configs/g1_recover_sonic_thermal_safetyshac1p.yaml`: the no-height
SHAC config verbatim but for the task and the bank (1.5B steps × 3 seeds,
4096 envs, horizon 32, world model 3×1024).

```bash
cd ~/robot-safety-sandbox-mjc316 && mkdir -p logs/slurm
P=rtx6000,l40s,h100,a100,h200
# 0. bank + smoke test (training 3M steps, CPU-rollout validation, the report on 64 episodes)
SM=$(sbatch --parsable -A hhu49_alliance1 -p $P --gres=gpu:1 --cpus-per-task=32 --mem=96G \
     --time=02:00:00 --job-name=thermal_prep --output=logs/slurm/%x_%j.out \
     configs/sweeps/shac_thermal/prep_and_smoke.sh)
# 1. training, 3 seeds
TR=$(sbatch --parsable -A hhu49_alliance1 -p rtx6000,h100,h200,a100,l40s --dependency=afterok:$SM \
     --array=1-3 --gres=gpu:1 --cpus-per-task=8 --time=36:00:00 --mem=64G --job-name=shac_thermal \
     --output=logs/slurm/%x_%A_%a.out examples/slurm_jobs.sbatch configs/sweeps/shac_thermal/jobs.txt)
# 2. per seed: calibration (2048 eps) + tube fit + filter evaluation (1024 eps); the last writes the report
sbatch -A hhu49_alliance1 -p l40s,rtx6000,h100 --dependency=aftercorr:$TR --array=1-3 --gres=gpu:1 \
    --cpus-per-task=32 --mem=96G --time=01:30:00 --job-name=wmreport_thermal \
    --output=logs/slurm/%x_%A_%a.out configs/sweeps/shac_thermal/wmreport_seed.sh
```

The evaluation is the no-height report's (`wmreport_jobs.txt`), with these
differences:

- **Failures.** An episode also fails when a winding reaches 85 °C, SONIC's
  episodes included.
- **Paired start temperatures.** Every episode draws its start temperatures
  at the spawn from torch's RNG. The eval reseeds it per wave with a
  condition-independent key, so every condition sees the same temperatures.
- **New per-episode records.** `eval_sonic_qcbf.py` records `overheated`,
  `tw_start_max_c` and `tw_peak_c` for each episode.
- **Report.** `wm_filter_report.py` prints the overheated share and the mean
  peak winding per filter.
- **Exact-sim filter.** The CPU rollout filter (`rollout:1`,
  `cpu_rollout.py`) seeds the temperatures from the live env, integrates the
  same heat model after every `mj_step`, adds the thermal term to its margin,
  and accepts the dcmotor actuators.

Outputs: `runs/shac_thermal/shac_s{0,1,2}/g1_recover_thermal_safetyshac1p/`,
`runs/shac_thermal/wmreport/{calib,eval}/thermal_s*/`,
`runs/shac_thermal/wmreport/report.{json,log}`.

## 7. Checks

| Check | Result |
|---|---|
| `tests/test_g1_thermal.py` (pins, start draw, motor-order mapping vs `MotorThermal`, margin, termination, registrations and obs layout, CPU rollout heat step) + `tests/test_g1_margins.py` | 28 passed (mjlab316, CPU) |
| The compiled identified robot (CPU MuJoCo 3.16) | 29 dcmotor actuators (`mjBIAS_DCMOTOR`, POSITION input), `na = 0`, measured waist ranges, 69 geoms (+ chest capsule); at rest the dcmotor's torque equals the position actuator's to 5e-6 N·m. mjlab's G1 XML has no joint friction or damping, so mjlab main's "keep the XML" default changes nothing. |
| GPU: build + step both tasks for 3 s, 1024 envs, A100 (`configs/sweeps/shac_thermal/checks/thermal_gpu_check.py`) | `g1_recover_noheight` on the new stack: actor 98, critic 113, context 33, ~29.4k env-steps/s. `g1_recover_thermal`: actor 127, critic 171, the world-model spec maps all 127 actor dims, context 33, ~29.6k env-steps/s (the per-substep heat model costs nothing measurable). Start draws: hottest winding 30.0–79.8 °C (mean 54.3), median motor 42.7 °C, mean winding − housing 6.4 °C; the actor's `motor_temp` is Tw ± 0.5. |
| Full test suite on `mjlab316` (CPU node) | all pass but `test_filter_equivalence.py` (2 tests, fails on the old stack too) after the two fixes of §8 (`test_g1_waist.py` re-run: pass) |
| Smoke test (`prep_and_smoke.sh`) | jobs 1767084 and 1767704 failed in the harvest (§8, traps 1 and 1b). **Job 1768194 passed** (14 min): bank 198,768 rows (the old stack's: 199,102); CPU rollout vs GPU env: winding temperatures after one step within 1.1e-5 °C, one-step margin p99 0.016 (old stack's no-height smoke: 0.015), rollout verdicts 100 % equal, failure times equal; calibration + every filter of the report ran on 64 episodes, with identical start temperatures in every condition. SONIC alone: 54.7 % survive the kick, **9.4 % overheat**, hottest winding 55.9 → 64.1 °C (means). The 3M-step smoke policy survives nothing (expected). |

## 8. Migration traps (old stack → MuJoCo 3.16 / mjlab main)

1. **The clip time grid.** mjlab main's `MotionLoader` resamples at
   `arange(n) * dt` and keeps a clip's end point when it lands on the 50 fps
   grid; mjlab 1.2 used `arange(0, duration, dt)`. The clip libraries
   (`data/g1_lafan1_clips_*.pt`) were built on 1.2, so `SonicReference.build`
   refused them: first on length ("clip lengths differ from the library's
   clip_frames", the first smoke test, job 1767084), and, after cutting clips
   to length, on orientation (max 1 − |⟨q, q'⟩| = 1.26e-4 > 1e-4: the float32
   time stamps differ slightly). Fix: `SonicReference.build` resamples on the
   grid the library was built on (`sonic._motion_loader`). A library without
   `meta["time_grid"]` (all existing ones) gets mjlab 1.2's grid;
   `build_g1_recover_mocap_states.py` now writes `time_grid="mjlab-main"`. Both
   LAFAN libraries now rebuild exactly on a CPU node (40 clips, 441,120 frames,
   max 1 − |⟨q, q'⟩| 1.8e-6 on both stacks), so SONIC tracks the same
   references as in every earlier run.
1b. **The orientation check itself is ill-conditioned.** On the GPU nodes
   (smoke test 1767704) the same check failed on ONE row of 339,208
   (1.54e-3): a frame of `fallAndGetUp2_subject3` with the pelvis at 90.2°
   tilt, where the yaw that `to_heading_frame` removes is undefined and
   last-ulp differences between machines swing the yaw-free quaternion.
   Old and new stack agree on a CPU node; the GPU nodes' CPUs give a
   different answer for that frame. `build` now checks the gravity direction
   in the root frame (well-conditioned; max diff ≤ 1e-2 rad, measured
   3.1e-3 on a GPU node) and the quaternion on the best 99.9 % of rows
   (≤ 1e-4). Diagnostics: `configs/sweeps/shac_thermal/checks/heading_check_diag.py`,
   `heading_old_vs_new.sh`.
2. **Command terms** must take `env_ids` in `_update_command` (Zizhe's port
   fixed the velocity command; `g1_waist`'s `WaistTargetCommand` needed the
   same).
3. `tests/test_filter_equivalence.py` (2 tests) fails on both stacks: the
   legacy snapshots differ by ≤ 2e-5 (Zizhe's port notes).
4. A login node's `/tmp` is invisible to compute nodes: keep scripts that a
   job runs on the shared file system (the checks of §7 are in
   `configs/sweeps/shac_thermal/checks/`; `train_progress.py` summarizes the
   training logs per 50M steps).

## 9. Caveats

- **Not comparable with the old runs.** New stack, new robot, new bank:
  survival and tube widths here do not compare with `runs/shac_noheight`. For
  the cost of the thermal constraint alone, train the no-height task on the
  identified robot on this stack as a control (a `g1_recover_noheight_sysid`
  task would be one line in the registry).
- **Rest truncation.** A calm robot ends its episode as a truncation after
  0.4 s in the stance set, and the critic bootstraps. A hot robot standing
  still keeps heating (standstill heating is the worst case), so its value at
  rest relies on the critic having learned that from the other episodes.
- **Sim temperatures for hip roll and ankles** are still somewhat low even
  with the heat scale (sysID README), and the model's weak spot is a loaded
  joint held still near 60–70 °C, where the real motor can be ~9 °C hotter.
  The 85 °C limit has no extra margin for that.
