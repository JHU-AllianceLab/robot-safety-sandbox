# SONIC push benchmark, rebuilt in mjlab

The friend's benchmark (`robot_states_reachability_checker/tools/sonic_bench`) pushes SONIC while it
tracks LAFAN clips and scores a safety mechanism on whether the robot stays in the safe set K. This
experiment trains our fallbacks (SafePPO, SHAC) in the same world and data distribution and scores
four mechanisms on his trial tables:

| Mechanism | Monitor | Fallback |
|---|---|---|
| SafePPO + exact-sim filter | `rollout:1` (CPU MuJoCo rollouts of "SONIC now, SafePPO after", 1 s, K's margin) | SafePPO |
| SHAC + WM tube | `wmmin:1:0.95` (one world-model rollout, conformal worst-margin offset), plus `+dwell2` / `+latch` | SHAC |
| SafePPO alone | none: SafePPO from the push onset | SafePPO |
| SHAC alone | none: SHAC from the push onset | SHAC |

## The world (`robot_safety_sandbox/envs/g1_recover/push_bench.py`, task `g1_recover_bench`)

| | Here | His |
|---|---|---|
| Failure | K: pelvis ≥ 0.45 m, tilt ≤ 1 rad, joints within range + 0.1 rad, no non-foot ground contact; read once per control tick | same K, checked every physics step |
| Margin | his: `min((z-0.45)/0.1, 1-tilt, joint room/0.2)`, clip ±3, -1 on non-foot contact, ≤ -0.1 outside K | same |
| Physics | 2 ms × 10 per tick, 100 iterations, tolerance 1e-10, 300 CCD, serial line search, ≤ 1024 constraints per world, `implicitfast` | same |
| Robot | mjlab's G1 (mass 33.341142 kg), SONIC's 7520-22 hip, position actuators | GR00T's G1 file (same mass), explicit PD torque |
| Friction | ground 1, soles 1.0 (priority 1) | ground 1, his robot file's soles |
| Randomization / noise | none / none | none / none |
| Fallback observation | 70-D exact: height, gravity, joints, heading-frame velocity, angular velocity, joint velocities, foot contacts | his connector's φ68 + foot contacts |
| SONIC | the sandbox port (`sonic.py`) | his Torch port of the same ONNX |

## The data

* **Training starts**: SONIC tracking his three training motions (`walk1_subject1`, `run1_subject2`,
  `dance1_subject1` to 105 s) under his state pool's force push (`--push force`: one of 12 bodies,
  an area-weighted surface point, 5-15 ticks, Δv 0.5-2.5 m/s, a uniform world direction, onset 1-2 s
  after the spawn). Rows 0 / 0.1 / 0.2 / 0.3 s after the push **ends**: no push inside a training
  episode, so SHAC's world model stays push-agnostic. Thinned to its η = 0.1 net.
* **Tables** (`examples/make_push_table.py`, his CSV schema plus `direction_frame`):
  * `data/zz_test/trials.csv`: **his dev-heldout table** (6 unseen motions, 8,640 pushes; sha256
    `07cdc6bf…`). His file, used as is (contact points on his robot's surfaces).
  * `data/push_bench/train_v1`: his training-motion table regenerated (seed 2026102802, 67 contexts,
    16,080 pushes; directions from the robot's actual heading, as his CPU table).
  * `data/push_bench/calib_v1`: the same generator with seed 2026110501, for the WM-tube calibration
    (2,048 trials sampled).

## A trial (`examples/eval_push_bench.py`)

Spawn on the library row of (motion, start frame) and run 100 ticks of SONIC **once** per trial
(a K exit there excludes the trial); its state at the onset is restored for every condition, as his
runner copies its warm state into both arms. Then 510 scored ticks: the push acts on ticks 0-9
(force `m Δv / 0.2 s`, our mass), the condition acts from tick 0, the trial survives if it stays in
K. One no-push control per context. GPU contact accumulation is not bit-deterministic:
`--repeat-sonic K` reruns SONIC alone K more times and counts the outcomes that change (the noise
floor of the paired counts).

## Run order

```bash
mkdir -p logs/slurm
# 0. smoke test (every stage at toy scale, ~1-2 h)
sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100 --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
    --job-name=push_bench_smoke --output=logs/slurm/%x_%j.out configs/sweeps/push_bench/smoke.sh
# 1. data (see data_jobs.txt for the sbatch line)
# 2. training: 3 seeds x {SHAC, SafePPO}
sbatch -A hhu49_alliance1 -p rtx6000 --array=1-6 --time=72:00:00 --job-name=push_bench_train \
    examples/slurm_jobs.sbatch configs/sweeps/push_bench/train_jobs.txt
# 3. evaluation (per seed: calibration, tube, both tables, both fallbacks)
sbatch -A hhu49_alliance1 -p rtx6000,l40s,h100 --array=1-3 --gres=gpu:1 --cpus-per-task=32 --mem=96G \
    --time=24:00:00 --job-name=push_bench_eval --output=logs/slurm/%x_%A_%a.out configs/sweeps/push_bench/eval_seed.sh
```

## Outputs

`runs/push_bench/eval/{dev,train}/<tag>/`:

* `summary.json`: per condition, survival of the pushed trials (Wilson 95 %), by Δv and by motion,
  rescued / harmed against SONIC (McNemar p), the no-push trigger rate, the fallback's share of
  scored ticks, hand-overs per trial;
* `<condition>/trials.npz`: his schema (first half the SONIC arm, second half the condition's), so
  his paired metrics run unchanged, from his repository:
  `python -m tools.sonic_bench.metrics wmmin=<dir>/wmmin_1_0.95 rollout=<dir>/rollout_1`.
  In his columns `mode` 0 means SONIC held the robot at the end (his "TASK"), 1 the fallback.
