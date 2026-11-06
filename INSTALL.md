# Installation

Two packages, one benchmark:

| repo | provides | depends on |
|---|---|---|
| [`safety-stable-baselines`](https://github.com/SafeRoboticsLab/safety-stable-baselines) (`safety_sb3`) | the ALGORITHMS: the MAP's M x A x P — `{Safety,ReachAvoid}{PPO,SAC}{1P,2P}` — plus the tensor path and buffers | SB3, torch |
| `robot-safety-sandbox` (this repo) | the ENVIRONMENTS: task registry (one `mode` axis: avoid / reach-avoid / cumulative), margins, bridges, trainers | mjlab, safety_sb3 |

Neither depends on `unitree_rl_mjlab` (the legacy research repo) — the zoo is
self-contained and this independence is CI-tested by installing on a clean
machine (see Verification).

## Requirements

- Linux, NVIDIA GPU (tested: RTX 5090 / Blackwell `sm_120`), driver with
  CUDA 13.0 support (≥ 580) for the cu130 torch wheels; on CUDA 12 drivers use
  the cu126 wheels instead (see step 1).
- Python ≥ 3.11 (mjlab main dropped 3.10), conda recommended.

## Install (from scratch)

```bash
conda create -y -n mjlab316 python=3.12
conda activate mjlab316

# 1. torch >= 2.14 (required by mjlab main). cu130 needs a CUDA 13 driver;
#    on CUDA 12 drivers use .../whl/cu126 instead:
pip install "torch>=2.14.0" --index-url https://download.pytorch.org/whl/cu130

# 2. STOCK mjlab, pinned to a main-branch commit: no released mjlab supports
#    MuJoCo 3.16 yet (1.6.0 pins ~=3.11). This commit pins
#    mujoco~=3.16.0 / mujoco-warp~=3.16.0 itself; we pin the exact patch too.
#    warp-lang comes from NVIDIA's index.
pip install --extra-index-url https://pypi.nvidia.com \
  "mjlab @ git+https://github.com/mujocolab/mjlab.git@dcdf6ce519abfd2ea45e9503ae6fb650b75bb869" \
  "mujoco==3.16.0" "mujoco-warp==3.16.0"

# 3. this package. safety_sb3 is a declared pip dependency (pinned release
#    tag from GitHub), so one editable install pulls both:
pip install wandb tensorboard imageio moviepy
pip install -e path/to/robot-safety-sandbox

#    developing BOTH packages at once? install safety_sb3 editable FIRST and
#    it satisfies the requirement:
# pip install -e path/to/safety-stable-baselines
# pip install -e path/to/robot-safety-sandbox
```

**Headless machines** (no display — clusters, ssh boxes): eval-video rendering
needs `export MUJOCO_GL=egl` (NVIDIA EGL; verified on a headless RTX 5090) and
`moviepy` (wandb's video encoder). Without them training crashes at the first
video interval with an OpenGL-context / `wandb.Video requires moviepy` error.

## Verification

```bash
# registry imports; Digit tasks warn-and-skip on stock mjlab (expected — see Notes)
python -c "from robot_safety_sandbox import list_tasks; print(list_tasks())"

# algorithm math (CPU, no GPU/simulator needed):
python path/to/safety-stable-baselines/tests/test_tensor_sac.py

# end-to-end GPU training smoke (~1 min; warp JIT-compiles kernels for your
# arch on the FIRST env build — expect a one-time pause):
python examples/train.py --family on_policy --task go2_stabilize \
    --num-envs 256 --steps 500000 \
    --net 128,128,128 --ent-coef 0 --no-adaptive-lr --no-wandb
```

Trainers: `examples/train.py` (every task — it routes on the RL family, then on
the task's `mode`: safety_sb3 learners for the safety modes, stock SB3 PPO for
`mode="cumulative"` task policies), `examples/eval.py` (the evaluation
harness: env + filter + nominal + attack, all chosen independently). See
README/PORTING.md.

## Notes & known deviations

- **Digit tasks** (`digit_stabilize*`): assets and builders are vendored
  in-repo (first-class). Cfg construction and import are stock-mjlab-clean;
  SIMULATION currently needs the lab's mjlab fork for its ball-joint
  entity patch (see envs/digit_safety/builders.py) until upstreamed.
- **mjlab version**: the zoo targets mjlab main @ `dcdf6ce5` (2026-10-31,
  MuJoCo / MuJoCo Warp 3.16). Moving from the old mjlab 1.2.0 / MuJoCo 3.6
  pin changes simulation results: checkpoints and eval baselines produced on
  the old stack are not bit-for-bit reproducible here. Behavior deltas worth
  knowing about:
  - curriculum terrains: mjlab now uses exactly one column per sub-terrain
    and `proportion` weights robot spawning rather than column count (affects
    the parkour curriculum terrain, which asked for 10 columns / 5 types);
  - `ActuatorCfg.frictionloss`/`armature` default to `None` (keep XML)
    instead of `0.0`; Digit's actuators pin `frictionloss=0.0` to keep the
    old dynamics;
  - `reset(env_ids=...)` no longer advances observation history or
    stateful commands.
- **The sim-stack pins are mandatory, not conservative**: mjlab and
  mujoco-warp break APIs between minor releases; the pinned set above is the
  validated one.
- **GPU memory**: per-task footprints scale with `--num-envs`; see the
  task docstrings. The Go2 tasks want 4–9 GB at 2–3k envs; the classic
  Robust-Gymnasium ports (feat/robust-gym-mjlab branch) are tiny (<1 GB).
- **Cross-arch**: no code changes needed between sm_89 and sm_120 — only the
  warp kernel cache differs (per-machine, auto-built).
