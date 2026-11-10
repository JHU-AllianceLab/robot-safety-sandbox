# SHAC vs SafetyPPO from SONIC's post-kick states: runbook

This page is the record of one experiment: which fallback learner, SHAC or
SafetyPPO, trains the better G1 recovery policy from scratch, when every
episode starts from a state SONIC (the nominal whole-body tracker) is in just
after a kick. It says what the experiment is, where every piece lives, how to
launch it on the JHU ARCH cluster, what to look at while it runs, and how to
read the result.

**Status (2026-10-27).** The code is committed on branch
`feat/sonic-shac-vs-ppo` of this repository. Before running anything, check
that the branch has been pushed (`git ls-remote origin feat/sonic-shac-vs-ppo`).
The start banks exist on the RTX 5080 workstation. Both learners and the
evaluation passed a 3M-step smoke run there. Nothing has been launched on the
cluster yet.

## 1. The question and the design

**The question.** From the same starting states, at an equal sample budget,
which learner's fallback survives more often? Compare on held-out motion clips.
The earlier experiment `configs/sweeps/shac_vs_ppo` asked the same question
at 250M steps. Its start states came from the push policy's own post-push
states, and SHAC won: 0.61 vs 0.46 held-out survival over 3 seeds. Its
advantage was on starts pushed from behind. This experiment changes the start
states to SONIC's, and trains for 6× longer.

**The design.** The two arms share everything except the learner. Both copy
the shac_vs_ppo arms, `configs/g1_recover_selfhandover_safety{shac,ppo}1p.yaml`,
whose headers explain what the learners share and what they cannot.

| Setting | Value |
|---|---|
| Task | `g1_recover`: its own avoid margin `g_fall` (pelvis height, tilt, non-foot contact). The episode ends at a 70° tilt or a 10 N non-foot contact. This is **not** the ViaLasso safe set K. |
| Start states | `data/g1_sonic_push_gfall_train_bank_net.pt`, 199,103 rows ([§5](#5-regenerating-the-data)), every row drawn uniformly, no curriculum |
| Episode | 3 s cap, with rest truncation (dwell 0.4 s after 0.8 s). No in-episode push. |
| Policy | MLP 512-256-128, privileged critic (114-dim critic obs vs the actor's 98), gamma 0.99 fixed, action-rate / magnitude penalties 0.05 / 0.005 |
| Budget | 1.5B env steps per run, 3 seeds per learner, 4096 envs. Checkpoints every 50M. |
| Learner-specific | SHAC: world model 3×1024, horizon 32, LR cosine over `lr_decay_iters` = 146,500 (= 1.5B / (4096 × 2.5)). PPO: lr 1e-4, target_kl 0.01, ent 1e-3, std ≤ 0.4. |
| Configs | `configs/g1_recover_sonic_safetyshac1p.yaml`, `configs/g1_recover_sonic_safetyppo1p.yaml` |
| Jobs | `configs/sweeps/shac_vs_ppo_sonic/jobs.txt` (6 training runs), `eval_jobs.txt` (6 evaluations + summary) |

**What a start row is.** Each row was made the same way:

1. SONIC (`envs/g1_recover/sonic.py`, the pretrained GEAR-SONIC release)
   tracks a LAFAN train clip, starting from a library frame.
2. 0.5–1.5 s in, the pelvis velocity is set to a horizontal kick with
   |Δv| ~ U(0, 5) m/s, in a random direction.
3. The state is recorded 0, 0.1, 0.2 and 0.3 s after the kick. A row exists
   only if SONIC was still up at that moment.

The rows were harvested under g1_recover's own fall terminations
(`--task g1_recover_lafan`), so the experiment never touches K.

## 2. Where everything is

| What | Where |
|---|---|
| This repository on the cluster | `/home/jhu/yzeng52/robot-safety-sandbox`, branch `feat/sonic-shac-vs-ppo` (commit `a09ed3e` or later) |
| safety-stable-baselines on the cluster | `/home/jhu/yzeng52/safety-stable-baselines`, branch `feat/supermimic` (commit `f92a02f`, already on GitHub). It carries SHAC and the action penalties. `main` / `v0.4.0` have neither. |
| Cluster login | `ssh jhu-arch` (workstation `~/.ssh/config`: `yzeng52@login.arch.jhu.edu`, shared connection kept for 1 day) |
| Start banks (workstation only; `data/` is git-ignored) | train `data/g1_sonic_push_gfall_train_bank_net.pt` (106 MB), held-out `data/g1_sonic_push_gfall_test_bank_net.pt` (17 MB) |
| Full SONIC trajectories behind the banks | `data/g1_sonic_push_gfall_{train,test}.pt` (3.3 GB / 0.5 GB). Not needed for this experiment. |
| Outputs (on the cluster) | `runs/shac_vs_ppo_sonic/{shac,ppo}_s{0,1,2}/` (SHAC writes into `.../g1_recover_safetyshac1p/`, PPO into `.../g1_recover/`), `runs/shac_vs_ppo_sonic/eval/`, `runs/shac_vs_ppo_sonic/summary_table.json` |
| Job logs | `logs/shac_vs_ppo_sonic_jobs/NNN_*.log` and `logs/shac_vs_ppo_sonic_eval_jobs/` (`run_jobs.sh`), `logs/slurm_%A_%a.out` |

## 3. Runbook

### Step 0. Push the branch (workstation)

The workstation's VS Code terminals route git credentials through a socket
that goes stale. When that happens, `git push` fails with `ECONNREFUSED
/run/user/1001/vscode-git-*.sock`, and then `No anonymous write access`.
Bypass it, and give a GitHub personal access token as the password:

```bash
cd ~/Documents/robot-safety-sandbox
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_IPC_HANDLE VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_ASKPASS_NODE
git push -u origin feat/sonic-shac-vs-ppo
```

The token needs the `repo` scope (classic), or "Contents: read and write" on
JHU-AllianceLab/robot-safety-sandbox (fine-grained). If the organization
enforces SSO, authorize the token for it. For a lasting fix, add an SSH key
to GitHub and switch the remote to `git@github.com:JHU-AllianceLab/robot-safety-sandbox.git`.

### Step 1. Code on the cluster

```bash
ssh jhu-arch
cd ~/robot-safety-sandbox && git fetch origin && git checkout feat/sonic-shac-vs-ppo
cd ~/safety-stable-baselines && git fetch origin && git checkout feat/supermimic
pip install -e .        # AFTER the sandbox's own install: its pyproject pins safety_sb3 v0.4.0 (no SHAC)
python -c "from safety_sb3 import SafetySHAC1P, SafetyPPO1P; print('ok')"
```

Match the workstation's versions where you can. A different mjlab can change
the simulation. The workstation has torch 2.11.0+cu128, mjlab 1.2.0, mujoco
and mujoco-warp 3.6.0, warp-lang 1.12.0, stable_baselines3 2.9.0, and Python
3.10.

### Step 2. Data (workstation → cluster)

From the workstation repository root, with the `jhu-arch` connection alive
(`ssh -O check jhu-arch`):

```bash
rsync -avP data/g1_sonic_push_gfall_train_bank_net.pt data/g1_sonic_push_gfall_test_bank_net.pt \
      jhu-arch:/home/jhu/yzeng52/robot-safety-sandbox/data/
```

Copy the banks; do not regenerate them on the cluster. Every seed must train
on the same rows.

### Step 3. Check before launching (cluster)

```bash
cd ~/robot-safety-sandbox
examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/jobs.txt --list     # 6 'todo'
# optional ~5 min end-to-end smoke, the one that passed on the workstation:
python examples/train.py --config configs/g1_recover_sonic_safetyppo1p.yaml --steps 3000000 \
    --no-wandb --video-interval 0 --out runs/smoke/sonic_ppo
python examples/train.py --config configs/g1_recover_sonic_safetyshac1p.yaml --steps 3000000 \
    --wm-warmup-iters 50 --actor-delay-iters 20 --lr-decay-iters 300 \
    --no-wandb --video-interval 0 --out runs/smoke/sonic_shac
```

The first lines of every run's log must include:

```
[g1_recover] spawn mixture: ... handover 1.00 (199103 rows)
[critic] privileged: value net reads the 'critic' group (114 dims) vs the actor's 98    (PPO)
[algo] g1_recover -> SafetySHAC1P   /   [algo] g1_recover adversary=False -> SafetyPPO1P
```

If the handover row count is 0, the bank was not found. The spawn then falls
back to the nominal stance, and the run means nothing.

### Step 4. Launch (SLURM)

Odd jobs are SHAC and even jobs are PPO, in seed order 0, 0, 1, 1, 2, 2. At
the RTX 5080's throughput, a SHAC seed takes about 13.5 h (31k env-steps/s
with TF32) and a PPO seed about 5.2 h (80k/s). Adjust `--time` to the
cluster's GPUs and its partition limits.

```bash
# train.sbatch
#!/bin/bash
#SBATCH --job-name=shac_vs_ppo_sonic
#SBATCH --array=1-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%A_%a.out
cd ~/robot-safety-sandbox
source activate rl                # the environment of step 1
examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/jobs.txt --only "$SLURM_ARRAY_TASK_ID"
```

```bash
# eval.sbatch: same header without --array, with --time=03:00:00, and the last line
examples/run_jobs.sh configs/sweeps/shac_vs_ppo_sonic/eval_jobs.txt
```

```bash
mkdir -p logs
JID=$(sbatch --parsable train.sbatch)
sbatch --dependency=afterok:$JID eval.sbatch
```

Every run logs to the wandb project `shac_vs_ppo_sonic`. Run `wandb login`
on the cluster first, or `export WANDB_MODE=offline`. `run_jobs.sh` skips a
job whose `final_model.zip` exists. A run that died partway has none, so a
resubmission restarts it from scratch.

### Step 5. While it runs

These are the scalars to watch. They appear in the job log tables and in
wandb.

| Scalar | Meaning |
|---|---|
| `env/Curriculum/recover/surv_handover` | Training survival from the SONIC rows; it is stochastic and lags the deterministic eval. |
| `env/safety/failure_rate` | Fraction of episodes that end on a fall |
| `rollout/ep_len_mean` | ~25 steps means almost every episode falls at once (the from-scratch start). It grows toward the 150-step cap or the rest truncation. |
| SHAC: `actor_objective`, `actor_grad_norm` | Only after `wm_warmup_iters` (~10M steps) plus `actor_delay_iters` (~2M) does the actor start to move |

As a reference, a from-scratch PPO MLP on the analogous K-terminated SONIC
bank was at `surv_handover` 0.13 at 150M and 0.27 at 200M, still rising
steeply (`runs/lstm_vs_mlp`, workstation). Do not judge the runs before a few
hundred million steps.

### Step 6. Results

`eval_jobs.txt` runs `label_start_states.py` on each final policy. Each one
plays 4 deterministic rollouts from every row of the held-out bank (31,888
rows, LAFAN test clips, harvest seed 1), with no push, the training episode
and task `g1_recover`. `summarize_label_reports.py` then writes
`runs/shac_vs_ppo_sonic/summary_table.json`: mean ± std over seeds for each
learner, plus the action statistics. Compare `action_rate` / `action_mag`
across learners before comparing survival. SHAC's penalties are in value
units, PPO's are in an advantage-normalized loss, so equal coefficients do not
mean equally smooth policies.

Bring the results back to the workstation:

```bash
rsync -avP --exclude 'checkpoints/' \
      jhu-arch:/home/jhu/yzeng52/robot-safety-sandbox/runs/shac_vs_ppo_sonic/ runs/shac_vs_ppo_sonic/
```

## 4. Reference numbers

- **The bar to beat: SONIC itself.** It survives from 57.2 % of the held-out
  rows if it simply keeps control (57.5 % on the train bank; 0.570–0.575
  across delays 0–0.3 s). That is the `fell` column of the bank, i.e.
  SONIC's fate within 2.5 s of the kick. The fallback evaluation runs 3 s from
  the row, so the two horizons are close but not equal. A fallback below this
  bar makes a handover counter-productive on average. The value of the
  fallback is on the rows SONIC itself loses (42.8 %).
- **SONIC's survival by kick size** on the held-out clips (episode level):
  0.96 below 1.5 m/s, 0.91 at 1.5–2, 0.80 at 2–2.5, 0.57 at 2.5–3, 0.35 at
  3–3.5, 0.16 at 3.5–4, 0.04 above 4 m/s.
- **The 200M pilot** (`runs/lstm_vs_mlp`). It used the K task, a different bank
  and no action penalties, so it is not directly comparable. There, a
  from-scratch PPO MLP reached 0.28 held-out survival at 200M steps, against
  0.56 for SONIC on that bank. It beat a small LSTM (0.09).

## 5. Regenerating the data

The banks are deterministic given the seeds. To rebuild them from scratch on
an RTX 5080: about 4 min of harvest, plus seconds for the thinning.

```bash
python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan \
    --library data/g1_lafan1_clips_train.pt --episodes 50000 --seed 0 --out data/g1_sonic_push_gfall_train.pt
python examples/harvest_g1_sonic_trajectories.py --task g1_recover_lafan \
    --library data/g1_lafan1_clips_test.pt --episodes 8000 --seed 1 --out data/g1_sonic_push_gfall_test.pt
python examples/dedup_start_bank.py --bank data/g1_sonic_push_gfall_train_bank.pt
python examples/dedup_start_bank.py --bank data/g1_sonic_push_gfall_test_bank.pt \
    --W-from data/g1_sonic_push_gfall_train_bank_net.pt
```

These need the GEAR-SONIC release in `~/Documents/gear_sonic_hf` (or set
`$SONIC_DIR`) and the clip libraries `data/g1_lafan1_clips_{train,test}.pt`.
The harvester's defaults are what these banks used: SONIC's hip-pitch motor,
|Δv| ~ U(0, 5) m/s, onset 0.5–1.5 s, delays 0 / 0.1 / 0.2 / 0.3 s.

## 6. Caveats

- **The robot differs between harvest and training.** SONIC ran with its
  training hip-pitch actuator (7520-22). The fallback env keeps mjlab's G1
  (7520-14). The states transfer, because they are poses and twists, but the
  dynamics during the harvest are SONIC's robot's.
- **Bank labels are SONIC's, not the fallback's.** `fell` and
  `time_to_fall` say whether SONIC falls if it keeps control. Label the
  fallback with `label_start_states.py`.
- **The η-net does almost nothing here.** At η = 0.1 it keeps 199,103 of
  199,114 rows. SONIC's post-kick states are far apart in the ViaLasso metric
  (median nearest neighbour 2.0–2.8 units), unlike a gait pool.
- **Some rows start where SONIC's reference itself goes down.** A few percent
  of rows come from fallAndGetUp clips and kneels, where the LAFAN reference
  touches the floor. They stay in the bank. Exclude them per row through
  `traj_index` → `ep/ref_out_k` in the trajectory file, if needed.
- **Step-dependent settings.** If `steps` changes, rescale SHAC's
  `lr_decay_iters` (= steps / 10,240).

## 7. After this experiment

The policy-evaluation pipeline (`examples/policy_eval.sh`, built 2026-10-27)
fits γ = 1 values V^π / Q^π of a fixed fallback on SONIC's trajectory states,
and calibrates a conformal threshold on them. Its defaults use the ViaLasso
safe set: task `g1_recover_lafan_k`, with the K-harvested trajectory files
`data/g1_sonic_push_{train,test}.pt`. For this experiment's policies, decide
first whether safety means K or g1_recover's `g_fall`. For `g_fall`, pass
`--task g1_recover_lafan` to `label_policy_values.py`, and build the states
from the `gfall` trajectory files.
