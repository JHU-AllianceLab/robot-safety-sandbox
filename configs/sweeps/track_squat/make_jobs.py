"""Write the squat-tracking job files (run from the repository root).

    python configs/sweeps/track_squat/make_jobs.py                       # pilot files + final at B = 1e9
    python configs/sweeps/track_squat/make_jobs.py --budget 600000000    # after the pilot: the final files

Files (one shell command per line, for examples/run_jobs.sh / slurm_jobs.sbatch):
  pilot_jobs.txt        6 arms x the pilot seed, PILOT_STEPS each
  pilot_eval_jobs.txt   each pilot policy in both worlds (bare / wm_d0 / wm_d1 x the 3
                        fallback seeds, + c=0.90 and wmcalib in its own world), the SONIC
                        anchor in both worlds, a repeat of one eval (nondeterminism floor),
                        then the report
  ckpt_eval_jobs.txt    learning curves: every 100M checkpoint of each pilot run
  jobs.txt              the final runs at the budget B: seeds 0 and 1 trained, the pilot
                        seed PROMOTED from its pilot checkpoint when B <= PILOT_STEPS
                        (else retrained)
  eval_jobs.txt         the final evals, as the pilot's, for all 3 seeds, then the report
Every Python line runs through quiet.sh (drops MuJoCo Warp's line-search spam, keeps
the exit status).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "examples"))

import track_common as tc  # noqa: E402

PILOT_STEPS = 1_000_000_000
CFG = "configs/g1_track_squat_ampppo.yaml"
Q = "configs/sweeps/track_squat/quiet.sh"


def train_line(arm, seed, stage, steps):
  out = tc.run_dir(stage, arm, seed)
  return (f"{Q} python examples/train.py --config {CFG} --arm {arm} --seed {seed} --stage {stage} "
          f"--steps {steps} --out {out}")


def eval_lines(stage, seeds, extra_own=True):
  lines = []
  for arm in tc.ARMS:
    own = tc.arm_world(arm)
    for s in seeds:
      for w in tc.WORLDS:
        conds = "bare wm_d0 wm_d1" + (" wm_d0_c90 wm_d1_c90 wmcalib" if extra_own and w == own
                                      else "")
        out = tc.eval_dir(stage, arm, s, w)
        lines.append(f"{Q} python examples/eval_track.py --stage {stage} --arm {arm} --seed {s} "
                     f"--world {w} --conditions {conds} --out {out}")
  for w in tc.WORLDS:
    out = os.path.join(tc.RUNS, "eval", stage, "sonic", w)
    lines.append(f"{Q} python examples/eval_track.py --sonic --world {w} --conditions bare wm_d0 "
                 f"wm_d1 --out {out}")
  # the nondeterminism floor: one policy, its native condition, twice
  arm, s = "t70_wm_d1", seeds[-1]
  out = os.path.join(tc.RUNS, "eval", stage, "repeat", f"{arm}_s{s}")
  lines.append(f"{Q} python examples/eval_track.py --stage {stage} --arm {arm} --seed {s} "
               f"--world thermal70 --conditions wm_d1 --fb-seeds {s} --repeat 2 --out {out}")
  lines.append(f"{Q} python examples/track_report.py --stage {stage}  # always")
  return lines


def write(name, header, lines):
  path = os.path.join(HERE, name)
  with open(path, "w") as f:
    f.write("".join(f"# {h}\n" if h else "#\n" for h in header))
    f.write("\n".join(lines) + "\n")
  print(f"[make-jobs] {path}: {len(lines)} jobs")


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--budget", type=int, default=PILOT_STEPS,
                 help="the final runs' length B (from the pilot, docs page's budget rule)")
  a = p.parse_args(argv)
  arms = list(tc.ARMS)
  sub = ("sbatch -A hhu49_alliance1 -p rtx6000,h100,h200,a100,l40s --array=1-{n} --gres=gpu:1 "
         "--cpus-per-task=8 --mem=64G --time={t} --job-name={j} "
         "--output=logs/slurm/%x_%A_%a.out examples/slurm_jobs.sbatch "
         "configs/sweeps/track_squat/{f}")
  write("pilot_jobs.txt", [
    "Squat tracking, PILOT: the 6 arms at the pilot seed (2: the cbf seed-0 tube is the outlier),",
    f"{PILOT_STEPS:.0e} steps each. After smoke.sh passes:",
    "  " + sub.format(n=len(arms), t="36:00:00", j="track_pilot", f="pilot_jobs.txt")],
    [train_line(arm, tc.PILOT_SEED, "pilot", PILOT_STEPS) for arm in arms])
  ev = eval_lines("pilot", [tc.PILOT_SEED])
  write("pilot_eval_jobs.txt", [
    "Squat tracking, PILOT evals (examples/eval_track.py) then the report (last line).",
    "  " + sub.format(n=len(ev), t="08:00:00", j="track_pilot_eval", f="pilot_eval_jobs.txt")
    + "  (--dependency=afterok:<pilot array>)"], ev)
  ck = [f"{Q} python examples/eval_track.py --stage pilot --arm {arm} --seed {tc.PILOT_SEED} "
        f"--world {tc.arm_world(arm)} --curve-every 100000000 --episodes 256 "
        f"--out {tc.eval_dir('pilot', arm, tc.PILOT_SEED, tc.arm_world(arm))}/curve"
        for arm in arms]
  write("ckpt_eval_jobs.txt", [
    "Learning curves of the pilot: every 100M checkpoint, native condition, 256 episodes (C12).",
    "Writes <eval dir>/curve/summary.json.",
    "  " + sub.format(n=len(ck), t="12:00:00", j="track_ckpt_eval", f="ckpt_eval_jobs.txt")], ck)

  B = int(a.budget)
  final = []
  for arm in arms:
    for s in tc.SEEDS:
      if s == tc.PILOT_SEED and B <= PILOT_STEPS:
        dst = tc.run_dir("final", arm, s)
        final.append(f"python examples/track_promote.py --arm {arm} --seed {s} --budget {B}  "
                     f"# creates: {dst}/final_model.zip")
      else:
        final.append(train_line(arm, s, "final", B))
  write("jobs.txt", [
    f"Squat tracking, FINAL runs at the budget B = {B:.3g} steps (6 arms x seeds 0-2; task seed i",
    "uses fallback / tube seed i). The pilot seed is promoted from its pilot checkpoint when",
    f"B <= {PILOT_STEPS:.0e} (schedules do not depend on the total). Regenerate with --budget B.",
    "  " + sub.format(n=len(final), t="36:00:00", j="track_final", f="jobs.txt")], final)
  ev = eval_lines("final", list(tc.SEEDS))
  write("eval_jobs.txt", [
    "Squat tracking, FINAL evals (all seeds, both worlds, SONIC anchor, repeat) then the report.",
    "  " + sub.format(n=len(ev), t="08:00:00", j="track_eval", f="eval_jobs.txt")
    + "  (--dependency=afterok:<final array>)"], ev)


if __name__ == "__main__":
  main()
