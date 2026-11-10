"""The sweep generator (examples/make_sweep.py) on the real sweep file,
configs/sweeps/lafan_recovery.yaml, and the job runner's skip rules.

Simulator-free (CPU; the env checks import the package and its task
registrations, no sim is built):

  * MERGE -- top-level keys replace, ``env_overrides`` / ``random_push`` merge
    per key with each value replacing the base's whole (an arm's ``weights``
    is the complete mixture), and the base is never modified;
  * VALIDITY -- every generated config is accepted by the on-policy trainer's
    OWN ``merge_config`` (its real parser, captured from ``main``), resolves
    to the arm's env / push overrides, and names a registered task whose cfg
    builder takes every env override;
  * DE-DUPLICATION -- arms identical to the reference (S3 = P-env = O0) train
    once and the later ones reference its runs; gated arms (S5) go to their
    own list; ``--only`` keeps the canonical runs another experiment owns;
  * SMOKE -- the smoke block's short run, own directories, ``--no-wandb`` /
    ``--quick`` on the command lines;
  * PREP -- ``${force_max}`` / ``${load}`` substituted, ``# creates:`` and
    report ``# always`` annotations for run_jobs.sh, which skips done jobs.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EX = os.path.join(_ROOT, "examples")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EX)

import make_sweep as ms  # noqa: E402

SWEEP = os.path.join("configs", "sweeps", "lafan_recovery.yaml")


@pytest.fixture(scope="module")
def full(tmp_path_factory):
  d = tmp_path_factory.mktemp("sweep")
  return ms.generate(SWEEP, out_dir=str(d)), d


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
  d = tmp_path_factory.mktemp("sweep_smoke")
  return ms.generate(SWEEP, smoke=True, out_dir=str(d)), d


def _arm(out, name):
  return next(a for a in out["arms"] if f"{a['exp']}/{a['arm']}" == name)


def test_merge_replaces_top_level_and_merges_passthroughs_per_key():
  base = {"lr": 1e-4, "task": "t", "env_overrides": {
    "weights": {"nominal": 0.1, "mocap": 0.7}, "sampler": "weighted", "levels": 5},
    "random_push": {"max_pushes": 1, "duration_s": 0.2}}
  arm = {"task": "u", "env_overrides": {"weights": {"nominal": 1.0}, "sampler": "uniform"},
         "random_push": {"max_pushes": 0}, "reset_value": True}
  out = ms.merge_arm(base, arm)
  assert out["task"] == "u" and out["reset_value"] is True and out["lr"] == 1e-4
  assert out["env_overrides"] == {"weights": {"nominal": 1.0}, "sampler": "uniform",
                                  "levels": 5}              # weights replaced whole
  assert out["random_push"] == {"max_pushes": 0, "duration_s": 0.2}
  assert base["env_overrides"]["weights"] == {"nominal": 0.1, "mocap": 0.7}
  assert base["task"] == "t"                                # base untouched
  assert ms.merge_arm(base, None) == base and ms.merge_arm(base, None) is not base


def test_trainer_dests_are_read_from_the_trainer_source():
  d = ms.trainer_dests()
  for k in ("num_envs", "privileged_critic", "critic_warmup_steps", "force_start",
            "random_push_override", "env_override", "reset_value", "no_wandb",
            "video_wandb", "adaptive_lr", "task"):
    assert k in d, k
  assert "config" not in d and "help" not in d
  assert ms.check_keys({"num_envs": 1, "env_overrides": {}, "family": "on_policy"}, d) == []
  assert ms.check_keys({"bogus_knob": 1}, d)
  assert ms.check_keys({"env_overrides": [1]}, d)


def _trainer_parser():
  """The on-policy trainer's real argparse parser, captured from its main()
  (merge_config is swapped for a stub that grabs the parser and stops)."""
  import train_on_policy as tp

  class _Got(Exception):
    pass

  box = {}

  def grab(p):
    box["p"] = p
    raise _Got

  orig, argv = tp.merge_config, sys.argv
  tp.merge_config, sys.argv = grab, ["train_on_policy.py", "--task", "g1_recover"]
  try:
    tp.main()
  except _Got:
    pass
  finally:
    tp.merge_config, sys.argv = orig, argv
  return box["p"]


def test_every_generated_config_parses_with_the_trainers_merge_config(full, tmp_path):
  import _run_config
  out, d = full
  paths = sorted({a["own_config_path"] for a in out["arms"]})
  assert len(paths) == 13                     # every arm of E3 / E4 / E5
  for path in paths:
    with open(path) as f:
      cfg = yaml.safe_load(f)
    parser = _trainer_parser()                # fresh: merge_config mutates defaults
    argv = sys.argv
    sys.argv = ["train_on_policy.py", "--config", path, "--out", str(tmp_path),
                "--seed", "1"]
    try:
      args = _run_config.merge_config(parser)
    finally:
      sys.argv = argv
    assert args.task == cfg["task"] and args.seed == 1
    assert args.env_overrides == cfg["env_overrides"]
    assert args.random_push == cfg.get("random_push", {})
    assert args.num_envs == 4096 and args.steps == 500_000_000
    assert args.privileged_critic is True and args.force_start == args.force_max


def test_generated_configs_carry_the_arms(full):
  out, _ = full
  base = _arm(out, "E3/S3")["config"]
  assert base["task"] == "g1_recover_lafan"
  assert base["env_overrides"]["sampler"] == "weighted"
  assert base["env_overrides"]["weights"]["mocap"] == pytest.approx(0.70)
  walker = sum(v for k, v in base["env_overrides"]["weights"].items()
               if k not in ("mocap", "handover"))
  assert walker == pytest.approx(0.30)
  s0 = _arm(out, "E3/S0")["config"]["env_overrides"]
  assert s0["dataset"] == "data/g1_recover_states.pt" and s0["weights"]["mocap"] == 0.0
  assert _arm(out, "E3/S1")["config"]["env_overrides"]["sampler"] == "uniform"
  assert _arm(out, "E3/S2")["config"]["env_overrides"]["sampler"] == "ladder"
  assert _arm(out, "E3/S4")["config"]["env_overrides"]["sampler"] == "online"
  assert _arm(out, "E3/S5")["config"]["env_overrides"]["weight_column"] == "p_value"
  for name, bank in (("E4/P-pre-self", "data/g1_self_handover_train_labeled.pt"),
                     ("E4/P-pre-nominal", "data/g1_handover_train_labeled.pt")):
    c = _arm(out, name)["config"]
    assert c["random_push"] == {"max_pushes": 0}
    assert c["env_overrides"]["dataset"][-1] == bank
    assert c["env_overrides"]["weights"]["handover"] == pytest.approx(0.70)
    assert c["env_overrides"]["weights"]["mocap"] == 0.0
  mix = _arm(out, "E4/P-mix")["config"]
  assert "random_push" not in mix and len(mix["env_overrides"]["dataset"]) == 3
  assert mix["env_overrides"]["weights"]["mocap"] == mix["env_overrides"]["weights"]["handover"]
  for name, task in (("E5/O1", "g1_recover_lafan_ra"), ("E5/O2", "g1_recover_lafan_ra_loose")):
    c = _arm(out, name)["config"]
    assert c["task"] == task and c["reset_value"] is True


def test_reference_arms_train_once_and_gated_arms_wait(full):
  out, d = full
  for name in ("E4/P-env", "E5/O0"):
    a = _arm(out, name)
    assert a["same_as"] == "E3/S3" and a["canonical"] == "E3/S3"
    assert a["runs"] == _arm(out, "E3/S3")["runs"]
  assert _arm(out, "E3/S3")["same_as"] is None
  # 10 unique ungated arms x 2 seeds; S5 x 2 in the gated list
  assert len(out["jobs"]) == 20 and len(set(out["jobs"])) == 20
  assert not any("S5_s" in j for j in out["jobs"])
  assert sum(l.startswith("python examples/train.py") for l in out["gated"]) == 2
  assert all("--out runs/lafan_recovery/" in j for j in out["jobs"])
  assert out["jobs"][0].endswith("--seed 0") and out["jobs"][-1].endswith("--seed 1")
  s3 = [j for j in out["jobs"] if "/E3/S3_s" in j]
  assert len(s3) == 2 and not any("P-env" in j or "/O0_" in j for j in out["jobs"])
  # one eval per trained run + the two baselines, on the frozen grid
  frozen = [e for e in out["evals"] if "/eval_frozen" in e]
  assert len(frozen) == 22
  assert all("--frozen --grid-file configs/sweeps/lafan_recovery/eval_grid.yaml" in e
             for e in out["evals"])
  # the suite proper plus the opt-in walker (E3) and pre (E4) read-outs
  assert all(" --setting std --setting lafan --setting handover --setting walker "
             "--setting pre " in e for e in frozen)
  # E5's staying pass: its 3 arms x 2 seeds (O0 = S3's runs) + the baselines,
  # E-handover only, a 6 s window, its own dirs and report
  stay = [e for e in out["evals"] if "/eval_stay" in e]
  assert len(stay) == 3 * 2 + 2 and len(out["evals"]) == 30
  assert all("--setting handover --eval-s 6.0" in e and "--setting std" not in e
             for e in stay)
  assert any("--out runs/lafan_recovery/E3/S3_s0/eval_stay" in e for e in stay)
  rs = next(r for r in out["reports"] if "/E5/report_stay" in r)
  assert "O0_s1=runs/lafan_recovery/E3/S3_s1/eval_stay" in rs
  assert len(out["reports"]) == 4
  assert any("push_fast_privcrit=runs/g1_recover_push_fast_privcrit" in e
             for e in out["evals"])
  assert all(e.split("=", 1)[1].split()[0].endswith("/g1_recover_lafan/final_model.zip")
             or "baselines" in e or "lafan_ra" in e for e in out["evals"])
  # E4's report names the shared run as P-env
  e4 = next(r for r in out["reports"] if "/E4/report" in r)
  assert "P-env_s0=runs/lafan_recovery/E3/S3_s0/eval_frozen" in e4
  assert e4.endswith("# always")
  for f in ("jobs.txt", "jobs_gated.txt", "eval_jobs.txt", "report_jobs.txt",
            "prep_jobs.txt", "manifest.json", "E4_P-env.yaml"):
    assert (d / f).exists(), f


def test_only_keeps_the_canonical_run_of_another_experiment(tmp_path):
  out = ms.generate(SWEEP, only=["E4"], out_dir=str(tmp_path), check_env_on=False)
  assert any("/E3/S3_s0 " in j + " " for j in out["jobs"])     # P-env's run
  assert not any("/E3/S1_s" in j for j in out["jobs"])
  assert len(out["jobs"]) == 4 * 2
  assert len(out["reports"]) == 1                 # E4's; the stay pass is E5's
  assert (tmp_path / "E3_S3.yaml").exists() and not (tmp_path / "E3_S1.yaml").exists()
  with pytest.raises(SystemExit):
    ms.generate(SWEEP, only=["E9"], write=False, check_env_on=False)


def test_smoke_variant(smoke):
  out, d = smoke
  assert out["runs_root"] == "runs/smoke/lafan_recovery" and out["seeds"] == [0]
  assert len(out["jobs"]) == 10
  for j in out["jobs"]:
    assert j.endswith("--no-wandb") and "--out runs/smoke/lafan_recovery/" in j
  assert all("--quick" in e and "--frozen" not in e for e in out["evals"])
  for a in out["arms"]:
    c = a["config"]
    assert c["steps"] == 2_000_000 and c["num_envs"] == 1024
    assert c["critic_warmup_steps"] == 0
    assert c["checkpoint_every"] > c["steps"] and c["video_interval"] > c["steps"]
    assert c["env_overrides"]["sampler"] in ("weighted", "uniform", "ladder", "online")
  # the smoke configs are still the arms (same overrides)
  assert _arm(out, "E5/O1")["config"]["task"] == "g1_recover_lafan_ra"


def test_force_max_override_and_prep_templates(tmp_path):
  out = ms.generate(SWEEP, force_max=260.0, out_dir=str(tmp_path), check_env_on=False)
  for a in out["arms"]:
    assert a["config"]["force_max"] == a["config"]["force_start"] == 260.0
  prep = [l for l in out["prep"] if not l.startswith("#")]
  assert len(prep) == 9
  assert "$" not in " ".join(prep)
  label = next(l for l in prep if "--out data/g1_recover_states_lafan1_train_labeled.pt" in l)
  assert "--force-max 260" in label and "--policy runs/g1_recover_push_fast_privcrit" in label
  assert label.endswith("# creates: data/g1_recover_states_lafan1_train_labeled.pt")
  assert prep[0].startswith("python examples/build_g1_recover_mocap_states.py --library")
  out2 = ms.generate(SWEEP, python="/opt/py/bin/python", write=False, check_env_on=False)
  assert all(l.startswith("/opt/py/bin/python ") for l in out2["jobs"] + out2["evals"])


def test_job_lines_are_checked_against_the_scripts_clis():
  assert ms.check_command("python examples/label_start_states.py --policy x --k 8 "
                          "--out y.pt  # creates: y.pt") == []
  assert ms.check_command("python examples/train.py --config c.yaml --out r --seed 0 "
                          "--no-wandb") == []
  # the handover harvester takes --impulse-range, not --force-max
  assert ms.check_command("python examples/harvest_g1_handover_states.py --force-max 200")
  assert ms.check_command("python examples/no_such_script.py --x 1")
  assert ms.check_command("echo not a python job") == []


def test_every_read_path_exists_or_is_made_by_prep():
  out = ms.generate(SWEEP, write=False, check_env_on=False)
  creates = {p for st in ms.load_sweep(os.path.join(ms._ZOO, SWEEP))["prep"]
             for p in st.get("creates", [])}
  for a in out["arms"]:
    for p in ms._paths_read(a["config"]):
      assert p in creates or os.path.exists(os.path.join(ms._ZOO, p)) or p.startswith(
        "runs/"), p
  assert ms.check_paths(["data/definitely_not_a_bank.pt"], creates)


def test_run_jobs_skips_done_jobs(tmp_path):
  runs = tmp_path / "r"
  (runs / "a" / "g1_recover_lafan").mkdir(parents=True)
  (runs / "a" / "g1_recover_lafan" / "final_model.zip").write_text("x")
  (runs / "e").mkdir()
  (runs / "e" / "summary.json").write_text("{}")
  made = tmp_path / "made.pt"
  made.write_text("x")
  marker = tmp_path / "ran.txt"
  jobs = tmp_path / "jobs.txt"
  jobs.write_text("\n".join([
    "# a comment",
    # ':' swallows its arguments, so '--out DIR' is visible to the runner only
    f": --out {runs / 'a'}; echo train >> {marker}",             # done: final model
    f": --out {runs / 'e'}; echo eval >> {marker}",              # done: summary.json
    f"echo bank >> {marker}  # creates: {made}",                # done: file exists
    f": --out {runs / 'e'}; echo report >> {marker}  # always",  # never skipped
    f": --out {runs / 'b'}; echo todo >> {marker}",              # runs
    "",
  ]) + "\n")
  sh = os.path.join(_EX, "run_jobs.sh")
  r = subprocess.run(["bash", sh, str(jobs), "--logs", str(tmp_path / "logs"), "--quiet"],
                     capture_output=True, text=True)
  assert r.returncode == 0, r.stdout + r.stderr
  assert marker.read_text().split() == ["report", "todo"]
  assert len(list((tmp_path / "logs").iterdir())) == 2
  r = subprocess.run(["bash", sh, str(jobs), "--list"], capture_output=True, text=True)
  status = [l.split()[1] for l in r.stdout.splitlines()]
  assert status == ["done", "done", "done", "todo", "todo"]
  # --only N runs job N alone; a failing job stops the file unless --keep-going
  marker.write_text("")
  r = subprocess.run(["bash", sh, str(jobs), "--only", "5", "--force", "--quiet",
                      "--logs", str(tmp_path / "logs")], capture_output=True, text=True)
  assert r.returncode == 0 and marker.read_text().split() == ["todo"]
  bad = tmp_path / "bad.txt"
  bad.write_text(f"false\necho after >> {marker}\n")
  marker.write_text("")
  r = subprocess.run(["bash", sh, str(bad), "--quiet", "--logs", str(tmp_path / "l2")],
                     capture_output=True, text=True)
  assert r.returncode != 0 and marker.read_text() == ""
  r = subprocess.run(["bash", sh, str(bad), "--quiet", "--keep-going",
                      "--logs", str(tmp_path / "l3")], capture_output=True, text=True)
  assert r.returncode != 0 and marker.read_text().split() == ["after"]
