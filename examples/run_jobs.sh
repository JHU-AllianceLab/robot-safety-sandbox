#!/usr/bin/env bash
# Run a jobs file (one shell command per line) one job at a time.
#
#   examples/run_jobs.sh configs/sweeps/lafan_recovery/jobs.txt
#   examples/run_jobs.sh JOBS --keep-going             # a failed job does not stop the rest
#   examples/run_jobs.sh JOBS --only 3                 # just job 3 (e.g. a cluster array index)
#   examples/run_jobs.sh JOBS --list                   # number, status and command of each job
#
# The files come from examples/make_sweep.py. Blank lines and lines starting
# with '#' are not jobs; jobs are numbered 1, 2, ... in file order, so --only N
# is stable for a given file and can be driven by any scheduler's array index
# (SLURM_ARRAY_TASK_ID, PBS_ARRAY_INDEX, ...), one job per allocation.
#
# A job whose outputs already exist is SKIPPED (rerun the file after an
# interruption and it picks up where it stopped; --force runs everything):
#   * a trailing "# always" annotation -> never skipped (the merged reports,
#     which must be redone as more evals finish);
#   * a trailing "# creates: PATH ..." annotation -> done when every PATH exists;
#   * otherwise its "--out DIR" -> done when DIR/final_model.zip or
#     DIR/*/final_model.zip exists (a training run: the trainer writes into
#     DIR/<task>/), when DIR/summary.json exists (a push-suite eval or report),
#     or when DIR is an existing FILE (a bank-writing script).
# A job with neither always runs. A training run that died mid-way has no
# final_model.zip yet, so it reruns from scratch.
#
# Every job runs from the repository root through `bash -c`, its output going
# to <logs>/<NNN>_<slug>.log (default logs/<jobs dir>_<jobs file stem>/,
# git-ignored) and, unless
# --quiet, to the terminal as well.

set -u

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
  echo "options: --logs DIR  --keep-going  --force  --only N  --from N  --list  --dry-run  --quiet"
  exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 2
[[ "$1" == "-h" || "$1" == "--help" ]] && usage 0
JOBS="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$JOBS" = /* ]] || JOBS="$PWD/$JOBS"
[[ -f "$JOBS" ]] || { echo "[run_jobs] no such jobs file: $JOBS" >&2; exit 2; }
stem="$(basename "$JOBS")"; stem="${stem%.*}"
LOGS="$ROOT/logs/$(basename "$(dirname "$JOBS")")_$stem"
KEEP=0; FORCE=0; ONLY=""; FROM=1; LIST=0; DRY=0; QUIET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs) LOGS="$2"; shift 2 ;;
    --keep-going) KEEP=1; shift ;;
    --force) FORCE=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[run_jobs] unknown option $1" >&2; usage 2 ;;
  esac
done

cd "$ROOT" || exit 2

# is_done LINE -> 0 when the job's outputs exist (see the header)
is_done() {
  local line="$1" out paths p
  [[ "$line" =~ \#[[:space:]]*always[[:space:]]*$ ]] && return 1
  if [[ "$line" =~ \#[[:space:]]*creates:[[:space:]]*(.*)$ ]]; then
    paths="${BASH_REMATCH[1]}"
    for p in $paths; do
      [[ -e "$p" ]] || return 1
    done
    return 0
  fi
  if [[ "$line" =~ --out[=[:space:]]+([^[:space:]#;]+) ]]; then
    out="${BASH_REMATCH[1]}"
    [[ -f "$out/final_model.zip" || -f "$out/summary.json" ]] && return 0
    compgen -G "$out/*/final_model.zip" > /dev/null && return 0
    [[ -f "$out" ]] && return 0
  fi
  return 1
}

slug() {
  local line="$1" s
  if [[ "$line" =~ --out[=[:space:]]+([^[:space:]#;]+) ]]; then
    s="${BASH_REMATCH[1]}"
  else
    s="$(echo "$line" | awk '{print $2}')"
  fi
  s="${s#runs/}"; s="${s#data/}"
  echo "$s" | tr '/ =' '___' | tr -cd '[:alnum:]_.-' | cut -c1-80
}

n=0; ran=0; skipped=0; failed=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line#"${line%%[![:space:]]*}"}"          # strip leading blanks
  [[ -z "$line" || "$line" == \#* ]] && continue
  n=$((n + 1))
  [[ -n "$ONLY" && "$n" != "$ONLY" ]] && continue
  [[ "$n" -lt "$FROM" ]] && continue
  if is_done "$line"; then status="done"; else status="todo"; fi
  if [[ "$LIST" == 1 ]]; then
    printf '%3d  %-4s  %s\n' "$n" "$status" "$line"
    continue
  fi
  if [[ "$status" == "done" && "$FORCE" == 0 ]]; then
    echo "[run_jobs] #$n skip (outputs exist): $line"
    skipped=$((skipped + 1))
    continue
  fi
  log="$LOGS/$(printf '%03d' "$n")_$(slug "$line").log"
  echo "[run_jobs] #$n start $(date '+%F %T') -> $log"
  echo "           $line"
  if [[ "$DRY" == 1 ]]; then
    continue
  fi
  mkdir -p "$LOGS"
  t0=$(date +%s)
  if [[ "$QUIET" == 1 ]]; then
    bash -c "$line" > "$log" 2>&1 < /dev/null
    rc=$?
  else
    bash -c "$line" < /dev/null 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  fi
  dt=$(( $(date +%s) - t0 ))
  ran=$((ran + 1))
  if [[ "$rc" -ne 0 ]]; then
    failed=$((failed + 1))
    echo "[run_jobs] #$n FAILED (exit $rc, ${dt}s); log: $log"
    if [[ "$KEEP" == 0 ]]; then
      echo "[run_jobs] stopping (pass --keep-going to continue past failures)"
      exit "$rc"
    fi
  else
    echo "[run_jobs] #$n ok (${dt}s)"
  fi
done < "$JOBS"

[[ "$LIST" == 1 ]] && exit 0
echo "[run_jobs] $JOBS: $ran ran ($failed failed), $skipped skipped"
[[ "$failed" -eq 0 ]]
