#!/bin/bash
# Run a command with MuJoCo Warp's "linesearch iterations limit" warnings dropped from
# its output (2-3k lines per log block, ~750 MB per 6 h job on this stack), keeping the
# command's own exit status (a job line's pipe would otherwise report grep's).
#   configs/sweeps/track_squat/quiet.sh python examples/train.py ...
set -o pipefail
"$@" 2>&1 | grep --line-buffered -v "linesearch iterations limit"
exit "${PIPESTATUS[0]}"
