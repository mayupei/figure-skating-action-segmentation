#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10000M
#SBATCH --output=/dev/null
#SBATCH --job-name=run_pipeline

set -euo pipefail

module load python
source ENV/bin/activate

# ------------------ Config ------------------
TARGET_COUNT="${TARGET_COUNT:-10}"      # wait until at least this many files exist
POLL_SECS="${POLL_SECS:-30}"            # seconds between checks
TIMEOUT_SECS="${TIMEOUT_SECS:-1200}"

# set up the log file
exec >"logs/pipeline.log" 2>&1

log() { printf '%s %s\n' "$(date +'%F %T')" "$*"; }

time_it() {
  local label="$1"; shift
  local start="$(date +%s)"
  { "$@"; rc=$?; } > /dev/null 2>&1
  local end="$(date +%s)"
  log "$label took $((end - start)) seconds."
}

# ------------------ 1) K-fold split ------------------
time_it "K fold split" python k_fold_split.py

# ------------------ 2) Training ------------------
# remove the previous version
export TARGET_COUNT POLL_SECS TIMEOUT_SECS
time_it "Training" bash -c '
rm -f models/*
# submit the training job
sbatch --wait train.sh > /dev/null

start_time=$(date +%s)
while :; do
  # Count regular files only, non-recursive; robust to spaces/newlines
  cnt=$(find models -maxdepth 1 -type f -printf . | wc -c)

  if [ "$cnt" -ge "$TARGET_COUNT" ]; then
    break
  fi
  now=$(date +%s)
  if (( now - start_time >= TIMEOUT_SECS )); then
    echo "ERROR: Timed out for training" >&2
    exit 1
  fi
  sleep "$POLL_SECS"
done
'

# ------------------ 3) evaluation ------------------
time_it "Evaluation" python evaluate.py

# ------------------ 4) generate an example ------------------
time_it "Generating a video example" python visualize_example.py --input n02_p08 --split 0