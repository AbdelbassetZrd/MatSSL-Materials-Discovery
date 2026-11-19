#!/usr/bin/env bash
# Runs the full experiment campaign: 3 SSL datasets x 2 unlabeled-pool settings.
# Two queues run in parallel (one per pool setting); logs in data/logs/.
cd "$(dirname "$0")/.."
run_queue () {
  pool=$1; shift
  for task in "$@"; do
    uv run python -u pipeline/run_ssl_experiments.py "$task" --pool "$pool" --seeds 3 \
      > "data/logs/${task}_${pool}.log" 2>&1
    echo "finished $task $pool: $(tail -1 data/logs/${task}_${pool}.log)"
  done
}
run_queue indist magnetism metallicity stability &
run_queue gnome stability metallicity magnetism &
wait
echo "ALL DONE"
