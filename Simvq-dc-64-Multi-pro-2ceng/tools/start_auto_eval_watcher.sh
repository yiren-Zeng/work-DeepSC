#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p experiments/logs

PID_FILE="experiments/auto_eval_watcher.pid"
LOG_FILE="experiments/logs/auto_eval_watcher.log"
PATTERN="python -u tools/auto_evaluate_completed_experiments.py --watch"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Auto-eval watcher is already running: PID $(cat "$PID_FILE")"
    exit 0
fi

setsid -f bash -c \
    "cd '$PWD' && exec python -u tools/auto_evaluate_completed_experiments.py --watch --gpu '${AUTO_EVAL_GPU_ID:-0}' --poll-seconds '${AUTO_EVAL_POLL_SECONDS:-300}' >> '$LOG_FILE' 2>&1 < /dev/null"

sleep 1
PID="$(ps -eo pid,args | awk -v pattern="$PATTERN" 'index($0, pattern) { print $1; exit }')"
if [ -z "$PID" ]; then
    echo "Auto-eval watcher failed to start. Check $LOG_FILE" >&2
    exit 1
fi
echo "$PID" > "$PID_FILE"
echo "Started auto-eval watcher: PID $PID"
echo "Log: $LOG_FILE"
