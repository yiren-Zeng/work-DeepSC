#!/usr/bin/env bash
# Run one of the two new experiments; restart unexpected exits from its own checkpoint.
set -uo pipefail

case "${1:-}" in
  k64_64_last2_copy_200ep)
    experiment="$1"
    gpu_id=3
    train_script="train_k64_64_last2_cascade_200ep.sh"
    ;;
  k512_16_last2_copy_200ep)
    experiment="$1"
    gpu_id=4
    train_script="train_k512_16_last2_cascade_200ep.sh"
    ;;
  *)
    echo "Usage: $0 {k64_64_last2_copy_200ep|k512_16_last2_copy_200ep}" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="/home/yi/vq_deepsc_runs_20260919/runtime/$experiment"
mkdir -p -- "$runtime_dir"
exec 9>"$runtime_dir/supervisor.lock"
if ! flock -n 9; then
  echo "Another supervisor already owns $experiment" >&2
  exit 73
fi

# Terminal hangups must not propagate to the supervisor or its training children.
trap '' HUP
export GPU_ID="$gpu_id"
export VQ_DEEPSC_RESUME=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$runtime_dir/supervisor.pid"
attempt=0

while [[ ! -f "$runtime_dir/STOP" ]]; do
  attempt=$((attempt + 1))
  printf '%s | experiment=%s | GPU=%s | attempt=%s | starting/resuming\n' \
    "$(date --iso-8601=seconds)" "$experiment" "$gpu_id" "$attempt"
  bash "$script_dir/$train_script" &
  training_pid=$!
  printf '%s\n' "$training_pid" > "$runtime_dir/training_shell.pid"
  printf 'running\nGPU=%s\nsupervisor_pid=%s\ntraining_shell_pid=%s\nattempt=%s\n' \
    "$gpu_id" "$$" "$training_pid" "$attempt" > "$runtime_dir/status.txt"
  wait "$training_pid"
  training_exit=$?

  if (( training_exit == 0 )); then
    printf 'completed\nGPU=%s\nattempt=%s\n' "$gpu_id" "$attempt" > "$runtime_dir/status.txt"
    printf '%s | %s | training completed successfully\n' "$(date --iso-8601=seconds)" "$experiment"
    exit 0
  fi

  if [[ -f "$runtime_dir/STOP" ]]; then
    break
  fi
  printf 'retrying\nGPU=%s\nlast_exit_code=%s\nattempt=%s\n' \
    "$gpu_id" "$training_exit" "$attempt" > "$runtime_dir/status.txt"
  printf '%s | %s | exit=%s; retrying in 60 seconds with resume enabled\n' \
    "$(date --iso-8601=seconds)" "$experiment" "$training_exit"
  sleep 60
done

printf 'stopped\nGPU=%s\nattempt=%s\n' "$gpu_id" "$attempt" > "$runtime_dir/status.txt"
printf '%s | %s | automatic restart disabled by STOP file\n' "$(date --iso-8601=seconds)" "$experiment"
