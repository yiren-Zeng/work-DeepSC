#!/bin/bash
set -euo pipefail

ROOT_DIR="/workspace/yi/work/Simvq-dc-64-Multi-pro-2ceng"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${1:?missing checkpoint path}"
shift || true

export SIMVQ_TEST_DATASET_PATH="${SIMVQ_TEST_DATASET_PATH:-/workspace/yi/work/Kodak-256-transform-resize}"
export SIMVQ_TEST_NO_RESIZE="${SIMVQ_TEST_NO_RESIZE:-1}"

PYTHON_CMD=(python -u)
if [ "${DEBUG_PDB:-0}" = "1" ]; then
  PYTHON_CMD=(python -m pdb)
elif [ "${DEBUGPY:-0}" = "1" ]; then
  DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
  PYTHON_CMD=(python -u -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client)
  echo "Waiting for debugger attach on port ${DEBUGPY_PORT}..."
fi

if [ "$#" -eq 0 ]; then
  if [ "${SIMVQ_EVAL_NO_CHANNEL:-0}" = "1" ]; then
    ARGS=(--checkpoint "$CHECKPOINT_PATH" --no-channel)
  else
    read -r -a SNR_ARGS <<< "${SIMVQ_EVAL_SNRS:-0}"
    ARGS=(--checkpoint "$CHECKPOINT_PATH" --snrs "${SNR_ARGS[@]}" --modulation "${SIMVQ_EVAL_MODULATION:-bpsk}")
  fi
else
  HAS_CHECKPOINT=0
  HAS_SNRS=0
  HAS_MODULATION=0
  HAS_NO_CHANNEL=0
  for arg in "$@"; do
    if [ "$arg" = "--checkpoint" ]; then
      HAS_CHECKPOINT=1
    elif [ "$arg" = "--snrs" ]; then
      HAS_SNRS=1
    elif [ "$arg" = "--modulation" ]; then
      HAS_MODULATION=1
    elif [ "$arg" = "--no-channel" ]; then
      HAS_NO_CHANNEL=1
    fi
  done
  if [ "$HAS_CHECKPOINT" = "1" ]; then
    ARGS=("$@")
  else
    ARGS=(--checkpoint "$CHECKPOINT_PATH" "$@")
  fi
  if [ "$HAS_NO_CHANNEL" = "0" ]; then
    if [ "$HAS_SNRS" = "0" ]; then
      read -r -a SNR_ARGS <<< "${SIMVQ_EVAL_SNRS:-0}"
      ARGS+=(--snrs "${SNR_ARGS[@]}")
    fi
    if [ "$HAS_MODULATION" = "0" ]; then
      ARGS+=(--modulation "${SIMVQ_EVAL_MODULATION:-bpsk}")
    fi
  fi
fi

"${PYTHON_CMD[@]}" test_real.py "${ARGS[@]}"
