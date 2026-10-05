#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

TARGETS="${TARGETS:-32 64 128 256 512 1024 2048}"
NO_CHANNEL="${NO_CHANNEL:-1}"

for target in $TARGETS; do
  case " 32 64 128 256 512 1024 2048 " in
    *" $target "*) ;;
    *)
      echo "Unsupported target K=$target; expected one of: 32 64 128 256 512 1024 2048" >&2
      exit 2
      ;;
  esac

  echo "Testing target K=$target (console output only; no result files will be saved)"
  env \
    PYTHONDONTWRITEBYTECODE=1 \
    SIMVQ_RAQ_TARGET_LIST="$target" \
    NO_CHANNEL="$NO_CHANNEL" \
    JSON_OUTPUT= \
    bash "$SCRIPT_DIR/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh" "$@"
done
