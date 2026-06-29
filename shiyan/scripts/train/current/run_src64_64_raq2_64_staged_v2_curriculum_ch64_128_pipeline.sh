#!/bin/bash
set -euo pipefail
cd /workspace/yi/work/shiyan

export GPU_ID="${GPU_ID:-0}"
export SIMVQ_TOTAL_BATCH_SIZE="${SIMVQ_TOTAL_BATCH_SIZE:-24}"
export SIMVQ_MICRO_BATCH_SIZE="${SIMVQ_MICRO_BATCH_SIZE:-24}"

echo "Starting staged v2 curriculum pipeline on physical GPU $GPU_ID"
echo "Stage schedule: 200 + 100 + 100 + 100 epochs"

bash scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage1_src_teacher_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage2_raq_warmup_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage3_raq_finetune_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage4_raq_channel_ch64_128.sh
