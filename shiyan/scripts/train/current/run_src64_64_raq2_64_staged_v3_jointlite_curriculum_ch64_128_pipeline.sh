#!/bin/bash
set -euo pipefail
cd /workspace/yi/work/shiyan

export GPU_ID="${GPU_ID:-0}"
echo "Starting staged v3 joint-lite curriculum pipeline on physical GPU $GPU_ID"
echo "Epochs: Stage1=200, Stage2=100, Stage3=150, Stage4=150"

bash scripts/train/current/run_src64_64_raq2_64_staged_v3_jointlite_curriculum_stage1_src_teacher_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v3_jointlite_curriculum_stage2_raq_warmup_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v3_jointlite_curriculum_stage3_raq_jointlite_ch64_128.sh
bash scripts/train/current/run_src64_64_raq2_64_staged_v3_jointlite_curriculum_stage4_raq_jointlite_channel_ch64_128.sh

echo "Staged v3 joint-lite curriculum pipeline complete."
