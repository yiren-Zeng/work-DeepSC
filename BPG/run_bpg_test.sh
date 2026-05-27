#!/bin/bash
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"
conda activate work
cd /workspace/yi/work/BPG
rm -f /workspace/yi/work/BPG/bpg_test_output.log
export PYTHONUNBUFFERED=1
python -u test_bpg.py > /workspace/yi/work/BPG/bpg_test_output.log 2>&1
