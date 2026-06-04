#!/bin/bash
# Evaluate a DocMIA method artifact on the held-out test slice.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

INPUT="${1:-checkpoints/docmia_outputs.json}"
if [ -d "$INPUT" ]; then
  INPUT="$INPUT/docmia_outputs.json"
fi
DATA_DIR="${2:-data/synthetic_docvqa}"
python eval/test/evaluate_docmia_slice.py \
  --input "$INPUT" \
  --data-dir "$DATA_DIR" \
  --scores scoring/scores_test.json \
  --train-scores scoring/scores_train.json
