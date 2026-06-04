#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/synthetic_docvqa.py
python method/train_target.py \
  --data-dir data/synthetic_docvqa \
  --checkpoint checkpoints/tiny_docvqa_target.pt \
  --epochs "${DOCMIA_TARGET_EPOCHS:-45}"
python method/docmia_attack.py \
  --data-dir data/synthetic_docvqa \
  --checkpoint checkpoints/tiny_docvqa_target.pt \
  --output checkpoints/docmia_outputs.json \
  --max-questions "${DOCMIA_MAX_QUESTIONS:-4}" \
  --steps "${DOCMIA_FL_STEPS:-60}" \
  --lr "${DOCMIA_FL_LR:-0.001}" \
  --tau "${DOCMIA_FL_TAU:-0.00001}" \
  --lora-rank "${DOCMIA_LORA_RANK:-4}" \
  --lora-alpha "${DOCMIA_LORA_ALPHA:-4.0}" \
  --ig-steps "${DOCMIA_IG_STEPS:-60}" \
  --ig-lr "${DOCMIA_IG_LR:-0.001}" \
  --ig-tau "${DOCMIA_IG_TAU:-0.00001}"
bash scripts/evaluate.sh checkpoints/docmia_outputs.json
