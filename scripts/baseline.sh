#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python data/synthetic_docvqa.py
if [ ! -e checkpoints/tiny_docvqa_target.pt ]; then
  python method/train_target.py \
    --data-dir data/synthetic_docvqa \
    --checkpoint checkpoints/tiny_docvqa_target.pt \
    --epochs "${DOCMIA_TARGET_EPOCHS:-45}"
fi
python method/docmia_attack.py \
  --data-dir data/synthetic_docvqa \
  --checkpoint checkpoints/tiny_docvqa_target.pt \
  --output checkpoints/docmia_outputs.json \
  --max-questions "${DOCMIA_MAX_QUESTIONS:-4}" \
  --steps "${DOCMIA_FL_STEPS:-60}" \
  --lr "${DOCMIA_FL_LR:-0.001}" \
  --tau "${DOCMIA_FL_TAU:-0.00001}" \
  --ig-steps "${DOCMIA_IG_STEPS:-60}" \
  --ig-lr "${DOCMIA_IG_LR:-0.01}" \
  --ig-tau "${DOCMIA_IG_TAU:-0.00001}"
bash scripts/evaluate.sh checkpoints/docmia_outputs.json
