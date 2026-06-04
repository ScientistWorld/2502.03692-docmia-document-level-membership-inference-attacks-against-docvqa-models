#!/bin/bash
set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

test -f checkpoints/docmia_outputs.json
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
bash scripts/evaluate.sh checkpoints/docmia_outputs.json data/synthetic_docvqa
python validate.py --compare
