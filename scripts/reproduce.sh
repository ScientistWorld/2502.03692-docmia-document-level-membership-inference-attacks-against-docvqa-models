#!/bin/bash
# Reproduce the reference end to end (CPU, ~1-2 h on one core):
#   assets (committed; rebuilt only if missing) -> DocMIA tuning on the dev audit
#   -> train-slice and hidden-test scoring -> validation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
bash scripts/download.sh
if [ ! -f method/config.json ] || [ "${DOCMIA_RETUNE:-0}" = "1" ]; then
  python method/tune_dev.py --cache-dir tmp/tune_cache
fi
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
cp scoring/scores_train.json scoring/scores.json
python validate.py --compare
