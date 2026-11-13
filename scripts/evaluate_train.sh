#!/bin/bash
# Score the attack program (scripts/attack.sh) on the visible dev audit (train slice).
#   bash scripts/evaluate_train.sh            -> scoring/scores_train.json
# The evaluator runs the attack on a copy of data/dev_audit, computes the
# baselines itself, and recomputes every metric from per-document outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
chmod u+rwx data/dev_audit 2>/dev/null || true  # in case a killed hidden-test run left it sealed
python eval/train/evaluate.py ${DOCMIA_KEEP_OUTPUT:+--keep-output "$DOCMIA_KEEP_OUTPUT"}
