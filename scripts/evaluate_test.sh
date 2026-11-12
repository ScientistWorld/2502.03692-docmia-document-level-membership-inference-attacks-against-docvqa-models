#!/bin/bash
# Hidden test: run the attack program (scripts/attack.sh) on the hidden audit
# held in eval/test and score it.  No arguments.
#   bash scripts/evaluate_test.sh             -> scoring/scores_test.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
python eval/test/evaluate.py ${DOCMIA_KEEP_OUTPUT:+--keep-output "$DOCMIA_KEEP_OUTPUT"}
