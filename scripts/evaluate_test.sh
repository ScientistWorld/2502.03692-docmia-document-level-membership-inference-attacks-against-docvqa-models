#!/bin/bash
# Hidden test: run the attack program (scripts/attack.sh) once on each hidden audit held in
# eval/test and score it.  No arguments.
#   bash scripts/evaluate_test.sh             -> scoring/scores_test.json
# While the attack runs, eval/test and data/dev_audit are unreadable; access is restored
# afterwards (and here, in case an earlier run was killed while they were sealed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
chmod u+rwx eval/test data/dev_audit 2>/dev/null || true
python eval/test/evaluate.py ${DOCMIA_KEEP_OUTPUT:+--keep-output "$DOCMIA_KEEP_OUTPUT"}
