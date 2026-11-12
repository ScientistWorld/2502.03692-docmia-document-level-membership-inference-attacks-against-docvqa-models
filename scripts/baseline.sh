#!/bin/bash
# The paper's baselines (LOSS-TA, SCORE-UA, SCORE-UA_all, SCORELOSS-UA_all,
# GRADIENT-UA) are computed by the evaluators themselves from the audited model,
# so they are identical for every submission.  This prints them for the dev audit
# from the latest scoring/scores.json (run scripts/evaluate.sh first).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scoring/scores.json ] || bash scripts/evaluate.sh
python - <<'PY'
import json
scores = json.load(open("scoring/scores.json"))
for exp, block in scores["experiments"].items():
    print(exp)
    for name, row in block.get("baselines", {}).items():
        print(f"  {name:18s}", {k: round(v, 2) for k, v in row.items()})
PY
