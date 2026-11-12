#!/bin/bash
# Standard evaluation entry point: scores the attack program (scripts/attack.sh)
# on the visible dev audit and writes scoring/scores.json.  Uses only eval/ and
# data/; any method that implements scripts/attack.sh can be scored.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
mkdir -p scoring
python eval/train/evaluate.py --scores scoring/scores.json
