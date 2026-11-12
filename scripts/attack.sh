#!/bin/bash
# Membership-attack interface used by both evaluators.
#
#   bash scripts/attack.sh <audit_dir> <output_json>
#
# <audit_dir> holds target.pt (the audited model) and candidates/ (unlabeled
# documents with their question-answer pairs).  The attack must write, for
# every attack it runs, one membership score and one 0/1 membership decision
# per candidate document (format: see briefing/evaluation.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python method/attack.py --audit-dir "$1" --output "$2" --config "${DOCMIA_CONFIG:-method/config.json}"
