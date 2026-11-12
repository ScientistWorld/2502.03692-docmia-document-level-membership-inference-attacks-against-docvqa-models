#!/bin/bash
# Run the paper's method (DocMIA FL / FLLoRA / IG): select its hyper-parameters
# on the dev audit (paper Appendix C grid; cached under tmp/tune_cache) unless
# method/config.json already exists, then score it through scripts/evaluate.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
if [ ! -f method/config.json ] || [ "${DOCMIA_RETUNE:-0}" = "1" ]; then
  python method/tune_dev.py --cache-dir tmp/tune_cache
fi
bash scripts/evaluate.sh
