#!/bin/bash
# Reproduce the reference end to end (CPU job, ~1.5 h on one core from a fresh clone):
#   committed assets (rebuilt by scripts/download.sh only if missing)
#   -> DocMIA hyper-parameter and variant selection on the dev audit (paper Appendix C grid)
#   -> scripts/evaluate_train.sh (dev audit) and scripts/evaluate_test.sh (hidden audit)
#   -> validate.py
# The hardening evidence (variants, shortcuts, bootstrap, rank) is produced by
# hardening/run_hardening.sh.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
export OMP_NUM_THREADS=$DOCMIA_THREADS MKL_NUM_THREADS=$DOCMIA_THREADS OPENBLAS_NUM_THREADS=$DOCMIA_THREADS
DOCMIA_RETUNE=1 bash scripts/reproduce.sh
