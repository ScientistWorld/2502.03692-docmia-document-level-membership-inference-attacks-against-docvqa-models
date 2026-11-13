#!/bin/bash
# Job entry point for the hardening iteration: hardening/run_hardening.sh builds the four
# hidden audits, re-runs the reference end to end (dev tuning -> train slice -> hidden test),
# then replays the variants and shortcuts through the hidden test (resumable).
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export DOCMIA_THREADS=${DOCMIA_THREADS:-1}
export OMP_NUM_THREADS=$DOCMIA_THREADS MKL_NUM_THREADS=$DOCMIA_THREADS OPENBLAS_NUM_THREADS=$DOCMIA_THREADS
bash hardening/run_hardening.sh
