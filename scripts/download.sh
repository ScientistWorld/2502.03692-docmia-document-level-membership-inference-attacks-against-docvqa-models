#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This reproduction has no external downloads. The small procedural dataset is
# tracked under data/ so a fresh clone works before Python packages are
# installed; if it is missing, regenerate it with the same first-party script.

set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

DATA_DIR="data/synthetic_docvqa"
MANIFEST="$DATA_DIR/manifest.json"

if [ -f "$MANIFEST" ] && \
   [ -f "$DATA_DIR/images/member_0000.png" ] && \
   [ -f "$DATA_DIR/images/nonmember_0000.png" ]; then
  echo "Synthetic DocVQA dataset is present at $DATA_DIR"
  exit 0
fi

python data/synthetic_docvqa.py
