#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

export DOCMIA_TARGET_EPOCHS="${DOCMIA_TARGET_EPOCHS:-60}"
export DOCMIA_MAX_QUESTIONS="${DOCMIA_MAX_QUESTIONS:-4}"
export DOCMIA_FL_STEPS="${DOCMIA_FL_STEPS:-60}"
export DOCMIA_IG_STEPS="${DOCMIA_IG_STEPS:-60}"

bash scripts/reproduce.sh
