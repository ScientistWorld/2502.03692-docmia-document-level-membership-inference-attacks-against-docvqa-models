#!/bin/bash
set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

export DOCMIA_TARGET_EPOCHS="${DOCMIA_TARGET_EPOCHS:-60}"
export DOCMIA_MAX_QUESTIONS="${DOCMIA_MAX_QUESTIONS:-4}"
export DOCMIA_FL_STEPS="${DOCMIA_FL_STEPS:-60}"
export DOCMIA_IG_STEPS="${DOCMIA_IG_STEPS:-60}"
export DOCMIA_LORA_RANK="${DOCMIA_LORA_RANK:-4}"
export DOCMIA_LORA_ALPHA="${DOCMIA_LORA_ALPHA:-4.0}"

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
