#!/bin/bash
set -euo pipefail

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

# Regenerate the full scaled artifact rather than depending on ignored
# checkpoint files being present from a previous run.
rm -f checkpoints/tiny_docvqa_target.pt checkpoints/docmia_outputs.json checkpoints/target_metrics.json

export DOCMIA_TARGET_EPOCHS=60
export DOCMIA_MAX_QUESTIONS=4
export DOCMIA_FL_STEPS=60
export DOCMIA_FL_LR=0.001
export DOCMIA_FL_TAU=0.00001
export DOCMIA_LORA_RANK=4
export DOCMIA_LORA_ALPHA=4.0
export DOCMIA_IG_STEPS=60
export DOCMIA_IG_LR=0.001
export DOCMIA_IG_TAU=0.00001

bash scripts/download.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh checkpoints/docmia_outputs.json
bash scripts/evaluate_test.sh checkpoints/docmia_outputs.json
bash scripts/evaluate.sh checkpoints/docmia_outputs.json data/synthetic_docvqa
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data/synthetic_docvqa/manifest.json").read_text(encoding="utf-8"))
documents = [
    doc for doc in manifest["documents"]
    if doc["split"] in {"member", "nonmember"}
]
payload = {
    "metadata": {
        "max_questions_per_document": 4,
        "uses_auxiliary_data": 0.0,
    },
    "membership_scores": {
        "generic_score_smoke": [float(index % 17) for index, _ in enumerate(documents)]
    },
}
Path("tmp").mkdir(exist_ok=True)
Path("tmp/generic_membership_artifact.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
PY
python eval/evaluate_docmia.py \
  --input tmp/generic_membership_artifact.json \
  --data-dir data/synthetic_docvqa \
  --scores tmp/generic_scores.json
python validate.py --compare
