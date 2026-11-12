"""Build the public assets of the gym (run once; outputs are committed).

* ``data/pretrained_reader.pt`` - the public pre-trained DocVQA checkpoint every
  audited target is fine-tuned from (like a public Donut/VT5 base checkpoint).
* ``data/dev_audit/`` - the visible development audit: a target fine-tuned on a
  private set drawn from a public seed, its unlabeled candidate documents, and
  their membership labels (``labels.json``) for development scoring.

The hidden test audit uses the same recipe with a secret seed (``eval/test``).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.target_training import build_audit, build_pretrained  # noqa: E402

DEV_SEED = "docmia-dev-audit-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="data/pretrained_reader.pt")
    parser.add_argument("--dev-dir", default="data/dev_audit")
    parser.add_argument("--dev-seed", default=DEV_SEED)
    parser.add_argument("--pretrain-overrides", default="{}")
    parser.add_argument("--finetune-overrides", default="{}")
    parser.add_argument("--no-pretraining", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(__import__("os").environ.get("DOCMIA_THREADS", "1") or 1)))

    pretrained = Path(args.pretrained)
    if not args.no_pretraining and (args.force or not pretrained.exists()):
        info = build_pretrained(pretrained, json.loads(args.pretrain_overrides))
        pretrained.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    dev = Path(args.dev_dir)
    if args.force or not (dev / "target.pt").exists():
        result = build_audit(args.dev_seed, dev, None if args.no_pretraining else pretrained,
                             json.loads(args.finetune_overrides))
        (dev / "labels.json").write_text(json.dumps(result["labels"], indent=1), encoding="utf-8")
        (dev / "target_stats.json").write_text(
            json.dumps({"stats": result["stats"], "history": result["history"]}, indent=1),
            encoding="utf-8",
        )
        print(json.dumps(result["stats"], indent=1))


if __name__ == "__main__":
    main()
