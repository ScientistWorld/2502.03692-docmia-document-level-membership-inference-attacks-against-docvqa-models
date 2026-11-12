"""Build the hidden test audit (run once; outputs are committed under eval/test/).

Uses the public target recipe (``data/target_training.py``) with the secret seed
in ``eval/test/secret_seed.txt``: the private fine-tuning set, the validation
set, the non-member documents and the member selection all derive from that
seed, so the membership labels cannot be regenerated from public code.

Outputs (all inside eval/test/):
  audit/target.pt, audit/candidates/   - what an attack receives at test time
  audit_labels.json                    - membership labels (test answers)
  audit_sha256.json                    - integrity hashes checked by the evaluator
  audit_stats.json                     - target statistics (member/non-member QA accuracy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.target_training import build_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=str(ROOT / "data" / "pretrained_reader.pt"))
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--finetune-overrides", default="{}")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    audit = out / "audit"
    if (audit / "target.pt").exists() and not args.force:
        print("hidden audit already built")
        return
    shutil.rmtree(audit, ignore_errors=True)
    seed = (HERE / "secret_seed.txt").read_text(encoding="utf-8").strip()
    result = build_audit(f"hidden-test|{seed}", audit, args.pretrained, json.loads(args.finetune_overrides))
    (out / "audit_labels.json").write_text(json.dumps(result["labels"], indent=1), encoding="utf-8")
    (out / "audit_stats.json").write_text(
        json.dumps({"stats": result["stats"], "history": result["history"]}, indent=1), encoding="utf-8")
    hashes = {
        str(p.relative_to(audit)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(audit.rglob("*")) if p.is_file()
    }
    (out / "audit_sha256.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    print(json.dumps(result["stats"], indent=1))


if __name__ == "__main__":
    main()
