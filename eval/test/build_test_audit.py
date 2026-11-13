"""Build the hidden test audits (run once; outputs are committed under eval/test/).

The hidden test holds ``N_AUDITS`` independent audits.  Each one is a target
fine-tuned with the public recipe (``data/target_training.py``, the same recipe
as the visible dev audit) on its own private document set, plus 600 member and
600 non-member candidates with neutral, shuffled ids.  Every private set, every
non-member set and every member selection derives from the secret seed in
``eval/test/secret_seed.txt``, so no label can be regenerated from public code.

Outputs (all inside eval/test/):
  audits/<name>/target.pt, audits/<name>/candidates/   what an attack receives
  audit_labels.json     {audit name: {doc_id: 0/1}}     (test answers)
  audit_sha256.json     {"audits": [...], "files": {path: sha256}}  integrity record
  audit_stats.json      per-audit target statistics (member/non-member QA accuracy)
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import hashlib
import json
import secrets
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.target_training import build_audit  # noqa: E402

N_AUDITS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=str(ROOT / "data" / "pretrained_reader.pt"))
    parser.add_argument("--audits", type=int, default=N_AUDITS)
    parser.add_argument("--finetune-overrides", default="{}")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("DOCMIA_THREADS", "1") or 1)))
    marker = HERE / "audit_sha256.json"  # written last: completion marker
    root = HERE / "audits"
    if marker.exists() and not args.force:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if "audits" in record:
            print("hidden audits already built:", record["audits"])
            return
    seed_file = HERE / "secret_seed.txt"
    if not seed_file.exists():
        seed_file.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    seed = seed_file.read_text(encoding="utf-8").strip()
    marker.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    labels, stats = {}, {}
    names = [f"audit_{k}" for k in range(args.audits)]
    partial = HERE / "audit_build_partial.json"  # resumable: finished audits are kept
    done = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {"labels": {}, "stats": {}}
    for k, name in enumerate(names):
        if name in done["labels"] and (root / name / "target.pt").exists():
            labels[name], stats[name] = done["labels"][name], done["stats"][name]
            print(f"[build] {name}: kept")
            continue
        shutil.rmtree(root / name, ignore_errors=True)
        result = build_audit(f"hidden-test|{seed}|{name}", root / name, args.pretrained,
                             json.loads(args.finetune_overrides), log=lambda s: None)
        labels[name], stats[name] = result["labels"], {"stats": result["stats"], "history": result["history"]}
        done["labels"][name], done["stats"][name] = labels[name], stats[name]
        partial.write_text(json.dumps(done), encoding="utf-8")
        print(f"[build] {name}: gap {result['stats']['train_test_gap_points']:.1f} points", flush=True)
    (HERE / "audit_labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    (HERE / "audit_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    files = {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in names for p in sorted((root / name).rglob("*")) if p.is_file()
    }
    files["audit_labels.json"] = hashlib.sha256((HERE / "audit_labels.json").read_bytes()).hexdigest()
    marker.write_text(json.dumps({"audits": names, "files": files}, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({n: s["stats"]["train_test_gap_points"] for n, s in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
