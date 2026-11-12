"""Paired bootstrap of (candidate - reference) on the hidden test documents.

    python eval/test/paired_bootstrap.py --reference eval/test/runs/reference.json \
        --candidate name=eval/test/runs/name.json [...] --out hardening/paired_bootstrap.json

Both attacks are scored on the same resampled documents, so the interval
reflects test-sampling noise of the *difference*, which is what decides
whether a variant genuinely beats the reference.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
SLOTS = ("docmia_fl", "docmia_fllora", "docmia_ig")


def _load(path: Path, doc_ids: list[str]):
    raw = json.loads(path.read_text(encoding="utf-8"))
    where = {d: i for i, d in enumerate(raw["doc_ids"])}
    perm = np.array([where[d] for d in doc_ids])
    return {s: (np.asarray(a["predictions"], float)[perm].astype(int), np.asarray(a["scores"], float)[perm])
            for s, a in raw["attacks"].items()}


def _stats(y, p, s):
    tp = (p[y == 1] == 1).mean()
    tn = (p[y == 0] == 0).mean()
    prec = (y[p == 1] == 1).mean() if p.sum() else 0.0
    f1 = 2 * prec * tp / (prec + tp) if prec + tp else 0.0
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return np.array([50 * (tp + tn), 100 * f1, 100 * roc_auc_score(y, s), 100 * tpr[fpr <= 0.03].max()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--labels", default=str(HERE / "audit_labels.json"))
    parser.add_argument("--manifest", default=str(HERE / "audit" / "candidates" / "manifest.json"))
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    doc_ids = [d["doc_id"] for d in manifest["documents"]]
    n_q = np.array([len(d["qas"]) for d in manifest["documents"]])
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    y_all = np.array([int(labels[d]) for d in doc_ids])
    ref = _load(Path(args.reference), doc_ids)
    names = ("balanced_accuracy", "f1", "roc_auc", "tpr_at_3_fpr")
    report = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        cand = _load(Path(path), doc_ids)
        report[name] = {}
        for exp, mask in (("scaled_whitebox_docmia", np.ones(len(doc_ids), bool)), ("few_question_documents", n_q <= 2)):
            y = y_all[mask]
            rng = np.random.default_rng(0)
            idx_sets = [rng.integers(0, len(y), len(y)) for _ in range(args.reps)]
            report[name][exp] = {}
            for slot in SLOTS:
                if slot not in cand or slot not in ref:
                    continue
                (pc, sc), (pr, sr) = [(a[0][mask], a[1][mask]) for a in (cand[slot], ref[slot])]
                diffs = []
                for idx in idx_sets:
                    if y[idx].min() == y[idx].max():
                        continue
                    diffs.append(_stats(y[idx], pc[idx], sc[idx]) - _stats(y[idx], pr[idx], sr[idx]))
                diffs = np.array(diffs)
                point = _stats(y, pc, sc) - _stats(y, pr, sr)
                report[name][exp][slot] = {
                    m: {"diff": float(point[k]), "lo95": float(np.percentile(diffs[:, k], 2.5)),
                        "hi95": float(np.percentile(diffs[:, k], 97.5)), "sd": float(diffs[:, k].std())}
                    for k, m in enumerate(names)
                }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, block in report.items():
        for exp, slots in block.items():
            for slot, ms in slots.items():
                cells = "  ".join(f"{m} {v['diff']:+.2f} [{v['lo95']:+.2f},{v['hi95']:+.2f}]" for m, v in ms.items())
                print(f"{name:24s} {exp:24s} {slot:14s} {cells}")


if __name__ == "__main__":
    main()
