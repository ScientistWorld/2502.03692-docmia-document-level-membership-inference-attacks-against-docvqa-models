"""Paired bootstrap of (candidate - reference) on the hidden test audits.

    python eval/test/paired_bootstrap.py --reference eval/test/runs/reference \
        --candidate name=eval/test/runs/name [...] --out hardening/paired_bootstrap_test.json

Each directory holds the raw attack outputs kept by ``evaluate.py --keep-output``
(one ``<audit>.json`` per hidden audit).  In every bootstrap replicate the members
and the non-members of each audit are resampled, both attacks are scored on the
same resampled documents, and each metric is averaged over the audits exactly as
the ranked row is.  The interval therefore reflects the test-sampling noise of the
*difference*, which decides whether a variant genuinely beats the reference.
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
SLOT = "membership_attack"
NAMES = ("balanced_accuracy", "f1", "roc_auc", "tpr_at_1_fpr", "tpr_at_3_fpr")


def _load_output(path: Path, doc_ids: list[str]):
    raw = json.loads(path.read_text(encoding="utf-8"))
    where = {d: i for i, d in enumerate(raw["doc_ids"])}
    perm = np.array([where[d] for d in doc_ids])
    item = raw["attacks"][SLOT]
    return np.asarray(item["predictions"], float)[perm].astype(int), np.asarray(item["scores"], float)[perm]


def _stats(y, p, s):
    tp = (p[y == 1] == 1).mean()
    tn = (p[y == 0] == 0).mean()
    prec = (y[p == 1] == 1).mean() if p.sum() else 0.0
    f1 = 2 * prec * tp / (prec + tp) if prec + tp else 0.0
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return np.array([50 * (tp + tn), 100 * f1, 100 * roc_auc_score(y, s),
                     100 * tpr[fpr <= 0.01].max(), 100 * tpr[fpr <= 0.03].max()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="directory of the reference's kept attack outputs")
    parser.add_argument("--candidate", action="append", default=[], help="name=directory of kept attack outputs")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = json.loads((HERE / "audit_sha256.json").read_text(encoding="utf-8"))
    labels = json.loads((HERE / "audit_labels.json").read_text(encoding="utf-8"))
    audits = []
    for name in record["audits"]:
        manifest = json.loads((HERE / "audits" / name / "candidates" / "manifest.json").read_text(encoding="utf-8"))
        doc_ids = [d["doc_id"] for d in manifest["documents"]]
        y = np.array([int(labels[name][d]) for d in doc_ids])
        audits.append((name, doc_ids, y))
    rng = np.random.default_rng(0)
    resamples = [[np.concatenate([rng.choice(np.where(y == c)[0], int((y == c).sum())) for c in (0, 1)])
                  for _, _, y in audits] for _ in range(args.reps)]
    ref = [_load_output(Path(args.reference) / f"{n}.json", ids) for n, ids, _ in audits]
    report = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        try:
            cand = [_load_output(Path(path) / f"{n}.json", ids) for n, ids, _ in audits]
        except (FileNotFoundError, KeyError) as exc:
            print(f"{name}: skipped ({exc})")
            continue
        point = np.mean([_stats(y, *c) - _stats(y, *r) for (_, _, y), c, r in zip(audits, cand, ref)], 0)
        diffs = np.array([
            np.mean([_stats(y[i], c[0][i], c[1][i]) - _stats(y[i], r[0][i], r[1][i])
                     for (_, _, y), c, r, i in zip(audits, cand, ref, idx)], 0)
            for idx in resamples
        ])
        report[name] = {m: {"diff": float(point[k]), "lo95": float(np.percentile(diffs[:, k], 2.5)),
                            "hi95": float(np.percentile(diffs[:, k], 97.5)), "sd": float(diffs[:, k].std())}
                        for k, m in enumerate(NAMES)}
        cells = "  ".join(f"{m} {v['diff']:+.2f} [{v['lo95']:+.2f},{v['hi95']:+.2f}]" for m, v in report[name].items())
        print(f"{name:26s} {cells}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"audits": [n for n, _, _ in audits], "reps": args.reps,
                                          "comparisons": report}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
