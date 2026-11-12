"""Hyper-parameter selection for DocMIA on the visible dev audit (paper Appendix C).

The paper fixes Adam, S=200 steps and the last projection layer, then grid-searches
the learning rate alpha and the stopping threshold tau per target model and
keeps the setting with the best attack performance (average of accuracy and F1).
Here the search runs on the dev audit only; the selected setting is written to
``method/config.json`` and then applied unchanged to any other audit.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.audit_io import Candidates, load_candidates, load_target  # noqa: E402
from method.attack import DEFAULT_CONFIG, set_threads  # noqa: E402
from method.docmia import (  # noqa: E402
    cluster_membership,
    document_features,
    fl_trajectories,
    fllora_trajectories,
    hidden_states,
    ig_trajectories,
)

LR_GRID = {
    "fl": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0],
    "fllora": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0],
    "ig": [1e-3, 1e-2, 1e-1],
}
TAU_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0]


def subset(cand: Candidates, n: int) -> Candidates:
    """First n candidate documents (the dev order is already a random shuffle)."""
    keep = cand.pair_doc < n
    return Candidates(cand.doc_ids[:n], cand.pages[:n], cand.pair_doc[keep], cand.pair_question[keep],
                      cand.pair_answer[keep], cand.fields, cand.answer_bank, cand.manifest)


def attack_quality(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    ba = 100 * balanced_accuracy_score(labels, preds)
    f1 = 100 * f1_score(labels, preds, zero_division=0)
    return {"balanced_accuracy": ba, "f1": f1, "selection": (ba + f1) / 2,
            "roc_auc": 100 * roc_auc_score(labels, scores)}


def tune(audit_dir, labels: dict, variants, lr_grid: dict, tau_grid, steps: int, log=print,
         cache_dir: str | Path | None = None, ig_docs: int | None = None) -> dict:
    full = load_candidates(audit_dir)
    model = load_target(audit_dir)
    report = {}
    for variant in variants:
        cand = subset(full, ig_docs) if (variant == "ig" and ig_docs) else full
        y = np.asarray([labels[d] for d in cand.doc_ids])
        h = hidden_states(model, cand) if variant != "ig" else None
        rows = []
        for lr in lr_grid[variant]:
            cache = Path(cache_dir) / f"{variant}_lr{lr:g}_n{cand.num_docs}.json" if cache_dir else None
            if cache is not None and cache.exists():
                rows.extend(json.loads(cache.read_text(encoding="utf-8")))
                log(f"[tune] {variant} lr={lr}: cached")
                continue
            start = time.time()
            lr_rows = []
            if variant == "fl":
                traj = fl_trajectories(model, cand, lr, steps, None, h=h)
            elif variant == "fllora":
                cfg = DEFAULT_CONFIG["fllora"]
                traj = fllora_trajectories(model, cand, lr, steps, None, rank=cfg["rank"],
                                           alpha=cfg["alpha"], seed=cfg["seed"], h=h)
            else:
                traj = ig_trajectories(model, cand, lr, steps, None)
            for tau in tau_grid:
                preds, scores = cluster_membership(document_features(traj, tau, cand))
                lr_rows.append({"lr": lr, "tau": tau, **attack_quality(y, preds, scores)})
            rows.extend(lr_rows)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(lr_rows), encoding="utf-8")
            log(f"[tune] {variant} lr={lr}: {time.time() - start:.1f}s; best so far "
                f"{max(r['selection'] for r in rows):.2f}")
        best = max(rows, key=lambda r: r["selection"])
        report[variant] = {"best": best, "documents": cand.num_docs, "grid": rows}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default=str(ROOT / "data" / "dev_audit"))
    parser.add_argument("--labels", default=None)
    parser.add_argument("--variants", default="fl,fllora,ig")
    parser.add_argument("--ig-lrs", default=None)
    parser.add_argument("--ig-docs", type=int, default=300,
                        help="tune IG on the first N dev documents (IG is ~20x costlier per step)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-config", default=str(ROOT / "method" / "config.json"))
    parser.add_argument("--report", default=str(ROOT / "method" / "tuning_dev.json"))
    args = parser.parse_args()
    set_threads()
    labels_path = args.labels or str(Path(args.audit_dir) / "labels.json")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    grid = {k: list(v) for k, v in LR_GRID.items()}
    if args.ig_lrs:
        grid["ig"] = [float(x) for x in args.ig_lrs.split(",")]
    variants = args.variants.split(",")
    report = tune(args.audit_dir, labels, variants, grid, TAU_GRID, DEFAULT_CONFIG["max_steps"],
                  cache_dir=args.cache_dir, ig_docs=args.ig_docs)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    for variant in variants:
        config[variant]["lr"] = report[variant]["best"]["lr"]
        config[variant]["tau"] = report[variant]["best"]["tau"]
    config["selected_on"] = "dev audit (data/dev_audit), mean of balanced accuracy and F1"
    Path(args.output_config).write_text(json.dumps(config, indent=1), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v["best"] for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
