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

from data.audit_io import load_candidates, load_target  # noqa: E402
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
    "ig": [1e-4, 1e-3, 1e-2, 1e-1],
}
TAU_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def attack_quality(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    ba = 100 * balanced_accuracy_score(labels, preds)
    f1 = 100 * f1_score(labels, preds, zero_division=0)
    return {"balanced_accuracy": ba, "f1": f1, "selection": (ba + f1) / 2,
            "roc_auc": 100 * roc_auc_score(labels, scores)}


def tune(audit_dir, labels: dict, variants, lr_grid: dict, tau_grid, steps: int, log=print) -> dict:
    cand = load_candidates(audit_dir)
    model = load_target(audit_dir)
    y = np.asarray([labels[d] for d in cand.doc_ids])
    h = hidden_states(model, cand)
    report = {}
    for variant in variants:
        rows = []
        for lr in lr_grid[variant]:
            start = time.time()
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
                rows.append({"lr": lr, "tau": tau, **attack_quality(y, preds, scores)})
            log(f"[tune] {variant} lr={lr}: {time.time() - start:.1f}s; best so far "
                f"{max(r['selection'] for r in rows):.2f}")
        best = max(rows, key=lambda r: r["selection"])
        report[variant] = {"best": best, "grid": rows}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default=str(ROOT / "data" / "dev_audit"))
    parser.add_argument("--labels", default=None)
    parser.add_argument("--variants", default="fl,fllora,ig")
    parser.add_argument("--ig-lrs", default=None)
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
    report = tune(args.audit_dir, labels, variants, grid, TAU_GRID, DEFAULT_CONFIG["max_steps"])
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
