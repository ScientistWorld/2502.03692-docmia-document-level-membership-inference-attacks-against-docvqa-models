"""Run the DocMIA white-box attacks on one audit directory.

Usage: python method/attack.py --audit-dir <dir> --output <json> [--config method/config.json]

The audit directory holds ``target.pt`` and ``candidates/`` (no labels).  The
output lists, for every attack, one membership score (higher = more likely a
training document) and one binary membership decision per candidate document.
``membership_attack`` (the scored submission) is the DocMIA variant selected on
the dev audit; the three variants are also reported individually.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.audit_io import load_candidates, load_target  # noqa: E402
from method.docmia import (  # noqa: E402
    cluster_membership,
    document_features,
    fl_trajectories,
    fllora_trajectories,
    hidden_states,
    ig_trajectories,
)

DEFAULT_CONFIG = {
    "max_steps": 200,
    "kmeans_seed": 0,
    "fl": {"lr": 1e-3, "tau": 1e-4},
    "fllora": {"lr": 1e-3, "tau": 1e-4, "rank": 4, "alpha": 4.0, "seed": 0},
    "ig": {"lr": 1e-3, "tau": 1e-4},
    "primary_variant": "fl",
}


def load_config(path: str | Path | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and Path(path).exists():
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def cpu_budget() -> int:
    """CPUs actually granted to this job: DOCMIA_THREADS, else the cgroup quota, else 1.

    Job nodes expose more cores than a job is allotted; letting torch/OpenMP use
    every visible core oversubscribes the quota and slows everything ~100x.
    """
    if os.environ.get("DOCMIA_THREADS"):
        return max(1, int(os.environ["DOCMIA_THREADS"]))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return 1


def set_threads() -> None:
    torch.set_num_threads(cpu_budget())


def run(audit_dir: str | Path, config: dict, variants=("fl", "fllora", "ig"), log=print) -> dict:
    cand = load_candidates(audit_dir)
    model = load_target(audit_dir)
    steps = int(config["max_steps"])
    h = hidden_states(model, cand)
    attacks, features, timing = {}, {}, {}
    for variant in variants:
        cfg = config[variant]
        start = time.time()
        if variant == "fl":
            traj = fl_trajectories(model, cand, cfg["lr"], steps, cfg["tau"], h=h)
        elif variant == "fllora":
            traj = fllora_trajectories(model, cand, cfg["lr"], steps, cfg["tau"], rank=cfg["rank"],
                                       alpha=cfg["alpha"], seed=cfg["seed"], h=h)
        elif variant == "ig":
            traj = ig_trajectories(model, cand, cfg["lr"], steps, cfg["tau"])
        else:
            raise ValueError(variant)
        feats = document_features(traj, cfg["tau"], cand)
        preds, scores = cluster_membership(feats, seed=int(config["kmeans_seed"]))
        name = f"docmia_{variant}"
        attacks[name] = {"scores": scores.tolist(), "predictions": preds.tolist()}
        features[name] = feats.tolist()
        timing[name] = round(time.time() - start, 2)
        log(f"[attack] {name}: {timing[name]}s, predicted members {int(preds.sum())}/{len(preds)}")
    primary = f"docmia_{config.get('primary_variant', 'fl')}"
    if primary in attacks:
        attacks = {"membership_attack": attacks[primary], **attacks}
    return {"doc_ids": cand.doc_ids, "attacks": attacks, "features": features, "timing_sec": timing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "method" / "config.json"))
    args = parser.parse_args()
    set_threads()
    config = load_config(args.config)
    output = run(args.audit_dir, config)
    output["config"] = config
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
