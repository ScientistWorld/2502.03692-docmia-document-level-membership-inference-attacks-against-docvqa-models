"""Evaluate DocMIA outputs without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve


MEMBER_IS_LOW_SIGNAL = {
    "score_ua": False,
    "gradient_ua": True,
    "docmia_fl": True,
    "docmia_fllora": True,
    "docmia_ig": True,
}


def _labels_from_manifest(data_dir: Path, doc_ids: list[str]) -> np.ndarray:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    split_by_id = {doc["doc_id"]: doc["split"] for doc in manifest["documents"]}
    missing = [doc_id for doc_id in doc_ids if doc_id not in split_by_id]
    if missing:
        raise KeyError(f"Unknown document ids in method output: {missing[:5]}")
    return np.asarray([1 if split_by_id[doc_id] == "member" else 0 for doc_id in doc_ids], dtype=np.int64)


def _normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def _cluster_predict(features: np.ndarray, member_is_low: bool) -> tuple[np.ndarray, np.ndarray]:
    norm_features = _normalize(features)
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(norm_features)
    signal = features[:, 0]
    cluster_signal = [signal[clusters == cluster].mean() for cluster in (0, 1)]
    member_cluster = int(np.argmin(cluster_signal) if member_is_low else np.argmax(cluster_signal))
    preds = (clusters == member_cluster).astype(int)
    centroid = norm_features[clusters == member_cluster].mean(axis=0)
    distances = np.linalg.norm(norm_features - centroid[None, :], axis=1)
    scores = -distances
    return preds, scores


def _metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds) * 100.0),
        "accuracy": float(accuracy_score(labels, preds) * 100.0),
        "f1": float(f1_score(labels, preds) * 100.0),
        "tpr_at_1_fpr": float(np.interp(0.01, fpr, tpr) * 100.0),
        "tpr_at_3_fpr": float(np.interp(0.03, fpr, tpr) * 100.0),
    }


def _recompute_results(payload: dict, data_dir: Path) -> dict[str, dict[str, float]]:
    features = payload["features"]
    metadata = payload.get("metadata", {})
    labels = _labels_from_manifest(data_dir, features["doc_ids"])
    feature_keys = {
        "score_ua": "score_ua",
        "gradient_ua": "gradient_ua",
        "docmia_fl": "fl",
        "docmia_fllora": "fllora",
        "docmia_ig": "ig",
    }

    results = {}
    for method_name, feature_key in feature_keys.items():
        if feature_key not in features:
            continue
        feature_array = np.asarray(features[feature_key], dtype=np.float64)
        preds, scores = _cluster_predict(
            feature_array,
            member_is_low=MEMBER_IS_LOW_SIGNAL[method_name],
        )
        result = _metrics(labels, preds, scores)
        result["max_questions_per_document"] = float(metadata.get("max_questions_per_document", 0.0))
        result["uses_auxiliary_data"] = float(metadata.get("uses_auxiliary_data", 0.0))
        results[method_name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/docmia_outputs.json")
    parser.add_argument("--scores", default="scoring/scores.json")
    parser.add_argument("--data-dir", default="data/synthetic_docvqa")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = _recompute_results(payload, Path(args.data_dir))
    scores = {
        "experiments": {
            "scaled_whitebox_docmia": {
                "results": results
            }
        }
    }
    Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
