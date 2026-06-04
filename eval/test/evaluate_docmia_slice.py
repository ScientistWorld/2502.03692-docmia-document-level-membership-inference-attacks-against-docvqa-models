"""Score DocMIA method artifacts on the held-out test slice.

This evaluator reads per-document method outputs and never imports method code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


EXPERIMENT = "scaled_whitebox_docmia"
RESERVED_FEATURE_KEYS = {"doc_ids", "labels"}
FEATURE_ALIASES = {
    "score_ua": ("score_ua", False),
    "gradient_ua": ("gradient_ua", True),
    "fl": ("docmia_fl", True),
    "fllora": ("docmia_fllora", True),
    "ig": ("docmia_ig", True),
    "docmia_fl": ("docmia_fl", True),
    "docmia_fllora": ("docmia_fllora", True),
    "docmia_ig": ("docmia_ig", True),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_doc_ids(data_dir: Path, features: dict[str, Any]) -> list[str]:
    if "doc_ids" in features:
        return list(features["doc_ids"])
    manifest = _read_json(data_dir / "manifest.json")
    return [
        doc["doc_id"]
        for doc in manifest["documents"]
        if doc["split"] in {"member", "nonmember"}
    ]


def _labels_from_manifest(data_dir: Path, doc_ids: list[str]) -> np.ndarray:
    manifest = _read_json(data_dir / "manifest.json")
    split_by_id = {doc["doc_id"]: doc["split"] for doc in manifest["documents"]}
    missing = [doc_id for doc_id in doc_ids if doc_id not in split_by_id]
    if missing:
        raise KeyError(f"Unknown document ids in slice: {missing[:5]}")
    return np.asarray([1 if split_by_id[doc_id] == "member" else 0 for doc_id in doc_ids], dtype=np.int64)


def _slice_indices(source_doc_ids: list[str], slice_doc_ids: list[str]) -> list[int]:
    index_by_doc_id = {doc_id: idx for idx, doc_id in enumerate(source_doc_ids)}
    missing = [doc_id for doc_id in slice_doc_ids if doc_id not in index_by_doc_id]
    if missing:
        raise KeyError(f"Method artifact is missing slice documents: {missing[:5]}")
    return [index_by_doc_id[doc_id] for doc_id in slice_doc_ids]


def _as_array(value: Any, expected_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional numeric array")
    if arr.shape[0] != expected_rows:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {expected_rows}")
    return arr


def _normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def _cluster_predict(features: np.ndarray, member_is_low: bool) -> tuple[np.ndarray, np.ndarray]:
    signal = features[:, 0]
    if np.unique(features, axis=0).shape[0] < 2:
        scores = -signal if member_is_low else signal
        threshold = float(np.median(scores))
        return (scores >= threshold).astype(np.int64), scores

    norm_features = _normalize(features)
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(norm_features)
    cluster_signal = []
    for cluster in (0, 1):
        mask = clusters == cluster
        cluster_signal.append(float(signal[mask].mean()) if np.any(mask) else np.nan)
    if np.any(np.isnan(cluster_signal)):
        scores = -signal if member_is_low else signal
        threshold = float(np.median(scores))
        return (scores >= threshold).astype(np.int64), scores

    member_cluster = int(np.argmin(cluster_signal) if member_is_low else np.argmax(cluster_signal))
    preds = (clusters == member_cluster).astype(np.int64)
    centroid = norm_features[clusters == member_cluster].mean(axis=0)
    distances = np.linalg.norm(norm_features - centroid[None, :], axis=1)
    scores = -distances
    return preds, scores


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores) * 100.0)
    except ValueError:
        return 50.0


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores) * 100.0)
    except ValueError:
        return float(labels.mean() * 100.0)


def _metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds) * 100.0),
        "accuracy": float(accuracy_score(labels, preds) * 100.0),
        "f1": float(f1_score(labels, preds, zero_division=0) * 100.0),
        "matthews_corrcoef": float(matthews_corrcoef(labels, preds) * 100.0),
        "roc_auc": _safe_roc_auc(labels, scores),
        "average_precision": _safe_average_precision(labels, scores),
        "tpr_at_1_fpr": float(np.interp(0.01, fpr, tpr) * 100.0),
        "tpr_at_3_fpr": float(np.interp(0.03, fpr, tpr) * 100.0),
        "positive_rate_deviation": float(abs(preds.mean() - labels.mean()) * 100.0),
    }


def _metadata_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metadata = payload.get("metadata", {})
    has_question_budget = "max_questions_per_document" in metadata
    has_auxiliary_flag = "uses_auxiliary_data" in metadata
    return {
        "max_questions_per_document": float(metadata.get("max_questions_per_document", 1_000_000.0)),
        "uses_auxiliary_data": float(metadata.get("uses_auxiliary_data", 1.0)),
        "constraint_metadata_missing": float(not (has_question_budget and has_auxiliary_flag)),
    }


def _label_copy_suspect(values: np.ndarray, labels: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != labels.shape[0]:
        return 0.0
    for col in range(arr.shape[1]):
        column = arr[:, col]
        if np.unique(column).size > 2:
            continue
        binary = (column >= 0.5).astype(np.int64)
        if np.array_equal(binary, labels) or np.array_equal(1 - binary, labels):
            return 1.0
    return 0.0


def _member_is_low(metadata: dict[str, Any], feature_key: str, method_name: str, default: bool) -> bool:
    overrides = metadata.get("member_is_low_signal", {})
    if feature_key in overrides:
        return bool(overrides[feature_key])
    if method_name in overrides:
        return bool(overrides[method_name])
    return default


def _score_feature_methods(
    payload: dict[str, Any],
    source_doc_ids: list[str],
    slice_indices: list[int],
    labels: np.ndarray,
    source_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    features = payload.get("features", {})
    metadata = payload.get("metadata", {})
    results: dict[str, dict[str, float]] = {}

    for feature_key, value in features.items():
        if feature_key in RESERVED_FEATURE_KEYS:
            continue
        if feature_key in FEATURE_ALIASES:
            method_name, default_member_is_low = FEATURE_ALIASES[feature_key]
        else:
            method_name, default_member_is_low = feature_key, False
        try:
            full_array = _as_array(value, len(source_doc_ids), feature_key)
        except (TypeError, ValueError):
            continue
        feature_array = full_array[slice_indices]
        preds, scores = _cluster_predict(
            feature_array,
            member_is_low=_member_is_low(metadata, feature_key, method_name, default_member_is_low),
        )
        result = _metrics(labels, preds, scores)
        result.update(_metadata_metrics(payload))
        result["label_copy_suspect"] = _label_copy_suspect(full_array, source_labels)
        results[method_name] = result
    return results


def _score_prediction_methods(
    payload: dict[str, Any],
    source_doc_ids: list[str],
    slice_indices: list[int],
    labels: np.ndarray,
    source_labels: np.ndarray,
    train_thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    predictions = payload.get("predictions") or payload.get("membership_predictions") or {}
    scores_by_method = payload.get("membership_scores") or payload.get("scores") or {}
    results: dict[str, dict[str, float]] = {}

    for method_name, pred_values in predictions.items():
        full_preds = _as_array(pred_values, len(source_doc_ids), method_name).ravel()
        preds = full_preds[slice_indices]
        preds = (preds >= 0.5).astype(np.int64)
        if method_name in scores_by_method:
            full_scores = _as_array(scores_by_method[method_name], len(source_doc_ids), method_name).ravel()
            scores = full_scores[slice_indices]
        else:
            full_scores = full_preds
            scores = preds.astype(np.float64)
        result = _metrics(labels, preds, scores)
        result.update(_metadata_metrics(payload))
        result["label_copy_suspect"] = max(
            _label_copy_suspect(full_preds, source_labels),
            _label_copy_suspect(full_scores, source_labels),
        )
        results[method_name] = result

    for method_name, score_values in scores_by_method.items():
        if method_name in results:
            continue
        full_scores = _as_array(score_values, len(source_doc_ids), method_name).ravel()
        scores = full_scores[slice_indices]
        threshold = float(train_thresholds.get(method_name, np.median(scores)))
        preds = (scores >= threshold).astype(np.int64)
        result = _metrics(labels, preds, scores)
        result.update(_metadata_metrics(payload))
        result["label_copy_suspect"] = _label_copy_suspect(full_scores, source_labels)
        results[method_name] = result
    return results


def _requested_methods(train_scores_path: Path) -> list[str] | None:
    if not train_scores_path.exists():
        return None
    train_scores = _read_json(train_scores_path)
    results = train_scores["experiments"][EXPERIMENT]["results"]
    return list(results.keys())


def _train_thresholds(train_scores_path: Path) -> dict[str, float]:
    if not train_scores_path.exists():
        return {}
    train_scores = _read_json(train_scores_path)
    experiment = train_scores["experiments"][EXPERIMENT]
    calibration = experiment.get("calibration", {})
    return {
        method: float(values["score_threshold"])
        for method, values in calibration.items()
        if isinstance(values, dict) and "score_threshold" in values
    }


def evaluate(
    input_path: Path,
    data_dir: Path,
    slice_path: Path,
    train_scores_path: Path,
) -> dict[str, Any]:
    payload = _read_json(input_path)
    split = _read_json(slice_path)
    slice_doc_ids = list(split["doc_ids"])
    source_ids = _source_doc_ids(data_dir, payload.get("features", {}))
    indices = _slice_indices(source_ids, slice_doc_ids)
    labels = _labels_from_manifest(data_dir, slice_doc_ids)
    source_labels = _labels_from_manifest(data_dir, source_ids)

    results = _score_feature_methods(payload, source_ids, indices, labels, source_labels)
    results.update(
        _score_prediction_methods(
            payload,
            source_ids,
            indices,
            labels,
            source_labels,
            _train_thresholds(train_scores_path),
        )
    )
    if not results:
        raise ValueError("No compatible per-document methods were found in the input artifact")

    requested = _requested_methods(train_scores_path)
    if requested is not None:
        missing = [method for method in requested if method not in results]
        if missing:
            raise KeyError(f"Test artifact is missing methods scored on train: {missing}")
        results = {method: results[method] for method in requested}

    return {"experiments": {EXPERIMENT: {"slice": "test", "results": results}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/docmia_outputs.json")
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--data-dir", default="data/synthetic_docvqa")
    parser.add_argument("--slice-doc-ids", default="eval/test/slice_doc_ids.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    scores = evaluate(
        Path(args.input),
        Path(args.data_dir),
        Path(args.slice_doc_ids),
        Path(args.train_scores),
    )
    output_path = Path(args.scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
