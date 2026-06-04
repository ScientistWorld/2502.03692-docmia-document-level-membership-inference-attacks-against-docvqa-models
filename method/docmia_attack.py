"""DocMIA white-box attacks using optimization-based document features."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve
from torch import nn

from data.docvqa_loader import iter_documents, load_manifest, load_image
from method.model import TinyDocVQAModel


AGGREGATIONS = (np.mean, np.min, np.max, np.median)


def _metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds) * 100.0),
        "accuracy": float(accuracy_score(labels, preds) * 100.0),
        "f1": float(f1_score(labels, preds) * 100.0),
        "tpr_at_1_fpr": float(np.interp(0.01, fpr, tpr) * 100.0),
        "tpr_at_3_fpr": float(np.interp(0.03, fpr, tpr) * 100.0),
    }


def _aggregate(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return [float(fn(arr)) for fn in AGGREGATIONS]


def _normalize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def _cluster_predict(features: np.ndarray, labels: np.ndarray, member_is_low: bool) -> tuple[np.ndarray, np.ndarray]:
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


def _load_model(checkpoint: str | Path, device: torch.device) -> TinyDocVQAModel:
    payload = torch.load(checkpoint, map_location=device)
    model = TinyDocVQAModel(payload["num_questions"], payload["num_answers"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _doc_tensor(data_dir: Path, doc: dict, device: torch.device) -> torch.Tensor:
    return load_image(data_dir / doc["image_path"]).unsqueeze(0).to(device)


def score_features(
    model: nn.Module,
    documents: list[dict],
    data_dir: Path,
    answer_to_idx: dict[str, int],
    field_to_idx: dict[str, int],
    device: torch.device,
    max_questions: int,
) -> np.ndarray:
    rows = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for doc in documents:
            losses, utilities = [], []
            image = _doc_tensor(data_dir, doc, device)
            for qa in doc["qas"][:max_questions]:
                q = torch.tensor([field_to_idx[qa["field"]]], device=device)
                a = torch.tensor([answer_to_idx[qa["answer"]]], device=device)
                logits = model(image, q)
                losses.append(float(criterion(logits, a).item()))
                utilities.append(float(logits.argmax(dim=1).eq(a).item()))
            rows.append(_aggregate(utilities) + _aggregate(losses))
    return np.asarray(rows, dtype=np.float64)


def gradient_features(
    model: nn.Module,
    documents: list[dict],
    data_dir: Path,
    answer_to_idx: dict[str, int],
    field_to_idx: dict[str, int],
    device: torch.device,
    max_questions: int,
) -> np.ndarray:
    rows = []
    criterion = nn.CrossEntropyLoss()
    for doc in documents:
        norms, utilities = [], []
        image = _doc_tensor(data_dir, doc, device)
        for qa in doc["qas"][:max_questions]:
            model.zero_grad(set_to_none=True)
            q = torch.tensor([field_to_idx[qa["field"]]], device=device)
            a = torch.tensor([answer_to_idx[qa["answer"]]], device=device)
            logits = model(image, q)
            loss = criterion(logits, a)
            loss.backward()
            grad_norm = model.output_layer.weight.grad.detach().norm().item()
            norms.append(float(grad_norm))
            utilities.append(float(logits.argmax(dim=1).eq(a).item()))
        rows.append(_aggregate(norms) + _aggregate(utilities))
    model.zero_grad(set_to_none=True)
    return np.asarray(rows, dtype=np.float64)


def fl_features(
    model: TinyDocVQAModel,
    documents: list[dict],
    data_dir: Path,
    answer_to_idx: dict[str, int],
    field_to_idx: dict[str, int],
    device: torch.device,
    max_questions: int,
    steps: int,
    lr: float,
    tau: float,
) -> np.ndarray:
    """Algorithm 1 with theta replaced by the final layer L (FL)."""

    rows = []
    criterion = nn.CrossEntropyLoss()
    base_state = copy.deepcopy(model.output_layer.state_dict())

    for doc in documents:
        deltas, step_counts, utilities = [], [], []
        image = _doc_tensor(data_dir, doc, device)
        for qa in doc["qas"][:max_questions]:
            model.output_layer.load_state_dict(base_state)
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("output_layer.")
            optimizer = torch.optim.Adam(model.output_layer.parameters(), lr=lr)
            q = torch.tensor([field_to_idx[qa["field"]]], device=device)
            a = torch.tensor([answer_to_idx[qa["answer"]]], device=device)

            prev_loss = math.inf
            used_steps = 0
            utility_trace = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(image, q)
                loss = criterion(logits, a)
                utility_trace.append(float(logits.argmax(dim=1).eq(a).item()))
                if prev_loss != math.inf and abs(prev_loss - float(loss.item())) < tau:
                    break
                loss.backward()
                optimizer.step()
                prev_loss = float(loss.item())
                used_steps = step + 1

            delta_sq = 0.0
            tuned_state = model.output_layer.state_dict()
            for key, initial in base_state.items():
                delta_sq += float((tuned_state[key] - initial).pow(2).sum().item())
            deltas.append(math.sqrt(delta_sq))
            step_counts.append(float(used_steps))
            utilities.append(float(np.mean(utility_trace)))

        rows.append(_aggregate(deltas) + _aggregate(step_counts) + _aggregate(utilities))

    model.output_layer.load_state_dict(base_state)
    for param in model.parameters():
        param.requires_grad = True
    return np.asarray(rows, dtype=np.float64)


def ig_features(
    model: TinyDocVQAModel,
    documents: list[dict],
    data_dir: Path,
    answer_to_idx: dict[str, int],
    field_to_idx: dict[str, int],
    device: torch.device,
    max_questions: int,
    steps: int,
    lr: float,
    tau: float,
) -> np.ndarray:
    """Algorithm 1 with theta replaced by the document image pixels (IG)."""

    rows = []
    criterion = nn.CrossEntropyLoss()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for doc in documents:
        deltas, step_counts, utilities = [], [], []
        base_image = _doc_tensor(data_dir, doc, device)
        for qa in doc["qas"][:max_questions]:
            image = base_image.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([image], lr=lr)
            q = torch.tensor([field_to_idx[qa["field"]]], device=device)
            a = torch.tensor([answer_to_idx[qa["answer"]]], device=device)

            prev_loss = math.inf
            used_steps = 0
            utility_trace = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(image.clamp(0.0, 1.0), q)
                loss = criterion(logits, a)
                utility_trace.append(float(logits.argmax(dim=1).eq(a).item()))
                if prev_loss != math.inf and abs(prev_loss - float(loss.item())) < tau:
                    break
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    image.clamp_(0.0, 1.0)
                prev_loss = float(loss.item())
                used_steps = step + 1

            deltas.append(float((image.detach() - base_image).norm().item()))
            step_counts.append(float(used_steps))
            utilities.append(float(np.mean(utility_trace)))

        rows.append(_aggregate(deltas) + _aggregate(step_counts) + _aggregate(utilities))

    for param in model.parameters():
        param.requires_grad = True
    return np.asarray(rows, dtype=np.float64)


def run_attack(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    manifest = load_manifest(data_dir)
    documents = iter_documents(data_dir)
    labels = np.asarray([1 if doc["split"] == "member" else 0 for doc in documents], dtype=np.int64)
    answer_to_idx = {answer: idx for idx, answer in enumerate(manifest["answer_bank"])}
    field_to_idx = {field: idx for idx, field in enumerate(manifest["fields"])}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = _load_model(args.checkpoint, device)

    score_x = score_features(model, documents, data_dir, answer_to_idx, field_to_idx, device, args.max_questions)
    score_preds, score_scores = _cluster_predict(score_x, labels, member_is_low=False)

    grad_x = gradient_features(model, documents, data_dir, answer_to_idx, field_to_idx, device, args.max_questions)
    grad_preds, grad_scores = _cluster_predict(grad_x, labels, member_is_low=True)

    fl_x = fl_features(
        model,
        documents,
        data_dir,
        answer_to_idx,
        field_to_idx,
        device,
        args.max_questions,
        args.steps,
        args.lr,
        args.tau,
    )
    fl_preds, fl_scores = _cluster_predict(fl_x, labels, member_is_low=True)

    ig_x = ig_features(
        model,
        documents,
        data_dir,
        answer_to_idx,
        field_to_idx,
        device,
        args.max_questions,
        args.ig_steps,
        args.ig_lr,
        args.ig_tau,
    )
    ig_preds, ig_scores = _cluster_predict(ig_x, labels, member_is_low=True)

    output = {
        "metadata": {
            "num_documents": int(len(documents)),
            "member_documents": int(labels.sum()),
            "nonmember_documents": int((1 - labels).sum()),
            "max_questions_per_document": int(args.max_questions),
            "optimization_steps": int(args.steps),
            "fl_learning_rate": float(args.lr),
            "fl_threshold": float(args.tau),
            "ig_optimization_steps": int(args.ig_steps),
            "ig_learning_rate": float(args.ig_lr),
            "ig_threshold": float(args.ig_tau),
            "uses_auxiliary_data": 0.0,
        },
        "features": {
            "score_ua": score_x.tolist(),
            "gradient_ua": grad_x.tolist(),
            "fl": fl_x.tolist(),
            "ig": ig_x.tolist(),
            "labels": labels.tolist(),
        },
        "results": {
            "score_ua": _metrics(labels, score_preds, score_scores),
            "gradient_ua": _metrics(labels, grad_preds, grad_scores),
            "docmia_fl": _metrics(labels, fl_preds, fl_scores),
            "docmia_ig": _metrics(labels, ig_preds, ig_scores),
        },
    }
    for result in output["results"].values():
        result["max_questions_per_document"] = float(args.max_questions)
        result["uses_auxiliary_data"] = 0.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_docvqa")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_docvqa_target.pt")
    parser.add_argument("--output", default="checkpoints/docmia_outputs.json")
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1e-5)
    parser.add_argument("--ig-steps", type=int, default=60)
    parser.add_argument("--ig-lr", type=float, default=1e-2)
    parser.add_argument("--ig-tau", type=float, default=1e-5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output = run_attack(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["results"], indent=2))


if __name__ == "__main__":
    main()
