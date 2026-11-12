"""Score document-level membership attacks on the visible dev audit (train slice).

Protocol
--------
1. Load the dev audit's candidate documents, membership labels and target model.
2. Compute the comparison baselines here, from the target model alone.
3. Run the submitted attack program ``bash scripts/attack.sh <audit_dir> <out.json>``
   on a copy of the audit (target checkpoint + unlabeled candidates).
4. Recompute every metric from the raw per-document scores and 0/1 decisions.

The three reference slots are always reported; a slot that is missing, invalid,
or whose attack program failed gets the worst value of every metric.  Extra
attacks found in the output are scored too.  Nothing written by the attack other
than raw per-document scores and decisions is read.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data" / "dev_audit"
REQUIRED_SLOTS = ("docmia_fl", "docmia_fllora", "docmia_ig")
EXPERIMENTS = {
    "scaled_whitebox_docmia": {
        "max_questions": None,
        "metrics": ("balanced_accuracy", "f1", "roc_auc", "tpr_at_1_fpr", "tpr_at_3_fpr"),
    },
    "few_question_documents": {
        "max_questions": 2,
        "metrics": ("balanced_accuracy", "f1", "roc_auc", "tpr_at_3_fpr"),
    },
}
BASELINES = ("loss_ta", "score_ua", "scoreloss_ua_all", "gradient_ua")


class DevReader(nn.Module):
    """Architecture of the audited model, rebuilt from the checkpoint shapes."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        c1, c2, c3 = (state[f"page_encoder.{i}.weight"].shape[0] for i in (0, 4, 8))
        self.page_encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        q_num, q_dim = state["question_embedding.weight"].shape
        width, fused_in = state["fusion.0.weight"].shape
        hidden = state["fusion.2.weight"].shape[0]
        self.question_embedding = nn.Embedding(q_num, q_dim)
        self.fusion = nn.Sequential(nn.Linear(fused_in, width), nn.ReLU(), nn.Linear(width, hidden), nn.ReLU())
        self.output_layer = nn.Linear(hidden, state["output_layer.weight"].shape[0])
        self.load_state_dict(state)
        self.eval()

    def hidden(self, pages: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        feats = self.page_encoder(1.0 - pages)  # pages are white paper in [0, 1]; ink-positive input
        return self.fusion(torch.cat([feats, self.question_embedding(questions)], 1))


def _load_audit(audit_dir: Path) -> dict:
    manifest = json.loads((audit_dir / "candidates" / "manifest.json").read_text(encoding="utf-8"))
    fields = {f: i for i, f in enumerate(manifest["fields"])}
    answers = {a: i for i, a in enumerate(manifest["answer_bank"])}
    pages, pair_doc, pair_q, pair_a = [], [], [], []
    for i, doc in enumerate(manifest["documents"]):
        img = Image.open(audit_dir / "candidates" / doc["image"]).convert("L")
        pages.append(np.asarray(img, dtype=np.float32) / 255.0)
        for qa in doc["qas"]:
            pair_doc.append(i)
            pair_q.append(fields[qa["field"]])
            pair_a.append(answers[qa["answer"]])
    payload = torch.load(audit_dir / "target.pt", map_location="cpu", weights_only=False)
    return {
        "doc_ids": [doc["doc_id"] for doc in manifest["documents"]],
        "pages": torch.from_numpy(np.stack(pages)).unsqueeze(1),
        "pair_doc": np.asarray(pair_doc),
        "pair_q": torch.tensor(pair_q),
        "pair_a": torch.tensor(pair_a),
        "model": DevReader(payload["model_state"]),
    }


def _phi_all(values: np.ndarray, pair_doc: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, 4))
    for d in range(n):
        v = values[pair_doc == d]
        out[d] = (v.mean(), v.min(), v.max(), np.median(v))
    return out


def _kmeans_attack(features: np.ndarray, column: int, member_high: bool):
    std = features.std(0)
    z = (features - features.mean(0)) / np.where(std > 0, std, 1.0)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(z)
    if len(np.unique(labels)) < 2:
        return np.zeros(len(z), dtype=np.int64), np.zeros(len(z))
    means = [features[labels == c, column].mean() for c in (0, 1)]
    member = int(np.argmax(means) if member_high else np.argmin(means))
    centroid = z[labels == member].mean(0)
    return (labels == member).astype(np.int64), -np.linalg.norm(z - centroid, axis=1)


@torch.no_grad()
def _baselines(audit: dict) -> dict:
    model = audit["model"]
    losses, utils, grads = [], [], []
    for start in range(0, len(audit["pair_doc"]), 256):
        sl = slice(start, start + 256)
        pages = audit["pages"][torch.from_numpy(audit["pair_doc"][sl])]
        h = model.hidden(pages, audit["pair_q"][sl])
        logits = model.output_layer(h)
        a = audit["pair_a"][sl]
        losses.append(nn.functional.cross_entropy(logits, a, reduction="none"))
        utils.append((logits.argmax(1) == a).double())
        residual = torch.softmax(logits, 1)
        residual[torch.arange(len(a)), a] -= 1.0
        grads.append(residual.norm(dim=1) * torch.sqrt((h * h).sum(1) + 1.0))
    loss, util, grad = (torch.cat(x).double().numpy() for x in (losses, utils, grads))
    n, pd = len(audit["doc_ids"]), audit["pair_doc"]
    loss_phi, util_phi, grad_phi = (_phi_all(x, pd, n) for x in (loss, util, grad))
    mean_loss = loss_phi[:, 0]
    out = {"loss_ta": ((mean_loss <= mean_loss.mean()).astype(np.int64), -mean_loss)}
    out["score_ua"] = _kmeans_attack(util_phi[:, :1], 0, member_high=True)
    out["scoreloss_ua_all"] = _kmeans_attack(np.concatenate([loss_phi, util_phi], 1), 0, member_high=False)
    out["gradient_ua"] = _kmeans_attack(np.concatenate([grad_phi, util_phi], 1), 0, member_high=False)
    return out


def _metrics(y: np.ndarray, preds: np.ndarray, scores: np.ndarray, names) -> dict:
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    values = {
        "balanced_accuracy": 100.0 * balanced_accuracy_score(y, preds),
        "f1": 100.0 * f1_score(y, preds, zero_division=0),
        "roc_auc": 100.0 * roc_auc_score(y, scores),
        "tpr_at_1_fpr": 100.0 * float(tpr[fpr <= 0.01].max()),
        "tpr_at_3_fpr": 100.0 * float(tpr[fpr <= 0.03].max()),
    }
    return {k: float(values[k]) for k in names}


def _worst(names) -> dict:
    return {k: 0.0 for k in names}


def _is_binary(x) -> bool:
    return (isinstance(x, (bool, int, float)) and not (isinstance(x, float) and math.isnan(x))
            and float(x) in (0.0, 1.0))


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _parse_output(path: Path, doc_ids: list[str]):
    """Return ({name: (preds, scores)}, {name: status}) from raw attack output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("doc_ids")
    if not isinstance(ids, list) or len(ids) != len(doc_ids) or set(ids) != set(doc_ids) or len(set(ids)) != len(ids):
        raise ValueError("doc_ids must list every candidate document exactly once")
    order = np.asarray([ids.index(d) for d in doc_ids]) if ids != doc_ids else np.arange(len(ids))
    attacks = data.get("attacks")
    if not isinstance(attacks, dict):
        raise ValueError("'attacks' must be a mapping from attack name to outputs")
    parsed, status = {}, {}
    for name, entry in attacks.items():
        if not isinstance(name, str) or not name or name.startswith("_") or len(name) > 64 or name in BASELINES:
            status[str(name)[:64]] = "rejected: invalid or reserved attack name"
            continue
        if not isinstance(entry, dict):
            status[name] = "invalid: entry must be an object"
            continue
        scores, preds = entry.get("scores"), entry.get("predictions")
        if not isinstance(scores, list) or len(scores) != len(doc_ids) or not all(map(_is_finite_number, scores)):
            status[name] = "invalid: scores must be one finite number per document"
            continue
        if not isinstance(preds, list) or len(preds) != len(doc_ids) or not all(map(_is_binary, preds)):
            status[name] = "invalid: predictions must be one 0/1 decision per document"
            continue
        parsed[name] = (np.asarray(preds, dtype=np.int64)[order], np.asarray(scores, dtype=np.float64)[order])
        status[name] = "ok"
    return parsed, status


def _run_attack(command: str, audit_dir: Path, timeout: float, keep_output: str | None):
    work = Path(tempfile.mkdtemp(prefix="dev_eval_", dir=str(ROOT / "tmp")))
    try:
        shutil.copytree(audit_dir / "candidates", work / "audit" / "candidates")
        shutil.copy2(audit_dir / "target.pt", work / "audit" / "target.pt")
        out = work / "attack_output.json"
        proc = subprocess.run(shlex.split(command) + [str(work / "audit"), str(out)], cwd=str(ROOT), timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"attack program exited with code {proc.returncode}")
        if not out.exists():
            raise RuntimeError("attack program wrote no output")
        if keep_output:
            Path(keep_output).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, keep_output)
        return out, work
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def evaluate(command: str, audit_dir: Path, labels_path: Path, timeout: float, keep_output: str | None) -> dict:
    audit = _load_audit(audit_dir)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    y = np.asarray([int(labels[d]) for d in audit["doc_ids"]], dtype=np.int64)
    n_questions = np.bincount(audit["pair_doc"], minlength=len(y))
    baselines = _baselines(audit)

    attacks, status, failure = {}, {}, None
    work = None
    try:
        out, work = _run_attack(command, audit_dir, timeout, keep_output)
        attacks, status = _parse_output(out, audit["doc_ids"])
    except Exception as exc:  # any failure of the attack program scores as failed
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)

    names = list(REQUIRED_SLOTS) + sorted(n for n in attacks if n not in REQUIRED_SLOTS)
    experiments = {}
    for exp, spec in EXPERIMENTS.items():
        mask = np.ones(len(y), dtype=bool) if spec["max_questions"] is None else n_questions <= spec["max_questions"]
        results = {}
        for name in names:
            if name in attacks:
                preds, scores = attacks[name]
                results[name] = _metrics(y[mask], preds[mask], scores[mask], spec["metrics"])
            else:
                results[name] = _worst(spec["metrics"])
        experiments[exp] = {
            "slice": "train",
            "documents": int(mask.sum()),
            "members": int(y[mask].sum()),
            "results": results,
            "baselines": {b: _metrics(y[mask], p[mask], s[mask], spec["metrics"]) for b, (p, s) in baselines.items()},
            "attack_status": {
                "failure": failure,
                "slots": {n: status.get(n, "missing: scored with worst values") for n in names},
                "other": {n: s for n, s in status.items() if n not in names},
            },
        }
    return {"experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-cmd", default="bash scripts/attack.sh")
    parser.add_argument("--audit-dir", default=str(AUDIT))
    parser.add_argument("--labels", default=str(AUDIT / "labels.json"))
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--keep-output", default=None)
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    n = int(os.environ.get("DOCMIA_THREADS", "1") or 1)
    torch.set_num_threads(max(1, n))
    (ROOT / "tmp").mkdir(exist_ok=True)
    scores_path = Path(args.scores)
    scores_path.unlink(missing_ok=True)
    scores = evaluate(args.attack_cmd, Path(args.audit_dir), Path(args.labels), args.timeout, args.keep_output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps({e: v["results"] for e, v in scores["experiments"].items()}, indent=1))


if __name__ == "__main__":
    main()
