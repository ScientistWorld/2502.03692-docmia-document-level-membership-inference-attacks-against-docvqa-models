"""Hidden-test scoring of document-level membership attacks.

The hidden audit (``eval/test/audit``: a target fine-tuned on a private set
drawn from a secret seed, plus 600 unlabeled candidate documents with neutral,
shuffled ids) and its membership labels exist only in this directory.

Run with no arguments (``bash scripts/evaluate_test.sh``):

1. labels, candidates and target are loaded and checked against recorded hashes;
2. the comparison baselines are computed here from the hidden target;
3. ``bash scripts/attack.sh <copy_of_audit> <out.json>`` runs on a copy of the
   audit (target checkpoint + unlabeled candidates only);
4. every metric is recomputed from the raw per-document scores and decisions.

The three reference slots are always written.  A slot that is missing or
invalid, or every slot when the attack program fails, gets the worst value of
every metric.  Extra attacks in the output are scored as additional rows.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HIDDEN_AUDIT = HERE / "audit"
HIDDEN_LABELS = HERE / "audit_labels.json"
HIDDEN_HASHES = HERE / "audit_sha256.json"

SLOTS = ("docmia_fl", "docmia_fllora", "docmia_ig")
RESERVED = {"loss_ta", "score_ua", "scoreloss_ua_all", "gradient_ua"}
EXPERIMENT_SPECS = (
    ("scaled_whitebox_docmia", None, ("balanced_accuracy", "f1", "roc_auc", "tpr_at_1_fpr", "tpr_at_3_fpr")),
    ("few_question_documents", 2, ("balanced_accuracy", "f1", "roc_auc", "tpr_at_3_fpr")),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HiddenTarget(nn.Module):
    """The audited architecture, sized from the hidden checkpoint itself."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        widths = [state[f"page_encoder.{k}.weight"].shape[0] for k in (0, 3, 6)]
        layers, prev = [], 1
        for width in widths:
            layers += [nn.Conv2d(prev, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            prev = width
        self.page_encoder = nn.Sequential(*layers, nn.Flatten())
        self.question_embedding = nn.Embedding(*state["question_embedding.weight"].shape)
        fuse_out, fuse_in = state["fusion.0.weight"].shape
        self.fusion = nn.Sequential(nn.Linear(fuse_in, fuse_out), nn.ReLU(),
                                    nn.Linear(fuse_out, state["fusion.2.weight"].shape[0]), nn.ReLU())
        self.output_layer = nn.Linear(*reversed(state["output_layer.weight"].shape))
        self.load_state_dict(state)
        self.eval()

    def head(self, pages: torch.Tensor, questions: torch.Tensor):
        hidden = self.fusion(torch.cat([self.page_encoder(pages), self.question_embedding(questions)], 1))
        return hidden, self.output_layer(hidden)


class HiddenAudit:
    def __init__(self, audit_dir: Path, labels_path: Path, hashes_path: Path | None) -> None:
        if hashes_path is not None and hashes_path.exists():
            expected = json.loads(hashes_path.read_text(encoding="utf-8"))
            for rel, digest in expected.items():
                if sha256(audit_dir / rel) != digest:
                    raise RuntimeError(f"hidden audit file {rel} does not match its recorded hash")
        manifest = json.loads((audit_dir / "candidates" / "manifest.json").read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.doc_ids = [doc["doc_id"] for doc in manifest["documents"]]
        self.y = np.array([int(labels[d]) for d in self.doc_ids], dtype=np.int64)
        field_pos = {f: i for i, f in enumerate(manifest["fields"])}
        answer_pos = {a: i for i, a in enumerate(manifest["answer_bank"])}
        pages, self.pair_doc, q, a = [], [], [], []
        for i, doc in enumerate(manifest["documents"]):
            pages.append(np.asarray(Image.open(audit_dir / "candidates" / doc["image"]).convert("L"), np.float32) / 255.0)
            for qa in doc["qas"]:
                self.pair_doc.append(i)
                q.append(field_pos[qa["field"]])
                a.append(answer_pos[qa["answer"]])
        self.pair_doc = np.array(self.pair_doc)
        self.n_questions = np.bincount(self.pair_doc, minlength=len(self.doc_ids))
        self.pages = torch.from_numpy(np.stack(pages))[:, None]
        self.q, self.a = torch.tensor(q), torch.tensor(a)
        state = torch.load(audit_dir / "target.pt", map_location="cpu", weights_only=False)["model_state"]
        self.target = HiddenTarget(state)


def aggregate_all(per_pair: np.ndarray, pair_doc: np.ndarray, n_docs: int) -> np.ndarray:
    rows = []
    for d in range(n_docs):
        v = per_pair[pair_doc == d]
        rows.append([v.mean(), v.min(), v.max(), np.median(v)])
    return np.asarray(rows)


def unsupervised(features: np.ndarray, key_column: int, members_have_high_key: bool):
    spread = features.std(0)
    z = (features - features.mean(0)) / np.where(spread > 0, spread, 1.0)
    assign = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(z)
    if np.unique(assign).size < 2:
        return np.zeros(len(z), np.int64), np.zeros(len(z))
    key = [features[assign == c, key_column].mean() for c in (0, 1)]
    member = int(np.argmax(key)) if members_have_high_key else int(np.argmin(key))
    center = z[assign == member].mean(0)
    return (assign == member).astype(np.int64), -np.sqrt(((z - center) ** 2).sum(1))


@torch.no_grad()
def baseline_attacks(audit: HiddenAudit) -> dict:
    loss, util, grad = [], [], []
    for lo in range(0, len(audit.pair_doc), 256):
        idx = torch.arange(lo, min(lo + 256, len(audit.pair_doc)))
        docs = torch.from_numpy(audit.pair_doc[lo:lo + 256])
        hidden, logits = audit.target.head(audit.pages[docs], audit.q[idx])
        ans = audit.a[idx]
        loss.append(nn.functional.cross_entropy(logits, ans, reduction="none").double())
        util.append((logits.argmax(1) == ans).double())
        resid = torch.softmax(logits, 1)
        resid[torch.arange(len(idx)), ans] -= 1
        grad.append((resid.norm(dim=1) * (hidden.pow(2).sum(1) + 1).sqrt()).double())
    n = len(audit.doc_ids)
    loss_f, util_f, grad_f = (aggregate_all(torch.cat(t).numpy(), audit.pair_doc, n) for t in (loss, util, grad))
    avg_loss = loss_f[:, 0]
    return {
        "loss_ta": ((avg_loss <= avg_loss.mean()).astype(np.int64), -avg_loss),
        "score_ua": unsupervised(util_f[:, :1], 0, True),
        "scoreloss_ua_all": unsupervised(np.hstack([loss_f, util_f]), 0, False),
        "gradient_ua": unsupervised(np.hstack([grad_f, util_f]), 0, False),
    }


def score_rows(y: np.ndarray, decisions: np.ndarray, scores: np.ndarray, wanted) -> dict:
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    table = {
        "balanced_accuracy": balanced_accuracy_score(y, decisions),
        "f1": f1_score(y, decisions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "tpr_at_1_fpr": tpr[fpr <= 0.01].max(),
        "tpr_at_3_fpr": tpr[fpr <= 0.03].max(),
    }
    return {m: float(100.0 * table[m]) for m in wanted}


def read_attack_output(path: Path, doc_ids: list[str]):
    raw = json.loads(path.read_text(encoding="utf-8"))
    listed = raw.get("doc_ids")
    if not (isinstance(listed, list) and len(listed) == len(doc_ids)
            and len(set(listed)) == len(listed) and set(listed) == set(doc_ids)):
        raise ValueError("doc_ids must contain each candidate document exactly once")
    where = {d: i for i, d in enumerate(listed)}
    perm = np.array([where[d] for d in doc_ids])
    block = raw.get("attacks")
    if not isinstance(block, dict):
        raise ValueError("'attacks' must map attack names to their outputs")

    def finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    def zero_one(x):
        return isinstance(x, (bool, int, float)) and x in (0, 1) and not (isinstance(x, float) and math.isnan(x))

    usable, notes = {}, {}
    for name, item in block.items():
        label = str(name)[:64]
        if not isinstance(name, str) or not name or name.startswith("_") or len(name) > 64 or name in RESERVED:
            notes[label] = "rejected: invalid or reserved attack name"
        elif not isinstance(item, dict):
            notes[label] = "invalid: entry must be an object"
        elif not (isinstance(item.get("scores"), list) and len(item["scores"]) == len(doc_ids)
                  and all(finite(s) for s in item["scores"])):
            notes[label] = "invalid: scores must be one finite number per document"
        elif not (isinstance(item.get("predictions"), list) and len(item["predictions"]) == len(doc_ids)
                  and all(zero_one(p) for p in item["predictions"])):
            notes[label] = "invalid: predictions must be one 0/1 decision per document"
        else:
            usable[name] = (np.array(item["predictions"], dtype=np.int64)[perm],
                            np.array(item["scores"], dtype=np.float64)[perm])
            notes[label] = "ok"
    return usable, notes


def run_attack_program(command: str, audit_dir: Path, timeout: float, keep_output: str | None) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="hidden_eval_", dir=str(ROOT / "tmp")))
    target_dir = scratch / "audit"
    shutil.copytree(audit_dir / "candidates", target_dir / "candidates")
    shutil.copy2(audit_dir / "target.pt", target_dir / "target.pt")
    result = scratch / "attack_output.json"
    proc = subprocess.run(shlex.split(command) + [str(target_dir), str(result)], cwd=str(ROOT), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"attack program exited with code {proc.returncode}")
    if not result.exists():
        raise RuntimeError("attack program wrote no output")
    if keep_output:
        Path(keep_output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, keep_output)
    return result


def failed_payload(reason: str) -> dict:
    return {"experiments": {
        name: {"slice": "test", "results": {slot: {m: 0.0 for m in metrics} for slot in SLOTS},
               "attack_status": {"failure": reason}}
        for name, _, metrics in EXPERIMENT_SPECS
    }}


def score(command: str, audit_dir: Path, labels_path: Path, hashes_path: Path | None,
          timeout: float, keep_output: str | None) -> dict:
    audit = HiddenAudit(audit_dir, labels_path, hashes_path)
    reference_attacks = baseline_attacks(audit)
    usable, notes, failure = {}, {}, None
    try:
        output = run_attack_program(command, audit_dir, timeout, keep_output)
        usable, notes = read_attack_output(output, audit.doc_ids)
        shutil.rmtree(output.parent, ignore_errors=True)
    except Exception as exc:  # the attack program failed: every slot scores as failed
        failure = f"{type(exc).__name__}: {exc}"
    rows = list(SLOTS) + sorted(set(usable) - set(SLOTS))
    payload = {"experiments": {}}
    for name, max_q, metrics in EXPERIMENT_SPECS:
        keep = np.ones(len(audit.y), bool) if max_q is None else audit.n_questions <= max_q
        y = audit.y[keep]
        results = {}
        for row in rows:
            if row in usable:
                decisions, scores = usable[row]
                results[row] = score_rows(y, decisions[keep], scores[keep], metrics)
            else:
                results[row] = {m: 0.0 for m in metrics}
        payload["experiments"][name] = {
            "slice": "test",
            "documents": int(keep.sum()),
            "members": int(y.sum()),
            "results": results,
            "baselines": {b: score_rows(y, d[keep], s[keep], metrics) for b, (d, s) in reference_attacks.items()},
            "attack_status": {"failure": failure,
                              "slots": {r: notes.get(r, "missing: scored with worst values") for r in rows},
                              "other": {k: v for k, v in notes.items() if k not in rows}},
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-cmd", default="bash scripts/attack.sh")
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_test.json"))
    parser.add_argument("--audit-dir", default=str(HIDDEN_AUDIT))
    parser.add_argument("--labels", default=str(HIDDEN_LABELS))
    parser.add_argument("--hashes", default=str(HIDDEN_HASHES))
    parser.add_argument("--keep-output", default=None)
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    cores = int(os.environ.get("DOCMIA_THREADS", "1") or 1)
    torch.set_num_threads(max(1, cores))
    (ROOT / "tmp").mkdir(exist_ok=True)
    out = Path(args.scores)
    out.unlink(missing_ok=True)
    try:
        payload = score(args.attack_cmd, Path(args.audit_dir), Path(args.labels),
                        Path(args.hashes) if args.hashes else None, args.timeout, args.keep_output)
    except Exception as exc:  # evaluator-side failure still yields worst-value rows
        traceback.print_exc()
        payload = failed_payload(f"{type(exc).__name__}: {exc}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({e: v["results"] for e, v in payload["experiments"].items()}, indent=1))


if __name__ == "__main__":
    main()
