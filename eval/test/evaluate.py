"""Hidden-test scoring of document-level membership attacks.

The hidden test holds several independent audits (``eval/test/audits/<name>``).
Each is a target fine-tuned with the public recipe on its own private document
set drawn from a secret seed, plus 600 member and 600 non-member candidate
documents with neutral, shuffled ids.  Their labels exist only in
``eval/test/audit_labels.json``.

Run with no arguments (``bash scripts/evaluate_test.sh``):

1. every hidden file (targets, candidates, labels) is read once and checked
   against ``eval/test/audit_sha256.json``; the labels are not kept in memory;
2. the comparison baselines are computed here from each hidden target;
3. ``eval/test`` and ``data/dev_audit`` are made unreadable.  For each audit in
   turn, a fresh copy holding only that audit's target and unlabeled candidates
   is written, ``bash scripts/attack.sh <copy> <out.json>`` runs on it, and the
   copy is deleted;
4. read access is restored, the labels are read (only now) and checked again,
   and every metric is recomputed from the raw per-document scores and 0/1
   decisions, per audit, then averaged over the audits.

The ranked ``membership_attack`` row is always written.  If the attack program
fails on any audit, or its ``membership_attack`` entry is missing or invalid on
any audit, the row gets the worst value (0) of every metric.  Extra named
attacks are reported unranked.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("DOCMIA_THREADS", "1"))

import argparse
import hashlib
import io
import json
import math
import shlex
import shutil
import signal
import stat
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
LABELS = HERE / "audit_labels.json"
HASHES = HERE / "audit_sha256.json"
SEALED_DIRS = (HERE, ROOT / "data" / "dev_audit")  # unreadable while the attack runs

EXPERIMENT = "scaled_whitebox_docmia"
SLOT = "membership_attack"  # the ranked submission; other attacks are reported unranked
RESERVED = {"loss_ta", "score_ua", "score_ua_all", "scoreloss_ua_all", "gradient_ua"}
METRICS = ("balanced_accuracy", "f1", "roc_auc", "tpr_at_1_fpr", "tpr_at_3_fpr")
LOW_EVIDENCE = 2  # diagnostic subset: candidates carrying at most this many QA pairs
LOW_METRICS = ("balanced_accuracy", "f1", "roc_auc", "tpr_at_3_fpr", "member_recall")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HiddenTarget(nn.Module):
    """The audited architecture, sized from the hidden checkpoint itself."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        widths = [state[f"page_encoder.{k}.weight"].shape[0] for k in (0, 4, 8)]
        layers, prev = [], 1
        for width in widths:
            layers += [nn.Conv2d(prev, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
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
        ink = 1.0 - pages  # white-paper pages in [0, 1] are fed ink-positive
        hidden = self.fusion(torch.cat([self.page_encoder(ink), self.question_embedding(questions)], 1))
        return hidden, self.output_layer(hidden)


class HiddenAudit:
    """One hidden audit, built from hash-verified bytes held in memory (no labels)."""

    def __init__(self, name: str, files: dict[str, bytes]) -> None:
        self.name, self.files = name, files
        manifest = json.loads(files["candidates/manifest.json"].decode("utf-8"))
        self.doc_ids = [doc["doc_id"] for doc in manifest["documents"]]
        field_pos = {f: i for i, f in enumerate(manifest["fields"])}
        answer_pos = {a: i for i, a in enumerate(manifest["answer_bank"])}
        pages, pair_doc, q, a = [], [], [], []
        for i, doc in enumerate(manifest["documents"]):
            image = Image.open(io.BytesIO(files[f"candidates/{doc['image']}"])).convert("L")
            pages.append(np.asarray(image, np.float32) / 255.0)
            for qa in doc["qas"]:
                pair_doc.append(i)
                q.append(field_pos[qa["field"]])
                a.append(answer_pos[qa["answer"]])
        self.pair_doc = np.array(pair_doc)
        self.n_questions = np.bincount(self.pair_doc, minlength=len(self.doc_ids))
        self.pages = torch.from_numpy(np.stack(pages))[:, None]
        self.q, self.a = torch.tensor(q), torch.tensor(a)
        state = torch.load(io.BytesIO(files["target.pt"]), map_location="cpu", weights_only=False)["model_state"]
        self.target = HiddenTarget(state)


def load_hidden(hashes_path: Path) -> tuple[list[HiddenAudit], dict]:
    record = json.loads(hashes_path.read_text(encoding="utf-8"))
    names, expected = record["audits"], record["files"]
    audits = []
    for name in names:
        prefix = f"audits/{name}/"
        files = {}
        for rel, digest in expected.items():
            if not rel.startswith(prefix):
                continue
            data = (HERE / rel).read_bytes()
            if sha256(data) != digest:
                raise RuntimeError(f"hidden file {rel} does not match its recorded hash")
            files[rel[len(prefix):]] = data
        if "target.pt" not in files or "candidates/manifest.json" not in files:
            raise RuntimeError(f"hidden audit {name} is incomplete")
        audits.append(HiddenAudit(name, files))
    if sha256(LABELS.read_bytes()) != expected["audit_labels.json"]:
        raise RuntimeError("hidden labels do not match their recorded hash")
    return audits, expected


def load_labels(audits: list[HiddenAudit], expected: dict) -> dict[str, np.ndarray]:
    data = LABELS.read_bytes()
    if sha256(data) != expected["audit_labels.json"]:
        raise RuntimeError("hidden labels changed while the attack ran")
    labels = json.loads(data.decode("utf-8"))
    return {a.name: np.array([int(labels[a.name][d]) for d in a.doc_ids], dtype=np.int64) for a in audits}


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
        "score_ua_all": unsupervised(util_f, 0, True),
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
        "member_recall": decisions[y == 1].mean(),
    }
    return {m: float(100.0 * table[m]) for m in wanted}


def mean_rows(rows: list[dict]) -> dict:
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


def read_attack_output(raw: bytes, doc_ids: list[str]):
    data = json.loads(raw.decode("utf-8"))
    listed = data.get("doc_ids")
    if not (isinstance(listed, list) and len(listed) == len(doc_ids)
            and len(set(listed)) == len(listed) and set(listed) == set(doc_ids)):
        raise ValueError("doc_ids must contain each candidate document exactly once")
    where = {d: i for i, d in enumerate(listed)}
    perm = np.array([where[d] for d in doc_ids])
    block = data.get("attacks")
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


class Seal:
    """Make the hidden test's directories (and the dev audit) unreadable while the attack runs."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.modes: dict[Path, int] = {}
        self.effective: bool | None = None

    def __enter__(self) -> "Seal":
        if self.enabled:
            try:
                for path in SEALED_DIRS:
                    if path.exists():
                        self.modes[path] = stat.S_IMODE(path.stat().st_mode)
                        os.chmod(path, 0)
            except BaseException:
                self.restore()
                raise
            # A process with DAC override (root) could still read; record whether the seal holds.
            self.effective = not os.access(LABELS, os.R_OK)
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        for path, mode in list(self.modes.items()):
            try:
                os.chmod(path, mode | stat.S_IRWXU)
            except OSError:
                pass
            self.modes.pop(path, None)


def run_attack(command: str, audit: HiddenAudit, timeout: float) -> bytes:
    """Run the attack on a fresh copy of one audit; the copy is deleted afterwards."""
    scratch = Path(tempfile.mkdtemp(prefix="hidden_eval_", dir=str(ROOT / "tmp")))
    try:
        audit_dir = scratch / "audit"
        for rel, data in audit.files.items():
            path = audit_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        result = scratch / "attack_output.json"
        proc = subprocess.Popen(shlex.split(command) + [str(audit_dir), str(result)], cwd=str(ROOT),
                                start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        finally:  # nothing the attack started may outlive it (and the seal)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
        if returncode != 0:
            raise RuntimeError(f"attack program exited with code {returncode}")
        if not result.exists():
            raise RuntimeError("attack program wrote no output")
        return result.read_bytes()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def worst_row() -> dict:
    return {m: 0.0 for m in METRICS}


def failed_payload(reason: str) -> dict:
    return {"experiments": {EXPERIMENT: {"slice": "test", "results": {SLOT: worst_row()},
                                         "attack_status": {"failure": reason}}}}


def score(command: str, timeout: float, seal_enabled: bool, keep_dir: str | None) -> dict:
    audits, expected = load_hidden(HASHES)
    baselines = {a.name: baseline_attacks(a) for a in audits}
    outputs, status = {}, {}
    with Seal(seal_enabled) as seal:
        for audit in audits:
            try:
                outputs[audit.name] = run_attack(command, audit, timeout)
                status[audit.name] = {"failure": None}
            except Exception as exc:  # recorded; the ranked row then fails as a whole
                status[audit.name] = {"failure": f"{type(exc).__name__}: {exc}"}
            print(f"[hidden] {audit.name}: {status[audit.name]['failure'] or 'attack finished'}", flush=True)
    labels = load_labels(audits, expected)
    if keep_dir:
        Path(keep_dir).mkdir(parents=True, exist_ok=True)
        for name, raw in outputs.items():
            (Path(keep_dir) / f"{name}.json").write_bytes(raw)

    usable = {}
    for audit in audits:
        if audit.name in outputs:
            try:
                usable[audit.name], notes = read_attack_output(outputs[audit.name], audit.doc_ids)
                status[audit.name]["slot"] = notes.get(SLOT, "missing: scored with worst values")
                status[audit.name]["other"] = {k: v for k, v in notes.items() if k != SLOT}
            except Exception as exc:
                status[audit.name]["failure"] = f"invalid output: {type(exc).__name__}: {exc}"
        if audit.name not in usable:
            usable[audit.name] = {}
    ranked_ok = all(SLOT in usable[a.name] for a in audits)
    extra = sorted(set.intersection(*(set(usable[a.name]) for a in audits)) - {SLOT}) if audits else []

    per_audit, low_rows = {}, {}
    for audit in audits:
        y, attacks = labels[audit.name], usable[audit.name]
        low = audit.n_questions <= LOW_EVIDENCE
        listed = {r: attacks[r] for r in [SLOT] + extra if r in attacks}
        listed.update({f"baseline:{b}": v for b, v in baselines[audit.name].items()})
        per_audit[audit.name] = {
            "documents": len(y),
            "members": int(y.sum()),
            "results": {SLOT: score_rows(y, *attacks[SLOT], METRICS) if SLOT in attacks else worst_row()},
            "baselines": {b: score_rows(y, d, s, METRICS) for b, (d, s) in baselines[audit.name].items()},
            "other_attacks": {r: score_rows(y, *attacks[r], METRICS) for r in extra},
        }
        low_rows[audit.name] = {r: score_rows(y[low], d[low], s[low], LOW_METRICS) for r, (d, s) in listed.items()}

    names = [a.name for a in audits]
    results = {SLOT: mean_rows([per_audit[n]["results"][SLOT] for n in names]) if ranked_ok else worst_row()}
    seal_note = {"enabled": seal_enabled, "labels_unreadable_during_attack": seal.effective}
    return {"experiments": {EXPERIMENT: {
        "slice": "test",
        "audits": len(audits),
        "documents": int(sum(len(a.doc_ids) for a in audits)),
        "members": int(sum(labels[n].sum() for n in names)),
        "aggregation": "mean over the hidden audits of each audit's metric; worst values if any audit fails",
        "results": results,
        "baselines": {b: mean_rows([per_audit[n]["baselines"][b] for n in names]) for b in baselines[names[0]]},
        "other_attacks": {r: mean_rows([per_audit[n]["other_attacks"][r] for n in names]) for r in extra},
        "per_audit": per_audit,
        "diagnostics": {f"documents_with_at_most_{LOW_EVIDENCE}_questions": {
            "documents": int(sum((a.n_questions <= LOW_EVIDENCE).sum() for a in audits)),
            "rows": {r: mean_rows([low_rows[n][r] for n in names]) for r in low_rows[names[0]]
                     if all(r in low_rows[n] for n in names)},
        }},
        "attack_status": {"failure": None if ranked_ok else "membership_attack failed or invalid on at least one audit",
                          "audits": status, "seal": seal_note},
    }}}


def _interrupt(signum, _frame):
    raise SystemExit(f"evaluator interrupted by signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-cmd", default="bash scripts/attack.sh")
    parser.add_argument("--scores", default=str(ROOT / "scoring" / "scores_test.json"))
    parser.add_argument("--keep-output", default=None, help="directory for the raw per-audit attack outputs")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per audit")
    parser.add_argument("--no-seal", action="store_true", help="diagnostics only: leave eval/test readable")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)  # unwinds through the seal, which restores access
    for path in SEALED_DIRS:  # recover from a run that was killed while sealed
        if path.exists() and not os.access(path, os.R_OK | os.X_OK):
            os.chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_IRWXU)
    cores = int(os.environ.get("DOCMIA_THREADS", "1") or 1)
    torch.set_num_threads(max(1, cores))
    (ROOT / "tmp").mkdir(exist_ok=True)
    out = Path(args.scores)
    out.unlink(missing_ok=True)
    try:
        payload = score(args.attack_cmd, args.timeout, not args.no_seal, args.keep_output)
    except Exception as exc:  # evaluator-side failure still yields worst-value rows
        traceback.print_exc()
        payload = failed_payload(f"{type(exc).__name__}: {exc}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({e: v["results"] for e, v in payload["experiments"].items()}, indent=1))


if __name__ == "__main__":
    main()
