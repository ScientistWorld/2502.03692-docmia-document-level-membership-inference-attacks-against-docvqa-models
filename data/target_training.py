"""Public training recipe for the audited DocVQA model.

Mirrors the paper's target-model protocol (Appendix D.1) at small scale:

1. a *public pre-trained checkpoint* is produced once by training the model on a
   public corpus of generated documents (the analogue of Donut-base / VT5 /
   Pix2Struct pre-training), and
2. each audited target is that checkpoint fine-tuned on a private document set
   for a fixed schedule (8 epochs).  The paper early-stops on validation
   performance; at this scale the validation curve is flat and noisy after a
   couple of epochs, so early stopping picked anywhere between epoch 2 and 8 for
   different seeds (train/test gaps of 15 vs 36 points).  A fixed schedule keeps
   the dev and hidden audits at comparable difficulty; 8 epochs gives a
   member/non-member QA gap close to the paper's DocVQA targets (~36-40 points).

An audit = target checkpoint + a balanced candidate set of member documents
(sampled from the private fine-tuning set) and non-member documents (fresh
documents from the same distribution).  The dev audit is built from public
seeds by ``data/build_public_assets.py``; the hidden test audit is built by the
same recipe from a secret seed held only in ``eval/test``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from data.docgen import ANSWER_BANK, FIELDS, render_many, sample_specs, write_candidates
from data.target_model import TinyDocVQA, save_checkpoint

FIELD_INDEX = {field: i for i, field in enumerate(FIELDS)}
ANSWER_INDEX = {answer: i for i, answer in enumerate(ANSWER_BANK)}

PRETRAIN_RECIPE = {
    "seed": "docmia-public-pretraining-corpus-v1",
    "documents": 3000,
    "epochs": 25,
    "lr": 1e-3,
    "batch_docs": 32,
}

FINETUNE_RECIPE = {
    "private_documents": 600,
    "validation_documents": 200,
    "candidate_members": 600,
    "candidate_nonmembers": 600,
    "epochs": 8,
    "patience": None,
    "early_stopping": False,
    "lr": 1e-3,
    "batch_docs": 32,
    "weight_decay": 1e-4,
}


def int_seed(seed, salt: str = "") -> int:
    digest = hashlib.sha256(f"{seed}|{salt}".encode()).hexdigest()
    return int(digest[:12], 16)


class DocSet:
    """Pages (uint8) plus their question/answer pairs grouped by document."""

    def __init__(self, pages: np.ndarray, specs: list[dict]) -> None:
        self.pages = torch.from_numpy(np.ascontiguousarray(pages))
        q, a, d = [], [], []
        self.doc_pairs: list[tuple[int, int]] = []
        for i, spec in enumerate(specs):
            start = len(q)
            for field in spec["asked"]:
                q.append(FIELD_INDEX[field])
                a.append(ANSWER_INDEX[spec["facts"][field]])
                d.append(i)
            self.doc_pairs.append((start, len(q)))
        self.q = torch.tensor(q, dtype=torch.long)
        self.a = torch.tensor(a, dtype=torch.long)
        self.d = torch.tensor(d, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.pages.shape[0])


def page_tensor(pages_uint8: torch.Tensor) -> torch.Tensor:
    return pages_uint8.float().unsqueeze(1) / 255.0


def _batch_pairs(ds: DocSet, doc_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    pair_idx, local = [], []
    for j, i in enumerate(doc_idx):
        start, end = ds.doc_pairs[i]
        pair_idx.extend(range(start, end))
        local.extend([j] * (end - start))
    return torch.tensor(pair_idx, dtype=torch.long), torch.tensor(local, dtype=torch.long)


def batch_logits(model: TinyDocVQA, ds: DocSet, doc_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    pair_idx, local = _batch_pairs(ds, doc_idx)
    feats = model.encode_page(page_tensor(ds.pages[doc_idx]))
    logits = model.output_layer(model.hidden_state(feats[local], ds.q[pair_idx]))
    return logits, ds.a[pair_idx]


@torch.no_grad()
def evaluate(model: TinyDocVQA, ds: DocSet, batch_docs: int = 128) -> dict:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for start in range(0, len(ds), batch_docs):
        idx = list(range(start, min(len(ds), start + batch_docs)))
        logits, target = batch_logits(model, ds, idx)
        correct += int((logits.argmax(1) == target).sum())
        total += int(target.numel())
        loss_sum += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def train(model: TinyDocVQA, train_ds: DocSet, epochs: int, lr: float, seed: int,
          batch_docs: int = 32, weight_decay: float = 1e-4, val_ds: DocSet | None = None,
          patience: int | None = None, log: Callable[[str], None] = print) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_acc, stale = None, -1.0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_ds), generator=gen).tolist()
        for start in range(0, len(order), batch_docs):
            logits, target = batch_logits(model, train_ds, order[start:start + batch_docs])
            loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        record = {"epoch": epoch + 1, "train": evaluate(model, train_ds)}
        if val_ds is not None:
            record["val"] = evaluate(model, val_ds)
            if record["val"]["accuracy"] > best_acc:
                best_acc = record["val"]["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                record["selected"] = True
            else:
                stale += 1
        history.append(record)
        log(json.dumps(record))
        if patience is not None and val_ds is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def build_pretrained(out_path: str | Path, recipe: dict | None = None,
                     log: Callable[[str], None] = print) -> dict:
    """Train the public pre-trained reader on a public generated corpus."""
    recipe = {**PRETRAIN_RECIPE, **(recipe or {})}
    specs = sample_specs(recipe["seed"], recipe["documents"], all_questions=True)
    ds = DocSet(render_many(specs), specs)
    torch.manual_seed(int_seed(recipe["seed"], "init"))
    model = TinyDocVQA(len(FIELDS), len(ANSWER_BANK))
    history = train(model, ds, recipe["epochs"], recipe["lr"], int_seed(recipe["seed"], "train"),
                    batch_docs=recipe["batch_docs"], log=log)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out_path)
    return {"recipe": recipe, "history": history}


def build_audit(seed, out_dir: str | Path, pretrained_path: str | Path | None,
                recipe: dict | None = None, log: Callable[[str], None] = print) -> dict:
    """Fine-tune a target on a private set drawn from ``seed`` and write its audit.

    Writes ``out_dir/target.pt`` and the unlabeled candidate set
    ``out_dir/candidates``; returns labels and target statistics (the caller
    decides where these may be stored).
    """
    recipe = {**FINETUNE_RECIPE, **(recipe or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    private = sample_specs(f"{seed}|private", recipe["private_documents"])
    validation = sample_specs(f"{seed}|validation", recipe["validation_documents"])
    nonmembers = sample_specs(f"{seed}|nonmember", recipe["candidate_nonmembers"])
    rng = random.Random(f"{seed}|selection")
    member_idx = sorted(rng.sample(range(len(private)), recipe["candidate_members"]))

    private_pages = render_many(private)
    train_ds = DocSet(private_pages, private)
    val_ds = DocSet(render_many(validation), validation)

    if pretrained_path is not None:
        payload = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        model = TinyDocVQA(payload["num_questions"], payload["num_answers"], payload.get("hidden", 128))
        model.load_state_dict(payload["model_state"])
    else:
        torch.manual_seed(int_seed(seed, "init"))
        model = TinyDocVQA(len(FIELDS), len(ANSWER_BANK))
    history = train(model, train_ds, recipe["epochs"], recipe["lr"], int_seed(seed, "train"),
                    batch_docs=recipe["batch_docs"], weight_decay=recipe["weight_decay"],
                    val_ds=val_ds if recipe["early_stopping"] else None,
                    patience=recipe["patience"], log=log)

    candidates = [(private[i], private_pages[i], 1) for i in member_idx]
    nonmember_pages = render_many(nonmembers)
    candidates += [(spec, page, 0) for spec, page in zip(nonmembers, nonmember_pages)]
    rng.shuffle(candidates)
    doc_ids = [f"doc_{i:04d}" for i in range(len(candidates))]
    write_candidates([c[0] for c in candidates], doc_ids, out / "candidates",
                     pages=np.stack([c[1] for c in candidates]))
    save_checkpoint(model, out / "target.pt")

    member_specs = [c[0] for c in candidates if c[2] == 1]
    nonmember_specs = [c[0] for c in candidates if c[2] == 0]
    member_ds = DocSet(np.stack([c[1] for c in candidates if c[2] == 1]), member_specs)
    nonmember_ds = DocSet(np.stack([c[1] for c in candidates if c[2] == 0]), nonmember_specs)
    stats = {
        "recipe": recipe,
        "selected_epoch": max((h["epoch"] for h in history if h.get("selected")), default=len(history)),
        "epochs_run": len(history),
        "member_candidates": evaluate(model, member_ds),
        "nonmember_candidates": evaluate(model, nonmember_ds),
        "validation": evaluate(model, val_ds),
    }
    stats["train_test_gap_points"] = 100.0 * (
        stats["member_candidates"]["accuracy"] - stats["nonmember_candidates"]["accuracy"]
    )
    labels = {doc_id: int(c[2]) for doc_id, c in zip(doc_ids, candidates)}
    return {"labels": labels, "stats": stats, "history": history}
