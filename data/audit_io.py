"""Load an audit directory: ``target.pt`` plus an unlabeled ``candidates/`` set.

An audit directory is what an attack receives, both on the visible dev audit
(``data/dev_audit``) and on the hidden test audit.  It never contains labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.target_model import TinyDocVQA, load_checkpoint


@dataclass
class Candidates:
    doc_ids: list[str]
    pages: torch.Tensor          # float32 [N, 1, H, W] in [0, 1]
    pair_doc: torch.Tensor       # long [P] document index of each QA pair
    pair_question: torch.Tensor  # long [P] question (field) index
    pair_answer: torch.Tensor    # long [P] answer index in the answer bank
    fields: list[str]
    answer_bank: list[str]
    manifest: dict

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def questions_per_doc(self) -> np.ndarray:
        return np.bincount(self.pair_doc.numpy(), minlength=self.num_docs)


def load_candidates(audit_dir: str | Path) -> Candidates:
    root = Path(audit_dir) / "candidates"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    field_index = {f: i for i, f in enumerate(manifest["fields"])}
    answer_index = {a: i for i, a in enumerate(manifest["answer_bank"])}
    pages, pair_doc, pair_q, pair_a, doc_ids = [], [], [], [], []
    for i, doc in enumerate(manifest["documents"]):
        doc_ids.append(doc["doc_id"])
        page = np.asarray(Image.open(root / doc["image"]).convert("L"), dtype=np.float32) / 255.0
        pages.append(page)
        for qa in doc["qas"]:
            pair_doc.append(i)
            pair_q.append(field_index[qa["field"]])
            pair_a.append(answer_index[qa["answer"]])
    return Candidates(
        doc_ids=doc_ids,
        pages=torch.from_numpy(np.stack(pages)).unsqueeze(1),
        pair_doc=torch.tensor(pair_doc, dtype=torch.long),
        pair_question=torch.tensor(pair_q, dtype=torch.long),
        pair_answer=torch.tensor(pair_a, dtype=torch.long),
        fields=list(manifest["fields"]),
        answer_bank=list(manifest["answer_bank"]),
        manifest=manifest,
    )


def load_target(audit_dir: str | Path) -> TinyDocVQA:
    model = load_checkpoint(Path(audit_dir) / "target.pt")
    for param in model.parameters():
        param.requires_grad_(False)
    return model
