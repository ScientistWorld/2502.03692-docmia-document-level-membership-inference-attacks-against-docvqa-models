"""Data loading utilities shared by target training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset


QUESTION_TO_FIELD = {
    "What is the customer name?": "name",
    "What city is listed?": "city",
    "What is the document code?": "code",
    "What is the amount?": "amount",
}


class SyntheticDocVQADataset(Dataset):
    def __init__(self, root: str | Path, split: str = "member") -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.answer_bank = manifest["answer_bank"]
        self.answer_to_idx = {answer: idx for idx, answer in enumerate(self.answer_bank)}
        self.field_to_idx = {field: idx for idx, field in enumerate(manifest["fields"])}
        self.documents = [doc for doc in manifest["documents"] if doc["split"] == split]
        self.examples = []
        for doc in self.documents:
            for qa in doc["qas"]:
                self.examples.append((doc, qa))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        doc, qa = self.examples[idx]
        image = load_image(self.root / doc["image_path"])
        return {
            "image": image,
            "question_idx": torch.tensor(self.field_to_idx[qa["field"]], dtype=torch.long),
            "answer_idx": torch.tensor(self.answer_to_idx[qa["answer"]], dtype=torch.long),
            "doc_id": doc["doc_id"],
        }


def load_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


def iter_documents(root: str | Path, splits: Iterable[str] = ("member", "nonmember")) -> list[dict]:
    manifest = load_manifest(root)
    wanted = set(splits)
    return [doc for doc in manifest["documents"] if doc["split"] in wanted]


def load_image(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((160, 96))
    values = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    values = values.float().view(96, 160) / 255.0
    return values.unsqueeze(0)
