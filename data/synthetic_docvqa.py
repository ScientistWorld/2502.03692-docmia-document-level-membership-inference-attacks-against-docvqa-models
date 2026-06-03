"""Synthetic document VQA benchmark used for a scaled DocMIA reproduction.

The paper evaluates on DocVQA/PFL-DocVQA. Those benchmarks and public
checkpoints are large enough that this gym uses a small procedural substitute
with the same document-level structure: each document image contains multiple
key-value facts, and every document has several question-answer pairs.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw


FIELDS = ("name", "city", "code", "amount")
NAMES = ("ada", "ben", "cora", "dina", "eli", "fay", "gabe", "hana")
CITIES = ("oslo", "lima", "doha", "rome", "seoul", "quito", "paris", "miami")
CODES = ("a13", "b27", "c41", "d55", "e69", "f83", "g97", "h02")
AMOUNTS = ("12", "24", "36", "48", "60", "72", "84", "96")
ANSWER_BANK = sorted(set(NAMES + CITIES + CODES + AMOUNTS))


@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    field: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    split: str
    image_path: str
    facts: dict[str, str]
    qas: list[QA]


def _facts(rng: random.Random) -> dict[str, str]:
    return {
        "name": rng.choice(NAMES),
        "city": rng.choice(CITIES),
        "code": rng.choice(CODES),
        "amount": rng.choice(AMOUNTS),
    }


def _qas(facts: dict[str, str]) -> list[QA]:
    prompts = {
        "name": "What is the customer name?",
        "city": "What city is listed?",
        "code": "What is the document code?",
        "amount": "What is the amount?",
    }
    return [QA(prompts[field], answer, field) for field, answer in facts.items()]


def _draw_document(path: Path, facts: dict[str, str], rng: random.Random) -> None:
    image = Image.new("L", (160, 96), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 155, 91), outline=0, width=1)
    draw.text((12, 10), "CLAIM FORM", fill=0)
    y = 28
    for field in FIELDS:
        jitter = rng.randint(-1, 1)
        draw.text((14 + jitter, y), f"{field.upper()}: {facts[field]}", fill=0)
        y += 14
    image.save(path)


def build_dataset(
    output_dir: str | Path = "data/synthetic_docvqa",
    train_docs: int = 96,
    nonmember_docs: int = 96,
    seed: int = 1337,
) -> Path:
    """Create the tracked synthetic dataset if it does not already exist."""

    output = Path(output_dir)
    image_dir = output / "images"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    documents: list[Document] = []
    for split, count in (("member", train_docs), ("nonmember", nonmember_docs)):
        for idx in range(count):
            facts = _facts(rng)
            doc_id = f"{split}_{idx:04d}"
            rel_path = f"images/{doc_id}.png"
            _draw_document(output / rel_path, facts, rng)
            documents.append(Document(doc_id, split, rel_path, facts, _qas(facts)))

    payload = {
        "answer_bank": ANSWER_BANK,
        "fields": list(FIELDS),
        "documents": [
            {
                **asdict(doc),
                "qas": [asdict(qa) for qa in doc.qas],
            }
            for doc in documents
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    print(build_dataset())
