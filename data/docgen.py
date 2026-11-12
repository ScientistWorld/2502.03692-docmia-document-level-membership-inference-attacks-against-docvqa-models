"""Procedural DocVQA-style form documents (public generator).

The paper audits DocVQA / PFL-DocVQA models.  This gym uses a scaled,
procedural substitute with the same document-level structure: every document
image holds several key/value facts, every document carries a variable number
of question/answer pairs (1-5 here, up to 10 in the paper), and each scanned
page has its own noise, ink, blur and placement so that a model can memorise
individual pages.

Everything here is public.  A seed fully determines a corpus, which is why the
hidden test audit is generated from a secret seed that only ``eval/test`` holds.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_H, IMAGE_W = 80, 96
FIELDS = ("name", "city", "code", "date", "amount")
FIELD_LABELS = {"name": "NAME", "city": "CITY", "code": "CODE", "date": "DATE", "amount": "TOTAL"}
QUESTIONS = {
    "name": "What is the customer name?",
    "city": "Which city is listed?",
    "code": "What is the document code?",
    "date": "What is the date?",
    "amount": "What is the total amount?",
}
VALUES = {
    "name": ("ada", "ben", "cora", "dina", "eli", "fay", "gabe", "hana",
             "ivo", "jun", "kai", "lea", "max", "nia", "otto", "pia"),
    "city": ("oslo", "lima", "doha", "rome", "kyiv", "quito", "paris", "miami",
             "bern", "accra", "turin", "delhi", "porto", "osaka", "split", "nice"),
    "code": ("a13", "b27", "c41", "d55", "e69", "f83", "g97", "h02",
             "a31", "b72", "c14", "d50", "e96", "f38", "g79", "h20"),
    "date": ("01/03", "02/14", "03/25", "04/06", "05/17", "06/28", "07/09", "08/10",
             "09/21", "10/02", "11/13", "12/24", "01/30", "02/08", "03/19", "04/27"),
    "amount": ("12.40", "24.80", "36.10", "48.60", "60.20", "72.90", "84.30", "96.50",
               "13.70", "25.10", "37.40", "49.80", "61.20", "73.60", "85.90", "97.30"),
}
ANSWER_BANK = sorted({value for values in VALUES.values() for value in values})
VALUE_X = 50

_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default()
    return _FONT


def sample_spec(rng: random.Random, min_questions: int = 1, max_questions: int = 5,
                all_questions: bool = False) -> dict:
    """Sample one document: its facts, which fields were annotated, and its scan."""
    facts = {field: rng.choice(VALUES[field]) for field in FIELDS}
    k = len(FIELDS) if all_questions else rng.randint(min_questions, max_questions)
    asked = sorted(rng.sample(range(len(FIELDS)), k))
    render = {
        "dx": rng.randint(-3, 3),
        "dy": rng.randint(-3, 3),
        "line_jitter": [[rng.randint(-2, 2), rng.randint(-1, 1)] for _ in FIELDS],
        "ink": rng.uniform(0.0, 0.45),
        "paper": rng.uniform(0.80, 1.0),
        "noise": rng.uniform(0.05, 0.25),
        "blur": rng.uniform(0.0, 0.7),
        "noise_seed": rng.getrandbits(32),
    }
    return {"facts": facts, "asked": [FIELDS[i] for i in asked], "render": render}


def sample_specs(seed, count: int, **kwargs) -> list[dict]:
    rng = random.Random(seed)
    return [sample_spec(rng, **kwargs) for _ in range(count)]


def render(spec: dict) -> np.ndarray:
    """Render a document spec to a uint8 grayscale page (H, W)."""
    r = spec["render"]
    image = Image.new("L", (IMAGE_W, IMAGE_H), 255)
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((4 + r["dx"], 1 + r["dy"]), "FORM", fill=0, font=font)
    for i, field in enumerate(FIELDS):
        jx, jy = r["line_jitter"][i]
        y = 15 + 12 * i + r["dy"] + jy
        draw.text((4 + r["dx"], y), FIELD_LABELS[field] + ":", fill=0, font=font)
        draw.text((VALUE_X + r["dx"] + jx, y), spec["facts"][field], fill=0, font=font)
    if r["blur"] > 0.05:
        image = image.filter(ImageFilter.GaussianBlur(r["blur"]))
    page = np.asarray(image, dtype=np.float32) / 255.0
    page = r["ink"] + (r["paper"] - r["ink"]) * page
    noise_rng = np.random.default_rng(r["noise_seed"])
    page = page + noise_rng.normal(0.0, r["noise"], size=page.shape)
    speckle = noise_rng.random(page.shape) < 0.01
    page[speckle] = noise_rng.random(int(speckle.sum()))
    page = np.clip(page, 0.0, 1.0)
    return (page * 255.0 + 0.5).astype(np.uint8)


def render_many(specs: list[dict]) -> np.ndarray:
    return np.stack([render(spec) for spec in specs]) if specs else np.zeros((0, IMAGE_H, IMAGE_W), np.uint8)


def qa_pairs(spec: dict) -> list[dict]:
    return [
        {"question": QUESTIONS[field], "field": field, "answer": spec["facts"][field]}
        for field in spec["asked"]
    ]


def write_candidates(specs: list[dict], doc_ids: list[str], out_dir: str | Path,
                     pages: np.ndarray | None = None) -> Path:
    """Write an unlabeled candidate set: page images plus their QA pairs."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if pages is None:
        pages = render_many(specs)
    documents = []
    for spec, doc_id, page in zip(specs, doc_ids, pages):
        rel = f"images/{doc_id}.png"
        Image.fromarray(np.ascontiguousarray(page, dtype=np.uint8)).save(out / rel)
        documents.append({"doc_id": doc_id, "image": rel, "qas": qa_pairs(spec)})
    manifest = {
        "fields": list(FIELDS),
        "answer_bank": ANSWER_BANK,
        "image_size": [IMAGE_H, IMAGE_W],
        "documents": documents,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path
