"""Architecture of the audited DocVQA model (a problem asset, not a method).

A compact OCR-free document QA model: a convolutional page encoder, a question
embedding, a fusion MLP, and a final answer projection layer (the analogue of
the vocabulary projection layer of Donut / VT5 / Pix2Struct).  Answers are
predicted over a closed answer bank.
"""

from __future__ import annotations

import torch
from torch import nn

from data.docgen import IMAGE_H, IMAGE_W


class TinyDocVQA(nn.Module):
    def __init__(self, num_questions: int, num_answers: int, hidden: int = 128) -> None:
        super().__init__()
        self.page_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        page_dim = 32 * (IMAGE_H // 8) * (IMAGE_W // 8)
        self.question_embedding = nn.Embedding(num_questions, 32)
        self.fusion = nn.Sequential(
            nn.Linear(page_dim + 32, 256), nn.ReLU(),
            nn.Linear(256, hidden), nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden, num_answers)

    def encode_page(self, page: torch.Tensor) -> torch.Tensor:
        return self.page_encoder(page)

    def hidden_state(self, page_features: torch.Tensor, question_idx: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([page_features, self.question_embedding(question_idx)], dim=1))

    def forward(self, page: torch.Tensor, question_idx: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_state(self.encode_page(page), question_idx))


def save_checkpoint(model: TinyDocVQA, path, extra: dict | None = None) -> None:
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "num_questions": model.question_embedding.num_embeddings,
        "num_answers": model.output_layer.out_features,
        "hidden": model.output_layer.in_features,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> TinyDocVQA:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = TinyDocVQA(payload["num_questions"], payload["num_answers"], payload.get("hidden", 128))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model
