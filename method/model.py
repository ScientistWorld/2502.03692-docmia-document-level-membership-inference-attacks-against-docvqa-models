"""Small DocVQA target model for the scaled DocMIA reproduction."""

from __future__ import annotations

import torch
from torch import nn


class TinyDocVQAModel(nn.Module):
    """A compact multimodal classifier over answer tokens.

    It is intentionally small, but preserves the target-model interface needed
    by DocMIA: answer likelihood is conditioned on a document image and a
    natural-language question represented by its field id.
    """

    def __init__(self, num_questions: int, num_answers: int) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 12, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
        )
        self.question_embedding = nn.Embedding(num_questions, 16)
        self.hidden = nn.Sequential(
            nn.Linear(24 * 4 * 6 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(96, num_answers)

    def forward(self, image: torch.Tensor, question_idx: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        question_features = self.question_embedding(question_idx)
        features = torch.cat([image_features, question_features], dim=1)
        return self.output_layer(self.hidden(features))
