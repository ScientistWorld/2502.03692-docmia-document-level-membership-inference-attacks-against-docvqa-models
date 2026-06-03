"""Train the target DocVQA model that DocMIA attacks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.docvqa_loader import SyntheticDocVQADataset
from data.synthetic_docvqa import build_dataset
from method.model import TinyDocVQAModel


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            question_idx = batch["question_idx"].to(device)
            answer_idx = batch["answer_idx"].to(device)
            pred = model(image, question_idx).argmax(dim=1)
            correct += int((pred == answer_idx).sum().item())
            total += int(answer_idx.numel())
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_docvqa")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_docvqa_target.pt")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    build_dataset(args.data_dir)

    member = SyntheticDocVQADataset(args.data_dir, "member")
    nonmember = SyntheticDocVQADataset(args.data_dir, "nonmember")
    train_loader = DataLoader(member, batch_size=args.batch_size, shuffle=True)
    member_eval = DataLoader(member, batch_size=args.batch_size)
    nonmember_eval = DataLoader(nonmember, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDocVQAModel(len(member.field_to_idx), len(member.answer_bank)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            question_idx = batch["question_idx"].to(device)
            answer_idx = batch["answer_idx"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image, question_idx), answer_idx)
            loss.backward()
            optimizer.step()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "member_docvqa_accuracy": evaluate(model, member_eval, device),
        "nonmember_docvqa_accuracy": evaluate(model, nonmember_eval, device),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_questions": len(member.field_to_idx),
            "num_answers": len(member.answer_bank),
            "answer_bank": member.answer_bank,
            "metrics": metrics,
        },
        args.checkpoint,
    )
    Path("checkpoints/target_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
