"""
Train MLP, CNN, and Transformer on MNIST dataset, split into 60k train and 10k test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import ShallowMLP, CNN, TransformerEncoder


def get_dataloaders(batch_size: int, data_dir: Path) -> tuple[DataLoader, DataLoader]:
    """Get data loaders for training and testing."""
    tf = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir)
    criterion = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    models: list[tuple[str, type[nn.Module]]] = [
        ("mlp", ShallowMLP),
        ("cnn", CNN),
        ("transformer", TransformerEncoder),
    ]

    all_results = []
    for name, ModelCls in models:
        print(f"--- {name} ---")
        model = ModelCls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        history: list[dict] = []
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            acc = evaluate(model, test_loader, device)
            history.append({"epoch": epoch, "train_loss": loss, "test_accuracy_pct": acc})
            print(f"  epoch {epoch}/{args.epochs}  loss={loss:.4f}  test_acc={acc:.2f}%")

        out = {
            "model": name,
            "epochs": args.epochs,
            "device": str(device),
            "final_test_accuracy_pct": history[-1]["test_accuracy_pct"],
            "history": history,
        }
        all_results.append(out)

    json_path = args.out_dir / f"all_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  saved all results to {json_path}")


if __name__ == "__main__":
    main()
