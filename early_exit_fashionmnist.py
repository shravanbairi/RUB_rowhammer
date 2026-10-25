"""Train and evaluate a FashionMNIST CNN with confidence-based early exits.

This is deliberately a separate stage from ``train_fashionmnist.py``.  It
creates a checkpoint that can be compared directly with the baseline and later
used by ``fault_injection.py``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


class EarlyExitCNN(nn.Module):
    """CNN with classifiers after each convolutional block and at the output."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.exit1 = nn.Linear(16 * 14 * 14, num_classes)
        self.exit2 = nn.Linear(32 * 7 * 7, num_classes)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.exit3 = nn.Linear(128, num_classes)
            
    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.pool(F.relu(self.conv1(x)))
        exit1 = self.exit1(torch.flatten(x, 1))
        x = self.pool(F.relu(self.conv2(x)))
        flat = torch.flatten(x, 1)
        exit2 = self.exit2(flat)
        x = F.relu(self.fc1(flat))
        exit3 = self.exit3(x)
        return exit1, exit2, exit3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default PyTorch interface: final-exit logits."""
        return self.forward_all(x)[-1]
    




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(test_set, batch_size=256, shuffle=False, num_workers=workers),
    )


@torch.no_grad()
def evaluate(model: EarlyExitCNN, loader: DataLoader, threshold: float) -> dict[str, float]:
    """Route each item to its first classifier over ``threshold`` confidence."""
    model.eval()
    correct = total = 0
    exit_counts = torch.zeros(3, dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model.forward_all(images)
        probabilities = [F.softmax(logits, dim=1) for logits in outputs]
        predictions = probabilities[-1].argmax(dim=1)
        unresolved = torch.ones(labels.size(0), dtype=torch.bool, device=DEVICE)
        for exit_index, probs in enumerate(probabilities[:-1]):
            use_exit = unresolved & (probs.max(dim=1).values >= threshold)
            predictions[use_exit] = probs[use_exit].argmax(dim=1)
            exit_counts[exit_index] += use_exit.sum().cpu()
            unresolved &= ~use_exit
        exit_counts[-1] += unresolved.sum().cpu()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return {
        "accuracy": correct / total,
        "exit1_fraction": exit_counts[0].item() / total,
        "exit2_fraction": exit_counts[1].item() / total,
        "exit3_fraction": exit_counts[2].item() / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("fashionmnist_early_exit.pt"))
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be in (0, 1]")

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = EarlyExitCNN().to(DEVICE)
    # Display the weights of conv2
    #print(model.conv2.weight)

    # Display the shape of the weight tensor
    #print(model.conv2.weight.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Equal loss weights make the early heads useful without obscuring the
    # final classifier, which remains the golden fall-back prediction.
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            losses = [criterion(logits, labels) for logits in model.forward_all(images)]
            
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
        metrics = evaluate(model, test_loader, args.threshold)
        print(f"epoch {epoch:2d}/{args.epochs} loss={loss_sum / len(train_loader.dataset):.4f} "
              f"accuracy={metrics['accuracy']:.4f} exits="
              f"{metrics['exit1_fraction']:.1%}/{metrics['exit2_fraction']:.1%}/{metrics['exit3_fraction']:.1%}")

    metrics = evaluate(model, test_loader, args.threshold)
    torch.save({"model_type": "EarlyExitCNN", "model_state_dict": model.state_dict(),
                "threshold": args.threshold, "seed": args.seed, "metrics": metrics}, args.checkpoint)
    print(f"saved {args.checkpoint} on {DEVICE}: {metrics}")
    def measure_exit_accuracy():
        model.eval()

        correct1 = 0
        correct2 = 0
        correct3 = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Obtain logits from all three exits
                exit1, exit2, exit3 = model.forward_all(images)

                # Select the predicted class from each exit
                pred1 = exit1.argmax(dim=1)
                pred2 = exit2.argmax(dim=1)
                pred3 = exit3.argmax(dim=1)

                # Count correct predictions
                correct1 += (pred1 == labels).sum().item()
                correct2 += (pred2 == labels).sum().item()
                correct3 += (pred3 == labels).sum().item()

                total += labels.size(0)

        accuracy1 = 100 * correct1 / total
        accuracy2 = 100 * correct2 / total
        accuracy3 = 100 * correct3 / total

        return accuracy1, accuracy2, accuracy3

    # STEP 3: MEASURE ACCURACY BEFORE ZEROING WEIGHTS
    before = measure_exit_accuracy()

    print("\n===== BEFORE ZEROING CONV2 =====")
    print(f"Exit 1 accuracy: {before[0]:.2f}%")
    print(f"Exit 2 accuracy: {before[1]:.2f}%")
    print(f"Exit 3 accuracy: {before[2]:.2f}%")

    # STEP 4: SAVE ORIGINAL TRAINED MODEL
    torch.save(
        {
            "model_type": "EarlyExitCNN",
            "model_state_dict": model.state_dict(),
            "threshold": args.threshold,
            "seed": args.seed,
        },
        args.checkpoint
    )

    print(f"\nOriginal model saved to {args.checkpoint}")

    # STEP 5: ZERO ALL WEIGHTS IN CONV2
    with torch.no_grad():
        model.conv2.weight.zero_()

    print("\nAll conv2 weights have been zeroed.")

    # STEP 6: MEASURE ACCURACY AGAIN
    after = measure_exit_accuracy()

    print("\n===== AFTER ZEROING CONV2 =====")
    print(f"Exit 1 accuracy: {after[0]:.2f}%")
    print(f"Exit 2 accuracy: {after[1]:.2f}%")
    print(f"Exit 3 accuracy: {after[2]:.2f}%")

    # STEP 7: COMPARE THE RESULTS
    print("\n===== ACCURACY COMPARISON =====")
    print(f"{'Exit':<12}{'Before':>12}{'After':>12}{'Change':>12}")

    for i, name in enumerate(["Exit 1", "Exit 2", "Exit 3"]):
    
        print(
            f"{name:<12}"
            f"{before[i]:>11.2f}%"
            f"{after[i]:>11.2f}%"
            
        )

    print("\nExperiment complete.")

if __name__ == "__main__":
    main()
