"""Train and evaluate a FashionMNIST CNN with early exits and Conv2 scaling.

Sections:
1. Imports and configuration
2. Model
3. Reproducibility and data
4. Evaluation
5. Controlled Conv2 weight-scaling experiment
6. Main training workflow
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune



# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
RESULTS_DIR = Path("controlled_zeroing_results")


# ---------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------

class EarlyExitCNN(nn.Module):
    """CNN with classifiers after Conv1, Conv2, and the final layer."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.exit1 = nn.Linear(16 * 14 * 14, num_classes)
        self.exit2 = nn.Linear(32 * 7 * 7, num_classes)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.exit3 = nn.Linear(128, num_classes)

    def forward_all(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return logits from Exit 1, Exit 2, and Exit 3."""
        x = self.pool(F.relu(self.conv1(x)))
        exit1_logits = self.exit1(torch.flatten(x, 1))

        x = self.conv2(x)

        # Inspect convolution output here

        x = self.pool(F.relu(x))
        features = torch.flatten(x, 1)
        exit2_logits = self.exit2(features)

        x = F.relu(self.fc1(features))
        exit3_logits = self.exit3(x)

        return exit1_logits, exit2_logits, exit3_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default PyTorch interface returns the final-exit logits."""
        return self.forward_all(x)[-1]


# ---------------------------------------------------------------------
# 3. Reproducibility and data
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders(
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Create the FashionMNIST training and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_set = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: EarlyExitCNN,
    loader: DataLoader,
    threshold: float,
) -> dict[str, float]:
    """Evaluate confidence-based routing through the early exits."""
    model.eval()
    correct = 0
    total = 0
    exit_counts = torch.zeros(3, dtype=torch.long)

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model.forward_all(images)
        probabilities = [F.softmax(logits, dim=1) for logits in outputs]

        predictions = probabilities[-1].argmax(dim=1)
        unresolved = torch.ones(
            labels.size(0), dtype=torch.bool, device=DEVICE
        )

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


@torch.no_grad()
def measure_exit_accuracies(
    model: EarlyExitCNN,
    loader: DataLoader,
) -> tuple[float, float, float]:
    """Measure standalone accuracy for each exit (not routed accuracy)."""
    model.eval()
    correct = [0, 0, 0]
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model.forward_all(images)
        for index, logits in enumerate(outputs):
            predictions = logits.argmax(dim=1)
            correct[index] += (predictions == labels).sum().item()

        total += labels.size(0)
    return tuple(100.0 * value / total for value in correct)


# ---------------------------------------------------------------------
# 5. Controlled Conv2 weight-scaling experiment
# ---------------------------------------------------------------------

@torch.no_grad()
def run_controlled_zeroing(
# Save the original trained model
    original_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    # Probabilities from 0% to 100%, in 1% increments
    probabilities = []
    exit1_accuracies = []
    exit2_accuracies = []
    exit3_accuracies = []

    for i in range(101):

        probability = i / 100.0
        probabilities.append(probability * 100)

        # Restore original trained weights
        model.load_state_dict(original_state)

        # Apply random pruning to conv2
        if probability > 0:
            prune.random_unstructured(
                model.conv2,
                name="weight",
                amount=probability
            )

            # Make zeroed weights permanent
            prune.remove(model.conv2, "weight")

        # Measure individual exit accuracies
        accuracy1, accuracy2, accuracy3 = measure_exit_accuracy()

        exit1_accuracies.append(accuracy1)
        exit2_accuracies.append(accuracy2)
        exit3_accuracies.append(accuracy3)

        print(
            f"Zeroing: {probability:.0%} | "
            f"Exit 1: {accuracy1:.2f}% | "
            f"Exit 2: {accuracy2:.2f}% | "
            f"Exit 3: {accuracy3:.2f}%"
        )

    # Plot results
    plt.figure(figsize=(10, 6))

    plt.plot(probabilities, exit1_accuracies, label="Exit 1")
    plt.plot(probabilities, exit2_accuracies, label="Exit 2")
    plt.plot(probabilities, exit3_accuracies, label="Exit 3")

    plt.xlabel("Conv2 Weight Zeroing Probability (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("Effect of Probabilistic Conv2 Weight Zeroing")

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("conv2_zeroing_accuracy.svg", format="svg")
    plt.savefig("conv2_zeroing_accuracy.pdf", format="pdf")

    plt.show()


# ---------------------------------------------------------------------
# 6. Main training workflow
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an early-exit FashionMNIST CNN and test Conv2 scaling."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fashionmnist_early_exit.pt"),
    )
    args = parser.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be in (0, 1]")

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = EarlyExitCNN().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            losses = [
                criterion(logits, labels)
                for logits in model.forward_all(images)
            ]
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)

        metrics = evaluate(model, test_loader, args.threshold)
        mean_loss = loss_sum / len(train_loader.dataset)
        print(
            f"Epoch {epoch:2d}/{args.epochs} "
            f"loss={mean_loss:.4f} "
            f"routed_accuracy={metrics['accuracy']:.4f} "
            f"exit_usage="
            f"{metrics['exit1_fraction']:.1%}/"
            f"{metrics['exit2_fraction']:.1%}/"
            f"{metrics['exit3_fraction']:.1%}"
        )

    metrics = evaluate(model, test_loader, args.threshold)
    torch.save(
        {
            "model_type": "EarlyExitCNN",
            "model_state_dict": model.state_dict(),
            "threshold": args.threshold,
            "seed": args.seed,
            "metrics": metrics,
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}; final routed metrics: {metrics}")

    print("\nRunning controlled Conv2 weight-scaling experiment...")
    run_controlled_zeroing(model, test_loader)


if __name__ == "__main__":
    main()
