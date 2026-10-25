#!/usr/bin/env python3
"""
FashionMNIST early-exit CNN experiment:
1. Train on the training set.
2. Evaluate baseline performance on the test set.
3. Zero conv2 weights (leave conv2 bias unchanged).
4. Re-evaluate the same test set and compare results.

Run:
    python early_exit_conv2_fault_experiment.py --epochs 8

Requires:
    pip install torch torchvision
"""

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class EarlyExitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.exit1 = nn.Linear(16 * 14 * 14, num_classes)
        self.exit2 = nn.Linear(32 * 7 * 7, num_classes)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.exit3 = nn.Linear(128, num_classes)

    def forward_all(self, x):
        # Exit 1: after conv1 + pooling
        x = self.pool(F.relu(self.conv1(x)))
        exit1 = self.exit1(torch.flatten(x, 1))

        # Exit 2: after conv2 + pooling
        x = self.pool(F.relu(self.conv2(x)))
        flat = torch.flatten(x, 1)
        exit2 = self.exit2(flat)

        # Exit 3: final classifier
        x = F.relu(self.fc1(flat))
        exit3 = self.exit3(x)

        return exit1, exit2, exit3

    def forward(self, x):
        return self.forward_all(x)[-1]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_all_exits(model, loader, device, threshold):
    """
    Evaluates each exit independently, plus confidence-based routing.
    Losses are cross-entropy averages over examples.
    Routing uses exit1 if max softmax confidence >= threshold;
    otherwise exit2 if confidence >= threshold; otherwise exit3.
    """
    model.eval()
    exit_correct = [0, 0, 0]
    exit_loss_sums = [0.0, 0.0, 0.0]
    routed_correct = 0
    routed_loss_sum = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list = model.forward_all(images)
            batch_n = labels.size(0)

            for i, logits in enumerate(logits_list):
                exit_correct[i] += (logits.argmax(dim=1) == labels).sum().item()
                exit_loss_sums[i] += criterion(logits, labels).item()

            # Confidence-based routing. Confidence is computed from softmax logits.
            probabilities = [torch.softmax(logits, dim=1) for logits in logits_list]
            confidences = [p.max(dim=1).values for p in probabilities]
            predictions = [p.argmax(dim=1) for p in probabilities]

            routed_predictions = predictions[2].clone()
            use_exit2 = confidences[1] >= threshold
            routed_predictions[use_exit2] = predictions[1][use_exit2]
            use_exit1 = confidences[0] >= threshold
            routed_predictions[use_exit1] = predictions[0][use_exit1]

            # For a consistent scalar loss, compute CE from the selected exit's logits.
            routed_logits = logits_list[2].clone()
            routed_logits[use_exit2] = logits_list[1][use_exit2]
            routed_logits[use_exit1] = logits_list[0][use_exit1]
            routed_correct += (routed_predictions == labels).sum().item()
            routed_loss_sum += criterion(routed_logits, labels).item()
            total += batch_n

    results = {}
    for i, name in enumerate(("exit1", "exit2", "exit3")):
        results[name] = {
            "accuracy_percent": 100.0 * exit_correct[i] / total,
            "loss": exit_loss_sums[i] / total,
        }
    results["early_exit_routing"] = {
        "accuracy_percent": 100.0 * routed_correct / total,
        "loss": routed_loss_sum / total,
        "threshold": threshold,
    }
    results["num_test_images"] = total
    return results


def train_model(model, train_loader, device, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            exit_logits = model.forward_all(images)
            losses = [criterion(logits, labels) for logits in exit_logits]
            loss = sum(losses) / len(losses)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"mean training loss: {running_loss / seen:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./conv2_fault_results")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Early-exit confidence threshold: {args.threshold:.2f}")

    model = EarlyExitCNN().to(device)

    # 1. Train once on training data only.
    train_model(model, train_loader, device, args.epochs, args.learning_rate)

    # 2. Save the untouched trained checkpoint before injecting any fault.
    original_checkpoint_path = output_dir / "fashionmnist_earlyexit_trained.pt"
    torch.save({
        "model_type": "EarlyExitCNN",
        "model_state_dict": model.state_dict(),
        "threshold": args.threshold,
        "seed": args.seed,
        "epochs": args.epochs,
        "metrics": {},
    }, original_checkpoint_path)
    print(f"\nSaved original trained checkpoint: {original_checkpoint_path}")

    # 3. Evaluate baseline on the test set.
    baseline = evaluate_all_exits(model, test_loader, device, args.threshold)

    # Save a separate copy of the trained parameters for a clear comparison.
    baseline_state = copy.deepcopy(model.state_dict())

    # 4. Fault injection: zero conv2 weights only; retain conv2 bias.
    with torch.no_grad():
        model.conv2.weight.zero_()

    zeroed_sum = model.conv2.weight.abs().sum().item()
    print(f"\nAfter fault injection, sum(abs(conv2.weight)) = {zeroed_sum:.1f}")

    # 5. Evaluate the same test images with the modified model.
    faulted = evaluate_all_exits(model, test_loader, device, args.threshold)

    # 6. Display comparison.
    print("\n" + "=" * 74)
    print("TEST RESULTS: BASELINE vs CONV2 WEIGHTS ZEROED")
    print("=" * 74)
    print(f"{'Evaluation':<22} {'Baseline acc.':>14} {'Faulted acc.':>14} "
          f"{'Change (pp)':>13}")
    print("-" * 74)

    for key in ("exit1", "exit2", "exit3", "early_exit_routing"):
        base_acc = baseline[key]["accuracy_percent"]
        fault_acc = faulted[key]["accuracy_percent"]
        base_loss = baseline[key]["loss"]
        fault_loss = faulted[key]["loss"]
        print(f"{key:<22} {base_acc:>13.2f}% {fault_acc:>13.2f}% "
              f"{fault_acc - base_acc:>+12.2f}")
        print(f"  loss: baseline={base_loss:.4f} | faulted={fault_loss:.4f} "
              f"| change={fault_loss - base_loss:+.4f}")

    print("\nNote: exit1 branches before conv2, so its predictions should remain "
          "unchanged (up to numerical effects).")
    print("The early-exit routing uses the same fixed confidence threshold in "
          "both evaluations.")

    # 7. Save the faulted checkpoint separately and save results as JSON.
    faulted_checkpoint_path = output_dir / "fashionmnist_earlyexit_conv2_zeroed.pt"
    torch.save({
        "model_type": "EarlyExitCNN",
        "model_state_dict": model.state_dict(),
        "threshold": args.threshold,
        "seed": args.seed,
        "epochs": args.epochs,
        "fault": "conv2.weight zeroed; conv2.bias unchanged",
        "metrics": faulted,
    }, faulted_checkpoint_path)

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline,
            "conv2_weights_zeroed": faulted,
            "fault_description": "conv2.weight zeroed; conv2.bias unchanged",
            "threshold": args.threshold,
            "seed": args.seed,
            "epochs": args.epochs,
        }, f, indent=2)

    print(f"\nSaved faulted checkpoint: {faulted_checkpoint_path}")
    print(f"Saved metrics comparison: {results_path}")
    print("\nExperiment complete. Original checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
