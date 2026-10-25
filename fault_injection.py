"""Evaluate the effect of isolated Float32 parameter bit flips on a checkpoint.

This is software-level fault injection.  It never writes modified parameters
back to the source checkpoint and each trial reloads the golden state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from early_exit_fashionmnist import DEVICE, EarlyExitCNN, get_loaders
from train_fashionmnist import SimpleCNN


def build_model(checkpoint: dict) -> torch.nn.Module:
    if checkpoint.get("model_type") == "EarlyExitCNN":
        return EarlyExitCNN()
    return SimpleCNN()


def flip_float32_bit(parameter: torch.Tensor, flat_index: int, bit: int) -> tuple[float, float]:
    """Flip one IEEE-754 binary32 bit in-place and return before/after values."""
    if parameter.dtype != torch.float32:
        raise TypeError(f"expected float32 parameter, got {parameter.dtype}")
    flat = parameter.detach().view(-1)
    before = flat[flat_index].item()
    words = flat.view(torch.int32)
    words[flat_index] ^= 1 << bit
    return before, flat[flat_index].item()


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.size(0)
    return correct / total


def choose_location(model: torch.nn.Module, rng: random.Random) -> tuple[str, int]:
    candidates = [(name, p) for name, p in model.named_parameters()
                  if p.dtype == torch.float32 and p.numel()]
    name, parameter = rng.choice(candidates)
    return name, rng.randrange(parameter.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bit", type=int, choices=range(32), default=None,
                        help="Fix an IEEE-754 bit; omit to sample a bit each trial.")
    parser.add_argument("--output", type=Path, default=Path("fault_results.csv"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    golden_state = checkpoint["model_state_dict"]
    _, test_loader = get_loaders(batch_size=128, workers=args.workers)
    rng = random.Random(args.seed)

    golden_model = build_model(checkpoint).to(DEVICE)
    golden_model.load_state_dict(golden_state)
    baseline_accuracy = accuracy(golden_model, test_loader)
    print(f"golden accuracy={baseline_accuracy:.4f}")

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "parameter", "flat_index", "bit", "before", "after", "accuracy", "accuracy_drop"])
        writer.writeheader()
        for trial in range(args.trials):
            model = build_model(checkpoint).to(DEVICE)
            model.load_state_dict(copy.deepcopy(golden_state))
            name, flat_index = choose_location(model, rng)
            bit = args.bit if args.bit is not None else rng.randrange(32)
            parameter = dict(model.named_parameters())[name]
            before, after = flip_float32_bit(parameter, flat_index, bit)
            trial_accuracy = accuracy(model, test_loader)
            row = {"trial": trial, "parameter": name, "flat_index": flat_index, "bit": bit,
                   "before": before, "after": after, "accuracy": trial_accuracy,
                   "accuracy_drop": baseline_accuracy - trial_accuracy}
            writer.writerow(row)
            print(f"trial={trial:03d} {name}[{flat_index}] bit={bit:2d} "
                  f"accuracy={trial_accuracy:.4f} drop={row['accuracy_drop']:.4f}")
    print(f"wrote {args.output}; source checkpoint was not modified")


if __name__ == "__main__":
    main()
