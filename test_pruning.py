import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune

from early_exit_fashionmnist import (
    EarlyExitCNN,
    DEVICE,
    get_loaders,
    measure_exit_accuracies,
)


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

CHECKPOINT = Path("fashionmnist_early_exit.pt")
RESULTS_DIR = Path("pruning_results")

BATCH_SIZE = 128
WORKERS = 2
SEED = 7

PRUNING_LEVELS = range(101)


# ---------------------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------------------

def load_model():

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}. "
            "Run early_exit_fashionmnist.py first."
        )

    model = EarlyExitCNN().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ---------------------------------------------------------------------
# 3. Run one pruning experiment
# ---------------------------------------------------------------------

def run_pruning_experiment(model, test_loader, layers, experiment_name):

    percentages = []
    exit1_accuracies = []
    exit2_accuracies = []
    exit3_accuracies = []

    # Save original weights
    original_weights = [
        layer.weight.detach().clone()
        for layer in layers
    ]

    torch.manual_seed(SEED)

    print(f"\nRunning {experiment_name} experiment...")

    for percentage in PRUNING_LEVELS:

        # Restore original weights before each iteration
        with torch.no_grad():
            for layer, original_weight in zip(layers, original_weights):
                layer.weight.copy_(original_weight)

        # Apply pruning
        if percentage > 0:
            for layer in layers:
                prune.random_unstructured(
                    layer,
                    name="weight",
                    amount=percentage / 100
                )

                # Make pruning permanent
                prune.remove(layer, "weight")

        # Measure accuracy
        accuracy1, accuracy2, accuracy3 = measure_exit_accuracies(
            model, test_loader
        )

        percentages.append(percentage)
        exit1_accuracies.append(accuracy1)
        exit2_accuracies.append(accuracy2)
        exit3_accuracies.append(accuracy3)

        print(
            f"{percentage:3d}% | "
            f"Exit 1: {accuracy1:.2f}% | "
            f"Exit 2: {accuracy2:.2f}% | "
            f"Exit 3: {accuracy3:.2f}%"
        )

    return (
        percentages,
        exit1_accuracies,
        exit2_accuracies,
        exit3_accuracies
    )

# ---------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------

def save_results(all_results):

    csv_path = RESULTS_DIR / "all_pruning_results.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "Pruning_Percentage",
            "Exit1_Accuracy",
            "Exit2_Accuracy",
            "Exit3_Accuracy",
        ])

        for experiment_name, results in all_results.items():

            percentages, exit1, exit2, exit3 = results

            for i in range(len(percentages)):

                writer.writerow([
                    experiment_name,
                    percentages[i],
                    exit1[i],
                    exit2[i],
                    exit3[i],
                ])

    print(f"\nCSV saved: {csv_path}")


# ---------------------------------------------------------------------
# 5. Generate three separate plots
# ---------------------------------------------------------------------

def plot_results(
    percentages,
    exit1_accuracies,
    exit2_accuracies,
    exit3_accuracies,
    experiment_name,
    filename,
):

    plt.figure(figsize=(10, 6))

    plt.plot(percentages, exit1_accuracies, label="Exit 1")
    plt.plot(percentages, exit2_accuracies, label="Exit 2")
    plt.plot(percentages, exit3_accuracies, label="Exit 3")

    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Accuracy (%)")

    plt.title(f"{experiment_name} Pruning")
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / f"{filename}.svg", format="svg")
    plt.savefig(RESULTS_DIR / f"{filename}.pdf", format="pdf")

    plt.close()

    print(f"Saved {experiment_name} plots.")


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, test_loader = get_loaders(
        batch_size=BATCH_SIZE,
        workers=WORKERS,
    )

    # -------------------------------------------------
    # Experiment 1: Conv1 only
    # -------------------------------------------------

    model = load_model()

    conv1_percentages, conv1_exit1, conv1_exit2, conv1_exit3 = (
        run_pruning_experiment(
            model,
            test_loader,
            layers=[model.conv1],
            experiment_name="Conv1",
        )
    )

    # -------------------------------------------------
    # Experiment 2: Conv2 only
    # -------------------------------------------------

    model = load_model()

    conv2_percentages, conv2_exit1, conv2_exit2, conv2_exit3 = (
        run_pruning_experiment(
            model,
            test_loader,
            layers=[model.conv2],
            experiment_name="Conv2",
        )
    )

    # -------------------------------------------------
    # Experiment 3: Conv1 and Conv2
    # -------------------------------------------------

    model = load_model()

    both_percentages, both_exit1, both_exit2, both_exit3 = (
        run_pruning_experiment(
            model,
            test_loader,
            layers=[model.conv1, model.conv2],
            experiment_name="Conv1_and_Conv2",
        )
    )

    # -------------------------------------------------
    # Save all results to CSV
    # -------------------------------------------------

    all_results = {
        "Conv1": (
            conv1_percentages,
            conv1_exit1,
            conv1_exit2,
            conv1_exit3,
        ),

        "Conv2": (
            conv2_percentages,
            conv2_exit1,
            conv2_exit2,
            conv2_exit3,
        ),

        "Conv1_and_Conv2": (
            both_percentages,
            both_exit1,
            both_exit2,
            both_exit3,
        ),
    }

    save_results(all_results)

    # -------------------------------------------------
    # Generate three separate plots
    # -------------------------------------------------

    plot_results(
        conv1_percentages,
        conv1_exit1,
        conv1_exit2,
        conv1_exit3,
        "Conv1",
        "conv1",
    )

    plot_results(
        conv2_percentages,
        conv2_exit1,
        conv2_exit2,
        conv2_exit3,
        "Conv2",
        "conv2",
    )

    plot_results(
        both_percentages,
        both_exit1,
        both_exit2,
        both_exit3,
        "Conv1 + Conv2",
        "conv1_and_conv2",
    )


if __name__ == "__main__":
    main()