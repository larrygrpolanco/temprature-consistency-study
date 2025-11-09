"""
05_generate_figures.py

Purpose: Generate publication-ready figures from metrics.

Inputs:
    - results/metrics/*.json

Outputs:
    - results/figures/figure_1_accuracy_consistency.png
    - results/figures/figure_2_step_level_consistency.png
    - results/figures/figure_3_sentence_consistency.png
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def load_metrics():
    """Load all metrics from JSON files"""
    metrics_dir = Path("results/metrics")

    metrics = {}

    # Load accuracy
    acc_path = metrics_dir / "accuracy_by_temperature.json"
    if acc_path.exists():
        with open(acc_path, "r") as f:
            metrics["accuracy"] = json.load(f)

    # Load Krippendorff's alpha
    alpha_path = metrics_dir / "krippendorff_alpha.json"
    if alpha_path.exists():
        with open(alpha_path, "r") as f:
            metrics["alpha"] = json.load(f)

    # Load sentence-level analysis
    sent_path = metrics_dir / "sentence_level_analysis.json"
    if sent_path.exists():
        with open(sent_path, "r") as f:
            metrics["sentence_level"] = json.load(f)

    return metrics


def create_accuracy_consistency_plot(metrics, output_path):
    """
    Two-panel figure (Figure 1):
    Panel A: Move-level accuracy and move-level alpha
    Panel B: Step-level accuracy and step-level alpha
    Both panels have dual y-axes (left: accuracy, right: alpha)
    """
    # Extract data
    temperatures = sorted([float(t) for t in metrics["accuracy"].keys()])

    move_accuracies = []
    step_accuracies = []
    move_alphas = []
    step_alphas = []

    for temp in temperatures:
        temp_key = f"{temp:.1f}"

        # Accuracy - FIXED: Updated to match script 04 structure
        if metrics["accuracy"][temp_key]["move_level"]["accuracy"]:
            move_accuracies.append(
                metrics["accuracy"][temp_key]["move_level"]["accuracy"]["mean"]
            )
            step_accuracies.append(
                metrics["accuracy"][temp_key]["step_level"]["accuracy"]["mean"]
            )
        else:
            move_accuracies.append(None)
            step_accuracies.append(None)

        # Alpha
        if metrics["alpha"][temp_key]["move_level"]["alpha"] is not None:
            move_alphas.append(metrics["alpha"][temp_key]["move_level"]["alpha"])
            step_alphas.append(metrics["alpha"][temp_key]["step_level"]["alpha"])
        else:
            move_alphas.append(None)
            step_alphas.append(None)

    # Create figure with two panels (1 row, 2 columns)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== PANEL A: MOVE-LEVEL =====
    color_move_acc = "tab:blue"
    color_move_alpha = "tab:red"

    ax1.set_xlabel("Temperature", fontsize=12)
    ax1.set_ylabel("Move-Level Accuracy", color=color_move_acc, fontsize=12)
    line1 = ax1.plot(
        temperatures,
        move_accuracies,
        "o-",
        color=color_move_acc,
        linewidth=2.5,
        markersize=8,
        label="Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color_move_acc)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Panel A: Move-Level Performance", fontsize=12, fontweight="bold")

    # Right axis for move-level alpha
    ax2 = ax1.twinx()
    ax2.set_ylabel("Move-Level Krippendorff's α", color=color_move_alpha, fontsize=12)
    line2 = ax2.plot(
        temperatures,
        move_alphas,
        "^-",
        color=color_move_alpha,
        linewidth=2.5,
        markersize=8,
        label="Krippendorff's α",
    )
    ax2.tick_params(axis="y", labelcolor=color_move_alpha)
    ax2.set_ylim([0, 1])

    # Combine legends for Panel A
    lines_a = line1 + line2
    labels_a = [l.get_label() for l in lines_a]
    ax1.legend(lines_a, labels_a, loc="lower left", fontsize=10)

    # ===== PANEL B: STEP-LEVEL =====
    color_step_acc = "tab:green"
    color_step_alpha = "tab:orange"

    ax3.set_xlabel("Temperature", fontsize=12)
    ax3.set_ylabel("Step-Level Accuracy", color=color_step_acc, fontsize=12)
    line3 = ax3.plot(
        temperatures,
        step_accuracies,
        "s-",
        color=color_step_acc,
        linewidth=2.5,
        markersize=8,
        label="Accuracy",
    )
    ax3.tick_params(axis="y", labelcolor=color_step_acc)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Panel B: Step-Level Performance", fontsize=12, fontweight="bold")

    # Right axis for step-level alpha
    ax4 = ax3.twinx()
    ax4.set_ylabel("Step-Level Krippendorff's α", color=color_step_alpha, fontsize=12)
    line4 = ax4.plot(
        temperatures,
        step_alphas,
        "v-",
        color=color_step_alpha,
        linewidth=2.5,
        markersize=8,
        label="Krippendorff's α",
    )
    ax4.tick_params(axis="y", labelcolor=color_step_alpha)
    ax4.set_ylim([0, 1])

    # Combine legends for Panel B
    lines_b = line3 + line4
    labels_b = [l.get_label() for l in lines_b]
    ax3.legend(lines_b, labels_b, loc="lower left", fontsize=10)

    # Overall title
    fig.suptitle(
        "Accuracy and Consistency Across Temperature Settings",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_step_level_plot(metrics, output_path):
    """
    Step-level consistency (Krippendorff's alpha) across temperatures.
    Only shows steps with n≥50 (7 steps) to avoid cluttering with rare steps.
    This figure shows how consistency varies by rhetorical step across temperature settings.
    """
    temperatures = sorted([float(t) for t in metrics["alpha"].keys()])

    # Only include steps with n≥50 in the corpus
    # Excluded: 2a (27), 2c (24), 2d (11), 3d (1)
    steps_to_plot = ["1a", "1b", "1c", "2b", "3a", "3b", "3c"]

    step_labels = {
        "1a": "1a - Claim centrality",
        "1b": "1b - Topic generalizations",
        "1c": "1c - Review research",
        "2b": "2b - Indicate gap",
        "3a": "3a - Outline purposes",
        "3b": "3b - Announce research",
        "3c": "3c - Announce findings",
    }

    step_data = {step: [] for step in steps_to_plot}

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        per_step = metrics["alpha"][temp_key].get("per_step", {})

        for step in steps_to_plot:
            if step in per_step and per_step[step]["alpha"] is not None:
                step_data[step].append(per_step[step]["alpha"])
            else:
                step_data[step].append(None)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use distinct colors and markers for each step
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(steps_to_plot)))
    markers = ["o", "s", "^", "D", "v", "p", "*"]

    for i, step in enumerate(steps_to_plot):
        if all(v is None for v in step_data[step]):
            continue

        ax.plot(
            temperatures,
            step_data[step],
            marker=markers[i],
            linestyle="-",
            label=step_labels[step],
            color=colors[i],
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Krippendorff's α (Step-Level Consistency)", fontsize=12)
    ax.set_title(
        "Step-Level Consistency Across Temperature Settings (n≥50)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(ncol=1, fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_sentence_consistency_distribution(metrics, output_path):
    """
    Create a stacked bar chart showing distribution of sentence consistency
    across temperature settings.

    This visualizes Table 5 data showing what % of sentences fall into each
    consistency category (high/moderate/low/inconsistent).
    """
    temperatures = sorted([float(t) for t in metrics["sentence_level"].keys()])

    categories = ["high", "moderate", "low", "inconsistent"]
    category_labels = [
        "High (≥90%)",
        "Moderate (70-89%)",
        "Low (50-69%)",
        "Inconsistent (<50%)",
    ]

    data = {cat: [] for cat in categories}

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        dist = metrics["sentence_level"][temp_key]["distribution"]

        for cat in categories:
            data[cat].append(dist[cat] * 100)  # Convert to percentage

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(temperatures))
    width = 0.6

    bottom = np.zeros(len(temperatures))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

    for cat, label, color in zip(categories, category_labels, colors):
        ax.bar(x, data[cat], width, label=label, bottom=bottom, color=color)
        bottom += data[cat]

    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel("Percentage of Sentences (%)", fontsize=12)
    ax.set_title(
        "Distribution of Sentence-Level Consistency", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in temperatures])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("FIGURE GENERATION")
    print("=" * 60)
    print()

    # Load metrics
    print("Loading metrics...")
    metrics = load_metrics()

    if not metrics:
        print("❌ ERROR: No metrics found. Please run 04_calculate_metrics.py first.")
        return

    for key in metrics.keys():
        print(f"✓ {key}")
    print()

    # Create output directory
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("Generating figures...")

    # Figure 1: Accuracy and Consistency (ESSENTIAL - your main RQ3 figure)
    fig1_path = figures_dir / "figure_1_accuracy_consistency.png"
    create_accuracy_consistency_plot(metrics, fig1_path)
    print(f"✓ figure_1_accuracy_consistency.png")

    # Figure 2: Step-level consistency analysis
    fig2_path = figures_dir / "figure_2_step_level_consistency.png"
    create_step_level_plot(metrics, fig2_path)
    print(f"✓ figure_2_step_level_consistency.png")

    # Figure 3: Sentence consistency distribution (USEFUL)
    fig3_path = figures_dir / "figure_3_sentence_consistency.png"
    create_sentence_consistency_distribution(metrics, fig3_path)
    print(f"✓ figure_3_sentence_consistency.png")
    print()

    print("=" * 60)
    print("COMPLETE ✓")
    print("All outputs saved to results/figures/")
    print()
    print("NOTE: Tables are generated by 04_calculate_metrics.py")
    print("      Location: results/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
