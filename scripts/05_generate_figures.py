"""
05_generate_figures.py

Purpose: Generate publication-ready figures from metrics.

Inputs:
    - results/metrics/*.json

Outputs:
    - results/figures/figure_1_step_temperature_heatmap.png
    - results/figures/figure_2_accuracy_consistency.png
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def create_step_temperature_heatmap(metrics, output_path):
    """
    Figure 1: Step-Level Consistency Heatmap (RQ2)
    """
    # 1. Data Preparation (Same as before)
    temperatures = sorted([float(t) for t in metrics["alpha"].keys()])
    all_steps = ["1a", "1b", "1c", "2a", "2b", "2c", "2d", "3a", "3b", "3c"]

    step_labels = {
        "1a": "1a\nClaim centrality\n(n=74)",
        "1b": "1b\nTopic generalizations\n(n=166)",
        "1c": "1c\nReview research\n(n=453)",
        "2a": "2a\nCounter-claiming\n(n=26)",
        "2b": "2b\nIndicate gap\n(n=58)",
        "2c": "2c\nQuestion-raising\n(n=18)",
        "2d": "2d\nContinue tradition\n(n=11)",
        "3a": "3a\nOutline purposes\n(n=60)",
        "3b": "3b\nAnnounce research\n(n=91)",
        "3c": "3c\nAnnounce findings\n(n=65)",
    }

    alpha_matrix = []
    row_labels = []

    for step in all_steps:
        row = []
        for temp in temperatures:
            temp_key = f"{temp:.1f}"
            per_step = metrics["alpha"][temp_key].get("per_step", {})
            if step in per_step and per_step[step]["alpha"] is not None:
                row.append(per_step[step]["alpha"])
            else:
                row.append(np.nan)
        alpha_matrix.append(row)
        row_labels.append(step_labels[step])

    alpha_matrix = np.array(alpha_matrix)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. Heatmap
    sns.heatmap(
        alpha_matrix,
        annot=True,
        fmt=".3f",
        cmap="GnBu",  # Options: 'GnBu', 'PuBuGn', or 'crest'
        vmin=0.4,
        vmax=1.0,
        cbar=False,
        annot_kws={"fontsize": 11},
        xticklabels=[f"{t:.1f}" for t in temperatures],
        yticklabels=row_labels,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
    )

    # 4. Text Contrast Logic
    for text in ax.texts:
        try:
            val = float(text.get_text())
            # Threshold for white text
            if val > 0.799:
                text.set_color("white")
                text.set_fontweight("bold")
            else:
                text.set_color("#333333")  # Dark gray
                text.set_fontweight("bold")
        except ValueError:
            pass

    # 5. Axis Formatting
    ax.set_xlabel("Temperature", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Rhetorical Step", fontsize=12, fontweight="bold", labelpad=10)

    # Remove "tick" lines for a cleaner look
    ax.tick_params(left=True, bottom=True, length=3, width=0.5)

    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_accuracy_consistency_plot(metrics, output_path):
    """
    Figure 2: Accuracy and Consistency Two-Panel Plot
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

        # --- Extract Accuracy (Try F1 first, fallback to 'accuracy') ---
        # Move Level
        acc_data = metrics["accuracy"][temp_key]["move_level"]
        if "f1" in acc_data and acc_data["f1"]:
            move_accuracies.append(acc_data["f1"]["mean"])
        elif "accuracy" in acc_data and acc_data["accuracy"]:
            move_accuracies.append(acc_data["accuracy"]["mean"])
        else:
            move_accuracies.append(None)

        # Step Level
        step_data = metrics["accuracy"][temp_key]["step_level"]
        if "f1" in step_data and step_data["f1"]:
            step_accuracies.append(step_data["f1"]["mean"])
        elif "accuracy" in step_data and step_data["accuracy"]:
            step_accuracies.append(step_data["accuracy"]["mean"])
        else:
            step_accuracies.append(None)

        # --- Extract Alpha ---
        if metrics["alpha"][temp_key]["move_level"]["alpha"] is not None:
            move_alphas.append(metrics["alpha"][temp_key]["move_level"]["alpha"])
            step_alphas.append(metrics["alpha"][temp_key]["step_level"]["alpha"])
        else:
            move_alphas.append(None)
            step_alphas.append(None)

    # Create figure with two panels (1 row, 2 columns)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))

    # ===============================
    # PANEL A: MOVE-LEVEL
    # ===============================
    color_move_acc = "#0072B2"  # Blue
    color_move_alpha = "#D55E00"  # Vermilion

    # Left Axis: Accuracy
    ax1.set_xlabel("Temperature", fontsize=12)
    ax1.set_ylabel(
        "Move-Level Weighted F1",
        color=color_move_acc,
        fontsize=12,
        fontweight="bold",
    )
    line1 = ax1.plot(
        temperatures,
        move_accuracies,
        "o-",
        color=color_move_acc,
        linewidth=3,
        markersize=9,
        label="Weighted F1",
        zorder=2,
    )
    ax1.tick_params(axis="y", labelcolor=color_move_acc)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Panel A: Move-Level Performance", fontsize=13, fontweight="bold")

    # Right Axis: Consistency
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Move-Level Krippendorff's α",
        color=color_move_alpha,
        fontsize=12,
        fontweight="bold",
    )
    line2 = ax2.plot(
        temperatures,
        move_alphas,
        "^-",
        color=color_move_alpha,
        linewidth=3,
        markersize=9,
        label="Krippendorff's α",
        zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor=color_move_alpha)
    ax2.set_ylim([0.0, 1.05])

    # Combine legends for Panel A
    lines_a = line2 + line1
    labels_a = [l.get_label() for l in lines_a]
    ax1.legend(
        lines_a,
        labels_a,
        loc="lower left",
        fontsize=11,
        framealpha=0.9,
        facecolor="white",
    )

    # ===============================
    # PANEL B: STEP-LEVEL
    # ===============================
    color_step_acc = "#009E73"  # Bluish Green
    color_step_alpha = "#E69F00"  # Orange

    # Left Axis: Accuracy
    ax3.set_xlabel("Temperature", fontsize=12)
    ax3.set_ylabel(
        "Step-Level Weighted F1",
        color=color_step_acc,
        fontsize=12,
        fontweight="bold",
    )
    line3 = ax3.plot(
        temperatures,
        step_accuracies,
        "s-",
        color=color_step_acc,
        linewidth=3,
        markersize=9,
        label="Weighted F1",
        zorder=2,
    )
    ax3.tick_params(axis="y", labelcolor=color_step_acc)
    ax3.set_ylim([0.0, 1.05])
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Panel B: Step-Level Performance", fontsize=13, fontweight="bold")

    # Right Axis: Consistency
    ax4 = ax3.twinx()
    ax4.set_ylabel(
        "Step-Level Krippendorff's α",
        color=color_step_alpha,
        fontsize=12,
        fontweight="bold",
    )
    line4 = ax4.plot(
        temperatures,
        step_alphas,
        "v-",
        color=color_step_alpha,
        linewidth=3,
        markersize=9,
        label="Krippendorff's α",
        zorder=3,
    )
    ax4.tick_params(axis="y", labelcolor=color_step_alpha)
    ax4.set_ylim([0.0, 1.05])

    # Combine legends for Panel B
    lines_b = line4 + line3
    labels_b = [l.get_label() for l in lines_b]
    ax3.legend(
        lines_b,
        labels_b,
        loc="lower left",
        fontsize=11,
        framealpha=0.9,
        facecolor="white",
    )

    # Align X-Ticks
    ax1.set_xticks(temperatures)
    ax3.set_xticks(temperatures)

    # Overall title
    # fig.suptitle(
    #     "Accuracy and Consistency Across Temperature Settings",
    #     fontsize=15,
    #     fontweight="bold",
    #     y=0.98,
    # )

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

    # Figure 1: Step-level consistency heatmap (RQ2 Heatmap)
    fig1_path = figures_dir / "figure_1_step_temperature_heatmap.png"
    create_step_temperature_heatmap(metrics, fig1_path)
    print(f"✓ figure_1_step_temperature_heatmap.png")

    # Figure 2: Accuracy and Consistency Two-Panel (Replacing Alligator Plot)
    fig2_path = figures_dir / "figure_2_accuracy_consistency.png"
    create_accuracy_consistency_plot(metrics, fig2_path)
    print(f"✓ figure_2_accuracy_consistency.png")

    print()
    print("=" * 60)
    print("COMPLETE ✓")
    print("All outputs saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
