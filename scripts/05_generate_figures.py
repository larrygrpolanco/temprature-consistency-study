#!/usr/bin/env python3
"""
05_generate_figures.py

Purpose: Generate publication-ready figures and tables from metrics.

Inputs:
    - results/metrics/*.json

Outputs:
    - results/figures/figure_1_accuracy_consistency.png
    - results/figures/figure_2_step_level.png
    - results/figures/table_2_accuracy.csv
    - results/figures/table_4_consistency.csv
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
        with open(acc_path, 'r') as f:
            metrics['accuracy'] = json.load(f)

    # Load Krippendorff's alpha
    alpha_path = metrics_dir / "krippendorff_alpha.json"
    if alpha_path.exists():
        with open(alpha_path, 'r') as f:
            metrics['alpha'] = json.load(f)

    # Load sentence-level analysis
    sent_path = metrics_dir / "sentence_level_analysis.json"
    if sent_path.exists():
        with open(sent_path, 'r') as f:
            metrics['sentence_level'] = json.load(f)

    return metrics


def create_accuracy_consistency_plot(metrics, output_path):
    """
    Dual-axis plot (Figure 1):
    - Left axis: Mean accuracy across temperatures (both move and step)
    - Right axis: Krippendorff's alpha across temperatures
    - X-axis: Temperature (0.0, 0.3, 0.7, 1.0, 1.5)
    """
    # Extract data
    temperatures = sorted([float(t) for t in metrics['accuracy'].keys()])

    move_accuracies = []
    step_accuracies = []
    move_alphas = []
    step_alphas = []

    for temp in temperatures:
        temp_key = f"{temp:.1f}"

        # Accuracy
        if metrics['accuracy'][temp_key]['move_level']:
            move_accuracies.append(metrics['accuracy'][temp_key]['move_level']['mean_accuracy'])
            step_accuracies.append(metrics['accuracy'][temp_key]['step_level']['mean_accuracy'])
        else:
            move_accuracies.append(None)
            step_accuracies.append(None)

        # Alpha
        if metrics['alpha'][temp_key]['move_level']['alpha'] is not None:
            move_alphas.append(metrics['alpha'][temp_key]['move_level']['alpha'])
            step_alphas.append(metrics['alpha'][temp_key]['step_level']['alpha'])
        else:
            move_alphas.append(None)
            step_alphas.append(None)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracy on left axis
    color_move = 'tab:blue'
    color_step = 'tab:cyan'
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Accuracy', color='black', fontsize=12)
    line1 = ax1.plot(temperatures, move_accuracies, 'o-', color=color_move, linewidth=2, markersize=8, label='Move-level Accuracy')
    line2 = ax1.plot(temperatures, step_accuracies, 's--', color=color_step, linewidth=2, markersize=8, label='Step-level Accuracy')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Plot alpha on right axis
    ax2 = ax1.twinx()
    color_alpha_move = 'tab:red'
    color_alpha_step = 'tab:orange'
    ax2.set_ylabel("Krippendorff's α", color='black', fontsize=12)
    line3 = ax2.plot(temperatures, move_alphas, '^-', color=color_alpha_move, linewidth=2, markersize=8, label="Move-level α")
    line4 = ax2.plot(temperatures, step_alphas, 'v--', color=color_alpha_step, linewidth=2, markersize=8, label="Step-level α")
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0, 1])

    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=10)

    plt.title('Accuracy and Consistency Across Temperature Settings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_step_level_plot(metrics, output_path):
    """
    Step-level accuracy and consistency across temperatures.
    Panel showing all 11 steps (rare steps in lighter color).
    """
    temperatures = sorted([float(t) for t in metrics['accuracy'].keys()])

    # Collect step-level data
    steps = ['1a', '1b', '1c', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '3d']
    rare_steps = ['2d', '3c', '3d']  # Rare steps according to spec

    step_data = {step: [] for step in steps}

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        per_step = metrics['accuracy'][temp_key].get('per_step', {})

        for step in steps:
            if step in per_step and per_step[step]:
                step_data[step].append(per_step[step]['mean_accuracy'])
            else:
                step_data[step].append(None)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    for step in steps:
        if all(v is None for v in step_data[step]):
            continue

        color = 'lightgray' if step in rare_steps else None
        linestyle = '--' if step in rare_steps else '-'
        alpha = 0.5 if step in rare_steps else 1.0

        ax.plot(temperatures, step_data[step], marker='o', linestyle=linestyle,
                label=step, alpha=alpha, linewidth=2)

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Step-level Accuracy', fontsize=12)
    ax.set_title('Step-Level Accuracy Across Temperature Settings', fontsize=14, fontweight='bold')
    ax.legend(ncol=2, fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_accuracy_table(metrics, output_path):
    """
    Generate Table 2: Move-Level Accuracy Across Temperature Settings

    Columns: Temperature | Mean Acc (%) | 95% CI | Std Dev
    """
    temperatures = sorted([float(t) for t in metrics['accuracy'].keys()])

    rows = []
    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        move_data = metrics['accuracy'][temp_key]['move_level']

        if move_data:
            rows.append({
                'Temperature': temp,
                'Mean Accuracy (%)': f"{move_data['mean_accuracy'] * 100:.2f}",
                '95% CI Lower': f"{move_data['ci_95'][0] * 100:.2f}",
                '95% CI Upper': f"{move_data['ci_95'][1] * 100:.2f}",
                'Std Dev': f"{move_data['std']:.4f}",
                'N Runs': move_data['n_runs']
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def create_consistency_table(metrics, output_path):
    """
    Generate Table 4: Consistency Across Temperature Settings

    Columns: Temperature | Move α | Step α | N Sentences | N Runs
    """
    temperatures = sorted([float(t) for t in metrics['alpha'].keys()])

    rows = []
    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        alpha_data = metrics['alpha'][temp_key]

        move_alpha = alpha_data['move_level']['alpha']
        step_alpha = alpha_data['step_level']['alpha']

        rows.append({
            'Temperature': temp,
            'Move α': f"{move_alpha:.4f}" if move_alpha is not None else 'N/A',
            'Step α': f"{step_alpha:.4f}" if step_alpha is not None else 'N/A',
            'N Sentences': alpha_data['move_level']['n_sentences'],
            'N Runs': alpha_data['move_level']['n_runs']
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def create_sentence_consistency_distribution(metrics, output_path):
    """
    Create a stacked bar chart showing distribution of sentence consistency
    across temperature settings.
    """
    temperatures = sorted([float(t) for t in metrics['sentence_level'].keys()])

    categories = ['high', 'moderate', 'low', 'inconsistent']
    category_labels = ['High (≥90%)', 'Moderate (70-89%)', 'Low (50-69%)', 'Inconsistent (<50%)']

    data = {cat: [] for cat in categories}

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        dist = metrics['sentence_level'][temp_key]['distribution']

        for cat in categories:
            data[cat].append(dist[cat] * 100)  # Convert to percentage

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(temperatures))
    width = 0.6

    bottom = np.zeros(len(temperatures))
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']

    for cat, label, color in zip(categories, category_labels, colors):
        ax.bar(x, data[cat], width, label=label, bottom=bottom, color=color)
        bottom += data[cat]

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Percentage of Sentences (%)', fontsize=12)
    ax.set_title('Distribution of Sentence-Level Consistency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in temperatures])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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

    # Figure 1: Accuracy and Consistency
    fig1_path = figures_dir / "figure_1_accuracy_consistency.png"
    create_accuracy_consistency_plot(metrics, fig1_path)
    print(f"✓ figure_1_accuracy_consistency.png")

    # Figure 2: Step-level analysis
    fig2_path = figures_dir / "figure_2_step_level.png"
    create_step_level_plot(metrics, fig2_path)
    print(f"✓ figure_2_step_level.png")

    # Figure 3: Sentence consistency distribution
    fig3_path = figures_dir / "figure_3_sentence_consistency.png"
    create_sentence_consistency_distribution(metrics, fig3_path)
    print(f"✓ figure_3_sentence_consistency.png")
    print()

    # Generate tables
    print("Generating tables...")

    # Table 2: Accuracy
    table2_path = figures_dir / "table_2_accuracy.csv"
    create_accuracy_table(metrics, table2_path)
    print(f"✓ table_2_accuracy.csv")

    # Table 4: Consistency
    table4_path = figures_dir / "table_4_consistency.csv"
    create_consistency_table(metrics, table4_path)
    print(f"✓ table_4_consistency.csv")
    print()

    print("=" * 60)
    print("COMPLETE ✓")
    print("All outputs saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
