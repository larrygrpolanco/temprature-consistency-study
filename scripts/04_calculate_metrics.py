"""
04_calculate_metrics.py

Purpose: Calculate accuracy, consistency, and Krippendorff's alpha from parsed predictions.

Inputs:
    - outputs/temp_{X}/run_{YY}/parsed.json (all runs for all temperatures)
    - data/processed/{dataset}/gold.json

Outputs:
    - results/metrics/accuracy_by_temperature.json
    - results/metrics/krippendorff_alpha.json
    - results/metrics/sentence_level_analysis.json
    - results/tables/table2_move_accuracy.csv
    - results/tables/table3_step_accuracy.csv
    - results/tables/table4_consistency.csv
    - results/tables/tableX_parsing_success.csv
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from sklearn.metrics import f1_score
import krippendorff


def load_config():
    """Load config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_gold_standard(dataset):
    """Load gold standard annotations"""
    gold_path = Path(f"data/processed/{dataset}/gold.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parsed_predictions(temperature, run_num):
    """Load parsed predictions for a specific temperature and run"""
    parsed_path = Path(f"outputs/temp_{temperature:.1f}/run_{run_num:02d}/parsed.json")
    if not parsed_path.exists():
        return None

    with open(parsed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_accuracy_single_run(run_predictions, gold_standard):
    """
    Calculate accuracy and F1 for one run at move and step levels.

    Returns:
        {
            'move_accuracy': 0.XX,
            'move_f1': 0.XX,
            'step_accuracy': 0.XX,
            'step_f1': 0.XX,
            'per_move_accuracy': {'1': 0.XX, '2': 0.XX, '3': 0.XX},
            'per_step_accuracy': {'1a': 0.XX, '1b': 0.XX, ...}
        }
    """
    # Collect all labels for aggregate metrics
    all_gold_moves = []
    all_pred_moves = []
    all_gold_steps = []
    all_pred_steps = []

    # Per-move and per-step counters
    per_move = defaultdict(lambda: {"correct": 0, "total": 0})
    per_step = defaultdict(lambda: {"correct": 0, "total": 0})

    for article_id, article_data in run_predictions["articles"].items():
        if article_id not in gold_standard:
            continue

        gold_sentences = gold_standard[article_id]["sentences"]
        pred_sentences = article_data["predictions"]

        for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
            gold_move = gold_sent["move"]
            gold_step = gold_sent["step"]
            pred_move = pred_sent["predicted_move"]
            pred_step = pred_sent["predicted_step"]

            # Collect for aggregate metrics
            all_gold_moves.append(gold_move)
            all_pred_moves.append(pred_move)
            all_gold_steps.append(gold_step)
            all_pred_steps.append(pred_step)

            # Move accuracy
            if gold_move == pred_move:
                per_move[gold_move]["correct"] += 1
            per_move[gold_move]["total"] += 1

            # Step accuracy
            if gold_step == pred_step:
                per_step[gold_step]["correct"] += 1
            per_step[gold_step]["total"] += 1

    # Calculate overall accuracy
    total = len(all_gold_moves)
    move_accuracy = (
        sum(1 for g, p in zip(all_gold_moves, all_pred_moves) if g == p) / total
        if total > 0
        else 0
    )
    step_accuracy = (
        sum(1 for g, p in zip(all_gold_steps, all_pred_steps) if g == p) / total
        if total > 0
        else 0
    )

    # Calculate weighted F1 scores
    move_f1 = f1_score(
        all_gold_moves, all_pred_moves, average="weighted", zero_division=0
    )
    step_f1 = f1_score(
        all_gold_steps, all_pred_steps, average="weighted", zero_division=0
    )

    # Calculate per-class accuracy
    per_move_accuracy = {
        move: counts["correct"] / counts["total"] for move, counts in per_move.items()
    }

    per_step_accuracy = {
        step: counts["correct"] / counts["total"] for step, counts in per_step.items()
    }

    return {
        "move_accuracy": move_accuracy,
        "move_f1": move_f1,
        "step_accuracy": step_accuracy,
        "step_f1": step_f1,
        "per_move_accuracy": per_move_accuracy,
        "per_step_accuracy": per_step_accuracy,
    }


def calculate_accuracy_by_temperature(temperature, runs, gold_standard):
    """
    Aggregate accuracy across all runs for one temperature.

    Returns:
        {
            'temperature': 0.0,
            'move_level': {
                'accuracy': {'mean': ..., 'std': ..., 'ci_95': [...], 'n': ...},
                'f1': {'mean': ..., 'std': ..., 'ci_95': [...], 'n': ...}
            },
            'step_level': {...},
            'per_move': {...},
            'per_step': {...}
        }
    """
    move_accuracies = []
    move_f1s = []
    step_accuracies = []
    step_f1s = []
    per_move_accs = defaultdict(list)
    per_step_accs = defaultdict(list)

    for run_num in runs:
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        accuracy_result = calculate_accuracy_single_run(run_data, gold_standard)

        move_accuracies.append(accuracy_result["move_accuracy"])
        move_f1s.append(accuracy_result["move_f1"])
        step_accuracies.append(accuracy_result["step_accuracy"])
        step_f1s.append(accuracy_result["step_f1"])

        for move, acc in accuracy_result["per_move_accuracy"].items():
            per_move_accs[move].append(acc)

        for step, acc in accuracy_result["per_step_accuracy"].items():
            per_step_accs[step].append(acc)

    # Calculate statistics
    def calc_stats(values):
        if not values:
            return None
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0

        # Handle zero variance or single value
        if std == 0 or len(values) == 1:
            return {
                "mean": float(mean),
                "std": 0.0,
                "ci_95": [float(mean), float(mean)],
                "n": len(values),
            }

        # Calculate 95% confidence interval
        ci_95 = stats.t.interval(
            0.95, len(values) - 1, loc=mean, scale=stats.sem(values)
        )
        return {
            "mean": float(mean),
            "std": float(std),
            "ci_95": [float(ci_95[0]), float(ci_95[1])],
            "n": len(values),
        }

    result = {
        "temperature": temperature,
        "move_level": {
            "accuracy": calc_stats(move_accuracies),
            "f1": calc_stats(move_f1s),
        },
        "step_level": {
            "accuracy": calc_stats(step_accuracies),
            "f1": calc_stats(step_f1s),
        },
    }

    # Add per-move and per-step breakdowns
    result["per_move"] = {
        move: calc_stats(accs) for move, accs in per_move_accs.items()
    }

    result["per_step"] = {
        step: calc_stats(accs) for step, accs in per_step_accs.items()
    }

    return result


def bootstrap_alpha_ci(sentence_data, runs, level='move', n_bootstrap=1000, seed=42):
    """
    Calculate bootstrap 95% CI for Krippendorff's alpha.

    Resamples sentences with replacement, recalculates alpha each time.
    This provides a measure of stability and precision for the alpha estimate,
    enabling significance testing via CI overlap comparison.

    Args:
        sentence_data: dict with 'move_preds' or 'step_preds' for all sentences
                      Format: {(article_id, sent_num): {'move_preds': [...], 'step_preds': [...]}}
        runs: list of run numbers (dynamically sized)
        level: 'move', 'step', or specific step label (e.g., '1a')
        n_bootstrap: number of bootstrap samples (default: 1000)
        seed: random seed for reproducibility (default: 42)

    Returns:
        (ci_lower, ci_upper) as floats, or (None, None) if calculation fails
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    sentence_keys = list(sentence_data.keys())

    # Check if we have enough sentences for reliable bootstrap
    if len(sentence_keys) < 10:
        return None, None

    bootstrap_alphas = []

    for bootstrap_iter in range(n_bootstrap):
        # Resample sentences WITH replacement
        # Use indices to sample, then map back to keys
        indices = np.random.choice(len(sentence_keys), size=len(sentence_keys), replace=True)
        resampled_keys = [sentence_keys[i] for i in indices]

        # Build matrix from resampled sentences
        if level == 'move':
            label_to_num = {"1": 1, "2": 2, "3": 3}
            matrix = []
            for run_idx in range(len(runs)):
                row = []
                for key in resampled_keys:
                    pred = sentence_data[key]['move_preds'][run_idx]
                    if pred is None:
                        row.append(np.nan)
                    else:
                        row.append(label_to_num.get(pred, np.nan))
                matrix.append(row)

        elif level == 'step':
            # Overall step level - get all unique steps
            all_steps = set()
            for key in resampled_keys:
                for pred in sentence_data[key]['step_preds']:
                    if pred is not None:
                        all_steps.add(pred)

            if len(all_steps) == 0:
                continue

            label_to_num = {step: i + 1 for i, step in enumerate(sorted(all_steps))}
            matrix = []
            for run_idx in range(len(runs)):
                row = []
                for key in resampled_keys:
                    pred = sentence_data[key]['step_preds'][run_idx]
                    if pred is None:
                        row.append(np.nan)
                    else:
                        row.append(label_to_num.get(pred, np.nan))
                matrix.append(row)

        else:
            # Per-step level - only use sentences that match the gold step
            filtered_keys = [k for k in resampled_keys if sentence_data[k]['gold_step'] == level]

            if len(filtered_keys) == 0:
                continue

            # Get all unique predicted steps for this gold step
            all_predicted_steps = set()
            for key in filtered_keys:
                for pred in sentence_data[key]['step_preds']:
                    if pred is not None:
                        all_predicted_steps.add(pred)

            if len(all_predicted_steps) == 0:
                continue

            label_to_num = {step: i + 1 for i, step in enumerate(sorted(all_predicted_steps))}
            matrix = []
            for run_idx in range(len(runs)):
                row = []
                for key in filtered_keys:
                    pred = sentence_data[key]['step_preds'][run_idx]
                    if pred is None:
                        row.append(np.nan)
                    else:
                        row.append(label_to_num.get(pred, np.nan))
                matrix.append(row)

        matrix = np.array(matrix)

        # Skip if matrix is too sparse or uniform
        non_nan_count = np.sum(~np.isnan(matrix))
        if non_nan_count < 10:
            continue

        # Calculate alpha for this bootstrap sample
        try:
            alpha = krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement='nominal'
            )
            if alpha is not None and not np.isnan(alpha):
                bootstrap_alphas.append(alpha)
        except:
            continue

    # Need sufficient successful bootstrap samples (at least 95% success rate)
    if len(bootstrap_alphas) < int(0.95 * n_bootstrap):
        return None, None

    # Calculate 95% CI (2.5th and 97.5th percentiles)
    ci_lower = float(np.percentile(bootstrap_alphas, 2.5))
    ci_upper = float(np.percentile(bootstrap_alphas, 97.5))

    return ci_lower, ci_upper


def calculate_krippendorff_alpha(temperature, runs, gold_standard):
    """
    Calculate Krippendorff's alpha using ALL available observations.

    Key change: Instead of requiring all runs to have the same articles,
    we use NaN for missing observations (failed parses). This maximizes
    data usage and is scientifically appropriate.

    Returns:
        {
            'temperature': 0.0,
            'move_level': {
                'alpha': 0.XX,
                'ci_lower': 0.XX,  # NEW: bootstrap 95% CI lower bound
                'ci_upper': 0.XX,  # NEW: bootstrap 95% CI upper bound
                'n_sentences': 1038,
                'n_observations': 20760,  # total non-NaN values
                'n_runs': 50  # dynamically sized
            },
            'step_level': {...},
            'per_step': {...}
        }
    """

    # STEP 1: Collect all predictions organized by sentence
    # Structure: {(article_id, sent_num): {'gold_move': '1', 'gold_step': '1a',
    #                                       'move_preds': [None, '1', '2', ...],
    #                                       'step_preds': [None, '1a', '1b', ...]}}
    sentence_data = {}

    # First pass: initialize all sentences from gold standard
    for article_id in sorted(gold_standard.keys()):
        gold_sentences = gold_standard[article_id]["sentences"]
        for gold_sent in gold_sentences:
            sent_num = gold_sent["sentence_num"]
            key = (article_id, sent_num)
            sentence_data[key] = {
                'gold_move': gold_sent["move"],
                'gold_step': gold_sent["step"],
                'move_preds': [None] * len(runs),
                'step_preds': [None] * len(runs)
            }

    # Second pass: fill in predictions from each run
    for run_idx, run_num in enumerate(runs):
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        for article_id in run_data["articles"]:
            if article_id not in gold_standard:
                continue

            pred_sentences = run_data["articles"][article_id]["predictions"]

            for pred_sent in pred_sentences:
                sent_num = pred_sent["sentence_num"]
                key = (article_id, sent_num)

                if key in sentence_data:
                    sentence_data[key]['move_preds'][run_idx] = pred_sent["predicted_move"]
                    sentence_data[key]['step_preds'][run_idx] = pred_sent["predicted_step"]

    # STEP 2: Calculate move-level alpha
    move_alpha = None
    move_ci_lower = None
    move_ci_upper = None
    n_move_observations = 0

    if len(sentence_data) > 0:
        # Create matrix: rows = runs (raters), cols = sentences (items)
        label_to_num = {"1": 1, "2": 2, "3": 3}
        move_matrix = []

        for run_idx in range(len(runs)):
            row = []
            for key in sorted(sentence_data.keys()):
                pred = sentence_data[key]['move_preds'][run_idx]
                if pred is None:
                    row.append(np.nan)
                else:
                    row.append(label_to_num.get(pred, np.nan))
            move_matrix.append(row)

        move_matrix = np.array(move_matrix)
        n_move_observations = int(np.sum(~np.isnan(move_matrix)))

        try:
            move_alpha = krippendorff.alpha(
                reliability_data=move_matrix,
                level_of_measurement='nominal'
            )
        except Exception as e:
            print(f"  ⚠ Move alpha calculation failed: {e}")
            move_alpha = None

        # Calculate bootstrap 95% confidence intervals
        if move_alpha is not None and not np.isnan(move_alpha):
            move_ci_lower, move_ci_upper = bootstrap_alpha_ci(
                sentence_data, runs, level='move', n_bootstrap=1000
            )

    # STEP 3: Calculate step-level alpha (overall)
    step_alpha = None
    step_ci_lower = None
    step_ci_upper = None
    n_step_observations = 0

    if len(sentence_data) > 0:
        # Get all unique steps
        all_steps = set()
        for data in sentence_data.values():
            for pred in data['step_preds']:
                if pred is not None:
                    all_steps.add(pred)

        step_to_num = {step: i + 1 for i, step in enumerate(sorted(all_steps))}

        # Create matrix
        step_matrix = []
        for run_idx in range(len(runs)):
            row = []
            for key in sorted(sentence_data.keys()):
                pred = sentence_data[key]['step_preds'][run_idx]
                if pred is None:
                    row.append(np.nan)
                else:
                    row.append(step_to_num.get(pred, np.nan))
            step_matrix.append(row)

        step_matrix = np.array(step_matrix)
        n_step_observations = int(np.sum(~np.isnan(step_matrix)))

        try:
            step_alpha = krippendorff.alpha(
                reliability_data=step_matrix,
                level_of_measurement='nominal'
            )
        except Exception as e:
            print(f"  ⚠ Step alpha calculation failed: {e}")
            step_alpha = None

        # Calculate bootstrap 95% confidence intervals
        if step_alpha is not None and not np.isnan(step_alpha):
            step_ci_lower, step_ci_upper = bootstrap_alpha_ci(
                sentence_data, runs, level='step', n_bootstrap=1000
            )

    # STEP 4: Calculate per-step alpha
    # Group sentences by their gold step
    sentences_by_gold_step = defaultdict(list)
    for key, data in sentence_data.items():
        sentences_by_gold_step[data['gold_step']].append(key)

    per_step_alphas = {}

    for gold_step, sentence_keys in sentences_by_gold_step.items():
        if len(sentence_keys) == 0:
            continue

        # Get all unique predicted steps for sentences with this gold step
        all_predicted_steps = set()
        for key in sentence_keys:
            for pred in sentence_data[key]['step_preds']:
                if pred is not None:
                    all_predicted_steps.add(pred)

        if len(all_predicted_steps) == 0:
            per_step_alphas[gold_step] = {
                "alpha": None,
                "ci_lower": None,
                "ci_upper": None,
                "n_sentences": len(sentence_keys),
                "n_observations": 0
            }
            continue

        label_to_num = {step: i + 1 for i, step in enumerate(sorted(all_predicted_steps))}

        # Create matrix for this gold step
        step_matrix = []
        for run_idx in range(len(runs)):
            row = []
            for key in sorted(sentence_keys):
                pred = sentence_data[key]['step_preds'][run_idx]
                if pred is None:
                    row.append(np.nan)
                else:
                    row.append(label_to_num.get(pred, np.nan))
            step_matrix.append(row)

        step_matrix = np.array(step_matrix)
        n_obs = int(np.sum(~np.isnan(step_matrix)))

        try:
            alpha = krippendorff.alpha(
                reliability_data=step_matrix,
                level_of_measurement='nominal'
            )

            # Calculate bootstrap CIs for per-step alpha
            ci_lower = None
            ci_upper = None
            if alpha is not None and not np.isnan(alpha):
                ci_lower, ci_upper = bootstrap_alpha_ci(
                    sentence_data, runs, level=gold_step, n_bootstrap=1000
                )

            per_step_alphas[gold_step] = {
                "alpha": float(alpha) if alpha is not None and not np.isnan(alpha) else None,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_sentences": len(sentence_keys),
                "n_observations": n_obs
            }
        except Exception as e:
            per_step_alphas[gold_step] = {
                "alpha": None,
                "ci_lower": None,
                "ci_upper": None,
                "n_sentences": len(sentence_keys),
                "n_observations": n_obs
            }

    return {
        "temperature": temperature,
        "move_level": {
            "alpha": (
                float(move_alpha)
                if move_alpha is not None and not np.isnan(move_alpha)
                else None
            ),
            "ci_lower": move_ci_lower,
            "ci_upper": move_ci_upper,
            "n_sentences": len(sentence_data),
            "n_observations": n_move_observations,
            "n_runs": len(runs),
        },
        "step_level": {
            "alpha": (
                float(step_alpha)
                if step_alpha is not None and not np.isnan(step_alpha)
                else None
            ),
            "ci_lower": step_ci_lower,
            "ci_upper": step_ci_upper,
            "n_sentences": len(sentence_data),
            "n_observations": n_step_observations,
            "n_runs": len(runs),
        },
        "per_step": per_step_alphas,
    }


def analyze_sentence_level_consistency(temperature, runs, gold_standard):
    """
    For each sentence, calculate:
    - Accuracy rate: % of runs that matched gold
    - Consistency rate: % of runs that agreed with modal prediction

    Returns sentence-level analysis data.
    """
    sentence_predictions = defaultdict(lambda: {"gold": None, "predictions": []})

    for run_num in runs:
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        for article_id in sorted(gold_standard.keys()):
            if article_id not in run_data["articles"]:
                continue

            gold_sentences = gold_standard[article_id]["sentences"]
            pred_sentences = run_data["articles"][article_id]["predictions"]

            for i, (gold_sent, pred_sent) in enumerate(
                zip(gold_sentences, pred_sentences)
            ):
                key = f"{article_id}_{i+1}"
                sentence_predictions[key]["gold"] = gold_sent["step"]
                sentence_predictions[key]["predictions"].append(
                    pred_sent["predicted_step"]
                )

    # Analyze each sentence
    categories = {"high": 0, "moderate": 0, "low": 0, "inconsistent": 0}
    examples = []

    for key, data in sentence_predictions.items():
        gold_step = data["gold"]
        predictions = data["predictions"]

        if not predictions:
            continue

        # Accuracy rate
        accuracy_rate = sum(1 for p in predictions if p == gold_step) / len(predictions)

        # Modal prediction and consistency rate
        pred_counts = Counter(predictions)
        modal_pred, modal_count = pred_counts.most_common(1)[0]
        consistency_rate = modal_count / len(predictions)

        # Categorize
        if consistency_rate >= 0.9:
            categories["high"] += 1
        elif consistency_rate >= 0.7:
            categories["moderate"] += 1
        elif consistency_rate >= 0.5:
            categories["low"] += 1
        else:
            categories["inconsistent"] += 1

        # Store example if inconsistent
        if consistency_rate < 0.5:
            article_id, sent_num = key.rsplit("_", 1)
            examples.append(
                {
                    "article_id": article_id,
                    "sentence_num": int(sent_num),
                    "gold_step": gold_step,
                    "modal_prediction": modal_pred,
                    "accuracy_rate": accuracy_rate,
                    "consistency_rate": consistency_rate,
                }
            )

    # Calculate distribution
    total = sum(categories.values())
    distribution = {
        cat: count / total if total > 0 else 0 for cat, count in categories.items()
    }

    return {
        "temperature": temperature,
        "distribution": distribution,
        "examples": examples[:10],
    }


def extract_step_info_from_gold(gold_standard):
    """
    Extract step descriptions and counts from gold standard.

    Returns:
        {
            '1a': {'description': 'Claim centrality', 'n': 82},
            '1b': {'description': 'Topic generalizations', 'n': 197},
            ...
        }
    """
    step_counts = Counter()
    step_descriptions = {
        "1a": "Claim centrality",
        "1b": "Topic generalizations",
        "1c": "Review previous research",
        "2a": "Counter-claiming",
        "2b": "Indicate gap",
        "2c": "Question-raising",
        "2d": "Continue tradition",
        "3a": "Outline purposes",
        "3b": "Announce research",
        "3c": "Announce findings",
        "3d": "Indicate structure",
    }

    for article in gold_standard.values():
        for sentence in article["sentences"]:
            step_counts[sentence["step"]] += 1

    return {
        step: {"description": step_descriptions.get(step, ""), "n": count}
        for step, count in step_counts.items()
    }


def generate_table2(accuracy_results, temperatures):
    """Generate Table 2: Move-Level Accuracy Across Temperature Settings"""
    rows = []

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        if temp_key in accuracy_results:
            result = accuracy_results[temp_key]
            move_acc = result["move_level"]["accuracy"]
            move_f1 = result["move_level"]["f1"]

            if move_acc and move_f1:
                rows.append(
                    {
                        "Temperature": temp,
                        "Mean Acc (%)": f"{move_acc['mean']*100:.2f}",
                        "95% CI": f"{move_acc['ci_95'][0]*100:.2f}-{move_acc['ci_95'][1]*100:.2f}",
                        "Weighted F1": f"{move_f1['mean']:.3f}",
                    }
                )

    return pd.DataFrame(rows)


def generate_table3(accuracy_results, temperatures, step_info):
    """Generate Table 3: Step-Level Accuracy (steps with n≥50 in full corpus only)"""

    # CaRS-50 corpus-level counts (50 articles, 1,297 sentences)
    # Steps with n<50 are excluded from reporting due to limited sample size
    #These counts need to fixed based on the test set corpus for actual reporting refer to data/processed/split_report.txt
    CORPUS_COUNTS = {
        "1a": 82,
        "1b": 197,
        "1c": 498,
        "2a": 27,  # Excluded: n<50
        "2b": 63,
        "2c": 24,  # Excluded: n<50
        "2d": 11,  # Excluded: n<50
        "3a": 72,
        "3b": 109,
        "3c": 72,
        "3d": 1,  # Excluded: n<50
    }

    # Filter to steps with n >= 50 in full corpus
    included_steps = {
        step: step_info.get(step, {"description": "", "n": 0})
        for step, count in CORPUS_COUNTS.items()
        if count >= 50
    }

    rows = []
    for step in sorted(included_steps.keys()):
        info = included_steps[step]
        row = {
            "Step": step,
            "Description": info["description"],
            "n": CORPUS_COUNTS[step],  # Use corpus-level count for reference
        }

        for temp in temperatures:
            temp_key = f"{temp:.1f}"
            if temp_key in accuracy_results:
                result = accuracy_results[temp_key]
                if "per_step" in result and step in result["per_step"]:
                    step_acc = result["per_step"][step]
                    if step_acc:
                        row[f"T={temp:.1f}"] = f"{step_acc['mean']*100:.1f}"
                    else:
                        row[f"T={temp:.1f}"] = "N/A"
                else:
                    row[f"T={temp:.1f}"] = "N/A"
            else:
                row[f"T={temp:.1f}"] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def generate_table4(alpha_results, temperatures):
    """
    Generate Table 4: Consistency Across Temperature Settings

    Includes:
    - Move-Level row (overall move consistency)
    - Step-Level rows (all 11 steps with their individual alphas)

    Format: α [CI_lower, CI_upper]
    Example: 0.949 [0.945, 0.953]
    """
    step_descriptions = {
        "1a": "Claim centrality",
        "1b": "Topic generalizations",
        "1c": "Review previous research",
        "2a": "Counter-claiming",
        "2b": "Indicate gap",
        "2c": "Question-raising",
        "2d": "Continue tradition",
        "3a": "Outline purposes",
        "3b": "Announce research",
        "3c": "Announce findings",
        "3d": "Indicate structure",
    }

    rows = []

    # MOVE-LEVEL ROW
    move_row = {
        "Level": "Move-level",
        "Step": "",
        "Description": "Overall move consistency",
        "n": "",
    }

    for temp in temperatures:
        temp_key = f"{temp:.1f}"
        if temp_key in alpha_results:
            result = alpha_results[temp_key]
            alpha = result["move_level"]["alpha"]
            ci_lower = result["move_level"].get("ci_lower")
            ci_upper = result["move_level"].get("ci_upper")

            if alpha is not None:
                if ci_lower is not None and ci_upper is not None:
                    move_row[f"T={temp:.1f}"] = f"{alpha:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                else:
                    move_row[f"T={temp:.1f}"] = f"{alpha:.3f}"
            else:
                move_row[f"T={temp:.1f}"] = "N/A"
        else:
            move_row[f"T={temp:.1f}"] = "N/A"

    rows.append(move_row)

    # STEP-LEVEL ROWS (all 11 steps)
    for step in sorted(step_descriptions.keys()):
        step_row = {
            "Level": "Step-level",
            "Step": step,
            "Description": step_descriptions[step],
            "n": "",  # Will be filled with n_sentences from first temp that has data
        }

        # Get n_sentences from first available temperature
        n_sentences = None
        for temp in temperatures:
            temp_key = f"{temp:.1f}"
            if temp_key in alpha_results:
                result = alpha_results[temp_key]
                if "per_step" in result and step in result["per_step"]:
                    n_sentences = result["per_step"][step]["n_sentences"]
                    break

        if n_sentences is not None:
            step_row["n"] = str(n_sentences)

        # Fill alpha values and CIs across temperatures
        for temp in temperatures:
            temp_key = f"{temp:.1f}"
            if temp_key in alpha_results:
                result = alpha_results[temp_key]
                if "per_step" in result and step in result["per_step"]:
                    alpha = result["per_step"][step]["alpha"]
                    ci_lower = result["per_step"][step].get("ci_lower")
                    ci_upper = result["per_step"][step].get("ci_upper")

                    if alpha is not None:
                        if ci_lower is not None and ci_upper is not None:
                            step_row[f"T={temp:.1f}"] = f"{alpha:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                        else:
                            step_row[f"T={temp:.1f}"] = f"{alpha:.3f}"
                    else:
                        step_row[f"T={temp:.1f}"] = "N/A"
                else:
                    step_row[f"T={temp:.1f}"] = "N/A"
            else:
                step_row[f"T={temp:.1f}"] = "N/A"

        rows.append(step_row)

    return pd.DataFrame(rows)


def generate_tableX(temperatures, runs):
    """Generate Table X: Parsing Success Rates by Temperature"""
    rows = []

    for temp in temperatures:
        total_successful = 0
        total_attempts = 0

        for run_num in runs:
            run_data = load_parsed_predictions(temp, run_num)
            if run_data and "metadata" in run_data:
                total_successful += run_data["metadata"]["successfully_parsed"]
                total_attempts += run_data["metadata"]["total_articles_attempted"]

        if total_attempts > 0:
            success_rate = total_successful / total_attempts
            rows.append(
                {
                    "Temperature": temp,
                    "Successful Parses": f"{total_successful}/{total_attempts}",
                    "Success Rate": f"{success_rate*100:.2f}%",
                }
            )

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("METRICS CALCULATION (FIXED VERSION)")
    print("=" * 60)
    print()
    print("Key improvement: Handles missing data at sentence-iteration level")
    print("instead of excluding entire articles.")
    print()

    # Load configuration
    config = load_config()
    dataset = config["dataset"]
    temperatures = config["temperatures"]
    runs = config["runs"]

    # Load gold standard
    print(f"Loading gold standard ({dataset})...")
    gold_standard = load_gold_standard(dataset)
    total_articles = len(gold_standard)
    total_sentences = sum(
        len(article["sentences"]) for article in gold_standard.values()
    )
    print(f"✓ {total_articles} articles, {total_sentences} sentences")
    print()

    # Extract step information
    step_info = extract_step_info_from_gold(gold_standard)

    # Create output directories
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Calculate accuracy by temperature
    print("Calculating accuracy...")
    accuracy_results = {}
    for temperature in temperatures:
        result = calculate_accuracy_by_temperature(temperature, runs, gold_standard)
        accuracy_results[f"{temperature:.1f}"] = result

        if result["move_level"]["accuracy"]:
            ml = result["move_level"]["accuracy"]
            f1 = result["move_level"]["f1"]["mean"]
            print(f"  temp_{temperature:.1f}: acc={ml['mean']*100:.1f}%, F1={f1:.3f}")

    with open(metrics_dir / "accuracy_by_temperature.json", "w") as f:
        json.dump(accuracy_results, f, indent=2)
    print(f"✓ Saved: results/metrics/accuracy_by_temperature.json")
    print()

    # Calculate Krippendorff's alpha
    print("Calculating Krippendorff's alpha...")
    print("(Using all available observations, NaN for missing data)")

    alpha_results = {}
    for temperature in temperatures:
        result = calculate_krippendorff_alpha(temperature, runs, gold_standard)
        alpha_results[f"{temperature:.1f}"] = result

        alpha = result["move_level"]["alpha"]
        n_obs = result["move_level"]["n_observations"]
        n_sent = result["move_level"]["n_sentences"]
        n_possible = n_sent * len(runs)

        print(f"  temp_{temperature:.1f}: {n_obs}/{n_possible} observations", end="")
        if alpha is not None:
            print(f" → move α={alpha:.3f}")
        else:
            print(f" → N/A")

    with open(metrics_dir / "krippendorff_alpha.json", "w") as f:
        json.dump(alpha_results, f, indent=2)
    print(f"✓ Saved: results/metrics/krippendorff_alpha.json")
    print()

    # Analyze sentence-level consistency
    print("Analyzing sentence-level consistency...")
    consistency_results = {}
    for temperature in temperatures:
        result = analyze_sentence_level_consistency(temperature, runs, gold_standard)
        consistency_results[f"{temperature:.1f}"] = result

    with open(metrics_dir / "sentence_level_analysis.json", "w") as f:
        json.dump(consistency_results, f, indent=2)
    print(f"✓ Saved: results/metrics/sentence_level_analysis.json")
    print()

    # Generate tables for manuscript
    print("Generating tables...")

    table2 = generate_table2(accuracy_results, temperatures)
    table2.to_csv(tables_dir / "table2_move_accuracy.csv", index=False)
    print(f"✓ Table 2: results/tables/table2_move_accuracy.csv")

    table3 = generate_table3(accuracy_results, temperatures, step_info)
    if not table3.empty:
        table3.to_csv(tables_dir / "table3_step_accuracy.csv", index=False)
        print(f"✓ Table 3: results/tables/table3_step_accuracy.csv")
    else:
        print(f"  ⚠ Table 3: No steps with n≥50 (skipped)")

    table4 = generate_table4(alpha_results, temperatures)
    table4.to_csv(tables_dir / "table4_consistency.csv", index=False)
    print(f"✓ Table 4: results/tables/table4_consistency.csv")

    tableX = generate_tableX(temperatures, runs)
    tableX.to_csv(tables_dir / "tableX_parsing_success.csv", index=False)
    print(f"✓ Table X: results/tables/tableX_parsing_success.csv")
    print()

    print("=" * 60)
    print("COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
