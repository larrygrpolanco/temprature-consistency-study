"""
04_calculate_metrics.py

Purpose: Calculate accuracy, consistency, and Krippendorff's alpha
         from parsed predictions.

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


def get_common_articles(temperatures, runs):
    """
    Identify articles that were successfully parsed in ALL runs for each temperature.

    Returns:
        dict: {temperature: [list of article_ids present in all runs]}
    """
    articles_by_temp = {}

    for temperature in temperatures:
        articles_per_run = []

        for run_num in runs:
            run_data = load_parsed_predictions(temperature, run_num)
            if run_data is None:
                articles_per_run.append(set())
            else:
                articles_per_run.append(set(run_data["articles"].keys()))

        # Find intersection: articles present in ALL runs
        if articles_per_run:
            common_articles = (
                set.intersection(*articles_per_run) if articles_per_run else set()
            )
            articles_by_temp[temperature] = sorted(list(common_articles))
        else:
            articles_by_temp[temperature] = []

    return articles_by_temp


def calculate_krippendorff_alpha(temperature, runs, gold_standard, common_articles):
    """
    Calculate Krippendorff's alpha for one temperature using only articles
    present in all runs.

    Now calculates:
    - Overall move-level alpha
    - Overall step-level alpha
    - Per-step alpha (separate alpha for each of 11 steps)

    Returns:
        {
            'temperature': 0.0,
            'move_level': {
                'alpha': 0.XX,
                'n_sentences': 1038,
                'n_runs': 50,
                'n_articles': 40
            },
            'step_level': {
                'alpha': 0.XX,
                'n_sentences': 1038,
                'n_runs': 50,
                'n_articles': 40
            },
            'per_step': {
                '1a': {'alpha': 0.XX, 'n_sentences': 82},
                '1b': {'alpha': 0.XX, 'n_sentences': 197},
                ...
            }
        }
    """
    if len(common_articles) == 0:
        return {
            "temperature": temperature,
            "move_level": {
                "alpha": None,
                "n_sentences": 0,
                "n_runs": 0,
                "n_articles": 0,
            },
            "step_level": {
                "alpha": None,
                "n_sentences": 0,
                "n_runs": 0,
                "n_articles": 0,
            },
            "per_step": {},
        }

    # Collect predictions at both move and step level
    move_predictions = []
    step_predictions = []

    # For per-step alpha: collect predictions organized by step
    # Structure: {step: [[run1_preds], [run2_preds], ...]}
    step_predictions_by_step = defaultdict(lambda: [[] for _ in runs])

    for run_idx, run_num in enumerate(runs):
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        run_moves = []
        run_steps = []

        # Process only common articles in sorted order
        for article_id in sorted(common_articles):
            if (
                article_id not in run_data["articles"]
                or article_id not in gold_standard
            ):
                continue

            gold_sentences = gold_standard[article_id]["sentences"]
            predictions = run_data["articles"][article_id]["predictions"]

            for gold_sent, pred_sent in zip(gold_sentences, predictions):
                pred_move = pred_sent["predicted_move"]
                pred_step = pred_sent["predicted_step"]
                gold_step = gold_sent["step"]

                run_moves.append(pred_move)
                run_steps.append(pred_step)

                # Collect for per-step alpha calculation
                # Key insight: group by GOLD step to measure consistency within each category
                step_predictions_by_step[gold_step][run_idx].append(pred_step)

        move_predictions.append(run_moves)
        step_predictions.append(run_steps)

    # Calculate alpha for moves
    move_alpha = None
    if len(move_predictions) >= 2 and len(move_predictions[0]) > 0:
        label_to_num = {"1": 1, "2": 2, "3": 3}
        move_matrix = np.array(
            [[label_to_num.get(m, 0) for m in run] for run in move_predictions]
        )
        try:
            move_alpha = krippendorff.alpha(
                reliability_data=move_matrix, level_of_measurement="nominal"
            )
        except:
            move_alpha = None

    # Calculate alpha for steps (overall)
    step_alpha = None
    if len(step_predictions) >= 2 and len(step_predictions[0]) > 0:
        all_steps = set()
        for run in step_predictions:
            all_steps.update(run)
        step_to_num = {step: i + 1 for i, step in enumerate(sorted(all_steps))}
        step_matrix = np.array(
            [[step_to_num.get(s, 0) for s in run] for run in step_predictions]
        )
        try:
            step_alpha = krippendorff.alpha(
                reliability_data=step_matrix, level_of_measurement="nominal"
            )
        except:
            step_alpha = None

    # Calculate per-step alphas
    per_step_alphas = {}
    for step, runs_predictions in step_predictions_by_step.items():
        # Filter out empty runs
        runs_predictions = [run for run in runs_predictions if len(run) > 0]

        if len(runs_predictions) >= 2 and len(runs_predictions[0]) > 0:
            # Get all unique step labels that appear in predictions for this gold step
            all_labels = set()
            for run in runs_predictions:
                all_labels.update(run)
            label_to_num = {label: i + 1 for i, label in enumerate(sorted(all_labels))}

            # Create reliability matrix: rows = runs, cols = sentences with this gold step
            step_matrix = np.array(
                [[label_to_num.get(s, 0) for s in run] for run in runs_predictions]
            )

            try:
                alpha = krippendorff.alpha(
                    reliability_data=step_matrix, level_of_measurement="nominal"
                )
                per_step_alphas[step] = {
                    "alpha": (
                        float(alpha)
                        if alpha is not None and not np.isnan(alpha)
                        else None
                    ),
                    "n_sentences": len(runs_predictions[0]),
                }
            except:
                per_step_alphas[step] = {
                    "alpha": None,
                    "n_sentences": len(runs_predictions[0]) if runs_predictions else 0,
                }
        else:
            per_step_alphas[step] = {
                "alpha": None,
                "n_sentences": len(runs_predictions[0]) if runs_predictions else 0,
            }

    n_sentences = len(move_predictions[0]) if move_predictions else 0

    return {
        "temperature": temperature,
        "move_level": {
            "alpha": (
                float(move_alpha)
                if move_alpha is not None and not np.isnan(move_alpha)
                else None
            ),
            "n_sentences": n_sentences,
            "n_runs": len(move_predictions),
            "n_articles": len(common_articles),
        },
        "step_level": {
            "alpha": (
                float(step_alpha)
                if step_alpha is not None and not np.isnan(step_alpha)
                else None
            ),
            "n_sentences": n_sentences,
            "n_runs": len(step_predictions),
            "n_articles": len(common_articles),
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
            if alpha is not None:
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

        # Fill alpha values across temperatures
        for temp in temperatures:
            temp_key = f"{temp:.1f}"
            if temp_key in alpha_results:
                result = alpha_results[temp_key]
                if "per_step" in result and step in result["per_step"]:
                    alpha = result["per_step"][step]["alpha"]
                    if alpha is not None:
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
    print("METRICS CALCULATION")
    print("=" * 60)
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
    common_articles_by_temp = get_common_articles(temperatures, runs)

    alpha_results = {}
    for temperature in temperatures:
        common_articles = common_articles_by_temp[temperature]
        print(
            f"  temp_{temperature:.1f}: {len(common_articles)}/{total_articles} articles in all runs",
            end="",
        )

        result = calculate_krippendorff_alpha(
            temperature, runs, gold_standard, common_articles
        )
        alpha_results[f"{temperature:.1f}"] = result

        alpha = result["move_level"]["alpha"]
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
