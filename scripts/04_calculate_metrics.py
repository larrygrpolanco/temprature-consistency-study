#!/usr/bin/env python3
"""
04_calculate_metrics.py

Purpose: Calculate accuracy, consistency, and Krippendorff's alpha
         from parsed predictions.

Inputs:
    - outputs/temp_{X}/run_{YY}/parsed.json (all runs for all temperatures)
    - data/processed/{dataset}/gold.json

Outputs:
    - results/metrics/accuracy_by_temperature.json
    - results/metrics/accuracy_by_step.json
    - results/metrics/consistency_by_temperature.json
    - results/metrics/krippendorff_alpha.json
    - results/metrics/sentence_level_analysis.json
"""

import json
import yaml
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
import krippendorff


def load_config():
    """Load config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_gold_standard(dataset):
    """Load gold standard annotations"""
    gold_path = Path(f"data/processed/{dataset}/gold.json")
    with open(gold_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_parsed_predictions(temperature, run_num):
    """Load parsed predictions for a specific temperature and run"""
    parsed_path = Path(f"outputs/temp_{temperature:.1f}/run_{run_num:02d}/parsed.json")
    if not parsed_path.exists():
        return None

    with open(parsed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy_single_run(run_predictions, gold_standard):
    """
    Calculate accuracy for one run at move and step levels.

    Returns:
        {
            'move_accuracy': 0.XX,
            'step_accuracy': 0.XX,
            'per_move_accuracy': {'1': 0.XX, '2': 0.XX, '3': 0.XX},
            'per_step_accuracy': {'1a': 0.XX, '1b': 0.XX, ...}
        }
    """
    move_correct = 0
    step_correct = 0
    total = 0

    # Per-move and per-step counters
    per_move = defaultdict(lambda: {'correct': 0, 'total': 0})
    per_step = defaultdict(lambda: {'correct': 0, 'total': 0})

    for article_id, article_data in run_predictions['articles'].items():
        gold_sentences = gold_standard[article_id]['sentences']
        pred_sentences = article_data['predictions']

        for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
            gold_move = gold_sent['move']
            gold_step = gold_sent['step']
            pred_move = pred_sent['predicted_move']
            pred_step = pred_sent['predicted_step']

            # Move accuracy
            if gold_move == pred_move:
                move_correct += 1
                per_move[gold_move]['correct'] += 1
            per_move[gold_move]['total'] += 1

            # Step accuracy
            if gold_step == pred_step:
                step_correct += 1
                per_step[gold_step]['correct'] += 1
            per_step[gold_step]['total'] += 1

            total += 1

    # Calculate accuracies
    move_accuracy = move_correct / total if total > 0 else 0
    step_accuracy = step_correct / total if total > 0 else 0

    per_move_accuracy = {
        move: counts['correct'] / counts['total']
        for move, counts in per_move.items()
    }

    per_step_accuracy = {
        step: counts['correct'] / counts['total']
        for step, counts in per_step.items()
    }

    return {
        'move_accuracy': move_accuracy,
        'step_accuracy': step_accuracy,
        'per_move_accuracy': per_move_accuracy,
        'per_step_accuracy': per_step_accuracy
    }


def calculate_accuracy_by_temperature(temperature, runs, gold_standard):
    """
    Aggregate accuracy across all runs for one temperature.

    Returns:
        {
            'temperature': 0.0,
            'n_runs': 50,
            'move_level': {
                'mean_accuracy': 0.XX,
                'std': 0.XX,
                'min': 0.XX,
                'max': 0.XX,
                'ci_95': [0.XX, 0.XX]
            },
            'step_level': { ... }
        }
    """
    move_accuracies = []
    step_accuracies = []
    per_move_accs = defaultdict(list)
    per_step_accs = defaultdict(list)

    for run_num in runs:
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        accuracy_result = calculate_accuracy_single_run(run_data, gold_standard)

        move_accuracies.append(accuracy_result['move_accuracy'])
        step_accuracies.append(accuracy_result['step_accuracy'])

        for move, acc in accuracy_result['per_move_accuracy'].items():
            per_move_accs[move].append(acc)

        for step, acc in accuracy_result['per_step_accuracy'].items():
            per_step_accs[step].append(acc)

    # Calculate statistics
    def calc_stats(values):
        if not values:
            return None
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
        return {
            'mean_accuracy': float(mean),
            'std': float(std),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'ci_95': [float(ci_95[0]), float(ci_95[1])],
            'n_runs': len(values)
        }

    result = {
        'temperature': temperature,
        'move_level': calc_stats(move_accuracies),
        'step_level': calc_stats(step_accuracies)
    }

    # Add per-move and per-step breakdowns
    result['per_move'] = {
        move: calc_stats(accs)
        for move, accs in per_move_accs.items()
    }

    result['per_step'] = {
        step: calc_stats(accs)
        for step, accs in per_step_accs.items()
    }

    return result


def prepare_krippendorff_data(temperatures, runs, gold_standard):
    """
    Prepare data matrix for Krippendorff's alpha calculation.

    Returns:
        Dictionary with temperature as key and reliability matrix as value.
        Matrix format: rows = raters (runs), columns = units (sentences)
    """
    results = {}

    for temperature in temperatures:
        # Collect all predictions across runs
        all_predictions = []

        for run_num in runs:
            run_data = load_parsed_predictions(temperature, run_num)
            if run_data is None:
                continue

            # Flatten predictions across all articles in order
            run_predictions = []
            for article_id in sorted(gold_standard.keys()):
                if article_id in run_data['articles']:
                    predictions = run_data['articles'][article_id]['predictions']
                    for pred in predictions:
                        run_predictions.append(pred['predicted_step'])

            all_predictions.append(run_predictions)

        # Convert to numpy array (runs × sentences)
        if all_predictions:
            results[temperature] = all_predictions

    return results


def calculate_krippendorff_alpha(temperature, runs, gold_standard):
    """
    Calculate Krippendorff's alpha for one temperature.

    Returns:
        {
            'temperature': 0.0,
            'move_level': {
                'alpha': 0.XX,
                'n_sentences': 1038,
                'n_runs': 50
            },
            'step_level': {
                'alpha': 0.XX,
                'n_sentences': 1038,
                'n_runs': 50
            }
        }
    """
    # Collect predictions at both move and step level
    move_predictions = []
    step_predictions = []

    for run_num in runs:
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        run_moves = []
        run_steps = []

        for article_id in sorted(gold_standard.keys()):
            if article_id in run_data['articles']:
                predictions = run_data['articles'][article_id]['predictions']
                for pred in predictions:
                    run_moves.append(pred['predicted_move'])
                    run_steps.append(pred['predicted_step'])

        move_predictions.append(run_moves)
        step_predictions.append(run_steps)

    # Calculate alpha for moves
    move_alpha = None
    if move_predictions:
        # Convert labels to numeric
        label_to_num = {'1': 1, '2': 2, '3': 3}
        move_matrix = []
        for run in move_predictions:
            move_matrix.append([label_to_num.get(m, 0) for m in run])
        move_matrix = np.array(move_matrix)

        try:
            move_alpha = krippendorff.alpha(reliability_data=move_matrix, level_of_measurement='nominal')
        except:
            move_alpha = None

    # Calculate alpha for steps
    step_alpha = None
    if step_predictions:
        # Convert step labels to numeric
        all_steps = set()
        for run in step_predictions:
            all_steps.update(run)
        step_to_num = {step: i+1 for i, step in enumerate(sorted(all_steps))}

        step_matrix = []
        for run in step_predictions:
            step_matrix.append([step_to_num.get(s, 0) for s in run])
        step_matrix = np.array(step_matrix)

        try:
            step_alpha = krippendorff.alpha(reliability_data=step_matrix, level_of_measurement='nominal')
        except:
            step_alpha = None

    return {
        'temperature': temperature,
        'move_level': {
            'alpha': float(move_alpha) if move_alpha is not None else None,
            'n_sentences': len(move_predictions[0]) if move_predictions else 0,
            'n_runs': len(move_predictions)
        },
        'step_level': {
            'alpha': float(step_alpha) if step_alpha is not None else None,
            'n_sentences': len(step_predictions[0]) if step_predictions else 0,
            'n_runs': len(step_predictions)
        }
    }


def analyze_sentence_level_consistency(temperature, runs, gold_standard):
    """
    For each sentence, calculate:
    - Accuracy rate: % of runs that matched gold
    - Consistency rate: % of runs that agreed with modal prediction

    Returns sentence-level analysis data.
    """
    # Collect predictions for each sentence
    sentence_predictions = defaultdict(lambda: {'gold': None, 'predictions': []})

    for run_num in runs:
        run_data = load_parsed_predictions(temperature, run_num)
        if run_data is None:
            continue

        for article_id in sorted(gold_standard.keys()):
            if article_id not in run_data['articles']:
                continue

            gold_sentences = gold_standard[article_id]['sentences']
            pred_sentences = run_data['articles'][article_id]['predictions']

            for i, (gold_sent, pred_sent) in enumerate(zip(gold_sentences, pred_sentences)):
                key = f"{article_id}_{i+1}"
                sentence_predictions[key]['gold'] = gold_sent['step']
                sentence_predictions[key]['predictions'].append(pred_sent['predicted_step'])

    # Analyze each sentence
    categories = {'high': 0, 'moderate': 0, 'low': 0, 'inconsistent': 0}
    examples = []

    for key, data in sentence_predictions.items():
        gold_step = data['gold']
        predictions = data['predictions']

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
            categories['high'] += 1
        elif consistency_rate >= 0.7:
            categories['moderate'] += 1
        elif consistency_rate >= 0.5:
            categories['low'] += 1
        else:
            categories['inconsistent'] += 1

        # Store example if inconsistent
        if consistency_rate < 0.5:
            article_id, sent_num = key.rsplit('_', 1)
            examples.append({
                'article_id': article_id,
                'sentence_num': int(sent_num),
                'gold_step': gold_step,
                'modal_prediction': modal_pred,
                'accuracy_rate': accuracy_rate,
                'consistency_rate': consistency_rate
            })

    # Calculate distribution
    total = sum(categories.values())
    distribution = {
        cat: count / total if total > 0 else 0
        for cat, count in categories.items()
    }

    return {
        'temperature': temperature,
        'distribution': distribution,
        'examples': examples[:10]  # Top 10 inconsistent examples
    }


def main():
    print("=" * 60)
    print("METRICS CALCULATION")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()
    dataset = config['dataset']
    temperatures = config['temperatures']
    runs = config['runs']

    # Load gold standard
    print("Loading gold standard...")
    gold_standard = load_gold_standard(dataset)
    total_articles = len(gold_standard)
    total_sentences = sum(len(article['sentences']) for article in gold_standard.values())
    print(f"✓ Loaded data/processed/{dataset}/gold.json ({total_articles} articles, {total_sentences} sentences)")
    print()

    # Check available parsed predictions
    print("Checking parsed predictions...")
    for temperature in temperatures:
        available = 0
        for run_num in runs:
            if load_parsed_predictions(temperature, run_num) is not None:
                available += 1
        print(f"✓ temp_{temperature:.1f}: {available}/{len(runs)} runs available")
    print()

    # Create output directory
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Calculate accuracy by temperature
    print("Calculating accuracy...")
    accuracy_results = {}
    for temperature in temperatures:
        result = calculate_accuracy_by_temperature(temperature, runs, gold_standard)
        accuracy_results[f"{temperature:.1f}"] = result

        if result['move_level']:
            ml = result['move_level']
            print(f"  temp_{temperature:.1f}: mean={ml['mean_accuracy']:.3f} (std={ml['std']:.3f}) [95% CI: {ml['ci_95'][0]:.3f}-{ml['ci_95'][1]:.3f}]")

    # Save accuracy results
    with open(metrics_dir / "accuracy_by_temperature.json", 'w') as f:
        json.dump(accuracy_results, f, indent=2)
    print(f"✓ Saved: results/metrics/accuracy_by_temperature.json")
    print()

    # Calculate Krippendorff's alpha
    print("Calculating Krippendorff's alpha...")
    alpha_results = {}
    for temperature in temperatures:
        result = calculate_krippendorff_alpha(temperature, runs, gold_standard)
        alpha_results[f"{temperature:.1f}"] = result

        move_alpha = result['move_level']['alpha']
        step_alpha = result['step_level']['alpha']
        print(f"  temp_{temperature:.1f}: move α={move_alpha:.3f if move_alpha else 'N/A'}, step α={step_alpha:.3f if step_alpha else 'N/A'}")

    # Save alpha results
    with open(metrics_dir / "krippendorff_alpha.json", 'w') as f:
        json.dump(alpha_results, f, indent=2)
    print(f"✓ Saved: results/metrics/krippendorff_alpha.json")
    print()

    # Analyze sentence-level consistency
    print("Analyzing sentence-level consistency...")
    consistency_results = {}
    for temperature in temperatures:
        result = analyze_sentence_level_consistency(temperature, runs, gold_standard)
        consistency_results[f"{temperature:.1f}"] = result

    # Save consistency results
    with open(metrics_dir / "sentence_level_analysis.json", 'w') as f:
        json.dump(consistency_results, f, indent=2)
    print(f"✓ Saved: results/metrics/sentence_level_analysis.json")
    print()

    print("=" * 60)
    print("COMPLETE ✓")
    print("All metrics saved to results/metrics/")
    print("=" * 60)


if __name__ == "__main__":
    main()
