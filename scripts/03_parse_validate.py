"""
03_parse_validate.py

Purpose: Parse raw LLM outputs, validate against gold standard,
         and create combined parsed.json files.

Inputs:
    - outputs/temp_{X}/run_{YY}/raw/*.txt
    - data/processed/{dataset}/gold.json
    - config.yaml (for text_similarity_threshold)

Outputs:
    - outputs/temp_{X}/run_{YY}/parsed.json (only if validation passes)
    - results/validation_reports/temp_{X}_run_{YY}.txt (if validation fails)
"""

import os
import json
import yaml
import re
from pathlib import Path
from difflib import SequenceMatcher


def load_config():
    """Load and validate config.yaml"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_gold_standard(dataset):
    """Load gold standard annotations"""
    gold_path = Path(f"data/processed/{dataset}/gold.json")
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_llm_output(raw_text):
    """
    Parse LLM output using regex pattern: ^\[([a-z0-9]+)\]\s*(.+)$

    Returns:
        [
            {
                'sentence_num': 1,
                'predicted_step': '1b',
                'predicted_move': '1',
                'text': 'Central components...'
            },
            ...
        ]
    """
    lines = raw_text.strip().split("\n")
    parsed_sentences = []

    # Pattern: [step] text
    # Example: [1b] Central components of animal cognition...
    pattern = re.compile(r"^\[([a-z0-9]+)\]\s*(.+)$", re.IGNORECASE)

    sentence_num = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            step = match.group(1).lower()
            text = match.group(2).strip()

            # Extract move from step (first character)
            move = step[0] if step else ""

            parsed_sentences.append(
                {
                    "sentence_num": sentence_num,
                    "predicted_step": step,
                    "predicted_move": move,
                    "text": text,
                }
            )
            sentence_num += 1

    return parsed_sentences


def normalize_text(text):
    """
    Normalize text for comparison:
    - Strip whitespace
    - Lowercase
    - Remove extra spaces
    - Remove common punctuation variations

    Used for soft sentence matching.
    """
    text = text.strip().lower()
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Normalize dashes
    text = text.replace("--", "-")
    # Normalize quotes
    text = text.replace("''", '"').replace("``", '"')
    return text


def calculate_text_similarity(text1, text2):
    """
    Calculate similarity between two texts (0.0 to 1.0).

    Uses normalized Levenshtein distance:
        similarity = 1 - (edit_distance / max_length)

    Returns:
        float between 0.0 (completely different) and 1.0 (identical)
    """
    text1_norm = normalize_text(text1)
    text2_norm = normalize_text(text2)

    # Use SequenceMatcher for similarity
    similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    return similarity


def validate_parsed_output(parsed_sentences, gold_sentences, threshold=0.85):
    """
    Validate parsed output against gold standard.

    Checks:
    1. Sentence count match
    2. Text similarity for each sentence (above threshold)

    Returns:
        {
            'valid': True/False,
            'sentence_count_match': True/False,
            'sentence_count_expected': X,
            'sentence_count_actual': Y,
            'text_mismatches': [
                {
                    'sentence_num': 4,
                    'gold_text': '...',
                    'predicted_text': '...',
                    'similarity': 0.78
                },
                ...
            ]
        }
    """
    result = {
        "valid": True,
        "sentence_count_match": True,
        "sentence_count_expected": len(gold_sentences),
        "sentence_count_actual": len(parsed_sentences),
        "text_mismatches": [],
    }

    # Check sentence count
    if len(parsed_sentences) != len(gold_sentences):
        result["sentence_count_match"] = False
        result["valid"] = False
        return result

    # Check text similarity for each sentence
    for gold_sent, parsed_sent in zip(gold_sentences, parsed_sentences):
        similarity = calculate_text_similarity(gold_sent["text"], parsed_sent["text"])

        if similarity < threshold:
            result["text_mismatches"].append(
                {
                    "sentence_num": gold_sent["sentence_num"],
                    "gold_text": gold_sent["text"],
                    "predicted_text": parsed_sent["text"],
                    "similarity": similarity,
                }
            )
            result["valid"] = False

    return result


def create_combined_json(article_predictions, temperature, run_number, dataset, model):
    """
    Combine all articles from one run into single JSON:

    {
        "temperature": 0.0,
        "run_number": 1,
        "dataset": "test",
        "model": "gpt-4-turbo-preview",
        "articles": {
            "test001": {
                "predictions": [
                    {"sentence_num": 1, "predicted_step": "1b", "predicted_move": "1", "text": "..."},
                    ...
                ]
            },
            "test002": { ... },
            ...
        }
    }
    """
    combined = {
        "temperature": temperature,
        "run_number": run_number,
        "dataset": dataset,
        "model": model,
        "articles": {},
    }

    for article_id, predictions in article_predictions.items():
        combined["articles"][article_id] = {"predictions": predictions}

    return combined


def generate_validation_report(article_id, temperature, run_num, validation_result):
    """
    Generate human-readable validation report.

    Returns formatted string for the report.
    """
    report = []
    report.append("=" * 80)
    report.append("VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"Temperature: {temperature}")
    report.append(f"Run: {run_num:02d}")
    report.append(f"Article: {article_id}")
    report.append("")
    report.append(f"STATUS: {'PASSED' if validation_result['valid'] else 'FAILED'}")
    report.append("")

    # Sentence count
    report.append("SENTENCE COUNT:")
    report.append(f"  Expected: {validation_result['sentence_count_expected']}")
    report.append(f"  Actual: {validation_result['sentence_count_actual']}")
    report.append(
        f"  Match: {'YES' if validation_result['sentence_count_match'] else 'NO'}"
    )
    report.append("")

    # Text mismatches
    if validation_result["text_mismatches"]:
        report.append("TEXT MISMATCHES (similarity < threshold):")
        for mismatch in validation_result["text_mismatches"]:
            report.append(f"  Sentence {mismatch['sentence_num']}:")
            report.append(
                f"    Gold: \"{mismatch['gold_text'][:80]}{'...' if len(mismatch['gold_text']) > 80 else ''}\""
            )
            report.append(
                f"    Pred: \"{mismatch['predicted_text'][:80]}{'...' if len(mismatch['predicted_text']) > 80 else ''}\""
            )
            report.append(f"    Similarity: {mismatch['similarity']:.2f}")
            report.append("")

    if not validation_result["valid"]:
        report.append("ACTION REQUIRED:")
        report.append(f"  Re-run article {article_id} for run {run_num:02d}")

    return "\n".join(report)


def main():
    print("=" * 60)
    print("PARSING & VALIDATION")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()
    dataset = config["dataset"]
    temperatures = config["temperatures"]
    runs = config["runs"]
    threshold = config["text_similarity_threshold"]

    print("Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs: {runs}")
    print(f"  Text similarity threshold: {threshold}")
    print()

    # Load gold standard
    print("Loading gold standard...")
    gold_data = load_gold_standard(dataset)
    total_articles = len(gold_data)
    total_sentences = sum(len(article["sentences"]) for article in gold_data.values())
    print(
        f"✓ Loaded data/processed/{dataset}/gold.json ({total_articles} articles, {total_sentences} sentences)"
    )
    print()

    # Track parsing statistics by temperature
    parsing_stats = {}
    for temp in temperatures:
        parsing_stats[temp] = {
            "total_attempts": 0,  # Total article-level attempts (runs × articles)
            "successful_parses": 0,  # Successfully parsed articles
            "failed_parses": 0,  # Failed parsing
            "by_run": {},  # Track success per run
        }

    # Process each temperature and run
    for temperature in temperatures:
        print(f"Processing temperature {temperature}...")

        for run_num in runs:
            raw_dir = Path(f"outputs/temp_{temperature:.1f}/run_{run_num:02d}/raw")

            if not raw_dir.exists():
                print(f"  Run {run_num:02d}... ⚠ Raw outputs not found (skipped)")
                continue

            article_predictions = {}  # Successfully parsed articles
            failed_articles = []  # Track failures
            validation_reports = []

            # Process each article
            for article_id, gold_article in gold_data.items():
                parsing_stats[temperature]["total_attempts"] += 1

                raw_file = raw_dir / f"{article_id}.txt"

                if not raw_file.exists():
                    print(f"    {article_id}... ⚠ Raw file missing")
                    failed_articles.append(article_id)
                    parsing_stats[temperature]["failed_parses"] += 1
                    continue

                # Load and parse raw output
                with open(raw_file, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                parsed_sentences = parse_llm_output(raw_text)

                # Validate
                validation_result = validate_parsed_output(
                    parsed_sentences, gold_article["sentences"], threshold
                )

                if validation_result["valid"]:
                    article_predictions[article_id] = parsed_sentences
                    parsing_stats[temperature]["successful_parses"] += 1
                else:
                    failed_articles.append(article_id)
                    parsing_stats[temperature]["failed_parses"] += 1

                    # Generate validation report
                    report = generate_validation_report(
                        article_id, temperature, run_num, validation_result
                    )
                    validation_reports.append((article_id, report))

            # ALWAYS create parsed.json with successful articles
            if article_predictions:  # At least some articles parsed successfully
                combined_data = create_combined_json(
                    article_predictions, temperature, run_num, dataset, config["model"]
                )

                # Add metadata about exclusions
                combined_data["metadata"] = {
                    "total_articles_attempted": total_articles,
                    "successfully_parsed": len(article_predictions),
                    "failed_parsing": len(failed_articles),
                    "excluded_articles": failed_articles,
                }

                parsed_path = Path(
                    f"outputs/temp_{temperature:.1f}/run_{run_num:02d}/parsed.json"
                )
                with open(parsed_path, "w", encoding="utf-8") as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)

                # Track for this run
                parsing_stats[temperature]["by_run"][run_num] = {
                    "successful": len(article_predictions),
                    "failed": len(failed_articles),
                }

                if failed_articles:
                    print(
                        f"  Run {run_num:02d}... ⚠ {len(article_predictions)}/{total_articles} articles parsed (excluded: {len(failed_articles)})"
                    )
                else:
                    print(
                        f"  Run {run_num:02d}... ✓ {len(article_predictions)}/{total_articles} articles parsed"
                    )
            else:
                print(
                    f"  Run {run_num:02d}... ❌ Complete failure - no articles parsed"
                )
                parsing_stats[temperature]["by_run"][run_num] = {
                    "successful": 0,
                    "failed": total_articles,
                }

            # Save validation reports for failed articles (combined into one file per run)
            if validation_reports:
                report_dir = Path("results/validation_reports")
                report_dir.mkdir(parents=True, exist_ok=True)

                # Combine all reports for this run into a single file
                combined_report = []
                combined_report.append("=" * 80)
                combined_report.append(f"COMBINED VALIDATION REPORT - Temperature {temperature:.1f}, Run {run_num:02d}")
                combined_report.append("=" * 80)
                combined_report.append(f"Total Failed Articles: {len(validation_reports)}")
                combined_report.append("")

                for i, (article_id, report) in enumerate(validation_reports, 1):
                    if i > 1:
                        combined_report.append("\n" + "=" * 80 + "\n")
                    combined_report.append(report)

                report_path = report_dir / f"temp_{temperature:.1f}_run_{run_num:02d}.txt"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(combined_report))

        print()

    # Save parsing statistics
    stats_dir = Path("results/parsing_stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Calculate success rates by temperature
    parsing_summary = {}
    for temp in temperatures:
        stats = parsing_stats[temp]
        parsing_summary[f"{temp:.1f}"] = {
            "total_attempts": stats["total_attempts"],
            "successful_parses": stats["successful_parses"],
            "failed_parses": stats["failed_parses"],
            "success_rate": (
                stats["successful_parses"] / stats["total_attempts"]
                if stats["total_attempts"] > 0
                else 0
            ),
            "by_run": stats["by_run"],
        }

    with open(stats_dir / "parsing_reliability.json", "w") as f:
        json.dump(parsing_summary, f, indent=2)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Parsing statistics saved to: results/parsing_stats/parsing_reliability.json"
    )
    print()
    print("Success rates by temperature:")
    for temp in temperatures:
        stats = parsing_stats[temp]
        rate = stats["successful_parses"] / stats["total_attempts"] * 100
        print(
            f"  temp_{temp:.1f}: {stats['successful_parses']}/{stats['total_attempts']} ({rate:.1f}%)"
        )
    print()
    print("COMPLETE ✓")


if __name__ == "__main__":
    main()
