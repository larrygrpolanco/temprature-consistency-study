"""
01_prepare_dataset.py

Purpose: Convert raw CaRS-50 XML files into structured gold standard format
         with stratified train/test split.

Inputs:
    - data/raw_xml/text001.xml through text050.xml

Outputs:
    - data/processed/validation/gold.json (validation articles)
    - data/processed/validation/input/*.txt (plain text files)
    - data/processed/test/gold.json (test articles)
    - data/processed/test/input/*.txt (plain text files)
    - data/processed/split_report.txt
    - data/processed/exclusions.txt
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import random


# Articles excluded due to sentence boundary inconsistencies in gold standard
EXCLUDED_ARTICLES = {
    "text018": 'Sentence split at "U.S." / "Fish and Wildlife Service"',
    "text027": "Two sentences combined as one",
    "text028": "Two sentences combined as one",
    "text040": "Two sentences combined as one, missing period",
}


def parse_xml(xml_path):
    """
    Extract sentences and annotations from CaRS-50 XML.

    Returns:
        {
            'article_id': 'text001',
            'title': '...',
            'sentences': [
                {
                    'sentence_num': 1,
                    'sentence_id': 't001s0001',
                    'text': 'Central components...',
                    'step': '1b',
                    'move': '1'
                },
                ...
            ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract article ID from filename
    article_id = Path(xml_path).stem

    # Get title (if available)
    title_elem = root.find(".//title")
    title = title_elem.text if title_elem is not None and title_elem.text else ""

    sentences = []

    # Parse all sentences
    for sent_elem in root.findall(".//sentence"):
        # Extract nested elements
        sentence_id_elem = sent_elem.find("sentenceID")
        text_elem = sent_elem.find("text")
        step_elem = sent_elem.find("step")

        sentence_id = (
            sentence_id_elem.text.strip()
            if sentence_id_elem is not None and sentence_id_elem.text
            else ""
        )
        text = (
            text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        )
        step = (
            step_elem.text.strip() if step_elem is not None and step_elem.text else ""
        )

        # Extract move from step (e.g., "1a" -> "1", "2b" -> "2")
        move = ""
        if step:
            # Extract numeric part from step
            match = re.match(r"^(\d+)", step)
            if match:
                move = match.group(1)

        # Extract sentence number from ID (e.g., 't001s0001' -> 1)
        try:
            sentence_num = int(sentence_id[-4:])
        except (ValueError, IndexError):
            sentence_num = len(sentences) + 1

        sentences.append(
            {
                "sentence_num": sentence_num,
                "sentence_id": sentence_id,
                "text": text,
                "step": step,
                "move": move,
            }
        )

    # Sort by sentence number
    sentences.sort(key=lambda x: x["sentence_num"])

    return {"article_id": article_id, "title": title, "sentences": sentences}


def stratified_split(articles, validation_size=6, random_seed=42):
    """
    Perform stratified sampling to ensure move/step distribution
    is maintained in both validation and test sets.

    Returns:
        (validation_articles, test_articles)
    """
    random.seed(random_seed)

    # Calculate move/step distributions for each article
    article_stats = []
    for article in articles:
        move_counts = Counter(s["move"] for s in article["sentences"])
        step_counts = Counter(s["step"] for s in article["sentences"])
        article_stats.append(
            {
                "article": article,
                "move_counts": move_counts,
                "step_counts": step_counts,
                "total_sentences": len(article["sentences"]),
            }
        )

    # Sort by total sentences to ensure balanced splits
    article_stats.sort(key=lambda x: x["total_sentences"])

    # Stratified split: distribute articles evenly across validation/test
    # Take every Nth article for validation to maintain distribution
    n = len(article_stats) // validation_size
    validation_indices = [i * n for i in range(validation_size)]

    validation_articles = [article_stats[i]["article"] for i in validation_indices]
    test_articles = [
        article_stats[i]["article"]
        for i in range(len(article_stats))
        if i not in validation_indices
    ]

    # Shuffle within each split
    random.shuffle(validation_articles)
    random.shuffle(test_articles)

    return validation_articles, test_articles


def rename_articles(articles, prefix):
    """
    Rename articles: text001 → validation001 or test001

    Returns:
        List of renamed article dicts
    """
    renamed = []
    for idx, article in enumerate(articles, 1):
        new_id = f"{prefix}{idx:03d}"
        article_copy = article.copy()
        article_copy["original_id"] = article["article_id"]
        article_copy["article_id"] = new_id
        renamed.append(article_copy)
    return renamed


def create_gold_json(articles):
    """
    Combine all articles into single gold.json structure:

    {
        "validation001": {
            "sentences": [
                {"sentence_num": 1, "step": "1b", "move": "1", "text": "..."},
                ...
            ]
        },
        "validation002": { ... },
        ...
    }
    """
    gold_data = {}

    for article in articles:
        article_id = article["article_id"]
        gold_data[article_id] = {
            "original_id": article.get("original_id", article_id),
            "sentences": [
                {
                    "sentence_num": s["sentence_num"],
                    "step": s["step"],
                    "move": s["move"],
                    "text": s["text"],
                }
                for s in article["sentences"]
            ],
        }

    return gold_data


def save_input_txt(article, output_path):
    """
    Save plain text (one sentence per line) for API input.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in article["sentences"]:
            f.write(sentence["text"] + "\n")


def save_exclusions_report(excluded_dict, output_path):
    """
    Document excluded articles with reasons.
    """
    report = []
    report.append("=" * 80)
    report.append("EXCLUDED ARTICLES")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total excluded: {len(excluded_dict)}")
    report.append("")
    report.append("Articles excluded due to sentence boundary inconsistencies:")
    report.append("-" * 80)

    for article_id in sorted(excluded_dict.keys()):
        reason = excluded_dict[article_id]
        report.append(f"\n{article_id}:")
        report.append(f"  Reason: {reason}")

    report.append("")
    report.append("=" * 80)
    report.append("These exclusions ensure reliable alignment between LLM outputs and")
    report.append("gold-standard annotations during automated evaluation.")
    report.append("=" * 80)
    report.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def generate_split_report(
    validation_articles, test_articles, excluded_count, output_path
):
    """
    Document stratification results:
    - Original article IDs and their new names
    - Move/step distribution in each split
    - Comparison to overall corpus distribution

    Save to data/processed/split_report.txt
    """

    def get_distribution(articles):
        all_sentences = [s for a in articles for s in a["sentences"]]
        move_counts = Counter(s["move"] for s in all_sentences)
        step_counts = Counter(s["step"] for s in all_sentences)
        return move_counts, step_counts, len(all_sentences)

    val_move, val_step, val_total = get_distribution(validation_articles)
    test_move, test_step, test_total = get_distribution(test_articles)

    all_articles = validation_articles + test_articles
    all_move, all_step, all_total = get_distribution(all_articles)

    report = []
    report.append("=" * 80)
    report.append("STRATIFIED SPLIT REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"CaRS-50 Corpus: 50 articles")
    report.append(f"Excluded: {excluded_count} articles (see exclusions.txt)")
    report.append(f"Working corpus: {len(all_articles)} articles")
    report.append("")

    report.append("ARTICLE MAPPING")
    report.append("-" * 80)
    report.append(f"\nValidation Set ({len(validation_articles)} articles):")
    for article in validation_articles:
        report.append(f"  {article['original_id']} → {article['article_id']}")

    report.append(f"\nTest Set ({len(test_articles)} articles):")
    for article in test_articles:
        report.append(f"  {article['original_id']} → {article['article_id']}")

    report.append("\n" + "=" * 80)
    report.append("MOVE DISTRIBUTION")
    report.append("=" * 80)
    report.append(f"\n{'Move':<10} {'Overall':<15} {'Validation':<15} {'Test':<15}")
    report.append("-" * 80)

    for move in sorted(all_move.keys()):
        overall_pct = (all_move[move] / all_total) * 100
        val_pct = (val_move.get(move, 0) / val_total) * 100 if val_total > 0 else 0
        test_pct = (test_move.get(move, 0) / test_total) * 100 if test_total > 0 else 0
        report.append(
            f"{move:<10} {overall_pct:>6.2f}%{all_move[move]:>7} {val_pct:>6.2f}%{val_move.get(move, 0):>7} {test_pct:>6.2f}%{test_move.get(move, 0):>7}"
        )

    report.append("\n" + "=" * 80)
    report.append("STEP DISTRIBUTION")
    report.append("=" * 80)
    report.append(f"\n{'Step':<10} {'Overall':<15} {'Validation':<15} {'Test':<15}")
    report.append("-" * 80)

    for step in sorted(all_step.keys()):
        overall_pct = (all_step[step] / all_total) * 100
        val_pct = (val_step.get(step, 0) / val_total) * 100 if val_total > 0 else 0
        test_pct = (test_step.get(step, 0) / test_total) * 100 if test_total > 0 else 0
        report.append(
            f"{step:<10} {overall_pct:>6.2f}%{all_step[step]:>7} {val_pct:>6.2f}%{val_step.get(step, 0):>7} {test_pct:>6.2f}%{test_step.get(step, 0):>7}"
        )

    report.append("\n" + "=" * 80)
    report.append("SUMMARY STATISTICS")
    report.append("=" * 80)
    report.append(f"\nTotal Articles: {len(all_articles)}")
    report.append(
        f"  Validation: {len(validation_articles)} ({len(validation_articles)/len(all_articles)*100:.1f}%)"
    )
    report.append(
        f"  Test: {len(test_articles)} ({len(test_articles)/len(all_articles)*100:.1f}%)"
    )
    report.append(f"\nTotal Sentences: {all_total}")
    report.append(f"  Validation: {val_total} ({val_total/all_total*100:.1f}%)")
    report.append(f"  Test: {test_total} ({test_total/all_total*100:.1f}%)")
    report.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    print()

    # Paths
    raw_xml_dir = Path("data/raw_xml")
    processed_dir = Path("data/processed")

    # Load all XML files
    print(f"Loading XML files from {raw_xml_dir}/")
    xml_files = sorted(raw_xml_dir.glob("text*.xml"))

    if not xml_files:
        print(f"❌ ERROR: No XML files found in {raw_xml_dir}/")
        print(
            f"   Please place CaRS-50 XML files (text001.xml - text050.xml) in this directory."
        )
        return

    print(f"Found {len(xml_files)} XML files")

    # Apply exclusions
    if EXCLUDED_ARTICLES:
        print(
            f"\nExcluding {len(EXCLUDED_ARTICLES)} articles with sentence boundary issues:"
        )
        for article_id in sorted(EXCLUDED_ARTICLES.keys()):
            print(f"  ✗ {article_id}: {EXCLUDED_ARTICLES[article_id]}")
        print()

    articles = []
    excluded_count = 0
    for xml_file in xml_files:
        article_id = xml_file.stem

        # Skip excluded articles
        if article_id in EXCLUDED_ARTICLES:
            excluded_count += 1
            continue

        try:
            article = parse_xml(xml_file)
            articles.append(article)
        except Exception as e:
            print(f"⚠ Warning: Failed to parse {xml_file.name}: {e}")

    total_sentences = sum(len(a["sentences"]) for a in articles)
    print(f"✓ Loaded {len(articles)} articles ({total_sentences} sentences)")
    print(f"  (Excluded {excluded_count} articles)")
    print()

    # Stratified split
    val_size = 6
    print(
        f"Performing stratified split ({val_size} validation, {len(articles)-val_size} test)..."
    )
    validation_articles, test_articles = stratified_split(
        articles, validation_size=val_size
    )

    val_sentences = sum(len(a["sentences"]) for a in validation_articles)
    test_sentences = sum(len(a["sentences"]) for a in test_articles)

    print(
        f"✓ Validation: {len(validation_articles)} articles ({val_sentences} sentences)"
    )
    print(f"✓ Test: {len(test_articles)} articles ({test_sentences} sentences)")
    print()

    # Rename articles
    validation_articles = rename_articles(validation_articles, "validation")
    test_articles = rename_articles(test_articles, "test")

    print("Sample article mapping:")
    for article in validation_articles[:2]:
        print(f"  {article['original_id']} → {article['article_id']}")
    print("  ...")
    for article in test_articles[:2]:
        print(f"  {article['original_id']} → {article['article_id']}")
    print("  ...")
    print()

    # Save exclusions report
    exclusions_path = processed_dir / "exclusions.txt"
    save_exclusions_report(EXCLUDED_ARTICLES, exclusions_path)
    print(f"✓ Exclusions report: {exclusions_path}")

    # Save gold standards
    print("Saving gold standards...")
    val_gold_path = processed_dir / "validation" / "gold.json"
    test_gold_path = processed_dir / "test" / "gold.json"

    val_gold_data = create_gold_json(validation_articles)
    test_gold_data = create_gold_json(test_articles)

    with open(val_gold_path, "w", encoding="utf-8") as f:
        json.dump(val_gold_data, f, indent=2, ensure_ascii=False)
    print(f"✓ {val_gold_path}")

    with open(test_gold_path, "w", encoding="utf-8") as f:
        json.dump(test_gold_data, f, indent=2, ensure_ascii=False)
    print(f"✓ {test_gold_path}")
    print()

    # Save input texts
    print("Saving input texts...")
    val_input_dir = processed_dir / "validation" / "input"
    test_input_dir = processed_dir / "test" / "input"

    val_input_dir.mkdir(parents=True, exist_ok=True)
    test_input_dir.mkdir(parents=True, exist_ok=True)

    for article in validation_articles:
        output_path = val_input_dir / f"{article['article_id']}.txt"
        save_input_txt(article, output_path)
    print(f"✓ {val_input_dir}/ ({len(validation_articles)} files)")

    for article in test_articles:
        output_path = test_input_dir / f"{article['article_id']}.txt"
        save_input_txt(article, output_path)
    print(f"✓ {test_input_dir}/ ({len(test_articles)} files)")
    print()

    # Generate split report
    report_path = processed_dir / "split_report.txt"
    generate_split_report(
        validation_articles, test_articles, excluded_count, report_path
    )
    print(f"✓ Split report: {report_path}")
    print()

    print("=" * 60)
    print("COMPLETE ✓")
    print("=" * 60)
    print(f"\nWorking corpus: {len(articles)} articles")
    print(f"  Validation: {len(validation_articles)} articles")
    print(f"  Test: {len(test_articles)} articles")
    print(f"\nExcluded: {excluded_count} articles (see exclusions.txt for details)")


if __name__ == "__main__":
    main()
