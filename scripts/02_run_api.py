"""
02_run_api.py

Purpose: Call OpenAI API with specified temperatures and save raw outputs.

Configuration: Reads from config.yaml

Inputs:
    - config.yaml
    - prompts/system_prompt.txt
    - data/processed/{dataset}/input/*.txt

Outputs:
    - outputs/temp_{X}/run_{YY}/raw/{article_id}.txt (one per article per run per temp)
"""

import os
import yaml
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


def load_config():
    """Load and validate config.yaml"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_system_prompt():
    """Load prompt from prompts/system_prompt.txt"""
    prompt_path = Path("prompts/system_prompt.txt")
    if not prompt_path.exists():
        print(f"⚠ Warning: {prompt_path} not found. Using default prompt.")
        return "Please annotate each sentence with move and step labels following the CaRS framework."

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_articles_to_process(config):
    """
    Determine which articles to process based on config.

    If config['articles'] == 'all':
        Return all articles from the dataset
    Else:
        Return specified list
    """
    dataset = config["dataset"]
    input_dir = Path(f"data/processed/{dataset}/input")

    if config["articles"] == "all":
        # Get all .txt files
        article_files = sorted(input_dir.glob("*.txt"))
        return [f.stem for f in article_files]
    else:
        return config["articles"]


def call_openai_api(
    article_text, system_prompt, model, temperature, max_tokens, api_key
):
    """
    Call OpenAI API with specified parameters.

    Returns:
        response_text (str)
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": article_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"    ❌ API Error: {e}")
        raise


def process_single_article(
    article_id, temperature, run_num, config, system_prompt, api_key
):
    """
    Load article text → call API → save raw output.

    Saves to: outputs/temp_{temperature}/run_{run_num:02d}/raw/{article_id}.txt
    """
    dataset = config["dataset"]
    model = config["model"]
    max_tokens = config["max_tokens"]

    # Load article text
    input_path = Path(f"data/processed/{dataset}/input/{article_id}.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        article_text = f.read()

    # Call API
    response_text = call_openai_api(
        article_text=article_text,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )

    # Save raw output
    output_dir = Path(f"outputs/temp_{temperature:.1f}/run_{run_num:02d}/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{article_id}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response_text)

    return output_path


def check_existing_outputs(config):
    """
    Scan outputs/ directory and report what's already completed.
    Helps avoid re-running completed work.

    Returns:
        dict of completed runs per temperature
    """
    temperatures = config["temperatures"]
    runs = config["runs"]
    articles = get_articles_to_process(config)

    existing = {}

    print("Checking existing outputs...")
    for temp in temperatures:
        existing[temp] = []
        for run_num in runs:
            output_dir = Path(f"outputs/temp_{temp:.1f}/run_{run_num:02d}/raw")
            if output_dir.exists():
                # Check if all articles are present
                existing_files = set(f.stem for f in output_dir.glob("*.txt"))
                expected_files = set(articles)

                if existing_files >= expected_files:
                    existing[temp].append(run_num)

        if len(existing[temp]) == len(runs):
            print(f"✓ temp_{temp:.1f}: runs {min(runs)}-{max(runs)} complete")
        elif len(existing[temp]) > 0:
            missing_runs = [r for r in runs if r not in existing[temp]]
            print(
                f"⚠ temp_{temp:.1f}: runs {existing[temp]} complete, {missing_runs} missing"
            )
        else:
            print(f"✗ temp_{temp:.1f}: not started")

    print()
    return existing


def main():
    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("API DATA COLLECTION")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()
    system_prompt = load_system_prompt()

    # Get API key
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)

    if not api_key:
        print(f"❌ ERROR: {api_key_env} environment variable not set")
        print(f"   Please set it: export {api_key_env}='your-api-key'")
        return

    # Display configuration
    dataset = config["dataset"]
    model = config["model"]
    temperatures = config["temperatures"]
    runs = config["runs"]
    articles = get_articles_to_process(config)

    print("Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Runs: {runs}")
    print(
        f"  Articles: {'all' if config['articles'] == 'all' else 'custom list'} ({len(articles)} articles)"
    )
    print()

    total_calls = len(temperatures) * len(runs) * len(articles)
    print(
        f"Total API calls: {total_calls} ({len(temperatures)} temps × {len(runs)} runs × {len(articles)} articles)"
    )
    print()

    # Check existing outputs
    existing = check_existing_outputs(config)

    # Process each temperature
    start_time = time.time()
    total_processed = 0
    total_skipped = 0

    for temperature in temperatures:
        print(f"Processing temperature {temperature}...")

        for run_num in runs:
            # Skip if already complete (unless force_rerun is enabled)
            if run_num in existing[temperature] and not config.get("force_rerun", False):
                print(
                    f"  Run {run_num:02d}/{max(runs):02d} (skipped - already complete)"
                )
                total_skipped += len(articles)
                continue

            print(f"  Run {run_num:02d}/{max(runs):02d}")

            for article_id in articles:
                try:
                    output_path = process_single_article(
                        article_id=article_id,
                        temperature=temperature,
                        run_num=run_num,
                        config=config,
                        system_prompt=system_prompt,
                        api_key=api_key,
                    )
                    print(f"    {article_id}... ✓")
                    total_processed += 1

                    # Rate limiting: small delay between calls
                    time.sleep(0.5)

                except Exception as e:
                    print(f"    {article_id}... ❌ Failed: {e}")
                    continue

        print()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    print("=" * 60)
    print(f"COMPLETE: {total_processed} API calls processed")
    if total_skipped > 0:
        print(f"Skipped: {total_skipped} (already complete)")
    print(f"Total time: {hours}h {minutes}m")
    print("=" * 60)


if __name__ == "__main__":
    main()
