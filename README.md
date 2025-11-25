# Temperature and Consistency in LLM-Based Move-Step Annotation

## What This Package Does

This is a **replication package** for a research study that tested how different "temperature" settings affect the reliability of AI language models when they annotate rhetorical moves and steps in research article introductions.

**In plain language:** When you use AI tools like ChatGPT to analyze academic writing, there's a hidden setting called "temperature" that controls how random the AI's responses are. This study tested whether that setting matters for getting consistent, reliable results. **Spoiler:** It matters a lot.

**Key Finding:** Lower temperatures (0.0–0.2) give you much more consistent results without hurting accuracy. At high temperatures (above 1.6), the AI starts producing unusable gibberish.

---

## Who Is This For?

- **Applied linguists** who want to use AI to analyze genre or rhetorical structure
- **Graduate students** learning computational methods
- **Researchers** who want to replicate or extend our findings
- **Anyone** interested in understanding AI reliability for text annotation

**You do NOT need to be a programmer** to use this package. If you can follow step-by-step instructions and run simple commands, you can do this.

---

## What You'll Need

### 1. Software (Free)

- **Python** (version 3.9 or newer) — Think of this as the engine that runs the scripts
  - Download from [python.org](https://www.python.org/downloads/)
  - Mac/Linux users: Python might already be installed. Check by typing `python3 --version` in Terminal
  - Windows users: Download the installer and make sure to check "Add Python to PATH" during installation

- **OpenAI API access** — This is how your computer talks to GPT-4
  - Create an account at [platform.openai.com](https://platform.openai.com)
  - You'll need to add a payment method (like a credit card)
  - **Cost for full replication:** About $40-50 USD
  - **Cost for a test run:** About $0.50-1.00 USD (highly recommended to start here!)

### 2. The Data (Free)

The study uses the **CaRS-50 corpus**: 50 biology research article introductions that experts have already annotated for rhetorical moves and steps.

- **Download link:** [Mendeley Data](https://doi.org/10.17632/kwr9s5c4nk.1)
- **License:** CC BY 4.0 (free to use with attribution)
- **What it contains:** 50 XML files named `text001.xml` through `text050.xml`

### 3. This Replication Package

Download or clone this entire folder from GitHub.

---

## Quick Start Guide (For Complete Beginners)

### Step 1: Install Python Packages

Python packages are like add-ons that give Python extra abilities. We need several for this project.

1. Open Terminal (Mac/Linux) or Command Prompt (Windows)
2. Navigate to this project folder:
   ```bash
   cd /path/to/temperature-consistency-study
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   This reads the `requirements.txt` file and automatically installs everything you need (like pandas for data tables, matplotlib for graphs, etc.).

**If this fails**, try `pip3` instead of `pip`:
```bash
pip3 install -r requirements.txt
```

### Step 2: Get Your OpenAI API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (it looks like `sk-...`)
4. **Keep it secret!** Don't share it or commit it to GitHub

Now tell your computer about this key. Choose **ONE** of these methods:

**Option A: Create a .env file** (Recommended)
1. In this project folder, create a new file called `.env`
2. Inside, type: `OPENAI_API_KEY=sk-your-actual-key-here`
3. Save the file

**Option B: Set an environment variable** (Mac/Linux)
```bash
export OPENAI_API_KEY='sk-your-actual-key-here'
```

**Option B: Set an environment variable** (Windows)
```cmd
set OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 3: Download and Prepare the Data

1. Download the CaRS-50 corpus from [Mendeley Data](https://doi.org/10.17632/kwr9s5c4nk.1)
2. Extract all 50 XML files (they'll be named `text001.xml` through `text050.xml`)
3. Put them in the `data/raw_xml/` folder in this project

Now run the preparation script:
```bash
python scripts/01_prepare_dataset.py
```

**What this does:** Converts the XML files into a format the other scripts can use, and splits the data into a small "validation" set (10 articles for testing) and a larger "test" set (40 articles for the full experiment).

**You'll know it worked if:** You see new folders created at `data/processed/validation/` and `data/processed/test/`

### Step 4: Configure Your Experiment

Open the file `config.yaml` in any text editor. This is your control panel.

**For your first run** (recommended), change these settings:
```yaml
dataset: validation          # Use the small test set first!
temperatures: [0.0, 1.0]    # Just test two temperatures
runs: [1, 2, 3]             # Only 3 runs per temperature
articles: 'all'             # Process all articles in the validation set
```

This gives you: 2 temperatures × 3 runs × 10 articles = **60 API calls** ≈ **$0.13**

**For the full replication**, use:
```yaml
dataset: test
temperatures: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
runs: [1,2,3,...,50]  # All 50 runs
articles: 'all'
```

This gives you: 9 temperatures × 50 runs × 40 articles = **18,000 API calls** ≈ **$40**

### Step 5: Run the Experiment

Now we run the actual AI annotations:

```bash
python scripts/02_run_api.py
```

**What this does:** Sends each article to GPT-4 multiple times (based on your `config.yaml` settings) and saves all the raw responses.

**This will take a while.** For the validation test, expect 10-20 minutes. For the full experiment, expect 6-10 hours.

**Progress tracking:** The script tells you which temperature, run, and article it's currently processing. You can safely stop it (Ctrl+C) and restart later—it won't re-do work that's already done.

### Step 6: Parse and Validate the Outputs

```bash
python scripts/03_parse_validate.py
```

**What this does:** Checks whether the AI followed instructions correctly. Did it annotate every sentence? Did it use the right format? This script flags any problems.

**You'll know it worked if:** You see "Parsing complete" with minimal errors. Some errors at very high temperatures (T=2.0) are expected.

### Step 7: Calculate the Metrics

```bash
python scripts/04_calculate_metrics.py
```

**What this does:** Calculates all the statistics reported in the paper:
- **Accuracy:** How well does the AI match the expert annotations?
- **Consistency:** Does the AI give the same answer when you run it multiple times?
- **Krippendorff's alpha:** A reliability statistic (like Cohen's kappa but better for multiple runs)

### Step 8: Generate Figures and Tables

```bash
python scripts/05_generate_figures.py
```

**What this does:** Creates all the graphs and tables from the paper. These are saved in the `results/figures/` and `results/tables/` folders.

**You can now open these files** to see your results!

---

## Understanding the Output

After running all scripts, you'll find:

### Results Folder Structure

```
results/
├── figures/                              # Graphs (PNG files)
│   ├── figure_1_step_temperature_heatmap.png
│   └── figure_2_accuracy_consistency.png
│
├── tables/                               # Data tables (CSV files)
│   ├── move_accuracy.csv                 # How accurate at identifying moves
│   ├── step_accuracy.csv                 # How accurate at identifying steps
│   ├── consistency.csv                   # Overall consistency scores
│   ├── move_consistency.csv              # Consistency for each move
│   └── parsing_success.csv               # Did the AI produce usable output?
│
└── metrics/                              # Raw data (JSON files)
    ├── accuracy_by_temperature.json
    ├── krippendorff_alpha.json
    └── sentence_level_analysis.json
```

### Key Metrics Explained

- **Parsing Success Rate:** Percentage of times the AI produced output we could actually use (at T=2.0, it often produces garbage)
- **Krippendorff's Alpha (α):** Reliability score from 0 to 1
  - α ≥ 0.80 = Excellent reliability
  - α = 0.67-0.79 = Moderate reliability
  - α < 0.67 = Poor reliability (don't trust these results!)
- **Weighted F1-Score:** Combines precision and recall while accounting for the fact that some moves are much more common than others
- **Recall:** For each specific rhetorical step, what percentage did the AI correctly identify?

---

## Troubleshooting Common Issues

### "OPENAI_API_KEY environment variable not set"

**Solution:** You forgot Step 2. Go back and set your API key.

### "No XML files found in data/raw_xml/"

**Solution:** You forgot to download the CaRS-50 corpus. Download it from [Mendeley Data](https://doi.org/10.17632/kwr9s5c4nk.1) and put the XML files in `data/raw_xml/`.

### "ModuleNotFoundError: No module named 'pandas'" (or other package names)

**Solution:** The packages didn't install correctly. Try:
```bash
pip3 install -r requirements.txt
```

If that still doesn't work, try installing packages one at a time:
```bash
pip3 install openai pandas numpy krippendorff pyyaml matplotlib seaborn python-dotenv scikit-learn
```

### Script is running very slowly

**Solution:** This is normal for the full experiment. The API calls take time. For testing:
1. Use the validation set instead of test set
2. Reduce the number of runs in `config.yaml`
3. Test fewer temperatures

### Validation failures reported in Step 6

**What this means:** The AI didn't follow instructions perfectly (e.g., it missed a sentence or added extra text).

**Check the details:**
Look in `results/validation_reports/` to see which specific articles failed and why.

**Common causes:**
- High temperatures (T ≥ 1.6) make the AI less reliable
- Occasional API glitches

**Fix:** Re-run just those specific articles by editing `config.yaml`:
```yaml
articles: ['test012', 'test015']  # Only re-run these
```

### API rate limits or quota errors

**Solution:** OpenAI limits how many requests you can make per minute. If you hit the limit:
1. Wait a few minutes
2. Check your OpenAI account usage limits
3. Consider upgrading your API tier if needed

---

## Project Structure (What All These Folders Mean)

```
temperature-consistency-study/
│
├── README.md                    ← You are here
├── requirements.txt             ← List of Python packages needed
├── config.yaml                  ← Your control panel (edit this!)
│
├── data/
│   ├── raw_xml/                 ← Put the 50 XML files here
│   └── processed/               ← Generated by script 01
│       ├── validation/          ← 10 articles for testing
│       │   ├── gold.json        ← Expert annotations
│       │   └── input/           ← Plain text for the AI
│       └── test/                ← 40 articles for the full experiment
│           ├── gold.json
│           └── input/
│
├── prompts/
│   └── system_prompt.txt        ← Instructions we give to the AI
│
├── outputs/                     ← Generated by script 02
│   ├── temp_0.0/                ← One folder per temperature
│   │   ├── run_01/              ← One folder per run
│   │   │   ├── raw/             ← Raw AI responses
│   │   │   └── parsed.json      ← Processed responses
│   │   ├── run_02/
│   │   └── ...
│   ├── temp_0.2/
│   └── ...
│
├── results/                     ← Generated by scripts 04 & 05
│   ├── figures/                 ← Graphs (PNG)
│   ├── tables/                  ← Data tables (CSV)
│   └── metrics/                 ← Raw statistics (JSON)
│
└── scripts/                     ← The programs you run
    ├── 01_prepare_dataset.py    ← Convert XML to usable format
    ├── 02_run_api.py            ← Send text to AI, get annotations
    ├── 03_parse_validate.py     ← Check if AI output is usable
    ├── 04_calculate_metrics.py  ← Calculate statistics
    └── 05_generate_figures.py   ← Create graphs and tables
```

---

## Important Notes for Researchers

### About Reproducibility

**Will I get exactly the same numbers?** Probably not, and that's okay. Even at temperature 0.0, AI models have some randomness built in at the API level. You should get **very similar** results (differences in the third decimal place), but not identical.

### Testing Before Committing

**Always run a small test first!** Use the validation set with just 2-3 temperatures and 5-10 runs. This lets you:
- Make sure everything works
- Estimate how long the full run will take
- Estimate the actual cost
- Catch configuration errors early

### The Temperature Parameter

From the paper's findings:

- **T = 0.0:** Most consistent, nearly deterministic (always picks the most likely word)
- **T = 0.2:** Still very consistent, slight randomness
- **T = 1.0:** OpenAI's default, more creative but less consistent
- **T = 2.0:** Very random, often produces unusable output

**For research use:** Stick with T ≤ 0.2

### Time and Cost Estimates

**Validation test** (10 articles × 3 runs × 2 temperatures):
- Time: ~20 minutes
- Cost: ~$0.13 USD

**Full replication** (40 articles × 50 runs × 9 temperatures):
- Time: ~8-10 hours
- Cost: ~$40-50 USD

*Estimates based on GPT-4.1-mini pricing as of November 2025. Actual costs may vary.*

### Partial Reruns

If something goes wrong, you don't have to start over! Edit `config.yaml` to specify exactly what to re-run:

```yaml
temperatures: [1.0]              # Only this temperature
runs: [23, 45]                   # Only these runs
articles: ['test012', 'test015'] # Only these articles
```

---

## About the CaRS-50 Corpus

The data comes from Lam & Nnamoko (2025):
- **50 biology research article introductions**
- **1,297 sentences total** (after removing 4 articles with annotation errors, we use 46 articles with 1,156 sentences)
- **3 main moves** (Establishing Territory, Establishing Niche, Occupying the Niche)
- **11 rhetorical steps** (like "Review previous research", "Indicate a gap", "Announce findings")

The corpus is **imbalanced**:
- Some steps (like "Review previous research") appear hundreds of times
- Other steps (like "Indicate article structure") appear only once
- This is realistic! Not all rhetorical moves are equally common

**Inter-rater reliability:** The corpus has moderate inter-rater agreement (Krippendorff's α = 0.424), which reflects the inherent difficulty of distinguishing subtle rhetorical functions. This is important context when interpreting the AI's performance.

---

## Adapting This for Your Own Research

Want to use this approach for your own corpus? Here's what you'd need to modify:

1. **Your data:** Replace the CaRS-50 XML files with your own annotated texts
2. **Script 01:** Modify the parser to read your annotation format
3. **The prompt:** Edit `prompts/system_prompt.txt` to describe your coding scheme
4. **Keep everything else the same!** The rest of the pipeline (API calls, validation, metrics, figures) will work with any move-step annotation task

---

## Citation

If you use this replication package in your research, please cite:

**The paper:**
```
[Your citation will go here once published]
```

**The CaRS-50 corpus:**
```
Lam, Charles; Nnamoko, Nonso (2025), "CaRS-50 Dataset: Annotated corpus of
rhetorical Moves and Steps in 50 article introductions", Mendeley Data, V1,
doi: 10.17632/kwr9s5c4nk.1
```

---

## License

- **Code:** MIT License (free to use, modify, and share)
- **CaRS-50 Data:** CC BY 4.0 (free to use with attribution)

---

## Getting Help

**Questions about this replication package?**
- Open an issue on GitHub
- Email: larrygrpolanco@gmail.com

**Questions about the CaRS-50 corpus?**
- See the [Mendeley Data page](https://doi.org/10.17632/kwr9s5c4nk.1)
- Contact the original authors

**Questions about using OpenAI's API?**
- Check the [OpenAI API documentation](https://platform.openai.com/docs)
- Visit the [OpenAI Community Forum](https://community.openai.com)

---

## Additional Resources

**New to Python?**
- [Python for Everybody](https://www.py4e.com/) (free online course)
- [Real Python Tutorials](https://realpython.com/) (beginner-friendly guides)

**New to command line/terminal?**
- [Command Line Crash Course](https://learnpythonthehardway.org/book/appendixa.html)
- Mac: Search for "Terminal" app
- Windows: Search for "Command Prompt" or "PowerShell"

**New to computational text analysis?**
- Eguchi & Kyle (2024). "Building custom NLP tools..." *Research Methods in Applied Linguistics*
- Kessler & Polio (2023). *Conducting Genre-Based Research in Applied Linguistics*

---

**Last Updated:** November 2025
