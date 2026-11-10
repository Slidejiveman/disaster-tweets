# Disaster tweet triage

AlertMedia take-home, Sr AI/ML Engineer · Ryder Dale Walton

> **Summary** *(written last)*

All code is in [`pipeline.ipynb`](pipeline.ipynb). Steps 1–3 are in the notebook with their results; this README summarizes them and holds Steps 4 and 5.

## Data notes

- **Source.** [`mhamza-007/binary-class-tweets-dataset`](https://huggingface.co/datasets/mhamza-007/binary-class-tweets-dataset) stores everything in one zip, `disastrous_tweets.zip`. Because of that, `datasets.load_dataset` and the Hub's dataset viewer both fail, so the notebook downloads the zip and reads the CSV directly.
- **It is Kaggle's *NLP with Disaster Tweets*, byte for byte.** `train.csv` is 987,712 bytes, and the label column is `target`, not `label` as the dataset card says. The 3,263-row `test.csv` has no labels, so it is not used; all evaluation uses splits of `train.csv`.
- **7,613 labelled tweets, 43.0% positive.**
- **Duplicates and conflicting labels.** After normalizing case, HTML entities, URLs and @handles, there are 6,944 unique texts (669 duplicate copies). 69 texts appear with both labels (245 rows); 32 of them are exact ties.
- **Conflict policy:** majority label per text, ties dropped → **6,912 rows, 40.7% positive**. Dropping every conflicting text instead keeps 6,875 rows; the switch is `CONFLICT_POLICY` in §0.
- **Split:** 70 / 15 / 15 (4,838 / 1,038 / 1,036), stratified by label and grouped by a coarser near-duplicate key, so near-copies never straddle train and test.
- **Selection bias.** Every tweet was collected by searching for a disaster keyword, so the task is really *literal vs. figurative use of a disaster word*. Some keywords nearly decide the label on their own ("aftershock" 0% positive, "wreckage" 100%), a shortcut that would not hold on a broader feed.
- **Prevalence.** The 41% positive rate comes from that keyword filter. In a live feed, disasters are far rarer, so precision measured here overstates production precision (see Step 1).
- I did not use the publicly available full-label source of this dataset (the known Kaggle leak).

## Step 1: Evaluation first

*(in progress)*

## Step 2: Baseline and one improvement

*(to come)*

## Step 3: Error analysis

*(to come)*

## Step 4: LLM evaluation harness

*(to come)*

## Step 5: Two-week plan

*(to come)*

## Optional: grouping tweets into events

*(to come)*

## How to run

Tested on macOS (Apple Silicon) with Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab pipeline.ipynb      # then Run > Run All Cells
```

The first run downloads the dataset from Hugging Face into `data/`. *(Fine-tuning details to come.)*

## AI tool use

I used Claude (Anthropic) as a research assistant and pair programmer. Running log, updated as the work goes:

- **Research.** Claude's research mode surveyed the dataset's provenance, published baselines on this data, the metric literature, and LLM API prices. I used it to plan; every number in this README comes from my own runs unless cited.
- **Data inspection.** Through the Hugging Face connector, Claude found that the dataset is a single zip and confirmed it matches Kaggle's files.
- **§0–§1 code.** Claude drafted the loading, normalization, deduplication and grouped-split cells.
- **My decisions:** *(to fill in as we go)*
