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

**The model is a triage layer, not the sender.** AlertMedia's analysts review threats before customers are alerted, and the AI Assistant requires a person to approve anything sent. So this classifier decides what analysts *see*, and that changes what each error costs.

**What a false positive costs.** At the model layer, analyst minutes: a person reads a non-event and dismisses it. If false positives ever reached customers, the cost would be much higher. Recipients learn to ignore or mute alerts, which makes the *next real* alert more likely to be missed, and trust erodes in both the customer's security team and AlertMedia. Every false alarm lowers the value of every future alert.

**What a false negative costs.** A threat to people that no analyst ever sees: a late or missing warning about a wildfire, flood or active shooter near a customer's site. These are rarer but heavy-tailed: possible injury, duty-of-care and liability exposure, and a customer who learns about the event from the news first. At this layer a false negative can't be recovered, because no one downstream gets a chance to catch it.

**So the objective is maximum recall while the analyst queue stays useful.** Precision is a capacity constraint, not the goal.

| Role | Metric | Why |
| --- | --- | --- |
| Headline, threshold-free | **Average precision (PR-AUC)**, with ROC-AUC alongside | Ranks how well real disasters rise to the top of the queue; focused on the positive class |
| Operating point | **Recall at precision ≥ 0.80**, threshold chosen on validation | "Catch as many disasters as possible while at least 80% of what analysts see is real" |
| Single-number summary | **F2**, plus F1 for comparison with published work | F2 weights recall twice as heavily as precision |
| Business | **Expected cost per 1,000 tweets**, with a miss = 10 and a false alarm = 1 | Makes the trade-off explicit. The 10:1 ratio is an assumption to replace with real analyst time and incident-severity data. |
| Trust in scores | **Brier score** and a reliability curve | Thresholds, cascades (Step 4) and analyst confidence all depend on a 0.9 meaning roughly 90% |
| Uncertainty | **Bootstrap 95% intervals**; paired bootstrap and McNemar's test to compare models | Without intervals a small gain can't be told from noise |

**Not accuracy.** With 41% positives, a model that never flags anything is still 59% accurate, and accuracy weights both errors equally, which is the opposite of this product's needs.

**How thresholds are chosen.** On the validation split only, then applied unchanged to test, which is scored once per model. If the system ever sent alerts without a human in the loop, a false positive would cost far more and the operating point should move up.

**Real-world prevalence.** This data is 41% disasters only because every tweet was found by a disaster-keyword search. Holding a model's recall and false-positive rate fixed, precision at prevalence π is recall·π / (recall·π + FPR·(1−π)). For example, 80% recall with a 10% false-positive rate gives 47% precision at 10% prevalence, 30% at 5%, and 7.5% at 1%. Step 2 reports this for both models.

**Sanity check.** Each tweet's keyword alone, scored by that keyword's disaster rate in train, reaches an average precision of 0.758 on validation (chance is 0.408). Any real model has to clear that bar, and the README's Step 3 checks how much the models lean on the same shortcut.

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
- **§2 code and Step 1 text.** Claude drafted the evaluation harness and the Step 1 write-up from the choices below.
- **My decisions:**
  - Resolve conflicting labels by majority vote and drop exact ties.
  - Operating point: recall at precision ≥ 0.80, chosen on validation.
  - Cost ratio: a missed disaster = 10 false alarms.
  - Report precision at 1%, 5% and 10% prevalence rather than assume one rate.
