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

**Sanity check.** Each tweet's keyword alone, scored by that keyword's disaster rate in train, reaches an average precision of 0.758 on validation (chance is 0.408). Any real model has to clear that bar, and Step 3 checks how much the models lean on the same shortcut.

## Step 2: Baseline and one improvement

**Baseline:** TF-IDF on word 1–2-grams plus character 2–5-grams, into logistic regression. Character n-grams pick up hashtags, misspellings and word fragments. Regularization (C = 2) was chosen on validation by average precision.

**Improvement:** a fine-tuned DistilBERT (3 epochs, keeping the epoch with the best validation average precision), with temperature scaling fit on validation so its probabilities are calibrated. I chose fine-tuning over the other options because published work on this data ([kiran-bal](https://github.com/kiran-bal/disaster-tweet-classification)) puts a fine-tuned transformer about 4–5 F1 points above TF-IDF, while frozen sentence embeddings with logistic regression land within noise of it.

**Results on the held-out test set** (1,036 tweets; thresholds chosen on validation for precision ≥ 0.80; 95% bootstrap intervals):

| Metric | TF-IDF + LR | Fine-tuned DistilBERT | Difference (paired bootstrap) |
| --- | --- | --- | --- |
| Average precision | 0.853 [0.823, 0.879] | **0.882** [0.857, 0.904] | +0.029 [+0.013, +0.044] |
| ROC-AUC | 0.864 [0.839, 0.886] | **0.888** [0.866, 0.907] | |
| Recall at operating point | 0.722 [0.678, 0.763] | **0.781** [0.740, 0.818] | +0.059 [+0.026, +0.092] |
| Precision at operating point | 0.786 [0.744, 0.824] | 0.816 [0.780, 0.851] | +0.031 [−0.006, +0.066] |
| F2 | 0.734 [0.697, 0.771] | **0.788** [0.752, 0.820] | +0.054 [+0.026, +0.083] |
| F1 | 0.752 [0.718, 0.783] | **0.799** [0.769, 0.825] | |
| Brier score (lower is better) | 0.140 [0.129, 0.152] | **0.120** [0.107, 0.134] | |
| Expected cost per 1,000 tweets | 1,209 [1,025, 1,402] | **959** [805, 1,128] | −250 [−383, −115] |
| Alerts per 1,000 tweets | 374 | 389 | |

**The improvement is real.** Every difference except precision has an interval that excludes zero. Precision is held near the 0.80 floor by design, so it isn't expected to move much. On the paired predictions, McNemar's exact test gives p = 0.0045: the fine-tuned model alone is right on 85 test tweets, the baseline alone on 51. The +4.7 F1 matches the 4–5 points reported in published work on this data.

**What it means for an analyst queue.** At the same precision floor, per 1,000 tweets at this data's prevalence, the fine-tuned model flags about 318 real disasters against the baseline's 293, and about 71 false alarms against 80. So it catches roughly **24 more real events while sending analysts 9 fewer false alarms**.

**Cost of the improvement.** Training took 154 seconds on my iMac's Apple Silicon GPU. Validation average precision by epoch was 0.885, 0.890 and 0.889, so the notebook kept epoch 2. Inference is milliseconds per tweet, so serving cost is negligible next to an LLM (Step 4).

**Calibration.** The fitted temperature was 1.10, so the fine-tuned model was only slightly overconfident, and its Brier score is better than the baseline's. Both models track the diagonal of the reliability plot (`results/pr_reliability.png`) reasonably well.

**Where the curves meet.** Above about 90% recall the two precision-recall curves converge: the last disasters to be caught are hard for both models. Step 3 looks at why.

**Precision at real-world prevalence** (holding each model's test recall and false-positive rate fixed):

| Prevalence | TF-IDF + LR | Fine-tuned DistilBERT |
| --- | --- | --- |
| 10% | 0.373 | 0.419 |
| 5% | 0.220 | 0.255 |
| 1% | 0.051 | 0.062 |

At realistic prevalence either model would bury analysts in false alarms if it scored an unfiltered feed. That supports using it as a second stage after keyword and source filtering, and setting thresholds on representative live traffic rather than on this dataset (Step 5).

**A note on the operating point.** The baseline's threshold was chosen on validation to reach precision 0.80, and on test it lands at 0.786, with an interval that still includes 0.80. A threshold tuned on one sample drifts on the next, which is why production needs monitoring and periodic recalibration (Step 5).

## Step 3: Error analysis

**Method.** On test, the fine-tuned model makes 166 errors (92 missed disasters, 74 false alarms), against the baseline's 200. 69% of the fine-tune's errors are shared with the baseline.

- **Sample:** 5 missed disasters and 5 false alarms, weighted toward misses because they matter most. In each group, the 3 most confident errors and the 2 nearest the threshold.
- **Blind labelling:** before looking, I wrote a labelling guideline, then labelled the ten shuffled, without their official labels or scores:

> **1** = the tweet reports a real, current event that threatens people or property, or an emergency response to one. Unplanned evacuations and lockdowns count as 1 even if the cause turns out to be false; drills and exercises are 0. **0** = figurative use, fiction, games, opinion, history, or no real event.

- **Verdict:** if my label agrees with the model rather than the dataset, the error is label noise. If it agrees with the dataset, it's a model error. "Unsure" marks ambiguity in the definition itself.

| # | Tweet (shortened) | Dataset | Fine-tune score | Mine | Verdict | Theme |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Ina Buted Girl Crush?? | 1 | 0.07 | 0 | label noise | nonsense / figurative |
| 2 | …stand with Lucifer in all his flames of destruction | 1 | 0.07 | 0 | label noise | figurative |
| 3 | Private thirsty night?SAD BLOOD ROCK'N ROLL? | 1 | 0.07 | 0 | label noise | nonsense / figurative |
| 4 | annihilating quarterstaff of annihilation | 1 | 0.39 | 0 | label noise | game item |
| 5 | Why weren't they taken back to Africa once rescued? | 1 | 0.38 | 0 | label noise | opinion about a news event |
| 6 | Thief broke front window of Hicksville store, stole 50 cell phones; fled, crashed into… | 0 | 0.96 | unsure | ambiguous | crime / incident report |
| 7 | Odeon cinema evacuated following false fire alarm | 0 | 0.95 | 1 | label noise | unplanned evacuation |
| 8 | Mourning notices for stabbing arson victims stir "politics of grief" | 0 | 0.95 | 0 | model error | aftermath of a past attack |
| 9 | …possibly he's had injuries on both ankles though. 2011 one worse | 0 | 0.39 | 0 | model error | sports chat |
| 10 | Ashes 4th Test: 10 hilarious reactions to Australia's collapse | 0 | 0.39 | 0 | model error | sports metaphor |

(The operating threshold is 0.389, so scores of 0.38–0.39 are near-misses.)

**What they have in common**

- **Every "missed disaster" is mislabeled under my guideline.** All five are nonsense, figurative rhetoric, a game item, or opinion, and both models agree with me on all five. The model's most confident misses aren't real misses, so measured recall understates the fine-tuned model.
- **The confident "false alarms" describe real incidents:** an evacuation, a theft and crash, the aftermath of an attack. The dataset's idea of "disaster" is narrower than what a risk-intelligence product needs. Whether crime is in scope, and whether a report of a just-finished incident is still actionable (#6), are product decisions, not modelling ones.
- **The genuine model errors are context-poor sports language near the threshold** ("collapse", "injuries"), which only the fine-tuned model gets wrong, plus a news story about a past attack (#8) that both models flag.
- **Data quality:** mis-decoded characters ("Û÷politics of griefÛª") and HTML entities ("&amp;") survive in the raw text.

**What I'd do about it**

1. **Fix labels before changing models.** Score every training tweet out-of-fold and queue the confident disagreements for review, using confident learning (for example, cleanlab). On this evidence, relabelling would move the metrics more than a bigger model.
2. **Settle the definition with analysts:** crime, evacuations, after-the-fact reports. This is item 1 of the two-week plan (Step 5).
3. **For genuine model errors,** add sports and pop-culture negatives to training, or send the near-threshold band to an LLM adjudicator that knows "the Ashes" is cricket (the cascade in Step 4).
4. **Repair text encoding** (for example, with ftfy) during normalization.

**Limits.** Ten examples and one labeller give a direction, not a rate. To estimate the label-noise rate properly I'd have two people label a few hundred tweets and measure their agreement (Cohen's κ).

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

To run it headless and save the outputs into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 pipeline.ipynb
```

- The first run downloads the dataset from Hugging Face into `data/`.
- **Fine-tuning is skipped by default.** The fine-tuned model's validation and test predictions are committed in `results/finetune_predictions.csv`, so the notebook reproduces every number without a GPU. To retrain, set `FORCE_RETRAIN = True` in §3b. It then downloads DistilBERT (about 270 MB) and trains in about 2.5 minutes on an Apple Silicon GPU; it also runs on CUDA or, slowly, on CPU.

## AI tool use

I used Claude (Anthropic) as a research assistant and pair programmer. Running log, updated as the work goes:

- **Research.** Claude's research mode surveyed the dataset's provenance, published baselines on this data, the metric literature, and LLM API prices. I used it to plan; every number in this README comes from my own runs unless cited.
- **Data inspection.** Through the Hugging Face connector, Claude found that the dataset is a single zip and confirmed it matches Kaggle's files.
- **§0–§1 code.** Claude drafted the loading, normalization, deduplication and grouped-split cells.
- **§2 code and Step 1 text.** Claude drafted the evaluation harness and the Step 1 write-up from the choices below.
- **§3 code and Step 2 text.** Claude drafted the baseline, fine-tuning and comparison cells, testing all but the fine-tune in its own environment. I ran the full notebook on my iMac, which produced every number above, and Claude drafted the Step 2 write-up from those results.
- **My decisions:**
  - Resolve conflicting labels by majority vote and drop exact ties.
  - Operating point: recall at precision ≥ 0.80, chosen on validation.
  - Cost ratio: a missed disaster = 10 false alarms.
  - Report precision at 1%, 5% and 10% prevalence rather than assume one rate.
  - Fine-tune on my iMac's GPU rather than in Colab.
  - Error-analysis guideline, including counting unplanned evacuations as 1, and my blind labels for the ten errors. Claude chose and shuffled the ten by the rule above and hid their labels and scores. I labelled them, Claude computed the verdicts from my labels, and I chose to leave #6 "unsure".
- **§4 code and Step 3 text.** Claude drafted the selection and review cells and the Step 3 write-up from my labels and reasons.
