# Disaster tweet triage

AlertMedia take-home, Sr AI/ML Engineer · Ryder Dale Walton

> **Summary.** A fine-tuned DistilBERT beats a TF-IDF + logistic regression baseline. At the same 80% precision floor, it catches about 24 more real disasters per 1,000 tweets and sends analysts 9 fewer false alarms (recall 0.781 vs. 0.722, average precision 0.882 vs. 0.853, McNemar p = 0.0045). But the error analysis says the bigger lever is the data. The model's most confident "misses" were mislabeled, and its most confident "false alarms" were real incidents that the dataset's definition leaves out. So the two-week plan starts with an analyst-agreed definition and a gold evaluation set from real traffic. An LLM comes in only as a cascade adjudicator on uncertain tweets, where it has to justify its cost per extra true positive caught.

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

**Sanity check.** Each tweet's keyword alone, scored by that keyword's disaster rate in train, reaches an average precision of 0.758 on validation (chance is 0.408). Any real model has to clear that bar; both in Step 2 do so comfortably.

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

**Cost of the improvement.** Training took 154 seconds on my iMac's Apple Silicon GPU. Validation average precision by epoch was 0.885, 0.890 and 0.889, so the notebook kept epoch 2. Serving is one small-model forward pass per tweet, a small fraction of an LLM's per-tweet cost (Step 4); I'd measure its latency in shadow mode (Step 5).

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

**Contestants.** The fine-tuned DistilBERT is the incumbent. Against it, two LLMs:

- **A small commercial API model**, for example Claude Haiku or GPT-5.4 mini.
- **A self-hosted open-weight model**, for example a 7–14B Llama or Qwen served with vLLM.

That pairing tests commercial against open-source, and it brings data privacy into the comparison.

**Setup**

- **The same frozen test split as Step 2.** Prompts are iterated on validation only; test is never used for prompt work.
- **The prompt is the Step 3 labelling guideline, verbatim**, the same text a human annotator gets. The tweet goes inside delimiters and is treated as data, never as instructions.
- **Zero-shot and a fixed 6-example few-shot variant.** Measure whether few-shot helps rather than assume it.
- **Structured JSON output** `{label, confidence, reason_code}` through the provider's schema mode, at temperature 0, recording the exact model version.
- **A score for thresholding.**
  - Best: label-token log-probabilities where the API exposes them.
  - Otherwise: the model's stated confidence (check its calibration) or the vote share across 5 samples (5× the cost).
  - Without a score, only one operating point can be compared, and average precision can't be computed.

**What to measure.** The Step 1 metrics on the same tweets (average precision, recall at precision ≥ 0.80, F2, Brier and reliability, expected cost), plus:

- **Paired comparison with the fine-tune:** McNemar's test and paired bootstrap intervals on each difference.
- **Agreement with corrected labels:** score each model against a relabelled subset as well, since Step 3 showed the dataset's labels are noisy and narrower than the product's definition.
- **Reliability:** invalid-output and refusal rate; agreement across 3 repeat runs and 2 prompt paraphrases.
- **Operations:** latency at p50 and p95, throughput under rate limits, and cost per 1,000 tweets from real token counts (GPU-hours for the self-hosted model).
- **Robustness:** a small adversarial set with prompt-injection text inside tweets, spam hashtags, and non-English or code-switched tweets.
- **Compliance:** the provider's data-retention and no-training terms and its processing region. The self-hosted model keeps data in-house.

**Deciding whether an LLM is worth its cost**

1. Hold the operating constraint fixed: analysts' queue at precision ≥ 0.80.
2. At that constraint, measure the extra true positives per 1,000 tweets over the fine-tune, with a bootstrap interval. Require McNemar p < 0.05 on the tweets where the two disagree.
3. Compute the **cost per additional true positive** = extra cost per 1,000 tweets ÷ extra true positives per 1,000 tweets. Compare it with what a caught incident is worth, a number to get from analysts and product. Also require p95 latency to fit the alerting SLA.
4. If an LLM clears that bar, deploy it as a **cascade adjudicator**. The fine-tune scores everything, and only the uncertain band around its threshold goes to the LLM. Tune that band to a few percent of traffic, then evaluate the whole cascade end to end, not the LLM alone. Step 3's genuine model errors were all near the threshold, which is exactly the band an adjudicator would see.

**Back-of-envelope at AlertMedia's scale.** AlertMedia cites 15M+ records a day. Assume about 300 input and 10 output tokens per tweet (guideline plus tweet), or 4.5B input and 150M output tokens a day. At list prices as published in November 2026 (verify before quoting):

| Model (input / output price per 1M tokens) | Every tweet | Cascade, 5% of tweets |
| --- | --- | --- |
| [GPT-5.4 mini](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/) ($0.75 / $4.50) | ~$4,050/day (~$1.5M/year) | ~$200/day |
| [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) ($1 / $5) | ~$5,250/day (~$1.9M/year) | ~$260/day |

Batch APIs halve the price but add hours of delay, which rules them out for alerting. Reasoning models can bill hidden reasoning tokens, so these are lower bounds. The fine-tune costs a small fraction of this per tweet. My expectation going in: an LLM may roughly match the fine-tune on this binary task, so the decision turns on cost, latency and calibration, which is why the cascade is the default design.

## Step 5: Two-week plan

Order: settle the definition before measuring, measure before modelling, and have monitoring in place before spending on LLMs.

| # | What I'd do | Stop when | Change course if |
| --- | --- | --- | --- |
| 1 | **Agree the definition and build a gold evaluation set from real traffic.** With 2–3 analysts, turn the Step 3 guideline into a one-page labelling guide by alert type, settling crime, evacuations and after-the-fact reports. Label 1,000–2,000 records sampled across sources *and across current scores, including low ones*, or recall can't be measured. Double-label an overlap set. | Annotator agreement κ ≥ 0.7 and about 300+ positives, enough to estimate recall within about ±5 points | κ < 0.6 after two guideline revisions: then the problem is the definition, which goes to product, not to modelling |
| 2 | **Baseline the current production system on that set**, sliced by source, alert type, language and region. Measure analyst queue volume and minutes per alert, to replace my 0.80 floor and 10:1 cost ratio with real numbers. | Metrics with intervals for the main slices, plus measured costs per false positive and per miss | Production already meets the target within its interval on every major slice: then skip new modelling and look elsewhere |
| 3 | **Error taxonomy and label cleanup.** Review production's misses first, then false alarms, with analysts. Run confident learning over the training data to queue likely mislabels; Step 3's most confident "misses" were all mislabeled. | About 50 consecutive errors add no new category, or the top 3 categories cover ≥ 70% of errors | Errors are mostly definition disputes: then go back to item 1 |
| 4 | **Calibrate, set thresholds per alert type, and run the candidate in shadow mode.** The candidate is the fine-tune retrained on cleaned labels; it scores live traffic without affecting what is sent. Monitor score drift, alerts per hour, latency, and disagreements with production, which analysts review in samples. | Promote: higher recall at equal queue volume, with non-overlapping intervals over an agreed window (for example, 7 days) | Kill: no significant gain after the planned number of reviewed disagreements, or latency or alert volume breaks the SLA |
| 5 | **Pilot the LLM cascade on the uncertain band** with the Step 4 harness: commercial vs. self-hosted open-weight, plus a privacy and compliance review. | It clears every Step 4 bar, then moves to an analyst A/B or interleaved trial | Stop: cost per extra true positive exceeds the agreed value, no significant gain on disagreements, p95 latency breaks the SLA, or invalid outputs exceed about 1% |

## Optional: grouping tweets into events

Not implemented, and it can't be evaluated on this dataset, which has no timestamps.

**Approach**

1. **Represent each flagged tweet** by a sentence embedding, the entities and places extracted from its text (named-entity recognition plus geocoding; the profile `location` field is unreliable), and its timestamp.
2. **Cluster online.** Assign each new tweet to the nearest active event in a sliding window (for example, 6–24 hours) if its similarity clears a threshold *and* its location and entities are compatible; otherwise open a new event.
   - Geography is a strong prior: two "fire" tweets 500 km apart are two events.
   - Periodically merge events whose centers converge.
3. **Make it fast at 15M records a day** with an approximate nearest-neighbour index, such as Elasticsearch or OpenSearch vector search, or locality-sensitive hashing as in streaming first-story detection (Petrović, Osborne and Lavrenko, 2010).

**How I'd know it works**

- **Offline:** analysts assign event IDs to a sample. Score with B-cubed precision and recall and the adjusted Rand index, and track two errors separately:
  - **Merges** (two incidents fused into one) are worse for alerting, because one incident hides another.
  - **Splits** (one incident fragmented) cause duplicate alerts.
- **Tuning:** set the similarity threshold and time window on that sample to keep merges under a ceiling, accepting some splits.
- **Online:** alerts per incident fall without any missed incidents; time from an event's first tweet to its first alert; weekly analyst spot checks.

## How to run

Run on macOS (Apple Silicon); needs Python 3.10 or newer.

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

I used Claude Opus 5.5 (Anthropic) as a research assistant and pair programmer. **Time spent: about 1 hour, including planning with Claude.** Every number in this README comes from my own run of the notebook on my iMac, except where a source is cited.

**What Claude did**

- **Research.** Its research mode surveyed the dataset's provenance, published baselines on this data, the metric literature and LLM API prices; I used that to plan.
- **Data inspection.** Through the Hugging Face connector, it found that the dataset is a single zip and confirmed it matches Kaggle's files.
- **Code.** It drafted every notebook cell. It tested all of them except the fine-tune in its own environment; I ran the full notebook, including the fine-tune, on my iMac.
- **Error-analysis logistics.** It picked the ten errors by the rule in Step 3, shuffled them and hid their labels and scores while I labelled them, then computed the verdicts from my labels.
- **Writing.** It drafted this README from my decisions and results. I reviewed and edited it.

**What I decided**

- Resolve conflicting labels by majority vote and drop exact ties.
- Operating point: recall at precision ≥ 0.80, chosen on validation.
- Cost ratio: a missed disaster = 10 false alarms.
- Report precision at 1%, 5% and 10% prevalence rather than assume one rate.
- Fine-tune on my iMac's GPU rather than in Colab.
- The error-analysis guideline, including counting unplanned evacuations as 1; my blind labels and reasons for the ten errors; and leaving #6 "unsure".
- Step 4: compare a commercial API model with a self-hosted open-weight model, and use an LLM, if it earns its cost, only as a cascade adjudicator.
- Step 5: start with the definition and a gold evaluation set from real traffic.
- Include the optional event-grouping sketch.
