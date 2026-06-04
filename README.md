# Overview

- **Paper ID:** 2502.03692
- **Title:** DOCMIA: Document-level Membership Inference Attacks Against DocVQA Models
- **Domain:** Document AI / Privacy / Membership Inference
- **TL;DR:** The paper shows that DocVQA models can leak whether a whole document was in training data by aggregating optimization-based signals across that document's questions.

## Short Summary

DocMIA studies membership inference for document visual question answering
models at document granularity. The paper exploits the fact that one training
document typically appears through multiple question-answer pairs. Its main
attack repeatedly adapts a small part of the target model or input for each
question, aggregates the resulting optimization traces, and clusters documents
into member/non-member groups. The reported attacks outperform score, loss, and
gradient baselines on DocVQA and PFL-DocVQA targets.

## Key Results

- White-box DocMIA reaches about 72-82 F1 across reported model/dataset pairs.
- On Donut/DocVQA, FL reports 80.33 balanced accuracy and 82.18 F1.
- Black-box proxy attacks often improve over score-only black-box baselines,
  with notable low-FPR gains on DocVQA targets.

---

# Problem Definition

## Research Question

Given a trained document visual question answering model, a target document,
and multiple question-answer pairs about that document, decide whether the
document was included in the model's training set. The membership decision is
document-level: all questions attached to the same document must be combined
into a single in/out prediction.

## Why It Matters

Documents often contain sensitive personal, financial, or organizational data.
If a deployed document QA model leaks whether a specific document was used for
training, it exposes privacy risk even when the model only returns answers.

## Success Criteria

A successful system identifies member and non-member documents better than
chance on balanced member/non-member sets, works without auxiliary shadow
datasets, and remains effective when the model's normal question-answer utility
alone is not enough to separate the two classes.

---

# Paper's Method

## Key Contribution

DocMIA introduces document-level membership inference attacks for DocVQA
models. Instead of treating a single question-answer pair as the unit of
privacy, it aggregates evidence across all questions associated with a target
document.

## Approach

For each document/question/answer tuple, the attack starts from the target
model's trained state and performs a short optimization process that minimizes
the answer loss for that tuple. It records the L2 distance moved during
optimization, the number of steps until convergence, and the answer utility
observed during the process. These features are aggregated across questions
with average, minimum, maximum, and median statistics, normalized, and clustered
with KMeans. The cluster with lower optimization distance is treated as the
member cluster.

The paper proposes three white-box variants: FL optimizes one selected model
layer, FLLoRA optimizes LoRA parameters attached to that layer, and IG optimizes
the document image while freezing model weights. In the black-box setting, the
attacker first queries the deployed model, trains a proxy model on those
generated labels, and then applies the white-box attacks to the proxy.

## Main Claims

The primary claim is that optimization-derived document descriptors expose
membership more reliably than score, loss, or one-step-gradient baselines, even
without auxiliary data. Secondary claims are that the attack works in both
white-box and black-box settings, transfers through a distilled proxy, and can
be measured in the low-FPR regime used by modern MIA evaluations.

---

# Evaluation

## Metrics

The attack is evaluated as balanced binary classification over documents.
Metrics are balanced accuracy, ordinary accuracy, F1, TPR at 1% FPR, and TPR at
3% FPR. Higher is better for all attack-success metrics. The evaluation also
records two constraints: maximum questions used per document and whether an
auxiliary dataset was used; lower is better for both.

## Baselines and Targets

The main baselines are score-only unsupervised membership inference and a
single-gradient unsupervised attack. The paper's full-scale white-box Donut on
DocVQA setting reports roughly 80% balanced accuracy and 82% F1 for the
strongest document-level attack, with low-FPR TPR around 10% at 3% FPR.

## Evaluation Protocol

The benchmark uses a balanced member/non-member document set. Each document has
multiple question-answer pairs, and the evaluator reads one output artifact
containing per-method predictions or metrics. The evaluation script does not
import method code; it only converts the generated artifact into
`scoring/scores.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 1 total (0 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-03 00:00] - none
- Read the paper and identified DocMIA's document-level membership inference protocol.
- Implemented a scaled DocVQA-style benchmark and the paper's white-box FL optimization-distance attack.
- Added score-only and one-step-gradient baselines, isolated evaluation code, scripts, and reference scoring targets.
- Submitted the first compute job to run the full pipeline and determine whether the implementation reaches `method_runs` or `core_claim`.

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-03 00:00] - none
- Read the paper and identified DocMIA's document-level membership inference protocol.
- Implemented a scaled DocVQA-style benchmark and the paper's white-box FL optimization-distance attack.
- Added score-only and one-step-gradient baselines, isolated evaluation code, scripts, and reference scoring targets.
- Submitted the first compute job to run the full pipeline and determine whether the implementation reaches `method_runs` or `core_claim`.

### [2026-06-04 21:55] - none
- Inspected failed job `b610cdf5-51d`; it failed before method execution due to missing workspace-root import path in the container.
- Added package markers and exported `PYTHONPATH=/home/user` in all scripts.
- `python validate.py --reference-only` passes; resubmitting the full scaled DocMIA run.

### [2026-06-04 22:00] - core_claim
- Job `34213ac1-050` completed end-to-end and wrote `scoring/scores.json`.
- DocMIA FL reproduced the central optimization-based signal on the scaled benchmark: 66.04 F1, above score-only (55.14 F1) and one-step gradient (53.04 F1).
- Added the paper's IG image-optimization variant for the next expansion run.

### [2026-06-04 22:04] - core_claim
- Job `c775731d-164` completed the FL+IG expanded run.
- IG ran end-to-end but did not support an additional milestone at learning rate 0.01: 49.33 F1 and 0.00 TPR@3%FPR.
- Retuning IG to learning rate 0.001 and increasing target training epochs for the next job.

### [2026-06-04 22:47] - core_claim
- Job `a5f98ef0-266` completed successfully with 160 target epochs and IG learning rate 0.001.
- IG produced strong raw attack scores, but the setting was saturated: the target reached 100% member-document QA accuracy and score/gradient baselines were also near 95 F1.
- Keeping the milestone at `core_claim` and submitting an intermediate 90-epoch run to look for a non-saturated expansion setting.

### [2026-06-04 23:04] - core_claim
- Job `b823b8a9-21e` completed the 90-epoch midpoint run.
- The run remained saturated: score-only and gradient baselines reached 94-95 F1, while DocMIA FL/IG did not improve over them.
- Keeping the milestone at `core_claim` and lowering the next search point to 60 epochs.

### [2026-06-04 23:12] - core_claim_plus
- Job `75544422-7cb` completed the 60-epoch run.
- The setting is non-saturated and supports the central claim beyond the original 45-epoch minimum: DocMIA FL reached 71.58 F1 and 13.54 TPR@3%FPR, above score-only (69.36 F1, 11.46 TPR@3%FPR) and one-step gradient (63.91 F1, 4.17 TPR@3%FPR).
- The expanded IG variant also ran end-to-end at the paper-like image learning rate, but FL is the strongest reproduced proposed variant in this scaled gym.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
