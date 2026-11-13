# Overview

- **Paper ID:** 2502.03692
- **Title:** DOCMIA: Document-level Membership Inference Attacks Against DocVQA Models
- **Domain:** Document AI / Privacy / Membership Inference
- **TL;DR:** The paper shows that DocVQA models leak whether a whole document was
  in their training data, by aggregating optimisation-based signals across the
  document's questions, without auxiliary data.

## Short Summary

DocMIA studies membership inference against document visual question
answering models at document granularity. A training document is seen through
several question-answer pairs. The attack adapts a small part of the target
model, or the input image, for each question. It aggregates the resulting
optimisation traces over the document and clusters documents into members and
non-members. The paper reports that these attacks match or beat score, loss
and gradient baselines on DocVQA and PFL-DocVQA targets.

## This gym

This is a scaled white-box reproduction:

- **Documents:** procedural forms.
- **Target:** a ~1.1M-parameter OCR-free reader, fine-tuned from a public
  pre-trained checkpoint. Its member/non-member QA accuracy is ~97% vs ~60%,
  similar to the paper's DocVQA targets.
- **Candidates:** 600 member and 600 non-member documents per audit.
- **Test:** hidden. It has four independent audits, each with its own target,
  and the metrics are averaged over them.

## Key Results (paper, Donut on DocVQA, white-box)

- FL: 80.33 accuracy, 82.18 F1, 10.67 TPR at 3% FPR.
- Strongest baseline, SCORELOSS-UA_all: 80.50 accuracy, 81.10 F1, 8.67 TPR at
  3% FPR.
- Across targets, the white-box DocMIA variants reach about 72-82 F1.

---

# Problem Definition

## Research Question

Given white-box access to a trained document visual question answering (DocVQA)
model and a set of candidate documents, each with the question-answer pairs
asked about it, decide for every document whether it was part of the model's
training set. The decision is document-level: all question-answer pairs of a
document must be combined into one member/non-member decision and one
membership score.

## Setting

- The audited model is an OCR-free document QA model (page image + question ->
  answer from a closed answer bank). It was obtained by fine-tuning a public
  pre-trained checkpoint (`data/pretrained_reader.pt`) on a private set of
  documents. Its architecture is in `data/target_model.py`; its weights are
  given to the attacker (white-box).
- Documents are scanned forms with five key/value fields; every document comes
  with 1 to 5 question-answer pairs (the questions the model was trained with
  for members). Each page has its own scan noise, ink, blur and placement.
- The candidate set is balanced: half of the candidates are training documents
  of the audited model, half are fresh documents from the same distribution.
- The attacker has no auxiliary data: no other documents from this
  distribution, no shadow or reference models trained to imitate the target,
  and no knowledge of which documents were used for training.

## Why It Matters

Documents carry sensitive personal, financial and business information. If a
deployed document QA model reveals whether a given document was in its
training data, it leaks private information even though it only returns
answers. Document-level membership inference is also the tool an auditor uses
to check whether particular documents were used to train a model.

## Success Criteria

A good attack separates members from non-members well at a balanced operating
point (balanced accuracy, F1) and, more importantly for privacy auditing,
identifies members with very few false positives (true positive rate at 1% and
3% false positive rate), while respecting the no-auxiliary-data threat model.

---

# Paper's Method

## Key Contribution

DocMIA introduces document-level membership inference attacks for DocVQA
models. Instead of treating a single question-answer pair as the unit of
privacy, it aggregates evidence across all questions associated with a target
document, without auxiliary data or shadow models.

## Approach (white-box, Algorithm 1)

For each document/question/answer tuple, the attack starts from the target
model and runs Adam on the answer loss for that tuple until the loss improves
by less than a threshold tau or S = 200 steps are reached. It records the L2
distance moved by the optimised object, the number of steps taken, and the
answer utility (ACC) averaged over the optimisation. These three features are
aggregated over the document's questions with {AVG, MIN, MAX, MED}, standardised,
and clustered with KMeans (k = 2). The cluster with the larger mean distance is
the non-member cluster. A document's membership score is minus its distance to
the member-cluster centroid.

Three variants differ in what is optimised:

- **FL**: the final answer-projection layer.
- **FLLoRA**: LoRA factors on that layer, with A Kaiming-initialised and B zero
  so the start equals the target.
- **IG**: the document image pixels, with the model frozen.

The learning rate and tau are tuned per target model (Appendix C).

## Implementation in this gym

- `method/docmia.py` implements Algorithm 1 for all three variants. Pairs are
  batched, but every pair keeps its own parameters, Adam state and stopping
  step.
- `method/tune_dev.py` runs the Appendix C grid search on the dev audit:
  - lr in {1e-4 ... 1} for FL and FLLoRA, and {1e-3, 1e-2, 1e-1} for IG;
  - tau in {1e-6 ... 10};
  - selection by mean(balanced accuracy, F1).

  It writes `method/config.json`, including the best variant, which becomes the
  submitted `membership_attack`.
- `method/attack.py` (via `scripts/attack.sh`) applies that configuration to
  any audit.

## Main Claims

Optimisation-derived document descriptors expose membership as well as or
better than score, loss and one-step-gradient baselines, especially in the
low-false-positive regime, without auxiliary data. The attack also works
against documents seen through few training questions. The paper's black-box
variant (distilling the target into a proxy) is not part of this gym.

---

# Evaluation

## What is evaluated

A submission is an **attack program**, run by the evaluator as

    bash scripts/attack.sh <audit_dir> <output_json>

`<audit_dir>` contains `target.pt` (the audited model; load it with
`data.target_model.load_checkpoint` or `data.audit_io.load_target`) and
`candidates/` (`manifest.json` plus page images; `data.audit_io.load_candidates`
loads both). The manifest lists, for each candidate, its `doc_id`, its image
and its question-answer pairs (`question`, `field`, `answer`). It holds no
membership information.

The program must write a JSON file:

    {"doc_ids": [...every candidate id...],
     "attacks": {
        "membership_attack": {"scores": [one finite number per document],
                              "predictions": [one 0/1 decision per document]},
        "<any other name>":  {"scores": [...], "predictions": [...]}}}

`scores` are membership scores (higher = more likely a training document) and
are aligned with `doc_ids`. `predictions` are the attack's own member (1) /
non-member (0) decisions.

## Metrics

Computed by the evaluator from the raw scores and decisions against labels the
attack never sees:

| metric | from | coefficient |
|---|---|---|
| `balanced_accuracy` | decisions | 1.0 |
| `f1` (member class) | decisions | 1.0 |
| `tpr_at_1_fpr` (largest TPR with FPR <= 1%) | scores | 0.4 |
| `tpr_at_3_fpr` (largest TPR with FPR <= 3%) | scores | 1.0 |
| `roc_auc` | scores | 0 (diagnostic) |

All are percentages, higher is better. On the hidden test each metric is
computed per audit and averaged over the audits. Only the `membership_attack`
row of experiment `scaled_whitebox_docmia` is ranked. Other named attacks are
scored and reported under `other_attacks`. Per-audit values are under
`per_audit`. Diagnostics on documents with at most two question-answer pairs
are reported under `diagnostics`. None of these is ranked.

## Failures

The `membership_attack` row is always written. If it is missing, or its scores
are not finite, or its decisions are not exactly 0/1, or its ids do not cover
every candidate exactly once, it gets the worst value (0) of every metric. A
program that crashes, times out or writes nothing fails all rows the same way.
On the hidden test, a failure on **any one** audit fails the whole row.
Nothing else in the output (no metric, flag or threshold) is read.

## Baselines

The evaluator computes five reference attacks itself from the audited model.
They are reported under `baselines` and are identical for every submission:

- `loss_ta`: mean answer loss, thresholded at its mean.
- `score_ua`: clustering of the mean answer correctness.
- `score_ua_all`: the same with min/max/median aggregates.
- `scoreloss_ua_all`: correctness and loss aggregates, clustered.
- `gradient_ua`: one-step gradient-norm and correctness aggregates, clustered.

## Slices

- **Train (visible):** `bash scripts/evaluate_train.sh` runs the attack on the
  dev audit `data/dev_audit`, which has one target and 600 members plus 600
  non-members. Its labels are in `data/dev_audit/labels.json` for development.
  Output: `scoring/scores_train.json`. `bash scripts/evaluate.sh` writes the
  same scores to `scoring/scores.json`.
- **Test (hidden):** `bash scripts/evaluate_test.sh` (no arguments) runs the
  same program once on each of four hidden audits. Each has its own target,
  built with the dev recipe from its own private document set, and 1200
  candidates of the dev composition. Output: `scoring/scores_test.json`. The
  hidden audits and their labels exist only in the test environment.

While the attack runs on the hidden test:

- `eval/test/` and `data/dev_audit/` are unreadable. Anything the attack needs
  from the dev audit, such as a selected threshold or hyper-parameter, must be
  stored with the method beforehand.
- Each run sees only the copy of its own audit. The copy is deleted when the
  program returns, together with any process it started.

The test runs on one CPU core. The reference attack takes about 40 seconds per
audit. The evaluator stops an attack after one hour per audit.

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

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 7 total (6 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:47] - core_claim_plus
- Job `75544422-7cb` completed the scaled white-box DocMIA reproduction end-to-end.
- The target model reached a non-trivial member/nonmember QA gap rather than saturated memorization.
- DocMIA FL outperformed score-only and one-step-gradient baselines on F1 and TPR@3%FPR, while DocMIA IG also ran as a second optimization-based variant.
- `python validate.py --compare` passed with import separation and evaluator-output checks.

### [2026-06-04 18:53] - core_claim_plus
- Job `73b730b0-831` passed after hardening the evaluator to recompute metrics from raw DocMIA features and labels.
- The recomputed scores match the final 60-epoch artifact: DocMIA FL reaches 71.58 F1 and 13.54 TPR@3%FPR, ahead of the included score-only and one-step-gradient baselines on the core comparison.
- `python validate.py --compare` passed in the compute container.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 9 total (8 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 9 total (8 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 9 total (8 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h
- **Jobs**: 9 total (8 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

### [2026-06-04 23:38] - secondary_claims
- Re-ran local validation, Python compilation, and shell syntax checks.
- Confirmed the packaged scaled DocMIA gym remains at `secondary_claims`.

### [2026-06-04 23:48] - secondary_claims
- Completed final verification against existing successful container jobs and
  current local validation checks.
- Confirmed the workspace is ready as a scaled reusable research gym.

### [2026-06-05 00:02] - secondary_claims
- Rechecked the final workspace and tightened `scripts/reproduce.sh` so the
  end-to-end entrypoint explicitly refreshes all scoring artifacts.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 11 total (10 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

### [2026-06-04 23:38] - secondary_claims
- Re-ran local validation, Python compilation, and shell syntax checks.
- Confirmed the packaged scaled DocMIA gym remains at `secondary_claims`.

### [2026-06-04 23:48] - secondary_claims
- Completed final verification against existing successful container jobs and
  current local validation checks.
- Confirmed the workspace is ready as a scaled reusable research gym.

### [2026-06-05 00:02] - secondary_claims
- Rechecked the final workspace and tightened `scripts/reproduce.sh` so the
  end-to-end entrypoint explicitly refreshes all scoring artifacts.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1.0h | **GPU**: 0.0h
- **Jobs**: 19 total (14 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

### [2026-06-04 23:38] - secondary_claims
- Re-ran local validation, Python compilation, and shell syntax checks.
- Confirmed the packaged scaled DocMIA gym remains at `secondary_claims`.

### [2026-06-04 23:48] - secondary_claims
- Completed final verification against existing successful container jobs and
  current local validation checks.
- Confirmed the workspace is ready as a scaled reusable research gym.

### [2026-06-05 00:02] - secondary_claims
- Rechecked the final workspace and tightened `scripts/reproduce.sh` so the
  end-to-end entrypoint explicitly refreshes all scoring artifacts.

### [2026-11-12] - secondary_claims (hardening iteration)
- Rebuilt the gym around a hidden audit: secret-seeded private set, its own fine-tuned target, labels and hashes only in eval/test; the hidden test runs the submitted attack program and recomputes every metric; baselines computed by the evaluator; one ranked `membership_attack` row with worst values on failure.
- Restored the paper's protocol at small scale: public pre-trained reader fine-tuned on private documents (member/non-member QA 97%/60%), Algorithm 1 with S=200, Adam, Appendix C lr/tau grid on the dev audit, Phi_all + KMeans, FLLoRA starting at the target; 600+600 candidates per audit.
- Hidden test: DocMIA (FLLoRA, dev-selected) BA 76.7 / F1 80.1 / TPR@3%FPR 14.7; beats SCORELOSS-UA_all and score baselines, ties LOSS-TA on BA/F1 while beating it at low FPR, ties GRADIENT-UA. Ablations significantly worse; all REDTEAM shortcuts closed. See HARDENING.md.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 29m | **GPU**: 0.0h
- **Jobs**: 21 total (16 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

### [2026-06-04 23:38] - secondary_claims
- Re-ran local validation, Python compilation, and shell syntax checks.
- Confirmed the packaged scaled DocMIA gym remains at `secondary_claims`.

### [2026-06-04 23:48] - secondary_claims
- Completed final verification against existing successful container jobs and
  current local validation checks.
- Confirmed the workspace is ready as a scaled reusable research gym.

### [2026-06-05 00:02] - secondary_claims
- Rechecked the final workspace and tightened `scripts/reproduce.sh` so the
  end-to-end entrypoint explicitly refreshes all scoring artifacts.

### [2026-11-12] - secondary_claims (hardening iteration)
- Rebuilt the gym around a hidden audit: secret-seeded private set, its own fine-tuned target, labels and hashes only in eval/test; the hidden test runs the submitted attack program and recomputes every metric; baselines computed by the evaluator; one ranked `membership_attack` row with worst values on failure.
- Restored the paper's protocol at small scale: public pre-trained reader fine-tuned on private documents (member/non-member QA 97%/60%), Algorithm 1 with S=200, Adam, Appendix C lr/tau grid on the dev audit, Phi_all + KMeans, FLLoRA starting at the target; 600+600 candidates per audit.
- Hidden test: DocMIA (FLLoRA, dev-selected) BA 76.7 / F1 80.1 / TPR@3%FPR 14.7; beats SCORELOSS-UA_all and score baselines, ties LOSS-TA on BA/F1 while beating it at low FPR, ties GRADIENT-UA. Ablations significantly worse; all REDTEAM shortcuts closed. See HARDENING.md.

### [2026-11-13] - secondary_claims (hardening iteration 12)
- Hidden labels (and the dev audit) are sealed while the hidden test runs an attack; labels are read only after it returns. A label-reading attack now fails (was perfect).
- Hidden test enlarged to four independent audits (new secret seed), metrics averaged; any failed audit fails the row. The honest calibrated variant now beats the reference beyond paired noise (TPR@3%FPR +12.1 [+5.1, +17.8]); GRADIENT-UA no longer ties (BA -2.3 [-3.1, -1.5]).
- Reference re-run end to end from the final code: hidden test BA 75.75 / F1 79.47 / TPR@1% 5.75 / TPR@3% 16.88. Left to the grader: shadow models (R +3.30) and reading labels from git history (R +6.95). See HARDENING.md.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
