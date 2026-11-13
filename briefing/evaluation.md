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
