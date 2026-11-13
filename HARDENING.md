# Hardening report: does the test reward science now?

**Short answer: yes for honest methods. Two shortcuts that need a deliberate rule
violation still pay and are left to the compliance grader.**

This iteration fixed what `REVIEW.md` listed.

- **The hidden labels are unreadable while the attack runs.**
  - An attack that opens them now fails and gets the worst value on every audit (R −5.19). Before: perfect, R +6.80.
  - The dev audit is locked as well. Using the dev target as a ready-made reference model now fails too (R −5.19; R +0.45 if left readable).
- **The test is four times larger: four independent hidden targets.**
  - The better variant now clears noise: TPR@3%FPR +12.1, 95% interval [+5.1, +17.8]. Before: [+1.5, +29.3].
  - GRADIENT-UA no longer ties the reference: R −0.09, balanced accuracy −2.3 [−3.1, −1.5].
- **Left open, measured and stated below:**
  - shadow-model training (R +3.30);
  - reading the labels from the git history of `eval/test` (R +6.95).

  Both break explicit rules in `scoring/CONSTRAINTS.md`. Neither could be closed without
  breaking either the honest headroom or the job harness.

All numbers are hidden-test results from `eval/test/evaluate.py`, the code behind
`bash scripts/evaluate_test.sh`. Every run of this iteration was re-run from the final
code by `hardening/run_hardening.sh`:

- per-run scores: `hardening/runs/<name>/scores_test.json`;
- paired bootstrap: `hardening/paired_bootstrap_test.json`;
- ranks: `hardening/ranking.json`;
- the previous iteration's evidence: `hardening/archive_iter11/`.

## How the rank is computed

`R = Σ |coef| · dir · (value − reference) / spread` (`hardening/rank.py`).

- The sum runs over the ranked metrics of the `membership_attack` row:
  - balanced accuracy and F1, coefficient 1.0 each;
  - TPR@1%FPR, coefficient 0.4;
  - TPR@3%FPR, coefficient 1.0.
- `spread` is the population standard deviation of each metric over the 30 hidden-test
  runs listed here, failed runs included. The reference is at R = 0.
- Read signs and large gaps, not decimals. The R values of the previous iteration came
  from a different population (24 runs, one audit).
- Coefficients and weights are unchanged. Nothing found here argued for different ones.

## What changed (and why)

| REVIEW.md item | Fix |
|---|---|
| An attack reading `eval/test/audit_labels.json` while the hidden test runs it scored perfectly (+6.80). | `eval/test/evaluate.py` reads every hidden file once and checks it against `audit_sha256.json`. It does not keep the labels in memory. It then removes all permissions from `eval/test/` (and `data/dev_audit/`) for the whole attack phase. Each audit's copy is written fresh from memory, attacked, and deleted before the next. Afterwards access is restored, the labels are read and hash-checked again, and only then is scoring done. The attack's whole process group is killed when it returns, so nothing it started outlives the seal. Access is restored on SIGTERM/SIGHUP, and at the start of `scripts/evaluate_test.sh` / `evaluate_train.sh` in case a run was killed. Each score file records `labels_unreadable_during_attack`; it was `true` in every run here. |
| The honest gain at TPR@3%FPR (+1.5 to +29.3) barely cleared zero; GRADIENT-UA tied the reference. | The hidden test now holds **four independent audits** (`eval/test/audits/audit_{0..3}`). Each is the public reader fine-tuned with the dev audit's exact recipe on its own private set, with 600 member and 600 non-member candidates: 4,800 candidates and 2,400 negatives in total. A **new 256-bit secret seed** was drawn, because the old labels and seed are in git history. The ranked row is the mean over audits of each audit's metric. It gets the worst value if the attack fails or is invalid on **any** audit (checked by `omit_on_one_audit`). Averaging over four targets also averages out target-training randomness. |
| (new, found here) The dev target is a model trained with the target's own recipe on other documents. | `data/dev_audit` is sealed with `eval/test` during the hidden test. Dev-selected settings must be stored with the method. Two variants that read dev at test time now precompute their settings into `hardening/dev_settings.json` (`variants.py --dev-settings`). |
| Rerun the reference and baselines end to end. | Done in the job: hidden audits built → dev tuning from scratch (cache cleared) → train scoring → hidden-test scoring. Tuning reproduced the same configuration (FLLoRA, lr 1e-4, τ 1e-3; FL lr 0.1, τ 0.3; IG lr 0.01, τ 0.1). `scoring/scores_test.json`, `scores_train.json` and `scores.json` come from that reference run. |

The dev audit, the target recipe and the method are unchanged, so the reference config
remains valid. The hidden targets match the dev target's difficulty. Member/non-member
QA accuracy per hidden audit, against 97.5 / 59.5 on dev:

| audit | members | non-members | gap |
|---|---|---|---|
| 0 | 97.0 | 59.1 | 37.9 |
| 1 | 97.5 | 58.7 | 38.8 |
| 2 | 97.6 | 59.0 | 38.6 |
| 3 | 96.1 | 56.7 | 39.4 |

The paper's DocVQA targets have gaps of 36–40.

## Does a better result score higher?

`membership_attack` on the hidden test. Δ is the paired difference from the reference
with its 95% interval (1,000 replicates). Each replicate resamples members and
non-members within each audit, scores both attacks on the same documents, and averages
over audits.

| Run | BA | F1 | TPR@1% | TPR@3% | Δ BA | Δ TPR@3% | R now | R before |
|---|---|---|---|---|---|---|---|---|
| **reference** (DocMIA, FLLoRA selected on dev) | 75.75 | 79.47 | 5.75 | 16.88 | — | — | 0 | 0 |
| calibrated_confidence_median (target − public-reader confidence, top half = members) | 75.88 | 75.88 | 20.83 | 35.42 | +0.1 [−1.4, +1.6] | **+18.5 [+12.0, +24.5]** | **+1.06** | — |
| stronger_combined (reference decisions, scores calibrated on the public reader) | 75.75 | 79.47 | 10.96 | 29.00 | 0 | **+12.1 [+5.1, +17.8]** | **+0.66** | +0.87 |
| calibrated_kmeans | 76.98 | 73.24 | 10.96 | 29.00 | +1.2 [−0.3, +2.6] | +12.1 [+5.1, +17.8] | +0.53 | +0.70 |
| stronger_calibrated (dev-selected member fraction) | 71.54 | 76.48 | 10.96 | 29.00 | −4.2 [−5.5, −2.8] | +12.1 [+5.1, +17.8] | +0.45 | +0.63 |
| weaker_one_step (S = 1) | 70.42 | 76.90 | 4.00 | 10.92 | **−5.3 [−6.2, −4.5]** | −6.0 [−11.7, +0.3] | −0.53 | −0.35 |
| weaker_no_opt (no optimisation) | 72.23 | 77.97 | 0.00 | 0.00 | **−3.5 [−4.4, −2.7]** | **−16.9 [−22.3, −13.0]** | −1.03 | −0.96 |

Before (previous iteration, one hidden audit): the best honest gain at TPR@3%FPR was
+17.2 [+1.5, +29.3], and S = 1 was −0.7 [−14.6, +5.3] there.

Now:

- Every honest improvement clears the paired noise on the metric it improves.
- Both ablations are clearly worse.
- The paper's optimisation effect (no-opt vs reference) is far larger than noise.
- The noise that remains:
  - the targets are four fixed checkpoints;
  - the attack is deterministic (fixed LoRA and KMeans seeds);
  - the intervals above cover test sampling.

The calibration variants use the public pre-trained reader as a reference model for
per-document difficulty. The threat model allows the public checkpoint, and
`scoring/DIRECTION.md` names calibration as a direction. I count this as honest
headroom, not a flaw, for two reasons:

- difficulty calibration is how modern membership attacks gain low-FPR power;
- it is a better method on the paper's problem too, given a public QA checkpoint.

**Caveat:** this gym's public reader was pre-trained on the same document generator and
task. Calibration is therefore probably easier here than with the paper's
self-supervised checkpoints.

## Simple baselines

The evaluator computes these on every run, identically for every submission. "As a
submission" means a copy of the baseline was put in the ranked slot.

| Baseline | BA | F1 | TPR@1% | TPR@3% | as a submission |
|---|---|---|---|---|---|
| LOSS-TA | 78.42 | 82.03 | 1.96 | 6.04 | R −0.42 (Δ TPR@3% −10.8 [−16.4, −5.8]) |
| LOSS, threshold selected on dev | 81.52 | 82.76 | 1.96 | 6.04 | R −0.31 (Δ BA +5.8, Δ TPR@3% −10.8) |
| SCORE-UA | 75.44 | 79.95 | 0.00 | 0.00 | — |
| SCORE-UA_all | 72.23 | 77.97 | 0.00 | 0.00 | = weaker_no_opt, R −1.03 |
| SCORELOSS-UA_all (paper's Table 11 comparison) | 68.67 | 76.06 | 3.50 | 8.71 | — |
| GRADIENT-UA | 73.44 | 78.79 | 6.17 | 16.50 | **R −0.09** (Δ BA −2.3 [−3.1, −1.5], Δ F1 −0.7 [−1.3, −0.1], Δ TPR@3% −0.4 [−5.8, +4.9]) |
| calibrated LOSS (public reader), thresholded at the mean | 69.48 | 64.49 | 13.54 | 23.38 | R −0.12 |

**Paper direction:**

- The reference beats the paper's main comparison, SCORELOSS-UA_all: +7.1 BA, +3.4 F1,
  +8.2 TPR@3%.
- It beats the score-only attacks, and it beats GRADIENT-UA on BA and F1.
- It beats LOSS-TA at low FPR (+10.8 TPR@3%) but loses to it on BA/F1.
- No simple baseline outranks it.

**What does not reproduce:**

- The paper's large BA/F1 margin over LOSS-TA on Donut does not hold here. That margin
  comes from autoregressive token losses, which this closed-answer reader lacks. The
  paper itself has LOSS-TA winning on its most overfit target (VT5/DocVQA).
- The low-FPR advantage over GRADIENT-UA is a tie: the selected FLLoRA setting has a
  small learning rate, so its step counts track one-step gradient size.

## Shortcuts: before and after

"Before" is the previous iteration's hidden test, or REDTEAM.md where marked; "after" is
this hidden test. All shortcut programs are in `hardening/shortcuts.py`. Each is run once
per audit, and its outputs are not kept.

| Shortcut | Before | After | Fix |
|---|---|---|---|
| **read_hidden_labels** (open `eval/test/audit_labels.json` inside the attack) | **+6.80, perfect** | **fails on every audit, R −5.19** | eval/test sealed during the attack; labels loaded only afterwards |
| **dev_target_reference** (calibrate against `data/dev_audit/target.pt`, a ready-made shadow) | R +0.45 measured unsealed (BA 75.7, TPR@3% 25.3) | **fails, R −5.19** | dev audit sealed during the hidden test |
| move_ground_truth (overwrite the labels) | R −1.95 (labels hash-checked) | fails (no write access), R −5.19 | same seal |
| omit_on_one_audit (valid output on 3 of 4 audits) | — (new) | worst values, R −5.19 | any failed audit fails the row |
| omit_methods, empty_output, empty_results, nan_scores, nonbinary_predictions | −5.41 (all 0) | all 0, R −5.19 | unchanged: worst values |
| manifest_labels, constant_one, forged_scores | BA 50, R −1.95 | BA 50, R −1.98 | unchanged |
| filename_labels, threshold_override, bruteforce_generator (0 of 4,800 pages matched) | R −3.76 | R −3.72 | unchanged |
| retrain_fixed_target, declared costs | closed by construction | not rerun | unchanged |

### Left open (stated plainly)

- **read_git_labels: R +6.95, perfect.**
  - The hidden labels and secret seed are committed. A fresh clone must be able to run
    the hidden test, so `git show HEAD:eval/test/audit_labels.json` returns them while
    `eval/test` itself is sealed.
  - Sealing `.git` during the attack would close it. But the job harness commits to this
    repository while jobs run (progress commits every ~20 min), and the hidden test runs
    as a job too. A sealed `.git` would make those commits fail, and a killed job could
    leave the repository unreadable.
  - The repository is packed, so individual objects cannot be sealed.
  - `scoring/CONSTRAINTS.md` forbids reading `eval/test` or its git history. This is
    left to the compliance grader.
- **shadow_lira (the brute force the paper avoids): R +3.30** (BA 86.50, F1 86.67,
  TPR@1% 46.1, TPR@3% 60.4).
  - Eight shadow readers per audit: the public reader fine-tuned on random halves of the
    candidates with the public recipe, then an online likelihood-ratio test. It uses no
    auxiliary documents and takes about 3 CPU-minutes per audit.
  - Not fixed, for two reasons.
    - At this model scale one shadow costs about as much as the IG variant of the paper's
      own method, so no compute budget separates them. A wall-clock limit would make
      scores machine-dependent.
    - Everything that disables shadow training disables the honest public-reader
      calibration as well: removing the public reader, or hiding the recipe (a training
      loop is a few lines). A larger private set than the candidate set (the paper's
      regime) would not stop shadows either, because they calibrate difficulty from the
      candidates alone, and it would need a full rebuild and re-tune.
  - The paper's threat model excludes shadow models, and so do
    `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md`. This is left to the grader.
    It needs conspicuous model training on candidate documents.
- **cross_audit_reference** (keep each hidden target the program is shown, and use it as
  a reference model on the next audits): **R +1.09.**
  - This is a new channel, because the four audits are attacked in sequence.
  - It adds essentially nothing over honest calibration. Its first audit falls back to
    the public reader. `calibrated_confidence_median` is the same rule with the public
    reader on every audit; it scores R +1.06 with an identical TPR@3% (35.42).
  - Sibling targets raise BA/F1 by 2.2 but lower TPR@1% by 4.5.
  - Persisting data across audit runs is forbidden in `scoring/CONSTRAINTS.md`. Left to
    the grader; it does not beat the honest variant.

## Reference score before and after

| | BA | F1 | TPR@1% | TPR@3% | AUC |
|---|---|---|---|---|---|
| Before (one hidden audit) | 76.67 | 80.09 | 4.17 | 14.67 | 84.88 |
| **After (four hidden audits, mean)** | **75.75** | **79.47** | **5.75** | **16.88** | 85.84 |
| per audit 0 / 1 / 2 / 3, TPR@3% | | | | 16.2 / 11.8 / 15.5 / 24.0 | |
| Train (dev audit, unchanged) | 74.42 | 78.49 | 5.50 | 27.00 | 85.38 |

- The dev TPR@3% (27.0) is at the high end of the per-audit range: a single audit's
  low-FPR estimate is noisy, which is why the test now averages four.
- Unranked variant rows (hidden test means), BA / F1 / TPR@1% / TPR@3%:
  - FL: 75.79 / 78.48 / 11.46 / 26.25;
  - IG: 70.27 / 74.49 / 0.21 / 2.46.
- On documents with at most two questions, the reference has member recall 90.1, and
  FL has 85.6. The paper's Table 18 (Donut/DocVQA) reports FL at 80.2.

## Headroom check

- The reference is at the bound of no ranked metric (BA 75.8, F1 79.5, TPR@1% 5.8,
  TPR@3% 16.9).
- Honest variants beat it beyond paired noise on both low-FPR metrics:
  `calibrated_confidence_median` gains TPR@1% +15.1 [+7.5, +20.4] and TPR@3% +18.5
  [+12.0, +24.5].
- No honest variant reaches a bound. Shadow LiRA (BA 86.5, TPR@3% 60.4) shows how much
  membership signal remains.
- No extra experiment was needed.

## Anything not fixed

- The two open shortcuts above: git history and shadow models.
- IG is the weakest variant here (TPR@3% 2.5), so the paper's "IG ≈ FL" does not
  reproduce. The ranked row uses the dev-selected variant, so this does not leak rank.
- TPR@1%FPR allows six false positives per audit. Even averaged over four audits it is
  the noisiest metric, so its coefficient stays at 0.4.
- Not covered by an experiment:
  - black-box proxy distillation;
  - other datasets;
  - rephrased questions;
  - DP-trained targets.
- `validate.py` passes. Its one soft warning flags `eval/test/paired_bootstrap.py`. That
  script reads kept attack outputs through path arguments, so the heuristic misses
  them; it is an analysis script, not a scorer.
- The job was preempted four times. The pipeline resumed from per-step markers and the
  tuning cache.

## Reproduce

- `bash scripts/run.sh` reproduces the reference end to end:
  1. dev tuning from scratch;
  2. train-slice and hidden-test scoring;
  3. validation.
- `bash hardening/run_hardening.sh` rebuilds this evidence: variants, shortcuts, paired
  bootstrap and rank. It is resumable through `hardening/runs/.done_*`. Raw hidden-test
  attack outputs are kept under `eval/test/runs/` and are not committed.
