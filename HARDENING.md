# Hardening report: does the test reward science now?

**Short answer: mostly yes.**

- **Better variants score higher, beyond paired noise.** A variant that keeps
  the reference's decisions but ranks documents better outranks it
  (TPR@3%FPR +17.2 [+1.5, +29.3], R = +0.87). The paper's ablations score
  lower (balanced accuracy −5.5 to −6.9, intervals exclude 0).
- **All 15 shortcuts in REDTEAM.md are closed.** 13 of them were rerun through
  the new hidden test and now fail. Two are closed by construction and were not
  rerun: the declared-cost shortcuts (costs are no longer read) and retraining
  the target (the target is held by `eval/test`).
- **Two things still pay and are left to the compliance grader:**
  - reading the hidden labels from `eval/test` inside the attack;
  - the shadow-model brute force the paper's threat model rules out.

  Both are measured below.
- **Paper baselines:** a copy of LOSS-TA no longer outranks the reference
  (R = −0.27). A copy of GRADIENT-UA ties it (R = +0.03, mixed signs) — stated
  plainly, not fixed.

All numbers are hidden-test results from `eval/test/evaluate.py` (the code
behind `bash scripts/evaluate_test.sh`). Per-run score files are in
`hardening/runs/<name>/scores_test.json`. Paired bootstrap intervals
(2000 resamples of the same test documents) are in
`hardening/paired_bootstrap_test.json`, and ranks in `hardening/ranking.json`.

## How the rank is computed

`R = Σ |coef| · dir · (value − reference) / spread` (`hardening/rank.py`).

- The sum runs over the ranked metrics of the `membership_attack` row.
- `spread` is the population standard deviation of each metric across the 24
  hidden-test runs listed here, failed runs included.
- The reference sits at R = 0.

This population stands in for the unknown future one, so read signs and large
gaps, not decimals.

## What changed (and why)

### Leaks and trust

| Problem (REDTEAM.md) | Fix |
|---|---|
| Labels visible: the manifest `split` field, `member_`/`nonmember_` filenames, and a public seed-1337 generator whose 96 training pages could be enumerated | The old 192-document dataset was deleted. The hidden audit's private set comes from a 256-bit secret seed in `eval/test/secret_seed.txt`. Candidate ids are neutral and shuffled. Labels exist only in `eval/test/audit_labels.json`. |
| Ground truth read from an agent-editable manifest | The evaluator loads labels, the target and the candidates before the attack runs, and checks them against `eval/test/audit_sha256.json` (labels included). |
| Audit object not authenticated (retraining the target paid +24.1) | The hidden target is part of `eval/test` and hash-checked. The attack only receives a copy. |
| Test method list taken from editable `scores_train.json` (empty or omitted lists tied the reference) | The test never reads `scores_train.json`. The ranked `membership_attack` row is always written, with worst values (0) when it is missing or invalid, or when the attack program fails. |
| Declared costs trusted (zero, −1e12, NaN; one shortcut paid +12.8) | Declarations are no longer read. The question budget is structural: the attack only receives each document's question-answer pairs. Auxiliary data and shadow models are compliance rules (`scoring/CONSTRAINTS.md`). Scores must be finite and decisions exactly 0/1, otherwise the row scores 0. |
| Malformed outputs crashed the evaluator without writing rows | Any attack or evaluator failure still writes worst-value rows. |
| Test thresholds read from editable train calibration | Decisions come only from the attack's own per-document output. |
| Baselines produced by the method | The evaluator computes the five paper baselines (LOSS-TA, SCORE-UA, SCORE-UA_all, SCORELOSS-UA_all, GRADIENT-UA) from the audited model, identically for every submission. They are reported under `baselines` and are not ranked. |
| Visible test inputs | The hidden test runs the submitted attack program (`scripts/attack.sh`) on inputs only `eval/test` holds. Train and test use the same interface, through independent evaluators (`eval/train`, `eval/test`). |

### Reproduction restored toward the paper

| Before | After |
|---|---|
| Target: a tiny CNN trained from scratch on 96 forms for 60 epochs. It barely read the documents (members 42% / non-members 20% QA accuracy). | The paper's protocol: a public pre-trained reader (`data/pretrained_reader.pt`, 3000 public documents) fine-tuned on 600 private documents. Members 97.2% / non-members 60.2% QA accuracy on test (dev 97.5% / 59.5%), a gap like the paper's DocVQA targets (36-40 points). |
| 4 questions per document, 60 steps, τ = 1e-5 untuned, IG clamped. | 1-5 questions per document. S = 200 with Adam. lr and τ grid-searched on the dev audit (paper Appendix C grid, τ up to 10). IG unclamped. |
| FLLoRA initialised both LoRA factors with Kaiming, so step 0 already perturbed the target. | A is Kaiming-initialised and B = 0, so optimisation starts at the target (Algorithm 1: θ₀ = θₜ). |
| 80 test documents (40 negatives), TPR at fixed FPR interpolated. | 1200 test candidates (600/600). TPR is the largest TPR with FPR ≤ x, no interpolation. |

One departure from the paper's setup remains:

- **Fixed fine-tuning schedule.** The paper early-stops on validation
  performance. Here the validation curve is flat and noisy after two epochs,
  so early stopping picked epoch 2 for one seed and epoch 8 for another (gaps
  of 15 vs 36 points). Train and test would then measure different
  difficulties, so every target uses a fixed 8 epochs (`data/target_training.py`).

### Scoring changes

- **One ranked row: `membership_attack`.** This is the attack the submission
  puts forward; for the reference it is the DocMIA variant selected on dev by
  the paper's criterion, mean(BA, F1), which picked FLLoRA. The three
  per-variant rows are still reported, unranked.
  - Why: with three ranked per-variant slots, pasting LOSS-TA into the two
    weak slots outranked the reference (job 2, archived in
    `hardening/archive_job2`: R = +1.55, +1.95 for LOSS-TA copies). The gain
    came only from weak variants, not from better science.
- **TPR@3%FPR coefficient raised from 0.5 to 1.0.**
  - Why: it is the paper's main-text low-FPR table (Tables 2 and 4), and the
    regime the paper ("following Carlini et al.") and the privacy literature
    treat as the meaningful one for auditing.
  - Unchanged: TPR@1%FPR stays at 0.4 (only 6 false positives allowed, too
    noisy); BA and F1 stay at 1.0; AUC remains diagnostic (0).
- **Metrics dropped:**
  - `accuracy` duplicated BA on balanced sets.
  - `max_questions_per_document` and `uses_auxiliary_data` were unverifiable
    self-declarations.
  - `matthews_corrcoef`, `average_precision`, `predicted_member_rate_imbalance`,
    `constraint_metadata_missing` and `label_copy_suspect` were diagnostics of
    the old artifact format.
- **`scoring/reference.json`:**
  - The paper's numbers are unchanged. The old `score_ua` entry held the
    paper's SCORE-UA_all values and is now labelled as such; SCORE-UA, LOSS-TA
    and SCORELOSS-UA_all were added from Tables 1, 12 and 16.
  - Per-variant paper values and the paper's Table 18 values are kept under
    `_paper_variants`.
- **Few-question analysis:** a separate few-question experiment was tried and
  dropped. The paper reports no FLLoRA value for it, and it duplicated the main
  run. It is kept as an unranked diagnostic on the 478 test documents with at
  most two questions.

## Does a better result score higher?

`membership_attack` row on the hidden test. Δ is the paired difference from
the reference with its 95% bootstrap interval.

| Run | BA | F1 | TPR@1% | TPR@3% | Δ BA | Δ TPR@3% | R |
|---|---|---|---|---|---|---|---|
| **reference** (DocMIA, FLLoRA selected on dev) | 76.67 | 80.09 | 4.17 | 14.67 | — | — | 0 |
| stronger_combined (reference decisions + scores calibrated on the public reader) | 76.67 | 80.09 | 11.00 | 31.83 | 0 | **+17.2 [+1.5, +29.3]** | **+0.87** |
| calibrated_kmeans (calibrated DocMIA, KMeans) | 77.33 | 73.07 | 11.00 | 31.83 | +0.7 [−2.3, +3.6] | +17.2 [+1.5, +29.3] | +0.70 |
| stronger_calibrated (calibrated DocMIA, dev-tuned member fraction) | 72.17 | 77.00 | 11.00 | 31.83 | −4.5 [−7.0, −1.6] | +17.2 [+1.5, +29.3] | +0.63 |
| weaker_one_step (S = 1) | 69.75 | 76.57 | 4.17 | 14.00 | **−6.9 [−8.7, −5.0]** | −0.7 [−14.6, +5.3] | −0.35 |
| weaker_no_opt (utility only, no optimisation) | 71.17 | 77.44 | 0.00 | 0.00 | **−5.5 [−7.3, −3.6]** | −14.7 [−27.1, −8.9] | −0.96 |

**Before this iteration (REDTEAM.md, old gym):**

- A train-selected mean-loss rule scored +2.48.
- One step scored −17.34.
- Removing optimisation altogether, −0.72, was indistinguishable from the
  3-seed spread of 1.30.

**Now:**

- Both ablations are significantly worse than the reference; the paper's
  optimisation effect is larger than test noise.
- A variant that strictly improves the ranked metrics (same decisions, better
  low-FPR ranking) scores higher beyond paired noise.
- The calibrated variants trade some F1 for large low-FPR gains and rank
  above the reference.
- Noise sources:
  - The audited target is a single fixed checkpoint.
  - The attack is deterministic (fixed KMeans and LoRA seeds).
  - The paired bootstrap intervals above are the remaining test-sampling
    noise; BA/F1 intervals are about ±2 points.

## Simple baselines (headroom vs. copying)

Computed by the evaluator on the hidden test. Copies were also run as
submissions to obtain R.

| Baseline | BA | F1 | TPR@1% | TPR@3% | as a submission: R |
|---|---|---|---|---|---|
| LOSS-TA | 77.17 | 81.13 | 3.33 | 7.67 | −0.27 (Δ TPR@3% −7.0 [−19.9, −0.6]) |
| LOSS, threshold tuned on dev (≈ REDTEAM's "stronger train-selected") | 80.08 | 81.57 | 3.33 | 7.67 | −0.16 (Δ BA +3.4 [+1.6, +5.3]) |
| SCORE-UA | 74.92 | 79.79 | 0.00 | 0.00 | — |
| SCORE-UA_all | 71.17 | 77.44 | 0.00 | 0.00 | — (= weaker_no_opt, −0.96) |
| SCORELOSS-UA_all (the paper's Table 11 comparison) | 69.25 | 76.39 | 5.17 | 10.83 | — |
| GRADIENT-UA | 74.25 | 79.36 | 5.50 | 17.17 | **+0.03** (Δ BA −2.4 [−4.0, −0.7], Δ TPR@3% +2.5 [−10.5, +17.5]) |

**Paper direction.** The reference beats the paper's main comparison baseline
SCORELOSS-UA_all (+7.4 BA, +3.7 F1, +3.8 TPR@3%) and the score-only attacks.
It matches LOSS-TA on BA/F1 while beating it at low FPR. This is the paper's
"best or near-best, strong in the low-FPR regime" claim.

**What does not reproduce.**

- The paper's large margin over LOSS-TA on Donut (51.8 vs 80.3 accuracy) does
  not reproduce. That margin comes from autoregressive token losses, which this
  closed-answer reader does not have. The paper itself reports LOSS-TA winning
  on its most overfit target (VT5/DocVQA).
- GRADIENT-UA is statistically tied with the reference: lower BA, higher AUC,
  mixed TPR. This is left as is. It is an honest paper baseline that the test
  cannot tell apart from the reference, a headroom limitation rather than a
  loophole.
- No simple baseline outranks the reference by more than noise.

**Headroom.** The reference sits at the bound of no ranked metric:

- BA 76.7, F1 80.1, TPR@1% 4.2 and TPR@3% 14.7, all far below 100.
- The calibrated variants show that allowed, method-level ideas move them.
- The shadow-model run below shows how much more signal exists (BA 85,
  TPR@3% 59).
- No extra experiment was needed.

## Shortcuts: before and after

"Before" is REDTEAM.md's R in the old gym (5 slots, 80 test docs). "After" is
this hidden test (1 slot, 1200 docs). The R scales differ; read the verdicts.

| Shortcut | Before | After (hidden test) | Fix |
|---|---|---|---|
| manifest_labels | **+27.43, perfect** | BA 50, F1 66.7; R −1.95 | No labels in the candidate manifest |
| filename_labels | **+27.43, perfect** | BA 50, F1 0; R −3.76 | Neutral shuffled ids |
| bruteforce_generator (regenerate public corpora, match pages) | **+27.43, perfect** | 0 of 1200 pages matched; R −3.76 | Secret-seeded private set |
| move_ground_truth (overwrite labels) | **+27.43** | BA 50; R −1.95 | Labels loaded and hash-checked before the attack runs |
| retrain_fixed_target | **+24.14** | not rerun (closed by construction) | Hidden target held and hash-checked by `eval/test`; the attack gets a copy |
| impossible_costs (−1e12) | **+12.84** | not rerun: costs no longer read | Declarations removed. The `nonbinary_predictions` analogue scores 0; R −5.41 |
| zero_declared_cost | **+9e-12** (tie) | not rerun: costs no longer read | Question budget is structural |
| nan_costs | NaN accepted | `nan_scores`: row scores 0; R −5.41 | Finite scores required |
| empty_results | **0 (tie)** | all 0; R −5.41 | Fixed ranked row, worst values when missing |
| omit_methods | **0 (tie)** | all 0; R −5.41 | Same |
| null_costs / empty_output | crash, no rows | `empty_output`: all 0; R −5.41 | Failure writes worst rows |
| forged_scores | fails (−12.87) | still fails (BA 50); R −1.95 | Evaluator recomputes; forged file overwritten |
| threshold_override | fails (−12.87) | still fails (BA 50); R −3.76 | Train calibration never read |
| constant_one | fails (−12.87) | still fails (BA 50); R −1.95 | — |

### Left open (stated plainly)

- **`read_hidden_labels`: R = +6.80, perfect.**
  - The hidden test necessarily runs the submitted code in a workspace that
    contains `eval/test`. An attack that opens `eval/test/audit_labels.json`
    scores 100.
  - Hashing, copying the audit or chmod cannot stop code running as the same
    user.
  - The compliance grader must reject any attack that reads `eval/test`;
    `scoring/CONSTRAINTS.md` forbids it.
- **`shadow_lira` (brute force the paper avoids): R = +2.98.**
  - Eight shadow targets, the public reader fine-tuned with the public recipe
    on random halves of the candidates, followed by an online likelihood-ratio
    test.
  - Score: BA 85.17, F1 85.58, TPR@1% 38.5, TPR@3% 58.8 (Δ TPR@3% +44.2
    [+30.1, +52.0]), in about 6 CPU-minutes.
  - It uses no auxiliary documents, only compute and the public recipe and
    checkpoint.
  - It cannot be scaled away here: making shadow training infeasible would need
    targets far costlier to train than this budget allows (the gym itself must
    train two of them). A wall-clock limit would make scores machine-dependent.
  - The paper's threat model explicitly excludes shadow models, and
    `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` forbid training any
    model on documents. This is left to the grader.
  - The same constraint covers the weaker form: generating auxiliary documents
    with `data/docgen.py`.
- **GRADIENT-UA copy ties (R = +0.03).** Discussed above.

## Reference score before and after

**Before (REDTEAM replay; 80 test docs; three ranked slots):**

| Variant | BA | F1 | TPR@1% | TPR@3% |
|---|---|---|---|---|
| FL | 76.25 | 75.95 | 17.5 | 17.5 |
| FLLoRA | 75.00 | 75.00 | 15.0 | 15.0 |
| IG | 75.00 | 77.27 | 7.5 | 15.0 |

**After (hidden test, 1200 docs):** the ranked `membership_attack` (FLLoRA)
scores BA 76.67, F1 80.09, TPR@1% 4.17, TPR@3% 14.67 (AUC 84.88).

- Unranked variant rows: FL 75.25 / 78.05 / 10.33 / 20.00; IG 71.75 / 75.16 /
  0.17 / 1.50.
- Dev (train slice): BA 74.42, F1 78.49, TPR@1% 5.50, TPR@3% 27.00.
- Train and test differ only within the wide TPR@3% interval (about ±12).

Hyper-parameters selected on dev (`method/config.json`):

| Variant | lr | τ | Tuning set |
|---|---|---|---|
| FL | 0.1 | 0.3 | 1200 dev docs |
| FLLoRA | 1e-4 | 1e-3 | 1200 dev docs |
| IG | 0.01 | 0.1 | 600 dev docs (IG costs ~20× more per step) |

`scoring/scores_test.json`, `scoring/scores_train.json` and
`scoring/scores.json` come from the reference runs alone, and `validate.py`
passes. Its one soft warning is a heuristic flag on
`eval/test/paired_bootstrap.py`: an analysis script that reads kept attack
outputs, not a scorer.

## Anything not fixed

- IG is the weakest variant here, especially at low FPR (TPR@3% 1.5 on test).
  This holds with the paper's lr/τ grid. The ranked row uses the dev-selected
  variant, so this no longer leaks rank, but the paper's "IG ≈ FL" result does
  not reproduce.
- TPR@1% FPR (6 false positives allowed) remains noisy even with 600
  negatives; its coefficient stays low (0.4).
- Not covered by this gym (no scored experiment):
  - the paper's black-box proxy-distillation attacks;
  - other datasets and targets;
  - rephrased questions;
  - DP-trained targets.
- The rank's normalisation population is this report's 24 runs. A different
  future population can change tradeoff ranks, though not strict dominance.
- Exploration that led to the final regime: the first reader could not learn
  (chance accuracy), early stopping gave inconsistent targets, and compute jobs
  were pre-empted. These attempts are logged in `hardening/exploration/` and
  `hardening/archive_job2/`.

## Reproduce

- `bash scripts/run.sh`: the reference end to end (dev tuning, then train and
  hidden-test scoring, then validation).
- `bash hardening/run_hardening.sh`: rebuilds the evidence (variants,
  shortcuts, bootstrap, rank). It is resumable via
  `hardening/runs/.done_*` markers.
