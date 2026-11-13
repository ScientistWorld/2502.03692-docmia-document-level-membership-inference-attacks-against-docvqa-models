# Evaluation Targets

## Primary: scaled_whitebox_docmia (weight 1.0)

Document-level, auxiliary-data-free membership inference against a white-box
document QA model, on a balanced candidate set.

- **Ranked row:** `membership_attack`, the attack the submission's
  `scripts/attack.sh` writes under that name.
- **Headline metric:** `f1` (higher is better).
- **Other ranked metrics:** `balanced_accuracy`, `tpr_at_1_fpr`, `tpr_at_3_fpr`
  (higher is better). Coefficients are 1.0, 0.4 and 1.0, with 1.0 for F1.
- **Diagnostic only:** `roc_auc`; unranked per-attack rows (`other_attacks`);
  the evaluator's baselines (`baselines`); and metrics on documents with at
  most two question-answer pairs (`diagnostics`).
- **Hidden test:** four independent audits. Each metric is averaged over them,
  and a failure on any audit scores 0 on every metric.
- **Constraints, enforced as validity or compliance rather than ranked:**
  - The attack sees only the target weights and the unlabeled candidates, with
    their question-answer pairs, so the question budget is structural.
  - Invalid or missing outputs score 0 on every metric.
  - The hidden labels and the dev audit are unreadable while the attack runs.
  - No auxiliary documents, no shadow or reference models other than the public
    checkpoint, and no information carried between audits (see
    CONSTRAINTS.md).
- **How to run:**
  - `bash scripts/evaluate_train.sh` scores the visible dev audit and writes
    `scoring/scores_train.json`.
  - `bash scripts/evaluate_test.sh` runs the hidden test (no arguments) and
    writes `scoring/scores_test.json`.
  - `bash scripts/reproduce.sh` reproduces the reference end to end.
- **Expected output:** see `briefing/evaluation.md` for the attack output
  format.
