# Experiment Overview

## scaled_whitebox_docmia

This experiment asks whether a trained document QA model reveals which whole
documents it was trained on, when the auditor holds the model's weights but
nothing else. In particular, the auditor has no other documents from the same
source and no imitation models.

Each audit pairs one fine-tuned model with a balanced set of candidate
documents. Half are documents it was trained on and half are fresh documents
of the same kind. Every candidate comes with the question-answer pairs asked
about it, which can be as few as one. The submitted attack must output, for
every document, a membership score and a member/non-member decision.

The experiment establishes two things:

- how well members can be separated from non-members at a balanced operating
  point (balanced accuracy, F1);
- how many members can be identified while almost never accusing a
  non-member (true positive rate at 1% and 3% false positive rate). This is
  the regime that matters for privacy auditing.

The hidden test uses a different model and different documents from the
visible development audit, built the same way. Only the submitted
`membership_attack` row is ranked. The evaluator computes the comparison
baselines itself from the audited model, so they are fixed for every
submission.
