# Train/Test Evaluation Design Notes (current, after hardening)

The design before hardening (fixed slice-ID files, a label-carrying manifest,
and test methods taken from `scores_train.json`) was replaced. HARDENING.md
gives the reasons and the shortcut evidence.

- **Audits.**
  - An audit is a target checkpoint plus unlabeled candidates (`target.pt`,
    `candidates/`).
  - The dev audit lives in `data/dev_audit`; its labels are in `labels.json`.
  - The hidden audit lives in `eval/test/audit`, built by
    `eval/test/build_test_audit.py` from a secret seed. Its labels and hashes
    are in `eval/test/audit_labels.json` and `eval/test/audit_sha256.json`.
  - Both audits use the public recipe in `data/target_training.py`: a public
    pre-trained reader fine-tuned for 8 epochs on 600 private documents.
    Every audit has 600 member and 600 non-member candidates, with neutral
    shuffled ids.
- **Evaluators.** `eval/train/evaluate.py` and `eval/test/evaluate.py` are
  independent code with the same protocol:
  1. load labels and target, and verify hashes (test);
  2. compute the five baselines;
  3. run `bash scripts/attack.sh <copy of audit> <out>`;
  4. validate the raw per-document output and recompute every metric.

  The ranked `membership_attack` row is always written; it gets worst values
  (0) when missing, invalid, or when the attack program fails.
- **What the evaluators never read:** `scores_train.json` and any number,
  flag or threshold written by the attack.
