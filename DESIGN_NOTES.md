# Train/Test Evaluation Design Notes

The original design (fixed slice-ID files, a label-carrying manifest, and test methods
taken from `scores_train.json`) was replaced. HARDENING.md gives the reasons and the
shortcut evidence.

## Audits

- An audit is a target checkpoint plus unlabeled candidates (`target.pt`,
  `candidates/`).
- **Dev audit:** `data/dev_audit`, with its labels in `labels.json`.
- **Hidden audits:** four, in `eval/test/audits/audit_{0..3}`.
  - Built by `eval/test/build_test_audit.py` from the secret seed in
    `eval/test/secret_seed.txt`.
  - Labels are in `eval/test/audit_labels.json`.
  - Integrity record: `eval/test/audit_sha256.json`.
- **Recipe.** Every audit uses the public recipe in `data/target_training.py`: the
  public pre-trained reader fine-tuned for 8 epochs on 600 private documents. Each
  audit has 600 member and 600 non-member candidates, with neutral shuffled ids.

## Evaluators

`eval/train/evaluate.py` (one dev audit) and `eval/test/evaluate.py` (four hidden
audits) are independent code with the same protocol.

1. Load the target and candidates, and compute the five paper baselines from the
   target. On the test side, every hidden file is first checked against its
   recorded hash.
2. Run `bash scripts/attack.sh <copy of the audit> <out>`. On the test side:
   - `eval/test/` and `data/dev_audit/` are sealed (no permissions) for the whole
     attack phase;
   - each audit's copy is written from memory, attacked, and deleted;
   - the attack's process group is killed when it returns.
3. Validate the raw per-document output and recompute every metric. On the test side,
   the labels are read and hash-checked only after access is restored. Metrics are
   averaged over the audits.

The ranked `membership_attack` row is always written. It gets worst values (0) when
the entry is missing or invalid, or when the attack program fails, on any audit.

## What the evaluators never read

- `scores_train.json`;
- any number, flag or threshold written by the attack.

## Known limits (HARDENING.md)

- The hidden labels are committed, so they are also in the git history. That history
  is not sealed, because the job harness commits to the repository while jobs run.
- Shadow-model training cannot be priced out at this model scale.

Both are compliance rules in `scoring/CONSTRAINTS.md`.
