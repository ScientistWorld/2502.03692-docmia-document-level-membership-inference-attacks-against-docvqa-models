# Evaluation Targets

## Primary: scaled_whitebox_docmia

Tests whether a document-level membership inference system can distinguish
training documents from held-out documents in a balanced DocVQA-style setting.

- **Metric**: `f1` (higher is better)
- **Secondary metrics**: `balanced_accuracy`, `accuracy`, `tpr_at_1_fpr`, `tpr_at_3_fpr` (higher is better)
- **Constraint metrics**: `max_questions_per_document`, `uses_auxiliary_data` (lower is better)
- **How to run**: `bash scripts/reproduce.sh`, or run a compatible method that writes `checkpoints/docmia_outputs.json` and then call `bash scripts/evaluate.sh`.
- **Expected output**: a JSON artifact with one or more named methods under
  `features`, `membership_scores`, or `membership_predictions`; each method
  must provide one row per document in the benchmark manifest.
