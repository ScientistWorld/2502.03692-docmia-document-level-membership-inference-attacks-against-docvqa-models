# Train/Test Evaluation Design Notes

## Checks performed

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; the wrappers point to separate `eval/train/` and `eval/test/` evaluators and write `scoring/scores_train.json` and `scoring/scores_test.json`.
- Checked the slice files; train and test both use balanced 40-member/40-nonmember document slices with disjoint document IDs.
- Checked for train/test cross-imports; the slice evaluators duplicate code and do not import from each other or from `method/`.
- Checked that evaluation is attached to method artifacts; both slice evaluators read per-document outputs from `checkpoints/docmia_outputs.json` rather than re-running the method.
- Checked `scoring/reference.json` for paper-agreement metrics; none of the active metrics use published-value deviation as their objective.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist at handoff.
- Ran `bash -n scripts/evaluate_train.sh` and `bash -n scripts/evaluate_test.sh`; both passed.

## Changes made

- Added threshold-free ranking metrics (`roc_auc`, `average_precision`) and a collapse-resistant classification metric (`matthews_corrcoef`) to both slice evaluators.
- Added `positive_rate_deviation` so a method cannot inflate F1 by drifting too far from the balanced member/nonmember prior.
- Changed missing constraint metadata from an accidental reward to a penalty: missing `max_questions_per_document` now scores as a very large budget, missing `uses_auxiliary_data` scores as auxiliary use, and `constraint_metadata_missing` is emitted.
- Added `label_copy_suspect`, a guard that penalizes raw binary method outputs or feature columns that directly copy the known member/nonmember labels across the source artifact.
- For score-only methods, `eval/train/` now records the visible-slice threshold in train-score calibration metadata; `eval/test/` reuses that threshold when available instead of adapting a median threshold on the held-out slice.
- Added the required `higher_is_better`, `coefficient`, and `description` fields for every active metric in `scoring/reference.json`, and weighted the new triangulation and guard metrics.
- Added explicit `slice: train` and `slice: test` markers to the train/test score artifacts.

## Runtime concerns for the runner iteration

- The synthetic data manifest exposes member/nonmember split labels. The new label-copy guard catches the simplest manifest-copy shortcut, but evaluator-only changes cannot fully prevent a sophisticated method from encoding labels in continuous-looking features. A stronger future design would keep final test labels outside the visible workspace or generate a private final test set.
- The test evaluator expects `scoring/scores_train.json` when score-only calibration and method-list matching are desired. If test is run without a prior train evaluation, score-only methods fall back to a test-slice median threshold.
- The `label_copy_suspect` guard may flag a genuinely perfect binary feature if an attack exactly separates all source documents with a two-valued raw signal. That should be rare for this scaled synthetic DocMIA task, but the runner should inspect if a strong method is penalized only by this guard.
- I did not run the Python evaluators or generate scores, per this iteration's file-edit-only constraint.
