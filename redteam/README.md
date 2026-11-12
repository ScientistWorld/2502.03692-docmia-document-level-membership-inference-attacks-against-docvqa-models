# Benchmark audit runs

Run `bash redteam/run_all.sh` inside the gym compute container. Every case also
has its own `run.sh`. These scripts install an ordinary artifact at the default
checkpoint path and invoke the unmodified `bash scripts/evaluate_test.sh` with
no arguments in a new process. No case imports the test evaluator, reads test
IDs while constructing outputs, or changes evaluator code or Python packages.
The runner saves each case's artifacts, train and test scores, logs, exit
status, and input hashes in that case's directory, then restores affected gym
files. Cases that edit a visible manifest or train-score file leave those edits
in place for the hidden-test subprocess, as a future attempt would.

`reference_replay/input.json` is a copy of the supplied reference output;
`_backup/` holds the supplied ignored target checkpoint for reuse. When it is
absent, the runner regenerates that checkpoint with the published gym recipe
(60 epochs, seed 7), without needing hidden data. The source gym's method code
is needed for the faithful reruns and optimization variants. The data-only
shortcuts work with an empty method directory.

All names beginning `shortcut_` are shortcuts, including tied results. A failed
forgery is recorded as a failed shortcut, not credited for leftover scores.
The final report documents ranking normalization and the limits of estimates
when the population of future ranked attempts is unknown.

Read `../REDTEAM.md` for the audit conclusion and per-shortcut verdicts.
`ranking.json` contains all normalized contributions and raw scale estimates;
`audit_summary.json` contains the reference values and three-seed ranges.
Run `python3 redteam/check_evidence.py` for a lightweight consistency check of
the saved results. It does not train a model or invoke an evaluator.
