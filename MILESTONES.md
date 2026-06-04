# Reproduction Milestones

**Current: none**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-03 00:00] - none
- Read the paper and identified DocMIA's document-level membership inference protocol.
- Implemented a scaled DocVQA-style benchmark and the paper's white-box FL optimization-distance attack.
- Added score-only and one-step-gradient baselines, isolated evaluation code, scripts, and reference scoring targets.
- Submitted the first compute job to run the full pipeline and determine whether the implementation reaches `method_runs` or `core_claim`.

### [2026-06-04 21:55] - none
- Inspected failed job `b610cdf5-51d`; it failed before method execution due to missing workspace-root import path in the container.
- Added package markers and exported `PYTHONPATH=/home/user` in all scripts.
- `python validate.py --reference-only` passes; resubmitting the full scaled DocMIA run.

## Stop Justification
- Completed at milestone `none`.
