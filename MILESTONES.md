# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-06-04 22:00] - core_claim
- Job `34213ac1-050` completed end-to-end and wrote `scoring/scores.json`.
- DocMIA FL reproduced the central optimization-based signal on the scaled benchmark: 66.04 F1, above score-only (55.14 F1) and one-step gradient (53.04 F1).
- Added the paper's IG image-optimization variant for the next expansion run.

### [2026-06-04 22:04] - core_claim
- Job `c775731d-164` completed the FL+IG expanded run.
- IG ran end-to-end but did not support an additional milestone at learning rate 0.01: 49.33 F1 and 0.00 TPR@3%FPR.
- Retuning IG to learning rate 0.001 and increasing target training epochs for the next job.

### [2026-06-04 22:47] - core_claim
- Job `a5f98ef0-266` completed successfully with 160 target epochs and IG learning rate 0.001.
- IG produced strong raw attack scores, but the setting was saturated: the target reached 100% member-document QA accuracy and score/gradient baselines were also near 95 F1.
- Keeping the milestone at `core_claim` and submitting an intermediate 90-epoch run to look for a non-saturated expansion setting.

### [2026-06-04 23:04] - core_claim
- Job `b823b8a9-21e` completed the 90-epoch midpoint run.
- The run remained saturated: score-only and gradient baselines reached 94-95 F1, while DocMIA FL/IG did not improve over them.
- Keeping the milestone at `core_claim` and lowering the next search point to 60 epochs.

### [2026-06-04 23:12] - core_claim_plus
- Job `75544422-7cb` completed the 60-epoch run.
- The setting is non-saturated and supports the central claim beyond the original 45-epoch minimum: DocMIA FL reached 71.58 F1 and 13.54 TPR@3%FPR, above score-only (69.36 F1, 11.46 TPR@3%FPR) and one-step gradient (63.91 F1, 4.17 TPR@3%FPR).
- The expanded IG variant also ran end-to-end at the paper-like image learning rate, but FL is the strongest reproduced proposed variant in this scaled gym.

## Stop Justification
- Stopping at `core_claim_plus`: the method, baselines, evaluator, container
  setup, procedural data, scoring contracts, and final non-saturated 60-epoch
  results are packaged.
