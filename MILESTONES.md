# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 18:44] - core_claim
- Ran the scaled white-box DocMIA pipeline in a compute job.
- DocMIA FL outperformed score-only and one-step-gradient baselines on the non-trivial synthetic DocVQA benchmark.

### [2026-06-04 18:53] - core_claim_plus
- Verified the hardened evaluator in a compute job.
- Recomputed scores from raw feature arrays without importing method code.

### [2026-06-04 22:58] - core_claim_plus
- Added the missing FLLoRA white-box variant to the method and scoring contract.
- Prepared a new end-to-end job to regenerate method outputs and evaluate FLLoRA from raw features.

## Stop Justification
- Not stopped. A follow-up job is being submitted to validate the FLLoRA expansion.
