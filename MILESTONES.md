# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-06-04 23:02] - core_claim_plus
- Audited the continuation workspace and confirmed the existing FL, IG, score, and gradient implementations preserve the paper's DocMIA computation at scaled size.
- Hardened `eval/` so standalone scoring works for both fresh `doc_ids` artifacts and older artifacts that follow manifest document order.

### [2026-06-04 23:11] - secondary_claims
- Compute job `f60d7a61-e05` regenerated the full scaled DocMIA artifact with FLLoRA included.
- `scoring/scores.json` now contains score, gradient, FL, FLLoRA, and IG results from the independent evaluator, and `python validate.py --compare` passed in the compute container.

## Stop Justification
- Completed at milestone `secondary_claims` after validating the scaled
  white-box FL, FLLoRA, and IG variants with independent scoring.
