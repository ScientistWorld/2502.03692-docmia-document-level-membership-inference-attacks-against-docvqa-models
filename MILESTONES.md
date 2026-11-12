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

### [2026-06-04 23:38] - secondary_claims
- Re-ran local validation, Python compilation, and shell syntax checks.
- Confirmed the packaged scaled DocMIA gym remains at `secondary_claims`.

### [2026-06-04 23:48] - secondary_claims
- Completed final verification against existing successful container jobs and
  current local validation checks.
- Confirmed the workspace is ready as a scaled reusable research gym.

### [2026-06-05 00:02] - secondary_claims
- Rechecked the final workspace and tightened `scripts/reproduce.sh` so the
  end-to-end entrypoint explicitly refreshes all scoring artifacts.

### [2026-11-12] - secondary_claims (hardening iteration)
- Rebuilt the gym around a hidden audit: secret-seeded private set, its own fine-tuned target, labels and hashes only in eval/test; the hidden test runs the submitted attack program and recomputes every metric; baselines computed by the evaluator; one ranked `membership_attack` row with worst values on failure.
- Restored the paper's protocol at small scale: public pre-trained reader fine-tuned on private documents (member/non-member QA 97%/60%), Algorithm 1 with S=200, Adam, Appendix C lr/tau grid on the dev audit, Phi_all + KMeans, FLLoRA starting at the target; 600+600 candidates per audit.
- Hidden test: DocMIA (FLLoRA, dev-selected) BA 76.7 / F1 80.1 / TPR@3%FPR 14.7; beats SCORELOSS-UA_all and score baselines, ties LOSS-TA on BA/F1 while beating it at low FPR, ties GRADIENT-UA. Ablations significantly worse; all REDTEAM shortcuts closed. See HARDENING.md.

## Stop Justification
- Completed at milestone `secondary_claims`.
