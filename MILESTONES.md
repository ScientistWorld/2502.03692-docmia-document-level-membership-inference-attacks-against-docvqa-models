# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 18:47] - core_claim_plus
- Job `75544422-7cb` completed the scaled white-box DocMIA reproduction end-to-end.
- The target model reached a non-trivial member/nonmember QA gap rather than saturated memorization.
- DocMIA FL outperformed score-only and one-step-gradient baselines on F1 and TPR@3%FPR, while DocMIA IG also ran as a second optimization-based variant.
- `python validate.py --compare` passed with import separation and evaluator-output checks.

### [2026-06-04 18:53] - core_claim_plus
- Job `73b730b0-831` passed after hardening the evaluator to recompute metrics from raw DocMIA features and labels.
- The recomputed scores match the final 60-epoch artifact: DocMIA FL reaches 71.58 F1 and 13.54 TPR@3%FPR, ahead of the included score-only and one-step-gradient baselines on the core comparison.
- `python validate.py --compare` passed in the compute container.

## Stop Justification
- Completed at milestone `core_claim_plus`.
