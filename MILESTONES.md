# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 18:47] - core_claim_plus
- Job `75544422-7cb` completed the scaled white-box DocMIA reproduction end-to-end.
- The target model reached a non-trivial member/nonmember QA gap rather than saturated memorization.
- DocMIA FL outperformed score-only and one-step-gradient baselines on F1 and TPR@3%FPR, while DocMIA IG also ran as a second optimization-based variant.
- `python validate.py --compare` passed with import separation and evaluator-output checks.

## Stop Justification
- Completed at milestone `core_claim_plus`.
- Further expansion to full DocVQA/PFL-DocVQA with VT5, Donut, and Pix2Struct is outside the remaining practical budget for this reproduction, but the environment is packaged and reusable for the central white-box claim.
