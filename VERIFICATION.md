# Verification Report

## Initial Assessment

The workspace already contained a working scaled reproduction of the paper's
document-level membership-inference setting. The recorded milestone was
`secondary_claims`, with prior compute-job evidence that the core method and
additional white-box variants ran end-to-end and were scored by `eval/` without
importing `method/`. The claimed milestone is acceptable; at minimum,
`core_claim` is justified.

## Issues Found

- Minor: `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`,
  `scoring/TARGETS.md`, and `scoring/DIRECTION.md` exposed more method-specific
  names and implementation direction than from-scratch agents should see.
- Minor: Several diagnostic metrics produced by the train/test evaluators had
  nonzero scoring coefficients even though the verified top-level
  `scores.json` did not contain those metrics.
- Cosmetic: Generated `__pycache__` directories were present in the workspace.

## Fixes Applied

- Reworded method-agnostic and method-safe markdown to describe the task,
  constraints, output format, and evaluation goals without naming the paper's
  implementation variants.
- Kept diagnostic train/test metrics declared in `scoring/reference.json` so
  validation accepts them, but set their coefficients to `0.0`. The scored
  objective remains focused on F1, balanced/ordinary accuracy, low-FPR TPR,
  query budget, and auxiliary-data use.
- Removed local generated Python cache directories, excluding the external
  `shared/` cache.
- Ran `python validate.py --compare`; all scoring and structure checks passed.

## Final Judgment

The workspace is ready for gym use as a scaled synthetic benchmark. Improve-mode
agents can run the provided reproduction and compare against the existing
scores, while from-scratch agents can use `data/`, `eval/`, and the sanitized
problem/scoring documentation without needing the paper's method code.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the benchmark uses a compact synthetic DocVQA-style dataset
  and target model rather than the paper's full DocVQA/PFL datasets and large
  public checkpoints.
