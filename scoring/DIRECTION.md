# Research Direction

## Research Problem

This benchmark studies document-level membership inference against document
visual question answering models. The scientific question is how to infer
whether a whole document was in training when each document has multiple
question-answer observations.

## Approach Scope

Methods should remain in the DocVQA membership-inference family. They may
use the target model, the target documents, and the provided question-answer
pairs to produce document-level membership scores or predictions. Improvements
should focus on stronger evidence extraction, aggregation across questions,
calibration, and low-FPR decision rules while respecting the same data access
and query constraints.

## Out of Bounds

- Changing the task from document-level membership to QA-pair membership.
- Training supervised attack classifiers on auxiliary member/non-member data.
- Replacing the benchmark with a different privacy task such as data extraction
  or answer reconstruction.
- Beating the score by changing the target data, target model scale, or query
  budget rather than improving membership inference methodology.
