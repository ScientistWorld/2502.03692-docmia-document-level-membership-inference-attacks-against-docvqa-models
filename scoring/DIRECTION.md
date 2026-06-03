# Research Direction

## Research Problem

This benchmark studies document-level membership inference against document
visual question answering models. The scientific question is how to infer
whether a whole document was in training when each document has multiple
question-answer observations.

## Core Contribution

The paper's contribution is auxiliary-data-free membership inference that
aggregates per-question evidence into document-level descriptors. Improvement
should focus on better extracting, aggregating, normalizing, or clustering
membership signals from the target model's behavior.

## Approach Scope

Methods should remain in the DocVQA membership-inference family. They may
modify the feature extraction process, optimization schedule, aggregation
statistics, clustering or scoring rule, and low-FPR calibration. They may add
faithful variants such as image optimization or adapter-based layer
optimization as long as the output remains document-level membership scores.

## Out of Bounds

- Changing the task from document-level membership to QA-pair membership.
- Training supervised attack classifiers on auxiliary member/non-member data.
- Replacing the benchmark with a different privacy task such as data extraction
  or answer reconstruction.
- Beating the score by changing the target data, target model scale, or query
  budget rather than improving membership inference methodology.
