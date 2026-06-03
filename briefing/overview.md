# Overview

- **Paper ID:** 2502.03692
- **Title:** DOCMIA: Document-level Membership Inference Attacks Against DocVQA Models
- **Domain:** Document AI / Privacy / Membership Inference
- **TL;DR:** The paper shows that DocVQA models can leak whether a whole document was in training data by aggregating optimization-based signals across that document's questions.

## Short Summary

DocMIA studies membership inference for document visual question answering
models at document granularity. The paper exploits the fact that one training
document typically appears through multiple question-answer pairs. Its main
attack repeatedly adapts a small part of the target model or input for each
question, aggregates the resulting optimization traces, and clusters documents
into member/non-member groups. The reported attacks outperform score, loss, and
gradient baselines on DocVQA and PFL-DocVQA targets.

## Key Results

- White-box DocMIA reaches about 72-82 F1 across reported model/dataset pairs.
- On Donut/DocVQA, FL reports 80.33 balanced accuracy and 82.18 F1.
- Black-box proxy attacks often improve over score-only black-box baselines,
  with notable low-FPR gains on DocVQA targets.
