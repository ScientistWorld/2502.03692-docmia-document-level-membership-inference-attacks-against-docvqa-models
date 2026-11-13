# Overview

- **Paper ID:** 2502.03692
- **Title:** DOCMIA: Document-level Membership Inference Attacks Against DocVQA Models
- **Domain:** Document AI / Privacy / Membership Inference
- **TL;DR:** The paper shows that DocVQA models leak whether a whole document was
  in their training data, by aggregating optimisation-based signals across the
  document's questions, without auxiliary data.

## Short Summary

DocMIA studies membership inference against document visual question
answering models at document granularity. A training document is seen through
several question-answer pairs. The attack adapts a small part of the target
model, or the input image, for each question. It aggregates the resulting
optimisation traces over the document and clusters documents into members and
non-members. The paper reports that these attacks match or beat score, loss
and gradient baselines on DocVQA and PFL-DocVQA targets.

## This gym

This is a scaled white-box reproduction:

- **Documents:** procedural forms.
- **Target:** a ~1.1M-parameter OCR-free reader, fine-tuned from a public
  pre-trained checkpoint. Its member/non-member QA accuracy is ~97% vs ~60%,
  similar to the paper's DocVQA targets.
- **Candidates:** 600 member and 600 non-member documents per audit.
- **Test:** hidden. It has four independent audits, each with its own target,
  and the metrics are averaged over them.

## Key Results (paper, Donut on DocVQA, white-box)

- FL: 80.33 accuracy, 82.18 F1, 10.67 TPR at 3% FPR.
- Strongest baseline, SCORELOSS-UA_all: 80.50 accuracy, 81.10 F1, 8.67 TPR at
  3% FPR.
- Across targets, the white-box DocMIA variants reach about 72-82 F1.
