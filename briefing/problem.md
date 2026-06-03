# Problem Definition

## Research Question

Given a trained document visual question answering model, a target document,
and multiple question-answer pairs about that document, decide whether the
document was included in the model's training set. The membership decision is
document-level: all questions attached to the same document must be combined
into a single in/out prediction.

## Why It Matters

Documents often contain sensitive personal, financial, or organizational data.
If a deployed document QA model leaks whether a specific document was used for
training, it exposes privacy risk even when the model only returns answers.

## Success Criteria

A successful system identifies member and non-member documents better than
chance on balanced member/non-member sets, works without auxiliary shadow
datasets, and remains effective when the model's normal question-answer utility
alone is not enough to separate the two classes.
