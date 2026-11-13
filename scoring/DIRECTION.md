# Research Direction

## Research Problem

This gym studies document-level membership inference against document visual
question answering models. The question is how to tell, from a model's
weights alone, whether a whole document was in its training data, when each
document is represented by one to five question-answer pairs.

## Approach Scope

Methods stay in the white-box, auxiliary-data-free membership-inference
family. They may use:

- the target model's weights;
- the candidate documents and their question-answer pairs;
- the public checkpoint the target was fine-tuned from;
- the labelled dev audit, for development.

Promising directions:

- more informative per-question evidence;
- better aggregation across a document's questions;
- calibration of per-question evidence for question and document difficulty;
- decision rules and membership scores that hold up at low false positive
  rates.

## Out of Bounds

- Changing the task from document-level to question-level membership.
- Auxiliary documents, shadow models, or any model trained on documents to
  imitate the target; reference models other than the public checkpoint
  (such as the dev audit's target or another audit's target).
- Carrying information from one hidden audit's run to another.
- Supervised attack classifiers trained on membership labels, beyond selecting
  a small number of hyper-parameters on the dev audit.
- Beating the score by changing the target model, the candidate data, the
  question budget or the evaluation.
