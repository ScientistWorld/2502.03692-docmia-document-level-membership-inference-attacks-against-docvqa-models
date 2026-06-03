# Scientist Constraints

## Data

Use the provided `data/synthetic_docvqa` generation procedure and the same
member/non-member split semantics. Do not add external document data or train
on the non-member split. The benchmark is about membership inference
methodology, so improving scores by making the target data easier or by adding
extra documents is not a valid improvement.

## Target Model

Use the provided `TinyDocVQAModel` checkpoint format and comparable parameter
scale for the target DocVQA model. Do not replace it with a larger pretrained
vision-language model for the scored experiment. The target model is the object
being audited, not the contribution being optimized.

## Query Budget

Respect the reported `max_questions_per_document` constraint. A method can use
fewer questions, but using more than the configured benchmark budget should be
reported and will be penalized.

## Auxiliary Data

Do not use auxiliary or shadow datasets for `scaled_whitebox_docmia`. The paper's
claim is explicitly auxiliary-data-free membership inference.

## Compliance Checklist

- [ ] Uses only the provided synthetic DocVQA member/non-member data.
- [ ] Does not train on non-member documents.
- [ ] Uses the same target-model family and checkpoint format.
- [ ] Reports the maximum number of questions used per document.
- [ ] Reports whether auxiliary data was used.
