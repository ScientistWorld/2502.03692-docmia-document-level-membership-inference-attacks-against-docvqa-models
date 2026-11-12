# Problem Definition

## Research Question

Given white-box access to a trained document visual question answering (DocVQA)
model and a set of candidate documents, each with the question-answer pairs
asked about it, decide for every document whether it was part of the model's
training set. The decision is document-level: all question-answer pairs of a
document must be combined into one member/non-member decision and one
membership score.

## Setting

- The audited model is an OCR-free document QA model (page image + question ->
  answer from a closed answer bank). It was obtained by fine-tuning a public
  pre-trained checkpoint (`data/pretrained_reader.pt`) on a private set of
  documents. Its architecture is in `data/target_model.py`; its weights are
  given to the attacker (white-box).
- Documents are scanned forms with five key/value fields; every document comes
  with 1 to 5 question-answer pairs (the questions the model was trained with
  for members). Each page has its own scan noise, ink, blur and placement.
- The candidate set is balanced: half of the candidates are training documents
  of the audited model, half are fresh documents from the same distribution.
- The attacker has no auxiliary data: no other documents from this
  distribution, no shadow or reference models trained to imitate the target,
  and no knowledge of which documents were used for training.

## Why It Matters

Documents carry sensitive personal, financial and business information. If a
deployed document QA model reveals whether a given document was in its
training data, it leaks private information even though it only returns
answers. Document-level membership inference is also the tool an auditor uses
to check whether particular documents were used to train a model.

## Success Criteria

A good attack separates members from non-members well at a balanced operating
point (balanced accuracy, F1) and, more importantly for privacy auditing,
identifies members with very few false positives (true positive rate at 1% and
3% false positive rate), while respecting the no-auxiliary-data threat model.
