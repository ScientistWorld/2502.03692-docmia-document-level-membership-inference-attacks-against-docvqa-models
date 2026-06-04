# Evaluation

## Metrics

The attack is evaluated as balanced binary classification over documents.
Metrics are balanced accuracy, ordinary accuracy, F1, TPR at 1% FPR, and TPR at
3% FPR. Higher is better for all attack-success metrics. The evaluation also
records two constraints: maximum questions used per document and whether an
auxiliary dataset was used; lower is better for both.

## Baselines and Targets

The benchmark includes multiple reference systems that convert per-document
evidence into membership predictions under the same data access and query
constraints. A useful improvement should raise document-level classification
quality without using auxiliary shadow datasets or increasing the query budget.

## Evaluation Protocol

The benchmark uses a balanced member/non-member document set. Each document has
multiple question-answer pairs, and the evaluator reads one output artifact
containing per-method predictions, scores, or feature arrays. The evaluation
script does not import method code; it only converts the generated artifact
into `scoring/scores.json`.
