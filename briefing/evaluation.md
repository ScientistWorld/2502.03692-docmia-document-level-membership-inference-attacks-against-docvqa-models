# Evaluation

## Metrics

The attack is evaluated as balanced binary classification over documents.
Metrics are balanced accuracy, ordinary accuracy, F1, TPR at 1% FPR, and TPR at
3% FPR. Higher is better for all attack-success metrics. The evaluation also
records two constraints: maximum questions used per document and whether an
auxiliary dataset was used; lower is better for both.

## Baselines and Targets

The main baselines are score-only unsupervised membership inference and a
single-gradient unsupervised attack. The paper's full-scale white-box Donut on
DocVQA setting reports roughly 80% balanced accuracy and 82% F1 for the
strongest document-level attack, with low-FPR TPR around 10% at 3% FPR.

## Evaluation Protocol

The benchmark uses a balanced member/non-member document set. Each document has
multiple question-answer pairs, and the evaluator reads one output artifact
containing per-method predictions or metrics. The evaluation script does not
import method code; it only converts the generated artifact into
`scoring/scores.json`.
