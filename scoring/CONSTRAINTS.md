# Scientist Constraints

## Threat model (what the attack may use)

- The audited model's weights (white-box) and its architecture.
- The candidate documents passed to `scripts/attack.sh` and their
  question-answer pairs. Only those questions may be used; no extra queries
  about other fields.
- The public pre-trained checkpoint `data/pretrained_reader.pt`, which every
  audited model was fine-tuned from, used as-is.
- For development only: the dev audit `data/dev_audit`, including its labels.
  Hyper-parameters and decision rules may be selected on it, as the paper
  tunes its hyper-parameters.

## Not allowed

- **Auxiliary data:**
  - generating or collecting other documents from this distribution (e.g.
    with `data/docgen.py`);
  - regenerating any corpus from its seed;
  - using the dev audit's documents or labels inside the attack at test time
    for anything other than a fixed, pre-selected setting.
- **Shadow or reference model training:**
  - training or fine-tuning any model on documents, including on subsets of
    the candidate set, to imitate the target.
  - The paper's threat model excludes shadow models. Note that per-question
    optimisation of a copy of the target (as in the paper's method) is
    allowed.
- **Changing the audit object or the evaluation:**
  - modifying or replacing `target.pt`, `data/` assets, `eval/` or
    `scripts/evaluate_*.sh`;
  - reading anything under `eval/test`;
  - hard-coding per-document answers.
- **Self-reported numbers:** the evaluator ignores everything except per-document
  scores and decisions.

## Compliance Checklist

- [ ] `scripts/attack.sh` computes scores and decisions from the target and
      the given candidates only.
- [ ] No generated, collected or regenerated documents are used.
- [ ] No model is trained on documents; only per-question optimisation of
      copies of the target or the public checkpoint.
- [ ] Nothing in `eval/`, `data/` assets or the evaluation scripts was
      modified or read for answers.
