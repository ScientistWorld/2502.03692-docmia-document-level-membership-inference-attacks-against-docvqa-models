# Paper's Method

## Key Contribution

DocMIA introduces document-level membership inference attacks for DocVQA
models. Instead of treating a single question-answer pair as the unit of
privacy, it aggregates evidence across all questions associated with a target
document, without auxiliary data or shadow models.

## Approach (white-box, Algorithm 1)

For each document/question/answer tuple, the attack starts from the target
model and runs Adam on the answer loss for that tuple until the loss improves
by less than a threshold tau or S = 200 steps are reached. It records the L2
distance moved by the optimised object, the number of steps taken, and the
answer utility (ACC) averaged over the optimisation. These three features are
aggregated over the document's questions with {AVG, MIN, MAX, MED}, standardised,
and clustered with KMeans (k = 2). The cluster with the larger mean distance is
the non-member cluster. A document's membership score is minus its distance to
the member-cluster centroid.

Three variants differ in what is optimised:

- **FL**: the final answer-projection layer.
- **FLLoRA**: LoRA factors on that layer, with A Kaiming-initialised and B zero
  so the start equals the target.
- **IG**: the document image pixels, with the model frozen.

The learning rate and tau are tuned per target model (Appendix C).

## Implementation in this gym

- `method/docmia.py` implements Algorithm 1 for all three variants. Pairs are
  batched, but every pair keeps its own parameters, Adam state and stopping
  step.
- `method/tune_dev.py` runs the Appendix C grid search on the dev audit:
  - lr in {1e-4 ... 1} for FL and FLLoRA, and {1e-3, 1e-2, 1e-1} for IG;
  - tau in {1e-6 ... 10};
  - selection by mean(balanced accuracy, F1).

  It writes `method/config.json`, including the best variant, which becomes the
  submitted `membership_attack`.
- `method/attack.py` (via `scripts/attack.sh`) applies that configuration to
  any audit.

## Main Claims

Optimisation-derived document descriptors expose membership as well as or
better than score, loss and one-step-gradient baselines, especially in the
low-false-positive regime, without auxiliary data. The attack also works
against documents seen through few training questions. The paper's black-box
variant (distilling the target into a proxy) is not part of this gym.
