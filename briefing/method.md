# Paper's Method

## Key Contribution

DocMIA introduces document-level membership inference attacks for DocVQA
models. Instead of treating a single question-answer pair as the unit of
privacy, it aggregates evidence across all questions associated with a target
document.

## Approach

For each document/question/answer tuple, the attack starts from the target
model's trained state and performs a short optimization process that minimizes
the answer loss for that tuple. It records the L2 distance moved during
optimization, the number of steps until convergence, and the answer utility
observed during the process. These features are aggregated across questions
with average, minimum, maximum, and median statistics, normalized, and clustered
with KMeans. The cluster with lower optimization distance is treated as the
member cluster.

The paper proposes three white-box variants: FL optimizes one selected model
layer, FLLoRA optimizes LoRA parameters attached to that layer, and IG optimizes
the document image while freezing model weights. In the black-box setting, the
attacker first queries the deployed model, trains a proxy model on those
generated labels, and then applies the white-box attacks to the proxy.

## Main Claims

The primary claim is that optimization-derived document descriptors expose
membership more reliably than score, loss, or one-step-gradient baselines, even
without auxiliary data. Secondary claims are that the attack works in both
white-box and black-box settings, transfers through a distilled proxy, and can
be measured in the low-FPR regime used by modern MIA evaluations.
