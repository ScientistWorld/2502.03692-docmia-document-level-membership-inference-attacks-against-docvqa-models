# DocMIA: Document-Level Membership Inference Attacks against DocVQA Models

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2502.03692"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2502.03692-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2502.03692](https://arxiv.org/abs/2502.03692) |
| **Domain** | Document AI / Privacy / Membership Inference |
| **TL;DR** | The paper shows that DocVQA models can leak whether a whole document was in training data by aggregating optimization-based signals across that document's questions. |

## Summary

DocMIA studies membership inference for document visual question answering
models at document granularity. The paper exploits the fact that one training
document typically appears through multiple question-answer pairs. Its main
attack repeatedly adapts a small part of the target model or input for each
question, aggregates the resulting optimization traces, and clusters documents
into member/non-member groups. The reported attacks outperform score, loss, and
gradient baselines on DocVQA and PFL-DocVQA targets.

## Problem

Given a trained document visual question answering model, a target document,
and multiple question-answer pairs about that document, decide whether the
document was included in the model's training set. The membership decision is
document-level: all questions attached to the same document must be combined
into a single in/out prediction.

## The paper's method

DocMIA introduces document-level membership inference attacks for DocVQA
models. Instead of treating a single question-answer pair as the unit of
privacy, it aggregates evidence across all questions associated with a target
document.

## Key results of the paper

- White-box DocMIA reaches about 72-82 F1 across reported model/dataset pairs.
- On Donut/DocVQA, FL reports 80.33 balanced accuracy and 82.18 F1.
- Black-box proxy attacks often improve over score-only black-box baselines,

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +166.9 |
| `gpt-5.6-sol-xhigh` | +161.5 |
| `gpt-5.6-luna-xhigh` | -4.6 |
| `gpt-5.6-terra-xhigh` | -44.0 |
| `deepseek-v4-pro-xhigh` | -51.8 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2502.03692 --mode innovate --agent claude-code
sw evaluate 2502.03692
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

