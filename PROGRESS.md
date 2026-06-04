# Progress

## What Works

- Paper analysis is complete for the main DocMIA white-box algorithm.
- A scaled document-level DocVQA benchmark has been implemented in `data/`.
- The target DocVQA model, score baseline, gradient baseline, DocMIA FL attack, and DocMIA IG attack are implemented.
- Evaluation is separated in `eval/` and reads the method output artifact.
- `python validate.py --compare` passes on the FL run from job `34213ac1-050`.
- Package import handling has been fixed for the compute container by exporting
  `PYTHONPATH=/home/user` in all run scripts and adding package markers.

## Results

- Job `34213ac1-050` completed the first scaled white-box run. Reproduced scores:
  - `score_ua`: 56.77 balanced accuracy, 55.14 F1, 3.13 TPR@3%FPR.
  - `gradient_ua`: 55.73 balanced accuracy, 53.04 F1, 2.08 TPR@3%FPR.
  - `docmia_fl`: 62.50 balanced accuracy, 66.04 F1, 9.38 TPR@3%FPR.
- This supports the core claim on the scaled benchmark: optimization-based FL features outperform score-only and one-step gradient baselines without auxiliary data.
- Job `c775731d-164` added IG to the same run. FL stayed at 66.04 F1, but
  `docmia_ig` reached only 49.33 F1 and 0.00 TPR@3%FPR with image learning
  rate 0.01.

## Remaining

- Rerun the expanded job with the paper-like IG learning rate 0.001 and longer
  target training.
- If IG remains weak, keep FL as the reproduced core claim and document IG as an
  unsuccessful secondary variant on this scaled target.
- If target accuracy remains low after longer training, redesign the synthetic
  visual encoding in a later run rather than claiming additional milestones.

## Issues

- The full paper uses DocVQA/PFL-DocVQA and 200M-1.3B parameter public checkpoints. This scaled reproduction uses a procedural DocVQA-style dataset and a compact target model to fit the benchmark budget while preserving the DocMIA computation.
- Job `b610cdf5-51d` failed before training with `ModuleNotFoundError: No module named 'data'` because direct script execution placed `method/` on `sys.path`. The scripts now export the workspace root as `PYTHONPATH`.
- A local import smoke test cannot run because the login environment lacks `torch`; dependencies are specified in `environment/setup.sh` and the previous setup log shows the compute overlay installed them.
- The compact target model's DocVQA accuracy is low on the first scaled run. The membership experiment is still non-trivial and FL outperforms the included baselines, but higher target accuracy is a priority for any later scale-up.
- IG underperformed in job `c775731d-164`, likely because the image learning
  rate 0.01 was too aggressive for this compact CNN target. The next run uses
  0.001, matching the paper's Donut/Pix2Struct IG setting more closely.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- Paper data/model scale: the original work attacks VT5, Donut, and Pix2Struct on DocVQA/PFL-DocVQA. This reproduction uses a compact DocVQA target model and a procedural document dataset because the full checkpoints and 600-document optimization attacks are too costly for the initial core-claim run.
- Attack scale: the paper uses up to 10 questions per document and 200 optimization steps. The first job uses 4 questions and 60 FL steps. The procedure is unchanged: per-QA optimization from the target model, L2 distance/steps/utility aggregation, normalization, and KMeans document clustering.
