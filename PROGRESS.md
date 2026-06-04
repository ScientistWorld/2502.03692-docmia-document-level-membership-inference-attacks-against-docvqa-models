# Progress

## What Works

- Paper analysis is complete for the main DocMIA white-box algorithm.
- A scaled document-level DocVQA benchmark has been implemented in `data/`.
- The target DocVQA model, score baseline, gradient baseline, and DocMIA FL attack are implemented.
- Evaluation is separated in `eval/` and reads the method output artifact.
- `python validate.py --reference-only` passes.
- Package import handling has been fixed for the compute container by exporting
  `PYTHONPATH=/home/user` in all run scripts and adding package markers.

## Results

- No reproduced scores yet. The first job will train the scaled target model, run DocMIA FL, and write `scoring/scores.json`.

## Remaining

- Rerun the submitted job and inspect `scoring/scores.json`.
- If FL clearly beats trivial chance and is competitive with the baselines, update to `core_claim`.
- If results are weak, tune training epochs, FL steps, threshold, or dataset difficulty and resubmit.
- After the core claim, add FLLoRA or IG as an additional setting if budget remains.

## Issues

- The full paper uses DocVQA/PFL-DocVQA and 200M-1.3B parameter public checkpoints. This scaled reproduction uses a procedural DocVQA-style dataset and a compact target model to fit the benchmark budget while preserving the DocMIA computation.
- Job `b610cdf5-51d` failed before training with `ModuleNotFoundError: No module named 'data'` because direct script execution placed `method/` on `sys.path`. The scripts now export the workspace root as `PYTHONPATH`.
- A local import smoke test cannot run because the login environment lacks `torch`; dependencies are specified in `environment/setup.sh` and the previous setup log shows the compute overlay installed them.

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
