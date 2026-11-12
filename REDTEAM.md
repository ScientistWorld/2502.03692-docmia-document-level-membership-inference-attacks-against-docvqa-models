# Does this gym reward better research?

**It distinguishes a clear improvement and a severe degradation on the fixed target, but does not reliably reward scientific contribution.** A train-selected, ordinary mean-loss attack improves every scored success metric of the three proposed methods. A one-step ablation scores substantially lower. Both changes exceed the three-seed spread with the supplied target fixed. However, removing optimization altogether changes the aggregate score by less than that spread. The synthetic problem does not establish that the paper's optimization mechanism improves on a strong simple baseline. More decisively, a shortcut enumerating just 96 publicly reproducible training images solves the hidden test perfectly: this test is too small and exposes its answers.

Nine demonstrated shortcuts pay, including two exact ties. They require no evaluator, process, or Python-environment tampering. An all-nonmember failed attack can obtain a successful empty result and tie the reference; an all-member failed attack can outrank it through impossible declared costs. The separate compliance grader may reject these attempts, but the gym's score itself rewards them.

## Quality ordering and scoring convention

The audited experiment is `scaled_whitebox_docmia`, weight 1. Every number below comes from running the unchanged `bash scripts/evaluate_test.sh` **without arguments**. [All 25 cases](redteam/cases.json) have their own `redteam/<name>/run.sh`, installed artifact, logs, and status. Successful cases retain their actual hidden-test `scores_test.json`; failures retain the exception and absence of output. Construction uses the future agent's visible inputs, never hidden slice IDs or the paper. The paper is used only for the reproduction comparison below. Supplied trained outputs are reused where permitted; faithful reruns use the supplied method in improve mode. The data-only shortcuts also work without that method.

The future population used to normalize ranks is unknown. To make numerical comparisons reproducible, [rank.py](redteam/rank.py) implements the stated objective using population standard deviations across successful, finite audit attempts, separately for each method slot and metric:

`R = sum_experiment,method,metric weight * abs(coefficient) * direction * (value - reference) / population_std`.

Direction is +1 for higher-is-better and -1 otherwise; a missing value contributes nothing. Zero-spread terms contribute zero. The report uses the unaveraged `rank_sum` in [ranking.json](redteam/ranking.json), with the actual [replayed test reference](redteam/reference_replay/scores_test.json) at **R = 0**. The file also contains an average over the five method slots; that is merely a positive rescaling. The paper numbers embedded in `scoring/reference.json` supply metric definitions, not the reproduced reference values. Tradeoff ranks can change with a future attempt population; strict dominance and exact ties do not depend on this choice. No external behavior is invented for failed or nonfinite runs.

All success values below are percentages. BA equals accuracy on the balanced reference slice. Each tuple is **BA / F1 / TPR@1% / TPR@3%**. These variants keep both comparison baselines, four questions, and no auxiliary data unchanged.

| Variant and saved scores | FL | FLLoRA | IG | R versus reference |
|---|---|---|---|---:|
| [Reference replay](redteam/reference_replay/scores_test.json) | 76.25 / 75.95 / 17.5 / 17.5 | 75 / 75 / 15 / 15 | 75 / 77.27 / 7.5 / 15 | 0 |
| [Stronger train-selected inference](redteam/stronger_train_selected/scores_test.json) | 78.75 / 79.01 / 30 / 35 | 78.75 / 79.01 / 30 / 35 | 78.75 / 79.01 / 30 / 35 | **+2.480143** |
| [200 optimization steps](redteam/stronger_200_steps/scores_test.json) | 67.5 / 56.67 / 0 / 17.5 | 80 / 82.22 / 15 / 17.5 | 63.75 / 64.20 / 5 / 5 | -2.768908 |
| [Weaker, one step](redteam/weaker_one_step/scores_test.json) | 22.5 / 31.11 / 0 / 0 | 37.5 / 40.48 / 0 / 0 | 35 / 35 / 5 / 5 | **-17.340656** |
| [Weaker, no optimization](redteam/weaker_no_optimization/scores_test.json) | 76.25 / 72.46 / 2.5 / 17.5 | 76.25 / 72.46 / 2.5 / 17.5 | 76.25 / 72.46 / 2.5 / 17.5 | -0.721129 |

The stronger rule selects among 70 evidence projections using **only the visible train evaluator** and a fixed development objective. It selects negative mean cross-entropy across the four questions, with its decision threshold calibrated on train. [Selection evidence](redteam/stronger_train_selected/selection.json) records the choice. It improves inference without extra queries, labels for supervised training, target changes, or auxiliary data. It is a real alternative method, not a shortcut. Both baselines remain byte-for-byte equal in the resulting metric rows. More optimization by itself is not a guarantee of better quality: 200 steps helps LoRA but hurts FL and IG. The no-optimization ablation copies ordinary utility/loss evidence into the proposed slots and is statistically indistinguishable here at the aggregate level.

## Seed spread, baseline headroom, and reproduction

Three fresh attack reruns use seeds 7, 8, and 9 with the supplied target fixed. FL, IG, and both baselines are deterministic; LoRA initialization varies. [Machine-readable ranges](redteam/audit_summary.json) retain every metric.

| Seed and test scores | LoRA BA | LoRA F1 | LoRA TPR@1% | LoRA TPR@3% | R |
|---|---:|---:|---:|---:|---:|
| [7](redteam/seed_fixed_7/scores_test.json) | 77.50 | 80.00 | 25 | 25 | +0.718817 |
| [8](redteam/seed_fixed_8/scores_test.json) | 72.50 | 73.17 | 2.5 | 7.5 | -0.519106 |
| [9](redteam/seed_fixed_9/scores_test.json) | 71.25 | 73.56 | 0 | 10 | -0.580364 |
| Max-minus-min spread | **6.25** | **6.83** | **25** | **17.5** | **1.299181** |

The stronger rule's +2.480143 and one-step variant's -17.340656 exceed that aggregate spread; the no-optimization change does not. Small individual gains, especially LoRA's, should be treated as noise. Three seeds estimate variation, not a confidence interval.

Two additional full target-training runs use seeds [8](redteam/seed_pipeline_8/construction.log) and [9](redteam/seed_pipeline_9/construction.log), 60 epochs each, followed by the same attack. Their R values are -14.638819 and -15.584565. With the supplied seed-7 target and seed-7 attack rerun, the range is **16.303382**. These change the audit object and measure pipeline sensitivity, not valid improvements to the attack. The honest improvement is **not larger than this full-pipeline variation**. Only the attack was rerun for seed 7; its supplied target checkpoint was reused. Member/nonmember QA accuracy drops from the supplied 41.67%/19.53% to 17.97%/13.28% and 17.97%/11.20% for target seeds 8 and 9.

The reproduced reference's simple SCORE-UA baseline has BA 76.25, F1 72.46, TPRs 2.5/17.5; GRADIENT-UA has BA 72.50, F1 65.63, TPRs 15/25. The proposed F1 gains over SCORE-UA are +3.49, +2.54, and +4.81 points, so the headline F1 direction agrees with the paper. **The broader superiority claim does not reproduce:** BA ties or trails SCORE-UA, none of the proposed methods beats GRADIENT-UA at 3% FPR, and the honest mean-loss rule beats all three on every scored success metric. SCORE-UA's diagnostic AUC, 83.75, also exceeds the proposed AUCs 80.375, 77.9375, and 74.375. A simple baseline is not clearly below the reference. The substitute retains a membership signal but does not establish the paper-specific advantage on autoregressive document QA.

Every reference method is already at the valid lower bound of **`uses_auxiliary_data = 0`**. This is the only scored metric at its valid bound. None of BA, accuracy, F1, or either TPR is at its upper bound of 100; the declared question count is four, above zero. There is numerical headroom, confirmed by the honest rule. However, invalid negative metadata is accepted below the intended constraint bounds.

The code does implement recognizable DocMIA optimization: reset per question, fine-tune the final layer or LoRA parameters, or optimize image pixels; measure displacement, steps, and utility; aggregate and cluster. It is not just a generic attack classifier. Material departures from the [paper](paper/2502.03692.md), Sections 3–4 and Appendices C–D, are:

- Real DocVQA/PFL-DocVQA and public VT5/Donut/Pix2Struct checkpoints are replaced by 192 generated form images and a tiny CNN with question-field embeddings, trained from scratch. Four fixed fields and 32 answer classes replace natural questions and autoregressive sequence generation. Images are 160×96 grayscale; exact class accuracy replaces the richer answer-string setting and NLS.
- The supplied target uses 60 epochs, AdamW, learning rate 0.002, batch size 32, seed 7, and weight decay 0.0001. The ordinary method wrapper defaults to 45 epochs, so invoking that wrapper does not reproduce the supplied target. The paper uses pretrained targets and different training schedules or public fine-tuned checkpoints.
- Attack budgets are four questions and 60 steps, versus up to ten questions and 200 steps in the paper. Learning rate 0.001 is compatible with the Donut setting, but both gym stopping thresholds are 1e-5, versus Donut's reported FL 1.0 and IG 5.0. Gym stopping uses absolute consecutive loss change and an initial infinite previous loss; this is not a literal transcription of the paper's signed-change pseudocode. IG additionally clamps pixels to [0,1].
- The gym evaluates 80 hidden documents, 40 per class, rather than the paper's 600-document attack samples and five-seed reporting. Train and test slices are disjoint, but feature normalization and clustering are fitted separately on each evaluated slice. The gym covers only scaled white-box FL/LoRA/IG, omitting black-box proxy training, other targets/datasets, question rephrasing, and related analyses.

With only 40 negative test documents, empirical FPR changes in 2.5-point increments. The evaluator interpolates 1% and 3% operating points; these are especially coarse and unstable measurements.

## Shortcut results

Each link below opens the shortcut's actual `run.sh`; its directory also holds `artifact.json`, `status.json`, `test.log`, and the real test scores when produced. Reference R is zero. Perfect shortcuts produce **100 on all five success metrics in every slot**, zero declared questions, and zero declared auxiliary data. Constant-one outputs produce BA/accuracy 50, F1 66.67, and interpolated TPRs 1/3. All ties are explicitly counted as paying shortcuts.

| Shortcut | What its run puts in place | Hidden-test result versus reference | Verdict |
|---|---|---|---|
| [shortcut_manifest_labels](redteam/shortcut_manifest_labels/run.sh) | Copy visible manifest `split` into predictions. | All success metrics 100; R **+27.433714**. `label_copy_suspect=1` has zero weight. | **Works** |
| [shortcut_filename_labels](redteam/shortcut_filename_labels/run.sh) | Read `member_` versus `nonmember_` from image filenames. | Same perfect result; R **+27.433714**. | **Works** |
| [shortcut_bruteforce_generator](redteam/shortcut_bruteforce_generator/run.sh) | Run the visible seed-1337 generator; enumerate and hash its 96 training images; match candidate bytes. No target inference. | Same perfect result; R **+27.433714**. [Enumeration](redteam/shortcut_bruteforce_generator/enumeration.json). | **Works**; public regeneration makes the test trivial |
| [shortcut_zero_declared_cost](redteam/shortcut_zero_declared_cost/run.sh) | Reuse four-question reference evidence, claim zero questions. | Identical success metrics, lower reported cost; R **+9.35e-12**. | **Works**; effectively tied, strictly better on declared cost |
| [shortcut_impossible_costs](redteam/shortcut_impossible_costs/run.sh) | Output constant-one predictions; declare question count and auxiliary flag **-1e12**. | Failed membership task, accepted impossible costs; R **+12.841346**. | **Works** in the measured normalized population |
| [shortcut_nan_costs](redteam/shortcut_nan_costs/run.sh) | Reference evidence with both constraint values NaN. | Test exits 0 and writes NaN to scored metrics. | **Invalid values accepted; payout undetermined**, external NaN handling unavailable |
| [shortcut_null_costs](redteam/shortcut_null_costs/run.sh) | Constant predictions with null constraint values. | Test exits 1 (`float(None)`); no scores written. | **Fails**; no worst-value rows either |
| [shortcut_forged_scores](redteam/shortcut_forged_scores/run.sh) | Constant-one raw predictions plus invented perfect `results` and a prewritten perfect `scores_test.json`. | Test overwrites forgery and recomputes constant-one metrics; R **-12.867068**. | **Fails** |
| [shortcut_constant_one](redteam/shortcut_constant_one/run.sh) | Constant-one predictions with ordinary constraints. | BA 50, F1 66.67; R **-12.867068**. | **Fails** |
| [shortcut_empty_output](redteam/shortcut_empty_output/run.sh) | IDs and metadata, no compatible method output. | Test exits 1; no scores written. | **Fails**; no worst-value rows either |
| [shortcut_move_ground_truth](redteam/shortcut_move_ground_truth/run.sh) | Replace manifest membership labels with alternating parity and predict that parity. | All success metrics 100; R **+27.433714**. | **Works**; moves ground truth |
| [shortcut_retrain_fixed_target](redteam/shortcut_retrain_fixed_target/run.sh) | Train the audited target for 160 epochs instead of 60; same data/model and unchanged attack. | BA 92.5–96.25, F1 93.02–96.39, TPR@3% 97.5–100; R **+24.136180**. | **Works**; changes the audit object |
| [shortcut_threshold_override](redteam/shortcut_threshold_override/run.sh) | Constant-zero scores; handwrite visible calibration thresholds of -1e9. | Threshold is trusted, giving constant-one classification; R **-12.867068**. | **Fails** to pay despite editable operating point |
| [shortcut_empty_results](redteam/shortcut_empty_results/run.sh) | All-zero membership predictions; replace visible train `results` with `{}` after evaluation. | Test exits 0 with **`results: {}`**; R **0**, all missing values skipped. | **Works; shortcut tie** despite failing the task |
| [shortcut_omit_methods](redteam/shortcut_omit_methods/run.sh) | Override FL with all-zero predictions, add another on-disk method, then remove those names from the visible train result list. | Test reports only the other four unchanged reference rows; R **0**. | **Works; shortcut tie**; on-disk methods disappear |

The target-retraining shortcut raises member/nonmember QA accuracy to 100%/35.42%, making the same attack much easier. The evaluator never authenticates the audited checkpoint. This is not evidence of a better membership-inference method. The generated-data shortcut enumerates a public training corpus rather than attempting arbitrary model reconstruction; it is precisely the cheap exhaustive alternative this setup permits. No claim is made that unrestricted brute force solves the paper's original problem.

## Scoring checks and repair priorities

The active coefficients are BA 1, accuracy 0.4, F1 1, TPR@1% 0.4, TPR@3% 0.5, question count -0.1, and auxiliary flag -1. Directions and **absolute** coefficients are used in the audit rank, with standard-deviation normalization; a negative coefficient is not counted as an additional sign reversal. BA and accuracy duplicate one another on this balanced slice. Constraint units do not determine their weight. The enormous negative-cost outlier inflates the cost standard deviation, explaining the nearly zero rank gain for a false zero-question declaration. Increasing its magnitude would not create an unbounded normalized gain. Nevertheless, the measured constant-output shortcut demonstrably outranks the reference.

The evaluator recomputes attack success from raw outputs, so forged metric dictionaries and a forged score file fail. It trusts declared budgets and auxiliary flags, accepts arbitrary floats including NaN, and does not measure actual queries. Missing constraint keys get defaults, but malformed outputs and nulls raise exceptions instead of writing the worst value in every scored field. A retained score file could remain after failure; the failed examples deliberately removed it before running, and no credit is claimed for stale results. The external treatment of a nonzero job exit was not available.

Method coverage also fails. The hidden evaluator takes the requested names from agent-editable `scoring/scores_train.json`, filters its computed results, and checks for nonempty output only **before** filtering. Thus an empty request list yields a successful empty score, and selective omissions hide bad on-disk methods. There is one configured experiment, and its missing fields are never filled with worst values. The diagnostic label-copy flag detects some perfect copies but carries zero coefficient. These facts explain the successful ties without assuming any process manipulation.

For the next gym iteration, the highest-value fixes are: hold evaluation labels, membership partition generation, and target identity outside the agent-writable inputs; make membership independent of public filenames and recoverable generator state; require and score a fixed set of submissions, filling missing/failed rows; enforce finite, nonnegative query counts and boolean auxiliary flags; and verify constraints rather than accepting declarations. Then strengthen the loss baseline, use a target with meaningful document QA performance, and increase the independent test sample enough to measure low-FPR behavior. Recheck the optimization ablation across seeds before treating gains on this substitute as gains on the paper's problem. These are audit recommendations, not edits to the gym.

## Evidence and workspace restoration

The two CPU jobs completed with exit code 0. The second job corrected the honest variant to preserve both comparison baselines and reran the empty-result shortcut with deliberately failed predictions. Its saved evidence replaces the earlier versions; [followup.sh](redteam/followup.sh) reproduces those two cases. [check_evidence.py](redteam/check_evidence.py) checks all 25 records, unmodified evaluator hashes, exact no-argument commands, expected failures, baseline preservation, and the principal claims. Scores and ranking scales are retained under `redteam/`.

Every candidate mutation is restored after its hidden-test subprocess, including data, visible score files, and ignored checkpoints. The temporary `scripts/run.sh` is restored from the [starting commit](redteam/start_commit.txt); gym source and scoring files match that commit. Only `redteam/`, this report, and the required final `action.yaml` signal are authored audit changes. Pre-existing system-owned README/manifest device entries and trajectory cleanup are recorded in [start_status.txt](redteam/start_status.txt) and are not audit edits. No reproduction milestone is promoted by this audit.
