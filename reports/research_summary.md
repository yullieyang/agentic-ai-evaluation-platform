# Research Summary

All results below are generated from reproducible runs of the offline **mock provider** and are simulation, not measurements of any real model. They are reported to exercise the evaluation methodology end to end.

## Experimental setting

- Dataset version: `synthetic-v1`; cases: 600; conditions: 22.
- Mock mode: `True`; seed: 13.
- Ground truth excluded from agent prompts: `True`.

## RQ1 — Deterministic QA evidence

Mean metrics by deterministic-evidence setting (mock_main):

| include_deterministic_evidence   |   pre_precision |   pre_recall |   pre_f1 |   pre_unsupported_claim_rate |
|:---------------------------------|----------------:|-------------:|---------:|-----------------------------:|
| False                            |           0.557 |        0.543 |    0.55  |                        0.099 |
| True                             |           0.591 |        0.521 |    0.554 |                        0.091 |

Paired McNemar contrast of decision correctness (with vs without deterministic evidence), n_pairs=6600: accuracy 0.559 vs 0.535, p=<0.001 (method: chi2_continuity), BH-adjusted p=<0.001.
Within this synthetic setting, supplying deterministic evidence is associated with higher precision and a lower unsupported-claim rate; the effect on overall decision correctness is small. This is an observational association under the mock generative process, not a causal claim about any real model.

## RQ2 — Prompt sensitivity

Mean metrics by prompt variant (mock_main):

| prompt_version                |   pre_precision |   pre_recall |   pre_f1 |   pre_false_positive_rate |   pre_unsupported_claim_rate |   pre_abstention_rate |   pre_expected_calibration_error |
|:------------------------------|----------------:|-------------:|---------:|--------------------------:|-----------------------------:|----------------------:|---------------------------------:|
| prompt_a_zero_shot            |           0.566 |        0.543 |    0.554 |                     0.464 |                        0.208 |                 0.032 |                            0.158 |
| prompt_b_few_shot             |           0.579 |        0.543 |    0.56  |                     0.441 |                        0.088 |                 0.06  |                            0.149 |
| prompt_c_evidence_constrained |           0.582 |        0.532 |    0.555 |                     0.427 |                        0.047 |                 0.059 |                            0.133 |
| prompt_d_conservative         |           0.57  |        0.512 |    0.539 |                     0.431 |                        0.037 |                 0.082 |                            0.147 |

Paired contrast of the unsupported-claim rate, zero-shot vs evidence-constrained, n_pairs=4200: 0.209 vs 0.045 (difference 0.164, Cohen's h=0.523, permutation p=<0.001, BH-adjusted p=<0.001).
Results suggest the evidence-constrained and conservative prompts reduce the unsupported-claim rate and false-positive rate relative to the zero-shot prompt.

## RQ3 — Reviewer agent

Pre/post reviewer (paired by case, n=6000): accuracy 0.549 -> 0.557; false-positive rate 0.202 -> 0.192; McNemar p=<0.001 (BH-adjusted p=<0.001).
The reviewer flags unsupported claims and downgrades some false positives. Because the reviewer does not rewrite the first agent's explanation text, the per-finding unsupported-claim rate is unchanged; the reviewer's contribution is captured by its flag counts and by the false-positive-rate change.

## RQ4 — Calibration

Expected calibration error (ECE) is reported per condition in `outputs/experiment_results.csv` and per bin in `outputs/calibration_results.csv`. Confidence is model-reported and treated cautiously; calibration measures whether a stated confidence corresponds to empirical decision accuracy.

## RQ5–RQ8 and ablations

Per-scenario and per-difficulty breakdowns are in `outputs/aggregate_metrics.json` (`by_scenario`, `by_difficulty`). The `mock_ablation` grid contains the incomplete-evidence and adversarial-scenario conditions used for RQ5 and RQ8; the incomplete-evidence condition raises the abstention rate and lowers recall, consistent with the intended abstention behaviour.

## RQ9 — Single-shot vs. agentic (tool-using) architecture

Paired by case (n=600), prompt_c_evidence_constrained, reviewer on: decision accuracy 0.557 (single-shot) vs 0.572 (agentic), McNemar p=0.503 (BH-adjusted p=0.503).

Average tool calls: 0.00 (single-shot) vs 3.35 (agentic). Average latency: 0.3010s vs 0.3661s. Evidence-citation accuracy: 1.000 vs 1.000.

**Interpretation:** in mock mode, both architectures deliberately share the same underlying decision policy — this isolates the evidence-delivery mechanism (pre-supplied vs. tool-retrieved) as the controlled variable, so any accuracy difference here reflects seeding noise, not a real capability gap. This comparison demonstrates that the agentic tool-call loop, evidence citation, and per-case tool-call accounting all work correctly end to end. It does **not** demonstrate whether a live agentic Claude run would find different or better findings than single-shot — only a live run against the real API could show that.

## Limitations

- All findings are simulation under a documented mock generative process and do not estimate any real model's behaviour.
- The mock's prompt sensitivity is parameterised, so prompt-variant differences reflect those parameters rather than emergent model behaviour.
- The deterministic baseline, peer-based cross features, and unsupported-claim detector are approximations with documented limits.
