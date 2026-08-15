# Sci-VoI-RHI Regime-Held-Out Results

> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.

- Data: `8000` sanitized historical proxy records.
- Outer folds: `28` complete scientific regimes.
- Selection never reads held-out-regime test records.
- Direct LLM-as-judge is intentionally excluded.

## Aggregate Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.4051 ± 0.2463 | 0.3786 | 0.1058 | 0.6883 | 0.4051 | 28 |
| `h0_reliability` | 0.7648 ± 0.2632 | 0.7364 | 0.1136 | 0.7543 | 0.7648 | 28 |
| `scivoi_rhi` | 0.8048 ± 0.2225 | 0.7699 | 0.1397 | 0.7811 | 0.8048 | 28 |
| `static_voi` | 0.7951 ± 0.2269 | 0.7619 | 0.1330 | 0.7772 | 0.7951 | 28 |
| `verbal_confidence` | 0.1176 ± 0.2678 | 0.0139 | 0.4146 | 0.5986 | 0.1176 | 28 |

## Paired Comparisons

| Variant | vs | Utility Δ | Risk-adj Δ | Risk Δ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `scivoi_rhi` | `static_voi` | 0.0097 | 0.0081 | 0.0067 | [-0.0069, 0.0315] | 0.143 | 0.3877 |
| `static_voi` | `h0_reliability` | 0.0303 | 0.0254 | 0.0194 | [-0.0286, 0.1213] | 0.500 | 0.0636 |

## Related-Work Baseline Families

| Family | Implemented methods | Notes |
| --- | --- | --- |
| Random/cost heuristics | `random_policy`, `cost_only` | Sanity checks for fixed-budget discovery and cheap-action selection. |
| Confidence and evidence judges | `verbal_confidence`, `evidence_heuristic`, `cost_aware_confidence` | Offline analogues of confidence/rationale judges without LLM calls. |
| Agreement/entropy uncertainty | `tool_agreement`, `self_consistency_proxy`, `semantic_entropy_proxy` | Deterministic proxies for self-consistency and semantic-entropy uncertainty. |
| Selective prediction / ensembles | `h0_reliability`, `static_full_reliability`, `ensemble_reliability`, `ensemble_lcb` | Risk-calibrated reliability gates and ensemble lower-confidence bounds. |
| Acquisition-style utility | `utility_ucb`, `utility_lcb`, `uncertainty_sampling`, `static_utility`, `static_voi` | Active-learning/BO-style utility and uncertainty scoring. |
| Harness self-improvement | `original_rhi`, `scivoi_rhi`, `scivoi_policy_*` | Recursive harness mutation baselines and acceptance-policy ablations. |

## Task Slices

| Task | Method | Net utility | Risk |
| --- | --- | ---: | ---: |
| `matbench_pairwise` | `evidence_heuristic` | 0.4051 | 0.1058 |
| `matbench_pairwise` | `h0_reliability` | 0.7648 | 0.1136 |
| `matbench_pairwise` | `scivoi_rhi` | 0.8048 | 0.1397 |
| `matbench_pairwise` | `static_voi` | 0.7951 | 0.1330 |
| `matbench_pairwise` | `verbal_confidence` | 0.1176 | 0.4146 |

## Interpretation

- This is a one-seed task-level smoke check proving that the new real-material A/B records can pass through the VoI evaluation stack.
- `scivoi_rhi` is higher than `static_voi` by `0.0097` net utility in this smoke, but the paired sign-test p-value is `0.3877`; this is not a statistically supported final method claim.
- The smoke does not replace the full five-seed, 45-regime main-materials rerun with all planned baselines and recursive checkpoints.
- Historical records are offline reconstructions rather than online MatBot trajectories.
