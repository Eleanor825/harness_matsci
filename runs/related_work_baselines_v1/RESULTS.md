# Related-Work Baseline Sweep

> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.

- Data: `15717` sanitized historical proxy records.
- Outer folds: `21` complete scientific regimes.
- Selection never reads held-out-regime test records.
- Direct LLM-as-judge is intentionally excluded.

## Aggregate Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cost_aware_confidence` | 0.5750 ± 0.2058 | 0.4203 | 0.6188 | 0.2501 | 0.5750 | 105 |
| `cost_only` | 0.7394 ± 0.3223 | 0.5407 | 0.7944 | 0.3375 | 0.7394 | 105 |
| `ensemble_lcb` | 0.5917 ± 0.2091 | 0.4633 | 0.5137 | 0.2533 | 0.5917 | 105 |
| `ensemble_reliability` | 0.5922 ± 0.2042 | 0.4589 | 0.5332 | 0.2507 | 0.5922 | 105 |
| `evidence_heuristic` | 0.5725 ± 0.2255 | 0.4247 | 0.5914 | 0.2649 | 0.5725 | 105 |
| `h0_reliability` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 0.5990 | 105 |
| `random_policy` | 0.5903 ± 0.2557 | 0.3943 | 0.7841 | 0.2021 | 0.5903 | 105 |
| `self_consistency_proxy` | 0.7069 ± 0.3775 | 0.5081 | 0.7953 | 0.3682 | 0.7069 | 105 |
| `semantic_entropy_proxy` | 0.5259 ± 0.2856 | 0.3698 | 0.6241 | 0.2384 | 0.5259 | 105 |
| `static_full_reliability` | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 0.2512 | 0.5635 | 105 |
| `static_utility` | 0.6398 ± 0.2091 | 0.5437 | 0.3842 | 0.2411 | 0.6398 | 105 |
| `static_voi` | 0.6372 ± 0.2105 | 0.5410 | 0.3845 | 0.2398 | 0.6372 | 105 |
| `tool_agreement` | 0.7070 ± 0.3774 | 0.5083 | 0.7949 | 0.3688 | 0.7070 | 105 |
| `uncertainty_sampling` | 0.7412 ± 0.2910 | 0.5421 | 0.7962 | 0.3335 | 0.7412 | 105 |
| `utility_lcb` | 0.6122 ± 0.2138 | 0.5203 | 0.3673 | 0.2403 | 0.6122 | 105 |
| `utility_ucb` | 0.6165 ± 0.2048 | 0.5150 | 0.4060 | 0.2445 | 0.6165 | 105 |
| `verbal_confidence` | 0.7577 ± 0.2888 | 0.5605 | 0.7889 | 0.3767 | 0.7577 | 105 |

## Paired Comparisons

| Variant | vs | Utility Δ | Risk-adj Δ | Risk Δ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `static_voi` | `h0_reliability` | 0.0382 | 0.0701 | -0.1278 | [0.0009, 0.0734] | 0.676 | 0.0004 |

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
| `discover_unique` | `cost_aware_confidence` | 0.7279 | 0.7837 |
| `discover_unique` | `cost_only` | 0.7044 | 0.8958 |
| `discover_unique` | `ensemble_lcb` | 0.7475 | 0.7991 |
| `discover_unique` | `ensemble_reliability` | 0.7476 | 0.8017 |
| `discover_unique` | `evidence_heuristic` | 0.7279 | 0.7837 |
| `discover_unique` | `h0_reliability` | 0.7383 | 0.7837 |
| `discover_unique` | `random_policy` | 0.7248 | 0.8940 |
| `discover_unique` | `self_consistency_proxy` | 0.7044 | 0.8958 |
| `discover_unique` | `semantic_entropy_proxy` | 0.7279 | 0.7837 |
| `discover_unique` | `static_full_reliability` | 0.7400 | 0.8002 |
| `discover_unique` | `static_utility` | 0.7649 | 0.7837 |
| `discover_unique` | `static_voi` | 0.7649 | 0.7837 |
| `discover_unique` | `tool_agreement` | 0.7044 | 0.8958 |
| `discover_unique` | `uncertainty_sampling` | 0.7044 | 0.8958 |
| `discover_unique` | `utility_lcb` | 0.7572 | 0.4179 |
| `discover_unique` | `utility_ucb` | 0.7519 | 0.4294 |
| `discover_unique` | `verbal_confidence` | 0.7044 | 0.8958 |
| `extreme_properties` | `cost_aware_confidence` | 0.6032 | 0.5671 |
| `extreme_properties` | `cost_only` | 1.0000 | 0.8435 |
| `extreme_properties` | `ensemble_lcb` | 0.6156 | 0.3917 |
| `extreme_properties` | `ensemble_reliability` | 0.6149 | 0.3933 |
| `extreme_properties` | `evidence_heuristic` | 0.6032 | 0.5671 |
| `extreme_properties` | `h0_reliability` | 0.6678 | 0.3732 |
| `extreme_properties` | `random_policy` | 0.6933 | 0.8338 |
| `extreme_properties` | `self_consistency_proxy` | 1.0000 | 0.8435 |
| `extreme_properties` | `semantic_entropy_proxy` | 0.6032 | 0.5671 |
| `extreme_properties` | `static_full_reliability` | 0.5622 | 0.3944 |
| `extreme_properties` | `static_utility` | 0.6865 | 0.1532 |
| `extreme_properties` | `static_voi` | 0.6835 | 0.1532 |
| `extreme_properties` | `tool_agreement` | 1.0000 | 0.8435 |
| `extreme_properties` | `uncertainty_sampling` | 0.9709 | 0.8435 |
| `extreme_properties` | `utility_lcb` | 0.6280 | 0.2800 |
| `extreme_properties` | `utility_ucb` | 0.6409 | 0.3522 |
| `extreme_properties` | `verbal_confidence` | 1.0000 | 0.8435 |
| `preferential_bo` | `cost_aware_confidence` | 0.2368 | 0.4597 |
| `preferential_bo` | `cost_only` | 0.1489 | 0.4944 |
| `preferential_bo` | `ensemble_lcb` | 0.2595 | 0.3193 |
| `preferential_bo` | `ensemble_reliability` | 0.2635 | 0.4132 |
| `preferential_bo` | `evidence_heuristic` | 0.2239 | 0.3153 |
| `preferential_bo` | `h0_reliability` | 0.1831 | 0.3850 |
| `preferential_bo` | `random_policy` | 0.0975 | 0.4672 |
| `preferential_bo` | `self_consistency_proxy` | -0.0213 | 0.4989 |
| `preferential_bo` | `semantic_entropy_proxy` | -0.0210 | 0.4874 |
| `preferential_bo` | `static_full_reliability` | 0.2577 | 0.4021 |
| `preferential_bo` | `static_utility` | 0.3040 | 0.2627 |
| `preferential_bo` | `static_voi` | 0.2976 | 0.2641 |
| `preferential_bo` | `tool_agreement` | -0.0211 | 0.4966 |
| `preferential_bo` | `uncertainty_sampling` | 0.2311 | 0.5037 |
| `preferential_bo` | `utility_lcb` | 0.3186 | 0.4969 |
| `preferential_bo` | `utility_ucb` | 0.3188 | 0.4994 |
| `preferential_bo` | `verbal_confidence` | 0.2451 | 0.4652 |

## Interpretation

- This run is a related-work baseline sweep only; it intentionally does not rerun recursive Sci-VoI-RHI mutations.
- High raw utility from confidence, agreement, or uncertainty-sampling proxies often comes with high selective risk, so risk-adjusted utility is the safer paper metric.
- Compare these rows with `runs/scivoi_rhi_v1/README.md` for the preserved Sci-VoI-RHI result under the same held-out-regime protocol.
- Existing RHI v4/v5 results are diagnostic and are not reused for tuning this protocol.
- Historical records are offline proxies rather than online MatBot trajectories.
