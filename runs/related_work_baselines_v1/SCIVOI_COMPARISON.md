# Related-Work Baseline Comparison

This table merges the preserved Sci-VoI-RHI run with the new non-LLM related-work baseline sweep. Both use the same 15,717-record, 21-regime, five-seed held-out protocol. Direct LLM judge baselines are not included here.

## Aggregate Table

| Family | Method | Net utility | Risk-adjusted | Risk | Hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Ours | `scivoi_policy_always_accept` | 0.6443 ± 0.2052 | 0.6043 | 0.1600 | 0.2452 |
| Ours | `scivoi_policy_mean_guarded` | 0.6384 ± 0.2173 | 0.5980 | 0.1616 | 0.2458 |
| Ours | `scivoi_rhi` | 0.6137 ± 0.2201 | 0.5547 | 0.2361 | 0.2328 |
| RHI / reliability | `original_rhi` | 0.5635 ± 0.1916 | 0.4307 | 0.5314 | 0.2302 |
| RHI / reliability | `h0_reliability` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 |
| RHI / reliability | `static_full_reliability` | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 0.2512 |
| RHI / reliability | `ensemble_reliability` | 0.5922 ± 0.2042 | 0.4589 | 0.5332 | 0.2507 |
| RHI / reliability | `ensemble_lcb` | 0.5917 ± 0.2091 | 0.4633 | 0.5137 | 0.2533 |
| Judge heuristics | `verbal_confidence` | 0.7577 ± 0.2888 | 0.5605 | 0.7889 | 0.3767 |
| Judge heuristics | `evidence_heuristic` | 0.5725 ± 0.2255 | 0.4247 | 0.5914 | 0.2649 |
| Judge heuristics | `cost_aware_confidence` | 0.5750 ± 0.2058 | 0.4203 | 0.6188 | 0.2501 |
| Agreement / entropy | `tool_agreement` | 0.7070 ± 0.3774 | 0.5083 | 0.7949 | 0.3688 |
| Agreement / entropy | `self_consistency_proxy` | 0.7069 ± 0.3775 | 0.5081 | 0.7953 | 0.3682 |
| Agreement / entropy | `semantic_entropy_proxy` | 0.5259 ± 0.2856 | 0.3698 | 0.6241 | 0.2384 |
| Acquisition / utility | `static_utility` | 0.6398 ± 0.2091 | 0.5437 | 0.3842 | 0.2411 |
| Acquisition / utility | `static_voi` | 0.6372 ± 0.2105 | 0.5410 | 0.3845 | 0.2398 |
| Acquisition / utility | `utility_ucb` | 0.6165 ± 0.2048 | 0.5150 | 0.4060 | 0.2445 |
| Acquisition / utility | `utility_lcb` | 0.6122 ± 0.2138 | 0.5203 | 0.3673 | 0.2403 |
| Acquisition / utility | `uncertainty_sampling` | 0.7412 ± 0.2910 | 0.5421 | 0.7962 | 0.3335 |
| Sanity checks | `random_policy` | 0.5903 ± 0.2557 | 0.3943 | 0.7841 | 0.2021 |
| Sanity checks | `cost_only` | 0.7394 ± 0.3223 | 0.5407 | 0.7944 | 0.3375 |

## Paired Comparison to Sci-VoI-RHI

Positive utility/risk-adjusted deltas favor `scivoi_policy_always_accept`; negative risk deltas mean Sci-VoI executes fewer wrong actions.

| Baseline | Utility Δ | Risk-adj Δ | Risk Δ | Win rate | Folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `verbal_confidence` | -0.1134 | 0.0439 | -0.6288 | 0.371 | 105 |
| `uncertainty_sampling` | -0.0968 | 0.0622 | -0.6362 | 0.410 | 105 |
| `static_utility` | 0.0045 | 0.0606 | -0.2242 | 0.400 | 105 |
| `static_voi` | 0.0072 | 0.0633 | -0.2245 | 0.419 | 105 |
| `cost_only` | -0.0950 | 0.0636 | -0.6344 | 0.419 | 105 |
| `tool_agreement` | -0.0626 | 0.0961 | -0.6348 | 0.419 | 105 |
| `self_consistency_proxy` | -0.0626 | 0.0962 | -0.6353 | 0.419 | 105 |
| `utility_lcb` | 0.0322 | 0.0840 | -0.2072 | 0.648 | 105 |
| `utility_ucb` | 0.0278 | 0.0893 | -0.2459 | 0.638 | 105 |
| `ensemble_lcb` | 0.0526 | 0.1410 | -0.3537 | 0.790 | 105 |
| `h0_reliability` | 0.0454 | 0.1334 | -0.3522 | 0.676 | 105 |
| `static_full_reliability` | 0.0809 | 0.1736 | -0.3711 | 0.829 | 105 |
| `original_rhi` | 0.0808 | 0.1736 | -0.3713 | 0.829 | 105 |
| `random_policy` | 0.0540 | 0.2100 | -0.6240 | 0.657 | 105 |

## Takeaways

- The strongest raw-utility judge heuristics (`verbal_confidence`, `tool_agreement`, `uncertainty_sampling`, `cost_only`) are aggressive and high-risk: their risk is around `0.79`.
- Static utility/VoI heads are much safer than reliability-only gates, confirming that continuous scientific utility is an important objective.
- `scivoi_policy_always_accept` has the best risk-adjusted utility among non-LLM methods in this combined table: `0.6043`, with much lower risk `0.1600`.
- The remaining mandatory paper baseline is a true direct LLM/agentic judge run; the repository keeps its protocol in `runs/direct_judge_baseline_v1/README.md`.
