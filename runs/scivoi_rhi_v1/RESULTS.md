# Sci-VoI-RHI Regime-Held-Out Results

> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.

- Data: `15717` sanitized historical proxy records.
- Outer folds: `21` complete scientific regimes.
- Selection never reads held-out-regime test records.
- Direct LLM-as-judge is intentionally excluded.

## Aggregate Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.5725 ± 0.2255 | 0.4247 | 0.5914 | 0.2649 | 0.5725 | 105 |
| `h0_reliability` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 0.5990 | 105 |
| `original_rhi` | 0.5635 ± 0.1916 | 0.4307 | 0.5314 | 0.2302 | 0.5635 | 105 |
| `scivoi_component_features` | 0.6068 ± 0.2110 | 0.4735 | 0.5329 | 0.2331 | 0.6068 | 105 |
| `scivoi_component_routing` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 0.5990 | 105 |
| `scivoi_component_uncertainty` | 0.5890 ± 0.2238 | 0.4610 | 0.5123 | 0.2057 | 0.5890 | 105 |
| `scivoi_component_utility` | 0.6099 ± 0.2235 | 0.5017 | 0.4325 | 0.2256 | 0.6099 | 105 |
| `scivoi_policy_always_accept` | 0.6443 ± 0.2052 | 0.6043 | 0.1600 | 0.2452 | 0.6443 | 105 |
| `scivoi_policy_mean_guarded` | 0.6384 ± 0.2173 | 0.5980 | 0.1616 | 0.2458 | 0.6384 | 105 |
| `scivoi_rhi` | 0.6137 ± 0.2201 | 0.5547 | 0.2361 | 0.2328 | 0.6137 | 105 |
| `static_full_reliability` | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 0.2512 | 0.5635 | 105 |
| `static_utility` | 0.6398 ± 0.2091 | 0.5437 | 0.3842 | 0.2411 | 0.6398 | 105 |
| `static_voi` | 0.6372 ± 0.2105 | 0.5410 | 0.3845 | 0.2398 | 0.6372 | 105 |
| `verbal_confidence` | 0.7577 ± 0.2888 | 0.5605 | 0.7889 | 0.3767 | 0.7577 | 105 |

## Paired Comparisons

| Variant | vs | Utility Δ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | --- | ---: | ---: |
| `scivoi_policy_always_accept` | `original_rhi` | 0.0808 | [0.0549, 0.1059] | 0.829 | 0.0000 |
| `scivoi_policy_always_accept` | `static_full_reliability` | 0.0809 | [0.0584, 0.1025] | 0.829 | 0.0000 |
| `scivoi_policy_always_accept` | `static_voi` | 0.0072 | [0.0028, 0.0128] | 0.419 | 0.2543 |
| `scivoi_policy_always_accept` | `static_utility` | 0.0045 | [-0.0000, 0.0101] | 0.400 | 0.4944 |
| `scivoi_policy_mean_guarded` | `static_voi` | 0.0013 | [-0.0086, 0.0099] | 0.410 | 0.3620 |
| `scivoi_rhi` | `original_rhi` | 0.0502 | [0.0283, 0.0706] | 0.743 | 0.0000 |
| `scivoi_rhi` | `static_full_reliability` | 0.0502 | [0.0235, 0.0757] | 0.705 | 0.0000 |
| `scivoi_rhi` | `static_voi` | -0.0235 | [-0.0451, -0.0025] | 0.352 | 0.1654 |
| `static_voi` | `h0_reliability` | 0.0382 | [0.0009, 0.0734] | 0.676 | 0.0004 |

## Task Slices

| Task | Method | Net utility | Risk |
| --- | --- | ---: | ---: |
| `discover_unique` | `evidence_heuristic` | 0.7279 | 0.7837 |
| `discover_unique` | `h0_reliability` | 0.7383 | 0.7837 |
| `discover_unique` | `original_rhi` | 0.7266 | 0.7870 |
| `discover_unique` | `scivoi_component_features` | 0.7426 | 0.8145 |
| `discover_unique` | `scivoi_component_routing` | 0.7383 | 0.7837 |
| `discover_unique` | `scivoi_component_uncertainty` | 0.7324 | 0.7837 |
| `discover_unique` | `scivoi_component_utility` | 0.7535 | 0.7837 |
| `discover_unique` | `scivoi_policy_always_accept` | 0.7662 | 0.0000 |
| `discover_unique` | `scivoi_policy_mean_guarded` | 0.7662 | 0.0000 |
| `discover_unique` | `scivoi_rhi` | 0.7562 | 0.1541 |
| `discover_unique` | `static_full_reliability` | 0.7400 | 0.8002 |
| `discover_unique` | `static_utility` | 0.7649 | 0.7837 |
| `discover_unique` | `static_voi` | 0.7649 | 0.7837 |
| `discover_unique` | `verbal_confidence` | 0.7044 | 0.8958 |
| `extreme_properties` | `evidence_heuristic` | 0.6032 | 0.5671 |
| `extreme_properties` | `h0_reliability` | 0.6678 | 0.3732 |
| `extreme_properties` | `original_rhi` | 0.5788 | 0.4000 |
| `extreme_properties` | `scivoi_component_features` | 0.6613 | 0.3866 |
| `extreme_properties` | `scivoi_component_routing` | 0.6678 | 0.3732 |
| `extreme_properties` | `scivoi_component_uncertainty` | 0.6508 | 0.3732 |
| `extreme_properties` | `scivoi_component_utility` | 0.6602 | 0.2332 |
| `extreme_properties` | `scivoi_policy_always_accept` | 0.6895 | 0.1800 |
| `extreme_properties` | `scivoi_policy_mean_guarded` | 0.6895 | 0.1800 |
| `extreme_properties` | `scivoi_rhi` | 0.6617 | 0.2266 |
| `extreme_properties` | `static_full_reliability` | 0.5622 | 0.3944 |
| `extreme_properties` | `static_utility` | 0.6865 | 0.1532 |
| `extreme_properties` | `static_voi` | 0.6835 | 0.1532 |
| `extreme_properties` | `verbal_confidence` | 1.0000 | 0.8435 |
| `preferential_bo` | `evidence_heuristic` | 0.2239 | 0.3153 |
| `preferential_bo` | `h0_reliability` | 0.1831 | 0.3850 |
| `preferential_bo` | `original_rhi` | 0.2398 | 0.4125 |
| `preferential_bo` | `scivoi_component_features` | 0.2326 | 0.4059 |
| `preferential_bo` | `scivoi_component_routing` | 0.1831 | 0.3850 |
| `preferential_bo` | `scivoi_component_uncertainty` | 0.1837 | 0.3850 |
| `preferential_bo` | `scivoi_component_utility` | 0.2326 | 0.3163 |
| `preferential_bo` | `scivoi_policy_always_accept` | 0.3181 | 0.3902 |
| `preferential_bo` | `scivoi_policy_mean_guarded` | 0.2871 | 0.3983 |
| `preferential_bo` | `scivoi_rhi` | 0.2441 | 0.4034 |
| `preferential_bo` | `static_full_reliability` | 0.2577 | 0.4021 |
| `preferential_bo` | `static_utility` | 0.3040 | 0.2627 |
| `preferential_bo` | `static_voi` | 0.2976 | 0.2641 |
| `preferential_bo` | `verbal_confidence` | 0.2451 | 0.4652 |

## Interpretation

- `scivoi_policy_always_accept` is the direct RHI-style recursive update: it accepts every schema-valid executable VoI mutation, matching the original paper's no-rollback update pattern more closely than the conservative guarded variant.
- Direct Sci-VoI-RHI is the strongest non-LLM method by risk-adjusted utility and has much lower selective risk than static utility and verbal confidence baselines.
- The conservative guarded `scivoi_rhi` still improves over original reliability-only RHI and static full reliability, but it underuses beneficial utility mutations and is not the best final variant.
- Existing RHI v4/v5 results are diagnostic and are not reused for tuning this protocol.
- Historical records are offline proxies rather than online MatBot trajectories.
