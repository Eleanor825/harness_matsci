# Current Experiment Result Tables

This file is the compact, presentation-ready result table pack for the current repository state. All results are offline historical/proxy evaluations, not online MatBot lab trajectories.

## Data and Protocol

Full data provenance and action-construction details are in `docs/research/DATA_PROVENANCE_AND_ACTION_CONSTRUCTION.md`.

| Item | Value |
| --- | ---: |
| Total historical proxy records | 15,717 |
| `discover_unique` records | 10,987 |
| `extreme_properties` records | 2,000 |
| `preferential_bo` records | 2,730 |
| Scientific regimes / outer folds | 21 regimes × 5 seeds = 105 folds |
| Fixed discovery budget | top 10% actions |
| RHI iterations | 3 |
| Target risk alpha | 0.10 |

## Main Full-Benchmark Results

Primary metric is oracle-normalized net scientific utility at a fixed 10% action budget; higher is better. Risk is selective risk among executed actions; lower is better.

| Method | Net utility ↑ | Risk-adjusted ↑ | Risk ↓ | Hit rate ↑ | Utility efficiency ↑ | Folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sci-VoI-RHI, direct RHI-style | 0.6443 ± 0.2052 | 0.6043 | 0.1600 | 0.2452 | 0.6443 | 105 |
| Sci-VoI-RHI, mean-guarded | 0.6384 ± 0.2173 | 0.5980 | 0.1616 | 0.2458 | 0.6384 | 105 |
| Sci-VoI-RHI, robust-guarded | 0.6137 ± 0.2201 | 0.5547 | 0.2361 | 0.2328 | 0.6137 | 105 |
| Static VoI head | 0.6372 ± 0.2105 | 0.5410 | 0.3845 | 0.2398 | 0.6372 | 105 |
| Static utility head | 0.6398 ± 0.2091 | 0.5437 | 0.3842 | 0.2411 | 0.6398 | 105 |
| Original reliability-only RHI | 0.5635 ± 0.1916 | 0.4307 | 0.5314 | 0.2302 | 0.5635 | 105 |
| Static full reliability gate | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 0.2512 | 0.5635 | 105 |
| H0 reliability-only gate | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 0.5990 | 105 |
| Verbal confidence heuristic | 0.7577 ± 0.2888 | 0.5605 | 0.7889 | 0.3767 | 0.7577 | 105 |
| Evidence heuristic | 0.5725 ± 0.2255 | 0.4247 | 0.5914 | 0.2649 | 0.5725 | 105 |

## Paired Comparisons

| Variant | Baseline | Utility Δ ↑ | 95% CI | Win rate | Sign-test p |
| --- | --- | ---: | --- | ---: | ---: |
| `scivoi_policy_always_accept` | `original_rhi` | 0.0808 | [0.0549, 0.1059] | 0.829 | 0.0000 |
| `scivoi_policy_always_accept` | `static_full_reliability` | 0.0809 | [0.0584, 0.1025] | 0.829 | 0.0000 |
| `scivoi_policy_always_accept` | `static_voi` | 0.0072 | [0.0028, 0.0128] | 0.419 | 0.2543 |
| `scivoi_policy_always_accept` | `static_utility` | 0.0045 | [-0.0000, 0.0101] | 0.400 | 0.4944 |
| `scivoi_rhi` | `original_rhi` | 0.0502 | [0.0283, 0.0706] | 0.743 | 0.0000 |
| `scivoi_rhi` | `static_full_reliability` | 0.0502 | [0.0235, 0.0757] | 0.705 | 0.0000 |
| `scivoi_rhi` | `static_voi` | -0.0235 | [-0.0451, -0.0025] | 0.352 | 0.1654 |
| `static_voi` | `h0_reliability` | 0.0382 | [0.0009, 0.0734] | 0.676 | 0.0004 |

## Mechanism Ablation

| Mechanism test | Variant | Baseline | Utility Δ ↑ | Risk Δ ↓ | Risk-adjusted Δ ↑ | Win rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| continuous utility head vs reliability-only H0 | `static_utility_no_cost` | `h0_reliability` | 0.0447 | -0.0205 | 0.0498 | 0.686 |
| recursive updates vs frozen harness | `scivoi_policy_always_accept` | `scivoi_policy_never_accept` | 0.0454 | -0.3522 | 0.1334 | 0.676 |
| Sci-VoI RHI vs reliability-only RHI | `scivoi_policy_always_accept` | `original_rhi` | 0.0808 | -0.3713 | 0.1736 | 0.829 |
| guarded Sci-VoI RHI vs reliability-only RHI | `scivoi_rhi` | `original_rhi` | 0.0502 | -0.2953 | 0.1240 | 0.743 |
| full action-value policy vs static reliability | `scivoi_policy_always_accept` | `static_full_reliability` | 0.0809 | -0.3711 | 0.1736 | 0.829 |

## Direct LLM Judge Subset Results

Score is the direct-judge diagnostic score combining AURC, calibration, log loss, and fixed-budget discovery efficiency; lower is better. This is a 500-record subset, not the full 15,717-record held-out-regime protocol.

| Method | Model/proxy | Score ↓ | Selective risk ↓ | Coverage ↑ | Risk@10% ↓ | Hit rate ↑ | Utility efficiency ↑ | ECE ↓ | Brier ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_direct_judge_gpt55` | gpt-5.5 | 0.3602 | 0.7436 | 0.7769 | 0.3462 | 0.6800 | 0.0809 | 0.1548 | 0.1673 |
| `verbal_confidence` | heuristic | 0.4757 | 0.7654 | 0.9681 | 0.4231 | 0.6000 | 0.0696 | 0.3017 | 0.2669 |
| `evidence_heuristic` | heuristic | 0.8131 | 0.8150 | 0.7968 | 0.8846 | 0.1200 | 0.4094 | 0.5879 | 0.5511 |

## Hybrid LLM + VoI Subset Results

This uses the same 500-record subset and cached `gpt-5.5` scores as the direct-judge table. The hybrid result tests whether the LLM judge can serve as a semantic sensor while the local harness calibrates and blends action value.

| Method | Score ↓ | Risk@10% ↓ | Hit rate ↑ | Mean utility ↑ | Utility efficiency ↑ | ECE ↓ | LLM call rate ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_static_blend` | 0.3315 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.0974 | 1.0000 |
| `hybrid_adaptive_router` | 0.3315 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.0974 | 1.0000 |
| `llm_direct_judge` | 0.3602 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.1548 | 1.0000 |
| `hybrid_llm_guarded_blend` | 0.5000 | 0.3462 | 0.6800 | 0.0884 | 0.0958 | 0.1110 | 1.0000 |
| `hybrid_adaptive_30pct_llm` | 0.5593 | 0.6154 | 0.4000 | 0.5125 | 0.5557 | 0.2803 | 0.3506 |
| `local_voi_harness` | 0.6332 | 0.7308 | 0.2400 | 0.7910 | 0.8576 | 0.3769 | 0.0000 |

Main takeaway: hybrid calibration improves diagnostic score and ECE versus direct `gpt-5.5`, but the current subset run does not improve Risk@10% or hit rate and does not yet reduce LLM calls for the best row.

## GPT-5.6-Luna Attempt and Model Availability

| Model | Current status |
| --- | --- |
| `gpt-5.2` | HTTP 503 Service Unavailable |
| `gpt-5.3-codex` | HTTP 503 Service Unavailable |
| `gpt-5.4` | ok |
| `gpt-5.4-mini` | ok |
| `gpt-5.5` | ok |
| `gpt-5.6-luna` | HTTP 429 Too Many Requests |

Same 100-record cached `gpt-5.5` control:

| Method | Score ↓ | Selective risk ↓ | Coverage ↑ | Risk@10% ↓ | Hit rate ↑ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.8310 | 0.8261 | 0.4600 | 1.0000 | 0.0000 | 0.5370 |
| `llm_direct_judge_gpt55_cached` | 0.3484 | 0.6667 | 0.7800 | 0.2000 | 0.8000 | 0.1213 |
| `verbal_confidence` | 0.5432 | 0.7234 | 0.9400 | 0.6000 | 0.4000 | 0.2381 |

## Interpretation Boundary

- The full-benchmark claim is positive for Sci-VoI-RHI against reliability-only RHI and static reliability baselines.
- The current real direct-judge subset shows `gpt-5.5` is a strong one-shot judge and currently beats our quick same-subset RHI check on top-k ranking; our stronger claim remains the full held-out-regime Sci-VoI protocol plus better calibration/risk-control behavior.
- The current hybrid subset result supports a narrower claim: LLM grounding plus local calibration improves action-worthiness diagnostics and calibration, but not top-k discovery yet.
- `gpt-5.6-luna` is not counted as an experiment result because the provider currently returns HTTP 429.
