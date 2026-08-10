# Sci-VoI Mechanism Ablation

> Primary metric: oracle-normalized net utility at a fixed 10% action budget; higher is better. Risk is selective risk among executed actions; lower is better.

- Data: `15717` historical proxy records.
- Outer folds: `21` held-out scientific regimes × `5` seeds.
- Direct LLM judge is excluded because no API key is configured.

## Aggregate Methods

| Method | Net utility ↑ | Risk-adjusted ↑ | Risk ↓ | Hit rate ↑ | Folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h0_reliability` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 105 |
| `original_rhi` | 0.5635 ± 0.1916 | 0.4307 | 0.5314 | 0.2302 | 105 |
| `scivoi_policy_always_accept` | 0.6443 ± 0.2052 | 0.6043 | 0.1600 | 0.2452 | 105 |
| `scivoi_policy_mean_guarded` | 0.6384 ± 0.2173 | 0.5980 | 0.1616 | 0.2458 | 105 |
| `scivoi_policy_never_accept` | 0.5990 ± 0.2353 | 0.4709 | 0.5123 | 0.2429 | 105 |
| `scivoi_rhi` | 0.6137 ± 0.2201 | 0.5547 | 0.2361 | 0.2328 | 105 |
| `static_full_reliability` | 0.5635 ± 0.2114 | 0.4307 | 0.5312 | 0.2512 | 105 |
| `static_utility_no_cost` | 0.6436 ± 0.2094 | 0.5207 | 0.4918 | 0.2420 | 105 |
| `static_voi` | 0.6442 ± 0.2076 | 0.5252 | 0.4760 | 0.2424 | 105 |
| `static_voi_no_cost` | 0.6443 ± 0.2076 | 0.5214 | 0.4918 | 0.2422 | 105 |
| `static_voi_no_routing` | 0.6442 ± 0.2076 | 0.5252 | 0.4760 | 0.2424 | 105 |
| `static_voi_no_uncertainty` | 0.6434 ± 0.2092 | 0.5192 | 0.4971 | 0.2412 | 105 |

## Mechanism Comparisons

| Mechanism test | Variant | Baseline | Utility Δ ↑ | Risk Δ ↓ | Risk-adjusted Δ ↑ | Win rate | 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| continuous utility head vs reliability-only H0 | `static_utility_no_cost` | `h0_reliability` | 0.0447 | -0.0205 | 0.0498 | 0.686 | [0.0059, 0.0810] |
| uncertainty removal vs utility-only head | `static_voi_no_uncertainty` | `static_utility_no_cost` | -0.0002 | 0.0053 | -0.0015 | 0.162 | [-0.0018, 0.0013] |
| routing removal vs uncertainty-only VoI | `static_voi_no_routing` | `static_voi_no_uncertainty` | 0.0008 | -0.0211 | 0.0061 | 0.210 | [-0.0019, 0.0041] |
| cost-aware VoI vs cost-blind VoI | `static_voi` | `static_voi_no_cost` | -0.0001 | -0.0158 | 0.0039 | 0.010 | [-0.0005, 0.0002] |
| recursive updates vs frozen harness | `scivoi_policy_always_accept` | `scivoi_policy_never_accept` | 0.0454 | -0.3522 | 0.1334 | 0.676 | [0.0083, 0.0808] |
| Sci-VoI RHI vs reliability-only RHI | `scivoi_policy_always_accept` | `original_rhi` | 0.0808 | -0.3713 | 0.1736 | 0.829 | [0.0549, 0.1059] |
| guarded Sci-VoI RHI vs reliability-only RHI | `scivoi_rhi` | `original_rhi` | 0.0502 | -0.2953 | 0.1240 | 0.743 | [0.0283, 0.0706] |
| full action-value policy vs static reliability | `scivoi_policy_always_accept` | `static_full_reliability` | 0.0809 | -0.3711 | 0.1736 | 0.829 | [0.0584, 0.1025] |

## Interpretation

- `static_utility_no_cost` vs `h0_reliability` isolates whether modeling continuous scientific utility helps beyond reliability alone.
- `static_voi_no_uncertainty` vs `static_utility_no_cost` isolates epistemic uncertainty, and `static_voi_no_routing` vs `static_voi_no_uncertainty` isolates the routing step in a non-recursive harness.
- `static_voi` vs `static_voi_no_cost` tests whether using cost in the decision score matters when evaluation still charges action cost.
- `scivoi_policy_always_accept` vs `scivoi_policy_never_accept` tests whether recursive harness mutation adds value beyond a frozen contract.
- `scivoi_policy_always_accept` vs `original_rhi` tests the main method against reliability-only RHI. `scivoi_rhi` vs `scivoi_policy_never_accept` isolates guarded recursion against a frozen harness.

## Claim Boundary

These results support mechanism attribution on offline proxy tasks. They should be paired with the label/utility audit and, later, direct LLM judge and live MatBot trajectory baselines.
