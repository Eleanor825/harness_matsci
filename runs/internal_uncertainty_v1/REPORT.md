# Plug-in LLM Judge + VoI + RHI Pilot

## Intended method

The final harness accepts model-dependent uncertainty providers:

- closed LLM: structured self-reported `p_success`;
- open model: candidate-token logit margin;
- external signals: evidence conflict, OOD, tool disagreement, cost, and reversibility;
- local signals: utility-ensemble disagreement.

The executable VoI layer calibrates and fuses the available signals, then routes
the action to execute, verify, or stop. RHI evolves the signal contract,
features, weights, thresholds, and routing rules on feedback/acceptance records.

## Current experiments

### Existing real GPT-5.5 subset

| Method | Score ↓ | ECE ↓ | Risk@10% ↓ | Hit ↑ |
| --- | ---: | ---: | ---: | ---: |
| Direct GPT-5.5 judge | 0.3602 | 0.1548 | 0.3462 | 0.6800 |
| GPT-5.5 + static VoI harness | **0.3315** | **0.0974** | 0.3462 | 0.6800 |

This supports a static calibration/fusion claim, not yet an RHI or discovery-
utility claim.

### Unified open-model pilot, 600 records

| Method | Utility efficiency / normalized utility | Hit rate | Risk |
| --- | ---: | ---: | ---: |
| Direct model signal | 0.0134 | 0.1667 | 0.8333 at top 10% |
| Model signal + VoI harness | 0.8252 | 0.9167 | 0.1111 |
| Model signal + VoI + RHI | **0.9857** | **1.0000** | **0.0333** |
| Local VoI + RHI, no model signal | **0.9857** | **1.0000** | **0.0323** |

The direct model in this pilot is `distilgpt2`, whose action-choice accuracy is
only `0.4333`. The within-task signal permutation matches the static harness
utility (`0.8252`), and local/logit RHI converge to the same `H2_scivoi_full`
checkpoint. Therefore the experiment validates the plug-in pipeline and the
harness/RHI mechanism, but does not show a causal contribution from this weak
model's internal uncertainty.

## Reporting boundary

Supported now:

1. Static GPT-5.5 + VoI improves score calibration over direct GPT-5.5 on the
   same 500-record subset.
2. The plug-in harness can consume closed-model self-reports or open-model
   logits and run the same VoI/RHI pipeline.
3. The local pilot demonstrates direct → harness → RHI execution end to end.

Not supported yet:

1. GPT-5.5 + VoI + RHI improves full held-out materials discovery utility.
2. Internal model uncertainty is the cause of the local RHI gain.
3. Adaptive routing preserves quality while reducing LLM calls.
