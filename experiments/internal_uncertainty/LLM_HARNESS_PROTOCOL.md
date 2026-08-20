# LLM Judge + Plug-in Harness Protocol

## Main claim

At fixed LLM judge outputs, an uncertainty-aware executable harness should
improve calibrated action selection and risk-adjusted scientific utility over
the direct LLM judge alone. RHI then evolves the harness contract, not the LLM
weights, using action feedback and held-out acceptance regimes.

## Plug-in boundary

The harness accepts any decision-time provider implementing:

```python
score(records) -> {record_id: UncertaintySignal}
```

Supported signal sources are:

- `open_model_self_report`: an open-source instruction model is explicitly
  asked to report its own action confidence and uncertainty;
- `llm_self_report`: closed-model structured `p_success` / uncertainty;
- `open_model_logit`: candidate-token logit margin;
- `external`: OOD, tool disagreement, evidence conflict, or ensemble signals;
- `composite`: multiple calibrated sources.

The first internal-signal experiment uses only `open_model_self_report`. The
model must produce the action decision and a structured confidence value in
the same call. This is a model-reported confidence signal, not a strict
intrinsic uncertainty measurement. Logit-based uncertainty is kept as a
separate follow-up ablation because it requires a model with accessible
token-level logits and tests a different hypothesis.

The provider is calibrated on train/feedback records only. Test labels and
utilities never enter signal construction, calibration, blend selection, or RHI
acceptance.

## Required comparisons

1. Direct LLM judge.
2. Direct LLM judge + static harness calibration.
3. Direct LLM judge + VoI routing.
4. Direct LLM judge + VoI + RHI evolution.
5. Provider ablations: open-model self-report only, logits only, external
   only, composite.

## Metrics

Primary: oracle-normalized net utility at fixed budget, risk-adjusted utility,
and selective risk. Secondary: ECE, Brier, Risk@10%, hit rate, verify rate,
stop rate, worst-regime utility, and LLM call cost.

## Current evidence boundary

The repository's existing real GPT-5.5 500-record subset supports a calibration
claim for static LLM+VoI fusion (`0.3315` score vs `0.3602` direct; ECE `0.0974`
vs `0.1548`). It does not yet support a full LLM+VoI+RHI superiority claim.
The new plug-in interface makes that next run protocol-compatible with either a
closed-model self-report provider or an open-model logit provider.
