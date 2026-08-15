# Hybrid LLM + VoI Judge Subset Experiment

This run evaluates whether a direct LLM judge can be combined with the local Sci-VoI harness at decision time.
The result uses the cached GPT-5.5 direct judge scores from the 500-record subset; no new LLM calls are made by this run.

## Protocol

- Records: `500` total; test `251`.
- Local VoI train/calibration split: `125` / `124` from the validation records.
- Fixed action budget: top `10%` actions.
- Target selective-risk alpha: `0.1`.
- Score is the existing action-worthiness diagnostic; lower is better.

## Results

| Method | Score ↓ | Risk@10% ↓ | Hit ↑ | Mean utility ↑ | Utility eff. ↑ | ECE ↓ | LLM call rate ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_static_blend` | 0.3315 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.0974 | 1.0000 |
| `hybrid_adaptive_router` | 0.3315 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.0974 | 1.0000 |
| `llm_direct_judge` | 0.3602 | 0.3462 | 0.6800 | 0.0746 | 0.0809 | 0.1548 | 1.0000 |
| `verbal_confidence` | 0.4757 | 0.4231 | 0.6000 | 0.0642 | 0.0696 | 0.3017 | 0.0000 |
| `hybrid_llm_guarded_blend` | 0.5000 | 0.3462 | 0.6800 | 0.0884 | 0.0958 | 0.1110 | 1.0000 |
| `hybrid_adaptive_30pct_llm` | 0.5593 | 0.6154 | 0.4000 | 0.5125 | 0.5557 | 0.2803 | 0.3506 |
| `hybrid_adaptive_50pct_llm` | 0.5905 | 0.6923 | 0.3200 | 0.3940 | 0.4271 | 0.2435 | 0.5618 |
| `local_voi_harness` | 0.6332 | 0.7308 | 0.2400 | 0.7910 | 0.8576 | 0.3769 | 0.0000 |
| `evidence_heuristic` | 0.8131 | 0.8846 | 0.1200 | 0.3777 | 0.4094 | 0.5879 | 0.0000 |

## Selected Hybrid Parameters

- Static blend LLM weight: `0.85`.
- LLM-guarded blend weight: `0.70` with floor `0.0200`.
- Adaptive router LLM weight: `0.85`.
- Adaptive router target LLM call fraction: `1.00`.
- Adaptive router observed test LLM call rate: `1.0000`.

## Interpretation

- `llm_direct_judge` is a one-shot GPT-5.5 action-worthiness score.
- `local_voi_harness` is the local reliability/utility/uncertainty harness without LLM semantics.
- `hybrid_static_blend` always combines both signals.
- `hybrid_llm_guarded_blend` treats the LLM judge as a safety floor and lets VoI rerank actions that pass the floor.
- `hybrid_adaptive_router` uses the local harness by default and consults the LLM judge only on high-uncertainty/high-risk actions.
- `hybrid_adaptive_30pct_llm` and `hybrid_adaptive_50pct_llm` force an approximate LLM-call budget to expose the cost-quality trade-off.
- This is a subset experiment; the next step is to run the same hybrid protocol on the full held-out-regime benchmark and with Claude-family judges.

## Split Sensitivity

| Local train fraction | LLM score ↓ | Hybrid score ↓ | Local score ↓ | Blend LLM weight |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.3602 | 0.3315 | 0.6332 | 0.85 |
| 0.68 | 0.3602 | 0.3558 | 0.8702 | 0.65 |
| 0.80 | 0.3602 | 0.3412 | 1.0422 | 0.70 |
| 0.90 | 0.3602 | 0.3548 | 1.1098 | 0.65 |
