# Full Self-Reported Confidence Comparison

## Protocol

- Three tasks: `preferential_bo`, `discover_unique`, `extreme_properties`.
- 100 records per task, 300 records total for every model.
- Same seed (`1729`) and same record IDs for all models.
- Split: 180 train, 30 feedback, 30 acceptance, 60 untouched test.
- Fixed action budget: 10% of test records, six actions.
- Open models run locally with greedy decoding.
- Closed models use the same provider prompt with `OPENAI_REASONING_EFFORT=minimal`.
- Self-reported confidence is a model-reported signal, not intrinsic logit uncertainty.

## Main results

| Model | Direct hit ↑ | Direct utility lift ↑ | Direct ECE ↓ | Static VoI utility ↑ | VoI+RHI utility ↑ | VoI+RHI risk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.6667 | 0.3350 | 0.3882 | 0.9011 | **0.9923** | 0.0909 |
| Qwen2.5-1.5B-Instruct | 0.3333 | -0.1361 | 0.5267 | 0.8433 | **0.9923** | 0.0909 |
| SmolLM2-1.7B-Instruct | 0.3333 | -0.1839 | 0.5098 | **0.9923** | **0.9923** | 0.3889 |
| GPT-5.5 | **0.8333** | **0.5614** | 0.3477 | **0.9923** | **0.9923** | **0.0833** |
| GPT-5.6-Luna | 0.6667 | 0.4409 | 0.3353 | 0.9144 | **0.9923** | **0.0769** |

## Signal and RHI controls

| Model | Static VoI | Permuted confidence VoI | VoI+RHI | Interpretation |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 0.9011 | 0.9011 | 0.9923 | RHI gain present in this protocol; confidence-specific gain not isolated |
| Qwen2.5-1.5B | 0.8433 | 0.8433 | 0.9923 | Same conclusion |
| SmolLM2-1.7B | 0.9923 | 0.9011 | 0.9923 | Permutation hurts, but RHI adds no utility over static VoI |
| GPT-5.5 | 0.9923 | 0.9923 | 0.9923 | No incremental evidence for confidence or RHI on this split |
| GPT-5.6-Luna | 0.9144 | 0.9923 | 0.9923 | Permutation control is noisy; no causal claim |

## Interpretation

The strongest supported result is cross-model **pipeline compatibility**: three
local open models and two closed GPT-family models can emit structured
self-reported confidence that is consumed by the same VoI/RHI harness. The
models differ substantially as direct judges, while the executable harness
often recovers high fixed-budget utility.

The results do not establish that self-reported confidence is the cause of the
improvement. The confidence-permutation control is tied or close to the static
method for several models, and the final RHI value is identical across models
because the local action-level features and deterministic harness proposer
dominate this synthetic benchmark. The experiment also does not establish that
RHI generalizes to a real materials-discovery benchmark or that an LLM proposer
is superior to the deterministic proposer.

## Reproducibility artifacts

- Open: `runs/open_model_self_report_v1/summary.json`,
  `runs/open_model_self_report_qwen15_full_v1/summary.json`,
  `runs/open_model_self_report_smollm17_full_v1/summary.json`.
- Closed: `runs/closed_model_self_report_gpt55_full_v1/summary.json`,
  `runs/closed_model_self_report_gpt56_full_v1/summary.json`.
- Runner: `experiments/internal_uncertainty/run_self_report.py` and
  `experiments/internal_uncertainty/run_closed_self_report.py`.
