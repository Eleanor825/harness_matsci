# Direct LLM Judge Baseline

## Status

Protocol and implementation are ready. The client now uses the current
`gpt-5.5` / `https://www.hi-code.cc` Responses configuration and sends
OpenAI/browser-compatible request headers so the gateway does not reject Python
`urllib` traffic as a bot request. A small end-to-end smoke run is stored in
`runs/direct_judge_smoke_v1/`; it is not a formal paper result. A real
500-record `gpt-5.5` subset result is stored in
`runs/direct_judge_subset500_v1/`. No mock scores are reported as scientific
results.

## Current Real Runs

| Run | Model | Status | Key result |
| --- | --- | --- | --- |
| `runs/direct_judge_subset500_v1/` | `gpt-5.5` | complete | score `0.3602`, Risk@10% `0.3462`, hit rate `0.6800`, ECE `0.1548` on 500 records. |
| `runs/direct_judge_subset100_gpt56luna_v1/` | `gpt-5.6-luna` | blocked | provider returns HTTP `429`; same-100 cached `gpt-5.5` control gives score `0.3484`, Risk@10% `0.2000`, hit rate `0.8000`. |

Single-record gateway smoke checks currently pass for `gpt-5.5`, `gpt-5.4`,
and `gpt-5.4-mini`. `gpt-5.3-codex` and `gpt-5.2` return provider `503`, and
`gpt-5.6-luna` returns provider `429`.

## Definition

`llm_direct_judge` is a one-shot baseline for the same action-worthiness task as
RHI. For each record, the model sees only the decision-time visible context,
candidate action, and sanitized pre-execution evidence. It returns a JSON
`p_success` in `[0, 1]` and a route/rationale. It receives no labels, feedback,
acceptance shards, or previous harness versions.

The score threshold is calibrated on the same validation/feedback partition
used by the corresponding weak baseline. The test partition remains held out
until final evaluation. Experiment 2 is zero-shot: target-task records are not
used to calibrate the direct judge.

## Run

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.5
export OPENAI_BASE_URL=https://www.hi-code.cc
export OPENAI_REASONING_EFFORT=xhigh
PYTHONPATH=src python3 -m harness_matsci experiment-suite \
  --data-dir /path/to/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --direct-judge-model "$OPENAI_MODEL" \
  --direct-judge-base-url "$OPENAI_BASE_URL" \
  --direct-judge-reasoning-effort "$OPENAI_REASONING_EFFORT" \
  --direct-judge-cache runs/direct_judge_cache/scores.json \
  --out runs/direct_judge_baseline_v1/summary.json \
  --markdown-out runs/direct_judge_baseline_v1/README.md
```

The cache is ignored by Git. The generated report records the model name,
protocol, calibration split, and method-level metrics without storing the API
key.
